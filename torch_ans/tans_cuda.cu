// The tANS family kernels (CUDA) and their launchers.
//
// Structure mirrors rans_cuda.cu: the algorithm itself lives in
// tans_utils.hpp (already written for a CUDA port - `TANS_API` / `TANS_INLINE`
// become `__device__` / `__device__ __forceinline__` behind RANS_CUDA_API,
// exactly like rans_utils.hpp), and this file is the tensor plumbing plus the
// kernels. The op wrappers in tans.hpp route on the tensor device, so the
// operator names, the signatures and the Python call sites stay identical on
// CPU and CUDA.
//
// Single lane only: a tANS stream holds one state and one bit buffer, and the
// interleaved variants stay CPU-only for now (the wrapper rejects them with an
// explanation). Every stream row is therefore one independent serial task, and
// one thread codes one row - the same "one thread per stream" organisation the
// non-interleaved rANS kernels use, and the reason the CUDA kernels are
// bit-compatible with the CPU ones (the loop bodies below are the single-lane
// CPU path verbatim).
//
// Table staging: the tANS table *is* the symbol lookup, and each symbol reads
// it at a state- or symbol-dependent offset, so it is the main memory-latency
// source of both directions. When the whole table fits a per-block budget it is
// staged into shared memory and the lookups are served from there; see
// tans_table_staging below for the budget and rans_warp_cuda.cuh's A9 note for
// why whole-table-or-nothing (a partial stage mixes two memories without
// removing the divergent global loads).
//
// Kernels and launchers are written against the ta:: compatibility layer
// (torch_ans/ta/torch_api.hpp), so this translation unit compiles in both build
// modes (the ATen/pybind11 build and the libtorch-stable-ABI build).
#include "ta/torch_api.hpp"

#if defined(WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(WITH_HIP)
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <cstdlib>
#include <type_traits>

// enable compiling tans_utils.hpp for kernel call
#define RANS_CUDA_API

#include "rans_build_config.hpp"
#include "tans.hpp"
#include "tans_utils.hpp"

// ---------------------------------------------------------------------------
// launch configuration
// ---------------------------------------------------------------------------
// One thread codes one stream row, so the block size trades two things:
//   * `blocks = ceil(batch / threads)` - how many SMs get work. A large block
//     leaves most of the device idle for a modest batch: the first version
//     defaulted to 64 threads, which codes 1024 rows in 16 blocks and measured
//     ~18% slower (decode) than 8 threads per block;
//   * the shared-memory staging, which is paid *per block*: a 13 KiB table
//     copied by 1-thread blocks means one copy per row (1024 of them) next to
//     ~4 KiB of payload per row, which is why staging wants fewer, larger
//     blocks while the global path wants more, smaller ones.
// `tans_num_threads_per_block` derives the block size from the batch size; its
// comment carries the measured table behind the formula.
// TORCH_ANS_TANS_BLOCK_THREADS overrides it so a sweep needs no rebuild.

//: Number of blocks a launch aims at (see tans_num_threads_per_block).
#ifndef TANS_BLOCKS_TARGET
#define TANS_BLOCKS_TARGET 128
#endif

//: Upper bound of the derived block size (a batch large enough to fill the
//: device wants a small number of wide blocks, not 1024-thread ones).
#ifndef TANS_MAX_THREADS_PER_BLOCK
#define TANS_MAX_THREADS_PER_BLOCK 128
#endif

// Shared-memory staging budget, in bytes per block. The default is 1/4 of the
// per-SM shared memory of the target architectures, i.e. the largest budget
// that keeps four blocks resident per SM (the same value and reasoning as A9 in
// rans_warp_cuda.cuh). `TORCH_ANS_TANS_SMEM_BUDGET` overrides it; 0 disables
// staging entirely, which is what an A/B measurement of the feature wants. Keep
// it at or below the device's per-block shared memory limit (48 KiB without an
// opt-in on sm_75), since nothing below sets the function attribute.
#ifndef TANS_SMEM_BUDGET_DEFAULT
#define TANS_SMEM_BUDGET_DEFAULT (16 * 1024)
#endif

namespace {

int tans_num_threads_per_block(int64_t batch_size) {
  const char* env = std::getenv("TORCH_ANS_TANS_BLOCK_THREADS");
  if (env != nullptr) {
    const long value = std::atol(env);
    // A non-positive or absurd value would make the launch fail; fall back to
    // the default instead of turning a measurement knob into a crash.
    if (value > 0 && value <= 1024) return (int)value;
  }
  // Measured on RTX 2080 Ti (tans32, 8 distributions, fp 7, alphabet 16, 1M
  // symbols, table in *global* memory - scripts/bench_tans_vs_rans.py
  // --smem-budget 0), encode MB/s per threads-per-block:
  //   batch    1     2     4     8    16    32    64   128   256
  //    256   381   372   357   424   412   399   395   347   283
  //   1024   557   603   599   596   588   574   563   575   539
  //   4096   395   449   472   472   477   483   485   478   466
  // The optimum sits around 128 blocks at every batch size - fewer SMs used
  // above it, more per-block staging cost below it - so the block size is the
  // batch over that target. Small batches therefore get 1-2 threads (where the
  // staging also measured best) and a device-filling batch gets wide blocks.
  const int64_t threads = (batch_size + TANS_BLOCKS_TARGET - 1) / TANS_BLOCKS_TARGET;
  if (threads < 1) return 1;
  if (threads > TANS_MAX_THREADS_PER_BLOCK) return TANS_MAX_THREADS_PER_BLOCK;
  return (int)threads;
}

int64_t tans_smem_budget_bytes() {
  const char* env = std::getenv("TORCH_ANS_TANS_SMEM_BUDGET");
  if (env != nullptr) return std::atoll(env);
  return TANS_SMEM_BUDGET_DEFAULT;
}

// Whether the whole table of `tables` (`num_dists x row_words` int32 words) is
// staged into shared memory, and how many bytes that needs. `words == 0` means
// "no staging" - the global-memory kernel is launched with no dynamic shared
// memory at all.
struct TansTableStaging {
  int64_t words;
  size_t bytes;
};

inline TansTableStaging tans_table_staging(const ta::Tensor& tables) {
  const int64_t words = tables.size(0) * tables.size(1);
  const int64_t budget = tans_smem_budget_bytes();
  if (budget > 0 && words * (int64_t)sizeof(int32_t) <= budget) {
    return TansTableStaging{words, (size_t)words * sizeof(int32_t)};
  }
  return TansTableStaging{0, 0};
}

// The pointer every table lookup goes through. With USE_SHARED_TABLE the whole
// table is copied into the block's shared memory first (one pass over
// `smem_words` words, the threads of the block in stride) and every thread then
// returns the shared pointer; without it this is the global pointer and compiles
// to nothing. Both kernels call it *before* their `b >= size0` early return,
// because the barrier inside has to be reached by every thread of the block.
template <bool USE_SHARED_TABLE>
inline __device__ const int32_t* tans_stage_table(const int32_t* __restrict__ tables_data,
                                                  int32_t* smem, int64_t smem_words) {
  if constexpr (USE_SHARED_TABLE) {
    for (int64_t k = (int64_t)threadIdx.x; k < smem_words; k += (int64_t)blockDim.x) {
      smem[k] = tables_data[k];
    }
    __syncthreads();
    return smem;
  } else {
    (void)smem;
    (void)smem_words;
    return tables_data;
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// push
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_SHARED_TABLE = false, size_t NUM_INTERLEAVES = 1>
__global__ void tans_push_indexed_cuda_kernel(
  ta::View2<int32_t> stream_view,
  const ta::View2<TORCH_TENSOR_TYPE> symbols_view,
  const ta::View2<TORCH_TENSOR_TYPE> indexes_view,
  const int32_t* __restrict__ tables_data,
  const int64_t tables_row_stride,
  const ta::View1<TORCH_TENSOR_TYPE> cdfs_sizes_view,
  const ta::View1<TORCH_TENSOR_TYPE> offsets_view,
  const int freq_precision,
  const bool bypass_coding,
  const int bypass_precision,
  const int64_t num_dists,
  const int64_t smem_table_words)
{
  // Single lane: the interleaved variants stay CPU-only for now (see the file
  // header), and the wrapper never routes them here.
  static_assert(NUM_INTERLEAVES == 1,
                "the tANS CUDA kernels are single-lane; interleaved tANS is CPU-only for now");
  constexpr int64_t NUM_STATES = (int64_t)NUM_INTERLEAVES;
  constexpr int64_t LANE = 0;

  extern __shared__ char tans_smem_raw[];
  const int32_t* const tables = tans_stage_table<USE_SHARED_TABLE>(
      tables_data, reinterpret_cast<int32_t*>(tans_smem_raw), smem_table_words);

  const int64_t b = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (b >= stream_view.size0) return;

  const uint32_t table_log = (uint32_t)freq_precision;
  const uint32_t ts = (uint32_t)1 << table_log;
  const uint32_t bpp = (uint32_t)bypass_precision;
  const bool do_bypass = bypass_coding;
  const int64_t num_symbols = symbols_view.size1;

  int32_t* const header = stream_view.row(b);
  const int64_t length = (int64_t)header[0];
  TANS_STREAM_TYPE* const base = reinterpret_cast<TANS_STREAM_TYPE*>(header + 1);
  TansBitWriter<TANS_STREAM_TYPE> writer{
      (uint64_t)(uint32_t)header[tans_bitbuf_slot(NUM_STATES)],
      (uint32_t)header[tans_nbits_slot(NUM_STATES)],
      base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};

  const TORCH_TENSOR_TYPE* const symbols_row = symbols_view.row(b);
  const TORCH_TENSOR_TYPE* const indexes_row = indexes_view.row(b);

  // 0 is not a legal encoder state (those live in [2**L, 2**(L+1))), so it is
  // the "stream has not been pushed to yet" marker - which is how
  // tans_init_stream can keep rans_init_stream's signature (it does not know
  // table_log).
  uint32_t state = (uint32_t)header[tans_state_slot(LANE)];
  if (state == 0) state = ts;

  // Reverse order: ANS is LIFO, so the decoder meets the symbols in forward
  // order (tans_cpu.cpp's single-lane path, verbatim).
  for (int64_t i = num_symbols - 1; i >= 0; --i) {
    // signed on purpose: the batched API pads with index -1, and comparing an
    // unsigned symbol dtype against zero is what nvcc calls a "pointless
    // comparison" (the CPU kernels spell it out; the cast is what keeps the
    // check identical for every dtype without the warning).
    const int64_t index = (int64_t)indexes_row[i];
    // check index range, skip on invalid indexes (batched padding)
    if (index < 0 || index >= num_dists) continue;
    const int32_t* const row = tables + index * tables_row_stride;
    const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
    const TORCH_TENSOR_TYPE offset = offsets_view.at(index);
    uint32_t value = 0;
    uint32_t raw_val = 0;
    bool bypassed = false;
    tans_cook_symbol<TORCH_TENSOR_TYPE>(symbols_row[i], cdf_size, offset, do_bypass,
                                        &value, &raw_val, &bypassed);
    // The bypass digits go in before the sentinel symbol; the decoder meets
    // them in the mirrored order (see tans_utils.hpp).
    if (bypassed) tans_push_raw_value<TANS_STREAM_TYPE>(&writer, raw_val, bpp);
    state = tans_encode_symbol<TANS_STREAM_TYPE>(state, row, ts, value, &writer);
  }

  header[tans_state_slot(LANE)] = (int32_t)state;
  // At rest the bit buffer holds fewer than STREAM_BITS bits (append() drains
  // while nbits >= STREAM_BITS), so both fit in the header words.
  header[tans_bitbuf_slot(NUM_STATES)] = (int32_t)(uint32_t)writer.buf;
  header[tans_nbits_slot(NUM_STATES)] = (int32_t)writer.nbits;
  // The payload is whole stream words only: the residual bits stay in the
  // header, so the length is unambiguous and the reader needs no trailer.
  header[0] = (int32_t)((writer.ptr - base + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
}

// ---------------------------------------------------------------------------
// pop
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_SHARED_TABLE = false, size_t NUM_INTERLEAVES = 1>
__global__ void tans_pop_indexed_cuda_kernel(
  ta::View2<int32_t> stream_view,
  ta::View2<TORCH_TENSOR_TYPE> symbols_view,
  const ta::View2<TORCH_TENSOR_TYPE> indexes_view,
  const int32_t* __restrict__ tables_data,
  const int64_t tables_row_stride,
  const ta::View1<TORCH_TENSOR_TYPE> cdfs_sizes_view,
  const ta::View1<TORCH_TENSOR_TYPE> offsets_view,
  const int freq_precision,
  const bool bypass_coding,
  const int bypass_precision,
  const int64_t num_dists,
  const int64_t smem_table_words)
{
  static_assert(NUM_INTERLEAVES == 1,
                "the tANS CUDA kernels are single-lane; interleaved tANS is CPU-only for now");
  constexpr int64_t NUM_STATES = (int64_t)NUM_INTERLEAVES;
  constexpr int64_t LANE = 0;

  extern __shared__ char tans_smem_raw[];
  const int32_t* const tables = tans_stage_table<USE_SHARED_TABLE>(
      tables_data, reinterpret_cast<int32_t*>(tans_smem_raw), smem_table_words);

  const int64_t b = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
  if (b >= stream_view.size0) return;

  const uint32_t table_log = (uint32_t)freq_precision;
  const uint32_t ts = (uint32_t)1 << table_log;
  const uint32_t bpp = (uint32_t)bypass_precision;
  const bool do_bypass = bypass_coding;
  const int64_t num_symbols = symbols_view.size1;

  // not const: the continuation point is written back below
  int32_t* const header = stream_view.row(b);
  const int64_t length = (int64_t)header[0];
  const TANS_STREAM_TYPE* const base =
      reinterpret_cast<const TANS_STREAM_TYPE*>(header + 1);
  // The reader walks the payload backwards from its end: the encoder wrote whole
  // words forward, the bits are consumed in reverse order.
  TansBitReader<TANS_STREAM_TYPE> reader{
      (uint64_t)(uint32_t)header[tans_bitbuf_slot(NUM_STATES)],
      (uint32_t)header[tans_nbits_slot(NUM_STATES)],
      base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};

  // The two directions use different ranges for the same state: the encoder's
  // lives in [ts, 2*ts), the decoder's in [0, ts). The state travels in the
  // header instead of in the bit stream, so the conversion is the subtraction.
  uint32_t state = (uint32_t)header[tans_state_slot(LANE)] - ts;

  TORCH_TENSOR_TYPE* const symbols_row = symbols_view.row(b);
  const TORCH_TENSOR_TYPE* const indexes_row = indexes_view.row(b);

  for (int64_t i = 0; i < num_symbols; ++i) {
    // see the push kernel: signed, so that the padding index -1 is caught for
    // an unsigned symbol dtype as well
    const int64_t index = (int64_t)indexes_row[i];
    // check index range, skip on invalid indexes (batched padding)
    if (index < 0 || index >= num_dists) {
      symbols_row[i] = 0;
      continue;
    }
    const int32_t* const row = tables + index * tables_row_stride;
    const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
    const TORCH_TENSOR_TYPE offset = offsets_view.at(index);
    const TORCH_TENSOR_TYPE max_value = cdf_size - 2;

    uint32_t symbol = 0;
    state = tans_decode_symbol<TANS_STREAM_TYPE>(state, row, ts, &reader, &symbol);
    TORCH_TENSOR_TYPE value = (TORCH_TENSOR_TYPE)symbol;

    if (do_bypass && symbol == (uint32_t)max_value) {
      const uint32_t raw_val = tans_pop_raw_value<TANS_STREAM_TYPE>(&reader, bpp);
      value = (TORCH_TENSOR_TYPE)tans_uncook_raw_value(raw_val, (int64_t)max_value);
    }

    symbols_row[i] = value + offset;
  }

  // Write the continuation point back, exactly like tans_cpu.cpp / rans_pop do:
  // the reader's position *is* a tANS header, so a second pop on the same stream
  // decodes the previous push - which is what the chunked decode path relies on.
  // The state goes back in the encoder's range (`+ ts`) so the header keeps one
  // meaning for both directions.
  header[tans_state_slot(LANE)] = (int32_t)(state + ts);
  header[0] = (int32_t)((reader.ptr - base + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
  header[tans_bitbuf_slot(NUM_STATES)] = (int32_t)(uint32_t)reader.buf;
  header[tans_nbits_slot(NUM_STATES)] = (int32_t)reader.nbits;
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
// The table tensor travels in the `cdfs` slot of the shared push/pop signature
// (see tans.hpp): it carries both directions and is always int32, so it is
// passed as a plain pointer plus its row stride instead of a `View2<scalar_t>`
// (the symbols/indexes dtype is only known through TA_DISPATCH_INTEGRAL).
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
void tans_push_indexed_cuda(
  ta::Tensor stream,
  const ta::Tensor& symbols,
  const ta::Tensor& indexes,
  const ta::Tensor& tables,
  const ta::Tensor& cdfs_sizes,
  const ta::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision)
{
  STD_TORCH_CHECK(ta::is_cuda(stream), "tans_push: the stream must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(symbols), "tans_push: symbols must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(indexes), "tans_push: indexes must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(tables), "tans_push: the table tensor must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(cdfs_sizes), "tans_push: cdfs_sizes must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(offsets), "tans_push: offsets must be a CUDA tensor");
  STD_TORCH_CHECK(tables.dim() == 2,
                  "tans_push: the table tensor must be 2-D (one row per distribution)");
  STD_TORCH_CHECK(tables.scalar_type() == ta::ScalarType::Int,
                  "tans_push: the table tensor must be int32 (as tans_build_tables "
                  "returns it)");
  // ArrayRef comparison is not part of the stable interface, so compare the two
  // dimensions explicitly.
  STD_TORCH_CHECK(indexes.size(0) == symbols.size(0) && indexes.size(1) == symbols.size(1),
                  "indexes and symbols must have the same shape");
  STD_TORCH_CHECK(cdfs_sizes.size(0) == tables.size(0) && offsets.size(0) == tables.size(0),
                  "tans_push: cdfs_sizes and offsets must have one entry per table row");

  // The kernels below launch on the tensor's device, so make it current first:
  // the stream handle from ta::current_stream belongs to that device and a
  // launch with a foreign device's stream fails with "invalid resource handle".
  const ta::DeviceGuard device_guard(ta::device_index(stream));

  TA_DISPATCH_INTEGRAL(indexes.scalar_type(), "tans_push_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    const TansTableStaging staging = tans_table_staging(tables);
    const int num_threads_per_block = tans_num_threads_per_block(batch_size);
    const int num_blocks = (int)((batch_size + num_threads_per_block - 1) /
                                 num_threads_per_block);
    const cudaStream_t cuda_stream = ta::current_stream(stream);

    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const int64_t num_dists = tables.size(0);

    // Two instantiations so that the staged path costs nothing when it is not
    // used (the staging decision is a runtime one - the table size is only known
    // with the tensors - but the per-symbol code must not carry the select).
    if (staging.words > 0) {
      tans_push_indexed_cuda_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, true, NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, staging.bytes, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              staging.words);
    } else {
      tans_push_indexed_cuda_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, false, NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              0);
    }
    TA_CUDA_CHECK(cudaGetLastError());
  });
}

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
ta::Tensor tans_pop_indexed_cuda(
  ta::Tensor stream,
  const ta::Tensor& indexes,
  const ta::Tensor& tables,
  const ta::Tensor& cdfs_sizes,
  const ta::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision)
{
  STD_TORCH_CHECK(ta::is_cuda(stream), "tans_pop: the stream must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(indexes), "tans_pop: indexes must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(tables), "tans_pop: the table tensor must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(cdfs_sizes), "tans_pop: cdfs_sizes must be a CUDA tensor");
  STD_TORCH_CHECK(ta::is_cuda(offsets), "tans_pop: offsets must be a CUDA tensor");
  STD_TORCH_CHECK(tables.dim() == 2,
                  "tans_pop: the table tensor must be 2-D (one row per distribution)");
  STD_TORCH_CHECK(tables.scalar_type() == ta::ScalarType::Int,
                  "tans_pop: the table tensor must be int32 (as tans_build_tables "
                  "returns it)");
  STD_TORCH_CHECK(cdfs_sizes.size(0) == tables.size(0) && offsets.size(0) == tables.size(0),
                  "tans_pop: cdfs_sizes and offsets must have one entry per table row");

  ta::Tensor symbols;

  // See tans_push_indexed_cuda: the launch target has to be the current device.
  const ta::DeviceGuard device_guard(ta::device_index(stream));

  TA_DISPATCH_INTEGRAL(indexes.scalar_type(), "tans_pop_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    const TansTableStaging staging = tans_table_staging(tables);
    const int num_threads_per_block = tans_num_threads_per_block(batch_size);
    const int num_blocks = (int)((batch_size + num_threads_per_block - 1) /
                                 num_threads_per_block);
    const cudaStream_t cuda_stream = ta::current_stream(stream);

    symbols = ta::zeros_like(indexes);

    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const int64_t num_dists = tables.size(0);

    if (staging.words > 0) {
      tans_pop_indexed_cuda_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, true, NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, staging.bytes, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              staging.words);
    } else {
      tans_pop_indexed_cuda_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, false, NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              0);
    }
    TA_CUDA_CHECK(cudaGetLastError());
  });

  return symbols;
}

// ---------------------------------------------------------------------------
// Instantiations required by the bindings in rans_bindings.hpp.
// ---------------------------------------------------------------------------
// The wrappers in tans.hpp only see the declarations above, so every bound
// family needs an explicit instantiation definition here - the same reason as
// the rANS blocks above and the tANS CPU blocks in tans_cpu.cpp (see
// rans_build_config.hpp). Only the single-lane variant is instantiated: the
// interleaved tANS kernels do not exist yet and `tans_push` / `tans_pop` reject
// them with an explanation before reaching a CUDA launcher (the `if constexpr`
// there is what keeps the ILV > 1 specialisations from being referenced).
#define TORCH_ANS_TANS_CUDA_INST_FAMILY(STATE, STREAM) \
  TORCH_ANS_INST_TANS_PUSH_CUDA(STATE, STREAM, 1); \
  TORCH_ANS_INST_TANS_POP_CUDA(STATE, STREAM, 1)

#if TORCH_ANS_WITH_TANS32
TORCH_ANS_TANS_CUDA_INST_FAMILY(uint32_t, uint16_t);
#endif

#if TORCH_ANS_WITH_TANS16
TORCH_ANS_TANS_CUDA_INST_FAMILY(uint16_t, uint8_t);
#endif

#if TORCH_ANS_WITH_TANS8
TORCH_ANS_TANS_CUDA_INST_FAMILY(uint8_t, uint8_t);
#endif

#undef TORCH_ANS_TANS_CUDA_INST_FAMILY
