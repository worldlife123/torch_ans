#pragma once

// Warp-level interleaved tANS kernels (CUDA/HIP device code), plus the launch
// and table-staging plumbing both tANS kernel families share.
//
// Technique reference: rans_warp_cuda.cuh (warp-cooperative interleaved rANS,
// itself modelled on 3rdparty/recoil). One group of NUM_INTERLEAVES lanes codes
// one batch row cooperatively: every lane owns one of the row's tANS states
// (they already live in the stream header, `tans_state_slot(lane)`) while the
// *bit layer* stays shared - which is the one structural difference to rANS,
// where each lane owns its own words.
//
// Why the bit layer cannot be split: a tANS state is fixed width (the encoder's
// lives in [2**L, 2**(L+1)), see tans_utils.hpp), so the bits it does not emit
// have to go somewhere else. They go into one bit buffer that the lanes share,
// exactly like an interleaved rANS stream shares its cursor - so the *bit* order
// is a single sequence and this kernel mirrors tans_cpu.cpp's interleaved path
// bit for bit:
//   * symbol s is coded on lane s % NUM_INTERLEAVES;
//   * the payload is the concatenation of the per-lane fields over the symbols
//     in *reverse* symbol order, and inside a symbol group the lanes are visited
//     from N-1 down to 0 (push), each contributing its bypass digits first and
//     then the low `nbBits` bits of its state;
//   * pop is the mirror: groups forward, lanes 0..N-1, each taking its own
//     `nbBits` bits from the front of the shared window;
//   * the leftover head (push: the top `num_symbols % N` symbols) and tail
//     (pop: the last ones) are coded one symbol at a time on lane `i % N`,
//     exactly as the CPU path does - the lane of a symbol is its index % N, not
//     "the lane the group walks to first", and the two differ for a partial
//     group (the per-lane state histories, and with them the bytes, depend on
//     the assignment).
// This is what `tests/test_tans.py` pins by comparing CPU and CUDA byte strings.
//
// Consequences for the implementation:
//   * every lane holds its own copy of the writer/reader (`buf`, `nbits`, and
//     the payload cursor) and all of them apply the *same* update - the field
//     to append is broadcast from its owner with __shfl_sync, so the arithmetic
//     is identical in every lane and no lane ever diverges inside the serial
//     section (the control flow there depends only on broadcast values);
//   * only one lane per group stores a payload word (`if (lane == 0)` inside
//     tans_warp_append): the word would otherwise be written once per lane -
//     the same value, but a race;
//   * the *parallel* half is the per-lane table work: the symbol lookup
//     (encode: symbolTT + nextState; decode: the two decoder planes) is issued
//     for all N lanes at once, which is the memory-level parallelism and, for
//     consecutive symbol indices, the coalescing that the one-thread-per-row
//     kernels cannot have (they read a row's symbols with a row-sized stride).
//
// The kernels are templated on <TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t,
// USE_SHARED_TABLE, NUM_INTERLEAVES> so that the per-lane states stay
// compile-time indexed (registers) and the shared-table decision costs nothing
// when unused, like the single-lane kernels in tans_cuda.cu.
//
// Kernels and launchers go through the ta:: compatibility layer, so this header
// compiles in both build modes (ATen / libtorch stable ABI) - it is included by
// tans_cuda.cu, which defines RANS_CUDA_API before including tans_utils.hpp.

#include "ta/torch_api.hpp"

#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "rans_utils.hpp"  // constexpr_for: the compile-time lane loop
#include "tans_utils.hpp"

namespace tans_warp {

// ---------------------------------------------------------------------------
// launch configuration / table staging
// ---------------------------------------------------------------------------
// This block used to live in tans_cuda.cu's anonymous namespace. It is here
// because the warp kernels need the very same staging helper, and a function
// defined *after* an include is not visible to the templates it defines (a
// dependent call to a name in an anonymous namespace is not found by ADL for
// fundamental-type arguments). One copy serves both kernel families.

//: Number of blocks a launch aims at (see tans_num_threads_per_block).
#ifndef TANS_BLOCKS_TARGET
#define TANS_BLOCKS_TARGET 128
#endif

//: Upper bound of the derived block size (a batch large enough to fill the
//: device wants a small number of wide blocks, not 1024-thread ones).
#ifndef TANS_MAX_THREADS_PER_BLOCK
#define TANS_MAX_THREADS_PER_BLOCK 128
#endif

//: Default block size of the warp-level interleaved kernels (N lanes code one
//: row, so the block size also decides how many rows share a block).
#ifndef TANS_WARP_BLOCK_THREADS_DEFAULT
#define TANS_WARP_BLOCK_THREADS_DEFAULT 256
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

inline int tans_num_threads_per_block(int64_t batch_size) {
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

// Block size of the warp-level interleaved kernels. Same "aim at
// TANS_BLOCKS_TARGET blocks" idea as tans_num_threads_per_block, with the batch
// counted in *lane groups* instead of rows: a group codes one row with
// num_interleaves threads, so spreading `batch_size` rows over the target number
// of blocks means `ceil(batch / target) * num_interleaves` threads - clamped to
// [num_interleaves, TANS_WARP_BLOCK_THREADS_DEFAULT] and rounded down to a
// multiple of the group size (a partial group would have idle lanes and, worse,
// a block-local group_mask that does not cover it).
//
// Why not the fixed 256 of the rANS warp kernels: with 256 threads a small batch
// becomes one or two blocks, i.e. one or two SMs of a device with far more. The
// rANS kernels never see that because their 32-way groups are one *warp* each
// and a warp is the smallest launchable unit anyway; a tANS group can be as
// small as two threads, so the derived size is worth the extra arithmetic.
// `TORCH_ANS_TANS_WARP_BLOCK_THREADS` overrides it, so a sweep needs no rebuild.
inline int tans_warp_num_threads_per_block(int64_t batch_size, int64_t num_interleaves) {
  const int64_t groups = (num_interleaves > 0) ? num_interleaves : 1;
  const char* env = std::getenv("TORCH_ANS_TANS_WARP_BLOCK_THREADS");
  if (env != nullptr) {
    const long value = std::atol(env);
    if (value > 0 && value <= 1024) {
      const int threads = (int)(value / groups) * (int)groups;
      if (threads >= (int)groups) return threads;
    }
  }
  const int64_t rows_per_block = (batch_size + TANS_BLOCKS_TARGET - 1) / TANS_BLOCKS_TARGET;
  int64_t threads = rows_per_block * groups;
  if (threads < groups) threads = groups;
  if (threads > TANS_WARP_BLOCK_THREADS_DEFAULT) threads = TANS_WARP_BLOCK_THREADS_DEFAULT;
  threads = (threads / groups) * groups;
  if (threads < groups) threads = groups;
  return (int)threads;
}

inline int64_t tans_smem_budget_bytes() {
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
// to nothing. Both kernel families call it *before* their `b >= size0` early
// return, because the barrier inside has to be reached by every thread of the
// block.
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

// ---------------------------------------------------------------------------
// the shared bit layer
// ---------------------------------------------------------------------------
// Mirrors TansBitWriter (tans_utils.hpp) with two differences: one lane of the
// group stores the payload words, and the window is 32 bits instead of 64. Only
// the lane difference is about correctness - without the guard each flushed word
// would be written once per lane, the same value but a race - while the narrower
// window is about cost, see below.
//
// Why the *shared* window narrows to 32 bits: at rest the writer holds fewer
// than STREAM_BITS bits (append() drains while nbits >= STREAM_BITS, and the
// header carries one such residual), and one digit is at most
// max(bypass_precision, freq_precision) <= TANS_MAX_TABLELOG == TANS_MAX == 15
// bits wide, so the register never holds more than (STREAM_BITS - 1) + 15 bits:
// 30 for a 16-bit stream word, 22 for an 8-bit one. Every lane of the group runs
// these updates NUM_INTERLEAVES times per symbol group (appending the bits of all
// the lanes keeps the group converged), so a 64-bit shift pair per update is a
// handful of instructions times that count - hence the specialised copy here
// rather than the shared TansBitWriter. tans_utils.hpp keeps its wider type for
// the CPU and the single-lane kernels, where the cost is paid once per symbol.
template <typename TANS_STREAM_TYPE>
struct TansWarpBitWriter
{
  uint32_t buf;
  uint32_t nbits;
  TANS_STREAM_TYPE* ptr;  // last committed word; the next one is *(ptr + 1)

  static constexpr uint32_t STREAM_BITS = sizeof(TANS_STREAM_TYPE) * 8;
};

template <typename TANS_STREAM_TYPE>
__device__ __forceinline__ void tans_warp_append(TansWarpBitWriter<TANS_STREAM_TYPE>& writer,
                                                 uint32_t value, uint32_t k, bool store) {
  writer.buf |= (value & ((k >= 32) ? 0xFFFFFFFFu : ((1u << k) - 1))) << writer.nbits;
  writer.nbits += k;
  while (writer.nbits >= TansWarpBitWriter<TANS_STREAM_TYPE>::STREAM_BITS) {
    TANS_STREAM_TYPE* const next = writer.ptr + 1;
    if (store) *next = (TANS_STREAM_TYPE)writer.buf;
    writer.ptr = next;
    writer.buf >>= TansWarpBitWriter<TANS_STREAM_TYPE>::STREAM_BITS;
    writer.nbits -= TansWarpBitWriter<TANS_STREAM_TYPE>::STREAM_BITS;
  }
}

// The reader's mirror, with the same 32-bit window, plus the random-access view
// the decode fast path reads through (TansWarpBitWindow below). The bit order is
// TansBitReader's exactly - the lanes share one window, so a value read here has
// to be the value the encoder appended there. `take` only refills while
// nbits < k <= 15, so the window stays below 15 + STREAM_BITS <= 31 bits.
template <typename TANS_STREAM_TYPE>
struct TansWarpBitReader
{
  uint32_t buf;
  uint32_t nbits;
  const TANS_STREAM_TYPE* ptr;  // last word still to consume

  static constexpr uint32_t STREAM_BITS = sizeof(TANS_STREAM_TYPE) * 8;

  __device__ __forceinline__ uint32_t take(uint32_t k) {
    while (nbits < k) {
      buf = (buf << STREAM_BITS) | (uint32_t)(*ptr--);
      nbits += STREAM_BITS;
    }
    nbits -= k;
    return (buf >> nbits) & ((1u << k) - 1);
  }
};

// The same takes against the warp kernel's narrow window: the body lives in
// tans_utils.hpp (tans_pop_raw_value_generic), because the digit sequence has to
// be the CPU/single-lane reader's bit for bit.
template <typename TANS_STREAM_TYPE>
__device__ __forceinline__ uint32_t tans_pop_raw_value(TansWarpBitReader<TANS_STREAM_TYPE>* br,
                                                       uint32_t bypass_precision) {
  return tans_pop_raw_value_generic(br, bypass_precision);
}

// ---------------------------------------------------------------------------
// random access to the shared bit window (the decode fast path)
// ---------------------------------------------------------------------------
// The same state TansWarpBitReader holds, spelled so that a lane can read *its
// own* bits without walking the lanes before it:
//   * `res` holds the last `res_bits` bits of the appended sequence (the not yet
//     consumed tail, < STREAM_BITS of them, exactly TansBitReader's `buf`/`nbits`
//     pair);
//   * `cursor` is the index of the last word still to consume, counted from
//     `base` - which is what TansBitReader's `ptr` names.
// A read addresses the sequence by *consumption offset* o: o == 0 is the bit the
// next take() would return, i.e. the newest appended bit. The first res_bits
// offsets live in `res` (top bit first: the register's bit res_bits-1 is offset
// 0), everything below comes out of the payload words going *down* from `cursor`,
// one whole word per STREAM_BITS offsets, high bit first - which is just what
// take()'s refill does, written as arithmetic instead of as a step.
//
// That is the whole trick of the decode fast path: the bits a lane needs are a
// function of the prefix sum of the lanes' nbBits, so every lane can load them
// itself, in parallel, instead of every lane replaying the whole group's takes.
// (Push cannot do the same: there the bits have to be *written*, and a stream word
// mixes the bits of several lanes - no lane owns a word the way it owns a read.)
template <typename TANS_STREAM_TYPE>
struct TansWarpBitWindow
{
  const TANS_STREAM_TYPE* base;
  int64_t cursor;
  uint32_t res;
  uint32_t res_bits;

  static constexpr uint32_t STREAM_BITS = sizeof(TANS_STREAM_TYPE) * 8;
  static constexpr uint32_t LOG_STREAM_BITS = (STREAM_BITS == 16u) ? 4u : 3u;

  // The `nb` bits starting at consumption offset `o`, most significant first.
  __device__ __forceinline__ uint32_t read(uint32_t o, uint32_t nb) const {
    uint32_t value = 0;
    uint32_t left = nb;
    uint32_t at = o;
    if (left > 0u && at < res_bits) {
      const uint32_t take = (left < res_bits - at) ? left : (res_bits - at);
      value = (res >> (res_bits - at - take)) & ((1u << take) - 1u);
      left -= take;
      at += take;
    }
    while (left > 0u) {
      // Bits below the residual: `u` is the offset into the refilled words, whose
      // number is u / STREAM_BITS and whose bit (counted from the word's top) is
      // u % STREAM_BITS. A lane reads at most ceil(15 / STREAM_BITS) + 1 of them,
      // i.e. two words for tans32 and three for tans16/tans8.
      const uint32_t u = at - res_bits;
      const int64_t word_index = cursor - (int64_t)(u >> LOG_STREAM_BITS);
      const uint32_t t = u & (STREAM_BITS - 1u);
      const uint32_t take = (left < STREAM_BITS - t) ? left : (STREAM_BITS - t);
      const uint32_t chunk =
          ((uint32_t)base[word_index] >> (STREAM_BITS - t - take)) & ((1u << take) - 1u);
      value = (value << take) | chunk;
      left -= take;
      at += take;
    }
    return value;
  }

  // Consume `consumed` bits exactly the way the takes of a serial walk would:
  // the window is refilled in whole words and only while it is short of the bits
  // asked for, so after the walk it holds the smallest (res_bits + k*STREAM_BITS)
  // bits that cover what was consumed. Uniform across the group - `consumed` is
  // the group's total - so every lane ends up with the identical window.
  __device__ __forceinline__ void advance(uint32_t consumed) {
    if (consumed == 0u) return;
    const uint32_t refilled =
        (consumed > res_bits)
            ? ((consumed - res_bits + STREAM_BITS - 1u) >> LOG_STREAM_BITS)
            : 0u;
    const uint32_t held = res_bits + refilled * STREAM_BITS;  // bits loaded
    const uint32_t next_bits = held - consumed;               // < STREAM_BITS
    const uint32_t next_res = read(consumed, next_bits);
    cursor -= (int64_t)refilled;
    res = next_res;
    res_bits = next_bits;
  }

  // Back to the sequential reader, for the paths that still have to walk (the
  // bypass runs, whose length is only discoverable while decoding).
  __device__ __forceinline__ TansWarpBitReader<TANS_STREAM_TYPE> as_reader() const {
    return TansWarpBitReader<TANS_STREAM_TYPE>{res, res_bits, base + cursor};
  }

  __device__ __forceinline__ void take_over(TansWarpBitReader<TANS_STREAM_TYPE>& r) {
    res = r.buf & ((r.nbits >= 32u) ? 0xFFFFFFFFu : ((1u << r.nbits) - 1u));
    res_bits = r.nbits;
    cursor = r.ptr - base;
  }
};

// ---------------------------------------------------------------------------
// one broadcast per lane instead of two
// ---------------------------------------------------------------------------
// The serial sections below need three things from every lane of the group: its
// nbBits, whether its symbol is valid, and the bits it contributes. Those fit in
// one 32-bit word, so one __shfl_sync per lane replaces the two (push: nbBits +
// state; pop: nothing at all today, but see below) the straight implementation
// needs - and SHFL issues at a quarter of the rate of a plain ALU op:
//   * bits 0..14  the low 15 bits of the lane's encoder state. The append masks
//     the value down to nb_bits bits and nb_bits <= 15, so nothing above bit 14
//     can reach the stream;
//   * bits 16..19 nb_bits itself;
//   * bit  20     the lane's index is valid (an invalid lane contributes
//     nothing, and packing the flag next to nb_bits is what lets one word carry
//     "how much" and "whether" at once - the old `nb_bits | 0x100` did the same
//     with a byte's worth of headroom);
//   * bit  21     the lane bypassed: its raw digits precede its state bits.
// A bypassing lane additionally publishes raw_val / the three digit counts, but
// only when the group's ballot says some lane actually bypassed.
constexpr uint32_t TANS_WARP_STATE_MASK = 0x7FFFu;
constexpr uint32_t TANS_WARP_NB_SHIFT = 16;
constexpr uint32_t TANS_WARP_NB_MASK = 0x0Fu;
constexpr uint32_t TANS_WARP_VALID = 1u << 20;
constexpr uint32_t TANS_WARP_BYPASS = 1u << 21;

__device__ __forceinline__ uint32_t tans_warp_pack_lane(uint32_t state, uint32_t nb_bits,
                                                        bool valid, bool bypassed) {
  return (state & TANS_WARP_STATE_MASK) | ((nb_bits & TANS_WARP_NB_MASK) << TANS_WARP_NB_SHIFT) |
         (valid ? TANS_WARP_VALID : 0u) | (bypassed ? TANS_WARP_BYPASS : 0u);
}

// Whether the serial section is emitted as a rolled loop rather than the fully
// unrolled constexpr_for. Measured on RTX 2080 Ti (tans32, `cuobjdump -res-usage`
// of this kernel): 8 lanes cost 80-86 registers, 32 lanes cost 229-232, i.e. one
// 256-thread block per SM instead of four (25% instead of 100% occupancy) and
// 14800 SASS instructions of body. The rolled version pays a loop counter per
// lane but keeps the register budget - and therefore the occupancy - of the small
// lane counts, which is what a warp this dependent on memory latency wants.
template <size_t NUM_INTERLEAVES>
constexpr bool tans_warp_roll_serial() {
  return NUM_INTERLEAVES >= 16;
}

// Whether the decode fast path (TansWarpBitWindow) replaces the lane walk.
// Measured on RTX 2080 Ti (tans32, alphabet 64, freq_precision 7, bypass on,
// decode us in `benchmarks/tans_interleaved_cuda_optimized.json`, old -> new):
//   8 rows:   4 lanes 101189 -> 87876,  8 lanes 103180 -> 53811, 32 lanes
//             203334 -> 17499 (the walk's dependent chain is the whole cost there)
//   4096 rows: 4 lanes 3547 -> 3390,  8 lanes 3593 -> 3388, 32 lanes 6064 -> 3430
// At 2 lanes it is a wash to a small loss (8 rows: 120589 -> 140402; 4096 rows:
// 3596 -> 3565): there is no chain worth breaking in a two-step walk, only the
// extra shuffles of the prefix sum and the window's own read. So the walk stays
// for 2 lanes and the window takes over from 4.
template <size_t NUM_INTERLEAVES>
constexpr bool tans_warp_parallel_read() {
  return NUM_INTERLEAVES >= 4;
}

// The digit decomposition of tans_push_raw_value (tans_utils.hpp), with the
// appends left out: n_bypass digits of bypass_precision bits carry the raw
// value, `rest` digits carry the digit count (one max_bypass_val per carry),
// and the total is what the group's serial section has to append. A device-side
// "simulate" helper in the spirit of rans_warp_simulate_raw_push: the count is
// needed *before* the sequential appends, because the trip count of that loop
// has to be the same in every lane of the group.
//
// NOTE: the shift guard is rans_utils.hpp::rans_push_raw_value_step's - the
// count is capped so that `n_bypass * bypass_precision` never reaches 32, which
// would be a shift of 32 (undefined behaviour, and the CPU and the GPU resolve
// it differently: wrap-around vs clamp).
__device__ __forceinline__ void tans_warp_raw_digit_count(uint32_t raw_val,
                                                          uint32_t bypass_precision,
                                                          uint32_t* n_bypass_out,
                                                          uint32_t* rest_out,
                                                          uint32_t* n_carries_out) {
  const uint32_t max_bypass_val = (1u << bypass_precision) - 1u;
  uint32_t n_bypass = 0;
  while (n_bypass * bypass_precision < 32 &&
         (raw_val >> (n_bypass * bypass_precision)) != 0) {
    ++n_bypass;
  }
  uint32_t val = n_bypass;
  uint32_t n_carries = 0;
  while (val >= max_bypass_val) {
    ++n_carries;
    val -= max_bypass_val;
  }
  *n_bypass_out = n_bypass;
  *rest_out = val;
  *n_carries_out = n_carries;
}

// The bypass digits of one lane, in tans_push_raw_value's order (high digit
// first, then the remainder, then one max_bypass_val per carry) - the order the
// decoder meets them in, mirrored. `n_bypass`/`rest`/`n_carries` come from
// tans_warp_raw_digit_count; the caller passes the *broadcasting* lane's values,
// so every lane walks the same `nd` iterations.
template <typename TANS_STREAM_TYPE>
__device__ __forceinline__ void tans_warp_append_raw_digits(
    TansWarpBitWriter<TANS_STREAM_TYPE>& writer, uint32_t raw_val, uint32_t n_bypass,
                                                            uint32_t rest, uint32_t n_carries,
                                                            uint32_t bypass_precision,
                                                            bool store) {
  const uint32_t max_bypass_val = (1u << bypass_precision) - 1u;
  const uint32_t num_digits = n_bypass + 1u + n_carries;
  for (uint32_t d = 0; d < num_digits; ++d) {
    const uint32_t digit = (d < n_bypass)
        ? ((raw_val >> ((n_bypass - 1u - d) * bypass_precision)) & max_bypass_val)
        : ((d == n_bypass) ? rest : max_bypass_val);
    tans_warp_append<TANS_STREAM_TYPE>(writer, digit, bypass_precision, store);
  }
}

// ---------------------------------------------------------------------------
// push (encode) kernel: one lane group per batch row, reverse symbol coding
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_SHARED_TABLE = false, size_t NUM_INTERLEAVES = 1>
__global__ void tans_warp_push_indexed_kernel(
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
    const int64_t smem_table_words) {
  static_assert(NUM_INTERLEAVES >= 1 && NUM_INTERLEAVES <= 32 &&
                    (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "NUM_INTERLEAVES must be a power of two no greater than the warp size");
  constexpr unsigned N = (unsigned)NUM_INTERLEAVES;
  constexpr int64_t ilv = (int64_t)NUM_INTERLEAVES;

  extern __shared__ char tans_smem_raw[];
  const int32_t* const tables = tans_stage_table<USE_SHARED_TABLE>(
      tables_data, reinterpret_cast<int32_t*>(tans_smem_raw), smem_table_words);

  // Group identity, the same arithmetic as rans_warp_cuda.cuh: a group does not
  // have to start at lane 0 of the warp, so the masks are group-local - a group
  // that exits early (its row is past the end of the batch) never breaks another
  // group's collectives.
  const unsigned groups_per_block = blockDim.x / N;
  const unsigned group_in_block = threadIdx.x / N;
  const unsigned lane = threadIdx.x % N;
  const unsigned lane_in_warp = threadIdx.x & 31u;
  const unsigned group_start_in_warp = lane_in_warp - lane;
  const unsigned group_mask = (N == 32u) ? 0xffffffffu : (((1u << N) - 1u) << group_start_in_warp);

  const int64_t b = (int64_t)blockIdx.x * (int64_t)groups_per_block + (int64_t)group_in_block;
  if (b >= stream_view.size0) return;  // uniform per group

  const uint32_t ts = (uint32_t)1 << (uint32_t)freq_precision;
  const uint32_t bpp = (uint32_t)bypass_precision;
  const bool do_bypass = bypass_coding;
  const int64_t num_symbols = symbols_view.size1;

  int32_t* const header = stream_view.row(b);
  const int64_t length = (int64_t)header[0];
  TANS_STREAM_TYPE* const base = reinterpret_cast<TANS_STREAM_TYPE*>(header + 1);
  // The bit buffer travels in the header, so the writer is seeded from it; every
  // lane of the group gets the identical writer and keeps it that way.
  TansWarpBitWriter<TANS_STREAM_TYPE> writer{
      (uint32_t)header[tans_bitbuf_slot(ilv)],
      (uint32_t)header[tans_nbits_slot(ilv)],
      base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};

  const TORCH_TENSOR_TYPE* const symbols_row = symbols_view.row(b);
  const TORCH_TENSOR_TYPE* const indexes_row = indexes_view.row(b);

  // 0 is not a legal encoder state (those live in [2**L, 2**(L+1))), so it marks
  // a fresh stream - tans_init_stream does not know table_log (tans_utils.hpp).
  uint32_t state = (uint32_t)header[tans_state_slot((int64_t)lane)];
  if (state == 0) state = ts;
  const bool store_lane = (lane == 0u);

  // --- head: the top (num_symbols % N) symbols, one at a time, lane k % N ---
  // Only the owning lane reads the inputs: the group does not need the symbol,
  // the index or the table row - it needs the digit description, and that travels
  // as one broadcast word (tans_warp_pack_lane). Having every lane load them
  // anyway costs N times the memory traffic for one lane's worth of result, and
  // with up to N-1 head symbols per row that is comparable to the whole row for
  // the workloads where head is not empty.
  //
  // Every lane still issues the *appends*: the bit buffer is shared, so the
  // sequence it sees has to be identical in all of them, and only the owning lane
  // advances its state afterwards.
  const int64_t first_interleave = num_symbols % ilv;
  for (int64_t k = num_symbols - 1; k >= num_symbols - first_interleave; --k) {
    const unsigned active = (unsigned)(k % ilv);
    const unsigned src = group_start_in_warp + active;
    uint32_t packed = 0;
    uint32_t owner_raw = 0;
    uint32_t owner_next = 0;
    if (lane == active) {
      const int64_t index = (int64_t)indexes_row[k];
      if (index >= 0 && index < num_dists) {
        const int32_t* const row = tables + index * tables_row_stride;
        const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
        const TORCH_TENSOR_TYPE offset = offsets_view.at(index);
        uint32_t value = 0;
        bool bypassed = false;
        tans_cook_symbol<TORCH_TENSOR_TYPE>(symbols_row[k], cdf_size, offset, do_bypass,
                                            &value, &owner_raw, &bypassed);
        const int32_t* const symbol_tt = tans_symbol_tt_row(row, ts, value);
        const uint32_t nb_bits = tans_encode_nb_bits(state, symbol_tt);
        owner_next =
            tans_encode_next_state(row, tans_encode_next_index(state, symbol_tt, nb_bits));
        packed = tans_warp_pack_lane(state, nb_bits, true, bypassed);
      }
    }
    packed = __shfl_sync(group_mask, packed, src);
    if ((packed & TANS_WARP_VALID) == 0u) continue;  // uniform: the owner's answer
    if (do_bypass && (packed & TANS_WARP_BYPASS) != 0u) {
      owner_raw = __shfl_sync(group_mask, owner_raw, src);
      uint32_t n_bypass = 0, rest = 0, n_carries = 0;
      tans_warp_raw_digit_count(owner_raw, bpp, &n_bypass, &rest, &n_carries);
      // The bypass digits go in before the sentinel symbol (tans_utils.hpp).
      tans_warp_append_raw_digits<TANS_STREAM_TYPE>(writer, owner_raw, n_bypass, rest,
                                                    n_carries, bpp, store_lane);
    }
    tans_warp_append<TANS_STREAM_TYPE>(
        writer, packed & TANS_WARP_STATE_MASK,
        (packed >> TANS_WARP_NB_SHIFT) & TANS_WARP_NB_MASK, store_lane);
    if (lane == active) state = owner_next;
  }

  // --- main loop: symbol groups high to low, lane L codes symbol i-(N-1-L) ---
  const int64_t i0 = num_symbols - first_interleave - 1;
  for (int64_t i = i0; i >= 0; i -= ilv) {
    const int64_t s = i - (int64_t)(N - 1u - lane);
    const int64_t index = (int64_t)indexes_row[s];
    const bool valid = (index >= 0 && index < num_dists);

    // Pass 1 (parallel): what this lane's symbol needs. The table row is a
    // pointer here; the two symbolTT words are loaded in pass 2 together with
    // every other lane's, and the bypass digits are only decomposed (no writer
    // involved) so that their count can be broadcast later.
    const int32_t* row = nullptr;
    uint32_t value = 0, raw_val = 0;
    uint32_t n_bypass = 0, rest = 0, n_carries = 0;
    bool bypassed = false;
    if (valid) {
      row = tables + index * tables_row_stride;
      const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
      const TORCH_TENSOR_TYPE offset = offsets_view.at(index);
      tans_cook_symbol<TORCH_TENSOR_TYPE>(symbols_row[s], cdf_size, offset, do_bypass,
                                          &value, &raw_val, &bypassed);
      if (bypassed) tans_warp_raw_digit_count(raw_val, bpp, &n_bypass, &rest, &n_carries);
    }

    // Pass 2 (parallel): nbBits and the next-state index. One dependent load
    // pair per lane, all of them in flight at the same time - with the symbol
    // lookup the main memory-latency source of the encoder.
    uint32_t nb_bits = 0;
    int64_t next_index = 0;
    if (valid) {
      const int32_t* const symbol_tt = tans_symbol_tt_row(row, ts, value);
      nb_bits = tans_encode_nb_bits(state, symbol_tt);
      next_index = tans_encode_next_index(state, symbol_tt, nb_bits);
    }

    // Pass 3 (serial, lane N-1 -> 0): the bits. The bit buffer is shared, so this
    // is the only part that has to walk the lanes in order, and every lane runs
    // it - which is why it costs one broadcast per lane instead of two separate
    // ones (tans_warp_pack_lane carries nbBits, the contribution and the validity
    // in a single word). The bypass digits of a lane are only asked for when the
    // group's ballot says some lane actually has them (A5, rans_warp_cuda.cuh).
    const unsigned packed_self = tans_warp_pack_lane(state, nb_bits, valid, bypassed);
    const unsigned bypass_lanes = do_bypass ? __ballot_sync(group_mask, bypassed) : 0u;
    auto serial_lane = [&](const unsigned l) {
      const unsigned src = group_start_in_warp + l;
      const unsigned packed = __shfl_sync(group_mask, packed_self, src);
      if ((packed & TANS_WARP_VALID) == 0u) return;
      if (do_bypass && ((bypass_lanes >> l) & 1u) != 0u) {
        const uint32_t raw_b = __shfl_sync(group_mask, raw_val, src);
        const uint32_t nbp = __shfl_sync(group_mask, n_bypass, src);
        const uint32_t rst = __shfl_sync(group_mask, rest, src);
        const uint32_t ncar = __shfl_sync(group_mask, n_carries, src);
        tans_warp_append_raw_digits<TANS_STREAM_TYPE>(writer, raw_b, nbp, rst, ncar, bpp,
                                                      store_lane);
      }
      tans_warp_append<TANS_STREAM_TYPE>(writer, packed & TANS_WARP_STATE_MASK,
                                         (packed >> TANS_WARP_NB_SHIFT) & TANS_WARP_NB_MASK,
                                         store_lane);
    };
    if constexpr (tans_warp_roll_serial<NUM_INTERLEAVES>()) {
      for (unsigned j = 0; j < N; ++j) serial_lane(N - 1u - j);  // highest lane first
    } else {
      constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
        serial_lane((unsigned)((size_t)NUM_INTERLEAVES - 1u - (size_t)jc));
      });
    }

    // Pass 4 (parallel): the next state of every lane - one dependent load each,
    // all of them in flight at the same time.
    if (valid) state = tans_encode_next_state(row, next_index);
  }

  // Every lane writes its own state word (the lanes' slots are consecutive, so
  // the stores coalesce), and a lane the symbol count never reached keeps the
  // `ts` it was seeded with - exactly what the CPU kernel writes back for it.
  header[tans_state_slot((int64_t)lane)] = (int32_t)state;
  // At rest the bit buffer holds fewer than STREAM_BITS bits (append() drains
  // while nbits >= STREAM_BITS), so both fit in the header words.
  if (store_lane) {
    header[tans_bitbuf_slot(ilv)] = (int32_t)(uint32_t)writer.buf;
    header[tans_nbits_slot(ilv)] = (int32_t)writer.nbits;
    // The payload is whole stream words only: the residual bits stay in the
    // header, so the length is unambiguous and the reader needs no trailer.
    header[0] = (int32_t)((writer.ptr - base + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
  }
}

// ---------------------------------------------------------------------------
// pop (decode) kernel: one lane group per batch row, forward symbol coding
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_SHARED_TABLE = false, size_t NUM_INTERLEAVES = 1>
__global__ void tans_warp_pop_indexed_kernel(
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
    const int64_t smem_table_words) {
  static_assert(NUM_INTERLEAVES >= 1 && NUM_INTERLEAVES <= 32 &&
                    (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "NUM_INTERLEAVES must be a power of two no greater than the warp size");
  constexpr unsigned N = (unsigned)NUM_INTERLEAVES;
  constexpr int64_t ilv = (int64_t)NUM_INTERLEAVES;

  extern __shared__ char tans_smem_raw[];
  const int32_t* const tables = tans_stage_table<USE_SHARED_TABLE>(
      tables_data, reinterpret_cast<int32_t*>(tans_smem_raw), smem_table_words);

  const unsigned groups_per_block = blockDim.x / N;
  const unsigned group_in_block = threadIdx.x / N;
  const unsigned lane = threadIdx.x % N;
  const unsigned lane_in_warp = threadIdx.x & 31u;
  const unsigned group_start_in_warp = lane_in_warp - lane;
  const unsigned group_mask = (N == 32u) ? 0xffffffffu : (((1u << N) - 1u) << group_start_in_warp);

  const int64_t b = (int64_t)blockIdx.x * (int64_t)groups_per_block + (int64_t)group_in_block;
  if (b >= stream_view.size0) return;  // uniform per group

  const uint32_t ts = (uint32_t)1 << (uint32_t)freq_precision;
  const uint32_t bpp = (uint32_t)bypass_precision;
  const bool do_bypass = bypass_coding;
  const int64_t num_symbols = symbols_view.size1;

  // not const: the continuation point is written back below
  int32_t* const header = stream_view.row(b);
  const int64_t length = (int64_t)header[0];
  const TANS_STREAM_TYPE* const base = reinterpret_cast<const TANS_STREAM_TYPE*>(header + 1);
  // The reader walks the payload backwards from its end: the encoder wrote whole
  // words forward, the bits are consumed in reverse order. Only reads below, so
  // the redundant per-lane copy is harmless - and the takes stay in lockstep
  // because their counts are broadcast. The window is the same state spelled for
  // random access, which is what lets the fast path skip the walk.
  const uint32_t seed_nbits = (uint32_t)header[tans_nbits_slot(ilv)];
  TansWarpBitWindow<TANS_STREAM_TYPE> window{
      base, length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1,
      (uint32_t)header[tans_bitbuf_slot(ilv)] &
          ((seed_nbits >= 32u) ? 0xFFFFFFFFu : ((1u << seed_nbits) - 1u)),
      seed_nbits};

  // The two directions use different ranges for the same state: the encoder's
  // lives in [ts, 2*ts), the decoder's in [0, ts) - the conversion is the
  // subtraction (tans_cpu.cpp), and the header keeps the encoder's range.
  uint32_t state = (uint32_t)header[tans_state_slot((int64_t)lane)] - ts;

  TORCH_TENSOR_TYPE* const symbols_row = symbols_view.row(b);
  const TORCH_TENSOR_TYPE* const indexes_row = indexes_view.row(b);
  const bool store_lane = (lane == 0u);

  // --- main loop: full groups, low to high, lane L codes symbol i+L ---
  const int64_t last_interleave = num_symbols - (ilv - 1);
  int64_t i = 0;
  for (; i < last_interleave; i += ilv) {
    const int64_t s = i + (int64_t)lane;
    const int64_t index = (int64_t)indexes_row[s];
    const bool valid = (index >= 0 && index < num_dists);

    // Pass 1 (parallel): the per-distribution metadata and the two decoder
    // planes. Both planes are indexed by this lane's own state, so the lanes'
    // loads are independent of each other - the main memory-latency source of
    // the decoder, and the reason this pass exists.
    const int32_t* row = nullptr;
    TORCH_TENSOR_TYPE max_value = 0, offset = 0;
    uint32_t word = 0, out_sym = 0, nb_bits = 0;
    if (valid) {
      row = tables + index * tables_row_stride;
      const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
      max_value = cdf_size - 2;
      offset = offsets_view.at(index);
      word = tans_decode_word(row, ts, state);
      out_sym = tans_decode_symbol_of(row, ts, state);
      nb_bits = tans_decode_nb_bits(word);
    }

    // The decoded symbol leaves the sequential section entirely: every lane owns
    // slot i+lane, so one coalescing store here replaces N predicated single-lane
    // stores inside the walk below - the same split rANS makes (its phase A stores
    // every lane's symbol, phase B only overwrites the bypass sentinels). A lane
    // whose own symbol turns out to be a bypass sentinel overwrites the slot in
    // the walk, which is the only order the two writes can meet in.
    if (valid) symbols_row[s] = (TORCH_TENSOR_TYPE)out_sym + offset;

    // A5 fast path: one ballot tells the group whether any lane has to read
    // bypass digits.
    const unsigned sentinel_lanes =
        do_bypass ? __ballot_sync(group_mask, valid && (out_sym == (uint32_t)max_value)) : 0u;

    uint32_t low_bits = 0;
    // The walk, shared by every path that cannot be turned into a prefix sum (see
    // tans_warp_parallel_read for which those are). The digits of a bypassing lane
    // follow its sentinel symbol and their count is only discoverable while
    // decoding, so the offsets of the lanes behind it are not known up front - the
    // same reason rANS' phase B is a loop. Every lane performs the identical walk
    // on its own copy of the reader; tans_warp_pack_lane carries the lane's bit
    // count and its validity in one broadcast word (nbBits <= 15).
    auto run_walk = [&]() {
      TansWarpBitReader<TANS_STREAM_TYPE> walk = window.as_reader();
      const unsigned packed_nb = tans_warp_pack_lane(0, nb_bits, valid, false);
      auto serial_lane = [&](const unsigned l) {
        const unsigned src = group_start_in_warp + l;
        const unsigned packed = __shfl_sync(group_mask, packed_nb, src);
        const uint32_t low = walk.take((packed >> TANS_WARP_NB_SHIFT) & TANS_WARP_NB_MASK);
        if (((sentinel_lanes >> l) & 1u) != 0u) {
          const uint32_t raw_val = tans_pop_raw_value<TANS_STREAM_TYPE>(&walk, bpp);
          if (lane == l) {
            symbols_row[i + (int64_t)l] =
                (TORCH_TENSOR_TYPE)tans_uncook_raw_value(raw_val, (int64_t)max_value) + offset;
          }
        }
        if (lane == l) low_bits = low;  // take(0) for an invalid lane consumed nothing
      };
      if constexpr (tans_warp_roll_serial<NUM_INTERLEAVES>()) {
        for (unsigned l = 0; l < N; ++l) serial_lane(l);  // lane-ascending: the mirror
      } else {
        constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>(
            [&](auto jc) { serial_lane((unsigned)jc); });
      }
      window.take_over(walk);
    };

    if constexpr (!tans_warp_parallel_read<NUM_INTERLEAVES>()) {
      run_walk();
    } else if (sentinel_lanes == 0u) {
      // --- pass 2a (parallel): every lane reads its own bits ---
      // No lane needs a bypass run, so the number of bits each lane consumes is
      // known before any of them is read - which makes the group's ordering a
      // prefix sum instead of a replay: log2(N) shuffles give every lane the
      // offset its bits start at, and TansWarpBitWindow turns that offset into a
      // load. The lanes' loads overlap (neighbouring lanes share a stream word),
      // so they cost one or two L1 hits each instead of N dependent takes.
      //
      // This is the counterpart of rANS' ballot+popc ordering (rans_warp_cuda.cuh
      // phase A): there a lane's contribution is a whole word and the ballot alone
      // places it; here it is a bit range and the prefix sum places it.
      unsigned acc = nb_bits;
      #pragma unroll
      for (unsigned off = 1; off < N; off <<= 1) {
        const unsigned up = __shfl_up_sync(group_mask, acc, off);
        if (lane >= off) acc += up;
      }
      const unsigned total_bits = __shfl_sync(group_mask, acc, group_start_in_warp + (N - 1u));
      const unsigned before = acc - nb_bits;  // exclusive prefix: bits read earlier
      if (valid) low_bits = window.read(before, nb_bits);
      window.advance(total_bits);  // uniform: every lane ends up with the same window
    } else {
      run_walk();
    }
    if (valid) state = tans_decode_new_state(word, low_bits);
  }

  // --- tail: the last (num_symbols % N) symbols, one at a time, lane i % N ---
  // See the head of the push kernel: only the owning lane reads the inputs, the
  // group gets one broadcast word carrying nbBits + the flags, and every lane
  // still issues the takes because the window is shared.
  for (; i < num_symbols; ++i) {
    const unsigned active = (unsigned)(i % ilv);
    const unsigned src = group_start_in_warp + active;
    uint32_t packed = 0;
    uint32_t word = 0, out_sym = 0;  // symbols stay unsigned: see pass 1 above
    TORCH_TENSOR_TYPE max_value = 0, offset = 0;
    if (lane == active) {
      const int64_t index = (int64_t)indexes_row[i];
      if (index >= 0 && index < num_dists) {
        const int32_t* const row = tables + index * tables_row_stride;
        const TORCH_TENSOR_TYPE cdf_size = cdfs_sizes_view.at(index);
        max_value = cdf_size - 2;
        offset = offsets_view.at(index);
        word = tans_decode_word(row, ts, state);
        out_sym = tans_decode_symbol_of(row, ts, state);
        packed = tans_warp_pack_lane(0, tans_decode_nb_bits(word), true,
                                     do_bypass && out_sym == (uint32_t)max_value);
      }
    }
    packed = __shfl_sync(group_mask, packed, src);
    if ((packed & TANS_WARP_VALID) == 0u) continue;  // uniform: the owner's answer
    const uint32_t nb = (packed >> TANS_WARP_NB_SHIFT) & TANS_WARP_NB_MASK;
    uint32_t low = 0;
    if constexpr (tans_warp_parallel_read<NUM_INTERLEAVES>()) {
      if ((packed & TANS_WARP_BYPASS) == 0u) {
        // One symbol, one lane: its bits start at offset 0 of the window, so the
        // read needs no prefix sum at all.
        low = window.read(0u, nb);
        window.advance(nb);
        if (lane == active) {
          state = tans_decode_new_state(word, low);
          symbols_row[i] = (TORCH_TENSOR_TYPE)out_sym + offset;
        }
        continue;
      }
    }
    // The bypass run's length is only discoverable while decoding (and at 2 lanes
    // the walk is the faster spelling anyway), so this one stays sequential.
    TansWarpBitReader<TANS_STREAM_TYPE> walk = window.as_reader();
    low = walk.take(nb);
    const uint32_t raw_val = ((packed & TANS_WARP_BYPASS) != 0u)
                                 ? tans_pop_raw_value<TANS_STREAM_TYPE>(&walk, bpp)
                                 : 0u;
    window.take_over(walk);
    if (lane == active) {
      state = tans_decode_new_state(word, low);
      symbols_row[i] = ((packed & TANS_WARP_BYPASS) != 0u
                            ? (TORCH_TENSOR_TYPE)tans_uncook_raw_value(raw_val, (int64_t)max_value)
                            : (TORCH_TENSOR_TYPE)out_sym) +
                       offset;
    }
  }

  // Write the continuation point back, exactly like tans_cuda.cu's single-lane
  // kernel: the reader's position *is* a tANS header, so a second pop on the
  // same stream decodes the previous push - which is what the chunked decode
  // path relies on. The state goes back in the encoder's range (`+ ts`) so the
  // header keeps one meaning for both directions.
  header[tans_state_slot((int64_t)lane)] = (int32_t)(state + ts);
  if (store_lane) {
    header[0] = (int32_t)((window.cursor + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
    header[tans_bitbuf_slot(ilv)] = (int32_t)window.res;
    header[tans_nbits_slot(ilv)] = (int32_t)window.res_bits;
  }
}

// ---------------------------------------------------------------------------
// host-side launchers
// ---------------------------------------------------------------------------
// Same signatures and the same argument checks as the single-lane launchers in
// tans_cuda.cu (which route here for NUM_INTERLEAVES > 1), so a stream coded by
// one is a stream the other can decode.

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
void tans_warp_push_indexed_cuda(ta::Tensor stream, const ta::Tensor& symbols,
                                 const ta::Tensor& indexes, const ta::Tensor& tables,
                                 const ta::Tensor& cdfs_sizes, const ta::Tensor& offsets,
                                 int64_t freq_precision, bool bypass_coding,
                                 int64_t bypass_precision) {
  static_assert(NUM_INTERLEAVES > 1 && NUM_INTERLEAVES <= 32 &&
                    (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "the warp kernels cover 2, 4, 8 and 32 interleaved lanes");
  // The kernels launch on the tensor's device and on torch's current stream,
  // exactly like the single-lane ones (the DeviceGuard matters: a launch with a
  // foreign device's stream fails with "invalid resource handle").
  const ta::DeviceGuard device_guard(ta::device_index(stream));
  const cudaStream_t cuda_stream = ta::current_stream(stream);

  TA_DISPATCH_INTEGRAL(indexes.scalar_type(), "tans_warp_push_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    const TansTableStaging staging = tans_table_staging(tables);
    const int num_threads_per_block =
        tans_warp_num_threads_per_block(batch_size, (int64_t)NUM_INTERLEAVES);
    const int groups_per_block = num_threads_per_block / (int)NUM_INTERLEAVES;
    const int num_blocks = (int)((batch_size + groups_per_block - 1) / groups_per_block);

    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const int64_t num_dists = tables.size(0);

    // Two instantiations so that the staged path costs nothing when it is not
    // used (the staging decision is a runtime one - the table size is only known
    // with the tensors - but the per-symbol code must not carry the select).
    if (staging.words > 0) {
      tans_warp_push_indexed_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, true,
                                    NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, staging.bytes, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              staging.words);
    } else {
      tans_warp_push_indexed_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, false,
                                    NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists, 0);
    }
    TA_CUDA_CHECK(cudaGetLastError());
  });
}

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
ta::Tensor tans_warp_pop_indexed_cuda(ta::Tensor stream, const ta::Tensor& indexes,
                                      const ta::Tensor& tables, const ta::Tensor& cdfs_sizes,
                                      const ta::Tensor& offsets, int64_t freq_precision,
                                      bool bypass_coding, int64_t bypass_precision) {
  static_assert(NUM_INTERLEAVES > 1 && NUM_INTERLEAVES <= 32 &&
                    (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "the warp kernels cover 2, 4, 8 and 32 interleaved lanes");

  ta::Tensor symbols;

  // See tans_warp_push_indexed_cuda: the launch target has to be the current
  // device - and the output tensor has to be allocated on that device too.
  const ta::DeviceGuard device_guard(ta::device_index(stream));
  const cudaStream_t cuda_stream = ta::current_stream(stream);

  TA_DISPATCH_INTEGRAL(indexes.scalar_type(), "tans_warp_pop_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    const TansTableStaging staging = tans_table_staging(tables);
    const int num_threads_per_block =
        tans_warp_num_threads_per_block(batch_size, (int64_t)NUM_INTERLEAVES);
    const int groups_per_block = num_threads_per_block / (int)NUM_INTERLEAVES;
    const int num_blocks = (int)((batch_size + groups_per_block - 1) / groups_per_block);

    symbols = ta::zeros_like(indexes);

    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const int64_t num_dists = tables.size(0);

    if (staging.words > 0) {
      tans_warp_pop_indexed_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, true,
                                   NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, staging.bytes, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists,
              staging.words);
    } else {
      tans_warp_pop_indexed_kernel<TANS_STATE_TYPE, TANS_STREAM_TYPE, scalar_t, false,
                                   NUM_INTERLEAVES>
          <<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
              ta::view2<int32_t>(stream), ta::view2<scalar_t>(symbols),
              ta::view2<scalar_t>(indexes), tables_data, tables_row_stride,
              ta::view1<scalar_t>(cdfs_sizes), ta::view1<scalar_t>(offsets),
              (int)freq_precision, bypass_coding, (int)bypass_precision, num_dists, 0);
    }
    TA_CUDA_CHECK(cudaGetLastError());
  });

  return symbols;
}

}  // namespace tans_warp
