#include <torch/extension.h>
#if defined(WITH_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#elif defined(WITH_HIP)
#include <hip/hip_runtime.h>
#endif

// enable compiling rans_utils for kernel call
#define RANS_CUDA_API

#include "rans_utils.hpp"
#include "rans_build_config.hpp"
#include "rans_warp_cuda.cuh"

// NOTE: using 256 threads per block causes significant slowdown,
// likely due to the fact that each thread processes different amount of work or
// there is not enough parallelism to keep the GPU fully utilized.
// Using 1 thread per block allows us to process each stream sequentially without the overhead of synchronization between threads,
// which can be more efficient for small batch sizes or short streams.
#define DEFAULT_NUM_THREADS_PER_BLOCK 1 // 256

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
__global__ void rans_push_indexed_cuda_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= stream_accessor.size(0)) return;

    const auto stream_length = stream_accessor[b][0];
    const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
    RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
    RANS_STREAM_TYPE* state_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr);
    RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;

    RANS_STATE_TYPE state_cache;
    RANS_STREAM_TYPE* state_cache_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(&state_cache);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_cache_stream_ptr[i] = state_stream_ptr[i];
    }

    auto symbols_ptr = symbols_accessor[b].data();
    auto indexes_ptr = indexes_accessor[b].data();
    int64_t num_symbols = symbols_accessor.size(1);

    // reverse coding
    for (auto i = num_symbols-1; i >= 0; i--) {
      auto index = indexes_ptr[i];
      // check index range, skip on invalid indexes
      if (index < 0 || index >= cdfs_accessor.size(0)) {
        continue;
      }
      auto cdf_ptr = cdfs_accessor[index].data();
      auto cdf_size = cdfs_sizes_accessor[index];
      auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
      rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE, TORCH_TENSOR_TYPE>(
        &state_cache, &stream_ptr, symbols_ptr[i],
        cdf_ptr, cdf_size, offsets_accessor[index],
        freq_precision, bypass_coding, bypass_precision,
        cdf_alias_remap_ptr
      );
    }

    // update stream length
    stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_stream_ptr[i] = state_cache_stream_ptr[i];
    }

}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
__global__ void rans_pop_indexed_cuda_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  int64_t freq_precision,
  bool bypass_coding, 
  int64_t bypass_precision,
  int64_t inverse_cdf_precision)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= stream_accessor.size(0)) return;

    const auto stream_length = stream_accessor[b][0];
    const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
    RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
    RANS_STREAM_TYPE* state_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr);

    RANS_STATE_TYPE state_cache;
    RANS_STREAM_TYPE* state_cache_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(&state_cache);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_cache_stream_ptr[i] = state_stream_ptr[i];
    }
  
    RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;

    auto symbols_ptr = symbols_accessor[b].data();
    auto indexes_ptr = indexes_accessor[b].data();
    int64_t num_symbols = symbols_accessor.size(1);

    for (int64_t i = 0; i < num_symbols; i++) {
      auto index = indexes_ptr[i];
      // check index range, skip on invalid indexes
      if (index < 0 || index >= cdfs_accessor.size(0)) {
        symbols_ptr[i] = 0;
        continue;
      }
      auto cdf_size = cdfs_sizes_accessor[index];
      auto cdf_ptr = cdfs_accessor[index].data();
      auto inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
      auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<TORCH_TENSOR_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
      // TODO: check index validity
      symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE, TORCH_TENSOR_TYPE>(
        &state_cache, &stream_ptr,
        cdf_ptr, cdf_size, offsets_accessor[index],
        freq_precision, bypass_coding, bypass_precision,
        inversed_cdf_ptr, cdf_alias_table_ptr, inverse_cdf_precision
      );
    }

    // update stream length
    stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_stream_ptr[i] = state_cache_stream_ptr[i];
    }

}



template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding, 
  int64_t bypass_precision)
{
  // TORCH_CHECK(stream.dtype() == TORCH_TENSOR_DTYPE);
  // TORCH_CHECK(symbols.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(stream.device().type() == torch::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(symbols.device().type() == torch::DeviceType::CUDA);
  
  TORCH_CHECK(indexes.sizes() == symbols.sizes());
  // TORCH_CHECK(indexes.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(indexes.device().type() == torch::DeviceType::CUDA);

  // TORCH_CHECK(cdfs.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(cdfs.device().type() == torch::DeviceType::CUDA);

  TORCH_CHECK(cdfs_sizes.size(0) == cdfs.size(0));
  // TORCH_CHECK(cdfs_sizes.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(cdfs_sizes.device().type() == torch::DeviceType::CUDA);

  TORCH_CHECK(offsets.size(0) == cdfs.size(0));
  // TORCH_CHECK(offsets.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == torch::DeviceType::CUDA);

  AT_DISPATCH_INTEGRAL_TYPES(indexes.scalar_type(), "rans_push_indexed_cuda", [&] {

    auto batch_size = stream.size(0);

    if constexpr (NUM_INTERLEAVES > 1) {
      // warp-level interleaved kernels (see rans_warp_cuda.cuh)
      if constexpr (std::is_same_v<RANS_STATE_TYPE, uint32_t> && std::is_same_v<RANS_STREAM_TYPE, uint16_t>) {
        rans_warp::rans_warp_push_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES>(
          stream, symbols, indexes, cdfs, cdfs_sizes, offsets, freq_precision, bypass_coding, bypass_precision);
      } else {
        TORCH_CHECK(false, "torch_ans: CUDA interleaved coding is only supported for the rans32_16 variant (uint32 state + uint16 stream)");
      }
      return;
    }

    {
      const int num_threads_per_block = DEFAULT_NUM_THREADS_PER_BLOCK;
      const int num_blocks = (batch_size + num_threads_per_block - 1) / num_threads_per_block;

      rans_push_indexed_cuda_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES><<<num_blocks, num_threads_per_block>>>(
          stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          freq_precision, bypass_coding, bypass_precision
      );
    }

  });
  
  // cudaDeviceSynchronize();
}



template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding, 
  int64_t bypass_precision,
  int64_t inverse_cdf_precision)
{

  // TORCH_CHECK(stream.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(stream.device().type() == torch::DeviceType::CUDA);

  // TORCH_CHECK(cdfs.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(cdfs.device().type() == torch::DeviceType::CUDA);

  TORCH_CHECK(cdfs_sizes.size(0) == cdfs.size(0));
  // TORCH_CHECK(cdfs_sizes.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(cdfs_sizes.device().type() == torch::DeviceType::CUDA);

  TORCH_CHECK(offsets.size(0) == cdfs.size(0));
  // TORCH_CHECK(offsets.dtype() == TORCH_TENSOR_DTYPE);
  TORCH_INTERNAL_ASSERT(offsets.device().type() == torch::DeviceType::CUDA);

  torch::Tensor symbols;

  AT_DISPATCH_INTEGRAL_TYPES(indexes.scalar_type(), "rans_pop_indexed_cuda", [&] {

    auto batch_size = stream.size(0);

    if constexpr (NUM_INTERLEAVES > 1) {
      // warp-level interleaved kernels (see rans_warp_cuda.cuh)
      if constexpr (std::is_same_v<RANS_STATE_TYPE, uint32_t> && std::is_same_v<RANS_STREAM_TYPE, uint16_t>) {
        symbols = rans_warp::rans_warp_pop_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
          stream, indexes, cdfs, cdfs_sizes, offsets, freq_precision, bypass_coding, bypass_precision,
          inverse_cdf_precision);
      } else {
        TORCH_CHECK(false, "torch_ans: CUDA interleaved coding is only supported for the rans32_16 variant (uint32 state + uint16 stream)");
      }
      return;
    }

    symbols = torch::zeros_like(indexes);

    {
      const int num_threads_per_block = DEFAULT_NUM_THREADS_PER_BLOCK;
      const int num_blocks = (batch_size + num_threads_per_block - 1) / num_threads_per_block;

      rans_pop_indexed_cuda_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES><<<num_blocks, num_threads_per_block>>>(
          stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          freq_precision, bypass_coding, bypass_precision, inverse_cdf_precision
      );
    }

  });

  return symbols;

}

// Kernel to fix CDF rows in parallel
__global__ void fix_cdf_batch_kernel(int32_t* cdf_ptr, int B, int N) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) return;
  int32_t* row = cdf_ptr + b * (N + 1);
  for (int i = 0; i < N; ++i) {
    if (row[i] == row[i + 1]) {
      int32_t best_freq = INT32_MAX;
      int best_steal = -1;
      for (int j = 0; j < N; ++j) {
        int32_t f = row[j + 1] - row[j];
        if (f > 1 && f < best_freq) {
          best_freq = f;
          best_steal = j;
        }
      }
      if (best_steal == -1) continue;
      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) {
          row[j] -= 1;
        }
      } else {
        if (best_steal > i) {
          for (int j = i + 1; j <= best_steal; ++j) {
            row[j] += 1;
          }
        }
      }
    }
  }
}


// B4: one block per distribution row does the whole pmf -> quantized cdf
// pipeline (round, prefix, rescale, inclusive scan, "steal frequency" fix-up).
// The previous version was ~10 torch ops plus a separate fix-up kernel, i.e.
// ~10 dispatches of ~40 us each, and init_params runs it on every
// encode/decode call of the dist_freqs API. Arithmetic mirrors the old chain
// bit-for-bit (int32 wrap, truncating division) - scripts/bench_cdf_build.py
// checks that against a torch-level reference.
//
// Shared memory layout: [rowsz int32 scan buffer][1 int32 total][nt int64
// partial sums].
template <typename PMF_T>
__global__ void pmf_to_quantized_cdf_kernel(const PMF_T* __restrict__ pmf, int64_t N,
                                            int64_t scale, int32_t* __restrict__ cdf) {
  extern __shared__ char smem_raw[];
  const int64_t rowsz = N + 1;
  const int tid = threadIdx.x;
  const int nt = blockDim.x;
  // NOTE: rowsz can be odd, so s_partial must not be laid out right after the
  // scan buffer - an int64 at a 4-byte-aligned offset faults. Layout:
  // [total (4B)][pad to 8][nt partial sums (8B each)][scan buffer (4B * rowsz)]
  int32_t* const s_total = reinterpret_cast<int32_t*>(smem_raw);
  int64_t* const s_partial =
      reinterpret_cast<int64_t*>(smem_raw + sizeof(int64_t));
  int32_t* const s_scan =
      reinterpret_cast<int32_t*>(smem_raw + sizeof(int64_t) + (size_t)nt * sizeof(int64_t));

  int32_t* const row = cdf + (int64_t)blockIdx.x * rowsz;
  const PMF_T* const pmf_row = pmf + (int64_t)blockIdx.x * N;

  // freq = round(pmf * 2**p); cdf = [0, freq...]
  if (tid == 0) row[0] = 0;
  for (int64_t j = tid; j < N; j += nt) {
    row[j + 1] = (int32_t)nearbyint((double)(pmf_row[j] * (PMF_T)scale));
  }
  __syncthreads();

  // total = sum(row) with 0 replaced by 1
  int64_t partial = 0;
  for (int64_t j = tid; j < rowsz; j += nt) partial += (int64_t)row[j];
  s_partial[tid] = partial;
  __syncthreads();
  if (tid == 0) {
    int64_t total = 0;
    for (int i = 0; i < nt; ++i) total += s_partial[i];
    const int32_t t32 = (int32_t)total;
    s_total[0] = (t32 == 0) ? 1 : t32;
  }
  __syncthreads();
  const int32_t total32 = s_total[0];

  // in-place inclusive scan of (row * 2**p) / total, using row/s_scan as a
  // double buffer (Hillis-Steele)
  for (int64_t j = tid; j < rowsz; j += nt) {
    const int32_t prod = (int32_t)((int64_t)row[j] * scale);
    s_scan[j] = prod / total32;
  }
  __syncthreads();
  for (int64_t off = 1; off < rowsz; off <<= 1) {
    for (int64_t j = tid; j < rowsz; j += nt) {
      row[j] = (j >= off) ? (int32_t)(s_scan[j] + s_scan[j - off]) : s_scan[j];
    }
    __syncthreads();
    for (int64_t j = tid; j < rowsz; j += nt) s_scan[j] = row[j];
    __syncthreads();
  }
  if (tid == 0) row[N] = (int32_t)scale;
  __syncthreads();

  // "steal frequency" fix-up: sequential per row, exactly like the old kernel
  if (tid == 0) {
    for (int64_t i = 0; i < N; ++i) {
      if (row[i] == row[i + 1]) {
        int32_t best_freq = INT32_MAX;
        int64_t best_steal = -1;
        for (int64_t j = 0; j < N; ++j) {
          const int32_t f = row[j + 1] - row[j];
          if (f > 1 && f < best_freq) {
            best_freq = f;
            best_steal = j;
          }
        }
        if (best_steal == -1) continue;
        if (best_steal < i) {
          for (int64_t j = best_steal + 1; j <= i; ++j) row[j] -= 1;
        } else if (best_steal > i) {
          for (int64_t j = i + 1; j <= best_steal; ++j) row[j] += 1;
        }
      }
    }
  }
}


// Batched PMF to quantized CDF (CUDA, parallel over batch)
torch::Tensor rans_pmf_to_quantized_cdf_cuda(const torch::Tensor& pmf, int64_t precision) {
  TORCH_CHECK(pmf.is_cuda(), "Input must be CUDA tensor");
  // TORCH_CHECK(pmf.dim() == 1 || pmf.dim() == 2, "pmf must be 1D or 2D tensor");
  torch::Tensor pmf_batched;
  int64_t B, N;
  if (pmf.dim() == 1) {
    pmf_batched = pmf.unsqueeze(0);
    B = 1;
    N = pmf.size(0);
  } 
  else 
  // If pmf has higher dimensions, we treat the last dimension as the symbol dimension and batch over the preceding dimensions
  {
    pmf_batched = pmf.reshape({-1, pmf.size(-1)});
    B = pmf_batched.size(0);
    N = pmf_batched.size(1);
  }
  const int64_t scale = (int64_t)1 << precision;
  auto cdf_contig = torch::zeros({B, N + 1},
      torch::TensorOptions().dtype(torch::kInt32).device(pmf.device()));
  const torch::Tensor pmf_contig = pmf_batched.contiguous();

  const int threads = 256;
  const size_t smem_bytes =
      sizeof(int64_t) + (size_t)threads * sizeof(int64_t) + (size_t)(N + 1) * sizeof(int32_t);

  AT_DISPATCH_FLOATING_TYPES(pmf_contig.scalar_type(), "rans_pmf_to_quantized_cdf_cuda", [&] {
    pmf_to_quantized_cdf_kernel<scalar_t><<<(int)B, threads, smem_bytes>>>(
        pmf_contig.data_ptr<scalar_t>(), N, scale, cdf_contig.data_ptr<int32_t>());
  });

  if (pmf.dim() == 1) {
    return cdf_contig[0];
  } else {
    auto sizes = std::vector<int64_t>(pmf.sizes().begin(), pmf.sizes().end()-1);
    sizes.push_back(N+1);
    return cdf_contig.reshape(sizes);
  }
}



// B2: one launch builds the combined "cdf ++ inverse-CDF table" tensor. The
// Python version needed 3-5 torch calls (~40 us of dispatch each on this
// machine); init_params runs per encode/decode call in the dist_freqs API, so
// that overhead used to exceed the coding time for small tensors.
template <typename T>
__global__ void build_inverse_cdf_kernel(const T* __restrict__ cdfs, int64_t num_cols,
                                         int64_t out_cols, int64_t shift,
                                         T* __restrict__ out, int64_t total) {
  const int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;
  const int64_t row = idx / out_cols;
  const int64_t col = idx - row * out_cols;
  const T* cdf_row = cdfs + row * num_cols;
  T* out_row = out + row * out_cols;
  if (col < num_cols) {  // plain copy of the cdf part
    out_row[col] = cdf_row[col];
    return;
  }
  // largest i with cdf[i] <= value (== count(cdf <= value) - 1); cdf[0] is
  // always 0 <= value, so this is the symbol the decoder must start from
  const int64_t value = (int64_t)(col - num_cols) << shift;
  int64_t lo = 0, hi = num_cols - 1;
  while (lo < hi) {
    const int64_t mid = lo + (hi - lo + 1) / 2;
    if ((int64_t)cdf_row[mid] <= value) lo = mid; else hi = mid - 1;
  }
  out_row[col] = (T)lo;
}


torch::Tensor rans_build_inverse_cdf_cuda(const torch::Tensor& cdfs, int64_t freq_precision,
                                          int64_t table_precision) {
  TORCH_CHECK(cdfs.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(table_precision >= 1 && table_precision <= freq_precision,
              "table_precision must be in [1, ", freq_precision, "], got ", table_precision);
  const bool was_1d = cdfs.dim() == 1;
  const torch::Tensor cdf = (was_1d ? cdfs.unsqueeze(0) : cdfs).contiguous();
  const int64_t B = cdf.size(0);
  const int64_t M = cdf.size(1);
  const int64_t T = (int64_t)1 << table_precision;
  const int64_t shift = freq_precision - table_precision;
  const int64_t out_cols = M + T;
  const int64_t total = B * out_cols;

  torch::Tensor out = torch::empty({B, out_cols}, cdf.options());
  const int threads = 256;
  const int blocks = (int)((total + threads - 1) / threads);

  AT_DISPATCH_INTEGRAL_TYPES(cdf.scalar_type(), "rans_build_inverse_cdf_cuda", [&] {
    build_inverse_cdf_kernel<scalar_t><<<blocks, threads>>>(
        cdf.data_ptr<scalar_t>(), M, out_cols, shift, out.data_ptr<scalar_t>(), total);
  });

  return was_1d ? out[0] : out;
}


// ---------------------------------------------------------------------------
// Instantiations required by the pybind bindings in rans_bindings.hpp.
//
// Whenever WITH_CUDA/WITH_HIP is defined, the rans.hpp wrappers reference the
// CUDA template for every bound combination, so the .cu translation unit must
// define each of them - a missing one is not a compile error but an undefined
// symbol, which makes the whole module fail to import. The blocks mirror both
// the bindings and the gates in rans_build_config.hpp.
// ---------------------------------------------------------------------------

#define TORCH_ANS_CUDA_INST_ILV(STATE, STREAM, ILV) \
  TORCH_ANS_INST_PUSH(rans_push_indexed_cuda, STATE, STREAM, false, ILV); \
  TORCH_ANS_INST_POP(rans_pop_indexed_cuda, STATE, STREAM, false, false, ILV)
#define TORCH_ANS_CUDA_INST_ILV_INVCDCDF(STATE, STREAM, ILV) \
  TORCH_ANS_INST_POP(rans_pop_indexed_cuda, STATE, STREAM, false, true, ILV)
#define TORCH_ANS_CUDA_INST_ILV_ALIAS(STATE, STREAM, ILV) \
  TORCH_ANS_INST_PUSH(rans_push_indexed_cuda, STATE, STREAM, true, ILV); \
  TORCH_ANS_INST_POP(rans_pop_indexed_cuda, STATE, STREAM, true, false, ILV)

#if TORCH_ANS_WITH_RANS64
TORCH_ANS_CUDA_INST_ILV(uint64_t, uint32_t, 1);
#  if TORCH_ANS_WITH_INVCDCDF
TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CUDA_INST_ILV_ALIAS(uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CUDA_INST_ILV(uint64_t, uint32_t, 2);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint64_t, uint32_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint64_t, uint32_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CUDA_INST_ILV(uint64_t, uint32_t, 4);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint64_t, uint32_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint64_t, uint32_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CUDA_INST_ILV(uint64_t, uint32_t, 8);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint64_t, uint32_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint64_t, uint32_t, 8);
#    endif
#  endif
#endif

#if TORCH_ANS_WITH_RANS32
TORCH_ANS_CUDA_INST_ILV(uint32_t, uint8_t, 1);
#  if TORCH_ANS_WITH_INVCDCDF
TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint8_t, 2);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint8_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint8_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint8_t, 4);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint8_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint8_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint8_t, 8);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint8_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint8_t, 8);
#    endif
#  endif
#endif

#if TORCH_ANS_WITH_RANS32_16
TORCH_ANS_CUDA_INST_ILV(uint32_t, uint16_t, 1);
#  if TORCH_ANS_WITH_INVCDCDF
TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint16_t, 2);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint16_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint16_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint16_t, 4);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint16_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint16_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint16_t, 8);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint16_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CUDA_INST_ILV_ALIAS(uint32_t, uint16_t, 8);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
  TORCH_ANS_CUDA_INST_ILV(uint32_t, uint16_t, 32);
#    if TORCH_ANS_WITH_INVCDCDF
  TORCH_ANS_CUDA_INST_ILV_INVCDCDF(uint32_t, uint16_t, 32);
#    endif
#  endif
#endif

#undef TORCH_ANS_CUDA_INST_ILV
#undef TORCH_ANS_CUDA_INST_ILV_INVCDCDF
#undef TORCH_ANS_CUDA_INST_ILV_ALIAS
