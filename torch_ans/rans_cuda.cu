#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// enable compiling rans_utils for kernel call
#define RANS_CUDA_API

#include "rans_utils.hpp"

#define DEFAULT_NUM_THREADS_PER_BLOCK 1 // 256
#define DEFAULT_SHARED_STREAM_CACHE_SIZE (4096 / sizeof(RANS_STREAM_TYPE))
#define DEFAULT_SHARED_CDF_CACHE_SIZE (4096 / sizeof(TORCH_TENSOR_TYPE))

#define USE_INTERLEAVED_KERNEL_THREADS (NUM_INTERLEAVES>1)

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
__global__ void rans_push_indexed_cuda_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
{
    int b = blockIdx.x * blockDim.x + (USE_INTERLEAVED_KERNEL_THREADS ? 0 : threadIdx.x);
    if (b >= stream_accessor.size(0)) return;

    const auto stream_length = stream_accessor[b][0];
    const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
    RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
    RANS_STREAM_TYPE* state_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr);
    // RANS_STATE_TYPE* state_ptr;
    // cudaMalloc(&state_ptr, 32);
    // *state_ptr = *reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
    RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;

    RANS_STATE_TYPE state_cache;
    RANS_STREAM_TYPE* state_cache_stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(&state_cache);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_cache_stream_ptr[i] = state_stream_ptr[i];
    }

    auto symbols_ptr = symbols_accessor[b].data();
    auto indexes_ptr = indexes_accessor[b].data();
    ssize_t num_symbols = symbols_accessor.size(1);

    // __shared__ RANS_STREAM_TYPE stream_cache_ptr[DEFAULT_SHARED_STREAM_CACHE_SIZE];
    // RANS_STREAM_TYPE* stream_cache_mutable_ptr = stream_cache_ptr;

    // TODO: interleave coding between threads (need to move stream_ptr out from step)
    // const auto state_idx = (USE_INTERLEAVED_KERNEL_THREADS ? threadIdx.x : 0);
    // __shared__ RANS_STATE_TYPE state_ptr[NUM_INTERLEAVES];
    // __shared__ const RANS_STREAM_TYPE stream_cache_ptr[DEFAULT_SHARED_STREAM_CACHE_SIZE];
    // __shared__ RANS_STREAM_TYPE* stream_cache_ptr_mutable = stream_cache;
    // const auto first_interleave = num_symbols % NUM_INTERLEAVES;
    // ssize_t i = num_symbols-state_idx-1;
    // for (; i >= 0; i-=NUM_INTERLEAVES) {
    //     const auto index = indexes_ptr[i-state_idx];
    //     const auto cdf_ptr = cdfs_accessor[index].data();
    //     const auto cdf_size = cdfs_sizes_accessor[index];
    //     const auto offsets = offsets_accessor[index];
    //     const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
    //     // TODO: check index validity
    //     rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
    //       state_ptr+(NUM_INTERLEAVES-1-state_idx), &stream_cache_ptr_mutable, symbols_ptr[i-state_idx],
    //       cdf_ptr, cdf_size, offsets,
    //       freq_precision, bypass_coding, bypass_precision,
    //       cdf_alias_remap_ptr
    //     );
    //     __syncthreads();
    //     // TODO: write stream_ptr in sync
    //   }
    // }
    
    // reverse coding
    for (auto i = num_symbols-1; i >= 0; i--) {
      auto index = indexes_ptr[i];
      auto cdf_ptr = cdfs_accessor[index].data();
      auto cdf_size = cdfs_sizes_accessor[index];
      auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
      // TODO: check index validity
      rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE, TORCH_TENSOR_TYPE>(
        &state_cache, &stream_ptr, symbols_ptr[i],
        cdf_ptr, cdf_size, offsets_accessor[index],
        freq_precision, bypass_coding, bypass_precision,
        cdf_alias_remap_ptr
      );
      // RANS_STATE_TYPE x = *state_ptr;
      // if (x >= RANS_STATE_LOWER_BOUND) {
      //   symbols_ptr[i] = 0;
      // } 
      
      // stream_ptr[0] = *state_ptr;
      // *state_ptr = index;
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
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
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

    const int num_threads_per_block = USE_INTERLEAVED_KERNEL_THREADS ? NUM_INTERLEAVES : DEFAULT_NUM_THREADS_PER_BLOCK;
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

  });
  
  // cudaDeviceSynchronize();
}



template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
__global__ void rans_pop_indexed_cuda_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
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

    // __shared__ TORCH_TENSOR_TYPE cdf_cache_ptr[DEFAULT_SHARED_STREAM_CACHE_SIZE];
    // for (ssize_t index = 0; index < cdfs_sizes_accessor.size(0); index++) {
    //   auto cdf_size = cdfs_sizes_accessor[index];
    //   auto cdf_ptr = cdfs_accessor[index].data();
    //   memcpy((cdf_cache_ptr + index * cdf_size), cdf_ptr, cdf_size*sizeof(TORCH_TENSOR_TYPE));
    // }

    auto symbols_ptr = symbols_accessor[b].data();
    auto indexes_ptr = indexes_accessor[b].data();
    ssize_t num_symbols = symbols_accessor.size(1);

    for (ssize_t i = 0; i < num_symbols; i++) {
      auto index = indexes_ptr[i];
      auto cdf_size = cdfs_sizes_accessor[index];
      auto cdf_ptr = cdfs_accessor[index].data();
      auto inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
      auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<TORCH_TENSOR_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
      // TODO: check index validity
      symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE, TORCH_TENSOR_TYPE>(
        &state_cache, &stream_ptr,
        cdf_ptr, cdf_size, offsets_accessor[index],
        freq_precision, bypass_coding, bypass_precision,
        inversed_cdf_ptr, cdf_alias_table_ptr
      );
    }

    // update stream length
    stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
    // update state in aligned memory
    for (auto i=0; i<(RANS_STATE_BITS/RANS_STREAM_BITS); i++) {
      state_stream_ptr[i] = state_cache_stream_ptr[i];
    }

}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
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

  torch::Tensor symbols = torch::zeros_like(indexes);

  AT_DISPATCH_INTEGRAL_TYPES(indexes.scalar_type(), "rans_pop_indexed_cuda", [&] {


    auto batch_size = stream.size(0);

    const int num_threads_per_block = USE_INTERLEAVED_KERNEL_THREADS ? NUM_INTERLEAVES : DEFAULT_NUM_THREADS_PER_BLOCK;
    const int num_blocks = (batch_size + num_threads_per_block - 1) / num_threads_per_block;

    rans_pop_indexed_cuda_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES><<<num_blocks, num_threads_per_block>>>(
        stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        freq_precision, bypass_coding, bypass_precision
    );

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


// Batched PMF to quantized CDF (CUDA, parallel over batch)
torch::Tensor rans_pmf_to_quantized_cdf_cuda(const torch::Tensor& pmf, int64_t precision) {
  TORCH_CHECK(pmf.is_cuda(), "Input must be CUDA tensor");
  TORCH_CHECK(pmf.dim() == 1 || pmf.dim() == 2, "pmf must be 1D or 2D tensor");
  auto dtype = torch::kInt32;
  torch::Tensor pmf_batched;
  int64_t B, N;
  if (pmf.dim() == 1) {
    pmf_batched = pmf.unsqueeze(0);
    B = 1;
    N = pmf.size(0);
  } else {
    pmf_batched = pmf;
    B = pmf.size(0);
    N = pmf.size(1);
  }
  auto freq = torch::round(pmf_batched * (1 << precision)).to(dtype);
  auto cdf = torch::zeros({B, N + 1}, torch::TensorOptions().dtype(dtype).device(pmf.device()));
  cdf.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}, freq);
  auto total = cdf.sum(1, true).to(dtype);
  total = torch::where(total == 0, torch::ones_like(total), total);
  cdf = ((cdf * (1 << precision)) / total).to(dtype);
  cdf = torch::cumsum(cdf, 1).to(dtype);
  cdf.index_put_({torch::indexing::Slice(), N}, 1 << precision);
  auto cdf_contig = cdf.contiguous();
  auto cdf_ptr = cdf_contig.data_ptr<int32_t>();

  // CUDA kernel for parallel batch CDF fix
  int threads = 256;
  int blocks = (B + threads - 1) / threads;
  fix_cdf_batch_kernel<<<blocks, threads>>>(cdf_ptr, B, N);
  cudaDeviceSynchronize();

  if (pmf.dim() == 1) {
    return cdf_contig[0];
  } else {
    return cdf_contig;
  }
}


TORCH_LIBRARY_IMPL(torch_ans, CUDA, m) {
    m.impl("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf_cuda);
    m.impl("rans64_push_indexed", &rans_push_indexed_cuda<uint64_t, uint32_t>);
    m.impl("rans64_pop_indexed", &rans_pop_indexed_cuda<uint64_t, uint32_t>);
    m.impl("rans64_i4_push_indexed", &rans_push_indexed_cuda<uint64_t, uint32_t, false, 4>);
    m.impl("rans64_i4_pop_indexed", &rans_pop_indexed_cuda<uint64_t, uint32_t, false, false, 4>);
    m.impl("rans64_alias_push_indexed", &rans_push_indexed_cuda<uint64_t, uint32_t, true>);
    m.impl("rans64_alias_pop_indexed", &rans_pop_indexed_cuda<uint64_t, uint32_t, true, false>);
    m.impl("rans64_invcdf_pop_indexed", &rans_pop_indexed_cuda<uint64_t, uint32_t, false, true>);
    m.impl("rans32_push_indexed", &rans_push_indexed_cuda<uint32_t, uint8_t>);
    m.impl("rans32_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint8_t>);
    m.impl("rans32_i4_push_indexed", &rans_push_indexed_cuda<uint32_t, uint8_t, false, 4>);
    m.impl("rans32_i4_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint8_t, false, false, 4>);
    m.impl("rans32_i4_invcdf_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint8_t, false, true, 4>);
    m.impl("rans32_alias_push_indexed", &rans_push_indexed_cuda<uint32_t, uint8_t, true>);
    m.impl("rans32_alias_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint8_t, true, false>);
    m.impl("rans32_invcdf_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint8_t, false, true>);
    m.impl("rans32_16_push_indexed", &rans_push_indexed_cuda<uint32_t, uint16_t>);
    m.impl("rans32_16_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint16_t>);
    m.impl("rans32_16_alias_push_indexed", &rans_push_indexed_cuda<uint32_t, uint16_t, true>);
    m.impl("rans32_16_alias_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint16_t, true, false>);
    m.impl("rans32_16_invcdf_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint16_t, false, true>);
    m.impl("rans32_16_i4_push_indexed", &rans_push_indexed_cuda<uint32_t, uint16_t, false, 4>);
    m.impl("rans32_16_i4_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint16_t, false, false, 4>);
    m.impl("rans32_16_i4_invcdf_pop_indexed", &rans_pop_indexed_cuda<uint32_t, uint16_t, false, true, 4>);
}