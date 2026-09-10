#pragma once

#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "rans_utils.hpp"
#include "rans_build_config.hpp"

// std::vector<DEFAULT_TORCH_TENSOR_TYPE> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision);

// template <size_t STATE_BITS, size_t STREAM_BITS>
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS=0>
torch::Tensor rans_init_stream(int64_t size, int64_t num_interleaves=1, int64_t preallocate_size=0);

// template <typename STATE_TYPE, typename STREAM_TYPE>
// std::vector<py::bytes> rans_flush_stream(// ANSStream stream,
//   torch::Tensor& stream);

std::vector<py::bytes> rans_stream_to_byte_strings(const torch::Tensor& stream);

torch::Tensor rans_byte_strings_to_stream(std::vector<py::bytes> byte_strings);

std::tuple<torch::Tensor, torch::Tensor> rans_alias_build_table(
  const torch::Tensor& cdfs, const torch::Tensor& cdfs_sizes,
  int64_t symbol_precision=8,
  int64_t freq_precision=16
);

#if defined(WITH_CUDA) || defined(WITH_HIP)
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1);

  // Batched PMF to quantized CDF declarations
torch::Tensor rans_pmf_to_quantized_cdf_cuda(const torch::Tensor& pmf, int64_t precision);

// B2: fused "cdf ++ inverse-CDF table" construction (device side, one launch)
torch::Tensor rans_build_inverse_cdf_cuda(const torch::Tensor& cdfs, int64_t freq_precision,
                                          int64_t table_precision);

#endif

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1);

// Batched PMF to quantized CDF declarations
torch::Tensor rans_pmf_to_quantized_cdf_cpu(const torch::Tensor& pmf, int64_t precision);

// B2: fused "cdf ++ inverse-CDF table" construction (host side)
torch::Tensor rans_build_inverse_cdf_cpu(const torch::Tensor& cdfs, int64_t freq_precision,
                                         int64_t table_precision);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push(// ANSStream stream,
  torch::Tensor stream, 
  torch::Tensor symbols, 
  std::optional<torch::Tensor> indexes, 
  std::optional<torch::Tensor> cdfs, 
  std::optional<torch::Tensor> cdfs_sizes, 
  std::optional<torch::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4)
{
  const int64_t batch_size = stream.size(0);
  const int64_t num_symbols = symbols.size(1);
  const bool use_cuda = stream.device().is_cuda();

  if (symbols.size(0) != batch_size) {
    throw py::value_error("Symbols batch size mismatch!");
  }

  AT_DISPATCH_INTEGRAL_TYPES(stream.scalar_type(), "rans_push_resize_stream", [&] {
    // Resize the stream to its worst-case size. Per coded symbol the stream
    // grows by at most:
    //   * freq_precision bits -- the rANS state absorbs
    //     log2(2^freq_precision / freq) <= freq_precision bits before a
    //     renormalization flush;
    //   * with bypass_coding, an out-of-range symbol additionally pushes a raw
    //     value in bypass_precision-sized chunks (see rans_push_raw_value_step):
    //     ceil(symbol_bits / bypass_precision) chunks for the value itself, one
    //     chunk holding the chunk count, plus one chunk per carry of that count
    //     (a carry every max_bypass_val).
    const int64_t max_byte_length =
        (int64_t)stream.index({torch::indexing::Slice(), 0}).max().item<scalar_t>();
    int64_t bits_per_symbol = freq_precision;
    if (bypass_coding) {
      TORCH_CHECK(bypass_precision >= 1, "bypass_precision must be >= 1");
      const int64_t symbol_bits = (int64_t)sizeof(scalar_t) * 8;
      const int64_t max_bypass_val = ((int64_t)1 << bypass_precision) - 1;
      const int64_t n_bypass = (symbol_bits + bypass_precision - 1) / bypass_precision;
      const int64_t n_chunks = n_bypass + 1 + n_bypass / max_bypass_val;
      bits_per_symbol += n_chunks * bypass_precision;
    }
    // +7 rounds the bit total up to whole bytes. On top of that, every rANS
    // state can hold up to one not-yet-flushed stream word, so reserve one word
    // per interleave (the amortized bit estimate above does not account for it),
    // plus one more word for the branch-free renormalization, which always
    // writes one word past the cursor (see APPEND_STATE_TO_STREAM_IF).
    const int64_t safe_byte_length = max_byte_length
        + (num_symbols * bits_per_symbol + 7) / 8
        + ((int64_t)NUM_INTERLEAVES + 1) * (int64_t)sizeof(RANS_STREAM_TYPE);

    if (safe_byte_length < max_byte_length) {
      throw py::value_error("Overflow!");
    }

    // NOTE: the computation above is deliberately kept in int64_t rather than a
    // platform-width type. The intermediate num_symbols * bits_per_symbol already
    // exceeds 2^31 for realistic workloads (1e8 symbols at 51 bits each is
    // ~5e9), and signed overflow is undefined behaviour, so a 32-bit type would
    // corrupt the result rather than merely trip the check below.
    //
    // The result is then range-checked against size_t, which *is* platform-width
    // (32-bit on 32-bit systems, where no allocation can exceed 4 GiB), so an
    // unrepresentable stream fails here instead of being silently truncated.
    const int64_t safe_tensor_length = safe_byte_length / (int64_t)sizeof(scalar_t) + 1;
    const uint64_t total_bytes =
        (uint64_t)batch_size * (uint64_t)safe_tensor_length * (uint64_t)sizeof(scalar_t);
    TORCH_CHECK(total_bytes <= (uint64_t)std::numeric_limits<size_t>::max(),
                "rans_push: the stream would need ", total_bytes,
                " bytes, which is not addressable on this platform");
    if (safe_tensor_length > stream.size(1)) {
      const auto stream_copy = stream.clone();
      stream = stream.resize_({batch_size, safe_tensor_length}).contiguous();
      stream.index_put_({torch::indexing::Slice(), torch::indexing::Slice(0, stream_copy.size(1))}, stream_copy);
    }
  });

  // indexed cdf
  if (indexes.has_value()) {

    TORCH_CHECK(cdfs.has_value());
    TORCH_CHECK(cdfs_sizes.has_value());

    torch::Tensor offsets_tensor;
    if (!offsets.has_value()) 
      offsets_tensor = torch::zeros({cdfs.value().size(0)}, indexes.value().options());
    else
      offsets_tensor = offsets.value();

    if (use_cuda) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
      return rans_push_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES>(
        stream.contiguous().cuda(), symbols.contiguous().cuda(), 
        indexes.value().contiguous().cuda(), cdfs.value().contiguous().cuda(), 
        cdfs_sizes.value().contiguous().cuda(), offsets_tensor.contiguous().cuda(), 
        freq_precision, bypass_coding, bypass_precision);
#else
      AT_ERROR("torch_ans is not compiled with GPU support!");
#endif
    }

    return rans_push_indexed_cpu<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES>(
      stream.contiguous(), symbols.contiguous(), 
      indexes.value().contiguous(), cdfs.value().contiguous(), 
      cdfs_sizes.value().contiguous(), offsets_tensor.contiguous(), 
      freq_precision, bypass_coding, bypass_precision);

  }
  else {
    // per-symbol cdf
    if (cdfs.has_value()) {
      // TORCH_CHECK(cdfs_sizes.has_value(), #cdfs_sizes " must exist");
      // TORCH_CHECK(offsets.has_value(), #offsets " must exist");

      // auto cdfs_accessor = cdfs.value().accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
      // auto cdfs_sizes_accessor = cdfs_sizes.value().accessor<DEFAULT_TORCH_TENSOR_TYPE, 1>();
      // auto offsets_accessor = offsets.value().accessor<DEFAULT_TORCH_TENSOR_TYPE, 1>();

      // at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      //     for (size_t b = start; b < end; b++) {
      //       auto stream_ptr_offset = stream_accessor[b][0] / sizeof(RANS_STREAM_TYPE);
      //       RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
      //       RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr+1) + stream_ptr_offset - 1;
            
      //       auto symbols_ptr = symbols_accessor[b].data();
      //       // reverse coding
      //       for (auto i = num_symbols-1; i >= 0; i--) {
      //         rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, DEFAULT_TORCH_TENSOR_TYPE, DEFAULT_TORCH_TENSOR_TYPE>(
      //           state_ptr, &stream_ptr, 
      //           symbols_ptr[i], cdfs_accessor[i].data(), cdfs_sizes_accessor[i], offsets_accessor[i],
      //           freq_precision, bypass_coding, bypass_precision
      //         );
      //       }
            
      //       // update stream length
      //       stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
      //     }
      // });

    }
    // direct coding
    else {
      // at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      //     for (size_t b = start; b < end; b++) {
      //       auto stream_ptr_offset = stream_accessor[b][0] / sizeof(RANS_STREAM_TYPE);
      //       RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
      //       RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr+1) + stream_ptr_offset - 1;
            
      //       auto symbols_ptr = symbols_accessor[b].data();
      //       // reverse coding
      //       for (auto i = num_symbols-1; i >= 0; i--) {
      //         rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, DEFAULT_TORCH_TENSOR_TYPE, DEFAULT_TORCH_TENSOR_TYPE>(
      //           state_ptr, &stream_ptr, 
      //           symbols_ptr[i], nullptr, (1 >> symbol_precision) + 1, 0,
      //           freq_precision, bypass_coding, bypass_precision
      //         );
      //       }

      //       // update stream length
      //       stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
      //     }
      // });

    }
  }



}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop(// ANSStream stream,
  torch::Tensor stream, 
  std::optional<torch::Tensor> indexes, 
  std::optional<torch::Tensor> cdfs, 
  std::optional<torch::Tensor> cdfs_sizes, 
  std::optional<torch::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1)
{
  torch::Tensor symbols;
  const bool use_cuda = stream.device().is_cuda();



  // indexed cdf
  if (indexes.has_value()) {

    TORCH_CHECK(cdfs.has_value());
    TORCH_CHECK(cdfs_sizes.has_value());

    torch::Tensor offsets_tensor;
    if (!offsets.has_value()) 
      offsets_tensor = torch::zeros({cdfs.value().size(0)}, indexes.value().options());
    else
      offsets_tensor = offsets.value();

    if (use_cuda) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
      return rans_pop_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
        stream.contiguous().cuda(),
        indexes.value().contiguous().cuda(), cdfs.value().contiguous().cuda(), 
        cdfs_sizes.value().contiguous().cuda(), offsets_tensor.contiguous().cuda(), 
        freq_precision, bypass_coding, bypass_precision, inverse_cdf_precision);
#else
      AT_ERROR("torch_ans is not compiled with GPU support!");
#endif
    }

    return rans_pop_indexed_cpu<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
        stream.contiguous(),
        indexes.value().contiguous(), cdfs.value().contiguous(), 
        cdfs_sizes.value().contiguous(), offsets_tensor.contiguous(), 
        freq_precision, bypass_coding, bypass_precision, inverse_cdf_precision);

  }
  else {
    // TODO:
    // throw py::not_implemented_error("");
  }
  // TODO: shrink stream

  return symbols;

}


inline torch::Tensor rans_pmf_to_quantized_cdf(const torch::Tensor& pmfs, int64_t precision) {
  if (pmfs.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return rans_pmf_to_quantized_cdf_cuda(pmfs, precision);
#else
    AT_ERROR("torch_ans is not compiled with GPU support!");
#endif
  } else {
    return rans_pmf_to_quantized_cdf_cpu(pmfs, precision);
  }
}

// B2: build the combined cdf ++ inverse-CDF table in a single op. Python used
// to spend 3-5 torch calls (arange/expand, searchsorted, sub, cat) on it, and
// each torch call costs ~40 us of dispatch on this machine - more than the
// coding itself for small tensors, because init_params runs per encode/decode
// call in the dist_freqs API.
inline torch::Tensor rans_build_inverse_cdf(const torch::Tensor& cdfs, int64_t freq_precision,
                                            int64_t table_precision) {
  if (cdfs.device().is_cuda()) {
#if defined(WITH_CUDA) || defined(WITH_HIP)
    return rans_build_inverse_cdf_cuda(cdfs, freq_precision, table_precision);
#else
    AT_ERROR("torch_ans is not compiled with GPU support!");
#endif
  } else {
    return rans_build_inverse_cdf_cpu(cdfs, freq_precision, table_precision);
  }
}



// NOTE: the pybind11 bindings used to be defined here as the
// TORCH_EXTENSION_RANS_BINDINGS(m) macro. They moved to rans_bindings.hpp,
// where they are a regular function and the operator set can be gated by the
// feature macros in rans_build_config.hpp (a macro body cannot contain #if).
