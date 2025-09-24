#pragma once

#include <torch/extension.h>

#include "rans_utils.hpp"

// std::vector<DEFAULT_TORCH_TENSOR_TYPE> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision);

// template <size_t STATE_BITS, size_t STREAM_BITS>
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS=0>
torch::Tensor rans_init_stream(ssize_t size, ssize_t num_interleaves=1, ssize_t preallocate_size=0);

// template <typename STATE_TYPE, typename STREAM_TYPE>
// std::vector<py::bytes> rans_flush_stream(// ANSStream stream,
//   torch::Tensor& stream);

std::vector<py::bytes> rans_stream_to_byte_strings(const torch::Tensor& stream);

torch::Tensor rans_byte_strings_to_stream(std::vector<py::bytes> byte_strings);

std::tuple<torch::Tensor, torch::Tensor> rans_alias_build_table(
  const torch::Tensor& cdfs, const torch::Tensor& cdfs_sizes,
  ssize_t symbol_precision=8,
  ssize_t freq_precision=16
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
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cuda(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4);

  // Batched PMF to quantized CDF declarations
torch::Tensor rans_pmf_to_quantized_cdf_cuda(const torch::Tensor& pmf, int64_t precision);

#endif

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4);

// Batched PMF to quantized CDF declarations
torch::Tensor rans_pmf_to_quantized_cdf_cpu(const torch::Tensor& pmf, int64_t precision);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push(// ANSStream stream,
  torch::Tensor stream, 
  torch::Tensor symbols, 
  std::optional<torch::Tensor> indexes, 
  std::optional<torch::Tensor> cdfs, 
  std::optional<torch::Tensor> cdfs_sizes, 
  std::optional<torch::Tensor> offsets,
  ssize_t symbol_precision=8,
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4)
{
  const ssize_t batch_size = stream.size(0);
  const ssize_t num_symbols = symbols.size(1);
  const bool use_cuda = stream.device().is_cuda();

  if (symbols.size(0) != batch_size) {
    throw py::value_error("Symbols batch size mismatch!");
  }

  AT_DISPATCH_INTEGRAL_TYPES(stream.scalar_type(), "rans_push_resize_stream", [&] {
    // resize stream to worse-case size
    auto max_byte_length = stream.index({torch::indexing::Slice(), 0}).max().item<scalar_t>();
    auto safe_byte_length = max_byte_length + (num_symbols * freq_precision + 7) / 8;
    // TODO: confirm the safe length for bypass_coding
    if (bypass_coding) {
      safe_byte_length += num_symbols * sizeof(scalar_t);
    }

    if (safe_byte_length < max_byte_length) {
      throw py::value_error("Overflow!");
    }

    const auto safe_tensor_length = safe_byte_length/((scalar_t) sizeof(scalar_t)) + 1;
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
  ssize_t symbol_precision=8,
  ssize_t freq_precision=16,
  bool bypass_coding=true, 
  ssize_t bypass_precision=4)
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
        freq_precision, bypass_coding, bypass_precision);
#else
      AT_ERROR("torch_ans is not compiled with GPU support!");
#endif
    }

    return rans_pop_indexed_cpu<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
        stream.contiguous(),
        indexes.value().contiguous(), cdfs.value().contiguous(), 
        cdfs_sizes.value().contiguous(), offsets_tensor.contiguous(), 
        freq_precision, bypass_coding, bypass_precision);

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


#define TORCH_EXTENSION_RANS_BINDINGS(m) \
    m.def("rans_stream_to_byte_strings", &rans_stream_to_byte_strings);\
    m.def("rans_byte_strings_to_stream", &rans_byte_strings_to_stream);\
    m.def("rans_alias_build_table", &rans_alias_build_table, py::arg("cdfs"), py::arg("cdfs_sizes"), py::arg("symbol_precision")=8, py::arg("freq_precision")=16);\
    m.def("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf, py::arg("pmfs"), py::arg("precision")=16);\
    m.def("rans64_init_stream", &rans_init_stream<uint64_t, uint32_t>, py::arg("size"), py::arg("num_interleaves")=1, py::arg("preallocate_size")=0);\
    m.def("rans64_push", &rans_push<uint64_t, uint32_t>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_pop", &rans_pop<uint64_t, uint32_t>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_i4_push", &rans_push<uint64_t, uint32_t, false, 4>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_i4_pop", &rans_pop<uint64_t, uint32_t, false, false, 4>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_alias_push", &rans_push<uint64_t, uint32_t, true>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_alias_pop", &rans_pop<uint64_t, uint32_t, true, false>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans64_invcdf_pop", &rans_pop<uint64_t, uint32_t, false, true>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_init_stream", &rans_init_stream<uint32_t, uint8_t>, py::arg("size"), py::arg("num_interleaves")=1, py::arg("preallocate_size")=0);\
    m.def("rans32_push", &rans_push<uint32_t, uint8_t>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_pop", &rans_pop<uint32_t, uint8_t>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_i4_push", &rans_push<uint32_t, uint8_t, false, 4>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_i4_pop", &rans_pop<uint32_t, uint8_t, false, false, 4>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_i4_invcdf_pop", &rans_pop<uint32_t, uint8_t, false, true, 4>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_alias_push", &rans_push<uint32_t, uint8_t, true>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_alias_pop", &rans_pop<uint32_t, uint8_t, true, false>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_invcdf_pop", &rans_pop<uint32_t, uint8_t, false, true>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_init_stream", &rans_init_stream<uint32_t, uint16_t>, py::arg("size"), py::arg("num_interleaves")=1, py::arg("preallocate_size")=0);\
    m.def("rans32_16_push", &rans_push<uint32_t, uint16_t>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_pop", &rans_pop<uint32_t, uint16_t>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_i4_push", &rans_push<uint32_t, uint16_t, false, 4>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_i4_pop", &rans_pop<uint32_t, uint16_t, false, false, 4>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_i4_invcdf_pop", &rans_pop<uint32_t, uint16_t, false, true, 4>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_alias_push", &rans_push<uint32_t, uint16_t, true>, py::arg("stream"), py::arg("symbols"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_alias_pop", &rans_pop<uint32_t, uint16_t, true, false>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\
    m.def("rans32_16_invcdf_pop", &rans_pop<uint32_t, uint16_t, false, true>, py::arg("stream"), py::arg("indexes")=py::none(), py::arg("cdfs")=py::none(), py::arg("cdfs_sizes")=py::none(), py::arg("offsets")=py::none(), py::arg("symbol_precision")=8, py::arg("freq_precision")=16, py::arg("bypass_coding")=true, py::arg("bypass_precision")=4);\

// #define TORCH_LIBRARY_RANS_BINDINGS(m)
//     m.def("rans64_init_stream", &rans_init_stream<uint64_t, uint32_t>);\
//     m.def("rans64_push", &rans_push<uint64_t, uint32_t>);\
//     m.def("rans64_pop", &rans_pop<uint64_t, uint32_t>);\
//     m.def("rans32_init_stream", &rans_init_stream<uint32_t, uint8_t>);\
//     m.def("rans32_push", &rans_push<uint32_t, uint8_t>);\
//     m.def("rans32_pop", &rans_pop<uint32_t, uint8_t>);\
//     m.def("rans_stream_to_byte_strings(Tensor stream) -> List[bytes] ");\
//     m.def("rans_byte_strings_to_stream(List[bytes]) -> Tensor ");\
