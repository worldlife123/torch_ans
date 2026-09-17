#pragma once

// Op wrappers, written once against the `ta::` compatibility layer
// (torch_ans/ta/torch_api.hpp): the ATen build today, the libtorch-stable-ABI
// build for the prebuilt wheel tomorrow.
// Nothing in this file may reach for ATen directly any more - the two
// `py::bytes` operators are the one exception, because the stable ABI has no
// schema type for them; they are behind `#ifndef TORCH_ANS_STABLE_ABI` and
// move to Python in stage 4 (migration section 1.5).
#include "ta/torch_api.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "rans_utils.hpp"
#include "rans_build_config.hpp"

// The worst-case stream growth every push shares (rANS and tANS alike); it used
// to be inline in rans_push below.
#include "stream_capacity.hpp"

// std::vector<DEFAULT_TORCH_TENSOR_TYPE> pmf_to_quantized_cdf(const std::vector<float> &pmf, int precision);

// template <size_t STATE_BITS, size_t STREAM_BITS>
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS=0>
ta::Tensor rans_init_stream(int64_t size, int64_t num_interleaves=1, int64_t preallocate_size=0);

// template <typename STATE_TYPE, typename STREAM_TYPE>
// std::vector<py::bytes> rans_flush_stream(// ANSStream stream,
//   ta::Tensor& stream);

#ifndef TORCH_ANS_STABLE_ABI
// pybind11-only: `std::vector<py::bytes>` has no stable-schema equivalent, so
// the stable build drops both operators and Python does the conversion instead
// (`torch_ans/_stable_C.py`, migration section 1.5). The ATen build keeps them
// until stage 4 switches the Python call sites over.
std::vector<py::bytes> rans_stream_to_byte_strings(const ta::Tensor& stream);

ta::Tensor rans_byte_strings_to_stream(std::vector<py::bytes> byte_strings);
#endif  // TORCH_ANS_STABLE_ABI

std::tuple<ta::Tensor, ta::Tensor> rans_alias_build_table(
  const ta::Tensor& cdfs, const ta::Tensor& cdfs_sizes,
  int64_t symbol_precision=8,
  int64_t freq_precision=16
);

#if TA_WITH_CUDA
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cuda(// ANSStream stream,
  ta::Tensor stream, 
  const ta::Tensor& symbols, 
  const ta::Tensor& indexes, 
  const ta::Tensor& cdfs, 
  const ta::Tensor& cdfs_sizes, 
  const ta::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
ta::Tensor rans_pop_indexed_cuda(// ANSStream stream,
  ta::Tensor stream, 
  const ta::Tensor& indexes, 
  const ta::Tensor& cdfs, 
  const ta::Tensor& cdfs_sizes, 
  const ta::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1);

  // Batched PMF to quantized CDF declarations
ta::Tensor rans_pmf_to_quantized_cdf_cuda(const ta::Tensor& pmf, int64_t precision);

// B2: fused "cdf ++ inverse-CDF table" construction (device side, one launch)
ta::Tensor rans_build_inverse_cdf_cuda(const ta::Tensor& cdfs, int64_t freq_precision,
                                       int64_t table_precision);

#endif

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cpu(// ANSStream stream,
  ta::Tensor stream, 
  const ta::Tensor& symbols, 
  const ta::Tensor& indexes, 
  const ta::Tensor& cdfs, 
  const ta::Tensor& cdfs_sizes, 
  const ta::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4);

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
ta::Tensor rans_pop_indexed_cpu(// ANSStream stream,
  ta::Tensor stream, 
  const ta::Tensor& indexes, 
  const ta::Tensor& cdfs, 
  const ta::Tensor& cdfs_sizes, 
  const ta::Tensor& offsets,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1);

// Batched PMF to quantized CDF declarations
ta::Tensor rans_pmf_to_quantized_cdf_cpu(const ta::Tensor& pmf, int64_t precision);

// B2: fused "cdf ++ inverse-CDF table" construction (host side)
ta::Tensor rans_build_inverse_cdf_cpu(const ta::Tensor& cdfs, int64_t freq_precision,
                                      int64_t table_precision);

// Encode `symbols` into `stream` (in place: `stream` is a `Tensor(a!)` argument,
// so the caller keeps using its own tensor object).
//
// `used_bytes` is the caller's upper bound on the bytes already coded into the
// longest row of `stream`; when it is given (>= 0) the caller also owns the
// capacity - the wrapper only checks it and throws instead of growing. Python
// passes it for every push it issues (see torch_ans/utils.py::_encode_func), so
// the hot path does not read the length word back (a blocking D2H on CUDA) and
// never calls `aten::resize_`, whose schema the stable ABI does not promise. The
// default -1 means "read the length word and grow the stream as needed", which is
// what a direct op call (tests, benchmarks, scripts) keeps getting.
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push(// ANSStream stream,
  ta::Tensor stream, 
  ta::Tensor symbols, 
  std::optional<ta::Tensor> indexes, 
  std::optional<ta::Tensor> cdfs, 
  std::optional<ta::Tensor> cdfs_sizes, 
  std::optional<ta::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t used_bytes=-1)
{
  const int64_t batch_size = stream.size(0);
  const int64_t num_symbols = symbols.size(1);
  const bool use_cuda = ta::is_cuda(stream);

  if (symbols.size(0) != batch_size) {
    // std::invalid_argument rather than STD_TORCH_CHECK: pybind11 maps it to
    // ValueError, which is what this call raised before the ta:: migration
    // (the callers in utils.py, and the tests, expect a ValueError).
    throw std::invalid_argument("Symbols batch size mismatch!");
  }

  // The worst case the stream must hold, and the compare/throw/grow that follows
  // from it, are shared with the tANS family: see stream_capacity.hpp for the
  // arithmetic (and torch_ans/utils.py::_stream_budget for the Python mirror
  // that reserves the capacity, pinned against it by
  // tests/test_capacity_contract.py).
  ans_ensure_stream_capacity<RANS_STATE_TYPE, RANS_STREAM_TYPE>(
      stream, num_symbols, freq_precision, bypass_coding, bypass_precision,
      NUM_INTERLEAVES, used_bytes);

  // indexed cdf
  if (indexes.has_value()) {

    STD_TORCH_CHECK(cdfs.has_value());
    STD_TORCH_CHECK(cdfs_sizes.has_value());

    ta::Tensor offsets_tensor;
    if (!offsets.has_value()) 
      offsets_tensor = ta::new_zeros_1d(indexes.value(), cdfs.value().size(0));
    else
      offsets_tensor = offsets.value();

    if (use_cuda) {
#if TA_WITH_CUDA
      // The tensors must live on the device `stream` lives on; `.cuda()` used
      // to mean "the current device", which is the same thing here because
      // use_cuda is derived from stream itself.
      const ta::Device device = ta::device_of(stream);
      return rans_push_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES>(
        ta::as_contiguous(ta::to(stream, device)), ta::as_contiguous(ta::to(symbols, device)), 
        ta::as_contiguous(ta::to(indexes.value(), device)), ta::as_contiguous(ta::to(cdfs.value(), device)), 
        ta::as_contiguous(ta::to(cdfs_sizes.value(), device)), ta::as_contiguous(ta::to(offsets_tensor, device)), 
        freq_precision, bypass_coding, bypass_precision);
#else
      STD_TORCH_CHECK(false, "torch_ans is not compiled with GPU support!");
#endif
    }

    return rans_push_indexed_cpu<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, NUM_INTERLEAVES>(
      ta::as_contiguous(stream), ta::as_contiguous(symbols), 
      ta::as_contiguous(indexes.value()), ta::as_contiguous(cdfs.value()), 
      ta::as_contiguous(cdfs_sizes.value()), ta::as_contiguous(offsets_tensor), 
      freq_precision, bypass_coding, bypass_precision);

  }
  else {
    // per-symbol cdf
    if (cdfs.has_value()) {
      // STD_TORCH_CHECK(cdfs_sizes.has_value(), #cdfs_sizes " must exist");
      // STD_TORCH_CHECK(offsets.has_value(), #offsets " must exist");

      // auto cdfs_accessor = ta::view2<DEFAULT_TORCH_TENSOR_TYPE>(cdfs.value());
      // auto cdfs_sizes_accessor = ta::view1<DEFAULT_TORCH_TENSOR_TYPE>(cdfs_sizes.value());
      // auto offsets_accessor = ta::view1<DEFAULT_TORCH_TENSOR_TYPE>(offsets.value());

      // ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      //     for (int64_t b = start; b < end; b++) {
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
      // ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      //     for (int64_t b = start; b < end; b++) {
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
ta::Tensor rans_pop(// ANSStream stream,
  ta::Tensor stream, 
  std::optional<ta::Tensor> indexes, 
  std::optional<ta::Tensor> cdfs, 
  std::optional<ta::Tensor> cdfs_sizes, 
  std::optional<ta::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true, 
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1)
{
  ta::Tensor symbols;
  const bool use_cuda = ta::is_cuda(stream);



  // indexed cdf
  if (indexes.has_value()) {

    STD_TORCH_CHECK(cdfs.has_value());
    STD_TORCH_CHECK(cdfs_sizes.has_value());

    ta::Tensor offsets_tensor;
    if (!offsets.has_value()) 
      offsets_tensor = ta::new_zeros_1d(indexes.value(), cdfs.value().size(0));
    else
      offsets_tensor = offsets.value();

    if (use_cuda) {
#if TA_WITH_CUDA
      const ta::Device device = ta::device_of(stream);
      return rans_pop_indexed_cuda<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
        ta::as_contiguous(ta::to(stream, device)),
        ta::as_contiguous(ta::to(indexes.value(), device)), ta::as_contiguous(ta::to(cdfs.value(), device)), 
        ta::as_contiguous(ta::to(cdfs_sizes.value(), device)), ta::as_contiguous(ta::to(offsets_tensor, device)), 
        freq_precision, bypass_coding, bypass_precision, inverse_cdf_precision);
#else
      STD_TORCH_CHECK(false, "torch_ans is not compiled with GPU support!");
#endif
    }

    return rans_pop_indexed_cpu<RANS_STATE_TYPE, RANS_STREAM_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, NUM_INTERLEAVES>(
        ta::as_contiguous(stream),
        ta::as_contiguous(indexes.value()), ta::as_contiguous(cdfs.value()), 
        ta::as_contiguous(cdfs_sizes.value()), ta::as_contiguous(offsets_tensor), 
        freq_precision, bypass_coding, bypass_precision, inverse_cdf_precision);

  }
  else {
    // TODO:
    // throw std::invalid_argument("");
  }
  // TODO: shrink stream

  return symbols;

}


inline ta::Tensor rans_pmf_to_quantized_cdf(const ta::Tensor& pmfs, int64_t precision) {
  if (ta::is_cuda(pmfs)) {
#if TA_WITH_CUDA
    return rans_pmf_to_quantized_cdf_cuda(pmfs, precision);
#else
    STD_TORCH_CHECK(false, "torch_ans is not compiled with GPU support!");
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
inline ta::Tensor rans_build_inverse_cdf(const ta::Tensor& cdfs, int64_t freq_precision,
                                         int64_t table_precision) {
  if (ta::is_cuda(cdfs)) {
#if TA_WITH_CUDA
    return rans_build_inverse_cdf_cuda(cdfs, freq_precision, table_precision);
#else
    STD_TORCH_CHECK(false, "torch_ans is not compiled with GPU support!");
#endif
  } else {
    return rans_build_inverse_cdf_cpu(cdfs, freq_precision, table_precision);
  }
}



// NOTE: the pybind11 bindings used to be defined here as the
// TORCH_EXTENSION_RANS_BINDINGS(m) macro. They moved to rans_bindings.hpp,
// where they are a regular function and the operator set can be gated by the
// feature macros in rans_build_config.hpp (a macro body cannot contain #if).
