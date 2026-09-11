#pragma once

// pybind11 bindings of the native operators.
//
// This supersedes the TORCH_EXTENSION_RANS_BINDINGS macro that used to live in
// rans.hpp. It is a regular (inline) function rather than a macro because the
// bound operator set is gated by the feature macros in rans_build_config.hpp:
// the gates are plain #if blocks, which a macro body cannot contain. See
// torch_ans/_dynamic_build.py for how the Python side selects a subset.
//
// The pybind argument lists are factored into macros so that the gated list
// stays readable, and the push/pop/invcdf bindings are factored into
// TORCH_ANS_BIND_PUSH_POP / TORCH_ANS_BIND_INVCDF so each gated block below
// is a couple of lines.

#include "rans.hpp"
#include "rans_build_config.hpp"

#define TORCH_ANS_INIT_ARGS \
  py::arg("size"), py::arg("num_interleaves") = 1, py::arg("preallocate_size") = 0

#define TORCH_ANS_PUSH_ARGS \
  py::arg("stream"), py::arg("symbols"), py::arg("indexes") = py::none(), \
  py::arg("cdfs") = py::none(), py::arg("cdfs_sizes") = py::none(), \
  py::arg("offsets") = py::none(), py::arg("symbol_precision") = 8, \
  py::arg("freq_precision") = 16, py::arg("bypass_coding") = true, \
  py::arg("bypass_precision") = 4

#define TORCH_ANS_POP_ARGS \
  py::arg("stream"), py::arg("indexes") = py::none(), \
  py::arg("cdfs") = py::none(), py::arg("cdfs_sizes") = py::none(), \
  py::arg("offsets") = py::none(), py::arg("symbol_precision") = 8, \
  py::arg("freq_precision") = 16, py::arg("bypass_coding") = true, \
  py::arg("bypass_precision") = 4, py::arg("inverse_cdf_precision") = -1

#define TORCH_ANS_BIND_INIT(m, PREFIX, STATE, STREAM) \
  m.def(PREFIX "_init_stream", &rans_init_stream<STATE, STREAM>, TORCH_ANS_INIT_ARGS)

#define TORCH_ANS_BIND_PUSH_POP(m, PREFIX, STATE, STREAM, ALIAS, ILV) \
  m.def(PREFIX "_push", &rans_push<STATE, STREAM, ALIAS, ILV>, TORCH_ANS_PUSH_ARGS); \
  m.def(PREFIX "_pop", &rans_pop<STATE, STREAM, ALIAS, false, ILV>, TORCH_ANS_POP_ARGS)

#define TORCH_ANS_BIND_INVCDF(m, PREFIX, STATE, STREAM, ILV) \
  m.def(PREFIX "_invcdf_pop", &rans_pop<STATE, STREAM, false, true, ILV>, \
        TORCH_ANS_POP_ARGS)

inline void torch_ans_bind_all(py::module& m) {
  // ---------------------------------------------------------------- common --
  // Not gated: the stream (de)serialization, the pmf -> quantized cdf
  // conversion, the inverse-CDF table builder and the alias table builder are
  // needed by every interface (and by the high-level API for any impl).
  m.def("rans_stream_to_byte_strings", &rans_stream_to_byte_strings);
  m.def("rans_byte_strings_to_stream", &rans_byte_strings_to_stream);
  m.def("rans_alias_build_table", &rans_alias_build_table,
        py::arg("cdfs"), py::arg("cdfs_sizes"),
        py::arg("symbol_precision") = 8, py::arg("freq_precision") = 16);
  m.def("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf,
        py::arg("pmfs"), py::arg("precision") = 16);
  m.def("rans_build_inverse_cdf", &rans_build_inverse_cdf,
        py::arg("cdfs"), py::arg("freq_precision") = 16,
        py::arg("table_precision") = 16);

  // --------------------------------------------------------------- rans64 --
  // 64-bit state, 32-bit stream words (freq_precision <= 31).
#if TORCH_ANS_WITH_RANS64
  TORCH_ANS_BIND_INIT(m, "rans64", uint64_t, uint32_t);
  TORCH_ANS_BIND_PUSH_POP(m, "rans64", uint64_t, uint32_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans64", uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_alias", uint64_t, uint32_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_i2", uint64_t, uint32_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans64_i2", uint64_t, uint32_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_alias_i2", uint64_t, uint32_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_i4", uint64_t, uint32_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans64_i4", uint64_t, uint32_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_alias_i4", uint64_t, uint32_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_i8", uint64_t, uint32_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans64_i8", uint64_t, uint32_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans64_alias_i8", uint64_t, uint32_t, true, 8);
#    endif
#  endif
#endif

  // --------------------------------------------------------------- rans32 --
  // 32-bit state, 8-bit stream words (freq_precision <= 23).
#if TORCH_ANS_WITH_RANS32
  TORCH_ANS_BIND_INIT(m, "rans32", uint32_t, uint8_t);
  TORCH_ANS_BIND_PUSH_POP(m, "rans32", uint32_t, uint8_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32", uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_alias", uint32_t, uint8_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_i2", uint32_t, uint8_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_i2", uint32_t, uint8_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_alias_i2", uint32_t, uint8_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_i4", uint32_t, uint8_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_i4", uint32_t, uint8_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_alias_i4", uint32_t, uint8_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_i8", uint32_t, uint8_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_i8", uint32_t, uint8_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_alias_i8", uint32_t, uint8_t, true, 8);
#    endif
#  endif
#endif

  // ------------------------------------------------------------ rans32_16 --
  // 32-bit state, 16-bit stream words (freq_precision <= 15). The only family
  // with a 32-way interleaved variant, which maps to the warp-level CUDA
  // kernels (`num_interleaves=32` is CUDA-only in practice).
#if TORCH_ANS_WITH_RANS32_16
  TORCH_ANS_BIND_INIT(m, "rans32_16", uint32_t, uint16_t);
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16", uint32_t, uint16_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_16", uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_alias", uint32_t, uint16_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_i2", uint32_t, uint16_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_16_i2", uint32_t, uint16_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_alias_i2", uint32_t, uint16_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_i4", uint32_t, uint16_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_16_i4", uint32_t, uint16_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_alias_i4", uint32_t, uint16_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_i8", uint32_t, uint16_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_16_i8", uint32_t, uint16_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_alias_i8", uint32_t, uint16_t, true, 8);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
  TORCH_ANS_BIND_PUSH_POP(m, "rans32_16_i32", uint32_t, uint16_t, false, 32);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_BIND_INVCDF(m, "rans32_16_i32", uint32_t, uint16_t, 32);
#    endif
#  endif
#endif
}
