#pragma once

// Registration of the native operators, in the two flavours the package ships:
//
//   * `torch_ans_bind_all(py::module&)` - the pybind11 bindings used by the
//     ATen build (install-time or runtime JIT). This supersedes the
//     TORCH_EXTENSION_RANS_BINDINGS macro that used to live in rans.hpp, and it
//     is a regular (inline) function rather than a macro because the bound
//     operator set is gated by the feature macros in rans_build_config.hpp: the
//     gates are plain #if blocks, which a macro body cannot contain. See
//     torch_ans/_dynamic_build.py for how the Python side selects a subset.
//
//   * `torch_ans_register_schemas` / `..._init_impls` / `..._family_impls` /
//     `..._cpu_impls` / `..._cuda_impls` - the libtorch-stable-ABI registration
//     (schema strings + boxed kernels) used by the prebuilt wheel. It is split
//     because the schema may
//     be declared exactly once in the process: the CPU .so (lib.cpp) declares
//     them and registers schemas + catch-all + CPU impls, while the CUDA .so
//     (lib_cuda.cpp) registers implementations only (migration section 1.2).
//     `..._family_impls` is the list both dispatch keys share, so their operator
//     sets cannot drift apart.
//
// Both lists carry the same operator set with the same gating; the schema
// default values below are copied character for character from the py::arg
// lists, and tests/test_stable_abi.py compares the registered schemas against
// _dynamic_build.profile_op_names(None) so the two cannot drift apart silently.
// The `py::bytes` pair is deliberately absent from the stable list: the stable
// ABI has no schema type for it, Python implements it (torch_ans/_stable_C.py).

#include "rans.hpp"
#include "rans_build_config.hpp"
#include "tans.hpp"

#ifdef TORCH_ANS_STABLE_ABI

// ---------------------------------------------------------------------------
// schema strings
// ---------------------------------------------------------------------------
// Default values are spelled exactly like the py::arg ones below, including the
// historical `freq_precision=16`, which is out of range for rans32_16 (its
// upper bound is 15). utils.py validates freq_precision per impl before the
// call, so changing the default here would silently move that boundary; the
// migration note (section 4, pit 3) is where the cleanup is tracked.

#define TORCH_ANS_SCHEMA_INIT(PREFIX) \
  PREFIX "_init_stream(int size, int num_interleaves=1, int preallocate_size=0) -> Tensor"

#define TORCH_ANS_SCHEMA_PUSH(PREFIX) \
  PREFIX "_push(Tensor(a!) stream, Tensor symbols, Tensor? indexes=None, " \
  "Tensor? cdfs=None, Tensor? cdfs_sizes=None, Tensor? offsets=None, " \
  "int symbol_precision=8, int freq_precision=16, bool bypass_coding=True, " \
  "int bypass_precision=4, int used_bytes=-1) -> ()"

#define TORCH_ANS_SCHEMA_POP(PREFIX) \
  PREFIX "_pop(Tensor(a!) stream, Tensor? indexes=None, Tensor? cdfs=None, " \
  "Tensor? cdfs_sizes=None, Tensor? offsets=None, int symbol_precision=8, " \
  "int freq_precision=16, bool bypass_coding=True, int bypass_precision=4, " \
  "int inverse_cdf_precision=-1) -> Tensor"

#define TORCH_ANS_SCHEMA_INVCDF_POP(PREFIX) \
  PREFIX "_invcdf_pop(Tensor(a!) stream, Tensor? indexes=None, Tensor? cdfs=None, " \
  "Tensor? cdfs_sizes=None, Tensor? offsets=None, int symbol_precision=8, " \
  "int freq_precision=16, bool bypass_coding=True, int bypass_precision=4, " \
  "int inverse_cdf_precision=-1) -> Tensor"

#define TORCH_ANS_DEF_INIT(m, PREFIX) m.def(TORCH_ANS_SCHEMA_INIT(PREFIX))
#define TORCH_ANS_DEF_PUSH_POP(m, PREFIX) \
  m.def(TORCH_ANS_SCHEMA_PUSH(PREFIX)); m.def(TORCH_ANS_SCHEMA_POP(PREFIX))
#define TORCH_ANS_DEF_INVCDF(m, PREFIX) m.def(TORCH_ANS_SCHEMA_INVCDF_POP(PREFIX))

// ---------------------------------------------------------------------------
// boxed implementations
// ---------------------------------------------------------------------------
// `TORCH_BOX(&fn)` turns the typed function into the `(StableIValue*, uint64_t,
// uint64_t)` calling convention `StableLibrary::impl` wants; the template
// specializations are the same ones the explicit-instantiation blocks in
// rans_cpu.cpp / rans_cuda.cu provide.

// NOTE: the extra parentheses around the template-id are required. TORCH_BOX
// is a function-like macro and the preprocessor does not understand `<...>`, so
// without them the comma between the template arguments would be read as a
// macro-argument separator ("macro TORCH_BOX passed 2 arguments, but takes
// just 1").
#define TORCH_ANS_IMPL_INIT(m, PREFIX, STATE, STREAM) \
  m.impl(PREFIX "_init_stream", TORCH_BOX((&rans_init_stream<STATE, STREAM>)))

#define TORCH_ANS_IMPL_PUSH_POP(m, PREFIX, STATE, STREAM, ALIAS, ILV) \
  m.impl(PREFIX "_push", TORCH_BOX((&rans_push<STATE, STREAM, ALIAS, ILV>))); \
  m.impl(PREFIX "_pop", TORCH_BOX((&rans_pop<STATE, STREAM, ALIAS, false, ILV>)))

#define TORCH_ANS_IMPL_INVCDF(m, PREFIX, STATE, STREAM, ILV) \
  m.impl(PREFIX "_invcdf_pop", \
         TORCH_BOX((&rans_pop<STATE, STREAM, false, true, ILV>)))

// tANS: the family *is* the (state, stream) pair and the symbol lookup is its
// own table, so the only other template argument is the interleave factor (see
// tans.hpp). The schema strings are the rANS ones on purpose - identical
// signatures are what let Python switch impl without touching a call site.
#define TORCH_ANS_TANS_IMPL_INIT(m, PREFIX, STATE, STREAM) \
  m.impl(PREFIX "_init_stream", TORCH_BOX((&tans_init_stream<STATE, STREAM>)))

#define TORCH_ANS_TANS_IMPL_PUSH_POP(m, PREFIX, STATE, STREAM) \
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, PREFIX, STATE, STREAM, 1)

#define TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, PREFIX, STATE, STREAM, ILV) \
  m.impl(PREFIX "_push", TORCH_BOX((&tans_push<STATE, STREAM, ILV>))); \
  m.impl(PREFIX "_pop", TORCH_BOX((&tans_pop<STATE, STREAM, ILV>)))

// ---------------------------------------------------------------------------
// registration
// ---------------------------------------------------------------------------
// Called from lib.cpp: the schemas from a STABLE_TORCH_LIBRARY(torch_ans) block,
// the implementations from STABLE_TORCH_LIBRARY_IMPL blocks (`init_stream` needs
// the catch-all key because it takes no Tensor argument - migration section 4,
// pit 9).
inline void torch_ans_register_schemas(
    torch::stable::detail::StableLibrary& m) {
  // ---------------------------------------------------------------- common --
  // Not gated: the alias table, the pmf -> quantized cdf conversion and the
  // inverse-CDF table builder are needed by every interface.
  m.def("rans_alias_build_table(Tensor cdfs, Tensor cdfs_sizes, "
        "int symbol_precision=8, int freq_precision=16) -> (Tensor, Tensor)");
  m.def("rans_pmf_to_quantized_cdf(Tensor pmfs, int precision=16) -> Tensor");
  m.def("rans_build_inverse_cdf(Tensor cdfs, int freq_precision=16, "
        "int table_precision=16) -> Tensor");
  // tANS builds both directions of its table in one host-side op; the three
  // tANS families share the result (tans_utils.hpp), so it is not gated.
  m.def("tans_build_tables(Tensor cdfs, Tensor cdfs_sizes, int table_log) -> Tensor");

  // --------------------------------------------------------------- rans64 --
  // 64-bit state, 32-bit stream words (freq_precision <= 31).
#if TORCH_ANS_WITH_RANS64
  TORCH_ANS_DEF_INIT(m, "rans64");
  TORCH_ANS_DEF_PUSH_POP(m, "rans64");
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans64");
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_alias");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_i2");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans64_i2");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_alias_i2");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_i4");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans64_i4");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_alias_i4");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_i8");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans64_i8");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans64_alias_i8");
#    endif
#  endif
#endif

  // --------------------------------------------------------------- rans32 --
  // 32-bit state, 8-bit stream words (freq_precision <= 23).
#if TORCH_ANS_WITH_RANS32
  TORCH_ANS_DEF_INIT(m, "rans32");
  TORCH_ANS_DEF_PUSH_POP(m, "rans32");
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32");
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_alias");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_i2");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_i2");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_alias_i2");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_i4");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_i4");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_alias_i4");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_i8");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_i8");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_alias_i8");
#    endif
#  endif
#endif

  // ------------------------------------------------------------ rans32_16 --
  // 32-bit state, 16-bit stream words (freq_precision <= 15). The only family
  // with a 32-way interleaved variant; 32-way is CUDA-only in practice.
#if TORCH_ANS_WITH_RANS32_16
  TORCH_ANS_DEF_INIT(m, "rans32_16");
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16");
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_16");
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_alias");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_i2");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_16_i2");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_alias_i2");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_i4");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_16_i4");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_alias_i4");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_i8");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_16_i8");
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_alias_i8");
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
  TORCH_ANS_DEF_PUSH_POP(m, "rans32_16_i32");
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_DEF_INVCDF(m, "rans32_16_i32");
#    endif
#  endif
#endif

  // ----------------------------------------------------------------- tANS --
  // tabled ANS, CPU only: one gate per (state, stream) pair, and the interleave
  // factors behind the gates the rANS families share (no 32-way - that one is
  // the CUDA warp path). The signatures are the rANS ones (see tans.hpp); only
  // the family is new.
#if TORCH_ANS_WITH_TANS32
  // 32-bit state, 16-bit stream words (freq_precision <= 15).
  TORCH_ANS_DEF_INIT(m, "tans32");
  TORCH_ANS_DEF_PUSH_POP(m, "tans32");
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "tans32_i2");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "tans32_i4");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "tans32_i8");
#  endif
#endif
#if TORCH_ANS_WITH_TANS16
  // 16-bit state, 8-bit stream words (freq_precision <= 15); the same tables as
  // tans32, a finer stream granularity.
  TORCH_ANS_DEF_INIT(m, "tans16");
  TORCH_ANS_DEF_PUSH_POP(m, "tans16");
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "tans16_i2");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "tans16_i4");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "tans16_i8");
#  endif
#endif
#if TORCH_ANS_WITH_TANS8
  // 8-bit state, 8-bit stream words (freq_precision <= 7): the small-table tier.
  TORCH_ANS_DEF_INIT(m, "tans8");
  TORCH_ANS_DEF_PUSH_POP(m, "tans8");
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_DEF_PUSH_POP(m, "tans8_i2");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_DEF_PUSH_POP(m, "tans8_i4");
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_DEF_PUSH_POP(m, "tans8_i8");
#  endif
#endif
}

// `init_stream` takes no Tensor, so the dispatcher has no backend key to derive
// and it has to be registered for the catch-all key (CompositeExplicitAutograd).
inline void torch_ans_register_init_impls(
    torch::stable::detail::StableLibrary& m) {
#if TORCH_ANS_WITH_RANS64
  TORCH_ANS_IMPL_INIT(m, "rans64", uint64_t, uint32_t);
#endif
#if TORCH_ANS_WITH_RANS32
  TORCH_ANS_IMPL_INIT(m, "rans32", uint32_t, uint8_t);
#endif
#if TORCH_ANS_WITH_RANS32_16
  TORCH_ANS_IMPL_INIT(m, "rans32_16", uint32_t, uint16_t);
#endif
#if TORCH_ANS_WITH_TANS32
  TORCH_ANS_TANS_IMPL_INIT(m, "tans32", uint32_t, uint16_t);
#endif
#if TORCH_ANS_WITH_TANS16
  TORCH_ANS_TANS_IMPL_INIT(m, "tans16", uint16_t, uint8_t);
#endif
#if TORCH_ANS_WITH_TANS8
  TORCH_ANS_TANS_IMPL_INIT(m, "tans8", uint8_t, uint8_t);
#endif
}

// ---------------------------------------------------------------------------
// the family x interleave implementations
// ---------------------------------------------------------------------------
// One list for both dispatch keys: rans_push / rans_pop (rans.hpp) inspect the
// tensor device and call the CPU or the CUDA kernel themselves, so the
// dispatcher only has to pick the key - the operator names, and therefore the
// Python side, are identical on CPU and CUDA. Registering the very same names
// under both keys is what lib_cuda.cpp does for the second .so.
//
// The gate structure has to stay a plain `#if` list (a macro body cannot contain
// preprocessor conditionals), which is why this is a function and not a macro.
inline void torch_ans_register_family_impls(
    torch::stable::detail::StableLibrary& m) {

  // --------------------------------------------------------------- rans64 --
#if TORCH_ANS_WITH_RANS64
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64", uint64_t, uint32_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans64", uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_alias", uint64_t, uint32_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_i2", uint64_t, uint32_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans64_i2", uint64_t, uint32_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_alias_i2", uint64_t, uint32_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_i4", uint64_t, uint32_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans64_i4", uint64_t, uint32_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_alias_i4", uint64_t, uint32_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_i8", uint64_t, uint32_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans64_i8", uint64_t, uint32_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans64_alias_i8", uint64_t, uint32_t, true, 8);
#    endif
#  endif
#endif

  // --------------------------------------------------------------- rans32 --
#if TORCH_ANS_WITH_RANS32
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32", uint32_t, uint8_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32", uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_alias", uint32_t, uint8_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_i2", uint32_t, uint8_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_i2", uint32_t, uint8_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_alias_i2", uint32_t, uint8_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_i4", uint32_t, uint8_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_i4", uint32_t, uint8_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_alias_i4", uint32_t, uint8_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_i8", uint32_t, uint8_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_i8", uint32_t, uint8_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_alias_i8", uint32_t, uint8_t, true, 8);
#    endif
#  endif
#endif

  // ------------------------------------------------------------ rans32_16 --
#if TORCH_ANS_WITH_RANS32_16
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16", uint32_t, uint16_t, false, 1);
#  if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_16", uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_alias", uint32_t, uint16_t, true, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_i2", uint32_t, uint16_t, false, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_16_i2", uint32_t, uint16_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_alias_i2", uint32_t, uint16_t, true, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_i4", uint32_t, uint16_t, false, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_16_i4", uint32_t, uint16_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_alias_i4", uint32_t, uint16_t, true, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_i8", uint32_t, uint16_t, false, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_16_i8", uint32_t, uint16_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_alias_i8", uint32_t, uint16_t, true, 8);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
  TORCH_ANS_IMPL_PUSH_POP(m, "rans32_16_i32", uint32_t, uint16_t, false, 32);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_IMPL_INVCDF(m, "rans32_16_i32", uint32_t, uint16_t, 32);
#    endif
#  endif
#endif

  // ----------------------------------------------------------------- tANS --
  // The wrappers themselves reject CUDA tensors with an explanation, so they are
  // registered for the CUDA key as well: that way a device tensor produces our
  // message instead of the dispatcher's bare "not implemented for CUDA".
#if TORCH_ANS_WITH_TANS32
  TORCH_ANS_TANS_IMPL_PUSH_POP(m, "tans32", uint32_t, uint16_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans32_i2", uint32_t, uint16_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans32_i4", uint32_t, uint16_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans32_i8", uint32_t, uint16_t, 8);
#  endif
#endif
#if TORCH_ANS_WITH_TANS16
  TORCH_ANS_TANS_IMPL_PUSH_POP(m, "tans16", uint16_t, uint8_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans16_i2", uint16_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans16_i4", uint16_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans16_i8", uint16_t, uint8_t, 8);
#  endif
#endif
#if TORCH_ANS_WITH_TANS8
  TORCH_ANS_TANS_IMPL_PUSH_POP(m, "tans8", uint8_t, uint8_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans8_i2", uint8_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans8_i4", uint8_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_IMPL_PUSH_POP_ILV(m, "tans8_i8", uint8_t, uint8_t, 8);
#  endif
#endif
}

// ---------------------------------------------------------------------------
// per-dispatch-key registration
// ---------------------------------------------------------------------------
// The CPU library (lib.cpp) is the only one that declares schemas - a second
// `def()` of the same name+overload is rejected by the dispatcher - and it
// registers the whole operator set. The CUDA library (lib_cuda.cpp) registers
// implementations only, for the family list above plus the two common
// operators that have CUDA kernels.
//
// `rans_alias_build_table` is deliberately absent from the CUDA list: it is
// host-side code (rans.cpp) with no CUDA kernel, and its input is a CPU tensor
// by design (utils.py builds the alias table before moving anything to the
// device), so it stays a CPU-key operator.
inline void torch_ans_register_cpu_impls(
    torch::stable::detail::StableLibrary& m) {
  // ---------------------------------------------------------------- common --
  m.impl("rans_alias_build_table", TORCH_BOX(&rans_alias_build_table));
  m.impl("rans_pmf_to_quantized_cdf", TORCH_BOX(&rans_pmf_to_quantized_cdf));
  m.impl("rans_build_inverse_cdf", TORCH_BOX(&rans_build_inverse_cdf));
  // host-side, CPU tensors by design (like rans_alias_build_table): deliberately
  // absent from the CUDA list below.
  m.impl("tans_build_tables", TORCH_BOX(&tans_build_tables));

  torch_ans_register_family_impls(m);
}

#if TA_WITH_CUDA
// Both wrappers dispatch on ta::is_cuda() internally (rans.hpp), so the same
// function pointers serve the CUDA key; there is no separate *_cuda operator
// name on the Python side.
inline void torch_ans_register_cuda_impls(
    torch::stable::detail::StableLibrary& m) {
  // ---------------------------------------------------------------- common --
  m.impl("rans_pmf_to_quantized_cdf", TORCH_BOX(&rans_pmf_to_quantized_cdf));
  m.impl("rans_build_inverse_cdf", TORCH_BOX(&rans_build_inverse_cdf));

  torch_ans_register_family_impls(m);
}
#endif  // TA_WITH_CUDA

#undef TORCH_ANS_DEF_INIT
#undef TORCH_ANS_DEF_PUSH_POP
#undef TORCH_ANS_DEF_INVCDF
#undef TORCH_ANS_IMPL_INIT
#undef TORCH_ANS_IMPL_PUSH_POP
#undef TORCH_ANS_IMPL_INVCDF
#undef TORCH_ANS_TANS_IMPL_INIT
#undef TORCH_ANS_TANS_IMPL_PUSH_POP
#undef TORCH_ANS_TANS_IMPL_PUSH_POP_ILV

#else  // TORCH_ANS_STABLE_ABI

// ---------------------------------------------------------------------------
// pybind11 bindings (ATen build)
// ---------------------------------------------------------------------------
// The pybind argument lists are factored into macros so that the gated list
// stays readable, and the push/pop/invcdf bindings are factored into
// TORCH_ANS_BIND_PUSH_POP / TORCH_ANS_BIND_INVCDF so each gated block below
// is a couple of lines.

#define TORCH_ANS_INIT_ARGS \
  py::arg("size"), py::arg("num_interleaves") = 1, py::arg("preallocate_size") = 0

#define TORCH_ANS_PUSH_ARGS \
  py::arg("stream"), py::arg("symbols"), py::arg("indexes") = py::none(), \
  py::arg("cdfs") = py::none(), py::arg("cdfs_sizes") = py::none(), \
  py::arg("offsets") = py::none(), py::arg("symbol_precision") = 8, \
  py::arg("freq_precision") = 16, py::arg("bypass_coding") = true, \
  py::arg("bypass_precision") = 4, py::arg("used_bytes") = -1

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

// tANS: same argument lists as rANS on purpose (see tans.hpp - pop's
// inverse_cdf_precision is accepted and ignored), so the shared
// TORCH_ANS_INIT_ARGS / _PUSH_ARGS / _POP_ARGS are reused verbatim.
#define TORCH_ANS_TANS_BIND_INIT(m, PREFIX, STATE, STREAM) \
  m.def(PREFIX "_init_stream", &tans_init_stream<STATE, STREAM>, TORCH_ANS_INIT_ARGS)

#define TORCH_ANS_TANS_BIND_PUSH_POP(m, PREFIX, STATE, STREAM) \
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, PREFIX, STATE, STREAM, 1)

#define TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, PREFIX, STATE, STREAM, ILV) \
  m.def(PREFIX "_push", &tans_push<STATE, STREAM, ILV>, TORCH_ANS_PUSH_ARGS); \
  m.def(PREFIX "_pop", &tans_pop<STATE, STREAM, ILV>, TORCH_ANS_POP_ARGS)

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
  // tANS builds both directions of its table in one host-side op; the three
  // tANS families share the result (tans_utils.hpp), so it is not gated.
  m.def("tans_build_tables", &tans_build_tables,
        py::arg("cdfs"), py::arg("cdfs_sizes"), py::arg("table_log"));

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

  // ----------------------------------------------------------------- tANS --
  // tabled ANS, CPU only, interleave 1/2/4/8. One gate per (state, stream) pair;
  // no alias and no inverse-CDF variant (the tANS table *is* the lookup).
#if TORCH_ANS_WITH_TANS32
  // 32-bit state, 16-bit stream words (freq_precision <= 15).
  TORCH_ANS_TANS_BIND_INIT(m, "tans32", uint32_t, uint16_t);
  TORCH_ANS_TANS_BIND_PUSH_POP(m, "tans32", uint32_t, uint16_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans32_i2", uint32_t, uint16_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans32_i4", uint32_t, uint16_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans32_i8", uint32_t, uint16_t, 8);
#  endif
#endif
#if TORCH_ANS_WITH_TANS16
  // 16-bit state, 8-bit stream words (freq_precision <= 15): the same tables as
  // tans32, a finer stream granularity.
  TORCH_ANS_TANS_BIND_INIT(m, "tans16", uint16_t, uint8_t);
  TORCH_ANS_TANS_BIND_PUSH_POP(m, "tans16", uint16_t, uint8_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans16_i2", uint16_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans16_i4", uint16_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans16_i8", uint16_t, uint8_t, 8);
#  endif
#endif
#if TORCH_ANS_WITH_TANS8
  // 8-bit state, 8-bit stream words (freq_precision <= 7): the small-table tier.
  TORCH_ANS_TANS_BIND_INIT(m, "tans8", uint8_t, uint8_t);
  TORCH_ANS_TANS_BIND_PUSH_POP(m, "tans8", uint8_t, uint8_t);
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans8_i2", uint8_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans8_i4", uint8_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_TANS_BIND_PUSH_POP_ILV(m, "tans8_i8", uint8_t, uint8_t, 8);
#  endif
#endif
}

#undef TORCH_ANS_INIT_ARGS
#undef TORCH_ANS_PUSH_ARGS
#undef TORCH_ANS_POP_ARGS
#undef TORCH_ANS_BIND_INIT
#undef TORCH_ANS_BIND_PUSH_POP
#undef TORCH_ANS_BIND_INVCDF
#undef TORCH_ANS_TANS_BIND_INIT
#undef TORCH_ANS_TANS_BIND_PUSH_POP
#undef TORCH_ANS_TANS_BIND_PUSH_POP_ILV

#endif  // TORCH_ANS_STABLE_ABI
