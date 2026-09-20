#pragma once

// ---------------------------------------------------------------------------
// Compile-time operator selection for the native torch_ans extension.
//
// A normal build (install-time `pip install .`, or the JIT "full" build used
// when dynamic compilation is disabled) compiles every operator. An
// incremental JIT build additionally defines TORCH_ANS_INCREMENTAL_BUILD plus
// one TORCH_ANS_WITH_* gate per feature an interface instance needs, so that
// e.g. `TorchANSInterface(impl="rans64")` only compiles the rans64 variants
// instead of every family x interleave x symbol-lookup combination. That is
// the difference between seconds and minutes of gcc/nvcc work on a low-end
// machine. The Python side of this lives in torch_ans/_dynamic_build.py
// (profile_defines / profile_module_name / profile_op_names).
//
// Gates (undefined means 1 unless TORCH_ANS_INCREMENTAL_BUILD is defined):
//   TORCH_ANS_WITH_RANS64 / _RANS32 / _RANS32_16
//       rANS state/stream variant family ("rans64", "rans32", "rans32_16")
//   TORCH_ANS_WITH_INTERLEAVE_2 / _4 / _8 / _32
//       interleaved push/pop (interleave 1 is the always-compiled core)
//   TORCH_ANS_WITH_ALIAS
//       alias-sampling push/pop (CPU only; the CUDA side only implements the
//       rans32_16 interleaved combination, as before)
//   TORCH_ANS_WITH_INVCDF
//       inverse-CDF table pop (`inverse_cdf_precision`)
//   TORCH_ANS_WITH_TANS32 / _TANS16 / _TANS8
//       tANS families (interleave 1 always, 2/4/8/32 behind the shared
//       TORCH_ANS_WITH_INTERLEAVE_* gates - 32 is the full-warp group of the CUDA
//       warp kernels, and it has CPU kernels too, like the rANS 32-way). The
//       three differ only in the state/stream width, which is also their
//       freq_precision cap:
//       tans32 <uint32, uint16> <= 15, tans16 <uint16, uint8> <= 15,
//       tans8 <uint8, uint8> <= 7. The table builder (tans_build_tables) is
//       *not* gated: all three share the same int32 tables - see
//       torch_ans/tans_utils.hpp - so it is part of the common operator set and
//       is compiled by every build. The gate covers the CPU *and* the CUDA
//       kernels: one thread codes a row at one lane, a lane group at several
//       (tans_warp_cuda.cuh).
//
// The gate names and the (family, interleave, lookup) combinations they cover
// are mirrored by the profile helpers in torch_ans/_dynamic_build.py; a gate
// the Python side never sets silently drops the corresponding operators from
// the module, which shows up as an AttributeError at the first use.
// ---------------------------------------------------------------------------

#if defined(TORCH_ANS_INCREMENTAL_BUILD)
#  ifndef TORCH_ANS_WITH_RANS64
#    define TORCH_ANS_WITH_RANS64 0
#  endif
#  ifndef TORCH_ANS_WITH_RANS32
#    define TORCH_ANS_WITH_RANS32 0
#  endif
#  ifndef TORCH_ANS_WITH_RANS32_16
#    define TORCH_ANS_WITH_RANS32_16 0
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_2
#    define TORCH_ANS_WITH_INTERLEAVE_2 0
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_4
#    define TORCH_ANS_WITH_INTERLEAVE_4 0
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_8
#    define TORCH_ANS_WITH_INTERLEAVE_8 0
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_32
#    define TORCH_ANS_WITH_INTERLEAVE_32 0
#  endif
#  ifndef TORCH_ANS_WITH_ALIAS
#    define TORCH_ANS_WITH_ALIAS 0
#  endif
#  ifndef TORCH_ANS_WITH_INVCDF
#    define TORCH_ANS_WITH_INVCDF 0
#  endif
#  ifndef TORCH_ANS_WITH_TANS32
#    define TORCH_ANS_WITH_TANS32 0
#  endif
#  ifndef TORCH_ANS_WITH_TANS16
#    define TORCH_ANS_WITH_TANS16 0
#  endif
#  ifndef TORCH_ANS_WITH_TANS8
#    define TORCH_ANS_WITH_TANS8 0
#  endif
#else
#  ifndef TORCH_ANS_WITH_RANS64
#    define TORCH_ANS_WITH_RANS64 1
#  endif
#  ifndef TORCH_ANS_WITH_RANS32
#    define TORCH_ANS_WITH_RANS32 1
#  endif
#  ifndef TORCH_ANS_WITH_RANS32_16
#    define TORCH_ANS_WITH_RANS32_16 1
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_2
#    define TORCH_ANS_WITH_INTERLEAVE_2 1
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_4
#    define TORCH_ANS_WITH_INTERLEAVE_4 1
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_8
#    define TORCH_ANS_WITH_INTERLEAVE_8 1
#  endif
#  ifndef TORCH_ANS_WITH_INTERLEAVE_32
#    define TORCH_ANS_WITH_INTERLEAVE_32 1
#  endif
#  ifndef TORCH_ANS_WITH_ALIAS
#    define TORCH_ANS_WITH_ALIAS 1
#  endif
#  ifndef TORCH_ANS_WITH_INVCDF
#    define TORCH_ANS_WITH_INVCDF 1
#  endif
#  ifndef TORCH_ANS_WITH_TANS32
#    define TORCH_ANS_WITH_TANS32 1
#  endif
#  ifndef TORCH_ANS_WITH_TANS16
#    define TORCH_ANS_WITH_TANS16 1
#  endif
#  ifndef TORCH_ANS_WITH_TANS8
#    define TORCH_ANS_WITH_TANS8 1
#  endif
#endif

// ---------------------------------------------------------------------------
// Explicit instantiation helpers.
//
// The push/pop/init implementations are defined in rans_cpu.cpp / rans_cuda.cu
// while the pybind wrappers (ans_bindings.hpp) only see their declarations,
// so every bound combination needs an explicit instantiation in the
// translation unit that defines it. This used to be implied by the
// TORCH_LIBRARY_IMPL blocks taking the address of each specialization. Those
// blocks registered nothing observable (torch.ops.torch_ans was always empty,
// there is no TORCH_LIBRARY schema), so they are gone; spelled out here the
// instantiation also survives -O3, which the address-of-an-unused-static idiom
// does not (gcc drops both the variable and the instantiation).
//
// The signatures below are spelled in ta::Tensor (the compatibility layer's
// tensor alias) instead of torch::Tensor, so that the same instantiation macros
// serve the ATen build and the libtorch-stable-ABI build. ta::Tensor *is*
// at::Tensor in the ATen build, so this is not a type change there. The layer
// is pulled in here rather than assumed, because rans_cuda.cu includes this
// header directly.
// ---------------------------------------------------------------------------

#include "ta/torch_api.hpp"

#define TORCH_ANS_INST_INIT(STATE, STREAM) \
  template ta::Tensor rans_init_stream<STATE, STREAM>(int64_t, int64_t, int64_t)

#define TORCH_ANS_INST_PUSH(FN, STATE, STREAM, ALIAS, ILV) \
  template void FN<STATE, STREAM, ALIAS, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, const ta::Tensor&, \
      int64_t, bool, int64_t)

#define TORCH_ANS_INST_POP(FN, STATE, STREAM, ALIAS, INV, ILV) \
  template ta::Tensor FN<STATE, STREAM, ALIAS, INV, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, int64_t, bool, int64_t, \
      int64_t)

// tANS: the family is the (state, stream) pair - see tans.hpp. The symbol lookup
// is the tANS table itself, so there is no alias or inverse-CDF parameter; the
// interleave factor is a template parameter like the rANS one (the per-lane
// states have to be compile-time indexed to stay in registers). The stream
// initializer takes `num_interleaves` as a runtime argument instead, so it is
// instantiated once per family.
#define TORCH_ANS_INST_TANS_INIT(STATE, STREAM) \
  template ta::Tensor tans_init_stream<STATE, STREAM>(int64_t, int64_t, int64_t)

#define TORCH_ANS_INST_TANS_PUSH(STATE, STREAM, ILV) \
  template void tans_push_indexed_cpu<STATE, STREAM, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, const ta::Tensor&, \
      int64_t, bool, int64_t)

#define TORCH_ANS_INST_TANS_POP(STATE, STREAM, ILV) \
  template ta::Tensor tans_pop_indexed_cpu<STATE, STREAM, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, int64_t, bool, int64_t)

// tANS on CUDA: the same signatures as the CPU ones above (the *launchers* in
// tans_cuda.cu), instantiated per interleave factor behind the same shared
// TORCH_ANS_WITH_INTERLEAVE_* gates the CPU side uses. The launchers route
// NUM_INTERLEAVES > 1 to the warp-level kernels in tans_warp_cuda.cuh, whose
// definitions are visible there - so instantiating the caller instantiates the
// callee and no separate instantiation macro is needed for them.
#define TORCH_ANS_INST_TANS_PUSH_CUDA(STATE, STREAM, ILV) \
  template void tans_push_indexed_cuda<STATE, STREAM, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, const ta::Tensor&, \
      int64_t, bool, int64_t)

#define TORCH_ANS_INST_TANS_POP_CUDA(STATE, STREAM, ILV) \
  template ta::Tensor tans_pop_indexed_cuda<STATE, STREAM, ILV>( \
      ta::Tensor, const ta::Tensor&, const ta::Tensor&, \
      const ta::Tensor&, const ta::Tensor&, int64_t, bool, int64_t)
