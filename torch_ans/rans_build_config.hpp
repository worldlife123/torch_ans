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
//   TORCH_ANS_WITH_INVCDCDF
//       inverse-CDF table pop (`inverse_cdf_precision`)
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
#  ifndef TORCH_ANS_WITH_INVCDCDF
#    define TORCH_ANS_WITH_INVCDCDF 0
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
#  ifndef TORCH_ANS_WITH_INVCDCDF
#    define TORCH_ANS_WITH_INVCDCDF 1
#  endif
#endif

// ---------------------------------------------------------------------------
// Explicit instantiation helpers.
//
// The push/pop/init implementations are defined in rans_cpu.cpp / rans_cuda.cu
// while the pybind wrappers (rans_bindings.hpp) only see their declarations,
// so every bound combination needs an explicit instantiation in the
// translation unit that defines it. This used to be implied by the
// TORCH_LIBRARY_IMPL blocks taking the address of each specialization. Those
// blocks registered nothing observable (torch.ops.torch_ans was always empty,
// there is no TORCH_LIBRARY schema), so they are gone; spelled out here the
// instantiation also survives -O3, which the address-of-an-unused-static idiom
// does not (gcc drops both the variable and the instantiation).
//
// Requires <torch/extension.h> (for torch::Tensor) to be included first, which
// rans.hpp, rans_cpu.cpp and rans_cuda.cu all do.
// ---------------------------------------------------------------------------

#define TORCH_ANS_INST_INIT(STATE, STREAM) \
  template torch::Tensor rans_init_stream<STATE, STREAM>(int64_t, int64_t, int64_t)

#define TORCH_ANS_INST_PUSH(FN, STATE, STREAM, ALIAS, ILV) \
  template void FN<STATE, STREAM, ALIAS, ILV>( \
      torch::Tensor, const torch::Tensor&, const torch::Tensor&, \
      const torch::Tensor&, const torch::Tensor&, const torch::Tensor&, \
      int64_t, bool, int64_t)

#define TORCH_ANS_INST_POP(FN, STATE, STREAM, ALIAS, INV, ILV) \
  template torch::Tensor FN<STATE, STREAM, ALIAS, INV, ILV>( \
      torch::Tensor, const torch::Tensor&, const torch::Tensor&, \
      const torch::Tensor&, const torch::Tensor&, int64_t, bool, int64_t, \
      int64_t)
