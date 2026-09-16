#pragma once

// Torch-free stand-ins for at::TensorAccessor / torch::PackedTensorAccessor32.
//
// The libtorch stable ABI has no accessor type: `torch::stable::Tensor` only
// exposes a data pointer plus sizes/strides (and no `operator[]`, because a
// stable ABI cannot hand out a reference into a TensorImpl). These two POD
// views are what the kernels get instead. They are
//
//   * trivially copyable, so they can be passed **by value** into a CUDA kernel
//     exactly like `PackedTensorAccessor32` is today, and
//   * built from raw int64_t strides, so the index arithmetic below constant
//     folds. That matters: the interleaved CPU kernels keep their rANS states
//     in registers and must not gain a runtime-indexed array access or an
//     indirection through a spilled view object (see AGENTS.md).
//
// Both backends use the *same* struct: ATen today, torch::stable under
// TORCH_ANS_STABLE_ABI. That is what keeps the CPU and the CUDA kernels
// bit-identical between the two build modes, and it is why `.accessor<>()` /
// `.packed_accessor32<>()` can be replaced mechanically at every call site
// (see STABLE_ABI_MIGRATION.md sections 2.3 / 2.4 / 2.5).
//
// The name of the file is stable history: it started as the POC's
// `poc/stable_abi/csrc/poc_view.hpp`, whose structs are the same ones.

#include <cstdint>

// The CPU build compiles this header with g++ and never sees __CUDACC__; in a
// .cu the accessors are the *only* thing a kernel calls into, so they have to
// be callable from device code (and from the host wrapper, hence __host__ too).
// Forgetting this is 80x "calling a __host__ function from a __global__
// function" (STABLE_ABI_MIGRATION.md section 4, pit 2).
#if defined(__CUDACC__)
#define TA_VIEW_API __host__ __device__
#else
#define TA_VIEW_API
#endif

namespace ta {

template <typename T>
struct View1 {
  T* data;
  int64_t numel;
  int64_t stride;

  TA_VIEW_API T* ptr() const { return data; }
  TA_VIEW_API T& at(int64_t i) const { return data[i * stride]; }
  TA_VIEW_API T& operator[](int64_t i) const { return at(i); }
};

template <typename T>
struct View2 {
  T* data;
  int64_t size0;
  int64_t size1;
  int64_t stride0;
  int64_t stride1;

  TA_VIEW_API T* row(int64_t i) const { return data + i * stride0; }
  TA_VIEW_API T& at(int64_t i, int64_t j) const {
    return data[i * stride0 + j * stride1];
  }
};

}  // namespace ta
