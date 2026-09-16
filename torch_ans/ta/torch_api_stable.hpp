#pragma once

// stable-ABI backend of the `ta::` compatibility layer.
//
// Everything a torch_ans kernel or op wrapper needs is expressed here in terms
// of the libtorch stable ABI: only the C shim (`aoti_torch_*` / `torch_*`, all
// in libtorch_cpu.so) and header-only code (`torch/headeronly/**`). Not a single
// `at::` / `c10::` C++ symbol is referenced, which is what makes a `.so` built
// from this header loadable by any torch >= TORCH_TARGET_VERSION.
//
// Conventions (shared with the ATen backend, see torch_api_aten.hpp):
//   * `ta::` free functions instead of member calls wherever the two backends
//     name things differently (`data_ptr`, reductions, ...), so a call site is
//     migrated by adding a `ta::` prefix and nothing else;
//   * `STD_TORCH_CHECK` is the one check macro - it comes from
//     torch/headeronly/util/Exception.h (a macro in the global namespace) and
//     takes the same `(cond, msg...)` shape as `TORCH_CHECK`;
//   * `TA_DISPATCH_INTEGRAL` / `TA_DISPATCH_FLOATING` inject `scalar_t` exactly
//     like `AT_DISPATCH_INTEGRAL_TYPES` / `AT_DISPATCH_FLOATING_TYPES` do;
//   * every call from one `ta::` helper to another is written `ta::foo(...)`,
//     never `foo(...)`. `ta::Tensor` *is* `torch::stable::Tensor`, so an
//     unqualified call would also find the `torch::stable::` overload of the
//     same name through argument-dependent lookup and become ambiguous (the
//     same applies to `at::` in the ATen backend).
//
// This file is deliberately the only place that knows about the stable ABI; it
// was derived from (and is kept in sync with) `poc/stable_abi/csrc/
// poc_torch_api.hpp`, which is the version that is actually covered by tests.
// See STABLE_ABI_MIGRATION.md section 1.4.

// Lowest ABI we claim to support (torch 2.10). setup.py passes the same value
// as -D; this default only matters for ad-hoc compiles (`-fsyntax-only`).
#ifndef TORCH_TARGET_VERSION
#define TORCH_TARGET_VERSION (((0ULL + 2) << 56) | ((0ULL + 10) << 48))
#endif

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/stableivalue_conversions.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/Dispatch.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>
#include <torch/headeronly/util/Exception.h>
#include <torch/headeronly/util/HeaderOnlyArrayRef.h>

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

// torch/csrc/stable/c/shim.h only declares the CUDA entries of the C shim
// (`aoti_torch_get_current_cuda_stream`, ...) when USE_CUDA is defined, and it
// is the *only* way to ask torch for its current stream without touching
// `at::cuda::getCurrentCUDAStream()`, which is not a stable symbol. Nothing
// else in the stable headers keys off USE_CUDA. Miss it and you get
// "aoti_torch_get_current_cuda_stream was not declared" (migration section 4,
// pit 1).
#if defined(__CUDACC__) || defined(USE_CUDA)
#include <cuda_runtime.h>
#endif

#include "view.hpp"

namespace ta {

// ---------------------------------------------------------------------------
// backend types
// ---------------------------------------------------------------------------
using Tensor = torch::stable::Tensor;
using ScalarType = torch::headeronly::ScalarType;
using Device = torch::stable::Device;
using DeviceType = torch::headeronly::DeviceType;

// ---------------------------------------------------------------------------
// dtype dispatch: AT_DISPATCH_INTEGRAL_TYPES / AT_DISPATCH_FLOATING_TYPES
// ---------------------------------------------------------------------------
// torch/headeronly ships the very same machinery libtorch uses, minus the
// libtorch parts (THO_ = "torch header only"), so the existing dispatch sites
// keep working with a one-line rename. The case lists mirror ATen's:
// AT_DISPATCH_INTEGRAL_TYPES is Byte/Char/Short/Int/Long and
// AT_DISPATCH_FLOATING_TYPES is Double/Float (Half is a separate macro there).
#define TA_DISPATCH_INTEGRAL(TYPE, NAME, ...)                                 \
  THO_DISPATCH_SWITCH(                                                        \
      TYPE, NAME,                                                             \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Byte, __VA_ARGS__)      \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Char, __VA_ARGS__)      \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Short, __VA_ARGS__)     \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Int, __VA_ARGS__)       \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Long, __VA_ARGS__))

#define TA_DISPATCH_FLOATING(TYPE, NAME, ...)                                 \
  THO_DISPATCH_SWITCH(                                                        \
      TYPE, NAME,                                                             \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Double, __VA_ARGS__)    \
      THO_DISPATCH_CASE(torch::headeronly::ScalarType::Float, __VA_ARGS__))

// ---------------------------------------------------------------------------
// metadata helpers
// ---------------------------------------------------------------------------
// The stable Tensor mirrors TensorBase, so sizes/strides/dim/size/scalar_type
// are already called through the object. These wrappers exist for the handful
// of cases where ATen and stable disagree in shape or name.
inline bool is_cuda(const Tensor& t) { return t.is_cuda(); }
inline bool is_cpu(const Tensor& t) { return t.is_cpu(); }
inline Device device_of(const Tensor& t) { return t.device(); }
inline int64_t device_index(const Tensor& t) { return t.get_device_index(); }

// ATen spells the mutable one `data_ptr<T>()`; the stable ABI calls it
// `mutable_data_ptr<T>()` and keeps `const_data_ptr<T>()` for the read-only
// pointer. The mutable accessor is const-qualified in both, so a `const
// Tensor&` is enough - and it is why a view can be built without a
// const_cast here (the ATen backend needs one).
template <typename T>
inline T* data_ptr(const Tensor& t) {
  return t.mutable_data_ptr<T>();
}

template <typename T>
inline const T* const_data_ptr(const Tensor& t) {
  return t.const_data_ptr<T>();
}

// ---------------------------------------------------------------------------
// views (accessor / packed_accessor replacement)
// ---------------------------------------------------------------------------
template <typename T>
inline View1<T> view1(const Tensor& t) {
  STD_TORCH_CHECK(t.dim() == 1, "view1: expected a 1-D tensor, got dim=", t.dim());
  auto sizes = t.sizes();
  auto strides = t.strides();
  return View1<T>{ta::data_ptr<T>(t), sizes[0], strides[0]};
}

template <typename T>
inline View2<T> view2(const Tensor& t) {
  STD_TORCH_CHECK(t.dim() == 2, "view2: expected a 2-D tensor, got dim=", t.dim());
  auto sizes = t.sizes();
  auto strides = t.strides();
  return View2<T>{ta::data_ptr<T>(t), sizes[0], sizes[1], strides[0], strides[1]};
}

// ---------------------------------------------------------------------------
// allocation / copies
// ---------------------------------------------------------------------------
// There is no `TensorOptions` in the stable ABI (the ATen backend keeps using
// the real one), so allocation goes through explicit dtype/device arguments or
// through a template tensor whose dtype and device are inherited.
inline Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype) {
  return torch::stable::empty(sizes, dtype);
}

inline Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype,
                    Device device) {
  return torch::stable::empty(sizes, dtype, std::nullopt, device);
}

inline Tensor empty_like(const Tensor& like) {
  return torch::stable::empty_like(like);
}

inline Tensor new_empty(const Tensor& like, const std::vector<int64_t>& sizes,
                        std::optional<ScalarType> dtype = std::nullopt) {
  return torch::stable::new_empty(like, sizes, dtype);
}

inline Tensor new_zeros(const Tensor& like, const std::vector<int64_t>& sizes,
                        std::optional<ScalarType> dtype = std::nullopt) {
  return torch::stable::new_zeros(like, sizes, dtype);
}

inline Tensor zero_(Tensor& t) { return torch::stable::zero_(t); }

// `torch::stable::zeros` does not exist: empty + zero_ is the two-call form the
// POC used, and it keeps the "inherit from a template tensor" overload below
// cheap.
inline Tensor zeros(const std::vector<int64_t>& sizes, ScalarType dtype) {
  Tensor out = ta::empty(sizes, dtype);
  ta::zero_(out);
  return out;
}

inline Tensor zeros(const std::vector<int64_t>& sizes, ScalarType dtype,
                    Device device) {
  Tensor out = ta::empty(sizes, dtype, device);
  ta::zero_(out);
  return out;
}

inline Tensor zeros_like(const Tensor& like) {
  Tensor out = ta::empty_like(like);
  ta::zero_(out);
  return out;
}

inline Tensor new_zeros_1d(const Tensor& like, int64_t n) {
  std::vector<int64_t> sizes{n};
  return ta::new_zeros(like, sizes);
}

// `torch::stable::contiguous()` is a dispatcher round trip plus a tensor handle
// allocation, i.e. ~1.5 us, while `is_contiguous()` is one shim call. The old
// ATen code calls `.contiguous()` unconditionally because it is nearly free
// there; under the stable ABI it is not, so gate it on the cheap check
// (+10.2 us -> +7.4 us on the POC's push benchmark).
inline Tensor as_contiguous(const Tensor& t) {
  return t.is_contiguous() ? t : torch::stable::contiguous(t);
}

inline Tensor contiguous(const Tensor& t) {
  return torch::stable::contiguous(t);
}

inline Tensor clone(const Tensor& t) { return torch::stable::clone(t); }

// NOTE: `narrow` and `copy_` take the destination by non-const reference, which
// is why the call sites in rans.hpp pass a mutable `stream`.
inline Tensor narrow(Tensor& t, int64_t dim, int64_t start, int64_t length) {
  return torch::stable::narrow(t, dim, start, length);
}

inline Tensor copy_(Tensor& dst, const Tensor& src) {
  return torch::stable::copy_(dst, src);
}

inline Tensor select(const Tensor& t, int64_t dim, int64_t index) {
  return torch::stable::select(t, dim, index);
}

inline Tensor reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  return torch::stable::reshape(t, shape);
}

inline Tensor unsqueeze(const Tensor& t, int64_t dim) {
  return torch::stable::unsqueeze(t, dim);
}

inline Tensor amax(const Tensor& t, int64_t dim, bool keepdim = false) {
  return torch::stable::amax(t, dim, keepdim);
}

inline Tensor to(const Tensor& t, Device device) {
  return torch::stable::to(t, device);
}

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------
// `.max().item<T>()` has no stable equivalent: at::Scalar has no StableIValue
// representation, so `aten::_local_scalar_dense` cannot be called through
// torch_call_dispatcher. Reduce with stable ops instead and read the single
// value through the data pointer.
//
// On CUDA the reduce runs on the device, so the one-element result has to come
// back to the host first: `.to(cpu)` is a blocking device->host copy (the
// default non_blocking=false), i.e. exactly the synchronisation `.item()` costs
// today. This is the *only* place the stable push wrapper synchronises.
template <typename T>
inline T max_scalar(const Tensor& t) {
  Tensor col = ta::select(t, 1, 0);
  Tensor reduced = ta::amax(col, int64_t(0), /*keepdim=*/false);
  if (!reduced.is_cpu()) {
    reduced = ta::to(reduced, Device(DeviceType::CPU, 0));
  }
  return *reduced.const_data_ptr<T>();
}

// ---------------------------------------------------------------------------
// parallel_for: the same shim as at::parallel_for (torch_parallel_for), i.e.
// torch's intra-op pool; the ATen backend forwards to the identical function.
// ---------------------------------------------------------------------------
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain, const F& f) {
  torch::stable::parallel_for(begin, end, grain, f);
}

// ---------------------------------------------------------------------------
// aten::resize_ - no stable wrapper exists, so go through the dispatcher.
// ---------------------------------------------------------------------------
// This is the one place the stable build has to leave the blessed path: the op
// signature is not covered by the ABI promise (only the *call mechanism* is).
// `aten::resize_` has had this schema for years; if it ever changes the failure
// is a loud TORCH_ERROR_CODE_CHECK at the call site, not silent corruption.
// STABLE_ABI_MIGRATION.md section 4, pit 4 records the alternative (return a
// grown stream instead of resizing in place).
inline void resize2d(Tensor& t, int64_t rows, int64_t cols) {
  std::vector<int64_t> size{rows, cols};
  std::array<StableIValue, 3> stack{
      torch::stable::detail::from(t),
      torch::stable::detail::from(size),
      torch::stable::detail::from(std::optional<torch::headeronly::MemoryFormat>{}),
  };
#if defined(TORCH_FEATURE_VERSION) && TORCH_FEATURE_VERSION >= TORCH_VERSION_2_10_0
  TORCH_ERROR_CODE_CHECK(
      torch_call_dispatcher("aten::resize_", "", stack.data(), TORCH_ABI_VERSION));
#else
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_call_dispatcher("aten::resize_", "", stack.data()));
#endif
  // `from(t)` allocated a fresh handle for the (already mutated) tensor; steal
  // it back into a Tensor so the reference is released instead of leaked.
  Tensor out = torch::stable::detail::to<Tensor>(stack[0]);
  (void)out;
}

// ---------------------------------------------------------------------------
// CUDA
// ---------------------------------------------------------------------------
// Guarded exactly like the header's CUDA include: a CPU-only translation unit
// has neither cudaStream_t nor the CUDA entries of the C shim. The CUDA build
// reaches this block through `-DUSE_CUDA=1` (and nvcc's __CUDACC__).
#if defined(__CUDACC__) || defined(USE_CUDA)
// The stable ABI has no `C10_CUDA_CHECK` (it lives in c10/cuda, i.e. behind a
// non-stable C++ symbol), so a single macro is enough - plus a
// `cudaGetLastError()` after every launch, which is what catches "invalid
// configuration" style failures that would otherwise be reported by the *next*
// unrelated CUDA call.
#define TA_CUDA_CHECK(call)                                                   \
  do {                                                                        \
    const cudaError_t ta_cuda_err_ = (call);                                  \
    STD_TORCH_CHECK(ta_cuda_err_ == cudaSuccess, "torch_ans CUDA error: ",     \
                    cudaGetErrorString(ta_cuda_err_));                        \
  } while (0)

// torch's current stream for the device the tensors live on - the stable
// equivalent of `at::cuda::getCurrentCUDAStream()`. Existing torch_ans kernels
// launch with `<<<blocks, threads>>>` (i.e. stream 0) and therefore ignore
// torch's current stream, a latent race with any non-default stream; the
// migrated kernels pass this value explicitly.
inline cudaStream_t current_stream(const Tensor& t) {
  void* stream_ptr = nullptr;
  TORCH_ERROR_CODE_CHECK(
      aoti_torch_get_current_cuda_stream(t.get_device_index(), &stream_ptr));
  return reinterpret_cast<cudaStream_t>(stream_ptr);
}
#endif  // __CUDACC__ / USE_CUDA

// Same constructor shape as the ATen backend's guard (a device index), so the
// wrappers can be written once.
using DeviceGuard = torch::stable::accelerator::DeviceGuard;

}  // namespace ta
