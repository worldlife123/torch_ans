#pragma once

// ATen backend of the `ta::` compatibility layer - the near-zero-cost
// forwarding side of the same interface the stable backend implements in
// torch_api_stable.hpp.
//
// This is what today's build (ATen + pybind11, compiled locally per torch
// version) keeps using, so behaviour here has to stay identical to the current
// code: every function is a one-line forward to the ATen call the sources use
// today, and the only deliberate difference is that the reductions go through
// the same `select + amax + const_data_ptr` implementation as the stable
// backend instead of `.item<T>()`. Keeping one implementation is what makes
// "the same kernel, two build modes" verifiable (the POC proved bit-identical
// results for the algorithm core).
//
// This header must not be included by a stable-ABI translation unit: it pulls
// in <torch/extension.h>, i.e. the whole of ATen.

#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <vector>

// The accelerator section at the bottom of this file (stream lookup,
// DeviceGuard, error macro) needs the c10/cuda (or c10/hip) headers, which only
// exist when a toolkit is in play - a CPU-only build must not see them.
// WITH_CUDA/WITH_HIP is what setup.py defines, nvcc defines __CUDACC__, hipcc
// __HIPCC__. These includes have to be outside `namespace ta` (and at the top
// with the rest), because the c10 headers declare their own namespaces.
//
// HIP: torch's ROCMExtension hipifies the .cu sources before compiling them, so
// the kernels' `cudaStream_t`/`cudaGetLastError()` are rewritten for us, but the
// names *this* header defines are not - hence the explicit branch. A HIP-only
// build (WITH_HIP, no WITH_CUDA) would otherwise have no ta::current_stream /
// ta::DeviceGuard / TA_CUDA_CHECK at all.
#if defined(__CUDACC__) || defined(WITH_CUDA) || defined(__HIPCC__) || defined(WITH_HIP)
#if defined(__HIPCC__) || defined(WITH_HIP)
#include <c10/hip/HIPException.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#else
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#endif
#endif

#include "view.hpp"

namespace ta {

// ---------------------------------------------------------------------------
// backend types
// ---------------------------------------------------------------------------
using Tensor = at::Tensor;
using ScalarType = at::ScalarType;
using Device = at::Device;
using DeviceType = at::DeviceType;

// ---------------------------------------------------------------------------
// dtype dispatch: the real ATen macros, one macro name per ta:: alias so that
// the dispatch sites look the same in both backends (the stable backend writes
// its own case lists over THO_DISPATCH_CASE).
// ---------------------------------------------------------------------------
#define TA_DISPATCH_INTEGRAL(TYPE, NAME, ...) \
  AT_DISPATCH_INTEGRAL_TYPES(TYPE, NAME, __VA_ARGS__)

#define TA_DISPATCH_FLOATING(TYPE, NAME, ...) \
  AT_DISPATCH_FLOATING_TYPES(TYPE, NAME, __VA_ARGS__)

// The stable backend gets this macro from torch/headeronly/util/Exception.h; in
// an ATen build the natural spelling is TORCH_CHECK (a c10::Error, i.e. exactly
// what the current sources throw). The guard matters on torch >= 2.10, where
// <torch/extension.h> may already have pulled the header-only definition in;
// overwriting it would be a redefinition warning and would silently change the
// exception type.
namespace detail {

// Builds the c10::Error for a failed `STD_TORCH_CHECK`. A function, not a macro
// sending `__VA_ARGS__` on: the message arguments have to be split by whoever
// parses the call, and MSVC's traditional preprocessor substitutes `__VA_ARGS__`
// as a *single* argument. With the previous spelling
// (`#define STD_TORCH_CHECK(...) TORCH_CHECK(__VA_ARGS__)`) the expansion became
// `if (!(cond, "message", value))` there - a comma expression whose value is its
// last operand, i.e. always true - so *every* check in an MSVC ATen build was a
// silent no-op. Measured in CI: `tans_init_stream` accepted
// `num_interleaves=0` on Windows (both the setuptools install and a fresh JIT
// build) while gcc and clang threw, and the checks whose tests did pass there
// (tans_utils.hpp's "sums to" / "no probability mass") come from an
// `std::string* error` out-parameter, not from this macro.
template <typename... Args>
[[noreturn]] inline void check_failed(const Args&... args) {
  if constexpr (sizeof...(Args) == 0) {
    // `STD_TORCH_CHECK(cond)` is a legal call (19 of the call sites use it), and
    // a trailing empty `__VA_ARGS__` does not compile on torch 2.7.
    TORCH_CHECK(false, "torch_ans: a ta:: precondition does not hold");
  } else {
    TORCH_CHECK(false, args...);
  }
}

}  // namespace detail

#ifndef STD_TORCH_CHECK
// The condition is a *named* parameter, so it is the caller's expression: that
// is the whole point (see `detail::check_failed` for what the variadic
// forwarding used to do on MSVC).
#define STD_TORCH_CHECK(cond, ...) \
  do {                             \
    if (!(cond)) {                 \
      ::ta::detail::check_failed(__VA_ARGS__); \
    }                              \
  } while (0)
#endif

// ---------------------------------------------------------------------------
// metadata helpers
// ---------------------------------------------------------------------------
inline bool is_cuda(const Tensor& t) { return t.is_cuda(); }
inline bool is_cpu(const Tensor& t) { return t.is_cpu(); }
inline Device device_of(const Tensor& t) { return t.device(); }
// ATen spells it `get_device()` (a c10::DeviceIndex), the stable ABI
// `get_device_index()` (an int32_t) - the difference is invisible after this.
inline int64_t device_index(const Tensor& t) { return t.get_device(); }

template <typename T>
inline T* data_ptr(const Tensor& t) {
  return t.data_ptr<T>();
}

template <typename T>
inline const T* const_data_ptr(const Tensor& t) {
  // `data_ptr<T>()` rather than ATen's `const_data_ptr<T>()`: the latter only
  // exists from torch 1.11 on, while `pyproject.toml` still promises
  // `torch>=1.10` and the CI matrix builds against 1.10.1 (where
  // `const_data_ptr` is not a member of at::Tensor at all). It is
  // const-qualified in every release and returns the same pointer - this helper
  // *is* the const spelling of it.
  return t.data_ptr<T>();
}

// ---------------------------------------------------------------------------
// views (accessor / packed_accessor replacement)
// ---------------------------------------------------------------------------
// Built from the raw pointer and the strides rather than from `.accessor<>()`:
// ATen's accessor would need the tensor's dtype as a template argument here
// (the call sites already know it), and using the same struct as the stable
// backend keeps the two builds bit-identical.
template <typename T>
inline View1<T> view1(const Tensor& t) {
  STD_TORCH_CHECK(t.dim() == 1, "view1: expected a 1-D tensor, got dim=", t.dim());
  return View1<T>{ta::data_ptr<T>(t), t.size(0), t.stride(0)};
}

template <typename T>
inline View2<T> view2(const Tensor& t) {
  STD_TORCH_CHECK(t.dim() == 2, "view2: expected a 2-D tensor, got dim=", t.dim());
  return View2<T>{ta::data_ptr<T>(t), t.size(0), t.size(1), t.stride(0), t.stride(1)};
}

// ---------------------------------------------------------------------------
// allocation / copies
// ---------------------------------------------------------------------------
// The stable ABI has no TensorOptions, so the interface cannot expose one; the
// ATen backend keeps the real at::TensorOptions hidden inside these helpers.
inline Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype) {
  return at::empty(sizes, at::TensorOptions().dtype(dtype));
}

inline Tensor empty(const std::vector<int64_t>& sizes, ScalarType dtype,
                    Device device) {
  return at::empty(sizes, at::TensorOptions().dtype(dtype).device(device));
}

inline Tensor empty_like(const Tensor& like) { return at::empty_like(like); }

inline Tensor new_empty(const Tensor& like, const std::vector<int64_t>& sizes,
                        std::optional<ScalarType> dtype = std::nullopt) {
  return at::empty(sizes, like.options().dtype(dtype.value_or(like.scalar_type())));
}

inline Tensor new_zeros(const Tensor& like, const std::vector<int64_t>& sizes,
                        std::optional<ScalarType> dtype = std::nullopt) {
  return at::zeros(sizes, like.options().dtype(dtype.value_or(like.scalar_type())));
}

inline Tensor zero_(Tensor& t) { return t.zero_(); }

inline Tensor zeros(const std::vector<int64_t>& sizes, ScalarType dtype) {
  return at::zeros(sizes, at::TensorOptions().dtype(dtype));
}

inline Tensor zeros(const std::vector<int64_t>& sizes, ScalarType dtype,
                    Device device) {
  return at::zeros(sizes, at::TensorOptions().dtype(dtype).device(device));
}

inline Tensor zeros_like(const Tensor& like) { return at::zeros_like(like); }

inline Tensor new_zeros_1d(const Tensor& like, int64_t n) {
  return at::zeros({n}, like.options());
}

// `.contiguous()` is nearly free when the tensor is already contiguous in ATen,
// but the short circuit is kept so that both backends take the same branches.
inline Tensor as_contiguous(const Tensor& t) {
  return t.is_contiguous() ? t : t.contiguous();
}

inline Tensor contiguous(const Tensor& t) { return t.contiguous(); }

inline Tensor clone(const Tensor& t) { return t.clone(); }

inline Tensor narrow(Tensor& t, int64_t dim, int64_t start, int64_t length) {
  return t.narrow(dim, start, length);
}

inline Tensor copy_(Tensor& dst, const Tensor& src) { return dst.copy_(src); }

inline Tensor select(const Tensor& t, int64_t dim, int64_t index) {
  return t.select(dim, index);
}

inline Tensor reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  return t.reshape(shape);
}

inline Tensor unsqueeze(const Tensor& t, int64_t dim) { return t.unsqueeze(dim); }

inline Tensor amax(const Tensor& t, int64_t dim, bool keepdim = false) {
  return t.amax(dim, keepdim);
}

inline Tensor to(const Tensor& t, Device device) { return t.to(device); }

// ---------------------------------------------------------------------------
// reductions
// ---------------------------------------------------------------------------
// Same implementation as the stable backend (select + amax + data pointer)
// instead of `.max().item<T>()`: one code path means one set of results and one
// synchronisation point (a blocking D2H on CUDA), so a difference between the
// two build modes can only come from the kernel, never from the wrapper.
template <typename T>
inline T max_scalar(const Tensor& t) {
  Tensor col = ta::select(t, 1, 0);
  Tensor reduced = ta::amax(col, int64_t(0), /*keepdim=*/false);
  if (!reduced.is_cpu()) {
    reduced = ta::to(reduced, Device(DeviceType::CPU, 0));
  }
  // through the helper, which is the torch >= 1.10 spelling of this read
  return *ta::const_data_ptr<T>(reduced);
}

// ---------------------------------------------------------------------------
// parallelism: at::parallel_for and torch::stable::parallel_for are the same
// shim (torch_parallel_for), so the thread pool does not change between builds.
// ---------------------------------------------------------------------------
template <typename F>
inline void parallel_for(int64_t begin, int64_t end, int64_t grain, const F& f) {
  at::parallel_for(begin, end, grain, f);
}

// ---------------------------------------------------------------------------
// in-place resize: the stable backend has to go through the dispatcher, ATen
// has the method. See torch_api_stable.hpp for the caveat.
// ---------------------------------------------------------------------------
inline void resize2d(Tensor& t, int64_t rows, int64_t cols) {
  t.resize_({rows, cols});
}

// ---------------------------------------------------------------------------
// CUDA / HIP
// ---------------------------------------------------------------------------
// The ATen build keeps using c10/cuda directly: C10_CUDA_CHECK for a runtime
// call, c10::cuda::getCurrentCUDAStream() for the stream (today's kernels used
// to ignore it, which is the race the migration fixes) and c10::cuda::CUDAGuard
// as the guard. The HIP flavour is the same three names from c10/hip - see the
// include block at the top for why it needs its own branch. It is only compiled
// when a toolkit is actually in play.
#if defined(__CUDACC__) || defined(WITH_CUDA) || defined(__HIPCC__) || defined(WITH_HIP)

#if defined(__HIPCC__) || defined(WITH_HIP)
#define TA_CUDA_CHECK(call) C10_HIP_CHECK(call)

inline hipStream_t current_stream(const Tensor& t) {
  return c10::hip::getCurrentHIPStream(t.get_device()).stream();
}

using DeviceGuard = c10::hip::HIPGuard;
#else
#define TA_CUDA_CHECK(call) C10_CUDA_CHECK(call)

inline cudaStream_t current_stream(const Tensor& t) {
  return c10::cuda::getCurrentCUDAStream(t.get_device()).stream();
}

using DeviceGuard = c10::cuda::CUDAGuard;
#endif

#endif  // __CUDACC__ / WITH_CUDA / __HIPCC__ / WITH_HIP

}  // namespace ta
