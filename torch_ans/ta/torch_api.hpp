#pragma once

// The single entry point of the `ta::` compatibility layer.
//
// torch_ans has to compile two ways for a while:
//
//   * ATen + pybind11, compiled locally per torch version  -> torch_api_aten.hpp
//   * libtorch stable ABI, one prebuilt .so for torch>=2.10 -> torch_api_stable.hpp
//
// Kernels and op wrappers include *this* header and are written once against
// `ta::*`; the difference between the two backends is confined to the
// `#ifdef` below plus the two backend headers. That is what allows "the same
// kernel, two build modes" to be checked bit for bit (the POC did it for the
// algorithm core in rans_utils.hpp and for the CPU/CUDA pair).
//
// Every source that touches torch goes through this header: rans.hpp,
// rans_cpu.cpp, rans_cuda.cu, rans_warp_cuda.cuh, rans.cpp, rans_bindings.hpp
// and lib.cpp / lib_cuda.cpp (migration stages 1-4 and 6). A new call site means
// adding a `ta::` primitive to *both* backends, never an ATen call from a kernel
// or an op wrapper - that is what keeps the two build modes bit-comparable.

#ifdef TORCH_ANS_STABLE_ABI
#include "torch_api_stable.hpp"
#else
#include "torch_api_aten.hpp"
#endif

// ---------------------------------------------------------------------------
// CUDA gating
// ---------------------------------------------------------------------------
// The sources used to write `#if defined(WITH_CUDA) || defined(WITH_HIP)`,
// which says "was a GPU kernel compiled into this extension" - a build-time
// fact, not a backend fact. Both build paths define WITH_CUDA the same way
// (setup.py, and the stable path's CUDA flavour too), so one macro covers them
// and the kernels do not need to know which backend they are built against.
#if defined(WITH_CUDA) || defined(WITH_HIP)
#define TA_WITH_CUDA 1
#else
#define TA_WITH_CUDA 0
#endif

namespace ta {

// ---------------------------------------------------------------------------
// stream growth: one implementation, in rans_push
// ---------------------------------------------------------------------------
// A `ta::safe_stream_byte_length` / `ta::ensure_stream_capacity` pair used to
// live here as a second copy of the wrapper's arithmetic. It was never called
// (only the syntax-check TU in scripts/check_ta_headers.py instantiated it) and
// it had to stay in step with two other places by hand, which is exactly the
// drift the capacity contract cannot afford. The three are now:
//
//   * `rans_push`          - the worst case, and the compare/throw that enforces
//                            the caller's capacity (torch_ans/rans.hpp);
//   * `_stream_budget`     - the same arithmetic in Python, which reserves the
//                            capacity (torch_ans/utils.py);
//   * tests/test_capacity_contract.py - pins the two against each other by using
//                            the wrapper's own growth as the oracle.
//
// The capacity contract is the option that removes both the per-push device->host
// read and the `aten::resize_` call the stable ABI does not promise. The
// alternatives were: keeping the dispatcher call (removes neither), having the
// operator return a grown stream (changes the public signature and every call
// site) and a strict pre-reservation that raises instead of growing (loses push's
// self-growing behaviour for direct callers). scripts/bench_stream_capacity.py
// measures what it buys - a small CUDA push: 157 -> 39 us.

}  // namespace ta
