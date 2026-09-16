// Module entry point, in the two flavours the package ships.
//
//   ATen build (`#else` below): a pybind11 module, `torch_ans._C`.
//   stable-ABI build: no Python module at all - the .so only registers ops with
//   the dispatcher and is loaded with `torch.ops.load_library()`. That is what
//   makes it CPython-agnostic and torch-version-agnostic
//   (STABLE_ABI_MIGRATION.md sections 1.1-1.3, stage 3).

#include "rans.hpp"
#include "rans_bindings.hpp"

#ifdef TORCH_ANS_STABLE_ABI

// The schemas are declared exactly once per process: the CUDA flavour of the
// wheel is a second .so (lib_cuda.cpp) that registers implementations only
// (migration section 1.2). Load order between the two does not matter. This
// library also carries the CPU (and catch-all) kernels, so a CPU-only wheel is
// just this one file.
STABLE_TORCH_LIBRARY(torch_ans, m) {
  torch_ans_register_schemas(m);
}

// `init_stream` has no Tensor argument, so the dispatcher cannot derive a
// backend key from the call; those operators live under the catch-all key
// (migration section 4, pit 9).
STABLE_TORCH_LIBRARY_IMPL(torch_ans, CompositeExplicitAutograd, m) {
  torch_ans_register_init_impls(m);
}

STABLE_TORCH_LIBRARY_IMPL(torch_ans, CPU, m) {
  torch_ans_register_cpu_impls(m);
}

#else

// TORCH_EXTENSION_NAME is provided by torch's build systems (BuildExtension
// defines it as the last component of the extension name, and JIT builds via
// torch.utils.cpp_extension.load define it as the module name). Using it here
// keeps the PyInit symbol in sync with JIT version bumps (e.g. `*_v1` after a
// CUDA->CPU fallback re-build in the same process).
#ifndef TORCH_EXTENSION_NAME
#define TORCH_EXTENSION_NAME _C
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.doc() = "PyTorch based ANS entropy coding library.";

    // Capability flag. Both build paths define WITH_CUDA for a CUDA build
    // (setup.py's CUDAExtension and the runtime build's extra_cuda_cflags) and
    // leave it undefined for a CPU-only one, so this attribute is the single
    // reliable source of truth for "can this module code CUDA tensors?" - the
    // Python layer uses it to explain the situation instead of letting the call
    // fail with a bare "not compiled with GPU support!" (see torch_ans/utils.py
    // and torch_ans/_lazy_C.py).
#if defined(WITH_CUDA) || defined(WITH_HIP)
    m.attr("_torch_ans_with_cuda") = py::bool_(true);
#else
    m.attr("_torch_ans_with_cuda") = py::bool_(false);
#endif

    // m.def("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf);

    // Gated by the TORCH_ANS_WITH_* feature macros: with incremental
    // compilation only the (impl, interleaves, lookup) subset an interface
    // needs is compiled. See rans_bindings.hpp / rans_build_config.hpp.
    torch_ans_bind_all(m);
}

#endif  // TORCH_ANS_STABLE_ABI


// Defines the operators
// NOTE: this used to fail because TORCH_LIBRARY could not express
// std::optional, which is why the bindings are pybind11. The stable ABI does
// support `Tensor?` (the POC verified it), so the equivalent registration now
// exists in the TORCH_ANS_STABLE_ABI branch above.
// TORCH_LIBRARY(torch_ans, m) {
//     TORCH_LIBRARY_RANS_BINDINGS(m);
// }
