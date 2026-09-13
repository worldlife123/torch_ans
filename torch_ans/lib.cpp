#include <torch/extension.h>

#include "rans.hpp"
#include "rans_bindings.hpp"


// Optimized: torch tensor input/output
// inline torch::Tensor rans_pmf_to_quantized_cdf(const torch::Tensor& pmf, int precision) {
//   TORCH_CHECK(pmf.dim() == 1 || pmf.dim() == 2, "pmf must be 1D or 2D tensor");
//   auto device = pmf.device();
//   auto dtype = torch::kInt32;

//   torch::Tensor pmf_batched;
//   int64_t B, N;
//   if (pmf.dim() == 1) {
//     pmf_batched = pmf.unsqueeze(0); // shape (1, N)
//     B = 1;
//     N = pmf.size(0);
//   } else {
//     pmf_batched = pmf;
//     B = pmf.size(0);
//     N = pmf.size(1);
//   }

//   auto freq = torch::round(pmf_batched * (1 << precision)).to(dtype);
//   auto cdf = torch::zeros({B, N + 1}, torch::TensorOptions().dtype(dtype).device(device));
//   cdf.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}, freq);
//   // Step 3: Normalize frequencies per batch
//   auto total = cdf.sum(1, true).to(dtype);
//   total = torch::where(total == 0, torch::ones_like(total), total); // avoid div by zero
//   cdf = ((cdf * (1 << precision)) / total).to(dtype);
//   // Step 4: Partial sum for CDF per batch
//   cdf = torch::cumsum(cdf, 1).to(dtype);
//   cdf.index_put_({torch::indexing::Slice(), N}, 1 << precision);
//   // Step 5: Ensure strictly increasing CDF per batch
//   auto cdf_contig = cdf.contiguous();
//   auto cdf_ptr = cdf_contig.data_ptr<int32_t>();
//   for (int b = 0; b < B; ++b) {
//     int32_t* row = cdf_ptr + b * (N + 1);
//     for (int i = 0; i < N; ++i) {
//       if (row[i] == row[i + 1]) {
//         int32_t best_freq = INT32_MAX;
//         int best_steal = -1;
//         for (int j = 0; j < N; ++j) {
//           int32_t f = row[j + 1] - row[j];
//           if (f > 1 && f < best_freq) {
//             best_freq = f;
//             best_steal = j;
//           }
//         }
//         TORCH_CHECK(best_steal != -1, "No symbol to steal frequency from");
//         if (best_steal < i) {
//           for (int j = best_steal + 1; j <= i; ++j) {
//             row[j] -= 1;
//           }
//         } else {
//           TORCH_CHECK(best_steal > i, "best_steal must be > i");
//           for (int j = i + 1; j <= best_steal; ++j) {
//             row[j] += 1;
//           }
//         }
//       }
//     }
//   }
//   if (pmf.dim() == 1) {
//     return cdf_contig[0];
//   } else {
//     return cdf_contig;
//   }
// }



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


// Defines the operators
// NOTE: this will fail as TORCH_LIBRARY do not support std::optional
// TORCH_LIBRARY(torch_ans, m) {

//     TORCH_LIBRARY_RANS_BINDINGS(m);
// }