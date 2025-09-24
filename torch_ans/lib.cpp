#include <torch/extension.h>

#include "rans.hpp"


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



PYBIND11_MODULE(_C, m){
    m.doc() = "PyTorch based ANS entropy coding library.";

    // m.def("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf);
    
    TORCH_EXTENSION_RANS_BINDINGS(m);
}

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}


// Defines the operators
// NOTE: this will fail as TORCH_LIBRARY do not support std::optional
// TORCH_LIBRARY(torch_ans, m) {

//     TORCH_LIBRARY_RANS_BINDINGS(m);
// }