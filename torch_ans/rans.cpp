
#include "rans.hpp"


std::vector<py::bytes> rans_stream_to_byte_strings(const torch::Tensor& stream)
{
  auto stream_safe = stream.cpu().contiguous();
  auto stream_accessor = stream_safe.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();

  std::vector<py::bytes> output(stream.size(0));
  // NOTE: this cause error with openmp! maybe python objects cannot use parallel_for?
  // at::parallel_for(0, stream.size(0), 0, [&](size_t start, size_t end) {
  size_t start=0;
  size_t end=stream.size(0);
      for (size_t b = start; b < end; b++) {
          output[b] = py::bytes(reinterpret_cast<char*>(stream_accessor[b].data()+1), stream_accessor[b][0]);
      }
  // });

  return output;
}

torch::Tensor rans_byte_strings_to_stream(std::vector<py::bytes> byte_strings)
{
  // NOTE: implicit convert py::bytes to std::string
  // const std::string max_length_bytes = *std::max_element(byte_strings.begin(), byte_strings.end(), [](std::string s1, std::string s2){return s1.size()<s2.size();});
  const ssize_t batch_size = byte_strings.size();
  size_t max_length = 0;
  for (std::string bytes : byte_strings)
  {
    if (bytes.size() > max_length) max_length = bytes.size();
  }

  const ssize_t stream_length = (max_length + sizeof(DEFAULT_TORCH_TENSOR_TYPE) - 1) / sizeof(DEFAULT_TORCH_TENSOR_TYPE) + 1;

  auto stream = torch::zeros({batch_size, stream_length}, torch::TensorOptions().dtype(DEFAULT_TORCH_TENSOR_DTYPE));
  auto stream_accessor = stream.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
  // stream.index_put_({Slice(), 0}, sizeof(RANS_STATE_TYPE));

  at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
        // TODO: does it copy memory?
        std::string bytes = byte_strings[b];
        const auto byte_length = bytes.size();
        const char* bytes_ptr = bytes.data();
        stream_accessor[b][0] = byte_length;
        char* stream_ptr = reinterpret_cast<char*>(stream_accessor[b].data()+1);
        std::copy(bytes_ptr, bytes_ptr + byte_length, stream_ptr);
      }
  });

  return stream;
}


std::tuple<torch::Tensor, torch::Tensor> rans_alias_build_table(
  const torch::Tensor& cdfs, const torch::Tensor& cdfs_sizes,
  ssize_t symbol_precision,
  ssize_t freq_precision
)
{
  auto cdfs_accessor = cdfs.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
  auto cdfs_sizes_accessor = cdfs_sizes.accessor<DEFAULT_TORCH_TENSOR_TYPE, 1>();
  const ssize_t batch_size = cdfs.size(0);
  const ssize_t max_cdf_size = cdfs.size(1);

  // TODO: confirm the max size of cdfs_with_alias_table
  auto cdfs_with_alias_table = torch::zeros({batch_size, static_cast<ssize_t>(max_cdf_size + max_cdf_size*(sizeof(RANSAliasSamplingCDFTableElement<DEFAULT_TORCH_TENSOR_TYPE>)/sizeof(DEFAULT_TORCH_TENSOR_TYPE)))}, 
    torch::TensorOptions().dtype(DEFAULT_TORCH_TENSOR_DTYPE));
  auto cdfs_with_alias_table_accessor = cdfs_with_alias_table.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
  auto cdfs_with_alias_remap = torch::zeros({batch_size, max_cdf_size + (1<<freq_precision)}, torch::TensorOptions().dtype(DEFAULT_TORCH_TENSOR_DTYPE));
  auto cdfs_with_alias_remap_accessor = cdfs_with_alias_remap.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();

  at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
        auto cdf_ptr = cdfs_accessor[b].data();
        auto cdf_size = cdfs_sizes_accessor[b];
        auto cdf_alias_table_ptr = cdfs_with_alias_table_accessor[b].data();
        auto cdf_alias_remap_ptr = cdfs_with_alias_remap_accessor[b].data();
        // copy existing cdf to new tensors
        std::copy(cdf_ptr, cdf_ptr + cdf_size, cdf_alias_table_ptr);
        cdf_alias_table_ptr += cdf_size;
        std::copy(cdf_ptr, cdf_ptr + cdf_size, cdf_alias_remap_ptr);
        cdf_alias_remap_ptr += cdf_size;
        build_alias_mapping<DEFAULT_TORCH_TENSOR_TYPE, DEFAULT_TORCH_TENSOR_TYPE>(
          cdf_ptr, cdf_size,
          reinterpret_cast<RANSAliasSamplingCDFTableElement<DEFAULT_TORCH_TENSOR_TYPE>*>(cdf_alias_table_ptr), 
          reinterpret_cast<DEFAULT_TORCH_TENSOR_TYPE*>(cdf_alias_remap_ptr), 
          symbol_precision, freq_precision
        );
      }
  });

  return std::tuple<torch::Tensor, torch::Tensor>(cdfs_with_alias_remap, cdfs_with_alias_table);
}