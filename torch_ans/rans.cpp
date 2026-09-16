// The two `py::bytes` operators and the alias-table builder.
//
// The bytes pair cannot be expressed in a stable schema - `std::vector<py::bytes>`
// has no StableIValue representation - so the stable build drops them and the
// Python layer does the conversion instead (torch_ans/_stable_C.py, migration
// section 1.5). The ATen build keeps them until the Python call sites switch
// over in stage 4.
//
// `rans_alias_build_table` is an ordinary tensor op (Tensor in, two Tensors
// out) and is shared by both builds.

#include "rans.hpp"

#ifndef TORCH_ANS_STABLE_ABI

std::vector<py::bytes> rans_stream_to_byte_strings(const ta::Tensor& stream)
{
  auto stream_safe = ta::as_contiguous(ta::to(stream, ta::Device(ta::DeviceType::CPU, 0)));
  ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream_safe);

  std::vector<py::bytes> output(stream.size(0));
  // NOTE: this cause error with openmp! maybe python objects cannot use parallel_for?
  // ta::parallel_for(0, stream.size(0), 0, [&](int64_t start, int64_t end) {
  int64_t start=0;
  int64_t end=stream.size(0);
      for (int64_t b = start; b < end; b++) {
          output[b] = py::bytes(reinterpret_cast<char*>(stream_view.row(b)+1), stream_view.at(b, 0));
      }
  // });

  return output;
}

ta::Tensor rans_byte_strings_to_stream(std::vector<py::bytes> byte_strings)
{
  // NOTE: implicit convert py::bytes to std::string
  // const std::string max_length_bytes = *std::max_element(byte_strings.begin(), byte_strings.end(), [](std::string s1, std::string s2){return s1.size()<s2.size();});
  const int64_t batch_size = byte_strings.size();
  size_t max_length = 0;
  for (std::string bytes : byte_strings)
  {
    if (bytes.size() > max_length) max_length = bytes.size();
  }

  const int64_t stream_length = (max_length + sizeof(int32_t) - 1) / sizeof(int32_t) + 1;

  ta::Tensor stream = ta::zeros({batch_size, stream_length}, ta::ScalarType::Int);
  ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream);
  // stream.index_put_({Slice(), 0}, sizeof(RANS_STATE_TYPE));

  ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; b++) {
        // TODO: does it copy memory?
        std::string bytes = byte_strings[b];
        const auto byte_length = bytes.size();
        const char* bytes_ptr = bytes.data();
        stream_view.at(b, 0) = byte_length;
        char* stream_ptr = reinterpret_cast<char*>(stream_view.row(b)+1);
        std::copy(bytes_ptr, bytes_ptr + byte_length, stream_ptr);
      }
  });

  return stream;
}

#endif  // TORCH_ANS_STABLE_ABI


std::tuple<ta::Tensor, ta::Tensor> rans_alias_build_table(
  const ta::Tensor& cdfs, const ta::Tensor& cdfs_sizes,
  int64_t symbol_precision,
  int64_t freq_precision
)
{
  ta::View2<int32_t> cdfs_view = ta::view2<int32_t>(cdfs);
  ta::View1<int32_t> cdfs_sizes_view = ta::view1<int32_t>(cdfs_sizes);
  const int64_t batch_size = cdfs.size(0);
  const int64_t max_cdf_size = cdfs.size(1);

  // TODO: confirm the max size of cdfs_with_alias_table
  ta::Tensor cdfs_with_alias_table = ta::zeros(
    {batch_size, static_cast<int64_t>(max_cdf_size + max_cdf_size*(sizeof(RANSAliasSamplingCDFTableElement<int32_t>)/sizeof(int32_t)))},
    ta::ScalarType::Int);
  ta::View2<int32_t> cdfs_with_alias_table_view = ta::view2<int32_t>(cdfs_with_alias_table);
  ta::Tensor cdfs_with_alias_remap = ta::zeros({batch_size, max_cdf_size + (1<<freq_precision)}, ta::ScalarType::Int);
  ta::View2<int32_t> cdfs_with_alias_remap_view = ta::view2<int32_t>(cdfs_with_alias_remap);

  ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; b++) {
        auto cdf_ptr = cdfs_view.row(b);
        auto cdf_size = cdfs_sizes_view[b];
        auto cdf_alias_table_ptr = cdfs_with_alias_table_view.row(b);
        auto cdf_alias_remap_ptr = cdfs_with_alias_remap_view.row(b);
        // copy existing cdf to new tensors
        std::copy(cdf_ptr, cdf_ptr + cdf_size, cdf_alias_table_ptr);
        cdf_alias_table_ptr += cdf_size;
        std::copy(cdf_ptr, cdf_ptr + cdf_size, cdf_alias_remap_ptr);
        cdf_alias_remap_ptr += cdf_size;
        // NOTE: build_alias_mapping reports failure (the alias scheme only works
        // when the cdf size is 2**symbol_precision + 1 and the frequencies sum
        // to exactly 2**freq_precision). Ignoring it used to leave a partially
        // built table behind, which silently produced a wrong stream on every
        // symbol instead of an error - so fail loudly.
        const bool built = build_alias_mapping<int32_t, int32_t>(
          cdf_ptr, cdf_size,
          reinterpret_cast<RANSAliasSamplingCDFTableElement<int32_t>*>(cdf_alias_table_ptr), 
          reinterpret_cast<int32_t*>(cdf_alias_remap_ptr), 
          symbol_precision, freq_precision
        );
        STD_TORCH_CHECK(built,
          "alias sampling table could not be built: it requires "
          "2**symbol_precision == cdfs_sizes[i] - 1 and a cdf whose frequencies sum to "
          "2**freq_precision. Got cdf_size=", cdf_size, " (symbol_precision=",
          symbol_precision, "), freq_precision=", freq_precision);
      }
  });

  return std::tuple<ta::Tensor, ta::Tensor>(cdfs_with_alias_remap, cdfs_with_alias_table);
}
