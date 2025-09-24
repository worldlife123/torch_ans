#include <torch/extension.h>

#include <x86intrin.h>
#include <vector>
// #include <span>
// #include <bit>

#include "rans.hpp"
#include "rans_utils.hpp"

// from https://artificial-mind.net/blog/2020/10/31/constexpr-for
template <auto Start, auto End, auto Inc, class F>
constexpr void constexpr_for(F&& f)
{
    if constexpr (Start < End)
    {
        f(std::integral_constant<decltype(Start), Start>());
        constexpr_for<Start + Inc, End, Inc>(f);
    }
}

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS=0>
torch::Tensor rans_init_stream(ssize_t size, ssize_t num_interleaves, ssize_t preallocate_size) 
{
  ssize_t stream_init_size = 1 + preallocate_size / sizeof(DEFAULT_TORCH_TENSOR_TYPE) + num_interleaves * ((sizeof(RANS_STATE_TYPE) + sizeof(DEFAULT_TORCH_TENSOR_TYPE) - 1) / sizeof(DEFAULT_TORCH_TENSOR_TYPE));
  auto stream = torch::zeros({size, stream_init_size}, torch::TensorOptions().dtype(DEFAULT_TORCH_TENSOR_DTYPE));
  auto stream_accessor = stream.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
  // stream.index_put_({Slice(), 0}, sizeof(RANS_STATE_TYPE));

  at::parallel_for(0, size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
          stream_accessor[b][0] = sizeof(RANS_STATE_TYPE) * num_interleaves;
          RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
          for (ssize_t i = 0; i < num_interleaves; i++) {
            state_ptr[i] = RANS_STATE_LOWER_BOUND;
          }
      }
  });

  return stream;
}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, size_t NUM_INTERLEAVES=1>
void rans_push_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
{

  AT_DISPATCH_INTEGRAL_TYPES(symbols.scalar_type(), "rans_push_indexed_cpu", [&] {

    // TORCH_CHECK(stream.dtype() == DEFAULT_TORCH_TENSOR_DTYPE);
    // TORCH_CHECK(symbols.dtype() == DEFAULT_TORCH_TENSOR_DTYPE);
    TORCH_INTERNAL_ASSERT(stream.device().type() == torch::DeviceType::CPU);
    TORCH_INTERNAL_ASSERT(symbols.device().type() == torch::DeviceType::CPU);
    
    TORCH_CHECK(indexes.sizes() == symbols.sizes());
    // TORCH_CHECK(indexes.dtype() == DEFAULT_TORCH_TENSOR_DTYPE);
    TORCH_INTERNAL_ASSERT(indexes.device().type() == torch::DeviceType::CPU);

    auto batch_size = stream.size(0);

    auto symbols_accessor = symbols.accessor<scalar_t, 2>();
    ssize_t num_symbols = symbols_accessor.size(1);

    auto stream_accessor = stream.accessor<scalar_t, 2>();
    auto indexes_accessor = indexes.accessor<scalar_t, 2>();
    auto cdfs_accessor = cdfs.accessor<scalar_t, 2>();
    auto cdfs_sizes_accessor = cdfs_sizes.accessor<scalar_t, 1>();
    auto offsets_accessor = offsets.accessor<scalar_t, 1>();

    at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
        const auto stream_length = stream_accessor[b][0];
        const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
        RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
        RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;
        
        auto symbols_ptr = symbols_accessor[b].data();
        auto indexes_ptr = indexes_accessor[b].data();
        // reverse coding
        const ssize_t first_interleave = num_symbols % NUM_INTERLEAVES;
        ssize_t i = num_symbols-1;
        for (; i >= num_symbols-first_interleave; i--) {
          const auto index = indexes_ptr[i];
          const auto cdf_ptr = cdfs_accessor[index].data();
          const auto cdf_size = cdfs_sizes_accessor[index];
          const auto offsets = offsets_accessor[index];
          const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
          // TODO: check index validity
          rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
            state_ptr+(i % NUM_INTERLEAVES), &stream_ptr, symbols_ptr[i],
            cdf_ptr, cdf_size, offsets,
            freq_precision, bypass_coding, bypass_precision,
            cdf_alias_remap_ptr
          );
        }

        for (; i >= 0; i-=NUM_INTERLEAVES) {

          if constexpr (NUM_INTERLEAVES>1) {
            // preprocess bypass coding before all interleaves
            std::array<scalar_t, NUM_INTERLEAVES> symbol_vals;
            if (bypass_coding) {
              for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
                const auto index = indexes_ptr[i-j];
                // const auto cdf_ptr = cdfs_accessor[index].data();
                const auto cdf_size = cdfs_sizes_accessor[index];
                const auto max_value = cdf_size - 2;
                scalar_t value = symbols_ptr[i-j] - offsets_accessor[index];
                scalar_t raw_val = 0;
                if (value < 0) {
                  raw_val = -2 * value - 1;
                  value = max_value;
                } else if (value >= max_value) {
                  raw_val = 2 * (value - max_value);
                  value = max_value;
                }
                symbol_vals[j] = value;
                if (value == max_value) {
                  rans_push_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t>(
                    state_ptr+(NUM_INTERLEAVES-1-j), &stream_ptr, raw_val, bypass_precision
                  );
                }
              }
              for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
                const auto index = indexes_ptr[i-j];
                const auto cdf_ptr = cdfs_accessor[index].data();
                const auto cdf_size = cdfs_sizes_accessor[index];
                const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
                rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
                  state_ptr+(NUM_INTERLEAVES-1-j), &stream_ptr, symbol_vals[j],
                  cdf_ptr, cdf_size, 0,
                  freq_precision, false, bypass_precision,
                  cdf_alias_remap_ptr
                );
              }
            }
            else {
              for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
                const auto index = indexes_ptr[i-j];
                const auto cdf_ptr = cdfs_accessor[index].data();
                const auto cdf_size = cdfs_sizes_accessor[index];
                const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
                rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
                  state_ptr+(NUM_INTERLEAVES-1-j), &stream_ptr, symbols_ptr[i-j],
                  cdf_ptr, cdf_size, offsets_accessor[index],
                  freq_precision, false, bypass_precision,
                  cdf_alias_remap_ptr
                );
              }
            }

          }
          else {
            const auto index = indexes_ptr[i];
            const auto cdf_ptr = cdfs_accessor[index].data();
            const auto cdf_size = cdfs_sizes_accessor[index];
            const auto offset = offsets_accessor[index];
            const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
            // TODO: check index validity
            rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
              state_ptr, &stream_ptr, symbols_ptr[i],
              cdf_ptr, cdf_size, offset,
              freq_precision, bypass_coding, bypass_precision,
              cdf_alias_remap_ptr
            );
          }


        }

        // update stream length
        stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);
      }
    });


  });


}

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF=false, bool USE_INVERSED_CDF=false, size_t NUM_INTERLEAVES=1>
torch::Tensor rans_pop_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  ssize_t freq_precision,
  bool bypass_coding, 
  ssize_t bypass_precision)
{
  torch::Tensor symbols = torch::zeros_like(indexes);

  AT_DISPATCH_INTEGRAL_TYPES(symbols.scalar_type(), "rans_pop_indexed_cpu", [&] {

    auto batch_size = stream.size(0);

    // TODO: accessor according to dtype
    auto stream_accessor = stream.accessor<scalar_t, 2>();
    auto indexes_accessor = indexes.accessor<scalar_t, 2>();
    auto cdfs_accessor = cdfs.accessor<scalar_t, 2>();
    auto cdfs_sizes_accessor = cdfs_sizes.accessor<scalar_t, 1>();
    auto offsets_accessor = offsets.accessor<scalar_t, 1>();

    auto symbols_accessor = symbols.accessor<scalar_t, 2>();
    ssize_t num_symbols = symbols_accessor.size(1);

    using RANS_SYMBOL_TYPE = scalar_t;
    using RANS_FREQ_TYPE = scalar_t;

    at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
        const auto stream_length = stream_accessor[b][0];
        const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
        RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
        RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;
        auto symbols_ptr = symbols_accessor[b].data();
        auto indexes_ptr = indexes_accessor[b].data();
        const ssize_t last_interleave = num_symbols - (NUM_INTERLEAVES-1);
        ssize_t i;

        for (i = 0; i < last_interleave; i+=NUM_INTERLEAVES) {

          if constexpr (NUM_INTERLEAVES>1) {

            for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
            // constexpr_for<0, NUM_INTERLEAVES, 1>([&](auto j){
              const auto index = indexes_ptr[i+j];
              const auto cdf_ptr = cdfs_accessor[index].data();
              const auto cdf_size = cdfs_sizes_accessor[index];
              const auto offset = offsets_accessor[index];
              const RANS_FREQ_TYPE* inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
              const auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
              // TODO: check index validity
              symbols_ptr[i+j] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_FREQ_TYPE>(
                state_ptr+j, &stream_ptr,
                cdf_ptr, cdf_size, offset,
                freq_precision, false, bypass_precision,
                inversed_cdf_ptr, cdf_alias_table_ptr
              );
            }//);

            // postprocess bypass coding after all interleaves (inverse to push step)
            if (bypass_coding) {
              for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
              // constexpr_for<0, NUM_INTERLEAVES, 1>([&](auto j){
                const auto index = indexes_ptr[i+j];
                // const auto cdf_ptr = cdfs_accessor[index].data();
                const auto cdf_size = cdfs_sizes_accessor[index];
                const auto offset = offsets_accessor[index];
                const auto max_value = cdf_size - 2;

                scalar_t value = symbols_ptr[i+j] - offset;
                if (value == max_value) {
                  const auto raw_val = rans_pop_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t>(
                    state_ptr+j, &stream_ptr, bypass_precision
                  );
                  value = raw_val >> 1;
                  if (raw_val & 1) {
                    value = -value - 1;
                  } else {
                    value += max_value;
                  }
                  symbols_ptr[i+j] = value + offset;
                }

              }//);
            }
          }
          else {
            const auto index = indexes_ptr[i];
            const auto cdf_ptr = cdfs_accessor[index].data();
            const auto cdf_size = cdfs_sizes_accessor[index];
            const auto offset = offsets_accessor[index];
            const RANS_FREQ_TYPE* inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
            const auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
            // TODO: check index validity
            symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_FREQ_TYPE>(
              state_ptr, &stream_ptr,
              cdf_ptr, cdf_size, offset,
              freq_precision, bypass_coding, bypass_precision,
              inversed_cdf_ptr, cdf_alias_table_ptr
            );
          }
        }
        
        // final symbols
        for (; i < num_symbols; i++) {
            const auto index = indexes_ptr[i];
            const auto cdf_ptr = cdfs_accessor[index].data();
            const auto cdf_size = cdfs_sizes_accessor[index];
            const auto offsets = offsets_accessor[index];
            const RANS_FREQ_TYPE* inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
            const auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
            // TODO: check index validity
            symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_FREQ_TYPE>(
              state_ptr+(i%NUM_INTERLEAVES), &stream_ptr,
              cdf_ptr, cdf_size, offsets,
              freq_precision, bypass_coding, bypass_precision,
              inversed_cdf_ptr, cdf_alias_table_ptr
            );
        }
        // update stream length
        stream_accessor[b][0] = (stream_ptr - reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + 1) * sizeof(RANS_STREAM_TYPE);

      }
    });

  });

  return symbols;
}


// Batched PMF to quantized CDF (CPU, parallel over batch)
torch::Tensor rans_pmf_to_quantized_cdf_cpu(const torch::Tensor& pmf, int64_t precision) {
  TORCH_CHECK(pmf.dim() == 1 || pmf.dim() == 2, "pmf must be 1D or 2D tensor");
  auto device = pmf.device();
  auto dtype = torch::kInt32;
  torch::Tensor pmf_batched;
  int64_t B, N;
  if (pmf.dim() == 1) {
    pmf_batched = pmf.unsqueeze(0);
    B = 1;
    N = pmf.size(0);
  } else {
    pmf_batched = pmf;
    B = pmf.size(0);
    N = pmf.size(1);
  }
  auto freq = torch::round(pmf_batched * (1 << precision)).to(dtype);
  auto cdf = torch::zeros({B, N + 1}, torch::TensorOptions().dtype(dtype).device(device));
  cdf.index_put_({torch::indexing::Slice(), torch::indexing::Slice(1, torch::indexing::None)}, freq);
  auto total = cdf.sum(1, true).to(dtype);
  total = torch::where(total == 0, torch::ones_like(total), total);
  cdf = ((cdf * (1 << precision)) / total).to(dtype);
  cdf = torch::cumsum(cdf, 1).to(dtype);
  cdf.index_put_({torch::indexing::Slice(), N}, 1 << precision);
  auto cdf_contig = cdf.contiguous();
  auto cdf_ptr = cdf_contig.data_ptr<int32_t>();
  at::parallel_for(0, B, 0, [&](size_t start, size_t end) {
    for (size_t b = start; b < end; b++) {
      int32_t* row = cdf_ptr + b * (N + 1);
      for (int i = 0; i < N; ++i) {
        if (row[i] == row[i + 1]) {
          int32_t best_freq = INT32_MAX;
          int best_steal = -1;
          for (int j = 0; j < N; ++j) {
            int32_t f = row[j + 1] - row[j];
            if (f > 1 && f < best_freq) {
              best_freq = f;
              best_steal = j;
            }
          }
          TORCH_CHECK(best_steal != -1, "No symbol to steal frequency from");
          if (best_steal < i) {
            for (int j = best_steal + 1; j <= i; ++j) {
              row[j] -= 1;
            }
          } else {
            TORCH_CHECK(best_steal > i, "best_steal must be > i");
            for (int j = i + 1; j <= best_steal; ++j) {
              row[j] += 1;
            }
          }
        }
      }
    }
  });
  if (pmf.dim() == 1) {
    return cdf_contig[0];
  } else {
    return cdf_contig;
  }
}


TORCH_LIBRARY_IMPL(torch_ans, CPU, m) {
    m.impl("rans64_init_stream", &rans_init_stream<uint64_t, uint32_t>);
    m.impl("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf_cpu);
    m.impl("rans64_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t>);
    m.impl("rans64_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t>);
    m.impl("rans64_i4_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t, false, 4>);
    m.impl("rans64_i4_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, false, false, 4>);
    m.impl("rans64_alias_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t, true>);
    m.impl("rans64_alias_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, true, false>);
    m.impl("rans64_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, false, true>);
    m.impl("rans32_init_stream", &rans_init_stream<uint32_t, uint8_t>);
    m.impl("rans32_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t>);
    m.impl("rans32_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t>);
    m.impl("rans32_i4_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t, false, 4>);
    m.impl("rans32_i4_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, false, 4>);
    m.impl("rans32_i4_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, true, 4>);
    m.impl("rans32_alias_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t, true>);
    m.impl("rans32_alias_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, true, false>);
    m.impl("rans32_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, true>);
    m.impl("rans32_16_init_stream", &rans_init_stream<uint32_t, uint16_t>);
    m.impl("rans32_16_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t>);
    m.impl("rans32_16_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t>);
    m.impl("rans32_16_i4_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, false, 4>);
    m.impl("rans32_16_i4_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, false, 4>);
    m.impl("rans32_16_i4_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, true, 4>);
    m.impl("rans32_16_alias_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, true>);
    m.impl("rans32_16_alias_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, true, false>);
    m.impl("rans32_16_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, true>);
}