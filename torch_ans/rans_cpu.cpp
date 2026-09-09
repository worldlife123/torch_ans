#include <torch/extension.h>

// #include <x86intrin.h>
#include <cmath>
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

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS>
torch::Tensor rans_init_stream(int64_t size, int64_t num_interleaves, int64_t preallocate_size) 
{
  int64_t stream_init_size = 1 + preallocate_size / sizeof(DEFAULT_TORCH_TENSOR_TYPE) + num_interleaves * ((sizeof(RANS_STATE_TYPE) + sizeof(DEFAULT_TORCH_TENSOR_TYPE) - 1) / sizeof(DEFAULT_TORCH_TENSOR_TYPE));
  auto stream = torch::zeros({size, stream_init_size}, torch::TensorOptions().dtype(DEFAULT_TORCH_TENSOR_DTYPE));
  auto stream_accessor = stream.accessor<DEFAULT_TORCH_TENSOR_TYPE, 2>();
  // stream.index_put_({Slice(), 0}, sizeof(RANS_STATE_TYPE));

  at::parallel_for(0, size, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
          stream_accessor[b][0] = sizeof(RANS_STATE_TYPE) * num_interleaves;
          RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
          for (int64_t i = 0; i < num_interleaves; i++) {
            state_ptr[i] = RANS_STATE_LOWER_BOUND;
          }
      }
  });

  return stream;
}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF, size_t NUM_INTERLEAVES>
void rans_push_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& symbols, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding, 
  int64_t bypass_precision)
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
    int64_t num_symbols = symbols_accessor.size(1);

    auto stream_accessor = stream.accessor<scalar_t, 2>();
    auto indexes_accessor = indexes.accessor<scalar_t, 2>();
    auto cdfs_accessor = cdfs.accessor<scalar_t, 2>();
    auto cdfs_sizes_accessor = cdfs_sizes.accessor<scalar_t, 1>();
    auto offsets_accessor = offsets.accessor<scalar_t, 1>();
    // std::cout << "num_threads" << at::get_num_threads();
    at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      // std::cout << "range:" << start << end << std::endl;
      for (size_t b = start; b < end; b++) {
        const auto stream_length = stream_accessor[b][0];
        const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
        RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
        RANS_STREAM_TYPE* stream_ptr = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr) + stream_ptr_offset - 1;
        
        auto symbols_ptr = symbols_accessor[b].data();
        auto indexes_ptr = indexes_accessor[b].data();
        // reverse coding
        const int64_t first_interleave = num_symbols % NUM_INTERLEAVES;
        int64_t i = num_symbols-1;
        for (; i >= num_symbols-first_interleave; i--) {
          const auto index = indexes_ptr[i];
          // check index range, skip on invalid indexes
          if (index < 0 || index >= cdfs_accessor.size(0)) {
            continue;
          }
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
                // check index range, skip on invalid indexes
                if (index < 0 || index >= cdfs_accessor.size(0)) {
                  continue;
                }
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
                // check index range, skip on invalid indexes
                if (index < 0 || index >= cdfs_accessor.size(0)) {
                  continue;
                }
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
                // check index range, skip on invalid indexes
                if (index < 0 || index >= cdfs_accessor.size(0)) {
                  continue;
                }
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
            // check index range, skip on invalid indexes
            if (index < 0 || index >= cdfs_accessor.size(0)) {
              continue;
            }
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

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF, bool USE_INVERSED_CDF, size_t NUM_INTERLEAVES>
torch::Tensor rans_pop_indexed_cpu(// ANSStream stream,
  torch::Tensor stream, 
  const torch::Tensor& indexes, 
  const torch::Tensor& cdfs, 
  const torch::Tensor& cdfs_sizes, 
  const torch::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding, 
  int64_t bypass_precision,
  int64_t inverse_cdf_precision)
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
    int64_t num_symbols = symbols_accessor.size(1);

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
        const int64_t last_interleave = num_symbols - (NUM_INTERLEAVES-1);
        int64_t i;

        for (i = 0; i < last_interleave; i+=NUM_INTERLEAVES) {

          if constexpr (NUM_INTERLEAVES>1) {

            for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
            // constexpr_for<0, NUM_INTERLEAVES, 1>([&](auto j){
              const auto index = indexes_ptr[i+j];
              // check index range, skip on invalid indexes
              if (index < 0 || index >= cdfs_accessor.size(0)) {
                symbols_ptr[i+j] = 0;
                continue;
              }
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
                inversed_cdf_ptr, cdf_alias_table_ptr, inverse_cdf_precision
              );
            }//);

            // postprocess bypass coding after all interleaves (inverse to push step)
            if (bypass_coding) {
              for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) {
              // constexpr_for<0, NUM_INTERLEAVES, 1>([&](auto j){
                const auto index = indexes_ptr[i+j];
                // check index range, skip on invalid indexes
                if (index < 0 || index >= cdfs_accessor.size(0)) {
                  symbols_ptr[i+j] = 0;
                  continue;
                }
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
            // check index range, skip on invalid indexes
            if (index < 0 || index >= cdfs_accessor.size(0)) {
              symbols_ptr[i] = 0;
              continue;
            }
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
              inversed_cdf_ptr, cdf_alias_table_ptr, inverse_cdf_precision
            );
          }
        }
        
        // final symbols
        for (; i < num_symbols; i++) {
            const auto index = indexes_ptr[i];
            // check index range, skip on invalid indexes
            if (index < 0 || index >= cdfs_accessor.size(0)) {
              symbols_ptr[i] = 0;
              continue;
            }

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
              inversed_cdf_ptr, cdf_alias_table_ptr, inverse_cdf_precision
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
//
// B4: the previous implementation was a chain of ~10 torch ops followed by a
// fix-up pass; every op costs a dispatch and a temporary, and init_params runs
// this on every encode/decode call of the dist_freqs API. The loop below does
// the whole thing per row in one pass and is bit-identical to the old chain
// (verified by scripts/bench_cdf_build.py):
//   freq = round(pmf * 2**p) -> cdf = [0, freq...] -> total = sum -> 0 becomes 1
//   -> cdf = (cdf * 2**p) // total  (int32 wrap, truncating division)
//   -> inclusive cumsum -> cdf[N] = 2**p -> "steal frequency" fix-up
torch::Tensor rans_pmf_to_quantized_cdf_cpu(const torch::Tensor& pmf, int64_t precision) {
  auto device = pmf.device();
  auto dtype = torch::kInt32;
  torch::Tensor pmf_batched;
  int64_t B, N;
  if (pmf.dim() == 1) {
    pmf_batched = pmf.unsqueeze(0);
    B = 1;
    N = pmf.size(0);
  } 
  else 
  // If pmf has higher dimensions, we treat the last dimension as the symbol dimension and batch over the preceding dimensions
  {
    pmf_batched = pmf.reshape({-1, pmf.size(-1)});
    B = pmf_batched.size(0);
    N = pmf_batched.size(1);
  }
  const int64_t scale = (int64_t)1 << precision;
  auto cdf_contig = torch::zeros({B, N + 1}, torch::TensorOptions().dtype(dtype).device(device));
  auto cdf_ptr = cdf_contig.data_ptr<int32_t>();

  const torch::Tensor pmf_contig = pmf_batched.contiguous();
  AT_DISPATCH_FLOATING_TYPES(pmf_contig.scalar_type(), "rans_pmf_to_quantized_cdf_cpu", [&] {
    const scalar_t* pmf_ptr = pmf_contig.data_ptr<scalar_t>();
    at::parallel_for(0, B, 0, [&](size_t start, size_t end) {
      for (size_t b = start; b < end; b++) {
        const scalar_t* pmf_row = pmf_ptr + b * N;
        int32_t* row = cdf_ptr + b * (N + 1);
        // freq -> cdf[1..N], cdf[0] = 0
        row[0] = 0;
        int64_t total = 0;
        for (int64_t j = 0; j < N; ++j) {
          const double rounded = std::nearbyint((double)(pmf_row[j] * (scalar_t)scale));
          const int32_t f = (int32_t)rounded;
          row[j + 1] = f;
          total += (int64_t)f;
        }
        int32_t total32 = (int32_t)total;
        if (total32 == 0) total32 = 1;
        // scale, then inclusive cumsum; both wrap like the int32 torch ops did
        int64_t acc = 0;
        for (int64_t j = 0; j <= N; ++j) {
          const int32_t prod = (int32_t)((int64_t)row[j] * scale);
          acc += (int64_t)(prod / total32);
          row[j] = (int32_t)acc;
        }
        row[N] = (int32_t)scale;
        // "steal frequency" fix-up: every symbol must own at least one unit
        for (int64_t i = 0; i < N; ++i) {
          if (row[i] == row[i + 1]) {
            int32_t best_freq = INT32_MAX;
            int best_steal = -1;
            for (int64_t j = 0; j < N; ++j) {
              int32_t f = row[j + 1] - row[j];
              if (f > 1 && f < best_freq) {
                best_freq = f;
                best_steal = (int)j;
              }
            }
            TORCH_CHECK(best_steal != -1, "No symbol to steal frequency from");
            if (best_steal < i) {
              for (int64_t j = best_steal + 1; j <= i; ++j) {
                row[j] -= 1;
              }
            } else {
              TORCH_CHECK(best_steal > i, "best_steal must be > i");
              for (int64_t j = i + 1; j <= best_steal; ++j) {
                row[j] += 1;
              }
            }
          }
        }
      }
    });
  });
  if (pmf.dim() == 1) {
    return cdf_contig[0];
  } else {
    auto sizes = std::vector<int64_t>(pmf.sizes().begin(), pmf.sizes().end()-1);
    sizes.push_back(N+1);
    return cdf_contig.reshape(sizes);
  }
}


// B2: build the combined "cdf ++ inverse-CDF table" tensor the kernels expect.
// Entry i of the table answers the query for the bucket start i << (p - q):
// the largest symbol index whose cdf value is <= that start. cdf[0] is always 0,
// so the binary search below (largest i with cdf[i] <= value) is exactly the
// value the decoder wants - no off-by-one fix-up afterwards.
torch::Tensor rans_build_inverse_cdf_cpu(const torch::Tensor& cdfs, int64_t freq_precision,
                                         int64_t table_precision) {
  TORCH_CHECK(table_precision >= 1 && table_precision <= freq_precision,
              "table_precision must be in [1, ", freq_precision, "], got ", table_precision);
  const bool was_1d = cdfs.dim() == 1;
  const torch::Tensor cdf = (was_1d ? cdfs.unsqueeze(0) : cdfs).contiguous();
  const int64_t B = cdf.size(0);
  const int64_t M = cdf.size(1);
  const int64_t T = (int64_t)1 << table_precision;
  const int64_t shift = freq_precision - table_precision;

  torch::Tensor out = torch::empty({B, M + T}, cdf.options());
  AT_DISPATCH_INTEGRAL_TYPES(cdf.scalar_type(), "rans_build_inverse_cdf_cpu", [&] {
    const scalar_t* cdf_ptr = cdf.data_ptr<scalar_t>();
    scalar_t* out_ptr = out.data_ptr<scalar_t>();
    at::parallel_for(0, B, 0, [&](int64_t begin, int64_t end) {
      for (int64_t b = begin; b < end; ++b) {
        const scalar_t* row = cdf_ptr + b * M;
        scalar_t* out_row = out_ptr + b * (M + T);
        for (int64_t c = 0; c < M; ++c) out_row[c] = row[c];
        // cdf and the bucket starts are both sorted, so a single merge sweep
        // (O(M + T)) replaces T independent binary searches (O(T log M))
        int64_t sym = 0;  // cdf[0] is always 0 <= the first bucket start
        for (int64_t bucket = 0; bucket < T; ++bucket) {
          const int64_t value = bucket << shift;
          while (sym + 1 < M && (int64_t)row[sym + 1] <= value) ++sym;
          out_row[M + bucket] = (scalar_t)sym;
        }
      }
    });
  });

  return was_1d ? out[0] : out;
}


TORCH_LIBRARY_IMPL(torch_ans, CPU, m) {
    m.impl("rans64_init_stream", &rans_init_stream<uint64_t, uint32_t>);
    m.impl("rans_pmf_to_quantized_cdf", &rans_pmf_to_quantized_cdf_cpu);
    m.impl("rans64_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t, false, 1>);
    m.impl("rans64_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t>);
    m.impl("rans64_i4_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t, false, 4>);
    m.impl("rans64_i4_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, false, false, 4>);
    m.impl("rans64_alias_push_indexed", &rans_push_indexed_cpu<uint64_t, uint32_t, true, 1>);
    m.impl("rans64_alias_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, true, false>);
    m.impl("rans64_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, false, true>);
    m.impl("rans64_i4_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint64_t, uint32_t, false, true, 4>);
    m.impl("rans32_init_stream", &rans_init_stream<uint32_t, uint8_t>);
    m.impl("rans32_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t, false, 1>);
    m.impl("rans32_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t>);
    m.impl("rans32_i4_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t, false, 4>);
    m.impl("rans32_i4_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, false, 4>);
    m.impl("rans32_i4_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, true, 4>);
    m.impl("rans32_alias_push_indexed", &rans_push_indexed_cpu<uint32_t, uint8_t, true, 1>);
    m.impl("rans32_alias_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, true, false>);
    m.impl("rans32_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint8_t, false, true>);
    m.impl("rans32_16_init_stream", &rans_init_stream<uint32_t, uint16_t>);
    m.impl("rans32_16_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, false, 1>);
    m.impl("rans32_16_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t>);
    m.impl("rans32_16_i4_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, false, 4>);
    m.impl("rans32_16_i4_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, false, 4>);
    m.impl("rans32_16_i4_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, true, 4>);
    m.impl("rans32_16_i32_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, false, 32>);
    m.impl("rans32_16_i32_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, false, 32>);
    m.impl("rans32_16_i32_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, true, 32>);
    m.impl("rans32_16_alias_push_indexed", &rans_push_indexed_cpu<uint32_t, uint16_t, true, 1>);
    m.impl("rans32_16_alias_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, true, false>);
    m.impl("rans32_16_invcdf_pop_indexed", &rans_pop_indexed_cpu<uint32_t, uint16_t, false, true>);
    
}
