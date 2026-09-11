#include <torch/extension.h>

// #include <x86intrin.h>
#include <cmath>
#include <cstring>
#include <vector>
// #include <span>
// #include <bit>

#include "rans.hpp"
#include "rans_utils.hpp"
#include "rans_build_config.hpp"

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, size_t RANS_STATE_VALID_BITS>
torch::Tensor rans_init_stream(int64_t size, int64_t num_interleaves, int64_t preallocate_size) 
{
  // +1 element of slack: the branch-free renormalization in
  // APPEND_STATE_TO_STREAM_IF always writes one word past the cursor (see
  // rans_utils.hpp), so the buffer must have room for it even when the last
  // append does not commit.
  int64_t stream_init_size = 2 + preallocate_size / sizeof(DEFAULT_TORCH_TENSOR_TYPE) + num_interleaves * ((sizeof(RANS_STATE_TYPE) + sizeof(DEFAULT_TORCH_TENSOR_TYPE) - 1) / sizeof(DEFAULT_TORCH_TENSOR_TYPE));
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

    // std::cout << "num_threads" << at::get_num_threads();
    at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      // Everything the inner loops read repeatedly is copied into plain locals
      // *of this body*.
      //
      // This is correctness-neutral but a large performance detail: the lambda
      // captures by reference, so the address of freq_precision and of the
      // accessors escapes and the compiler must assume the words this loop
      // stores into the stream may modify them. It then reloaded freq_precision
      // (through a pointer, i.e. two dependent loads) and the cdf base/stride
      // *per lane*, chaining every lane behind the previous lane's store - the
      // main reason interleaving used to be a slowdown. A local whose address
      // never escapes cannot alias the stream, so it stays in a register.
      scalar_t* const cdfs_data = cdfs.data_ptr<scalar_t>();
      const int64_t cdfs_row_stride = cdfs.stride(0);
      const scalar_t* const cdfs_sizes_data = cdfs_sizes.data_ptr<scalar_t>();
      const int64_t cdfs_sizes_stride = cdfs_sizes.stride(0);
      const scalar_t* const offsets_data = offsets.data_ptr<scalar_t>();
      const int64_t offsets_stride = offsets.stride(0);
      const int64_t num_dists = cdfs.size(0);
      const int64_t num_syms = num_symbols;
      const int64_t fp = freq_precision;
      const bool do_bypass = bypass_coding;
      const int64_t bpp = bypass_precision;
      // std::cout << "range:" << start << end << std::endl;
      for (size_t b = start; b < end; b++) {
        const auto stream_length = stream_accessor[b][0];
        const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
        RANS_STATE_TYPE* state_mem = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
        RANS_STREAM_TYPE* const stream_base = reinterpret_cast<RANS_STREAM_TYPE*>(state_mem);
        RANS_STREAM_TYPE* stream_ptr = stream_base + stream_ptr_offset - 1;

        auto symbols_ptr = symbols_accessor[b].data();
        auto indexes_ptr = indexes_accessor[b].data();
        // reverse coding
        const int64_t first_interleave = num_syms % NUM_INTERLEAVES;
        int64_t i = num_syms-1;
        // The leftover head is coded straight out of the stream tensor: it runs
        // before the main loop, so the states can then be copied into locals
        // *without* any variable-index access to them (see below).
        for (; i >= num_syms-first_interleave; i--) {
          const auto index = indexes_ptr[i];
          // check index range, skip on invalid indexes
          if (index < 0 || index >= num_dists) {
            continue;
          }
          const auto cdf_ptr = cdfs_data + index * cdfs_row_stride;
          const auto cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
          const auto offset = offsets_data[index * offsets_stride];
          const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
          // TODO: check index validity
          rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
            state_mem+(i % NUM_INTERLEAVES), &stream_ptr, symbols_ptr[i],
            cdf_ptr, cdf_size, offset,
            fp, do_bypass, bpp,
            cdf_alias_remap_ptr
          );
        }

        // The interleaved states are kept in locals instead of in the stream
        // tensor they live in. A state inside the tensor aliases the words this
        // loop appends, so the compiler would have to reload every state after
        // each appended word and could never let the (independent) lanes
        // overlap - the very thing interleaving exists for. Every access below
        // uses a compile-time lane index, which is what allows the array to be
        // kept in registers at all.
        RANS_STATE_TYPE states[NUM_INTERLEAVES];
        for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) states[j] = state_mem[j];

        for (; i >= 0; i-=NUM_INTERLEAVES) {

          if constexpr (NUM_INTERLEAVES>1) {
            // Pass 1: gather what every lane needs. None of these loads depend
            // on a rANS state or on the stream cursor, so all NUM_INTERLEAVES of
            // them (the random cdf row accesses in particular, which are the
            // main cache-miss source) are in flight at the same time instead of
            // being serialized behind the previous lane's state update.
            //
            // NOTE: the per-lane loops below are constexpr_for, i.e. fully
            // unrolled with a compile-time lane index. That is not cosmetic: with
            // a runtime index gcc keeps these small arrays (and therefore the
            // states) on the stack, which both adds store-to-load forwarding to
            // every lane and stops it from hoisting the next lane's work - the
            // measured difference is the whole interleaving gain.
            const scalar_t* cdf_ptrs[NUM_INTERLEAVES] = {};
            scalar_t cdf_sizes[NUM_INTERLEAVES] = {};
            scalar_t starts[NUM_INTERLEAVES] = {};
            scalar_t freqs[NUM_INTERLEAVES] = {};
            scalar_t raw_vals[NUM_INTERLEAVES] = {};
            bool bypass_hits[NUM_INTERLEAVES] = {};
            constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
              constexpr size_t j = jc;
              const auto index = indexes_ptr[i-j];
              // check index range, skip on invalid indexes
              if (index >= 0 && index < num_dists) {
                const scalar_t* cdf_ptr = cdfs_data + index * cdfs_row_stride;
                const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
                scalar_t value = symbols_ptr[i-j] - offsets_data[index * offsets_stride];
                scalar_t raw_val = 0;
                if (do_bypass) {
                  const scalar_t max_value = cdf_size - 2;
                  if (value < 0) {
                    raw_val = -2 * value - 1;
                    value = max_value;
                    bypass_hits[j] = true;
                  } else if (value >= max_value) {
                    raw_val = 2 * (value - max_value);
                    value = max_value;
                    bypass_hits[j] = true;
                  }
                }
                cdf_ptrs[j] = cdf_ptr;
                cdf_sizes[j] = cdf_size;
                starts[j] = cdf_ptr[value];
                freqs[j] = cdf_ptr[value + 1] - starts[j];
                raw_vals[j] = raw_val;
              }
            });

            // Pass 2: the bypass raw values of all lanes go in before any
            // symbol of the group (the decoder undoes them in the mirrored
            // order), exactly as the previous two-loop version did.
            if (do_bypass) {
              constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                constexpr size_t j = jc;
                if (bypass_hits[j]) {
                  rans_push_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t>(
                    states+(NUM_INTERLEAVES-1-j), &stream_ptr, raw_vals[j], bpp
                  );
                }
              });
            }

            // Pass 3: the symbol pushes. Only the renormalize/encode arithmetic
            // is left here, all table lookups already happened above.
            constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
              constexpr size_t j = jc;
              if (cdf_ptrs[j] != nullptr) {
                rans_push_step_freq<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t>(
                  states+(NUM_INTERLEAVES-1-j), &stream_ptr, starts[j], freqs[j],
                  fp,
                  (USE_ALIAS_SAMPLING_CDF) ? cdf_ptrs[j] + cdf_sizes[j] : nullptr
                );
              }
            });
          }
          else {
            const auto index = indexes_ptr[i];
            // check index range, skip on invalid indexes
            if (index < 0 || index >= num_dists) {
              continue;
            }
            const auto cdf_ptr = cdfs_data + index * cdfs_row_stride;
            const auto cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
            const auto offset = offsets_data[index * offsets_stride];
            const auto cdf_alias_remap_ptr = (USE_ALIAS_SAMPLING_CDF) ? cdf_ptr + cdf_size : nullptr;
            // TODO: check index validity
            rans_push_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, scalar_t>(
              states, &stream_ptr, symbols_ptr[i],
              cdf_ptr, cdf_size, offset,
              fp, do_bypass, bpp,
              cdf_alias_remap_ptr
            );
          }


        }

        for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) state_mem[j] = states[j];
        // update stream length
        stream_accessor[b][0] = (stream_ptr - stream_base + 1) * sizeof(RANS_STREAM_TYPE);
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

    auto symbols_accessor = symbols.accessor<scalar_t, 2>();
    int64_t num_symbols = symbols_accessor.size(1);

    using RANS_SYMBOL_TYPE = scalar_t;
    using RANS_FREQ_TYPE = scalar_t;

    at::parallel_for(0, batch_size, 0, [&](size_t start, size_t end) {
      // copied into locals of this body, see rans_push_indexed_cpu for why this
      // matters (captured-by-reference values are reloaded after every store
      // into the stream, which serializes the interleaved lanes)
      scalar_t* const cdfs_data = cdfs.data_ptr<scalar_t>();
      const int64_t cdfs_row_stride = cdfs.stride(0);
      const scalar_t* const cdfs_sizes_data = cdfs_sizes.data_ptr<scalar_t>();
      const int64_t cdfs_sizes_stride = cdfs_sizes.stride(0);
      const scalar_t* const offsets_data = offsets.data_ptr<scalar_t>();
      const int64_t offsets_stride = offsets.stride(0);
      const int64_t num_dists = cdfs.size(0);
      const int64_t num_syms = num_symbols;
      const int64_t fp = freq_precision;
      const bool do_bypass = bypass_coding;
      const int64_t bpp = bypass_precision;
      const int64_t inv_fp = inverse_cdf_precision;
      for (size_t b = start; b < end; b++) {
        const auto stream_length = stream_accessor[b][0];
        const auto stream_ptr_offset = stream_length / sizeof(RANS_STREAM_TYPE);
        RANS_STATE_TYPE* state_mem = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data()+1);
        RANS_STREAM_TYPE* const stream_base = reinterpret_cast<RANS_STREAM_TYPE*>(state_mem);
        RANS_STREAM_TYPE* stream_ptr = stream_base + stream_ptr_offset - 1;
        auto symbols_ptr = symbols_accessor[b].data();
        auto indexes_ptr = indexes_accessor[b].data();
        const int64_t last_interleave = num_syms - (NUM_INTERLEAVES-1);
        int64_t i;

        // states in locals, see rans_push_indexed_cpu for why
        RANS_STATE_TYPE states[NUM_INTERLEAVES];
        for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) states[j] = state_mem[j];

        // Stand-in row for lanes whose index is invalid (only produced by the
        // padding of the last block). It makes every lookup of the batched
        // decode below land on entry 0 and terminate immediately, so the fast
        // path needs no per-lane validity test; such lanes are then dropped
        // when the states are advanced. It is padded to the window size because
        // the gap-closing loads of those lanes are still issued (their results
        // are discarded, but the reads must stay in bounds).
        scalar_t dummy_cdf_row[2 + RANS_INVCDF_WINDOW] = {0};
        for (size_t _k = 1; _k < sizeof(dummy_cdf_row) / sizeof(dummy_cdf_row[0]); _k++) {
          dummy_cdf_row[_k] = std::numeric_limits<scalar_t>::max();
        }

        for (i = 0; i < last_interleave; i+=NUM_INTERLEAVES) {

          if constexpr (NUM_INTERLEAVES>1) {
            // Pass 1: the per-distribution metadata of all lanes. Independent
            // of the states, so the loads are issued back to back and the lane
            // searches below can start as soon as their row is there. Lanes with
            // an invalid (padding) index are pointed at a dummy row so that the
            // batched lookup below needs no per-lane guard; they are dropped
            // again in pass 3.
            scalar_t* cdf_ptrs[NUM_INTERLEAVES];
            scalar_t cdf_sizes[NUM_INTERLEAVES] = {};
            scalar_t offs[NUM_INTERLEAVES] = {};
            scalar_t cums[NUM_INTERLEAVES] = {};
            scalar_t out[NUM_INTERLEAVES] = {};
            bool valid[NUM_INTERLEAVES] = {};
            const RANS_STATE_TYPE mask = (1ull << fp) - 1;
            constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
              constexpr size_t j = jc;
              const auto index = indexes_ptr[i+j];
              // check index range, skip on invalid indexes
              if (index >= 0 && index < num_dists) {
                valid[j] = true;
                cdf_ptrs[j] = cdfs_data + index * cdfs_row_stride;
                cdf_sizes[j] = cdfs_sizes_data[index * cdfs_sizes_stride];
                offs[j] = offsets_data[index * offsets_stride];
                cums[j] = (scalar_t)(states[j] & mask);
              } else {
                cdf_ptrs[j] = dummy_cdf_row;
              }
            });

            // Pass 2: locate the symbol of every lane, *step by step across the
            // lanes* instead of lane by lane.
            //
            // This is where interleaving finally pays off on the decode side.
            // The lookup is a chain of ~log2(cdf_size) dependent loads (~5
            // cycles each in L1, more further out) and it is by far the longest
            // dependency in a pop. Running the lanes one after another leaves
            // that chain exposed - the reorder buffer cannot cover four such
            // chains - while walking all lanes through step k before any lane
            // takes step k+1 puts NUM_INTERLEAVES independent loads in flight at
            // every step, so a whole group costs about one chain instead of
            // NUM_INTERLEAVES of them.
            uint32_t bases[NUM_INTERLEAVES] = {};
            uint32_t last[NUM_INTERLEAVES] = {};
            uint32_t cut_bits = 0;
            if constexpr (USE_ALIAS_SAMPLING_CDF) {
              // Alias sampling: the table entry for the bucket
              // cum_freq / cut_size already names the symbol (self or `other`)
              // and the rank within it, so there is no search at all - but the
              // bucket index is a division by a runtime value, i.e. the
              // non-pipelined divider, and that is what keeps this lookup from
              // overlapping across lanes. Gather the entries of all lanes
              // together so the (16 byte, cache-missing) table loads overlap.
              constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                constexpr size_t j = jc;
                const auto* tbl = reinterpret_cast<const RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(
                    cdf_ptrs[j] + cdf_sizes[j]);
                // the dummy row of an invalid (padding) lane has cdf_size 0
                const scalar_t cdf_size_j = cdf_sizes[j] ? cdf_sizes[j] : (scalar_t)2;
                // The bucket is cum_freq / cut_size with
                // cut_size = 2**freq_precision / (cdf_size - 1), and the alias
                // builder only accepts a cdf_size whose (cdf_size - 1) is a
                // power of two - so cut_size is a power of two and the bucket is
                // a shift. Doing it as a division instead costs the
                // non-pipelined integer divider on every symbol, which is what
                // caps the gain from interleaving this lookup. The shift is the
                // exponent of the float representation of that power of two.
                const float cut_size_f = (float)(((uint32_t)1 << fp) / ((uint32_t)cdf_size_j - 1));
                std::memcpy(&cut_bits, &cut_size_f, sizeof(cut_bits));
                const int bucket_shift = (int)((cut_bits >> 23) - 127);
                const RANS_FREQ_TYPE bucket = (RANS_FREQ_TYPE)((uint32_t)cums[j] >> bucket_shift);
                const RANS_FREQ_TYPE cut_cdf = tbl[bucket].cut_cdf;
                const bool to_other = cums[j] >= cut_cdf;
                bases[j] = (uint32_t)(to_other ? tbl[bucket].other_symbol : bucket);
                cums[j] -= to_other ? tbl[bucket].other_alias_offset
                                    : tbl[bucket].self_alias_offset;
              });
            }
            else {
              if constexpr (USE_INVERSED_CDF) {
                const int64_t inv_cdf_precision = (inv_fp > 0) ? inv_fp : fp;
                const int shift = (int)(fp - inv_cdf_precision);
                // one independent table load per lane ...
                constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                  constexpr size_t j = jc;
                  bases[j] = (uint32_t)cdf_ptrs[j][cdf_sizes[j] + (cums[j] >> shift)];
                });
                // ... then the gap closing of every lane, again step by step
                // across the lanes. Unlike the linear walk it replaces, the
                // window loads of a step are independent (they all come from
                // the table's lower bound), so a step issues NUM_INTERLEAVES
                // parallel loads instead of a serial chain, and the count that
                // replaces the walk's comparison is branch-free.
                uint32_t advance[NUM_INTERLEAVES] = {};
                constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                  constexpr size_t j = jc;
                  last[j] = (uint32_t)cdf_sizes[j] - 1;
                });
                for (uint32_t k = 1; k <= RANS_INVCDF_WINDOW; ++k) {
                  constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                    constexpr size_t j = jc;
                    const uint32_t idx = bases[j] + k;
                    // cdf[last] == 1 << fp > cum_freq, so clamping the index
                    // keeps the load in range and contributes 0
                    const uint32_t idx_clamped = (idx < last[j]) ? idx : last[j];
                    advance[j] += (uint32_t)(cdf_ptrs[j][idx_clamped] <= cums[j]);
                  });
                }
                constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                  constexpr size_t j = jc;
                  bases[j] += advance[j];
                  // rare remainder (a bucket spanning more than the window)
                  while (cdf_ptrs[j][bases[j] + 1] <= cums[j]) ++bases[j];
                });
              }
              else {
                // Branch-free search for the last index with cdf[idx] <= cum_freq,
                // identical in outcome to the scalar one in rans_pop_step.
                uint32_t lens[NUM_INTERLEAVES];
                uint32_t max_len = 1;
                constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                  constexpr size_t j = jc;
                  lens[j] = valid[j] ? (uint32_t)cdf_sizes[j] - 1 : 1u;
                  max_len = (lens[j] > max_len) ? lens[j] : max_len;
                });
                // every lane's len follows the same halving recurrence, so it
                // reaches 1 no later than max_len does; lanes that are already
                // done simply take half == 0 steps
                while (max_len > 1) {
                  constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                    constexpr size_t j = jc;
                    const uint32_t half = lens[j] >> 1;
                    bases[j] = (cdf_ptrs[j][bases[j] + half] <= cums[j]) ? (bases[j] + half) : bases[j];
                    lens[j] -= half;
                  });
                  max_len -= max_len >> 1;
                }
              }

              // turn the symbol index into its rank within that symbol, so that
              // the state advance below is shared by all three lookups
              constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                constexpr size_t j = jc;
                cums[j] -= cdf_ptrs[j][bases[j]];
              });
            }

            // Pass 3: advance the states. This is the only part that has to
            // walk the lanes in order (they share the stream cursor), and it
            // is now just a multiply, an add and a branch-free renormalize.
            constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
              constexpr size_t j = jc;
              if (valid[j]) {
                const scalar_t* cdf_ptr = cdf_ptrs[j];
                const RANS_FREQ_TYPE freq = cdf_ptr[bases[j] + 1] - cdf_ptr[bases[j]];
                RANS_STATE_TYPE x = (states[j] >> fp) * freq + cums[j];
                RANS_POP_STATE_RENORM(x, (&stream_ptr));
                states[j] = x;
                out[j] = (RANS_SYMBOL_TYPE)bases[j] + offs[j];
              }
            });

            // postprocess bypass coding after all interleaves (inverse to push step)
            if (do_bypass) {
              constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
                constexpr size_t j = jc;
                if (valid[j]) {
                  const auto max_value = cdf_sizes[j] - 2;
                  scalar_t value = out[j] - offs[j];
                  if (value == max_value) {
                    const auto raw_val = rans_pop_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t>(
                      states+j, &stream_ptr, bpp
                    );
                    value = raw_val >> 1;
                    if (raw_val & 1) {
                      value = -value - 1;
                    } else {
                      value += max_value;
                    }
                    out[j] = value + offs[j];
                  }
                }
              });
            }

            // The decoded symbols are buffered in locals and only written back
            // after the group: a store into the output tensor may alias the cdf
            // rows, which would stop the compiler from overlapping the lanes.
            constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
              constexpr size_t j = jc;
              symbols_ptr[i+j] = out[j];
            });
          }
          else {
            const auto index = indexes_ptr[i];
            // check index range, skip on invalid indexes
            if (index < 0 || index >= num_dists) {
              symbols_ptr[i] = 0;
              continue;
            }
            const auto cdf_ptr = cdfs_data + index * cdfs_row_stride;
            const auto cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
            const auto offset = offsets_data[index * offsets_stride];
            const RANS_FREQ_TYPE* inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
            const auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
            // TODO: check index validity
            symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_FREQ_TYPE>(
              states, &stream_ptr,
              cdf_ptr, cdf_size, offset,
              fp, do_bypass, bpp,
              inversed_cdf_ptr, cdf_alias_table_ptr, inv_fp
            );
          }
        }
        
        for (size_t j = 0; j < static_cast<size_t>(NUM_INTERLEAVES); j++) state_mem[j] = states[j];

        // The leftover tail is decoded straight out of the stream tensor. It has
        // to run after the states are written back so that the loop above never
        // touches `states` with a variable lane index - one such access is
        // enough to make gcc keep the whole array on the stack instead of in
        // registers, and with it goes the interleaving gain.
        for (; i < num_syms; i++) {
            const auto index = indexes_ptr[i];
            // check index range, skip on invalid indexes
            if (index < 0 || index >= num_dists) {
              symbols_ptr[i] = 0;
              continue;
            }

            const auto cdf_ptr = cdfs_data + index * cdfs_row_stride;
            const auto cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
            const auto offsets = offsets_data[index * offsets_stride];
            const RANS_FREQ_TYPE* inversed_cdf_ptr = (USE_INVERSED_CDF) ? (cdf_ptr + cdf_size) : nullptr;
            const auto cdf_alias_table_ptr = (USE_ALIAS_SAMPLING_CDF) ? reinterpret_cast<RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>*>(cdf_ptr + cdf_size) : nullptr;
            // TODO: check index validity
            symbols_ptr[i] = rans_pop_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_FREQ_TYPE>(
              state_mem+(i%NUM_INTERLEAVES), &stream_ptr,
              cdf_ptr, cdf_size, offsets,
              fp, do_bypass, bpp,
              inversed_cdf_ptr, cdf_alias_table_ptr, inv_fp
            );
        }

        // update stream length
        stream_accessor[b][0] = (stream_ptr - stream_base + 1) * sizeof(RANS_STREAM_TYPE);

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


// ---------------------------------------------------------------------------
// Instantiations required by the pybind bindings in rans_bindings.hpp.
//
// The wrappers in rans.hpp only see the declarations of the functions above,
// so every bound combination needs an explicit instantiation definition here
// (see rans_build_config.hpp for why these are not just address-takings any
// more). The blocks are gated exactly like the bindings themselves, so an
// incremental build emits only the requested subset.
// ---------------------------------------------------------------------------

#define TORCH_ANS_CPU_INST_ILV(STATE, STREAM, ILV) \
  TORCH_ANS_INST_PUSH(rans_push_indexed_cpu, STATE, STREAM, false, ILV); \
  TORCH_ANS_INST_POP(rans_pop_indexed_cpu, STATE, STREAM, false, false, ILV)
#define TORCH_ANS_CPU_INST_ILV_INVCDF(STATE, STREAM, ILV) \
  TORCH_ANS_INST_POP(rans_pop_indexed_cpu, STATE, STREAM, false, true, ILV)
#define TORCH_ANS_CPU_INST_ILV_ALIAS(STATE, STREAM, ILV) \
  TORCH_ANS_INST_PUSH(rans_push_indexed_cpu, STATE, STREAM, true, ILV); \
  TORCH_ANS_INST_POP(rans_pop_indexed_cpu, STATE, STREAM, true, false, ILV)

#if TORCH_ANS_WITH_RANS64
TORCH_ANS_INST_INIT(uint64_t, uint32_t);
TORCH_ANS_CPU_INST_ILV(uint64_t, uint32_t, 1);
#  if TORCH_ANS_WITH_INVCDF
TORCH_ANS_CPU_INST_ILV_INVCDF(uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CPU_INST_ILV_ALIAS(uint64_t, uint32_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CPU_INST_ILV(uint64_t, uint32_t, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint64_t, uint32_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint64_t, uint32_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CPU_INST_ILV(uint64_t, uint32_t, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint64_t, uint32_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint64_t, uint32_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CPU_INST_ILV(uint64_t, uint32_t, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint64_t, uint32_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint64_t, uint32_t, 8);
#    endif
#  endif
#endif

#if TORCH_ANS_WITH_RANS32
TORCH_ANS_INST_INIT(uint32_t, uint8_t);
TORCH_ANS_CPU_INST_ILV(uint32_t, uint8_t, 1);
#  if TORCH_ANS_WITH_INVCDF
TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint8_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint8_t, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint8_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint8_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint8_t, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint8_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint8_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint8_t, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint8_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint8_t, 8);
#    endif
#  endif
#endif

#if TORCH_ANS_WITH_RANS32_16
TORCH_ANS_INST_INIT(uint32_t, uint16_t);
TORCH_ANS_CPU_INST_ILV(uint32_t, uint16_t, 1);
#  if TORCH_ANS_WITH_INVCDF
TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_ALIAS
TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint16_t, 1);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_2
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint16_t, 2);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint16_t, 2);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint16_t, 2);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint16_t, 4);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint16_t, 4);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint16_t, 4);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint16_t, 8);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint16_t, 8);
#    endif
#    if TORCH_ANS_WITH_ALIAS
  TORCH_ANS_CPU_INST_ILV_ALIAS(uint32_t, uint16_t, 8);
#    endif
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
  TORCH_ANS_CPU_INST_ILV(uint32_t, uint16_t, 32);
#    if TORCH_ANS_WITH_INVCDF
  TORCH_ANS_CPU_INST_ILV_INVCDF(uint32_t, uint16_t, 32);
#    endif
#  endif
#endif

#undef TORCH_ANS_CPU_INST_ILV
#undef TORCH_ANS_CPU_INST_ILV_INVCDF
#undef TORCH_ANS_CPU_INST_ILV_ALIAS
