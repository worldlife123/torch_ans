// The tANS family kernels (CPU) and the shared table construction.
//
// Everything algorithmic lives in tans_utils.hpp; this file is the tensor
// plumbing: the table builder (host, parallel over distributions), the stream
// initializer and the two indexed kernels. The op wrappers are in tans.hpp.
//
// The three exported families differ only in (state type, stream word type) -
// see tans.hpp - and are instantiated at the bottom behind the
// TORCH_ANS_WITH_TANS* gates of ans_build_config.hpp.

#include "ta/torch_api.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "ans_build_config.hpp"
#include "tans.hpp"
#include "tans_utils.hpp"
// `constexpr_for`: the compile-time lane loop the interleaved kernels need. It
// lives in rans_utils.hpp next to the rANS kernels that introduced it, and is
// reused instead of being copied here - see rans_cpu.cpp for why a *runtime*
// lane index costs the whole interleaving gain (the lane arrays then live on
// the stack and the lanes cannot overlap).
#include "rans_utils.hpp"

// ---------------------------------------------------------------------------
// table construction
// ---------------------------------------------------------------------------
// One row per distribution, both directions in one tensor: the encoder's
// nextState (spread slots) + symbolTT and the decoder's symbol/newState/nbBits.
// See tans_utils.hpp for the layout and tans_build_tables_row for the algorithm
// (FSE's spread with the normalization step dropped - the cdf is already
// normalized to 2**table_log because table_log == freq_precision here).
//
// A host-only op like rans_alias_build_table: it is built once per init_params
// and then moved to wherever the coding happens, so it stays a CPU operator even
// when the coding itself would move to a GPU.
ta::Tensor tans_build_tables(const ta::Tensor& cdfs, const ta::Tensor& cdfs_sizes,
                             int64_t table_log)
{
  STD_TORCH_CHECK(table_log >= 1 && table_log <= (int64_t)TANS_MAX_TABLELOG,
                  "tans_build_tables: table_log must be in [1, ", (int64_t)TANS_MAX_TABLELOG,
                  "] (got ", table_log, "); the encoder's fixed point arithmetic "
                  "(nbBits = (state + deltaNbBits) >> 16 with minStatePlus in "
                  "(2**table_log, 2**(table_log+1)]) overflows past 15");
  STD_TORCH_CHECK(cdfs.dim() == 2 && cdfs_sizes.dim() == 1 &&
                  cdfs.size(0) == cdfs_sizes.size(0),
                  "tans_build_tables: cdfs must be 2-D with one row per distribution "
                  "and cdfs_sizes 1-D with the same length");
  STD_TORCH_CHECK(ta::is_cpu(cdfs) && ta::is_cpu(cdfs_sizes),
                  "tans_build_tables: the table is built on the host, so cdfs and "
                  "cdfs_sizes must be CPU tensors (move the built table to the "
                  "coding device afterwards, like the alias table does)");
  STD_TORCH_CHECK(cdfs.scalar_type() == ta::ScalarType::Int,
                  "tans_build_tables: cdfs must be an int32 tensor (that is what "
                  "rans_pmf_to_quantized_cdf returns, and what the kernels read)");

  const ta::Tensor cdf = ta::as_contiguous(cdfs);
  const ta::Tensor sizes = ta::as_contiguous(cdfs_sizes);
  ta::Tensor tables;

  TA_DISPATCH_INTEGRAL(sizes.scalar_type(), "tans_build_tables", [&] {
    const int64_t batch = cdf.size(0);
    const int64_t max_symbols = cdf.size(1);  // symbolTT width (the widest cdf row)
    const int64_t ts = (int64_t)1 << table_log;
    const int64_t row_words = tans_table_row_words(ts, max_symbols);

    // zeros: the symbolTT slots of symbols a distribution does not have are
    // never read, but zeroing keeps a row deterministic (see
    // tans_build_tables_row).
    tables = ta::zeros({batch, row_words}, ta::ScalarType::Int);

    const int32_t* const cdf_ptr = ta::const_data_ptr<int32_t>(cdf);
    const int64_t cdf_stride = cdf.stride(0);
    const scalar_t* const sizes_ptr = ta::const_data_ptr<scalar_t>(sizes);
    const int64_t sizes_stride = sizes.stride(0);
    int32_t* const tables_ptr = ta::data_ptr<int32_t>(tables);

    // Errors are collected per row instead of thrown from the parallel body:
    // at::parallel_for's exception support is not part of what this code can
    // rely on across torch versions, and a bad row is a user error we want to
    // report with its distribution index.
    std::vector<std::string> errors((size_t)batch);

    ta::parallel_for(0, batch, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; ++b) {
        const int32_t* const cdf_row = cdf_ptr + b * cdf_stride;
        // cdfs_sizes counts the terminator (and the bypass sentinel when bypass
        // coding is on), so the number of *symbols* to spread is one less than
        // that.
        const int64_t cdf_size = (int64_t)sizes_ptr[b * sizes_stride];
        const int64_t num_symbols = cdf_size - 1;
        int32_t* const row = tables_ptr + b * row_words;
        std::string error;
        if (!tans_build_tables_row(cdf_row, num_symbols, max_symbols, row,
                                   table_log, ts, &error)) {
          errors[(size_t)b] = "tans: distribution " + std::to_string(b) + ": " + error;
        }
      }
    });

    for (const std::string& error : errors) {
      if (!error.empty()) throw std::invalid_argument(error);
    }
  });

  return tables;
}

// ---------------------------------------------------------------------------
// stream initialization
// ---------------------------------------------------------------------------
// The interleave factors with a bound (and instantiated) kernel pair. The
// initializer rejects anything else with that list rather than building a stream
// no kernel can read. 32 is part of the set for the same reason it is for
// rans32_16: the CUDA warp kernels code one row per full warp with it, and the
// CPU kernels below instantiate it too - the stream is a plain interleaved one
// either way, so a 32-lane stream is interchangeable between the two.
#define TANS_ILV_IS_SUPPORTED(num_interleaves) \
  ((num_interleaves) == 1 || (num_interleaves) == 2 || \
   (num_interleaves) == 4 || (num_interleaves) == 8 || (num_interleaves) == 32)

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
ta::Tensor tans_init_stream(int64_t size, int64_t num_interleaves, int64_t preallocate_size)
{
  STD_TORCH_CHECK(TANS_ILV_IS_SUPPORTED(num_interleaves),
                  "tans_init_stream: num_interleaves must be 1, 2, 4, 8 or 32 for tANS "
                  "(got ", num_interleaves, "); 32 is the full-warp group of the CUDA "
                  "interleaved kernels, which the CPU kernels implement as well");
  STD_TORCH_CHECK(preallocate_size >= 0, "tans_init_stream: preallocate_size must be >= 0");

  // +1 for the length word, + the header (one state word per lane plus the
  // shared bitbuf/nbits - see tans_utils.hpp).
  const int64_t header_words = tans_header_words(num_interleaves);
  const int64_t stream_init_size =
      1 + header_words + preallocate_size / (int64_t)sizeof(int32_t);
  ta::Tensor stream = ta::zeros({size, stream_init_size}, ta::ScalarType::Int);
  ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream);
  const int32_t header_bytes = (int32_t)tans_header_bytes(num_interleaves);

  ta::parallel_for(0, size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; ++b) {
      // ta::zeros already cleared every lane's state (0 = "fresh", the first
      // push initializes it to 2**table_log), the bit buffer and its count. The
      // length word counts bytes from the word after it, so it is exactly the
      // header - the same convention rans_init_stream uses for its state words,
      // which is what makes rans_stream_to_byte_strings work unchanged (and why
      // it has to be a whole number of stream words, see tans_utils.hpp).
      stream_view.at(b, 0) = header_bytes;
    }
  });

  return stream;
}

// ---------------------------------------------------------------------------
// push
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
void tans_push_indexed_cpu(ta::Tensor stream, const ta::Tensor& symbols,
                           const ta::Tensor& indexes, const ta::Tensor& tables,
                           const ta::Tensor& cdfs_sizes, const ta::Tensor& offsets,
                           int64_t freq_precision, bool bypass_coding,
                           int64_t bypass_precision)
{
  TA_DISPATCH_INTEGRAL(symbols.scalar_type(), "tans_push_indexed_cpu", [&] {
    STD_TORCH_CHECK(ta::is_cpu(stream), "tans_push: the stream must be a CPU tensor");
    STD_TORCH_CHECK(ta::is_cpu(symbols), "tans_push: symbols must be a CPU tensor");
    STD_TORCH_CHECK(ta::is_cpu(indexes), "tans_push: indexes must be a CPU tensor");
    // ArrayRef comparison (indexes.sizes() == symbols.sizes()) is not part of
    // the stable interface, so compare the two dimensions explicitly.
    STD_TORCH_CHECK(indexes.size(0) == symbols.size(0) && indexes.size(1) == symbols.size(1),
                    "indexes and symbols must have the same shape");

    const uint32_t table_log = (uint32_t)freq_precision;
    const uint32_t ts = (uint32_t)1 << table_log;
    const int64_t batch_size = stream.size(0);
    const int64_t num_syms = symbols.size(1);

    ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream);
    ta::View2<scalar_t> symbols_view = ta::view2<scalar_t>(symbols);
    ta::View2<scalar_t> indexes_view = ta::view2<scalar_t>(indexes);

    // Everything the inner loop reads repeatedly is copied into plain locals of
    // the parallel body - the lambda captures by reference, so the address of
    // these values escapes and the compiler would otherwise assume a store into
    // the stream may modify them (see rans_push_indexed_cpu for the measured
    // effect of that).
    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const scalar_t* const cdfs_sizes_data = ta::const_data_ptr<scalar_t>(cdfs_sizes);
    const int64_t cdfs_sizes_stride = cdfs_sizes.stride(0);
    const scalar_t* const offsets_data = ta::const_data_ptr<scalar_t>(offsets);
    const int64_t offsets_stride = offsets.stride(0);
    const int64_t num_dists = tables.size(0);
    const bool do_bypass = bypass_coding;
    const uint32_t bpp = (uint32_t)bypass_precision;
    // The shared bit buffer's slots: they follow the per-lane states.
    const int64_t bitbuf_slot = tans_bitbuf_slot((int64_t)NUM_INTERLEAVES);
    const int64_t nbits_slot = tans_nbits_slot((int64_t)NUM_INTERLEAVES);

    ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; ++b) {
        int32_t* const header = stream_view.row(b);
        const int64_t length = (int64_t)header[0];
        TANS_STREAM_TYPE* const base = reinterpret_cast<TANS_STREAM_TYPE*>(header + 1);
        TansBitWriter<TANS_STREAM_TYPE> writer{
            (uint64_t)(uint32_t)header[bitbuf_slot],
            (uint32_t)header[nbits_slot],
            base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};

        const scalar_t* const symbols_row = symbols_view.row(b);
        const scalar_t* const indexes_row = indexes_view.row(b);

        // Reverse order: ANS is LIFO, so the decoder meets the symbols in
        // forward order. The lane of a symbol is its index % NUM_INTERLEAVES,
        // exactly as in the rANS kernels.
        const int64_t first_interleave = num_syms % NUM_INTERLEAVES;
        int64_t i = num_syms - 1;
        // The leftover head is coded straight out of the stream tensor: it runs
        // before the main loop, so the states can then be copied into locals
        // *without* any variable-index access to them (see below).
        for (; i >= num_syms - first_interleave; --i) {
          const scalar_t index = indexes_row[i];
          // check index range, skip on invalid indexes (batched padding)
          if (index < 0 || index >= num_dists) continue;
          const int64_t lane = i % NUM_INTERLEAVES;
          const int32_t* const row = tables_data + index * tables_row_stride;
          const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
          const scalar_t offset = offsets_data[index * offsets_stride];
          uint32_t value = 0;
          uint32_t raw_val = 0;
          bool bypassed = false;
          tans_cook_symbol<scalar_t>(symbols_row[i], cdf_size, offset, do_bypass,
                                     &value, &raw_val, &bypassed);
          uint32_t state = (uint32_t)header[tans_state_slot(lane)];
          // 0 is not a legal encoder state (those live in [2**L, 2**(L+1))), so
          // it is the "stream has not been pushed to yet" marker - which is how
          // tans_init_stream can keep rans_init_stream's signature (it does not
          // know table_log).
          if (state == 0) state = ts;
          // The bypass digits go in before the sentinel symbol; the decoder
          // meets them in the mirrored order (see tans_utils.hpp).
          if (bypassed) tans_push_raw_value<TANS_STREAM_TYPE>(&writer, raw_val, bpp);
          state = tans_encode_symbol<TANS_STREAM_TYPE>(state, row, ts, value, &writer);
          header[tans_state_slot(lane)] = (int32_t)state;
        }

        // The interleaved states are kept in locals instead of in the stream
        // tensor they live in. A state inside the tensor aliases the words this
        // loop appends, so the compiler would have to reload every state after
        // each appended word and could never let the (independent) lanes
        // overlap - the very thing interleaving exists for. Every access below
        // uses a compile-time lane index, which is what allows the array to be
        // kept in registers at all.
        uint32_t states[NUM_INTERLEAVES];
        constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
          constexpr size_t j = jc;
          const uint32_t state = (uint32_t)header[tans_state_slot((int64_t)j)];
          states[j] = (state == 0) ? ts : state;
        });

        for (; i >= 0; i -= (int64_t)NUM_INTERLEAVES) {
          // Pass 1: gather what every lane needs. None of these loads depends
          // on a tANS state or on the bit cursor, so all NUM_INTERLEAVES of them
          // (the random table row accesses in particular, which are the main
          // cache-miss source) are in flight at the same time instead of being
          // serialized behind the previous lane's state update.
          const int32_t* rows[NUM_INTERLEAVES] = {};
          const int32_t* symbol_tt[NUM_INTERLEAVES] = {};
          uint32_t raw_vals[NUM_INTERLEAVES] = {};
          bool bypass_hits[NUM_INTERLEAVES] = {};
          bool valid[NUM_INTERLEAVES] = {};
          // The array is indexed by *lane*: the lane of a symbol is its index %
          // NUM_INTERLEAVES, and a group covers the ILV symbols i-ILV+1 .. i, so
          // lane L belongs to the symbol i-ILV+1+L.
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            const int64_t symbol_index = i - ((int64_t)NUM_INTERLEAVES - 1 - (int64_t)j);
            const scalar_t index = indexes_row[symbol_index];
            // check index range, skip on invalid indexes (batched padding)
            if (index < 0 || index >= num_dists) return;
            valid[j] = true;
            rows[j] = tables_data + index * tables_row_stride;
            const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
            const scalar_t offset = offsets_data[index * offsets_stride];
            uint32_t value = 0;
            tans_cook_symbol<scalar_t>(symbols_row[symbol_index], cdf_size, offset,
                                       do_bypass, &value, &raw_vals[j], &bypass_hits[j]);
            // only a pointer here: the two words are loaded in pass 2, together
            // with every other lane's
            symbol_tt[j] = tans_symbol_tt_row(rows[j], ts, value);
          });

          // Pass 2: nbBits and the next-state index, from the transformation
          // table every lane just pointed at. It depends on that lane's state
          // only, so the table loads of all lanes overlap.
          uint32_t nb_bits[NUM_INTERLEAVES] = {};
          int64_t next_index[NUM_INTERLEAVES] = {};
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            if (!valid[j]) return;
            nb_bits[j] = tans_encode_nb_bits(states[j], symbol_tt[j]);
            next_index[j] = tans_encode_next_index(states[j], symbol_tt[j], nb_bits[j]);
          });

          // Pass 3: the bits themselves. The bit buffer is shared, so this is
          // the only part that has to walk the lanes in order - and it is no
          // more than a shift, a mask and a store per lane. The bypass digits of
          // a lane go in before its sentinel symbol, mirrored by the decoder.
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            constexpr size_t lane = NUM_INTERLEAVES - 1 - j;  // highest lane first
            if (!valid[lane]) return;
            if (bypass_hits[lane]) {
              tans_push_raw_value<TANS_STREAM_TYPE>(&writer, raw_vals[lane], bpp);
            }
            writer.append(states[lane], nb_bits[lane]);
          });

          // Pass 4: the next state of every lane - one dependent load each, all
          // of them in flight at the same time.
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            if (valid[j]) states[j] = tans_encode_next_state(rows[j], next_index[j]);
          });
        }

        constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
          constexpr size_t j = jc;
          header[tans_state_slot((int64_t)j)] = (int32_t)states[j];
        });
        // At rest the bit buffer holds fewer than STREAM_BITS bits (append()
        // drains while nbits >= STREAM_BITS), so both fit in the header words.
        header[bitbuf_slot] = (int32_t)(uint32_t)writer.buf;
        header[nbits_slot] = (int32_t)writer.nbits;
        // The payload is whole stream words only: the residual bits stay in the
        // header, so the length is unambiguous and the reader needs no trailer.
        header[0] = (int32_t)((writer.ptr - base + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
      }
    });
  });
}

// ---------------------------------------------------------------------------
// pop
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE, size_t NUM_INTERLEAVES>
ta::Tensor tans_pop_indexed_cpu(ta::Tensor stream, const ta::Tensor& indexes,
                                const ta::Tensor& tables, const ta::Tensor& cdfs_sizes,
                                const ta::Tensor& offsets, int64_t freq_precision,
                                bool bypass_coding, int64_t bypass_precision)
{
  ta::Tensor symbols = ta::zeros_like(indexes);

  TA_DISPATCH_INTEGRAL(symbols.scalar_type(), "tans_pop_indexed_cpu", [&] {
    STD_TORCH_CHECK(ta::is_cpu(stream), "tans_pop: the stream must be a CPU tensor");
    STD_TORCH_CHECK(ta::is_cpu(indexes), "tans_pop: indexes must be a CPU tensor");

    const uint32_t table_log = (uint32_t)freq_precision;
    const uint32_t ts = (uint32_t)1 << table_log;
    const int64_t batch_size = stream.size(0);
    const int64_t num_syms = symbols.size(1);

    ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream);
    ta::View2<scalar_t> symbols_view = ta::view2<scalar_t>(symbols);
    ta::View2<scalar_t> indexes_view = ta::view2<scalar_t>(indexes);

    const int32_t* const tables_data = ta::const_data_ptr<int32_t>(tables);
    const int64_t tables_row_stride = tables.stride(0);
    const scalar_t* const cdfs_sizes_data = ta::const_data_ptr<scalar_t>(cdfs_sizes);
    const int64_t cdfs_sizes_stride = cdfs_sizes.stride(0);
    const scalar_t* const offsets_data = ta::const_data_ptr<scalar_t>(offsets);
    const int64_t offsets_stride = offsets.stride(0);
    const int64_t num_dists = tables.size(0);
    const bool do_bypass = bypass_coding;
    const uint32_t bpp = (uint32_t)bypass_precision;
    const int64_t bitbuf_slot = tans_bitbuf_slot((int64_t)NUM_INTERLEAVES);
    const int64_t nbits_slot = tans_nbits_slot((int64_t)NUM_INTERLEAVES);

    ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; ++b) {
        // not const: the continuation point is written back below
        int32_t* const header = stream_view.row(b);
        const int64_t length = (int64_t)header[0];
        const TANS_STREAM_TYPE* const base =
            reinterpret_cast<const TANS_STREAM_TYPE*>(header + 1);
        // The reader walks the payload backwards from its end: the encoder wrote
        // whole words forward, the bits are consumed in reverse order.
        TansBitReader<TANS_STREAM_TYPE> reader{
            (uint64_t)(uint32_t)header[bitbuf_slot],
            (uint32_t)header[nbits_slot],
            base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};
        // The two directions use different ranges for the same state: the
        // encoder's lives in [ts, 2*ts), the decoder's in [0, ts). FSE bridges
        // that by flushing only the low tableLog bits of the final encoder state
        // (bit L is always 1 there), i.e. the decoder's initial state is
        // `encoder_state - ts` - which is exactly the conversion below, with the
        // encoder state travelling in the header instead of in the bit stream.
        uint32_t states[NUM_INTERLEAVES];
        constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
          constexpr size_t j = jc;
          states[j] = (uint32_t)header[tans_state_slot((int64_t)j)] - ts;
        });

        scalar_t* const symbols_row = symbols_view.row(b);
        const scalar_t* const indexes_row = indexes_view.row(b);

        const int64_t last_interleave = num_syms - ((int64_t)NUM_INTERLEAVES - 1);
        int64_t i = 0;
        for (; i < last_interleave; i += (int64_t)NUM_INTERLEAVES) {
          // Pass 1: the per-distribution metadata and the two table planes of
          // every lane. The loads are indexed by that lane's own state, so they
          // are independent of each other and all NUM_INTERLEAVES of them are in
          // flight at the same time - the table row of a lane is the main
          // cache-miss source and the reason this pass exists.
          const int32_t* rows[NUM_INTERLEAVES] = {};
          scalar_t max_values[NUM_INTERLEAVES] = {};
          scalar_t offs[NUM_INTERLEAVES] = {};
          uint32_t words[NUM_INTERLEAVES] = {};
          uint32_t out_syms[NUM_INTERLEAVES] = {};
          bool valid[NUM_INTERLEAVES] = {};
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            const int64_t symbol_index = i + (int64_t)j;
            const scalar_t index = indexes_row[symbol_index];
            // check index range, skip on invalid indexes (batched padding)
            if (index < 0 || index >= num_dists) return;
            valid[j] = true;
            rows[j] = tables_data + index * tables_row_stride;
            const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
            max_values[j] = cdf_size - 2;
            offs[j] = offsets_data[index * offsets_stride];
            words[j] = tans_decode_word(rows[j], ts, states[j]);
            out_syms[j] = tans_decode_symbol_of(rows[j], ts, states[j]);
          });

          // Pass 2: take the bits of every lane - in lane order, because the
          // buffer is one bit sequence shared by all of them (the encoder
          // appended them in exactly this order, see tans_push_indexed_cpu). The
          // bypass digits of a lane follow its sentinel symbol.
          uint32_t low_bits[NUM_INTERLEAVES] = {};
          scalar_t values[NUM_INTERLEAVES] = {};
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            if (!valid[j]) return;
            low_bits[j] = reader.take(tans_decode_nb_bits(words[j]));
            scalar_t value = (scalar_t)out_syms[j];
            if (do_bypass && out_syms[j] == (uint32_t)max_values[j]) {
              const uint32_t raw_val = tans_pop_raw_value<TANS_STREAM_TYPE>(&reader, bpp);
              value = (scalar_t)tans_uncook_raw_value(raw_val, (int64_t)max_values[j]);
            }
            values[j] = value;
          });

          // Pass 3: the new state and the decoded symbol of every lane. Both are
          // arithmetic now, so the lanes run in parallel; the stores are buffered
          // in `values` because a store into the output tensor may alias the
          // table rows.
          constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
            constexpr size_t j = jc;
            if (!valid[j]) return;
            states[j] = tans_decode_new_state(words[j], low_bits[j]);
            symbols_row[i + (int64_t)j] = values[j] + offs[j];
          });
        }

        // The leftover tail is decoded straight out of the stream tensor. It has
        // to run after the states are written back so that the loop above never
        // touches `states` with a variable lane index - one such access is
        // enough to make gcc keep the whole array on the stack instead of in
        // registers, and with it goes the interleaving gain.
        constexpr_for<size_t(0), size_t(NUM_INTERLEAVES), size_t(1)>([&](auto jc) {
          constexpr size_t j = jc;
          header[tans_state_slot((int64_t)j)] = (int32_t)(states[j] + ts);
        });
        for (; i < num_syms; ++i) {
          const scalar_t index = indexes_row[i];
          // check index range, skip on invalid indexes (batched padding)
          if (index < 0 || index >= num_dists) {
            symbols_row[i] = 0;
            continue;
          }
          const int64_t lane = i % NUM_INTERLEAVES;
          const int32_t* const row = tables_data + index * tables_row_stride;
          const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
          const scalar_t offset = offsets_data[index * offsets_stride];
          const scalar_t max_value = cdf_size - 2;

          uint32_t state = (uint32_t)header[tans_state_slot(lane)] - ts;
          uint32_t symbol = 0;
          state = tans_decode_symbol<TANS_STREAM_TYPE>(state, row, ts, &reader, &symbol);
          scalar_t value = (scalar_t)symbol;

          if (do_bypass) {
            if (symbol == (uint32_t)max_value) {
              const uint32_t raw_val =
                  tans_pop_raw_value<TANS_STREAM_TYPE>(&reader, bpp);
              value = (scalar_t)tans_uncook_raw_value(raw_val, (int64_t)max_value);
            }
          }

          symbols_row[i] = value + offset;
          header[tans_state_slot(lane)] = (int32_t)(state + ts);
        }

        // Write the continuation point back, exactly like rans_pop does (it
        // rewinds the length word and leaves the state in place): the reader's
        // position *is* a tANS header, so a second pop on the same stream
        // decodes the previous push - which is what the chunked decode path
        // relies on. The state goes back in the encoder's range (`+ ts`) so the
        // header keeps one meaning for both directions; a decoder state of 0 is
        // a legal state and `+ ts` keeps it out of the "fresh stream" sentinel.
        //
        // The residual bits (`reader.buf`) hold the highest bits still unread,
        // which is the same thing the encoder's writer holds between pushes, so
        // the header round-trips through a push/pop pair unchanged.
        header[0] = (int32_t)((reader.ptr - base + 1) * (int64_t)sizeof(TANS_STREAM_TYPE));
        header[bitbuf_slot] = (int32_t)(uint32_t)reader.buf;
        header[nbits_slot] = (int32_t)reader.nbits;
      }
    });
  });

  return symbols;
}

// ---------------------------------------------------------------------------
// Instantiations required by the bindings in ans_bindings.hpp.
// ---------------------------------------------------------------------------
// The wrappers in tans.hpp only see the declarations above, so every bound
// family needs an explicit instantiation definition here - the same reason as
// the rANS blocks in rans_cpu.cpp (ans_build_config.hpp spells it out). The
// family gates are per family, the interleave ones are shared with the rANS
// families, so an incremental build compiles only the requested subset.
#define TORCH_ANS_TANS_INST_ILV(STATE, STREAM, ILV) \
  TORCH_ANS_INST_TANS_PUSH(STATE, STREAM, ILV); \
  TORCH_ANS_INST_TANS_POP(STATE, STREAM, ILV)

#if TORCH_ANS_WITH_TANS32
TORCH_ANS_INST_TANS_INIT(uint32_t, uint16_t);
TORCH_ANS_TANS_INST_ILV(uint32_t, uint16_t, 1);
#  if TORCH_ANS_WITH_INTERLEAVE_2
TORCH_ANS_TANS_INST_ILV(uint32_t, uint16_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
TORCH_ANS_TANS_INST_ILV(uint32_t, uint16_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
TORCH_ANS_TANS_INST_ILV(uint32_t, uint16_t, 8);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
// 32 lanes is the full warp of the CUDA interleaved kernels (tans_warp_cuda.cuh);
// the CPU kernels implement it from the same constexpr_for lane loop, so a 32-lane
// stream is interchangeable between the two.
TORCH_ANS_TANS_INST_ILV(uint32_t, uint16_t, 32);
#  endif
#endif

#if TORCH_ANS_WITH_TANS16
TORCH_ANS_INST_TANS_INIT(uint16_t, uint8_t);
TORCH_ANS_TANS_INST_ILV(uint16_t, uint8_t, 1);
#  if TORCH_ANS_WITH_INTERLEAVE_2
TORCH_ANS_TANS_INST_ILV(uint16_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
TORCH_ANS_TANS_INST_ILV(uint16_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
TORCH_ANS_TANS_INST_ILV(uint16_t, uint8_t, 8);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
TORCH_ANS_TANS_INST_ILV(uint16_t, uint8_t, 32);
#  endif
#endif

#if TORCH_ANS_WITH_TANS8
TORCH_ANS_INST_TANS_INIT(uint8_t, uint8_t);
TORCH_ANS_TANS_INST_ILV(uint8_t, uint8_t, 1);
#  if TORCH_ANS_WITH_INTERLEAVE_2
TORCH_ANS_TANS_INST_ILV(uint8_t, uint8_t, 2);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_4
TORCH_ANS_TANS_INST_ILV(uint8_t, uint8_t, 4);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_8
TORCH_ANS_TANS_INST_ILV(uint8_t, uint8_t, 8);
#  endif
#  if TORCH_ANS_WITH_INTERLEAVE_32
TORCH_ANS_TANS_INST_ILV(uint8_t, uint8_t, 32);
#  endif
#endif

#undef TORCH_ANS_TANS_INST_ILV
