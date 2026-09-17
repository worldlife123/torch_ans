// The tANS family kernels (CPU) and the shared table construction.
//
// Everything algorithmic lives in tans_utils.hpp; this file is the tensor
// plumbing: the table builder (host, parallel over distributions), the stream
// initializer and the two indexed kernels. The op wrappers are in tans.hpp.
//
// The three exported families differ only in (state type, stream word type) -
// see tans.hpp - and are instantiated at the bottom behind the
// TORCH_ANS_WITH_TANS* gates of rans_build_config.hpp.

#include "ta/torch_api.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans_build_config.hpp"
#include "tans.hpp"
#include "tans_utils.hpp"

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
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
ta::Tensor tans_init_stream(int64_t size, int64_t num_interleaves, int64_t preallocate_size)
{
  STD_TORCH_CHECK(num_interleaves == 1,
                  "tans_init_stream: only num_interleaves=1 is implemented for tANS "
                  "(got ", num_interleaves, "); an interleaved tANS stream needs one "
                  "state per lane *and* a shared bit buffer, which the kernels do not "
                  "implement yet");
  STD_TORCH_CHECK(preallocate_size >= 0, "tans_init_stream: preallocate_size must be >= 0");

  // +1 for the length word, + the header (state/bitbuf/nbits, see tans_utils.hpp).
  const int64_t stream_init_size =
      1 + TANS_HEADER_WORDS + preallocate_size / (int64_t)sizeof(int32_t);
  ta::Tensor stream = ta::zeros({size, stream_init_size}, ta::ScalarType::Int);
  ta::View2<int32_t> stream_view = ta::view2<int32_t>(stream);

  ta::parallel_for(0, size, 0, [&](int64_t start, int64_t end) {
    for (int64_t b = start; b < end; ++b) {
      // ta::zeros already cleared the state (0 = "fresh", the first push
      // initializes it to 2**table_log), the bit buffer and its count. The
      // length word counts bytes from the word after it, so it is exactly the
      // header - the same convention rans_init_stream uses for its state words,
      // which is what makes rans_stream_to_byte_strings work unchanged.
      stream_view.at(b, 0) = (int32_t)TANS_HEADER_BYTES;
    }
  });

  return stream;
}

// ---------------------------------------------------------------------------
// push
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
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

    ta::parallel_for(0, batch_size, 0, [&](int64_t start, int64_t end) {
      for (int64_t b = start; b < end; ++b) {
        int32_t* const header = stream_view.row(b);
        const int64_t length = (int64_t)header[0];
        TANS_STREAM_TYPE* const base = reinterpret_cast<TANS_STREAM_TYPE*>(header + 1);
        TansBitWriter<TANS_STREAM_TYPE> writer{
            (uint64_t)(uint32_t)header[TANS_BITBUF_SLOT],
            (uint32_t)header[TANS_NBITS_SLOT],
            base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};
        uint32_t state = (uint32_t)header[TANS_STATE_SLOT];
        // 0 is not a legal encoder state (those live in [2**L, 2**(L+1))), so it
        // is the "stream has not been pushed to yet" marker - which is how
        // tans_init_stream can keep rans_init_stream's signature (it does not
        // know table_log).
        if (state == 0) state = ts;

        const scalar_t* const symbols_row = symbols_view.row(b);
        const scalar_t* const indexes_row = indexes_view.row(b);
        // Reverse order: ANS is LIFO, so the decoder meets the symbols in
        // forward order.
        for (int64_t i = num_syms - 1; i >= 0; --i) {
          const scalar_t index = indexes_row[i];
          // check index range, skip on invalid indexes (batched padding)
          if (index < 0 || index >= num_dists) continue;
          const int32_t* const row = tables_data + index * tables_row_stride;
          const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
          const scalar_t offset = offsets_data[index * offsets_stride];
          const scalar_t max_value = cdf_size - 2;
          scalar_t value = symbols_row[i] - offset;
          bool bypassed = false;
          uint32_t raw_val = 0;
          if (do_bypass) {
            // Same zig-zag as the rANS bypass (rans_utils.hpp):
            //   raw = -2 * value - 1     for value < 0
            //   raw =  2 * (value - max) for value >= max
            // The distance from the coded range must stay below
            // 2^(bits(symbol) - 2), i.e. ~2^30 for int32 - beyond that the
            // multiply overflows and the symbol is silently mis-coded.
            if (value < 0) {
              raw_val = (uint32_t)(-2 * (int64_t)value - 1);
              value = max_value;
              bypassed = true;
            } else if (value >= max_value) {
              raw_val = (uint32_t)(2 * ((int64_t)value - max_value));
              value = max_value;
              bypassed = true;
            }
          } else {
            // No bypass coding: an out-of-range symbol has no representation.
            // Clamp it into [0, max_value] rather than indexing the symbolTT row
            // out of bounds (with bypass *off* max_value is a legal symbol - it
            // is the last cdf entry - so only values above it are clamped).
            if (value < 0) value = 0;
            else if (value > max_value) value = max_value;
          }
          // The bypass digits go in before the sentinel symbol; the decoder
          // meets them in the mirrored order (see tans_utils.hpp).
          if (bypassed) tans_push_raw_value<TANS_STREAM_TYPE>(&writer, raw_val, bpp);
          state = tans_encode_symbol<TANS_STREAM_TYPE>(state, row, ts, (uint32_t)value,
                                                       &writer);
        }

        header[TANS_STATE_SLOT] = (int32_t)state;
        // At rest the bit buffer holds fewer than STREAM_BITS bits (append()
        // drains while nbits >= STREAM_BITS), so both fit in the header words.
        header[TANS_BITBUF_SLOT] = (int32_t)(uint32_t)writer.buf;
        header[TANS_NBITS_SLOT] = (int32_t)writer.nbits;
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
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
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
            (uint64_t)(uint32_t)header[TANS_BITBUF_SLOT],
            (uint32_t)header[TANS_NBITS_SLOT],
            base + length / (int64_t)sizeof(TANS_STREAM_TYPE) - 1};
        // The two directions use different ranges for the same state: the
        // encoder's lives in [ts, 2*ts), the decoder's in [0, ts). FSE bridges
        // that by flushing only the low tableLog bits of the final encoder state
        // (bit L is always 1 there), i.e. the decoder's initial state is
        // `encoder_state - ts` - which is exactly the conversion below, with the
        // encoder state travelling in the header instead of in the bit stream.
        uint32_t state = (uint32_t)header[TANS_STATE_SLOT] - ts;

        scalar_t* const symbols_row = symbols_view.row(b);
        const scalar_t* const indexes_row = indexes_view.row(b);
        for (int64_t i = 0; i < num_syms; ++i) {
          const scalar_t index = indexes_row[i];
          // check index range, skip on invalid indexes (batched padding)
          if (index < 0 || index >= num_dists) {
            symbols_row[i] = 0;
            continue;
          }
          const int32_t* const row = tables_data + index * tables_row_stride;
          const scalar_t cdf_size = cdfs_sizes_data[index * cdfs_sizes_stride];
          const scalar_t offset = offsets_data[index * offsets_stride];
          const scalar_t max_value = cdf_size - 2;

          uint32_t symbol = 0;
          state = tans_decode_symbol<TANS_STREAM_TYPE>(state, row, ts, &reader, &symbol);
          scalar_t value = (scalar_t)symbol;

          if (do_bypass) {
            if (symbol == (uint32_t)max_value) {
              const uint32_t raw_val =
                  tans_pop_raw_value<TANS_STREAM_TYPE>(&reader, bpp);
              value = (scalar_t)(raw_val >> 1);
              if (raw_val & 1u) {
                value = -value - 1;
              } else {
                value = value + max_value;
              }
            }
          }

          symbols_row[i] = value + offset;
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
        header[TANS_STATE_SLOT] = (int32_t)(state + ts);
        header[TANS_BITBUF_SLOT] = (int32_t)(uint32_t)reader.buf;
        header[TANS_NBITS_SLOT] = (int32_t)reader.nbits;
      }
    });
  });

  return symbols;
}

// ---------------------------------------------------------------------------
// Instantiations required by the bindings in rans_bindings.hpp.
// ---------------------------------------------------------------------------
// The wrappers in tans.hpp only see the declarations above, so every bound
// family needs an explicit instantiation definition here - the same reason as
// the rANS blocks in rans_cpu.cpp (rans_build_config.hpp spells it out). The
// gate is per family, so an incremental build compiles only the requested one.
#if TORCH_ANS_WITH_TANS32
TORCH_ANS_INST_TANS_INIT(uint32_t, uint16_t);
TORCH_ANS_INST_TANS_PUSH(uint32_t, uint16_t);
TORCH_ANS_INST_TANS_POP(uint32_t, uint16_t);
#endif

#if TORCH_ANS_WITH_TANS16
TORCH_ANS_INST_TANS_INIT(uint16_t, uint8_t);
TORCH_ANS_INST_TANS_PUSH(uint16_t, uint8_t);
TORCH_ANS_INST_TANS_POP(uint16_t, uint8_t);
#endif

#if TORCH_ANS_WITH_TANS8
TORCH_ANS_INST_TANS_INIT(uint8_t, uint8_t);
TORCH_ANS_INST_TANS_PUSH(uint8_t, uint8_t);
TORCH_ANS_INST_TANS_POP(uint8_t, uint8_t);
#endif
