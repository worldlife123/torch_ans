#pragma once

// ---------------------------------------------------------------------------
// tANS (tabled ANS / FSE) core: table construction, the encode/decode steps and
// the bit layer. Plain C++, no torch: the file is host-only today (the tANS
// family is CPU-only, see tans.hpp) but is written the way rans_utils.hpp is,
// so a CUDA port can include it verbatim.
//
// Attribution: the table construction and the two symbol steps are the
// algorithm of FSE (Yann Collet, "FiniteStateEntropy", BSD-2 / GPLv2 dual
// licence) as it is also spelled out in 3rdparty/ans/tans.cpp of this
// repository. Only the algorithm is taken - FSE itself is *not* compiled into
// torch_ans (this project is MIT), and the bit layer below is torch_ans's own
// (word-oriented, sharing the rANS stream tensor layout) rather than FSE's
// byte-oriented bit container, so the two are not bitstream-compatible.
//
// Why the bit layer is different: a rANS state absorbs the not-yet-flushed bits
// (that is what the renormalization threshold is about), so a rANS stream only
// ever holds whole stream words. A tANS state is *fixed width* (x in
// [2**L, 2**(L+1))), so the bits it emits per symbol (0..L of them) have to go
// somewhere else. They go into a bit buffer that lives in the stream header
// next to the state (see TANS_HEADER_WORDS below), which keeps the two
// properties the rest of the library relies on:
//   * the residual bits are part of the serialized byte string, exactly like a
//     rANS state is - so there is no flush operator and no terminator bit;
//   * the payload itself is only ever written in whole stream words, so the
//     reader's start position is unambiguous (no trailer padding to skip).
// ---------------------------------------------------------------------------

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// Marked like rans_utils.hpp's RANS_API: `static` on the host, `__device__`
// under RANS_CUDA_API, so a future CUDA port reuses this header unchanged.
#ifdef RANS_CUDA_API
#define TANS_API __device__
#define TANS_INLINE __device__ __forceinline__
#else
#define TANS_API static
#define TANS_INLINE inline
#endif

// ---------------------------------------------------------------------------
// limits and table layout
// ---------------------------------------------------------------------------
// `L` below is table_log (== freq_precision for this implementation, see
// tans_build_tables). It is bounded by the 16.16 fixed point arithmetic of the
// encode step: nbBits = (x + deltaNbBits) >> 16 with deltaNbBits =
// (maxBitsOut << 16) - minStatePlus, minStatePlus in (2**L, 2**(L+1)] and
// x in [2**L, 2**(L+1)), so the sum stays non-negative only while
// 2**16 - 2**L > 0, i.e. L <= 15 - whatever the state's own width. The state
// width adds its own bound (x must fit: L <= state_bits - 1), which is what
// makes tans8 (8-bit state) stop at 7.
#define TANS_MAX_TABLELOG 15

// Encoder state range, as a reminder of the two roles a tANS state plays:
// encoding x in [2**L, 2**(L+1)), decoding x in [0, 2**L).
// (No macro here: `ts` is computed from table_log at runtime.)

// ---------------------------------------------------------------------------
// table row layout (int32 words, built by tans_build_tables)
// ---------------------------------------------------------------------------
// The state tables come *first* and the per-symbol transformation table last,
// so that every offset follows from `table_log` alone - the kernels never need
// to know `max_symbols` (the row width of the symbolTT part), which is why the
// tANS push/pop operators can keep the rANS signatures instead of taking the
// table geometry as extra arguments.
//
//   ts = 1 << table_log          (number of spread slots)
//   M  = cdfs.size(1)            (symbolTT width in symbols)
//
//   [0,     ts)    nextState[ts]                  encoder
//   [ts,  2*ts)   newState | (nbBits << 16)       decoder
//   [2*ts, 3*ts)  symbol[ts]                      decoder
//   [3*ts, 3*ts + 2*M)  (deltaNbBits[s], deltaFindState[s]) interleaved
//
// `newState < ts <= 2**15` and `nbBits <= 15`, so the two fit in one word
// (bits 0..15 and 16..20) - that is what keeps the decode table at two planes
// instead of three.
#define TANS_NBITS_SHIFT 16
#define TANS_NBITS_MASK 0x0000FFFFu  // bits >= 16 are nbBits
#define TANS_NEW_STATE_MASK 0x0000FFFFu

TANS_API TANS_INLINE int64_t tans_next_state_offset() { return 0; }
TANS_API TANS_INLINE int64_t tans_decode_offset(int64_t ts) { return ts; }
TANS_API TANS_INLINE int64_t tans_symbol_offset(int64_t ts) { return 2 * ts; }
TANS_API TANS_INLINE int64_t tans_symbol_tt_offset(int64_t ts) { return 3 * ts; }

// Row width of the table tensor for a distribution whose symbolTT must hold
// `max_symbols` symbols (the widest cdf row of the batch).
TANS_API TANS_INLINE int64_t tans_table_row_words(int64_t ts, int64_t max_symbols) {
  return 3 * ts + 2 * max_symbols;
}

// ---------------------------------------------------------------------------
// stream header
// ---------------------------------------------------------------------------
// Per row, right after the length word (which counts bytes from row+1 and does
// include this header):
//
//   word 0: encoder state (a valid state is in [2**L, 2**(L+1)), so 0 means
//           "fresh stream" - the first push initializes it to 2**L; this is
//           what lets tans_init_stream keep rans_init_stream's signature, which
//           does not know table_log);
//   word 1: bitbuf, the not-yet-committed bits (< 2**stream_bits at rest);
//   word 2: nbits, how many bits of bitbuf are valid (< stream_bits at rest).
//
// A decoded stream keeps this header (pop does not rewind the length word -
// unlike rANS, the payload position cannot be recovered from a bit cursor);
// pushing onto a decoded stream is not a supported flow, same as for rANS.
#define TANS_HEADER_WORDS 3
// Absolute word indices inside a stream row - word 0 is the length - so the
// kernels can index them straight off the row pointer without an offset (which
// is also why they are NOT 0/1/2: the first one would be the length word and a
// write to it would destroy the length the header is counted by).
#define TANS_STATE_SLOT 1
#define TANS_BITBUF_SLOT 2
#define TANS_NBITS_SLOT 3
#define TANS_HEADER_BYTES (TANS_HEADER_WORDS * 4)

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
// BIT_highbit32 of FSE, with the one extension this header relies on:
// highbit(0) == 0, which makes the `norm == 1` case fall out of the general
// symbolTT formula (maxBitsOut = table_log - 0 = table_log, minStatePlus =
// 1 << table_log) exactly as FSE's explicit `case 1:` does.
TANS_API TANS_INLINE uint32_t tans_highbit32(uint32_t value) {
  uint32_t bit = 0;
  while (value >>= 1) ++bit;
  return bit;
}

// FSE_TABLESTEP: the stride of the symbol spread. Coprime with a power of two
// table size, so the spread visits every slot.
TANS_API TANS_INLINE uint32_t tans_table_step(uint32_t table_size) {
  return (table_size >> 1) + (table_size >> 3) + 3;
}

// ---------------------------------------------------------------------------
// bit layer
// ---------------------------------------------------------------------------
// Encoder: bits are appended at increasing positions of a 64-bit register and
// committed as whole stream words (low word first), so the payload is the bit
// sequence b0 b1 b2 ... laid out little-endian, word by word.
//
// The register is a local during a push; at rest only the residual (<
// stream_bits) is written back to the header. Appending up to 15 bits to a
// residual of at most stream_bits-1 needs at most 15 + 15 = 30 bits (the widest
// stream word is 16 bits), so 64 bits are always enough.
template <typename TANS_STREAM_TYPE>
struct TansBitWriter
{
  uint64_t buf;
  uint32_t nbits;
  TANS_STREAM_TYPE* ptr;  // last committed word; the next one is *(ptr + 1)

  static constexpr uint32_t STREAM_BITS = sizeof(TANS_STREAM_TYPE) * 8;

  TANS_INLINE void append(uint32_t value, uint32_t k) {
    // `value` is expected to be a k-bit digit; masking here keeps the bit
    // positions above k clean without a branch at every call site.
    buf |= (uint64_t)(value & ((k >= 32) ? 0xFFFFFFFFu : ((1u << k) - 1))) << nbits;
    nbits += k;
    while (nbits >= STREAM_BITS) {
      *++ptr = (TANS_STREAM_TYPE)buf;
      buf >>= STREAM_BITS;
      nbits -= STREAM_BITS;
    }
  }
};

// Decoder: the mirror. `buf` holds `nbits` bits with the *next* bits to read at
// the top, so a read takes the top k of them; when there are not enough, whole
// words are loaded from the end of the payload (the encoder's words are
// consumed back to front, which is the LIFO order ANS decoding needs).
//
// The window is ordered by *time*: a higher bit position is a later bit, in both
// directions (the encoder appends at `<< nbits`, i.e. above what it already
// holds). A refill therefore has to shift the current window *up* and put the
// newly loaded - older - word at the bottom: OR-ing the word in at `<< nbits`
// would place it above the pending bits, i.e. claim it is newer than them, which
// silently reorders every read that happens after a refill.
//
// `while` (not `if`): an 8-bit stream word needs two refills for a 15-bit read,
// which is what tans16/tans8 with a large table_log do. A refill only happens
// while nbits < k <= 15, so the shifted window never grows past 15 + 16 = 31
// bits, well inside the 64-bit register.
template <typename TANS_STREAM_TYPE>
struct TansBitReader
{
  uint64_t buf;
  uint32_t nbits;
  const TANS_STREAM_TYPE* ptr;  // last word still to consume

  static constexpr uint32_t STREAM_BITS = sizeof(TANS_STREAM_TYPE) * 8;

  TANS_INLINE uint32_t take(uint32_t k) {
    while (nbits < k) {
      buf = (buf << STREAM_BITS) | (uint64_t)(*ptr--);
      nbits += STREAM_BITS;
    }
    nbits -= k;
    return (uint32_t)((buf >> nbits) & ((1u << k) - 1));
  }
};

// ---------------------------------------------------------------------------
// bypass (out-of-range) coding
// ---------------------------------------------------------------------------
// The layout of the digits is the reference's (3rdparty/ans/tans.cpp): the
// decoder reads the digit count, then the raw value's digits least significant
// first, and pushes them into the *bit buffer* as plain bits rather than
// through a uniform bypass table as FSE does. For a uniform alphabet both cost
// exactly bypass_precision bits per digit, so the stream is the same size while
// the table (and the extra "row" carrying it) disappears - and nothing depends
// on freq_precision any more.
//
// The digit count is written as repeated max_bypass_val "carries" plus a
// remainder, and the whole sequence is emitted *before* the sentinel symbol:
// the decoder sees the sentinel first (last written, first read), then the
// carries, then the remainder, then the digits - each mirrored, which is why
// the loops below run in the opposite order to the decoder's.
TANS_API TANS_INLINE uint32_t tans_max_bypass_digits(uint32_t bypass_precision) {
  // ceil(32 / bypass_precision): how many digits a 32-bit raw value can need,
  // i.e. the largest n_bypass the encoder can produce. The decoder clamps to it
  // so that a corrupted stream cannot push a shift out of range.
  return (32u + bypass_precision - 1) / bypass_precision;
}

template <typename TANS_STREAM_TYPE>
TANS_INLINE void tans_push_raw_value(TansBitWriter<TANS_STREAM_TYPE>* bw,
                                     uint32_t raw_val, uint32_t bypass_precision) {
  const uint32_t max_bypass_val = (1u << bypass_precision) - 1;

  // Number of bypass_precision-sized digits of raw_val. The shift guard is
  // rans_utils.hpp::rans_push_raw_value_step's: for a 32-bit symbol with
  // bypass_precision=4 the count reaches 8, i.e. a shift of 32, which is
  // undefined behaviour (on x86 the count wraps to 0 and the loop never
  // terminates).
  uint32_t n_bypass = 0;
  while (n_bypass * bypass_precision < 32 &&
         (raw_val >> (n_bypass * bypass_precision)) != 0) {
    ++n_bypass;
  }

  uint32_t val = n_bypass;
  uint32_t n_carries = 0;
  while (val >= max_bypass_val) {
    ++n_carries;
    val -= max_bypass_val;
  }
  const uint32_t rest = val;

  for (int64_t j = (int64_t)n_bypass - 1; j >= 0; --j) {
    bw->append(raw_val >> (j * bypass_precision), bypass_precision);
  }
  bw->append(rest, bypass_precision);
  for (uint32_t j = 0; j < n_carries; ++j) {
    bw->append(max_bypass_val, bypass_precision);
  }
}

template <typename TANS_STREAM_TYPE>
TANS_INLINE uint32_t tans_pop_raw_value(TansBitReader<TANS_STREAM_TYPE>* br,
                                        uint32_t bypass_precision) {
  const uint32_t max_bypass_val = (1u << bypass_precision) - 1;
  uint32_t val = br->take(bypass_precision);
  uint32_t n_bypass = val;
  while (val == max_bypass_val) {
    val = br->take(bypass_precision);
    n_bypass += val;
  }
  const uint32_t max_digits = tans_max_bypass_digits(bypass_precision);
  if (n_bypass > max_digits) n_bypass = max_digits;  // corrupted stream guard
  uint32_t raw_val = 0;
  for (uint32_t j = 0; j < n_bypass; ++j) {
    raw_val |= br->take(bypass_precision) << (j * bypass_precision);
  }
  return raw_val;
}

// ---------------------------------------------------------------------------
// the two symbol steps
// ---------------------------------------------------------------------------
// Encoder: emit the state bits of `symbol` and advance the state. `row` is the
// table row, `ts` = 1 << table_log. The index arithmetic is signed on purpose -
// deltaFindState = total - norm[s] is negative for the first symbols - and the
// reference relies on it: the index lands inside the symbol's own slot range
// [total, total + norm[s]), the invariant the tables were built for.
template <typename TANS_STREAM_TYPE>
TANS_INLINE uint32_t tans_encode_symbol(uint32_t state, const int32_t* row,
                                        uint32_t ts, uint32_t symbol,
                                        TansBitWriter<TANS_STREAM_TYPE>* bw) {
  const int32_t* const symbol_tt = row + tans_symbol_tt_offset(ts) + 2 * (int64_t)symbol;
  const int32_t delta_nb_bits = symbol_tt[0];
  const int32_t delta_find_state = symbol_tt[1];

  // nbBits = maxBitsOut - [state < minStatePlus], computed in the reference's
  // way: the sum is non-negative by construction (see TANS_MAX_TABLELOG).
  const uint32_t nb_bits = (uint32_t)(((int64_t)state + (int64_t)delta_nb_bits) >> 16);
  bw->append(state, nb_bits);  // the low nb_bits of the state, masked in append()

  const int64_t next_index = (int64_t)(state >> nb_bits) + (int64_t)delta_find_state;
  return (uint32_t)row[tans_next_state_offset() + next_index];
}

// Decoder: the mirror; returns the new state and writes the symbol out.
template <typename TANS_STREAM_TYPE>
TANS_INLINE uint32_t tans_decode_symbol(uint32_t state, const int32_t* row,
                                        uint32_t ts, TansBitReader<TANS_STREAM_TYPE>* br,
                                        uint32_t* symbol_out) {
  const uint32_t word =
      (uint32_t)row[tans_decode_offset(ts) + (int64_t)state];
  const uint32_t nb_bits = (word >> TANS_NBITS_SHIFT) & 0x1Fu;
  *symbol_out = (uint32_t)row[tans_symbol_offset(ts) + (int64_t)state];
  const uint32_t low_bits = br->take(nb_bits);
  return (word & TANS_NEW_STATE_MASK) + low_bits;
}

// ---------------------------------------------------------------------------
// table construction (host only: it allocates temporaries per distribution)
// ---------------------------------------------------------------------------
// Builds one distribution's row of the tANS table tensor: the spread, the
// encoder's nextState and symbol transformation tables and the decoder's
// (symbol, newState, nbBits) table.
//
// `norm` is *given*: `cdf[s + 1] - cdf[s]` for s in [0, num_symbols), i.e. the
// quantized counter torch_ans already builds with its "steal frequency" fix-up.
// That is why FSE's FSE_normalizeCount/FSE_normalizeM2 are not ported: those
// renormalize an arbitrary histogram to 2**table_log, while the cdf is already
// normalized to 2**freq_precision and `table_log == freq_precision` here. It
// also removes FSE's `-1` (low probability) branch: every symbol owns at least
// one unit, so every count is >= 1 and the spread covers the table without a
// reserved low-probability area.
//
// `num_symbols` is max_value + 1 (the last symbol is the bypass sentinel when
// bypass coding is on) while `max_symbols` is the tensor-wide width of the
// symbolTT part - entries of symbols this distribution does not have are zeroed
// so the row is deterministic, and the kernels never read them (value <
// num_symbols always).
//
// Returns false and fills `error` when the row is not a usable normalized
// counter (a zero-frequency symbol, or a cdf that does not sum to 2**table_log
// - which is what a cdf built with a different freq_precision looks like).
inline bool tans_build_tables_row(const int32_t* cdf, int64_t num_symbols,
                                  int64_t max_symbols, int32_t* out_row,
                                  int64_t table_log, int64_t ts, std::string* error) {
  if (num_symbols < 1 || num_symbols > max_symbols) {
    *error = "tans: cdfs_sizes - 1 must be in [1, " + std::to_string(max_symbols) +
             "] (got " + std::to_string(num_symbols) + ")";
    return false;
  }

  // --- normalized counter ---------------------------------------------------
  std::vector<uint32_t> norm((size_t)num_symbols);
  int64_t total = 0;
  for (int64_t s = 0; s < num_symbols; ++s) {
    const int64_t freq = (int64_t)cdf[s + 1] - (int64_t)cdf[s];
    if (freq <= 0) {
      *error = "tans: symbol " + std::to_string(s) +
               " has no probability mass (freq=0); tANS needs every symbol to own at "
               "least one of the 2**table_log slots (torch_ans' quantized cdf "
               "guarantees that, a hand-made cdf may not)";
      return false;
    }
    norm[(size_t)s] = (uint32_t)freq;
    total += freq;
  }
  if (total != ts) {
    *error = "tans: the cdf row sums to " + std::to_string(total) + " but a table_log of " +
             std::to_string(table_log) + " needs 2**table_log = " + std::to_string(ts) +
             " (build the cdf with freq_precision == table_log)";
    return false;
  }

  int32_t* const next_state = out_row + tans_next_state_offset();
  int32_t* const decode_word = out_row + tans_decode_offset(ts);
  int32_t* const symbol_plane = out_row + tans_symbol_offset(ts);
  int32_t* const symbol_tt = out_row + tans_symbol_tt_offset(ts);

  // --- spread --------------------------------------------------------------
  // FSE's "distribute symbols over the table" (Tans_buildCTable /
  // FSE_buildCTable): walk the table with a stride coprime with its size and
  // give every symbol its `norm[s]` consecutive turns. `slot_start[s]` is the
  // first slot of symbol s in the order the spread produces them, i.e. the rank
  // the encoder indexes with (x >> nbBits) + deltaFindState.
  std::vector<uint32_t> table_symbol((size_t)ts);
  std::vector<uint32_t> slot_start((size_t)num_symbols);
  {
    uint32_t acc = 0;
    for (int64_t s = 0; s < num_symbols; ++s) {
      slot_start[(size_t)s] = acc;
      acc += norm[(size_t)s];
    }
  }
  {
    const uint32_t step = tans_table_step((uint32_t)ts);
    const uint32_t mask = (uint32_t)(ts - 1);
    uint32_t position = 0;
    for (int64_t s = 0; s < num_symbols; ++s) {
      for (uint32_t k = 0; k < norm[(size_t)s]; ++k) {
        table_symbol[position] = (uint32_t)s;
        position = (position + step) & mask;
      }
    }
    if (position != 0) {
      *error = "tans: the symbol spread did not close (the normalized counter is "
               "inconsistent with the table size)";
      return false;
    }
  }

  // --- encoder: nextState --------------------------------------------------
  // next_state[slot] = ts + (spread position), i.e. always in [ts, 2*ts).
  {
    std::vector<uint32_t> cursor = slot_start;
    for (int64_t u = 0; u < ts; ++u) {
      const uint32_t symbol = table_symbol[(size_t)u];
      next_state[cursor[symbol]++] = (int32_t)(ts + u);
    }
  }

  // --- decoder: symbol + (newState, nbBits) --------------------------------
  // `next` is the rank of the state within its symbol's slots; nbBits is the
  // number of bits needed to count up to it, so that `newState + lowBits`
  // stays in [0, ts) - the decoder's state range.
  {
    std::vector<uint32_t> symbol_next = norm;
    for (int64_t u = 0; u < ts; ++u) {
      const uint32_t symbol = table_symbol[(size_t)u];
      const uint32_t next = symbol_next[symbol]++;
      const uint32_t nb_bits = (uint32_t)table_log - tans_highbit32(next);
      const uint32_t new_state = (next << nb_bits) - (uint32_t)ts;
      symbol_plane[u] = (int32_t)symbol;
      decode_word[u] = (int32_t)(new_state | (nb_bits << TANS_NBITS_SHIFT));
    }
  }

  // --- encoder: symbol transformation table --------------------------------
  // deltaNbBits = (maxBitsOut << 16) - minStatePlus with
  // minStatePlus = norm << maxBitsOut in (2**L, 2**(L+1)], and
  // deltaFindState = total - norm[s] (negative for the earliest symbols, which
  // is the signed index arithmetic tans_encode_symbol relies on).
  {
    uint32_t running = 0;
    for (int64_t s = 0; s < max_symbols; ++s) {
      if (s < num_symbols) {
        const uint32_t n = norm[(size_t)s];
        const uint32_t max_bits_out = (uint32_t)table_log - tans_highbit32(n - 1);
        const uint32_t min_state_plus = n << max_bits_out;
        symbol_tt[2 * s] = (int32_t)((int64_t)(max_bits_out << 16) - (int64_t)min_state_plus);
        symbol_tt[2 * s + 1] = (int32_t)((int64_t)running - (int64_t)n);
        running += n;
      } else {
        // A symbol this distribution does not have: never coded, zeroed so that
        // a reused buffer cannot leak a stale table into a debugger.
        symbol_tt[2 * s] = 0;
        symbol_tt[2 * s + 1] = 0;
      }
    }
  }

  return true;
}

