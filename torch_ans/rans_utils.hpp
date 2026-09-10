#pragma once

#include <torch/extension.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifdef RANS_CUDA_API
  #define RANS_API __device__
  // using libcu++
  // #include <cuda/std/iostream>
  // #include <cuda/std/cmath>
  // namespace std = cuda::std
#else
  #define RANS_API static
#endif

// TODO: support multiple dtypes!
#define DEFAULT_TORCH_TENSOR_DTYPE torch::kInt32
#define DEFAULT_TORCH_TENSOR_TYPE int32_t

// #define DEBUG_STEPS

// L ('l' in the paper) is the lower bound of our normalization interval.
// Between this and our 32-bit-aligned emission, we use 63 (not 64!) bits.
// This is done intentionally because exact reciprocals for 63-bit uints
// fit in 64-bit uints: this permits some optimizations during encoding.
// #define RANS_STATE_BITS sizeof(RANS_STATE_TYPE)*8-1 // f

// NOTE: we could use fewer bits (less than RANS_STATE_BITS-freq_precision) to allow state overflow during rans_push
// This might be useful for interleaved coding 
#define RANS_STATE_BITS sizeof(RANS_STATE_TYPE)*8
#define RANS_STATE_USED_BITS RANS_STATE_BITS-1 // ((RANS_STATE_VALID_BITS>0) ? RANS_STATE_VALID_BITS : RANS_STATE_BITS)
#define RANS_STREAM_BITS sizeof(RANS_STREAM_TYPE)*8

// lower bound of our normalization interval
#define RANS_STATE_LOWER_BOUND (1ull << (RANS_STATE_USED_BITS - RANS_STREAM_BITS)) // (1ull << 31)  

// NOTE: we do this inversely for convenience in tensor manipulation
#define APPEND_STATE_TO_STREAM(state, pptr) \
    *pptr += 1; **pptr = (RANS_STREAM_TYPE) state; state >>= RANS_STREAM_BITS; // std::cout<<"appendstate"<<std::endl;
#define POP_STATE_FROM_STREAM(state, pptr) \
    state = (state << RANS_STREAM_BITS) | **pptr; *pptr -= 1; // std::cout<<"popstate"<<std::endl;

// Branch-free single-word renormalization (host builds).
//
// Whether a renormalization is needed is close to a coin flip for the usual
// parameter choices - freq_precision=15 bits consumed per symbol against a
// 32-bit stream word renormalizes about every second symbol - so an `if` here
// mispredicts on roughly half of the symbols. That is what kept interleaved
// coding from paying off: a mispredict discards the whole out-of-order window,
// which is exactly the window the (otherwise independent) interleaved lanes
// need in order to overlap. The variants below are branch-free instead:
//   * append: store the word into the *next* slot unconditionally and commit
//     the cursor only when the renormalization was really needed. When it was
//     not, a garbage word is left one past the cursor; the next append
//     overwrites it, and the stream length is derived from the cursor, so it
//     never becomes part of the bitstream. The buffers keep one word of slack
//     for the very last append (see rans_init_stream / rans_push).
//   * pop: load the word unconditionally (the cursor always points at a word
//     inside the row) and select.
// Both emit/consume exactly the words the branchy version does, so streams stay
// bit-compatible between implementations (CPU <-> CUDA included).
#ifdef RANS_CUDA_API
#define APPEND_STATE_TO_STREAM_IF(state, pptr, cond) \
    { if (cond) { APPEND_STATE_TO_STREAM(state, pptr) } }
#define POP_STATE_FROM_STREAM_IF(state, pptr, cond) \
    { if (cond) { POP_STATE_FROM_STREAM(state, pptr) } }
#else
#define APPEND_STATE_TO_STREAM_IF(state, pptr, cond) \
    { const bool _rans_need = (cond); \
      (*pptr)[1] = (RANS_STREAM_TYPE) state; \
      *pptr += _rans_need; \
      state = _rans_need ? (RANS_STATE_TYPE) (state >> RANS_STREAM_BITS) : state; }
#define POP_STATE_FROM_STREAM_IF(state, pptr, cond) \
    { const bool _rans_need = (cond); \
      const RANS_STATE_TYPE _rans_word = (RANS_STATE_TYPE) **pptr; \
      state = _rans_need ? (RANS_STATE_TYPE) ((state << RANS_STREAM_BITS) | _rans_word) : state; \
      *pptr -= _rans_need; }
#endif

#define RANS_APPEND_STATE_RENORM(state, freq, freq_precision, pptr) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {APPEND_STATE_TO_STREAM_IF(state, pptr, state >= ((RANS_STATE_LOWER_BOUND >> freq_precision) << RANS_STREAM_BITS) * freq)} \
    else \
      {while (state >= ((RANS_STATE_LOWER_BOUND >> freq_precision) << RANS_STREAM_BITS) * freq) {APPEND_STATE_TO_STREAM(state, pptr);}}

#define RANS_APPEND_STATE_RENORM_OVERFLOW(state, pptr) \
    if (RANS_STATE_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {APPEND_STATE_TO_STREAM_IF(state, pptr, state >= (RANS_STATE_LOWER_BOUND << RANS_STREAM_BITS))} \
    else \
      {while (state >= (RANS_STATE_LOWER_BOUND << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}}

#define RANS_POP_STATE_RENORM(state, pptr) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {POP_STATE_FROM_STREAM_IF(state, pptr, state < RANS_STATE_LOWER_BOUND)} \
    else \
      {while (state < RANS_STATE_LOWER_BOUND) {POP_STATE_FROM_STREAM(state, pptr);}}

#define RANS_APPEND_BITS(state, pptr, value, nbits) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {APPEND_STATE_TO_STREAM_IF(state, pptr, state >= ((RANS_STATE_LOWER_BOUND >> nbits) << RANS_STREAM_BITS))} \
    else \
      {while (state >= ((RANS_STATE_LOWER_BOUND >> nbits) << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}} \
    state = (state << (nbits)) | (value);

#define RANS_POP_BITS(state, pptr, value, nbits) \
    value = (RANS_SYMBOL_TYPE) (state & ((1u << (nbits)) - 1)); state = state >> (nbits); RANS_POP_STATE_RENORM(state, pptr)


// Compile-time unrolled loop (used to give the interleaved lanes a compile-time
// lane index, which is what allows the state arrays to live in registers).
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


// How many cdf entries the inverse-CDF gap closing loads speculatively (see
// rans_invcdf_advance). See the discussion there for how the value is chosen.
#ifndef RANS_INVCDF_WINDOW
#define RANS_INVCDF_WINDOW 2
#endif


// Turn the inverse-CDF table's lower bound into the exact symbol index.
//
// `inversed_cdf[cum_freq >> shift]` gives the largest i with
// cdf[i] <= (the bucket start), which is a lower bound of the answer; the exact
// index is the largest i with cdf[i] <= cum_freq. The gap is at most 2**shift
// symbols, but in practice it is a couple of them (measured with the auto
// precision: mean 0.5-2 steps, max 4-10 depending on the alphabet size), and the
// obvious `while (cdf[idx + 1] <= cum_freq) ++idx;` is a data-dependent branch -
// the same kind of coin flip that keeps the interleaved lanes from overlapping.
//
// So instead of walking, load a fixed window of RANS_INVCDF_WINDOW cdf entries
// *in parallel* (they all come from the lower bound, so they do not depend on
// each other, unlike the walk's serial loads) and count how many of them are
// <= cum_freq: since the cdf is monotone that count is exactly the number of
// steps the walk would have taken. A loop for the rare remainder keeps this
// exact for any distribution (limit case: a bucket holding many symbols).
template <typename RANS_FREQ_TYPE>
RANS_API inline uint32_t rans_invcdf_advance(const RANS_FREQ_TYPE* cdf, uint32_t last,
                                             uint32_t lb, RANS_FREQ_TYPE cum_freq)
{
#ifdef RANS_CUDA_API
    uint32_t idx = lb;
    while (cdf[idx + 1] <= cum_freq) ++idx;
    return idx;
#else
    uint32_t advance = 0;
    constexpr_for<size_t(1), size_t(RANS_INVCDF_WINDOW) + 1, size_t(1)>([&](auto kc) {
        constexpr size_t k = kc;
        // cdf[last] == 1 << freq_precision > cum_freq, so clamping the read
        // index (instead of testing the bound) keeps every load in range and
        // contributes nothing for entries past the end.
        const uint32_t j = lb + k;
        const uint32_t j_clamped = (j < last) ? j : last;
        advance += (uint32_t)(cdf[j_clamped] <= cum_freq);
    });
    uint32_t idx = lb + advance;
    // Rare: only when a bucket spans more than RANS_INVCDF_WINDOW symbols.
    while (cdf[idx + 1] <= cum_freq) ++idx;
    return idx;
#endif
}


// x / freq and x % freq in one go.
//
// The hardware integer divider is the most expensive instruction of the push
// step by a wide margin and - unlike everything else in the loop - it is *not*
// pipelined (a 64-bit div has a reciprocal throughput of ~21-74 cycles on
// Broadwell, ~30 on Zen), so interleaving N states merely queues N divisions on
// the same unit and cannot overlap anything. The floating-point divider is
// pipelined (~4-8 cycles throughput), and after the pre-renormalization the
// quotient is bounded by 2**(RANS_STATE_USED_BITS - freq_precision), which
// leaves the double result off by at most one for the usual precisions. The
// fix-up loops below (normally zero iterations, and perfectly predicted because
// they almost never trigger) make the result exact for *any* input, so this is
// a drop-in replacement rather than an approximation.
template <typename RANS_STATE_TYPE, typename RANS_FREQ_TYPE>
RANS_API inline RANS_STATE_TYPE rans_divmod(RANS_STATE_TYPE x, RANS_FREQ_TYPE freq,
    RANS_STATE_TYPE* remainder)
{
#ifdef RANS_CUDA_API
    *remainder = x % (RANS_STATE_TYPE) freq;
    return x / (RANS_STATE_TYPE) freq;
#else
    // (int64_t) casts keep both conversions single-instruction: the state never
    // uses its top bit (RANS_STATE_USED_BITS == RANS_STATE_BITS - 1), while
    // unsigned <-> double conversions would compile to a branchy sequence.
    RANS_STATE_TYPE q = (RANS_STATE_TYPE) (int64_t) ((double) (int64_t) x / (double) (int64_t) freq);
    int64_t r = (int64_t) (x - q * (RANS_STATE_TYPE) freq);
    while (r < 0) { --q; r += (int64_t) freq; }
    while (r >= (int64_t) freq) { ++q; r -= (int64_t) freq; }
    *remainder = (RANS_STATE_TYPE) r;
    return q;
#endif
}



// ============================================================================
// Alias sampling table element
// ============================================================================

template <typename RANS_FREQ_TYPE>
struct RANSAliasSamplingCDFTableElement
{
  RANS_FREQ_TYPE cut_cdf;
  // RANS_FREQ_TYPE self_symbol;
  RANS_FREQ_TYPE other_symbol;
  RANS_FREQ_TYPE self_alias_offset;
  RANS_FREQ_TYPE other_alias_offset;
};

template <typename RANS_SYMBOL_TYPE, typename RANS_FREQ_TYPE>
RANS_API inline RANS_SYMBOL_TYPE binary_search_cdf(RANS_FREQ_TYPE cum_freq, 
    const RANS_FREQ_TYPE* cdf, RANS_FREQ_TYPE cdf_size) {
    RANS_FREQ_TYPE cum_freq_offset = cum_freq;
    RANS_FREQ_TYPE low = 0;
    RANS_FREQ_TYPE high = cdf_size - 1;
    RANS_SYMBOL_TYPE cdf_idx = (low + high) / 2;
    while (high>low)
    {
      if (cum_freq>=cdf[cdf_idx])
        low = cdf_idx + 1;
      else // if (cum_freq<cdf[cdf_idx])
        high = cdf_idx;
      // else break;
      cdf_idx = (low + high) / 2;
      // std::cout << cum_freq << " " << cdf[cdf_idx] << " " << cdf_idx << " "  << low << " "  << high << std::endl;
    }
    cdf_idx--;
    return cdf_idx;
}

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline void rans_push_raw_value_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    RANS_SYMBOL_TYPE raw_val, 
    int64_t bypass_precision
    )
{
    // RANS_SYMBOL_TYPE raw_val = 0;
    // if (value < 0) {
    //   raw_val = -2 * value - 1;
    //   value = max_value;
    // } else if (value >= max_value) {
    //   raw_val = 2 * (value - max_value);
    //   value = max_value;
    // }

    const RANS_SYMBOL_TYPE max_bypass_val = (1u << bypass_precision) - 1;

    // NOTE: avoid using vector for cuda compability!
    // std::vector<RANS_SYMBOL_TYPE> bypass_syms;
    /* Determine the number of bypasses (in bypass_precision size) needed to
    * encode the raw value. */
    // NOTE: the shift amount must stay below the symbol width. For a 32-bit
    // symbol with bypass_precision=4 the count reaches 8, i.e. a shift of 32,
    // which is undefined behaviour -- on x86 the count wraps to 0 and the loop
    // never terminates. n_bypass is therefore capped at
    // ceil(symbol_bits / bypass_precision), which is the largest value the
    // bypass digits can represent anyway.
    RANS_SYMBOL_TYPE n_bypass = 0;
    while ((int64_t)(n_bypass * bypass_precision) < (int64_t)(sizeof(RANS_SYMBOL_TYPE) * 8) &&
           ((raw_val >> (n_bypass * bypass_precision)) != 0)) {
      ++n_bypass;
    }

    /* Encode number of bypasses */
    RANS_SYMBOL_TYPE val = n_bypass;
    RANS_SYMBOL_TYPE n_codes_n_bypass = 0;
    RANS_SYMBOL_TYPE rest_n_bypass = 0;
    while (val >= max_bypass_val) {
      // bypass_syms.push_back(max_bypass_val);
      n_codes_n_bypass += 1;
      val -= max_bypass_val;
    }
    rest_n_bypass = val;
    // bypass_syms.push_back(val);

    RANS_STATE_TYPE x = *state_ptr;
    for (int64_t j = n_bypass-1; j >= 0; j--) {
      const RANS_SYMBOL_TYPE bypass_val =
          (raw_val >> (j * bypass_precision)) & max_bypass_val;
      // RANS_FREQ_TYPE freq = 1 << (freq_precision - bypass_precision);
      // RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);
      /* x = C(s, x) */
      // *state_ptr = (x << bypass_precision) | val;
      RANS_APPEND_BITS(x, stream_pptr, bypass_val, bypass_precision);
      // bypass_syms.push_back(val);
    }
    RANS_APPEND_BITS(x, stream_pptr, rest_n_bypass, bypass_precision);
    for (int64_t j = n_codes_n_bypass-1; j >= 0; j--) {
      RANS_APPEND_BITS(x, stream_pptr, max_bypass_val, bypass_precision);
    }

#ifdef DEBUG_STEPS
    std::cout << "PUSH BYPASS: state_ptr:" << state_ptr << ", stream_ptr:" << (void*)(*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << x << std::endl;
#endif
    *state_ptr = x;
    
}


// The arithmetic core of a push, with the (start, freq) pair already looked up.
//
// Split out of rans_push_step so that the interleaved coders can gather the cdf
// entries of *all* lanes first: those loads only depend on the input tensors,
// not on the rANS states or on the stream cursor, so hoisting them makes the
// (cache-missing) cdf accesses of the lanes overlap instead of being serialized
// behind each other's state updates.
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_FREQ_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline void rans_push_step_freq(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr,
    const RANS_FREQ_TYPE start,
    const RANS_FREQ_TYPE freq,
    int64_t freq_precision,
    const RANS_FREQ_TYPE* cdf_alias_remap
    )
{
    RANS_STATE_TYPE x = *state_ptr;
    RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);

    RANS_STATE_TYPE rem;
    const RANS_STATE_TYPE quo = rans_divmod<RANS_STATE_TYPE, RANS_FREQ_TYPE>(x, freq, &rem);
    if (cdf_alias_remap != nullptr) {
      x = (quo << freq_precision) + cdf_alias_remap[rem + start];
    }
    else {
      // x = C(s,x), written as x + quo * (2**freq_precision - freq) + start:
      // algebraically the same as (quo << freq_precision) + rem + start, but it
      // keeps the remainder off the dependency chain (it is only needed for the
      // exactness check inside rans_divmod, which is a not-taken branch), which
      // is worth a few cycles per symbol on the encode critical path.
      x += quo * (((RANS_STATE_TYPE)1 << freq_precision) - (RANS_STATE_TYPE)freq) + start;
    }
#ifdef DEBUG_STEPS
    std::cout << "state_ptr:" << state_ptr << ", stream_ptr:" << (void*) (*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << x << ", start: " << start << ", freq: " << freq << std::endl;
#endif
    *state_ptr = x;
}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, typename RANS_FREQ_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline void rans_push_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    RANS_SYMBOL_TYPE symbol, 
    const RANS_FREQ_TYPE* cdf, 
    const RANS_SYMBOL_TYPE cdf_size, 
    const RANS_SYMBOL_TYPE offset, 
    int64_t freq_precision,
    bool bypass_coding, 
    int64_t bypass_precision,
    const RANS_FREQ_TYPE* cdf_alias_remap
    )
{
    static_assert(RANS_STATE_USED_BITS <= RANS_STATE_BITS);
    static_assert(RANS_STATE_USED_BITS > RANS_STREAM_BITS);
    assert(freq_precision <= RANS_STATE_USED_BITS-RANS_STREAM_BITS);

    RANS_SYMBOL_TYPE value = symbol - offset;

    RANS_SYMBOL_TYPE max_value = cdf_size - 2;
    if (bypass_coding) {
        assert(bypass_precision < freq_precision);
    }
    assert(max_value >= 0);

    // Bypass coding range limit. An out-of-range symbol is zig-zag encoded into
    // a raw value held in RANS_SYMBOL_TYPE (int32 for the default tensor
    // dtype):
    //     raw_val = 2 * (value - max_value)      for value >= max_value
    //     raw_val = -2 * value - 1               for value < 0
    // so the distance from the coded range [0, max_value) must stay below
    // 2^(8*sizeof(RANS_SYMBOL_TYPE)-2) -- about 2^30 (~1.07e9) for int32
    // symbols. Beyond that the multiply overflows and the symbol is silently
    // mis-coded (no error is raised). The bypass digits themselves carry a full
    // width value; only this zig-zag mapping is the limit. Note this is far
    // above the point where the digit counter used to overflow (2^28), which is
    // fixed by the shift guard below.
    RANS_SYMBOL_TYPE raw_val = 0;
    if (bypass_coding) {
      if (value < 0) {
        raw_val = -2 * value - 1;
        value = max_value;
      } else if (value >= max_value) {
        raw_val = 2 * (value - max_value);
        value = max_value;
      }
    }
    // else {
    //   // Avoid memerr but may reduce speed
    //   if (value < 0 || value >= max_value) value = max_value;
    // }

    assert(value >= 0);
    assert(value < cdf_size - 1);

    if (bypass_coding) {

      /* Bypass coding mode (value == max_value -> sentinel flag) */
      if (value == max_value) {
        rans_push_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_STATE_VALID_BITS>(
          state_ptr, stream_pptr, raw_val, bypass_precision
        );

        // const RANS_SYMBOL_TYPE max_bypass_val = (1u << bypass_precision) - 1;

        // NOTE: avoid using vector for cuda compability!
        // std::vector<RANS_SYMBOL_TYPE> bypass_syms;
        /* Determine the number of bypasses (in bypass_precision size) needed to
        * encode the raw value. */
        // RANS_SYMBOL_TYPE n_bypass = 0;
        // while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        //   ++n_bypass;
        // }

        // /* Encode number of bypasses */
        // RANS_SYMBOL_TYPE val = n_bypass;
        // RANS_SYMBOL_TYPE n_codes_n_bypass = 0;
        // RANS_SYMBOL_TYPE rest_n_bypass = 0;
        // while (val >= max_bypass_val) {
        //   // bypass_syms.push_back(max_bypass_val);
        //   n_codes_n_bypass += 1;
        //   val -= max_bypass_val;
        // }
        // rest_n_bypass = val;
        // // bypass_syms.push_back(val);

        // RANS_STATE_TYPE x = *state_ptr;
        // for (RANS_SYMBOL_TYPE j = n_bypass-1; j >= 0; j--) {
        //   const RANS_SYMBOL_TYPE val =
        //       (raw_val >> (j * bypass_precision)) & max_bypass_val;
        //   // RANS_FREQ_TYPE freq = 1 << (freq_precision - bypass_precision);
        //   // RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);
        //   /* x = C(s, x) */
        //   // *state_ptr = (x << bypass_precision) | val;
        //   RANS_APPEND_BITS(x, stream_pptr, val, bypass_precision);
        //   // bypass_syms.push_back(val);
        // }
        // RANS_APPEND_BITS(x, stream_pptr, rest_n_bypass, bypass_precision);
        // for (RANS_SYMBOL_TYPE j = n_codes_n_bypass-1; j >= 0; j--) {
        //   RANS_APPEND_BITS(x, stream_pptr, max_bypass_val, bypass_precision);
        // }
        // *state_ptr = x;

        // /* Encode raw value */
        // for (RANS_SYMBOL_TYPE j = 0; j < n_bypass; ++j) {
        //   const RANS_SYMBOL_TYPE val =
        //       (raw_val >> (j * bypass_precision)) & max_bypass_val;
        //   bypass_syms.push_back(val);
        // }

        // // bypass_syms should be encoded in reverse order!
        // while (!bypass_syms.empty()) {
        //   const RANS_SYMBOL_TYPE val = bypass_syms.back();
        // //   Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
        //   RANS_STATE_TYPE x = *state_ptr;
        //   // RANS_FREQ_TYPE freq = 1 << (freq_precision - bypass_precision);
        //   // RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);
        //   /* x = C(s, x) */
        //   // *state_ptr = (x << bypass_precision) | val;
        //   RANS_APPEND_BITS(x, stream_pptr, val, bypass_precision);
        //   *state_ptr = x;

        //   bypass_syms.pop_back();
        // }

      }
    }

    // directly put bits
    if (cdf == nullptr) {
      const RANS_FREQ_TYPE freq = (1 << freq_precision) / max_value;
      const RANS_FREQ_TYPE start = freq * value;
      rans_push_step_freq<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_FREQ_TYPE, RANS_STATE_VALID_BITS>(
        state_ptr, stream_pptr, start, freq, freq_precision, nullptr);
    }
    // cdf-based coding
    else {
      // Rans64EncPut(state_ptr, stream_pptr, cdf[value], cdf[value + 1] - cdf[value], freq_precision);
      const RANS_FREQ_TYPE start = cdf[value];
      const RANS_FREQ_TYPE freq = cdf[value + 1] - start;
      rans_push_step_freq<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_FREQ_TYPE, RANS_STATE_VALID_BITS>(
        state_ptr, stream_pptr, start, freq, freq_precision, cdf_alias_remap);
    }

}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline RANS_SYMBOL_TYPE rans_pop_raw_value_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    int64_t bypass_precision
    )
{
    const RANS_SYMBOL_TYPE max_bypass_val = (1u << bypass_precision) - 1;

    RANS_STATE_TYPE x = *state_ptr;
    /* Bypass decoding mode */
    RANS_SYMBOL_TYPE val;
    // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
    // x = x >> bypass_precision;
    // RANS_POP_STATE_RENORM(x, stream_pptr);
    // *state_ptr = x;
    RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
    auto n_bypass = val;

    while (val == max_bypass_val) {
      // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
      // x = x >> bypass_precision;
      // RANS_POP_STATE_RENORM(x, stream_pptr);
      // *state_ptr = x;
      RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
      n_bypass += val;
    }

    RANS_SYMBOL_TYPE raw_val = 0;
    for (int j = 0; j < n_bypass; ++j) {
      // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
      // x = x >> bypass_precision;
      // RANS_POP_STATE_RENORM(x, stream_pptr);
      // *state_ptr = x;
      RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
      assert(val <= max_bypass_val);
      raw_val |= val << (j * bypass_precision);
    }
    // value = raw_val >> 1;
    // if (raw_val & 1) {
    //   value = -value - 1;
    // } else {
    //   value += max_value;
    // }
#ifdef DEBUG_STEPS
    std::cout << "POP BYPASS: state_ptr:" << state_ptr << ", stream_ptr:" << (void*)(*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << x << std::endl;
#endif
    *state_ptr = x;

    return raw_val;
    
}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, typename RANS_FREQ_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline RANS_SYMBOL_TYPE rans_pop_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    const RANS_FREQ_TYPE* cdf, 
    const RANS_SYMBOL_TYPE cdf_size, 
    const RANS_SYMBOL_TYPE offset, 
    int64_t freq_precision,
    bool bypass_coding, 
    int64_t bypass_precision,
    const RANS_FREQ_TYPE* inversed_cdf,
    const RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>* cdf_alias_table,
    int64_t inverse_cdf_precision=-1
    )
{
    static_assert(RANS_STATE_USED_BITS <= RANS_STATE_BITS);
    static_assert(RANS_STATE_USED_BITS > RANS_STREAM_BITS);
    assert(freq_precision <= RANS_STATE_USED_BITS-RANS_STREAM_BITS);

    RANS_SYMBOL_TYPE max_value = cdf_size - 2;
    if (bypass_coding) {
        assert(bypass_precision < freq_precision);
    }
    assert(max_value >= 0);

    RANS_STATE_TYPE x = *state_ptr;
    const RANS_STATE_TYPE mask = (1ull << freq_precision) - 1;
    const RANS_FREQ_TYPE cum_freq = (x & mask);

    // std::vector<RANS_FREQ_TYPE> cdf_vec(cdf, cdf + cdf_size);
    // const auto it = std::find_if(cdf_vec.begin(), cdf_vec.begin() + cdf_size,
    //                              [cum_freq](int v) { return v > cum_freq; });
    // const RANS_FREQ_TYPE cdf_idx = std::distance(cdf_vec.begin(), it) - 1;

    // find symbol by searching inversed_cdf table
    RANS_STATE_TYPE cdf_idx; // NOTE: use RANS_STATE_TYPE to ensure no overflow!
    RANS_FREQ_TYPE cum_freq_offset = cum_freq;
    if (cdf_alias_table != nullptr){
      const RANS_FREQ_TYPE cut_size = (1<<freq_precision) / (cdf_size-1); // ((1<<freq_precision)+cdf_size-2) / (cdf_size-1);
      const RANS_FREQ_TYPE alias_map_id = cum_freq / cut_size; // ((cdf_size-1) * cum_freq) >> freq_precision;
      const RANS_FREQ_TYPE cut_cdf = cdf_alias_table[alias_map_id].cut_cdf;
      cdf_idx = (cum_freq >= cut_cdf) ? cdf_alias_table[alias_map_id].other_symbol : alias_map_id;
      const RANS_FREQ_TYPE alias_start = (cum_freq >= cut_cdf) ? cdf_alias_table[alias_map_id].other_alias_offset : cdf_alias_table[alias_map_id].self_alias_offset;
      cum_freq_offset -= alias_start;
    }
    else if (inversed_cdf != nullptr) {
      // Sparse inverse CDF: entry i answers the query for the bucket start
      // i << (freq_precision - inverse_cdf_precision). A bucket start never
      // exceeds cum_freq, so the entry is a lower bound of the true symbol; the
      // remaining gap is closed branch-free in rans_invcdf_advance. The result
      // is the same as the linear walk (and as rans_warp_pop_lookup in
      // rans_warp_cuda.cuh), so the CPU and the warp kernels stay
      // bit-compatible.
      const int64_t inv_cdf_precision = (inverse_cdf_precision > 0) ? inverse_cdf_precision : freq_precision;
      const int shift = (int)(freq_precision - inv_cdf_precision);
      cdf_idx = rans_invcdf_advance<RANS_FREQ_TYPE>(
          cdf, (uint32_t)cdf_size - 1, (uint32_t)inversed_cdf[cum_freq >> shift], cum_freq);
      cum_freq_offset -= cdf[cdf_idx];
    }
    else {
#ifndef RANS_CUDA_API
      // Branch-free search for the last index with cdf[idx] <= cum_freq.
      //
      // The divided/binary search below spends ~log2(cdf_size) *data dependent*
      // branches per symbol, about half of them mispredicted (~10 cycles each,
      // i.e. the bulk of the decode time for a 256 symbol alphabet), and every
      // mispredict also throws away the work the other interleaved lanes had
      // already started - which is why interleaving could not pay off. Here the
      // trip count only depends on cdf_size (so the loop branch is predictable)
      // and the update is a cmov, leaving a pure chain of dependent loads that
      // different lanes overlap freely.
      //
      // Invariant: the answer lies in [base, base+len-1] and cdf[base] <=
      // cum_freq holds (cdf[0] == 0). cdf[cdf_size-1] == 1<<freq_precision is
      // always > cum_freq, hence len starts at cdf_size-1 and the result is
      // identical to the binary search it replaces.
      uint32_t base = 0;
      uint32_t len = (uint32_t)cdf_size - 1;
      while (len > 1) {
        const uint32_t half = len >> 1;
        base = (cdf[base + half] <= cum_freq) ? (base + half) : base;
        len -= half;
      }
      cdf_idx = base;
      cum_freq_offset -= cdf[cdf_idx];
#else
      // divided search (seems to be fastest)
      cdf_idx = (cdf_size * cum_freq) >> freq_precision;
      // if (cdf_idx < 0 || cdf_idx >= cdf_size) cdf_idx = cdf_size / 2;
      // bool lb, ub;
      // do  {
      //   lb = (cdf[cdf_idx+1] > cum_freq);
      //   ub = (cdf[cdf_idx] <= cum_freq);
      //   if (lb) cdf_idx--;
      //   if (ub) cdf_idx++;
      // } while (!(lb&&ub));
      uint32_t low = 0;
      uint32_t high = cdf_size - 1;
      while (high>low)
      {
        if (cum_freq>=cdf[cdf_idx])
          low = cdf_idx + 1;
        else // if (cum_freq<cdf[cdf_idx])
          high = cdf_idx;
        // else break;
        cdf_idx = (low + high) / 2;
        // std::cout << cum_freq << " " << cdf[cdf_idx] << " " << cdf_idx << " "  << low << " "  << high << std::endl;
      }
      cdf_idx--;
      cum_freq_offset -= cdf[cdf_idx];

      // RANS_SYMBOL_TYPE cdf_idx = 1;
      // for (cdf_idx = 1; cdf_idx < cdf_size; cdf_idx++) {
      //   if (cdf[cdf_idx] > cum_freq) break;
      // }
      // cdf_idx--;
// #else
//     // NOTE: this seems much faster than for loop!
//     const auto it = std::find_if(cdf, cdf + cdf_size,
//                                  [cum_freq](int v) { return v > cum_freq; });
//     const RANS_SYMBOL_TYPE cdf_idx = std::distance(cdf, it) - 1;
// #endif
#endif

    }
    
    RANS_SYMBOL_TYPE value = static_cast<RANS_SYMBOL_TYPE>(cdf_idx);
    // const RANS_FREQ_TYPE start = cdf[value];
    const RANS_FREQ_TYPE freq = cdf[cdf_idx+1] - cdf[cdf_idx];
    // Rans64DecAdvance(state_ptr, stream_pptr, start, freq, freq_precision);


    // s, x = D(x)
    // std::cout << "(x >> freq_precision) * static_cast<RANS_STATE_TYPE>(freq):" << (x >> freq_precision) * static_cast<RANS_STATE_TYPE>(freq) << ", (x & mask):" << (x & mask) << ", start:" << static_cast<RANS_STATE_TYPE>(start) << std::endl;
    x = (x >> freq_precision) * freq + cum_freq_offset; // cum_freq - start;
    // std::cout << "newstate:" << x << std::endl;
    RANS_POP_STATE_RENORM(x, stream_pptr);
    // std::cout << "newstate_renorm:" << x << std::endl;
#ifdef DEBUG_STEPS
    std::cout << "state_ptr:" << state_ptr << ", stream_ptr:" << (void*)(*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << x << ", cum_freq:" << cum_freq << ", freq: " << freq << ", cum_freq_offset: " << cum_freq_offset << ", value: " << value << std::endl;
#endif
    *state_ptr = x;

    if (bypass_coding) {

      if (value == max_value) {
        const RANS_SYMBOL_TYPE raw_val = rans_pop_raw_value_step<RANS_STATE_TYPE, RANS_STREAM_TYPE, RANS_SYMBOL_TYPE, RANS_STATE_VALID_BITS>(
          state_ptr, stream_pptr, bypass_precision
        );

        // const RANS_SYMBOL_TYPE max_bypass_val = (1u << bypass_precision) - 1;
        // /* Bypass decoding mode */
        // RANS_SYMBOL_TYPE val;
        // // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
        // // x = x >> bypass_precision;
        // // RANS_POP_STATE_RENORM(x, stream_pptr);
        // // *state_ptr = x;
        // RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
        // auto n_bypass = val;

        // while (val == max_bypass_val) {
        //   // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
        //   // x = x >> bypass_precision;
        //   // RANS_POP_STATE_RENORM(x, stream_pptr);
        //   // *state_ptr = x;
        //   RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
        //   n_bypass += val;
        // }

        // RANS_SYMBOL_TYPE raw_val = 0;
        // for (int j = 0; j < n_bypass; ++j) {
        //   // val = (RANS_SYMBOL_TYPE) (x & ((1u << bypass_precision) - 1));
        //   // x = x >> bypass_precision;
        //   // RANS_POP_STATE_RENORM(x, stream_pptr);
        //   // *state_ptr = x;
        //   RANS_POP_BITS(x, stream_pptr, val, bypass_precision);
        //   assert(val <= max_bypass_val);
        //   raw_val |= val << (j * bypass_precision);
        // }

        value = raw_val >> 1;
        if (raw_val & 1) {
          value = -value - 1;
        } else {
          value += max_value;
        }

        // *state_ptr = x;
      }

    }

    return value + offset;
}



// See https://github.com/skal65535/fsc/blob/master/alias.c
template <typename RANS_SYMBOL_TYPE, typename RANS_FREQ_TYPE>
inline bool build_alias_mapping(
  const RANS_FREQ_TYPE* cdf, 
  const RANS_SYMBOL_TYPE cdf_size, 
  RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>* cdf_alias_table,
  RANS_FREQ_TYPE* cdf_alias_remap,
  int64_t symbol_precision,
  int64_t freq_precision
)
{
  auto max_symbols = cdf_size - 1;
  // auto max_symbols_log2 = (1 << symbol_precision);
  auto max_table_size = (1 << freq_precision);
  // TODO: non-equal cuts to avoid 2^n num_cuts limit!
  // const RANS_SYMBOL_TYPE num_cuts = 1 << symbol_precision;
  // const RANS_SYMBOL_TYPE cut = 1 << (freq_precision-symbol_precision); // 1/n
  const RANS_SYMBOL_TYPE cut = max_table_size / max_symbols; // (max_table_size + max_symbols - 1) / max_symbols;   // 1/n
  const RANS_SYMBOL_TYPE num_cuts = (1 << symbol_precision); // (max_table_size + cut - 1) / cut;
  if ((num_cuts * cut) != max_table_size) return false;
  
  std::vector<RANS_SYMBOL_TYPE> symbols(num_cuts);
  int l = num_cuts, s = 0;
  std::vector<RANS_FREQ_TYPE> proba(num_cuts);
  RANS_SYMBOL_TYPE total = 0;
  assert(num_cuts >= max_symbols);
  assert((num_cuts * cut) >= max_table_size);
  // assert((max_table_size % max_symbols_log2) == 0);
  // if (max_symbols > max_symbols_log2 || max_symbols <= 0) return false;
  if (num_cuts < max_symbols || (num_cuts * cut) < max_table_size || max_symbols <= 0) return false;


  int i;
  for (i = 0; i < num_cuts; ++i) {
    proba[i] = (i < max_symbols) ? (cdf[i+1]-cdf[i]) : 0;
    total += proba[i];
    if (proba[i] >= cut) {
      symbols[--l] = i;
    } else {
      symbols[s++] = i;
    }
    assert(s <= l);
  }
  assert(s == l);
  // std::cout << "num_cuts:" << num_cuts << ", cut:" << cut << ", max_symbols:" << max_symbols << ", max_table_size:" << max_table_size << std::endl;
  if (total != max_table_size) return 0;   // unnormalized

  while (s > 0) {
    const int S = symbols[--s];
    const int L = symbols[l++];
    assert(proba[S] < cut);       // check that S is a small one
    const int cut_cdf = proba[S] + S * cut;
    if (cut_cdf >= max_table_size) std::cout << "CDF limit detected on small! Symbol:" << S << ", Other:" << L << std::endl;
    cdf_alias_table[S].cut_cdf = (cut_cdf >= max_table_size) ? max_table_size : cut_cdf ;
    cdf_alias_table[S].other_symbol = L;
    proba[L] -= cut - proba[S];   // decrease large proba
    if (proba[L] >= cut) {
      --l;                // large symbol stays large. Reuse the slot.
    } else {
      symbols[s++] = L;   // large becomes small
    }
    // The rest bucket from (large becomes small) cause overflow! Leave it be!
    if (l==num_cuts) {
      std::cout << "Large symbol overflow! Stopping... small ptr at " << s << std::endl;
      // break;
      return false;
    }
  }
  while (l < num_cuts) {
    const int L = symbols[l++];
    cdf_alias_table[L].other_symbol = L;
    const int cut_cdf = cut + L * cut;
    if (cut_cdf >= max_table_size) std::cout << "CDF limit detected on large! Symbol:" << L << ", Other:" << L << std::endl;
    cdf_alias_table[L].cut_cdf = (cut_cdf >= max_table_size) ? max_table_size : cut_cdf ;  // large symbols with max proba
  }

  // TODO: If cuts cannot cover the whole range, leave the final one smaller 
  // int L = num_cuts - 1;
  // while (s > 0) {
  //   const int S = symbols[--s];
  //   const int L = symbols[0];
  //   assert(proba[S] < cut);       // check that S is a small one
  //   const int cut_cdf = proba[S] + S * cut;
  //   cdf_alias_table[S].cut_cdf = (cut_cdf >= max_table_size) ? max_table_size : cut_cdf ;
  //   if (cut_cdf >= max_table_size) std::cout << "CDF limit detected on small! Symbol:" << S << ", Other:" << L << std::endl;
  //   cdf_alias_table[S].other_symbol = L;
  //   if (S != L) {
  //     proba[L] -= cut - proba[S];   // decrease large proba
  //     assert(proba[L] > 0);       // check that L is still valid
  //   }
  // }

  // Accumulate counts and compute the start_.
  std::vector<RANS_FREQ_TYPE> c(num_cuts, 0);
  for (s = 0; s < num_cuts; ++s) {
    if (s * cut >= max_table_size) break;
    const int other = cdf_alias_table[s].other_symbol;
    const int cut_cdf = cdf_alias_table[s].cut_cdf;
    const int count_s = cut_cdf - s * cut;
    const int count_other = ((s+1) * cut >= max_table_size) ? (max_table_size - cut_cdf) : (cut - count_s);    // complement to 'cut'
    cdf_alias_table[s].self_alias_offset = s * cut - c[s];
    cdf_alias_table[s].other_alias_offset = s * cut + count_s - c[other];
    c[s]     += count_s;
    c[other] += count_other;
  }

  // build remap
  for (RANS_FREQ_TYPE r = 0; r < max_table_size; ++r) {
    const RANS_FREQ_TYPE alias_map_id = r / cut;
    const RANS_FREQ_TYPE cut_cdf = cdf_alias_table[alias_map_id].cut_cdf;
    const RANS_FREQ_TYPE cdf_idx = (r >= cut_cdf) ? cdf_alias_table[alias_map_id].other_symbol : alias_map_id;
    const RANS_FREQ_TYPE alias_start = (r >= cut_cdf) ? cdf_alias_table[alias_map_id].other_alias_offset : cdf_alias_table[alias_map_id].self_alias_offset;
    cdf_alias_remap[r - alias_start + cdf[cdf_idx]] = r;
  }
  return true;
}
