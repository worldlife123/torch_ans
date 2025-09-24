#pragma once

#include <torch/extension.h>

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

#define RANS_APPEND_STATE_RENORM(state, freq, freq_precision, pptr) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {if (state >= ((RANS_STATE_LOWER_BOUND >> freq_precision) << RANS_STREAM_BITS) * freq) {APPEND_STATE_TO_STREAM(state, pptr);}} \
    else \
      {while (state >= ((RANS_STATE_LOWER_BOUND >> freq_precision) << RANS_STREAM_BITS) * freq) {APPEND_STATE_TO_STREAM(state, pptr);}}

#define RANS_APPEND_STATE_RENORM_OVERFLOW(state, pptr) \
    if (RANS_STATE_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {if (state >= (RANS_STATE_LOWER_BOUND << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}} \
    else \
      {while (state >= (RANS_STATE_LOWER_BOUND << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}}

#define RANS_POP_STATE_RENORM(state, pptr) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {if (state < RANS_STATE_LOWER_BOUND) {POP_STATE_FROM_STREAM(state, pptr);}} \
    else \
      {while (state < RANS_STATE_LOWER_BOUND) {POP_STATE_FROM_STREAM(state, pptr);}}

#define RANS_APPEND_BITS(state, pptr, value, nbits) \
    if (RANS_STATE_USED_BITS - RANS_STREAM_BITS < RANS_STREAM_BITS) \
      {if (state >= ((RANS_STATE_LOWER_BOUND >> nbits) << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}} \
    else \
      {while (state >= ((RANS_STATE_LOWER_BOUND >> nbits) << RANS_STREAM_BITS)) {APPEND_STATE_TO_STREAM(state, pptr);}} \
    state = (x << nbits) | value;

#define RANS_POP_BITS(state, pptr, value, nbits) \
    value = (RANS_SYMBOL_TYPE) (state & ((1u << nbits) - 1)); state = state >> bypass_precision; RANS_POP_STATE_RENORM(state, pptr)


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
    ssize_t bypass_precision
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
    RANS_SYMBOL_TYPE n_bypass = 0;
    while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
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
    for (ssize_t j = n_bypass-1; j >= 0; j--) {
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
    for (ssize_t j = n_codes_n_bypass-1; j >= 0; j--) {
      RANS_APPEND_BITS(x, stream_pptr, max_bypass_val, bypass_precision);
    }

#ifdef DEBUG_STEPS
    std::cout << "PUSH BYPASS: state_ptr:" << state_ptr << ", stream_ptr:" << (void*)(*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << x << std::endl;
#endif
    *state_ptr = x;
    
}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, typename RANS_FREQ_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline void rans_push_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    RANS_SYMBOL_TYPE symbol, 
    const RANS_FREQ_TYPE* cdf, 
    const RANS_SYMBOL_TYPE cdf_size, 
    const RANS_SYMBOL_TYPE offset, 
    ssize_t freq_precision,
    bool bypass_coding, 
    ssize_t bypass_precision,
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

    // TODO: enable use_post_renorm may reduce efficiency! Maybe consider fixing freq_precision as template?
    const bool use_post_renorm = false; // (RANS_STATE_USED_BITS + freq_precision) <= (sizeof(RANS_STATE_TYPE) * 8);

    RANS_STATE_TYPE x = *state_ptr;
    // directly put bits
    if (cdf == nullptr) {
      const RANS_FREQ_TYPE freq = (1 << freq_precision) / max_value;
      const RANS_FREQ_TYPE start = freq * value;
      if (!use_post_renorm) {RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);}
      /* x = C(s, x) */
      x = ((x / freq) << freq_precision) + (x % freq) + start;
      if (use_post_renorm) {RANS_APPEND_STATE_RENORM_OVERFLOW(x, stream_pptr);}
    }
    // cdf-based coding
    else {

      // Rans64EncPut(state_ptr, stream_pptr, cdf[value], cdf[value + 1] - cdf[value], freq_precision);
      const RANS_FREQ_TYPE start = cdf[value];
      const RANS_FREQ_TYPE freq = cdf[value + 1] - start;
      if (!use_post_renorm) {RANS_APPEND_STATE_RENORM(x, freq, freq_precision, stream_pptr);}

      if (cdf_alias_remap != nullptr) {
        const RANS_STATE_TYPE new_state = ((x / freq) << freq_precision) + cdf_alias_remap[(x % freq) + start];
#ifdef DEBUG_STEPS
        std::cout << "state_ptr:" << state_ptr << ", stream_ptr:" << (void*)(*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << new_state << ", symbol:" << symbol << ", start: " << start << ", freq: " << freq << ", cdf_alias: " << cdf_alias_remap[(x % freq) + start] << ", cum_freq_offset: " << (x % freq) << ", value: " << value << std::endl;
#endif
        x = new_state;
      }
      else {
        // x = C(s,x)
        const RANS_STATE_TYPE new_state = ((x / freq) << freq_precision) + (x % freq) + start;
#ifdef DEBUG_STEPS
        std::cout << "state_ptr:" << state_ptr << ", stream_ptr:" << (void*) (*stream_pptr) << ", state:" << *state_ptr << ", newstate:" << new_state << ", symbol:" << symbol << ", start: " << start << ", freq: " << freq << ", value: " << value << std::endl;
#endif
        x = new_state;
      }

      if (use_post_renorm) {RANS_APPEND_STATE_RENORM_OVERFLOW(x, stream_pptr);}
    }
    *state_ptr = x;

}


template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename RANS_SYMBOL_TYPE, size_t RANS_STATE_VALID_BITS=0>
RANS_API inline RANS_SYMBOL_TYPE rans_pop_raw_value_step(RANS_STATE_TYPE* state_ptr, RANS_STREAM_TYPE** stream_pptr, 
    ssize_t bypass_precision
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
    ssize_t freq_precision,
    bool bypass_coding, 
    ssize_t bypass_precision,
    const RANS_FREQ_TYPE* inversed_cdf,
    const RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>* cdf_alias_table
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
      cdf_idx = inversed_cdf[cum_freq];
      cum_freq_offset -= cdf[cdf_idx];
    }
    else {
// #ifdef RANS_CUDA_API
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
RANS_API inline bool build_alias_mapping(
  const RANS_FREQ_TYPE* cdf, 
  const RANS_SYMBOL_TYPE cdf_size, 
  RANSAliasSamplingCDFTableElement<RANS_FREQ_TYPE>* cdf_alias_table,
  RANS_FREQ_TYPE* cdf_alias_remap,
  ssize_t symbol_precision,
  ssize_t freq_precision
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
  
  RANS_SYMBOL_TYPE symbols[num_cuts];
  int l = num_cuts, s = 0;
  RANS_FREQ_TYPE proba[num_cuts];
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
  RANS_FREQ_TYPE c[num_cuts] = { 0 };
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