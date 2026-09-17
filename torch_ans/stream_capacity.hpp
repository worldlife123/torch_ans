#pragma once

// ---------------------------------------------------------------------------
// The worst-case stream growth of a push, shared by every ANS family.
// ---------------------------------------------------------------------------
// This used to live inline in `rans_push`, which was fine while rANS was the
// only family. `tans_push` needs exactly the same arithmetic and the same
// compare/throw/grow behaviour, and a second copy is precisely what drifts (the
// capacity contract fails loudly - writing past the row or rejecting a good
// call - which is the worst kind of drift to debug). So there is one C++
// implementation here, one Python mirror (`torch_ans/utils.py::_stream_budget`)
// and `tests/test_capacity_contract.py` pins the two against each other by
// using the wrapper's own growth as the oracle.
//
// It is family-independent because the bound is: per symbol at most
// `freq_precision` bits (rANS: log2(2**freq_precision / freq) <= freq_precision;
// tANS: nbBits <= table_log == freq_precision) plus, with bypass coding, the
// raw value as `bypass_precision`-sized chunks - `ceil(symbol_bits /
// bypass_precision)` of them for the value itself, one chunk holding the chunk
// count and one more per carry of that count (a carry every max_bypass_val).
//
// `used_bytes >= 0` is the caller's upper bound on the bytes already coded into
// the longest row; with it the caller owns the capacity and this function only
// *compares* - no `select`+`amax` (a blocking device->host copy on CUDA) and no
// `aten::resize_`, whose schema the stable ABI does not promise (migration
// section 4, pit 4). -1 keeps the historical behaviour: read the length word
// and grow in place, which is what a direct op call (tests, benchmarks,
// scripts) gets.

#include "ta/torch_api.hpp"

#include <cstdint>
#include <limits>
#include <stdexcept>

// ---------------------------------------------------------------------------
// `RANS_STATE_TYPE` / `RANS_STREAM_TYPE` only supply the sizes of the slack
// terms; the arithmetic itself is indifferent to the family.
// ---------------------------------------------------------------------------
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE>
inline void ans_ensure_stream_capacity(ta::Tensor& stream, int64_t num_symbols,
                                       int64_t freq_precision, bool bypass_coding,
                                       int64_t bypass_precision, int64_t num_interleaves,
                                       int64_t used_bytes)
{
  // The length word is read (when the caller does not own the capacity) in the
  // stream tensor's own dtype, so the bound has to be computed per dtype. Only
  // the *sizes* differ: an integral tensor is what every family uses.
  TA_DISPATCH_INTEGRAL(stream.scalar_type(), "ans_ensure_stream_capacity", [&] {
    const int64_t batch_size = stream.size(0);
    const bool caller_owns_capacity = used_bytes >= 0;
    const int64_t max_byte_length = caller_owns_capacity
        ? used_bytes
        : (int64_t)ta::max_scalar<scalar_t>(stream);
    int64_t bits_per_symbol = freq_precision;
    if (bypass_coding) {
      STD_TORCH_CHECK(bypass_precision >= 1, "bypass_precision must be >= 1");
      const int64_t symbol_bits = (int64_t)sizeof(scalar_t) * 8;
      const int64_t max_bypass_val = ((int64_t)1 << bypass_precision) - 1;
      const int64_t n_bypass = (symbol_bits + bypass_precision - 1) / bypass_precision;
      const int64_t n_chunks = n_bypass + 1 + n_bypass / max_bypass_val;
      bits_per_symbol += n_chunks * bypass_precision;
    }
    // +7 rounds the bit total up to whole bytes. On top of that, every ANS state
    // can hold up to one not-yet-flushed stream word (rANS: the state itself;
    // tANS: the bit buffer in the stream header), so reserve one word per
    // interleave (the amortized bit estimate above does not account for it),
    // plus one more word for rANS's branch-free renormalization, which always
    // writes one word past the cursor (see APPEND_STATE_TO_STREAM_IF).
    const int64_t safe_byte_length = max_byte_length
        + (num_symbols * bits_per_symbol + 7) / 8
        + ((int64_t)num_interleaves + 1) * (int64_t)sizeof(RANS_STREAM_TYPE);

    if (safe_byte_length < max_byte_length) {
      throw std::invalid_argument("Overflow!");
    }

    // NOTE: the computation above is deliberately kept in int64_t rather than a
    // platform-width type. The intermediate num_symbols * bits_per_symbol already
    // exceeds 2^31 for realistic workloads (1e8 symbols at 51 bits each is
    // ~5e9), and signed overflow is undefined behaviour, so a 32-bit type would
    // corrupt the result rather than merely trip the check below.
    //
    // The result is then range-checked against size_t, which *is* platform-width
    // (32-bit on 32-bit systems, where no allocation can exceed 4 GiB), so an
    // unrepresentable stream fails here instead of being silently truncated.
    const int64_t safe_tensor_length = safe_byte_length / (int64_t)sizeof(scalar_t) + 1;
    const uint64_t total_bytes =
        (uint64_t)batch_size * (uint64_t)safe_tensor_length * (uint64_t)sizeof(scalar_t);
    STD_TORCH_CHECK(total_bytes <= (uint64_t)std::numeric_limits<size_t>::max(),
                    "ans_ensure_stream_capacity: the stream would need ", total_bytes,
                    " bytes, which is not addressable on this platform");
    if (safe_tensor_length > stream.size(1)) {
      if (caller_owns_capacity) {
        // The caller reserved the capacity, so a stream that is still too small
        // is a bug on its side: report it loudly rather than papering over it
        // with a resize (which would need the length word this parameter exists
        // to avoid reading) - and certainly rather than letting the kernel write
        // past the row.
        throw std::invalid_argument(
            "ans_ensure_stream_capacity: the stream has " +
            std::to_string(stream.size(1)) + " columns but this call needs " +
            std::to_string(safe_tensor_length) + " (used_bytes=" +
            std::to_string(used_bytes) +
            "); when used_bytes is given the caller owns the stream capacity");
      }
      // `resize_` in place, then copy the old contents back into the first
      // stream_copy.size(1) columns. ATen's index_put_ with a Slice pair has no
      // stable equivalent; narrow + copy_ is the same operation (migration
      // section 2.2), and ta::resize2d is the one dispatcher call the layer
      // keeps (no stable wrapper for aten::resize_ exists).
      const ta::Tensor stream_copy = ta::clone(stream);
      ta::resize2d(stream, batch_size, safe_tensor_length);
      stream = ta::as_contiguous(stream);
      ta::Tensor dst = ta::narrow(stream, 1, 0, stream_copy.size(1));
      ta::copy_(dst, stream_copy);
    }
  });
}
