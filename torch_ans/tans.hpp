#pragma once

// ---------------------------------------------------------------------------
// The tANS (FSE-style tabled ANS) family: op wrappers.
// ---------------------------------------------------------------------------
// Structurally the same as rans.hpp - one wrapper per direction that validates
// the arguments, applies the capacity contract and routes to the family kernels
// - so that `TorchANSInterface` can switch impl without a single call-site
// change. Three families are exported, differing only in the state and stream
// word width (see tans_utils.hpp::TANS_MAX_TABLELOG and the state-width bound
// for the resulting freq_precision caps):
//
//   tans32: uint32 state, uint16 stream words, freq_precision <= 15
//   tans16: uint16 state, uint8  stream words, freq_precision <= 15
//   tans8 : uint8  state, uint8  stream words, freq_precision <= 7
//
// The tables are *shared* by all three: they only depend on the (cdf,
// table_log) pair, which is why tans_build_tables takes no family argument and
// lands in the common operator set.
//
// Unlike rANS there is no alias-sampling and no inverse-CDF variant (both are
// rANS lookups that tANS replaces with its own table), and interleaving is not
// implemented yet: `tans_init_stream` rejects num_interleaves != 1 rather than
// silently producing a stream the kernels do not understand.
//
// CUDA is not implemented either: the wrappers reject device tensors with an
// explanation instead of dispatching to a missing kernel.

#include "ta/torch_api.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "rans_build_config.hpp"
#include "stream_capacity.hpp"
#include "tans_utils.hpp"

// ---------------------------------------------------------------------------
// table construction (host, CPU-only - the same shape as rans_alias_build_table)
// ---------------------------------------------------------------------------
// One row per distribution, holding both directions of the table (the encoder's
// nextState + symbolTT and the decoder's symbol/newState/nbBits), so that the
// table tensor can be handed to `push` and `pop` unchanged - see tans_utils.hpp
// for the layout and why the state tables come first.
//
// `table_log` must equal the precision the cdfs were quantized with (this
// implementation does not renormalize them, see tans_build_tables_row).
ta::Tensor tans_build_tables(const ta::Tensor& cdfs, const ta::Tensor& cdfs_sizes,
                             int64_t table_log);

// ---------------------------------------------------------------------------
// kernels (defined in tans_cpu.cpp)
// ---------------------------------------------------------------------------
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
ta::Tensor tans_init_stream(int64_t size, int64_t num_interleaves=1, int64_t preallocate_size=0);

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
void tans_push_indexed_cpu(
  ta::Tensor stream,
  const ta::Tensor& symbols,
  const ta::Tensor& indexes,
  const ta::Tensor& tables,
  const ta::Tensor& cdfs_sizes,
  const ta::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision);

template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
ta::Tensor tans_pop_indexed_cpu(
  ta::Tensor stream,
  const ta::Tensor& indexes,
  const ta::Tensor& tables,
  const ta::Tensor& cdfs_sizes,
  const ta::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision);

// ---------------------------------------------------------------------------
// op wrappers
// ---------------------------------------------------------------------------

// The largest table_log this family can code with: the 16.16 fixed point
// arithmetic of the encode step bounds it at 15 for every family, and the state
// width at state_bits - 1 (the encoder state lives in [2**L, 2**(L+1))).
template <typename TANS_STATE_TYPE>
inline int64_t tans_max_table_log() {
  const int64_t state_bound = (int64_t)sizeof(TANS_STATE_TYPE) * 8 - 1;
  return state_bound < (int64_t)TANS_MAX_TABLELOG ? state_bound : (int64_t)TANS_MAX_TABLELOG;
}

// Shared validation of the two coding directions: the invariants a caller can
// break (the Python layer already validates what it can, this is the second
// line of defence for direct op calls in tests and scripts).
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
inline void tans_check_coding_args(int64_t freq_precision, bool bypass_coding,
                                   int64_t bypass_precision, const char* what) {
  STD_TORCH_CHECK(freq_precision >= 1 && freq_precision <= tans_max_table_log<TANS_STATE_TYPE>(),
                  what, ": freq_precision must be in [1, ",
                  tans_max_table_log<TANS_STATE_TYPE>(), "] for a ",
                  (int64_t)sizeof(TANS_STATE_TYPE) * 8, "-bit tANS state (got ",
                  freq_precision, "); the table has 2**freq_precision slots, so a "
                  "larger value is both too big to allocate per distribution and "
                  "outside the range the encoder's fixed point arithmetic handles");
  if (bypass_coding) {
    STD_TORCH_CHECK(bypass_precision >= 1 && bypass_precision <= freq_precision,
                    what, ": bypass_precision must be in [1, freq_precision] (got ",
                    bypass_precision, " with freq_precision=", freq_precision, ")");
  }
}

// One push, with the same trailing `used_bytes` capacity contract as rans_push
// (see stream_capacity.hpp).
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
void tans_push(// tANS stream
  ta::Tensor stream,
  ta::Tensor symbols,
  std::optional<ta::Tensor> indexes,
  std::optional<ta::Tensor> tables,
  std::optional<ta::Tensor> cdfs_sizes,
  std::optional<ta::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true,
  int64_t bypass_precision=4,
  int64_t used_bytes=-1)
{
  (void)symbol_precision;  // part of the shared signature, unused by tANS

  const int64_t num_symbols = symbols.size(1);
  const int64_t batch_size = stream.size(0);

  if (symbols.size(0) != batch_size) {
    // std::invalid_argument rather than STD_TORCH_CHECK: pybind11 maps it to
    // ValueError, like rans_push (the callers in utils.py and the tests expect
    // a ValueError).
    throw std::invalid_argument("Symbols batch size mismatch!");
  }

  STD_TORCH_CHECK(!ta::is_cuda(stream) && !ta::is_cuda(symbols),
                  "tans_push: the tANS family is CPU-only for now; the tensors are "
                  "on a CUDA device. Move them to the CPU, or use impl=\"rans64\" / "
                  "\"rans32\" / \"rans32_16\", whose CUDA kernels are implemented.");
  STD_TORCH_CHECK(indexes.has_value(),
                  "tans_push: tANS codes through the indexed pattern, so `indexes` "
                  "is required (the per-symbol cdf pattern has no tANS equivalent)");
  STD_TORCH_CHECK(tables.has_value() && cdfs_sizes.has_value(),
                  "tans_push: `cdfs` must be the table tensor built by "
                  "tans_build_tables and `cdfs_sizes` the matching per-distribution "
                  "symbol counts");
  STD_TORCH_CHECK(tables.value().scalar_type() == ta::ScalarType::Int,
                  "tans_push: the table tensor must be int32 (as tans_build_tables "
                  "returns it)");

  tans_check_coding_args<TANS_STATE_TYPE, TANS_STREAM_TYPE>(
      freq_precision, bypass_coding, bypass_precision, "tans_push");

  ans_ensure_stream_capacity<TANS_STATE_TYPE, TANS_STREAM_TYPE>(
      stream, num_symbols, freq_precision, bypass_coding, bypass_precision,
      /*num_interleaves=*/1, used_bytes);

  ta::Tensor offsets_tensor;
  if (!offsets.has_value())
    offsets_tensor = ta::new_zeros_1d(indexes.value(), tables.value().size(0));
  else
    offsets_tensor = offsets.value();

  return tans_push_indexed_cpu<TANS_STATE_TYPE, TANS_STREAM_TYPE>(
    ta::as_contiguous(stream), ta::as_contiguous(symbols),
    ta::as_contiguous(indexes.value()), ta::as_contiguous(tables.value()),
    ta::as_contiguous(cdfs_sizes.value()), ta::as_contiguous(offsets_tensor),
    freq_precision, bypass_coding, bypass_precision);
}

// One pop. `inverse_cdf_precision` is part of the shared rans_pop signature and
// ignored here (tANS looks its symbol up in its own table, so there is no
// inverse-CDF variant).
template <typename TANS_STATE_TYPE, typename TANS_STREAM_TYPE>
ta::Tensor tans_pop(// tANS stream
  ta::Tensor stream,
  std::optional<ta::Tensor> indexes,
  std::optional<ta::Tensor> tables,
  std::optional<ta::Tensor> cdfs_sizes,
  std::optional<ta::Tensor> offsets,
  int64_t symbol_precision=8,
  int64_t freq_precision=16,
  bool bypass_coding=true,
  int64_t bypass_precision=4,
  int64_t inverse_cdf_precision=-1)
{
  (void)symbol_precision;
  (void)inverse_cdf_precision;

  STD_TORCH_CHECK(!ta::is_cuda(stream),
                  "tans_pop: the tANS family is CPU-only for now; the stream is on "
                  "a CUDA device. Move it to the CPU, or use impl=\"rans64\" / "
                  "\"rans32\" / \"rans32_16\", whose CUDA kernels are implemented.");
  STD_TORCH_CHECK(indexes.has_value(),
                  "tans_pop: tANS decodes through the indexed pattern, so `indexes` "
                  "is required (the per-symbol cdf pattern has no tANS equivalent)");
  STD_TORCH_CHECK(tables.has_value() && cdfs_sizes.has_value(),
                  "tans_pop: `cdfs` must be the table tensor built by "
                  "tans_build_tables and `cdfs_sizes` the matching per-distribution "
                  "symbol counts");
  STD_TORCH_CHECK(tables.value().scalar_type() == ta::ScalarType::Int,
                  "tans_pop: the table tensor must be int32 (as tans_build_tables "
                  "returns it)");

  tans_check_coding_args<TANS_STATE_TYPE, TANS_STREAM_TYPE>(
      freq_precision, bypass_coding, bypass_precision, "tans_pop");

  ta::Tensor offsets_tensor;
  if (!offsets.has_value())
    offsets_tensor = ta::new_zeros_1d(indexes.value(), tables.value().size(0));
  else
    offsets_tensor = offsets.value();

  return tans_pop_indexed_cpu<TANS_STATE_TYPE, TANS_STREAM_TYPE>(
    ta::as_contiguous(stream), ta::as_contiguous(indexes.value()),
    ta::as_contiguous(tables.value()), ta::as_contiguous(cdfs_sizes.value()),
    ta::as_contiguous(offsets_tensor),
    freq_precision, bypass_coding, bypass_precision);
}
