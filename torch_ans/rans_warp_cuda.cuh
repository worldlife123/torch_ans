#pragma once

// Warp-level interleaved rANS kernels (CUDA/HIP device code).
//
// Technique reference: 3rdparty/recoil (warp-cooperative interleaved rANS,
// decode-only there). One warp — or a power-of-two sub-warp group of
// NUM_INTERLEAVES lanes — codes one batch row cooperatively: every lane owns
// one interleaved rANS state and the group shares a single bitstream cursor
// held in registers. Renormalization words are placed by warp-collective
// accounting (__ballot_sync / __shfl_xor_sync), so no shared memory and no
// __syncthreads are needed.
//
// The kernels are bit-compatible with the CPU interleaved implementation in
// rans_cpu.cpp (same stream layout, same word order):
//   - symbol s is coded on lane s % NUM_INTERLEAVES (tail and main loop);
//   - push walks symbol groups from high to low, appending forward, with the
//     lanes of a group visited from N-1 down to 0 (bypass raw-value runs of
//     each lane first, then the symbol pushes);
//   - pop walks groups from low to high, consuming the stream backward, with
//     lane-ascending order (parallel ballot for symbol pops, sequential
//     lane-by-lane for bypass raw-value runs whose lengths are only
//     discoverable while decoding).
//
// Only instantiate with the rans32_16 configuration
// (RANS_STATE_TYPE=uint32_t, RANS_STREAM_TYPE=uint16_t): it guarantees at
// most one stream word per renormalization step for all allowed precisions
// (freq_precision <= 15), which the ballot/prefix scheme relies on.

#include <torch/extension.h>
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "rans_utils.hpp"

namespace rans_warp {

constexpr int WARP_NUM_THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// Bypass raw-value coding, restructured for per-lane local simulation.
// Mirrors rans_push_raw_value_step / rans_pop_raw_value_step (rans_utils.hpp)
// without owning the stream pointer: the caller decides where the produced
// words land (warp-ranked position) and how far the shared cursor moves.
// Each RANS_APPEND_BITS / RANS_POP_BITS step emits/consumes at most one
// stream word for the rans32_16 configuration.
// ---------------------------------------------------------------------------

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE>
__device__ __forceinline__ unsigned rans_warp_simulate_raw_push(
    RANS_STATE_TYPE& state, TORCH_TENSOR_TYPE raw_val, const int bypass_precision, RANS_STREAM_TYPE* out)
{
  const TORCH_TENSOR_TYPE max_bypass_val = (TORCH_TENSOR_TYPE)((1u << bypass_precision) - 1);

  // NOTE: keep the shift amount below the symbol width (see
  // rans_push_raw_value_step in rans_utils.hpp): the count is capped at
  // ceil(symbol_bits / bypass_precision), the largest value the bypass digits
  // can represent. Without the cap the shift reaches 32, which is UB and
  // resolves differently on the CPU (wrap-around -> infinite loop) and on the
  // GPU (clamped -> 0), so the two backends would disagree.
  TORCH_TENSOR_TYPE n_bypass = 0;
  while ((int64_t)(n_bypass * bypass_precision) < (int64_t)(sizeof(TORCH_TENSOR_TYPE) * 8) &&
         ((raw_val >> (n_bypass * bypass_precision)) != 0)) {
    ++n_bypass;
  }

  TORCH_TENSOR_TYPE val = n_bypass;
  TORCH_TENSOR_TYPE n_codes_n_bypass = 0;
  TORCH_TENSOR_TYPE rest_n_bypass = 0;
  while (val >= max_bypass_val) {
    ++n_codes_n_bypass;
    val -= max_bypass_val;
  }
  rest_n_bypass = val;

  RANS_STATE_TYPE x = state;
  unsigned count = 0;
  // one RANS_APPEND_BITS step: renorm-append at most one word, then shift in
  auto append_bits = [&](const TORCH_TENSOR_TYPE value) {
    if (x >= ((RANS_STATE_LOWER_BOUND >> bypass_precision) << RANS_STREAM_BITS)) {
      if (out != nullptr) out[count] = (RANS_STREAM_TYPE)x;
      x >>= RANS_STREAM_BITS;
      ++count;
    }
    x = (x << bypass_precision) | (RANS_STATE_TYPE)value;
  };

  for (int64_t j = (int64_t)n_bypass - 1; j >= 0; j--) {
    const TORCH_TENSOR_TYPE bypass_val = (raw_val >> (j * bypass_precision)) & max_bypass_val;
    append_bits(bypass_val);
  }
  append_bits(rest_n_bypass);
  for (int64_t j = (int64_t)n_codes_n_bypass - 1; j >= 0; j--) {
    append_bits(max_bypass_val);
  }

  state = x;
  return count;
}

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE>
__device__ __forceinline__ unsigned rans_warp_simulate_raw_pop(
    RANS_STATE_TYPE& state, const RANS_STREAM_TYPE* cursor, const int bypass_precision,
    TORCH_TENSOR_TYPE& raw_val_out)
{
  const TORCH_TENSOR_TYPE max_bypass_val = (TORCH_TENSOR_TYPE)((1u << bypass_precision) - 1);

  RANS_STATE_TYPE x = state;
  unsigned consumed = 0;
  // one RANS_POP_BITS step: extract the low bits, then renorm-pop at most one
  // word (single-word pop is guaranteed for freq_precision <= RANS_STREAM_BITS)
  auto pop_bits = [&]() -> TORCH_TENSOR_TYPE {
    const TORCH_TENSOR_TYPE val = (TORCH_TENSOR_TYPE)(x & ((1u << bypass_precision) - 1));
    x >>= bypass_precision;
    if (x < RANS_STATE_LOWER_BOUND) {
      x = (x << RANS_STREAM_BITS) | (RANS_STATE_TYPE)cursor[-(int)consumed];
      ++consumed;
    }
    return val;
  };

  TORCH_TENSOR_TYPE val = pop_bits();
  auto n_bypass = val;
  while (val == max_bypass_val) {
    val = pop_bits();
    n_bypass += val;
  }

  TORCH_TENSOR_TYPE raw_val = 0;
  for (int64_t j = 0; j < (int64_t)n_bypass; ++j) {
    val = pop_bits();
    raw_val |= val << (j * bypass_precision);
  }

  state = x;
  raw_val_out = raw_val;
  return consumed;
}

// Decode-side symbol lookup by cumulative frequency. Mirrors the three lookup
// branches of rans_pop_step (rans_utils.hpp): alias table, inversed CDF, and
// the default divided search + binary refine.
//
// The inversed CDF table may be *sparse*: it holds 2^inverse_cdf_precision
// entries only, entry i answering the query for the bucket start
// `i << (freq_precision - inverse_cdf_precision)`. A bucket start never exceeds
// the queried cum_freq, so the table entry is a lower bound of the true symbol
// and a short linear walk closes the remaining gap. This trades the
// 2^freq_precision space of a dense table for a bounded walk, and - unlike the
// divided search - its cost no longer depends on the symbol distribution.
//
// Walk bound: with `W = 1 << (freq_precision - inverse_cdf_precision)` and
// `lo = table[cum_freq >> ...]`, the CDF construction guarantees every symbol
// owns at least one frequency unit, and cdf[lo + 1] > bucket_start by
// definition of lo, so `cdf_idx - lo <= W`. A dense table (W == 1) degenerates
// to the single-load O(1) path.
// ``inversed_cdf_ptr`` is passed in instead of being derived as
// ``cdf_ptr + cdf_size`` because A9 may keep the CDF and the inverse-CDF table
// in different memories (CDF in shared, table still in global).
template <typename TORCH_TENSOR_TYPE, bool USE_ALIAS_SAMPLING_CDF, bool USE_INVERSED_CDF>
__device__ __forceinline__ TORCH_TENSOR_TYPE rans_warp_pop_lookup(
    const TORCH_TENSOR_TYPE* cdf_ptr, const TORCH_TENSOR_TYPE* inversed_cdf_ptr,
    const TORCH_TENSOR_TYPE cdf_size,
    const TORCH_TENSOR_TYPE cum_freq, const int freq_precision,
    const int inverse_cdf_precision, TORCH_TENSOR_TYPE& cum_freq_offset)
{
  TORCH_TENSOR_TYPE cum_freq_offset_local = cum_freq;
  TORCH_TENSOR_TYPE cdf_idx;  // NOTE: kept unsigned internally like rans_pop_step
  if constexpr (USE_ALIAS_SAMPLING_CDF) {
    const auto cdf_alias_table_ptr =
        reinterpret_cast<const RANSAliasSamplingCDFTableElement<TORCH_TENSOR_TYPE>*>(cdf_ptr + cdf_size);
    const TORCH_TENSOR_TYPE cut_size = (TORCH_TENSOR_TYPE)((1 << freq_precision) / (cdf_size - 1));
    const TORCH_TENSOR_TYPE alias_map_id = cum_freq / cut_size;
    const TORCH_TENSOR_TYPE cut_cdf = cdf_alias_table_ptr[alias_map_id].cut_cdf;
    cdf_idx = (cum_freq >= cut_cdf) ? cdf_alias_table_ptr[alias_map_id].other_symbol : alias_map_id;
    const TORCH_TENSOR_TYPE alias_start = (cum_freq >= cut_cdf)
        ? cdf_alias_table_ptr[alias_map_id].other_alias_offset
        : cdf_alias_table_ptr[alias_map_id].self_alias_offset;
    cum_freq_offset_local -= alias_start;
  } else if constexpr (USE_INVERSED_CDF) {
    const int shift = freq_precision - inverse_cdf_precision;
    cdf_idx = inversed_cdf_ptr[cum_freq >> shift];
    // sparse table: walk forward from the bucket's symbol to the exact one.
    // NOTE: measured alternatives - a fixed-length predicated walk (uniform
    // trip count, SIMT-friendly in principle) was ~5-16% SLOWER on the warp
    // kernels: SIMT already runs the divergent walk at the group's slowest
    // lane, so the fixed loop only adds unconditional loads/comparisons per
    // symbol. The early-exit while stays.
    while (cdf_ptr[cdf_idx + 1] <= cum_freq) ++cdf_idx;
    cum_freq_offset_local -= cdf_ptr[cdf_idx];
  } else {
    // divided search (seems to be fastest)
    cdf_idx = (TORCH_TENSOR_TYPE)((cdf_size * cum_freq) >> freq_precision);
    uint32_t low = 0;
    uint32_t high = (uint32_t)(cdf_size - 1);
    while (high > low) {
      if (cum_freq >= cdf_ptr[cdf_idx])
        low = cdf_idx + 1;
      else
        high = cdf_idx;
      cdf_idx = (low + high) / 2;
    }
    cdf_idx--;
    cum_freq_offset_local -= cdf_ptr[cdf_idx];
  }
  cum_freq_offset = cum_freq_offset_local;
  return cdf_idx;
}

// ---------------------------------------------------------------------------
// Push (encode) kernel: one lane group per batch row, reverse coding.
// ---------------------------------------------------------------------------

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_ALIAS_SAMPLING_CDF = false, size_t NUM_INTERLEAVES = 1>
__global__ void rans_warp_push_indexed_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  const int freq_precision, const bool bypass_coding, const int bypass_precision)
{
  static_assert(NUM_INTERLEAVES <= 32 && (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "NUM_INTERLEAVES must be a power of two no greater than the warp size");
  constexpr unsigned N = (unsigned)NUM_INTERLEAVES;

  const unsigned groups_per_block = blockDim.x / N;
  const unsigned group_in_block = threadIdx.x / N;
  const unsigned lane = threadIdx.x % N;
  const unsigned lane_in_warp = threadIdx.x & 31u;
  const unsigned group_start_in_warp = lane_in_warp - lane;
  // masks are group-local so that groups exiting early never break collectives
  const unsigned group_mask = (N == 32u) ? 0xffffffffu : (((1u << N) - 1u) << group_start_in_warp);
  // bit mask of the group-local lanes [0, lane], shifted into the group's
  // position inside the warp (the group does not necessarily start at lane 0)
  const unsigned mask_le = group_mask & ((((1u << lane) - 1u) | (1u << lane)) << group_start_in_warp);

  const int64_t b = (int64_t)blockIdx.x * groups_per_block + group_in_block;
  if (b >= stream_accessor.size(0)) return;  // uniform per group

  const int64_t num_symbols = symbols_accessor.size(1);
  const int64_t num_cdfs = cdfs_accessor.size(0);
  const int p = freq_precision;

  const auto stream_length = stream_accessor[b][0];
  RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data() + 1);
  RANS_STREAM_TYPE* base = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr);
  // the cursor names the last written word; appends start at cursor + 1
  RANS_STREAM_TYPE* cursor = base + stream_length / sizeof(RANS_STREAM_TYPE) - 1;

  RANS_STATE_TYPE state = state_ptr[lane];

  const TORCH_TENSOR_TYPE* symbols_ptr = symbols_accessor[b].data();
  const TORCH_TENSOR_TYPE* indexes_ptr = indexes_accessor[b].data();

  // --- tail: the top (num_symbols % N) symbols, one at a time, lane s % N ---
  const int64_t first_interleave = num_symbols % (int64_t)N;
  for (int64_t s = num_symbols - 1; s >= num_symbols - first_interleave; s--) {
    const unsigned active_lane = (unsigned)(s % (int64_t)N);
    const bool is_active = (lane == active_lane);
    unsigned run_words = 0;
    unsigned sym_words = 0;
    TORCH_TENSOR_TYPE start = 0, freq = 0;
    // alias sampling replaces the low bits the symbol contributes with the
    // remap table entry (see rans_push_step_freq in rans_utils.hpp); the table
    // lives right behind the cdf row
    const TORCH_TENSOR_TYPE* alias_remap_ptr = nullptr;
    bool valid = false;
    bool renorm_pred = false;
    if (is_active) {
      const auto index = indexes_ptr[s];
      valid = (index >= 0 && index < num_cdfs);
      if (valid) {
        const auto cdf_size = cdfs_sizes_accessor[index];
        const auto cdf_ptr = cdfs_accessor[index].data();
        TORCH_TENSOR_TYPE value = symbols_ptr[s] - offsets_accessor[index];
        if (bypass_coding) {
          const auto max_value = cdf_size - 2;
          TORCH_TENSOR_TYPE raw_val = 0;
          if (value < 0) {
            raw_val = -2 * value - 1;
            value = max_value;
          } else if (value >= max_value) {
            raw_val = 2 * (value - max_value);
            value = max_value;
          }
          if (value == max_value) {
            run_words = rans_warp_simulate_raw_push<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE>(
              state, raw_val, bypass_precision, cursor + 1);
          }
        }
        start = cdf_ptr[value];
        freq = cdf_ptr[value + 1] - start;
        if constexpr (USE_ALIAS_SAMPLING_CDF) {
          alias_remap_ptr = cdf_ptr + cdf_size;
        }
        renorm_pred =
            state >= ((RANS_STATE_LOWER_BOUND >> p) << RANS_STREAM_BITS) * (unsigned long long)freq;
      }
    }
    cursor += __shfl_sync(group_mask, run_words, group_start_in_warp + active_lane);
    if (valid && renorm_pred) {
      cursor[1] = (RANS_STREAM_TYPE)state;
      state >>= RANS_STREAM_BITS;
      sym_words = 1;
    }
    cursor += __shfl_sync(group_mask, sym_words, group_start_in_warp + active_lane);
    if (valid) {
      if constexpr (USE_ALIAS_SAMPLING_CDF) {
        state = ((state / freq) << p) + (RANS_STATE_TYPE)alias_remap_ptr[(state % freq) + start];
      } else {
        state = ((state / freq) << p) + (state % freq) + start;
      }
    }
  }

  // --- main loop: symbol groups from high to low, lane L codes symbol gN+L ---
  const int64_t i0 = num_symbols - first_interleave - 1;
  for (int64_t i = i0; i >= 0; i -= (int64_t)N) {
    const int64_t s = i - (int64_t)(N - 1 - lane);
    const auto index = indexes_ptr[s];
    const bool valid = (index >= 0 && index < num_cdfs);
    TORCH_TENSOR_TYPE cdf_value = 0, raw_val = 0;
    unsigned run_words = 0;
    bool is_sentinel = false;
    if (valid) {
      const auto cdf_size = cdfs_sizes_accessor[index];
      if (bypass_coding) {
        const auto max_value = cdf_size - 2;
        TORCH_TENSOR_TYPE value = symbols_ptr[s] - offsets_accessor[index];
        // zig-zag raw value; |value| must stay below ~2^30 for int32 symbols,
        // see the range-limit note in rans_push_step (rans_utils.hpp)
        if (value < 0) {
          raw_val = -2 * value - 1;
          value = max_value;
        } else if (value >= max_value) {
          raw_val = 2 * (value - max_value);
          value = max_value;
        }
        cdf_value = value;
        is_sentinel = (value == max_value);
        if (is_sentinel) {
          RANS_STATE_TYPE tmp = state;
          run_words = rans_warp_simulate_raw_push<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE>(
            tmp, raw_val, bypass_precision, (RANS_STREAM_TYPE*)nullptr);
        }
      } else {
        cdf_value = symbols_ptr[s] - offsets_accessor[index];
      }
    }

    if (bypass_coding) {
      // bypass raw-value runs: lane-descending order, each run consecutive
      // (exclusive prefix of run_words over lanes > self). The shfl_up scan is
      // inclusive, so lane L ends up with sum(run_words[0..L]).
      unsigned c32 = run_words;
      #pragma unroll
      for (unsigned off = 1; off < N; off <<= 1) {
        const unsigned up = __shfl_up_sync(group_mask, c32, off);
        if (lane >= off) c32 += up;
      }
      const unsigned total = __shfl_sync(group_mask, c32, group_start_in_warp + N - 1);
      const unsigned desc_excl = total - c32;
      // NOTE: run the full simulation for sentinel lanes even when it emits
      // zero words — the raw-value bits are still folded into the state
      if (is_sentinel) {
        rans_warp_simulate_raw_push<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE>(
          state, raw_val, bypass_precision, cursor + 1 + desc_excl);
      }
      cursor += total;
    }

    // symbol pushes: at most one word per lane, lane-descending order
    bool pred = false;
    TORCH_TENSOR_TYPE start = 0, freq = 0;
    const TORCH_TENSOR_TYPE* alias_remap_ptr = nullptr;
    if (valid) {
      const auto cdf_ptr = cdfs_accessor[index].data();
      start = cdf_ptr[cdf_value];
      freq = cdf_ptr[cdf_value + 1] - start;
      if constexpr (USE_ALIAS_SAMPLING_CDF) {
        alias_remap_ptr = cdf_ptr + cdfs_sizes_accessor[index];
      }
      pred = state >= ((RANS_STATE_LOWER_BOUND >> p) << RANS_STREAM_BITS) * (unsigned long long)freq;
    }
    const unsigned vote = __ballot_sync(group_mask, pred);
    const unsigned count = __popc(vote);
    if (pred) {
      // lane-descending placement: the highest renorming lane pushes first
      const unsigned rank_desc = count - __popc(vote & mask_le);
      cursor[1 + rank_desc] = (RANS_STREAM_TYPE)state;
      state >>= RANS_STREAM_BITS;
    }
    cursor += count;
    if (valid) {
      if constexpr (USE_ALIAS_SAMPLING_CDF) {
        state = ((state / freq) << p) + (RANS_STATE_TYPE)alias_remap_ptr[(state % freq) + start];
      } else {
        state = ((state / freq) << p) + (state % freq) + start;
      }
    }
  }

  if (lane == 0) {
    stream_accessor[b][0] = (TORCH_TENSOR_TYPE)((cursor - base + 1) * (int64_t)sizeof(RANS_STREAM_TYPE));
  }
  state_ptr[lane] = state;
}

// ---------------------------------------------------------------------------
// Pop (decode) kernel: one lane group per batch row, forward coding.
// ---------------------------------------------------------------------------

// USE_SHARED_CDF is a compile-time flag rather than a runtime ``smem_cdf_cols
// > 0`` test: measured on RTX 2080 Ti, keeping the decision at runtime cost the
// inverse-CDF paths ~8% even when nothing was staged (a few extra address
// selects per symbol on a loop that only has a handful of instructions left).
template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, typename TORCH_TENSOR_TYPE,
          bool USE_ALIAS_SAMPLING_CDF = false, bool USE_INVERSED_CDF = false,
          bool USE_SHARED_CDF = false, size_t NUM_INTERLEAVES = 1>
__global__ void rans_warp_pop_indexed_kernel(
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> stream_accessor,
  torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> symbols_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> indexes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 2, torch::RestrictPtrTraits> cdfs_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> cdfs_sizes_accessor,
  const torch::PackedTensorAccessor32<TORCH_TENSOR_TYPE, 1, torch::RestrictPtrTraits> offsets_accessor,
  const int freq_precision, const bool bypass_coding, const int bypass_precision,
  const int inverse_cdf_precision, const int smem_cdf_cols)
{
  static_assert(NUM_INTERLEAVES <= 32 && (NUM_INTERLEAVES & (NUM_INTERLEAVES - 1)) == 0,
                "NUM_INTERLEAVES must be a power of two no greater than the warp size");
  constexpr unsigned N = (unsigned)NUM_INTERLEAVES;

  // --- A9: stage whole CDF rows in shared memory. Symbol lookup is a handful
  // of *divergent* loads - every lane asks for a different cdf/table entry -
  // which costs many L1 wavefronts per warp instruction; shared memory serves
  // the same random pattern with only bank conflicts, and takes the pressure
  // off the L1/TEX pipe (62.6%, the highest of all pipes before A5). Staged
  // before the early return below so that every thread reaches the
  // __syncthreads. A whole row is always staged (cdf + table), never a part.
  extern __shared__ char rans_smem_raw[];
  TORCH_TENSOR_TYPE* const scdf = reinterpret_cast<TORCH_TENSOR_TYPE*>(rans_smem_raw);
  if constexpr (USE_SHARED_CDF) {
    const int64_t n_cdf = cdfs_accessor.size(0);
    const int64_t total = n_cdf * (int64_t)smem_cdf_cols;
    for (int64_t k = (int64_t)threadIdx.x; k < total; k += (int64_t)blockDim.x) {
      scdf[k] = cdfs_accessor[k / smem_cdf_cols][k % smem_cdf_cols];
    }
    __syncthreads();
  }

  const unsigned groups_per_block = blockDim.x / N;
  const unsigned group_in_block = threadIdx.x / N;
  const unsigned lane = threadIdx.x % N;
  const unsigned lane_in_warp = threadIdx.x & 31u;
  const unsigned group_start_in_warp = lane_in_warp - lane;
  const unsigned group_mask = (N == 32u) ? 0xffffffffu : (((1u << N) - 1u) << group_start_in_warp);
  // bit mask of the group-local lanes [0, lane], shifted into the group's
  // position inside the warp (the group does not necessarily start at lane 0)
  const unsigned mask_le = group_mask & ((((1u << lane) - 1u) | (1u << lane)) << group_start_in_warp);

  const int64_t b = (int64_t)blockIdx.x * groups_per_block + group_in_block;
  if (b >= stream_accessor.size(0)) return;  // uniform per group

  const int64_t num_symbols = symbols_accessor.size(1);
  const int64_t num_cdfs = cdfs_accessor.size(0);
  const int p = freq_precision;

  const auto stream_length = stream_accessor[b][0];
  RANS_STATE_TYPE* state_ptr = reinterpret_cast<RANS_STATE_TYPE*>(stream_accessor[b].data() + 1);
  RANS_STREAM_TYPE* base = reinterpret_cast<RANS_STREAM_TYPE*>(state_ptr);
  // the cursor names the last unconsumed word; pops read at cursor and below
  RANS_STREAM_TYPE* cursor = base + stream_length / sizeof(RANS_STREAM_TYPE) - 1;

  RANS_STATE_TYPE state = state_ptr[lane];

  TORCH_TENSOR_TYPE* symbols_ptr = symbols_accessor[b].data();
  const TORCH_TENSOR_TYPE* indexes_ptr = indexes_accessor[b].data();

  const int64_t last_interleave = num_symbols - (int64_t)(N - 1);
  int64_t i = 0;
  for (; i < last_interleave; i += (int64_t)N) {
    // --- phase A: parallel symbol pops, lane L codes symbol i+L ---
    const int64_t s = i + lane;
    const auto index = indexes_ptr[s];
    const bool valid = (index >= 0 && index < num_cdfs);
    RANS_STATE_TYPE new_state = 0;
    TORCH_TENSOR_TYPE out_value = 0, offset_v = 0, max_value = 0;
    if (valid) {
      const auto cdf_size = cdfs_sizes_accessor[index];
      // with USE_SHARED_CDF the whole row (cdf + appended inverse-CDF table) is
      // staged, so both pointers come from shared memory
      const TORCH_TENSOR_TYPE* cdf_ptr;
      if constexpr (USE_SHARED_CDF) {
        cdf_ptr = scdf + (int64_t)index * smem_cdf_cols;
      } else {
        cdf_ptr = cdfs_accessor[index].data();
      }
      const TORCH_TENSOR_TYPE* const inv_cdf_ptr = cdf_ptr + cdf_size;
      offset_v = offsets_accessor[index];
      max_value = cdf_size - 2;
      const RANS_STATE_TYPE mask_p = (RANS_STATE_TYPE)((1ull << p) - 1);
      const TORCH_TENSOR_TYPE cum_freq = (TORCH_TENSOR_TYPE)(state & mask_p);
      TORCH_TENSOR_TYPE cum_freq_offset = cum_freq;
      const TORCH_TENSOR_TYPE cdf_idx = rans_warp_pop_lookup<TORCH_TENSOR_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF>(
        cdf_ptr, inv_cdf_ptr, cdf_size, cum_freq, p, inverse_cdf_precision, cum_freq_offset);
      const TORCH_TENSOR_TYPE freq = cdf_ptr[cdf_idx + 1] - cdf_ptr[cdf_idx];
      new_state = (state >> p) * (RANS_STATE_TYPE)freq + (RANS_STATE_TYPE)cum_freq_offset;
      out_value = cdf_idx;
    }
    const bool pred = valid && (new_state < RANS_STATE_LOWER_BOUND);
    const unsigned vote = __ballot_sync(group_mask, pred);
    const unsigned count = __popc(vote);
    if (valid) {
      // the state is always replaced by the advanced one; the renorm ballot
      // only decides whether an extra word is folded in (RANS_POP_STATE_RENORM).
      // Lane-ascending consumption: the k-th renorming lane (1-based rank)
      // reads the word k-1 below the cursor.
      state = pred ? ((new_state << RANS_STREAM_BITS) | (RANS_STATE_TYPE)cursor[1 - (int)__popc(vote & mask_le)])
                   : new_state;
    }
    cursor -= count;
    symbols_ptr[s] = valid ? (out_value + offset_v) : (TORCH_TENSOR_TYPE)0;

    // --- phase B: bypass raw-value runs, lane-ascending, one lane at a time
    // (run lengths are only discoverable while decoding, so the mirror of the
    // encoder's lane-descending runs is inherently sequential).
    //
    // A5 fast path: every lane already knows from phase A whether its own
    // symbol is the bypass sentinel, so the group ballots the sentinel mask
    // once instead of re-loading indexes[]/symbols[] once per lane. When no
    // lane is a sentinel the whole loop disappears; previously it cost N
    // iterations x (2 global loads + a shuffle) for every N symbols even when
    // nothing in the group needed a raw value. No __syncwarp is needed either:
    // phase B no longer reads symbols written by other lanes.
    if (bypass_coding) {
      unsigned remaining = __ballot_sync(group_mask, valid && (out_value == max_value));
      while (remaining) {
        // __ballot_sync indexes by lane-in-warp; convert back to group-local
        const unsigned l = (__ffs(remaining) - 1u) - group_start_in_warp;
        remaining &= remaining - 1u;  // clear the lowest set bit, lane-ascending
        unsigned consumed = 0;
        if (lane == l) {
          // lane == l means i + lane == i + l, so the lane's own phase A
          // offset_v / max_value are exactly the ones the old code reloaded
          TORCH_TENSOR_TYPE raw_val = 0;
          consumed = rans_warp_simulate_raw_pop<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE>(
              state, cursor, bypass_precision, raw_val);
          TORCH_TENSOR_TYPE value = raw_val >> 1;
          if (raw_val & 1) {
            value = -value - 1;
          } else {
            value += max_value;
          }
          symbols_ptr[i + l] = value + offset_v;
        }
        cursor -= __shfl_sync(group_mask, consumed, group_start_in_warp + l);
      }
    }
  }

  // --- final symbols: the last (num_symbols % N) ones, one at a time ---
  for (; i < num_symbols; i++) {
    const unsigned active_lane = (unsigned)(i % (int64_t)N);
    const bool is_active = (lane == active_lane);
    unsigned consumed = 0;
    if (is_active) {
      const auto index = indexes_ptr[i];
      if (index >= 0 && index < num_cdfs) {
        const auto cdf_size = cdfs_sizes_accessor[index];
        const TORCH_TENSOR_TYPE* cdf_ptr;
        if constexpr (USE_SHARED_CDF) {
          cdf_ptr = scdf + (int64_t)index * smem_cdf_cols;
        } else {
          cdf_ptr = cdfs_accessor[index].data();
        }
        const TORCH_TENSOR_TYPE* const inv_cdf_ptr = cdf_ptr + cdf_size;
        const auto offset_v = offsets_accessor[index];
        const auto max_value = cdf_size - 2;
        const RANS_STATE_TYPE mask_p = (RANS_STATE_TYPE)((1ull << p) - 1);
        const TORCH_TENSOR_TYPE cum_freq = (TORCH_TENSOR_TYPE)(state & mask_p);
        TORCH_TENSOR_TYPE cum_freq_offset = cum_freq;
        const TORCH_TENSOR_TYPE cdf_idx =
            rans_warp_pop_lookup<TORCH_TENSOR_TYPE, USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF>(
              cdf_ptr, inv_cdf_ptr, cdf_size, cum_freq, p, inverse_cdf_precision, cum_freq_offset);
        const TORCH_TENSOR_TYPE freq = cdf_ptr[cdf_idx + 1] - cdf_ptr[cdf_idx];
        const RANS_STATE_TYPE new_state =
            (state >> p) * (RANS_STATE_TYPE)freq + (RANS_STATE_TYPE)cum_freq_offset;
        // the state is always replaced by the advanced one; renorm only
        // decides whether an extra word is folded in (RANS_POP_STATE_RENORM)
        state = new_state;
        if (new_state < RANS_STATE_LOWER_BOUND) {
          state = (new_state << RANS_STREAM_BITS) | (RANS_STATE_TYPE)cursor[0];
          consumed = 1;
        }
        TORCH_TENSOR_TYPE value = cdf_idx;
        if (bypass_coding && value == max_value) {
          TORCH_TENSOR_TYPE raw_val = 0;
          consumed += rans_warp_simulate_raw_pop<RANS_STATE_TYPE, RANS_STREAM_TYPE, TORCH_TENSOR_TYPE>(
            state, cursor - consumed, bypass_precision, raw_val);
          value = raw_val >> 1;
          if (raw_val & 1) {
            value = -value - 1;
          } else {
            value += max_value;
          }
        }
        symbols_ptr[i] = value + offset_v;
      } else {
        symbols_ptr[i] = 0;
      }
    }
    cursor -= __shfl_sync(group_mask, consumed, group_start_in_warp + active_lane);
  }

  if (lane == 0) {
    stream_accessor[b][0] = (TORCH_TENSOR_TYPE)((cursor - base + 1) * (int64_t)sizeof(RANS_STREAM_TYPE));
  }
  state_ptr[lane] = state;
}

// ---------------------------------------------------------------------------
// Host-side launchers
// ---------------------------------------------------------------------------

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF = false,
          size_t NUM_INTERLEAVES = 1>
void rans_warp_push_indexed_cuda(
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
  TORCH_CHECK(freq_precision > 0 && freq_precision <= RANS_STATE_USED_BITS - RANS_STREAM_BITS,
              "freq_precision must be in (0, ", RANS_STATE_USED_BITS - RANS_STREAM_BITS,
              "] for the warp-level interleaved kernels");
  if (bypass_coding) {
    TORCH_CHECK(bypass_precision >= 1 && bypass_precision < freq_precision,
                "bypass_precision must be in [1, freq_precision)");
  }

  AT_DISPATCH_INTEGRAL_TYPES(indexes.scalar_type(), "rans_warp_push_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    constexpr int num_threads_per_block = WARP_NUM_THREADS_PER_BLOCK;
    constexpr int groups_per_block = num_threads_per_block / (int)NUM_INTERLEAVES;
    const int num_blocks = (int)((batch_size + groups_per_block - 1) / groups_per_block);

    rans_warp_push_indexed_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t, USE_ALIAS_SAMPLING_CDF,
                                  NUM_INTERLEAVES><<<num_blocks, num_threads_per_block>>>(
        stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
        cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
        (int)freq_precision, bypass_coding, (int)bypass_precision);
  });
}

template <typename RANS_STATE_TYPE, typename RANS_STREAM_TYPE, bool USE_ALIAS_SAMPLING_CDF = false,
          bool USE_INVERSED_CDF = false, size_t NUM_INTERLEAVES = 1>
torch::Tensor rans_warp_pop_indexed_cuda(
  torch::Tensor stream,
  const torch::Tensor& indexes,
  const torch::Tensor& cdfs,
  const torch::Tensor& cdfs_sizes,
  const torch::Tensor& offsets,
  int64_t freq_precision,
  bool bypass_coding,
  int64_t bypass_precision,
  int64_t inverse_cdf_precision = -1)
{
  TORCH_CHECK(freq_precision > 0 && freq_precision <= RANS_STATE_USED_BITS - RANS_STREAM_BITS,
              "freq_precision must be in (0, ", RANS_STATE_USED_BITS - RANS_STREAM_BITS,
              "] for the warp-level interleaved kernels");
  if (bypass_coding) {
    TORCH_CHECK(bypass_precision >= 1 && bypass_precision < freq_precision,
                "bypass_precision must be in [1, freq_precision)");
  }
  // inverse_cdf_precision <= 0 means "dense table", the historical behaviour:
  // the lookup is a single load with no linear walk.
  const int64_t inv_cdf_precision =
      (inverse_cdf_precision > 0) ? inverse_cdf_precision : freq_precision;
  if constexpr (USE_INVERSED_CDF) {
    TORCH_CHECK(inv_cdf_precision >= 1 && inv_cdf_precision <= freq_precision,
                "inverse_cdf_precision must be in [1, ", freq_precision, "]");
  }

  torch::Tensor symbols = torch::zeros_like(indexes);

  AT_DISPATCH_INTEGRAL_TYPES(indexes.scalar_type(), "rans_warp_pop_indexed_cuda", [&] {
    const int64_t batch_size = stream.size(0);
    // A6 tried to spread a small batch over more, smaller blocks; rejected, see
    // SPARSE_INVCDF_SUMMARY.md (A6 实测记录): warps == rows is fixed, so block
    // packing cannot raise occupancy, and it was at most +4% for tiny batches
    // while costing 33-62% on large ones.
    constexpr int num_threads_per_block = WARP_NUM_THREADS_PER_BLOCK;
    constexpr int groups_per_block = num_threads_per_block / (int)NUM_INTERLEAVES;
    const int num_blocks = (int)((batch_size + groups_per_block - 1) / groups_per_block);

    // A9: stage a whole CDF row (CDF + inverse-CDF table) in shared memory
    // when it fits the budget, otherwise don't stage at all.
    //
    // The budget is 1/4 of the per-SM shared memory, i.e. what four 256-thread
    // blocks may use while keeping occupancy at the current 4 blocks/SM (ncu
    // Occupancy section: Block Limit Shared Mem must stay >= 4).
    //
    // Only whole rows are worth staging: measured on RTX 2080 Ti (rows=2048,
    // 16K symbols/stream, bypass on), staging CDF-only while leaving the table
    // in global memory was 3-10% SLOWER for the inverse-CDF paths, while a
    // fully staged row was 3.4% faster for q=7 and nothing lost elsewhere.
    // Partial staging mixes two memories without removing the divergent global
    // loads, so the extra per-symbol address work is pure overhead there.
    // TORCH_ANS_WARP_SMEM_BUDGET overrides the budget (bytes per block). 0
    // disables A9 staging entirely - handy for A/B measuring the feature
    // without rebuilding.
    int64_t shared_budget_bytes = 16 * 1024;
    if (const char* env_budget = std::getenv("TORCH_ANS_WARP_SMEM_BUDGET")) {
      shared_budget_bytes = std::atoll(env_budget);
    }
    const int64_t n_cdf = cdfs.size(0);
    const int64_t row_width = cdfs.size(1);
    int smem_cdf_cols = 0;
    if (shared_budget_bytes > 0 &&
        n_cdf * row_width * (int64_t)sizeof(scalar_t) <= shared_budget_bytes) {
      smem_cdf_cols = (int)row_width;
    }
    const size_t smem_bytes =
        (size_t)smem_cdf_cols * (size_t)n_cdf * sizeof(scalar_t);

    // two instantiations so that the staged path costs nothing when unused
    if (smem_cdf_cols > 0) {
      rans_warp_pop_indexed_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t,
                                   USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, true,
                                   NUM_INTERLEAVES><<<num_blocks, num_threads_per_block, smem_bytes>>>(
          stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          (int)freq_precision, bypass_coding, (int)bypass_precision, (int)inv_cdf_precision,
          smem_cdf_cols);
    } else {
      rans_warp_pop_indexed_kernel<RANS_STATE_TYPE, RANS_STREAM_TYPE, scalar_t,
                                   USE_ALIAS_SAMPLING_CDF, USE_INVERSED_CDF, false,
                                   NUM_INTERLEAVES><<<num_blocks, num_threads_per_block>>>(
          stream.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          symbols.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          indexes.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          cdfs_sizes.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          offsets.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
          (int)freq_precision, bypass_coding, (int)bypass_precision, (int)inv_cdf_precision,
          0);
    }
  });

  return symbols;
}

}  // namespace rans_warp
