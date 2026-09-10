#!/usr/bin/env python3
"""Benchmark CPU encode/decode throughput for different num_interleaves.

Interleaved rANS keeps several independent rANS states per parallel stream so
that the CPU's out-of-order engine can overlap their (mostly independent)
renormalize/push/pop chains. This script measures the effect on this machine.

Run it twice to also see the symbol-lookup choice, since the two lookups respond
differently to interleaving (the inverse-CDF table scales less but its absolute
throughput is higher, so it wins at every num_interleaves):

    python scripts/bench_interleaves_cpu.py [--symbols 1048576] [--reps 20]
    python scripts/bench_interleaves_cpu.py --inverse-cdf auto

Note the per-call overhead (tensor dtype/device conversion, stream <-> bytes, the
merge/split of the byte strings) is included, so the reported interleave gain is
diluted; measure the raw ops for the kernel-only numbers.
"""
import argparse
import time

import torch

from torch_ans.utils import TorchANSInterface

SUPPORTED = {
    "rans64": [1, 2, 4, 8],
    "rans32": [1, 2, 4, 8],
    "rans32_16": [1, 2, 4, 8, 32],
}


def build_data(num_symbols, num_distributions, alphabet_size, seed=0, target=1 << 15):
    g = torch.Generator().manual_seed(seed)
    symbols = torch.randint(0, alphabet_size, (num_symbols,), generator=g)
    indexes = torch.randint(0, num_distributions, (num_symbols,), generator=g)
    freqs = torch.rand(num_distributions, alphabet_size, generator=g) + 1e-3
    # largest-remainder quantization: guarantees every symbol freq >= 1 and
    # each row sums exactly to 2**15
    target = target
    scaled = freqs / freqs.sum(-1, keepdim=True) * target
    base = scaled.floor().int()
    base = base.clamp(min=1)  # every symbol gets at least one count
    residual = target - base.sum(-1)
    # give the residual counts to the symbols with the largest fractions
    frac = scaled - torch.floor(scaled)
    for i in range(num_distributions):
        r = int(residual[i])
        if r == 0:
            continue
        _, order = torch.sort(frac[i], descending=True)
        base[i, order[:r]] += 1
    freqs = base
    assert freqs.min() > 0 and bool((freqs.sum(-1) == target).all())
    num_freqs = torch.full((num_distributions,), alphabet_size, dtype=torch.int32)
    offsets = torch.zeros(num_distributions, dtype=torch.int32)
    return symbols, indexes, freqs, num_freqs, offsets


def bench_one(impl, num_interleaves, symbols, indexes, freqs, num_freqs,
              offsets, num_parallel_states, reps, bypass=True, freq_precision=16,
              inverse_cdf_precision=None):
    coder = TorchANSInterface(
        impl=impl,
        bypass_coding=bypass,
        bypass_precision=4,
        freq_precision=freq_precision,
        inverse_cdf_precision=inverse_cdf_precision,
        num_parallel_states=num_parallel_states,
        num_interleaves=num_interleaves,
        mode="encdec",
        dtype=torch.int32,
        device="cpu",
    )
    coder.init_params(freqs, num_freqs, offsets)

    # warmup + correctness check
    encoded = coder.encode_with_indexes(symbols, indexes)
    decoded = coder.decode_with_indexes(encoded, indexes)
    assert torch.equal(decoded.to(symbols.dtype), symbols), \
        f"roundtrip mismatch for {impl} x{num_interleaves}"

    times_enc, times_dec = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        encoded = coder.encode_with_indexes(symbols, indexes)
        t1 = time.perf_counter()
        decoded = coder.decode_with_indexes(encoded, indexes)
        t2 = time.perf_counter()
        times_enc.append(t1 - t0)
        times_dec.append(t2 - t1)
    return min(times_enc), min(times_dec), len(encoded)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbols", type=int, default=1 << 20)
    p.add_argument("--dists", type=int, default=256)
    p.add_argument("--alphabet", type=int, default=256)
    p.add_argument("--parallel-states", type=int, default=8)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads; 1 isolates the OoO effect "
                        "from ATen-level parallelism")
    p.add_argument("--no-bypass", action="store_true",
                   help="disable bypass coding")
    p.add_argument("--precision", type=int, default=15)
    p.add_argument("--inverse-cdf", default=None,
                   help="inverse CDF table precision: int, 'auto', or omitted "
                        "to use the divided-search decode (default)")
    args = p.parse_args()
    inverse_cdf = None if args.inverse_cdf is None else (
        args.inverse_cdf if args.inverse_cdf == "auto"
        else int(args.inverse_cdf))

    torch.set_num_threads(args.threads)
    symbols, indexes, freqs, num_freqs, offsets = build_data(
        args.symbols, args.dists, args.alphabet, target=1 << args.precision)
    raw_mb = args.symbols / 1e6  # one byte per symbol

    print(f"CPU benchmark: {args.symbols} symbols, alphabet={args.alphabet}, "
          f"dists={args.dists}, parallel_states={args.parallel_states}, "
          f"torch_threads={args.threads}, reps={args.reps} (best of)")
    print(f"{'impl':<10} {'interleaves':>11} {'enc MB/s':>10} {'dec MB/s':>10} "
          f"{'enc us':>9} {'dec us':>9} {'size':>10}")
    results = {}
    for impl, interleaves in SUPPORTED.items():
        for k in interleaves:
            te, td, size = bench_one(
                impl, k, symbols, indexes, freqs, num_freqs, offsets,
                args.parallel_states, args.reps,
                bypass=not args.no_bypass, freq_precision=args.precision,
                inverse_cdf_precision=inverse_cdf)
            enc_mbs = raw_mb / te
            dec_mbs = raw_mb / td
            results[(impl, k)] = (enc_mbs, dec_mbs)
            print(f"{impl:<10} {k:>11} {enc_mbs:>10.1f} {dec_mbs:>10.1f} "
                  f"{te * 1e6:>9.1f} {td * 1e6:>9.1f} {size:>10}")

    print("\nSpeedup vs num_interleaves=1 (same impl):")
    for impl, interleaves in SUPPORTED.items():
        base = results.get((impl, 1))
        if base is None:
            continue
        for k in interleaves:
            if k == 1:
                continue
            e, d = results[(impl, k)]
            print(f"  {impl} x{k}: encode {e / base[0]:.2f}x, "
                  f"decode {d / base[1]:.2f}x")


if __name__ == "__main__":
    main()
