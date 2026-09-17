#!/usr/bin/env python3
"""Compare tANS and rANS coding throughput, at matched parameters.

tANS is a *table-driven* coder: `freq_precision` is also its table_log, so it
spends `3 * 2**freq_precision + 2 * cdf_row` int32 words per distribution on
tables and gets a single table access per symbol instead of rANS' division
(encode) or search (decode). rANS spends nothing on tables and can afford a much
larger `freq_precision`. Which one wins therefore depends on the parameters,
and this script measures it:

    python scripts/bench_tans_vs_rans.py
    python scripts/bench_tans_vs_rans.py --json benchmarks/tans_vs_rans_cpu.json
    python scripts/bench_tans_vs_rans.py --alphabet 64 --precision 10 12

What it reports per configuration:

* encode / decode throughput (MB/s of *symbols* coded, and the raw time);
* the coded size, as bits/symbol - tANS at a small `freq_precision` codes a
  coarser distribution, so "faster" can mean "worse compression" and the two
  numbers must be read together;
* the per-distribution state the impl carries (`table KB`), i.e. the memory
  tANS trades for its table lookups.

Configurations a family cannot code are skipped with the reason printed rather
than silently measured wrong: tANS needs every symbol (plus the bypass sentinel)
to own at least one of its `2**freq_precision` slots, and `tans8` stops at
`freq_precision=7` (its state has to hold `[2**L, 2**(L+1))`).

Like `scripts/bench_interleaves_cpu.py` this measures the high-level API, so the
per-call overhead (tensor conversion, stream <-> bytes, the byte-string merge)
is included; `--threads 1` (the default) isolates the out-of-order effect of
interleaving from ATen-level parallelism.
"""
import argparse
import json
import platform
import time

import torch

from torch_ans.utils import TorchANSInterface

#: family -> (state, stream) description, freq_precision cap, interleave factors
FAMILIES = {
    "tans32": ("uint32 state / uint16 stream", 15, (1, 2, 4, 8)),
    "tans16": ("uint16 state / uint8 stream", 15, (1, 2, 4, 8)),
    "tans8": ("uint8 state / uint8 stream", 7, (1, 2, 4, 8)),
    "rans64": ("uint64 state / uint32 stream", 31, (1, 2, 4, 8)),
    "rans32": ("uint32 state / uint8 stream", 23, (1, 2, 4, 8)),
    "rans32_16": ("uint32 state / uint16 stream", 15, (1, 2, 4, 8)),
}

DEFAULT_PRECISIONS = (7, 10, 12, 15)
DEFAULT_ALPHABETS = (64, 256)


def build_data(num_symbols, num_distributions, alphabet_size, precision, seed=0,
               from_pmf=True):
    """Symbols/indexes/freqs whose cdf quantizes to exactly `2**precision`.

    Largest-remainder quantization: every symbol keeps at least one unit and each
    row sums to `2**precision`, which is exactly what `tans_build_tables` demands
    (the cdf differences *are* the normalized counter). A plain
    `pmf_to_quantized_cdf` of random freqs can give a symbol 0 units, which tANS
    rejects - and it would also make the comparison unfair, since the two
    families would not be coding the same distribution.

    The symbols are drawn *from* those distributions (unless `from_pmf` is
    False), because the run also reports bits/symbol: coding uniform symbols with
    a random pmf measures the cross-entropy of two unrelated distributions, which
    grows with `freq_precision` and says nothing about the coder.

    Returns (symbols, indexes, freqs, num_freqs, offsets, entropy) with
    `entropy` the mean per-distribution entropy in bits - the number a coder
    coding these symbols with these distributions cannot go below.
    """
    generator = torch.Generator().manual_seed(seed)
    indexes = torch.randint(0, num_distributions, (num_symbols,), generator=generator)
    freqs = torch.rand(num_distributions, alphabet_size, generator=generator) + 1e-3
    target = 1 << precision
    scaled = freqs / freqs.sum(-1, keepdim=True) * target
    base = scaled.floor().int().clamp(min=1)
    frac = scaled - torch.floor(scaled)
    for i in range(num_distributions):
        diff = target - int(base[i].sum())
        if diff == 0:
            continue
        if diff > 0:
            # the usual largest-remainder step: hand the leftover units out
            _, order = torch.sort(frac[i], descending=True)
            base[i, order[:diff]] += 1
        else:
            # `clamp(min=1)` can push the row *above* the target when the
            # alphabet is as large as the table (the fp=7 / alphabet=127 case),
            # so the excess has to come back off - from the smallest fractions,
            # never taking a symbol's last unit.
            _, order = torch.sort(frac[i])
            for j in order:
                if diff == 0:
                    break
                if int(base[i, j]) > 1:
                    base[i, j] -= 1
                    diff += 1
    assert base.min() > 0 and bool((base.sum(-1) == target).all())

    probs = base.float() / target
    entropy = float(-(probs * (probs + 1e-12).log2()).sum(-1).mean())

    if from_pmf:
        # one multinomial per distribution: the symbols a row sees are the ones
        # its own cdf describes
        counts = torch.bincount(indexes, minlength=num_distributions)
        symbols = torch.empty(num_symbols, dtype=torch.int64)
        for i in range(num_distributions):
            n = int(counts[i])
            if n == 0:
                continue
            symbols[indexes == i] = torch.multinomial(
                probs[i], n, replacement=True, generator=generator)
        symbols = symbols.to(torch.int32)
    else:
        symbols = torch.randint(0, alphabet_size, (num_symbols,),
                                generator=generator).to(torch.int32)

    num_freqs = torch.full((num_distributions,), alphabet_size, dtype=torch.int32)
    offsets = torch.zeros(num_distributions, dtype=torch.int32)
    return symbols, indexes, base, num_freqs, offsets, entropy


def config_is_codable(family, alphabet, precision, bypass):
    """(ok, reason) - whether `family` can code this parameter set."""
    _, cap, _ = FAMILIES[family]
    if precision > cap:
        return False, f"freq_precision {precision} > the {cap} cap of {family}"
    if family.startswith("tans"):
        # the spread has to give every symbol (plus the bypass sentinel) a slot
        needed = alphabet + (1 if bypass else 0)
        if needed > (1 << precision):
            return False, (f"alphabet {alphabet} (+sentinel) needs "
                           f"{needed} > 2**{precision} slots")
    return True, ""


def bench_one(impl, num_interleaves, symbols, indexes, freqs, num_freqs, offsets,
              num_parallel_states, reps, precision, bypass):
    coder = TorchANSInterface(
        impl=impl,
        bypass_coding=bypass,
        bypass_precision=4,
        freq_precision=precision,
        num_parallel_states=num_parallel_states,
        num_interleaves=num_interleaves,
        mode="encdec",
        dtype=torch.int32,
        device="cpu",
    )
    coder.init_params(freqs, num_freqs, offsets)

    # warmup + correctness check: a configuration that does not round-trip is
    # worthless as a throughput number
    encoded = coder.encode_with_indexes(symbols, indexes)
    decoded = coder.decode_with_indexes(encoded, indexes)
    if not torch.equal(decoded.to(symbols.dtype), symbols):
        raise AssertionError(f"roundtrip mismatch for {impl} x{num_interleaves} "
                             f"(precision={precision})")

    times_enc, times_dec = [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        encoded = coder.encode_with_indexes(symbols, indexes)
        t1 = time.perf_counter()
        coder.decode_with_indexes(encoded, indexes)
        t2 = time.perf_counter()
        times_enc.append(t1 - t0)
        times_dec.append(t2 - t1)
    # the per-distribution state the impl carries: for tANS this *is* the table
    table_bytes = coder.cdfs.numel() * coder.cdfs.element_size()
    return {
        "enc_s": min(times_enc),
        "dec_s": min(times_dec),
        "coded_bytes": len(encoded),
        "table_bytes": table_bytes,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", type=int, default=1 << 20)
    p.add_argument("--dists", type=int, default=256)
    p.add_argument("--alphabet", type=int, nargs="+", default=list(DEFAULT_ALPHABETS))
    p.add_argument("--precision", type=int, nargs="+", default=list(DEFAULT_PRECISIONS))
    p.add_argument("--interleaves", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--impl", nargs="+", default=list(FAMILIES))
    p.add_argument("--parallel-states", type=int, default=8)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads; 1 isolates the OoO effect of "
                        "interleaving from ATen-level parallelism")
    p.add_argument("--no-bypass", action="store_true")
    p.add_argument("--uniform-symbols", action="store_true",
                   help="draw the symbols uniformly instead of from the "
                        "distributions (then bits/symbol is not meaningful)")
    p.add_argument("--json", default=None, help="write the raw results here")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    bypass = not args.no_bypass
    num_symbols = args.symbols

    print(f"tANS vs rANS (CPU): {num_symbols} symbols, {args.dists} distributions, "
          f"parallel_states={args.parallel_states}, torch_threads={args.threads}, "
          f"bypass={bypass}, reps={args.reps} (best of)")
    print(f"machine: {platform.machine()} / torch {torch.__version__}, "
          f"python {platform.python_version()}")
    print()

    results = []
    skipped = []
    for alphabet in args.alphabet:
        for precision in args.precision:
            if alphabet + (1 if bypass else 0) > (1 << precision):
                # No tANS family can code this (every symbol needs a slot), and a
                # rANS-only row would not be a like-for-like comparison - so the
                # whole (alphabet, precision) pair is left out.
                skipped.append(("*", 0, alphabet, precision,
                                f"alphabet {alphabet} (+sentinel) does not fit "
                                f"2**{precision} slots"))
                continue
            symbols, indexes, freqs, num_freqs, offsets, entropy = build_data(
                num_symbols, args.dists, alphabet, precision,
                from_pmf=not args.uniform_symbols)
            raw_mb = num_symbols / 1e6
            for impl in args.impl:
                if impl not in FAMILIES:
                    raise SystemExit(f"unknown impl {impl}")
                _, _, interleaves = FAMILIES[impl]
                for num_interleaves in args.interleaves:
                    if num_interleaves not in interleaves:
                        continue
                    codable, reason = config_is_codable(impl, alphabet, precision, bypass)
                    if not codable:
                        skipped.append((impl, num_interleaves, alphabet, precision, reason))
                        continue
                    measured = bench_one(
                        impl, num_interleaves, symbols, indexes, freqs, num_freqs,
                        offsets, args.parallel_states, args.reps, precision, bypass)
                    results.append({
                        "impl": impl,
                        "interleaves": num_interleaves,
                        "alphabet": alphabet,
                        "freq_precision": precision,
                        "bypass": bypass,
                        "symbols": num_symbols,
                        "distributions": args.dists,
                        "entropy": entropy,
                        **measured,
                    })

    header = (f"{'impl':<10} {'ilv':>3} {'fp':>3} {'alpha':>5} {'enc MB/s':>10} "
              f"{'dec MB/s':>10} {'enc us':>9} {'dec us':>9} {'bits/sym':>9} "
              f"{'H':>6} {'over':>6} {'table KB':>9}")
    print(header)
    print("-" * len(header))
    by_key = {}
    for row in results:
        enc_mbs = raw_mb / row["enc_s"]
        dec_mbs = raw_mb / row["dec_s"]
        bits_per_symbol = 8.0 * row["coded_bytes"] / num_symbols
        overhead = bits_per_symbol - row["entropy"]
        print(f"{row['impl']:<10} {row['interleaves']:>3} {row['freq_precision']:>3} "
              f"{row['alphabet']:>5} {enc_mbs:>10.1f} {dec_mbs:>10.1f} "
              f"{row['enc_s'] * 1e6:>9.1f} {row['dec_s'] * 1e6:>9.1f} "
              f"{bits_per_symbol:>9.3f} {row['entropy']:>6.3f} {overhead:>6.3f} "
              f"{row['table_bytes'] / 1024.0:>9.1f}")
        by_key[(row["impl"], row["interleaves"], row["alphabet"],
                row["freq_precision"])] = (enc_mbs, dec_mbs)

    if skipped:
        print(f"\nskipped {len(skipped)} configuration(s) a family cannot code:")
        for impl, ilv, alphabet, precision, reason in skipped[:20]:
            print(f"  {impl} x{ilv} alphabet={alphabet} fp={precision}: {reason}")

    # 1. the interleave gain, per family and precision (same family, same data)
    print("\nInterleave gain vs 1 lane (same impl / alphabet / precision):")
    for impl in args.impl:
        for alphabet in args.alphabet:
            for precision in args.precision:
                base = by_key.get((impl, 1, alphabet, precision))
                if base is None:
                    continue
                gains = []
                for num_interleaves in args.interleaves:
                    if num_interleaves == 1:
                        continue
                    cur = by_key.get((impl, num_interleaves, alphabet, precision))
                    if cur is None:
                        continue
                    gains.append(f"x{num_interleaves}: {cur[0] / base[0]:.2f}/"
                                 f"{cur[1] / base[1]:.2f}")
                if gains:
                    print(f"  {impl:<10} alphabet={alphabet:<4} fp={precision:<3} "
                          f"enc/dec {', '.join(gains)}")

    # 2. tANS against rANS at the same precision (the fair comparison: same
    #    quantization, same alphabet - only the coder differs)
    print("\ntANS vs rANS at the same freq_precision (encode / decode speedup):")
    rans_families = [f for f in args.impl if f.startswith("rans")]
    for alphabet in args.alphabet:
        for precision in args.precision:
            for num_interleaves in args.interleaves:
                tans_rows = [(f, by_key[(f, num_interleaves, alphabet, precision)])
                             for f in args.impl
                             if f.startswith("tans")
                             and (f, num_interleaves, alphabet, precision) in by_key]
                rans_rows = [(f, by_key[(f, num_interleaves, alphabet, precision)])
                             for f in rans_families
                             if (f, num_interleaves, alphabet, precision) in by_key]
                if not tans_rows or not rans_rows:
                    continue
                best_rans_e = max(v[0] for _, v in rans_rows)
                best_rans_d = max(v[1] for _, v in rans_rows)
                best_rans_name = max(rans_rows, key=lambda kv: kv[1][1])[0]
                for name, (enc, dec) in tans_rows:
                    print(f"  alphabet={alphabet:<4} fp={precision:<3} x{num_interleaves} "
                          f"{name:<8} vs fastest rANS ({best_rans_name}): "
                          f"enc {enc / best_rans_e:.2f}x, dec {dec / best_rans_d:.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "machine": {
                    "machine": platform.machine(),
                    "torch": torch.__version__,
                    "python": platform.python_version(),
                    "threads": args.threads,
                },
                "config": {
                    "symbols": num_symbols,
                    "distributions": args.dists,
                    "parallel_states": args.parallel_states,
                    "reps": args.reps,
                    "bypass": bypass,
                },
                "results": results,
                "skipped": [
                    {"impl": i, "interleaves": k, "alphabet": a,
                     "freq_precision": fp, "reason": r}
                    for i, k, a, fp, r in skipped
                ],
            }, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
