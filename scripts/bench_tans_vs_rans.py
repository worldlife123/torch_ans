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
    python scripts/bench_tans_vs_rans.py --device cuda \
        --json benchmarks/tans_vs_rans_cuda.json
    # the interleaved CUDA kernels (the warp paths) against each other, over a
    # batch sweep: the numbers behind benchmarks/tans_interleaved_cuda.json
    python scripts/bench_tans_vs_rans.py --device cuda --interleaves 1 2 4 8 32 \
        --parallel-states 8 64 512 4096 \
        --json benchmarks/tans_interleaved_cuda.json

What it reports per configuration:

* encode / decode throughput (MB/s of *symbols* coded, and the raw time);
* the coded size, as bits/symbol - tANS at a small `freq_precision` codes a
  coarser distribution, so "faster" can mean "worse compression" and the two
  numbers must be read together;
* the per-distribution state the impl carries (`table KB`), i.e. the memory
  tANS trades for its table lookups;
* whether the tANS kernels staged that table into shared memory (`smem`), see
  below - the `--smem-budget` / `--block-threads` knobs are what makes the
  staging an A/B instead of a compile-time constant.

Configurations a family cannot code are skipped with the reason printed rather
than silently measured wrong: tANS needs every symbol (plus the bypass sentinel)
to own at least one of its `2**freq_precision` slots, and `tans8` stops at
`freq_precision=7` (its state has to hold `[2**L, 2**(L+1))`).

On CUDA (`--device cuda`) the same matrix runs on the device kernels, with the
differences the two implementations have there:

* tANS codes a row with **one lane group** per stream for `--interleaves 2/4/8/32`
  (the warp-level kernels in `torch_ans/tans_warp_cuda.cuh`) and with one thread
  per row at `--interleaves 1`, while rANS interleaving above one lane is the
  rans32_16 warp path. Both write the same payload as their CPU counterpart, so
  the two devices are directly comparable - which is what the "interleave gain"
  and the "vs the fastest rANS" sections below report;
* the table staging is decided per launch
  (`num_dists * row_words * 4 <= TORCH_ANS_TANS_SMEM_BUDGET`, 16 KiB by default,
  see torch_ans/tans_cuda.cu), so a small `--dists` and a small `--precision`
  are what turn it on: `--smem-budget 0` disables it and `--smem-budget` above
  the table size forces it, and `--block-threads` sets the launch block size
  (one thread codes one stream row, so it also decides how many blocks there
  are). The interleaved kernels derive their block size from the batch and the
  lane count instead (`TORCH_ANS_TANS_WARP_BLOCK_THREADS` overrides it), which is
  what `--warp-block-threads` sweeps;
* `--parallel-states` takes several batch sizes: the interleaved kernels trade
  rows per block against lanes per row, so the batch size (the number of
  independent stream rows, i.e. `num_parallel_states` of the high-level API) is a
  first-class dimension here rather than a constant;
* every timed call synchronizes (`torch.cuda.synchronize`), because the
  byte-string conversion of the high-level API already copies the stream back to
  the host and the number has to include that.

Like `scripts/bench_interleaves_cpu.py` this measures the high-level API, so the
per-call overhead (tensor conversion, stream <-> bytes, the byte-string merge)
is included; `--threads 1` (the default) isolates the out-of-order effect of
interleaving from ATen-level parallelism.

NOTE: run as `python scripts/bench_tans_vs_rans.py`, the script directory becomes
`sys.path[0]` and `torch_ans` is imported from site-packages - i.e. from the
*installed* package, which silently shadows the working tree when it is older.
Prefix `PYTHONPATH=$PWD` to measure the sources (that is how the committed
`benchmarks/*.json` were produced).
"""
import argparse
import json
import os
import platform
import time
from itertools import product

import torch

from torch_ans.utils import TorchANSInterface

#: family -> (state, stream) description, freq_precision cap, interleave factors.
#: 32-way is the full-warp group of the CUDA warp kernels, and both the kernel
#: families that have one - rans32_16 and the three tANS families - implement it
#: on the CPU as well, so it is listed for them here.
FAMILIES = {
    "tans32": ("uint32 state / uint16 stream", 15, (1, 2, 4, 8, 32)),
    "tans16": ("uint16 state / uint8 stream", 15, (1, 2, 4, 8, 32)),
    "tans8": ("uint8 state / uint8 stream", 7, (1, 2, 4, 8, 32)),
    "rans64": ("uint64 state / uint32 stream", 31, (1, 2, 4, 8)),
    "rans32": ("uint32 state / uint8 stream", 23, (1, 2, 4, 8)),
    "rans32_16": ("uint32 state / uint16 stream", 15, (1, 2, 4, 8, 32)),
}

DEFAULT_PRECISIONS = (7, 10, 12, 15)
DEFAULT_ALPHABETS = (64, 256)

#: Mirrors TANS_SMEM_BUDGET_DEFAULT in torch_ans/tans_cuda.cu: the default the
#: kernel uses when TORCH_ANS_TANS_SMEM_BUDGET is unset, needed here to report
#: whether a table would be staged.
DEFAULT_SMEM_BUDGET = 16 * 1024

#: Substrings that mean "this (family, device, interleave) combination is not
#: implemented" rather than "the measurement failed". The matrix deliberately
#: spans combinations that exist on one device only, so those are reported as
#: skipped instead of aborting the run - but anything else propagates.
UNSUPPORTED_MARKERS = (
    "Unsupported num_interleaves",
    "only supported for the rans32_16",
    "not compiled with GPU support",
    "not available for impl",
)


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


def config_is_codable(family, alphabet, precision, bypass, device="cpu",
                      num_interleaves=1):
    """(ok, reason) - whether `family` can code this parameter set on `device`."""
    _, cap, interleaves = FAMILIES[family]
    if precision > cap:
        return False, f"freq_precision {precision} > the {cap} cap of {family}"
    if num_interleaves not in interleaves:
        return False, (f"num_interleaves {num_interleaves} is not a {family} "
                       f"configuration (supported: {interleaves})")
    if device != "cpu":
        # The interleaved CUDA kernels are the warp paths: every tANS family
        # (tans_warp_cuda.cuh) and rans32_16 (rans_warp_cuda.cuh). This is a
        # second line of defence - the interface raises first - and it keeps the
        # run from paying for a build that could not have the operators.
        if not family.startswith("tans") and num_interleaves > 1 and family != "rans32_16":
            return False, (f"CUDA interleaved rANS is only supported for "
                           f"rans32_16, not {family}")
    if family.startswith("tans"):
        # the spread has to give every symbol (plus the bypass sentinel) a slot
        needed = alphabet + (1 if bypass else 0)
        if needed > (1 << precision):
            return False, (f"alphabet {alphabet} (+sentinel) needs "
                           f"{needed} > 2**{precision} slots")
    return True, ""


def _is_unsupported(exc):
    """Whether `exc` reports an unimplemented configuration (see the markers)."""
    message = str(exc)
    return any(marker in message for marker in UNSUPPORTED_MARKERS)


def _smem_budget():
    """The staging budget the kernels will read (see DEFAULT_SMEM_BUDGET)."""
    return int(os.environ.get("TORCH_ANS_TANS_SMEM_BUDGET", DEFAULT_SMEM_BUDGET))


def bench_one(impl, num_interleaves, symbols, indexes, freqs, num_freqs, offsets,
              num_parallel_states, reps, precision, bypass, device="cpu"):
    """Times one configuration; (result, reason) with result None when skipped.

    A configuration the implementation does not support is reported rather than
    raised (`_is_unsupported`): the matrix spans combinations that only exist on
    one device. Anything else - in particular a round-trip mismatch - raises.
    """

    def on_device(tensor):
        return tensor if device == "cpu" else tensor.to(device)

    def synchronize():
        # Every timed call is a high-level encode/decode, which already copies
        # the stream to the host for the byte-string conversion; the sync is
        # what makes the wall-clock number cover the kernel as well.
        if device != "cpu":
            torch.cuda.synchronize()

    symbols, indexes = on_device(symbols), on_device(indexes)
    try:
        coder = TorchANSInterface(
            impl=impl,
            bypass_coding=bypass,
            bypass_precision=4,
            freq_precision=precision,
            num_parallel_states=num_parallel_states,
            num_interleaves=num_interleaves,
            mode="encdec",
            dtype=torch.int32,
            device=device,
        )
        coder.init_params(on_device(freqs), on_device(num_freqs),
                          on_device(offsets))
        # warmup + correctness check: a configuration that does not round-trip is
        # worthless as a throughput number
        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)
    except Exception as exc:  # noqa: BLE001 - classified right here
        if _is_unsupported(exc):
            return None, f"{type(exc).__name__}: {exc}"
        raise
    if not torch.equal(decoded.to(symbols.dtype), symbols):
        raise AssertionError(f"roundtrip mismatch for {impl} x{num_interleaves} "
                             f"(precision={precision}, device={device})")

    times_enc, times_dec = [], []
    for _ in range(reps):
        synchronize()
        t0 = time.perf_counter()
        encoded = coder.encode_with_indexes(symbols, indexes)
        synchronize()
        t1 = time.perf_counter()
        coder.decode_with_indexes(encoded, indexes)
        synchronize()
        t2 = time.perf_counter()
        times_enc.append(t1 - t0)
        times_dec.append(t2 - t1)
    # the per-distribution state the impl carries: for tANS this *is* the table
    # the kernels look up, and the staging decision is the whole-table test
    # tans_cuda.cu::tans_table_staging performs (`None` for the families that
    # have no table to stage).
    table_bytes = coder.cdfs.numel() * coder.cdfs.element_size()
    staged = None
    if impl.startswith("tans"):
        budget = _smem_budget()
        staged = bool(budget > 0 and table_bytes <= budget)
    return {
        "enc_s": min(times_enc),
        "dec_s": min(times_dec),
        "coded_bytes": len(encoded),
        "table_bytes": table_bytes,
        "staged": staged,
    }, ""


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbols", type=int, default=1 << 20)
    p.add_argument("--dists", type=int, default=256)
    p.add_argument("--alphabet", type=int, nargs="+", default=list(DEFAULT_ALPHABETS))
    p.add_argument("--precision", type=int, nargs="+", default=list(DEFAULT_PRECISIONS))
    p.add_argument("--interleaves", type=int, nargs="+", default=[1, 2, 4, 8, 32])
    p.add_argument("--impl", nargs="+", default=list(FAMILIES))
    p.add_argument("--parallel-states", type=int, nargs="+", default=[8],
                   help="batch sizes (stream rows) to measure. Several values sweep "
                        "the whole matrix at each of them, which is what the `ps` "
                        "column of the table shows: the interleaved kernels trade "
                        "rows per block against lanes per row, so this is the "
                        "dimension that decides whether a device is filled.")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--threads", type=int, default=1,
                   help="torch intra-op threads; 1 isolates the OoO effect of "
                        "interleaving from ATen-level parallelism")
    p.add_argument("--device", default="cpu", choices=("cpu", "cuda"),
                   help="where to code; interleaving above one lane is the warp "
                        "path on the device (every tANS family, and rans32_16)")
    p.add_argument("--no-bypass", action="store_true")
    p.add_argument("--uniform-symbols", action="store_true",
                   help="draw the symbols uniformly instead of from the "
                        "distributions (then bits/symbol is not meaningful)")
    p.add_argument("--smem-budget", type=int, default=None,
                   help="TORCH_ANS_TANS_SMEM_BUDGET for the tANS CUDA kernels "
                        "(bytes per block; 0 disables the table staging, a value "
                        "above the table size forces it). Unset leaves the "
                        "kernel default (16 KiB) in place.")
    p.add_argument("--block-threads", type=int, nargs="+", default=[None],
                   help="TORCH_ANS_TANS_BLOCK_THREADS for the single-lane tANS "
                        "CUDA kernels (threads per block; each thread codes one "
                        "stream row, so this also sets the number of blocks). "
                        "Several values sweep the whole matrix at each of them, "
                        "which is what the `blk` column of the table shows. "
                        "Omitting the flag keeps the compiled-in default.")
    p.add_argument("--warp-block-threads", type=int, nargs="+", default=[None],
                   help="TORCH_ANS_TANS_WARP_BLOCK_THREADS for the interleaved "
                        "tANS CUDA kernels (threads per block; a block holds "
                        "blockDim.x / num_interleaves lane groups, so the batch "
                        "and this value decide how many blocks there are). The "
                        "kernel derives it from the batch when unset; the `wblk` "
                        "column shows what was requested.")
    p.add_argument("--json", default=None, help="write the raw results here")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    bypass = not args.no_bypass
    num_symbols = args.symbols

    # The tANS CUDA knobs are environment variables because that is where the
    # kernels read them (per launch, so an A/B does not need a rebuild).
    if args.smem_budget is not None:
        os.environ["TORCH_ANS_TANS_SMEM_BUDGET"] = str(args.smem_budget)

    device_name = torch.cuda.get_device_name() if args.device != "cpu" else "CPU"
    print(f"tANS vs rANS ({args.device}): {num_symbols} symbols, "
          f"{args.dists} distributions, parallel_states={args.parallel_states}, "
          f"torch_threads={args.threads}, bypass={bypass}, reps={args.reps} "
          f"(best of)")
    print(f"machine: {platform.machine()} / {device_name} / torch "
          f"{torch.__version__}, python {platform.python_version()}")
    if args.device != "cpu":
        print(f"tANS kernel knobs: smem budget {_smem_budget()} bytes/block "
              f"(TORCH_ANS_TANS_SMEM_BUDGET), block threads "
              f"{args.block_threads} (TORCH_ANS_TANS_BLOCK_THREADS), warp block "
              f"threads {args.warp_block_threads} "
              f"(TORCH_ANS_TANS_WARP_BLOCK_THREADS); None = the compiled-in "
              f"default")
    print()

    results = []
    skipped = []
    skipped_seen = set()
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
            # A sweep is for the *launch* knobs only: neither the block size nor
            # the batch changes what a stream looks like (the streams are compared
            # below through their round-trip), so repeating the matrix at each of
            # them is a pure A/B of the launch shape.
            for num_parallel_states, block_threads, warp_block_threads in product(
                    args.parallel_states, args.block_threads,
                    args.warp_block_threads):
                if block_threads is None:
                    os.environ.pop("TORCH_ANS_TANS_BLOCK_THREADS", None)
                else:
                    os.environ["TORCH_ANS_TANS_BLOCK_THREADS"] = str(block_threads)
                if warp_block_threads is None:
                    os.environ.pop("TORCH_ANS_TANS_WARP_BLOCK_THREADS", None)
                else:
                    os.environ["TORCH_ANS_TANS_WARP_BLOCK_THREADS"] = str(
                        warp_block_threads)
                for impl in args.impl:
                    if impl not in FAMILIES:
                        raise SystemExit(f"unknown impl {impl}")
                    _, _, interleaves = FAMILIES[impl]
                    for num_interleaves in args.interleaves:
                        if num_interleaves not in interleaves:
                            continue
                        # skips are keyed without the launch knobs: a
                        # configuration the implementation does not have is not
                        # worth repeating at every batch/block size
                        config_key = (impl, num_interleaves, alphabet, precision)
                        codable, reason = config_is_codable(
                            impl, alphabet, precision, bypass, device=args.device,
                            num_interleaves=num_interleaves)
                        if not codable:
                            if config_key not in skipped_seen:
                                skipped_seen.add(config_key)
                                skipped.append((*config_key, reason))
                            continue
                        measured, reason = bench_one(
                            impl, num_interleaves, symbols, indexes, freqs,
                            num_freqs, offsets, num_parallel_states, args.reps,
                            precision, bypass, device=args.device)
                        if measured is None:
                            # The device rejected it (an operator set compiled
                            # without the family, say); report, do not crash.
                            if config_key not in skipped_seen:
                                skipped_seen.add(config_key)
                                skipped.append((*config_key, reason))
                            continue
                        results.append({
                            "impl": impl,
                            "interleaves": num_interleaves,
                            "alphabet": alphabet,
                            "freq_precision": precision,
                            "bypass": bypass,
                            "symbols": num_symbols,
                            "distributions": args.dists,
                            "parallel_states": num_parallel_states,
                            "entropy": entropy,
                            "device": args.device,
                            "block_threads": block_threads,
                            "warp_block_threads": warp_block_threads,
                            **measured,
                        })

    header = (f"{'impl':<10} {'ilv':>3} {'fp':>3} {'alpha':>5} {'ps':>5} "
              f"{'enc MB/s':>10} {'dec MB/s':>10} {'enc us':>9} {'dec us':>9} "
              f"{'bits/sym':>9} {'H':>6} {'over':>6} {'table KB':>9} {'smem':>5} "
              f"{'blk':>4} {'wblk':>5}")
    print(header)
    print("-" * len(header))
    # The summary sections at the bottom compare *the first* batch size and the
    # first launch knobs; a sweep shows up as repeated rows (and is what the `ps`
    # / `blk` / `wblk` columns are for).
    first_batch = args.parallel_states[0]
    first_block_threads = args.block_threads[0]
    first_warp_block_threads = args.warp_block_threads[0]
    by_key = {}
    for row in results:
        enc_mbs = raw_mb / row["enc_s"]
        dec_mbs = raw_mb / row["dec_s"]
        bits_per_symbol = 8.0 * row["coded_bytes"] / num_symbols
        overhead = bits_per_symbol - row["entropy"]
        staged = row["staged"]
        smem = "-" if staged is None else ("yes" if staged else "no")
        blk = "-" if row["block_threads"] is None else str(row["block_threads"])
        wblk = ("-" if row["warp_block_threads"] is None
                else str(row["warp_block_threads"]))
        print(f"{row['impl']:<10} {row['interleaves']:>3} {row['freq_precision']:>3} "
              f"{row['alphabet']:>5} {row['parallel_states']:>5} "
              f"{enc_mbs:>10.1f} {dec_mbs:>10.1f} "
              f"{row['enc_s'] * 1e6:>9.1f} {row['dec_s'] * 1e6:>9.1f} "
              f"{bits_per_symbol:>9.3f} {row['entropy']:>6.3f} {overhead:>6.3f} "
              f"{row['table_bytes'] / 1024.0:>9.1f} {smem:>5} {blk:>4} {wblk:>5}")
        if (row["parallel_states"] == first_batch
                and row["block_threads"] == first_block_threads
                and row["warp_block_threads"] == first_warp_block_threads):
            by_key[(row["impl"], row["interleaves"], row["alphabet"],
                    row["freq_precision"])] = (enc_mbs, dec_mbs)

    if skipped:
        print(f"\nskipped {len(skipped)} configuration(s) not codable on "
              f"{args.device}:")
        for impl, ilv, alphabet, precision, reason in skipped[:20]:
            print(f"  {impl} x{ilv} alphabet={alphabet} fp={precision}: {reason}")

    print(f"\n(the sections below use parallel_states={first_batch}, blk="
          f"{first_block_threads}, wblk={first_warp_block_threads})")

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

    # 3. the two warp paths head to head at 32 lanes (the widest group both
    #    families have): rans32_16 is the only rANS family with a 32-way kernel,
    #    so it is the baseline this comparison is about.
    print("\ntANS vs rANS at 32 lanes (encode / decode speedup):")
    for alphabet in args.alphabet:
        for precision in args.precision:
            base = by_key.get(("rans32_16", 32, alphabet, precision))
            if base is None:
                continue
            for name in args.impl:
                if not name.startswith("tans"):
                    continue
                cur = by_key.get((name, 32, alphabet, precision))
                if cur is None:
                    continue
                print(f"  alphabet={alphabet:<4} fp={precision:<3} {name:<8} "
                      f"vs rans32_16 x32: enc {cur[0] / base[0]:.2f}x, "
                      f"dec {cur[1] / base[1]:.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump({
                "machine": {
                    "machine": platform.machine(),
                    "device": args.device,
                    "device_name": device_name,
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "python": platform.python_version(),
                    "threads": args.threads,
                },
                "config": {
                    "symbols": num_symbols,
                    "distributions": args.dists,
                    "parallel_states": args.parallel_states,
                    "reps": args.reps,
                    "bypass": bypass,
                    # the tANS CUDA launch knobs, as the kernels will read them;
                    # None means "the value compiled into the kernels" for the
                    # block sizes (the budget has a fixed default, see
                    # DEFAULT_SMEM_BUDGET)
                    "tans_smem_budget": (args.smem_budget
                                         if args.smem_budget is not None
                                         else _smem_budget()),
                    "tans_block_threads": args.block_threads,
                    "tans_warp_block_threads": args.warp_block_threads,
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
