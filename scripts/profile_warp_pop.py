"""A1 — warp pop kernel profiling driver (ncu) + counter-free ablation harness.

Two modes:

``profile``
    Set up one fixed configuration and run N identical decode launches. Use it
    under ``ncu`` so that the profiler sees exactly the launches of interest::

        ncu --target-processes all -k regex:runs_warp_pop_indexed_kernel \\
            -c 3 --set full ./scripts/run_ncu_warp.sh --strategy sparse --q 11

    Without ``ncu`` it simply prints the wall-clock throughput of that config,
    which is handy as a sanity check of what the profiled run measures.

``ablation``
    Counter-free substitute for the Nsight metrics: sweeps rows / lookup
    strategy / bypass / alphabet / freq_precision and prints, for every point,
    the launch geometry (blocks, warps, SM coverage) next to the achieved
    throughput and DRAM traffic. Comparing the columns isolates whether the
    kernel is limited by occupancy (launch geometry), by the serial bypass
    phase (sync), by symbol lookup (arithmetic) or by stream traffic (memory).

Usage examples::

    python scripts/profile_warp_pop.py profile --strategy sparse --q 11
    python scripts/profile_warp_pop.py ablation --rows 256 1024 4096 \\
        --strategies binary dense sparse --json /tmp/a1_ablation.json
"""

import argparse
import json
import time

import torch

from torch_ans._C import (
    rans_pmf_to_quantized_cdf,
    rans32_16_init_stream,
    rans32_16_i32_push,
    rans32_16_i32_pop,
    rans32_16_i32_invcdf_pop,
)
from torch_ans.utils import inverse_quantized_cdf

# rans_warp_pop_indexed_kernel launches NUM_INTERLEAVES=32 lane groups per row
# with WARP_NUM_THREADS_PER_BLOCK=256 threads (see rans_warp_cuda.cuh).
NUM_INTERLEAVES = 32
THREADS_PER_BLOCK = 256
GROUPS_PER_BLOCK = THREADS_PER_BLOCK // NUM_INTERLEAVES  # 8
WARPS_PER_BLOCK = THREADS_PER_BLOCK // 32  # 8

# dense table == q == p; sparse == 1 <= q < p; binary == no table at all
STRATEGIES = ("binary", "dense", "sparse")


def device_props():
    p = torch.cuda.get_device_properties(0)
    return {
        "name": p.name,
        "sm_count": p.multi_processor_count,
        "max_threads_per_sm": p.max_threads_per_multi_processor,
        "max_threads_per_block": p.max_threads_per_block,
        "l2_cache_size": p.L2_cache_size,
    }


def geometry(rows):
    blocks = (rows + GROUPS_PER_BLOCK - 1) // GROUPS_PER_BLOCK
    return {
        "rows": rows,
        "num_blocks": blocks,
        "threads_per_block": THREADS_PER_BLOCK,
        "warps_per_block": WARPS_PER_BLOCK,
        "total_warps": blocks * WARPS_PER_BLOCK,
    }


def build_case(rows, symbols_per_row, num_symbols, freq_precision, seed=0,
               bypass=True, in_range=False):
    """Encode one stream on the CPU, return everything needed to decode it.

    ``in_range`` keeps every symbol inside the CDF so that the bypass (raw
    value) path is never taken — that is the only way to encode with
    ``bypass_coding=False``, and it makes the two bypass ablations comparable.
    """
    num_dists = 8
    bypass_precision = 4
    g = torch.Generator().manual_seed(seed)
    pmf = torch.randint(1, 512, (num_dists, num_symbols), generator=g).float()
    pmf = torch.cat([pmf.clone(), torch.ones(num_dists, 1)], dim=1)
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)
    cdfs = rans_pmf_to_quantized_cdf(pmf, freq_precision)
    cdfs_sizes = torch.zeros(num_dists, dtype=torch.int32) + num_symbols + 2
    offsets = torch.randint(-4, 4, (num_dists,), generator=g, dtype=torch.int32)

    if in_range:
        # 4 <= data < num_symbols - 8 keeps data - offset inside [0, max_value)
        # for offsets in [-4, 4), so no symbol becomes a bypass sentinel
        data = torch.randint(4, max(num_symbols - 8, 5), (rows, symbols_per_row),
                             generator=g, dtype=torch.int32)
    else:
        data = torch.randint(-4, num_symbols + 8, (rows, symbols_per_row),
                             generator=g, dtype=torch.int32)
    indexes = torch.randint(0, num_dists, (rows, symbols_per_row),
                            generator=g, dtype=torch.int32)

    stream = rans32_16_init_stream(rows, NUM_INTERLEAVES)
    rans32_16_i32_push(stream, data, indexes, cdfs, cdfs_sizes, offsets,
                       freq_precision=freq_precision, bypass_coding=bypass,
                       bypass_precision=bypass_precision)
    torch.cuda.synchronize()
    stream = stream.cuda()
    return {
        "stream": stream,
        "data": data,
        "indexes": indexes.cuda(),
        "cdfs": cdfs,
        "cdfs_sizes": cdfs_sizes,
        "offsets": offsets,
        "bypass_precision": bypass_precision,
        "coded_bytes": float(stream[:, 0].sum().cpu().item()),
        "bypass": bypass,
        "in_range": in_range,
    }


def make_decode(case, strategy, freq_precision, q, bypass_coding, device="cuda"):
    """Return a callable that decodes a cloned stream once."""
    if strategy == "binary":
        table_precision, op = None, rans32_16_i32_pop
    elif strategy == "dense":
        table_precision, op = freq_precision, rans32_16_i32_invcdf_pop
    elif strategy == "sparse":
        if not 1 <= q < freq_precision:
            raise ValueError(f"sparse needs 1 <= q < p, got q={q}, p={freq_precision}")
        table_precision, op = q, rans32_16_i32_invcdf_pop
    else:
        raise ValueError(f"unknown strategy {strategy}")

    cdfs = case["cdfs"]
    if table_precision is not None:
        cdfs = torch.cat([cdfs, inverse_quantized_cdf(
            cdfs, freq_precision=freq_precision, table_precision=table_precision)], dim=-1)
    cdfs_d = cdfs.to(device)
    cdfs_sizes_d = case["cdfs_sizes"].to(device)
    offsets_d = case["offsets"].to(device)
    indexes_d = case["indexes"]

    kwargs = {"freq_precision": freq_precision, "bypass_coding": bypass_coding,
              "bypass_precision": case["bypass_precision"]}
    if table_precision is not None:
        kwargs["inverse_cdf_precision"] = table_precision

    def decode(stream_):
        return op(stream_, indexes_d, cdfs_d, cdfs_sizes_d, offsets_d, **kwargs)

    return decode


def time_decode(decode, stream, repeats, warmup, runs, stream_bytes):
    """Best-of-`runs` mean seconds per decode (clone cost excluded).

    ``repeats`` is capped so that one timed pass never holds more than ~1.5 GB
    of stream clones (a clone is a full copy of the stream tensor).
    """
    repeats = max(2, min(repeats, int(1_500_000_000 // max(stream_bytes, 1))))
    for _ in range(warmup):
        decode(stream.clone())
    torch.cuda.synchronize()
    best = None
    for _ in range(runs):
        clones = [stream.clone() for _ in range(repeats)]
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for s in clones:
            decode(s)
        end.record()
        torch.cuda.synchronize()
        secs = start.elapsed_time(end) / 1000.0 / len(clones)
        best = secs if best is None else min(best, secs)
    return best


_CASE_CACHE = {}


def get_case(rows, symbols_per_row, num_symbols, freq_precision, seed, bypass,
             in_range):
    """Cached build_case: encoding is CPU-side and dominates the sweep cost."""
    key = (rows, symbols_per_row, num_symbols, freq_precision, seed, bypass,
           in_range)
    if key not in _CASE_CACHE:
        _CASE_CACHE[key] = build_case(rows, symbols_per_row, num_symbols,
                                      freq_precision, seed, bypass, in_range)
    return _CASE_CACHE[key]


def clear_cases():
    """Drop cached (GPU) cases — the stream tensors are the memory hog."""
    _CASE_CACHE.clear()
    torch.cuda.empty_cache()


def measure(rows, symbols_per_row, num_symbols, freq_precision, strategy, q,
            bypass_coding, repeats, warmup, runs, seed=0, in_range=False):
    case = get_case(rows, symbols_per_row, num_symbols, freq_precision, seed,
                    bypass_coding, in_range)
    decode = make_decode(case, strategy, freq_precision, q, bypass_coding)

    ref = decode(case["stream"].clone())
    if not torch.equal(ref.cpu(), case["data"]):
        raise RuntimeError(f"decode mismatch: rows={rows} sym={num_symbols} "
                           f"p={freq_precision} strategy={strategy} q={q}")

    total_symbols = rows * symbols_per_row
    stream_bytes = case["stream"].numel() * case["stream"].element_size()
    secs = time_decode(decode, case["stream"], repeats, warmup, runs, stream_bytes)
    # decode reads the coded stream + indexes and writes the symbols (int32)
    bytes_moved = case["coded_bytes"] + 2 * 4 * total_symbols
    res = {
        "strategy": strategy,
        "q": (q if strategy == "sparse" else (freq_precision if strategy == "dense" else None)),
        "rows": rows,
        "symbols_per_row": symbols_per_row,
        "num_symbols": num_symbols,
        "freq_precision": freq_precision,
        "bypass_coding": bypass_coding,
        "in_range": in_range,
        "sec_per_decode": secs,
        "msymbols_per_s": total_symbols / secs / 1e6,
        "ns_per_symbol": secs / total_symbols * 1e9,
        "dram_gb_per_s": bytes_moved / secs / 1e9,
        "coded_bytes": case["coded_bytes"],
        "bits_per_symbol": case["coded_bytes"] * 8 / total_symbols,
    }
    res.update(geometry(rows))
    return res


def cmd_profile(args):
    torch.manual_seed(args.seed)
    case = build_case(args.rows, args.symbols_per_row, args.num_symbols,
                      args.freq_precision, args.seed, args.bypass, args.in_range)
    decode = make_decode(case, args.strategy, args.freq_precision, args.q, args.bypass)
    out = decode(case["stream"].clone())
    if not torch.equal(out.cpu(), case["data"]):
        raise RuntimeError("decode mismatch - profiling an invalid configuration")
    for _ in range(args.warmup):
        decode(case["stream"].clone())
    torch.cuda.synchronize()

    # the launch(es) ncu should profile: identical work, no clone in between
    # beyond the stream copy that every iteration needs
    streams = [case["stream"].clone() for _ in range(args.iters)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for s in streams:
        decode(s)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total = args.rows * args.symbols_per_row
    print(f"[profile] strategy={args.strategy} q={args.q} p={args.freq_precision} "
          f"sym={args.num_symbols} rows={args.rows} per_row={args.symbols_per_row} "
          f"bypass={args.bypass} iters={args.iters}")
    print(f"[profile] blocks={geometry(args.rows)['num_blocks']} "
          f"warps={geometry(args.rows)['total_warps']}")
    print(f"[profile] {args.iters} decodes in {elapsed * 1e3:.2f} ms -> "
          f"{total * args.iters / elapsed / 1e6:.1f} Msymbol/s")


def warmup_gpu(seconds=1.5, **case_kwargs):
    """Spin the GPU up before the first timed measurement.

    Clocks ramp up over the first seconds of load, which made the *first*
    configuration measured in a process read 10-20% slow (measured: binary
    11067 cold vs 13015 warm). Decode a small case until the budget elapsed.
    """
    case = build_case(rows=64, symbols_per_row=4096, num_symbols=256,
                      freq_precision=15, **case_kwargs)
    decode = make_decode(case, "binary", 15, 11, True)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        decode(case["stream"].clone())
    torch.cuda.synchronize()


def cmd_ablation(args):
    torch.manual_seed(args.seed)
    warmup_gpu()
    props = device_props()
    print(f"device: {props['name']} ({props['sm_count']} SMs, "
          f"{props['max_threads_per_sm']} threads/SM)")
    rows_list = args.rows
    max_blocks_per_sm = max(1, props["max_threads_per_sm"] // THREADS_PER_BLOCK)
    header = (f"{'rows':>6}{'blk':>5}{'blk/SM':>8}{'occ%':>6}{'strategy':>10}{'q':>4}{'p':>4}"
              f"{'sym':>5}{'Msym/s':>9}{'ns/sym':>8}{'GB/s':>8}{'bit/sym':>9}")
    print(header)
    print("-" * len(header))
    collected = []
    for rows in rows_list:
        clear_cases()
        geo = geometry(rows)
        sm_pct = min(1.0, geo["num_blocks"] / props["sm_count"]) * 100
        blocks_per_sm = geo["num_blocks"] / props["sm_count"]
        occ_pct = min(1.0, blocks_per_sm / max_blocks_per_sm) * 100
        for strategy in args.strategies:
            qs = [None] if strategy != "sparse" else args.q
            for q in qs:
                try:
                    r = measure(rows, args.symbols_per_row, args.num_symbols,
                                args.freq_precision, strategy,
                                q if q is not None else args.q[0],
                                not args.no_bypass, args.repeats, args.warmup,
                                args.runs, args.seed, args.in_range)
                except ValueError as exc:
                    print(f"  skip {strategy}: {exc}")
                    continue
                r["sm_coverage_pct"] = sm_pct
                r["occupancy_pct"] = occ_pct
                print(f"{r['rows']:>6}{r['num_blocks']:>5}{blocks_per_sm:>8.2f}{occ_pct:>6.1f}"
                      f"{r['strategy']:>10}{str(r['q'] or '-'):>4}{r['freq_precision']:>4}"
                      f"{r['num_symbols']:>5}"
                      f"{r['msymbols_per_s']:>9.1f}{r['ns_per_symbol']:>8.3f}"
                      f"{r['dram_gb_per_s']:>8.1f}{r['bits_per_symbol']:>9.2f}")
                collected.append(r)
        print()

    # --- cross-cutting ablations at the widest row count -----------------
    if args.cross:
        base_rows = args.cross_rows or min(max(rows_list), 2048)
        base_per_row = args.cross_symbols_per_row
        print("cross-cutting ablations (everything else fixed):")
        print(f"  base: rows={base_rows} p={args.freq_precision} "
              f"sym={args.num_symbols} per_row={base_per_row}")
        # each variant: label -> (bypass, in_range, num_symbols, freq_precision)
        variants = [
            ("A out-of-range, byp on", True, False, args.num_symbols, args.freq_precision),
            ("B in-range, byp on", True, True, args.num_symbols, args.freq_precision),
            ("C in-range, byp off", False, True, args.num_symbols, args.freq_precision),
            ("D in-range off sym16", False, True, 16, args.freq_precision),
            ("E in-range off sym64", False, True, 64, args.freq_precision),
            ("F in-range off sym1024", False, True, 1024, args.freq_precision),
            ("G out-of-range sym16", True, False, 16, args.freq_precision),
            ("H in-range off p=12", False, True, args.num_symbols, 12),
            ("I in-range off p=10", False, True, args.num_symbols, 10),
        ]
        for label, bypass, in_range, num_symbols, freq_precision in variants:
            clear_cases()
            for strategy in ("binary", "dense"):
                try:
                    r = measure(base_rows, base_per_row, num_symbols,
                                freq_precision, strategy, args.q[0],
                                bypass, args.repeats, args.warmup, args.runs,
                                args.seed, in_range)
                    print(f"    {label:<22} {strategy:<8} "
                          f"{r['msymbols_per_s']:>8.1f} Msym/s  "
                          f"{r['ns_per_symbol']:>6.3f} ns/sym  "
                          f"{r['dram_gb_per_s']:>6.1f} GB/s  "
                          f"{r['bits_per_symbol']:>5.2f} bit/sym")
                    collected.append(r)
                except (ValueError, RuntimeError) as exc:
                    print(f"    {label:<22} {strategy:<8} skipped: {exc}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"device": props, "results": collected}, f, indent=2)
        print(f"wrote {args.json}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--symbols-per-row", type=int, default=65536)
        p.add_argument("--num-symbols", type=int, default=256)
        p.add_argument("--freq-precision", type=int, default=15)
        p.add_argument("--repeats", type=int, default=20)
        p.add_argument("--warmup", type=int, default=2)
        p.add_argument("--runs", type=int, default=3)
        p.add_argument("--seed", type=int, default=0)

    pp = sub.add_parser("profile", help="single config, for use under ncu")
    common(pp)
    pp.add_argument("--rows", type=int, default=256)
    pp.add_argument("--q", type=int, default=11)
    pp.add_argument("--strategy", choices=STRATEGIES, default="sparse")
    pp.add_argument("--bypass", dest="bypass", action="store_true", default=True)
    pp.add_argument("--no-bypass", dest="bypass", action="store_false")
    pp.add_argument("--in-range", action="store_true",
                    help="encode in-range symbols only (needed with --no-bypass)")
    pp.add_argument("--iters", type=int, default=3,
                    help="number of identical decode launches to profile")
    pp.set_defaults(func=cmd_profile)

    pa = sub.add_parser("ablation", help="counter-free sweep")
    common(pa)
    pa.add_argument("--rows", type=int, nargs="+", default=[256, 1024, 4096])
    pa.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                    default=["binary", "dense", "sparse"])
    pa.add_argument("--q", type=int, nargs="+", default=[11])
    pa.add_argument("--no-bypass", action="store_true")
    pa.add_argument("--in-range", action="store_true",
                    help="encode in-range symbols only (needed with --no-bypass)")
    pa.add_argument("--cross", action="store_true",
                    help="also run the bypass/alphabet/freq_precision ablations")
    pa.add_argument("--cross-rows", type=int, default=None,
                    help="rows used by the cross ablations (default: min(max(rows), 2048))")
    pa.add_argument("--cross-symbols-per-row", type=int, default=16384,
                    help="symbols per row used by the cross ablations")
    pa.add_argument("--json", default=None)
    pa.set_defaults(func=cmd_ablation)

    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for warp-kernel profiling")
    args.func(args)


if __name__ == "__main__":
    main()
