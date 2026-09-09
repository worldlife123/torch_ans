"""Decode-throughput benchmark comparing rANS symbol-lookup strategies.

Three lookup strategies for turning a cumulative frequency into a symbol:

  * divided-search + binary refine   -- baseline, plain ``rans32_16*_pop``
  * dense inverse CDF                -- ``inverse_cdf_precision == freq_precision``
  * sparse inverse CDF               -- ``1 <= inverse_cdf_precision < freq_precision``,
                                        ``2**inverse_cdf_precision``-entry table plus a
                                        bounded linear walk (see rans_warp_pop_lookup /
                                        rans_pop_step and utils.inverse_quantized_cdf)

Matrix: device (cpu/cuda) x num_interleaves (1/32) x freq_precision (12/15) x
alphabet size (16/256). Decoding only rewinds the stream cursor and never
rewrites the coded words, so the encoded stream is cloned once per repeat up
front and the clone cost stays outside the timed region.

Usage:
    python scripts/bench_sparse_invcdf.py [--devices cpu cuda] [--repeats 20]
                                          [--json bench.json]

Output: per-configuration lines with the per-row table size, decode throughput
(symbols/s and MB/s of coded bytes) and speed-up vs the binary-search baseline.
"""

import argparse
import json
import time

import torch

from torch_ans._C import (
    rans_pmf_to_quantized_cdf,
    rans32_16_init_stream,
    rans32_16_push,
    rans32_16_pop,
    rans32_16_invcdf_pop,
    rans32_16_i4_push,
    rans32_16_i4_pop,
    rans32_16_i4_invcdf_pop,
    rans32_16_i32_push,
    rans32_16_i32_pop,
    rans32_16_i32_invcdf_pop,
)
from torch_ans.utils import inverse_quantized_cdf

OPS = {
    1: (rans32_16_push, rans32_16_pop, rans32_16_invcdf_pop),
    4: (rans32_16_i4_push, rans32_16_i4_pop, rans32_16_i4_invcdf_pop),
    32: (rans32_16_i32_push, rans32_16_i32_pop, rans32_16_i32_invcdf_pop),
}


def cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        torch.zeros(1, dtype=torch.int32).cuda()
        return True
    except Exception:
        return False


def time_decode(device, fn, clones):
    """Mean seconds per decode over the given stream clones (one pop each)."""
    if device == "cuda":
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for s in clones:
            fn(s)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000.0 / len(clones)
    t0 = time.perf_counter()
    for s in clones:
        fn(s)
    return (time.perf_counter() - t0) / len(clones)


def run_configuration(device, interleaves, freq_precision, num_symbols,
                      rows, symbols_per_row, repeats, warmup, runs, q_drops,
                      seed=0):
    push, pop, invcdf_pop = OPS[interleaves]
    num_dists = 8
    bypass = True

    g = torch.Generator().manual_seed(seed)
    pmf = torch.randint(1, 512, (num_dists, num_symbols), generator=g).float()
    pmf = torch.cat([pmf.clone(), torch.ones(num_dists, 1)], dim=1)
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)
    cdfs = rans_pmf_to_quantized_cdf(pmf, freq_precision)
    cdfs_sizes = torch.zeros(num_dists, dtype=torch.int32) + num_symbols + 2
    offsets = torch.randint(-4, 4, (num_dists,), generator=g, dtype=torch.int32)

    data = torch.randint(-4, num_symbols + 8, (rows, symbols_per_row),
                         generator=g, dtype=torch.int32)
    indexes = torch.randint(0, num_dists, (rows, symbols_per_row),
                            generator=g, dtype=torch.int32)

    stream = rans32_16_init_stream(rows, interleaves)
    push(stream, data, indexes, cdfs, cdfs_sizes, offsets,
         freq_precision=freq_precision, bypass_coding=bypass, bypass_precision=4)
    if device == "cuda":
        torch.cuda.synchronize()
        stream = stream.cuda()

    sanity = pop(stream.clone(), indexes.to(device), cdfs.to(device),
                 cdfs_sizes.to(device), offsets.to(device),
                 freq_precision=freq_precision, bypass_coding=bypass,
                 bypass_precision=4)
    if not torch.equal(data, sanity.cpu()):
        raise RuntimeError("baseline decode mismatch - benchmark is invalid")

    coded_bytes = float(stream[:, 0].sum().cpu().item())
    total_symbols = rows * symbols_per_row

    # baseline first (no table), then dense, then the sparse ladder
    strategies = [("binary", None), (f"full_invcdf(p={freq_precision})", freq_precision)]
    for d in q_drops:
        q = freq_precision - d
        if q >= 1:
            strategies.append((f"sparse_q={q}", q))

    results = []
    for label, table_precision in strategies:
        cdfs_d = (cdfs if table_precision is None else torch.cat(
            [cdfs, inverse_quantized_cdf(cdfs, freq_precision=freq_precision,
                                         table_precision=table_precision)], dim=-1)).to(device)
        indexes_d = indexes.to(device)
        decode_op = pop if table_precision is None else invcdf_pop

        def decode(stream_, cdfs_d=cdfs_d, table_precision=table_precision,
                   decode_op=decode_op):
            kwargs = {"freq_precision": freq_precision, "bypass_coding": bypass,
                      "bypass_precision": 4}
            if table_precision is not None:
                kwargs["inverse_cdf_precision"] = table_precision
            return decode_op(stream_, indexes_d, cdfs_d, cdfs_sizes.to(device),
                             offsets.to(device), **kwargs)

        out = decode(stream.clone())
        if not torch.equal(data, out.cpu()):
            raise RuntimeError(f"{label} decode mismatch - benchmark is invalid")

        for _ in range(warmup):
            decode(stream.clone())
        best = None
        for _ in range(runs):
            # fresh clones every timed pass: decoding consumes the stream
            clones = [stream.clone() for _ in range(repeats)]
            secs = time_decode(device, decode, clones)
            best = secs if best is None else min(best, secs)
        results.append({
            "strategy": label,
            "table_entries_per_row": (1 << table_precision)
                                     if table_precision is not None else 0,
            "sec_per_decode": best,
            "throughput_msymbols_per_s": total_symbols / best / 1e6,
            "throughput_mbytes_per_s": coded_bytes / best / 1e6,
        })
    base = results[0]["sec_per_decode"]
    for r in results:
        r["speedup_vs_binary"] = base / r["sec_per_decode"]
        r.update({"device": device, "interleaves": interleaves,
                  "freq_precision": freq_precision, "num_symbols": num_symbols,
                  "coded_bytes": coded_bytes})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None,
                    help="devices to benchmark (cpu, cuda); default: both when CUDA works")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--runs", type=int, default=3,
                    help="number of timed passes per strategy (the best one is kept)")
    ap.add_argument("--freq-precision", nargs="+", type=int, default=[12, 15])
    ap.add_argument("--num-symbols", nargs="+", type=int, default=[16, 256])
    ap.add_argument("--interleaves", nargs="+", type=int, default=[1, 32])
    ap.add_argument("--q-drops", nargs="+", type=int, default=[1, 2, 4, 6, 8],
                    help="sparse precisions to test, as freq_precision - drop")
    ap.add_argument("--rows", type=int, default=None,
                    help="number of parallel streams (default: 32 cpu / 256 cuda)")
    ap.add_argument("--symbols-per-row", type=int, default=8192)
    ap.add_argument("--json", default=None, help="optional JSON output path")
    args = ap.parse_args()

    devices = args.devices or (["cpu", "cuda"] if cuda_usable() else ["cpu"])
    devices = [d for d in devices if d == "cpu" or cuda_usable()]

    header = (f"{'device':<6}{'ilv':<4}{'p':>3}{'sym':>5}{'strategy':<20}"
              f"{'tbl/row':>8}{'Msym/s':>9}{'MB/s':>9}{'vs bin':>9}")
    print(header)
    print("-" * len(header))
    collected = []
    for device in devices:
        rows = args.rows or (32 if device == "cpu" else 256)
        for interleaves in args.interleaves:
            for freq_precision in args.freq_precision:
                for num_symbols in args.num_symbols:
                    results = run_configuration(
                        device, interleaves, freq_precision, num_symbols,
                        rows, args.symbols_per_row, args.repeats, args.warmup,
                        args.runs, args.q_drops)
                    for r in results:
                        print(f"{r['device']:<6}{r['interleaves']:<4}"
                              f"{r['freq_precision']:>3}{r['num_symbols']:>5}"
                              f"{r['strategy']:<20}{r['table_entries_per_row']:>8}"
                              f"{r['throughput_msymbols_per_s']:>9.1f}"
                              f"{r['throughput_mbytes_per_s']:>9.1f}"
                              f"{r['speedup_vs_binary']:>9.2f}")
                        collected.append(r)
                    print()
        if len(devices) > 1:
            print()
    if args.json:
        with open(args.json, "w") as f:
            json.dump(collected, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
