"""B1/B2 — inverse-CDF table construction: cost and alternatives.

The table turns a cumulative frequency into a symbol. ``inverse_quantized_cdf``
currently evaluates it as a broadcast compare-and-count,

    (arange(2**q) << shift).unsqueeze(-1) >= cdf.unsqueeze(1)   # (N, 2**q, M)
    .sum(-1) - 1

which materialises an ``N * 2**q * M`` intermediate: 270 MB already for a
dense table with 8 distributions and 256 symbols, and gigabytes for larger
alphabets. B1 replaces it with a batched ``torch.searchsorted``
(``O(2**q log M)``, no big intermediate); B2 would move the whole thing into a
CUDA kernel so that the table is produced device-side in one pass.

This script cross-checks the alternatives against the reference for bit-exact
equality and times them, so a change can only land if it is both faster and
identical.

Usage::

    python scripts/bench_table_build.py [--devices cpu cuda] [--json out.json]
"""

import argparse
import json
import time

import torch

from torch_ans.utils import inverse_quantized_cdf, rans_pmf_to_quantized_cdf


def build_cdfs(num_dists, num_symbols, freq_precision, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    pmf = torch.randint(1, 512, (num_dists, num_symbols), generator=g).float()
    pmf = torch.cat([pmf.clone(), torch.ones(num_dists, 1)], dim=1)
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)
    return rans_pmf_to_quantized_cdf(pmf, freq_precision).to(device).contiguous()


def table_broadcast(quantized_cdf, freq_precision, table_precision):
    """Reference: the original broadcast compare-and-count (kept here so the
    A/B stays meaningful after B1 lands in ``utils.inverse_quantized_cdf``)."""
    table_precision = freq_precision if table_precision is None else int(table_precision)
    shift = freq_precision - table_precision
    table_size = (1 << table_precision)
    freq_range = (torch.arange(table_size).unsqueeze(0) << shift).type_as(quantized_cdf)
    return ((freq_range.unsqueeze(-1) >= quantized_cdf.unsqueeze(1))
            .sum(-1, dtype=quantized_cdf.dtype) - 1)


def table_searchsorted(quantized_cdf, freq_precision, table_precision):
    """B1: batched binary search instead of the (N, 2**q, M) broadcast."""
    table_precision = freq_precision if table_precision is None else int(table_precision)
    shift = freq_precision - table_precision
    table_size = 1 << table_precision
    values = (torch.arange(table_size, device=quantized_cdf.device) << shift)
    values = values.unsqueeze(0).expand(quantized_cdf.size(0), table_size).contiguous()
    idx = torch.searchsorted(quantized_cdf.contiguous(), values, right=True)
    return (idx - 1).to(quantized_cdf.dtype)


def table_cdf_kernel(quantized_cdf, freq_precision, table_precision):
    """B2 placeholder: counted once per table element on the device.

    Same arithmetic as the reference but without the (N, 2**q, M) temporary -
    this is what a dedicated CUDA kernel would do, simulated here with a
    Python loop-free gather so the timing is dominated by the search itself.
    """
    return table_searchsorted(quantized_cdf, freq_precision, table_precision)


def timeit(fn, device, repeats):
    if device == "cuda":
        fn()
        torch.cuda.synchronize()
        best = None
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            out = fn()
            end.record()
            torch.cuda.synchronize()
            secs = start.elapsed_time(end) / 1000.0
            best = secs if best is None else min(best, secs)
        return out, best
    out = fn()
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        secs = time.perf_counter() - t0
        best = secs if best is None else min(best, secs)
    return out, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--num-dists", type=int, nargs="+", default=[8])
    ap.add_argument("--num-symbols", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--freq-precision", type=int, nargs="+", default=[15])
    ap.add_argument("--q", type=int, nargs="+", default=[7, 9, 11, 15])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    header = (f"{'dev':<5}{'dist':>5}{'sym':>6}{'p':>4}{'q':>4}{'tbl':>8}"
              f"{'broadcast':>12}{'searchsorted':>14}{'speedup':>9}{'same':>6}")
    print(header)
    print("-" * len(header))
    collected = []
    for device in devices:
        for num_dists in args.num_dists:
            for num_symbols in args.num_symbols:
                for p in args.freq_precision:
                    cdfs = build_cdfs(num_dists, num_symbols, p, device)
                    for q in args.q:
                        if q > p:
                            continue
                        ref, t_ref = timeit(
                            lambda: table_broadcast(cdfs, freq_precision=p, table_precision=q),
                            device, args.repeats)
                        new, t_new = timeit(
                            lambda: inverse_quantized_cdf(cdfs, freq_precision=p,
                                                          table_precision=q),
                            device, args.repeats)
                        same = torch.equal(ref, new)
                        print(f"{device:<5}{num_dists:>5}{num_symbols:>6}{p:>4}{q:>4}"
                              f"{ref.numel():>8}{t_ref * 1e3:>11.2f}m{t_new * 1e3:>13.2f}m"
                              f"{t_ref / max(t_new, 1e-9):>8.2f}x{str(same):>6}")
                        collected.append({
                            "device": device, "num_dists": num_dists,
                            "num_symbols": num_symbols, "freq_precision": p, "q": q,
                            "table_entries": ref.numel(),
                            "sec_broadcast": t_ref, "sec_searchsorted": t_new,
                            "identical": bool(same),
                        })
                        if not same:
                            print("   !!! MISMATCH - not bit-exact, B1 must not land")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(collected, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
