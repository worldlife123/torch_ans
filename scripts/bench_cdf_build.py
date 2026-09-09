"""B4 — fused ``pmf -> quantized CDF``: correctness cross-check and timing.

``rans_pmf_to_quantized_cdf`` used to be a chain of ~10 torch ops
(round / zeros / index_put / sum / where / mul / div / cumsum / index_put plus a
fix-up kernel). On CUDA each torch call costs ~40 us of dispatch here, and
``init_params`` runs that chain on every encode/decode call of the
``dist_freqs`` API - it was 0.27 ms of the 0.44 ms init. B4 replaces it with a
single fused pass (one kernel per row on CUDA, one parallel loop per row on
CPU), which is only allowed to land if it is bit-identical to the old chain.

This script keeps the old chain as a torch-level reference and compares it
against the fused op for a range of shapes, precisions, dtypes and edge cases
(all-zero pmf, many-zero pmf -> the "steal frequency" fix-up path, 1D input,
float64 input), then times both.

Usage::

    python scripts/bench_cdf_build.py [--devices cpu cuda] [--json out.json]
"""

import argparse
import json
import time

import torch

from torch_ans._C import rans_pmf_to_quantized_cdf


def reference_pmf_to_quantized_cdf(pmf, precision):
    """The original torch-op chain, kept verbatim as the oracle."""
    was_1d = pmf.dim() == 1
    batched = pmf.unsqueeze(0) if was_1d else pmf.reshape(-1, pmf.size(-1))
    B, N = batched.size(0), batched.size(1)
    dtype = torch.int32
    freq = torch.round(batched * (1 << precision)).to(dtype)
    cdf = torch.zeros((B, N + 1), dtype=dtype, device=batched.device)
    cdf[:, 1:] = freq
    total = cdf.sum(1, keepdim=True).to(dtype)
    total = torch.where(total == 0, torch.ones_like(total), total)
    cdf = ((cdf * (1 << precision)) / total).to(dtype)
    cdf = torch.cumsum(cdf, 1).to(dtype)
    cdf[:, N] = 1 << precision
    # the fix-up loop ("steal frequency" from the cheapest symbol)
    rows = cdf
    for b in range(B):
        row = rows[b]
        for i in range(N):
            if row[i] == row[i + 1]:
                best_freq, best_steal = None, -1
                for j in range(N):
                    f = int(row[j + 1]) - int(row[j])
                    if f > 1 and (best_freq is None or f < best_freq):
                        best_freq, best_steal = f, j
                if best_steal == -1:
                    continue
                if best_steal < i:
                    row[best_steal + 1:i + 1] -= 1
                elif best_steal > i:
                    row[i + 1:best_steal + 1] += 1
    cdf = cdf.contiguous()
    return cdf[0] if was_1d else cdf.reshape(*pmf.shape[:-1], N + 1)


def make_pmf(shape, kind, dtype, device, seed=0):
    g = torch.Generator().manual_seed(seed)
    if kind == "random":
        pmf = torch.rand(shape, generator=g, dtype=torch.float64).to(dtype)
        pmf = pmf / pmf.sum(-1, keepdim=True)
    elif kind == "uniform_int":
        pmf = torch.randint(1, 512, shape, generator=g).to(dtype)
        pmf = pmf / pmf.sum(-1, keepdim=True)
    elif kind == "all_zero":
        pmf = torch.zeros(shape, dtype=dtype)
    elif kind == "many_zeros":
        # half the mass spread over a handful of symbols -> many equal cdf
        # entries, exercises the fix-up path
        pmf = torch.zeros(shape, dtype=dtype)
        pmf[..., :: max(shape[-1] // 4, 1)] = 1.0
        pmf = pmf / pmf.sum(-1, keepdim=True)
    elif kind == "one_hot":
        pmf = torch.zeros(shape, dtype=dtype)
        pmf[..., 0] = 1.0
    return pmf.to(device)


def timeit(fn, device, repeats):
    if device == "cuda":
        fn()
        torch.cuda.synchronize()
        best = None
        for _ in range(repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            out = fn()
            e.record()
            torch.cuda.synchronize()
            t = s.elapsed_time(e) / 1000.0
            best = t if best is None else min(best, t)
        return out, best
    out = fn()
    best = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t = time.perf_counter() - t0
        best = t if best is None else min(best, t)
    return out, best


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--devices", nargs="+", default=None)
    ap.add_argument("--shapes", nargs="+", default=["8,258", "8,1026", "1,66"])
    ap.add_argument("--precisions", type=int, nargs="+", default=[8, 12, 15])
    ap.add_argument("--kinds", nargs="+",
                    default=["uniform_int", "random", "many_zeros", "one_hot"])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
    header = (f"{'dev':<5}{'shape':<10}{'p':>3}{'kind':<13}{'dtype':<9}"
              f"{'ref':>10}{'fused':>10}{'speedup':>9}{'same':>6}")
    print(header)
    print("-" * len(header))
    collected = []
    ok = True
    for device in devices:
        for shape in args.shapes:
            dims = [int(x) for x in shape.split(",")]
            for p in args.precisions:
                for kind in args.kinds:
                    for dtype in (torch.float32, torch.float64):
                        if (1 << p) < dims[-1] + 1:
                            # degenerate: 2**p frequency units cannot give every
                            # symbol at least one -> the fix-up has no donor and
                            # both implementations raise
                            continue
                        pmf = make_pmf(dims, kind, dtype, device)
                        ref, t_ref = timeit(
                            lambda: reference_pmf_to_quantized_cdf(pmf, p), device, args.repeats)
                        new, t_new = timeit(
                            lambda: rans_pmf_to_quantized_cdf(pmf, p), device, args.repeats)
                        same = torch.equal(ref, new)
                        ok = ok and same
                        print(f"{device:<5}{shape:<10}{p:>3}{kind:<13}"
                              f"{str(dtype).replace('torch.', ''):<9}"
                              f"{t_ref * 1e3:>9.3f}m{t_new * 1e3:>9.3f}m"
                              f"{t_ref / max(t_new, 1e-9):>8.2f}x{str(same):>6}")
                        collected.append({
                            "device": device, "shape": shape, "precision": p,
                            "kind": kind, "dtype": str(dtype),
                            "sec_reference": t_ref, "sec_fused": t_new, "identical": bool(same),
                        })
    # 1D input
    for device in devices:
        pmf = make_pmf([66], "uniform_int", torch.float32, device)
        ref = reference_pmf_to_quantized_cdf(pmf, 12)
        new = rans_pmf_to_quantized_cdf(pmf, 12)
        print(f"{device:<5} 1D input{'':<2} identical={torch.equal(ref, new)}")
        ok = ok and torch.equal(ref, new)

    # Degenerate case (2**p < N+1 -> no symbol can donate a frequency unit).
    # Pre-existing divergence preserved on purpose: the host path raises
    # (TORCH_CHECK), the CUDA path silently skips the row.
    for device in devices:
        pmf = make_pmf([4, 300], "uniform_int", torch.float32, device)
        expect_raise = (device == "cpu")
        try:
            rans_pmf_to_quantized_cdf(pmf, 8)
            raised = False
        except RuntimeError:
            raised = True
        print(f"{device:<5} degenerate (2**p < N+1): raises={raised} "
              f"(expected {expect_raise}) -> {'OK' if raised == expect_raise else 'CHANGED'}")
        ok = ok and (raised == expect_raise)

    print("\nALL IDENTICAL" if ok else "\n!!! MISMATCH - B4 must not land")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(collected, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
