#!/usr/bin/env python3
"""Cross-version rANS operator benchmark matrix (v0.2.1 vs v0.3.0).

The same script is run twice, once against each version of the package
(typically by checking out the release tag into a worktree, building the
extension there and running this file from that checkout), and the two JSON
outputs are the input of `benchmark_status.md`.

Why a dedicated script instead of `pytest-benchmark`:

* it drives the **raw operators** (`rans*_push` / `rans*_pop` and friends) so
  that configurations that only exist in one version can be skipped instead of
  failing (missing op -> `status: "unsupported"`, wrong result ->
  `status: "mismatch"`);
* it keeps the stream clones and the table construction outside the timed
  region, so the numbers are kernel time, not Python/dispatch time;
* it verifies every configuration decodes back to the source symbols before
  timing it, so a "fast" configuration that decodes garbage is reported as an
  error instead of a speed-up.

Sections:

* `codec`      -- encode/decode throughput per (device, family, interleave,
                  symbol lookup);
* `params`     -- parameter build cost: `pmf -> quantized cdf` and the inverse
                  CDF table (v0.3.0 fuses both into native ops);
* `high_level` -- end-to-end `TorchANSInterface.encode_with_indexes` /
                  `decode_with_indexes`, i.e. what an unchanged user program
                  gets after upgrading.

Usage:
    python scripts/bench_version_matrix.py --version-label 0.3.0 \
        --json bench_030.json
"""

from __future__ import annotations

import argparse
import json
import platform
import time

import torch

FAMILIES = ("rans64", "rans32", "rans32_16")
MAX_PRECISION = {"rans64": 31, "rans32": 23, "rans32_16": 15}

# alphabet 255 + 1 tail column -> cdfs_sizes = 257 -> cdfs_sizes - 1 = 256 =
# 2**symbol_precision, which is what `rans_alias_build_table` requires.
ALPHABET = 255
SYMBOL_PRECISION = 8
NUM_DISTS = 8
FREQ_PRECISION = 15
BYPASS_PRECISION = 4
SEED = 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def cpu_model():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def native(name):
    """Return the native op `name`, or None when this version does not have it."""
    from torch_ans import _C

    return getattr(_C, name, None)


def supports_inverse_cdf_arg(op):
    """True when the pop op takes `inverse_cdf_precision` (v0.3.0 and later)."""
    return "inverse_cdf_precision" in (getattr(op, "__doc__", "") or "")


def util(name):
    import torch_ans.utils as utils

    return getattr(utils, name, None)


def op_names(family, interleaves, lookup):
    """Operator names for a (family, interleave, lookup) combination."""
    suffix = "" if interleaves == 1 else f"_i{interleaves}"
    if lookup == "alias":
        prefix = f"{family}_alias{suffix}"
        return f"{prefix}_push", f"{prefix}_pop"
    prefix = f"{family}{suffix}"
    if lookup == "binary":
        return f"{prefix}_push", f"{prefix}_pop"
    return f"{prefix}_push", f"{prefix}_invcdf_pop"


def build_params(freq_precision, seed=SEED):
    g = torch.Generator().manual_seed(seed)
    pmf = torch.rand(NUM_DISTS, ALPHABET, generator=g) + 0.05
    # tail column (bypass): a tiny but non-zero mass so it keeps >= 1 count
    pmf = torch.cat([pmf, torch.full((NUM_DISTS, 1), 0.01)], dim=1)
    pmf = pmf / pmf.sum(dim=-1, keepdim=True)

    cdfs = native("rans_pmf_to_quantized_cdf")(pmf, freq_precision)
    cdfs_sizes = torch.full((NUM_DISTS,), ALPHABET + 2, dtype=torch.int32)
    offsets = torch.zeros(NUM_DISTS, dtype=torch.int32)
    return pmf, cdfs, cdfs_sizes, offsets


def build_data(rows, symbols_per_row, seed=SEED):
    g = torch.Generator().manual_seed(seed + 1)
    data = torch.randint(0, ALPHABET, (rows, symbols_per_row),
                         generator=g, dtype=torch.int32)
    indexes = torch.randint(0, NUM_DISTS, (rows, symbols_per_row),
                            generator=g, dtype=torch.int32)
    return data, indexes


def build_inverse_table(cdfs, freq_precision, table_precision):
    """cdf ++ inverse table, using the fastest builder this version has."""
    builder = native("rans_build_inverse_cdf")
    if builder is not None:
        return builder(cdfs, freq_precision, table_precision)
    inverse = util("inverse_quantized_cdf")
    if inverse is None:
        return None
    if table_precision != freq_precision:
        return None  # sparse tables do not exist before v0.3.0
    return torch.cat([cdfs, inverse(cdfs, freq_precision=freq_precision)], dim=-1)


def timed(device, make_batch, body, repeats, runs, warmup, spin_seconds):
    """Best-of-`runs` seconds per call; `body` runs on `repeats` fresh clones."""
    for _ in range(warmup):
        body(*make_batch())
    if device == "cuda":
        torch.cuda.synchronize()

    if spin_seconds > 0:  # let the GPU clocks settle before the first timing
        deadline = time.perf_counter() + spin_seconds
        while time.perf_counter() < deadline:
            body(*make_batch())
        if device == "cuda":
            torch.cuda.synchronize()

    events = None
    if device == "cuda":
        events = (torch.cuda.Event(enable_timing=True),
                  torch.cuda.Event(enable_timing=True))

    best = None
    for _ in range(runs):
        args = make_batch()
        if device == "cuda":
            torch.cuda.synchronize()
            events[0].record()
            body(*args)
            events[1].record()
            torch.cuda.synchronize()
            secs = events[0].elapsed_time(events[1]) / 1000.0 / repeats
        else:
            t0 = time.perf_counter()
            body(*args)
            secs = (time.perf_counter() - t0) / repeats
        best = secs if best is None else min(best, secs)
    return best


# --------------------------------------------------------------------------- #
# section 1: raw codec operators
# --------------------------------------------------------------------------- #
def run_codec(device, rows, symbols_per_row, args, records):
    freq_precision = min(FREQ_PRECISION, 15)
    _, cdfs, cdfs_sizes, offsets = build_params(freq_precision)
    data, indexes = build_data(rows, symbols_per_row)

    cdfs = cdfs.to(device)
    cdfs_sizes = cdfs_sizes.to(device)
    offsets = offsets.to(device)
    data_d = data.to(device)
    indexes_d = indexes.to(device)

    alias_pair = None
    build_alias = native("rans_alias_build_table")
    if build_alias is not None:
        # the table builder is a CPU implementation: build it before the
        # tensors are moved to the device
        alias_pair = build_alias(cdfs.cpu(), cdfs_sizes.cpu(),
                                 symbol_precision=SYMBOL_PRECISION,
                                 freq_precision=freq_precision)
        alias_pair = tuple(t.contiguous().to(device) for t in alias_pair)

    total_symbols = rows * symbols_per_row

    for family in FAMILIES:
        for interleaves in args.interleaves:
            for lookup in args.lookups:
                push_name, pop_name = op_names(family, interleaves, lookup)
                push, pop = native(push_name), native(pop_name)
                rec = {
                    "section": "codec", "device": device, "family": family,
                    "interleaves": interleaves, "lookup": lookup,
                    "freq_precision": freq_precision, "rows": rows,
                    "symbols_per_row": symbols_per_row,
                    "push_op": push_name, "pop_op": pop_name,
                }
                if push is None or pop is None:
                    rec.update(status="unsupported", encode=None, decode=None,
                               encoded_bytes=None)
                    records.append(rec)
                    print(f"{family:<10} x{interleaves:<3} {lookup:<14} "
                          f"unsupported")
                    continue
                if lookup == "alias" and alias_pair is None:
                    rec.update(status="unsupported", encode=None, decode=None,
                               encoded_bytes=None)
                    records.append(rec)
                    continue

                if lookup == "alias":
                    push_cdfs, pop_cdfs = alias_pair
                elif lookup == "binary":
                    push_cdfs = pop_cdfs = cdfs
                else:
                    table_precision = (freq_precision if lookup == "invcdf_dense"
                                       else args.sparse_q)
                    push_cdfs = cdfs
                    pop_cdfs = build_inverse_table(cdfs, freq_precision,
                                                   table_precision)
                    if pop_cdfs is None:
                        rec.update(status="unsupported", encode=None,
                                   decode=None, encoded_bytes=None)
                        records.append(rec)
                        print(f"{family:<10} x{interleaves:<3} {lookup:<14} "
                              f"unsupported")
                        continue

                init_stream = native(f"{family}_init_stream")
                common = dict(symbol_precision=SYMBOL_PRECISION,
                              freq_precision=freq_precision,
                              bypass_coding=True,
                              bypass_precision=BYPASS_PRECISION)

                def encode(stream_):
                    push(stream_, data_d, indexes_d, push_cdfs, cdfs_sizes,
                         offsets, **common)

                def decode(stream_):
                    kwargs = dict(common)
                    if lookup.startswith("invcdf") and supports_inverse_cdf_arg(pop):
                        # v0.2.1 has no sparse tables and no such argument:
                        # its dense table is implied by the appended table.
                        kwargs["inverse_cdf_precision"] = (
                            freq_precision if lookup == "invcdf_dense"
                            else args.sparse_q)
                    return pop(stream_, indexes_d, pop_cdfs, cdfs_sizes,
                               offsets, **kwargs)

                def make_streams(encoded=None):
                    base = encoded if encoded is not None else init_stream(
                        rows, interleaves).to(device)
                    return [[base.clone() for _ in range(args.repeats)]]

                try:
                    stream = init_stream(rows, interleaves).to(device)
                    encode(stream)
                    if device == "cuda":
                        torch.cuda.synchronize()
                    encoded = stream
                    out = decode(encoded.clone())
                    if device == "cuda":
                        torch.cuda.synchronize()
                    if not torch.equal(out.cpu(), data):
                        raise RuntimeError("decoded symbols differ from input")
                except Exception as exc:  # unsupported / broken combination
                    rec.update(status=f"error: {str(exc).splitlines()[0][:120]}",
                               encode=None, decode=None, encoded_bytes=None)
                    records.append(rec)
                    print(f"{family:<10} x{interleaves:<3} {lookup:<14} "
                          f"error: {str(exc).splitlines()[0][:60]}")
                    continue

                encoded_bytes = float(encoded[:, 0].sum().cpu().item())

                def enc_body(clones):
                    for s in clones:
                        encode(s)

                def dec_body(clones):
                    for s in clones:
                        decode(s)

                enc_s = timed(device, lambda: make_streams(), enc_body,
                              args.repeats, args.runs, args.warmup,
                              args.gpu_spin if device == "cuda" else 0.0)
                dec_s = timed(device, lambda: make_streams(encoded), dec_body,
                              args.repeats, args.runs, args.warmup, 0.0)

                rec.update(status="ok", encode=total_symbols / enc_s / 1e6,
                           decode=total_symbols / dec_s / 1e6,
                           encoded_mb=encoded_bytes / 1e6)
                records.append(rec)
                print(f"{family:<10} x{interleaves:<3} {lookup:<14} "
                      f"enc {rec['encode']:>8.1f} Msym/s  "
                      f"dec {rec['decode']:>8.1f} Msym/s")


# --------------------------------------------------------------------------- #
# section 2: parameter build
# --------------------------------------------------------------------------- #
def run_params(device, args, records):
    pmf, cdfs, _, _ = build_params(FREQ_PRECISION)
    pmf = pmf.to(device)
    cdfs_dev = cdfs.to(device)

    def measure(fn, runs=20, warmup=5):
        for _ in range(warmup):
            fn()
        if device == "cuda":
            torch.cuda.synchronize()
        best = None
        for _ in range(runs):
            if device == "cuda":
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                fn()
                end.record()
                torch.cuda.synchronize()
                secs = start.elapsed_time(end) / 1000.0
            else:
                t0 = time.perf_counter()
                fn()
                secs = time.perf_counter() - t0
            best = secs if best is None else min(best, secs)
        return best * 1e3  # ms

    to_cdf = native("rans_pmf_to_quantized_cdf")
    ms = measure(lambda: to_cdf(pmf, FREQ_PRECISION))
    records.append(dict(section="params", device=device, step="pmf_to_cdf",
                        table_precision=None, ms=ms, status="ok"))
    print(f"{'pmf_to_cdf':<28} {ms:>8.3f} ms")

    for label, q in (("invcdf_dense", FREQ_PRECISION),
                     ("invcdf_sparse", args.sparse_q)):
        builder = native("rans_build_inverse_cdf")
        if builder is not None:
            fn = lambda q=q: builder(cdfs_dev, FREQ_PRECISION, q)
        else:
            inverse = util("inverse_quantized_cdf")
            if inverse is None or q != FREQ_PRECISION:
                records.append(dict(section="params", device=device, step=label,
                                    table_precision=q, ms=None,
                                    status="unsupported"))
                print(f"{label:<28} unsupported")
                continue
            fn = lambda: inverse(cdfs_dev, freq_precision=FREQ_PRECISION)
        ms = measure(fn)
        records.append(dict(section="params", device=device, step=label,
                            table_precision=q, ms=ms, status="ok"))
        print(f"{label + f' (q={q})':<28} {ms:>8.3f} ms")


# --------------------------------------------------------------------------- #
# section 3: high level API
# --------------------------------------------------------------------------- #
def run_high_level(device, rows, symbols_per_row, args, records):
    from torch_ans.utils import TorchANSInterface

    freq_precision = FREQ_PRECISION
    _, _, _, _ = build_params(freq_precision)
    data, indexes = build_data(rows, symbols_per_row)
    data = data.to(device)
    indexes = indexes.to(device)

    # pmf in "counts" form for init_params
    g = torch.Generator().manual_seed(SEED)
    freqs = torch.randint(1, 4096, (NUM_DISTS, ALPHABET + 1), generator=g)
    num_freqs = torch.full((NUM_DISTS,), ALPHABET + 1, dtype=torch.int32)
    offsets = torch.zeros(NUM_DISTS, dtype=torch.int32)

    variants = [
        ("rans64 default", dict(impl="rans64")),
        ("rans32_16 default", dict(impl="rans32_16")),
    ]
    if args.best_effort:
        variants.append((
            "rans32_16 i4 + auto invcdf",
            dict(impl="rans32_16", num_interleaves=4,
                 inverse_cdf_precision="auto"),
        ))

    for label, kwargs in variants:
        rec = {"section": "high_level", "device": device, "variant": label}
        try:
            coder = TorchANSInterface(freq_precision=freq_precision,
                                      device=device, mode="encdec",
                                      **kwargs)
            coder.init_params(freqs, num_freqs, offsets)
            encoded = coder.encode_with_indexes(data, indexes)
            decoded = coder.decode_with_indexes(encoded, indexes)
            if device == "cuda":
                torch.cuda.synchronize()
            if not torch.equal(decoded.to(data.dtype).cpu(), data.cpu()):
                raise RuntimeError("roundtrip mismatch")
        except Exception as exc:
            rec.update(status=f"error: {str(exc).splitlines()[0][:120]}",
                       encode=None, decode=None)
            records.append(rec)
            print(f"{label:<28} error: {str(exc).splitlines()[0][:60]}")
            continue

        def enc():
            coder.encode_with_indexes(data, indexes)

        def dec():
            coder.decode_with_indexes(encoded, indexes)

        total = rows * symbols_per_row
        enc_s = timed(device, lambda: [None], lambda _: enc(), 1, args.runs,
                      args.warmup, args.gpu_spin if device == "cuda" else 0.0)
        dec_s = timed(device, lambda: [None], lambda _: dec(), 1, args.runs,
                      args.warmup, 0.0)
        rec.update(status="ok", encode=total / enc_s / 1e6,
                   decode=total / dec_s / 1e6)
        records.append(rec)
        print(f"{label:<28} enc {rec['encode']:>8.1f} Msym/s  "
              f"dec {rec['decode']:>8.1f} Msym/s")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version-label", default="unknown",
                   help="version tag written into every record")
    p.add_argument("--json", default=None, help="JSON output path")
    p.add_argument("--devices", nargs="+", default=None)
    p.add_argument("--interleaves", nargs="+", type=int, default=[1, 4, 32])
    p.add_argument("--lookups", nargs="+",
                   default=["binary", "invcdf_dense", "invcdf_sparse", "alias"])
    p.add_argument("--sparse-q", type=int, default=7,
                   help="sparse inverse-CDF table precision (v0.3.0 only)")
    p.add_argument("--rows-cpu", type=int, default=48)
    p.add_argument("--rows-cuda", type=int, default=2048)
    p.add_argument("--symbols-per-row", type=int, default=8192)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--gpu-spin", type=float, default=1.0,
                   help="seconds of untimed GPU work before the first timing")
    p.add_argument("--sections", nargs="+",
                   default=["codec", "params", "high_level"])
    p.add_argument("--best-effort", action="store_true",
                   help="also measure the fastest v0.3.0 configuration")
    args = p.parse_args()

    devices = args.devices or (["cpu", "cuda"] if torch.cuda.is_available()
                               else ["cpu"])
    devices = [d for d in devices if d == "cpu" or torch.cuda.is_available()]

    import torch_ans

    records = []
    print(f"=== torch_ans {args.version_label} | torch {torch.__version__} | "
          f"python {platform.python_version()} | threads {torch.get_num_threads()}")
    print(f"    package: {torch_ans.__file__}")
    for device in devices:
        rows = args.rows_cpu if device == "cpu" else args.rows_cuda
        print(f"--- device={device} rows={rows} "
              f"symbols/row={args.symbols_per_row}")
        if "codec" in args.sections:
            run_codec(device, rows, args.symbols_per_row, args, records)
        if "params" in args.sections:
            run_params(device, args, records)
        if "high_level" in args.sections:
            run_high_level(device, rows, args.symbols_per_row, args, records)
        print()

    payload = {
        "version": args.version_label,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cpu": cpu_model(),
        "sparse_q": args.sparse_q,
        "records": records,
    }
    if args.json:
        with open(args.json, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
