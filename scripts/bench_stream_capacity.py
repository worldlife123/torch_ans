"""Per-call cost of the stream-capacity contract (STABLE_ABI_MIGRATION.md
section 7, decision B): reserved capacity vs the wrapper growing the stream.

Both arms run on the *same* kernel and the same pre-grown stream; the only
difference is whether the call carries `used_bytes`:

  legacy    `used_bytes` left at its default (-1): the wrapper reads the length
            word back (a blocking device->host copy on CUDA) and resizes the
            stream when it does not fit;
  reserved  `used_bytes` is the upper bound Python tracks, so the wrapper only
            compares it against the stream's column count.

The delta is what the migration buys per call - the same thing
`tests/test_capacity_contract.py` proves is still bit-exact.

Usage::

    python scripts/bench_stream_capacity.py                          # CPU, all sizes
    python scripts/bench_stream_capacity.py --device cuda            # CUDA
    TORCH_ANS_STABLE_LIB_DIR=build/stable_abi python scripts/bench_stream_capacity.py \
        --device cuda --json bench_capacity.json

The prebuilt stable-ABI libraries are the interesting target (`_C_cpu.so` /
`_C_cuda.so`); without them the script measures this release's runtime build, and
fails on CUDA unless that build has CUDA support.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Measure the checkout this script lives in, not whatever `torch_ans` happens to
# be installed in site-packages (scripts/ is put on sys.path first, which would
# otherwise pick the installed copy).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

from torch_ans import _stable_C, utils  # noqa: E402

#: The family the CUDA side of the migration targets (32-way warp path included).
FAMILY = "rans32_16"
FREQ_PRECISION = 8
ALPHABET = 8
NUM_DISTS = 4
BATCH = 8


def _op_names(num_interleaves):
    suffix = "" if num_interleaves == 1 else f"_i{num_interleaves}"
    return f"{FAMILY}_init_stream", f"{FAMILY}{suffix}_push"


def _module(device):
    stable = _stable_C.load_stable(require_cuda=(device == "cuda"))
    if stable is not None:
        return stable, "stable-ABI"
    from torch_ans import _C as native
    return native, "runtime/ATen"


def _params(device, num_symbols):
    generator = torch.Generator().manual_seed(0)
    scale = 1 << FREQ_PRECISION
    cdf = torch.arange(ALPHABET + 1, dtype=torch.int64) * scale // ALPHABET
    cdf[-1] = scale
    cdfs = cdf.to(torch.int32).unsqueeze(0).repeat(NUM_DISTS, 1)
    cdfs_sizes = torch.full((NUM_DISTS,), ALPHABET + 1, dtype=torch.int32)
    offsets = torch.zeros(NUM_DISTS, dtype=torch.int32)
    symbols = torch.randint(0, ALPHABET, (BATCH, num_symbols), dtype=torch.int32,
                            generator=generator)
    indexes = torch.randint(0, NUM_DISTS, (BATCH, num_symbols), dtype=torch.int32,
                            generator=generator)
    move = (lambda t: t) if device == "cpu" else (lambda t: t.cuda())
    return (move(cdfs), move(cdfs_sizes), move(offsets), move(symbols), move(indexes))


def _time(call, repeats, device):
    if device == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        call()
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / repeats * 1e6


def _measure(module, device, num_symbols, num_interleaves, repeats):
    init_name, push_name = _op_names(num_interleaves)
    init_op = getattr(module, init_name)
    push_op = getattr(module, push_name)
    cdfs, cdfs_sizes, offsets, symbols, indexes = _params(device, num_symbols)
    state_bytes, word_bytes = utils._STREAM_SIZES[FAMILY]

    # Enough room for every repetition, so growth is never what is measured:
    # `_stream_budget` returns (columns, bytes), and the bytes are the upper bound
    # the next push starts from.
    used = state_bytes * num_interleaves
    columns = 0
    for _ in range(repeats):
        columns, _ = utils._stream_budget(
            used, num_symbols, FREQ_PRECISION, True, 4, num_interleaves, word_bytes, 4)
        _, used = utils._stream_budget(used, num_symbols, FREQ_PRECISION, True, 4,
                                       num_interleaves, word_bytes, 4)
    if device == "cuda":
        stream = init_op(BATCH, num_interleaves, 0).cuda()
    else:
        stream = init_op(BATCH, num_interleaves, 0)
    utils._grow_stream(stream, columns)

    def legacy():
        push_op(stream, symbols, indexes, cdfs, cdfs_sizes, offsets,
                8, FREQ_PRECISION, True, 4)

    # the mirror a caller keeps: an upper bound per push, recomputed the same way
    tracker = {"used": state_bytes * num_interleaves}

    def reserved():
        value = tracker["used"]
        push_op(stream, symbols, indexes, cdfs, cdfs_sizes, offsets,
                8, FREQ_PRECISION, True, 4, value)
        tracker["used"] = utils._stream_budget(
            value, num_symbols, FREQ_PRECISION, True, 4, num_interleaves,
            word_bytes, 4)[1]

    # both arms code the same number of symbols; rewind the length word between
    # runs so neither starts from a stream the other has filled
    initial_length = int(stream[:, 0].max().item())
    legacy_us = _time(legacy, repeats, device)
    stream[:, 0] = initial_length
    tracker["used"] = state_bytes * num_interleaves
    reserved_us = _time(reserved, repeats, device)

    return {
        "device": device,
        "family": FAMILY,
        "num_interleaves": num_interleaves,
        "num_symbols": num_symbols,
        "batch": BATCH,
        "repeats": repeats,
        "legacy_us_per_call": round(legacy_us, 2),
        "reserved_us_per_call": round(reserved_us, 2),
        "saved_us_per_call": round(legacy_us - reserved_us, 2),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--symbols", type=int, nargs="+", default=[64, 1024, 8192])
    parser.add_argument("--interleaves", type=int, nargs="+", default=[1])
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--json", default=None, help="also write the results here")
    args = parser.parse_args(argv)

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda needs a GPU")

    module, flavour = _module(args.device)
    print(f"native module: {flavour} ({type(module).__name__}), "
          f"device={args.device}, repeats={args.repeats}")
    rows = []
    for num_interleaves in args.interleaves:
        for num_symbols in args.symbols:
            row = _measure(module, args.device, num_symbols, num_interleaves,
                           args.repeats)
            rows.append(row)
            print(f"  ilv={num_interleaves:<2} symbols={num_symbols:<6} "
                  f"legacy={row['legacy_us_per_call']:>9.2f} us  "
                  f"reserved={row['reserved_us_per_call']:>9.2f} us  "
                  f"saved={row['saved_us_per_call']:>8.2f} us")

    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"flavour": flavour, "rows": rows}, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
