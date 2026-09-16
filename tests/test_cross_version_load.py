"""One prebuilt `.so`, every `torch >= 2.10`.

This guards the claim the migration exists for: the wheel is compiled once
(`TORCH_TARGET_VERSION` = torch 2.10, see `setup.py`) and has to load and code
*identically* under any later torch. Nothing here needs a compiler and nothing
rebuilds anything - it goes through the prebuilt `_C_cpu.so`/`_C_cuda.so` via
`torch_ans/_stable_C.py`, exactly like an installed wheel does.

One process cannot see two torch versions, so there are two entry points:

* pytest - the probe runs against the prebuilt extension and round-trips, and when
  `TORCH_ANS_CROSS_VERSION_REFERENCE` names a JSON file produced elsewhere it
  compares its digest with that reference (that is how a local check works);
* command line, used by the `cross-version-load` CI job - ``--json out.json``
  writes the probe result for one torch version, ``--compare a.json b.json ...``
  asserts that every digest agrees and fails the job when they do not.

The parameter set and the digest are the POC's (`poc/stable_abi/tests/
cross_version_probe.py`), kept byte-compatible: a hand-built CDF, out-of-range
symbols (bypass coding), and the coded part of every row hashed.

Imports the package the way anything else does, so CI exercises the *installed
wheel*; in a source checkout that is not installed, run it as
``PYTHONPATH=. python tests/test_cross_version_load.py ...`` (``python -m pytest``
already puts the checkout root on the path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

import pytest
import torch

from torch_ans import _stable_C

#: Where the probe result of another torch version is, for the pytest entry point.
REFERENCE_ENV_VAR = "TORCH_ANS_CROSS_VERSION_REFERENCE"

#: The family every flavour of the wheel provides, at the highest freq_precision
#: it supports (15) - so the same probe works on a CPU-only wheel too.
FAMILY = "rans32_16"
FREQ_PRECISION = 15
NUM_DISTS = 8
NUM_SYMBOLS = 64
BATCH = 4
SEQ = 1000

#: Loading the prebuilt extension registers operators process-wide: one answer
#: per session, like tests/test_stable_abi.py.
_STABLE = _stable_C.load_stable()

needs_stable = pytest.mark.skipif(
    _STABLE is None,
    reason="no prebuilt stable-ABI extension found (build one with "
           "WITH_STABLE_ABI=1, or point TORCH_ANS_STABLE_LIB_DIR at one)")


def build_params():
    """A batch plus a CDF the probe can use with nothing but torch.

    A hand-built quantized CDF, as in the POC probe: it has to be strictly
    increasing and end at exactly ``2 ** freq_precision`` (a decreasing CDF makes
    a frequency negative, which the encoder cannot recover from), and the symbols
    deliberately run past the alphabet so bypass coding is exercised too.
    """
    torch.manual_seed(1234)
    total = 1 << FREQ_PRECISION
    inc = torch.randint(1, total // (2 * NUM_SYMBOLS), (NUM_DISTS, NUM_SYMBOLS),
                        dtype=torch.int64)
    cum = torch.cumsum(inc, dim=1)
    cdfs = torch.zeros((NUM_DISTS, NUM_SYMBOLS + 2), dtype=torch.int32)
    cdfs[:, 1:-1] = cum.to(torch.int32)
    cdfs[:, -1] = total
    assert bool((cdfs[:, 1:] > cdfs[:, :-1]).all()), "cdf must be strictly increasing"
    cdfs_sizes = torch.full((NUM_DISTS,), NUM_SYMBOLS + 2, dtype=torch.int32)
    offsets = torch.randint(-8, 8, (NUM_DISTS,), dtype=torch.int32)
    indexes = torch.randint(0, NUM_DISTS, (BATCH, SEQ), dtype=torch.int32)
    symbols = torch.randint(-6, NUM_SYMBOLS + 6, (BATCH, SEQ), dtype=torch.int32)
    return symbols, indexes, cdfs, cdfs_sizes, offsets


def stream_digest(stream) -> str:
    """Hash the coded part of every row (the length word plus what it counts)."""
    data = stream.contiguous().cpu()
    lengths = data[:, 0].tolist()
    raw = data.view(torch.uint8)
    out = []
    for row, length in enumerate(lengths):
        # .tolist() rather than .numpy(): the probe must not need numpy
        out.append(bytes(raw[row, 4:4 + int(length)].tolist()))
    return hashlib.sha256(b"".join(out)).hexdigest()


def probe() -> dict:
    """Encode and decode the fixed batch through the prebuilt extension."""
    module = _stable_C.load_stable()
    if module is None:
        raise RuntimeError(
            "no prebuilt stable-ABI extension: build one with WITH_STABLE_ABI=1 "
            "or point TORCH_ANS_STABLE_LIB_DIR at a directory containing "
            "_C_cpu* / _C_cuda*")
    symbols, indexes, cdfs, cdfs_sizes, offsets = build_params()
    init_op = getattr(module, f"{FAMILY}_init_stream")
    push_op = getattr(module, f"{FAMILY}_push")
    pop_op = getattr(module, f"{FAMILY}_pop")

    stream = init_op(BATCH)
    push_op(stream, symbols, indexes, cdfs, cdfs_sizes, offsets,
            symbol_precision=8, freq_precision=FREQ_PRECISION)
    # Sample the coded stream *before* decoding: the pop rewinds the length word,
    # so a digest taken afterwards would only cover the decoder's leftovers.
    coded_lengths = [int(value) for value in stream[:, 0].tolist()]
    digest = stream_digest(stream)
    stream_shape = [int(value) for value in stream.shape]

    decoded = pop_op(stream, indexes, cdfs, cdfs_sizes, offsets,
                     symbol_precision=8, freq_precision=FREQ_PRECISION)

    return {
        "torch": torch.__version__,
        "extension": list(getattr(module, "_loaded_paths", ())),
        "with_cuda": bool(getattr(module, "_torch_ans_with_cuda", False)),
        "family": FAMILY,
        "freq_precision": FREQ_PRECISION,
        "stream_shape": stream_shape,
        "coded_lengths": coded_lengths,
        "digest": digest,
        "roundtrip": bool(torch.equal(decoded.reshape(symbols.shape), symbols)),
    }


@needs_stable
def test_the_prebuilt_extension_round_trips_the_probe_batch():
    result = probe()
    assert result["roundtrip"], "the prebuilt extension could not decode its own stream"
    assert len(result["digest"]) == 64
    # The digest has to cover real coding, not just the state word: SEQ symbols at
    # FREQ_PRECISION bits each can never be a handful of bytes.
    assert min(result["coded_lengths"]) > SEQ, result["coded_lengths"]


@needs_stable
def test_the_digest_matches_the_cross_version_reference():
    """Compare with a probe recorded under another torch (`--json`)."""
    reference_path = os.getenv(REFERENCE_ENV_VAR)
    if not reference_path:
        pytest.skip(f"set {REFERENCE_ENV_VAR} to a probe JSON to compare against")
    with open(reference_path) as handle:
        reference = json.load(handle)

    result = probe()
    assert result["digest"] == reference["digest"], (
        f"the same prebuilt .so coded differently under torch {result['torch']} "
        f"and torch {reference['torch']}: {result['digest']} != {reference['digest']}")
    assert result["coded_lengths"] == reference["coded_lengths"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", default=None,
                        help="write this torch version's probe result here")
    parser.add_argument("--compare", nargs="+", default=None, metavar="JSON",
                        help="assert that every listed probe result has the same "
                             "digest (the .so was not rebuilt between them)")
    args = parser.parse_args(argv)

    if args.compare:
        results = {}
        for path in args.compare:
            with open(path) as handle:
                results[path] = json.load(handle)
        for path, entry in results.items():
            print(f"{path}: torch {entry['torch']} digest {entry['digest']}")
        digests = {entry["digest"] for entry in results.values()}
        if len(digests) != 1:
            print(f"FAIL: {len(digests)} different digests from one .so: "
                  f"{sorted(digests)}")
            return 1
        print(f"PASS: {len(results)} torch versions, one digest ({digests.pop()})")
        return 0

    result = probe()
    print(f"torch          : {result['torch']}")
    print(f"extension      : {', '.join(result['extension']) or '<none>'}")
    print(f"stream shape   : {tuple(result['stream_shape'])}")
    print(f"coded lengths  : {result['coded_lengths']}")
    print(f"stream digest  : {result['digest']}")
    print(f"round trip     : {result['roundtrip']}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump(result, handle, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
