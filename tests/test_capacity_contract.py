"""The stream-capacity contract (STABLE_ABI_MIGRATION.md section 7, option B).

`TorchANSInterface` reserves the capacity of a stream in Python and the native
push only *compares* it (the `used_bytes` parameter), which is what keeps both the
per-push read of the length word (a blocking device->host copy on CUDA) and
`aten::resize_` out of the prebuilt extension. That leaves one thing to get right:
Python's worst case has to match the one the wrapper would have computed. If it is
smaller the kernel writes past the row; if the formula drifts the other way a
perfectly good call is rejected.

These tests pin the two computations against each other (the wrapper's own growth
is the oracle: it allocates exactly what its formula asks for), pin the coded
bytes of the reserved path against the legacy one, and check the mirror - the
upper bound of the bytes coded into the longest row - against the stream itself.
"""
from __future__ import annotations

import pytest
import torch

from torch_ans import utils
from torch_ans.utils import TorchANSInterface

BATCH = 2
SEQ = 32
NUM_DISTS = 4
ALPHABET = 16

#: (family, interleaves) pairs whose CPU push is always compiled; the 32-way
#: variant is CUDA-only in practice and is covered by the CUDA test below.
FAMILIES = [("rans64", 1), ("rans64", 4), ("rans32", 1),
            ("rans32_16", 1), ("rans32_16", 4)]


def _setup(impl="rans64", num_interleaves=1, freq_precision=12, device="cpu", **kwargs):
    torch.manual_seed(0)
    pmf = torch.randint(1, 100, (NUM_DISTS, ALPHABET), dtype=torch.int32)
    num_freqs = torch.full((NUM_DISTS,), ALPHABET, dtype=torch.int32)
    offsets = torch.zeros(NUM_DISTS, dtype=torch.int32)
    coder = TorchANSInterface(impl=impl, num_interleaves=num_interleaves,
                              freq_precision=freq_precision, device=device, **kwargs)
    coder.init_params(pmf, num_freqs, offsets)
    symbols = torch.randint(0, ALPHABET, (BATCH, SEQ), dtype=torch.int32, device=device)
    indexes = torch.randint(0, NUM_DISTS, (BATCH, SEQ), dtype=torch.int32, device=device)
    return coder, symbols, indexes


def _legacy_push(coder, stream, symbols, indexes):
    """Push without the capacity parameter: the wrapper grows the stream itself."""
    coder.ans_encode_func(
        stream, symbols, indexes, coder.cdfs, coder.cdfs_sizes, coder.offsets,
        freq_precision=coder.freq_precision, bypass_coding=coder.bypass_coding,
        bypass_precision=coder.bypass_precision)


def _push(coder, stream, symbols, indexes):
    """Push through the interface, i.e. with the capacity contract."""
    coder._encode_push(stream, symbols, indexes, coder.cdfs, coder.cdfs_sizes,
                       coder.offsets)


def _used_bytes(stream):
    """The length word of the longest row: a byte count."""
    return int(stream[:, 0].max().item())


def _coded_rows(stream):
    """Per row: the length word plus the bytes it counts (no padding)."""
    lengths = stream[:, 0].tolist()
    raw = stream.view(torch.uint8).reshape(stream.shape[0], -1)
    return [bytes(raw[row, 4:4 + int(length)].tolist())
            for row, length in enumerate(lengths)]


def _budget(coder, used_bytes, num_symbols, stream):
    """The interface's own worst-case computation, for the assertions here."""
    _, word_bytes = coder._stream_sizes()
    return utils._stream_budget(
        used_bytes, num_symbols, coder.freq_precision, coder.bypass_coding,
        coder.bypass_precision, coder.num_interleaves, word_bytes,
        stream.element_size())


def _requires_capacity_parameter(coder):
    if not coder._push_takes_used_bytes:
        pytest.skip("this native extension predates the used_bytes parameter")


def _cuda_capable_native_available() -> bool:
    """Whether a CUDA-capable extension is already loaded/loadable.

    Deliberately only *looks*: asking an interface for a CUDA module when none is
    there starts a runtime build (minutes, and it can legitimately fail on a
    machine whose nvcc does not match its torch), which is not what this test is
    about. Evaluated at collection time, like `tests/test_stable_abi.py` does for
    the prebuilt extension.
    """
    from torch_ans import _stable_C
    try:
        if _stable_C.load_stable(require_cuda=True) is not None:
            return True
    except Exception:
        pass
    try:
        from torch_ans import _C as compiled
    except Exception:
        return False
    return utils.module_has_cuda(compiled) is True


@pytest.mark.parametrize("impl,num_interleaves", FAMILIES)
def test_python_budget_matches_the_capacity_the_wrapper_allocates(impl, num_interleaves):
    """The wrapper's own growth is the oracle for Python's arithmetic."""
    coder, symbols, indexes = _setup(impl=impl, num_interleaves=num_interleaves)
    stream = coder._init_stream(num_parallel_states=BATCH)
    initial_capacity = int(stream.shape[1])
    initial_used = _used_bytes(stream)

    _legacy_push(coder, stream, symbols, indexes)

    expected_columns, expected_bytes = _budget(coder, initial_used, symbols.size(1),
                                               stream)
    # the wrapper resized to exactly its own worst case, unless the stream was
    # already bigger
    assert int(stream.shape[1]) == max(initial_capacity, expected_columns)
    # ... and the length word can never exceed the byte count used to size it
    assert _used_bytes(stream) <= expected_bytes


@pytest.mark.parametrize("impl,num_interleaves", FAMILIES)
def test_the_reserved_path_codes_the_same_bytes_as_the_growing_one(impl, num_interleaves):
    """Reserving in Python must not change a single coded byte."""
    coder, symbols, indexes = _setup(impl=impl, num_interleaves=num_interleaves)
    _requires_capacity_parameter(coder)

    # the oracle: the wrapper reads the length word and grows the stream itself
    reference = coder._init_stream(num_parallel_states=BATCH)
    reference_initial = int(reference.shape[1])
    _legacy_push(coder, reference, symbols, indexes)

    # the contract: Python reserves, the wrapper only compares
    stream = coder._init_stream(num_parallel_states=BATCH)
    initial_used = _used_bytes(stream)
    _push(coder, stream, symbols, indexes)

    expected_columns, _ = _budget(coder, initial_used, symbols.size(1), stream)
    assert int(stream.shape[1]) == max(reference_initial, expected_columns)
    assert _coded_rows(stream) == _coded_rows(reference)

    # the mirror describes the stream it was adopted for: an upper bound, and the
    # exact number the next push has to build on
    assert coder._mirrored_stream is stream
    assert coder._mirrored_used_bytes >= _used_bytes(stream)


def test_a_stream_that_is_too_small_is_reported_not_overrun():
    """`used_bytes` is a promise: breaking it must fail loudly, not corrupt."""
    coder, symbols, indexes = _setup()
    _requires_capacity_parameter(coder)
    stream = coder._init_stream(num_parallel_states=BATCH)

    with pytest.raises(ValueError, match="caller owns the stream capacity"):
        coder.ans_encode_func(
            stream, symbols, indexes, coder.cdfs, coder.cdfs_sizes, coder.offsets,
            freq_precision=coder.freq_precision, bypass_coding=coder.bypass_coding,
            bypass_precision=coder.bypass_precision,
            used_bytes=_used_bytes(stream))


def test_the_capacity_mirror_stays_an_upper_bound():
    coder, symbols, indexes = _setup()
    stream = coder._init_stream(num_parallel_states=BATCH)
    # a fresh stream is adopted without a read: rans_init_stream's own length word
    assert coder._mirrored_stream is stream
    assert coder._mirrored_used_bytes >= _used_bytes(stream)

    _push(coder, stream, symbols, indexes)
    assert coder._mirrored_used_bytes >= _used_bytes(stream)

    # a push into another stream moves the mirror over (each stream is tracked by
    # identity, so a stream this interface has not coded is read once)
    other = coder._init_stream(num_parallel_states=BATCH)
    _push(coder, other, symbols, indexes)
    assert coder._mirrored_stream is other
    assert coder._mirrored_used_bytes >= _used_bytes(other)

    # the pop rewinds the length word, so the mirror is dropped rather than
    # carried over (which would inflate it on every encode/decode cycle)
    decoded = coder._decode_func(stream, dist_indexes=indexes)
    assert torch.equal(decoded.reshape(BATCH, SEQ), symbols)
    assert coder._mirrored_stream is None

    # ... and the next push adopts it again
    _push(coder, stream, symbols, indexes)
    assert coder._mirrored_stream is stream
    assert coder._mirrored_used_bytes >= _used_bytes(stream)


def test_streaming_pushes_reuse_the_reservation():
    """Two pushes into one stream: the second builds on the first, no re-read."""
    coder, symbols, indexes = _setup()
    _requires_capacity_parameter(coder)
    stream = coder._init_stream(num_parallel_states=BATCH)

    _push(coder, stream, symbols, indexes)
    used_after_first = coder._mirrored_used_bytes
    capacity_after_first = int(stream.shape[1])

    second_symbols = torch.randint(0, ALPHABET, (BATCH, SEQ), dtype=torch.int32)
    second_indexes = torch.randint(0, NUM_DISTS, (BATCH, SEQ), dtype=torch.int32)
    _push(coder, stream, second_symbols, second_indexes)

    expected_columns, expected_bytes = _budget(coder, used_after_first,
                                               second_symbols.size(1), stream)
    assert coder._mirrored_used_bytes == expected_bytes
    assert int(stream.shape[1]) == max(capacity_after_first, expected_columns)
    assert coder._mirrored_used_bytes >= _used_bytes(stream)

    # the last push comes back first (ANS is LIFO)
    decoded = coder._decode_func(stream, dist_indexes=second_indexes)
    assert torch.equal(decoded.reshape(BATCH, SEQ), second_symbols)


def test_an_extension_without_the_parameter_keeps_working(monkeypatch):
    """The contract is opt-in: an older extension grows the stream as before."""
    coder, symbols, indexes = _setup()
    monkeypatch.setattr(coder, "_push_takes_used_bytes", False)
    stream = coder._init_stream(num_parallel_states=BATCH)
    capacity_before = int(stream.shape[1])

    _push(coder, stream, symbols, indexes)

    assert int(stream.shape[1]) > capacity_before, (
        "without the capacity parameter the wrapper has to grow the stream itself")
    decoded = coder._decode_func(stream, dist_indexes=indexes)
    assert torch.equal(decoded.reshape(BATCH, SEQ), symbols)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not _cuda_capable_native_available(),
    reason="needs a GPU and an already available CUDA-capable extension")
def test_the_contract_holds_on_cuda_and_the_32_way_warp_path():
    """The path the contract exists for: the length word is on the device."""
    coder, symbols, indexes = _setup(impl="rans32_16", num_interleaves=32,
                                     device="cuda")
    if utils.module_has_cuda(coder._native) is False:
        pytest.skip("the loaded native extension cannot code CUDA tensors")
    _requires_capacity_parameter(coder)

    stream = coder._init_stream(num_parallel_states=BATCH)
    _push(coder, stream, symbols, indexes)
    assert coder._mirrored_stream is stream
    assert coder._mirrored_used_bytes >= _used_bytes(stream)

    decoded = coder._decode_func(stream, dist_indexes=indexes)
    assert torch.equal(decoded.reshape(BATCH, SEQ).cpu(), symbols.cpu())
