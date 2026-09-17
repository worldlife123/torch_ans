"""The tANS family: `impl="tans32" / "tans16" / "tans8"`.

tANS (table-driven ANS, the algorithm of FSE) differs from rANS in two ways
these tests pin down:

* it is **table driven**. ``tans_build_tables`` turns the quantized cdf into a
  spread table of ``2**freq_precision`` slots per distribution plus the encoder's
  symbol transformation table, and ``push``/``pop`` are then a single table
  access per symbol instead of a search (decode) or a division (encode). The
  table only depends on ``(cdf, table_log)`` - which is also why all three
  families share it - and it carries *both* directions, so the same tensor
  travels in the ``cdfs`` slot of push and pop.
* a tANS state is **fixed width** (``[2**L, 2**(L+1))``), so it cannot absorb
  the bits it does not emit the way a rANS state does. Those bits live in a
  3-word stream header (state, residual bit buffer, bit count) next to the
  length word, which is what keeps a tANS stream in the same
  ``stream[b][0] = byte length`` layout a rANS stream uses: the byte-string
  conversion, the capacity contract and the high-level API are unchanged.

The three families differ only in the state / stream word width, which is also
their ``freq_precision`` cap: ``tans32`` (uint32 + uint16) and ``tans16``
(uint16 + uint8) stop at 15 - the limit of the encoder's 16.16 fixed point -
while ``tans8`` (uint8 + uint8) stops at 7 (its state has to hold
``[2**L, 2**(L+1))``). It is the small-table tier: ``2**7`` slots per
distribution instead of ``2**15``, at a coarser probability resolution.

Every test needs a native build that provides the family. A prebuilt extension
from before tANS existed makes them skip; otherwise the runtime build compiles a
module for the profile on first use (~30 s, then cached in the torch extension
directory).
"""
from __future__ import annotations

import pytest
import torch

from torch_ans import _dynamic_build as build_config
from torch_ans import utils
from torch_ans.utils import TorchANSInterface

#: (family, freq_precision cap) - see rans_build_config.hpp / tans_utils.hpp
FAMILIES = (("tans32", 15), ("tans16", 15), ("tans8", 7))

#: one module (or one failure) per profile, so a family is resolved once per
#: session. Caching the *failure* matters as much as caching the module: with
#: neither a prebuilt extension nor a working toolchain, resolving a family
#: starts a JIT build (~30 s up to a few minutes when it fails), and without this
#: each of the tests below would pay for it again.
_MODULES = {}
_UNRESOLVED = {}


def _tans_module(family):
    """The native module that provides `family`; skips when there is none."""
    if family in _UNRESOLVED:
        pytest.skip(_UNRESOLVED[family])
    module = _MODULES.get(family)
    if module is None:
        try:
            module = utils._native_module(build_config.BuildProfile(family=family))
        except Exception as exc:  # no prebuilt extension and the build failed
            _UNRESOLVED[family] = ("the %s family could not be resolved, so no "
                                   "native build provides it: %s" % (family, exc))
            pytest.skip(_UNRESOLVED[family])
        _MODULES[family] = module
    if not hasattr(module, "%s_push" % family):
        _UNRESOLVED[family] = ("this native build does not provide the %s family "
                              "(a prebuilt torch_ans/_C from before it existed?)"
                              % family)
        pytest.skip(_UNRESOLVED[family])
    return module


def _distribution(module, alphabet, freq_precision, bypass_coding=True, num_dists=3,
                  seed=0, spread=100):
    """A batch of quantized cdfs plus the matching `cdfs_sizes`.

    Built the way `TorchANSInterface.init_params` does: an extra symbol with a
    tail mass when bypass coding is on (the sentinel), then the "steal
    frequency" quantization.
    """
    generator = torch.Generator().manual_seed(seed)
    freqs = torch.randint(1, spread, (num_dists, alphabet), generator=generator)
    if bypass_coding:
        freqs = torch.cat([freqs, torch.zeros(num_dists, 1) + 1e-10], dim=-1)
    pmf = freqs.float() / freqs.sum(-1, keepdim=True)
    cdfs = module.rans_pmf_to_quantized_cdf(pmf, freq_precision)
    sizes = torch.full((num_dists,), alphabet + (2 if bypass_coding else 1),
                       dtype=torch.int32)
    return cdfs, sizes


def _roundtrip(family, alphabet, freq_precision, bypass_coding, batch=3,
               num_symbols=257, seed=0):
    """init + build_tables + push + pop; returns (symbols, indexes, decoded, stream)."""
    module = _tans_module(family)
    cdfs, cdfs_sizes = _distribution(module, alphabet, freq_precision, bypass_coding)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, freq_precision)

    generator = torch.Generator().manual_seed(seed)
    indexes = torch.randint(0, cdfs.size(0), (batch, num_symbols), dtype=torch.int32,
                            generator=generator)
    if bypass_coding:
        symbols = torch.randint(-40, alphabet + 40, (batch, num_symbols),
                                dtype=torch.int32, generator=generator)
    else:
        symbols = torch.randint(0, alphabet, (batch, num_symbols), dtype=torch.int32,
                                generator=generator)
    offsets = torch.zeros(cdfs.size(0), dtype=torch.int32)

    stream = getattr(module, family + "_init_stream")(symbols.size(0), 1, 0)
    getattr(module, family + "_push")(stream, symbols, indexes, tables, cdfs_sizes,
                                      offsets, 8, freq_precision, bypass_coding, 4)
    decoded = getattr(module, family + "_pop")(stream, indexes, tables, cdfs_sizes,
                                               offsets, 8, freq_precision, bypass_coding, 4)
    return symbols, indexes, decoded, stream


def _table_invariants(cdfs, cdfs_sizes, tables, table_log):
    """The properties the spread has to have for the coder to be correct.

    * every symbol owns exactly `norm[s] = cdf[s+1] - cdf[s]` slots (the spread
      visits each slot once), which is what `pop`'s symbol plane has to show;
    * `nextState` is a permutation of `[ts, 2*ts)` - the encoder's state range;
    * a decoded state plus the bits it reads stays inside `[0, ts)`, the
      decoder's range (otherwise the state would index out of the table);
    * `deltaNbBits` / `deltaFindState` are the ones the cdf implies.
    """
    ts = 1 << table_log
    max_symbols = cdfs.size(1)
    assert tables.size(1) == 3 * ts + 2 * max_symbols

    for d in range(cdfs_sizes.size(0)):
        row = tables[d]
        next_state = row[0:ts]
        decode_word = row[ts:2 * ts]
        symbol_plane = row[2 * ts:3 * ts]

        assert int(next_state.min()) >= ts and int(next_state.max()) < 2 * ts
        assert torch.equal(next_state.sort().values,
                           torch.arange(ts, 2 * ts, dtype=next_state.dtype))

        num_symbols = int(cdfs_sizes[d]) - 1  # the bypass sentinel is a symbol too
        norms = (cdfs[d, 1:] - cdfs[d, :-1])[:num_symbols].to(torch.int64)
        assert int(norms.sum()) == ts
        counts = torch.bincount(symbol_plane.to(torch.int64),
                                minlength=max_symbols)[:num_symbols].to(torch.int64)
        assert torch.equal(counts, norms), "the spread does not follow the cdf"

        nb_bits = (decode_word >> 16) & 0x1F
        new_state = decode_word & 0xFFFF
        assert int(nb_bits.max()) <= table_log
        # `newState + lowBits` has to stay below ts for every possible `lowBits`
        assert int((new_state + (1 << nb_bits) - 1).max()) < ts

        symbol_tt = row[3 * ts:].view(max_symbols, 2)
        running = 0
        for s in range(num_symbols):
            norm = int(norms[s])
            # highbit(0) == 0 in the builder, which is what makes norm == 1 fall
            # out of the general formula (maxBitsOut = table_log)
            max_bits_out = table_log - ((norm - 1).bit_length() - 1 if norm > 1 else 0)
            min_state_plus = norm << max_bits_out
            assert int(symbol_tt[s, 0]) == (max_bits_out << 16) - min_state_plus
            assert int(symbol_tt[s, 1]) == running - norm
            running += norm


@pytest.mark.parametrize("family,cap", FAMILIES)
@pytest.mark.parametrize("alphabet,bypass", [
    (8, True), (8, False), (2, True), (17, True), (3, False),
])
def test_the_tables_are_a_spread_of_the_cdf(family, cap, alphabet, bypass):
    if alphabet + (2 if bypass else 1) > (1 << cap):
        pytest.skip("the alphabet does not fit the table size")
    module = _tans_module(family)
    cdfs, cdfs_sizes = _distribution(module, alphabet, cap, bypass)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, cap)
    _table_invariants(cdfs, cdfs_sizes, tables, cap)


@pytest.mark.parametrize("family,cap", FAMILIES)
@pytest.mark.parametrize("alphabet,bypass", [
    (2, True), (8, True), (8, False), (17, False), (256, True), (256, False),
])
def test_push_pop_roundtrips(family, cap, alphabet, bypass):
    if alphabet + (2 if bypass else 1) > (1 << cap):
        pytest.skip("the alphabet does not fit the table size")
    symbols, _, decoded, stream = _roundtrip(family, alphabet, cap, bypass)
    assert torch.equal(decoded, symbols)
    # every row still describes a stream, and the length word is a byte count
    assert int(stream[:, 0].min()) > 0
    assert int(stream[:, 0].max()) <= stream.size(1) * stream.element_size()


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_bypass_codes_symbols_outside_the_alphabet(family, cap):
    """The sentinel path: negative values and values above the cdf's range."""
    module = _tans_module(family)
    alphabet, fp = 16, min(cap, 8)
    cdfs, cdfs_sizes = _distribution(module, alphabet, fp, bypass_coding=True)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, fp)
    offsets = torch.tensor([0, 1000, -250], dtype=torch.int32)
    generator = torch.Generator().manual_seed(7)
    # distribution 1 codes [1000, 1015] and distribution 2 [-250, -235], so the
    # same symbols are far out of range for one of them; distribution 0 also sees
    # values below 0 and above 15
    symbols = torch.randint(0, alphabet, (3, 128), dtype=torch.int32,
                            generator=generator) + offsets.unsqueeze(1)
    indexes = torch.arange(3, dtype=torch.int32).unsqueeze(1).expand(3, 128).contiguous()

    stream = getattr(module, family + "_init_stream")(3, 1, 0)
    getattr(module, family + "_push")(stream, symbols, indexes, tables, cdfs_sizes,
                                      offsets, 8, fp, True, 4)
    decoded = getattr(module, family + "_pop")(stream, indexes, tables, cdfs_sizes,
                                               offsets, 8, fp, True, 4)
    assert torch.equal(decoded, symbols)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_padding_indexes_are_skipped(family, cap):
    """The batched API pads with index -1; those lanes must stay untouched."""
    module = _tans_module(family)
    alphabet, fp = 8, min(cap, 8)
    cdfs, cdfs_sizes = _distribution(module, alphabet, fp)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, fp)
    indexes = torch.tensor([[0, 1, -1, 2, -1, 0]], dtype=torch.int32)
    symbols = torch.tensor([[3, 5, 99, 1, 42, 7]], dtype=torch.int32)
    offsets = torch.zeros(3, dtype=torch.int32)

    stream = getattr(module, family + "_init_stream")(1, 1, 0)
    getattr(module, family + "_push")(stream, symbols, indexes, tables, cdfs_sizes,
                                      offsets, 8, fp, True, 4)
    decoded = getattr(module, family + "_pop")(stream, indexes, tables, cdfs_sizes,
                                               offsets, 8, fp, True, 4)
    expected = torch.tensor([[3, 5, 0, 1, 0, 7]], dtype=torch.int32)
    assert torch.equal(decoded, expected)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_two_pushes_share_one_stream(family, cap):
    """A push continues where the previous one stopped, and a pop rewinds.

    The header carries the conversion point, so popping twice walks the two
    pushes back - which is the property rans_pop gets from rewriting the length
    word and the state.
    """
    module = _tans_module(family)
    alphabet, fp = 8, min(cap, 10)
    cdfs, cdfs_sizes = _distribution(module, alphabet, fp)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, fp)
    offsets = torch.zeros(3, dtype=torch.int32)
    generator = torch.Generator().manual_seed(11)
    first = torch.randint(0, alphabet, (2, 64), dtype=torch.int32, generator=generator)
    second = torch.randint(0, alphabet, (2, 64), dtype=torch.int32, generator=generator)
    first_idx = torch.randint(0, 3, (2, 64), dtype=torch.int32, generator=generator)
    second_idx = torch.randint(0, 3, (2, 64), dtype=torch.int32, generator=generator)

    stream = getattr(module, family + "_init_stream")(2, 1, 0)
    push = getattr(module, family + "_push")
    pop = getattr(module, family + "_pop")
    push(stream, first, first_idx, tables, cdfs_sizes, offsets, 8, fp, True, 4)
    length_after_first = int(stream[:, 0].max())
    push(stream, second, second_idx, tables, cdfs_sizes, offsets, 8, fp, True, 4)
    assert int(stream[:, 0].max()) > length_after_first

    # ANS is LIFO: the second push comes back first, then the first one
    assert torch.equal(pop(stream, second_idx, tables, cdfs_sizes, offsets, 8, fp, True, 4),
                       second)
    assert torch.equal(pop(stream, first_idx, tables, cdfs_sizes, offsets, 8, fp, True, 4),
                       first)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_byte_string_roundtrip(family, cap):
    """`flush()`/`set_stream()` go through the byte strings; they must survive.

    Pushed here rather than through `_roundtrip`: that helper pops, and a pop
    rewrites the stream's continuation point (the length word and the header), so
    the byte string of a *decoded* stream is not the encoded one any more.
    """
    module = _tans_module(family)
    fp, alphabet = min(cap, 8), 8
    cdfs, cdfs_sizes = _distribution(module, alphabet, fp, True)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, fp)
    offsets = torch.zeros(3, dtype=torch.int32)
    generator = torch.Generator().manual_seed(13)
    indexes = torch.randint(0, 3, (3, 257), dtype=torch.int32, generator=generator)
    symbols = torch.randint(-40, alphabet + 40, (3, 257), dtype=torch.int32,
                            generator=generator)

    stream = getattr(module, family + "_init_stream")(3, 1, 0)
    getattr(module, family + "_push")(stream, symbols, indexes, tables, cdfs_sizes,
                                      offsets, 8, fp, True, 4)
    byte_strings = module.rans_stream_to_byte_strings(stream)
    # each byte string is exactly the coded bytes (the length word counts the
    # header, which the byte string includes)
    raw = stream.view(torch.uint8).reshape(stream.shape[0], -1)
    lengths = stream[:, 0].tolist()
    for row, byte_string in enumerate(byte_strings):
        assert bytes(byte_string) == bytes(raw[row, 4:4 + int(lengths[row])].tolist())

    restored = module.rans_byte_strings_to_stream(byte_strings)
    decoded = getattr(module, family + "_pop")(restored, indexes, tables, cdfs_sizes,
                                               offsets, 8, fp, True, 4)
    assert torch.equal(decoded, symbols)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_compression_is_close_to_the_entropy(family, cap):
    """A table-driven coder must actually compress: bits/symbol ~ H + epsilon."""
    module = _tans_module(family)
    alphabet, fp, num_symbols = 16, min(cap, 12), 4096
    cdfs, cdfs_sizes = _distribution(module, alphabet, fp, bypass_coding=False,
                                     num_dists=1, spread=1000)
    tables = module.tans_build_tables(cdfs, cdfs_sizes, fp)
    offsets = torch.zeros(1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(3)
    freqs = (cdfs[0, 1:] - cdfs[0, :-1]).to(torch.float64)
    probs = freqs / freqs.sum()
    symbols = torch.multinomial(probs.float(), num_symbols, replacement=True,
                                generator=generator).to(torch.int32).unsqueeze(0)
    indexes = torch.zeros_like(symbols)

    stream = getattr(module, family + "_init_stream")(1, 1, 0)
    getattr(module, family + "_push")(stream, symbols, indexes, tables, cdfs_sizes,
                                      offsets, 8, fp, False, 4)
    # the length counts the 12-byte header; whatever is left is payload words (the
    # residual bits stay in the header and are not part of it)
    coded_bits = (int(stream[0, 0]) - 12) * 8 / num_symbols
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * nonzero.log2()).sum())
    # the spread quantizes each probability to a multiple of 1/2**fp, and the
    # coder is an arithmetic coder over that quantized distribution, so the
    # cross-entropy it achieves sits just above the entropy
    assert entropy - 0.25 <= coded_bits <= entropy + 0.5, (entropy, coded_bits)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_high_level_api(family, cap):
    """`TorchANSInterface` drives tANS exactly like a rANS family."""
    # The interface resolves its operators from `torch_ans._C`, which is the
    # prebuilt extension when there is one - so the skip has to be asked for
    # through the same profile resolution the rest of this file uses.
    _tans_module(family)
    freq_precision = min(cap, 10)
    interface = TorchANSInterface(impl=family, num_interleaves=1,
                                  freq_precision=freq_precision)
    alphabet, num_dists = 8, 4
    generator = torch.Generator().manual_seed(5)
    freqs = torch.randint(1, 50, (num_dists, alphabet), generator=generator).to(torch.int32)
    num_freqs = torch.full((num_dists,), alphabet, dtype=torch.int32)
    offsets = torch.zeros(num_dists, dtype=torch.int32)
    interface.init_params(freqs, num_freqs, offsets)

    symbols = torch.randint(0, alphabet, (2, 128), dtype=torch.int32, generator=generator)
    indexes = torch.randint(0, num_dists, (2, 128), dtype=torch.int32, generator=generator)
    encoded = interface.encode_with_indexes(symbols, indexes)
    assert isinstance(encoded, bytes) and len(encoded) > 0
    assert torch.equal(interface.decode_with_indexes(encoded, indexes), symbols)

    # the set_stream/decode_stream path
    interface.set_stream(encoded)
    assert torch.equal(interface.decode_stream(indexes), symbols)

    # the dist_freqs pattern (a distribution per symbol)
    flat_indexes = indexes.reshape(-1)
    per_symbol_freqs = freqs[flat_indexes].reshape(2, 128, alphabet)
    per_symbol_num = num_freqs[flat_indexes].reshape(2, 128)
    per_symbol_offsets = offsets[flat_indexes].reshape(2, 128)
    encoded = interface.encode_with_freqs(symbols, per_symbol_freqs, per_symbol_num,
                                          per_symbol_offsets)
    decoded = interface.decode_with_freqs(encoded, per_symbol_freqs, per_symbol_num,
                                          per_symbol_offsets)
    assert torch.equal(decoded, symbols)


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_the_family_caps_freq_precision(family, cap):
    _tans_module(family)  # see test_high_level_api
    interface = TorchANSInterface(impl=family, num_interleaves=1, freq_precision=16)
    assert interface.freq_precision == cap


@pytest.mark.parametrize("family,cap", FAMILIES)
def test_unsupported_options_are_rejected(family, cap):
    """tANS has no interleaving, no alias table and no inverse-CDF variant."""
    with pytest.raises(NotImplementedError, match="num_interleaves=2"):
        TorchANSInterface(impl=family, num_interleaves=2)
    with pytest.raises(NotImplementedError, match="alias sampling"):
        TorchANSInterface(impl=family, num_interleaves=1, alias_sampling=True)
    with pytest.raises(NotImplementedError, match="inverse CDF"):
        TorchANSInterface(impl=family, num_interleaves=1, inverse_cdf_precision=4)


def test_the_table_builder_validates_its_input():
    module = _tans_module("tans32")
    cdfs, sizes = _distribution(module, 8, 10)
    # a cdf quantized with another precision does not sum to 2**table_log
    with pytest.raises((ValueError, RuntimeError), match="sums to"):
        module.tans_build_tables(cdfs, sizes, 11)
    # a symbol with no probability mass has no slot to be coded with
    broken = cdfs.clone()
    broken[0, 1] = broken[0, 0]
    with pytest.raises((ValueError, RuntimeError), match="no probability mass"):
        module.tans_build_tables(broken, sizes, 10)


def test_table_log_above_the_fixed_point_limit_is_rejected():
    module = _tans_module("tans32")
    cdfs, sizes = _distribution(module, 8, 10)
    with pytest.raises((ValueError, RuntimeError), match="table_log"):
        module.tans_build_tables(cdfs, sizes, 16)


def test_num_interleaves_other_than_one_is_rejected():
    module = _tans_module("tans32")
    with pytest.raises((ValueError, RuntimeError), match="num_interleaves=1"):
        module.tans32_init_stream(2, 4, 0)


def test_a_cuda_stream_is_reported_not_silently_mis_coded():
    """tANS is CPU-only for now: device tensors must produce a clear message."""
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU to build a CUDA tensor")
    module = _tans_module("tans32")
    cdfs, sizes = _distribution(module, 8, 10)
    tables = module.tans_build_tables(cdfs, sizes, 10)
    stream = module.tans32_init_stream(1, 1, 0).cuda()
    symbols = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    indexes = torch.zeros((1, 4), dtype=torch.int32, device="cuda")
    offsets = torch.zeros(3, dtype=torch.int32, device="cuda")
    with pytest.raises((RuntimeError, ValueError), match="CPU-only"):
        module.tans32_push(stream, symbols, indexes, tables.cuda(), sizes.cuda(),
                           offsets, 8, 10, True, 4)
