"""Tests for the sparse inverse-CDF symbol lookup.

The decoder turns a cumulative frequency into a symbol index (`cdf_idx`). Three
lookup strategies must agree bit for bit:

  * divided search + binary refine   (default, data dependent O(logN))
  * dense inverse CDF                (O(1), 2**freq_precision table entries)
  * sparse inverse CDF               (this file: 2**inverse_cdf_precision table
                                      entries + a bounded linear walk)

The sparse table stores, for every bucket start ``i << (freq_precision -
inverse_cdf_precision)``, the symbol that owns it. A bucket start never exceeds
the queried cumulative frequency, so the table entry is a lower bound of the
true symbol; the decoder walks forward from there.
"""

import unittest

import torch

from torch_ans._C import (
    rans_pmf_to_quantized_cdf,
    rans32_16_init_stream,
    rans32_16_push,
    rans32_16_pop,
    rans32_16_invcdf_pop,
    rans32_16_i32_push,
    rans32_16_i32_pop,
    rans32_16_i32_invcdf_pop,
)
from torch_ans.utils import TorchANSInterface, auto_inverse_cdf_precision, inverse_quantized_cdf

NUM_DISTS = 6
NUM_SYMBOLS = 9  # in-range values are 0..NUM_SYMBOLS-1


def _generate_rans_params(freq_precision=12, bypass=True, num_symbols=NUM_SYMBOLS,
                          num_dists=NUM_DISTS):
    """Quantized CDFs (with a bypass tail) plus sizes and offsets."""
    g = torch.Generator().manual_seed(1234)
    pmfs = torch.randint(1, 1024, (num_dists, num_symbols), generator=g).float()
    pmfs = torch.cat([pmfs.clone(), torch.ones(num_dists, 1)], dim=1)
    pmfs = pmfs / pmfs.sum(dim=-1, keepdim=True)
    cdfs = rans_pmf_to_quantized_cdf(pmfs, precision=freq_precision)
    cdfs_sizes = torch.zeros(num_dists, dtype=torch.int32) + num_symbols + 2
    if bypass:
        offsets = torch.randint(-4, 4, (num_dists,), generator=g, dtype=torch.int32)
    else:
        offsets = torch.zeros(num_dists, dtype=torch.int32)
    return cdfs, cdfs_sizes, offsets


def _with_inverse_cdf_table(cdfs, freq_precision, table_precision):
    """Appends the (sparse or dense) inverse-CDF table to every CDF row."""
    return torch.cat([cdfs, inverse_quantized_cdf(
        cdfs, freq_precision=freq_precision, table_precision=table_precision)], dim=-1)


def _generate_data(shape, seed, invalid_ratio=0.0, clamp_to_range=False,
                   num_symbols=NUM_SYMBOLS, num_dists=NUM_DISTS):
    """Symbols covering in-range, sentinel and out-of-range (bypass) values."""
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(-6, num_symbols + 10, shape, generator=g, dtype=torch.int32)
    indexes = torch.randint(0, num_dists, shape, generator=g, dtype=torch.int32)
    if clamp_to_range:
        data = data.clamp(0, num_symbols - 1)
    if invalid_ratio > 0:
        invalid_mask = torch.rand(shape, generator=g) < invalid_ratio
        indexes = indexes.masked_fill(invalid_mask, -1)
        data = data.masked_fill(invalid_mask, 123)  # skipped by the coder
    expected = data.clone()
    expected[indexes < 0] = 0
    return data, indexes, expected


def _cuda_usable():
    if not torch.cuda.is_available():
        return False
    try:
        _ = torch.zeros(1, dtype=torch.int32).cuda()
        return True
    except Exception:
        return False


class TestSparseInverseCDFTable(unittest.TestCase):
    """Python-side table construction (no native extension involved)."""

    def test_default_is_dense_and_unchanged(self):
        cdfs = _generate_rans_params(10)[0]
        self.assertTrue(torch.equal(
            inverse_quantized_cdf(cdfs, freq_precision=10),
            inverse_quantized_cdf(cdfs, freq_precision=10, table_precision=10)))
        self.assertTrue(torch.equal(
            inverse_quantized_cdf(cdfs, freq_precision=10),
            inverse_quantized_cdf(cdfs, freq_precision=10, table_precision=None)))

    def test_sparse_table_samples_dense_table_at_bucket_starts(self):
        for freq_precision in (6, 8, 10):
            for table_precision in range(1, freq_precision + 1):
                with self.subTest(freq_precision=freq_precision,
                                  table_precision=table_precision):
                    cdfs = _generate_rans_params(freq_precision)[0]
                    dense = inverse_quantized_cdf(cdfs, freq_precision=freq_precision)
                    sparse = inverse_quantized_cdf(
                        cdfs, freq_precision=freq_precision,
                        table_precision=table_precision)
                    self.assertEqual(sparse.shape[-1], 1 << table_precision)
                    shift = freq_precision - table_precision
                    sampled = dense[:, ::(1 << shift)]
                    self.assertTrue(torch.equal(sparse, sampled))

    def test_invalid_table_precision_raises(self):
        cdfs = _generate_rans_params(8)[0]
        for bad in (0, -1, 9, 32):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    inverse_quantized_cdf(cdfs, freq_precision=8, table_precision=bad)

    def test_reference_lookup_matches_dense_table(self):
        """Exhaustively replays the kernel's sparse lookup in torch.

        For every cumulative frequency in [0, 2**freq_precision) the
        "table lookup + bounded linear walk" must land on exactly the same
        symbol as the dense inverse CDF.
        """
        for freq_precision in (6, 8):
            for num_symbols in (4, 16):
                cdfs, cdfs_sizes, _ = _generate_rans_params(
                    freq_precision, bypass=False, num_symbols=num_symbols, num_dists=3)
                cdf_size = int(cdfs_sizes[0])
                cdf = cdfs[:, :cdf_size].long()
                dense = inverse_quantized_cdf(cdfs, freq_precision=freq_precision).long()
                cum = torch.arange(1 << freq_precision).unsqueeze(0).expand(cdf.size(0), -1)

                for table_precision in range(1, freq_precision + 1):
                    with self.subTest(freq_precision=freq_precision,
                                      num_symbols=num_symbols,
                                      table_precision=table_precision):
                        shift = freq_precision - table_precision
                        sparse = inverse_quantized_cdf(
                            cdfs, freq_precision=freq_precision,
                            table_precision=table_precision).long()
                        idx = sparse.gather(1, cum >> shift).clone()
                        steps = 0
                        for _ in range(1 << shift):
                            nxt = (idx + 1).clamp(max=cdf_size - 1)
                            advance = cdf.gather(1, nxt) <= cum
                            if not advance.any():
                                break
                            idx = torch.where(advance & (nxt <= cdf_size - 1), nxt, idx)
                            steps += 1
                        self.assertTrue(torch.equal(idx, dense.gather(1, cum)))
                        # the walk length is bounded by the bucket width
                        self.assertLessEqual(steps, 1 << shift)


class TestSparseInverseCDFRoundtrip(unittest.TestCase):
    """End-to-end: encode once, decode with every lookup strategy."""

    def _run(self, device, freq_precision, table_precisions, bypass, num_symbols,
             shape, seed, interleaves=32, invalid_ratio=0.0):
        data, indexes, expected = _generate_data(
            shape, seed, invalid_ratio=invalid_ratio, clamp_to_range=not bypass,
            num_symbols=num_symbols)
        cdfs, cdfs_sizes, offsets = _generate_rans_params(
            freq_precision, bypass=bypass, num_symbols=num_symbols)

        if interleaves == 32:
            init, push, pop, invcdf_pop = (rans32_16_init_stream, rans32_16_i32_push,
                                           rans32_16_i32_pop, rans32_16_i32_invcdf_pop)
        else:
            init, push, pop, invcdf_pop = (rans32_16_init_stream, rans32_16_push,
                                           rans32_16_pop, rans32_16_invcdf_pop)

        stream = init(shape[0], interleaves).to(device)
        push(stream, data.to(device), indexes.to(device), cdfs.to(device),
             cdfs_sizes.to(device), offsets.to(device),
             freq_precision=freq_precision, bypass_coding=bypass, bypass_precision=4)
        if device == "cuda":
            torch.cuda.synchronize()

        reference = pop(stream.clone(), indexes.to(device), cdfs.to(device),
                        cdfs_sizes.to(device), offsets.to(device),
                        freq_precision=freq_precision, bypass_coding=bypass,
                        bypass_precision=4)
        self.assertTrue(torch.equal(expected, reference.cpu()),
                        "divided-search baseline mismatch")

        for q in table_precisions:
            with self.subTest(device=device, freq_precision=freq_precision,
                              interleaves=interleaves, table_precision=q):
                cdfs_with_table = _with_inverse_cdf_table(cdfs, freq_precision, q)
                decoded = invcdf_pop(
                    stream.clone(), indexes.to(device), cdfs_with_table.to(device),
                    cdfs_sizes.to(device), offsets.to(device),
                    freq_precision=freq_precision, bypass_coding=bypass,
                    bypass_precision=4, inverse_cdf_precision=q)
                self.assertTrue(torch.equal(expected, decoded.cpu()))
                # identical to the baseline: the lookup must not change results
                self.assertTrue(torch.equal(reference.cpu(), decoded.cpu()))

    def test_cpu_warp32_sparse_sweep(self):
        # q == freq_precision is the dense table, q < freq_precision is sparse
        self._run("cpu", 12, [12, 10, 8, 6, 4, 1], True, NUM_SYMBOLS, (3, 200), seed=1)
        self._run("cpu", 8, [8, 6, 4, 2], True, NUM_SYMBOLS, (2, 77), seed=2)
        self._run("cpu", 15, [15, 13, 11, 8], True, NUM_SYMBOLS, (2, 96), seed=3)

    def test_cpu_non_warp_sparse_sweep(self):
        self._run("cpu", 12, [12, 8, 4], True, NUM_SYMBOLS, (3, 100), seed=4, interleaves=1)
        self._run("cpu", 10, [10, 7, 3], False, 5, (2, 64), seed=5, interleaves=1)

    def test_cpu_decoder_handles_tail_and_invalid_indexes(self):
        # (2, 77) leaves a 13-symbol tail (77 % 32 != 0) and 30% invalid indexes
        self._run("cpu", 12, [12, 8, 5], True, NUM_SYMBOLS, (2, 77), seed=6,
                  invalid_ratio=0.3)

    def test_cuda_warp32_sparse_sweep(self):
        if not _cuda_usable():
            self.skipTest("CUDA is not available")
        self._run("cuda", 12, [12, 10, 8, 6, 4, 1], True, NUM_SYMBOLS, (3, 200), seed=7)
        self._run("cuda", 15, [15, 11, 8, 4], True, NUM_SYMBOLS, (2, 96), seed=8)
        self._run("cuda", 8, [8, 5, 2], True, NUM_SYMBOLS, (5, 257), seed=9)

    def test_cuda_warp32_no_bypass(self):
        if not _cuda_usable():
            self.skipTest("CUDA is not available")
        self._run("cuda", 12, [12, 9, 6], False, NUM_SYMBOLS, (2, 128), seed=10)

    def test_cuda_non_warp_sparse_sweep(self):
        if not _cuda_usable():
            self.skipTest("CUDA is not available")
        self._run("cuda", 12, [12, 8, 4], True, NUM_SYMBOLS, (3, 100), seed=11,
                  interleaves=1)

    def test_cpu_cuda_agree_bit_for_bit(self):
        """The same stream decodes identically on CPU and on the warp kernel."""
        if not _cuda_usable():
            self.skipTest("CUDA is not available")
        data, indexes, expected = _generate_data((4, 300), seed=12)
        cdfs, cdfs_sizes, offsets = _generate_rans_params(12)
        stream = rans32_16_init_stream(4, 32)
        rans32_16_i32_push(stream, data, indexes, cdfs, cdfs_sizes, offsets,
                           freq_precision=12, bypass_coding=True, bypass_precision=4)

        for q in (12, 9, 6):
            with self.subTest(table_precision=q):
                cdfs_with_table = _with_inverse_cdf_table(cdfs, 12, q)
                on_cpu = rans32_16_i32_invcdf_pop(
                    stream.clone(), indexes, cdfs_with_table, cdfs_sizes, offsets,
                    freq_precision=12, bypass_coding=True, bypass_precision=4,
                    inverse_cdf_precision=q)
                on_gpu = rans32_16_i32_invcdf_pop(
                    stream.clone().cuda(), indexes.cuda(), cdfs_with_table.cuda(),
                    cdfs_sizes.cuda(), offsets.cuda(),
                    freq_precision=12, bypass_coding=True, bypass_precision=4,
                    inverse_cdf_precision=q)
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(expected, on_cpu))
                self.assertTrue(torch.equal(on_cpu, on_gpu.cpu()))


class TestTorchANSInterfaceSparseInverseCDF(unittest.TestCase):

    def _params(self, seed=20, num_symbols=NUM_SYMBOLS):
        g = torch.Generator().manual_seed(seed)
        freqs = torch.randint(1, 100, (NUM_DISTS, num_symbols), generator=g).float()
        num_freqs = torch.zeros(NUM_DISTS, dtype=torch.int32) + num_symbols
        offsets = torch.randint(-2, 2, (NUM_DISTS,), generator=g, dtype=torch.int32)
        return freqs, num_freqs, offsets

    def _roundtrip(self, device, inverse_cdf_precision, freq_precision=12, seed=21):
        freqs, num_freqs, offsets = self._params(seed=seed)
        coder = TorchANSInterface(
            impl="rans32_16", freq_precision=freq_precision, bypass_coding=True,
            device=device, num_interleaves=32,
            inverse_cdf_precision=inverse_cdf_precision)
        coder.init_params(freqs, num_freqs, offsets)

        g = torch.Generator().manual_seed(seed + 1)
        shape = (4, 150)
        symbols = torch.randint(0, NUM_SYMBOLS, shape, generator=g, dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)

        stream = coder.encode_with_indexes(symbols.to(device), indexes.to(device))
        decoded = coder.decode_with_indexes(stream, indexes.to(device))
        self.assertTrue(torch.equal(symbols, decoded.cpu()))
        # the appended table must be 2**inverse_cdf_precision wide
        expected_width = num_freqs.max().item() + 2 + (1 << inverse_cdf_precision)
        self.assertEqual(coder.cdfs.size(-1), expected_width)

    def test_cpu_high_level_sparse(self):
        for q in (12, 8, 5):
            with self.subTest(table_precision=q):
                self._roundtrip("cpu", q)

    def test_cuda_high_level_sparse(self):
        if not _cuda_usable():
            self.skipTest("CUDA is not available")
        for q in (12, 8, 5):
            with self.subTest(table_precision=q):
                self._roundtrip("cuda", q)

    def test_no_inverse_cdf_keeps_plain_cdfs(self):
        freqs, num_freqs, offsets = self._params(seed=30)
        coder = TorchANSInterface(impl="rans32_16", freq_precision=12, device="cpu")
        coder.init_params(freqs, num_freqs, offsets)
        self.assertEqual(coder.cdfs.size(-1), num_freqs.max().item() + 2)
        self.assertFalse(coder.impl_use_inverse_cdf)

    def test_rans64_i4_inverse_cdf_supported(self):
        # C2: the rans64 + 4-interleaves inverse-CDF combination used to be
        # missing; it must roundtrip like every other supported combination
        for device in ("cpu", "cuda"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            with self.subTest(device=device):
                self._roundtrip(device, inverse_cdf_precision=8)

    def test_inverse_cdf_precision_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            TorchANSInterface(impl="rans32_16", freq_precision=12, device="cpu",
                              inverse_cdf_precision=13)
        with self.assertRaises(ValueError):
            TorchANSInterface(impl="rans32_16", freq_precision=12, device="cpu",
                              inverse_cdf_precision=0)


class TestAutoInverseCDF(unittest.TestCase):
    """C1: inverse_cdf_precision="auto" picks the precision from the alphabet."""

    def _run(self, device, num_symbols, freq_precision=15, seed=31):
        g = torch.Generator().manual_seed(seed)
        freqs = torch.randint(1, 100, (NUM_DISTS, num_symbols), generator=g).float()
        num_freqs = torch.zeros(NUM_DISTS, dtype=torch.int32) + num_symbols
        offsets = torch.randint(-2, 2, (NUM_DISTS,), generator=g, dtype=torch.int32)
        coder = TorchANSInterface(
            impl="rans32_16", freq_precision=freq_precision, bypass_coding=True,
            device=device, num_interleaves=32, inverse_cdf_precision="auto")
        coder.init_params(freqs.to(device), num_freqs.to(device), offsets.to(device))

        g = torch.Generator().manual_seed(seed + 1)
        shape = (4, 150)
        symbols = torch.randint(0, num_symbols, shape, generator=g, dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)
        stream = coder.encode_with_indexes(symbols.to(device), indexes.to(device))
        decoded = coder.decode_with_indexes(stream, indexes.to(device))
        return coder, symbols, decoded

    def test_auto_roundtrip_and_precision(self):
        devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
        for device in devices:
            for num_symbols in (256, 1024):
                with self.subTest(device=device, num_symbols=num_symbols):
                    coder, symbols, decoded = self._run(device, num_symbols)
                    self.assertTrue(torch.equal(symbols, decoded.cpu()))
                    q = coder.inverse_cdf_precision
                    self.assertIsInstance(q, int)
                    self.assertGreaterEqual(q, 1)
                    self.assertLessEqual(q, coder.freq_precision)
                    # the appended table must be 2**q wide
                    table_entries = coder.cdfs.size(-1) - (num_symbols + 2)
                    self.assertEqual(table_entries, 1 << q)

    def test_auto_small_alphabet_skips_table(self):
        # below min_alphabet=64 the divided search is within ~3% - no table
        coder, symbols, decoded = self._run("cpu", num_symbols=32)
        self.assertFalse(coder.impl_use_inverse_cdf)
        self.assertIsNone(coder.inverse_cdf_precision)
        self.assertTrue(torch.equal(symbols, decoded.cpu()))

    def test_auto_expected_precision(self):
        # cols budget = 16 KB / (num_dists * 4 B); whole-row staging prefers a
        # walk of ~2 steps (2**q ~= alphabet/2), see auto_inverse_cdf_precision
        self.assertEqual(auto_inverse_cdf_precision(258, 15, num_distributions=6), 8)
        self.assertEqual(auto_inverse_cdf_precision(1026, 15, num_distributions=6), 9)
        self.assertIsNone(auto_inverse_cdf_precision(17, 15))
        self.assertEqual(auto_inverse_cdf_precision(17, 15, min_alphabet=16), 4)
        # never denser than the frequency precision allows
        self.assertEqual(auto_inverse_cdf_precision(600, 8, num_distributions=6), 7)


class TestFusedBuilders(unittest.TestCase):
    """B1/B2/B4: the fused builders must match the original torch-op chains."""

    @staticmethod
    def _quantized_cdf(num_dists, num_symbols, freq_precision, device="cpu", seed=3):
        g = torch.Generator().manual_seed(seed)
        pmf = torch.rand(num_dists, num_symbols, generator=g)
        return rans_pmf_to_quantized_cdf(pmf, freq_precision).to(device)

    @staticmethod
    def _reference_inverse_table(cdf, freq_precision, table_precision):
        # the original broadcast compare-and-count chain
        shift = freq_precision - table_precision
        table_size = 1 << table_precision
        freq_range = (torch.arange(table_size).unsqueeze(0) << shift).type_as(cdf)
        return ((freq_range.unsqueeze(-1) >= cdf.unsqueeze(1))
                .sum(-1, dtype=cdf.dtype) - 1)

    @staticmethod
    def _reference_pmf_to_quantized_cdf(pmf, precision):
        """The original torch-op chain, including the fix-up loop."""
        batched = pmf.unsqueeze(0) if pmf.dim() == 1 else pmf.reshape(-1, pmf.size(-1))
        B, N = batched.size(0), batched.size(1)
        freq = torch.round(batched * (1 << precision)).to(torch.int32)
        cdf = torch.zeros((B, N + 1), dtype=torch.int32, device=batched.device)
        cdf[:, 1:] = freq
        total = cdf.sum(1, keepdim=True).to(torch.int32)
        total = torch.where(total == 0, torch.ones_like(total), total)
        cdf = ((cdf * (1 << precision)) / total).to(torch.int32)
        cdf = torch.cumsum(cdf, 1).to(torch.int32)
        cdf[:, N] = 1 << precision
        for b in range(B):
            row = cdf[b]
            for i in range(N):
                if row[i] == row[i + 1]:
                    best_freq, best_steal = None, -1
                    for j in range(N):
                        f = int(row[j + 1]) - int(row[j])
                        if f > 1 and (best_freq is None or f < best_freq):
                            best_freq, best_steal = f, j
                    if best_steal == -1:
                        continue  # CUDA behaviour; the host path raises instead
                    if best_steal < i:
                        row[best_steal + 1:i + 1] -= 1
                    elif best_steal > i:
                        row[i + 1:best_steal + 1] += 1
        return cdf

    def test_build_cdf_with_inverse_table_matches_reference(self):
        from torch_ans.utils import build_cdf_with_inverse_table
        devices = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
        for device in devices:
            for num_symbols, q in ((256, 7), (256, 15), (64, 6)):
                with self.subTest(device=device, num_symbols=num_symbols, q=q):
                    cdf = self._quantized_cdf(6, num_symbols, 15, device)
                    combined = build_cdf_with_inverse_table(cdf, 15, q)
                    expected = torch.cat(
                        [cdf, self._reference_inverse_table(cdf, 15, q)], dim=-1)
                    self.assertEqual(combined.shape, expected.shape)
                    self.assertTrue(torch.equal(combined, expected))

    def test_inverse_quantized_cdf_matches_broadcast(self):
        devices = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
        for device in devices:
            for q in (6, 9, 15):
                with self.subTest(device=device, q=q):
                    cdf = self._quantized_cdf(4, 258, 15, device)
                    ref = self._reference_inverse_table(cdf, 15, q)
                    new = inverse_quantized_cdf(cdf, freq_precision=15, table_precision=q)
                    self.assertTrue(torch.equal(ref, new))

    def test_fused_pmf_to_quantized_cdf_matches_reference(self):
        devices = ("cpu",) + (("cuda",) if torch.cuda.is_available() else ())
        for device in devices:
            for num_symbols, precision in ((66, 12), (258, 15)):
                for kind in ("uniform", "many_zeros", "one_hot"):
                    with self.subTest(device=device, num_symbols=num_symbols,
                                      precision=precision, kind=kind):
                        g = torch.Generator().manual_seed(
                            hash((kind, num_symbols)) % (2 ** 31))
                        if kind == "uniform":
                            pmf = torch.rand(4, num_symbols, generator=g)
                        elif kind == "many_zeros":
                            pmf = torch.zeros(4, num_symbols)
                            pmf[:, :: max(num_symbols // 4, 1)] = 1.0
                        else:
                            pmf = torch.zeros(4, num_symbols)
                            pmf[:, 0] = 1.0
                        pmf = (pmf / pmf.sum(-1, keepdim=True)).to(device)
                        ref = self._reference_pmf_to_quantized_cdf(pmf, precision)
                        new = rans_pmf_to_quantized_cdf(pmf, precision)
                        self.assertTrue(torch.equal(ref, new))


if __name__ == "__main__":
    unittest.main()
