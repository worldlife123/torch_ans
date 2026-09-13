"""Tests for the index range check feature (0.2.1).

The C++/CUDA indexed rans_push/rans_pop implementations skip invalid
distribution indexes (index < 0 or index >= cdfs.size(0)) instead of
reading out of bounds: push silently skips such symbols, pop writes 0
for them. The high-level API additionally exposes ``check_validity``
which raises ValueError for out-of-range ``dist_indexes`` before coding.

This also covers the parallel-states padding use case, which relies on
this check: padding indexes are filled with -1 ("invalid") and skipped
during coding.
"""
import unittest

import torch

from cuda_helpers import is_cuda_unavailable_error, require_cuda_coding


class TestIndexRangeCheckLowLevel(unittest.TestCase):

    def _prepare_params(self, num_dists=4, num_symbols=255, freq_precision=15, device="cpu"):
        from torch_ans._C import rans_pmf_to_quantized_cdf
        pmfs = torch.randint(1, 1024, (num_dists, num_symbols), dtype=torch.float32)
        pmfs = torch.cat([pmfs, torch.ones(num_dists, 1).type_as(pmfs)], dim=-1)
        pmfs = pmfs / pmfs.sum(dim=-1, keepdim=True)
        cdfs = rans_pmf_to_quantized_cdf(pmfs, precision=freq_precision).to(device)
        cdfs_sizes = torch.full((num_dists,), num_symbols + 2, dtype=torch.int32, device=device)
        offsets = torch.zeros(num_dists, dtype=torch.int32, device=device)
        return cdfs, cdfs_sizes, offsets

    def _roundtrip_with_corrupted_indexes(self, device, corrupt, num_interleaves=1):
        from torch_ans._C import rans64_init_stream, rans64_push, rans64_pop
        num_batch, num_dists, seq_len = 8, 4, 128
        num_symbols = 255
        cdfs, cdfs_sizes, offsets = self._prepare_params(num_dists, num_symbols, device=device)
        data = torch.randint(0, num_symbols, (num_batch, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (num_batch, seq_len), dtype=torch.int32)
        if corrupt in ("negative", "mixed"):
            indexes[:, ::7] = -1
        if corrupt in ("too_large", "mixed"):
            indexes[:, 3::7] = num_dists  # first out-of-range index

        data, indexes = data.to(device), indexes.to(device)
        stream = rans64_init_stream(num_batch, num_interleaves).to(device)
        # Must not segfault or raise on out-of-range indexes
        rans64_push(stream, data, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes,
                    offsets=offsets, freq_precision=15)
        decoded = rans64_pop(stream, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes,
                             offsets=offsets, freq_precision=15)

        valid = (indexes >= 0) & (indexes < num_dists)
        # Valid positions must round-trip exactly
        self.assertTrue((decoded[valid] == data[valid]).all(),
                        "valid symbols must round-trip when invalid indexes are skipped")
        # Invalid positions are decoded as 0
        self.assertTrue((decoded[~valid] == 0).all(),
                        "out-of-range indexes must decode to 0")

    def test_cpu_push_pop_negative_indexes(self):
        self._roundtrip_with_corrupted_indexes("cpu", "negative")

    def test_cpu_push_pop_too_large_indexes(self):
        self._roundtrip_with_corrupted_indexes("cpu", "too_large")

    def test_cpu_push_pop_mixed_invalid_indexes(self):
        self._roundtrip_with_corrupted_indexes("cpu", "mixed")

    def test_cpu_push_pop_mixed_invalid_indexes_32way(self):
        # 32-way interleaved (warp-level) variant shares the same range checks
        self._roundtrip_with_corrupted_indexes("cpu", "mixed", num_interleaves=32)

    def test_cpu_pop_all_invalid_indexes(self):
        from torch_ans._C import rans64_init_stream, rans64_push, rans64_pop
        num_batch, num_dists, seq_len = 4, 4, 64
        cdfs, cdfs_sizes, offsets = self._prepare_params(num_dists)
        data = torch.randint(0, 255, (num_batch, seq_len), dtype=torch.int32)
        indexes = torch.full((num_batch, seq_len), num_dists, dtype=torch.int32)
        stream = rans64_init_stream(num_batch)
        rans64_push(stream, data, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes,
                    offsets=offsets, freq_precision=15)
        decoded = rans64_pop(stream, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes,
                             offsets=offsets, freq_precision=15)
        self.assertTrue((decoded == 0).all())

    def test_cuda_push_pop_mixed_invalid_indexes(self):
        require_cuda_coding(self)
        try:
            self._roundtrip_with_corrupted_indexes("cuda", "mixed")
        except RuntimeError as e:
            if is_cuda_unavailable_error(e):
                self.skipTest(f"CUDA coding is unavailable: {e}")
            raise


class TestIndexRangeCheckHighLevel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def _make_coder(self, num_dists=4, num_symbols=16, batch_size=2, seq_len=32, **coder_kwargs):
        from torch_ans.utils import TorchANSInterface
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)
        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu", **coder_kwargs)
        coder.init_params(pmf, num_freqs, offsets)
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)
        return coder, symbols, indexes

    def test_encode_with_indexes_raises_on_out_of_range(self):
        coder, symbols, indexes = self._make_coder(check_validity=True)
        indexes[:, ::5] = 4  # == num_dists
        with self.assertRaises(ValueError):
            coder.encode_with_indexes(symbols, indexes)

    def test_encode_with_indexes_raises_on_negative(self):
        coder, symbols, indexes = self._make_coder(check_validity=True)
        indexes[:, ::5] = -1
        with self.assertRaises(ValueError):
            coder.encode_with_indexes(symbols, indexes)

    def test_decode_raises_on_out_of_range(self):
        coder, symbols, indexes = self._make_coder(check_validity=True)
        encoded = coder.encode_with_indexes(symbols, indexes)
        bad_indexes = indexes.clone()
        bad_indexes[:, 0] = 4  # == num_dists
        with self.assertRaises(ValueError):
            coder.decode_with_indexes(encoded, bad_indexes)

    def test_valid_indexes_pass_check(self):
        coder, symbols, indexes = self._make_coder(check_validity=True)
        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)
        self.assertTrue(torch.equal(decoded, symbols))

    def test_default_no_check_skips_invalid_indexes(self):
        # Default check_validity=False: no exception, C++ skips invalid indexes
        coder, symbols, indexes = self._make_coder()
        indexes[:, ::5] = 4
        indexes[:, 7::5] = -1
        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)
        valid = (indexes >= 0) & (indexes < 4)
        self.assertTrue((decoded[valid] == symbols[valid]).all())
        self.assertTrue((decoded[~valid] == 0).all())

    def test_validity_check_tolerates_parallel_padding(self):
        # Padding indexes (-1) are injected after the validity check, so
        # check_validity=True must not reject non-divisible parallel states
        coder, symbols, indexes = self._make_coder(
            batch_size=4, seq_len=32, num_parallel_states=5, check_validity=True)
        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)
        self.assertTrue(torch.equal(decoded, symbols))


if __name__ == "__main__":
    unittest.main()
