import unittest
import torch

from torch_ans.utils import TorchANSInterface


class TestTorchANSHigherLevelAPI(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def test_encode_decode_basic(self):
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 32
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)

        self.assertTrue(torch.equal(decoded, symbols))

    def test_encode_with_cache_and_flush(self):
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 32
        symbols1 = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes1 = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)
        symbols2 = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes2 = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        # Cache two separate batches
        ret1 = coder.encode_with_indexes(symbols1, indexes1, cache=True)
        self.assertIsNone(ret1)
        ret2 = coder.encode_with_indexes(symbols2, indexes2, cache=True)
        self.assertIsNone(ret2)

        # Flush and decode the stacked segments
        encoded = coder.flush()
        self.assertIsInstance(encoded, (bytes, bytearray))

        stacked_indexes = torch.stack([indexes1, indexes2], dim=0)
        decoded = coder.decode_with_indexes(encoded, stacked_indexes)

        expected = torch.stack([symbols1, symbols2], dim=0)
        self.assertTrue(torch.equal(decoded, expected))

    def test_set_stream_and_decode_stream(self):
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 32
        symbols1 = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes1 = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        coder.encode_with_indexes(symbols1, indexes1, cache=True)
        encoded = coder.flush()

        coder2 = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder2.init_params(pmf, num_freqs, offsets)
        coder2.set_stream(encoded)

        decoded = coder2.decode_stream(indexes1)
        self.assertTrue(torch.equal(decoded, symbols1))

    def test_different_implementations(self):
        # Test different rANS implementations
        implementations = ["rans64", "rans32", "rans32_16"]
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        seq_len = 32
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        for impl in implementations:
            with self.subTest(impl=impl):
                coder = TorchANSInterface(impl=impl, freq_precision=12, device="cpu")
                coder.init_params(pmf, num_freqs, offsets)

                encoded = coder.encode_with_indexes(symbols, indexes)
                decoded = coder.decode_with_indexes(encoded, indexes)

                self.assertTrue(torch.equal(decoded, symbols))

    def test_bypass_coding(self):
        # Test bypass coding functionality
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, bypass_coding=True, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 32
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)

        self.assertTrue(torch.equal(decoded, symbols))

    def test_different_freq_precisions(self):
        # Test different frequency precisions
        freq_precisions = [8, 12, 16]
        batch_size = 2
        num_dists = 4
        num_symbols = 8  # Smaller for lower precision
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        seq_len = 16
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        for freq_precision in freq_precisions:
            with self.subTest(freq_precision=freq_precision):
                coder = TorchANSInterface(impl="rans32", freq_precision=freq_precision, device="cpu")
                coder.init_params(pmf, num_freqs, offsets)

                encoded = coder.encode_with_indexes(symbols, indexes)
                decoded = coder.decode_with_indexes(encoded, indexes)

                self.assertTrue(torch.equal(decoded, symbols))

    def test_set_cdfs_directly(self):
        # Test setting CDFs directly
        batch_size = 2
        num_dists = 4
        num_symbols = 16

        # Create CDFs manually
        cdfs = torch.zeros(num_dists, num_symbols + 2, dtype=torch.int32)
        for i in range(num_dists):
            cdfs[i] = torch.linspace(0, 256, num_symbols + 2, dtype=torch.int32)

        cdfs_sizes = torch.full((num_dists,), num_symbols + 1, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=8, device="cpu")
        coder.set_cdfs(cdfs, cdfs_sizes, offsets)

        seq_len = 32
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)

        self.assertTrue(torch.equal(decoded, symbols))

    def test_invalid_implementation(self):
        # Test invalid implementation name
        with self.assertRaises(NotImplementedError):
            TorchANSInterface(impl="invalid_impl")

    def test_single_symbol_distribution(self):
        # Test with single symbol distributions
        batch_size = 2
        num_dists = 4
        num_symbols = 1
        pmf = torch.ones(num_dists, num_symbols, dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 8
        symbols = torch.zeros(batch_size, seq_len, dtype=torch.int32)  # Only symbol 0
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)

        self.assertTrue(torch.equal(decoded, symbols))

    def test_reset_cache(self):
        # Test cache reset functionality
        batch_size = 2
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 16
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        # Cache some data
        coder.encode_with_indexes(symbols, indexes, cache=True)
        self.assertTrue(len(coder.cache_encode_with_indexes) > 0)

        # Reset cache
        coder.reset_cache()
        self.assertEqual(len(coder.cache_encode_with_indexes), 0)

    def test_num_parallel_states(self):
        # Test with explicit num_parallel_states
        num_parallel_states = 4
        num_dists = 4
        num_symbols = 16
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        coder = TorchANSInterface(impl="rans64", freq_precision=12, num_parallel_states=num_parallel_states, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        seq_len = 32
        symbols = torch.randint(0, num_symbols, (num_parallel_states, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (num_parallel_states, seq_len), dtype=torch.int32)

        encoded = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(encoded, indexes)

        self.assertTrue(torch.equal(decoded, symbols))


if __name__ == "__main__":
    unittest.main()
