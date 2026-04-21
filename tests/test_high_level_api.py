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


if __name__ == "__main__":
    unittest.main()
