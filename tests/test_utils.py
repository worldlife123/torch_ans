import unittest
import torch
import struct
import io

from torch_ans.utils import _get_bytes_format, merge_bytes, split_merged_bytes, pmf_to_quantized_cdf_batched, inverse_quantized_cdf


class TestUtils(unittest.TestCase):

    def test_get_bytes_format(self):
        # Test valid byte formats
        self.assertEqual(_get_bytes_format(1), "B")
        self.assertEqual(_get_bytes_format(2), "<H")
        self.assertEqual(_get_bytes_format(4), "<I")
        self.assertEqual(_get_bytes_format(8), "<L")

        # Test invalid byte format
        with self.assertRaises(ValueError):
            _get_bytes_format(3)

    def test_merge_bytes_basic(self):
        # Test basic merging
        data = [b"hello", b"world"]
        result = merge_bytes(data)
        expected = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world"
        self.assertEqual(result, expected)

    def test_merge_bytes_with_num_segments(self):
        # Test merging with num_segments
        data = [b"hello", b"world", b"!"]
        result = merge_bytes(data, num_segments=3)
        # Should only write lengths for first 2 segments
        expected = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world" + b"!"
        self.assertEqual(result, expected)

    def test_merge_bytes_custom_length(self):
        # Test with custom num_bytes_length
        data = [b"hi", b"ok"]
        result = merge_bytes(data, num_bytes_length=2)
        expected = struct.pack("<H", 2) + b"hi" + struct.pack("<H", 2) + b"ok"
        self.assertEqual(result, expected)

    def test_split_merged_bytes_basic(self):
        # Test basic splitting
        data = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world"
        result = split_merged_bytes(data)
        expected = [b"hello", b"world"]
        self.assertEqual(result, expected)

    def test_split_merged_bytes_with_num_segments(self):
        # Test splitting with num_segments
        data = struct.pack("<I", 5) + b"hello" + struct.pack("<I", 5) + b"world" + b"!"
        result = split_merged_bytes(data, num_segments=3)
        expected = [b"hello", b"world", b"!"]
        self.assertEqual(result, expected)

    def test_split_merged_bytes_custom_length(self):
        # Test with custom num_bytes_length
        data = struct.pack("<H", 2) + b"hi" + struct.pack("<H", 2) + b"ok"
        result = split_merged_bytes(data, num_bytes_length=2)
        expected = [b"hi", b"ok"]
        self.assertEqual(result, expected)

    def test_merge_split_roundtrip(self):
        # Test roundtrip merge -> split
        original = [b"test1", b"test2", b"test3"]
        merged = merge_bytes(original, num_segments=3)
        result = split_merged_bytes(merged, num_segments=3)
        self.assertEqual(result, original)

    def test_pmf_to_quantized_cdf_batched_basic(self):
        # Test basic PMF to CDF conversion
        pmf = torch.tensor([[0.25, 0.25, 0.25, 0.25]], dtype=torch.float32)
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8, add_tail=False)
        # Should be [0, 64, 128, 192, 256] for uniform distribution
        expected = torch.tensor([[0, 64, 128, 192, 256]], dtype=torch.int32)
        self.assertTrue(torch.equal(cdf, expected))

    def test_pmf_to_quantized_cdf_batched_nonuniform(self):
        # Test non-uniform PMF
        pmf = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8)
        # Check that CDF is monotonically increasing and ends at 256
        self.assertEqual(cdf[0, 0], 0)
        self.assertEqual(cdf[0, -1], 256)
        self.assertTrue(torch.all(cdf[0, 1:] >= cdf[0, :-1]))

    def test_pmf_to_quantized_cdf_batched_batch(self):
        # Test batched conversion
        pmf = torch.tensor([[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8, add_tail=False)
        self.assertEqual(cdf.shape, (2, 5))
        self.assertTrue(torch.all(cdf[:, -1] == 256))

    def test_pmf_to_quantized_cdf_batched_with_tail(self):
        # Test with tail symbol
        pmf = torch.tensor([[0.3, 0.3, 0.3]], dtype=torch.float32)
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8, add_tail=True)
        self.assertEqual(cdf.shape, (1, 5))  # 3 symbols + tail + end
        self.assertEqual(cdf[0, -1], 256)

    def test_pmf_to_quantized_cdf_batched_normalize(self):
        # Test normalization
        pmf = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)  # Not normalized
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8, normalize=True)
        self.assertEqual(cdf[0, -1], 256)

    def test_inverse_quantized_cdf_basic(self):
        # Test inverse CDF computation
        cdf = torch.tensor([[0, 64, 128, 192, 256]], dtype=torch.int32)
        inverse = inverse_quantized_cdf(cdf, freq_precision=8)
        # For uniform distribution, each value should map to its symbol
        self.assertEqual(inverse.shape, (1, 256))
        # Check some values
        self.assertEqual(inverse[0, 0], 0)  # 0-63 -> 0
        self.assertEqual(inverse[0, 64], 1)  # 64-127 -> 1
        self.assertEqual(inverse[0, 128], 2)  # 128-191 -> 2
        self.assertEqual(inverse[0, 192], 3)  # 192-255 -> 3

    def test_inverse_quantized_cdf_nonuniform(self):
        # Test inverse CDF with non-uniform distribution
        cdf = torch.tensor([[0, 26, 77, 154, 256]], dtype=torch.int32)
        inverse = inverse_quantized_cdf(cdf, freq_precision=8)
        self.assertEqual(inverse.shape, (1, 256))
        # Check ranges
        self.assertTrue(torch.all(inverse[0, 0:26] == 0))
        self.assertTrue(torch.all(inverse[0, 26:77] == 1))
        self.assertTrue(torch.all(inverse[0, 77:154] == 2))
        self.assertTrue(torch.all(inverse[0, 154:256] == 3))


if __name__ == "__main__":
    unittest.main()