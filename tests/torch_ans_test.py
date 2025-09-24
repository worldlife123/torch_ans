
import torch
# import multiprocessing
# torch.set_num_threads(multiprocessing.cpu_count())

import torch
import numpy as np
import time
import unittest
import functools

class TestTorchANS(unittest.TestCase):

    def _generate_rans_params(self, num_dists, num_symbols, freq_precision=16, device="cpu"):
        from torch_ans._C import rans_pmf_to_quantized_cdf
        from torch_ans.utils import pmf_to_quantized_cdf_batched, inverse_quantized_cdf
        pmfs = torch.randint(1, 1024, (num_dists, num_symbols), dtype=torch.float32, device=device) #.repeat(num_dists, 1).contiguous()
        # cdfs = pmf_to_quantized_cdf_batched(pmfs, freq_precision=freq_precision)
        pmfs = torch.cat([pmfs.clone(), torch.ones(pmfs.shape[0], 1).type_as(pmfs)], dim=1)
        pmfs = pmfs / pmfs.sum(dim=-1, keepdim=True) # normalize
        cdfs = rans_pmf_to_quantized_cdf(pmfs, precision=freq_precision)
        # inversed_cdfs = inverse_quantized_cdf(cdfs)
        # cdfs = torch.cat([cdfs, inversed_cdfs], dim=-1)
        cdfs_sizes = torch.zeros(num_dists, dtype=torch.int32) + num_symbols + 2
        # TODO: test different offsets
        # offsets = torch.zeros(num_dists, dtype=torch.int32)
        offsets = torch.randint(-8, 8, (num_dists, ), dtype=torch.int32) #.repeat(num_dists, 1).contiguous()
        self._cdfs, self._cdfs_sizes, self._offsets = cdfs, cdfs_sizes, offsets
        return cdfs, cdfs_sizes, offsets

    def test_import(self):
        import torch
        from torch_ans._C import rans_stream_to_byte_strings, rans_byte_strings_to_stream, rans_alias_build_table
        from torch_ans._C import rans64_init_stream, rans64_push, rans64_pop, rans64_alias_push, rans64_alias_pop, rans64_invcdf_pop
        from torch_ans._C import rans32_init_stream, rans32_push, rans32_pop, rans32_alias_push, rans32_alias_pop, rans32_invcdf_pop

    def test_utils_pmf_to_quantized_cdf_batched(self):
        from torch_ans.utils import pmf_to_quantized_cdf_batched, inverse_quantized_cdf
        pmf = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
        cdf = pmf_to_quantized_cdf_batched(pmf, precision=8)
        self.assertEqual(cdf.shape[0], pmf.shape[0])
        self.assertTrue((cdf[:, 0] == 0).all())
        # TODO: test inverse_quantized_cdf
        # inv = inverse_quantized_cdf(cdf)
        # self.assertEqual(inv.shape, cdf.shape)

    # def test_utils_invalid_input(self):
    #     from torch_ans.utils import pmf_to_quantized_cdf_batched
    #     with self.assertRaises(Exception):
    #         pmf_to_quantized_cdf_batched(torch.tensor([]), freq_precision=8)
    #     with self.assertRaises(Exception):
    #         pmf_to_quantized_cdf_batched(torch.tensor([[1.0, 2.0]]), freq_precision=8)

    def test_rans_pmf_to_quantized_cdf_basic(self):
        from torch_ans._C import rans_pmf_to_quantized_cdf
        # Simple PMF: uniform over 4 symbols
        pmf = torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float32)
        precision = 8
        cdf = rans_pmf_to_quantized_cdf(pmf, precision)
        self.assertEqual((cdf == torch.tensor([0, 64, 128, 192, 256])).all(), True)

    def test_rans_pmf_to_quantized_cdf_nonuniform(self):
        from torch_ans._C import rans_pmf_to_quantized_cdf
        pmf = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float32)
        precision = 8
        cdf = rans_pmf_to_quantized_cdf(pmf, precision)
        self.assertEqual((cdf == torch.tensor([0, 26, 77, 154, 256])).all(), True)

    def test_rans_params_edge_cases(self):
        # Test with num_dists=1, num_symbols=1
        cdfs, cdfs_sizes, offsets = self._generate_rans_params(1, 1, freq_precision=4)
        self.assertEqual(cdfs.shape, (1, 3))
        self.assertEqual(cdfs_sizes.shape, (1,))
        self.assertEqual(offsets.shape, (1,))

    def test_rans_params_extreme_values(self):
        # Test with large num_dists and num_symbols
        cdfs, cdfs_sizes, offsets = self._generate_rans_params(32, 128, freq_precision=16)
        self.assertEqual(cdfs.shape[0], 32)
        self.assertEqual(cdfs.shape[1], 130)

    def test_cuda_availability(self):
        import torch
        if torch.cuda.is_available():
            x = torch.tensor([1, 2, 3])
            x_cuda = x.cuda()
            self.assertTrue(x_cuda.is_cuda)
        else:
            self.assertFalse(torch.cuda.is_available())

    def test_rans_batch_coding_param(self):
        # Parameterized test for different freq_precision and symbol_precision
        for freq_precision, symbol_precision in [(8, 7), (12, 10)]:
            num_batch, num_dists, num_symbols = 10, 4, (1 << symbol_precision) - 1
            data_shape = (num_batch, 16)
            data = torch.randint(0, num_symbols, data_shape, dtype=torch.int32)
            indexes = torch.randint(0, num_dists, data_shape, dtype=torch.int32)
            cdfs, cdfs_sizes, offsets = self._generate_rans_params(num_dists, num_symbols, freq_precision=freq_precision)
            from torch_ans._C import rans32_init_stream, rans32_push, rans32_pop
            stream = rans32_init_stream(num_batch)
            rans32_push(stream, data, indexes, cdfs, cdfs_sizes, offsets, freq_precision=freq_precision)
            from torch_ans._C import rans_stream_to_byte_strings, rans_byte_strings_to_stream
            byte_strings = rans_stream_to_byte_strings(stream)
            stream = rans_byte_strings_to_stream(byte_strings)
            decoded = rans32_pop(stream, indexes, cdfs, cdfs_sizes, offsets, freq_precision=freq_precision)
            self.assertFalse((data.cpu() != decoded.cpu()).any())
    
    def _test_torch_ans(self, name, data, init_func, push_func, pop_func, device="cpu"):
        from torch_ans._C import rans_stream_to_byte_strings, rans_byte_strings_to_stream, rans_alias_build_table
        data = data.to(device)
        start_time = time.time()
        stream = init_func()
        if hasattr(stream, 'to'):
            stream = stream.to(device)
        print(f"{name} init time: {time.time() - start_time}")

        start_time = time.time()
        push_func(stream, data)
        byte_strings = rans_stream_to_byte_strings(stream.cpu() if device == "cuda" else stream)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{name} encoding time: {time.time() - start_time}")
        print(f"{name} encoding bytes: {stream[:, 0].sum()}")
        start_time = time.time()
        stream = rans_byte_strings_to_stream(byte_strings)
        if device == "cuda" and hasattr(stream, 'cuda'):
            stream = stream.cuda()
        decoded = pop_func(stream)
        if device == "cuda":
            torch.cuda.synchronize()
        print(f"{name} decoding time: {time.time() - start_time}")

        self.assertFalse((data.cpu() != decoded.cpu()).any())
        # self.assertEqual(data.reshape(-1).tolist(), decoded.reshape(-1).tolist())

    def test_rans_batch_coding(self):
        from torch_ans._C import (
            rans_stream_to_byte_strings, rans_byte_strings_to_stream, rans_alias_build_table,
            rans64_init_stream, rans64_push, rans64_pop, rans64_i4_push, rans64_i4_pop, rans64_alias_push, rans64_alias_pop,
            rans32_init_stream, rans32_push, rans32_pop, rans32_i4_push, rans32_i4_pop, rans32_alias_push, rans32_alias_pop,
            rans32_16_init_stream, rans32_16_push, rans32_16_pop, rans32_16_i4_push, rans32_16_i4_pop
        )

        symbol_precision = 8
        freq_precision = 15
        num_batch, num_dists, num_symbols = 1000, 8, (1 << symbol_precision) - 1
        bypass_num = 0
        data_num = 3 * 1024

        data_shape = (num_batch, data_num)
        data = torch.randint(0, num_symbols + bypass_num, data_shape, dtype=torch.int32)
        indexes = torch.randint(0, num_dists, data_shape, dtype=torch.int32)
        cdfs, cdfs_sizes, offsets = self._generate_rans_params(num_dists, num_symbols, freq_precision=freq_precision)
        cdfs_with_alias_remap, cdfs_with_alias_table = rans_alias_build_table(
            cdfs, cdfs_sizes, symbol_precision=symbol_precision, freq_precision=freq_precision
        )

        # Non-alias methods
        non_alias_methods = [
            ("Rans32", "cpu", functools.partial(rans32_init_stream, num_batch), rans32_push, rans32_pop),
            ("Rans32", "cuda", functools.partial(rans32_init_stream, num_batch), rans32_push, rans32_pop),
            # Interleaved methods (CPU only)
            ("Rans32(4way-interleaved)", "cpu", functools.partial(rans32_init_stream, num_batch, 4), rans32_i4_push, rans32_i4_pop),
            ("Rans32/16", "cpu", functools.partial(rans32_16_init_stream, num_batch), rans32_16_push, rans32_16_pop),
            ("Rans32/16", "cuda", functools.partial(rans32_16_init_stream, num_batch), rans32_16_push, rans32_16_pop),
            ("Rans32/16(4way-interleaved)", "cpu", functools.partial(rans32_16_init_stream, num_batch, 4), rans32_16_i4_push, rans32_16_i4_pop),
            ("Rans64", "cpu", functools.partial(rans64_init_stream, num_batch), rans64_push, rans64_pop),
            ("Rans64", "cuda", functools.partial(rans64_init_stream, num_batch), rans64_push, rans64_pop),
            ("Rans64(4way-interleaved)", "cpu", functools.partial(rans64_init_stream, num_batch, 4), rans64_i4_push, rans64_i4_pop),
        ]

        # Alias methods
        alias_methods = [
            ("Rans32(alias)", "cpu",
                functools.partial(rans32_init_stream, num_batch),
                functools.partial(rans32_alias_push, indexes=indexes, cdfs=cdfs_with_alias_remap, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                functools.partial(rans32_alias_pop, indexes=indexes, cdfs=cdfs_with_alias_table, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision)),
            ("Rans32(alias)", "cuda",
                functools.partial(rans32_init_stream, num_batch),
                functools.partial(rans32_alias_push, indexes=indexes, cdfs=cdfs_with_alias_remap, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                functools.partial(rans32_alias_pop, indexes=indexes, cdfs=cdfs_with_alias_table, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision)),
            ("Rans64(alias)", "cpu",
                functools.partial(rans64_init_stream, num_batch),
                functools.partial(rans64_alias_push, indexes=indexes, cdfs=cdfs_with_alias_remap, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                functools.partial(rans64_alias_pop, indexes=indexes, cdfs=cdfs_with_alias_table, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision)),
            ("Rans64(alias)", "cuda",
                functools.partial(rans64_init_stream, num_batch),
                functools.partial(rans64_alias_push, indexes=indexes, cdfs=cdfs_with_alias_remap, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                functools.partial(rans64_alias_pop, indexes=indexes, cdfs=cdfs_with_alias_table, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision)),
        ]

        with torch.no_grad():
            # Test non-alias methods
            for name, device, init_func, push_func, pop_func in non_alias_methods:
                if not torch.cuda.is_available() and device == "cuda":
                    continue
                self._test_torch_ans(
                    name + f" ({device})",
                    data,
                    init_func=init_func,
                    push_func=functools.partial(push_func, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                    pop_func=functools.partial(pop_func, indexes=indexes, cdfs=cdfs, cdfs_sizes=cdfs_sizes, offsets=offsets, freq_precision=freq_precision),
                    device=device
                )

            # Test alias methods
            for name, device, init_func, push_func, pop_func in alias_methods:
                self._test_torch_ans(
                    name + f" ({device})",
                    data,
                    init_func=init_func,
                    push_func=push_func,
                    pop_func=pop_func,
                    device=device
                )

    # Test for high-level API (TorchANSInterface)
    def test_high_level_api_encode_decode(self):
        import torch
        from torch_ans.utils import TorchANSInterface

        # Prepare PMF and convert to quantized CDF
        batch_size = 8
        num_dists = 8
        num_symbols = 256
        pmf = torch.randint(1, 100, (num_dists, num_symbols), dtype=torch.int32)
        num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32)
        offsets = torch.zeros(num_dists, dtype=torch.int32)

        # Create coder interface
        coder = TorchANSInterface(impl="rans64", freq_precision=16, device="cpu")
        coder.init_params(pmf, num_freqs, offsets)

        # Prepare symbols and indexes
        seq_len = 1024
        symbols = torch.randint(0, num_symbols, (batch_size, seq_len), dtype=torch.int32)
        indexes = torch.randint(0, num_dists, (batch_size, seq_len), dtype=torch.int32)

        # Encode
        encoded = coder.encode_with_indexes(symbols, indexes)

        # Decode
        decoded = coder.decode_with_indexes(encoded, indexes)

        # Check correctness
        self.assertTrue(torch.equal(decoded, symbols), "High-level API decode does not match input symbols")

    # def test_torchac_cuda(self):
        # import arithmetic

        # start_time = time.time()
        # cdfs_float = cdfs[indexes.long().reshape(-1)].float().cuda().reshape(-1, cdfs.shape[-1]) / (1<<freq_precision)
        # data_in = data.short().cuda().reshape(-1)
        # print(f"torchac(cuda) init time: {time.time() - start_time}")
        # start_time = time.time()
        # # print(stream, data)
        # (byte_stream_torch, cnt_torch) = arithmetic.arithmetic_encode(
        #     data_in,
        #     cdfs_float,
        #     int(cdfs_float.shape[0] / num_batch),
        #     int(cdfs_float.shape[0]),
        #     int(cdfs_float.shape[1])
        # )
        
        # # byte_strings = rans_stream_to_byte_strings(stream.cpu())
        # # print(stream, data)
        # torch.cuda.synchronize()
        # print(f"torchac(cuda) encoding time: {time.time() - start_time}")
        # print(f"torchac(cuda) encoding bytes: {len(byte_stream_torch)+len(cnt_torch)*4}")
        # start_time = time.time()

        # decoded = arithmetic.arithmetic_decode(
        #     cdfs_float,
        #     byte_stream_torch,
        #     cnt_torch,
        #     int(cdfs_float.shape[0] / num_batch),
        #     int(cdfs_float.shape[0]),
        #     int(cdfs_float.shape[1])
        # )
        # # print(stream, decoded)
        # torch.cuda.synchronize()
        # print(f"torchac(cuda) decoding time: {time.time() - start_time}")
        
        # self.assertFalse((data_in.cpu() != decoded.cpu()).any())



if __name__ == "__main__":
    unittest.main()