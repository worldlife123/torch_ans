"""Tests for the warp-level 32-way interleaved rANS CUDA kernels.

The CUDA warp kernels (rans_warp_cuda.cuh) are designed to be bit-compatible
with the CPU interleaved implementation (rans_cpu.cpp with
NUM_INTERLEAVES=32): for the same input they produce byte-identical streams
and can decode each other's streams. These tests verify that invariant.
"""

import unittest

import torch

from torch_ans._C import (
    rans_pmf_to_quantized_cdf,
    rans_stream_to_byte_strings,
    rans_byte_strings_to_stream,
    rans32_16_init_stream,
    rans32_16_push,
    rans32_16_pop,
    rans32_16_i4_push,
    rans32_16_i4_pop,
    rans32_16_i32_push,
    rans32_16_i32_pop,
    rans64_i4_push,
)
from torch_ans.utils import TorchANSInterface

NUM_DISTS = 8
NUM_SYMBOLS = 7  # symbols per distribution (in-range values 0..6)


def _generate_rans_params(freq_precision=12, bypass=True):
    """Builds quantized CDFs (with bypass tail) following tests/torch_ans_test.py.

    For bypass=False the caller must guarantee (symbol - offset) stays in
    [0, cdf_size - 2); zero offsets are used to make that trivially true.
    """
    pmfs = torch.randint(1, 1024, (NUM_DISTS, NUM_SYMBOLS), dtype=torch.float32)
    pmfs = torch.cat([pmfs.clone(), torch.ones(NUM_DISTS, 1).type_as(pmfs)], dim=1)
    pmfs = pmfs / pmfs.sum(dim=-1, keepdim=True)
    cdfs = rans_pmf_to_quantized_cdf(pmfs, precision=freq_precision)
    cdfs_sizes = torch.zeros(NUM_DISTS, dtype=torch.int32) + NUM_SYMBOLS + 2  # bypass tail included
    if bypass:
        offsets = torch.randint(-4, 4, (NUM_DISTS,), dtype=torch.int32)
    else:
        offsets = torch.zeros(NUM_DISTS, dtype=torch.int32)
    return cdfs, cdfs_sizes, offsets


def _generate_data(shape, seed, invalid_ratio=0.0, clamp_to_range=False):
    """Generates symbols covering in-range, sentinel and out-of-range (bypass) values.

    Returns (data, indexes, expected) where expected is the decode reference:
    symbols at invalid index positions decode to 0. With clamp_to_range=True,
    symbols are clamped to the in-range [0, NUM_SYMBOLS) before the expected
    reference is computed (required for bypass=False coding).
    """
    g = torch.Generator().manual_seed(seed)
    data = torch.randint(-6, NUM_SYMBOLS + 10, shape, generator=g, dtype=torch.int32)
    indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)
    if clamp_to_range:
        data = data.clamp(0, NUM_SYMBOLS - 1)
    if invalid_ratio > 0:
        invalid_mask = torch.rand(shape, generator=g) < invalid_ratio
        indexes = indexes.masked_fill(invalid_mask, -1)
        data = data.masked_fill(invalid_mask, 123)  # skipped by the coder
    expected = data.clone()
    expected[indexes < 0] = 0
    return data, indexes, expected


def _push(pop_params, init_interleaves=32):
    """Returns the push function bound to fresh per-test parameters."""
    cdfs, cdfs_sizes, offsets = pop_params
    init = rans32_16_init_stream
    push = rans32_16_i32_push if init_interleaves == 32 else rans32_16_i4_push
    pop = rans32_16_i32_pop if init_interleaves == 32 else rans32_16_i4_pop

    def do_push(stream, data, indexes, freq_precision, bypass):
        push(stream, data, indexes, cdfs, cdfs_sizes, offsets,
             freq_precision=freq_precision, bypass_coding=bypass, bypass_precision=4)

    def do_pop(stream, indexes, freq_precision, bypass):
        return pop(stream, indexes, cdfs, cdfs_sizes, offsets,
                   freq_precision=freq_precision, bypass_coding=bypass, bypass_precision=4)

    def do_init(batch):
        # no preallocate_size: rans_push grows the stream to a worst-case size
        # before coding, so the initial tensor only needs to hold the states
        return init(batch, init_interleaves)

    return do_init, do_push, do_pop


class TestRansWarpCuda(unittest.TestCase):

    def _roundtrip(self, shape, seed, freq_precision=12, bypass=True, invalid_ratio=0.0,
                   interleaves=32):
        """GPU (or CPU) push -> serialize -> deserialize -> pop roundtrip."""
        data, indexes, expected = _generate_data(shape, seed, invalid_ratio,
                                                 clamp_to_range=not bypass)
        params = _generate_rans_params(freq_precision, bypass=bypass)
        do_init, do_push, do_pop = _push(params, interleaves)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        stream = do_init(shape[0]).to(device)
        try:
            do_push(stream, data.to(device), indexes.to(device), freq_precision, bypass)
        except RuntimeError as e:
            if device == "cuda" and "not compiled with GPU support" in str(e):
                self.skipTest("torch_ans is not compiled with GPU support")
            raise
        if device == "cuda":
            torch.cuda.synchronize()

        byte_strings = rans_stream_to_byte_strings(stream.cpu())
        stream = rans_byte_strings_to_stream(byte_strings).to(device)

        try:
            decoded = do_pop(stream, indexes.to(device), freq_precision, bypass)
        except RuntimeError as e:
            if device == "cuda" and "not compiled with GPU support" in str(e):
                self.skipTest("torch_ans is not compiled with GPU support")
            raise
        if device == "cuda":
            torch.cuda.synchronize()

        self.assertTrue(torch.equal(expected, decoded.cpu()),
                        f"roundtrip mismatch for shape={shape}, seed={seed}, "
                        f"precision={freq_precision}, bypass={bypass}, invalid={invalid_ratio}")
    # ------------------------------------------------------------------
    # CPU 32-way interleaved roundtrip (runs everywhere; also exercises the
    # CPU NUM_INTERLEAVES=32 instantiation used as bit-compat reference)
    # ------------------------------------------------------------------

    def test_cpu_i32_roundtrip(self):
        data, indexes, expected = _generate_data((3, 200), seed=1, invalid_ratio=0.1)
        params = _generate_rans_params(12)
        do_init, do_push, do_pop = _push(params, 32)
        stream = do_init(3)
        do_push(stream, data, indexes, 12, True)
        decoded = do_pop(stream, indexes, 12, True)
        self.assertTrue(torch.equal(expected, decoded))

    def test_cpu_i32_roundtrip_no_bypass(self):
        data, indexes, expected = _generate_data((2, 96), seed=2, clamp_to_range=True)
        params = _generate_rans_params(8, bypass=False)
        do_init, do_push, do_pop = _push(params, 32)
        stream = do_init(2)
        do_push(stream, data, indexes, 8, False)
        decoded = do_pop(stream, indexes, 8, False)
        self.assertTrue(torch.equal(expected, decoded))

    # ------------------------------------------------------------------
    # GPU warp-32 roundtrips across a parameter sweep
    # ------------------------------------------------------------------

    def test_cuda_warp32_roundtrip_sweep(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        shapes = [(1, 1), (3, 2), (2, 31), (2, 32), (1, 33), (2, 64), (5, 257), (7, 1000)]
        seed = 10
        for shape in shapes:
            for freq_precision in (8, 15):
                for bypass in (True, False):
                    self._roundtrip(shape, seed, freq_precision=freq_precision, bypass=bypass)
                    seed += 1

    def test_cuda_warp32_roundtrip_invalid_indexes(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        self._roundtrip((3, 130), seed=50, invalid_ratio=0.3)

    # ------------------------------------------------------------------
    # Bit-compatibility: CPU i32 stream == GPU warp32 stream, cross-decoding
    # ------------------------------------------------------------------

    def _bit_compat_case(self, shape, seed, freq_precision, bypass, invalid_ratio,
                         interleaves=32, data=None):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        if data is None:
            data, indexes, expected = _generate_data(shape, seed, invalid_ratio,
                                                     clamp_to_range=not bypass)
        else:
            # caller-supplied (symbols, indexes, expected) triple
            data, indexes, expected = data
        params = _generate_rans_params(freq_precision, bypass=bypass)
        do_init, do_push, do_pop = _push(params, interleaves)

        cpu_stream = do_init(shape[0])
        do_push(cpu_stream, data, indexes, freq_precision, bypass)

        gpu_stream = do_init(shape[0]).cuda()
        do_push(gpu_stream, data.cuda(), indexes.cuda(), freq_precision, bypass)
        torch.cuda.synchronize()

        # byte-identical streams (lengths and used bytes)
        cpu_bytes = rans_stream_to_byte_strings(cpu_stream)
        gpu_bytes = rans_stream_to_byte_strings(gpu_stream.cpu())
        self.assertEqual(cpu_bytes, gpu_bytes)
        self.assertTrue(torch.equal(cpu_stream[:, 0], gpu_stream.cpu()[:, 0]))

        # cross-decoding: CPU stream on GPU, GPU stream on CPU
        decoded_on_gpu = do_pop(gpu_stream, indexes.cuda(), freq_precision, bypass)
        decoded_on_cpu = do_pop(cpu_stream, indexes, freq_precision, bypass)
        self.assertTrue(torch.equal(expected, decoded_on_gpu.cpu()))
        self.assertTrue(torch.equal(expected, decoded_on_cpu))

    def test_cuda_warp32_bit_compat_bypass(self):
        self._bit_compat_case((4, 300), seed=60, freq_precision=12, bypass=True, invalid_ratio=0.0)

    def test_cuda_warp32_bit_compat_no_bypass(self):
        self._bit_compat_case((4, 300), seed=61, freq_precision=15, bypass=False, invalid_ratio=0.0)

    def test_cuda_warp32_bit_compat_tail_and_invalid(self):
        # num_symbols % 32 != 0 exercises the sequential tail path
        self._bit_compat_case((3, 77), seed=62, freq_precision=8, bypass=True, invalid_ratio=0.2)

    def test_cuda_warp32_bit_compat_all_bypass(self):
        """Every symbol is out of range, so every lane emits a raw-value run.

        This is the regression case for the bypass-run placement: the runs of a
        symbol group must be laid out consecutively in lane-descending order.
        Overlapping (or gapped) runs only show up when several lanes of the
        same group emit words at once.
        """
        g = torch.Generator().manual_seed(63)
        shape = (4, 256)
        # all values are >= max_value -> sentinel + raw-value bypass coding
        data = torch.randint(NUM_SYMBOLS + 40, NUM_SYMBOLS + 4000, shape, generator=g,
                             dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)
        self._bit_compat_case(shape, seed=63, freq_precision=12, bypass=True,
                              invalid_ratio=0.0, data=(data, indexes, data))

    def test_cuda_i4_bit_compat(self):
        """Sub-warp (4-lane) groups: regression for the group-local lane masks.

        The lane masks are computed relative to the group's position inside the
        warp, which only differs from the warp-relative mask when
        NUM_INTERLEAVES < 32.
        """
        self._bit_compat_case((4, 300), seed=64, freq_precision=12, bypass=True,
                              invalid_ratio=0.0, interleaves=4)

    def test_cuda_i4_bit_compat_all_bypass(self):
        g = torch.Generator().manual_seed(65)
        shape = (4, 128)
        data = torch.randint(NUM_SYMBOLS + 40, NUM_SYMBOLS + 4000, shape, generator=g,
                             dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)
        self._bit_compat_case(shape, seed=65, freq_precision=12, bypass=True,
                              invalid_ratio=0.0, interleaves=4, data=(data, indexes, data))

    # ------------------------------------------------------------------
    # 4-way interleaved CUDA (regression for the sub-warp group path)
    # ------------------------------------------------------------------

    def test_cuda_i4_roundtrip(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        data, indexes, expected = _generate_data((6, 140), seed=70, invalid_ratio=0.1)
        params = _generate_rans_params(12)
        do_init, do_push, do_pop = _push(params, 4)
        stream = do_init(6).cuda()
        do_push(stream, data.cuda(), indexes.cuda(), 12, True)
        torch.cuda.synchronize()
        decoded = do_pop(stream, indexes.cuda(), 12, True)
        self.assertTrue(torch.equal(expected, decoded.cpu()))

    def test_cuda_i4_unsupported_variant_raises(self):
        # interleaved CUDA coding is rans32_16-only; other variants must fail loudly
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        data = torch.randint(0, 4, (2, 64), dtype=torch.int32)
        indexes = torch.randint(0, 2, (2, 64), dtype=torch.int32)
        cdfs = torch.tensor([[0, 128, 256], [0, 64, 256]], dtype=torch.int32)
        cdfs_sizes = torch.tensor([3, 3], dtype=torch.int32)
        offsets = torch.zeros(2, dtype=torch.int32)
        stream = torch.zeros(2, 4096, dtype=torch.int32).cuda()
        stream[:, 0] = 8  # one rans64 state
        with self.assertRaises(RuntimeError):
            rans64_i4_push(
                stream, data.cuda(), indexes.cuda(), cdfs.cuda(), cdfs_sizes.cuda(), offsets.cuda(),
                freq_precision=8, bypass_coding=False)

    def test_bypass_large_out_of_range_values(self):
        """Regression for the bypass digit counter shifting by >= 32.

        A symbol far out of range makes the raw value need
        ceil(symbol_bits / bypass_precision) bypass digits, i.e. a shift amount
        of 32 in the digit-counting loop of rans_push_raw_value_step. That is
        undefined behaviour: x86 wraps the shift count (infinite loop on CPU)
        while CUDA clamps it, so the two backends also disagreed.
        """
        for excess in (10 ** 6, 2 * 10 ** 8, 10 ** 9):
            for interleaves in (4, 32):
                for freq_precision in (8, 15):
                    with self.subTest(excess=excess, interleaves=interleaves,
                                      freq_precision=freq_precision):
                        params = _generate_rans_params(freq_precision, bypass=True)
                        do_init, do_push, do_pop = _push(params, interleaves)
                        shape = (2, 64)
                        data = torch.zeros(shape, dtype=torch.int32) + NUM_SYMBOLS + excess
                        indexes = torch.zeros(shape, dtype=torch.int32)
                        expected = data.clone()

                        cpu_stream = do_init(shape[0])
                        do_push(cpu_stream, data, indexes, freq_precision, True)
                        self.assertTrue(
                            torch.equal(expected, do_pop(cpu_stream, indexes, freq_precision, True)))

                        if not torch.cuda.is_available():
                            continue
                        gpu_stream = do_init(shape[0]).cuda()
                        do_push(gpu_stream, data.cuda(), indexes.cuda(), freq_precision, True)
                        torch.cuda.synchronize()
                        decoded = do_pop(gpu_stream, indexes.cuda(), freq_precision, True)
                        self.assertTrue(torch.equal(expected, decoded.cpu()))
                        # the CPU and the warp kernel must agree bit for bit
                        self.assertEqual(rans_stream_to_byte_strings(cpu_stream),
                                         rans_stream_to_byte_strings(gpu_stream.cpu()))

    def test_stream_resize_covers_worst_case(self):
        """rans_push's auto-resize must cover the worst-case bit rate.

        A symbol with frequency 1 costs the full ``freq_precision`` bits, which
        is exactly what the resize bound assumes per symbol, so it leaves no
        headroom for an off-by-one in the estimate.
        """
        for freq_precision in (8, 15):
            # cdf [0, 1, 2^p] puts all the mass on symbol 0 with frequency 1
            cdfs = torch.tensor([[0, 1, 1 << freq_precision]], dtype=torch.int32)
            cdfs_sizes = torch.tensor([3], dtype=torch.int32)
            offsets = torch.zeros(1, dtype=torch.int32)
            for interleaves in (4, 32):
                push = rans32_16_i32_push if interleaves == 32 else rans32_16_i4_push
                pop = rans32_16_i32_pop if interleaves == 32 else rans32_16_i4_pop
                for num_symbols in (64, 500, 2000):
                    with self.subTest(freq_precision=freq_precision, interleaves=interleaves,
                                      num_symbols=num_symbols):
                        shape = (2, num_symbols)
                        data = torch.zeros(shape, dtype=torch.int32)
                        indexes = torch.zeros(shape, dtype=torch.int32)
                        stream = rans32_16_init_stream(shape[0], interleaves)
                        push(stream, data, indexes, cdfs, cdfs_sizes, offsets,
                             freq_precision=freq_precision, bypass_coding=False,
                             bypass_precision=4)
                        # the tensor must be large enough for the bytes actually used
                        used_bytes = int(stream[:, 0].max())
                        self.assertLessEqual((used_bytes + 3) // 4, stream.size(1))
                        decoded = pop(stream, indexes, cdfs, cdfs_sizes, offsets,
                                      freq_precision=freq_precision, bypass_coding=False,
                                      bypass_precision=4)
                        self.assertTrue(torch.equal(data, decoded))

    # ------------------------------------------------------------------
    # High-level TorchANSInterface wiring
    # ------------------------------------------------------------------

    def _make_coder_params(self, seed=80):
        g = torch.Generator().manual_seed(seed)
        freqs = torch.randint(1, 100, (NUM_DISTS, NUM_SYMBOLS), generator=g).float()
        num_freqs = torch.zeros(NUM_DISTS, dtype=torch.int32) + NUM_SYMBOLS
        offsets = torch.randint(-2, 2, (NUM_DISTS,), generator=g, dtype=torch.int32)
        return freqs, num_freqs, offsets

    def test_high_level_warp32_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        freqs, num_freqs, offsets = self._make_coder_params()
        coder = TorchANSInterface(
            impl="rans32_16", freq_precision=12, bypass_coding=True,
            device="cuda", num_interleaves=32)
        coder.init_params(freqs, num_freqs, offsets)

        g = torch.Generator().manual_seed(81)
        shape = (4, 150)
        symbols = torch.randint(0, NUM_SYMBOLS, shape, generator=g, dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)

        stream = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(stream, indexes)
        self.assertTrue(torch.equal(symbols, decoded.cpu()))

    def test_high_level_warp32_streaming_cuda(self):
        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available")
        freqs, num_freqs, offsets = self._make_coder_params(seed=82)
        coder = TorchANSInterface(
            impl="rans32_16", freq_precision=12, bypass_coding=True,
            device="cuda", num_interleaves=32)
        coder.init_params(freqs, num_freqs, offsets)

        g = torch.Generator().manual_seed(83)
        shapes = [(4, 64), (4, 36)]
        symbols1 = torch.randint(0, NUM_SYMBOLS, shapes[0], generator=g, dtype=torch.int32)
        symbols2 = torch.randint(0, NUM_SYMBOLS, shapes[1], generator=g, dtype=torch.int32)
        indexes1 = torch.randint(0, NUM_DISTS, shapes[0], generator=g, dtype=torch.int32)
        indexes2 = torch.randint(0, NUM_DISTS, shapes[1], generator=g, dtype=torch.int32)

        # cache=True returns None; flush encodes the queue in LIFO order, i.e.
        # symbols2 is pushed first and symbols1 last, so rANS (last-in-first-out)
        # pops symbols1 first
        self.assertIsNone(coder.encode_with_indexes(symbols1, indexes1, cache=True))
        self.assertIsNone(coder.encode_with_indexes(symbols2, indexes2, cache=True))
        stream = coder.encode_flush()
        self.assertIsInstance(stream, torch.Tensor)

        # both segments live on the same stream: two push calls, one pop call
        stacked_indexes = torch.cat([indexes1, indexes2], dim=1)
        decoded = coder.decode(stream, dist_indexes=stacked_indexes)
        expected = torch.cat([symbols1, symbols2], dim=1)
        self.assertTrue(torch.equal(expected, decoded.cpu()))

    def test_high_level_warp32_cpu(self):
        freqs, num_freqs, offsets = self._make_coder_params(seed=84)
        coder = TorchANSInterface(
            impl="rans32_16", freq_precision=12, bypass_coding=True,
            device="cpu", num_interleaves=32)
        coder.init_params(freqs, num_freqs, offsets)

        g = torch.Generator().manual_seed(85)
        shape = (3, 100)
        symbols = torch.randint(0, NUM_SYMBOLS, shape, generator=g, dtype=torch.int32)
        indexes = torch.randint(0, NUM_DISTS, shape, generator=g, dtype=torch.int32)

        stream = coder.encode_with_indexes(symbols, indexes)
        decoded = coder.decode_with_indexes(stream, indexes)
        self.assertTrue(torch.equal(symbols, decoded))

    def test_high_level_rans32_16_impl_resolution(self):
        # regression: impl="rans32_16" must resolve to the rans32_16 functions
        # (previously shadowed by the "rans32" startswith branch)
        coder = TorchANSInterface(impl="rans32_16", device="cpu")
        from torch_ans._C import rans32_16_push as bound_push
        self.assertIs(coder.ans_encode_func, bound_push)
        self.assertEqual(coder.freq_precision, min(coder.freq_precision, 15))


if __name__ == "__main__":
    unittest.main()
