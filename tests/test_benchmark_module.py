import unittest

import torch

from torch_ans.benchmark import benchmark_parallel_states

from cuda_helpers import is_cuda_unavailable_error, require_cuda_coding


class TestTorchANSBenchmarkCLI(unittest.TestCase):

    def test_benchmark_parallel_states_cpu_returns_results(self):
        results = benchmark_parallel_states(batch_sizes=[1, 16], data_size_mb=1.0, device="cpu")
        self.assertTrue(isinstance(results, list))
        self.assertTrue(len(results) == 2)
        for batch_size, elapsed, throughput in results:
            self.assertTrue(batch_size in (1, 16))
            self.assertTrue(elapsed >= 0.0)
            self.assertTrue(throughput >= 0.0)


    def test_benchmark_parallel_states_cuda_skips_when_unavailable(self):
        require_cuda_coding(self)
        try:
            results = benchmark_parallel_states(batch_sizes=[1024], data_size_mb=1.0, device="cuda")
            self.assertTrue(len(results) == 1)
            self.assertTrue(results[0][0] == 1024)
        except RuntimeError as exc:
            if is_cuda_unavailable_error(exc):
                self.skipTest(f"CUDA coding is unavailable: {exc}")
            raise
        

    def test_benchmark_parallel_states_cpu_pop_mode(self):
        results = benchmark_parallel_states(batch_sizes=[1, 8], data_size_mb=1.0, device="cpu", mode="pop")
        self.assertEqual(len(results), 2)
        for batch_size, elapsed, throughput in results:
            self.assertIn(batch_size, (1, 8))
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertGreaterEqual(throughput, 0.0)

    def test_benchmark_parallel_states_cpu_both_mode(self):
        results = benchmark_parallel_states(batch_sizes=[2], data_size_mb=1.0, device="cpu", mode="both")
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0], 2)
