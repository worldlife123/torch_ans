import argparse
import time
from typing import Iterable, List, Tuple

import torch
# import torch_ans
# from torch_ans._C import rans_pmf_to_quantized_cdf, rans64_init_stream, rans64_push, rans64_pop
from torch_ans.utils import TorchANSInterface

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_DEVICES = ["cpu", "cuda"]
DEFAULT_MODES = ["push", "pop", "both"]
# (impl, num_interleaves) pairs that torch_ans can build. Interleaved CUDA
# coding is rans32_16-only and maps to the warp-level kernels (rans_warp_cuda.cuh);
# the CPU supports the same interleavings via rans_cpu.cpp.
DEFAULT_INTERLEAVES = [1, 4, 32]


def benchmark_parallel_states(
    batch_sizes: Iterable[int],
    data_size_mb: float = 50.0,
    device: str = "cpu",
    mode: str = "both",
    num_symbols: int = 256,
    num_dists: int = 8,
    freq_precision: int = 16,
    impl: str = "rans64",
    num_interleaves: int = 1,
    warmup: int = 0,
    repeat: int = 1,
) -> List[Tuple[int, float, float]]:
    """Benchmark rANS throughput for a list of parallel batch sizes.

    `impl` and `num_interleaves` select the rANS variant: "rans32_16" with
    num_interleaves=32 uses the warp-level interleaved kernels on CUDA. Note that
    freq_precision is capped per implementation (rans64: 31, rans32: 23,
    rans32_16: 15), so pass a freq_precision valid for every implementation when
    comparing them.
    """
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    if device == "cpu":
        print("num threads:", torch.get_num_threads())  # Ensure torch is initialized to get accurate CPU performance

    if mode not in DEFAULT_MODES:
        raise ValueError(f"Invalid benchmark mode: {mode}. Supported modes: {DEFAULT_MODES}")

    freqs = torch.randint(0, num_symbols, (num_dists, num_symbols), dtype=torch.float32, device=device)
    num_freqs = torch.full((num_dists,), num_symbols, dtype=torch.int32, device=device)
    # cdfs = rans_pmf_to_quantized_cdf(freqs / freqs.sum(dim=-1, keepdim=True), freq_precision)
    # cdfs_sizes = torch.full((num_dists,), cdfs.size(-1), dtype=torch.int32, device=device)
    offsets = torch.zeros(num_dists, dtype=torch.int32, device=device)
    results: List[Tuple[int, float, float]] = []

    for batch_size in batch_sizes:
        # num_data = int(data_size_mb * 1024 * 1024 / batch_size / 4)
        # symbols = torch.randint(0, num_symbols, (batch_size, num_data), dtype=torch.int32, device=device)
        symbols = torch.randint(0, num_symbols, (int(data_size_mb * 1024 * 1024 / 4) - 5, ), dtype=torch.int32, device=device)
        # stream = rans64_init_stream(batch_size).to(device=device)
        indexes = torch.zeros_like(symbols)

        ans_interface = TorchANSInterface(
            num_parallel_states=batch_size, 
            device=device, 
            bypass_coding=False, 
            freq_precision=freq_precision,
            impl=impl,
            num_interleaves=num_interleaves,
        )
        ans_interface.init_params(freqs, num_freqs, offsets)
        # ans_interface.set_cdfs(cdfs, cdfs_sizes, offsets)

        def _timed_call() -> float:
            """Runs one iteration and returns the seconds spent in the timed part."""
            if mode == "push":
                start = time.time()
                ans_interface.encode(symbols, indexes)
                if device == "cuda":
                    torch.cuda.synchronize()
                return time.time() - start
            if mode == "pop":
                # decoding consumes the stream, so encode a fresh one per run
                # (outside of the timed region, as in the single-shot version)
                stream_ = ans_interface.encode(symbols, indexes)
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.time()
                ans_interface.decode(stream_, indexes)
                if device == "cuda":
                    torch.cuda.synchronize()
                return time.time() - start
            start = time.time()
            stream_ = ans_interface.encode(symbols, indexes)
            ans_interface.decode(stream_, indexes)
            if device == "cuda":
                torch.cuda.synchronize()
            return time.time() - start

        for _ in range(max(0, warmup)):
            _timed_call()

        repeats = max(1, repeat)
        elapsed = sum(_timed_call() for _ in range(repeats)) / repeats
        throughput = data_size_mb / elapsed if elapsed > 0 else float("inf")
        results.append((batch_size, elapsed, throughput))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark torch_ans rANS throughput across batch sizes and devices."
    )
    parser.add_argument(
        "-b",
        "--batch-sizes",
        nargs="+",
        type=int,
        default=DEFAULT_BATCH_SIZES,
        help="Space-separated batch sizes to test.",
    )
    parser.add_argument(
        "-d",
        "--devices",
        nargs="+",
        # default=["cpu"],
        default=DEFAULT_DEVICES,
        help="Devices to benchmark. Use cpu and/or cuda.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=DEFAULT_MODES,
        default="push",
        help="Benchmark mode: push, pop, or both.",
    )
    parser.add_argument(
        "--data-size-mb",
        type=float,
        default=200.0,
        help="Total data size per benchmark iteration in megabytes.",
    )
    parser.add_argument(
        "--num-symbols",
        type=int,
        default=256,
        help="Number of symbols in the uniform distribution.",
    )
    parser.add_argument(
        "--freq-precision",
        type=int,
        default=16,
        help="Frequency precision for the quantized CDF.",
    )
    parser.add_argument(
        "--impl",
        default="rans64",
        help="rANS implementation: rans64, rans32 or rans32_16.",
    )
    parser.add_argument(
        "-i",
        "--num-interleaves",
        type=int,
        default=1,
        help="Number of interleaved rANS states (1, 4 or 32; 32 requires rans32_16).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of untimed warm-up iterations per batch size.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of timed iterations per batch size (the mean is reported).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("torch_ans benchmark")
    print(f"  impl: {args.impl}")
    print(f"  num interleaves: {args.num_interleaves}")
    print(f"  mode: {args.mode}")
    print(f"  data size: {args.data_size_mb} MB")
    print(f"  num symbols: {args.num_symbols}")
    print(f"  freq precision: {args.freq_precision}")
    print(" ")

    for device in args.devices:
        print(f"Benchmarking on {device}:")
        try:
            results = benchmark_parallel_states(
                batch_sizes=args.batch_sizes,
                data_size_mb=args.data_size_mb,
                device=device,
                mode=args.mode,
                num_symbols=args.num_symbols,
                freq_precision=args.freq_precision,
                impl=args.impl,
                num_interleaves=args.num_interleaves,
                warmup=args.warmup,
                repeat=args.repeat,
            )
        except RuntimeError as exc:
            print(f"  skipped {device}: {exc}")
            continue

        for batch_size, elapsed, throughput in results:
            print(
                f"  Batch size: {batch_size:4d} | Time: {elapsed:.4f} s | Throughput: {throughput:.2f} MB/s"
            )
        print(" ")


if __name__ == "__main__":
    main()
