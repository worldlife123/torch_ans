import argparse
import time
from typing import Iterable, List, Tuple

import torch
import torch_ans

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
DEFAULT_DEVICES = ["cpu", "cuda"]
DEFAULT_MODES = ["push", "pop", "both"]


def benchmark_parallel_states(
    batch_sizes: Iterable[int],
    data_size_mb: float = 50.0,
    device: str = "cpu",
    mode: str = "both",
    num_symbols: int = 256,
    freq_precision: int = 16,
) -> List[Tuple[int, float, float]]:
    """Benchmark rANS throughput for a list of parallel batch sizes."""
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment")

    if device == "cpu":
        print("num threads:", torch.get_num_threads())  # Ensure torch is initialized to get accurate CPU performance

    if mode not in DEFAULT_MODES:
        raise ValueError(f"Invalid benchmark mode: {mode}. Supported modes: {DEFAULT_MODES}")

    pmf = torch.ones(1, num_symbols, dtype=torch.float32, device=device) / num_symbols
    cdfs = torch_ans.rans_pmf_to_quantized_cdf(pmf, freq_precision)
    cdfs_sizes = torch.full((1,), cdfs.size(-1), dtype=torch.int32, device=device)
    offsets = torch.zeros(1, dtype=torch.int32, device=device)
    results: List[Tuple[int, float, float]] = []

    for batch_size in batch_sizes:
        num_data = int(data_size_mb * 1024 * 1024 / batch_size / 4)
        symbols = torch.randint(0, num_symbols, (batch_size, num_data), dtype=torch.int32, device=device)
        stream = torch_ans.rans64_init_stream(batch_size).to(device=device)
        indexes = torch.zeros_like(symbols)

        if mode == "push":
            start = time.time()
            torch_ans.rans64_push(
                stream,
                symbols,
                indexes,
                cdfs,
                cdfs_sizes,
                offsets,
                freq_precision,
                bypass_coding=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
        elif mode == "pop":
            torch_ans.rans64_push(
                stream,
                symbols,
                indexes,
                cdfs,
                cdfs_sizes,
                offsets,
                freq_precision,
                bypass_coding=False,
            )
            start = time.time()
            decoded = torch_ans.rans64_pop(
                stream,
                indexes,
                cdfs,
                cdfs_sizes,
                offsets,
                freq_precision,
                bypass_coding=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            _ = decoded
        else:  # both
            start = time.time()
            torch_ans.rans64_push(
                stream,
                symbols,
                indexes,
                cdfs,
                cdfs_sizes,
                offsets,
                freq_precision,
                bypass_coding=False,
            )
            decoded = torch_ans.rans64_pop(
                stream,
                indexes,
                cdfs,
                cdfs_sizes,
                offsets,
                freq_precision,
                bypass_coding=False,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            _ = decoded

        elapsed = time.time() - start
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
        default=50.0,
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
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print("torch_ans benchmark")
    print(f"  impl: rans64") # TODO: support other implementations in the future
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
