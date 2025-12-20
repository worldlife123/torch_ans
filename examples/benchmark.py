import torch
import torch_ans
import time

def benchmark_parallel_states(data_size_mb=50, device="cpu"):
  num_symbols = 256
  freq_precision = 16
  sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
  pmf = torch.ones(1, num_symbols, dtype=torch.float32, device=device) / num_symbols
  cdfs = torch_ans.rans_pmf_to_quantized_cdf(pmf, freq_precision)
  cdfs_sizes = torch.full((1,), cdfs.size(-1), dtype=torch.int32, device=device)
  offsets = torch.zeros(1, dtype=torch.int32, device=device)
  results = []
  for batch_size in sizes:
    num_data = int(data_size_mb * 1024 * 1024 / batch_size / 4)
    symbols = torch.randint(0, num_symbols, (batch_size, num_data), dtype=torch.int32, device=device)
    stream = torch_ans.rans64_init_stream(batch_size).to(device=device)
    start = time.time()
    torch_ans.rans64_push(stream, symbols, torch.zeros_like(symbols), cdfs, cdfs_sizes, offsets, freq_precision, bypass_coding=False)
    if device == "cuda":
      torch.cuda.synchronize()
    elapsed = time.time() - start
    results.append((batch_size, elapsed))
    print(f"Batch size: {batch_size:4d} | Time: {elapsed:.4f} s | Throughput: {(data_size_mb/elapsed):.2f} MB/s")
  return results

# Example usage:
print("Benchmarking on CPU:")
benchmark_parallel_states(data_size_mb=50, device="cpu")
# For GPU:
print("Benchmarking on GPU:")
benchmark_parallel_states(data_size_mb=50, device="cuda")