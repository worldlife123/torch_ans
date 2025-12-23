<!-- # Badges

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyPI](https://img.shields.io/pypi/v/torch_ans?label=PyPI)
![License](https://img.shields.io/badge/license-TODO-blue) -->

# torch_ans

**torch_ans** is a high-performance PyTorch extension for ANS (Asymmetric Numeral Systems) entropy coding, supporting both C++ and CUDA backends. It enables fast, parallel rANS (range ANS) compression and decompression for deep learning and data compression workflows.
This extension is designed as an efficient, extensible replacement to [torchac](https://github.com/fab-jul/torchac) and [CompressAI](https://github.com/InterDigitalInc/CompressAI) range coding.

## Features

- **High-speed ANS compression/decompression**: Efficient encoding and decoding using rANS, suitable for large-scale data and neural network applications.
- **Parallel rANS on CPU and GPU**: Leverages PyTorch's CPU parallelism and CUDA acceleration for batch processing and high throughput.
- **Flexible rANS variants**: Supports multiple state sizes, stream sizes, frequency precisions, interleaved coding schemes and decoding acceleration tricks (alias coding, inverse cdf).
- **Low-level and high-level APIs**: Exposes both granular tensor-based operations and user-friendly interfaces for integration.
- **Off-the-shelf for research**: Modular design using C++ templates, allowing for rapid prototyping and extension of new rANS variants without compromise on efficiency.

## Installation


### From PyPI
It is recommended to install a specific version of [PyTorch](https://pytorch.org/get-started/locally/) first, then install torch_ans with `--no-build-isolation`
```bash
pip install torch_ans --no-build-isolation
```

### From source
```bash
pip install .
```

To force CUDA build (if not auto-detected):

```bash
FORCE_CUDA=1 pip install .
```

Note that this process may install the latest CPU-only PyTorch version. If you would like to use a CUDA-enabled PyTorch version, or you have already install a specific PyTorch version according to the [official website](https://pytorch.org/get-started/locally/), install without isolated build system. This is important for PyTorch version compability!
```bash
pip install pybind11
pip install . --no-build-isolation
```

## Usage

torch_ans supports a wide range of rANS variants, including different state, stream, and frequency sizes, interleaved coding, and parallel coding on CPU or GPU. 

Below we list currently supported variants:

| Variant                | State Bits | Stream Bits | Max Freq Bits | Interleaved States | Device Support | init_func                | push_func                | pop_func                |
|------------------------|------------|-------------|---------------|--------------------|---------------|--------------------------|--------------------------|-------------------------|
| rans64                 | 64         | 32          | 16            | 1                  | CPU, CUDA     | rans64_init_stream       | rans64_push              | rans64_pop              |
| rans64_i4              | 64         | 32          | 16            | 4                  | CPU           | -                        | rans64_i4_push           | rans64_i4_pop           |
| rans64_alias           | 64         | 32          | 16            | 1                  | CPU, CUDA     | -                        | rans64_alias_push        | rans64_alias_pop        |
| rans64_invcdf          | 64         | 32          | 16            | 1                  | CPU, CUDA     | -                        | rans64_push              | rans64_invcdf_pop       |
| rans32                 | 32         | 8           | 16             | 1                  | CPU, CUDA     | rans32_init_stream       | rans32_push              | rans32_pop              |
| rans32_i4              | 32         | 8           | 16             | 4                  | CPU           | -                        | rans32_i4_push           | rans32_i4_pop           |
| rans32_alias           | 32         | 8           | 16             | 1                  | CPU, CUDA     | -                        | rans32_alias_push        | rans32_alias_pop        |
| rans32_invcdf          | 32         | 8           | 16             | 1                  | CPU, CUDA     | -                        | rans32_push              | rans32_invcdf_pop       |
| rans32_16              | 32         | 16          | 15            | 1                  | CPU, CUDA     | rans32_16_init_stream    | rans32_16_push           | rans32_16_pop           |
| rans32_16_i4           | 32         | 16          | 15            | 4                  | CPU           | -                        | rans32_16_i4_push        | rans32_16_i4_pop        |
| rans32_16_alias        | 32         | 8           | 15            | 1                  | CPU, CUDA     | -                        | rans32_16_alias_push     | rans32_16_alias_pop     |
| rans32_16_invcdf       | 32         | 8           | 15            | 1                  | CPU, CUDA     | -                        | rans32_16_push           | rans32_16_invcdf_pop    |

**Legend:**
- *State Bits*: Number of bits in the ANS state. This affects initial stream length, thereby impacting compression ratio when there are less symbols.
- *Stream Bits*: Number of bits per stream element. This affects the frequency of overflowed state to be written into/read from bitstream, slightly affecting speed.
- *Max Freq Bits*: Maximum supported frequency precision. This affects the accuracy of entropy estimation, thereby impacting compression ratio (higher is better). However, higher frequency precision also leads to larger memory occupation by CDF tables. In torch_ans implementation, State Bits > Stream Bits + Max Freq Bits.
- *Interleaved States*: Number of interleaved states for sequential coding in one step. Effective when combined with SIMD instructions (not implemented in torch_ans).
- *Device Support*: Indicates if variant is available on CPU and/or CUDA GPU.
- *init_func/push_func/pop_func*: Main API functions for this variant.

In addition to standard and interleaved rANS, two advanced coding types are supported: alias coding and inverse CDF coding.

**Alias Coding**: Alias coding modifies both the push (encode) and pop (decode) steps. It accelerates the pop (decode) process by enabling constant-time symbol lookup, but increases memory usage during the push (encode) step due to the need for additional alias tables.

**Inverse CDF decoding**: Inverse CDF decoding only changes the pop (decode) step. It accelerates decoding by allowing direct symbol lookup from the state, but requires much more memory during the pop step because of large inverse CDF tables. However, it seems that reading large inverse CDF tables does not result in prominent acceleration compared to standard coding.

Use alias coding for fast decoding when memory usage during encoding is not a concern. Use inverse CDF coding for maximum decoding speed when memory usage during decoding is acceptable.

### Parallel ANS stream

All encoding (push) and decoding (pop) operations in torch_ans are fully parallelized. The batch size of the rANS stream tensor (`stream.shape[0]`) determines the number of parallel states processed simultaneously. This enables efficient utilization of multi-core CPUs and GPUs for large-scale data compression.

- The stream tensor is typically `int32` (defined in DEFAULT_TORCH_TENSOR_TYPE and DEFAULT_TORCH_TENSOR_DTYPE in `rans_utils.hpp`) and has shape `(B, L)`, where `B` is the batch size (number of parallel states) and `L` is the stream length. Note that due to int32 range, each stream length could not exceed 2GB.
- Initial stream length `L` is computed as `1 + ceil(num_interleaves * sizeof(RANS_STATE_TYPE) / 4)`. The first element stores the stream length, and the remaining elements store the ANS states. Therefore using 32 state bits could reduce overhead than 64 state bits, especially when a large number of parallel stream is used on small amount of data.
- When calling `rans*_push` or `rans*_pop`, symbols are distributed along the batch dimension and processed in parallel across all states. 
- On CPU you can set the number of threads for parallel coding with `torch.set_num_threads()`.

### Speed/Compression Considerations

Increasing the number of parallel states (`B`) generally improves throughput, but may slightly reduce compression ratio due to larger initial states being stored in the bitstream. The optimal number of parallel states depends on your hardware and data size:

- For small datasets (<100MB), CPUs with fewer parallel states (e.g., 8 for desktop, 32 for server) are usually optimal.
- For large datasets, GPUs become advantageous only with a large number of parallel states (typically >256).
- Example: On an i7-6800k (6C12T) CPU and RTX 2080Ti GPU, rans64 encoding speed is similar for CPU and GPU with 128 parallel states, while with 256 parallel states GPU is 2 times faster than CPU.

#### Benchmark Example: Find Your Optimal Parallel States

Use the following script to benchmark encoding speed for different numbers of parallel states on your machine:

```python
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
```

Example output on i7-6800k (6C12T) CPU and RTX 2080Ti GPU:
```
Benchmarking on CPU:
Batch size:    1 | Time: 0.1711 s | Throughput: 292.26 MB/s
Batch size:    2 | Time: 0.1121 s | Throughput: 446.11 MB/s
Batch size:    4 | Time: 0.0794 s | Throughput: 629.38 MB/s
Batch size:    8 | Time: 0.0722 s | Throughput: 692.59 MB/s
Batch size:   16 | Time: 0.0809 s | Throughput: 618.17 MB/s
Batch size:   32 | Time: 0.0678 s | Throughput: 737.36 MB/s
Batch size:   64 | Time: 0.0664 s | Throughput: 753.02 MB/s
Batch size:  128 | Time: 0.0627 s | Throughput: 797.97 MB/s
Batch size:  256 | Time: 0.0534 s | Throughput: 937.06 MB/s
Batch size:  512 | Time: 0.0501 s | Throughput: 998.68 MB/s
Batch size: 1024 | Time: 0.0473 s | Throughput: 1058.06 MB/s
Batch size: 2048 | Time: 0.0479 s | Throughput: 1044.59 MB/s
Batch size: 4096 | Time: 0.0500 s | Throughput: 999.69 MB/s
Benchmarking on GPU:
Batch size:    1 | Time: 6.5701 s | Throughput: 7.61 MB/s
Batch size:    2 | Time: 4.8248 s | Throughput: 10.36 MB/s
Batch size:    4 | Time: 1.4710 s | Throughput: 33.99 MB/s
Batch size:    8 | Time: 0.7357 s | Throughput: 67.96 MB/s
Batch size:   16 | Time: 0.3679 s | Throughput: 135.90 MB/s
Batch size:   32 | Time: 0.1841 s | Throughput: 271.66 MB/s
Batch size:   64 | Time: 0.0922 s | Throughput: 542.46 MB/s
Batch size:  128 | Time: 0.0463 s | Throughput: 1081.05 MB/s
Batch size:  256 | Time: 0.0235 s | Throughput: 2128.31 MB/s
Batch size:  512 | Time: 0.0128 s | Throughput: 3898.49 MB/s
Batch size: 1024 | Time: 0.0081 s | Throughput: 6149.65 MB/s
Batch size: 2048 | Time: 0.0081 s | Throughput: 6161.39 MB/s
Batch size: 4096 | Time: 0.0080 s | Throughput: 6222.82 MB/s
```

This will help you determine the best batch size (number of parallel states) for your hardware and data size. For most users, start with CPU and increase batch size until speed plateaus or memory usage becomes excessive.


### Low-level API Reference

All low-level functions operate directly on PyTorch tensors for maximum performance and flexibility. Below are the main entry points:

**torch_ans.rans*_init_stream**

```python
torch_ans.rans*_init_stream(size: int, num_interleaves: int, preallocate_size: int) -> torch.Tensor
```
Initializes a parallel rANS stream tensor.
- **size** (int): Number of parallel states (batch size).
- **num_interleaves** (int): Number of interleaved states per stream.
- **preallocate_size** (int): Preallocated stream size (bytes). Optional; improves performance for large batches.
- **Returns**: `torch.Tensor` of shape `(size, L)`.

**torch_ans.rans*_push**

```python
torch_ans.rans*_push(
  stream: torch.Tensor,
  symbols: torch.Tensor,
  indexes: torch.Tensor,
  cdfs: torch.Tensor,
  cdfs_sizes: torch.Tensor,
  offsets: torch.Tensor,
  freq_precision: int,
  bypass_coding: bool,
  bypass_precision: int
) -> None
```
Encodes symbols into the rANS stream in parallel.
- **stream** (`torch.Tensor`): rANS stream tensor.
- **symbols** (`torch.Tensor`): Symbols to encode (int tensor).
- **indexes** (`torch.Tensor`): CDF table index for each symbol.
- **cdfs** (`torch.Tensor`): Batched CDF tables.
- **cdfs_sizes** (`torch.Tensor`): Size of each CDF table.
- **offsets** (`torch.Tensor`): Offset for each distribution.
- **freq_precision** (int): Frequency precision in bits.
- **bypass_coding** (bool): Enable bypass coding for out-of-range symbols.
- **bypass_precision** (int): Precision for bypass coding.

**torch_ans.rans*_pop**

```python
torch_ans.rans*_pop(
  stream: torch.Tensor,
  indexes: torch.Tensor,
  cdfs: torch.Tensor,
  cdfs_sizes: torch.Tensor,
  offsets: torch.Tensor,
  freq_precision: int,
  bypass_coding: bool,
  bypass_precision: int
) -> torch.Tensor
```
Decodes symbols from the rANS stream in parallel.
- **stream** (`torch.Tensor`): rANS stream tensor.
- **indexes** (`torch.Tensor`): CDF table index for each symbol.
- **cdfs** (`torch.Tensor`): Batched CDF tables.
- **cdfs_sizes** (`torch.Tensor`): Size of each CDF table.
- **offsets** (`torch.Tensor`): Offset for each distribution.
- **freq_precision** (int): Frequency precision in bits.
- **bypass_coding** (bool): Enable bypass coding for out-of-range symbols.
- **bypass_precision** (int): Precision for bypass coding.
- **Returns**: Decoded symbols as `torch.Tensor`.

**torch_ans.rans_stream_to_byte_strings**

```python
torch_ans.rans_stream_to_byte_strings(stream: torch.Tensor) -> List[bytes]
```
Converts a stream tensor to a list of Python `bytes` objects for serialization or storage.

**torch_ans.rans_byte_strings_to_stream**

```python
torch_ans.rans_byte_strings_to_stream(byte_strings: List[bytes]) -> torch.Tensor
```
Reconstructs a stream tensor from a list of Python `bytes` objects.

See `torch_ans_test.py` for more advanced usage and validation.


### High-level API Reference

The high-level API provides a user-friendly interface for rANS coding similar to CompressAI, abstracting away tensor management and stream details. The main entry point is `TorchANSInterface` in `torch_ans.utils`, which supports encoding and decoding with various rANS implementations.

#### TorchANSInterface

```python
from torch_ans.utils import TorchANSInterface
```

**Initialization:**

```python
coder = TorchANSInterface(impl="rans64", freq_precision=16, device="cpu")
```
- **impl** (str): rANS variant (e.g., "rans64", "rans32", "rans32_16", etc.)
- **freq_precision** (int): Frequency precision in bits.
- **device** (str): "cpu" or "cuda".
- ... (More parameters in [torch_ans/utils.py](torch_ans/utils.py))

**Parameter Setup:**

```python
coder.init_params(freqs, num_freqs, offsets)
```
- **freqs** (`torch.Tensor`): Batched distribution probability mass functions (PMFs), shape `(K, S)`. `K` is number of distributions/tables, `S` is maximum size of symbol alphabet.
- **num_freqs** (`torch.Tensor`): Size of symbol alphabet per distribution, shape `(K,)`.
- **offsets** (`torch.Tensor`): Offset for each alphabet, shape `(K,)`.

**Encoding:**

```python
encoded = coder.encode_with_indexes(symbols, indexes)
```
- **symbols** (`torch.Tensor`): Symbols to encode, shape `(B, N)`. `B` is the same with ANS state batch size (equal to number of parallel states), `N` is number of symbols.
- **indexes** (`torch.Tensor`): Distribution/table index for each symbol, should have same shape with **symbols**.
- **Returns**: Encoded stream (typically a list of `bytes` or tensor).

**Decoding:**

```python
decoded = coder.decode_with_indexes(encoded, indexes)
```
- **encoded**: Encoded stream from `encode_with_indexes`.
- **indexes** (`torch.Tensor`): Distribution/table index for each symbol, shape `(B, N)`. `B` is the same with ANS state batch size (equal to number of parallel states), `N` is number of symbols.
- **Returns**: Decoded symbols as `torch.Tensor`.

**Other Features:**
- Supports both CPU and CUDA parallel coding transparently.
- Handles CDF/PMF conversion internally during init_params.
- Provides methods for serialization and deserialization of streams.

See `torch_ans_test.py` for advanced usage, batch processing, and custom coding schemes.



### Full API Reference Table

| Function/Class                | Description                                                      |
|-------------------------------|------------------------------------------------------------------|
| rans*_init_stream             | Initialize rANS stream tensor                                    |
| rans*_push            | Encode symbols into rANS stream                                  |
| rans*_pop             | Decode symbols from rANS stream                                  |
| rans_stream_to_byte_strings   | Convert stream tensor to list of bytes                           |
| rans_byte_strings_to_stream   | Convert list of bytes to stream tensor                           |
| TorchANSInterface             | High-level API for encoding/decoding with rANS variants          |
| TorchANSInterface.init_params | Set PMF, alphabet size, and offsets for coder                   |
| TorchANSInterface.encode_with_indexes | Encode symbols using high-level API                    |
| TorchANSInterface.decode_with_indexes | Decode symbols using high-level API                    |



## Cookbook

A high-level usage example with TorchANSInterface:
```python
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

assert torch.equal(decoded, symbols)
```

## Troubleshooting

- Ensure PyTorch and pybind11 are installed and compatible with your Python version.
- For CUDA support, verify CUDA toolkit is installed and `torch.cuda.is_available()` returns `True`.
- If you encounter compilation errors, try cleaning the build directory and reinstalling:
  ```bash
  rm -rf build/ torch_ans.egg-info/
  pip install .
  ```


## FAQ

### Algorithmic

**Q: What is the difference between parallel and interleaved ANS state?**

A: Parallel states corresponds to individual bitstreams, so they require extra space to store bitstream lengths. Despite from this drawback, parallel states are more efficient (especially with a large number of ANS states), easy to implement (device specific code is hardly required) and robust to bit error (corruption in one bitstream will not propagate to another).

In constrast, interleaved states corresponds to a single bitstream, and is sequentially processed. In fact, SIMD ops (such as AVX2 and AVX512) could be used to accelerate interleaved ANS coding, but in our experiments 8 parallel states has better acceleration than the 8 interleaved states with AVX2 ops. 

**Q: How to choose parameters like State Bits, Stream Bits and Freq Bits? What is their relation to stream size and memory occupation?**

To be simple, always use rans64 configuration if data size is not extremely small (less than 1KB) and you are not sensitive about memory occupation!
If data size is extremely small, you could try rans32 or rans32_16 to reduce initial state.


### Technical

**Q: What Python and PyTorch versions are supported?**

A: Python >= 3.8 and PyTorch >= 1.12 are recommended. Earlier versions may work but are not tested.

**Q: How do I enable CUDA support?**

A: Install the CUDA toolkit and ensure PyTorch is built with CUDA. Use `FORCE_CUDA=1 pip install .` to force CUDA build if not auto-detected. Check with `torch.cuda.is_available()`.

**Q: Why do I get compilation errors during installation?**

A: Make sure you have a C++17 compiler, pybind11, and compatible PyTorch. Try cleaning the build directory: `rm -rf build/ torch_ans.egg-info/` and reinstall.

**Q: How do I choose between CPU and GPU?**

A: Set the `device` argument in API calls to "cpu" or "cuda". For large batches, GPU is recommended; for small data, CPU may be faster.

**Q: What is the difference between low-level and high-level APIs?**

A: Low-level APIs operate directly on tensors for maximum control, but improper use may lead to segmentation fault. High-level APIs (TorchANSInterface) is designed with a similar API to CompressAI's entropy coder, which simplify usage and enables auto paralellization, ANS stream management and serialization for you.

**Q: How do I serialize and deserialize streams in torch.Tensor format?**

A: Use `torch_ans.rans_stream_to_byte_strings` and `torch_ans.rans_byte_strings_to_stream` for conversion between tensors and Python bytes.

**Q: How do I run tests?**

A: Run `python tests/torch_ans_test.py` or use `pytest` in the project directory.

**Q: Where can I find more usage examples?**

A: See the README Usage section and `torch_ans_test.py` for practical examples and test cases. For practical usage of ANS coding in neural compression, see [CompressAI](https://github.com/InterDigitalInc/CompressAI).

**Q: What can I do if segmentation fault occurs?**

This library is developed for research-purpose only, and not as a robust everyday software, so errors may occur occasionally. If you are using the low-level API, try to switch to the high-level API and enable bypass coding, which might be more robust. If error still persists, try checking middle variables such as cdf (should be incremental positive integers and less than 2^freq_precision).

## Development

- Requires Python >= 3.8, PyTorch, and a C++17 compiler.
- CUDA toolkit required for GPU support.
- Run tests with:
  ```bash
  python tests/torch_ans_test.py
  ```

### TODO
- Fix interleave ANS state on CUDA
- Implement per-symbol coding (encode/decode_with_freqs) and direct symbol coding with pre-defined distributions ((encode/decode_symbols)) in high-level API
- Implement tANS and its variants (similar to [FSAR](https://github.com/alipay/Finite_State_Autoregressive_Entropy_Coding))
- Support for other backends supported in PyTorch (such as ROCM)
- Add more examples such as neural compression

# Related Projects

- [PyTorch](https://pytorch.org/): Deep learning framework used for tensor operations and GPU support.
- [pybind11](https://github.com/pybind/pybind11): Python bindings for C++ used in this extension.
- [CompressAI](https://github.com/InterDigitalInc/CompressAI): Neural compression library with entropy coding.
- [torchac](https://github.com/fab-jul/torchac): Fast arithmetic coding library for PyTorch, includes ANS variants.
- [ryg_rans](https://github.com/rygorous/ryg_rans): Reference C implementation of rANS.
- [fsc](https://github.com/skal65535/fsc): rANS implementation with alias mapping.
- [Recoil](https://github.com/lin-toto/recoil): A header-only C++20 rANS library including parallel rANS algorithms.
- [FiniteStateEntropy](https://github.com/Cyan4973/FiniteStateEntropy): Reference C implementation of tANS/FSE by Yann Collet.
- [FSAR](https://github.com/alipay/Finite_State_Autoregressive_Entropy_Coding): Provides a numpy-based unified rANS/tANS interface implementation.

For more ANS implementations, see [Jarek Duda's blog on encode.su](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations)

## License

MIT License