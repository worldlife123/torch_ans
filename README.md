# torch_ans

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![PyPI](https://img.shields.io/pypi/v/torch_ans?label=PyPI)
![License](https://img.shields.io/badge/license-MIT-green)

**torch_ans** is a high-performance PyTorch extension for ANS (Asymmetric Numeral Systems) entropy coding, supporting both C++ and CUDA backends across multiple platforms including Linux, macOS, and Windows. It enables fast, parallel rANS (range ANS) compression and decompression for deep learning and data compression workflows.
This extension is designed as an efficient, extensible replacement to [torchac](https://github.com/fab-jul/torchac) and [CompressAI](https://github.com/InterDigitalInc/CompressAI) range coding.

## Features

- **High-speed ANS compression/decompression**: Efficient encoding and decoding using rANS, suitable for large-scale data and neural network applications.
- **Parallel rANS on CPU and GPU**: Leverages PyTorch's CPU parallelism and CUDA acceleration/asynchronous operators for batch processing and high throughput.
- **Warp-level decoding on CUDA**: `rans32_16` with `num_interleaves=32` maps each stream onto the 32 lanes of a warp, stages the CDF and lookup tables in shared memory and picks the symbol with a tunable inverse-CDF lookup — achieving **up to 25.8 Gsymbol/s decode (~29 GB/s of payload) on an RTX 2080 Ti**.
- **Multi-platform support**: Compatible with Linux/macOS/Windows OS and x86_64/aarch64 based platforms, provided that PyTorch and the necessary build tools (C++ compiler, CUDA toolkit if applicable) are available on that platform.
- **Flexible rANS variants**: Supports multiple state sizes, stream sizes, frequency precisions, interleaved coding schemes and decoding acceleration tricks (alias coding, inverse cdf).
- **Low-level and high-level APIs**: Exposes both granular tensor-based operations and user-friendly interfaces for integration.
[CompressAI](https://github.com/InterDigitalInc/CompressAI) like API are also included, see [Cookbook](#cookbook).
- **Off-the-shelf for research**: Modular design using C++ templates, allowing for rapid prototyping and extension of new rANS variants without compromise on efficiency.

## Installation

`torch_ans` is distributed as a **source package**. The native extension is compiled **on your machine**, either at install time (`--no-build-isolation`, which needs a C++ toolchain) or lazily on first use (see [Runtime dynamic build](#runtime-dynamic-build)). Nothing is pinned to a particular PyTorch build: the extension is always compiled against the PyTorch you have installed, so there is no wheel-to-torch version to match. Pre-compiled wheels are deliberately not published - a wheel built against one torch version is not loadable by another.

It is therefore recommended to install a specific version of [PyTorch](https://pytorch.org/get-started/locally/) first, then install torch_ans with `--no-build-isolation`. This is important for PyTorch version compability!


### From PyPI

```bash
pip install torch_ans --no-build-isolation
```

This compiles the extension during installation, using the PyTorch that is already installed. Without `--no-build-isolation` the package still installs, but the extension is compiled later, on first use.

To build with CUDA support:

```bash
WITH_CUDA=1 pip install torch_ans --no-build-isolation
```

To build with ROCm/AMDGPU support when using a ROCm-enabled PyTorch installation (not tested yet):

```bash
WITH_HIP=1 pip install torch_ans --no-build-isolation
```

### From source
```bash
pip install . --no-build-isolation
```

To build with CUDA support:

```bash
WITH_CUDA=1 pip install . --no-build-isolation
```

To build with ROCm/AMDGPU support when using a ROCm-enabled PyTorch installation (not tested yet):

```bash
WITH_HIP=1 pip install . --no-build-isolation
```

### CUDA support

CUDA coding is **opt-in at build time**: `WITH_CUDA=1` is required, a visible GPU
is not enough (the same holds for both install paths above). The CUDA toolkit must
match the CUDA version your PyTorch build uses (`nvcc --version` vs
`torch.version.cuda`), and `nvcc` must be on `PATH`.

A CPU-only extension (the default) can still be used on a GPU machine, but only
with CPU tensors. If it is asked to code CUDA tensors it does not fail silently:

1. `torch_ans` emits a `RuntimeWarning` and compiles a CUDA extension for the
   local torch on the spot (this needs the matching `nvcc`, and takes a few
   minutes; the result is cached in `~/.cache/torch_extensions/`).
2. If no CUDA extension can be produced, it raises a `RuntimeError` listing the
   exact commands to fix it - install the toolkit, or rebuild from source with
   `WITH_CUDA=1 pip install . --no-build-isolation`, or set
   `TORCH_ANS_FORCE_RUNTIME_BUILD=1` to ignore a pre-built extension and always
   compile locally.

`TORCH_ANS_FORCE_RUNTIME_BUILD=1` is also useful on its own: it makes `torch_ans`
ignore an existing compiled extension (for example one built by an earlier
`pip install`) and compile for the local torch instead.

### Tested PyTorch versions

The CI matrix builds and runs the full test suite against **PyTorch 1.10.1 (Python 3.7), 2.1.1 (Python 3.9) and 2.7.1 (Python 3.11)** on Linux, macOS and Windows, plus an ARM64 Linux job and a lazy-compile job that installs without a compiler. Newer releases (e.g. 2.11) work for the same reason as any other: the extension is compiled locally against the torch that is installed.

## Usage

torch_ans supports a wide range of rANS variants, including different state, stream, and frequency sizes, interleaved coding, and parallel coding on CPU or GPU. 

Below we list currently supported variants:

| Variant                | State Bits | Stream Bits | Max Freq Bits | Interleaved States | Device Support | init_func                | push_func                | pop_func                |
|------------------------|------------|-------------|---------------|--------------------|---------------|--------------------------|--------------------------|-------------------------|
| rans64                 | 64         | 32          | 16            | 1                  | CPU, CUDA     | rans64_init_stream       | rans64_push              | rans64_pop              |
| rans64_i4              | 64         | 32          | 16            | 4                  | CPU           | -                        | rans64_i4_push           | rans64_i4_pop           |
| rans64_alias           | 64         | 32          | 16            | 1                  | CPU, CUDA     | -                        | rans64_alias_push        | rans64_alias_pop        |
| rans64_invcdf          | 64         | 32          | 16            | 1                  | CPU, CUDA     | -                        | rans64_push              | rans64_invcdf_pop       |
| rans64_i4_invcdf       | 64         | 32          | 16            | 4                  | CPU, CUDA     | -                        | rans64_i4_push           | rans64_i4_invcdf_pop    |
| rans32                 | 32         | 8           | 16             | 1                  | CPU, CUDA     | rans32_init_stream       | rans32_push              | rans32_pop              |
| rans32_i4              | 32         | 8           | 16             | 4                  | CPU           | -                        | rans32_i4_push           | rans32_i4_pop           |
| rans32_alias           | 32         | 8           | 16             | 1                  | CPU, CUDA     | -                        | rans32_alias_push        | rans32_alias_pop        |
| rans32_invcdf          | 32         | 8           | 16             | 1                  | CPU, CUDA     | -                        | rans32_push              | rans32_invcdf_pop       |
| rans32_16              | 32         | 16          | 15            | 1                  | CPU, CUDA     | rans32_16_init_stream    | rans32_16_push           | rans32_16_pop           |
| rans32_16_i4           | 32         | 16          | 15            | 4                  | CPU, CUDA     | -                        | rans32_16_i4_push        | rans32_16_i4_pop        |
| rans32_16_i32          | 32         | 16          | 15            | 32                 | CPU, CUDA     | -                        | rans32_16_i32_push       | rans32_16_i32_pop       |
| rans32_16_alias        | 32         | 16           | 15            | 1                  | CPU, CUDA     | -                        | rans32_16_alias_push     | rans32_16_alias_pop     |
| rans32_16_invcdf       | 32         | 16           | 15            | 1                  | CPU, CUDA     | -                        | rans32_16_push           | rans32_16_invcdf_pop    |

**Legend:**
- *State Bits*: Number of bits in the ANS state. This affects initial stream length, thereby impacting compression ratio when there are less symbols.
- *Stream Bits*: Number of bits per stream element. This affects the frequency of overflowed state to be written into/read from bitstream, slightly affecting speed.
- *Max Freq Bits*: Maximum supported frequency precision. This affects the accuracy of entropy estimation, thereby impacting compression ratio (higher is better). However, higher frequency precision also leads to larger memory occupation by CDF tables. In torch_ans implementation, State Bits > Stream Bits + Max Freq Bits.
- *Interleaved States*: Number of interleaved states for sequential coding in one step. On CPU this maps to SIMD-friendly sequential interleaving. On CUDA, `rans32_16_i4` uses 4-lane sub-warp groups and `rans32_16_i32` maps its 32 interleaved states onto the 32 lanes of a warp using warp-level primitives (`__ballot_sync`/`__shfl_xor_sync`, technique referenced from [Recoil](https://github.com/lin-toto/recoil)), with one shared bitstream cursor per warp. Interleaved streams are bit-compatible between the CPU and CUDA implementations (same layout and word order), so streams encoded on one device decode on the other.
- *Device Support*: Indicates if variant is available on CPU and/or CUDA GPU.
- *init_func/push_func/pop_func*: Main API functions for this variant.

In addition to standard and interleaved rANS, two advanced coding types are supported: alias coding and inverse CDF coding.

**Alias Coding**: Alias coding modifies both the push (encode) and pop (decode) steps. It accelerates the pop (decode) process by enabling constant-time symbol lookup, but increases memory usage during the push (encode) step due to the need for additional alias tables.

**Inverse CDF decoding**: Inverse CDF decoding only changes the pop (decode) step. Instead of the data-dependent divided search + binary refine, the decoder loads the symbol from a table keyed by the low bits of the ANS state. The table precision is tunable via `inverse_cdf_precision`:
- `inverse_cdf_precision = freq_precision` builds a *dense* table (one entry per quantized frequency, `2**freq_precision` entries per distribution — exact `O(1)` lookup);
- any smaller value builds a *sparse* table with only `2**q` entries per distribution plus a short bounded linear walk (`2**(freq_precision - q)` steps at most) — `2**(freq_precision - q)`x less memory, usually as fast or faster;
- `inverse_cdf_precision = "auto"` picks `q` from the alphabet size at `init_params` time (about `log2(alphabet) - 1`, tuned so the row fits the shared-memory staging budget on CUDA), and skips the table entirely for alphabets below 64 symbols where it buys less than ~3%.

With a sparse table the memory footprint is modest and the decode speedup is real: up to **+80% on CUDA** (large alphabets) and up to **2.8x on CPU** over the default lookup (measured, see `scripts/bench_sparse_invcdf.py`).

Use alias coding for fast decoding when memory usage during encoding is not a concern. Use inverse CDF coding (usually with `"auto"`) for maximum decoding speed when a small per-distribution table is acceptable.

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

On CUDA, throughput of the warp-level interleaved path (`rans32_16` + `num_interleaves=32`) is governed by **how many warps the batch provides**: each stream is decoded by one warp, so occupancy is `rows / (32 * SM count)`. On a 68-SM GPU, `rows >= 2048` saturates the GPU; below that, decode throughput scales almost linearly with `rows` (measured: 6.6 -> 25.8 Gsymbol/s going from 256 to 2048 rows, `inverse_cdf_precision="auto"`). See the [performance FAQ](#faq) for the full checklist.


### Command-line benchmark tool
After installing the package, you can run a built-in benchmark tool directly with Python:

```bash
python -m torch_ans.benchmark
```

This runs the rANS throughput benchmark across the specified batch sizes and devices. The benchmark supports `push`, `pop`, and `both` modes with `--mode`.

Example:
```bash
python -m torch_ans.benchmark -d cpu cuda --mode push
```

Example output on i7-6800k (6C12T) CPU and RTX 2080Ti GPU:
```
torch_ans benchmark
  impl: rans64
  mode: push
  data size: 50.0 MB
  num symbols: 256
  freq precision: 16
 
Benchmarking on cpu:
num threads: 6
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

Benchmarking on cuda:
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

  > **Range limit:** an out-of-range symbol is zig-zag encoded into a raw value
  > stored in the symbol dtype (`int32` by default), so its distance from the
  > coded range `[offset, offset + cdf_size - 2)` must stay below 2^(bits-2) —
  > about **2^30 (~1.07e9)** for `int32`. Symbols further out of range overflow
  > the raw value and are silently mis-coded.

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

# Fastest GPU decode path: warp-level 32-way interleaved coding + auto lookup
gpu_coder = TorchANSInterface(impl="rans32_16", freq_precision=15, device="cuda",
                              num_interleaves=32, inverse_cdf_precision="auto")
gpu_coder.init_params(freqs, num_freqs, offsets)
gpu_stream = gpu_coder.encode_with_indexes(
    symbols.to("cuda"), indexes.to("cuda"))
gpu_decoded = gpu_coder.decode_with_indexes(gpu_stream, indexes.to("cuda"))
assert torch.equal(gpu_decoded.cpu(), symbols)
```

Notes on the GPU example:
- `num_interleaves=32` selects the warp-level kernels (one warp per stream).
- `freq_precision=15` is the maximum supported by the warp-level path.
- `inverse_cdf_precision="auto"` builds a symbol-lookup table sized from the alphabet (and skips it for alphabets < 64 symbols).
- Batch at least ~2048 rows (`symbols.shape[0]`) to saturate the GPU; streams encoded on CPU decode on CUDA and vice versa.

## Performance Tuning

In order of impact (measured on an RTX 2080 Ti and a 6-core CPU):

### GPU

1. **Use `impl="rans32_16"` with `num_interleaves=32`** — this selects the warp-level interleaved CUDA kernels (one warp per stream). It requires `freq_precision <= 15`.
2. **Batch enough rows.** Each stream is decoded by one warp, so occupancy is `rows / (32 * SM count)`: use `rows >= 4 * SM count` (e.g. >= 2048 on a 68-SM GPU) to saturate the GPU. Below that, decode throughput scales almost linearly with `rows`, and no launch-configuration trick can recover it.
3. **Enable the inverse-CDF lookup for faster decoding** with `inverse_cdf_precision="auto"`: for alphabets with >= 64 symbols it builds a `2**q` table (`q ~= log2(alphabet) - 1`) sized so the whole row fits the shared-memory staging budget. Worth +5% to +80% over the default binary search depending on the alphabet.
4. **Keep `freq_precision <= 15`** — smaller states renormalize less often, and `rans32_16` is the only variant with the warp-level path.
5. **Avoid bypass coding when symbols are guaranteed in range** — the bypass phase is inherently sequential across the warp lanes (its per-symbol skeleton was worth 2.5-3.3x in our ablations).
6. `init_params` rebuilds the quantized CDFs and the lookup table and costs ~0.2 ms on CUDA. It runs on every call of the `dist_freqs` API, so for many small tensors prefer passing `dist_indexes` (precomputed CDF indexes), or reuse the coder.

On CUDA (RTX 2080 Ti, `rans32_16`, 1024 rows x 4096 symbols, alphabet 255, p=15, Gsymbol/s):

| lookup | 1-way enc/dec | 4-way enc/dec | 32-way enc/dec |
|---|---|---|---|
| divided search | 1.16 / 0.84 | 3.69 / 1.64 | 10.10 / 6.19 |
| inverse-CDF | 1.57 / **1.42** | 3.66 / 2.30 | 10.49 / **8.85** |
| alias sampling | 1.01 / 1.24 | 1.98 / 2.56 | n/a |

1. **Interleaving matters far more on CUDA than on CPU**: 4-way is ~3x and the 32-way warp path is ~7-9x over 1-way (each stream is coded by one warp with a shared cursor, which also coalesces the stream traffic).
2. **Decoding**: the inverse-CDF table wins at every interleave factor (1.7x over the divided search at 1-way with 256 distributions, 1.4x at 32-way). Alias sampling is also faster than the divided search (1.5x) but slower than the table, and it scales worst (~2x from 1-way to 4-way, against ~3x) because its bucket lookup is a division by a runtime value and it needs an extra dependent table load.
3. **Encoding**: alias sampling is the slowest by up to 1.9x - the push reads a remap table indexed by the whole `2**freq_precision` range, which is 33 MB for 256 distributions. There is also no 32-way alias variant.
4. So the recommendation is the same on both devices: `inverse_cdf_precision="auto"` (or a dense table) for decode, and no alias sampling except where its 1-way convenience is worth the throughput. Alias needs `cdfs_sizes - 1` to be a power of two, e.g. alphabet 255 with bypass coding (a cdf row of 257).

### CPU

1. `torch.set_num_threads(n)` controls the OpenMP parallelism used by push/pop.
2. `inverse_cdf_precision="auto"` also helps on CPU decoding — up to ~2.5x at `num_interleaves=1` and ~1.2-1.6x on top of the interleaved gain (see the next point); the table is built with a single sorted merge, so it is cheap.
3. **Interleaved variants boost CPU performance via OoO (Out-of-Order) execution** — with `num_interleaves=4`, the independent rANS states per stream let the out-of-order engine overlap their renormalize/push/pop chains. Measured on an i7-6800K (1M symbols, alphabet 256, 256 distributions, single thread, best-of-5, `scripts/bench_interleaves_cpu.py`):
   - **Encoding: ~1.25-1.4x** with `num_interleaves=4` (1.5x without bypass coding).
   - **Decoding: ~1.8-2.0x** with the default (divided-search) symbol lookup, ~1.4-1.8x with the inverse-CDF table — the table's *absolute* decode throughput is the highest of the two (about 1.2-1.6x better than the divided search at the same `num_interleaves`), it just has less left to gain from interleaving because a single state is already fast.
   - `num_interleaves=8` performs about the same as 4; `num_interleaves=32` is **not** useful on CPU (encode ~0.8x, register pressure) — it exists for the warp-level CUDA path. **Use 4 (or 8) on CPU.**
   - The inverse-CDF table (`inverse_cdf_precision="auto"` or an integer) is the fastest CPU decode configuration measured, at every alphabet size and every `num_interleaves`, *provided* the auto precision is used (a dense table is far worse: for 256 distributions it is tens of MB and thrashes the cache). This required removing the data-dependent linear walk that used to close the table's gap — see below.
   - The gains come from the fact that the coder is written to be overlap-friendly — see *Why interleaving helps* below.

   **Why interleaving helps (and what it needs).** Interleaved coding only pays off if the independent states can actually execute in parallel, and three things used to prevent that:
   - *Data-dependent branches.* Whether a renormalization is needed is close to a coin flip, so the renormalize `if` mispredicts on about every second symbol and flushes the out-of-order window — exactly the window the other lanes need. The coders therefore use branch-free renormalization (a speculative store on the encode side, an unconditional load on the decode side), a branch-free monotone search for the symbol lookup, and a branch-free way to close the inverse-CDF table's gap: the table gives a lower bound and the remaining couple of symbols are resolved by loading a small fixed window of cdf entries *in parallel* and counting how many are `<= cum_freq`, instead of walking one cdf entry at a time.
   - *Non-pipelined 64-bit division.* The integer divider is not pipelined, so N interleaved states merely queue on it. Encoding now uses a pipelined floating-point divide plus a rarely-taken exact integer fix-up.
   - *Aliasing and table lookups.* The states live in registers (locals accessed with a compile-time lane index) instead of in the stream tensor they are written to, and all per-lane table lookups (`index -> cdf row`, `symbol -> (start, freq)`) are gathered for the whole interleave group before any state is updated, so the cache misses and the symbol searches of different lanes overlap.

   Note the branch-free renormalization stores one word *past* the cursor speculatively and the launch configuration keeps the interleaved states inline in the stream buffer, so every buffer keeps one word of slack (see `rans_init_stream` / `rans_push`).

4. `rans64` remains the most robust default; `rans32_16` reduces the initial-state overhead for small payloads. For CPU decode throughput the best combination measured is **`inverse_cdf_precision="auto"` + `num_interleaves=4`** (or 8); for encode, `num_interleaves=4` (the table does not affect encode). See `scripts/bench_interleaves_cpu.py` to reproduce.
5. **Symbol lookup choice.** There are three decode lookups; on CPU the inverse-CDF table wins, and alias sampling is the slowest (it is opt-in via `alias_sampling=True`). Measured on an i7-6800K (1M symbols, 256 distributions, alphabet 255, p=15, single thread, `num_interleaves=4`, cyc/symbol):

   | lookup | encode | decode | tables |
   |---|---|---|---|
   | divided search (default) | 25.8 | 63.9 | cdf 257 KB |
   | inverse-CDF (`"auto"`) | 26.7 | **54.4** | 385 KB |
   | alias sampling | 101.6 | 75.8 | 1.3 MB + 32 MB remap |

   Alias sampling *does* benefit from interleaving, just less than the other two (decode 118.2 -> 75.8 cyc/symbol from 1 to 4 interleaved states, i.e. 1.56x, against 1.76x for the inverse-CDF table; encode 212.9 -> 101.6, 2.1x), for two structural reasons: its bucket lookup divides by a runtime value (a non-pipelined integer division - it is a shift here because the builder only accepts a cdf size of `2**k + 1`, but the shift has to be recovered per symbol), and its encode side reads a *remap* table indexed by the whole `2**freq_precision` range, which is 32 MB for 256 distributions and misses cache on every symbol. It also requires `cdfs_sizes - 1` to be a power of two (e.g. alphabet 255 with bypass coding), unlike the other lookups.


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

In constrast, interleaved states corresponds to a single bitstream, and is sequentially processed. See [Fabian's paper](https://ar5iv.labs.arxiv.org/html/1402.3392) for more information. In fact, SIMD ops (such as AVX2 and AVX512) could be used to accelerate interleaved ANS coding, but in our experiments 8 parallel states has better acceleration than the 8 interleaved states with AVX2 ops. 

**Q: How to choose parameters like State Bits, Stream Bits and Freq Bits? What is their relation to stream size and memory occupation?**

A: To be simple, always use rans64 configuration if data size is not extremely small (less than 1KB) and you are not sensitive about memory occupation!
If data size is extremely small, you could try rans32 or rans32_16 to reduce initial state.

**Q: Why isn't GPU throughput significantly higher than CPU? Some existing implementation like [dietgpu](https://github.com/facebookresearch/dietgpu) achieve over 200GB/s but torch_ans have only 6GB/s throughput!**

A: This was true for the old non-interleaved path (one thread per stream, divergent branches, `DEFAULT_NUM_THREADS_PER_BLOCK=1`), which still exists for `rans64`/`rans32`. The `rans32_16` variant now has a dedicated warp-level interleaved CUDA path (technique from [Recoil](https://github.com/lin-toto/recoil)/[dietgpu](https://github.com/facebookresearch/dietgpu)): each stream is decoded by one warp, the CDF and lookup tables are staged in shared memory, the symbol lookup uses a tunable inverse-CDF table, and the bypass phase is skipped by a group ballot when no lane needs it.

On an RTX 2080 Ti this reaches **~25.8 Gsymbol/s decode (~29 GB/s of payload at ~9 bits/symbol)** with `rows >= 2048`, which is the same order of magnitude as dietgpu's numbers. Two things to keep in mind when comparing:

- rANS decode throughput in symbols/s depends on the payload entropy: at ~9 bits/symbol, 25.8 Gsymbol/s already moves ~29 GB/s; benchmarks reporting 200 GB/s usually assume fixed 2-byte symbols on data-center GPUs (A100/H100) and no per-symbol distribution routing or bypass coding.
- torch_ans targets the learned-compression use case: multiple distributions per tensor, `int32` symbols with per-symbol CDF indexes, and bypass coding for out-of-range values.

### Technical

**Q: What Python and PyTorch versions are supported?**

A: Python >= 3.7 and PyTorch >= 1.10 are recommended. Earlier versions may work but are not tested.

**Q: What platforms are supported?**

A: torch_ans supports Linux (x86_64/aarch64), macOS (x86_64/aarch64), and Windows (x86_64) platforms. Ensure you have a compatible C++ compiler and optionally CUDA toolkit (for GPU support) installed on your system.

**Q: How do I enable CUDA support?**

A: Install the CUDA toolkit and ensure PyTorch is built with CUDA. Use `WITH_CUDA=1 pip install .` to enable CUDA build. Also check with `torch.cuda.is_available()`.

**Q: RuntimeError: The detected CUDA version (x.x) mismatches the version that was used to compile PyTorch (y.y). Please make sure to use the same CUDA versions.**

A: Ensure that your installed CUDA toolkit version align with PyTorch CUDA version. Or if you do not need CUDA for coding, remove `WITH_CUDA` environment variable to disable CUDA when building.

**Q: Why do I get compilation errors during installation?**

A: Make sure you have a C++17 compiler, pybind11, and compatible PyTorch. Try cleaning the build directory: `rm -rf build/ torch_ans.egg-info/` and reinstall.

**Q: ImportError: ...torch_ans/_C... Undefined Symbol xxxx.**

A: In most cases this is caused by version mismatch between PyTorch during runtime and building torch_ans. Reinstalling torch_ans would hopefully solve this.


**Q: How do I choose between CPU and GPU?**

A: Set the `device` argument in API calls to "cpu" or "cuda". For large batches, GPU is recommended; for small data, CPU may be faster.
Also, as GPU coding process is asynchronous, if some other tasks (such as neural networks in neural compression) are running meanwhile, using GPU coding may increase the overall throughput.
For the fastest GPU decode when you have massive data, use `impl="rans32_16"`, `num_interleaves=32`, `freq_precision <= 15`, `inverse_cdf_precision="auto"` and `rows >= 2048` (see the performance FAQ above).

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

- Requires Python >= 3.7, PyTorch, and a C++17 compatible compiler.
- CUDA toolkit required for GPU support.
- Run tests with:
  ```bash
  pytest -q
  ```

### Known Issues
- Rans64 cuda coding test fails on some newer GPU architectures.
- **No pre-compiled wheels are published**: installing requires a C++ toolchain (and `nvcc` for CUDA coding), and the extension is compiled against your local PyTorch. This is intentional - a wheel compiled against one torch build cannot be loaded by another, and shipping one would break the `import` on every other version.
- **CUDA coding is off unless the extension was built with `WITH_CUDA=1`**: a CPU-only extension on a GPU machine only codes CPU tensors. Asking it for CUDA tensors triggers the automatic local CUDA build described in [CUDA support](#cuda-support); if that is not possible you get the commands to install/fix it.
- **CUDA build fails with `std_function.h: parameter packs not expanded with '...'`**: `nvcc <= 12.1` cannot parse the `std::function` headers shipped with `libstdc++` from GCC >= 11.4 (e.g. Ubuntu 22.04.3+), so compiling CUDA extensions with the default `g++` fails. Fixes (any one of them):
  - Use an older host compiler for nvcc, e.g. `g++-10` (install with `apt install g++-10`). The runtime dynamic build detects this automatically: when a CUDA build fails, it retries with `g++-10`/`g++-9`/`g++-8` (`-ccbin`) before falling back to CPU-only, and remembers the working configuration in the torch extensions cache directory (`cuda_build_state`).
  - Use `nvcc >= 12.2`, which supports the newer libstdc++ headers.
  - Also make sure the CUDA toolkit version matches the one your PyTorch wheel was built with (e.g. `nvcc` 11.8 for `torch ...+cu118`); mismatched toolchains are unsupported.

### Testing and coverage

Install the development dependencies and run the Python unit tests:

```bash
pip install . --no-build-isolation
pip install pytest pytest-cov gcovr
pytest -q
```

Run Python coverage for the package:

```bash
pytest --cov=torch_ans --cov-report=term-missing --cov-report=html
```

For release-to-release throughput comparisons see `benchmark_status.md`: it is measured locally (CI runners are virtualised CPUs and too noisy for this) with

```bash
python scripts/bench_version_matrix.py --version-label <ver> --json <ver>.json
python scripts/generate_benchmark_report.py --baseline <old>.json \
    --candidate <new>.json --output benchmark_status.md
```

To collect native C/C++ coverage for the compiled extension, build with coverage instrumentation and run tests from the repository root:

```bash
ENABLE_COVERAGE=1 python setup.py build_ext --inplace
ENABLE_COVERAGE=1 pytest -q
```

Then generate a native coverage report with `gcovr`:

```bash
gcovr -r . --html-details -o native_coverage.html
```

This produces Python coverage output in `htmlcov/` and native C/C++ coverage output in `native_coverage.html`.

## Runtime dynamic build

If a compiled extension is already present (for example one built by `pip install . --no-build-isolation`),
it is used as is. Otherwise - e.g. when the package was installed without a compiler - `torch_ans` compiles
the native C++/CUDA sources at runtime using PyTorch's `cpp_extension`, on the first *use* of an operator.

An extension that cannot be loaded (built against another PyTorch, or with a different ABI) does not break
the import: `torch_ans` warns and falls back to compiling one for the local torch. Set
`TORCH_ANS_FORCE_RUNTIME_BUILD=1` to always take that path and ignore any pre-built extension.

- Trigger runtime build programmatically:

```py
import torch_ans
```

- Notes:
  - The build is triggered by the first use of a native operator: creating a `TorchANSInterface` (or `torch_ans._C.<operator>` / `from torch_ans._C import <operator>`, which compile the full extension). Importing `torch_ans.utils` alone no longer compiles anything. The built module is cached under the torch extensions directory (`~/.cache/torch_extensions/`) and reused by later processes.
  - **Incremental compilation (default)**: only the operators the interface needs are compiled. `TorchANSInterface(impl="rans64")` with default settings builds the `rans64` push/pop plus the shared helpers rather than all three rANS variants with every interleave factor and symbol lookup (about 4x less compiler work; measured 25s vs 100s for that configuration). Every configuration has its own cached module (`torch_ans_C_r64`, `torch_ans_C_r64_i4_invcdf`, ...), and any already loaded module that exports the requested operators is reused, so switching configurations does not duplicate work and a previously built full extension satisfies all of them. Pass `incremental_compile=False` or set `TORCH_ANS_INCREMENTAL_COMPILE=0` to compile every operator up front (this is also what a wheel built at install time always contains).
  - Runtime compilation requires a C/C++ toolchain and `ninja` (installed automatically as a dependency).
  - The runtime build tries CUDA first and falls back to CPU-only automatically when no usable CUDA runtime or toolchain is available. For CUDA builds you must have a compatible CUDA toolkit and driver installed (see Known Issues for nvcc/GCC compatibility).
  - CPU parallel coding relies on OpenMP: `at::parallel_for` in the native code is multi-threaded only when the extension is compiled with OpenMP enabled. The runtime build enables it automatically on Linux (GCC `-fopenmp`) and Windows (MSVC `/openmp`); on macOS it uses Homebrew libomp (`-Xpreprocessor -fopenmp -lomp`) when installed, and falls back to single-threaded coding otherwise.
  - If your runtime PyTorch ABI differs from the build-time one, `torch_ans` will warn by default. Set `TORCH_ANS_STRICT_CHECK=1` to re-enable a strict ImportError on mismatch.
  - A compiled `torch_ans/_C*.so` shadows `torch_ans/_C.py` completely, so the lazy implementation lives in `torch_ans/_lazy_C.py` (always importable). That is what the fallbacks switch to: an extension that fails to load, and a CPU-only extension that is asked to code CUDA tensors (see [CUDA support](#cuda-support)).
  - To keep CI/tests stable, the bundled test for dynamic build is guarded; enable it with `RUN_DYNAMIC_BUILD_TEST=1` when you want to run the rebuild test locally.



### TODO
- Docs and examples for high-level API
- ~~Fix interleave ANS state on CUDA, and possibly implement with [CUDA Warp-level Primitives](https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/)~~ Done for `rans32_16_i32` / `rans32_16_i4`: warp-level 32-way (and 4-way sub-warp) interleaved coding on CUDA, bit-compatible with the CPU interleaved streams (see `torch_ans/rans_warp_cuda.cuh`, technique referenced from [Recoil](https://github.com/lin-toto/recoil))
- ~~Implement lookup table logic in push/pop steps for possible acceleration~~ Done: inverse-CDF lookup with a tunable (dense/sparse/`"auto"`) `2**q` table on CPU and CUDA, warp-level shared-memory staging on CUDA, and a `"auto"` precision heuristic.
- Implement tANS and its variants with similar high-level API (refer to [FSAR](https://github.com/alipay/Finite_State_Autoregressive_Entropy_Coding))
- Test other backends supported in PyTorch (such as ROCm)
- Add more examples such as neural compression

# Related Projects

- [PyTorch](https://pytorch.org/): Deep learning framework used for tensor operations and GPU support.
- [pybind11](https://github.com/pybind/pybind11): Python bindings for C++ used in this extension.
- [CompressAI](https://github.com/InterDigitalInc/CompressAI): Neural compression library with entropy coding.
- [torchac](https://github.com/fab-jul/torchac): Fast arithmetic coding library for PyTorch, includes ANS variants.
- [ryg_rans](https://github.com/rygorous/ryg_rans): Reference C implementation of rANS.
- [fsc](https://github.com/skal65535/fsc): rANS implementation with alias mapping.
- [dietgpu](https://github.com/facebookresearch/dietgpu): An ultra-fast parallel rANS implementation (over 200GB/s) on NVIDIA GPUs.
- [Recoil](https://github.com/lin-toto/recoil): A header-only C++20 rANS library including parallel rANS algorithms.
- [FiniteStateEntropy](https://github.com/Cyan4973/FiniteStateEntropy): Reference C implementation of tANS/FSE by Yann Collet.
- [FSAR](https://github.com/alipay/Finite_State_Autoregressive_Entropy_Coding): Provides a numpy-based unified rANS/tANS interface implementation.

For more ANS implementations, see [Jarek Duda's blog on encode.su](https://encode.su/threads/2078-List-of-Asymmetric-Numeral-Systems-implementations)

## License

MIT License