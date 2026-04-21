<!-- .github/copilot-instructions.md: Guidance for AI coding agents working on this repo -->
# Copilot instructions — torch_ans

Goal: help an AI coding agent be immediately productive in this PyTorch C++/CUDA extension.

- **Big picture**: This repo implements a PyTorch extension (C++/CUDA + pybind11) providing parallel rANS (range ANS) entropy coding. Low-level, high-performance primitives live in the native extension (C++ files in `torch_ans/`), and a thin Python layer (`torch_ans/utils.py`) exposes both a low-level API (rans*_init_stream / rans*_push / rans*_pop) and a convenience interface `TorchANSInterface`.

- **Key components & files**:
  - `torch_ans/rans_cpu.cpp`, `torch_ans/rans.cpp`, `torch_ans/rans.hpp`, `torch_ans/rans_utils.hpp` — core native implementations and helpers.
  - `torch_ans/utils.py` — high-level Python interface, PMF/CDF helpers, caching, and `TorchANSInterface` used by tests and examples.
  - `setup.py`, `pyproject.toml` — extension build config. `setup.py` writes the build-time PyTorch version to `torch_ans/_torch_build_version.py`.
  - `tests/torch_ans_test.py`, `tests/test_high_level_api.py` — concrete examples and test coverage for both low- and high-level APIs.

- **Important runtime/data conventions** (explicit, checked in code):
  - The rANS *stream tensor* stores its current stream length in `stream[b][0]`; subsequent elements hold ANS states and stream data. See `rans_init_stream` in [torch_ans/rans_cpu.cpp](torch_ans/rans_cpu.cpp).
  - Default tensor dtype for stream and CDF tables is `torch::kInt32` (see `DEFAULT_TORCH_TENSOR_DTYPE` in [torch_ans/rans_utils.hpp](torch_ans/rans_utils.hpp)).
  - Low-level exported names follow `rans<state>_init_stream`, `rans<state>_push_indexed`, `rans<state>_pop_indexed` (e.g., `rans64_push_indexed`). Tests call these from Python via `torch_ans._C`.

- **Build / local dev workflow** (practical rules discovered from source):
  - Normal build: `pip install .` (this triggers C++/CUDA extension build via setuptools / torch.utils.cpp_extension).
  - Avoid PEP517 isolation pulling an unwanted `torch`: when building locally against an existing environment, use `pip install . --no-build-isolation` (recommended in README). To force a CUDA build set `FORCE_CUDA=1` in the environment.
  - When diagnosing build-time PyTorch mismatch, check `torch_ans/_torch_build_version.py` generated at build time.

- **Testing & quick checks**:
  - Run unit tests with `pytest -q tests` (the tests exercise both `TorchANSInterface` and low-level calls).
  - Example benchmarks are in `examples/benchmark.py` and useful for reproducing performance/parallel-state behaviour described in README.

- **Common code patterns & guidance for edits**:
  - CPU/CUDA code paths are separated by compile-time macros and by `setup.py` selecting `CppExtension` vs `CUDAExtension`. When adding GPU kernels, follow existing naming conventions (`rans_*_cuda.cu`) and the `WITH_CUDA` macro.
  - Many functions are templated by state type / interleaves. Prefer adding new variants by instantiating template wrappers (see `TORCH_LIBRARY_IMPL` at the bottom of `rans_cpu.cpp`).
  - PMF → quantized CDF conversion has CPU and Python implementations: prefer using the native `rans_pmf_to_quantized_cdf` when present, but tests reveal a Python fallback (`pmf_to_quantized_cdf_batched`) is used for experimentation.

- **Integration points**:
  - Python → native boundary: `torch_ans/_C` exposes functions used by `utils.py`. When changing signatures of native functions, update the import and call sites in `torch_ans/utils.py` and tests.
  - Serialization: `rans_stream_to_byte_strings` and `rans_byte_strings_to_stream` convert streams to/from Python `bytes` — if changing stream layout adjust these converters and tests.

- **Examples to follow (copyable patterns)**
  - Low-level push (from Python): call `stream = torch_ans.rans64_init_stream(batch_size)` then `torch_ans.rans64_push(stream, symbols, indexes, cdfs, cdfs_sizes, offsets, freq_precision, bypass_coding=False)`.
  - High-level: use `TorchANSInterface(impl="rans64", freq_precision=16)` and `encode_with_indexes(symbols, indexes)` / `decode_with_indexes(encoded, indexes)` — see `torch_ans/utils.py` and `tests/test_high_level_api.py`.

- **What NOT to change lightly**:
  - Stream tensor layout and `DEFAULT_TORCH_TENSOR_DTYPE` — many C++ accessors and casts assume these exactly.
  - Function signatures exported to `torch_ans._C` without updating the Python wrapper and tests.

If any detail here is unclear or you'd like the file to include more concrete code snippets (e.g., exact call-site examples with shapes/dtypes), tell me which area to expand and I'll iterate.
