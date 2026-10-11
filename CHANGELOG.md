# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and this project adheres to semantic versioning.

## [0.4.0] - 2026-10-11

v0.4.0 completes the libtorch **stable ABI** migration: the native code compiles once against the stable ABI
(torch 2.10), so a single prebuilt wheel serves every `torch >= 2.10`; source installs and `torch < 2.10` are
unchanged. Stream capacity is now a Python-side contract instead of a per-call computation, and tANS joins rANS
as a coder family on CPU and CUDA.

### Added

- **Prebuilt libtorch-stable-ABI wheel**, now published to PyPI on every `v*.*.*` tag for Linux, macOS (arm64/x86_64) and Windows. One `py3-none-<platform>` wheel covers all `torch >= 2.10` and any Python version, so no compilation happens on install or first use; a CUDA flavour can be built locally with `WITH_CUDA=1`. `Requires-Dist` stays `torch>=1.10` on purpose, so pip never force-upgrades torch for users on an older release - the floor is enforced at runtime instead.
  - `torch_ans/_stable_C.py` finds and loads the prebuilt library ahead of any compilation; every failure mode (old torch, missing or unloadable file, a CUDA request without a CUDA library) falls back to the normal runtime build instead of erroring. `TORCH_ANS_NO_STABLE_ABI` / `TORCH_ANS_STABLE_LIB_DIR` control the lookup.
  - `scripts/check_stable_abi_wheel.py` verifies the wheel shape, its undefined-symbol whitelist and its shared-library dependencies; missing platform tools are announced instead of silently skipped.
  - New CI jobs build the wheel, retag it, check it, load it under several torch versions and compare the digests.
- `torch_ans/ta/`: a compatibility layer so kernels and bindings exist once for both the ATen and the stable-ABI build, keeping the two modes bit-comparable.
- **tANS** (the table-driven, FSE-style ANS coder) as a new family, selected with `impl="tans32" / "tans16" / "tans8"` and driven through the existing high-level API unchanged. It trades table memory for throughput: `freq_precision` is the table log, and one table is built once in `init_params` and shared by both directions. There is no `alias_sampling` / `inverse_cdf_precision` variant - the table *is* the symbol lookup.
  - The stream layout, byte-string helpers and capacity contract are identical to rANS; bypass coding, offsets, per-symbol distributions, batched states and multi-push streams all work. See the README's [tANS section](README.md#tans-table-driven-ans) for the compression/memory trade-off.
  - **Interleaved variants** at `num_interleaves` 1/2/4/8/32, selected like the rANS ones, with one state per lane and a shared bit buffer.
  - **tANS on CUDA**, byte-identical to the CPU kernels at every interleave factor, including a warp-level interleaved path (`_i2/i4/i8/i32_*`) where a lane group per stream row shares the bit buffer. The 32-lane factor is the fastest at small and medium batches; the kernels support shared-memory table staging and a runtime block-size override.
  - Compile gates `TORCH_ANS_WITH_TANS32` / `_TANS16` / `_TANS8` keep incremental builds to the families an interface actually uses.
- Stream capacity is now a contract between Python and the native push: the interface reserves the worst-case capacity and passes it in, so the wrapper only compares and raises - removing a blocking device->host copy from every CUDA push and keeping `aten::resize_` out of the hot path. Omitting the new argument keeps the old behaviour bit for bit.
- Benchmarking scripts `scripts/bench_stream_capacity.py` (the per-push cost of the capacity contract) and `scripts/bench_tans_vs_rans.py` (tANS vs rANS across family, `num_interleaves`, `freq_precision`, alphabet and bypass, reporting bits/symbol and per-distribution table size next to throughput); `scripts/bench_interleaves_cpu.py` covers tANS too. Raw results and the discussion live in `benchmarks/` and `benchmark_status.md`.
- Tests for the new paths: `tests/test_stable_abi.py`, `tests/test_capacity_contract.py` (tANS families added), `tests/test_cross_version_load.py` and `tests/test_ta_headers.py`.

### Changed

- The native operator set and bindings are now declared side by side in `ans_bindings.hpp`, with `lib.cpp` (CPU) and `lib_cuda.cpp` (CUDA) as the two entry points. `pybind11` is no longer a build or runtime dependency: the ordinary build takes the headers from torch itself and the stable build needs none.
- `torch < 2.10` with an installed wheel emits one warning and falls back to the runtime build instead of skipping the prebuilt library silently - which is what makes publishing one wheel per platform safe for old-torch users.
- The stable-ABI build is toolchain-portable across Linux, macOS and Windows, and `scripts/check_stable_abi_wheel.py` verifies each platform with that platform's own tools.

### Fixed

- CUDA kernels now launch on torch's current stream instead of stream 0, removing a race with any non-default `torch.cuda.Stream()`, and a launch error is checked instead of surfacing on the next unrelated CUDA call.
- macOS runtime builds no longer abort on the first parallel operation: the crashing OpenMP variant is detected by running the operator once in a child process, and the build falls back to the flags without a second OpenMP runtime. The outcome is remembered in the extensions cache.
- `STD_TORCH_CHECK` forwards its arguments verbatim, so builds against torch 2.7 no longer die with a template error before reaching a kernel.
- `scripts/bench_tans_vs_rans.py` documents that it measures the *installed* package (`PYTHONPATH=$PWD` to measure the working tree instead).

### Removed

- An unused second copy of the stream-growth arithmetic from the `ta::` layer, which would otherwise have had to stay in step with the native wrapper and the Python budget by hand.

## [0.3.0] - 2026-09-14

v0.3.0 includes a major performance overhaul for both CPU and GPU.

### Added

- Warp-level 32-way interleaved rANS coding on CUDA for `rans32_16` (`rans32_16_i32_*`), one warp sharing a single bitstream cursor, bit-compatible with the CPU interleaved layout. Also fixed the broken 4-way CUDA interleaved path (routed to 4-lane sub-warp kernels; other variants raise a clear error), exposed `num_interleaves` in `TorchANSInterface`, and fixed `impl="rans32_16"` being shadowed by the `"rans32"` prefix match.
- Runtime-build fallbacks so a compiled extension is never a dead end: an extension that cannot be imported (compiled against another torch) falls back to compiling one for the local torch instead of failing the import, and a CPU-only extension that is asked to code CUDA tensors warns and compiles a CUDA extension locally - or raises with the exact commands to fix the toolchain instead of a bare "not compiled with GPU support" from C++. The new `TORCH_ANS_FORCE_RUNTIME_BUILD=1` ignores any pre-built extension and always compiles for the local torch.
- A sparse inverse-CDF symbol lookup for decode (`inverse_cdf_precision` int or `"auto"`): `2**q`-entry table plus a bounded walk, staged into shared memory on the CUDA 32-way warp path (decode ~6 -> ~26 Gsymbol/s on RTX 2080 Ti). Also fuses `pmf -> quantized cdf` and `cdf ++ inverse table` into single-pass implementations (2-4x faster `init_params`), adds the missing `rans64_i4_invcdf_pop` binding.
- Interleaved variants of the alias-sampling coder (`*_alias_i2/i4/i8_*`, new `alias_sampling=True` option), with batched symbol lookup; `rans_alias_build_table` now fails loudly on invalid input.
- `scripts/bench_version_matrix.py` (raw-operator matrix: device x family x `num_interleaves` x symbol lookup, plus the `init_params` build steps and the high-level API; driven through the low-level ops so operators that exist in only one release are reported as `n/a` instead of failing, and every configuration is round-trip checked before it is timed) and `scripts/generate_benchmark_report.py`, which renders `benchmark_status.md` from two of its JSON outputs. `benchmark_status.md` now holds a measured v0.2.1 -> v0.3.0 comparison (i7-6800K / RTX 2080 Ti): CPU decode 1.27-1.44x at the default configuration and 2.40-2.75x at `num_interleaves=4`, CUDA 1-way unchanged, new CUDA 32-way warp path at ~26 Gsymbol/s, `pmf -> cdf` 6-11x faster and inverse-CDF table build 60-600x faster.

### Changed
- Lazy compile is upgraded to incremental (per-interface) JIT build: `TorchANSInterface` now compiles only the operators its configuration needs, nothing before first use (new `incremental_compile=True` argument, `TORCH_ANS_INCREMENTAL_COMPILE=0` to disable). Bindings are gated by `TORCH_ANS_WITH_*` macros in `ans_build_config.hpp` / `ans_bindings.hpp`, with matching explicit instantiations in `rans_cpu.cpp`/`rans_cuda.cu`.
- CPU interleaved coding is now much faster: ~1.25-1.4x encode / ~1.8-2.1x decode at `num_interleaves=4` (previously a slight slowdown). Achieved via branch-free renormalization, pipelined division (`rans_divmod`), and register-resident interleaved states (see `rans_cpu.cpp` / `rans_utils.hpp`).
- New default symbol lookup is branch-free (monotone predicate search), removing binary-search mispredicts in non-interleaved decoding.
- Inverse-CDF table gap closing is branch-free (`rans_invcdf_advance`, fixed `RANS_INVCDF_WINDOW`=2 parallel loads), making the table the fastest CPU decode lookup at every alphabet size. `num_interleaves=32` remains CUDA-only.
- Package metadata now comes from `pyproject.toml` (PEP 621, `setuptools>=61`), including the project URLs. `setup.py` only mirrors it when the installed setuptools is older than 61, which ignores `[project]` entirely - that is the `pip install . --no-build-isolation` case, where dropping the mirror would silently produce an `UNKNOWN` distribution with no dependencies. `tests/test_packaging_metadata.py` keeps the two in sync.
- The extension compiled at install time is described by a compiled-in `_torch_ans_with_cuda` flag (see `lib.cpp`) rather than by generated metadata, so the Python layer can tell whether the loaded build can code CUDA tensors.

### Fixed

- CUDA build failed to import (`undefined symbol`) for `*_i2_*`/`*_i8_*` and interleaved alias/inverse-CDF combinations due to missing CUDA instantiations; all bound combinations now have one, unsupported ones raise a clear error (interleaved CUDA coding is rans32_16-only).
- The sdist could miss the native headers it needs to build (`ans_bindings.hpp`, `ans_build_config.hpp`): setuptools does not recompute the manifest when new source files appear, and the file list was never declared. `MANIFEST.in` now lists the sources explicitly and excludes generated or stale artifacts (`torch_ans/_torch_build_version.py`, `*.so`) so a leftover local `_C*.so` can never shadow `torch_ans/_C.py` in a distribution.
- Importing the Python API stays lazy when no compiled extension is present. The import machinery probes `__path__` on the module it is about to import names from (`from ._lazy_C import ...`), and the shim answered that probe with a full build, so a plain `import torch_ans.utils` compiled the whole extension - CUDA included - before any use. Dunder probes now raise `AttributeError` without building, while `from torch_ans._C import <op>` still compiles the full extension as documented.

### Removed

- The CI-generated benchmark table: the full-matrix workflow no longer runs `pytest-benchmark`, no longer uploads benchmark artifacts and no longer commits `benchmark_status.md` (it now only commits `build_status.md`); `tests/test_benchmarks.py`, `scripts/generate_benchmark_status.py` and the `pytest-benchmark` dev dependency are gone. The runners are virtualised CPUs whose throughput varies by tens of percent between runs, so the table could only ever show virtualisation noise rather than a regression. `benchmark_status.md` is measured locally on real hardware and committed instead.


## [0.2.1.post1] - 2026-09-07

- Packaging fixes surfaced by the v0.2.1 CI run: supply `-std=c++17` only for torch 1.x (torch >= 2.x selects its own C++ standard, up to C++20 for newer releases, and a user-supplied `-std=` would override it), silence Xcode 16's `-Winvalid-specialization` error for torch 2.7 headers on macOS, and skip `brew update` in CI.
- Fix runtime build on machines where every CUDA configuration fails: the `cuda_build_state="cpu"` shortcut no longer compiles the CPU-only extension with `-DWITH_CUDA` (it produced a `.so` with undefined symbols that failed to load and forced a second, pure-CPU rebuild on every fresh import — ~76s per process). Also pass `-DWITH_CUDA` to nvcc explicitly (torch only forwards `extra_cflags` to the C++ compiler and, unlike `WITH_HIP`, does not define it itself), so a successful CUDA toolchain no longer goes undetected.

## [0.2.1] - 2026-09-07

- (Beta) Add lazy compile mode during installation: Users can now directly use `pip install torch_ans` without `--no-build-isolation` to skip building when install. In this case the building will occur during first-time import. A dedicated `lazy-compile-test` CI job covers this flow on Linux/Windows/macOS, and the runtime build works around the nvcc <= 12.1 vs GCC >= 11.4 libstdc++ header incompatibility (see README Known Issues).
- Runtime dynamic build: platform-appropriate compile flags (MSVC / Apple clang / GCC), OpenMP kept enabled so `at::parallel_for` stays multi-threaded for CPU coding (with Homebrew libomp support on macOS), and build results cached in the torch extensions directory.
- Add index range check in C++/CUDA rans_push/pop code, improving its stability and reduce "Segmentation fault" issues.
- Fix cached encode in high-level API: adding CompressAI params to encode queue, and properly reset cache after flush.


## [0.2.0] - 2026-05-11

- Reworked high-level API, adding a unified encode/decode API supporting different coding patterns, keeping CompressAI-like encode/decode_with_indexes API for compability.

## [0.1.3] - 2026-04-26

- Added ROCm extension support (not tested)
- Change default setup compiler to only compile CPU code. To enable CUDA or ROCm support, you should manually set envvar WITH_CUDA=1 or WITH_HIP=1 during setup.
- Added a simple CLI benchmark tool torch_ans.benchmark

## [0.1.2] - 2026-04-24

- Updated high-level API to support CompressAI-like coding API.
- Widen dependency of PyTorch>=1.10 and Python>=3.7
- Added arm64 support by temporarily removing x86intrin headers.
- Added multi-platform support.

## [0.1.1] - 2025-12-20

- Bumped package version to 0.1.1.
- Minor packaging metadata updates.

## [0.1.0] - (initial)

- Initial release (C++/CUDA PyTorch extension for rANS compression).
