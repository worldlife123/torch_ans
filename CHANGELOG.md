# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and this project adheres to semantic versioning.

## [0.4.0] - 2026-10-11

v0.4.0 completes the libtorch **stable ABI** migration: the native code compiles once against the stable ABI
(`TORCH_TARGET_VERSION` = torch 2.10), so one `.so` serves every `torch >= 2.10`; source installs and `torch < 2.10`
are unchanged and keep compiling per torch. Stream capacity becomes a Python-side contract instead of a per-call
computation, and tANS joins rANS as a coder family on CPU and CUDA - `scripts/bench_stream_capacity.py` measures what
the capacity contract buys.

### Added

- A prebuilt libtorch-stable-ABI wheel: `WITH_STABLE_ABI=1 pip install . --no-build-isolation` builds `torch_ans/_C_cpu.so` (all 103 schemas - every family and interleave variant, i.e. `_dynamic_build.profile_op_names(None)` minus the two `py::bytes` operators Python implements - plus the CPU and catch-all implementations), and adding `WITH_CUDA=1` also builds `torch_ans/_C_cuda.so`, which registers implementations only (a second schema declaration is rejected by the dispatcher). Neither library exports CPython or ATen/C++ symbols (only the stable C shim's `aoti_torch_*`/`torch_*`), so they are independent of the torch release and the Python version, and the wheel is tagged `py3-none-<platform>`.
- `torch_ans/_stable_C.py` finds and loads the prebuilt libraries with `torch.ops.load_library`, ahead of any compilation; every failure mode (old torch, missing or unloadable file, a CUDA request without a CUDA library) is a warning plus the normal runtime build instead of an error. `TORCH_ANS_NO_STABLE_ABI=1` and `TORCH_ANS_STABLE_LIB_DIR` control the lookup, and `incremental_compile=` has no effect on this path.
- `torch_ans/ta/`: the compatibility layer between the ATen build and the stable-ABI build (`torch_api.hpp` plus one header per backend and the POD `view.hpp`), so kernels and op wrappers exist once and both build modes are bit-comparable. Compile-checked in both modes and with nvcc by `scripts/check_ta_headers.py` / `tests/test_ta_headers.py`.
- `scripts/check_stable_abi_wheel.py` (wheel shape, `nm -D -u` symbol whitelist, `readelf -d` NEEDED check) and `scripts/bench_stream_capacity.py` (the per-push cost of the capacity contract, before/after); the measured numbers are in the capacity-contract entry below.
- Tests for the new path: `tests/test_stable_abi.py` (registration, CPU/CUDA bit equality, the Python `bytes` operators, the CUDA flavour incl. warp-32 and the sparse inverse-CDF table), `tests/test_capacity_contract.py` and `tests/test_cross_version_load.py` (one wheel, several torch versions, one digest).
- CI in `cibuildwheel.yml`: `build-stable-wheel` (build, retag to `py3-none-<platform>`, symbol whitelist, then install it and check that using it compiles nothing), `cross-version-load` (torch 2.10 / the build version / the newest release) and `cross-version-compare` (all digests must agree). A test asserts the matrix floor against `_stable_C.STABLE_ABI_MIN_TORCH`, so CI cannot drift below the ABI the `.so` targets.
- **The prebuilt wheel is published to PyPI** (`.github/workflows/release_wheels.yml`, on every `v*.*.*` tag): `cibuildwheel` builds the CPU flavour for `manylinux_2_28_x86_64`, `macosx_11_0_arm64`, `macosx_11_0_x86_64` and `win_amd64`, the wheels are retagged `py3-none-<platform>`, checked (`check_stable_abi_wheel.py` + `twine check`), installed and used on their own platform, and then uploaded next to the sdist `release_sdist.yml` publishes. `workflow_dispatch` with `publish=false` builds and verifies without uploading. The CUDA flavour is not published yet (TODO at the end of that workflow).
- `Requires-Dist` stays `torch>=1.10` on purpose: pip reads the dependencies of the one best-matching wheel of a version and a platform wheel beats `py3-none-any`, so declaring `torch>=2.10` would make pip **upgrade torch** for anyone on an older release. The floor is enforced at runtime instead.
- **tANS** (the table-driven, FSE-style ANS coder) as a new family: `tans32` (uint32 state / uint16 stream words, `freq_precision <= 15`), `tans16` (uint16 / uint8, `<= 15`) and `tans8` (uint8 / uint8, `<= 7`, the small-table tier), each with `*_init_stream` / `*_push` / `*_pop` plus the shared host-side table builder `tans_build_tables`. Selected with `impl="tans32" / "tans16" / "tans8"`, it drives the existing high-level API unchanged: `table_log` is `freq_precision`, so the quantized CDF is spread into the FSE tables once in `init_params`, and both directions share one table tensor carried in the `cdfs` slot.
- A tANS stream carries a header: a fixed-width tANS state cannot absorb the bits it does not emit, so those travel in one state word per interleave lane followed by the residual bit buffer and its bit count, all part of the serialized byte string. See the README's [tANS section](README.md#tans-table-driven-ans) for the memory trade-off (`3 * 2**freq_precision + 2 * cdf_row` int32 words per distribution).
- `tans_pop` writes the header back, so sequential pops walk the pushes back like `rans_pop`; the stream layout, the byte-string helpers and the capacity contract stay identical to rANS, and bypass coding, `offsets`, per-symbol distributions, batched states and multi-push streams all work. There is no `alias_sampling` / `inverse_cdf_precision` variant - the table *is* the symbol lookup.
- **Interleaved tANS** (`num_interleaves` 1/2/4/8/32): `tans{32,16,8}_i2/i4/i8/i32_push|pop`, selected exactly like the rANS ones, with one state per lane and the bit buffer shared by the lanes - the tANS counterpart of rANS' single stream cursor, which keeps a 1-lane stream byte-for-byte what it was. The lane of a symbol is its index % `num_interleaves`, the two directions walk the lanes in mirrored order, and the header is `num_interleaves + 2` words, so the initial length word is `4 * (num_interleaves + 2)` bytes (`_STREAM_SIZES` plus `_TANS_HEADER_EXTRA_BYTES` on the Python side).
- **tANS on CUDA** (`torch_ans/tans_cuda.cu`): the three families code device tensors through the same operator names, stream layout and capacity contract, with one thread per stream row at one lane (above one lane the launcher routes to the warp kernels below). The CUDA kernels are the CPU loop rebased onto the device, so CPU and CUDA produce byte-identical payloads for the same table at every interleave factor - `tests/test_tans.py` compares the byte strings, not just the decoded symbols, and cross-decodes both ways. `init_params` builds the table on the host and moves it to `self.device`.
- **Warp-level interleaved tANS on CUDA** (`torch_ans/tans_warp_cuda.cuh`): `_i2/i4/i8/i32_push|pop` code device tensors too - one lane group of `num_interleaves` lanes per stream row, each lane owning its state, the bit buffer and the payload cursor shared. The per-lane table work runs in parallel while the bit layer is committed in lane order with `__shfl_sync`, so the payload is bit-identical to `tans_cpu.cpp` at every factor, head/tail included.
- Interleave gain on the device (RTX 2080 Ti, 4M symbols, 64 distributions, table in global memory): it is largest where one thread per row cannot fill the device (8 stream rows) and shrinks to single-digit percent at 4096 rows, where the win is the memory-level parallelism and the coalesced `symbols`/`indexes` loads rather than extra warps.
- The 32-lane factor was the outlier in the first measurements: 32 sequential bit-layer appends per group and 229 registers/thread (one 256-thread block per SM). Reworking the bit layer - one packed `__shfl_sync` broadcast per lane, a rolled serial section, and a parallel bit read on decode - brings it to 42-44 registers, and 32 lanes is now the fastest factor at 8 and 512 stream rows (2.59x encode / 11.6x decode over one lane at 8 rows).
- The 136-byte header of a 32-lane row is what remains: ~1.1 bits/symbol at 1024-symbol rows (6.96 vs 5.99), which is why its encode is 0.94x of one lane at 4096 rows. Structurally a tANS row shares one bit buffer where every rANS lane renormalizes its own words (placed by `__ballot_sync`), so the `rans32_16` 32-way warp path still leads on encode at a tiny batch. Before/after tables: `benchmarks/tans_interleaved_cuda_optimized.json`, discussed in `benchmark_status.md`.
- **Shared-memory table staging for the tANS CUDA kernels**: a tANS lookup *is* its table and every symbol reads it at a state- or symbol-dependent offset, so the kernel stages the whole table into shared memory when it fits `TORCH_ANS_TANS_SMEM_BUDGET` (default 16 KiB per block, `0` disables; the same whole-table-or-nothing pattern as `rans_warp_cuda.cuh`'s A9). `TORCH_ANS_TANS_BLOCK_THREADS` overrides the block size, which also decides how often the per-block staging copy is paid.
- Single-lane tANS decodes 1.0-1.5x faster than single-lane rANS at `alphabet >= 64` while encoding within ±5% of it (0.85-1.05x), except at `alphabet 16` / `fp=15`; the small-table long-stream sweeps in `benchmarks/` give 1.15-1.33x, 2.3x. The two device interleaved kernels are compared head to head in `benchmark_status.md`.
- `torch_ans/stream_capacity.hpp`: the worst-case stream growth and the caller-owned-capacity compare/grow, extracted from `rans_push` so `tans_push` shares one implementation instead of adding a third copy of the arithmetic. `utils._stream_budget` and `tests/test_capacity_contract.py` are unchanged, with the tANS families added to that test's matrix.
- Compile gates `TORCH_ANS_WITH_TANS32` / `_TANS16` / `_TANS8` (one per family, like the rANS ones) so an incremental build compiles only the families an interface asks for. `tans_build_tables` is deliberately not gated - all three families share it - and is part of `COMMON_OPS`; the push/pop instantiation macros now take the interleave factor and are gated by the shared `TORCH_ANS_WITH_INTERLEAVE_2/4/8/32`.
- `scripts/bench_tans_vs_rans.py`: the tANS vs rANS comparison (family × `num_interleaves` × `freq_precision` × alphabet × bypass), reporting encode/decode throughput next to bits/symbol and the per-distribution table size - tANS "faster" must never be read without its compression and memory cost. Unsupported configurations are skipped with the reason printed and every measured configuration is round-trip checked; `--device cuda` adds `--smem-budget` / `--block-threads` / `--warp-block-threads` and treats `--parallel-states` as a sweep. `scripts/bench_interleaves_cpu.py` covers tANS too, raw results live in `benchmarks/`, and the full discussion is in `benchmark_status.md`.

### Changed

- Stream capacity is now a contract between Python and the native push: `TorchANSInterface` reserves it (`utils._stream_budget`, pinned against the wrapper's arithmetic by `tests/test_capacity_contract.py`) and passes it as the new trailing `used_bytes` argument, so the wrapper only compares and raises instead of reading the stream's length word back and resizing. That removes a blocking device->host copy from every CUDA push (157 -> 39 us per small push) and keeps `aten::resize_` - not promised by the stable ABI - out of the hot path. Omitting `used_bytes` (the default, -1) keeps the old behaviour bit for bit.
- The native operator set, the pybind11 bindings and the stable schemas/implementations are now declared side by side in `ans_bindings.hpp`, with `lib.cpp` (CPU flavour: schemas + catch-all + CPU implementations) and the new `lib_cuda.cpp` (CUDA flavour: implementations only) as the two entry points. `pybind11` is no longer a build or runtime dependency: the ordinary build takes the headers from torch itself and the stable build needs none.
- `tests/test_warp_cuda.py::test_high_level_rans32_16_impl_resolution` compares the resolved encoder against the module the interface actually resolved, instead of against `torch_ans._C` (a different module whenever a compiled extension and a prebuilt one coexist).
- `torch < 2.10` no longer skips the prebuilt library silently: `_stable_C._load` emits one `RuntimeWarning` (upgrade torch, or `TORCH_ANS_NO_STABLE_ABI=1`) and then falls back to the runtime build. Only an *installed wheel* warns - a source install has no library to skip, so it stays quiet. This is what makes publishing one wheel per platform safe for old-torch users.
- The stable-ABI build is toolchain-portable: `setup.py`'s `_stable_abi_platform_flags` spells "optimise" and "do not depend on one torch build" per platform (Linux `-O3` + `-Wl,--as-needed`, macOS `-O3` + `-Wl,-rpath,@loader_path/../torch/lib`, Windows `/O2` and no link flag), and `_stable_abi_drop_libraries` removes the `-ltorch_python` that `cpp_extension` links unconditionally - ld64 has no `--as-needed` and link.exe has no such concept, so without this the macOS link aborts and the Windows wheel is tied to one CPython *and* one torch build. Linux flags are unchanged.
- `scripts/check_stable_abi_wheel.py` infers the platform from the wheel tag (`--platform` overrides) and uses that toolchain's tools: `otool -L` + `nm -u` on macOS (plus a check that no install name is an absolute build-machine path), `dumpbin /dependents` on Windows, `nm -D -u` + `readelf -d` on Linux as before. Missing tools are announced and skipped instead of silently passing.

### Fixed

- CUDA kernels now launch on torch's current stream (`ta::current_stream`) instead of stream 0, which was a race with any non-default `torch.cuda.Stream()` and made coding invisible to stream ordering, and a launch error is checked (`cudaGetLastError`) instead of surfacing on the next unrelated CUDA call.
- macOS runtime builds no longer abort the process on the first parallel operation: the OpenMP variant they get from Homebrew libomp links and imports fine and then dies with SIGABRT inside `rans_pmf_to_quantized_cdf` (torch 2.14 only), which turned the full-matrix macOS arm64 jobs red.
- As that crash is invisible to the build, the runtime build now runs the operator once in a child process (a build cache hit plus one call) before accepting the variant, and otherwise falls back to the flags without a second OpenMP runtime - the ones `setup.py` has always used on macOS, with `at::parallel_for` on torch's own thread pool. The outcome is remembered in an `openmp_build_state` file in the torch extensions cache directory; delete it to try OpenMP again.
- `STD_TORCH_CHECK` now forwards its arguments verbatim (`TORCH_CHECK(__VA_ARGS__)`) instead of appending the variadic part to a spelled-out condition. The common single-argument form expanded to `TORCH_CHECK(cond, )`, whose empty trailing argument only compiles on the newer torch releases; on torch 2.7 it failed with "expected primary-expression before ')' token" from `c10/util/Exception.h`, so every build against that release died before reaching a kernel. Behaviour is unchanged wherever the macro already compiled.
- `scripts/bench_tans_vs_rans.py` now documents which `torch_ans` it measures: run as `python scripts/bench_tans_vs_rans.py`, `scripts/` becomes `sys.path[0]`, so the script imports the *installed* package and an older install silently shadows the working tree (the first CUDA run here died with `NotImplementedError: Unknown impl tans32` for exactly that reason). Measuring the sources needs `PYTHONPATH=$PWD`.

### Removed

- `ta::safe_stream_byte_length` / `ta::ensure_stream_capacity`: an unused second copy of the stream-growth arithmetic (only the header syntax check instantiated it) that would have had to stay in step with the native wrapper and the Python budget by hand.

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
