# Changelog

All notable changes to this project will be documented in this file.

The format is based on "Keep a Changelog" and this project adheres to semantic versioning.

## [Unreleased]

- (Add bullet points for changes that will go into the next release)

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
