"""Dynamic runtime loader for the native torch_ans C++/CUDA extension.

Provides `build_extension` which compiles the native extension at runtime
using `torch.utils.cpp_extension.load` when the pre-built `torch_ans._C`
module is missing (lazy compile mode).

This is intentionally small and conservative: it mirrors enough of
`setup.py`'s decisions so runtime compilation behaves similarly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

def _detect_cuda_torch():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _cuda_state_path(module_name: str) -> Optional[str]:
    """Path of the CUDA build state marker inside the extension build cache.

    Records the outcome of previous CUDA build attempts so later first-time
    imports in fresh processes do not re-pay doomed ones:
      - "cpu"    -> the CUDA build failed with every available configuration;
                    build CPU-only directly.
      - "ccbin:<path>" -> the CUDA build only works with the host compiler
                    at <path> (nvcc <= 12.1 cannot parse the std::function
                    headers of gcc >= 11.4); start attempts with it.
    Returns None if torch's build directory cannot be determined (no state).
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory
        return os.path.join(_get_build_directory(module_name, verbose=False), "cuda_build_state")
    except Exception:
        return None


def _older_host_compilers():
    """Host compilers known to work around nvcc/libstdc++ header incompatibilities."""
    import shutil
    compilers = []
    for candidate in ("g++-10", "g++-9", "g++-8"):
        path = shutil.which(candidate)
        if path:
            compilers.append(path)
    return compilers


def _reset_jit_versioner(module_name: str) -> None:
    """Make the next torch JIT build reuse the base module name.

    torch's versioner appends a `_vN` suffix when the same module name is
    built again with different arguments (e.g. a build retry with different
    flags), which would leave previously built artifacts under a name that no
    longer matches later imports. Best effort: falls back to torch's default
    bump behavior on torch versions without this private API.
    """
    try:
        import torch.utils.cpp_extension as _cpp_ext
        versioner = getattr(_cpp_ext, "JIT_EXTENSION_VERSIONER", None)
        if versioner is None:
            versioner = getattr(_cpp_ext, "_JIT_EXTENSION_VERSIONER", None)
        if versioner is not None:
            versioner.entries.pop(module_name, None)
    except Exception:
        pass


def build_extension(module_name: str = "torch_ans_C_ext", with_cuda: Optional[bool] = None, verbose: bool = False):
    """Build the native extension under `module_name` and return the module.

    This function purposely does not try to import `torch_ans._C` first to avoid
    recursion when used from a shim module. If `with_cuda` is truthy, CUDA
    support is used when torch reports a working CUDA runtime, otherwise a
    CPU-only extension is built.

    NOTE: `module_name` must not contain dots. torch's JIT import machinery
    derives the required `PyInit_*` symbol from the module name, and pybind11
    cannot paste dotted names into that symbol. Undotted names additionally
    stay importable when the build is repeated with changed arguments.

    CUDA availability is detected BEFORE importing torch.utils.cpp_extension:
    that module-level import performs the process's first CUDA call, and doing
    it with hidden devices (CUDA_VISIBLE_DEVICES="") would initialize the CUDA
    runtime with zero visible devices, permanently disabling CUDA for the
    whole process even after the env var is restored.
    """
    # Detect CUDA availability before any CUDA call is made for us.
    if with_cuda:
        with_cuda = _detect_cuda_torch()
        if verbose:
            print(f"CUDA runtime available: {with_cuda}")

    try:
        from torch.utils.cpp_extension import load as torch_ext_load
    except Exception as e:
        raise RuntimeError("torch.utils.cpp_extension.load is required for runtime compilation: " + str(e))

    pkg_dir = Path(__file__).resolve().parent
    cpp_sources = [str(p) for p in pkg_dir.glob("*.cpp")]
    cu_sources = [str(p) for p in pkg_dir.glob("*.cu")]
    if len(cpp_sources) == 0:
        raise RuntimeError("No C++ sources found in package for runtime compilation")

    extra_cflags = ["-std=c++17", "-O3", "-fopenmp"]

    # setup.py's CUDAExtension defines WITH_CUDA for install-time builds; the
    # JIT loader does not, so the CUDA dispatch branches in rans.hpp would be
    # compiled out and CUDA tensors would hit "not compiled with GPU support".
    if with_cuda:
        extra_cflags = extra_cflags + ["-DWITH_CUDA"]

    def _load(sources, extra_cuda_cflags):
        _reset_jit_versioner(module_name)
        try:
            return torch_ext_load(
                name=module_name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                verbose=verbose,
            )
        except TypeError:
            # some torch versions have different signature; try without extra_cuda_cflags
            return torch_ext_load(
                name=module_name,
                sources=sources,
                extra_cflags=extra_cflags,
                verbose=verbose,
            )

    if not with_cuda:
        module = _load(cpp_sources, [])
        # Record build metadata for the runtime torch-version check in __init__.
        try:
            import torch
            build_ver_path = pkg_dir / "_torch_build_version.py"
            with open(build_ver_path, "w") as f:
                f.write(f"BUILD_TORCH_VERSION = {repr(torch.__version__)}\n")
                f.write(f"BUILD_WITH_CUDA = {repr(False)}\n")
                f.write(f"BUILD_WITH_HIP = {repr(False)}\n")
        except Exception:
            pass
        return module

    # CUDA build orchestration with failure memory (see _cuda_state_path).
    state_path = _cuda_state_path(module_name)
    state = None
    if state_path is not None and os.path.exists(state_path):
        try:
            with open(state_path) as f:
                state = f.read().strip()
        except OSError:
            state = None
    if state == "cpu":
        if verbose:
            print(
                "Previous CUDA build attempts all failed; building a CPU-only extension. "
                "Remove the cuda_build_state file in the torch extensions cache directory "
                "after fixing the CUDA toolchain to retry."
            )
        return _load(cpp_sources, [])

    # Try the remembered-working configuration first (if any), then the plain
    # build, then older host compilers that work around nvcc <= 12.1 vs
    # gcc >= 11.4 libstdc++ header incompatibilities ("parameter packs not
    # expanded with '...'" in std_function.h).
    plain = None
    ccbin_candidates = _older_host_compilers()
    if state and state.startswith("ccbin:"):
        preferred = state[len("ccbin:"):]
        if preferred in ccbin_candidates:
            ccbin_candidates.remove(preferred)
        attempt_order = [preferred, plain]
    else:
        attempt_order = [plain]
    attempt_order += ccbin_candidates

    extra_cuda_cflags = ["-O3", "-std=c++17"]
    last_err = None
    for cc in attempt_order:
        flags = extra_cuda_cflags if cc is None else extra_cuda_cflags + ["-ccbin", cc]
        try:
            module = _load(cpp_sources + cu_sources, flags)
        except Exception as e:
            last_err = e
            continue
        if state_path is not None:
            try:
                if cc is None:
                    os.remove(state_path)  # plain build works: clear any stale state
                else:
                    with open(state_path, "w") as f:
                        f.write(f"ccbin:{cc}\n")
            except OSError:
                pass
        if cc is not None and verbose:
            print(f"CUDA build succeeded with host compiler {cc}")

        # Record build metadata for the runtime torch-version check in __init__.
        try:
            import torch
            build_ver_path = pkg_dir / "_torch_build_version.py"
            with open(build_ver_path, "w") as f:
                f.write(f"BUILD_TORCH_VERSION = {repr(torch.__version__)}\n")
                f.write(f"BUILD_WITH_CUDA = {repr(True)}\n")
                f.write(f"BUILD_WITH_HIP = {repr(False)}\n")
        except Exception:
            pass
        return module

    # Every CUDA configuration failed: remember to build CPU-only next time.
    if state_path is not None:
        try:
            with open(state_path, "w") as f:
                f.write("cpu\n")
        except OSError:
            pass
    raise last_err
