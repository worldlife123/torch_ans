"""Dynamic runtime loader for the native torch_ans C++/CUDA extension.

Provides `build_extension` which compiles the native extension at runtime
using `torch.utils.cpp_extension.load` when the pre-built `torch_ans._C`
module is missing (lazy compile mode).

This is intentionally small and conservative: it mirrors enough of
`setup.py`'s decisions so runtime compilation behaves similarly.
"""
from __future__ import annotations

import os
import platform
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


def _darwin_cpu_flag_variants():
    """CPU compile flag variants for macOS, fastest (OpenMP) first.

    `rans_cpu.cpp` parallelizes batch coding with `at::parallel_for`, which is
    multi-threaded only when the extension is compiled with OpenMP enabled
    (`_OPENMP` selects ATen's AT_PARALLEL_OPENMP backend; ~3-4x throughput on
    many-core machines). Apple clang has no built-in OpenMP, so when Homebrew
    libomp is available use the standard `-Xpreprocessor -fopenmp` + `-lomp`
    recipe; otherwise (or if the OpenMP attempt fails) fall back to flags
    that still build, at the cost of a serial at::parallel_for.
    """
    base = ["-O3", "-mmacosx-version-min=10.14"]
    variants = []
    try:
        import subprocess
        out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, timeout=10)
        prefix = out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        prefix = ""
    if prefix and os.path.exists(os.path.join(prefix, "include", "omp.h")):
        variants.append((
            base + ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"],
        ))
    variants.append((base, []))
    return variants


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


def _patch_file_baton_stale_lock(stale_seconds: int = 600) -> None:
    """Make torch's FileBaton treat ancient lock files as stale.

    A build killed mid-way (CI timeout, OOM, Ctrl-C) leaves the baton lock
    file behind, and every later import would spin on ``baton.wait()``
    forever (it has no timeout). With this patch, acquiring fails only for
    *fresh* locks; stale ones are removed and the acquire retried, so
    imports never deadlock. Best effort: silently skipped on torch versions
    where the patch cannot be applied.
    """
    try:
        import time
        from torch.utils import file_baton as _fb

        if getattr(_fb.FileBaton, "_torch_ans_stale_patch", False):
            return
        original_try_acquire = _fb.FileBaton.try_acquire

        def try_acquire(self):
            ok = original_try_acquire(self)
            if not ok:
                try:
                    age = time.time() - os.path.getmtime(self.lock_file_path)
                    if age > stale_seconds:
                        os.remove(self.lock_file_path)
                        ok = original_try_acquire(self)
                except OSError:
                    pass
            return ok

        _fb.FileBaton.try_acquire = try_acquire
        _fb.FileBaton._torch_ans_stale_patch = True
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

    # A previous build killed mid-way leaves a stale baton lock behind, which
    # would make every later import spin on FileBaton.wait() forever.
    _patch_file_baton_stale_lock()

    pkg_dir = Path(__file__).resolve().parent
    cpp_sources = [str(p) for p in pkg_dir.glob("*.cpp")]
    cu_sources = [str(p) for p in pkg_dir.glob("*.cu")]
    if len(cpp_sources) == 0:
        raise RuntimeError("No C++ sources found in package for runtime compilation")

    # Per-platform CPU compile flags, as (cflags, ldflags) variants tried in
    # order. NOTE: no `-std=` flag is passed here on purpose — torch's
    # cpp_extension always appends the C++ standard matching its own headers
    # (c++14/17/20 depending on the torch version), and a user-supplied
    # `-std=` comes later on the command line and would override it, breaking
    # builds against newer torch releases that require C++20.
    if sys.platform == "win32":
        cpu_flag_variants = [(["/O2", "/openmp"], [])]
    elif sys.platform == "darwin":
        cpu_flag_variants = _darwin_cpu_flag_variants()
    else:
        cflags = ["-O3", "-fopenmp"]
        if platform.machine() == "x86_64":
            cflags.append("-march=native")
        cpu_flag_variants = [(cflags, [])]

    # setup.py's CUDAExtension defines WITH_CUDA for install-time builds; the
    # JIT loader does not, so the CUDA dispatch branches in rans.hpp would be
    # compiled out and CUDA tensors would hit "not compiled with GPU support".
    if with_cuda:
        cpu_flag_variants = [(c + ["-DWITH_CUDA"], ld) for c, ld in cpu_flag_variants]

    def _load(sources, extra_cuda_cflags, extra_cflags, extra_ldflags=None):
        _reset_jit_versioner(module_name)
        try:
            return torch_ext_load(
                name=module_name,
                sources=sources,
                extra_cflags=extra_cflags,
                extra_cuda_cflags=extra_cuda_cflags,
                extra_ldflags=extra_ldflags,
                verbose=verbose,
            )
        except TypeError:
            # some torch versions have different signature; try without the
            # optional flag arguments
            return torch_ext_load(
                name=module_name,
                sources=sources,
                extra_cflags=extra_cflags,
                verbose=verbose,
            )

    if not with_cuda:
        last_err = None
        for extra_cflags, extra_ldflags in cpu_flag_variants:
            try:
                module = _load(cpp_sources, [], extra_cflags, extra_ldflags)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err
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
        cuda_cflags, cuda_ldflags = cpu_flag_variants[0]
        return _load(cpp_sources, [], cuda_cflags, cuda_ldflags)

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

    extra_cuda_cflags = ["-O3"]
    base_cflags, base_ldflags = cpu_flag_variants[0]
    last_err = None
    for cc in attempt_order:
        flags = extra_cuda_cflags if cc is None else extra_cuda_cflags + ["-ccbin", cc]
        try:
            module = _load(cpp_sources + cu_sources, flags, base_cflags, base_ldflags)
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
