"""Dynamic runtime loader for the native torch_ans C++/CUDA extension.

Provides `build_extension` which compiles the native extension at runtime
using `torch.utils.cpp_extension.load` when the pre-built `torch_ans._C`
module is missing (lazy compile mode).

This is intentionally small and conservative: it mirrors enough of
`setup.py`'s decisions so runtime compilation behaves similarly.
"""
from __future__ import annotations

import dataclasses
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Incremental build profiles
#
# A profile is the operator subset a single TorchANSInterface instance needs.
# It maps to (a) the preprocessor gates in torch_ans/rans_build_config.hpp
# (profile_defines), (b) the JIT module name/build directory
# (profile_module_name) and (c) the operators the built module must export,
# which is also what a pre-existing module is checked against before it is
# reused (profile_op_names). The three must agree with the gated bindings in
# torch_ans/rans_bindings.hpp.
# ---------------------------------------------------------------------------

#: Environment variable that turns incremental compilation off (``0``/``false``).
INCREMENTAL_ENV_VAR = "TORCH_ANS_INCREMENTAL_COMPILE"

#: rANS family -> the macro that enables it, see rans_build_config.hpp
FAMILY_MACROS = {
    "rans64": "TORCH_ANS_WITH_RANS64",
    "rans32": "TORCH_ANS_WITH_RANS32",
    "rans32_16": "TORCH_ANS_WITH_RANS32_16",
}

#: short family code used in the module name / build directory
FAMILY_TAGS = {"rans64": "r64", "rans32": "r32", "rans32_16": "r3216"}

#: operator-name infix per interleave factor (``rans64_i4_push``, ...)
OP_SUFFIX = {1: "", 2: "_i2", 4: "_i4", 8: "_i8", 32: "_i32"}

#: interleave factors supported per family, and those with alias variants
#: (32-way interleaving is rans32_16-only and has no alias binding)
SUPPORTED_INTERLEAVES = {
    "rans64": (1, 2, 4, 8),
    "rans32": (1, 2, 4, 8),
    "rans32_16": (1, 2, 4, 8, 32),
}
ALIAS_INTERLEAVES = {
    "rans64": (1, 2, 4, 8),
    "rans32": (1, 2, 4, 8),
    "rans32_16": (1, 2, 4, 8),
}

#: operators that are not gated (needed by every interface)
COMMON_OPS = (
    "rans_stream_to_byte_strings",
    "rans_byte_strings_to_stream",
    "rans_alias_build_table",
    "rans_pmf_to_quantized_cdf",
    "rans_build_inverse_cdf",
)

#: JIT name of the full build. Not "torch_ans_C_<something>": it is the name
#: used before incremental compilation existed, so cached full builds keep
#: being reused and `torch_ans._C` keeps its historical meaning.
FULL_MODULE_NAME = "torch_ans_C"

#: Key of the CUDA toolchain state file. Deliberately not the module name:
#: whether the CUDA toolchain works (and with which host compiler) is a
#: property of the machine, while incremental builds create one module per
#: profile - without a shared key every new profile would re-attempt a build
#: that is already known to fail.
_CUDA_STATE_KEY = "torch_ans_cuda_toolchain"


@dataclasses.dataclass(frozen=True)
class BuildProfile:
    """Operator subset a build must provide.

    ``family=None`` selects the minimal "common operators only" profile; the
    full build is ``None`` (no profile at all) rather than a BuildProfile.
    """

    family: Optional[str] = None
    interleave: int = 1
    alias: bool = False
    invcdf: bool = False


def incremental_enabled(flag: Optional[bool] = None) -> bool:
    """Whether a build should contain only the operators it needs.

    ``flag`` (the ``incremental_compile=`` argument of
    :class:`~torch_ans.utils.TorchANSInterface`) wins; otherwise
    ``TORCH_ANS_INCREMENTAL_COMPILE`` is consulted, defaulting to enabled.
    """
    if flag is not None:
        return bool(flag)
    value = os.environ.get(INCREMENTAL_ENV_VAR)
    if value is None:
        return True
    return value.strip().lower() not in ("0", "false", "no", "off")


def profile_defines(profile: Optional[BuildProfile]) -> Dict[str, int]:
    """Preprocessor defines that select the operators of ``profile``.

    ``None`` is the full build and needs no defines: without
    ``TORCH_ANS_INCREMENTAL_BUILD`` every gate in rans_build_config.hpp
    defaults to 1.
    """
    if profile is None:
        return {}
    defines = {"TORCH_ANS_INCREMENTAL_BUILD": 1}
    if profile.family is not None:
        defines[FAMILY_MACROS[profile.family]] = 1
    if profile.interleave != 1:
        defines["TORCH_ANS_WITH_INTERLEAVE_%d" % profile.interleave] = 1
    if profile.alias:
        defines["TORCH_ANS_WITH_ALIAS"] = 1
    if profile.invcdf:
        defines["TORCH_ANS_WITH_INVCDF"] = 1
    return defines


def profile_module_name(profile: Optional[BuildProfile]) -> str:
    """JIT module name of ``profile`` (also its build cache directory)."""
    if profile is None:
        return FULL_MODULE_NAME
    if profile.family is None:
        return FULL_MODULE_NAME + "_common"
    parts = [FULL_MODULE_NAME, FAMILY_TAGS[profile.family]]
    if profile.interleave != 1:
        parts.append("i%d" % profile.interleave)
    if profile.alias:
        parts.append("alias")
    if profile.invcdf:
        parts.append("invcdf")
    return "_".join(parts)


def _ilv_op_names(family: str, interleave: int, alias: bool, invcdf: bool,
                  alias_ok: bool) -> List[str]:
    suffix = OP_SUFFIX[interleave]
    names = ["%s%s_push" % (family, suffix), "%s%s_pop" % (family, suffix)]
    if invcdf:
        names.append("%s%s_invcdf_pop" % (family, suffix))
    if alias and alias_ok:
        names += ["%s_alias%s_push" % (family, suffix),
                  "%s_alias%s_pop" % (family, suffix)]
    return names


def profile_op_names(profile: Optional[BuildProfile] = None) -> Tuple[str, ...]:
    """Operators a module built for ``profile`` must export.

    ``None`` asks for the full operator set. This is what a previously built
    module is checked against (with ``hasattr``) before it is reused for a
    different profile, so it must never under-report: a missing name only
    costs a rebuild, a name that is claimed but not exported would break the
    first use of the module.
    """
    names = list(COMMON_OPS)
    if profile is None:
        for family, interleaves in SUPPORTED_INTERLEAVES.items():
            names.append("%s_init_stream" % family)
            for interleave in interleaves:
                names += _ilv_op_names(
                    family, interleave, alias=True, invcdf=True,
                    alias_ok=interleave in ALIAS_INTERLEAVES[family])
        return tuple(names)
    if profile.family is not None:
        names.append("%s_init_stream" % profile.family)
        names += _ilv_op_names(
            profile.family, profile.interleave, alias=profile.alias,
            invcdf=profile.invcdf,
            alias_ok=profile.interleave in ALIAS_INTERLEAVES.get(profile.family, ()))
    return tuple(names)


def _define_flags(defines: Optional[Dict[str, object]]) -> List[str]:
    """Turn a macro dict into compiler flags (``-DNAME=VALUE``)."""
    if not defines:
        return []
    prefix = "/D" if sys.platform == "win32" else "-D"
    flags = []
    for name, value in defines.items():
        if value is None or value == "":
            flags.append(prefix + name)
        else:
            flags.append("%s%s=%s" % (prefix, name, value))
    return flags


def _detect_cuda_torch():
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _build_state_path(state_key: str, filename: str) -> Optional[str]:
    """Path of a build state marker inside the extension build cache.

    `state_key` picks the cache directory (a module name, or a key shared by
    every profile on purpose) and `filename` the marker in it. Returns None if
    torch's build directory cannot be determined (then there is no state).
    """
    try:
        from torch.utils.cpp_extension import _get_build_directory
        build_dir = _get_build_directory(state_key, verbose=False)
        os.makedirs(build_dir, exist_ok=True)
        return os.path.join(build_dir, filename)
    except Exception:
        return None


def _cuda_state_path(module_name: str) -> Optional[str]:
    """Path of the CUDA build state marker inside the extension build cache.

    `module_name` is really a state key: build_extension passes the constant
    ``_CUDA_STATE_KEY`` so that all profiles share one state (the toolchain
    works or it does not - it does not depend on which operators are built).

    Records the outcome of previous CUDA build attempts so later first-time
    imports in fresh processes do not re-pay doomed ones:
      - "cpu"    -> the CUDA build failed with every available configuration;
                    build CPU-only directly.
      - "ccbin:<path>" -> the CUDA build only works with the host compiler
                    at <path> (nvcc <= 12.1 cannot parse the std::function
                    headers of gcc >= 11.4); start attempts with it.
    Returns None if torch's build directory cannot be determined (no state).
    """
    return _build_state_path(module_name, "cuda_build_state")


#: State key of the macOS OpenMP probe (see _openmp_state_path). Deliberately
#: not a module name, like the CUDA key: whether a second OpenMP runtime aborts
#: here is a property of the machine, not of the operators being built.
_OMP_STATE_KEY = "torch_ans_macos_openmp"

#: Environment variable set for the child process of the OpenMP probe. The build
#: that child performs must not probe itself again (that would recurse).
_OMP_PROBE_ENV = "TORCH_ANS_OPENMP_PROBE"


def _openmp_state_path() -> Optional[str]:
    """Path of the macOS OpenMP state marker inside the extension build cache."""
    return _build_state_path(_OMP_STATE_KEY, "openmp_build_state")


def _openmp_state() -> Optional[str]:
    """First line of the recorded OpenMP probe outcome, or None.

    "broken" is followed by a human-readable explanation; "ok <torch version>"
    is what the probe writes when the module survived.
    """
    path = _openmp_state_path()
    if path is None or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return f.readline().strip() or None
    except OSError:
        return None


def _openmp_broken() -> bool:
    """Whether an earlier build found the OpenMP runtime unusable on this machine."""
    return (_openmp_state() or "").startswith("broken")


def _openmp_probe_is_redundant() -> bool:
    """Whether the probe already passed for the torch version in use.

    The probe spawns a process (torch import included), so it must not run on
    every first use; the recorded "ok" is keyed by the torch version because
    that is what the crash it looks for depends on. Deleting the file forces a
    fresh probe.
    """
    return _openmp_state() == f"ok {_torch_version()}"


def _write_openmp_state(status: str, explanation: str = "") -> None:
    path = _openmp_state_path()
    if path is None:
        return
    try:
        with open(path, "w") as f:
            f.write(status + "\n")
            if explanation:
                f.write(explanation)
    except OSError:
        pass


def _remember_openmp_ok() -> None:
    """Record that the OpenMP module survived, so later builds skip the probe."""
    _write_openmp_state(f"ok {_torch_version()}")


def _remember_openmp_broken() -> None:
    """Record that the OpenMP build aborted, so later builds skip that variant.

    Only failures are written with a reason, and they apply to the machine they
    happened on: the file keeps the fallback (which still parallelizes, through
    torch's own thread pool) until it is deleted.
    """
    _write_openmp_state(
        "broken",
        "The extension built with Homebrew's libomp aborted this machine on its "
        "first parallel op, so runtime builds use the flags that do not add a "
        "second OpenMP runtime. Delete this file to try OpenMP again.\n")


def _run_probe_child(code: str, env: Dict[str, str]):
    """Run the OpenMP probe's child process (separate so tests can replace it).

    The child repeats the build (a cache hit) and one operator call, so it needs
    seconds; the timeout only bounds a hang (a wedged OpenMP runtime), and a
    timeout keeps the OpenMP build like any other probe failure.
    """
    return subprocess.run(
        [sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, timeout=300)


def _openmp_variant_is_usable(module_name: str, defines, verbose: bool = False) -> bool:
    """Whether the OpenMP variant can code without killing the process here.

    macOS takes its OpenMP runtime from Homebrew's libomp, a *second* OpenMP
    runtime next to the one libtorch links. With torch 2.14 that combination
    aborts on the first `at::parallel_for` - SIGABRT inside
    rans_pmf_to_quantized_cdf, no exception, and nothing at build time notices,
    because the module links and imports perfectly well (torch 2.13 and older
    with the same libomp work, and the build without a second runtime never had
    the problem).

    So build and use the module *in a child process*, where a crash is only an
    exit status, and let the caller fall back to the flags without a second
    runtime - exactly what setup.py has always used on macOS. The child runs the
    same `build_extension(module_name, defines)` call, i.e. it takes the first
    variant too and leaves it in the shared JIT cache, so when the probe passes
    the parent's own build is a cache hit.

    Probing *before* the parent builds anything is what makes the fallback work
    in the same process: `dlopen` keys on the file path, and the two variants
    share one module name, so a module the parent had already loaded would be
    the one that came back after the rebuild.

    Returns True when the child survived, and also when the probe could not be
    run at all: it is a guess about the local toolchain, not a reason to reject
    a module that may work fine. The outcome is recorded so the child is spawned
    once per (machine, torch version) rather than on every first use.
    """
    if os.environ.get(_OMP_PROBE_ENV):
        return True  # we are the child: never probe recursively
    if _openmp_probe_is_redundant():
        return True  # already verified here for this torch version
    code = (
        "import torch\n"
        "import torch_ans._dynamic_build as db\n"
        f"m = db.build_extension(module_name={module_name!r}, with_cuda=False, "
        f"verbose=False, defines={defines!r})\n"
        "torch.manual_seed(0)\n"
        "pmf = torch.rand(8, 256)\n"
        "pmf = pmf / pmf.sum(-1, keepdim=True)\n"
        "m.rans_pmf_to_quantized_cdf(pmf, 16)\n"
        "print('torch_ans: the OpenMP build coded a pmf')\n"
    )
    env = dict(os.environ)
    env[_OMP_PROBE_ENV] = "1"
    if verbose:
        print("torch_ans: checking the OpenMP build against its first parallel op")
    try:
        proc = _run_probe_child(code, env)
    except Exception as exc:  # no interpreter, timeout, ...
        print(f"torch_ans: the OpenMP probe could not be run ({exc}); keeping OpenMP")
        return True
    if proc.returncode == 0:
        _remember_openmp_ok()
        return True
    signal_note = f", signal {-proc.returncode}" if proc.returncode < 0 else ""
    tail = (proc.stdout or b"").decode("utf-8", "replace").strip().splitlines()[-15:]
    print(
        "torch_ans: the OpenMP build failed its first parallel op (exit "
        f"{proc.returncode}{signal_note}); rebuilding without a second OpenMP "
        "runtime - the child said:\n  " + "\n  ".join(tail))
    _remember_openmp_broken()
    return False


def _older_host_compilers():
    """Host compilers known to work around nvcc/libstdc++ header incompatibilities."""
    import shutil
    compilers = []
    for candidate in ("g++-10", "g++-9", "g++-8"):
        path = shutil.which(candidate)
        if path:
            compilers.append(path)
    return compilers


def _libomp_prefix() -> str:
    """Homebrew's libomp prefix, or "" when it is not installed."""
    try:
        out = subprocess.run(["brew", "--prefix", "libomp"], capture_output=True, text=True, timeout=10)
        prefix = out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""
    return prefix if os.path.exists(os.path.join(prefix, "include", "omp.h")) else ""


def _darwin_cpu_flag_variants():
    """CPU compile flag variants for macOS, fastest (OpenMP) first.

    `rans_cpu.cpp` parallelizes batch coding with `at::parallel_for`, which is
    multi-threaded only when the extension is compiled with OpenMP enabled
    (`_OPENMP` selects ATen's AT_PARALLEL_OPENMP backend; ~3-4x throughput on
    many-core machines). Apple clang has no built-in OpenMP, so when Homebrew
    libomp is available use the standard `-Xpreprocessor -fopenmp` + `-lomp`
    recipe; otherwise (or if the OpenMP attempt fails) fall back to flags
    that still build, at the cost of a serial at::parallel_for.

    The OpenMP variant is dropped once a build has found it unusable on this
    machine (see _openmp_coding_survives): it links and imports fine but the
    process aborts on the first parallel op, which a build failure check cannot
    see.
    """
    base = ["-O3", "-mmacosx-version-min=10.14"]
    variants = []
    prefix = _libomp_prefix()
    if prefix and not _openmp_broken():
        variants.append((
            base + ["-Xpreprocessor", "-fopenmp", f"-I{prefix}/include"],
            [f"-L{prefix}/lib", "-lomp", f"-Wl,-rpath,{prefix}/lib"],
        ))
    variants.append((base, []))
    return variants


def _record_build_metadata(pkg_dir: Path, build_with_cuda: bool) -> None:
    """Record build metadata for the runtime torch-version check in __init__."""
    try:
        import torch
        build_ver_path = pkg_dir / "_torch_build_version.py"
        with open(build_ver_path, "w") as f:
            f.write(f"BUILD_TORCH_VERSION = {repr(torch.__version__)}\n")
            f.write(f"BUILD_WITH_CUDA = {repr(build_with_cuda)}\n")
            f.write(f"BUILD_WITH_HIP = {repr(False)}\n")
    except Exception:
        pass


def _torch_major_version() -> int:
    """Major version of the installed torch (2 when it cannot be determined)."""
    try:
        import torch
        return int(torch.__version__.split("+")[0].split(".")[0])
    except Exception:
        return 2


def _torch_version() -> str:
    """Version of the installed torch ("unknown" when it cannot be determined)."""
    try:
        import torch
        return str(torch.__version__)
    except Exception:
        return "unknown"


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


def build_extension(module_name: str = "torch_ans_C_ext", with_cuda: Optional[bool] = None, verbose: bool = False, defines: Optional[Dict[str, object]] = None):
    """Build the native extension under `module_name` and return the module.

    This function purposely does not try to import `torch_ans._C` first to avoid
    recursion when used from a shim module. If `with_cuda` is truthy, CUDA
    support is used when torch reports a working CUDA runtime, otherwise a
    CPU-only extension is built.

    NOTE: `module_name` must not contain dots. torch's JIT import machinery
    derives the required `PyInit_*` symbol from the module name, and pybind11
    cannot paste dotted names into that symbol. Undotted names additionally
    stay importable when the build is repeated with changed arguments.

    `defines` maps preprocessor macro names to values and is added to both the
    C++ and the nvcc command lines. Incremental builds use it for the
    TORCH_ANS_WITH_* gates (see profile_defines); the module name must already
    identify the profile, since a given name caches exactly one define set.

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
    # order. The C++ standard is chosen as follows: torch's cpp_extension
    # always appends the standard matching its own headers (c++14 for torch
    # 1.x, c++17 for 2.x up to ~2.13, c++20 for newer) and a user-supplied
    # `-std=` comes later on the command line and would override it — so we
    # supply `-std=c++17` (needed for `std::optional`/`if constexpr` in our
    # headers) only when torch's own default is older (torch 1.x), and never
    # downgrade newer torch releases that require C++20.
    torch_major = _torch_major_version()
    std_flags = ["/std:c++17" if sys.platform == "win32" else "-std=c++17"] if torch_major < 2 else []

    # Xcode 16's clang turns the std::is_arithmetic specialization in torch
    # 2.7's c10/util/strong_type.h into an error (-Winvalid-specialization).
    darwin_extra = ["-Wno-invalid-specialization"] if sys.platform == "darwin" else []

    # Feature gates of an incremental build (empty for a full build). They go
    # to the C++ compiler and to nvcc alike, since the gates are used by both
    # translation units.
    define_flags = _define_flags(defines)

    if sys.platform == "win32":
        # /bigobj: rans_cpu.cpp instantiates the interleaved kernels for every
        # (family, interleave, lookup) combination and its object outgrows the
        # section limit of the default object format ("fatal error C1128: number
        # of sections exceeded object file format limit"). Install-time builds do
        # not hit this: distutils adds /GL, which emits code at link time instead
        # of one section per function, and that is also why setup.py does not
        # need the flag here.
        # /Zc:lambda (the conforming MSVC lambda processor): the legacy one, the
        # default in C++17 mode, cannot see the declarations of an `if constexpr`
        # branch inside rans_cpu.cpp's nested generic lambdas and fails the build
        # with C2065. Only /std:c++20 (torch >= 2.14) turns it on implicitly, so
        # a runtime build against an older torch needs it explicitly. See setup.py.
        cpu_flag_variants = [(["/O2", "/openmp", "/bigobj", "/Zc:lambda"] + std_flags + define_flags, [])]
    elif sys.platform == "darwin":
        cpu_flag_variants = [(c + darwin_extra + define_flags, ld) for c, ld in _darwin_cpu_flag_variants()]
    else:
        cflags = ["-O3", "-fopenmp"] + std_flags + define_flags
        if platform.machine() == "x86_64":
            cflags.append("-march=native")
        cpu_flag_variants = [(cflags, [])]

    # setup.py's CUDAExtension defines WITH_CUDA for install-time builds; the
    # JIT loader does not, so the CUDA dispatch branches in rans.hpp would be
    # compiled out and CUDA tensors would hit "not compiled with GPU support".
    # The macro is only correct for the full CUDA build (cpp + cu sources):
    # the guarded dispatch calls functions defined in the .cu sources, so a
    # cpp-only build with it produces a .so with undefined symbols that fails
    # to load. `cpu_only_flag_variants` stays free of it and is used for
    # every cpp-only build below (including the failed-CUDA shortcut).
    cpu_only_flag_variants = cpu_flag_variants
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

    def _build_cpu_only():
        last_err = None
        for extra_cflags, extra_ldflags in cpu_only_flag_variants:
            # An OpenMP variant (the darwin one; only macOS links a second OpenMP
            # runtime) can build, import and still abort on the first parallel op.
            # Test it in a child process first - the probe builds the module
            # there, so nothing the fallback may replace is loaded here yet - and
            # fall back to the variant below when it does not survive.
            if "-Xpreprocessor" in extra_cflags and not _openmp_variant_is_usable(
                    module_name, defines, verbose):
                last_err = RuntimeError(
                    "the extension built with Homebrew's libomp aborts on its first "
                    "parallel op on this machine")
                continue
            try:
                module = _load(cpp_sources, [], extra_cflags, extra_ldflags)
                last_err = None
                break
            except Exception as e:
                last_err = e
        if last_err is not None:
            raise last_err
        _record_build_metadata(pkg_dir, False)
        return module

    if not with_cuda:
        return _build_cpu_only()

    # CUDA build orchestration with failure memory (see _cuda_state_path).
    state_path = _cuda_state_path(_CUDA_STATE_KEY)
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
        # CPU-only flags (no -DWITH_CUDA): defining it here would reference the
        # CUDA implementations that live in the .cu sources and are not linked
        # into this build, making the .so fail to load.
        return _build_cpu_only()

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

    # WITH_CUDA must reach nvcc explicitly: torch's cpp_extension passes
    # extra_cflags to the C++ compiler only and (unlike WITH_HIP) does not
    # define WITH_CUDA itself, and rans_cuda.cu is empty without it.
    extra_cuda_cflags = ["-O3", "-DWITH_CUDA"] + define_flags
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

        _record_build_metadata(pkg_dir, True)
        return module

    # Every CUDA configuration failed: remember to build CPU-only next time.
    if state_path is not None:
        try:
            with open(state_path, "w") as f:
                f.write("cpu\n")
        except OSError:
            pass
    raise last_err
