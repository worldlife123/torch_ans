from setuptools import setup
import os
import sys
import platform
from glob import glob


# ---------------------------------------------------------------------------
# Build flavours
#
# The default build (no switches below) is the ATen/pybind11 extension compiled
# against the local torch: `pip install .` and the runtime JIT both use it, and
# nothing in this file changes that path.
#
# WITH_STABLE_ABI=1 produces the prebuilt libtorch-stable-ABI libraries instead:
# one `.so` per flavour that every
# torch >= 2.10 can load, because it only talks to the stable C shim. torch is
# needed for its headers at build time and not at all at runtime.
# ---------------------------------------------------------------------------
WITH_STABLE_ABI = os.getenv("WITH_STABLE_ABI", "0") == "1"

#: Minimum libtorch ABI a prebuilt .so targets, encoded as
#: [MAJ][MIN][PATCH][ABI tag] for -DTORCH_TARGET_VERSION (torch 2.10).
STABLE_ABI_TORCH_TARGET_VERSION = "0x020a000000000000"

#: Archs a CUDA flavour is compiled for. Explicit on purpose: torch's own
#: `torch.cuda.get_arch_list()` is whatever that torch build shipped (6 archs on
#: the local cu130 build -> ~55-60 MB of kernels instead of ~14 MB), and a wheel
#: has to name its archs anyway. 7.5 (2080 Ti) through 9.0 plus a PTX fallback
#: covers every card from the 20-series up. Override with TORCH_CUDA_ARCH_LIST
#: (e.g. `TORCH_CUDA_ARCH_LIST=7.5` for a fast local build).
STABLE_ABI_CUDA_ARCH_LIST = os.getenv("TORCH_CUDA_ARCH_LIST", "7.5;8.0;8.6;9.0+PTX")


def write_build_metadata(extension_dir, torch_version, with_cuda, with_hip,
                         stable_abi=False):
    """Record how the native extension was built (read by torch_ans/__init__.py).

    The ATen build records the torch it compiled against, because such an
    extension is bound to that ABI and the runtime check in `__init__.py` warns
    when the two differ.

    The stable-ABI build must *not* record it: being torch-version independent is
    the whole point (TORCH_TARGET_VERSION = torch 2.10), so a "compiled with torch
    X, running torch Y" warning would be exactly the false positive the migration
    removes. It writes BUILD_STABLE_ABI instead and leaves BUILD_TORCH_VERSION
    absent, which makes that check a no-op. BUILD_WITH_CUDA still matters:
    `utils.module_has_cuda()` falls back to it for an extension too old to carry
    the `_torch_ans_with_cuda` attribute.
    """
    try:
        build_ver_path = os.path.join(extension_dir, "_torch_build_version.py")
        with open(build_ver_path, "w") as handle:
            if not stable_abi:
                handle.write(f"BUILD_TORCH_VERSION = {repr(torch_version)}\n")
            handle.write(f"BUILD_STABLE_ABI = {repr(bool(stable_abi))}\n")
            handle.write(f"BUILD_WITH_CUDA = {repr(bool(with_cuda))}\n")
            handle.write(f"BUILD_WITH_HIP = {repr(bool(with_hip))}\n")
    except Exception:
        # If writing fails for any reason, continue; the runtime check will be skipped.
        pass


def get_extension_config():
    """Import torch and torch extension builders only when we are building extensions.

    This avoids importing torch at setup.py import time which otherwise forces
    pip to install torch into the PEP 517 isolated build environment when
    `pyproject.toml` lists torch in build-system.requires.
    """
    # Import here so an environment that already has torch can be used for local builds
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    try:
        from torch.utils.cpp_extension import ROCMExtension
    except ImportError:
        ROCMExtension = None
    import torch

    extension_dir = "torch_ans"
    # lib_cuda.cpp is the entry point of the *stable-ABI* CUDA library
    # (WITH_STABLE_ABI=1, get_stable_abi_extension_config below). It compiles to an
    # empty translation unit here (see the file), so leaving it out only keeps this
    # build's source list exactly what it was before the migration.
    sources = [path for path in glob(f"{extension_dir}/*.cpp")
               if os.path.basename(path) != "lib_cuda.cpp"]
    include_dirs = [extension_dir]

    # extra_compile_args = {"cxx": ["-std=c++17", "-O3", '-fopenmp', '-march=native']}
    extra_compile_args = dict()
    extra_link_args = []

    # torch's cpp_extension appends its own -std (c++14 for torch 1.x, c++17
    # /c++20 for newer) and our headers need C++17 (std::optional / if
    # constexpr), so supply -std=c++17 only when torch's own default is older
    # (torch 1.x); for torch >= 2.0 never pass one — a user-supplied -std
    # comes later on the command line and would override torch's choice,
    # breaking builds against newer torch releases that require C++20.
    # (See torch_ans/_dynamic_build.py for the same logic in the runtime
    # build.)
    try:
        _torch_major = int(torch.__version__.split("+")[0].split(".")[0])
    except Exception:
        _torch_major = 2
    std_flag = "/std:c++17" if sys.platform == "win32" else "-std=c++17"
    std_flags = [std_flag] if _torch_major < 2 else []
    # Xcode 16's clang turns the std::is_arithmetic specialization in torch
    # 2.7's c10/util/strong_type.h into an error (-Winvalid-specialization).
    darwin_extra = ["-Wno-invalid-specialization"] if sys.platform == "darwin" else []

    if sys.platform == "win32":
        # /Zc:lambda selects MSVC's conforming lambda processor. The legacy one,
        # which is the default in C++17 mode (what torch <= 2.13 asks for on
        # Windows), fails to see the declarations made inside an `if constexpr`
        # branch of the nested generic lambdas in rans_cpu.cpp's interleaved pop
        # path and reports them as undeclared identifiers (C2065). The
        # conforming processor handles them; it is implied by /std:c++20,
        # /std:c++latest and /permissive- only, so ask for it explicitly. It is
        # a no-op once one of those is in effect (torch 2.14+ builds extensions
        # with /std:c++20).
        extra_compile_args["cxx"] = ["/O2", "/openmp", "/Zc:lambda"] + std_flags
    elif sys.platform == "darwin":
        extra_compile_args["cxx"] = ["-O3", "-mmacosx-version-min=10.14"] + darwin_extra + std_flags
    else:
        extra_compile_args["cxx"] = ["-O3", "-fopenmp"] + std_flags

    if platform.machine() == "x86_64":
        extra_compile_args["cxx"] += ["-march=native"]

    define_macros = []
    if (torch.cuda.is_available() and os.getenv("WITH_CUDA", "0") == "1") or os.getenv("FORCE_CUDA", "0") == "1":
        sources += glob(f"{extension_dir}/*.cu")
        ext_type = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = ["-O3"]
    elif (getattr(torch.version, "hip", None) is not None and os.getenv("WITH_HIP", "0") == "1") or os.getenv("FORCE_ROCM", "0") == "1":
        sources += glob(f"{extension_dir}/*.cu")
        if ROCMExtension is not None:
            ext_type = ROCMExtension
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["hipcc"] = ["-O3"]
        else:
            ext_type = CppExtension  # fallback if ROCMExtension not available
    else:
        ext_type = CppExtension

    if os.getenv("ENABLE_COVERAGE", "0") == "1":
        extra_compile_args["cxx"] += ["-fprofile-arcs", "-ftest-coverage"]
        extra_link_args += ["-fprofile-arcs", "-ftest-coverage"]
        if sys.platform != "darwin":
            extra_link_args += ["-lgcov"]
        if "nvcc" in extra_compile_args:
            extra_compile_args["nvcc"] += ["-Xcompiler", "-fprofile-arcs", "-Xcompiler", "-ftest-coverage"]

    ext_modules = [
        ext_type(
            "torch_ans._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
    # Record the torch version used at build time so we can verify at runtime.
    write_build_metadata(extension_dir, torch.__version__,
                         with_cuda="nvcc" in extra_compile_args,
                         with_hip="hipcc" in extra_compile_args)
    return ext_modules, cmdclass


def get_stable_abi_extension_config():
    """The prebuilt libtorch-stable-ABI libraries (migration section 5).

    Same shape as the POC (poc/stable_abi/setup.py), which is the configuration
    that was measured:

      * ``torch_ans._C_cpu``  - always: every schema plus the CPU and catch-all
        implementations;
      * ``torch_ans._C_cuda`` - with WITH_CUDA=1: CUDA implementations only, no
        schema (registering the same operator twice is rejected by the
        dispatcher, migration section 4, pit 8).

    Differences from the ATen build, each with its reason in the migration doc:
    no pybind11 (the .so exports no Python symbols at all), no ``-fopenmp`` (a
    second OpenMP runtime next to torch's gives the process two thread pools and
    aborts on macOS - pit 5), no ``-march=native`` (a wheel has to run on any
    x86-64, and it was measured to make no difference - section 7, risk 4) and
    ``-Wl,--as-needed``, so the ``-ltorch_python`` that cpp_extension links
    unconditionally is dropped instead of becoming a DT_NEEDED entry.
    """
    from torch.utils import cpp_extension
    import torch

    # The CUDA flavour is built on request only - the same switch the ATen path
    # uses - and a wheel builder without a visible GPU sets FORCE_CUDA=1.
    with_cuda = bool(
        (torch.cuda.is_available() and os.getenv("WITH_CUDA", "0") == "1")
        or os.getenv("FORCE_CUDA", "0") == "1"
    )

    keep_cuda_check = os.getenv("TORCH_ANS_STABLE_KEEP_CUDA_CHECK", "0").strip().lower() in (
        "1", "true", "yes", "on")
    if with_cuda and not keep_cuda_check:
        # cpp_extension refuses a toolkit whose *major* version differs from the
        # one torch was built with. That is the right default for an ordinary
        # extension - it links libcudart *and* ATen's CUDA headers, which do have
        # to match - but a stable-ABI .so only talks to the C shim, and building
        # it with CUDA 11.8 while the running torch is cu130 is a configuration
        # that was measured to work (poc/stable_abi/README.md section 3.4). Set
        # TORCH_ANS_STABLE_KEEP_CUDA_CHECK=1 to keep torch's behaviour.
        print("[torch_ans] WITH_STABLE_ABI: bypassing "
              "torch.utils.cpp_extension._check_cuda_version (a stable-ABI "
              "extension does not link against ATen's CUDA headers)")
        cpp_extension._check_cuda_version = lambda *args, **kwargs: None

    package_dir = "torch_ans"
    common_defines = [
        "-DTORCH_ANS_STABLE_ABI=1",
        "-DTORCH_TARGET_VERSION=" + STABLE_ABI_TORCH_TARGET_VERSION,
    ]
    extra_link_args = ["-Wl,--as-needed"]

    ext_modules = [
        cpp_extension.CppExtension(
            "torch_ans._C_cpu",
            sources=[os.path.join(package_dir, name)
                     for name in ("rans_cpu.cpp", "rans.cpp", "lib.cpp")],
            include_dirs=[package_dir],
            extra_compile_args={"cxx": ["-O3"] + common_defines},
            extra_link_args=extra_link_args,
        )
    ]

    if with_cuda:
        # -DUSE_CUDA makes the stable C shim declare its CUDA entries and
        # -DWITH_CUDA sets the ta:: layer's TA_WITH_CUDA gate; the two are
        # separate switches and both are required (migration section 4, pit 1).
        # The host compiler matters too: nvcc refuses a gcc it does not know
        # (CUDA 11.8 + gcc 11 is the usual case), so an explicit
        # `-ccbin <path>` is supported through TORCH_ANS_STABLE_CUDA_CCBIN
        # CUDA_HOME selects the toolkit itself, as always.
        cuda_defines = common_defines + ["-DWITH_CUDA=1", "-DUSE_CUDA=1"]
        nvcc_flags = ["-O3"] + cuda_defines
        cuda_ccbin = os.getenv("TORCH_ANS_STABLE_CUDA_CCBIN")
        if cuda_ccbin:
            nvcc_flags += ["-ccbin", cuda_ccbin]
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", STABLE_ABI_CUDA_ARCH_LIST)
        ext_modules.append(
            cpp_extension.CUDAExtension(
                "torch_ans._C_cuda",
                # rans_cpu.cpp belongs to the CUDA library on purpose: rans_push /
                # rans_pop are *one* wrapper each that routes on the tensor device
                # at runtime (rans.hpp), so the CPU kernels have to link here too.
                # The duplicated CPU code is ~3 MB next to ~14 MB of kernels, and
                # it keeps the two dispatch keys registering the same wrappers.
                # rans.cpp (the host-only alias table builder) rides along so that
                # the CPU-registration function - which is only *defined* here -
                # can never leave an unresolved symbol behind.
                sources=[os.path.join(package_dir, name)
                         for name in ("rans_cuda.cu", "rans_cpu.cpp", "rans.cpp",
                                      "lib_cuda.cpp")],
                include_dirs=[package_dir],
                extra_compile_args={
                    "cxx": ["-O3"] + cuda_defines,
                    "nvcc": nvcc_flags,
                },
                extra_link_args=extra_link_args,
            )
        )

    cmdclass = {"build_ext": cpp_extension.BuildExtension}
    return ext_modules, cmdclass, with_cuda


# Only import/build extensions when building distributions or extensions.
# This prevents importing torch during metadata-only operations and allows
# local installs to use an existing torch in the environment by using
# `pip install . --no-build-isolation` if desired.
ext_modules = []
cmdclass = {}
# Allow skipping building of native extensions at install time via env var.
SKIP_BUILD_EXT = os.getenv("SKIP_BUILD_EXT", "0") == "1"
# Detect whether we are actually able to import torch at build time. When
# pip runs a PEP 517 isolated build it creates a temporary environment that
# often does not contain the user's installed `torch`. In that case we
# should avoid trying to import torch (which would raise) and skip building
# the extension so the package can still be installed as a pure-Python
# distribution and the native extension can be compiled later at runtime.
want_build = any(arg in sys.argv for arg in ("build_ext", "bdist_wheel", "bdist_egg", "install", "develop"))
if SKIP_BUILD_EXT:
    print("SKIP_BUILD_EXT=1: skipping build of C++/CUDA extensions at install time")
elif want_build:
    if WITH_STABLE_ABI:
        # Unlike the ATen path below, a failure here is not a "pure-Python
        # install that compiles later": it would silently produce a wheel without
        # the extension the switch exists to ship. Only the missing-torch case is
        # tolerated (with a message); a broken toolchain propagates.
        try:
            import importlib
            _torch = importlib.import_module("torch")
        except Exception:
            print(
                "torch is not importable during build; cannot produce the prebuilt "
                "libtorch-stable-ABI libraries. Install torch in the build environment "
                "(e.g. use `pip install . --no-build-isolation`) or drop WITH_STABLE_ABI=1."
            )
        else:
            ext_modules, cmdclass, _stable_with_cuda = get_stable_abi_extension_config()
            # No BUILD_TORCH_VERSION on purpose: a stable-ABI .so targets torch
            # >= 2.10 and must not trigger the "compiled with torch X" warning.
            write_build_metadata("torch_ans", _torch.__version__,
                                 with_cuda=_stable_with_cuda, with_hip=False,
                                 stable_abi=True)
    else:
        # Try to import torch in a safe manner; if it's not available we assume
        # an isolated build environment and skip building the native extension.
        try:
            import importlib
            _torch = importlib.import_module("torch")
            # If import succeeded, proceed to get extension config and build.
            ext_modules, cmdclass = get_extension_config()
        except Exception:
            print(
                "torch is not importable during build; skipping native extension build. "
                "If you want to build the extension during install, ensure torch is pre-installed "
                "in the build environment (e.g. use `pip install . --no-build-isolation`)."
            )


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "README.md"), encoding="utf-8") as _f:
    _long_description = _f.read()


def _setuptools_reads_pyproject_metadata() -> bool:
    """True when the running setuptools reads `[project]` from pyproject.toml.

    Isolated builds always have it (`build-system.requires` pins
    `setuptools>=61`), but `pip install . --no-build-isolation` runs with
    whatever setuptools the user already has, and versions below 61 ignore the
    `[project]` table completely.
    """
    try:
        import setuptools
        major, minor = (int(part) for part in setuptools.__version__.split(".")[:2])
        return (major, minor) >= (61, 0)
    except Exception:
        return False


#: Mirrors `[project]` in pyproject.toml and is passed *only* on the legacy
#: setuptools path, where `[project]` is ignored: without it that build would
#: silently produce an "UNKNOWN" distribution with no dependencies. Keeping the
#: mirror out of the modern path leaves `[project]` as the single source of truth
#: and avoids setuptools' "overwritten in pyproject.toml" warnings. Bump the
#: version in both places: tests/test_packaging_metadata.py checks they agree.
_LEGACY_METADATA = {
    "name": "torch_ans",
    "version": "0.4.0",
    "description": "PyTorch extension for parallel-enabled ANS-based compression (C++/CUDA)",
    "long_description": _long_description,
    "long_description_content_type": "text/markdown",
    "license": "MIT",
    "author": "worldlife",
    "author_email": "worldlife@sjtu.edu.cn",
    "url": "https://github.com/worldlife123/torch_ans",
    "install_requires": ["torch>=1.10", "ninja"],
    "python_requires": ">=3.7",
}

setup(
    packages=["torch_ans"],
    include_package_data=True,
    package_data={
        "torch_ans": [
            "*.hpp",
            "*.cpp",
            "*.h",
            # "*.hh",
            # "*.inl",
            # "*.ipp",
            "*.cu",
            "*.cuh",
        ]
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    **({} if _setuptools_reads_pyproject_metadata() else _LEGACY_METADATA),
)