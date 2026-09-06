from setuptools import setup
import os
import sys
import platform
from glob import glob


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
    sources = glob(f"{extension_dir}/*.cpp")
    include_dirs = [extension_dir]

    # extra_compile_args = {"cxx": ["-std=c++17", "-O3", '-fopenmp', '-march=native']}
    extra_compile_args = dict()
    extra_link_args = []

    if sys.platform == "win32":
        extra_compile_args["cxx"] = ["/std:c++17", "/O2", "/openmp"]
    elif sys.platform == "darwin":
        extra_compile_args["cxx"] = ["-std=c++17", "-O3", "-mmacosx-version-min=10.14"]
    else:
        extra_compile_args["cxx"] = ["-std=c++17", "-O3", "-fopenmp"]

    if platform.machine() == "x86_64":
        extra_compile_args["cxx"] += ["-march=native"]

    define_macros = []
    if (torch.cuda.is_available() and os.getenv("WITH_CUDA", "0") == "1") or os.getenv("FORCE_CUDA", "0") == "1":
        sources += glob(f"{extension_dir}/*.cu")
        ext_type = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = ["-O3", "-std=c++17"]
    elif (getattr(torch.version, "hip", None) is not None and os.getenv("WITH_HIP", "0") == "1") or os.getenv("FORCE_ROCM", "0") == "1":
        sources += glob(f"{extension_dir}/*.cu")
        if ROCMExtension is not None:
            ext_type = ROCMExtension
            define_macros += [("WITH_HIP", None)]
            extra_compile_args["hipcc"] = ["-O3", "-std=c++17"]
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
    try:
        build_ver_path = os.path.join(extension_dir, "_torch_build_version.py")
        with open(build_ver_path, "w") as _bv:
            _bv.write(f"BUILD_TORCH_VERSION = {repr(torch.__version__)}\n")
            _bv.write(f"BUILD_WITH_CUDA = {repr('nvcc' in extra_compile_args)}\n")
            _bv.write(f"BUILD_WITH_HIP = {repr('hipcc' in extra_compile_args)}\n")
    except Exception:
        # If writing fails for any reason, continue; the runtime check will be skipped.
        pass
    return ext_modules, cmdclass


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


setup(
    name="torch_ans",
    version="0.2.1",
    description="PyTorch extension for parallel-enabled ANS-based compression (C++/CUDA)",
    author="worldlife",
    author_email="worldlife@sjtu.edu.cn",
    url="https://github.com/worldlife123/torch_ans",
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
    install_requires=["torch>=1.10", "pybind11", "ninja"],
    python_requires=">=3.7",
)