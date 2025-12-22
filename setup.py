from setuptools import setup
import os
import sys
from glob import glob


def get_extension_config():
    """Import torch and torch extension builders only when we are building extensions.

    This avoids importing torch at setup.py import time which otherwise forces
    pip to install torch into the PEP 517 isolated build environment when
    `pyproject.toml` lists torch in build-system.requires.
    """
    # Import here so an environment that already has torch can be used for local builds
    from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
    import torch
    # TODO: Add ROCMExtension import
    # try:
    #     from torch.utils.cpp_extension import ROCMExtension
    # except ImportError:
    #     ROCMExtension = None

    extension_dir = "torch_ans"
    sources = glob(f"{extension_dir}/*.cpp")
    include_dirs = [extension_dir]

    extra_compile_args = {"cxx": ["-std=c++17", "-O3", '-fopenmp', '-march=native']}
    define_macros = []

    if torch.cuda.is_available() or os.getenv("FORCE_CUDA", "0") == "1":
        sources += glob(f"{extension_dir}/*.cu")
        ext_type = CUDAExtension
        define_macros += [("WITH_CUDA", None)]
        extra_compile_args["nvcc"] = ["-O3", "-std=c++17"]
    # elif getattr(torch.version, "hip", None) is not None or os.getenv("FORCE_ROCM", "0") == "1":
    #     sources += glob(f"{extension_dir}/*.hip")
    #     if ROCMExtension is not None:
    #         ext_type = ROCMExtension
    #         define_macros += [("WITH_HIP", None)]
    #         extra_compile_args["hipcc"] = ["-O3", "-std=c++17"]
    #     else:
    #         ext_type = CppExtension  # fallback if ROCMExtension not available
    else:
        ext_type = CppExtension

    ext_modules = [
        ext_type(
            "torch_ans._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    cmdclass = {"build_ext": BuildExtension}
    # Record the torch version used at build time so we can verify at runtime.
    try:
        build_ver_path = os.path.join(extension_dir, "_torch_build_version.py")
        with open(build_ver_path, "w") as _bv:
            _bv.write(f"BUILD_TORCH_VERSION = {repr(torch.__version__)}\n")
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
if any(arg in sys.argv for arg in ("build_ext", "bdist_wheel", "bdist_egg", "install", "develop")):
    ext_modules, cmdclass = get_extension_config()


setup(
    name="torch_ans",
    version="0.1.1",
    description="PyTorch extension for parallel-enabled ANS-based compression (C++/CUDA)",
    author="worldlife",
    author_email="worldlife@sjtu.edu.cn",
    url="https://github.com/worldlife123/torch_ans",
    packages=["torch_ans"],
    include_package_data=True,
    package_data={
        "torch_ans": [
            "*.hpp",
            "*.h",
            "*.hh",
            "*.inl",
            "*.ipp",
            "*.cu",
            "*.cuh",
            "*.cpp",
        ]
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    install_requires=["torch>=1.12", "pybind11"],
    python_requires=">=3.8",
)