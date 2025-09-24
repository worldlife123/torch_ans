from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import torch
import os
from glob import glob

# TODO: Add ROCMExtension import
# try:
#     from torch.utils.cpp_extension import ROCMExtension
# except ImportError:
#     ROCMExtension = None

extension_dir = "torch_ans"
sources = glob(f"{extension_dir}/*.cpp")
include_dirs = [extension_dir]

extra_compile_args = {"cxx": ["-std=c++17", "-O3", '-fopenmp', '-march=native']} # -fopenmp is required for cpu parallel acceleration
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

setup(
    name="torch_ans",
    version="0.1.0",
    description="PyTorch extension for parallel-enabled ANS-based compression (C++/CUDA)",
    author="worldlife",
    author_email="worldlife@sjtu.edu.cn",
    url="https://github.com/worldlife123/torch_ans",
    packages=["torch_ans"],
    ext_modules=[
        ext_type(
            "torch_ans._C",
            sources=sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    install_requires=["torch>=1.12", "pybind11"],
    python_requires=">=3.8",
)