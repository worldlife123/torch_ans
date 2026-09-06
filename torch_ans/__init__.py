"""torch_ans package initializer with runtime build-version check.

This module checks whether the runtime `torch` version matches the
`torch` version recorded at build time (if present). If they differ, it
raises an informative ImportError to avoid obscure ABI/runtime failures.
"""

# Try to import the recorded build-time torch version (written by setup.py).
BUILD_TORCH_VERSION = None
try:
	from ._torch_build_version import BUILD_TORCH_VERSION
except Exception:
	BUILD_TORCH_VERSION = None

import os
from typing import Optional

# Try to determine the runtime torch version without importing the full torch
# package (importing torch can initialize CUDA and trigger driver checks).
runtime_torch_version = None
try:
	# Python 3.8+: importlib.metadata
	from importlib import metadata as _metadata
	runtime_torch_version = _metadata.version("torch")
except Exception:
	try:
		import torch as _torch
		runtime_torch_version = getattr(_torch, "__version__", None)
	except Exception:
		runtime_torch_version = None

def _major_minor(v: str) -> str:
	v = str(v).split('+', 1)[0]
	parts = v.split('.')
	return '.'.join(parts[:2]) if len(parts) >= 2 else v

import warnings

if BUILD_TORCH_VERSION is not None:
	if runtime_torch_version is None or _major_minor(BUILD_TORCH_VERSION) != _major_minor(runtime_torch_version):
		warnings.warn(
			(
				f"torch_ans was compiled with torch {BUILD_TORCH_VERSION} but the runtime "
				f"torch is {runtime_torch_version}. This may cause ABI/runtime incompatibilities. "
				"Attempting to continue: the package may fall back to dynamic compilation at runtime. "
				"If you want a strict check, set the environment variable TORCH_ANS_STRICT_CHECK=1."
			),
			RuntimeWarning,
		)
		# Optionally enforce strict check if requested
		if os.getenv("TORCH_ANS_STRICT_CHECK", "0") == "1":
			raise ImportError(
				f"torch_ans was compiled with torch {BUILD_TORCH_VERSION} but the runtime "
				f"torch is {runtime_torch_version}. Install a wheel compiled against your "
				"installed PyTorch or rebuild this package against your torch version. "
			)

# Import compiled extension (may raise if ABI mismatches)
# from . import _C

# rans_pmf_to_quantized_cdf = _C.rans_pmf_to_quantized_cdf
# rans_stream_to_byte_strings = _C.rans_stream_to_byte_strings
# rans_byte_strings_to_stream = _C.rans_byte_strings_to_stream
# rans64_init_stream = _C.rans64_init_stream
# rans64_push = _C.rans64_push
# rans64_pop = _C.rans64_pop
# rans32_init_stream = _C.rans32_init_stream
# rans32_push = _C.rans32_push
# rans32_pop = _C.rans32_pop
# rans32_16_init_stream = _C.rans32_16_init_stream
# rans32_16_push = _C.rans32_16_push
# rans32_16_pop = _C.rans32_16_pop

# Optionally import Python utilities
# from .utils import pmf_to_quantized_cdf_batched, inverse_quantized_cdf, TorchANSInterface


# def load_native(force_rebuild: bool = False, with_cuda: Optional[bool] = None, verbose: bool = False):
# 	"""Load or build the native extension `torch_ans._C` at runtime.

# 	This is a convenience wrapper around `torch_ans._dynamic_build.load_or_get_extension`.
# 	Calling this will attempt to import `torch_ans._C` and, if missing, compile it
# 	using `torch.utils.cpp_extension.load`.
# 	"""
# 	try:
# 		from ._dynamic_build import load_or_get_extension
# 	except Exception as e:
# 		raise ImportError("Dynamic build loader not available: " + str(e))
# 	return load_or_get_extension(force_rebuild=force_rebuild, with_cuda=with_cuda, verbose=verbose)

