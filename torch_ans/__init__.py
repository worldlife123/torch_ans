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

import torch as _torch

def _major_minor(v: str) -> str:
	v = str(v).split('+', 1)[0]
	parts = v.split('.')
	return '.'.join(parts[:2]) if len(parts) >= 2 else v

if BUILD_TORCH_VERSION is not None:
	if _major_minor(BUILD_TORCH_VERSION) != _major_minor(_torch.__version__):
		raise ImportError(
			f"torch_ans was compiled with torch {BUILD_TORCH_VERSION} but the runtime "
			f"torch is {_torch.__version__}. Install a wheel compiled against your "
			"installed PyTorch or rebuild this package against your torch version. "
			"For local development, you can `pip install . --no-build-isolation` after "
			"pre-installing the desired torch in your environment."
		)

# Import compiled extension (may raise if ABI mismatches)
from ._C import rans_pmf_to_quantized_cdf
from ._C import rans_stream_to_byte_strings, rans_byte_strings_to_stream
from ._C import rans64_init_stream, rans64_push, rans64_pop
from ._C import rans32_init_stream, rans32_push, rans32_pop
from ._C import rans32_16_init_stream, rans32_16_push, rans32_16_pop

# Optionally import Python utilities
# from .utils import pmf_to_quantized_cdf_batched, inverse_quantized_cdf, TorchANSInterface
