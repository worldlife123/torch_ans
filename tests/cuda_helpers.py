"""Shared CUDA-availability helpers for the test suite.

``torch.cuda.is_available()`` is not enough to decide whether a test can
exercise the CUDA kernels: they are compiled in only when the extension was
built with ``WITH_CUDA=1``, so a machine with a GPU can still run a CPU-only
build (the default of ``pip install .``, or a driver whose CUDA version has no
matching nvcc). Tests that only checked ``is_available()`` failed there instead
of skipping.
"""

import torch

from torch_ans import _C as _native_module
from torch_ans.utils import module_has_cuda

#: error markers that mean "this environment cannot code CUDA tensors"
CUDA_UNAVAILABLE_MARKERS = (
    "not compiled with GPU support",       # C++ AT_ERROR raised from rans.hpp
    "CUDA coding was requested",           # RuntimeWarning/Error from utils.py
    "could not be built for the local torch",
)


def cuda_coding_skip_reason():
    """Why CUDA coding cannot be exercised, or None when it can.

    A module that cannot report its capability - the lazy shim, which compiles
    the extension on demand - is given the benefit of the doubt and the test is
    allowed to run.
    """
    if not torch.cuda.is_available():
        return "CUDA is not available"
    if module_has_cuda(_native_module) is False:
        return ("the torch_ans extension was compiled without CUDA support "
                "(build it with WITH_CUDA=1 to run this test)")
    return None


def is_cuda_unavailable_error(error):
    """Whether `error` reports an environment without CUDA coding support."""
    message = str(error)
    return any(marker in message for marker in CUDA_UNAVAILABLE_MARKERS)


def require_cuda_coding(case):
    """``case.skipTest(...)`` unless the CUDA kernels can actually be run.

    Works with a ``unittest.TestCase`` (pass ``self``) and explains which of the
    two situations caused the skip.
    """
    reason = cuda_coding_skip_reason()
    if reason is not None:
        case.skipTest(reason)
