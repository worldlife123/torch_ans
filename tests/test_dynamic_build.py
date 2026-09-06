import os
import sys
import importlib
import shutil
import pytest


RUN_FLAG = os.getenv("RUN_DYNAMIC_BUILD_TEST", "0")


def _have_compiler():
    # Check for common C/C++ compilers
    return any(shutil.which(c) for c in ("g++", "clang++", "cc", "gcc"))


@pytest.mark.skipif(RUN_FLAG != "1", reason="Dynamic build test disabled (set RUN_DYNAMIC_BUILD_TEST=1 to enable)")
def test_dynamic_build_cpu():
    # Require torch's cpp_extension to be available
    try:
        import torch
        from torch.utils import cpp_extension
    except Exception:
        pytest.skip("torch or torch.utils.cpp_extension not available")

    if not _have_compiler():
        pytest.skip("No C/C++ compiler found on PATH")

    # Ensure we attempt a CPU-only build
    if "torch_ans._C" in sys.modules:
        del sys.modules["torch_ans._C"]


    # Force a rebuild/load (loader may reuse cache)
    import torch_ans._C as mod
    # mod = torch_ans.load_native(with_cuda=False, verbose=True)
    assert mod is not None
    assert hasattr(mod, "rans64_init_stream")
    assert callable(getattr(mod, "rans64_init_stream"))
