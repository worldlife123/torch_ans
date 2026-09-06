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


def _install_fake_load(monkeypatch, captured):
    """Replace torch's JIT `load` with a recorder so these tests never compile."""
    import torch.utils.cpp_extension as cpp_ext

    def fake_load(name, sources, extra_cflags, extra_cuda_cflags, extra_ldflags, verbose):
        captured["name"] = name
        captured["sources"] = list(sources)
        captured["extra_cflags"] = list(extra_cflags)
        captured["extra_cuda_cflags"] = list(extra_cuda_cflags)
        captured["extra_ldflags"] = list(extra_ldflags or [])
        captured["calls"] = captured.get("calls", 0) + 1
        return object()

    monkeypatch.setattr(cpp_ext, "load", fake_load)


def test_cpu_state_shortcut_builds_cpu_only(monkeypatch, tmp_path):
    # Regression: the cuda_build_state=="cpu" shortcut must build cpp-only
    # WITHOUT -DWITH_CUDA. With the flag the .so referenced the CUDA
    # implementations (defined only in the .cu sources), failed to load, and
    # forced a full CPU rebuild on every fresh import.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    captured = {}
    recorded = []
    _install_fake_load(monkeypatch, captured)
    monkeypatch.setattr(db, "_detect_cuda_torch", lambda: True)
    monkeypatch.setattr(db, "_record_build_metadata", lambda pkg_dir, with_cuda: recorded.append(with_cuda))
    state_file = tmp_path / "cuda_build_state"
    state_file.write_text("cpu\n")
    monkeypatch.setattr(db, "_cuda_state_path", lambda name: str(state_file))

    module = db.build_extension(module_name="torch_ans_C_test", with_cuda=True, verbose=False)

    assert captured["calls"] == 1
    assert captured["sources"]
    assert not any(s.endswith(".cu") for s in captured["sources"])
    assert "-DWITH_CUDA" not in captured["extra_cflags"]
    assert recorded == [False]
    assert module is not None


def test_cuda_build_defines_with_cuda_for_nvcc(monkeypatch):
    # The CUDA JIT build must pass -DWITH_CUDA to nvcc explicitly: torch
    # forwards extra_cflags to the C++ compiler only and does not define
    # WITH_CUDA itself (unlike WITH_HIP), and rans_cuda.cu compiles empty
    # without it, leaving the .so with undefined symbols.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    captured = {}
    recorded = []
    _install_fake_load(monkeypatch, captured)
    monkeypatch.setattr(db, "_detect_cuda_torch", lambda: True)
    monkeypatch.setattr(db, "_record_build_metadata", lambda pkg_dir, with_cuda: recorded.append(with_cuda))
    monkeypatch.setattr(db, "_cuda_state_path", lambda name: None)

    module = db.build_extension(module_name="torch_ans_C_test", with_cuda=True, verbose=False)

    assert captured["calls"] == 1
    assert any(s.endswith(".cu") for s in captured["sources"])
    assert "-DWITH_CUDA" in captured["extra_cflags"]
    assert "-DWITH_CUDA" in captured["extra_cuda_cflags"]
    assert recorded == [True]
    assert module is not None


def test_cpu_build_without_cuda_has_no_cuda_flags(monkeypatch):
    # with_cuda=False builds cpp-only and must never see -DWITH_CUDA.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    captured = {}
    recorded = []
    _install_fake_load(monkeypatch, captured)
    monkeypatch.setattr(db, "_record_build_metadata", lambda pkg_dir, with_cuda: recorded.append(with_cuda))

    module = db.build_extension(module_name="torch_ans_C_test", with_cuda=False, verbose=False)

    assert captured["calls"] == 1
    assert not any(s.endswith(".cu") for s in captured["sources"])
    assert "-DWITH_CUDA" not in captured["extra_cflags"]
    assert recorded == [False]
    assert module is not None
