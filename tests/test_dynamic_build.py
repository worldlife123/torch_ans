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


def test_incremental_defines_reach_cxx_and_nvcc(monkeypatch):
    # The TORCH_ANS_WITH_* gates of an incremental build must be seen by the
    # CUDA sources too (rans_cuda.cu uses the same gates), and a full build must
    # pass none of them so every gate in rans_build_config.hpp stays at 1.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    captured = {}
    _install_fake_load(monkeypatch, captured)
    monkeypatch.setattr(db, "_detect_cuda_torch", lambda: True)
    monkeypatch.setattr(db, "_record_build_metadata", lambda pkg_dir, with_cuda: None)
    monkeypatch.setattr(db, "_cuda_state_path", lambda name: None)

    profile = db.BuildProfile(family="rans32_16", interleave=32, invcdf=True)
    db.build_extension(module_name=db.profile_module_name(profile), with_cuda=True,
                       verbose=False, defines=db.profile_defines(profile))

    # `_define_flags` spells macros the way the local toolchain expects: `/D` for
    # MSVC, `-D` elsewhere. Take the prefix from it so the two cannot drift.
    prefix = db._define_flags({"PROBE": 1})[0].split("PROBE")[0]
    for flag in ("TORCH_ANS_INCREMENTAL_BUILD=1", "TORCH_ANS_WITH_RANS32_16=1",
                 "TORCH_ANS_WITH_INTERLEAVE_32=1", "TORCH_ANS_WITH_INVCDF=1"):
        assert prefix + flag in captured["extra_cflags"], flag
        assert prefix + flag in captured["extra_cuda_cflags"], flag
    assert prefix + "TORCH_ANS_WITH_RANS64=1" not in captured["extra_cflags"]
    assert any(s.endswith(".cu") for s in captured["sources"])

    db.build_extension(module_name=db.FULL_MODULE_NAME, with_cuda=False, verbose=False,
                       defines=db.profile_defines(None))
    assert not any(f.startswith(prefix + "TORCH_ANS_") for f in captured["extra_cflags"])
    assert not any(s.endswith(".cu") for s in captured["sources"])


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


class _ProbeResult:
    """Stand-in for the child process result of the OpenMP probe."""

    def __init__(self, returncode, stdout=b""):
        self.returncode = returncode
        self.stdout = stdout


def _tmp_openmp_state(monkeypatch, db, tmp_path):
    """Point the OpenMP state file at tmp_path so tests never touch the cache."""
    path = str(tmp_path / "openmp_build_state")
    monkeypatch.setattr(db, "_openmp_state_path", lambda: path)
    return path


def _darwin_variants(monkeypatch, db, broken):
    monkeypatch.setattr(db, "_libomp_prefix", lambda: "/opt/homebrew/opt/libomp")
    monkeypatch.setattr(db, "_openmp_broken", lambda: broken)
    return db._darwin_cpu_flag_variants()


def test_macos_openmp_variant_is_dropped_once_recorded_broken(monkeypatch):
    # The OpenMP variant builds and imports fine and still aborts at use time, so
    # the recorded outcome - not a build failure - is what has to drop it.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    with_omp = _darwin_variants(monkeypatch, db, broken=False)
    assert any("-Xpreprocessor" in cflags for cflags, _ in with_omp)
    assert any(ldflags for _, ldflags in with_omp)

    without = _darwin_variants(monkeypatch, db, broken=True)
    assert not any("-Xpreprocessor" in cflags for cflags, _ in without)
    assert without, "the variant without a second OpenMP runtime must stay"


def test_openmp_probe_rejects_a_crashing_build(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    monkeypatch.delenv(db._OMP_PROBE_ENV, raising=False)
    state = _tmp_openmp_state(monkeypatch, db, tmp_path)
    monkeypatch.setattr(db, "_run_probe_child", lambda code, env: _ProbeResult(-6, b"OMP: Error #15\n"))

    assert db._openmp_variant_is_usable("torch_ans_C", None) is False
    # the outcome is recorded, and it is the variant-dropping one
    assert open(state).read().startswith("broken")
    assert db._openmp_broken() is True


def test_openmp_probe_keeps_a_working_build_and_does_not_repeat_it(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    monkeypatch.delenv(db._OMP_PROBE_ENV, raising=False)
    _tmp_openmp_state(monkeypatch, db, tmp_path)
    spawned = []

    def fake_child(code, env):
        spawned.append(code)
        return _ProbeResult(0, b"ok\n")

    monkeypatch.setattr(db, "_run_probe_child", fake_child)

    assert db._openmp_variant_is_usable("torch_ans_C", None) is True
    assert len(spawned) == 1
    # the probe spawns a process, so a later build must not pay it again
    assert db._openmp_variant_is_usable("torch_ans_C", None) is True
    assert len(spawned) == 1
    assert db._openmp_broken() is False


def test_openmp_probe_reruns_after_a_torch_version_change(monkeypatch, tmp_path):
    # The crash it looks for depends on the torch version, so a recorded "ok"
    # from another torch release must not be trusted.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    state = _tmp_openmp_state(monkeypatch, db, tmp_path)
    with open(state, "w") as f:
        f.write("ok 2.13.0\n")
    monkeypatch.setattr(db, "_torch_version", lambda: "2.14.0")
    monkeypatch.delenv(db._OMP_PROBE_ENV, raising=False)
    monkeypatch.setattr(db, "_run_probe_child", lambda code, env: _ProbeResult(0, b"ok\n"))

    assert db._openmp_probe_is_redundant() is False
    assert db._openmp_variant_is_usable("torch_ans_C", None) is True
    assert open(state).read().startswith("ok 2.14.0")


def test_openmp_probe_never_recurses(monkeypatch, tmp_path):
    # The child builds again (cache hit); that build must not probe again.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    _tmp_openmp_state(monkeypatch, db, tmp_path)
    monkeypatch.setenv(db._OMP_PROBE_ENV, "1")

    def boom(code, env):
        raise AssertionError("the probe spawned a child from inside the child")

    monkeypatch.setattr(db, "_run_probe_child", boom)
    assert db._openmp_variant_is_usable("torch_ans_C", None) is True


def test_openmp_probe_keeps_the_build_when_it_cannot_run(monkeypatch, tmp_path):
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    _tmp_openmp_state(monkeypatch, db, tmp_path)
    monkeypatch.delenv(db._OMP_PROBE_ENV, raising=False)

    def boom(code, env):
        raise OSError("no interpreter")

    monkeypatch.setattr(db, "_run_probe_child", boom)
    assert db._openmp_variant_is_usable("torch_ans_C", None) is True


def test_macos_openmp_build_falls_back_and_is_remembered(monkeypatch, tmp_path):
    # End to end through build_extension: the OpenMP variant does not survive its
    # probe (a fake child that "dies"), so the build must not load it at all,
    # must use the variant without a second runtime instead, and record that -
    # the next build must not even offer the OpenMP variant.
    pytest.importorskip("torch")
    import torch_ans._dynamic_build as db

    captured = {}
    _install_fake_load(monkeypatch, captured)
    state = _tmp_openmp_state(monkeypatch, db, tmp_path)
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(db, "_libomp_prefix", lambda: "/opt/homebrew/opt/libomp")
    monkeypatch.delenv(db._OMP_PROBE_ENV, raising=False)
    monkeypatch.setattr(db, "_run_probe_child", lambda code, env: _ProbeResult(-6, b"OMP: Error #15\n"))
    monkeypatch.setattr(db, "_record_build_metadata", lambda pkg_dir, with_cuda: None)

    module = db.build_extension(module_name="torch_ans_C_test", with_cuda=False, verbose=False)

    assert module is not None
    # the probe builds the OpenMP module in the child, so the parent builds and
    # loads only the fallback (a module the parent had already loaded could not
    # be replaced: both variants share one module name and dlopen keys on paths)
    assert captured["calls"] == 1
    assert "-Xpreprocessor" not in captured["extra_cflags"]
    assert open(state).read().startswith("broken")
    assert not any("-Xpreprocessor" in cflags for cflags, _ in db._darwin_cpu_flag_variants())

