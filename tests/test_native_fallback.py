"""Tests for the native-module fallbacks in torch_ans.utils.

Four situations are covered, none of which compiles anything:

1. the pre-built ``torch_ans._C`` cannot be imported at all (e.g. it was
   compiled against another torch) -> the lazy shim takes over;
2. ``TORCH_ANS_FORCE_RUNTIME_BUILD=1`` bypasses the pre-built extension;
3. CUDA coding is requested from a module that was built without CUDA support
   -> warn and switch to a locally compiled CUDA module, or explain what to
   install when that is not possible;
4. the same contract for the prebuilt libtorch-stable-ABI extension
   (``torch_ans/_stable_C.py``, migration section 5.2): it is preferred over
   compiling, and every way of not having a usable one - an old torch, the
   opt-out switch, a missing or unloadable file, a CUDA request without a CUDA
   library - has to return None so that the runtime build takes over instead of
   the call failing.
"""
import contextlib
import importlib.util
import os
import sys
import types
import warnings

import pytest

from torch_ans import _dynamic_build as db
from torch_ans import utils


def _fake_tensor(is_cuda):
    """Minimal stand-in for a tensor (only `.is_cuda` is inspected)."""
    return types.SimpleNamespace(is_cuda=is_cuda)


class _FakeInterface:
    """Just enough of TorchANSInterface for the CUDA switch."""

    def __init__(self, native, device=None):
        self._native = native
        self.device = device


@contextlib.contextmanager
def _no_warnings():
    """Fail if any warning is emitted inside the block."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


@pytest.fixture(autouse=True)
def _no_prebuilt_stable_extension(monkeypatch):
    """No prebuilt stable-ABI extension for the tests in this module.

    `_lazy_C.ensure_full`/`ensure_module` return the prebuilt module before
    `_compile` is reached (torch_ans/_stable_C.py, migration section 1.3), which
    would bypass the shim/runtime-build behaviour asserted below whenever one is
    installed or pointed at with TORCH_ANS_STABLE_LIB_DIR. `_stable_module` is the
    shim's own seam onto the loader, so the section-6 tests - which call
    `_stable_C.load_stable` directly to cover the loader itself - keep working.
    """
    from torch_ans import _lazy_C

    monkeypatch.setattr(_lazy_C, "_stable_module", lambda require_cuda=False: None)


# ---------------------------------------------------------------------------
# 1. the pre-built extension cannot be imported
# ---------------------------------------------------------------------------

def test_import_native_c_falls_back_to_lazy_shim(monkeypatch):
    """A pre-built extension that fails to import must not break the package."""
    import torch_ans

    # Otherwise the force flag short-circuits before the import is attempted.
    monkeypatch.delenv(utils.FORCE_RUNTIME_BUILD_ENV_VAR, raising=False)
    monkeypatch.delattr(torch_ans, "_C", raising=False)
    # `None` in sys.modules makes the import machinery raise ImportError, which
    # is what a torch-ABI mismatch of a pre-built extension looks like.
    monkeypatch.setitem(sys.modules, "torch_ans._C", None)

    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        native = utils._import_native_c()

    assert native is utils._lazy_C()
    assert callable(getattr(native, "ensure_module", None))
    assert callable(getattr(native, "ensure_full", None))


# ---------------------------------------------------------------------------
# 2. forcing the runtime build
# ---------------------------------------------------------------------------

def test_force_runtime_build_skips_the_prebuilt_extension(monkeypatch):
    monkeypatch.setenv(utils.FORCE_RUNTIME_BUILD_ENV_VAR, "1")
    assert utils._import_native_c() is utils._lazy_C()


def test_C_forwarding_layer_keeps_the_old_shim_api(monkeypatch):
    """`torch_ans/_C.py` must forward to `_lazy_C` (a compiled .so shadows it)."""
    lazy = utils._lazy_C()
    monkeypatch.setattr(lazy, "ensure_full",
                        lambda verbose=True: types.SimpleNamespace(rans64_push="OP"))

    path = os.path.join(os.path.dirname(utils.__file__), "_C.py")
    spec = importlib.util.spec_from_file_location("torch_ans._C_forward_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module.ensure_module is lazy.ensure_module
    assert module.ensure_full is lazy.ensure_full
    # module-level __getattr__ is what `from torch_ans._C import <op>` and
    # scripts/bench_version_matrix.py rely on
    assert module.rans64_push == "OP"


def _record_builds(monkeypatch):
    """Make both build entry points record instead of compiling."""
    lazy = utils._lazy_C()
    builds = []
    monkeypatch.setattr(lazy, "ensure_full",
                        lambda *args, **kwargs: builds.append("full"))
    monkeypatch.setattr(lazy, "ensure_module",
                        lambda *args, **kwargs: builds.append("profile"))
    return lazy, builds


def test_dunder_attribute_probes_never_build(monkeypatch):
    """Import-machinery probes must not compile the extension.

    `from ._lazy_C import ensure_full, ensure_module` (in torch_ans/_C.py) makes
    the import system ask `hasattr(_lazy_C, "__path__")`; answering that with
    `ensure_full()` would turn `import torch_ans.utils` into a full JIT build.
    """
    lazy, builds = _record_builds(monkeypatch)

    assert hasattr(lazy, "__path__") is False
    assert hasattr(lazy, "__all__") is False
    with pytest.raises(AttributeError):
        lazy.__getattr__("__path__")

    assert builds == []


def test_importing_the_C_forwarder_does_not_build(monkeypatch):
    """Importing `torch_ans/_C.py` itself must stay lazy (regression test)."""
    lazy, builds = _record_builds(monkeypatch)

    path = os.path.join(os.path.dirname(utils.__file__), "_C.py")
    spec = importlib.util.spec_from_file_location("torch_ans._C_no_build_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert builds == [], "importing torch_ans._C compiled the extension"
    assert module.ensure_module is lazy.ensure_module
    assert module.ensure_full is lazy.ensure_full


def test_operator_attribute_access_still_builds(monkeypatch):
    """`torch_ans._C.<op>` keeps compiling the full extension: documented behaviour."""
    lazy, builds = _record_builds(monkeypatch)
    monkeypatch.setattr(lazy, "ensure_full", lambda *args, **kwargs: (
        builds.append("full") or types.SimpleNamespace(rans64_push="OP")))

    assert lazy.rans64_push == "OP"
    assert builds == ["full"]


@pytest.mark.parametrize("value", ["0", "false", "", "no", "off"])
def test_force_runtime_build_is_opt_in(monkeypatch, value):
    monkeypatch.setenv(utils.FORCE_RUNTIME_BUILD_ENV_VAR, value)
    import torch_ans
    assert utils._import_native_c() is torch_ans._C


# ---------------------------------------------------------------------------
# 3. CUDA capability detection
# ---------------------------------------------------------------------------

def test_module_has_cuda_reads_the_compiled_flag():
    assert utils.module_has_cuda(types.SimpleNamespace(_torch_ans_with_cuda=True)) is True
    assert utils.module_has_cuda(types.SimpleNamespace(_torch_ans_with_cuda=False)) is False
    # the shim has no flag and no compiled module behind it yet: unknown
    assert utils.module_has_cuda(utils._lazy_C()) is None


def test_prebuilt_extension_without_flag_uses_build_metadata(monkeypatch):
    """Extensions predating the flag recorded it in torch_ans._torch_build_version."""
    monkeypatch.setitem(sys.modules, "torch_ans._torch_build_version",
                        types.SimpleNamespace(BUILD_WITH_CUDA=True))
    assert utils.module_has_cuda(types.SimpleNamespace()) is True


def test_module_has_cuda_is_unknown_without_flag_or_metadata(monkeypatch):
    """A module that cannot report its CUDA support is unknown, not CPU-only."""
    monkeypatch.setitem(sys.modules, "torch_ans._torch_build_version", None)
    assert utils.module_has_cuda(types.SimpleNamespace()) is None


def test_native_module_switches_to_runtime_build_for_cuda(monkeypatch):
    cpu_only = types.SimpleNamespace(_torch_ans_with_cuda=False)
    monkeypatch.setattr(utils, "_native_C", cpu_only)
    calls = []

    def fake_runtime(profile=None, incremental_compile=None, require_cuda=False):
        calls.append(require_cuda)
        return "runtime-module"

    monkeypatch.setattr(utils, "_runtime_native_module", fake_runtime)

    # without a CUDA requirement the pre-built extension is used as is
    assert utils._native_module() is cpu_only
    assert calls == []

    assert utils._native_module(require_cuda=True) == "runtime-module"
    assert calls == [True]


def test_requests_cuda_checks_device_and_tensors():
    assert utils._requests_cuda("cuda:0") is True
    assert utils._requests_cuda("cpu") is False
    assert utils._requests_cuda(None) is False
    assert utils._requests_cuda(None, _fake_tensor(is_cuda=True)) is True
    assert utils._requests_cuda(None, _fake_tensor(is_cuda=False)) is False
    # a device string that torch cannot parse must not raise
    assert utils._requests_cuda(object()) is False


# ---------------------------------------------------------------------------
# 4. the interface-level switch
# ---------------------------------------------------------------------------

def test_interface_switches_to_cuda_build_when_needed(monkeypatch):
    interface = _FakeInterface(types.SimpleNamespace(_torch_ans_with_cuda=False),
                               device="cuda")
    calls = []
    interface._resolve_native = lambda require_cuda=False: calls.append(require_cuda)
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    with pytest.warns(RuntimeWarning, match="CUDA coding was requested"):
        utils.TorchANSInterface._ensure_native_for(interface)

    assert calls == [True]


def test_interface_cuda_switch_is_skipped_when_not_needed(monkeypatch):
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)
    calls = []

    # a) the module already understands CUDA
    capable = _FakeInterface(types.SimpleNamespace(_torch_ans_with_cuda=True),
                             device="cuda")
    capable._resolve_native = lambda require_cuda=False: calls.append(require_cuda)
    # b) the coding runs on the CPU
    cpu = _FakeInterface(types.SimpleNamespace(_torch_ans_with_cuda=False),
                         device="cpu")
    cpu._resolve_native = lambda require_cuda=False: calls.append(require_cuda)

    with _no_warnings():
        utils.TorchANSInterface._ensure_native_for(capable)
        utils.TorchANSInterface._ensure_native_for(cpu)

    assert calls == []


def test_interface_switch_is_skipped_without_a_gpu(monkeypatch):
    """No GPU here: the C++ error describes the real problem, so stay put."""
    interface = _FakeInterface(types.SimpleNamespace(_torch_ans_with_cuda=False),
                               device="cuda")
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    calls = []
    interface._resolve_native = lambda require_cuda=False: calls.append(require_cuda)

    with _no_warnings():
        utils.TorchANSInterface._ensure_native_for(interface)

    assert calls == []


def test_interface_reports_actionable_error_when_cuda_build_fails(monkeypatch):
    interface = _FakeInterface(types.SimpleNamespace(_torch_ans_with_cuda=False),
                               device="cuda")
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: True)

    def boom(require_cuda=False):
        raise RuntimeError("nvcc not found")

    interface._resolve_native = boom

    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError) as excinfo:
            utils.TorchANSInterface._ensure_native_for(interface)

    message = str(excinfo.value)
    for expected in ("WITH_CUDA=1 pip install . --no-build-isolation",
                     utils.FORCE_RUNTIME_BUILD_ENV_VAR,
                     "cuda_build_state"):
        assert expected in message


# ---------------------------------------------------------------------------
# 5. the shim itself
# ---------------------------------------------------------------------------

def test_lazy_shim_does_not_reuse_a_cpu_only_module_for_cuda(monkeypatch):
    shim = utils._lazy_C()
    monkeypatch.setattr(shim, "_modules", {})
    profile = db.BuildProfile(family="rans64")
    built = []

    def fake_compile(module_name, defines, verbose=True, require_cuda=False):
        built.append((module_name, require_cuda))
        return types.SimpleNamespace(
            _torch_ans_with_cuda=require_cuda,
            **{name: name for name in db.profile_op_names(profile)})

    monkeypatch.setattr(shim, "_compile", fake_compile)

    cpu = shim.ensure_module(profile)
    assert utils.module_has_cuda(cpu) is False

    cuda = shim.ensure_module(profile, require_cuda=True)
    assert cuda is not cpu
    assert utils.module_has_cuda(cuda) is True

    # the CPU-only module is still served from the cache for CPU requests
    assert shim.ensure_module(profile) is cpu
    assert built == [(db.profile_module_name(profile), False),
                     (db.profile_module_name(profile) + "_cuda", True)]


def test_lazy_shim_rejects_a_cpu_only_result_when_cuda_is_required(monkeypatch):
    shim = utils._lazy_C()

    def fake_build_extension(module_name, with_cuda=None, verbose=False, defines=None):
        # what build_extension returns after a remembered CUDA failure
        return types.SimpleNamespace(_torch_ans_with_cuda=False)

    monkeypatch.setattr(db, "build_extension", fake_build_extension)

    with pytest.raises(RuntimeError, match="CPU-only"):
        shim._compile("torch_ans_cuda_test", None, verbose=False, require_cuda=True)


def test_lazy_shim_does_not_swallow_cuda_build_errors_when_required(monkeypatch):
    shim = utils._lazy_C()
    attempts = []

    def fake_build_extension(module_name, with_cuda=None, verbose=False, defines=None):
        attempts.append(with_cuda)
        raise RuntimeError("nvcc not found")

    monkeypatch.setattr(db, "build_extension", fake_build_extension)

    with pytest.raises(RuntimeError, match="nvcc not found"):
        shim._compile("torch_ans_cuda_test", None, verbose=False, require_cuda=True)
    # the CPU-only retry must not happen when CUDA was explicitly required
    assert attempts == [True]


def test_interface_resolve_native_requests_a_cuda_module(monkeypatch):
    """`_resolve_native(require_cuda=True)` forwards the requirement and rebinds."""
    shim = utils._lazy_C()
    monkeypatch.setattr(utils, "_native_C", shim)
    monkeypatch.setattr(shim, "_modules", {})
    profile_names = db.profile_op_names(db.BuildProfile(family="rans64"))
    seen = []

    def fake_compile(module_name, defines, verbose=True, require_cuda=False):
        seen.append(require_cuda)
        return types.SimpleNamespace(
            _torch_ans_with_cuda=require_cuda,
            **{name: name for name in profile_names})

    monkeypatch.setattr(shim, "_compile", fake_compile)

    interface = _FakeInterface(None)
    interface.num_interleaves = 1
    interface.alias_sampling = False
    interface._configured_invcdf_decode = False
    interface._impl_key = ("rans64", 1)
    interface._incremental_compile = True
    interface._inverse_cdf_auto = False
    interface._build_profile = lambda: db.BuildProfile(family="rans64")

    utils.TorchANSInterface._resolve_native(interface, require_cuda=True)

    assert seen == [True]
    assert utils.module_has_cuda(interface._native) is True
    # the operators were rebound to the new module
    assert interface.ans_encode_func == "rans64_push"


# ---------------------------------------------------------------------------
# 6. the prebuilt libtorch-stable-ABI extension
# ---------------------------------------------------------------------------

@pytest.fixture
def stable_loader(monkeypatch, tmp_path):
    """`_stable_C` with an isolated search path and a fresh answer cache.

    The torch version is forced up: the gate itself is tested separately, and the
    rest of these tests must not depend on the interpreter's torch being >= 2.10
    (the prebuilt extension is usually exercised from a source checkout, which can
    run on an older torch).
    """
    from torch_ans import _stable_C

    monkeypatch.setattr(_stable_C, "_loaded", {})
    monkeypatch.setattr(_stable_C, "_search_dirs", lambda: [tmp_path])
    monkeypatch.setattr(_stable_C, "_torch_version_tuple", lambda: (3, 0, 0))
    monkeypatch.delenv(_stable_C.DISABLE_ENV_VAR, raising=False)
    return _stable_C, tmp_path


def _fake_load_library(monkeypatch, recorder=None, fail_for=()):
    """Replace `torch.ops.load_library` (which would register real operators)."""
    import torch

    def load(path):
        if recorder is not None:
            recorder.append(path)
        for marker in fail_for:
            if marker in path:
                raise OSError(f"{marker}: cannot load, wrong ABI")

    monkeypatch.setattr(torch.ops, "load_library", load)


def test_stable_loader_is_disabled_by_the_opt_out_switch(stable_loader, monkeypatch):
    stable, directory = stable_loader
    (directory / "_C_cpu.so").write_bytes(b"")
    monkeypatch.setenv(stable.DISABLE_ENV_VAR, "1")

    assert stable.load_stable() is None


def test_stable_loader_requires_torch_2_10(stable_loader, monkeypatch):
    """Below the target ABI the prebuilt .so must not even be tried."""
    stable, directory = stable_loader
    (directory / "_C_cpu.so").write_bytes(b"")
    monkeypatch.setattr(stable, "_torch_version_tuple", lambda: (2, 9, 1))

    assert stable.load_stable() is None


def test_stable_loader_without_the_extension_falls_back(stable_loader):
    stable, _ = stable_loader

    assert stable.load_stable() is None, "no _C_cpu*.so -> the runtime build"


def test_stable_loader_prefers_and_caches_the_cpu_library(stable_loader, monkeypatch):
    stable, directory = stable_loader
    cpu = directory / "_C_cpu.so"
    cpu.write_bytes(b"")
    loaded = []
    _fake_load_library(monkeypatch, loaded)

    module = stable.load_stable()
    assert module is not None
    assert loaded == [str(cpu)]
    assert module._torch_ans_with_cuda is False
    # the answer is cached: loading a library registers process-wide operators
    assert stable.load_stable() is module


def test_stable_loader_refuses_a_cuda_request_without_the_cuda_library(
        stable_loader, monkeypatch):
    """A CUDA request must reach the runtime build, not a CPU-only library."""
    stable, directory = stable_loader
    (directory / "_C_cpu.so").write_bytes(b"")
    loaded = []
    _fake_load_library(monkeypatch, loaded)

    assert stable.load_stable(require_cuda=True) is None
    assert loaded == [], "the CPU library is not even loaded for a CUDA request"
    # ... while a plain CPU request is served, and reports the truth
    assert stable.load_stable()._torch_ans_with_cuda is False


def test_stable_loader_warns_and_falls_back_when_loading_fails(
        stable_loader, monkeypatch):
    stable, directory = stable_loader
    (directory / "_C_cpu.so").write_bytes(b"")
    _fake_load_library(monkeypatch, fail_for=("_C_cpu.so",))

    with pytest.warns(RuntimeWarning, match="could not be loaded"):
        assert stable.load_stable() is None


def test_stable_loader_warns_when_only_the_cuda_library_is_unusable(
        stable_loader, monkeypatch):
    """A broken CUDA library must not break CPU coding."""
    stable, directory = stable_loader
    (directory / "_C_cpu.so").write_bytes(b"")
    (directory / "_C_cuda.so").write_bytes(b"")
    _fake_load_library(monkeypatch, fail_for=("_C_cuda.so",))

    with pytest.warns(RuntimeWarning, match="CUDA extension"):
        assert stable.load_stable(require_cuda=True) is None

    # a CPU request still succeeds, with the same warning and without CUDA
    monkeypatch.setattr(stable, "_loaded", {})
    with pytest.warns(RuntimeWarning, match="CUDA extension"):
        module = stable.load_stable()
    assert module is not None and module._torch_ans_with_cuda is False


def test_stable_loader_exposes_the_python_bytes_operators():
    """The two `py::bytes` operators live in Python (migration section 1.5)."""
    from torch_ans import _stable_C

    module = _stable_C._StableModule(with_cuda=False, loaded_paths=[])
    for name in _stable_C.PYTHON_OPS:
        assert getattr(module, name) is getattr(_stable_C, name)
        assert name in dir(module)
    # anything else has to name what was loaded instead of failing obscurely
    # (a deliberately impossible name: a real operator may be registered by
    # another test in the same session)
    with pytest.raises(AttributeError, match="does not provide"):
        module.this_operator_does_not_exist


def test_lazy_shim_prefers_the_prebuilt_extension(monkeypatch):
    """No compilation when the prebuilt extension is there (migration 1.3).

    `_stable_module` is the shim's seam onto `torch_ans._stable_C`: stubbing it
    here (rather than the loader) keeps this test independent of the module-wide
    switch-off in the autouse fixture above.
    """
    from torch_ans import _lazy_C

    sentinel = types.SimpleNamespace(_torch_ans_with_cuda=True)
    monkeypatch.setattr(_lazy_C, "_stable_module",
                        lambda require_cuda=False: sentinel)
    monkeypatch.setattr(_lazy_C, "_modules", {})
    monkeypatch.setattr(db, "build_extension",
                        lambda *args, **kwargs: pytest.fail(
                            "the prebuilt extension was ignored"))

    assert _lazy_C.ensure_full() is sentinel
    assert _lazy_C.ensure_module(db.BuildProfile(family="rans64")) is sentinel
    # and a CUDA requirement is forwarded to the loader
    seen = []
    monkeypatch.setattr(_lazy_C, "_stable_module",
                        lambda require_cuda=False: seen.append(require_cuda) or sentinel)
    assert _lazy_C.ensure_module(db.BuildProfile(family="rans64"),
                                 require_cuda=True) is sentinel
    assert seen == [True]
