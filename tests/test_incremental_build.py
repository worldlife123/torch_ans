"""Unit tests for incremental (per-interface) compilation.

Nothing here compiles a native extension: the profile bookkeeping is checked
against the C++ gate header, the shim's module resolution is driven through a
fake ``_compile``, and ``TorchANSInterface`` is pointed at a fake
``torch_ans._C`` so the :class:`BuildProfile` it selects can be observed.

The end-to-end behaviour (one module per profile, only the requested operators
exported, streams round-tripping) is covered by the build itself; see
``tests/test_dynamic_build.py`` for the guarded real-build test.
"""
import importlib.util
import os
import types

import pytest

from torch_ans import _dynamic_build as db
from torch_ans import utils


def _load_shim():
    """Import `torch_ans/_lazy_C.py` even when a pre-built .so exists.

    The shim lives in `_lazy_C.py` because a compiled `torch_ans/_C*.so` shadows
    `_C.py` completely (see torch_ans/_lazy_C.py).
    """
    path = os.path.join(os.path.dirname(utils.__file__), "_lazy_C.py")
    spec = importlib.util.spec_from_file_location("torch_ans._lazy_C_shim_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeShim:
    """Stand-in for the `torch_ans._C` shim: records and "builds" profiles."""

    def __init__(self):
        self.requests = []
        self.cuda_requests = []

    def ensure_module(self, profile=None, verbose=True, require_cuda=False):
        self.requests.append(profile)
        self.cuda_requests.append(require_cuda)
        # `_torch_ans_with_cuda` mirrors the flag the compiled extension exposes
        # (see torch_ans/lib.cpp); without it the shim reports "unknown" and
        # utils.module_has_cuda() would treat the module as CUDA-incapable.
        return types.SimpleNamespace(
            _torch_ans_with_cuda=bool(require_cuda),
            **{n: n for n in db.profile_op_names(profile)})


@pytest.fixture
def fake_native(monkeypatch):
    shim = FakeShim()
    monkeypatch.setattr(utils, "_native_C", shim)
    return shim


def _all_profiles():
    profiles = [db.BuildProfile()]
    for family, interleaves in db.SUPPORTED_INTERLEAVES.items():
        for interleave in interleaves:
            for alias in (False, True):
                if alias and interleave not in db.ALIAS_INTERLEAVES[family]:
                    continue
                for invcdf in (False, True):
                    if alias and invcdf:
                        continue  # alias sampling replaces the inverse-CDF table
                    profiles.append(db.BuildProfile(
                        family=family, interleave=interleave, alias=alias, invcdf=invcdf))
    return profiles


def test_profiles_have_distinct_module_names():
    names = [db.profile_module_name(p) for p in _all_profiles()]
    assert len(names) == len(set(names))
    assert db.profile_module_name(None) == db.FULL_MODULE_NAME
    assert db.profile_module_name(None) not in names


def test_profile_gates_exist_in_the_cpp_header():
    # The Python side and rans_build_config.hpp must not drift apart: a gate
    # the header does not know about would silently disable nothing.
    header = os.path.join(os.path.dirname(utils.__file__), "rans_build_config.hpp")
    with open(header) as handle:
        text = handle.read()
    used = set()
    for profile in _all_profiles():
        used.update(db.profile_defines(profile))
    assert used, "no gates at all?"
    for macro in used:
        if macro == "TORCH_ANS_INCREMENTAL_BUILD":
            continue
        assert macro in text, macro


def test_full_profile_is_a_superset_of_every_profile():
    full = set(db.profile_op_names(None))
    for profile in _all_profiles():
        missing = set(db.profile_op_names(profile)) - full
        assert not missing, (profile, missing)


def test_profile_defines_enable_only_the_requested_gates():
    profile = db.BuildProfile(family="rans32_16", interleave=4, invcdf=True)
    defines = db.profile_defines(profile)
    assert defines["TORCH_ANS_INCREMENTAL_BUILD"] == 1
    assert defines["TORCH_ANS_WITH_RANS32_16"] == 1
    assert defines["TORCH_ANS_WITH_INTERLEAVE_4"] == 1
    assert defines["TORCH_ANS_WITH_INVCDF"] == 1
    assert "TORCH_ANS_WITH_ALIAS" not in defines
    assert "TORCH_ANS_WITH_RANS64" not in defines
    # interleave 1 is the always-compiled core, so it needs no gate
    assert "TORCH_ANS_WITH_INTERLEAVE_1" not in db.profile_defines(db.BuildProfile(family="rans64"))
    # the full build is not incremental and needs no gates at all
    assert db.profile_defines(None) == {}


def test_default_interface_requests_the_impl_subset(fake_native):
    coder = utils.TorchANSInterface(impl="rans64")
    assert fake_native.requests == [db.BuildProfile(family="rans64", interleave=1)]
    assert coder.ans_encode_func == "rans64_push"
    assert coder.ans_decode_func == "rans64_pop"
    assert coder._binary_decode_func == "rans64_pop"


def test_interleave_and_inverse_cdf_flags_reach_the_profile(fake_native):
    coder = utils.TorchANSInterface(
        impl="rans32_16", num_interleaves=32, inverse_cdf_precision="auto")
    assert fake_native.requests == [
        db.BuildProfile(family="rans32_16", interleave=32, invcdf=True)]
    # "auto" may still fall back to the plain pop at init_params time
    assert coder._binary_decode_func == "rans32_16_i32_pop"
    assert coder.ans_decode_func == "rans32_16_i32_invcdf_pop"


def test_alias_sampling_replaces_the_inverse_cdf_profile(fake_native):
    coder = utils.TorchANSInterface(
        impl="rans64", num_interleaves=4, inverse_cdf_precision=8, alias_sampling=True)
    assert fake_native.requests == [
        db.BuildProfile(family="rans64", interleave=4, alias=True)]
    assert coder.ans_encode_func == "rans64_alias_i4_push"
    assert coder.ans_decode_func == "rans64_alias_i4_pop"
    assert coder.impl_use_inverse_cdf is False


def test_incremental_compile_argument_and_env_var(fake_native, monkeypatch):
    monkeypatch.setenv(db.INCREMENTAL_ENV_VAR, "0")
    # the environment disables it ...
    utils.TorchANSInterface(impl="rans64")
    assert fake_native.requests == [None]
    # ... and the argument wins over the environment in both directions
    utils.TorchANSInterface(impl="rans64", incremental_compile=True)
    assert fake_native.requests[-1] == db.BuildProfile(family="rans64")
    monkeypatch.delenv(db.INCREMENTAL_ENV_VAR)
    utils.TorchANSInterface(impl="rans64", incremental_compile=False)
    assert fake_native.requests[-1] is None


def test_unsupported_combination_fails_before_building(fake_native):
    with pytest.raises(NotImplementedError):
        utils.TorchANSInterface(impl="rans64", num_interleaves=32)
    with pytest.raises(NotImplementedError):
        utils.TorchANSInterface(impl="rans32_16", num_interleaves=32, alias_sampling=True)
    with pytest.raises(ValueError):
        utils.TorchANSInterface(impl="rans64", inverse_cdf_precision=99)
    assert fake_native.requests == []


def test_legacy_operator_attributes_still_resolve(fake_native):
    # names that used to be imported at module import time
    assert utils.rans_pmf_to_quantized_cdf == "rans_pmf_to_quantized_cdf"
    assert fake_native.requests[-1] == db.BuildProfile()  # common-only build
    assert utils.rans64_push == "rans64_push"
    assert fake_native.requests[-1] is None  # impl ops need the full build
    with pytest.raises(AttributeError):
        utils.definitely_not_an_operator
    assert fake_native.requests[-1] is None


def test_shim_reuses_an_equivalent_or_larger_module(monkeypatch):
    shim = _load_shim()
    built = []

    def fake_compile(module_name, defines, verbose=True, require_cuda=False):
        built.append(module_name)
        for profile in _all_profiles():
            if db.profile_module_name(profile) == module_name:
                ops = db.profile_op_names(profile)
                break
        else:
            ops = db.profile_op_names(None)  # the full build
        return types.SimpleNamespace(**{n: n for n in ops})

    monkeypatch.setattr(shim, "_compile", fake_compile)
    small = db.BuildProfile(family="rans64")
    larger = db.BuildProfile(family="rans64", interleave=4, invcdf=True)

    first = shim.ensure_module(small)
    assert built == [db.profile_module_name(small)]
    # a different configuration needs its own module ...
    second = shim.ensure_module(larger)
    assert built[-1] == db.profile_module_name(larger)
    assert second is not first
    # ... and an already requested profile is served from the cache
    assert shim.ensure_module(small) is first
    assert len(built) == 2

    # the full build satisfies every profile without building anything new
    full = shim.ensure_full()
    assert built[-1] == db.FULL_MODULE_NAME
    assert shim.ensure_module(db.BuildProfile(family="rans32")) is full
    # the common-only profile is satisfied by any module that has the shared
    # operators (the rans64 one here), so no further build happens either
    common_only = shim.ensure_module(db.BuildProfile())
    assert all(hasattr(common_only, name) for name in db.COMMON_OPS)
    assert len(built) == 3
