"""Stage-0 guard for the `ta::` compatibility layer (STABLE_ABI_MIGRATION.md).

`torch_ans/ta/` is not included by any source yet, so without these tests the
four headers would only ever be compile-checked once stages 1/2/6 start
replacing call sites - which is exactly when a missing overload or a
`__host__`-only view method is most expensive. Two kinds of checks:

* static ones (no compiler involved): the four headers exist, `torch_api.hpp`
  really dispatches on `TORCH_ANS_STABLE_ABI`, and the stable backend stays free
  of ATen/c10 symbols - the property that makes a prebuilt `.so` loadable on any
  `torch >= 2.10`;
* a real compile of both backends, driven by `scripts/check_ta_headers.py`: the
  two `g++` combinations always, the two `nvcc` ones when a usable toolkit is
  installed (skipped, not failed, otherwise).

The compile checks are slower than the rest of the suite (ATen's headers alone
take seconds to parse), and they are the only thing that proves the layer
compiles in both build modes, so they are worth the time.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TA_DIR = ROOT / "torch_ans" / "ta"
CHECKER = ROOT / "scripts" / "check_ta_headers.py"

#: The four files STABLE_ABI_MIGRATION.md stage 0 asks for.
HEADERS = ("torch_api.hpp", "torch_api_aten.hpp", "torch_api_stable.hpp",
           "view.hpp")


def _load_checker():
    """Import scripts/check_ta_headers.py to reuse its nvcc discovery."""
    spec = importlib.util.spec_from_file_location("_ta_header_checker", CHECKER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_checker(*args):
    return subprocess.run([sys.executable, str(CHECKER), *args],
                          capture_output=True, text=True, cwd=ROOT)


def _strip_line_comments(source):
    """Drop `//` comments: the headers discuss `at::`/`c10::` in prose."""
    return "\n".join(line.split("//", 1)[0] for line in source.splitlines())


def test_the_layer_has_its_four_headers():
    for name in HEADERS:
        assert (TA_DIR / name).is_file(), f"torch_ans/ta/{name} is missing"


def test_torch_api_dispatches_on_the_stable_abi_macro():
    source = _strip_line_comments(
        (TA_DIR / "torch_api.hpp").read_text(encoding="utf-8"))
    stable_include = source.index('#include "torch_api_stable.hpp"')
    ifdef = source.index("#ifdef TORCH_ANS_STABLE_ABI")
    else_branch = source.index("#else", ifdef)
    aten_include = source.index('#include "torch_api_aten.hpp"')
    assert ifdef < stable_include < else_branch < aten_include


def test_stable_backend_never_reaches_for_aten():
    """A prebuilt .so is only version-independent while this stays true."""
    code = _strip_line_comments(
        (TA_DIR / "torch_api_stable.hpp").read_text(encoding="utf-8"))
    for forbidden in ("torch/extension.h", "<ATen/", "<c10/",
                      "<torch/csrc/autograd"):
        assert forbidden not in code, (
            f"the stable backend must not include {forbidden}: staying off ATen "
            "is what makes the .so loadable on every torch >= 2.10")
    for symbol in ("at::", "c10::"):
        assert symbol not in code, (
            f"the stable backend must not use {symbol} C++ symbols")


def test_aten_backend_is_the_aten_one():
    code = _strip_line_comments(
        (TA_DIR / "torch_api_aten.hpp").read_text(encoding="utf-8"))
    assert "#include <torch/extension.h>" in code
    assert "namespace ta" in code


def test_headers_compile_with_gpp_in_both_modes():
    """The actual stage-0 acceptance criterion (migration section 3)."""
    result = _run_checker("--no-cuda")
    assert result.returncode == 0, result.stdout + result.stderr


def test_headers_compile_with_nvcc_when_available():
    nvcc, _ = _load_checker()._find_nvcc(None)
    if nvcc is None:
        pytest.skip("no usable nvcc; run scripts/check_ta_headers.py "
                    "--cuda-home <toolkit> locally to check the CUDA build")
    result = _run_checker("--only-cuda")
    assert result.returncode == 0, result.stdout + result.stderr
