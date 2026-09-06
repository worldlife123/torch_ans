"""Pure-Python shim for `torch_ans._C`.

When `torch_ans._C` (the compiled extension) is not present, importing
this shim will trigger a runtime compilation of the native extension and
then forward attribute access to the compiled module.

This enables `from torch_ans._C import rans64_push` to work even when the
compiled extension was not built at install time.
"""
from __future__ import annotations

import sys
import importlib
from types import ModuleType
from typing import Iterator

_compiled_mod: ModuleType | None = None

# NOTE: the JIT build uses an undotted module name on purpose: torch derives
# the required PyInit_* symbol from this name and pybind11 cannot handle dots
# in it (see _dynamic_build.build_extension). The module is registered under
# the dotted "torch_ans._C" name in sys.modules below.
_JIT_MODULE_NAME = "torch_ans_C"


def _ensure_compiled():
    global _compiled_mod
    if _compiled_mod is not None:
        return _compiled_mod

    # Build the extension under a temporary module name, then register it
    # as 'torch_ans._C' in sys.modules and return it.
    try:
        from . import _dynamic_build
    except Exception as e:
        raise ImportError("dynamic build helper not available: " + str(e))

    try:
        mod = _dynamic_build.build_extension(module_name=_JIT_MODULE_NAME, with_cuda=True, verbose=True)
    except Exception:
        # Fallback: try disable CUDA if the build failed
        print("CUDA build failed, retrying with CPU-only configuration: " + str(sys.exc_info()[1]))
        mod = _dynamic_build.build_extension(module_name=_JIT_MODULE_NAME, with_cuda=False, verbose=True)

    # Register compiled module under the standard name
    sys.modules["torch_ans._C"] = mod
    _compiled_mod = mod
    return mod

# Load the extension as soon as the package is imported
_C = _ensure_compiled()

def __getattr__(name: str):
    """Forward attribute access to the compiled extension after ensuring it exists."""
    mod = _ensure_compiled()
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise AttributeError(f"module 'torch_ans._C' has no attribute '{name}'") from e


def __dir__() -> Iterator[str]:
    # Provide a combined view of attributes from this shim and the compiled module
    names = set(globals().keys())
    try:
        mod = _ensure_compiled()
        names.update(dir(mod))
    except Exception:
        pass
    return sorted(names)
