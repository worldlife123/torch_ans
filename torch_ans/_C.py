"""Compatibility shim for `torch_ans._C`.

The lazy-compile implementation lives in `torch_ans/_lazy_C.py`; this module
only forwards to it, so that

* `import torch_ans._C` (and `from torch_ans._C import <op>`) keeps working when
  no compiled extension is present, exactly as before, and
* the shim stays importable as `torch_ans._lazy_C` when a compiled extension
  *is* present: `torch_ans/_C*.so` shadows this file completely, so the Python
  layer could otherwise never fall back to a runtime build (see
  `torch_ans/utils.py::_import_native_c`).

When a pre-built extension shadows this file, nothing here runs at all.
"""
from __future__ import annotations

from typing import Any, Iterator

# Re-exported for callers that use them to drive the incremental build.
from ._lazy_C import ensure_full, ensure_module  # noqa: F401

__all__ = ["ensure_full", "ensure_module"]


def __getattr__(name: str) -> Any:
    """Forward every operator lookup to the lazy implementation."""
    from . import _lazy_C
    return getattr(_lazy_C, name)


def __dir__() -> Iterator[str]:
    from . import _lazy_C
    return dir(_lazy_C)
