"""Loader for the prebuilt libtorch-stable-ABI extension.

A wheel can ship one `.so` compiled against the libtorch stable ABI
(`TORCH_TARGET_VERSION` = torch 2.10) that serves every `torch >= 2.10`,
instead of compiling per torch version. This
module is the Python side of that path:

* :func:`load_stable` finds `_C_cpu` (always) and `_C_cuda` (only when a GPU is
  visible or CUDA is required), loads them with `torch.ops.load_library` and
  returns a module-shaped object for `torch.ops.torch_ans`;
* :class:`_StableModule` exposes the same attribute names the compiled `_C`
  module does, so `utils.py` does not need to know which path it got;
* failure of any kind returns ``None`` - falling back to the runtime build,
  the same philosophy as `utils._import_native_c` (migration section 5.2).

The two `py::bytes` operators are implemented here in Python: the stable ABI
has no schema type for `std::vector<py::bytes>`, so C++ drops them (migration
section 1.5). Their output is byte-for-byte the C++ one - `tests/test_stable_abi.py`
compares both against the compiled `_C` as the golden reference.

Environment:
  * ``TORCH_ANS_NO_STABLE_ABI=1``   - never use the prebuilt extension
    (debugging/CI: forces the runtime build);
  * ``TORCH_ANS_STABLE_LIB_DIR``    - extra directory to look for `_C_cpu*` /
    `_C_cuda*` in, before the package directory (used by the tests, which build
    the stable extension outside the source tree).
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import torch

#: Lowest torch whose stable ABI the prebuilt `.so` targets; the same value is
#: passed to the compiler as -DTORCH_TARGET_VERSION (see setup.py and
#: poc/stable_abi/setup.py).
STABLE_ABI_MIN_TORCH = (2, 10)

#: Set to 1/true/yes to ignore the prebuilt extension entirely.
DISABLE_ENV_VAR = "TORCH_ANS_NO_STABLE_ABI"

#: Extra search directory for `_C_cpu*` / `_C_cuda*` (see the module docstring).
LIB_DIR_ENV_VAR = "TORCH_ANS_STABLE_LIB_DIR"

#: The operators C++ cannot export through a stable schema; see the docstring.
PYTHON_OPS = ("rans_stream_to_byte_strings", "rans_byte_strings_to_stream")


def _truthy(value: Optional[str]) -> bool:
    return value is not None and value.strip().lower() in ("1", "true", "yes", "on")


def _torch_version_tuple() -> tuple:
    """`torch.__version__` as a 3-tuple of ints ('2.11.0+cu130' -> (2, 11, 0))."""
    parts: List[int] = []
    for chunk in torch.__version__.split("+")[0].split(".")[:3]:
        digits = ""
        for char in chunk:
            if not char.isdigit():
                break
            digits += char
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


def _search_dirs() -> Sequence[Path]:
    dirs: List[Path] = []
    override = os.getenv(LIB_DIR_ENV_VAR)
    if override:
        dirs.append(Path(override))
    dirs.append(Path(__file__).resolve().parent)
    return dirs


def _find_extension(stem: str) -> Optional[Path]:
    """Locate `_C_cpu` / `_C_cuda` (`.so`, `.dylib` or `.pyd`)."""
    for directory in _search_dirs():
        if not directory.is_dir():
            continue
        for pattern in (stem + ".*", stem + "*"):
            matches = sorted(path for path in directory.glob(pattern)
                             if path.is_file() and path.suffix in (".so", ".dylib", ".pyd"))
            if matches:
                return matches[0]
    return None


class _StableModule:
    """`torch.ops.torch_ans` behind the attribute names of the compiled `_C`.

    Only the operators and the capability flag are exposed; in particular there
    is deliberately no `ensure_module` / `ensure_full` attribute, because
    `utils.module_has_cuda` treats their presence as "this is the lazy shim" and
    would refuse to read `_torch_ans_with_cuda`.
    """

    def __init__(self, with_cuda: bool, loaded_paths: Sequence[Path]):
        self._torch_ans_with_cuda = bool(with_cuda)
        self._loaded_paths = tuple(str(path) for path in loaded_paths)

    def __getattr__(self, name: str):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
        if name in PYTHON_OPS:
            return globals()[name]
        namespace = getattr(torch.ops, "torch_ans", None)
        if namespace is not None:
            try:
                return getattr(namespace, name)
            except AttributeError:
                pass
        raise AttributeError(
            f"the stable-ABI extension does not provide {name!r}; it was loaded "
            f"from {', '.join(self._loaded_paths) or '<nothing>'}")

    def __dir__(self):
        names = set(PYTHON_OPS) | {"_torch_ans_with_cuda"}
        namespace = getattr(torch.ops, "torch_ans", None)
        if namespace is not None:
            names.update(dir(namespace))
        return sorted(names)


# ---------------------------------------------------------------------------
# the two operators the stable ABI cannot express
# ---------------------------------------------------------------------------
def rans_stream_to_byte_strings(stream: torch.Tensor) -> List[bytes]:
    """Per-row byte strings of a coding stream (C++ `rans_stream_to_byte_strings`).

    `stream[b][0]` is the byte length of row `b`, the bytes themselves start at
    `stream[b][1]` (see AGENTS.md, "stream tensor layout").
    """
    if not isinstance(stream, torch.Tensor):
        raise TypeError("stream must be a torch.Tensor")
    if stream.dtype != torch.int32:
        raise TypeError(f"stream must be int32, got {stream.dtype}")
    safe = stream.detach().to(device="cpu").contiguous()
    raw = safe.view(torch.uint8).reshape(safe.shape[0], -1)
    lengths = safe[:, 0].tolist()
    return [bytes(raw[row, 4:4 + int(length)].tolist())
            for row, length in enumerate(lengths)]


def rans_byte_strings_to_stream(byte_strings: Sequence[bytes]) -> torch.Tensor:
    """Pack per-row byte strings into a coding stream (C++ `rans_byte_strings_to_stream`).

    The inverted operation of :func:`rans_stream_to_byte_strings`, with the same
    allocation rule as C++: `ceil(max_length / 4) + 1` int32 words per row.
    """
    data = [bytes(item) for item in byte_strings]
    batch_size = len(data)
    max_length = max((len(item) for item in data), default=0)
    stream_length = (max_length + 3) // 4 + 1
    stream = torch.zeros((batch_size, stream_length), dtype=torch.int32)
    raw = stream.view(torch.uint8).reshape(batch_size, stream_length * 4)
    for row, item in enumerate(data):
        stream[row, 0] = len(item)
        if item:
            raw[row, 4:4 + len(item)] = torch.frombuffer(bytearray(item), dtype=torch.uint8)
    return stream


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
#: require_cuda -> module or None. `torch.ops.load_library` registers into the
#: process-wide dispatcher, so each path is loaded at most once.
_loaded: dict = {}


def _load(require_cuda: bool) -> Optional[_StableModule]:
    if _truthy(os.getenv(DISABLE_ENV_VAR)):
        return None
    if _torch_version_tuple() < STABLE_ABI_MIN_TORCH:
        return None

    cpu_path = _find_extension("_C_cpu")
    if cpu_path is None:
        return None
    cuda_path = _find_extension("_C_cuda")
    if require_cuda and cuda_path is None:
        # A CUDA request cannot be served by the CPU library; leave it to the
        # runtime build, which probes the local CUDA toolchain (utils.py's
        # fallback, incl. the _CUDA_UNAVAILABLE_HINT).
        return None

    paths = [cpu_path]
    if cuda_path is not None and (require_cuda or torch.cuda.is_available()):
        paths.append(cuda_path)

    loaded: List[Path] = []
    with_cuda = False
    for path in paths:
        try:
            torch.ops.load_library(str(path))
        except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
            if path is cpu_path:
                warnings.warn(
                    f"the prebuilt stable-ABI extension {path.name} could not be "
                    f"loaded ({type(exc).__name__}: {exc}); falling back to a "
                    f"local build", RuntimeWarning, stacklevel=3)
                return None
            warnings.warn(
                f"the prebuilt stable-ABI CUDA extension {path.name} could not be "
                f"loaded ({type(exc).__name__}: {exc}); CUDA coding will need a "
                f"local build", RuntimeWarning, stacklevel=3)
            if require_cuda:
                return None
            continue
        loaded.append(path)
        if path is cuda_path:
            with_cuda = True

    if require_cuda and not with_cuda:
        return None
    return _StableModule(with_cuda, loaded)


def load_stable(require_cuda: bool = False) -> Optional[_StableModule]:
    """The prebuilt stable-ABI module, or None when it is unavailable.

    Cached per `require_cuda` value; the answer is not cached across the
    "extension missing" case in any expensive way, since finding it is a couple
    of `glob` calls.
    """
    key = bool(require_cuda)
    if key not in _loaded:
        _loaded[key] = _load(key)
    return _loaded[key]
