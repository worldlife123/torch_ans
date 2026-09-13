"""Lazy (runtime-compiled) implementation of the `torch_ans._C` module.

This module is the *shim* half of `torch_ans._C`:

* when no compiled extension is present, `import torch_ans._C` resolves to
  `torch_ans/_C.py`, which forwards to this module, so both names behave the
  same;
* when a compiled extension **is** present, it shadows `torch_ans/_C.py`
  entirely, and `torch_ans._C` becomes the compiled module. The shim stays
  reachable as `torch_ans._lazy_C`, which is what lets the Python layer switch
  to a locally compiled extension after the fact - e.g. when a pre-built
  extension turns out to be unloadable (torch ABI mismatch) or was compiled
  without CUDA support while CUDA coding is requested.

Nothing is compiled until an attribute is accessed (which builds the *full*
module, the historical behaviour) or :func:`ensure_module` is called with a
build profile, which compiles only the operators that profile needs (see
torch_ans/_dynamic_build.py and rans_build_config.hpp).

Any already loaded module that exports everything a request asks for is reused,
so the full build satisfies every profile and creating a second interface with
a different configuration never rebuilds operators that are already there.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Any, Dict, Iterator, Optional

# NOTE: the JIT build uses an undotted module name on purpose: torch derives
# the required PyInit_* symbol from this name and pybind11 cannot handle dots
# in it (see _dynamic_build.build_extension). The module is reachable as
# `torch_ans._C` because this shim stays registered under that name and
# forwards attribute access.
_FULL_MODULE_NAME = "torch_ans_C"

#: cached module name -> loaded native module. The name is suffixed for builds
#: that are required to contain CUDA support, so a CUDA-capable and a CPU-only
#: build of the same profile can coexist (see _cache_key).
_modules: Dict[str, ModuleType] = {}


def module_has_cuda(module) -> Optional[bool]:
    """Whether `module` can code CUDA tensors, or None when it does not say.

    The flag is compiled into the extension (see torch_ans/lib.cpp), so it is
    correct for both install-time and runtime builds. None means the module
    predates the flag (or is not a native module at all).
    """
    flag = getattr(module, "_torch_ans_with_cuda", None)
    return None if flag is None else bool(flag)


def _cache_key(module_name: str, require_cuda: bool) -> str:
    return f"{module_name}::cuda" if require_cuda else module_name


def _build_name(module_name: str, require_cuda: bool) -> str:
    # A distinct build name keeps the CUDA and the CPU-only variant of one
    # profile in separate torch extension cache directories, so neither can be
    # reused for the other (torch's JIT cache is keyed by name + build flags).
    return f"{module_name}_cuda" if require_cuda else module_name


def _compile(module_name: str, defines, verbose: bool = True,
             require_cuda: bool = False):
    """Build `module_name` for the local torch, optionally insisting on CUDA."""
    from . import _dynamic_build

    try:
        module = _dynamic_build.build_extension(
            module_name=module_name, with_cuda=True, verbose=verbose,
            defines=defines)
    except Exception:
        if require_cuda:
            # Let the caller add its own guidance: the CUDA toolchain is either
            # missing or mismatched with the local torch.
            raise
        # Fallback: try disable CUDA if the build failed
        print("CUDA build failed, retrying with CPU-only configuration: " + str(sys.exc_info()[1]))
        return _dynamic_build.build_extension(
            module_name=module_name, with_cuda=False, verbose=verbose, defines=defines)

    if require_cuda and module_has_cuda(module) is False:
        # The CUDA build did not raise but produced a CPU-only extension: this
        # happens when a previous CUDA attempt already failed on this machine
        # and the outcome is remembered in the cuda_build_state file.
        raise RuntimeError(
            "CUDA coding was requested but the runtime build produced a "
            "CPU-only extension (a previous CUDA build attempt failed on this "
            "machine, see the cuda_build_state file in the torch extensions "
            "cache directory)")
    return module


def _usable(module, needed, require_cuda: bool) -> bool:
    if require_cuda and module_has_cuda(module) is False:
        return False
    return all(hasattr(module, name) for name in needed)


def ensure_full(verbose: bool = True, require_cuda: bool = False):
    """Build/load the native extension that contains every operator."""
    from . import _dynamic_build

    key = _cache_key(_FULL_MODULE_NAME, require_cuda)
    module = _modules.get(key)
    if module is None:
        module = _compile(_build_name(_FULL_MODULE_NAME, require_cuda),
                          _dynamic_build.profile_defines(None), verbose,
                          require_cuda=require_cuda)
        _modules[key] = module
    return module


def ensure_module(profile=None, verbose: bool = True, require_cuda: bool = False):
    """Build/load a native module that provides the operators `profile` needs.

    `profile` is a :class:`torch_ans._dynamic_build.BuildProfile`; `None` means
    the full module. A module that is already loaded and exports every requested
    operator (in particular the full build, or a profile built earlier) is
    reused instead of being rebuilt. With ``require_cuda`` a module that is
    known to lack CUDA support is never reused and the build insists on CUDA.
    """
    if profile is None:
        return ensure_full(verbose=verbose, require_cuda=require_cuda)

    from . import _dynamic_build

    needed = _dynamic_build.profile_op_names(profile)
    for module in _modules.values():
        if _usable(module, needed, require_cuda):
            return module

    module_name = _dynamic_build.profile_module_name(profile)
    key = _cache_key(module_name, require_cuda)
    module = _modules.get(key)
    if module is None:
        module = _compile(_build_name(module_name, require_cuda),
                          _dynamic_build.profile_defines(profile), verbose,
                          require_cuda=require_cuda)
        _modules[key] = module
    return module


def __getattr__(name: str) -> Any:
    """Forward attribute access to the full build (compiling it if needed)."""
    module = ensure_full()
    try:
        return getattr(module, name)
    except AttributeError as e:
        raise AttributeError(f"module 'torch_ans._C' has no attribute '{name}'") from e


def __dir__() -> Iterator[str]:
    # Provide a combined view of attributes from this shim and the compiled module
    names = set(globals().keys())
    try:
        names.update(dir(ensure_full()))
    except Exception:
        pass
    return sorted(names)
