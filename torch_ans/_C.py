"""Pure-Python shim for `torch_ans._C`.

When the compiled extension (`torch_ans/_C*.so`) is not present, `import
torch_ans._C` resolves to this shim. Importing it is cheap: nothing is
compiled until an attribute is accessed (which compiles the *full* module, the
historical behaviour) or :func:`ensure_module` is called with a build profile,
which compiles only the operators that profile needs (see
torch_ans/_dynamic_build.py and rans_build_config.hpp).

Any already loaded module that exports everything a request asks for is reused,
so the full build satisfies every profile and creating a second interface with
a different configuration never rebuilds operators that are already there.

A pre-built extension shadows this file entirely: `torch_ans._C` is then the
compiled module, which always contains every operator.
"""
from __future__ import annotations

import sys
from types import ModuleType
from typing import Dict, Iterator

# NOTE: the JIT build uses an undotted module name on purpose: torch derives
# the required PyInit_* symbol from this name and pybind11 cannot handle dots
# in it (see _dynamic_build.build_extension). The module is reachable as
# `torch_ans._C` because this shim stays registered under that name and
# forwards attribute access.
_FULL_MODULE_NAME = "torch_ans_C"

#: module name -> loaded native module
_modules: Dict[str, ModuleType] = {}


def _compile(module_name: str, defines, verbose: bool = True):
    from . import _dynamic_build

    try:
        return _dynamic_build.build_extension(
            module_name=module_name, with_cuda=True, verbose=verbose, defines=defines)
    except Exception:
        # Fallback: try disable CUDA if the build failed
        print("CUDA build failed, retrying with CPU-only configuration: " + str(sys.exc_info()[1]))
        return _dynamic_build.build_extension(
            module_name=module_name, with_cuda=False, verbose=verbose, defines=defines)


def ensure_full(verbose: bool = True):
    """Build/load the native extension that contains every operator."""
    module = _modules.get(_FULL_MODULE_NAME)
    if module is None:
        from . import _dynamic_build
        module = _compile(_FULL_MODULE_NAME, _dynamic_build.profile_defines(None), verbose)
        _modules[_FULL_MODULE_NAME] = module
    return module


def ensure_module(profile=None, verbose: bool = True):
    """Build/load a native module that provides the operators `profile` needs.

    `profile` is a :class:`torch_ans._dynamic_build.BuildProfile`; `None` means
    the full module. A module that is already loaded and exports every
    requested operator (in particular the full build, or a profile built
    earlier) is reused instead of being rebuilt.
    """
    if profile is None:
        return ensure_full(verbose=verbose)

    from . import _dynamic_build

    needed = _dynamic_build.profile_op_names(profile)
    for module in _modules.values():
        if all(hasattr(module, name) for name in needed):
            return module

    module_name = _dynamic_build.profile_module_name(profile)
    module = _modules.get(module_name)
    if module is None:
        module = _compile(module_name, _dynamic_build.profile_defines(profile), verbose)
        _modules[module_name] = module
    return module


def __getattr__(name: str):
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
