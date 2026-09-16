"""Check a wheel built with `WITH_STABLE_ABI=1` (migration stage 8, section 6.5).

The point of the prebuilt wheel is that one `.so` serves every `torch >= 2.10` and
every CPython, so the things that could quietly break that promise are checked
here rather than left to a runtime failure on a user's machine:

* the wheel is platform-tagged but CPython-agnostic (`py3-none-<platform>`);
* it really ships the prebuilt libraries (`_C_cpu`, plus `_C_cuda` when the CUDA
  flavour was requested) and not a compiled `torch_ans/_C*.so`;
* no library references a C++ symbol outside the stable ABI: `nm -D -u` must not
  list `at::`, `c10::`, `torch::python` or any `Py*` symbol;
* `readelf -d` must not list `libtorch_python` / `libpython` as NEEDED.

The symbol checks need `nm`/`readelf` (binutils); they are skipped with a warning
when those are unavailable, so the script is usable on macOS for the packaging
half. Exit status is non-zero as soon as one check fails.

Usage::

    python scripts/check_stable_abi_wheel.py wheelhouse/*.whl
    python scripts/check_stable_abi_wheel.py wheelhouse/*.whl --require-cuda
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

#: Undefined symbols a stable-ABI library must never reference. `_ZN...` is the
#: Itanium mangling of a C++ symbol (`_ZNK` for const methods); `Py*` is the
#: CPython C API, which is what would tie the wheel to one interpreter.
FORBIDDEN_SYMBOL_PATTERNS = [
    re.compile(r"^_ZNK?2at"),        # at::
    re.compile(r"^_ZNK?3c10"),       # c10::
    re.compile(r"^_ZNK?5torch8python"),  # torch::python
    re.compile(r"^Py[A-Z_]"),        # CPython API
]

#: Libraries that must not be DT_NEEDED entries of a prebuilt `.so`.
FORBIDDEN_NEEDED = ("libtorch_python", "libpython")


def _run(command):
    result = subprocess.run([str(part) for part in command],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(str(p) for p in command)} failed:\n"
                           f"{result.stdout}{result.stderr}")
    return result.stdout


def _undefined_symbols(path: Path):
    return [line.split()[-1] for line in _run(["nm", "-D", "-u", path]).splitlines()
            if line.split()]


def _needed_libraries(path: Path):
    needed = []
    for line in _run(["readelf", "-d", path]).splitlines():
        if "NEEDED" not in line:
            continue
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            needed.append(match.group(1))
    return needed


def check_wheel(wheel: Path, require_cuda: bool) -> None:
    failures = []

    if "py3-none-" not in wheel.name:
        failures.append(
            f"{wheel.name}: a stable-ABI wheel has no Python symbols, so it must be "
            f"tagged py3-none-<platform>")

    with tempfile.TemporaryDirectory(prefix="stable_wheel_check_") as tmp:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            archive.extractall(tmp)

        libraries = sorted(Path(tmp).rglob("*.so"))
        library_names = [path.name for path in libraries]
        if not any("_C_cpu" in name for name in library_names):
            failures.append(f"no prebuilt CPU library in {wheel.name}: {library_names}")
        if require_cuda and not any("_C_cuda" in name for name in library_names):
            failures.append(f"no prebuilt CUDA library in {wheel.name}: {library_names}")
        # A compiled extension would shadow _C.py and bind the wheel to one torch
        compiled = [name for name in library_names
                    if name.startswith("_C.") or name.startswith("_C.cpython")]
        if compiled:
            failures.append(
                f"{wheel.name} ships a locally compiled extension {compiled}: that is "
                f"the ATen build, not the stable-ABI one")

        if not libraries:
            failures.append(f"{wheel.name} ships no native library at all")
        else:
            tools = [tool for tool in ("nm", "readelf") if shutil.which(tool) is None]
            if tools:
                print(f"  (skipping the symbol checks: {', '.join(tools)} not found)")
            for library in libraries:
                print(f"  {library.name}")
                if not tools:
                    undefined = _undefined_symbols(library)
                    bad = [symbol for symbol in undefined
                           if any(pattern.match(symbol)
                                  for pattern in FORBIDDEN_SYMBOL_PATTERNS)]
                    if bad:
                        failures.append(
                            f"{library.name} references non-stable symbols: {bad[:5]}"
                            f"{' ...' if len(bad) > 5 else ''}"
                            f" ({len(undefined)} undefined symbols in total)")
                    needed = _needed_libraries(library)
                    print(f"    NEEDED: {', '.join(needed) or '<none>'}")
                    forbidden = [entry for entry in needed
                                 if any(bad_name in entry for bad_name in FORBIDDEN_NEEDED)]
                    if forbidden:
                        failures.append(
                            f"{library.name} links {forbidden}: the wheel would be tied "
                            f"to one CPython / one torch build")

        names = [name for name in names if name.endswith((".so", ".pyd", ".dylib"))]
        print(f"  native files in the wheel: {names or '<none>'}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(f"OK: {wheel.name} ships the prebuilt stable-ABI extension and nothing else")


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("wheels", nargs="+", help="wheel(s) to check")
    parser.add_argument("--require-cuda", action="store_true",
                        help="the CUDA flavour of the wheel: _C_cuda.so must be there")
    args = parser.parse_args(argv)

    for wheel in args.wheels:
        print(f"checking {wheel}")
        check_wheel(Path(wheel), args.require_cuda)
    return 0


if __name__ == "__main__":
    sys.exit(main())
