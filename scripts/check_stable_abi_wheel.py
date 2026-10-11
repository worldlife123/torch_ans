"""Check a wheel built with `WITH_STABLE_ABI=1` (migration stage 8, section 6.5).

The point of the prebuilt wheel is that one `.so` serves every `torch >= 2.10` and
every CPython, so the things that could quietly break that promise are checked
here rather than left to a runtime failure on a user's machine:

* the wheel is platform-tagged but CPython-agnostic (`py3-none-<platform>`);
* it really ships the prebuilt libraries (`_C_cpu`, plus `_C_cuda` when the CUDA
  flavour was requested) and not a compiled `torch_ans/_C*.so`;
* no library references a C++ symbol outside the stable ABI: the undefined
  symbols must not contain `at::`, `c10::`, `torch::python` or any `Py*`;
* no library is linked against `libtorch_python` / `libpython` (the two that
  would tie the wheel to one torch build and one CPython).

The last two need the platform's own tools - `nm -D -u` + `readelf -d` on Linux,
`otool -L` + `nm -u` on macOS and `dumpbin /dependents` on Windows - and the
platform is taken from the wheel's tag (`--platform` overrides it). Missing tools
are reported and the checks are skipped, so the packaging half still runs; that
is the normal case for the symbol half on Windows, where the real gate is the
install-and-load job of the release workflow. Exit status is non-zero as soon as
one check fails.

Usage::

    python scripts/check_stable_abi_wheel.py wheelhouse/*.whl
    python scripts/check_stable_abi_wheel.py wheelhouse/*.whl --require-cuda
    python scripts/check_stable_abi_wheel.py wheelhouse/*.whl --platform macos
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

#: On top of `FORBIDDEN_NEEDED`, what "tied to one torch build" looks like per
#: platform. Windows spells the same libraries without the `lib` prefix and
#: links the CPython import library, which shows up as `python3xx.dll`. macOS
#: keeps `libtorch_cpu` on purpose (it carries the stable C shim) and reaches it
#: through `@rpath/../torch/lib`, so what must be caught there is an *absolute*
#: path to the build machine's torch - see `ABSOLUTE_LINK_WHITELIST`.
EXTRA_FORBIDDEN_NEEDED = {
    "linux": (),
    "macos": (),
    "windows": ("torch_python", "python3"),
}

#: macOS: the only absolute install names a shippable library may carry.
ABSOLUTE_LINK_WHITELIST = ("/usr/lib", "/System/Library")

#: Tools the symbol/dependency half needs, per platform.
PLATFORM_TOOLS = {
    "linux": ("nm", "readelf"),
    "macos": ("otool", "nm"),
    "windows": ("dumpbin",),
}

#: Suffixes a native extension has, per platform (setuptools: `.so` on Linux
#: *and* macOS, `.pyd` on Windows).
NATIVE_SUFFIXES = (".so", ".pyd", ".dylib")


def _platform_of(wheel: Path) -> str:
    """linux / macos / windows, from the wheel's platform tag."""
    name = wheel.name.lower()
    if "macosx" in name:
        return "macos"
    if name.endswith(".whl") and ("win32" in name or "win_amd64" in name
                                  or "win_arm64" in name):
        return "windows"
    if "manylinux" in name or "musllinux" in name or "linux" in name:
        return "linux"
    return "unknown"


def _run(command):
    result = subprocess.run([str(part) for part in command],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(str(p) for p in command)} failed:\n"
                           f"{result.stdout}{result.stderr}")
    return result.stdout


def _undefined_symbols(path: Path, platform: str):
    # `-D` ("dynamic symbols only") is a GNU-only switch; macOS `nm` gets the
    # same answer from `-u`. Windows has no nm at all, so the caller skips it.
    command = ["nm", "-u", path] if platform == "macos" else ["nm", "-D", "-u", path]
    return [line.split()[-1] for line in _run(command).splitlines() if line.split()]


def _needed_libraries(path: Path):
    needed = []
    for line in _run(["readelf", "-d", path]).splitlines():
        if "NEEDED" not in line:
            continue
        match = re.search(r"\[([^\]]+)\]", line)
        if match:
            needed.append(match.group(1))
    return needed


def _linked_libraries(path: Path, platform: str):
    """The libraries `path` is linked against, whatever the platform's spelling.

    Linux: DT_NEEDED (`readelf -d`). macOS: the install names `otool -L` lists.
    Windows: the DLLs `dumpbin /dependents` lists.
    """
    if platform == "macos":
        lines = _run(["otool", "-L", path]).splitlines()
        # The first line is "<path>:"; the rest are "<install name> (compat ...)".
        return [line.strip().split()[0].rsplit("/", 1)[-1]
                for line in lines[1:] if line.strip()]
    if platform == "windows":
        return [line.strip() for line in _run(["dumpbin", "/dependents", path]).splitlines()
                if line.strip().lower().endswith(".dll")]
    return _needed_libraries(path)


def check_wheel(wheel: Path, require_cuda: bool, platform: str = "auto") -> None:
    failures = []

    if platform in ("auto", None):
        platform = _platform_of(wheel)
    print(f"  platform: {platform}")

    if "py3-none-" not in wheel.name:
        failures.append(
            f"{wheel.name}: a stable-ABI wheel has no Python symbols, so it must be "
            f"tagged py3-none-<platform>")

    with tempfile.TemporaryDirectory(prefix="stable_wheel_check_") as tmp:
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            archive.extractall(tmp)

        libraries = sorted(path for suffix in NATIVE_SUFFIXES
                           for path in Path(tmp).rglob("*" + suffix))
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
            tools = [tool for tool in PLATFORM_TOOLS.get(platform, PLATFORM_TOOLS["linux"])
                     if shutil.which(tool) is None]
            if tools:
                print(f"  (skipping the symbol/dependency checks: "
                      f"{', '.join(tools)} not found)")
            forbidden_names = FORBIDDEN_NEEDED + EXTRA_FORBIDDEN_NEEDED.get(platform, ())
            for library in libraries:
                print(f"  {library.name}")
                if tools:
                    continue
                if platform != "windows":
                    # no `nm` on Windows: the dependency list is what is checked
                    undefined = _undefined_symbols(library, platform)
                    bad = [symbol for symbol in undefined
                           if any(pattern.match(symbol)
                                  for pattern in FORBIDDEN_SYMBOL_PATTERNS)]
                    if bad:
                        failures.append(
                            f"{library.name} references non-stable symbols: {bad[:5]}"
                            f"{' ...' if len(bad) > 5 else ''}"
                            f" ({len(undefined)} undefined symbols in total)")
                needed = _linked_libraries(library, platform)
                print(f"    linked: {', '.join(needed) or '<none>'}")
                forbidden = [entry for entry in needed
                             if any(bad_name in entry for bad_name in forbidden_names)]
                if forbidden:
                    failures.append(
                        f"{library.name} links {forbidden}: the wheel would be tied "
                        f"to one CPython / one torch build")
                if platform == "macos":
                    absolute = [entry for entry in needed
                                if entry.startswith("/")
                                and not entry.startswith(ABSOLUTE_LINK_WHITELIST)]
                    if absolute:
                        failures.append(
                            f"{library.name} links {absolute}: an absolute path to "
                            f"the build machine's libraries does not exist on the "
                            f"user's (use @rpath, like setup.py does)")

        names = [name for name in names if name.endswith(NATIVE_SUFFIXES)]
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
    parser.add_argument("--platform", default="auto",
                        choices=["auto", "linux", "macos", "windows"],
                        help="which toolchain's checks to run (auto: from the "
                             "wheel's platform tag)")
    args = parser.parse_args(argv)

    for wheel in args.wheels:
        print(f"checking {wheel}")
        check_wheel(Path(wheel), args.require_cuda, args.platform)
    return 0


if __name__ == "__main__":
    sys.exit(main())
