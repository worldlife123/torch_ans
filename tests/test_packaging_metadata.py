"""Guard the two metadata sources and the sdist file list.

Two independent code paths can produce a distribution:

* ``setuptools >= 61`` reads the static PEP 621 ``[project]`` table from
  ``pyproject.toml`` (this is what an isolated ``python -m build`` uses);
* ``pip install . --no-build-isolation`` runs with whatever setuptools the user
  has, and setuptools < 61 ignores ``[project]`` completely, so ``setup.py``
  passes a mirror of that table through ``_LEGACY_METADATA`` in that case.

Both copies must agree, otherwise the same release ships two different
distributions. The same applies to ``MANIFEST.in``: the native headers
``rans_bindings.hpp`` / ``rans_build_config.hpp`` are required to build the
extension from the sdist, and setuptools does not recompute the manifest when
new source files appear, so they have to be listed explicitly.
"""

import ast
import fnmatch
import os
import sys

import pytest

try:  # Python >= 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None

pytestmark = pytest.mark.skipif(
    tomllib is None, reason="tomllib is only available on Python >= 3.11")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_DIR = os.path.join(ROOT, "torch_ans")

#: source extensions that must be shipped for the extension to build from an sdist
NATIVE_SUFFIXES = (".hpp", ".cuh", ".h", ".cpp", ".cu")

#: `[project]` keys that setup.py mirrors in `_LEGACY_METADATA`
MIRRORED_FIELDS = ("name", "version", "description", "python_requires",
                   "install_requires", "license", "url")


def _read_pyproject():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as handle:
        return tomllib.load(handle)


def _setup_tree():
    with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as handle:
        return ast.parse(handle.read(), filename="setup.py")


def _setup_call():
    for node in ast.walk(_setup_tree()):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            return node
    raise AssertionError("no setup() call found in setup.py")


def _setup_kwargs():
    """Literal keyword arguments of ``setup()``; ``**expr`` is stored as ``"**"``."""
    kwargs = {}
    for keyword in _setup_call().keywords:
        if keyword.arg is None:  # `**expr` expansion
            kwargs["**"] = ast.unparse(keyword.value)
            continue
        try:
            kwargs[keyword.arg] = ast.literal_eval(keyword.value)
        except ValueError:
            kwargs[keyword.arg] = None
    return kwargs


def _legacy_metadata():
    """The `_LEGACY_METADATA` mirror, values that are not literals skipped."""
    for node in _setup_tree().body:
        if not (isinstance(node, ast.Assign)
                and any(getattr(target, "id", None) == "_LEGACY_METADATA"
                        for target in node.targets)):
            continue
        assert isinstance(node.value, ast.Dict), "_LEGACY_METADATA must be a dict literal"
        metadata = {}
        for key, value in zip(node.value.keys, node.value.values):
            try:
                metadata[ast.literal_eval(key)] = ast.literal_eval(value)
            except ValueError:
                continue  # e.g. long_description, built at runtime
        return metadata
    raise AssertionError("no _LEGACY_METADATA assignment found in setup.py")


def test_setup_py_mirrors_pyproject_metadata():
    project = _read_pyproject()["project"]
    legacy = _legacy_metadata()

    assert legacy["name"] == project["name"]
    assert legacy["version"] == project["version"], (
        "setup.py and pyproject.toml disagree on the version; both are used "
        "depending on the setuptools version")
    assert legacy["description"] == project["description"]
    assert legacy["python_requires"] == project["requires-python"]
    assert sorted(legacy["install_requires"]) == sorted(project["dependencies"])
    assert legacy["url"] in set(project["urls"].values())
    assert legacy["license"] == "MIT"


def test_legacy_metadata_is_only_used_without_pep621_support():
    """`[project]` must be the single source of truth on modern setuptools."""
    kwargs = _setup_kwargs()
    for field in MIRRORED_FIELDS:
        assert field not in kwargs, (
            f"{field} is passed unconditionally to setup(); it would be "
            "overwritten by [project] and trigger setuptools warnings")
    expansion = kwargs.get("**", "")
    assert "_LEGACY_METADATA" in expansion
    assert "_setuptools_reads_pyproject_metadata" in expansion


def test_setup_py_build_logic_is_kept():
    """The extension must still be built by setup.py (pyproject only has metadata)."""
    kwargs = _setup_kwargs()
    # ext_modules/cmdclass are computed at runtime, so they are not literals:
    # assert on the key names and on the source instead.
    assert "ext_modules" in kwargs
    assert "cmdclass" in kwargs
    assert kwargs["packages"] == ["torch_ans"]
    with open(os.path.join(ROOT, "setup.py"), encoding="utf-8") as handle:
        source = handle.read()
    assert "package_data" in source
    assert "get_extension_config" in source


def test_manifest_ships_every_native_source_extension():
    with open(os.path.join(ROOT, "MANIFEST.in"), encoding="utf-8") as handle:
        patterns = [line.strip() for line in handle
                    if line.strip() and not line.lstrip().startswith("#")]

    # collect the patterns that apply to the package directory
    package_patterns = []
    for pattern in patterns:
        parts = pattern.split()
        if len(parts) >= 3 and parts[0] == "recursive-include" and parts[1] == "torch_ans":
            package_patterns.extend(parts[2:])

    shipped = set()
    for name in os.listdir(PACKAGE_DIR):
        extension = os.path.splitext(name)[1]
        if extension in NATIVE_SUFFIXES:
            shipped.add(extension)

    for extension in sorted(shipped):
        assert any(fnmatch.fnmatch("*" + extension, p) or p == "*" + extension
                   for p in package_patterns), (
            f"MANIFEST.in does not ship torch_ans/*{extension}")

    assert "global-exclude *.so *.pyc" in patterns, (
        "a stray local _C*.so must never be packaged: it shadows torch_ans/_C.py")
    assert "exclude torch_ans/_torch_build_version.py" in patterns, (
        "the generated build record must be regenerated by setup.py, not shipped "
        "stale from the working tree")


def test_license_is_declared_for_both_paths():
    project = _read_pyproject()["project"]
    assert project["license"]["file"] == "LICENSE"
    assert os.path.exists(os.path.join(ROOT, project["license"]["file"]))
    assert _legacy_metadata()["license"] == "MIT"
