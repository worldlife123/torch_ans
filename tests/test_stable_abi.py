"""The prebuilt libtorch-stable-ABI extension: registration and bit-exactness.

The stable-ABI build turns the migration into something observable:
one `.so` per flavour registers every operator through `STABLE_TORCH_LIBRARY`
and the Python layer prefers it over compiling anything. This module checks the
claims that matter, and the runtime ones are skipped entirely when no stable-ABI
extension is present (the normal case for a source checkout and for CI - build
one via `WITH_STABLE_ABI=1 pip install .`, or point `TORCH_ANS_STABLE_LIB_DIR` at
a directory that contains `_C_cpu*`):

1. every operator `_dynamic_build.profile_op_names(None)` promises is really
   registered, minus the two `py::bytes` operators the stable ABI cannot express
   (Python implements those, see `torch_ans/_stable_C.py`);
2. `torch_ans.utils._native_module()` picks the prebuilt module instead of
   compiling (migration section 1.3, "the profile machinery is bypassed");
3. push/pop through the stable module produce the *same stream* as the compiled
   ATen extension, bit for bit (the whole point of the dual build);
4. the Python `bytes` operators agree byte for byte with the C++ ones;
5. with a CUDA flavour present (stage 6), the CUDA kernels are reachable through
   the same operator names, encode the same bytes as the CPU kernels and are
   interchangeable with them (CPU encode -> CUDA decode and back), including the
   32-way warp path and the sparse inverse-CDF lookup;
6. the packaging side of the switch: `setup.py` really has a `WITH_STABLE_ABI`
   build producing `_C_cpu` / `_C_cuda` with the stable-ABI flags and without the
   ATen-only ones, and `pyproject.toml` no longer depends on pybind11 (stage 7).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_ans import _dynamic_build, _stable_C

from stream_helpers import uint8_view

ROOT = Path(__file__).resolve().parents[1]

#: Loading the prebuilt extension registers its operators process-wide, so the
#: answer is stable for the whole session.
_STABLE = _stable_C.load_stable()

needs_stable = pytest.mark.skipif(
    _STABLE is None,
    reason="no prebuilt stable-ABI extension found (build one and/or set "
           "TORCH_ANS_STABLE_LIB_DIR)")

#: A CUDA flavour that is both loaded and usable (only a `_C_cuda` library
#: registers CUDA kernels; `_stable_C` reports that as `_torch_ans_with_cuda`).
needs_stable_cuda = pytest.mark.skipif(
    _STABLE is None or not getattr(_STABLE, "_torch_ans_with_cuda", False)
    or not torch.cuda.is_available(),
    reason="the prebuilt stable-ABI module has no CUDA library, or there is no GPU")


def _compiled_extension():
    """The ATen/pybind11 extension, when it is importable (the golden side)."""
    try:
        from torch_ans import _C as compiled
    except Exception:  # pragma: no cover - depends on the local build
        return None
    if not hasattr(compiled, "rans32_16_init_stream"):
        return None
    return compiled


needs_compiled = pytest.mark.skipif(
    _compiled_extension() is None,
    reason="the compiled ATen extension is not importable (no golden reference)")


def _compiled_cuda_extension():
    """The compiled ATen extension, but only when *it* can code CUDA tensors."""
    compiled = _compiled_extension()
    if compiled is None or not torch.cuda.is_available():
        return None
    if getattr(compiled, "_torch_ans_with_cuda", None) is not True:
        return None
    return compiled


needs_compiled_cuda = pytest.mark.skipif(
    _compiled_cuda_extension() is None,
    reason="no CUDA-capable compiled ATen extension (no golden reference)")


def _params(batch=4, num_symbols=64, alphabet=8, freq_precision=8, seed=0):
    """A batch of symbols plus a uniform CDF wide enough for all of them."""
    generator = torch.Generator().manual_seed(seed)
    scale = 1 << freq_precision
    cdf = (torch.arange(alphabet + 1, dtype=torch.int64) * scale // alphabet)
    cdf[-1] = scale
    cdfs = cdf.to(torch.int32).unsqueeze(0)
    cdfs_sizes = torch.tensor([alphabet + 1], dtype=torch.int32)
    offsets = torch.zeros(1, dtype=torch.int32)
    indexes = torch.zeros((batch, num_symbols), dtype=torch.int32)
    symbols = torch.randint(0, alphabet, (batch, num_symbols), dtype=torch.int32,
                            generator=generator)
    return cdfs, cdfs_sizes, offsets, indexes, symbols, freq_precision


def _mover(device):
    """`tensor -> tensor on `device``; a coding op is dispatched by its arguments."""
    return (lambda tensor: tensor) if device == "cpu" else (lambda tensor: tensor.to(device))


def _push_only(module, params, device="cpu", interleave=1):
    """init + push through `module`, returning the *encoded* stream.

    Use this - not `_push_pop` - for byte comparisons: `pop` rewinds the length
    word, so a decoded stream exposes its state words and nothing else of the
    bitstream.
    """
    cdfs, cdfs_sizes, offsets, indexes, symbols, freq_precision = params
    move = _mover(device)
    suffix = "" if interleave == 1 else f"_i{interleave}"
    push_name = f"rans32_16{suffix}_push"
    if not hasattr(module, push_name):
        pytest.skip(f"this build does not provide {push_name}")

    stream = move(module.rans32_16_init_stream(symbols.shape[0], interleave, 0))
    getattr(module, push_name)(stream, move(symbols), move(indexes), move(cdfs),
                               move(cdfs_sizes), move(offsets), 8, freq_precision,
                               True, 4)
    return stream


def _push_pop(module, params, device="cpu", interleave=1, inverse_cdf_precision=-1):
    """init + push + pop through `module`, returning (stream, decoded).

    `device` moves the tensors before the calls, `interleave` selects the `_i<N>`
    operator suffix and `inverse_cdf_precision` the `_invcdf_pop` variant - i.e.
    the two CUDA-only features of stage 6: the 32-way warp kernels and the sparse
    inverse-CDF table.
    """
    cdfs, cdfs_sizes, offsets, indexes, symbols, freq_precision = params
    move = _mover(device)
    suffix = "" if interleave == 1 else f"_i{interleave}"
    pop_name = f"rans32_16{suffix}_{'invcdf_' if inverse_cdf_precision > 0 else ''}pop"
    if not hasattr(module, pop_name):
        pytest.skip(f"this build does not provide {pop_name}")

    stream = _push_only(module, params, device=device, interleave=interleave)
    decoded = getattr(module, pop_name)(
        stream, move(indexes), move(cdfs), move(cdfs_sizes), move(offsets),
        8, freq_precision, True, 4, inverse_cdf_precision)
    return stream, decoded


def _coded_bytes(stream):
    """The used part of every row of `stream`.

    A push grows the stream in place and only copies the old contents back, so
    the padding beyond `stream[b][0]` is whatever the allocator handed out - it
    differs between two runs and carries no information. The used part is the
    length word in element 0 plus the bytes that word counts (the states and the
    bitstream, starting at the second element).
    """
    lengths = stream[:, 0].tolist()
    raw = uint8_view(stream)
    return [bytes(raw[row, 4:4 + int(length)].tolist())
            for row, length in enumerate(lengths)]


@needs_stable
def test_every_promised_operator_is_registered():
    """The schema list must match the profile list the Python layer relies on."""
    # NOTE: dir(torch.ops.torch_ans) does not list the registered operators (the
    # namespace object only publishes a handful of attributes), so ask the
    # dispatcher for the schemas it holds instead.
    registered = {schema.name.split("::", 1)[1].split(".")[0]
                  for schema in torch._C._jit_get_all_schemas()
                  if schema.name.startswith("torch_ans::")}
    expected = set(_dynamic_build.profile_op_names(None)) - set(_stable_C.PYTHON_OPS)
    assert registered == expected, (
        f"missing from the stable extension: {sorted(expected - registered)}; "
        f"not promised by _dynamic_build: {sorted(registered - expected)}")


@needs_stable
def test_the_python_layer_prefers_the_prebuilt_extension():
    """The lazy path must hand out the prebuilt module instead of compiling.

    `utils._native_module()` only consults the shim when `torch_ans._C` is not a
    compiled extension (a prebuilt one is used as-is, as always); the shim - i.e.
    the shipped-wheel case, where `_C.py` is what `torch_ans._C` resolves to - is
    tested here through `_lazy_C`, for both the full and a profile build.
    """
    from torch_ans import _lazy_C, utils

    assert _lazy_C.ensure_full() is _STABLE
    profile = _dynamic_build.BuildProfile(family="rans64", interleave=4, invcdf=True)
    assert _lazy_C.ensure_module(profile) is _STABLE
    # and the request reaches it through utils as well
    assert utils._runtime_native_module(profile) is _STABLE
    # The capability flag has to be readable without going through the ops.
    assert isinstance(_STABLE._torch_ans_with_cuda, bool)


@needs_stable
def test_bytes_operators_are_implemented_in_python():
    for name in _stable_C.PYTHON_OPS:
        assert getattr(_STABLE, name) is getattr(_stable_C, name)


@needs_stable
@needs_compiled
def test_push_pop_bitstream_matches_the_compiled_extension():
    """One stream, two build modes: the coded bytes have to be identical."""
    compiled = _compiled_extension()
    params = _params(seed=1)

    _, stable_decoded = _push_pop(_STABLE, params)
    _, compiled_decoded = _push_pop(compiled, params)

    assert _coded_bytes(_push_only(_STABLE, params)) == \
        _coded_bytes(_push_only(compiled, params)), (
            "the stable-ABI and the ATen push produced different streams")
    assert torch.equal(stable_decoded, compiled_decoded)
    assert torch.equal(stable_decoded, params[4])


@needs_stable
@needs_compiled
def test_bytes_operators_match_the_compiled_extension():
    """The Python replacements for the two `py::bytes` ops are drop-in."""
    compiled = _compiled_extension()
    stream, _ = _push_pop(compiled, _params(seed=2))

    reference = compiled.rans_stream_to_byte_strings(stream)
    implemented = _stable_C.rans_stream_to_byte_strings(stream)
    assert implemented == reference

    assert torch.equal(compiled.rans_byte_strings_to_stream(reference),
                       _stable_C.rans_byte_strings_to_stream(reference))
    # and they round-trip the coded bytes they came from
    assert _coded_bytes(_stable_C.rans_byte_strings_to_stream(implemented)) == implemented


# ---------------------------------------------------------------------------
# 5. the CUDA flavour
# ---------------------------------------------------------------------------
# The CUDA library is a second .so that registers implementations *only* for the
# CUDA dispatch key; the operator names - and therefore every call below - are the
# same ones the CPU library exports. `rans32_16` is the family these tests pin
# down (interleaved coding there is the warp path, and among the rANS families it
# exists for rans32_16 only; the tANS families have their own warp path and are
# covered by tests/test_tans.py).

@needs_stable_cuda
def test_cuda_push_pop_matches_the_cpu_one_bit_for_bit():
    """Same bytes on both devices, and each side decodes the other's stream."""
    params = _params(seed=3)
    cdfs, cdfs_sizes, offsets, indexes, symbols, freq_precision = params

    # encode on both devices (compared before any decode: see _push_only)
    cpu_encoded = _push_only(_STABLE, params)
    cuda_encoded = _push_only(_STABLE, params, device="cuda")
    assert _coded_bytes(cuda_encoded.cpu()) == _coded_bytes(cpu_encoded)

    # each device decodes its own stream
    _, cpu_decoded = _push_pop(_STABLE, params)
    _, cuda_decoded = _push_pop(_STABLE, params, device="cuda")
    assert torch.equal(cpu_decoded, symbols)
    assert torch.equal(cuda_decoded.cpu(), symbols)

    # CPU-encoded stream -> GPU decode. The op is dispatched by the device of its
    # *arguments*, so this is a CUDA kernel reading what the CPU wrote.
    decoded_on_gpu = _STABLE.rans32_16_pop(
        cpu_encoded.cuda(), indexes.cuda(), cdfs.cuda(), cdfs_sizes.cuda(),
        offsets.cuda(), 8, freq_precision, True, 4)
    assert torch.equal(decoded_on_gpu.cpu(), symbols)

    # GPU-encoded stream -> CPU decode
    decoded_on_cpu = _STABLE.rans32_16_pop(
        cuda_encoded.cpu(), indexes, cdfs, cdfs_sizes, offsets,
        8, freq_precision, True, 4)
    assert torch.equal(decoded_on_cpu, symbols)


@needs_stable_cuda
@needs_compiled_cuda
def test_cuda_push_matches_the_compiled_extension_bit_for_bit():
    """The CUDA kernel of the prebuilt .so vs the one compiled locally."""
    params = _params(seed=7)
    stable_stream = _push_only(_STABLE, params, device="cuda")
    compiled_stream = _push_only(_compiled_cuda_extension(), params, device="cuda")
    assert _coded_bytes(stable_stream.cpu()) == _coded_bytes(compiled_stream.cpu())


@needs_stable_cuda
def test_cuda_warp32_interleaved_matches_the_cpu_kernel():
    """The 32-way warp path (CUDA) against the CPU interleaved kernels.

    70 symbols on a 32-lane group exercises both the tail and the main loop.
    """
    params = _params(batch=2, num_symbols=70, seed=4)

    cpu_encoded = _push_only(_STABLE, params, interleave=32)
    cuda_encoded = _push_only(_STABLE, params, device="cuda", interleave=32)
    assert _coded_bytes(cuda_encoded.cpu()) == _coded_bytes(cpu_encoded)

    _, cpu_decoded = _push_pop(_STABLE, params, interleave=32)
    _, cuda_decoded = _push_pop(_STABLE, params, device="cuda", interleave=32)
    assert torch.equal(cpu_decoded, params[4])
    assert torch.equal(cuda_decoded.cpu(), params[4])


@needs_stable_cuda
def test_cuda_sparse_inverse_cdf_matches_the_cpu_path():
    """`rans32_16_invcdf_pop` with a sparse table, on both devices.

    The table is fused into the CDF tensor by rans_build_inverse_cdf (cdf ++
    table), which is what the high-level API passes as `cdfs`; a table_precision
    below freq_precision is the sparse path.
    """
    params = _params(seed=5)
    cdfs, cdfs_sizes, offsets, indexes, symbols, freq_precision = params
    table_precision = 3  # 2**3 entries, shift = freq_precision - 3 = 5
    fused = _STABLE.rans_build_inverse_cdf(cdfs, freq_precision, table_precision)
    fused_params = (fused, cdfs_sizes, offsets, indexes, symbols, freq_precision)

    _, cpu_decoded = _push_pop(_STABLE, fused_params,
                               inverse_cdf_precision=table_precision)
    _, cuda_decoded = _push_pop(_STABLE, fused_params, device="cuda",
                                inverse_cdf_precision=table_precision)
    assert torch.equal(cpu_decoded, symbols)
    assert torch.equal(cuda_decoded.cpu(), symbols)

    # On the same encoded bytes, the sparse-table lookup and the divided search
    # have to return the same symbols.
    table_stream = _push_only(_STABLE, fused_params, device="cuda")
    plain_stream = _push_only(_STABLE, params, device="cuda")
    with_table = _STABLE.rans32_16_invcdf_pop(
        table_stream, indexes.cuda(), fused.cuda(), cdfs_sizes.cuda(),
        offsets.cuda(), 8, freq_precision, True, 4, table_precision)
    divided_search = _STABLE.rans32_16_pop(
        plain_stream, indexes.cuda(), cdfs.cuda(), cdfs_sizes.cuda(),
        offsets.cuda(), 8, freq_precision, True, 4)
    assert torch.equal(with_table.cpu(), symbols)
    assert torch.equal(divided_search.cpu(), symbols)


@needs_stable_cuda
def test_cuda_coding_works_inside_a_non_default_stream():
    """Regression guard for the stream fix behind stage 6.

    The kernels take `ta::current_stream(tensor)` and launch with
    `<<<blocks, threads, 0, stream>>>`; the old code launched on stream 0, i.e.
    outside torch's stream semantics entirely. That the *argument* is torch's
    stream is a compile-time property (the POC measured it with a dedicated
    probe), so what this checks is that coding inside a side stream stays ordered
    with it and produces the same result.
    """
    params = _params(seed=6)
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        stream, decoded = _push_pop(_STABLE, params, device="cuda")
    side.synchronize()

    assert torch.equal(decoded.cpu(), params[4])
    assert _coded_bytes(stream.cpu()) == _coded_bytes(_push_pop(_STABLE, params)[0])


# ---------------------------------------------------------------------------
# 6. packaging: the WITH_STABLE_ABI build
# ---------------------------------------------------------------------------
# These are source-level checks: they pin the shape of the build switch without
# needing a compiler, and they are the only part of this module that runs when no
# prebuilt library is around (i.e. in a plain source checkout).

def _setup_source():
    return (ROOT / "setup.py").read_text(encoding="utf-8")


def _slice(source, start_marker, end_marker):
    """`source` from `start_marker` up to the next `end_marker` after it."""
    start = source.index(start_marker)
    return source[start:source.index(end_marker, start)]


def _code(source, start_marker, end_marker):
    """`_slice` with every docstring removed (only the code is asserted on)."""
    region = _slice(source, start_marker, end_marker)
    # odd indexes of a `"""` split are the docstrings, even ones the code
    return "".join(part for index, part in enumerate(region.split('"""'))
                   if index % 2 == 0)


def test_setup_py_builds_the_two_stable_abi_libraries():
    source = _setup_source()
    # `_stable_abi_platform_flags` / `_stable_abi_drop_libraries` belong to the
    # same region: they are where the flags and the library list come from, and
    # they are what lets one recipe build the wheel on Linux, macOS and Windows.
    # the docstrings explain *why* the flags below are absent, so only the code
    # is inspected
    code = _code(source, "def _stable_abi_platform_flags(",
                 "# Only import/build extensions")

    assert "torch_ans._C_cpu" in code, "the CPU library must always be built"
    assert "torch_ans._C_cuda" in code, "the CUDA library must be built too"
    for source_file in ("rans_cpu.cpp", "rans.cpp", "lib.cpp", "lib_cuda.cpp",
                        "rans_cuda.cu"):
        assert source_file in code, f"{source_file} is not part of the build"
    for define in ("TORCH_ANS_STABLE_ABI=1", "TORCH_TARGET_VERSION=",
                   "USE_CUDA=1", "WITH_CUDA=1"):
        assert define in code, f"the stable build must define {define}"
    assert "-Wl,--as-needed" in code, (
        "without --as-needed the -ltorch_python cpp_extension links becomes a "
        "DT_NEEDED entry, and the .so stops being CPython-agnostic")
    for forbidden in ("-fopenmp", "-march=native"):
        assert forbidden not in code, (
            f"{forbidden} must not be used for a prebuilt wheel "
            "(migration sections 4/5, pits 5 and 7)")


def test_stable_abi_flags_are_spelled_per_platform():
    """One recipe, three toolchains: GNU ld, ld64 and MSVC disagree."""
    source = _setup_source()
    code = _code(source, "def _stable_abi_platform_flags(",
                 "def get_stable_abi_extension_config(")

    assert code.count("-Wl,--as-needed") == 1, (
        "--as-needed is a GNU ld option: it must stay in the Linux branch only "
        "(ld64 aborts with 'unknown option', link.exe has no such concept)")
    assert "@loader_path/../torch/lib" in code, (
        "macOS has no --as-needed, so libtorch_cpu stays linked and dyld has to "
        "find it through an rpath into the *user's* torch install")
    assert "/O2" in code, "MSVC's optimisation level is /O2; -O3 is rejected"
    assert "-g0" in code, (
        "the manylinux image's CPython CFLAGS carry -g, and the auditwheel run "
        "that would strip the result is skipped on purpose: without -g0 the "
        "first published wheel shipped a 69.6 MB .so, 64.5 MB of it .debug_*")
    for platform_ in ("win32", "darwin"):
        assert f'sys.platform == "{platform_}"' in code, (
            f"{platform_} has no branch: it would get the Linux flags")
    # and the library that would freeze one CPython into the wheel is gone
    libraries = _slice(source, "def _stable_abi_drop_libraries(",
                       "def get_stable_abi_extension_config(")
    assert "torch_python" in libraries, (
        "cpp_extension links torch_python unconditionally; dropping it is what "
        "keeps the wheel CPython- and torch-build-independent")


def test_a_prebuilt_library_is_skipped_with_one_warning_on_an_old_torch(monkeypatch, tmp_path):
    """`torch < STABLE_ABI_MIN_TORCH`: say why the wheel is not used, once.

    This is the case the published wheel creates: the platform wheel is picked
    by pip whatever torch is installed, so the fallback has to be announced
    instead of silently compiling.
    """
    import warnings

    library = tmp_path / "_C_cpu.so"
    library.write_bytes(b"")
    monkeypatch.setattr(_stable_C, "_torch_version_tuple", lambda: (2, 9, 0))
    monkeypatch.setattr(_stable_C, "_find_extension",
                        lambda stem: library if stem == "_C_cpu" else None)
    monkeypatch.setattr(_stable_C, "_warned_old_torch", False)
    monkeypatch.setattr(_stable_C, "_loaded", {})

    with pytest.warns(RuntimeWarning, match="torch >= 2.10"):
        assert _stable_C.load_stable() is None
    # once: a CUDA request and a second interface must not repeat it
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _stable_C.load_stable(require_cuda=True) is None


def test_an_old_torch_without_a_prebuilt_library_says_nothing(monkeypatch):
    """A source install on an old torch has nothing to skip - stay quiet."""
    import warnings

    monkeypatch.setattr(_stable_C, "_torch_version_tuple", lambda: (2, 9, 0))
    monkeypatch.setattr(_stable_C, "_find_extension", lambda stem: None)
    monkeypatch.setattr(_stable_C, "_warned_old_torch", False)
    monkeypatch.setattr(_stable_C, "_loaded", {})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _stable_C.load_stable() is None


def test_the_opt_out_silences_the_old_torch_warning(monkeypatch, tmp_path):
    import warnings

    library = tmp_path / "_C_cpu.so"
    library.write_bytes(b"")
    monkeypatch.setenv(_stable_C.DISABLE_ENV_VAR, "1")
    monkeypatch.setattr(_stable_C, "_torch_version_tuple", lambda: (2, 9, 0))
    monkeypatch.setattr(_stable_C, "_find_extension",
                        lambda stem: library if stem == "_C_cpu" else None)
    monkeypatch.setattr(_stable_C, "_warned_old_torch", False)
    monkeypatch.setattr(_stable_C, "_loaded", {})

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _stable_C.load_stable() is None


def test_stable_abi_build_records_no_torch_version():
    """A stable-ABI .so targets torch >= 2.10: the ABI warning must stay silent."""
    source = _setup_source()
    metadata = _slice(source, "def write_build_metadata(",
                      "def get_extension_config(")
    assert "if not stable_abi:" in metadata, (
        "BUILD_TORCH_VERSION has to be conditional, or the cross-version warning "
        "the migration removes would come back")
    assert "BUILD_STABLE_ABI" in metadata


def test_cuda_registration_mirrors_the_cpu_family_list():
    """The CUDA library registers the same families, minus the host-only op."""
    bindings = (ROOT / "torch_ans" / "ans_bindings.hpp").read_text(encoding="utf-8")
    cpu = _slice(bindings, "inline void torch_ans_register_cpu_impls",
                 "#if TA_WITH_CUDA")
    cuda = _slice(bindings, "inline void torch_ans_register_cuda_impls",
                  "#endif  // TA_WITH_CUDA")

    assert "torch_ans_register_family_impls(m)" in cpu
    assert "torch_ans_register_family_impls(m)" in cuda, (
        "the CUDA library must register the shared family list, or the two "
        "dispatch keys drift apart")
    # rans_alias_build_table is host-only code (rans.cpp): it has no CUDA kernel
    # and its input is a CPU tensor by design.
    assert "rans_alias_build_table" in cpu
    assert "rans_alias_build_table" not in cuda
    # the two common operators that *do* have CUDA kernels
    assert "rans_pmf_to_quantized_cdf" in cuda
    assert "rans_build_inverse_cdf" in cuda


def test_lib_cuda_registers_implementations_only():
    """A second `def()` of the same operator is rejected by the dispatcher."""
    source = (ROOT / "torch_ans" / "lib_cuda.cpp").read_text(encoding="utf-8")
    assert "STABLE_TORCH_LIBRARY_IMPL(torch_ans, CUDA, m)" in source
    assert "STABLE_TORCH_LIBRARY(torch_ans" not in source, (
        "lib_cuda.cpp must not declare schemas: they belong to the CPU library")
    assert "torch_ans_register_cuda_impls" in source


def test_the_cross_version_matrix_starts_at_the_target():
    """The ABI gate has to start at the oldest torch the .so claims to serve."""
    import re

    workflow = (ROOT / ".github" / "workflows" / "cibuildwheel.yml").read_text(
        encoding="utf-8")
    job = _slice(workflow, "  cross-version-load:", "  cross-version-compare:")
    versions = re.findall(r"\"(\d+\.\d+\.\d+)\"", job)
    assert len(versions) >= 2, (
        f"the cross-version job has to try more than one torch:\n{job}")
    floor = tuple(int(part) for part in versions[0].split("."))[:2]
    assert floor == _stable_C.STABLE_ABI_MIN_TORCH, (
        f"the cross-version matrix starts at {versions[0]}, but the prebuilt .so "
        f"targets {'.'.join(str(part) for part in _stable_C.STABLE_ABI_MIN_TORCH)}"
        " (TORCH_TARGET_VERSION in setup.py / STABLE_ABI_MIN_TORCH)")


def _wheel_checker():
    """`scripts/check_stable_abi_wheel.py` as a module (CI runs it as a script)."""
    import importlib.util

    path = ROOT / "scripts" / "check_stable_abi_wheel.py"
    spec = importlib.util.spec_from_file_location("torch_ans_wheel_checker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_wheel_checker_rejects_what_the_stable_abi_promise_forbids():
    checker = _wheel_checker()
    patterns = checker.FORBIDDEN_SYMBOL_PATTERNS
    for symbol in ("_ZN2at6TensorD1Ev", "_ZNK3c1010TensorImplD2Ev",
                   "_ZN5torch8python10somethingEv", "PyExc_RuntimeError"):
        assert any(pattern.match(symbol) for pattern in patterns), (
            f"{symbol} would tie the wheel to one torch / CPython but is accepted")
    for symbol in ("aoti_torch_empty", "torch_call_dispatcher", "cudaMalloc",
                   "_ZNSt6vectorIiSaIiEE"):
        assert not any(pattern.match(symbol) for pattern in patterns), (
            f"{symbol} is a stable-ABI (or unrelated) symbol but is rejected")
    assert "libtorch_python" in checker.FORBIDDEN_NEEDED
    assert "libpython" in checker.FORBIDDEN_NEEDED


def test_the_wheel_checker_fails_a_wheel_without_the_prebuilt_library(tmp_path):
    import subprocess
    import sys
    import zipfile

    wheel = tmp_path / "torch_ans-0.4.0-cp311-cp311-linux_x86_64.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("torch_ans/__init__.py", "")

    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_stable_abi_wheel.py"), str(wheel)],
        capture_output=True, text=True)
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "no prebuilt CPU library" in output, output
    # the CPython tag is not what the stable ABI promises either
    assert "py3-none" in output, output


def test_pyproject_drops_pybind11():
    """torch ships the pybind11 headers; the stable build needs none at all."""
    tomllib = pytest.importorskip("tomllib", reason="needs Python >= 3.11")
    with open(ROOT / "pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)
    assert "pybind11" not in data["build-system"]["requires"]
    assert "pybind11" not in data["project"]["dependencies"]
    assert "ninja" in data["project"]["dependencies"], (
        "ninja is still needed by the runtime JIT path")
