"""Packaging and public-API contracts: the things a user hits before any crypto runs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

import triton_crypto

ROOT = Path(__file__).parent.parent


def test_version_matches_pyproject():
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert triton_crypto.__version__ == meta["project"]["version"]


def test_lms_is_a_package():
    """Without lms/__init__.py, setuptools.find_packages() drops the subpackage from wheels."""
    import setuptools

    packages = set(setuptools.find_packages(where=str(ROOT)))
    assert "triton_crypto.lms" in packages
    assert {
        "triton_crypto.mlkem",
        "triton_crypto.mlkem.kernels",
        "triton_crypto.mlkem.reference",
        "triton_crypto.primitives",
        "triton_crypto.primitives.ntt",
    } <= packages
    # The primitive modules are plain .py files, but they must exist in a wheel too.
    from pathlib import Path as _Path

    root = _Path(triton_crypto.__file__).parent
    for module in ("fields.py", "chacha20.py", "ntt/__init__.py", "ntt/domain.py"):
        assert (root / "primitives" / module).is_file(), f"primitives/{module} missing"

    from triton_crypto.lms import ALL_SETS, LMSSet, Verifier, from_name, get, verify_batch

    assert ALL_SETS and LMSSet and Verifier and verify_batch
    s = from_name("LMS_SHA256_M32_H10:LMOTS_SHA256_N32_W4")
    assert get(s.lms_type, s.ots_type) == s


@pytest.mark.parametrize(
    "names,module",
    [
        (["Signer", "sign_batch"], "triton_crypto.mldsa"),
        (["Verifier", "verify_batch"], "triton_crypto.slhdsa"),
        (["Verifier", "verify_batch"], "triton_crypto.lms"),
    ],
)
def test_public_exports(names, module):
    import importlib

    mod = importlib.import_module(module)
    for name in names:
        assert name in mod.__all__, f"{name} missing from {module}.__all__"
        assert callable(getattr(mod, name))


def test_readme_imports_execute():
    """Every `from triton_crypto... import ...` line in the README must be valid."""
    lines = []
    for raw in (ROOT / "README.md").read_text(encoding="utf-8").splitlines():
        line = raw.strip().removeprefix(">>>").strip()
        if re.match(r"^(from triton_crypto[\w.]* import [\w, ]+|import triton_crypto)$", line):
            lines.append(line)
    assert lines, "no import lines found in README"
    for line in lines:
        exec(compile(line, "README.md", "exec"), {})


def test_input_normalization_contract():
    from triton_crypto._inputs import as_batch_bytes, as_bytes_list

    assert as_bytes_list(b"abc") == [b"abc"]
    assert as_bytes_list([b"a", b"b"]) == [b"a", b"b"]
    assert as_batch_bytes(b"abc", 3).shape == (1, 3)
    assert as_batch_bytes([b"ab", b"cd"], 2).tolist() == [[97, 98], [99, 100]]
    assert as_batch_bytes([], 4).shape == (0, 4)

    with pytest.raises(ValueError, match="32 bytes"):
        as_batch_bytes([b"a" * 31, b"b" * 33], 32)
    with pytest.raises(ValueError, match="differing lengths"):
        as_batch_bytes([b"a", b"bb"])
    with pytest.raises(TypeError, match="bytes-like"):
        as_bytes_list("not bytes")

    torch = pytest.importorskip("torch")
    t = torch.arange(6, dtype=torch.uint8).reshape(2, 3)
    assert as_bytes_list(t) == [b"\x00\x01\x02", b"\x03\x04\x05"]
    assert as_batch_bytes(t, 3).shape == (2, 3)


def test_generated_source_cache_permissions(tmp_path):
    pytest.importorskip("triton")
    from triton_crypto._codegen import _cache_dir
    from triton_crypto.primitives.sponge import get_sponge

    get_sponge(136, 256)  # a small, fast instance that will already be cached
    d = _cache_dir()
    assert not (d.stat().st_mode & 0o077), oct(d.stat().st_mode)
    for f in d.glob("*.py"):
        assert not (f.stat().st_mode & 0o077), (f, oct(f.stat().st_mode))


def test_deprecated_module_paths_point_at_the_new_ones():
    """Old root paths stay valid for one release, but must be the same objects."""
    from triton_crypto.mlkem.kernels.ntt import byte_encode12 as new_be
    from triton_crypto.mlkem.pipeline import Decapsulator as NewDec
    from triton_crypto.mlkem.reference.sampling_cpu import cbd as new_cbd

    with pytest.warns(DeprecationWarning):
        import triton_crypto.kernels.ntt as old_kernels

        import triton_crypto.pipeline as old_pipeline
        import triton_crypto.sampling_cpu as old_sampling

    assert old_pipeline.Decapsulator is NewDec
    assert old_kernels.byte_encode12 is new_be
    assert old_sampling.cbd is new_cbd
