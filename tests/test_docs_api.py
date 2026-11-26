"""The README's code blocks, as executable claims.

Every snippet in `README.md` is illustrative -- it references `dk`, `pk`, `messages` that the
reader supplies -- but the *shape* of every call in it is a promise: these names exist, these
constructors take these arguments in this order, and these constants have these values. A rename
that leaves the README untouched is exactly the kind of drift a documentation test catches and a
test suite does not.

The check is `inspect.signature` rather than execution, so it stays cheap and needs no keys; the
constants in block 4 are checked by value because they are quoted.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from triton_crypto import Decapsulator, Encapsulator, generate_keypairs
from triton_crypto.primitives import BABYBEAR_NTT, DILITHIUM, KYBER


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_mlkem_block():
    """README block 0: `Decapsulator(scheme, dk)`, `Encapsulator(scheme, ek)`,
    `generate_keypairs(scheme, n)`."""
    # the README writes `Decapsulator("ML-KEM-768", dk)`, so the two positional names matter
    assert _params(Decapsulator)[:2] == ["ps", "dk"]
    assert _params(Encapsulator)[:2] == ["ps", "ek"]
    assert _params(generate_keypairs)[:2] == ["ps", "count"]
    assert hasattr(Decapsulator, "decaps")
    assert hasattr(Encapsulator, "encapsulate")


def test_mldsa_block():
    """README block 1: `Verifier(ps, pk)` with `.verify`, `Signer(ps, sk)` with `.sign(.., ctx,
    prehash, rnd)`."""
    from triton_crypto.mldsa import Signer, Verifier

    assert _params(Verifier)[0] in ("ps", "scheme")
    assert "ctx" in _params(Signer.sign)
    assert "prehash" in _params(Signer.sign)
    assert "rnd" in _params(Signer.sign)


@pytest.mark.parametrize(("module", "cls"), [("slhdsa", "Verifier"), ("lms", "Verifier")])
def test_other_scheme_blocks(module, cls):
    """README blocks 2 and 3: the two verifiers the snippets import."""
    mod = __import__(f"triton_crypto.{module}", fromlist=[cls])
    target = getattr(mod, cls)
    assert hasattr(target, "verify")


def test_primitives_block():
    """README block 4, including the two constant tuples it quotes by value."""
    assert (KYBER.q, KYBER.zeta, KYBER.layers, KYBER.wide) == (3329, 17, 7, False)
    assert (DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.layers, DILITHIUM.wide) == (
        8380417,
        1753,
        8,
        True,
    )
    assert BABYBEAR_NTT.q == 2013265921

    from triton_crypto.primitives import SHA3_256, chacha20_keystream, hash_np_batch

    # the README used to show `(key, nonce, nbytes)`, but the API is batched and takes plurals --
    # one key broadcasts against many nonces, which a singular name would hide
    assert _params(chacha20_keystream)[:3] == ["keys", "nonces", "nbytes"]
    assert "counters" in _params(chacha20_keystream)
    assert _params(hash_np_batch)[:2] == ["msgs", "hf"]
    assert SHA3_256.rate == 136


def test_the_package_parses_as_python_3_10():
    """`pyproject.toml` claims `requires-python = ">=3.10"`; this repository runs 3.14.

    The claim is only checked by the grammar version here -- no 3.10 interpreter is available on
    this machine -- so it catches syntax, not standard-library additions. That is the failure mode
    worth catching: a PEP 695 type parameter or a same-quote f-string nests in 3.12 and produces a
    `SyntaxError` for the very users the metadata promises.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    bad = []
    paths = list((root / "triton_crypto").rglob("*.py")) + list((root / "benchmarks").glob("*.py"))
    for p in paths:
        try:
            ast.parse(p.read_text(), filename=str(p), feature_version=(3, 10))
        except SyntaxError as exc:
            bad.append(f"{p.relative_to(root)}:{exc.lineno} {exc.msg}")
    assert not bad, "syntax newer than the declared floor: " + "; ".join(bad)
