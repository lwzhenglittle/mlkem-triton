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

import inspect

import pytest

from triton_crypto import Decapsulator, Encapsulator, generate_keypairs
from triton_crypto.primitives import BABYBEAR_NTT, DILITHIUM, KYBER


def _params(fn) -> list[str]:
    return [p for p in inspect.signature(fn).parameters]


def test_mlkem_block():
    """README block 0: `Decapsulator(scheme, dk)`, `Encapsulator(scheme, ek)`,
    `generate_keypairs(scheme, n)`."""
    assert _params(Decapsulator)[:2] == ["ps", "dk"] or _params(Decapsulator)[:2] == ["scheme", "dk"]
    assert _params(Encapsulator)[:2] == ["ps", "ek"] or _params(Encapsulator)[:2] == ["scheme", "ek"]
    assert _params(generate_keypairs)[0] in ("ps", "scheme")
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
