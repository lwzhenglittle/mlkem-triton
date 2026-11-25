"""Reference-implementation wrapper: uniform ref_keygen / ref_encaps / ref_decaps API.

Backed by **kyber-py** (a third-party FIPS 203 implementation, optional dependency):
    pip install mlkem-triton[ref]        # or: pip install kyber-py

It exposes deterministic internal entry points that line up exactly with the ACVP vectors:
    _keygen_internal(d, z) -> (ek, dk)
    _encaps_internal(ek, m) -> (K, c)
    _decaps_internal(dk, c) -> K

**Not required**: the primary correctness gate uses the official NIST test vectors in
`data/acvp/` (`mlkem_triton.acvp`) and needs no third-party implementation. kyber-py is
only needed when you want to randomly generate keypairs for round-trip tests.
"""

from __future__ import annotations

import hashlib
import os

from .ntt_ref import byte_decode
from .params import PARAMS, Q

_KEM = None


def _kems() -> dict:
    """Import kyber-py lazily; if it is missing, raise an error you can act on directly."""
    global _KEM
    if _KEM is None:
        try:
            from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "This feature requires kyber-py (an optional dependency). Install it:\n"
                "    pip install mlkem-triton[ref]\n"
                "If you only want correctness verification, use mlkem_triton.acvp + the\n"
                "official NIST test vectors under data/acvp/; kyber-py is not needed."
            ) from e
        _KEM = {
            "ML-KEM-512": ML_KEM_512,
            "ML-KEM-768": ML_KEM_768,
            "ML-KEM-1024": ML_KEM_1024,
        }
    return _KEM


def kem(ps: str):
    kems = _kems()
    if ps not in kems:
        raise KeyError(f"unknown parameter set {ps}")
    return kems[ps]


# ---------------- public API ----------------


def ref_keygen(ps: str, d: bytes | None = None, z: bytes | None = None):
    """Return (ek, dk). Deterministic when d/z are given."""
    d = os.urandom(32) if d is None else d
    z = os.urandom(32) if z is None else z
    return kem(ps)._keygen_internal(d, z)


def ref_encaps(ps: str, ek: bytes, m: bytes | None = None):
    """Return (K, c); note that K comes first."""
    m = os.urandom(32) if m is None else m
    return kem(ps)._encaps_internal(ek, m)


def ref_decaps(ps: str, dk: bytes, c: bytes) -> bytes:
    """Return the shared secret K (implicit rejection included)."""
    return kem(ps)._decaps_internal(dk, c)


# ---------------- FIPS 203 §7 input validation ----------------


def decaps_key_check(ps: str, dk: bytes) -> bool:
    """§7.3 decapsulation key check: length + embedded H(ek) matches ek."""
    p = PARAMS[ps]
    if len(dk) != p.dk_size:
        return False
    ek = dk[384 * p.k : 768 * p.k + 32]
    h = dk[768 * p.k + 32 : 768 * p.k + 64]
    return hashlib.sha3_256(ek).digest() == h


def encaps_key_check(ps: str, ek: bytes) -> bool:
    """§7.2 encapsulation key check: length + modulus check (every 12-bit
    coefficient must be < q)."""
    p = PARAMS[ps]
    if len(ek) != p.ek_size:
        return False
    # the t̂ part of ek is k polynomials, 384 bytes each, each needs its own ByteDecode12
    ok = True
    for i in range(p.k):
        vals = byte_decode(ek[384 * i : 384 * (i + 1)], 12)
        ok = ok and bool((vals < Q).all())
    return ok


# ---------------- helpers ----------------


def split_dk(ps: str, dk: bytes):
    """Split dk as in FIPS 203: dk_PKE ‖ ek ‖ H(ek) ‖ z."""
    p = PARAMS[ps]
    k = p.k
    return {
        "dk_pke": dk[: 384 * k],
        "ek": dk[384 * k : 768 * k + 32],
        "h_ek": dk[768 * k + 32 : 768 * k + 64],
        "z": dk[768 * k + 64 : 768 * k + 96],
    }


def implicit_reject_key(ps: str, dk: bytes, c: bytes) -> bytes:
    """K̄ = J(z ‖ c) = SHAKE256(z‖c, 32)."""
    z = split_dk(ps, dk)["z"]
    return hashlib.shake_256(z + c).digest(32)


__all__ = [
    "decaps_key_check",
    "encaps_key_check",
    "implicit_reject_key",
    "kem",
    "ref_decaps",
    "ref_encaps",
    "ref_keygen",
    "split_dk",
]
