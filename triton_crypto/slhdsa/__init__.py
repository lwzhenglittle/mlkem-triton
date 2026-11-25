"""SLH-DSA (FIPS 205) — signature verification, and the numpy reference behind it.

Status: **reference implementation and vectors first, device kernels in progress.** What is here
is the whole verification semantics (all 12 parameter sets, SHA-2 and SHAKE, the pure / pre-hash /
internal interfaces) checked against the NIST ACVP vectors — see `tests/test_slhdsa_ref.py`.

    from triton_crypto.slhdsa import SLH, PARAMS

    v = SLH("SLH-DSA-SHAKE-128f")           # ~7k hashes per verification, pure Python
    ok = v.verify(message, signature, public_key, ctx=b"", prehash=None)
"""

from __future__ import annotations

from .params import PARAMS, SLHParams, get
from .ref import OID, PH, SLH, Adrs

__all__ = ["OID", "PARAMS", "PH", "SLH", "Adrs", "SLHParams", "get"]
