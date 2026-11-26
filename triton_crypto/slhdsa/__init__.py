"""SLH-DSA (FIPS 205) — signature verification, and the numpy reference behind it.

Status: **device verification plus a numpy reference**, for all twelve parameter sets
(SHA-2 and SHAKE, pure / pre-hash / internal interfaces). The device path is checked
against the NIST ACVP vectors; `SLH` is the pure-Python reference.

    from triton_crypto.slhdsa import SLH, PARAMS, Verifier

    v = Verifier("SLH-DSA-SHAKE-128f", pk)     # device batch verification
    ok = v.verify(messages, signatures)        # -> np.ndarray[bool]
    ref = SLH("SLH-DSA-SHAKE-128f")           # ~7k hashes per verification, pure Python
    ok1 = ref.verify(message, signature, public_key, ctx=b"", prehash=None)
"""

from __future__ import annotations

from .params import PARAMS, SLHParams, get
from .ref import OID, PH, SLH, Adrs
from .verify import Verifier, verify_batch

__all__ = [
    "OID",
    "PARAMS",
    "PH",
    "SLH",
    "Adrs",
    "SLHParams",
    "Verifier",
    "get",
    "verify_batch",
]
