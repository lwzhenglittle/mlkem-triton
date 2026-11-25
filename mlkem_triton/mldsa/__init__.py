"""ML-DSA (FIPS 204) signature verification on the GPU, plus a numpy reference.

    from mlkem_triton.mldsa import Verifier
    v = Verifier("ML-DSA-65", pk)             # pk preprocessing, once
    ok = v.verify(messages, signatures)       # -> np.ndarray[bool], batch

The GPU side implements **verification only** (Alg. 3): signing needs the rejection
loop and secret-key material, which is a different (and much less batch-friendly)
problem. The numpy reference in `.ref` implements all of FIPS 204 — keygen, signing and
verification — and is byte-exact against the ACVP keyGen/sigGen vectors; it is what the
kernel tests compare against.

What is reused from the shared layer: `primitives.sponge` for every SHAKE call and
`primitives.ntt` for the transform over q = 8380417. What is ML-DSA-specific and lives
here: SampleInBall, Decompose/UseHint, and the bit-packings of z / w1 / hints.
"""

from __future__ import annotations

from .params import PARAMS, D, DSAParams, N, Q
from .ref import MLDSA, PREHASH, Ref, pack_bits, unpack_bits
from .verify import Verifier, verify_batch

__all__ = [
    "MLDSA",
    "PARAMS",
    "PREHASH",
    "D",
    "DSAParams",
    "N",
    "Q",
    "Ref",
    "Verifier",
    "pack_bits",
    "unpack_bits",
    "verify_batch",
]
