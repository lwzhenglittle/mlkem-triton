"""ML-DSA (FIPS 204) signing and verification on the GPU, plus a numpy reference.

    from triton_crypto.mldsa import Signer, Verifier
    v = Verifier("ML-DSA-65", pk)             # pk preprocessing, once
    ok = v.verify(messages, signatures)       # -> np.ndarray[bool], batch
    sigs = Signer("ML-DSA-65", sk).sign(messages)

The GPU side implements signing (Alg. 7, with the rejection loop batched by speculation)
and verification (Alg. 3). The numpy reference in `.ref` implements all of FIPS 204 —
keygen, signing and verification — and is byte-exact against the ACVP keyGen/sigGen
vectors; it is what the kernel tests compare against.

What is reused from the shared layer: `primitives.sponge` for every SHAKE call and
`primitives.ntt` for the transform over q = 8380417. What is ML-DSA-specific and lives
here: SampleInBall, Decompose/UseHint, and the bit-packings of z / w1 / hints.
"""

from __future__ import annotations

from .params import PARAMS, D, DSAParams, N, Q
from .ref import MLDSA, PREHASH, Ref, pack_bits, unpack_bits
from .sign import Signer, sign_batch
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
    "Signer",
    "Verifier",
    "pack_bits",
    "sign_batch",
    "unpack_bits",
    "verify_batch",
]
