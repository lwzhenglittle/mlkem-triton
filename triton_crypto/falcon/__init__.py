"""Falcon-512 signature verification (FIPS 206 draft / the NIST additional-signatures spec).

Verification only: signing needs Falcon's floating-point FFT and Gaussian sampler, which the
project deliberately does not port (Triton does not expose FMA fusion control, so a cross-backend
bit-exact signature is not a property this code base can claim). Verification is pure integer
arithmetic, so it is bit-exact by construction.

    from triton_crypto.falcon import Verifier
    v = Verifier(pk)                       # 897 bytes, header 0x09
    ok = v.verify(messages, signatures)    # -> numpy bool array

The transform reuses `primitives.ntt` with q = 12289, N = 512 -- the parameterisation needed
no new kernel. The signature decoder (`comp_decode`) runs on the host (35x faster than a device
scan kernel), while `HashToPoint` is dual-path (device kernel for uniform-length batches, host
fallback for mixed lengths); see `verify.py` for the device-only entry point.
"""

from __future__ import annotations

from .params import FALCON_512, FALCON_PADDED_512, FalconParams
from .ref import comp_decode, hash_to_point, modq_decode
from .ref import verify as verify_np
from .verify import Verifier

__all__ = [
    "FALCON_512",
    "FALCON_PADDED_512",
    "FalconParams",
    "Verifier",
    "comp_decode",
    "hash_to_point",
    "modq_decode",
    "verify_np",
]
