"""Falcon-512 and Falcon-1024 signature verification (FIPS 206 draft).

Verification only: signing needs Falcon's floating-point FFT and Gaussian sampler, which the
project deliberately does not port (Triton does not expose FMA fusion control, so a cross-backend
bit-exact signature is not a property this code base can claim). Verification is pure integer
arithmetic, so it is bit-exact by construction.

    from triton_crypto.falcon import Verifier
    v = Verifier(pk)                       # Falcon-512 (897 B) or Falcon-1024 (1793 B)
    ok = v.verify(messages, signatures)    # -> numpy bool array

The transform reuses `primitives.ntt` with q = 12289, N = 512 or 1024 -- the parameterisation
needed no new NTT kernel. Both the variable-length prefix decoder (`_comp_decode_kernel`, using
a 64-bit per-lane register bit reservoir and branchless 16-bit CLZ over transposed uint32 words)
and `HashToPoint` (for uniform-length batches) run on the device alongside the NTT/norm core,
avoiding host decode and CPU round-trips.
"""

from __future__ import annotations

from .params import (
    FALCON_512,
    FALCON_1024,
    FALCON_PADDED_512,
    FALCON_PADDED_1024,
    FalconParams,
)
from .ref import comp_decode, hash_to_point, modq_decode
from .ref import verify as verify_np
from .verify import Verifier, comp_decode_batch, comp_decode_batch_torch

__all__ = [
    "FALCON_512",
    "FALCON_1024",
    "FALCON_PADDED_512",
    "FALCON_PADDED_1024",
    "FalconParams",
    "Verifier",
    "comp_decode",
    "comp_decode_batch",
    "comp_decode_batch_torch",
    "hash_to_point",
    "modq_decode",
    "verify_np",
]
