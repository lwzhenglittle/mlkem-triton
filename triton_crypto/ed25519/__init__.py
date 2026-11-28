"""Ed25519 batched signature verification (RFC 8032) and X25519 Diffie-Hellman (RFC 7748).

    from triton_crypto.ed25519 import Verifier, verify_batch, x25519_torch, x25519_base_torch

    v = Verifier(pk)                       # 32-byte Ed25519 public key, precomputed once
    ok = v.verify(messages, signatures)    # -> numpy bool array

    shared = x25519_torch(scalars, peers)  # [B, 32] uint8 -> [B, 32] uint8

All finite-field arithmetic over F_{2^255 - 19} uses a signed 10-limb radix-2^25.5 representation
in `int64` registers (`ref10` layout), requiring no 128-bit integer or `umulhi` intrinsics so the
exact same pure-Triton kernels compile on Intel XPU, NVIDIA CUDA, and ROCm/DCU. Ed25519
verification computes `SHA-512(R || A || M)` on-device via `primitives.sha2` (for uniform-length
batches), reduces mod L on-device, and evaluates `s*B - k*A == R` via branchless 253-step Shamir's
trick with complete twisted Edwards addition (`add-2008-hwcd-3`).
"""

from __future__ import annotations

from .ref import (
    CURVE_D,
    CURVE_L,
    CURVE_P,
    ed25519_public_key_np,
    ed25519_sign_np,
    ed25519_verify_np,
    x25519_base_np,
    x25519_np,
)
from .verify import (
    Verifier,
    verify_batch,
    x25519_base_torch,
    x25519_torch,
)

__all__ = [
    "CURVE_D",
    "CURVE_L",
    "CURVE_P",
    "Verifier",
    "ed25519_public_key_np",
    "ed25519_sign_np",
    "ed25519_verify_np",
    "verify_batch",
    "x25519_base_np",
    "x25519_base_torch",
    "x25519_np",
    "x25519_torch",
]
