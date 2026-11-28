"""MAYO multivariate signature verification (MAYO-1, MAYO-2, MAYO-3, MAYO-5).

    from triton_crypto.mayo import Verifier
    v = Verifier(cpk)                      # auto-detects MAYO-1/2/3/5 from compact pk length
    ok = v.verify(messages, signatures)    # -> numpy bool array

Public-key expansion (`AES-128-CTR` of `pk_seed` into `P1, P2`, concatenated with `P3`) runs once
at `Verifier` construction and packs `P` into `[n, n, m_u32_words]` `uint32` words (8 GF(16)
nibbles per word). Verification runs `SHAKE256(M)` and `SHAKE256(md || salt)` on-device via
`primitives.sponge`, evaluates `SPS = S * P * S^T` via fused packed-nibble GF(16) arithmetic
(`_mayo_sps_kernel`), and folds mod `f(z)` + checks against `t` in `_mayo_fold_check_kernel`.
"""

from __future__ import annotations

from .params import (
    ALL_SETS,
    BY_NAME,
    MAYO_1,
    MAYO_2,
    MAYO_3,
    MAYO_5,
    MayoParams,
    from_pk_len,
)
from .ref import decode_nibbles, encode_nibbles, eval_public_map_np, expand_pk_np, verify_np
from .verify import Verifier, verify_batch

__all__ = [
    "ALL_SETS",
    "BY_NAME",
    "MAYO_1",
    "MAYO_2",
    "MAYO_3",
    "MAYO_5",
    "MayoParams",
    "Verifier",
    "decode_nibbles",
    "encode_nibbles",
    "eval_public_map_np",
    "expand_pk_np",
    "from_pk_len",
    "verify_batch",
    "verify_np",
]
