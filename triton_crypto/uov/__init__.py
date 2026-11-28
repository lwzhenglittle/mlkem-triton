"""UOV (Unbalanced Oil and Vinegar) multivariate signature verification (OV-Is, OV-Ip, OV-III, OV-V).

    from triton_crypto.uov import Verifier
    v = Verifier(pk)                       # auto-detects OV-Is/Ip/III/V (classic or -pkc)
    ok = v.verify(messages, signatures)    # -> numpy bool array

Supports all four NIST Round 2 parameter sets over both GF(16) (`OV-Is`) and GF(256)
(`OV-Ip`, `OV-III`, `OV-V`), accepting either uncompressed classic public keys or `-pkc`
compressed public keys (`pk_seed || P3`, expanded once via `AES-128-CTR` on construction).
Verification runs `SHAKE256(M || salt)` on-device via `primitives.sponge` and evaluates the
upper-triangular quadratic public map via packed `uint32` finite-field kernels (`_uov_eval_gf16_kernel`
and `_uov_eval_gf256_kernel`, 8 GF(16) nibbles or 4 GF(256) bytes per `uint32` lane).
"""

from __future__ import annotations

from .params import (
    ALL_SETS,
    BY_NAME,
    OV_III,
    OV_IP,
    OV_IS,
    OV_V,
    UOVParams,
    from_pk_len,
)
from .ref import eval_public_map_np, expand_pk_np, verify_np
from .verify import Verifier, verify_batch

__all__ = [
    "ALL_SETS",
    "BY_NAME",
    "OV_III",
    "OV_IP",
    "OV_IS",
    "OV_V",
    "UOVParams",
    "Verifier",
    "eval_public_map_np",
    "expand_pk_np",
    "from_pk_len",
    "verify_batch",
    "verify_np",
]
