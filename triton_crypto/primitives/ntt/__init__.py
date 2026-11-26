"""Generic negacyclic NTT over Z_q[X]/(X^N+1): domains, kernel and numpy reference.

Three modules, because they change for different reasons:

    domain.py   `NTTDomain` (q, N, ζ, layers) + twiddle tables — pure data
    kernel.py   the in-register Triton kernel for N <= 1024, forward and inverse
    ref.py      `ntt_np` / `intt_np` / `mul_np` / `negacyclic_conv_np`

A 2^20-point transform is a different algorithm (multi-pass, global memory), not a bigger
tile: `large.py` is that entry point -- one launch per stage, `LargeNTT`.

Primitives:
    NTTDomain      q, N, ζ, layers  (+ cached twiddle tables, both directions)
    ntt_np/intt_np numpy reference (same table layout as the kernel)
    ntt_torch      batched Triton kernel: [P, N] int32 -> [P, N] int32
    mul_np         pointwise (Hadamard) product mod q
    negacyclic_conv_np  schoolbook O(N²) product, used to test the transform itself

Instances: `KYBER` (q = 3329), `DILITHIUM` (q = 8380417), `BABYBEAR_NTT` (q = 2^31−2^27+1).
"""

from __future__ import annotations

from .domain import (
    BABYBEAR_NTT,
    DILITHIUM,
    KYBER,
    NTTDomain,
    bitrev,
    mldsa_domain,
    mlkem_domain,
)
from .kernel import intt_torch, ntt_torch
from .large import LargeNTT, large_ntt_np, large_ntt_torch
from .ref import intt_np, mul_np, negacyclic_conv_np, ntt_np

__all__ = [
    "BABYBEAR_NTT",
    "DILITHIUM",
    "KYBER",
    "LargeNTT",
    "NTTDomain",
    "bitrev",
    "intt_np",
    "intt_torch",
    "large_ntt_np",
    "large_ntt_torch",
    "mldsa_domain",
    "mlkem_domain",
    "mul_np",
    "negacyclic_conv_np",
    "ntt_np",
    "ntt_torch",
]
