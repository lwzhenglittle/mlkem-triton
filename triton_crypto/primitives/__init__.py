"""Scheme-independent GPU primitives.

Two blocks of mathematics are needed by *every* lattice scheme in the NIST PQC family,
and both were originally written for ML-KEM alone inside `triton_crypto.kernels`:

    primitives.keccak   Keccak-f[1600] permutation — the permutation itself, plus a
                        pure-Python reference
    primitives.sponge   the sponge around it: pad10*1, multi-block absorb, multi-block
                        squeeze, and the named SHA3/SHAKE instances
    primitives.ntt      negacyclic NTT over Z_q[X]/(X^N+1), parameterised by the modulus
                        and the 2N-th root of unity, forward *and* inverse

ML-KEM (`triton_crypto.kernels.*`, `triton_crypto.pipeline`) and ML-DSA
(`triton_crypto.mldsa`) are both thin layers on top of these; they keep only what is
scheme-specific — ML-KEM's base-case multiply in the quadratic extension, ML-DSA's
SampleInBall / Decompose / hint encoding.

The two domains in use:

    >>> from triton_crypto.primitives import KYBER, DILITHIUM
    >>> KYBER.q, KYBER.zeta, KYBER.wide
    (3329, 17, False)
    >>> DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.wide
    (8380417, 1753, True)
"""

from __future__ import annotations

from . import keccak, ntt, sponge
from .ntt import (
    DILITHIUM,
    KYBER,
    NTTDomain,
    intt_np,
    intt_torch,
    mul_np,
    negacyclic_conv_np,
    ntt_np,
    ntt_torch,
)
from .sponge import (
    DOMAIN_SHA3,
    DOMAIN_SHAKE,
    HASHES,
    SHA3_256,
    SHA3_512,
    SHAKE128,
    SHAKE256,
    HashFn,
    get_sponge,
    hash_np_batch,
    pad10star1,
    pad_batch,
    pad_torch,
    padded_len,
    sponge_padded,
)

__all__ = [
    "DILITHIUM",
    "DOMAIN_SHA3",
    "DOMAIN_SHAKE",
    "HASHES",
    "KYBER",
    "SHA3_256",
    "SHA3_512",
    "SHAKE128",
    "SHAKE256",
    "HashFn",
    "NTTDomain",
    "get_sponge",
    "hash_np_batch",
    "intt_np",
    "intt_torch",
    "keccak",
    "mul_np",
    "negacyclic_conv_np",
    "ntt",
    "ntt_np",
    "ntt_torch",
    "pad10star1",
    "pad_batch",
    "pad_torch",
    "padded_len",
    "sponge",
    "sponge_padded",
]
