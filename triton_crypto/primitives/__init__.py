"""Scheme-independent GPU primitives.

Two blocks of mathematics are needed by every scheme in this package, and both were
originally written for ML-KEM alone inside the old `triton_crypto.kernels` package (now
`triton_crypto.mlkem.kernels`):

    primitives.keccak   Keccak-f[1600] permutation — the permutation itself, plus a
                        pure-Python reference
    primitives.sponge   the sponge around it: pad10*1, multi-block absorb, multi-block
                        squeeze, and the named SHA3/SHAKE instances
    primitives.ntt      negacyclic NTT over Z_q[X]/(X^N+1), parameterised by the modulus
                        and the 2N-th root of unity, forward *and* inverse

ML-KEM (`triton_crypto.mlkem`), ML-DSA (`triton_crypto.mldsa`), SLH-DSA
(`triton_crypto.slhdsa`) and LMS/HSS (`triton_crypto.lms`) are thin layers on top of
these; they keep only what is scheme-specific — ML-KEM's base-case multiply and Toeplitz
GEMMs, ML-DSA's SampleInBall / Decompose, FIPS 205's thash, LMS's Winternitz chains.

The two domains in use:

    >>> from triton_crypto.primitives import KYBER, DILITHIUM
    >>> KYBER.q, KYBER.zeta, KYBER.wide
    (3329, 17, False)
    >>> DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.wide
    (8380417, 1753, True)
"""

from __future__ import annotations

from . import keccak, ntt, sha2, sponge
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
from .sha2 import (
    IV256,
    IV512,
    K256,
    K512,
    compress_np,
    hash_np,
    sha_absorb,
)
from .sponge import (
    DOMAIN_SHA3,
    DOMAIN_SHAKE,
    HASHES,
    SHA3_224,
    SHA3_256,
    SHA3_384,
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
    "IV256",
    "IV512",
    "K256",
    "K512",
    "KYBER",
    "SHA3_224",
    "SHA3_256",
    "SHA3_384",
    "SHA3_512",
    "SHAKE128",
    "SHAKE256",
    "HashFn",
    "NTTDomain",
    "compress_np",
    "get_sponge",
    "hash_np",
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
    "sha2",
    "sha_absorb",
    "sponge",
    "sponge_padded",
]
