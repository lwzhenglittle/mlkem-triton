"""Scheme-independent GPU primitives.

Two blocks of mathematics are needed by every scheme in this package, and both were
originally written for ML-KEM alone inside the old `triton_crypto.kernels` package (now
`triton_crypto.mlkem.kernels`):

    primitives.keccak   Keccak-f[1600] permutation — the permutation itself, plus a
                        pure-Python reference
    primitives.sponge   the sponge around it: pad10*1, multi-block absorb, multi-block
                        squeeze, and the named SHA3/SHAKE instances
    primitives.ntt      negacyclic NTT over Z_q[X]/(X^N+1), parameterised by the modulus
                        and the 2N-th root of unity, forward *and* inverse; a package
                        (`domain` / `kernel` / `ref`) so a large-N transform can land
                        beside the in-register one
    primitives.fields   prime-field arithmetic: the 32-bit Montgomery product every
                        modular primitive shares (`PrimeField`, `mont_mul`)
    primitives.chacha20 the ARX stream cipher (RFC 8439), the first non-PQC primitive
    primitives.poseidon2 Poseidon2-BabyBear/16, the algebraic hash ZK systems use (Plonky3)
    primitives.poly1305 Poly1305 (RFC 8439) and the ChaCha20-Poly1305 AEAD
    primitives.aes      AES-128-CTR through a T-table: a gather measurement, not a fast path
    primitives.gemm     exact int8 matrix multiply: the substrate for a GF(2^k) GEMM
    primitives.blake3   BLAKE3: the tree hash, one launch per block index and per tree level

ML-KEM (`triton_crypto.mlkem`), ML-DSA (`triton_crypto.mldsa`), SLH-DSA
(`triton_crypto.slhdsa`) and LMS/HSS (`triton_crypto.lms`) are thin layers on top of
these; they keep only what is scheme-specific — ML-KEM's base-case multiply and Toeplitz
GEMMs, ML-DSA's SampleInBall / Decompose, FIPS 205's thash, LMS's Winternitz chains.

Three domains in use:

    >>> from triton_crypto.primitives import KYBER, DILITHIUM, BABYBEAR_NTT
    >>> KYBER.q, KYBER.zeta, KYBER.wide
    (3329, 17, False)
    >>> DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.wide
    (8380417, 1753, True)
    >>> BABYBEAR_NTT.q
    2013265921
"""

from __future__ import annotations

from . import (
    aes,
    blake3,
    chacha20,
    fields,
    gemm,
    keccak,
    ntt,
    poly1305,
    poseidon2,
    sha2,
    sponge,
)
from .aes import (
    aes128_ctr_np,
    aes128_ctr_torch,
    aes128_encrypt_block_np,
    aes128_expand_key_np,
)
from .blake3 import blake3_hash_np, blake3_hash_torch
from .chacha20 import (
    chacha20_keystream,
    chacha20_keystream_np,
    chacha20_keystream_torch,
    chacha20_xor,
)
from .fields import BABYBEAR, MONT_R, PrimeField, mont_mul
from .gemm import gemm_int8_np, gemm_int8_torch
from .ntt import (
    BABYBEAR_NTT,
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
from .poly1305 import (
    chacha20_poly1305_open_torch,
    chacha20_poly1305_torch,
    poly1305_np,
    poly1305_np_batch,
    poly1305_torch,
)
from .poseidon2 import (
    BABYBEAR_POSEIDON2_16,
    Poseidon2,
    poseidon2_permute_np,
    poseidon2_permute_torch,
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
    "BABYBEAR",
    "BABYBEAR_NTT",
    "BABYBEAR_POSEIDON2_16",
    "DILITHIUM",
    "DOMAIN_SHA3",
    "DOMAIN_SHAKE",
    "HASHES",
    "IV256",
    "IV512",
    "K256",
    "K512",
    "KYBER",
    "MONT_R",
    "SHA3_224",
    "SHA3_256",
    "SHA3_384",
    "SHA3_512",
    "SHAKE128",
    "SHAKE256",
    "HashFn",
    "NTTDomain",
    "Poseidon2",
    "PrimeField",
    "aes",
    "aes128_ctr_np",
    "aes128_ctr_torch",
    "aes128_encrypt_block_np",
    "aes128_expand_key_np",
    "blake3",
    "blake3_hash_np",
    "blake3_hash_torch",
    "chacha20",
    "chacha20_keystream",
    "chacha20_keystream_np",
    "chacha20_keystream_torch",
    "chacha20_poly1305_open_torch",
    "chacha20_poly1305_torch",
    "chacha20_xor",
    "compress_np",
    "fields",
    "gemm",
    "gemm_int8_np",
    "gemm_int8_torch",
    "get_sponge",
    "hash_np",
    "hash_np_batch",
    "intt_np",
    "intt_torch",
    "keccak",
    "mont_mul",
    "mul_np",
    "negacyclic_conv_np",
    "ntt",
    "ntt_np",
    "ntt_torch",
    "pad10star1",
    "pad_batch",
    "pad_torch",
    "padded_len",
    "poly1305",
    "poly1305_np",
    "poly1305_np_batch",
    "poly1305_torch",
    "poseidon2",
    "poseidon2_permute_np",
    "poseidon2_permute_torch",
    "sha2",
    "sha_absorb",
    "sponge",
    "sponge_padded",
]
