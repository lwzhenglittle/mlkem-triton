"""ML-KEM (FIPS 203) reference helpers in numpy.

The **transform itself is generic** and lives in `triton_crypto.primitives.ntt`; what is
kept here is what is specific to ML-KEM:

- `multiply_ntts`, the base-case multiplication in the quadratic extension
  Z_q[X]/(X²−γ) (FIPS 203 Alg. 11/12) — even/odd coefficients form two channels;
- `GAMMAS`, the γ_i table that multiplication needs;
- ByteEncode/ByteDecode_d and Compress/Decompress_d (§4.2);
- `toeplitz`, the identity the whole project rests on:

      T(a) @ b  ==  a * b   (in Z_q[X]/(X^256 + 1))

  where T(a)[i][j] = a[(i-j) mod 256] * (+1 if i >= j else -1).
"""

from __future__ import annotations

import numpy as np

from ..primitives.ntt import KYBER, intt_np, negacyclic_conv_np, ntt_np
from ..primitives.ntt import bitrev as _bitrev
from .params import N, Q

ZETA = KYBER.zeta  # primitive 256th root of unity modulo 3329


def bitrev7(i: int) -> int:
    """7-bit bit reversal."""
    return _bitrev(i, 7)


# zetas[i] = ζ^BitRev7(i), i = 0..127 (used by FIPS 203 Algorithm 9)
ZETAS = KYBER.zetas

# γ_i for BaseCaseMultiply = ζ^(2·BitRev7(i)+1)
GAMMAS = np.array([pow(ZETA, 2 * bitrev7(i) + 1, Q) for i in range(128)], dtype=np.int64)

INV128 = KYBER.scale  # = 3303


def ntt(f):
    """FIPS 203 Algorithm 9 (delegates to the generic primitive)."""
    return ntt_np(f, KYBER)


def intt(f):
    """FIPS 203 Algorithm 10 (delegates to the generic primitive)."""
    return intt_np(f, KYBER)


def base_case_multiply(a0, a1, b0, b1, gamma):
    """FIPS 203 Algorithm 12."""
    c0 = (a0 * b0 + a1 * b1 % Q * gamma) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    return c0, c1


def multiply_ntts(f, g):
    """FIPS 203 Algorithm 11: pointwise multiply over 128 quadratic extensions."""
    f = np.asarray(f, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    h = np.zeros(N, dtype=np.int64)
    a0, a1 = f[0::2], f[1::2]
    b0, b1 = g[0::2], g[1::2]
    h[0::2] = (a0 * b0 + (a1 * b1 % Q) * GAMMAS) % Q
    h[1::2] = (a0 * b1 + a1 * b0) % Q
    return h


def negacyclic_conv(a, b):
    """O(N^2) schoolbook multiplication in Z_q[X]/(X^256+1) (test only)."""
    return negacyclic_conv_np(a, b, KYBER)


def toeplitz(a):
    """T(a)[i][j] = a[(i-j) mod 256] * (+1 if i >= j else -1)."""
    return KYBER.toeplitz(a)


# ---------------- encode/decode / compress (FIPS 203 §4.2) ----------------


def byte_encode(vals, d: int) -> bytes:
    """ByteEncode_d: 256 integers in 0..2^d-1 -> 32*d bytes (little-endian bit packing)."""
    mask = (1 << d) - 1
    x = 0
    for i, v in enumerate(vals):
        v = int(v) & mask
        if v != int(vals[i]):
            raise ValueError(f"coefficient {i} out of range for {d} bits")
        x |= v << (i * d)
    return x.to_bytes(32 * d, "little")


def byte_decode(data: bytes, d: int):
    """ByteDecode_d: 32*d bytes -> 256 integers."""
    if len(data) != 32 * d:
        raise ValueError(f"expected {32 * d} bytes, got {len(data)}")
    x = int.from_bytes(data, "little")
    mask = (1 << d) - 1
    return np.array([(x >> (i * d)) & mask for i in range(N)], dtype=np.int64)


def compress(x, d: int):
    """Compress_d(x) = round(2^d/q * x) mod 2^d (q is odd, so no ties)."""
    x = np.asarray(x, dtype=np.int64)
    return ((x << d) + Q // 2) // Q % (1 << d)


def decompress(y, d: int):
    """Decompress_d(y) = round(q/2^d * y) (q is odd, so no ties)."""
    y = np.asarray(y, dtype=np.int64)
    return ((y * Q) + (1 << (d - 1))) >> d


__all__ = [
    "GAMMAS",
    "INV128",
    "ZETA",
    "ZETAS",
    "N",
    "Q",
    "base_case_multiply",
    "bitrev7",
    "byte_decode",
    "byte_encode",
    "compress",
    "decompress",
    "intt",
    "multiply_ntts",
    "negacyclic_conv",
    "ntt",
    "toeplitz",
]
