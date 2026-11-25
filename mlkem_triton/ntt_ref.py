"""FIPS 203 的 NTT / INTT / MultiplyNTTs 参考实现（numpy）。

同时提供 `negacyclic_conv` 与 `toeplitz`，用来验证整个方案的核心等式：

    T(a) @ b  ==  a * b   (在 Z_q[X]/(X^256 + 1) 中)

其中 T(a)[i][j] = a[(i-j) mod 256] * (+1 if i >= j else -1)。
"""

from __future__ import annotations

import numpy as np

from .params import N, Q

ZETA = 17  # 3329 模下的 256 次本原单位根


def bitrev7(i: int) -> int:
    """7 bit 位反转。"""
    return int(format(i, "07b")[::-1], 2)


# zetas[i] = ζ^BitRev7(i)，i = 0..127（FIPS 203 Algorithm 9 用）
ZETAS = np.array([pow(ZETA, bitrev7(i), Q) for i in range(128)], dtype=np.int64)

# BaseCaseMultiply 的 γ_i = ζ^(2·BitRev7(i)+1)
GAMMAS = np.array([pow(ZETA, 2 * bitrev7(i) + 1, Q) for i in range(128)], dtype=np.int64)

INV128 = pow(128, -1, Q)  # = 3303


def ntt(f):
    """FIPS 203 Algorithm 9。"""
    f = np.array(f, dtype=np.int64) % Q
    i = 1
    length = 128
    while length >= 2:
        for start in range(0, N, 2 * length):
            zeta = int(ZETAS[i])
            i += 1
            j = np.arange(start, start + length)
            t = (zeta * f[j + length]) % Q
            f[j + length] = (f[j] - t) % Q
            f[j] = (f[j] + t) % Q
        length //= 2
    return f


def intt(f):
    """FIPS 203 Algorithm 10。"""
    f = np.array(f, dtype=np.int64) % Q
    i = 127
    length = 2
    while length <= 128:
        for start in range(0, N, 2 * length):
            zeta = int(ZETAS[i])
            i -= 1
            j = np.arange(start, start + length)
            t = f[j].copy()
            f[j] = (f[j] + f[j + length]) % Q
            f[j + length] = (zeta * (f[j + length] - t)) % Q
        length *= 2
    return (f * INV128) % Q


def base_case_multiply(a0, a1, b0, b1, gamma):
    """FIPS 203 Algorithm 12。"""
    c0 = (a0 * b0 + a1 * b1 % Q * gamma) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    return c0, c1


def multiply_ntts(f, g):
    """FIPS 203 Algorithm 11：128 个二次域上逐点相乘。"""
    f = np.asarray(f, dtype=np.int64)
    g = np.asarray(g, dtype=np.int64)
    h = np.zeros(N, dtype=np.int64)
    a0, a1 = f[0::2], f[1::2]
    b0, b1 = g[0::2], g[1::2]
    h[0::2] = (a0 * b0 + (a1 * b1 % Q) * GAMMAS) % Q
    h[1::2] = (a0 * b1 + a1 * b0) % Q
    return h


def negacyclic_conv(a, b):
    """Z_q[X]/(X^256+1) 中的 O(N^2) 朴素乘法（仅用于测试）。"""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    idx = np.arange(N)[:, None] + np.arange(N)[None, :]
    sign = np.where(idx < N, 1, -1)
    c = np.zeros(N, dtype=np.int64)
    np.add.at(c, idx % N, np.outer(a, b) * sign)
    return c % Q


def toeplitz(a):
    """T(a)[i][j] = a[(i-j) mod 256] * (+1 if i >= j else -1)。"""
    a = np.asarray(a, dtype=np.int64)
    i = np.arange(N)[:, None]
    j = np.arange(N)[None, :]
    return a[(i - j) % N] * np.where(i >= j, 1, -1)


# ---------------- 编解码 / 压缩（FIPS 203 §4.2） ----------------


def byte_encode(vals, d: int) -> bytes:
    """ByteEncode_d：256 个 0..2^d-1 的整数 -> 32*d 字节（小端位打包）。"""
    mask = (1 << d) - 1
    x = 0
    for i, v in enumerate(vals):
        v = int(v) & mask
        if v != int(vals[i]):
            raise ValueError(f"系数 {i} 超出 {d} bit 范围")
        x |= v << (i * d)
    return x.to_bytes(32 * d, "little")


def byte_decode(data: bytes, d: int):
    """ByteDecode_d：32*d 字节 -> 256 个整数。"""
    if len(data) != 32 * d:
        raise ValueError(f"需要 {32 * d} 字节，得到 {len(data)}")
    x = int.from_bytes(data, "little")
    mask = (1 << d) - 1
    return np.array([(x >> (i * d)) & mask for i in range(N)], dtype=np.int64)


def compress(x, d: int):
    """Compress_d(x) = round(2^d/q * x) mod 2^d（q 为奇数，无平局）。"""
    x = np.asarray(x, dtype=np.int64)
    return ((x << d) + Q // 2) // Q % (1 << d)


def decompress(y, d: int):
    """Decompress_d(y) = round(q/2^d * y)（q 为奇数，无平局）。"""
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
