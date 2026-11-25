"""Phase 3.2：位解包、(de)compress、常数时间选择。

全部用整数精确运算（无浮点），与 FIPS 203 的取整定义逐位一致
（q 为奇数，Compress/Decompress 不会出现平局，见 tests/test_ntt.py）。
"""

from __future__ import annotations

import numpy as np
import torch

from ..params import N, Q

# ---------------- numpy 版（CPU 参考 / 位打包） ----------------


def decompress_np(x, d: int) -> np.ndarray:
    """Decompress_d(x) = round(q/2^d · x)，x ∈ [0, 2^d)。"""
    x = np.asarray(x, dtype=np.int64)
    return ((x * Q) + (1 << (d - 1))) >> d


def compress_np(x, d: int) -> np.ndarray:
    """Compress_d(x) = round(2^d/q · x) mod 2^d，x ∈ [0, q)。"""
    x = np.asarray(x, dtype=np.int64) % Q
    return (((x << d) + (Q // 2)) // Q) & ((1 << d) - 1)


def _unpack_bits(data: np.ndarray, d: int, count: int) -> np.ndarray:
    """data [B, nbytes] uint8 -> [count, B] int64（小端位序）。"""
    bits = np.unpackbits(data, axis=1, bitorder="little")[:, : count * d]
    bits = bits.reshape(data.shape[0], count, d).astype(np.int64)
    weights = (1 << np.arange(d)).astype(np.int64)
    return np.einsum("bcd,d->cb", bits, weights)


def unpack_ct(cts: np.ndarray, k: int, du: int, dv: int):
    """cts [B, ct_len] uint8 -> c1 [k,256,B], c2 [256,B]，均为 [0,2^d) 的整数。"""
    cts = np.asarray(cts, dtype=np.uint8)
    if cts.ndim == 1:
        cts = cts[None, :]
    c1_bytes, c2_bytes = 32 * du * k, 32 * dv
    assert cts.shape[1] == c1_bytes + c2_bytes, (cts.shape, c1_bytes + c2_bytes)
    c1 = _unpack_bits(cts[:, :c1_bytes], du, 256 * k).reshape(k, N, cts.shape[0])
    c2 = _unpack_bits(cts[:, c1_bytes:], dv, N)
    return c1, c2


def pack_bits(vals: np.ndarray, d: int) -> np.ndarray:
    """vals [count, B] -> [B, 32*d] uint8（ByteEncode_d 的逆）。"""
    vals = np.asarray(vals, dtype=np.int64)
    count, B = vals.shape
    shifts = np.arange(d, dtype=np.int64)
    bits = ((vals[:, :, None] >> shifts) & 1).astype(np.uint8)  # [count,B,d]
    bits = np.transpose(bits, (0, 2, 1)).reshape(count * d, B).T  # [B, count*d]
    pad = (-bits.shape[1]) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros((B, pad), np.uint8)], axis=1)
    return np.packbits(bits, axis=1, bitorder="little")


def select_bytes(k_prime: np.ndarray, k_bar: np.ndarray, eq: np.ndarray) -> np.ndarray:
    """常数时间选择：K = where(eq, K', K̄)，逐字节 mask，不在 host 侧分支。"""
    m = eq.astype(np.uint8)[:, None] * 0xFF
    return (k_prime & m) | (k_bar & ~m)


# ---------------- torch 版（GPU / 与 kernel 融合用） ----------------


def decompress_t(x: torch.Tensor, d: int) -> torch.Tensor:
    return ((x.to(torch.int64) * Q) + (1 << (d - 1))) >> d


def compress_t(x: torch.Tensor, d: int) -> torch.Tensor:
    x = x.to(torch.int64) % Q
    return (((x << d) + (Q // 2)) // Q) & ((1 << d) - 1)


__all__ = [
    "compress_np",
    "compress_t",
    "decompress_np",
    "decompress_t",
    "pack_bits",
    "select_bytes",
    "unpack_ct",
]
