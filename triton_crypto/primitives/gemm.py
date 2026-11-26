"""Exact int8 matrix multiply: the substrate a finite-field GEMM is built from.

This is the generic half of what `mlkem/kernels/gemm_modq.py` does. That module is ML-KEM's:
`Q = 3329`, two epilogues (`W @ X + E` for re-encryption, `E - W @ X` for decryption) and a
float-assisted reduction. The part worth sharing is narrower and is what a GF(2^8) or GF(2^16)
GEMM actually needs:

    C = A @ B      A [M, K] int8, B [K, N] int8, C [M, N] int32 -- exact, no rounding

Exactness is the whole point. XMX int8 dot products accumulate in int32 with no rounding, so
`int8 x int8 -> int32` is bit-exact for any inputs; that is what makes the bit-plane trick
work: split each GF(2^8) element into 8 binary planes, run 8 (or fewer, with the standard
CarryLessMultiply-free decomposition) int8 GEMMs, and recombine with XOR. Whether that is
*faster* than a bitsliced CUDA-core version is exactly what `PLAN_ffgemm.md` is designed to
measure; this module only guarantees the arithmetic the measurement needs.

The tile table comes from `devices.gemm_cfg`, the same one ML-KEM's GEMM uses, because the
shape constraints (BM/BN multiples of 16) are the hardware's, not the scheme's.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from .. import devices as _devices

__all__ = ["gemm_int8_np", "gemm_int8_torch"]


def gemm_int8_np(a, b) -> np.ndarray:
    """Reference: `int64` accumulation, then the range check that makes "exact" meaningful."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError(f"shapes {a.shape} and {b.shape} are not a matrix product")
    if a.min() < -128 or a.max() > 127 or b.min() < -128 or b.max() > 127:
        raise ValueError("gemm_int8_np takes int8 operands")
    out = a @ b
    limit = np.int64(2**31)
    if out.size and (out.max() >= limit or out.min() < -limit):
        raise OverflowError("int32 accumulator would overflow; split the reduction")
    return out.astype(np.int32)


@triton.jit
def _gemm_int8_kernel(
    A,
    B,
    OUT,
    M,
    N,
    K,
    sa_m,
    sa_k,
    sb_k,
    sb_n,
    so_m,
    so_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """One program per `BM x BN` tile, `BK`-deep steps, int32 accumulate, no epilogue."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BK < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BK < K) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sa_k
        b_ptrs += BK * sb_k

    o_ptrs = OUT + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    tl.store(o_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def gemm_int8_torch(a, b, device=None, out=None, tile=None):
    """`[M,K] int8 x [K,N] int8 -> [M,N] int32`, exact, on the device."""
    import torch

    for operand, name in ((a, "a"), (b, "b")):
        # A device tensor must not be routed through numpy: `np.asarray(xpu_tensor)` raises
        # "can't convert xpu:0 device type tensor to numpy", which is a confusing way to say
        # "your dtype is fine". Read the dtype where the operand lives.
        dtype = operand.dtype if isinstance(operand, torch.Tensor) else np.asarray(operand).dtype
        if dtype != torch.int8 and dtype != np.int8:
            raise TypeError(f"{name} must be int8 (the contract is exact int8 x int8), got {dtype}")

    import torch

    dev = _devices.resolve_device(device)
    x = torch.as_tensor(a, dtype=torch.int8, device=dev)
    y = torch.as_tensor(b, dtype=torch.int8, device=dev)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[0]:
        raise ValueError(f"shapes {tuple(x.shape)} and {tuple(y.shape)} are not a product")
    m, k = x.shape
    n = y.shape[1]
    if out is None:
        out = torch.empty((m, n), dtype=torch.int32, device=dev)
    cfg = _devices.gemm_cfg(dev)
    bm, bn, bk = tile or (cfg["BM"], cfg["BN"], cfg["BK"])
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn))
    _gemm_int8_kernel[grid](
        x,
        y,
        out,
        m,
        n,
        k,
        x.stride(0),
        x.stride(1),
        y.stride(0),
        y.stride(1),
        out.stride(0),
        out.stride(1),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


def _reduce_mask(k: int) -> int:
    """`x^k mod 0x11B` as a bit mask over `x^0..x^7`, for `0 <= k <= 14`."""
    m = 1 << k
    while m & ~0xFF:
        top = m.bit_length() - 1
        m ^= 1 << top
        m ^= 0x1B << (top - 8)  # x^top = x^(top-8) * (x^4 + x^3 + x + 1)
    return m


def gf2_8_matmul_torch(a, b, device=None, planes=None):
    """Product in `GF(2^8)` with the AES polynomial, assembled from bit-plane int8 GEMMs.

    For binary operands an int8 GEMM *is* a `GF(2)` matrix product, so an `GF(2^8)` product is
    64 of them -- one per pair of bit planes -- XOR-ed through the reduction masks. This is the
    strategy `PLAN_ffgemm.md` is built on, and the point of exposing it here is to have an
    end-to-end number: what an `GF(2^8)` multiply actually costs when the matrix unit does the
    bit work.

    `a` and `b` are `[M, K]` and `[K, N]` arrays of byte values. Pass `planes=1` to run only the
    low bit plane, which is what a caller who knows its operands are binary would use.
    """
    import torch

    from .. import devices as _devices

    dev = _devices.resolve_device(device)
    a = np.asarray(a, dtype=np.uint16)
    b = np.asarray(b, dtype=np.uint16)
    m, k = a.shape
    k2, n = b.shape
    if k != k2:
        raise ValueError(f"shapes {a.shape} and {b.shape} are not a product")
    nplanes = 8 if planes is None else planes
    out = torch.zeros((m, n), dtype=torch.int32, device=dev)
    a_planes = [torch.as_tensor(((a >> i) & 1).astype(np.int8), device=dev) for i in range(nplanes)]
    b_planes = [torch.as_tensor(((b >> j) & 1).astype(np.int8), device=dev) for j in range(nplanes)]
    for i in range(nplanes):
        for j in range(nplanes):
            prod = gemm_int8_torch(a_planes[i], b_planes[j], device=dev)
            mask = _reduce_mask(i + j)
            if mask == 0:
                continue
            # the mask is a set of bits of x^(i+j) shifted back into x^0..x^7, so the plane's
            # entries are XOR-ed into every bit position the mask names
            for bit in range(8):
                if mask >> bit & 1:
                    out ^= (prod & 1) << bit
    return out
