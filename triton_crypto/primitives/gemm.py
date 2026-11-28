"""Exact int8 matrix multiply and Karatsuba finite-field GEMMs over GF(16) and GF(2^8).

This is the generic half of what `mlkem/kernels/gemm_modq.py` does. That module is ML-KEM's:
`Q = 3329`, two epilogues (`W @ X + E` for re-encryption, `E - W @ X` for decryption) and a
float-assisted reduction. The part worth sharing is narrower and is what a GF(2^k) GEMM needs:

    C = A @ B      A [M, K] int8, B [K, N] int8, C [M, N] int32 -- exact, no rounding

Exactness is the whole point. XMX / Tensor-Core int8 dot products accumulate in int32 with no
rounding, so binary `{0, 1}` planes multiplied via `tl.dot(int8, int8)` compute exact `GF(2)`
matrix products. Using recursive Karatsuba over `GF(2)[x]`:

- `gf16_matmul_torch` (`GF(2^4)` mod `x^4 + x + 1`): **9** binary `tl.dot` planes instead of 16
  (1.78× fewer MACs), launched as one 3-D batched XMX GEMM (`_gemm_int8_batched_kernel`).
- `gf2_8_matmul_torch` (`GF(2^8)` mod `x^8 + x^4 + x^3 + x + 1`): **27** binary `tl.dot` planes
  instead of 64 (2.37× fewer MACs), launched as one 3-D batched XMX GEMM.

The tile table comes from `devices.gemm_cfg`, the same one ML-KEM's GEMM uses, because the
shape constraints (BM/BN multiples of 16) are the hardware's, not the scheme's.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from .. import devices as _devices

__all__ = [
    "gemm_int8_np",
    "gemm_int8_torch",
    "gf2_8_matmul_torch",
    "gf16_matmul_torch",
]


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


@triton.jit
def _gemm_int8_batched_kernel(
    A,
    B,
    OUT,
    M,
    N,
    K,
    sa_p,
    sa_m,
    sa_k,
    sb_p,
    sb_k,
    sb_n,
    so_p,
    so_m,
    so_n,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Batched 3-D binary-plane GEMM: `[P, M, K] int8 @ [P, K, N] int8 -> [P, M, N] uint8` (mod 2)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_p = tl.program_id(2)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    a_ptrs = A + pid_p * sa_p + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
    b_ptrs = B + pid_p * sb_p + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BK)):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k * BK < K), other=0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] + k * BK < K) & (offs_n[None, :] < N), other=0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BK * sa_k
        b_ptrs += BK * sb_k

    o_ptrs = OUT + pid_p * so_p + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    tl.store(o_ptrs, (acc & 1).to(tl.uint8), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def _gf16_kara9_planes_kernel(X, PLANES, n_elems, BLOCK: tl.constexpr):
    """Expand `[n_elems]` uint8 GF(16) values into 9 Karatsuba binary planes `[9, n_elems]` int8."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_elems
    v = tl.load(X + idx, mask=m, other=0).to(tl.int8)
    b0 = v & 1
    b1 = (v >> 1) & 1
    b2 = (v >> 2) & 1
    b3 = (v >> 3) & 1
    b02 = b0 ^ b2
    b13 = b1 ^ b3
    tl.store(PLANES + 0 * n_elems + idx, b0, mask=m)
    tl.store(PLANES + 1 * n_elems + idx, b1, mask=m)
    tl.store(PLANES + 2 * n_elems + idx, b0 ^ b1, mask=m)
    tl.store(PLANES + 3 * n_elems + idx, b2, mask=m)
    tl.store(PLANES + 4 * n_elems + idx, b3, mask=m)
    tl.store(PLANES + 5 * n_elems + idx, b2 ^ b3, mask=m)
    tl.store(PLANES + 6 * n_elems + idx, b02, mask=m)
    tl.store(PLANES + 7 * n_elems + idx, b13, mask=m)
    tl.store(PLANES + 8 * n_elems + idx, b02 ^ b13, mask=m)


@triton.jit
def _gf16_kara9_recombine_kernel(PLANES, OUT, n_elems, BLOCK: tl.constexpr):
    """Recombine 9 Karatsuba mod-2 planes `[9, n_elems]` uint8 -> `[n_elems]` uint8 mod `x^4+x+1`."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_elems
    l0 = tl.load(PLANES + 0 * n_elems + idx, mask=m, other=0)
    l2 = tl.load(PLANES + 1 * n_elems + idx, mask=m, other=0)
    l1 = tl.load(PLANES + 2 * n_elems + idx, mask=m, other=0) ^ l0 ^ l2
    h0 = tl.load(PLANES + 3 * n_elems + idx, mask=m, other=0)
    h2 = tl.load(PLANES + 4 * n_elems + idx, mask=m, other=0)
    h1 = tl.load(PLANES + 5 * n_elems + idx, mask=m, other=0) ^ h0 ^ h2
    m0 = tl.load(PLANES + 6 * n_elems + idx, mask=m, other=0)
    m2 = tl.load(PLANES + 7 * n_elems + idx, mask=m, other=0)
    m1 = tl.load(PLANES + 8 * n_elems + idx, mask=m, other=0) ^ m0 ^ m2

    c0 = l0
    c1 = l1
    c2 = l2 ^ m0 ^ l0 ^ h0
    c3 = m1 ^ l1 ^ h1
    c4 = h0 ^ m2 ^ l2 ^ h2
    c5 = h1
    c6 = h2

    r0 = (c0 ^ c4) & 1
    r1 = (c1 ^ c4 ^ c5) & 1
    r2 = (c2 ^ c5 ^ c6) & 1
    r3 = (c3 ^ c6) & 1
    tl.store(OUT + idx, r0 | (r1 << 1) | (r2 << 2) | (r3 << 3), mask=m)


@triton.jit
def _gf256_kara27_planes_kernel(X, PLANES, n_elems, BLOCK: tl.constexpr):
    """Expand `[n_elems]` uint8 GF(2^8) values into 27 Karatsuba binary planes `[27, n_elems]` int8."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_elems
    v = tl.load(X + idx, mask=m, other=0).to(tl.int8)
    b0 = v & 1
    b1 = (v >> 1) & 1
    b2 = (v >> 2) & 1
    b3 = (v >> 3) & 1
    b4 = (v >> 4) & 1
    b5 = (v >> 5) & 1
    b6 = (v >> 6) & 1
    b7 = (v >> 7) & 1

    # Low nibble (0..3) -> planes 0..8
    tl.store(PLANES + 0 * n_elems + idx, b0, mask=m)
    tl.store(PLANES + 1 * n_elems + idx, b1, mask=m)
    tl.store(PLANES + 2 * n_elems + idx, b0 ^ b1, mask=m)
    tl.store(PLANES + 3 * n_elems + idx, b2, mask=m)
    tl.store(PLANES + 4 * n_elems + idx, b3, mask=m)
    tl.store(PLANES + 5 * n_elems + idx, b2 ^ b3, mask=m)
    tl.store(PLANES + 6 * n_elems + idx, b0 ^ b2, mask=m)
    tl.store(PLANES + 7 * n_elems + idx, b1 ^ b3, mask=m)
    tl.store(PLANES + 8 * n_elems + idx, b0 ^ b1 ^ b2 ^ b3, mask=m)

    # High nibble (4..7) -> planes 9..17
    tl.store(PLANES + 9 * n_elems + idx, b4, mask=m)
    tl.store(PLANES + 10 * n_elems + idx, b5, mask=m)
    tl.store(PLANES + 11 * n_elems + idx, b4 ^ b5, mask=m)
    tl.store(PLANES + 12 * n_elems + idx, b6, mask=m)
    tl.store(PLANES + 13 * n_elems + idx, b7, mask=m)
    tl.store(PLANES + 14 * n_elems + idx, b6 ^ b7, mask=m)
    tl.store(PLANES + 15 * n_elems + idx, b4 ^ b6, mask=m)
    tl.store(PLANES + 16 * n_elems + idx, b5 ^ b7, mask=m)
    tl.store(PLANES + 17 * n_elems + idx, b4 ^ b5 ^ b6 ^ b7, mask=m)

    # Mid nibble (low ^ high) -> planes 18..26
    m0 = b0 ^ b4
    m1 = b1 ^ b5
    m2 = b2 ^ b6
    m3 = b3 ^ b7
    tl.store(PLANES + 18 * n_elems + idx, m0, mask=m)
    tl.store(PLANES + 19 * n_elems + idx, m1, mask=m)
    tl.store(PLANES + 20 * n_elems + idx, m0 ^ m1, mask=m)
    tl.store(PLANES + 21 * n_elems + idx, m2, mask=m)
    tl.store(PLANES + 22 * n_elems + idx, m3, mask=m)
    tl.store(PLANES + 23 * n_elems + idx, m2 ^ m3, mask=m)
    tl.store(PLANES + 24 * n_elems + idx, m0 ^ m2, mask=m)
    tl.store(PLANES + 25 * n_elems + idx, m1 ^ m3, mask=m)
    tl.store(PLANES + 26 * n_elems + idx, m0 ^ m1 ^ m2 ^ m3, mask=m)


@triton.jit
def _kara4_from9(PLANES, base, idx, n_elems, m):
    l0 = tl.load(PLANES + (base + 0) * n_elems + idx, mask=m, other=0)
    l2 = tl.load(PLANES + (base + 1) * n_elems + idx, mask=m, other=0)
    l1 = tl.load(PLANES + (base + 2) * n_elems + idx, mask=m, other=0) ^ l0 ^ l2
    h0 = tl.load(PLANES + (base + 3) * n_elems + idx, mask=m, other=0)
    h2 = tl.load(PLANES + (base + 4) * n_elems + idx, mask=m, other=0)
    h1 = tl.load(PLANES + (base + 5) * n_elems + idx, mask=m, other=0) ^ h0 ^ h2
    m0 = tl.load(PLANES + (base + 6) * n_elems + idx, mask=m, other=0)
    m2 = tl.load(PLANES + (base + 7) * n_elems + idx, mask=m, other=0)
    m1 = tl.load(PLANES + (base + 8) * n_elems + idx, mask=m, other=0) ^ m0 ^ m2
    return (
        l0,
        l1,
        l2 ^ m0 ^ l0 ^ h0,
        m1 ^ l1 ^ h1,
        h0 ^ m2 ^ l2 ^ h2,
        h1,
        h2,
    )


@triton.jit
def _gf256_kara27_recombine_kernel(PLANES, OUT, n_elems, BLOCK: tl.constexpr):
    """Recombine 27 Karatsuba mod-2 planes `[27, n_elems]` uint8 -> `[n_elems]` int32 mod `0x11B`."""
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = idx < n_elems
    L0, L1, L2, L3, L4, L5, L6 = _kara4_from9(PLANES, 0, idx, n_elems, m)
    H0, H1, H2, H3, H4, H5, H6 = _kara4_from9(PLANES, 9, idx, n_elems, m)
    M0, M1, M2, M3, M4, M5, M6 = _kara4_from9(PLANES, 18, idx, n_elems, m)

    d0 = L0
    d1 = L1
    d2 = L2
    d3 = L3
    d4 = L4 ^ M0 ^ L0 ^ H0
    d5 = L5 ^ M1 ^ L1 ^ H1
    d6 = L6 ^ M2 ^ L2 ^ H2
    d7 = M3 ^ L3 ^ H3
    d8 = H0 ^ M4 ^ L4 ^ H4
    d9 = H1 ^ M5 ^ L5 ^ H5
    d10 = H2 ^ M6 ^ L6 ^ H6
    d11 = H3
    d12 = H4
    d13 = H5
    d14 = H6

    # Reduce mod x^8 + x^4 + x^3 + x + 1 (0x11B)
    y0 = (d0 ^ d8 ^ d12 ^ d13) & 1
    y1 = (d1 ^ d8 ^ d9 ^ d12 ^ d14) & 1
    y2 = (d2 ^ d9 ^ d10 ^ d13) & 1
    y3 = (d3 ^ d8 ^ d10 ^ d11 ^ d12 ^ d13 ^ d14) & 1
    y4 = (d4 ^ d8 ^ d9 ^ d11 ^ d14) & 1
    y5 = (d5 ^ d9 ^ d10 ^ d12) & 1
    y6 = (d6 ^ d10 ^ d11 ^ d13) & 1
    y7 = (d7 ^ d11 ^ d12 ^ d14) & 1

    val = (
        y0.to(tl.int32)
        | (y1.to(tl.int32) << 1)
        | (y2.to(tl.int32) << 2)
        | (y3.to(tl.int32) << 3)
        | (y4.to(tl.int32) << 4)
        | (y5.to(tl.int32) << 5)
        | (y6.to(tl.int32) << 6)
        | (y7.to(tl.int32) << 7)
    )
    tl.store(OUT + idx, val, mask=m)


def gemm_int8_torch(a, b, device=None, out=None, tile=None):
    """`[M,K] int8 x [K,N] int8 -> [M,N] int32`, exact, on the device."""
    import torch

    for operand, name in ((a, "a"), (b, "b")):
        dtype = operand.dtype if isinstance(operand, torch.Tensor) else np.asarray(operand).dtype
        if dtype != torch.int8 and dtype != np.int8:
            raise TypeError(f"{name} must be int8 (the contract is exact int8 x int8), got {dtype}")

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


def _batched_binary_gemm(a_planes, b_planes, dev, tile=None):
    """`[P, M, K] int8 @ [P, K, N] int8 -> [P, M, N] uint8` (mod 2) in one 3-D XMX launch."""
    import torch

    p, m, k = a_planes.shape
    n = b_planes.shape[2]
    out = torch.empty((p, m, n), dtype=torch.uint8, device=dev)
    cfg = _devices.gemm_cfg(dev)
    bm, bn, bk = tile or (cfg["BM"], cfg["BN"], cfg["BK"])
    grid = (triton.cdiv(m, bm), triton.cdiv(n, bn), p)
    _gemm_int8_batched_kernel[grid](
        a_planes,
        b_planes,
        out,
        m,
        n,
        k,
        a_planes.stride(0),
        a_planes.stride(1),
        a_planes.stride(2),
        b_planes.stride(0),
        b_planes.stride(1),
        b_planes.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BM=bm,
        BN=bn,
        BK=bk,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


def gf16_expand_planes_torch(x_u8, device=None):
    """Expand `[M, K]` uint8 GF(16) tensor into 9 Karatsuba binary planes `[9, M, K]` int8."""
    import torch

    dev = _devices.resolve_device(device)
    t = (
        x_u8.to(device=dev, dtype=torch.uint8).contiguous()
        if isinstance(x_u8, torch.Tensor)
        else torch.from_numpy(np.array(x_u8, dtype=np.uint8, copy=True)).to(dev)
    )
    planes = torch.empty((9, *t.shape), dtype=torch.int8, device=dev)
    n_elems = t.numel()
    if n_elems:
        _gf16_kara9_planes_kernel[(triton.cdiv(n_elems, 1024),)](t, planes, n_elems, BLOCK=1024)
    return planes


def gf16_matmul_from_planes_torch(a_planes, b_planes, device=None, tile=None):
    """Multiply pre-expanded `[9, M, K]` and `[9, K, N]` int8 Karatsuba planes -> `[M, N]` uint8."""
    import torch

    dev = _devices.resolve_device(device)
    m = a_planes.shape[1]
    n = b_planes.shape[2]
    c_planes = _batched_binary_gemm(a_planes, b_planes, dev, tile=tile)
    out = torch.empty((m, n), dtype=torch.uint8, device=dev)
    n_elems = m * n
    if n_elems:
        _gf16_kara9_recombine_kernel[(triton.cdiv(n_elems, 1024),)](
            c_planes, out, n_elems, BLOCK=1024
        )
    return out


def gf16_matmul_torch(a, b, device=None, tile=None):
    """Product in `GF(16)` mod `x^4 + x + 1` via 2-level Karatsuba (**9** int8 GEMMs vs 16)."""
    import torch

    dev = _devices.resolve_device(device)
    ta = (
        a.to(device=dev, dtype=torch.uint8).contiguous()
        if isinstance(a, torch.Tensor)
        else torch.from_numpy(np.array(a, dtype=np.uint8, copy=True)).to(dev)
    )
    tb = (
        b.to(device=dev, dtype=torch.uint8).contiguous()
        if isinstance(b, torch.Tensor)
        else torch.from_numpy(np.array(b, dtype=np.uint8, copy=True)).to(dev)
    )
    if ta.ndim != 2 or tb.ndim != 2 or ta.shape[1] != tb.shape[0]:
        raise ValueError(f"shapes {tuple(ta.shape)} and {tuple(tb.shape)} are not a product")
    a_planes = gf16_expand_planes_torch(ta, device=dev)
    b_planes = gf16_expand_planes_torch(tb, device=dev)
    return gf16_matmul_from_planes_torch(a_planes, b_planes, device=dev, tile=tile)


def gf2_8_matmul_torch(a, b, device=None, planes=None, tile=None):
    """Product in `GF(2^8)` with the AES polynomial (`0x11B`) via 3-level Karatsuba (**27** int8 GEMMs).

    For full `GF(2^8)` (`planes=None` or `planes=8`), expands operands on-device into 27 binary
    Karatsuba planes (2.37× fewer MACs than the 64-plane schoolbook product), executes one 3-D
    batched XMX/Tensor-Core GEMM, and recombines mod `0x11B` into `[M, N]` int32.

    Pass `planes=1` for a single binary `GF(2)` matrix multiply (or `planes=4` for 4-bit).
    """
    import torch

    dev = _devices.resolve_device(device)
    ta = (
        a.to(device=dev, dtype=torch.uint8).contiguous()
        if isinstance(a, torch.Tensor)
        else torch.from_numpy(np.array(a, dtype=np.uint8, copy=True)).to(dev)
    )
    tb = (
        b.to(device=dev, dtype=torch.uint8).contiguous()
        if isinstance(b, torch.Tensor)
        else torch.from_numpy(np.array(b, dtype=np.uint8, copy=True)).to(dev)
    )
    if ta.ndim != 2 or tb.ndim != 2 or ta.shape[1] != tb.shape[0]:
        raise ValueError(f"shapes {tuple(ta.shape)} and {tuple(tb.shape)} are not a product")
    m, k = ta.shape
    n = tb.shape[1]
    nplanes = 8 if planes is None else planes

    if nplanes == 1:
        prod = gemm_int8_torch(
            (ta & 1).to(torch.int8), (tb & 1).to(torch.int8), device=dev, tile=tile
        )
        return prod & 1

    if nplanes == 8:
        a_planes = torch.empty((27, m, k), dtype=torch.int8, device=dev)
        b_planes = torch.empty((27, k, n), dtype=torch.int8, device=dev)
        if m * k:
            _gf256_kara27_planes_kernel[(triton.cdiv(m * k, 1024),)](
                ta, a_planes, m * k, BLOCK=1024
            )
        if k * n:
            _gf256_kara27_planes_kernel[(triton.cdiv(k * n, 1024),)](
                tb, b_planes, k * n, BLOCK=1024
            )
        c_planes = _batched_binary_gemm(a_planes, b_planes, dev, tile=tile)
        out = torch.empty((m, n), dtype=torch.int32, device=dev)
        if m * n:
            _gf256_kara27_recombine_kernel[(triton.cdiv(m * n, 1024),)](
                c_planes, out, m * n, BLOCK=1024
            )
        return out

    # Generic fallback for arbitrary partial `planes` in 2..7
    def _reduce_mask(deg: int) -> int:
        mask = 1 << deg
        while mask & ~0xFF:
            top = mask.bit_length() - 1
            mask ^= (1 << top) ^ (0x1B << (top - 8))
        return mask

    out = torch.zeros((m, n), dtype=torch.int32, device=dev)
    a_p = [((ta >> i) & 1).to(torch.int8) for i in range(nplanes)]
    b_p = [((tb >> j) & 1).to(torch.int8) for j in range(nplanes)]
    for i in range(nplanes):
        for j in range(nplanes):
            prod = gemm_int8_torch(a_p[i], b_p[j], device=dev, tile=tile) & 1
            mask = _reduce_mask(i + j)
            for bit in range(8):
                if (mask >> bit) & 1:
                    out ^= prod << bit
    return out
