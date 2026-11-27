"""Fused kernels on the encaps side: `GEMM + mod q + Compress + ByteEncode`.

With `W_enc` (the Toeplitz-expanded re-encryption matrix) in hand, ML-KEM
encapsulation is:

    [u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]      (coefficient domain, no NTT)
    c1 = ByteEncode_du(Compress_du(u))                   (256 coeffs per row → 32·du bytes)
    c2 = ByteEncode_dv(Compress_dv(v))                   (256 coeffs → 32·dv bytes)
    c  = c1 ‖ c2

The preceding step is exactly the GEMM of the decaps re-encryption pass
(`fused_enc_gemm`); only the epilogue changes, from "compare against the
reference value, count per column" to "compress and pack into a ciphertext". So
we reuse `fused_gemm._gemm_tile` / `_tile_id` and write only the new epilogue
plus one packing kernel.

**Bit packing is not fused into the GEMM epilogue**: with `du = 10` a byte spans
1~2 coefficients, and coefficients run along the tile rows, so taking a byte
across rows needs reordering inside the tile (the same class of problem as the
Toeplitz expansion). So it takes two steps: the GEMM first writes
`comp [M, B] int16` (8.4 MB @ B=4096), then the packing kernel regroups it with
"32·D bytes per polynomial" alignment. Measured, those two steps together cost
about 15 ns/op more than the decaps pass, which is acceptable.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...devices import warps
from .fused_gemm import _fused_cfg, _gemm_tile, _tile_id

Q = 3329
QL = tl.constexpr(Q)


# --------------------------------------------------------------- compression epilogue
@triton.jit
def _compress_epi(
    acc,
    E,
    OUT,
    offm,
    cbase,
    D: tl.constexpr,
    BN: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    """`(acc + E) mod q` -> Compress_D -> written as uint16 into comp [M, N]."""
    cols = cbase + tl.arange(0, BN)
    if MASK_N:
        m = cols[None, :] < N
        e = tl.load(E + offm[:, None] * N + cols[None, :], mask=m, other=0)
    else:
        e = tl.load(E + offm[:, None] * N + cols[None, :])

    x = acc.to(tl.int32) + e.to(tl.int32)
    x = x % QL
    x = tl.where(x < 0, x + QL, x)
    comp = (((x << D) + 1664) // QL) & ((1 << D) - 1)
    if MASK_N:
        tl.store(OUT + offm[:, None] * N + cols[None, :], comp.to(tl.int16), mask=cols[None, :] < N)
    else:
        tl.store(OUT + offm[:, None] * N + cols[None, :], comp.to(tl.int16))


@triton.jit
def encaps_kernel(
    W,
    Y,
    E,
    OUT,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    PIVOT: tl.constexpr,
    DU: tl.constexpr,
    DV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _tile_id(pid, M, N, BM, BN, GM)
    acc = _gemm_tile(W, Y, pid_m, pid_n, K, N, BM, BN, BK)
    offm = pid_m * BM + tl.arange(0, BM)
    if pid_m * BM < PIVOT:
        _compress_epi(acc, E, OUT, offm, pid_n * BN, DU, BN, N, BM, MASK_N)
    else:
        _compress_epi(acc, E, OUT, offm, pid_n * BN, DV, BN, N, BM, MASK_N)


# --------------------------------------------------------------- bit packing
@triton.jit
def pack_seg_kernel(
    COMP, OUT, B, CTLEN, OUTOFF, NSEG, D: tl.constexpr, BLK: tl.constexpr, OBLK: tl.constexpr
):
    """Write the compressed coefficients of `NSEG` polynomials into the
    ciphertext buffer, `32*D` bytes per polynomial.

    `COMP` [NSEG*256, B] int16; `OUT` [B, CTLEN] uint8 (already zeroed).

    One byte covers 8 consecutive bits and may span **3** coefficients (when
    D < 8, e.g. dv=5): D=10/11 spans 2, D=5 spans 3, D=4 spans exactly 2. So
    always unroll 3 terms and mask by the shift amount -- do not hardcode "2
    coefficients": my first version hardcoded 2, and it was all correct for
    du=10/11 and for dv=4, while dv=5 was wrong in 40/160 bytes.
    """
    pid_b = tl.program_id(0)
    pid_o = tl.program_id(1)
    cols = pid_b * BLK + tl.arange(0, BLK)
    bmask = cols < B
    o = pid_o * OBLK + tl.arange(0, OBLK)
    segb = 32 * D
    omask = o < NSEG * segb

    seg = o // segb
    bit0 = (o - seg * segb) * 8
    c0 = bit0 // D
    s0 = bit0 - c0 * D

    acc = tl.zeros((OBLK, BLK), dtype=tl.int32)
    for t in tl.static_range(3):
        cc = c0 + t
        cc = tl.where(cc < 256, cc, c0)  # clamp inside the segment, no out-of-bounds read
        v = tl.load(
            COMP + (seg * 256 + cc)[:, None] * B + cols[None, :],
            mask=omask[:, None] & bmask[None, :],
            other=0,
        ).to(tl.int32)
        sh = (t * D - s0)[:, None]  # which output bit this coefficient's bit0 lands on
        a = tl.where(sh >= 0, v << sh, v >> (-sh))
        acc |= tl.where((sh < 8) & (sh > -D), a, 0)
    byte = acc & 0xFF
    tl.store(
        OUT + cols[None, :] * CTLEN + OUTOFF + o[:, None],
        byte.to(tl.uint8),
        mask=omask[:, None] & bmask[None, :],
    )


# --------------------------------------------------------------- entry points
def fused_encaps(W, Y, E, B: int, du: int, dv: int, pivot: int | None = None):
    """→ comp [M, B] int16 (already Compressed), or None (constraints
    unsatisfied, nothing written)."""
    if W.dim() != 2 or Y.dim() != 2 or E.dim() != 2:
        return None
    M, K = W.shape
    if M <= 0 or K <= 0 or B <= 0:
        return None
    if Y.shape != (K, B) or E.shape != (M, B):
        return None
    if W.dtype != torch.float16 or Y.dtype != torch.float16 or E.dtype != torch.int16:
        return None
    if not (W.is_contiguous() and Y.is_contiguous() and E.is_contiguous()):
        return None
    if pivot is None:
        pivot = M - 256
    if not (0 <= pivot <= M) or not (1 <= du <= 15 and 1 <= dv <= 15):
        return None
    cfg = _fused_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM or pivot % BM:
        return None
    out = torch.empty((M, B), dtype=torch.int16, device=W.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(B, BN),)
    # Always mask the epilogue load/store (same Intel triton-xpu VA 0x0 guard as fused_enc_gemm:
    # at B=4096 with MASK_N=False and num_stages=2, encaps_kernel wedged the GPU with
    # UR_RESULT_ERROR_DEVICE_LOST). `_fused_cfg` also disables pipelining on XPU while leaving
    # CUDA/ROCm untouched.
    encaps_kernel[grid](
        W,
        Y,
        E,
        out,
        M,
        K,
        B,
        pivot,
        du,
        dv,
        BM,
        BN,
        BK,
        cfg["GM"],
        True,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


def pack_ciphertext(
    comp: torch.Tensor,
    padded: torch.Tensor,
    k: int,
    du: int,
    dv: int,
    ct_size: int,
    blk: int = 32,
    oblk: int = 128,
) -> None:
    """Pack `comp [M, B]` into `[0, ct_size)` of `padded [B, ≥ct_size]`.

    The u segment (k polynomials × 32·du bytes) uses D=du; the c2 segment
    (1 polynomial × 32·dv bytes) uses D=dv.
    """
    B = comp.shape[1]
    grid_u = (triton.cdiv(B, blk), triton.cdiv(k * 32 * du, oblk))
    pack_seg_kernel[grid_u](
        comp,
        padded,
        B,
        padded.shape[1],
        0,
        k,
        D=du,
        BLK=blk,
        OBLK=oblk,
        num_warps=warps("ct_pack", 4),
    )
    grid_v = (triton.cdiv(B, blk), triton.cdiv(32 * dv, oblk))
    pack_seg_kernel[grid_v](
        comp[k * 256 :],
        padded,
        B,
        padded.shape[1],
        k * 32 * du,
        1,
        D=dv,
        BLK=blk,
        OBLK=oblk,
        num_warps=warps("ct_pack", 4),
    )


__all__ = ["encaps_kernel", "fused_encaps", "pack_ciphertext", "pack_seg_kernel"]
