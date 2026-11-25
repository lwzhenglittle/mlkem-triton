"""封装侧的融合内核：`GEMM + mod q + Compress + ByteEncode`。

ML-KEM 的封装在有了 `W_enc`（Toeplitz 展开的重加密矩阵）之后是：

    [u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]      （系数域，无 NTT）
    c1 = ByteEncode_du(Compress_du(u))                   （每行 256 系数 → 32·du 字节）
    c2 = ByteEncode_dv(Compress_dv(v))                   （256 系数 → 32·dv 字节）
    c  = c1 ‖ c2

前一步正好是 decaps 重加密那一趟（`fused_enc_gemm`）的 GEMM，只是 epilogue 从
「与参考值比较、按列计数」换成「压缩、打包成密文」。所以这里复用
`fused_gemm._gemm_tile` / `_tile_id`，只写新的 epilogue 和一个打包内核。

**位打包没有融进 GEMM 的 epilogue**：`du = 10` 时一个字节横跨 1~2 个系数，
而系数在 tile 的行方向上，跨行取字节需要在 tile 内部做重排（和 Toeplitz 展开
遇到的是同一类问题）。所以走两步：GEMM 先写出 `comp [M, B] int16`（8.4 MB @
B=4096），再由打包内核按「每个多项式 32·D 字节对齐」重组。实测这两步加起来
比 decaps 那一趟贵约 15 ns/op，可以接受。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..devices import warps
from .fused_gemm import _gemm_tile, _pick_cfg, _tile_id

Q = 3329
QL = tl.constexpr(Q)


# --------------------------------------------------------------- 压缩 epilogue
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
    """`(acc + E) mod q` → Compress_D → 写成 uint16 到 [M, N] 的 comp。"""
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


# --------------------------------------------------------------- 位打包
@triton.jit
def pack_seg_kernel(
    COMP, OUT, B, CTLEN, OUTOFF, NSEG, D: tl.constexpr, BLK: tl.constexpr, OBLK: tl.constexpr
):
    """把 `NSEG` 个多项式的压缩系数按 `32*D` 字节/多项式写进密文缓冲。

    `COMP` [NSEG*256, B] int16；`OUT` [B, CTLEN] uint8（已清零）。

    一个字节覆盖 8 个连续比特，可能横跨 **3 个**系数（D < 8 时，例如 dv=5）：
    D=10/11 是 2 个，D=5 是 3 个，D=4 恰好 2 个。所以固定展开 3 项再按移位量
    屏蔽，别按「2 个系数」硬编码 —— 我第一版就是硬编码 2 个，du=10/11 全对、
    dv=4 全对、dv=5 错 40/160 字节。
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
        cc = tl.where(cc < 256, cc, c0)  # 夹在段内，避免越界读
        v = tl.load(
            COMP + (seg * 256 + cc)[:, None] * B + cols[None, :],
            mask=omask[:, None] & bmask[None, :],
            other=0,
        ).to(tl.int32)
        sh = (t * D - s0)[:, None]  # 该系数 bit0 落在输出的第几位
        a = tl.where(sh >= 0, v << sh, v >> (-sh))
        acc |= tl.where((sh < 8) & (sh > -D), a, 0)
    byte = acc & 0xFF
    tl.store(
        OUT + cols[None, :] * CTLEN + OUTOFF + o[:, None],
        byte.to(tl.uint8),
        mask=omask[:, None] & bmask[None, :],
    )


# --------------------------------------------------------------- 入口
def fused_encaps(W, Y, E, B: int, du: int, dv: int, pivot: int | None = None):
    """→ comp [M, B] int16（已 Compress），或 None（约束不满足，未写输出）。"""
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
    cfg = _pick_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM or pivot % BM:
        return None
    out = torch.empty((M, B), dtype=torch.int16, device=W.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(B, BN),)
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
        (B % BN) != 0,
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
    """把 `comp [M, B]` 打包进 `padded [B, ≥ct_size]` 的 `[0, ct_size)`。

    u 段（k 个多项式 × 32·du 字节）用 D=du；c2 段（1 个多项式 × 32·dv 字节）用 D=dv。
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
