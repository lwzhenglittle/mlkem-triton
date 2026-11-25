"""两趟 GEMM 的融合内核：fp32 累加 + 就地 epilogue。

两趟都是「固定权重矩阵 × 每批不同的变量」，所以用同一副骨架，只换 epilogue：

**ENC（重加密）** `fused_enc_gemm`

    替换掉这三步：
        acc2 = torch.mm(W_enc, Y, out_dtype=fp32)     # [1024, B] fp32 = 16.8 MB
        cmp(acc2, E, c1r, ...)                        # du 段
        cmp(acc2, E, c2r, ...)                        # dv 段
    融合后 acc2 完全不落显存，只写一个 [M/BM, B] int32 的 partial 缓冲。

**DEC（解密）** `fused_dec_gemm`

    替换掉这两步：
        acc = torch.mm(W_dec, U, out_dtype=fp32)      # [256, B] fp32 = 4.2 MB
        m_bytes = compress_pack(V, acc, B)            # mod q → compress 1 bit → 打包
    融合后同样不落 acc。实测（B=4096，Intel Xe3）比两步快 **1.36×**
    （0.213 → 0.157 ms），输出逐字节一致。

为什么两个 epilogue 都要 `x %= q` 并补负
----------------------------------------
Triton / C 的 `int % int` 是向零取整（`-7 % 3 == -1`），而我们要的是
`[0, q)` 的规范代表，所以统一 `x = x % q; if x < 0: x += q`。

关于 `out_dtype=torch.float32`
------------------------------
`torch.mm` 没有这个公开参数（那是 torch-xpu 的私有 `aten::mm.dtype`）。
而且 fp16 输出在这里不可用：`|sum|` 峰值约 1.8e5 > fp16 上限 65504 → inf。
所以这两趟必须走我们自己的 Triton 内核。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..devices import gemm_cfg

Q = 3329
QL = tl.constexpr(Q)
BLOCK_OUT = 128  # partial 合并内核的列块


# --------------------------------------------------------------- 共用 GEMM 骨架
@triton.jit
def _gemm_tile(
    W,
    Y,
    pid_m,
    pid_n,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """一块 (BM, BN) 的 fp16×fp16 → fp32 累加。M 只用来算行偏移，不需要传。"""
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)

    a_ptrs = W + offm[:, None] * K + offk[None, :]
    b_ptrs = Y + offk[:, None] * N + offn[None, :]
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        acc = tl.dot(tl.load(a_ptrs), tl.load(b_ptrs), acc, out_dtype=tl.float32)
        a_ptrs += BK
        b_ptrs += BK * N
    return acc


@triton.jit
def _tile_id(
    pid, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr
):
    """L2 友好的骨牌式分组（grouped ordering），返回 (pid_m, pid_n)。"""
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    ng = GM * npn
    gid = pid // ng
    gsz = min(npm - gid * GM, GM)
    pid_m = gid * GM + ((pid % ng) % gsz)
    pid_n = (pid % ng) // gsz
    return pid_m, pid_n


# --------------------------------------------------------------- ENC epilogue
@triton.jit
def _epi_enc(
    acc,
    E,
    REF,
    OUT,
    offm,
    rrow0,
    cbase,
    D: tl.constexpr,
    BN: tl.constexpr,
    N: tl.constexpr,
    MASK_N: tl.constexpr,
    BM: tl.constexpr,
    DSTOFF,
):
    """mod q → compress → 与参考值比较 → 按列归约 → 写 partial。"""
    cols = cbase + tl.arange(0, BN)
    if MASK_N:
        m = cols[None, :] < N
        e = tl.load(E + offm[:, None] * N + cols[None, :], mask=m, other=0)
    else:
        e = tl.load(E + offm[:, None] * N + cols[None, :])

    x = acc.to(tl.int32) + e.to(tl.int32)
    x = x % QL
    x = tl.where(x < 0, x + QL, x)  # Python 语义的 mod
    comp = (((x << D) + 1664) // QL) & ((1 << D) - 1)  # x >= 0，// 即向零取整

    if MASK_N:
        r = tl.load(REF + (offm - rrow0)[:, None] * N + cols[None, :], mask=m, other=-1)
    else:
        r = tl.load(REF + (offm - rrow0)[:, None] * N + cols[None, :])

    s = tl.sum((comp == r).to(tl.int32), axis=0)  # [BN] int32
    if MASK_N:
        tl.store(OUT + DSTOFF + cols, s, mask=cols < N)
    else:
        tl.store(OUT + DSTOFF + cols, s)


@triton.jit
def fused_enc_kernel(
    W,
    Y,
    E,
    ref1,
    ref2,
    parts,
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
        _epi_enc(acc, E, ref1, parts, offm, 0, pid_n * BN, DU, BN, N, MASK_N, BM, pid_m * N)
    else:
        _epi_enc(acc, E, ref2, parts, offm, PIVOT, pid_n * BN, DV, BN, N, MASK_N, BM, pid_m * N)


@triton.jit
def _combine_parts(parts, cnt, NPM: tl.constexpr, N: tl.constexpr, BN: tl.constexpr):
    """[NPM, N] 的 row-block 部分和 → 原子累加到 cnt[N]。只有 N*NPM 个整数。"""
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    m = cols < N
    s = tl.zeros((BN,), dtype=tl.int32)
    for i in tl.static_range(NPM):
        s += tl.load(parts + i * N + cols, mask=m, other=0)
    tl.atomic_add(cnt + cols, s, mask=m)


# --------------------------------------------------------------- DEC epilogue
@triton.jit
def _epi_dec(
    acc,
    V,
    MB,
    offm,
    cbase,
    ccol0,
    NB: tl.constexpr,
    BN: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    """`(V - acc) mod q` → compress 到 1 bit → 8 行拼成一个字节 → 写 m_bytes。

    m_bytes[b][j] 的第 t 位来自第 `8j+t` 行 —— 和 `glue.compress_pack_kernel`
    的语义逐位一致（那里按 8 行循环，这里用一次 reshape 归约）。

    `tl.reshape([BM, BN] -> [BM//8, 8, BN])` 是行主序的拆分，所以
    元素 (i, j) 落到 (i//8, i%8, j)，正好是我们要的分组。
    """
    cols = cbase + tl.arange(0, BN)
    if MASK_N:
        m = cols[None, :] < N
        v = tl.load(V + offm[:, None] * N + cols[None, :], mask=m, other=0)
    else:
        v = tl.load(V + offm[:, None] * N + cols[None, :])

    x = v.to(tl.int32) - acc.to(tl.int32)
    x = x % QL
    x = tl.where(x < 0, x + QL, x)
    comp = (((x << 1) + 1664) // QL) & 1

    r = tl.reshape(comp, (BM // 8, 8, BN))
    w = (1 << tl.arange(0, 8))[None, :, None]
    byte = tl.sum(r * w, axis=1)  # [BM//8, BN]
    j = ccol0 + tl.arange(0, BM // 8)
    tl.store(MB + cols[None, :] * NB + j[:, None], byte.to(tl.uint8), mask=cols[None, :] < N)


@triton.jit
def fused_dec_kernel(
    W,
    U,
    V,
    MB,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _tile_id(pid, M, N, BM, BN, GM)
    acc = _gemm_tile(W, U, pid_m, pid_n, K, N, BM, BN, BK)
    offm = pid_m * BM + tl.arange(0, BM)
    _epi_dec(acc, V, MB, offm, pid_n * BN, pid_m * (BM // 8), M // 8, BN, N, BM, MASK_N)


# --------------------------------------------------------------- 配置选择
def _pick_cfg(M: int, dev) -> dict:
    """挑 tile。

    先取该后端的基准配置（`devices.gemm_cfg`），再按问题规模修一下：
    M 很小时（DEC 的 M=256、ML-KEM-512 的 M=512）128 宽的 CTA 太少，
    改成 BN=64 实测快 8~17%（B=512: 48.2 vs 57.1 µs；B=2048: 116 vs 133 µs）。
    """
    cfg = gemm_cfg(dev)
    if 2 * cfg["BM"] >= M:
        cfg["BN"] = min(cfg["BN"], 64)
        cfg["GM"] = min(cfg["GM"], 4)
    return cfg


# --------------------------------------------------------------- ENC 入口
def fused_enc_gemm(W, Y, E, ref, cnt, B: int, du: int, dv: int, pivot: int | None = None) -> bool:
    """ENC 重加密的 GEMM + mod q + compress + 比较 + 按列累加，一个内核。

    参数
    ----
    W    : [M, K] fp16，重加密矩阵（已按行展平）
    Y    : [K, B] fp16，采样出的 y
    E    : [M, B] int16，E1 || E2（E2 里已加过 1665*m_bits）
    ref  : (c1r, c2r) 两个 int16 张量，或单个 [M, B] int16
    cnt  : [B] int32，**累加**（调用前清零）
    du, dv : 两个压缩位宽（ENC: du 覆盖前 pivot 行，dv 覆盖剩下 256 行）
    pivot  : du 段的行数，默认 M - 256

    返回
    ----
    True 表示已执行；False 表示形状/dtype 不满足约束、**未写任何输出**，
    调用方应回退到「通用 GEMM + 逐元素比较」。
    """
    if W.dim() != 2 or Y.dim() != 2 or E.dim() != 2:
        return False
    M, K = W.shape
    if M <= 0 or K <= 0 or B <= 0:
        return False
    if Y.shape[0] != K or Y.shape[1] != B:
        return False
    if E.shape != (M, B) or cnt.shape != (B,) or cnt.dtype != torch.int32:
        return False
    if W.dtype != torch.float16 or Y.dtype != torch.float16:
        return False
    if E.dtype != torch.int16 or not (
        W.is_contiguous() and Y.is_contiguous() and E.is_contiguous() and cnt.is_contiguous()
    ):
        return False
    if pivot is None:
        pivot = M - 256
    if not (0 <= pivot <= M):
        return False
    if isinstance(ref, (tuple, list)):
        if len(ref) != 2:
            return False
        ref1, ref2 = ref
        if ref1.dtype != torch.int16 or ref2.dtype != torch.int16:
            return False
        if not (ref1.is_contiguous() and ref2.is_contiguous()):
            return False
        if tuple(ref1.shape) != (pivot, B) or tuple(ref2.shape) != (M - pivot, B):
            return False
    else:
        if ref.dtype != torch.int16 or tuple(ref.shape) != (M, B) or not ref.is_contiguous():
            return False
        ref1, ref2 = ref[:pivot], ref[pivot:]

    cfg = _pick_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM or pivot % BM:
        return False
    if not (1 <= du <= 15 and 1 <= dv <= 15):
        return False
    if not (W.device == E.device == Y.device == cnt.device):
        return False

    npm = M // BM
    grid = (npm * triton.cdiv(B, BN),)
    mask_n = (B % BN) != 0
    parts = torch.empty((npm, B), dtype=torch.int32, device=cnt.device)

    fused_enc_kernel[grid](
        W,
        Y,
        E,
        ref1,
        ref2,
        parts,
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
        mask_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    _combine_parts[(triton.cdiv(B, BLOCK_OUT),)](parts, cnt, npm, B, BLOCK_OUT, num_warps=1)
    return True


# --------------------------------------------------------------- DEC 入口
def fused_dec_gemm(W, U, V, B: int):
    """DEC 解密的 GEMM + `(V - W@U) mod q` + compress(1 bit) + 打包。

    Args:
        W: [M, K] fp16，解密矩阵（`keyprep` 产出的 `W_dec`）
        U: [K, B] fp16，解压后的 u
        V: [M, B] int16，解压后的 v

    Returns:
        m_bytes [B, M/8] uint8，或 **None**（形状/dtype/布局不满足约束，
        此时未写任何输出，调用方应回退）。
    """
    if W.dim() != 2 or U.dim() != 2 or V.dim() != 2:
        return None
    M, K = W.shape
    if M <= 0 or K <= 0 or B <= 0 or M % 8:
        return None
    if U.shape != (K, B) or V.shape != (M, B):
        return None
    if W.dtype != torch.float16 or U.dtype != torch.float16 or V.dtype != torch.int16:
        return None
    if not (W.is_contiguous() and U.is_contiguous() and V.is_contiguous()):
        return None
    if not (W.device == U.device == V.device):
        return None

    cfg = _pick_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM:
        return None

    nb = M // 8
    out = torch.empty((B, nb), dtype=torch.uint8, device=V.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(B, BN),)
    fused_dec_kernel[grid](
        W,
        U,
        V,
        out,
        M,
        K,
        B,
        BM,
        BN,
        BK,
        cfg["GM"],
        (B % BN) != 0,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


__all__ = [
    "fused_dec_gemm",
    "fused_dec_kernel",
    "fused_enc_gemm",
    "fused_enc_kernel",
]
