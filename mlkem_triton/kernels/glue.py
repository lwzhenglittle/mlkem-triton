"""decaps 流水线里两块「纯 PyTorch glue」的融合 Triton 内核。

**(A) `compress_pack`** —— 替换

    w = (V.to(int32) - acc.to(int32)) % Q      # DEC 那趟 mod q
    m_bits  = compress_gpu(w, 1)               # [256,B] ∈ {0,1}
    m_bytes = pack_bits_gpu(m_bits, 1)         # [B,32] uint8

一个 program 负责「BN 列 × 8 行」= 每列 1 个字节：8 行按位移 OR 拼字节，
不需要 reshape / 转置，也不落 m_bits 中间张量（下游 `build_ye` 直接读 m_bytes）。

**(B) `build_ye`** —— 替换

    Y  = y.permute(1,2,0).reshape(k*256, B).to(fp16)
    E1 = e[:, :k].permute(1,2,0).reshape(k*256, B).to(int16)
    E2 = (e[:, k].T + 1665 * m_bits).to(int16)
    E  = torch.cat([E1, E2], dim=0)

一个 launch 直接从 `y`/`e`/`m_bytes` 生成 `Y [256k,B] fp16` 与
`E [256(k+1),B] int16`，不再经过 permute+reshape+cat 的中间张量。
网格是 (列块, k+1)：前 k 个 program 各转置一个多项式，最后一个做
`e[:,k] + 1665·bit`（bit 从 m_bytes 就地取，每个字节被广播给 8 行）。

注意 `Y` 这里输出的是**连续**张量。旧代码 `y.permute(1,2,0).reshape(...).to(fp16)`
实际返回 stride=(1, 256k) 的转置视图（reshape 是 view、`.to(fp16)` 保留 memory format），
值相同但非连续 —— 而 `kernels/fused_enc_gemm.py` 要求 Y 连续，所以 ENC 那趟融合
GEMM 之前一直没生效、整条流水线走的是 mm + `_cmp_kernel` 回退路径。
形状/dtype/设备不满足约束时 `compress_pack` / `build_ye` 返回 None，调用方回退。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..devices import warps

Q = 3329
QL = tl.constexpr(Q)
NCOEF = 256

# 取自 bench/gl_sweep.py 的扫描（B=4096, ML-KEM-768）
PACK_BN = 256
PACK_WARPS = 2
YE_BN = 32
YE_CHUNK = 64
YE_WARPS = 2


# --------------------------------------------------------------- (A) compress+pack
@triton.jit
def compress_pack_kernel(acc, V, mbytes, B, BN: tl.constexpr):
    """acc [256,B] fp32, V [256,B] int16 -> mbytes [B,32] uint8。

    x = (v - acc) mod q（负数按 Python 语义补回非负），再 compress 到 1 bit，
    8 行拼一个字节（第 j 个字节的 bit t 来自第 8j+t 行）。
    """
    pid_b = tl.program_id(0)
    j = tl.program_id(1)
    cols = pid_b * BN + tl.arange(0, BN)
    m = cols < B
    byte = tl.zeros((BN,), dtype=tl.int32)
    for t in tl.static_range(8):
        off = (j * 8 + t) * B + cols
        a = tl.load(acc + off, mask=m, other=0.0)
        v = tl.load(V + off, mask=m, other=0)
        x = v.to(tl.int32) - a.to(tl.int32)
        x = x % QL
        x = tl.where(x < 0, x + QL, x)
        byte |= ((((x << 1) + 1664) // QL) & 1) << t
    tl.store(mbytes + cols * 32 + j, byte.to(tl.uint8), mask=m)


def can_compress_pack(V, acc, B: int) -> bool:
    """形状/dtype/布局是否满足内核约束（只看元信息，静态可判定）。"""
    if not isinstance(B, int) or B <= 0:
        return False
    if V.dim() != 2 or acc.dim() != 2:
        return False
    if tuple(V.shape) != (NCOEF, B) or tuple(acc.shape) != (NCOEF, B):
        return False
    if V.dtype != torch.int16 or acc.dtype != torch.float32:
        return False
    if not (V.is_contiguous() and acc.is_contiguous()):
        return False
    return V.device == acc.device


def compress_pack(V, acc, B: int) -> torch.Tensor:
    """[256,B] int16 / fp32 -> m_bytes [B,32] uint8。"""
    out = torch.empty((B, 32), dtype=torch.uint8, device=V.device)
    compress_pack_kernel[(triton.cdiv(B, PACK_BN), 32)](
        acc, V, out, B, BN=PACK_BN, num_warps=warps("glue_pack", PACK_WARPS)
    )
    return out


# --------------------------------------------------------------- (B) build Y / E
@triton.jit
def build_ye_kernel(
    y,
    e,
    mbytes,
    Y,
    E,
    B,
    K: tl.constexpr,
    BN: tl.constexpr,
    CHUNK: tl.constexpr,
    N: tl.constexpr = 256,
):
    """y [B,K,256] int16, e [B,K+1,256] int16, mbytes [B,32] uint8
    -> Y [K*256,B] fp16 (行序 r*256+n), E [(K+1)*256,B] int16。"""
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    cols = pid_b * BN + tl.arange(0, BN)
    m = cols < B
    for c0 in range(0, N, CHUNK):
        n = c0 + tl.arange(0, CHUNK)
        if pid_r < K:
            # Y 段：y[b, r, n] -> Y[r*256+n, b]
            v = tl.load(
                y + cols[:, None] * (K * N) + (pid_r * N + n)[None, :], mask=m[:, None], other=0
            )
            tl.store(
                Y + (pid_r * N + n)[:, None] * B + cols[None, :],
                tl.trans(v).to(tl.float16),
                mask=m[None, :],
            )
            ve = tl.load(
                e + cols[:, None] * ((K + 1) * N) + (pid_r * N + n)[None, :],
                mask=m[:, None],
                other=0,
            )
            tl.store(
                E + (pid_r * N + n)[:, None] * B + cols[None, :], tl.trans(ve), mask=m[None, :]
            )
        else:
            # E2 段：e[b, K, n] + 1665 * bit(n, b)
            ve = tl.load(
                e + cols[:, None] * ((K + 1) * N) + (K * N + n)[None, :], mask=m[:, None], other=0
            )
            mb = tl.load(mbytes + cols[:, None] * 32 + (n // 8)[None, :], mask=m[:, None], other=0)
            bits = (mb.to(tl.int32) >> (n % 8)[None, :]) & 1
            tl.store(
                E + (K * N + n)[:, None] * B + cols[None, :],
                tl.trans(ve.to(tl.int32) + 1665 * bits).to(tl.int16),
                mask=m[None, :],
            )


def can_build_ye(y, e, mbytes, B: int, K: int) -> bool:
    if not isinstance(B, int) or not isinstance(K, int):
        return False
    if B <= 0 or K <= 0:
        return False
    if y.dim() != 3 or e.dim() != 3 or mbytes.dim() != 2:
        return False
    if tuple(y.shape) != (B, K, NCOEF) or tuple(e.shape) != (B, K + 1, NCOEF):
        return False
    if tuple(mbytes.shape) != (B, 32):
        return False
    if y.dtype != torch.int16 or e.dtype != torch.int16 or mbytes.dtype != torch.uint8:
        return False
    if not (y.is_contiguous() and e.is_contiguous() and mbytes.is_contiguous()):
        return False
    return y.device == e.device == mbytes.device


def build_ye(y, e, mbytes, B: int, K: int):
    """-> (Y [K*256,B] fp16, E [(K+1)*256,B] int16)，都是新分配的连续张量。"""
    dev = y.device
    Y = torch.empty((K * NCOEF, B), dtype=torch.float16, device=dev)
    E = torch.empty(((K + 1) * NCOEF, B), dtype=torch.int16, device=dev)
    build_ye_kernel[(triton.cdiv(B, YE_BN), K + 1)](
        y, e, mbytes, Y, E, B, K=K, BN=YE_BN, CHUNK=YE_CHUNK, num_warps=warps("glue_ye", YE_WARPS)
    )
    return Y, E


def unpack_mbits(mbytes: torch.Tensor, B: int) -> torch.Tensor:
    """m_bytes [B,32] uint8 -> m_bits [256,B] int32（回退路径用）。"""
    b = mbytes.to(torch.int32)
    shifts = torch.arange(8, device=mbytes.device, dtype=torch.int32)
    bits = (b[:, :, None] >> shifts) & 1  # [B,32,8]
    return bits.reshape(B, 256).T.contiguous()


# ------------------------------------------------- (C) CBD 直接写出 Y / E
# 把 §B 再往前推一步：sponge 出来的 y_buf/e_buf 经 CBD 之后不再落 y/e 中间张量，
# 而是直接转置写进 Y/E。流量（B=4096,k=3,eta=2）：48.4 MB -> 15.2 MB。
CBD_BN = 32
CBD_WARPS = 4


@triton.jit
def cbd_ye_kernel(
    bufs,
    mbytes,
    OUT,
    B,
    K: tl.constexpr,
    RSTRIDE: tl.constexpr,
    BN: tl.constexpr,
    ETA: tl.constexpr,
    F16: tl.constexpr,
    E2: tl.constexpr,
    N: tl.constexpr = 256,
):
    """一只内核两种用法：

    - `F16=True, E2=False`：y_buf [B*K,64η] uint8 -> Y [K*256,B] fp16
    - `F16=False, E2=True`：e_buf [B*(K+1),64η] uint8 -> E [(K+1)*256,B] int16
      且当 pid_r == K 那一段（E2）要加上 1665·bit（bit 从 m_bytes 就地取）。

    bufs 的行序是 b*R + r（和 `cbd_gpu(...).reshape(B, R, 256)` 一致），
    所以一个 program 固定 r、走 BN 个 b，产出 [256, BN] 的块，转置后写
    Y/E 的行 r*256+n、列 b —— 与 §B 的布局完全一样。
    """
    pid_b = tl.program_id(0)
    pid_r = tl.program_id(1)
    cols = pid_b * BN + tl.arange(0, BN)
    m = cols < B
    nb = tl.arange(0, 64 * ETA)
    v = tl.load(
        bufs + cols[:, None] * RSTRIDE + pid_r * (64 * ETA) + nb[None, :], mask=m[:, None], other=0
    ).to(tl.int32)
    lo = v & 0x0F
    hi = (v >> 4) & 0x0F
    c0 = ((lo & 1) + ((lo >> 1) & 1)) - (((lo >> 2) & 1) + ((lo >> 3) & 1))
    c1 = ((hi & 1) + ((hi >> 1) & 1)) - (((hi >> 2) & 1) + ((hi >> 3) & 1))
    v = tl.interleave(c0, c1)  # [BN, 256] ∈ [-2,2]
    n = tl.arange(0, N)
    if E2:
        if pid_r == K:
            mb = tl.load(mbytes + cols[:, None] * 32 + (n // 8)[None, :], mask=m[:, None], other=0)
            bits = (mb.to(tl.int32) >> (n % 8)[None, :]) & 1
            v = v + 1665 * bits
    dst = OUT + (pid_r * N + n)[:, None] * B + cols[None, :]
    if F16:
        tl.store(dst, tl.trans(v).to(tl.float16), mask=m[None, :])
    else:
        tl.store(dst, tl.trans(v).to(tl.int16), mask=m[None, :])


def can_cbd_y(y_buf, B: int, K: int, eta: int) -> bool:
    if eta != 2 or B <= 0 or K <= 0:
        return False
    if y_buf.dim() != 2 or y_buf.dtype != torch.uint8 or not y_buf.is_contiguous():
        return False
    return tuple(y_buf.shape) == (B * K, 64 * eta)


def can_cbd_e(e_buf, mbytes, B: int, K: int, eta: int) -> bool:
    if eta != 2 or B <= 0 or K <= 0:
        return False
    if e_buf.dim() != 2 or e_buf.dtype != torch.uint8 or not e_buf.is_contiguous():
        return False
    if mbytes.dim() != 2 or mbytes.dtype != torch.uint8 or not mbytes.is_contiguous():
        return False
    if tuple(mbytes.shape) != (B, 32):
        return False
    return tuple(e_buf.shape) == (B * (K + 1), 64 * eta) and e_buf.device == mbytes.device


def cbd_y(y_buf, B: int, K: int, eta: int = 2) -> torch.Tensor:
    Y = torch.empty((K * NCOEF, B), dtype=torch.float16, device=y_buf.device)
    cbd_ye_kernel[(triton.cdiv(B, CBD_BN), K)](
        y_buf,
        y_buf,
        Y,
        B,
        K=K,
        RSTRIDE=K * 64 * eta,
        BN=CBD_BN,
        ETA=eta,
        F16=True,
        E2=False,
        num_warps=warps("glue_cbd", CBD_WARPS),
    )
    return Y


def cbd_e(e_buf, mbytes, B: int, K: int, eta: int = 2) -> torch.Tensor:
    E = torch.empty(((K + 1) * NCOEF, B), dtype=torch.int16, device=e_buf.device)
    cbd_ye_kernel[(triton.cdiv(B, CBD_BN), K + 1)](
        e_buf,
        mbytes,
        E,
        B,
        K=K,
        RSTRIDE=(K + 1) * 64 * eta,
        BN=CBD_BN,
        ETA=eta,
        F16=False,
        E2=True,
        num_warps=warps("glue_cbd", CBD_WARPS),
    )
    return E


__all__ = [
    "CBD_BN",
    "CBD_WARPS",
    "PACK_BN",
    "PACK_WARPS",
    "YE_BN",
    "YE_CHUNK",
    "YE_WARPS",
    "build_ye",
    "build_ye_kernel",
    "can_build_ye",
    "can_cbd_e",
    "can_cbd_y",
    "can_compress_pack",
    "cbd_e",
    "cbd_y",
    "cbd_ye_kernel",
    "compress_pack",
    "compress_pack_kernel",
    "unpack_mbits",
]
