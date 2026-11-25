"""Triton 上的 Kyber/ML-KEM 正向 NTT（FIPS 203 Alg. 9）。

**这是本仓库里唯一和 Toeplitz 技巧无关的部件。** decaps / encaps 都不需要 NTT
（多项式乘法被折叠成固定矩阵的 GEMM），但 keygen 不行：`ek` 里必须放 `t̂` 的
NTT 域表示，而 `s`/`e` 每个密钥都是新的，没有任何「跨操作固定的矩阵」可用。
所以这里就是老老实实一遍蝴蝶。

映射：一个 program 处理 `BR` 条多项式（每条 256 个 int32）。
7 级蝴蝶（length = 128, 64, …, 2），每级用 `tl.gather` 做 `idx ^ L` 的配对交换。

配对索引 `idx ^ L` 是等价的：在按 `2L` 对齐的块内，`idx ^ L` 正好把两个半区互换。
`ZETAS` 每级各有 `2^s` 个，偏移也是 `2^s`（i 从 1 开始累加 1,2,4,…）。

实测（GB10，B=8192 等价规模）：见 `benchmarks/bench_keygen.py`。
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..devices import warps
from ..ntt_ref import GAMMAS, ZETAS

Q = 3329
QL = tl.constexpr(Q)
N = 256


_GM: dict = {}


_ZT_FLAT: dict = {}


def _zt_flat(dev):
    """每个 stage × 每个元素位置的 ζ，展平成 [7*256]，一次 load 就够。

    原来每级每个元素都要算 `idx // (2*L)` 再 gather 常数表 —— 7 级就是 7 次整数
    除法 + 7 次小表 gather。整除在 GPU 上是 20+ 条指令。
    """
    t = _ZT_FLAT.get(dev)
    if t is None:
        import numpy as _np

        z = _np.asarray([int(v) for v in ZETAS], dtype=_np.int32)
        flat = _np.empty(7 * 256, dtype=_np.int32)
        L = 128
        off = 1
        for s_i in range(7):
            idx = _np.arange(256)
            flat[s_i * 256 : (s_i + 1) * 256] = z[off + idx // (2 * L)]
            L //= 2
            off *= 2
        t = torch.from_numpy(flat).to(dev)
        _ZT_FLAT[dev] = t
    return t


def _gammas(dev):
    t = _GM.get(dev)
    if t is None:
        t = torch.tensor([int(g) for g in GAMMAS[:128]], dtype=torch.int32, device=dev)
        _GM[dev] = t
    return t


@triton.jit
def _modq(v):
    v = v % QL
    return tl.where(v < 0, v + QL, v)


@triton.jit
def ntt_kernel(X, OUT, ZT, P, N: tl.constexpr, BR: tl.constexpr):
    """[P, 256] 系数域 int32 → NTT 域 int32。原地不必要，输入输出可同一块。"""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < P
    idx = tl.arange(0, N)
    x = tl.load(X + rows[:, None] * N + idx[None, :], mask=m[:, None], other=0).to(tl.int32)
    x = _modq(x)

    L = N // 2
    for _s in tl.static_range(7):
        pv = tl.gather(x, (idx ^ L)[None, :].broadcast_to((BR, N)), axis=1)
        z = tl.load(ZT + _s * N + idx).to(tl.int32)[None, :]
        lo = ((idx & L) == 0)[None, :]
        # 低半 out = a + z·b，高半 out = a − z·b（a 是块内第 j 个、b 是第 j+L 个）。
        # 把 a/b 用 where 归一到两个半区，就只需要 **一次** 乘法 + 两次 modq，
        # 而不是「两个分支各算一遍」（那样是两次乘法 + 四次 modq）。
        # 实测：这一改 + BR 从 16 降到 2，NTT 从 15.2 ms 降到 1.5 ms（P=24576）。
        a = tl.where(lo, x, pv)
        b = tl.where(lo, pv, x)
        sg = tl.where(lo, 1, -1)
        x = _modq(a + sg * _modq(z * b))
        L = L // 2

    tl.store(OUT + rows[:, None] * N + idx[None, :], x, mask=m[:, None])


@triton.jit
def pow_mac_kernel(
    A, S, E, OUT, GAM, P, K: tl.constexpr, N: tl.constexpr, HALF: tl.constexpr, BR: tl.constexpr
):
    """`t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]`（FIPS 203 Alg. 11/12）。

    `multiply_ntts` 是 128 个**二次扩域**乘法：

        c0 = a0·b0 + (a1·b1 mod q)·γ
        c1 = a0·b1 + a1·b0

    所以偶数/奇数系数是两个独立的通道。**直接在加载时按 stride=2 取**，就完全
    不需要 gather —— 这一版把 16 次 `tl.gather` 换成了纯 strided load/store。
    （GB10 上 keygen 里这一步原本占 33%，是整个 keygen 最大的一块。）
    """
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < P
    h = tl.arange(0, HALF)
    b = rows // K
    i = rows - b * K
    arow = b * (K * K) + i * K
    srow = b * K

    e_base = E + rows[:, None] * N
    acc0 = tl.load(e_base + 2 * h[None, :], mask=m[:, None], other=0).to(tl.int32)
    acc1 = tl.load(e_base + 2 * h[None, :] + 1, mask=m[:, None], other=0).to(tl.int32)
    g = tl.load(GAM + h).to(tl.int32)[None, :]

    for j in tl.static_range(4):  # K ∈ {2,3,4}，跑满 4 次
        if j < K:
            ab = A + (arow + j)[:, None] * N + 2 * h[None, :]
            sb = S + (srow + j)[:, None] * N + 2 * h[None, :]
            a0 = tl.load(ab, mask=m[:, None], other=0).to(tl.int32)
            a1 = tl.load(ab + 1, mask=m[:, None], other=0).to(tl.int32)
            b0 = tl.load(sb, mask=m[:, None], other=0).to(tl.int32)
            b1 = tl.load(sb + 1, mask=m[:, None], other=0).to(tl.int32)
            acc0 = _modq(acc0 + _modq(a0 * b0 + _modq(_modq(a1 * b1) * g)))
            acc1 = _modq(acc1 + _modq(a0 * b1 + a1 * b0))

    o_base = OUT + rows[:, None] * N
    tl.store(o_base + 2 * h[None, :], acc0, mask=m[:, None])
    tl.store(o_base + 2 * h[None, :] + 1, acc1, mask=m[:, None])


def forward_ntt(x: torch.Tensor, br: int = 2) -> torch.Tensor:
    """[P, 256] int32（系数域，[0,q)）→ [P, 256] int32（NTT 域）。"""
    P = x.shape[0]
    out = torch.empty_like(x, dtype=torch.int32)
    ntt_kernel[(triton.cdiv(P, br),)](
        x, out, _zt_flat(x.device), P, N, br, num_warps=warps("ntt", 2)
    )
    return out


def pow_mac(a, s, e, k: int, br: int = 16) -> torch.Tensor:
    """[B*k*k,256] × [B*k,256] → t̂ [B*k,256]；必须已经过 NTT。"""
    P = s.shape[0]
    out = torch.empty_like(s, dtype=torch.int32)
    pow_mac_kernel[(triton.cdiv(P, br),)](
        a,
        s,
        e,
        out,
        _gammas(s.device),
        P,
        k,
        N,
        N // 2,
        br,
        num_warps=warps("pow_mac", 4),
    )
    return out


__all__ = ["forward_ntt", "ntt_kernel", "pow_mac", "pow_mac_kernel"]


# --------------------------------------------------------------- ByteEncode_12
@triton.jit
def encode12_kernel(V, OUT, N: tl.constexpr, BYTES: tl.constexpr, OBLK: tl.constexpr):
    """`ByteEncode_12`：一条多项式的 256 个 [0,4096) 系数 → 384 字节。

    D=12 > 8，所以一个字节最多横跨 **2** 个系数（和封装那边 dv=5 会跨 3 个不同）。
    """
    p = tl.program_id(0)
    o = tl.program_id(1) * OBLK + tl.arange(0, OBLK)
    m = o < BYTES
    bit0 = o * 8
    c0 = bit0 // 12
    s0 = bit0 - c0 * 12
    c1 = tl.minimum(c0 + 1, N - 1)
    v0 = tl.load(V + p * N + c0, mask=m, other=0).to(tl.int32)
    v1 = tl.load(V + p * N + c1, mask=m, other=0).to(tl.int32)
    byte = ((v0 >> s0) | (v1 << (12 - s0))) & 0xFF
    tl.store(OUT + p * BYTES + o, byte.to(tl.uint8), mask=m)


def byte_encode12(vals: torch.Tensor) -> torch.Tensor:
    """[P, 256] int32（[0,4096)）→ [P, 384] uint8。"""
    P = vals.shape[0]
    out = torch.empty((P, 384), dtype=torch.uint8, device=vals.device)
    encode12_kernel[(P, 3)](vals, out, N, 384, 128, num_warps=warps("encode12", 4))
    return out


__all__ += ["byte_encode12", "encode12_kernel"]
