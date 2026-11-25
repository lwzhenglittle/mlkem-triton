"""在 GPU 上从 `dk` 重建 `W_dec` / `W_enc`（对照 cuPQC 的「无状态、每条密文现算 Â」）。

**这个模块不用于生产路径。** 它是为了回答一个测量问题：

    cuPQC 的 decaps 是无状态 API，每条密文都要重新跑 k² 次 SampleNTT
    （SHAKE-128，840 B/次）来得到 Â。我们的主路径把 Â 预先展开成 Toeplitz
    矩阵常驻显存，每条密文零 SHAKE。所以「我们比 cuPQC 快 1.61 倍」这个数字
    里，有多少来自「跳过 Â 生成」、有多少来自「Toeplitz GEMM 而不是 NTT」，
    分不出来。

把重建也搬进计时区间，就分得清了。

流程（全部在设备上，只有 1.5 KB 的输入从主机来）：

    dk ──┬─ rho(32B) ──▶ SHAKE-128(rate 168) × k² ──▶ 拒绝采样 ──▶ Â_hat
         ├─ t_hat   ──▶ ByteDecode12 ──┐
         └─ s_hat   ──▶ ByteDecode12 ──┤
                                       ▼
                              INTT(256×256 矩阵乘)
                                       ▼
                            中心化 ──▶ Toeplitz 展开 ──▶ W_dec / W_enc (fp16)

INTT 用一个预计算的 256×256 线性变换矩阵做，在 **fp64** 下精确：
矩阵元素与输入都 ≤ 3329，单项 ≤ 3329² = 1.1e7，256 项和 ≤ 2.84e9 < 2^53。
（也可以在 GPU 上写 8 级蝴蝶，但那样得到的是同样的数、更长的代码；这里要的是
**成本**，而 15 条 256 点的矩阵乘成本可以忽略。）

验证：`tests/test_rebuild.py` 要求重建出的 `W_dec` / `W_enc` 与
`keyprep.parse_key`（CPU numpy 路径）**逐元素相等**。
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ..devices import resolve_device, warps
from ..ntt_ref import intt
from ..params import PARAMS, N, Q

RATE_SHAKE128 = 168
DOMAIN = 0x1F

# ---------------------------------------------------------------- INTT 矩阵
_INTT_M: dict[torch.device, torch.Tensor] = {}


def intt_matrix(dev) -> torch.Tensor:
    """返回 **转置** 后的 INTT 矩阵 `MT`，使 `x @ MT == intt(x)`。

    `intt` 在 Z_q 上是线性的，所以 `intt(x)[i] = Σ_j M[i,j]·x[j]`，其中
    `M[:,j] = intt(e_j)`。批量做的时候用 `x @ Mᵀ` 更顺手（行是样本），
    所以这里直接缓存 `Mᵀ`。
    """
    t = _INTT_M.get(dev)
    if t is None:
        m = np.empty((N, N), dtype=np.float64)
        for j in range(N):
            e = np.zeros(N, dtype=np.int64)
            e[j] = 1
            m[:, j] = intt(e)  # 第 j 列 = intt(e_j)
        t = torch.from_numpy(np.ascontiguousarray(m.T)).to(dev)
        _INTT_M[dev] = t
    return t


# ---------------------------------------------------------------- SHAKE-128 输入
def _padded_msgs(rho: bytes, k: int) -> np.ndarray:
    """k² 条 `rho ‖ j ‖ i` 的 pad10*1 输入，[k², 168] uint8（主机侧，1.5 KB）。"""
    msgs = np.zeros((k * k, RATE_SHAKE128), dtype=np.uint8)
    n = 0
    for i in range(k):
        for j in range(k):
            msgs[n, :32] = np.frombuffer(rho, np.uint8)
            msgs[n, 32] = j
            msgs[n, 33] = i
            msgs[n, 34] = DOMAIN
            msgs[n, RATE_SHAKE128 - 1] |= 0x80
            n += 1
    return msgs


# ---------------------------------------------------------------- 拒绝采样
@triton.jit
def _sample_ntt_kernel(
    BUF, OUT, NB: tl.constexpr, N: tl.constexpr, QL: tl.constexpr, NCAND: tl.constexpr
):
    """840 B → 256 个 [0,q) 的 NTT 域系数（FIPS 203 Alg. 7）。

    一个 program 处理一条消息：把 280 组三元组展开成 560 个候选值，
    `tl.cumsum` 求前缀位置，再按位置散射，只保留前 256 个合法值。
    """
    pid = tl.program_id(0)
    base = BUF + pid * NB
    c = tl.arange(0, NCAND)  # 候选序号 0..639（>560）
    tri = c // 2  # 三元组序号
    valid = c < 2 * (NB // 3)
    b0 = tl.load(base + 3 * tri, mask=valid, other=0).to(tl.int32)
    b1 = tl.load(base + 3 * tri + 1, mask=valid, other=0).to(tl.int32)
    b2 = tl.load(base + 3 * tri + 2, mask=valid, other=0).to(tl.int32)
    d1 = b0 + 256 * (b1 & 0x0F)
    d2 = (b1 >> 4) + 16 * b2
    val = tl.where((c % 2) == 0, d1, d2)
    ok = valid & (val < QL)
    pos = tl.cumsum(ok.to(tl.int32), axis=0) - 1  # 从 0 开始的位置
    tl.store(OUT + pid * N + pos, val, mask=ok & (pos >= 0) & (pos < N))


# ---------------------------------------------------------------- Toeplitz 展开
@triton.jit
def _toeplitz_kernel(A, META, OUT, K, N: tl.constexpr, BR: tl.constexpr):
    """把一个 256×256 的 Toeplitz 分块写进 W。

    `T(a)[i][j] = a[(i-j) mod N] * (+1 若 i >= j 否则 -1)`，`a` 已中心化。
    A: [n_blocks, N] int32（中心化后的系数）；META: [n_blocks, 2] = (行块, 列块)。
    """
    pid = tl.program_id(0)
    nrow = tl.cdiv(N, BR)
    blk = pid // nrow  # 第几个 Toeplitz 分块
    r0 = (pid % nrow) * BR
    src = tl.load(META + blk * 3 + 0)  # 用 coef 里的哪一行
    row0 = tl.load(META + blk * 3 + 1)  # 写到 W 的哪个行块
    col0 = tl.load(META + blk * 3 + 2)  # 写在 W 的哪个列块
    row = r0 + tl.arange(0, BR)[:, None]
    col = tl.arange(0, N)[None, :]
    # 注意：Triton 的 int32 `%` 是 C 语义（向零取整），`(row-col) % 256` 对负数
    # 会给负索引。N 是 2 的幂，所以直接用位与。numpy 那边是 floor-mod，两者
    # 在这里不等价 —— 这个坑踩过一次。
    idx = (row - col) & (N - 1)
    a = tl.load(A + src * N + idx)
    sgn = tl.where(row >= col, 1, -1)
    dst = OUT + (row0 * N + row) * K + (col0 * N + col)
    tl.store(dst, (a * sgn).to(tl.float16))


# ---------------------------------------------------------------- 主入口
class GpuKeyRebuilder:
    """从 `dk` 在 GPU 上重建 `W_dec` / `W_enc`，接口与 `keyprep.parse_key` 对齐。"""

    def __init__(self, ps: str, device=None):
        self.ps = ps
        self.p = PARAMS[ps]
        self.k = self.p.k
        self.device = resolve_device(device)
        self.INTT = intt_matrix(self.device)
        # 元数据三元组：(coef 行号, W 行块, W 列块)。一次性算好、常驻。
        # coef 的行序 = [Â(0..k²-1) 行主序, t̂(k²..k²+k-1), ŝ(k²+k..k²+2k-1)]
        k = self.k
        enc, dec = [], []
        for i in range(k):  # u' = Âᵀ y  → 行块 i、列块 j 用 Â[j][i]
            for j in range(k):
                enc.append((j * k + i, i, j))  # 注意 A[j][i]，不是 A[i][j]
        for j in range(k):  # v' = t̂ᵀ y
            enc.append((k * k + j, k, j))
        for j in range(k):  # w = -ŝᵀ u
            dec.append((k * k + k + j, 0, j))
        self.meta_enc = torch.tensor(enc, dtype=torch.int32, device=self.device)
        self.meta_dec = torch.tensor(dec, dtype=torch.int32, device=self.device)
        self._sha = None

    # -- SHAKE-128 内核（第一次调用时编译） --
    def _shake(self):
        if self._sha is None:
            from .shake import get_sponge

            self._sha = get_sponge(RATE_SHAKE128, 5 * RATE_SHAKE128)
        return self._sha

    @staticmethod
    def rho_of(dk: bytes) -> bytes:
        """从 `dk` 里取 ρ（ek 的最后 32 字节）。"""
        k = (len(dk) - 96) // 768
        return dk[768 * k - 32 : 768 * k]

    def _intt_batch(self, x_hat: torch.Tensor) -> torch.Tensor:
        """[n, 256] NTT 域 → 系数域，再中心化到 [-1664, 1664]（int32）。"""
        c = torch.matmul(x_hat.to(torch.float64), self.INTT)  # x @ Mᵀ == intt(x)
        c = c.to(torch.int64) % Q
        c = torch.where(c > (Q - 1) // 2, c - Q, c)
        return c.to(torch.int32)

    def rebuild(self, dk: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (W_dec [256, 256k] fp16, W_enc [256(k+1), 256k] fp16)，都已中心化。"""
        k, dev = self.k, self.device
        dk_pke = dk[: 384 * k]
        ek = dk[384 * k : 768 * k + 32]
        rho = ek[384 * k :]
        assert len(rho) == 32

        # 1) Â_hat：SHAKE-128 × k² + 拒绝采样
        n = k * k
        outw = 5 * RATE_SHAKE128 // 8  # 840/8 = 105
        msgs = torch.from_numpy(_padded_msgs(rho, k)).to(dev)
        out64 = torch.empty((n, outw), dtype=torch.int64, device=dev)
        self._shake()[(triton.cdiv(n, 128),)](
            msgs.view(torch.int64),
            out64,
            n,
            1,
            RATE_SHAKE128 // 8,
            OUTW=outw,
            BLOCK=128,
            num_warps=warps("shake", 4),
        )
        buf = out64.view(torch.uint8).view(n, 5 * RATE_SHAKE128)

        a_hat = torch.empty((n, N), dtype=torch.int32, device=dev)
        _sample_ntt_kernel[(n,)](
            buf, a_hat, 5 * RATE_SHAKE128, N, Q, 1024, num_warps=warps("sample_ntt", 4)
        )

        # 2) t_hat / s_hat：dk 里的 12-bit 字段
        from ..pipeline import unpack_bits_gpu

        # 行序要和 coef 对齐：[t̂(k 条), ŝ(k 条)]
        ts = torch.cat(
            [
                torch.from_numpy(np.frombuffer(ek[: 384 * k], np.uint8).reshape(k, 384).copy()).to(
                    dev
                ),
                torch.from_numpy(np.frombuffer(dk_pke, np.uint8).reshape(k, 384).copy()).to(dev),
            ]
        )
        ts_hat = unpack_bits_gpu(ts, 12, N).to(torch.int32)  # [2k, 256]

        # 3) 全部 INTT 到系数域并中心化
        coef = self._intt_batch(torch.cat([a_hat, ts_hat]))  # [k²+2k, 256]

        # 4) Toeplitz 展开
        M, K = 256 * (k + 1), 256 * k
        W_enc = torch.empty((M, K), dtype=torch.float16, device=dev)
        W_dec = torch.empty((256, K), dtype=torch.float16, device=dev)
        BR = 32
        _toeplitz_kernel[(self.meta_enc.numel() // 3 * (N // BR),)](
            coef, self.meta_enc, W_enc, K, N, BR, num_warps=warps("toeplitz", 4)
        )
        _toeplitz_kernel[(self.meta_dec.numel() // 3 * (N // BR),)](
            coef, self.meta_dec, W_dec, K, N, BR, num_warps=warps("toeplitz", 4)
        )
        return W_dec, W_enc


__all__ = ["GpuKeyRebuilder", "intt_matrix"]
