"""ML-KEM 封装（FIPS 203 Alg. 17），全设备常驻。

和 decaps 一样，核心是**用 Toeplitz 矩阵乘代替 NTT**：

    [u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]

`W_enc` 只依赖公钥 `ek`（即 `Â = SampleNTT(ρ)` 和 `t`），对同一个 `ek` 是固定的。
所以「同一把公钥、海量封装」的形态下可以一次展开、反复用 —— 这正是我们相对
NTT 实现的结构性优势所在。**纯 Triton，没有 NTT。**

数据流：

    m(32B) ─┬─▶ pad(m ‖ H(ek)) ─▶ sponge(SHA3-512) ─▶ K_pre ‖ r_seed
            │                        K = K_pre（FIPS 203 没有 KDF！）
            │                          ┌───────────────────┴──────────┐
            │                          ▼                              ▼
            │                PRF(r_seed, 0..k-1)          PRF(r_seed, k..2k)
            │                          ▼                              ▼
            │                     CBD_η1 → r                   CBD_η2 → e1, e2
            │                          │                              │
            └──▶ m_bits ───────────────┼────── e2 + 1665·m_bits ─────┘
                                       ▼
                            W_enc @ r + [e1; e2']  ──▶ mod q ──▶ Compress
                                       ▼
                              ByteEncode_du/dv ──▶ c (1088 B)
"""

from __future__ import annotations

import numpy as np
import torch

from . import keyprep, ops
from .devices import resolve_device
from .kernels.encaps import fused_encaps, pack_ciphertext
from .kernels.shake import SHA3_DOMAIN, SHAKE_DOMAIN
from .params import PARAMS
from .pipeline import RATE_SHA3_512, RATE_SHAKE, _pad2_gpu, _pad_prf_gpu, sponge_gpu

RATE_SHA3_256 = 136


class Encapsulator:
    """把一把封装公钥装到设备上，之后可以批量封装。

    Args:
        ps: 参数集名。
        ek: 封装公钥字节串（长度 ``384k+32``）。
        device: ``None`` 自动选。
    """

    def __init__(self, ps: str, ek: bytes, device=None):
        p = PARAMS[ps]
        self.ps, self.p = ps, p
        self.device = resolve_device(device)
        self.kp = keyprep.parse_ek(ps, ek)
        self.ek = ek
        dev = self.device
        self.h_ek = torch.frombuffer(bytearray(self.kp.h_ek), dtype=torch.uint8).to(dev)
        W = torch.from_numpy(np.ascontiguousarray(self.kp.W_enc))
        assert W.abs().max().item() <= 2048
        self.W_enc = W.to(torch.float16).to(dev)

    # ---- 主流程 ----
    def encapsulate(self, msgs) -> tuple[np.ndarray, np.ndarray]:
        """批量封装。

        Args:
            msgs: ``[B, 32]`` uint8 数组 / ``list[bytes]`` / 一整块 ``bytes``
                  （每条 32 字节的消息，调用方负责随机性）。

        Returns:
            ``(c, K)``：密文 ``[B, ct_size]`` uint8，共享密钥 ``[B, 32]`` uint8。
        """
        arr = _to_msgs2d(msgs)
        B = arr.shape[0]
        m = torch.from_numpy(arr).to(self.device)
        c, K = self.encapsulate_torch(m, B)
        return c.cpu().numpy(), K.cpu().numpy()

    def encapsulate_torch(self, m: torch.Tensor, B: int):
        """纯设备侧流程。`m` 是 ``[B, 32]`` uint8。"""
        p, k, dev = self.p, self.p.k, self.device
        if m.dtype != torch.uint8 or m.shape != (B, 32):
            raise ValueError(f"m 需要是 [B,32] uint8，收到 {tuple(m.shape)} {m.dtype}")

        # 1. (K_pre, r_seed) = G(m ‖ H(ek))     —— SHA3-512, rate 72
        gr = sponge_gpu(_pad2_gpu(m, self.h_ek, B, RATE_SHA3_512, SHA3_DOMAIN), RATE_SHA3_512, 64)
        k_pre, r_seed = gr[:, :32], gr[:, 32:]

        # 2. PRF：r_seed ‖ [N] → CBD 的输入字节流
        eta1, eta2 = p.eta1, p.eta2
        y_buf = sponge_gpu(
            _pad_prf_gpu(r_seed, B * k, RATE_SHAKE, SHAKE_DOMAIN, mod=k, off=0),
            RATE_SHAKE,
            64 * eta1,
        )
        e_buf = sponge_gpu(
            _pad_prf_gpu(r_seed, B * (k + 1), RATE_SHAKE, SHAKE_DOMAIN, mod=k + 1, off=k),
            RATE_SHAKE,
            64 * eta2,
        )

        # 3. CBD：r → Y [256k, B] fp16；e1/e2 → E [256(k+1), B] int16
        #    （`cbd_e` 顺带把 1665·m_bits 加在 E2 上，正好是 Decompress_1(m)）
        if eta1 == 2:
            Y = ops.cbd_y(y_buf, B, k)
        else:  # ML-KEM-512 的 η1=3：通用路径
            from .pipeline import cbd_gpu

            Y = (
                cbd_gpu(y_buf, eta1)
                .reshape(B, k, 256)
                .permute(1, 2, 0)
                .reshape(k * 256, B)
                .to(torch.float16)
                .contiguous()
            )
        if eta2 == 2:
            E = ops.cbd_e(e_buf, m, B, k)
        else:
            from .kernels.glue import unpack_mbits
            from .pipeline import cbd_gpu

            e = cbd_gpu(e_buf, eta2).reshape(B, k + 1, 256)
            E1 = e[:, :k].permute(1, 2, 0).reshape(k * 256, B).to(torch.int16)
            E2 = (e[:, k].T + 1665 * unpack_mbits(m, B)).to(torch.int16)
            E = torch.cat([E1, E2], dim=0)

        # 4. 融合：GEMM + mod q + Compress
        comp = fused_encaps(self.W_enc, Y, E, B, p.du, p.dv)
        if comp is None:  # 形状不满足约束时的通用回退
            acc = torch.mm(self.W_enc.to(torch.float32), Y.to(torch.float32))
            x = (acc.to(torch.int32) + E.to(torch.int32)) % 3329
            x = torch.where(x < 0, x + 3329, x)
            du, dv, pivot = p.du, p.dv, 256 * k
            xu = (((x[:pivot] << du) + 1664) // 3329) & ((1 << du) - 1)
            xv = (((x[pivot:] << dv) + 1664) // 3329) & ((1 << dv) - 1)
            comp = torch.cat([xu, xv], dim=0).to(torch.int16)

        # 5. ByteEncode → c，同时把 H(c) 需要的 pad10*1 一起准备好
        #    （省掉单独一次 pad kernel：密文本身是 1088 B，加域分隔字节 1089 B，
        #     SHA3-256 的 rate 是 136，所以补到 9 块 = 1224 B）。
        ct_size = p.ct_size
        total = ((ct_size + 1 + RATE_SHA3_256 - 1) // RATE_SHA3_256) * RATE_SHA3_256
        padded = torch.zeros((B, total), dtype=torch.uint8, device=dev)
        pack_ciphertext(comp, padded, k, p.du, p.dv, ct_size)
        padded[:, ct_size] = SHA3_DOMAIN  # 域分隔
        padded[:, total - 1] = 0x80  # pad10*1 末字节
        c = padded[:, :ct_size]  # 视图，零拷贝

        # 6. FIPS 203 的 Encaps **没有** KDF 那一步。
        #    Kyber round-3 是 K = KDF(K̄ ‖ H(c))，FIPS 203 (Alg 17) 直接
        #    `return (K, c)`，K 就是 G(m‖H(ek)) 的前 32 字节。
        #    （我按 round-3 写了一版，`c` 全对、`K` 全错，ACVP 才查出来。）
        return c, k_pre


def _to_msgs2d(msgs) -> np.ndarray:
    if isinstance(msgs, np.ndarray):
        a = np.ascontiguousarray(msgs)
    elif isinstance(msgs, (bytes, bytearray, memoryview)):
        buf = bytes(msgs)
        if len(buf) % 32:
            raise ValueError(f"消息总长 {len(buf)} 不是 32 的整数倍")
        a = np.frombuffer(buf, np.uint8).reshape(-1, 32).copy()
    else:
        lst = list(msgs)
        a = np.frombuffer(b"".join(lst), np.uint8).reshape(len(lst), 32).copy()
    if a.ndim == 1:
        a = a[None, :]
    if a.shape[1] != 32:
        raise ValueError(f"每条消息应为 32 字节，收到 {a.shape}")
    return a


def encapsulate(ps: str, ek: bytes, msgs, device=None):
    """一次性批量封装。"""
    return Encapsulator(ps, ek, device=device).encapsulate(msgs)


__all__ = ["Encapsulator", "encapsulate"]
