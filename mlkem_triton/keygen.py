"""ML-KEM 密钥生成（FIPS 203 Alg. 16），全设备常驻。

这是三个操作里**唯一必须用 NTT 的一个**：

    (ρ, σ) ← G(d ‖ k)                    ← 那个 k 是参数集域分隔，不能漏
    Â[i][j] ← SampleNTT(ρ‖j‖i)          ← k² 次 SHAKE-128（840 B）
    s ← CBD_η1(PRF(σ, 0..k-1))
    e ← CBD_η1(PRF(σ, k..2k-1))
    ŝ ← NTT(s),  ê ← NTT(e)              ← 这里是 Toeplitz 技巧帮不上忙的地方
    t̂ ← Â ∘ ŝ + ê
    ek ← ByteEncode12(t̂) ‖ ρ
    dk ← ByteEncode12(ŝ) ‖ ek ‖ H(ek) ‖ z

decaps / encaps 不需要 NTT（乘法折成了固定矩阵的 GEMM）；keygen 不行：
`ek` 里必须放 `t̂` 的 NTT 域表示，而 `s`/`e` 每个密钥都是新的，没有可复用的矩阵。
所以这里就是一个普通的 NTT 实现 —— 见 `kernels/ntt.py` 的说明。
"""

from __future__ import annotations

import numpy as np
import torch
import triton

from . import ops
from .devices import resolve_device
from .kernels.ntt import byte_encode12, forward_ntt, pow_mac
from .kernels.rebuild import RATE_SHAKE128, _sample_ntt_kernel
from .kernels.shake import SHA3_DOMAIN, SHAKE_DOMAIN
from .params import PARAMS, Q
from .pipeline import RATE_SHA3_512, RATE_SHAKE, _pad_prf_gpu, sponge_gpu

N = 256
RATE_SHA3_256 = 136


class KeyGenerator:
    """批量 ML-KEM 密钥生成。"""

    def __init__(self, ps: str, device=None):
        self.ps = ps
        self.p = PARAMS[ps]
        self.k = self.p.k
        self.device = resolve_device(device)

    # ---- 设备侧 ----
    def generate_torch(self, d: torch.Tensor, z: torch.Tensor):
        """`d`/`z` 都是 ``[B, 32]`` uint8。返回 ``(ek [B,384k+32], dk [B,768k+96])``。"""
        p, k, dev = self.p, self.k, self.device
        if d.shape != z.shape or d.shape[1] != 32:
            raise ValueError("d/z 需要是 [B,32] uint8")
        B = d.shape[0]

        # 1. (ρ, σ) = G(d ‖ k)   —— SHA3-512, rate 72
        #    **注意那个 `k`**：FIPS 203 Alg. 13 用 `d ‖ k` 做不同参数集之间的域分隔。
        #    我第一版照 round-3 的 `G(d)` 写，ρ 全错 —— 拿 kyber-py 逐字节对拍才查出来。
        gpad = torch.zeros((B, RATE_SHA3_512), dtype=torch.uint8, device=dev)
        gpad[:, :32] = d
        gpad[:, 32] = k
        gpad[:, 33] = SHA3_DOMAIN
        gpad[:, RATE_SHA3_512 - 1] = 0x80
        g = sponge_gpu(gpad, RATE_SHA3_512, 64)
        rho, sigma = g[:, :32], g[:, 32:]

        # 2. Â = SampleNTT(ρ‖j‖i) × k²
        n = B * k * k
        msgs = torch.zeros((n, RATE_SHAKE128), dtype=torch.uint8, device=dev)
        msgs[:, :32] = rho.repeat_interleave(k * k, dim=0)
        ji = torch.arange(k * k, device=dev)
        msgs[:, 32] = (ji % k).repeat(B)
        msgs[:, 33] = (ji // k).repeat(B)
        msgs[:, 34] = SHAKE_DOMAIN
        msgs[:, RATE_SHAKE128 - 1] = 0x80
        outw = 5 * RATE_SHAKE128 // 8  # 840/8 = 105
        buf64 = torch.empty((n, outw), dtype=torch.int64, device=dev)
        from .kernels.shake import get_sponge

        get_sponge(RATE_SHAKE128, 5 * RATE_SHAKE128)[(triton.cdiv(n, 128),)](
            msgs.view(torch.int64),
            buf64,
            n,
            1,
            RATE_SHAKE128 // 8,
            OUTW=outw,
            BLOCK=128,
            num_warps=4,
        )
        a_hat = torch.empty((n, N), dtype=torch.int32, device=dev)
        _sample_ntt_kernel[(n,)](
            buf64.view(torch.uint8).view(n, 5 * RATE_SHAKE128),
            a_hat,
            5 * RATE_SHAKE128,
            N,
            Q,
            1024,
            num_warps=4,
        )

        # 3. s / e = CBD_η1(PRF(σ, 0..k-1)) 和 PRF(σ, k..2k-1)
        eta = p.eta1
        yb = sponge_gpu(
            _pad_prf_gpu(sigma, B * k, RATE_SHAKE, SHAKE_DOMAIN, mod=k, off=0), RATE_SHAKE, 64 * eta
        )
        eb = sponge_gpu(
            _pad_prf_gpu(sigma, B * k, RATE_SHAKE, SHAKE_DOMAIN, mod=k, off=k), RATE_SHAKE, 64 * eta
        )
        if eta == 2:
            sbuf = ops.cbd(yb, eta)  # [B*k, 256] int16
            ebuf = ops.cbd(eb, eta)
        else:  # ML-KEM-512 的 η1=3
            from .pipeline import cbd_gpu

            sbuf = cbd_gpu(yb, eta)
            ebuf = cbd_gpu(eb, eta)
        coef = torch.cat([sbuf, ebuf], dim=0).to(torch.int32)  # [B*2k, 256]

        # 4. ŝ = NTT(s), ê = NTT(e)
        coef_ntt = forward_ntt(coef)

        # 5. t̂ = Â ∘ ŝ + ê
        s_hat = coef_ntt[: B * k]
        e_hat = coef_ntt[B * k :]
        t_hat = pow_mac(a_hat, s_hat, e_hat, k)

        # 6. ek = ByteEncode12(t̂) ‖ ρ
        enc_t = byte_encode12(t_hat).reshape(B, 384 * k)
        ek = torch.cat([enc_t, rho], dim=1)

        # 7. H(ek) —— SHA3-256；把 pad10*1 一起准备好，省一次 kernel
        ct_len = p.ek_size
        total = ((ct_len + 1 + RATE_SHA3_256 - 1) // RATE_SHA3_256) * RATE_SHA3_256
        padded = torch.zeros((B, total), dtype=torch.uint8, device=dev)
        padded[:, :ct_len] = ek
        padded[:, ct_len] = SHA3_DOMAIN
        padded[:, total - 1] = 0x80
        h_ek = sponge_gpu(padded, RATE_SHA3_256, 32)

        # 8. dk = ByteEncode12(ŝ) ‖ ek ‖ H(ek) ‖ z
        enc_s = byte_encode12(s_hat).reshape(B, 384 * k)
        dk = torch.cat([enc_s, ek, h_ek, z], dim=1)
        return ek, dk

    # ---- 主机侧入口 ----
    def generate(self, count: int, seeds=None):
        """随机（或给定 `d`）批量生成。返回 ``(eks, dks)`` 两个 ``np.ndarray``。"""
        if seeds is None:
            d = np.random.default_rng().integers(0, 256, (count, 32), dtype=np.uint8)
        else:
            d = np.asarray(seeds, dtype=np.uint8).reshape(count, 32)
        z = np.random.default_rng().integers(0, 256, (count, 32), dtype=np.uint8)
        dev = self.device
        ek, dk = self.generate_torch(
            torch.from_numpy(np.ascontiguousarray(d)).to(dev),
            torch.from_numpy(np.ascontiguousarray(z)).to(dev),
        )
        return ek.cpu().numpy(), dk.cpu().numpy()


def generate_keypairs(ps: str, count: int = 1, device=None):
    """便捷入口。注意不要叫 `keygen` —— 那会和本模块同名，导入时无限递归。"""
    return KeyGenerator(ps, device=device).generate(count)


__all__ = ["KeyGenerator", "generate_keypairs"]
