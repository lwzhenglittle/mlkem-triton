"""封装（FIPS 203 Alg. 17）验收：必须逐字节通过 NIST ACVP 向量。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from mlkem_triton import acvp, keyprep
from mlkem_triton.encaps import Encapsulator

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="需要 GPU")

PS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


def _cases(ps):
    return [c for c in acvp.load() if c["function"] == "encapsulation" and c["parameterSet"] == ps]


def _h(v):
    return bytes.fromhex(v)


@pytest.mark.parametrize("ps", PS)
def test_parse_ek_builds_enc_matrix(ps):
    """`parse_ek` 的 W_enc 必须和 `parse_key` 里那段公式给出同一个矩阵。

    做法：从封装向量的 ek 造一个假 dk（把 ek 塞进去），走 `parse_key` 对比。
    """
    cs = _cases(ps)
    ek = _h(cs[0]["inputs"]["ek"])
    ep = keyprep.parse_ek(ps, ek)
    k = ep.k
    # 用 ek 自己的 Â/t 复算一遍，确认 W_enc 的分块布局和 build_W_enc 一致
    t_hat = np.stack([keyprep.byte_decode(ek[384 * j : 384 * (j + 1)], 12) for j in range(k)])
    A_coef = np.stack([[keyprep.intt(ep.A_hat[i, j]) for j in range(k)] for i in range(k)])
    t_coef = np.stack([keyprep.intt(t_hat[j]) for j in range(k)])
    assert np.array_equal(ep.W_enc, keyprep.build_W_enc(A_coef, t_coef, k))
    assert ep.h_ek is not None and len(ep.h_ek) == 32


@pytest.mark.parametrize("ps", PS)
def test_acvp_all_vectors(ps):
    """25 条官方向量，密文与共享密钥都要逐字节相等。"""
    cs = _cases(ps)
    assert len(cs) == 25
    for c in cs:
        enc = Encapsulator(ps, _h(c["inputs"]["ek"]), device=DEV)
        got_c, got_k = enc.encapsulate([_h(c["inputs"]["m"])])
        assert got_c[0].tobytes() == _h(c["expected"]["c"]), f"tcId {c['tcId']} 密文"
        assert got_k[0].tobytes() == _h(c["expected"]["k"]), f"tcId {c['tcId']} K"


@pytest.mark.parametrize("ps", PS)
def test_batch_same_message(ps):
    """同一条消息重复 B 次，输出必须全等于 ACVP 期望值。"""
    cs = _cases(ps)
    enc = Encapsulator(ps, _h(cs[0]["inputs"]["ek"]), device=DEV)
    m = _h(cs[0]["inputs"]["m"])
    got_c, got_k = enc.encapsulate([m] * 37)
    assert got_c.shape == (37, enc.p.ct_size)
    for i in range(37):
        assert got_c[i].tobytes() == _h(cs[0]["expected"]["c"])
        assert got_k[i].tobytes() == _h(cs[0]["expected"]["k"])


@pytest.mark.parametrize("ps", PS)
def test_roundtrip_with_our_own_decaps(ps):
    """我们的 encaps → 我们的 decaps 必须回到同一个 K。

    还要用 kyber-py 独立 decaps 一遍 —— 两个方向都对，才说明封装是真对的
    （只验 ACVP 的 c 相等已经很强，但那是同一批向量；这里换随机密钥）。
    """
    pytest.importorskip("kyber_py", reason="需要 kyber-py 生成随机密钥对")
    from mlkem_triton import refimpl
    from mlkem_triton.pipeline import Decapsulator

    ek, dk = refimpl.ref_keygen(ps)
    rng = np.random.default_rng(7)
    msgs = [rng.integers(0, 256, 32, dtype=np.uint8).tobytes() for _ in range(16)]

    got_c, got_k = Encapsulator(ps, ek, device=DEV).encapsulate(msgs)
    dec = Decapsulator(ps, dk, device=DEV)
    mine = dec.decaps(got_c)
    for i, _m in enumerate(msgs):
        assert mine[i].tobytes() == got_k[i].tobytes(), f"第 {i} 条自环不一致"
        assert refimpl.ref_decaps(ps, dk, got_c[i].tobytes()) == got_k[i].tobytes(), (
            f"第 {i} 条 kyber-py 不认"
        )


def test_pack_handles_odd_bit_widths():
    """du/dv = 11/5 时一个字节横跨 3 个系数，打位必须对（回归测试）。"""
    from mlkem_triton.kernels.encaps import pack_ciphertext

    for du, dv, k in ((10, 4, 3), (11, 5, 4)):
        ct = 32 * (du * k + dv)
        B = 2
        comp = torch.randint(0, 1 << du, (256 * (k + 1), B), dtype=torch.int16, device=DEV)
        comp[k * 256 :] = torch.randint(0, 1 << dv, (256, B), dtype=torch.int16, device=DEV)
        padded = torch.zeros((B, ((ct + 136) // 136) * 136), dtype=torch.uint8, device=DEV)
        pack_ciphertext(comp, padded, k, du, dv, ct)
        got = padded[:, :ct].cpu().numpy()
        cc = comp.cpu().numpy()
        for b in range(B):
            bits = []
            for j in range(k):
                for c in range(256):
                    bits += [(int(cc[j * 256 + c, b]) >> t) & 1 for t in range(du)]
            for c in range(256):
                bits += [(int(cc[k * 256 + c, b]) >> t) & 1 for t in range(dv)]
            want = np.array(
                [sum(bits[o * 8 + t] << t for t in range(8)) for o in range(ct)], np.uint8
            )
            assert np.array_equal(got[b], want), (du, dv, b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
