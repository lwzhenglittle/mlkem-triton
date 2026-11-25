"""`kernels/glue.py` 的验收测试：两块 glue 的融合内核必须与旧实现逐字节相等。

- (A) `compress_pack`：mod q + compress(1 bit) + 位打包 → m_bytes [B,32] uint8
- (B) `build_ye`：y/e/m_bytes → Y [256k,B] fp16 + E [256(k+1),B] int16
另外覆盖：回退判定、回退路径（从 m_bytes 还原 m_bits）的等价性、
以及 `build_ye` 产出的连续 Y 确实能被 `fused_enc_gemm` 接受。
"""

from __future__ import annotations

import pytest
import torch
from _common import DEV

from mlkem_triton.kernels import glue

pytest.importorskip("kyber_py", reason="refimpl 需要可选的 kyber-py")

pytest.importorskip("triton")
XPU = True  # 见 _common.DEV：能 import 就说明有可用 GPU
pytestmark = pytest.mark.skipif(not XPU, reason="需要 GPU")

Q = 3329


def _acc_data(rows, B, seed=0):
    """V ∈ [0,Q)，acc 取到实际范围（|W_dec|≤2、u<q、K=768 → |acc| < 5.2e6）。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    V = torch.randint(0, Q, (rows, B), generator=g, dtype=torch.int32).to(torch.int16).to(DEV)
    acc = (torch.rand((rows, B), generator=g) * 12.0e6 - 6.0e6).to(DEV)
    return V, acc


def _ref_pack(V, acc, B):
    from mlkem_triton.pipeline import compress_gpu, pack_bits_gpu

    w = (V.to(torch.int32) - acc.to(torch.int32)) % Q
    m_bits = compress_gpu(w, 1)
    return pack_bits_gpu(m_bits, 1), m_bits


def _ye_data(k, B, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    y = torch.randint(-3, 4, (B, k, 256), generator=g, dtype=torch.int16).to(DEV)
    e = torch.randint(-2, 3, (B, k + 1, 256), generator=g, dtype=torch.int16).to(DEV)
    mb = torch.randint(0, 256, (B, 32), generator=g, dtype=torch.uint8).to(DEV)
    return y, e, mb


def _ref_ye(y, e, mb, B, k):
    m_bits = glue.unpack_mbits(mb, B)
    Y = y.permute(1, 2, 0).reshape(k * 256, B).to(torch.float16)
    E1 = e[:, :k].permute(1, 2, 0).reshape(k * 256, B).to(torch.int16)
    E2 = (e[:, k].T + 1665 * m_bits).to(torch.int16)
    return Y, torch.cat([E1, E2], dim=0)


@pytest.mark.parametrize("B", [1, 37, 512, 1024, 2048, 4096])
def test_compress_pack_bitwise(B):
    V, acc = _acc_data(256, B, seed=B)
    mb_ref, _ = _ref_pack(V, acc, B)
    assert glue.can_compress_pack(V, acc, B)
    mb = glue.compress_pack(V, acc, B)
    assert mb.dtype == torch.uint8 and tuple(mb.shape) == (B, 32)
    assert torch.equal(mb, mb_ref)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("B", [1, 37, 512, 1024])
def test_build_ye_bitwise(k, B):
    y, e, mb = _ye_data(k, B, seed=k * 100 + B)
    Yr, Er = _ref_ye(y, e, mb, B, k)
    assert glue.can_build_ye(y, e, mb, B, k)
    Y, E = glue.build_ye(y, e, mb, B, k)
    assert Y.dtype == torch.float16 and tuple(Y.shape) == (k * 256, B)
    assert E.dtype == torch.int16 and tuple(E.shape) == ((k + 1) * 256, B)
    assert Y.is_contiguous() and E.is_contiguous()
    assert torch.equal(Y, Yr)
    assert torch.equal(E, Er)


def test_unpack_mbits_roundtrip():
    V, acc = _acc_data(256, 512, seed=5)
    mb_ref, m_bits = _ref_pack(V, acc, 512)
    assert torch.equal(glue.unpack_mbits(mb_ref, 512), m_bits)


def test_fallback_checks():
    V, acc = _acc_data(256, 128, seed=1)
    # 行数不对
    assert not glue.can_compress_pack(V[:128], acc[:128], 128)
    # dtype 不对
    assert not glue.can_compress_pack(V, acc.half(), 128)
    # 非连续
    assert not glue.can_compress_pack(V, acc.T.contiguous().T, 128)  # 全零 stride 的转置视图
    # B 不对
    assert not glue.can_compress_pack(V, acc, 64)
    assert not glue.can_compress_pack(V, acc, 0)

    y, e, mb = _ye_data(3, 128, seed=2)
    assert not glue.can_build_ye(y, e, mb.half(), 128, 3)
    assert not glue.can_build_ye(y, e, mb, 128, 2)
    assert not glue.can_build_ye(y[:, :, :64].contiguous(), e, mb, 128, 3)
    assert not glue.can_build_ye(y, e, mb, 0, 3)
    # 非连续 y
    assert not glue.can_build_ye(y.transpose(0, 1).contiguous().transpose(0, 1), e, mb, 128, 3)


def test_fused_enc_gemm_accepts_built_ye():
    """回归测试：build_ye 的 Y 必须是连续的，否则 fused_enc_gemm 静默回退。"""
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm

    k, B = 3, 512
    y, e, mb = _ye_data(k, B, seed=9)
    Y, E = glue.build_ye(y, e, mb, B, k)
    W = torch.randint(-1664, 1665, (1024, 768), device=DEV, dtype=torch.int16).to(torch.float16)
    ref1 = torch.zeros((768, B), device=DEV, dtype=torch.int16)
    ref2 = torch.zeros((256, B), device=DEV, dtype=torch.int16)
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    assert fused_enc_gemm(W, Y, E, (ref1, ref2), cnt, B, 10, 4) is True


def test_decaps_matches_legacy_formula():
    """整条 decaps 与「旧写法」在随机输入上逐字节一致（B=37，两种参数集）。"""
    from mlkem_triton import refimpl
    from mlkem_triton.pipeline import Decapsulator

    for ps in ("ML-KEM-768", "ML-KEM-1024"):
        ek, dk = refimpl.ref_keygen(ps)
        dec = Decapsulator(ps, dk)
        cts, exp = [], []
        for _ in range(37):
            K, c = refimpl.ref_encaps(ps, ek)
            cts.append(c)
            exp.append(K)
        got = dec.decaps(cts)
        assert [got[i].tobytes() for i in range(37)] == exp


# ---------------------------------------------------------------- CBD → Y/E
def _buf_data(k, B, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    y_buf = torch.randint(0, 256, (B * k, 128), generator=g, dtype=torch.uint8).to(DEV)
    e_buf = torch.randint(0, 256, (B * (k + 1), 128), generator=g, dtype=torch.uint8).to(DEV)
    mb = torch.randint(0, 256, (B, 32), generator=g, dtype=torch.uint8).to(DEV)
    return y_buf, e_buf, mb


def _ref_cbd_ye(y_buf, e_buf, mb, B, k):
    from mlkem_triton.pipeline import cbd_gpu

    y = cbd_gpu(y_buf, 2).reshape(B, k, 256)
    e = cbd_gpu(e_buf, 2).reshape(B, k + 1, 256)
    return glue.build_ye(y, e, mb, B, k)


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("B", [1, 37, 512, 1024, 4096])
def test_cbd_ye_bitwise(k, B):
    y_buf, e_buf, mb = _buf_data(k, B, seed=k * 7 + B)
    Yr, Er = _ref_cbd_ye(y_buf, e_buf, mb, B, k)
    assert glue.can_cbd_y(y_buf, B, k, 2) and glue.can_cbd_e(e_buf, mb, B, k, 2)
    Y, E = glue.cbd_y(y_buf, B, k), glue.cbd_e(e_buf, mb, B, k)
    assert Y.dtype == torch.float16 and Y.is_contiguous()
    assert E.dtype == torch.int16 and E.is_contiguous()
    assert torch.equal(Y, Yr)
    assert torch.equal(E, Er)


def test_cbd_ye_fallback_checks():
    y_buf, e_buf, mb = _buf_data(3, 64, seed=3)
    assert not glue.can_cbd_y(y_buf, 64, 3, 3)  # eta=3 没有融合内核
    assert not glue.can_cbd_y(y_buf, 64, 2, 2)  # 行数不对
    assert not glue.can_cbd_e(e_buf, mb.half(), 64, 3, 2)
    assert not glue.can_cbd_e(e_buf, mb, 64, 3, 3)
    assert not glue.can_cbd_e(e_buf, torch.zeros((64, 64), dtype=torch.uint8, device=DEV), 64, 3, 2)
    # 非连续 bufs
    nc = torch.zeros((y_buf.shape[0], 256), dtype=torch.uint8, device=DEV)[:, :128]
    assert not glue.can_cbd_y(nc, 64, 3, 2)


def test_all_paramsets_decaps_bytewise():
    """三个参数集（含 ML-KEM-512 的 eta1=3 回退路径）端到端逐字节。"""
    from mlkem_triton import refimpl
    from mlkem_triton.pipeline import Decapsulator

    for ps in ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"):
        ek, dk = refimpl.ref_keygen(ps)
        dec = Decapsulator(ps, dk)
        cts = []
        for _ in range(17):
            _, c = refimpl.ref_encaps(ps, ek)
            cts.append(c)
        got = dec.decaps(cts)
        exp = [refimpl.ref_decaps(ps, dk, c) for c in cts]
        assert [got[i].tobytes() for i in range(17)] == exp


def test_decaps_fallback_paths(monkeypatch):
    """把融合判定强制关掉，验证两条回退链路仍逐字节正确。"""
    from mlkem_triton import refimpl
    from mlkem_triton.kernels import glue
    from mlkem_triton.pipeline import Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    cts, exp = [], []
    for _ in range(9):
        K, c = refimpl.ref_encaps(ps, ek)
        cts.append(c)
        exp.append(K)
    ref = [refimpl.ref_decaps(ps, dk, c) for c in cts]

    # 1) 只关 CBD 融合 → 走 cbd_gpu + build_ye
    monkeypatch.setattr(glue, "can_cbd_y", lambda *a, **k: False)
    got = dec.decaps(cts)
    assert [got[i].tobytes() for i in range(9)] == ref

    # 2) 再关 compress_pack 和 build_ye → 全回退（含 unpack_mbits 还原 E2）
    monkeypatch.setattr(glue, "can_compress_pack", lambda *a, **k: False)
    monkeypatch.setattr(glue, "can_build_ye", lambda *a, **k: False)
    got = dec.decaps(cts)
    assert [got[i].tobytes() for i in range(9)] == ref
    assert [got[i].tobytes() for i in range(9)] == exp
