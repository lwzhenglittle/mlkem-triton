"""`kernels/fused_gemm.py` 的验收测试。

两个内核都必须与「未融合的多步实现」逐位相等：

- `fused_enc_gemm` ↔ 通用 fp32 GEMM + `_cmp_kernel` 两发
- `fused_dec_gemm` ↔ 通用 fp32 GEMM + `glue.compress_pack`
"""

from __future__ import annotations

import pytest
import torch
import triton
from _common import DEV, HAS_GPU

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="需要 GPU")

Q = 3329
K = 768


def _data(M, B, pivot, seed=0):
    torch.manual_seed(seed)
    W = torch.randint(-1664, 1665, (M, K), device=DEV, dtype=torch.int16).to(torch.float16)
    Y = torch.randint(-2, 3, (K, B), device=DEV, dtype=torch.int16).to(torch.float16)
    E = torch.empty((M, B), device=DEV, dtype=torch.int16)
    E[:pivot] = torch.randint(-2, 3, (pivot, B), device=DEV, dtype=torch.int16)
    E[pivot:] = torch.randint(-2, 1668, (M - pivot, B), device=DEV, dtype=torch.int16)
    return W, Y, E


def _ref(W, Y, E, B, pivot, du, dv, ref1, ref2):
    from mlkem_triton.pipeline import _cmp_kernel, _gemm_fp32

    M = W.shape[0]
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    acc2 = _gemm_fp32(W, Y)
    if pivot:
        _cmp_kernel[(pivot // 128, triton.cdiv(B, 64))](
            acc2, E, ref1, cnt, B, 0, pivot, D=du, BM=128, BN=64
        )
    if M - pivot:
        _cmp_kernel[((M - pivot) // 128, triton.cdiv(B, 64))](
            acc2, E, ref2, cnt, B, pivot, M - pivot, D=dv, BM=128, BN=64
        )
    return cnt


@pytest.mark.parametrize("M", [1024, 768, 256, 1280])
@pytest.mark.parametrize("B", [256, 512, 4096, 1000])
def test_bitwise_equal(M, B):
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm

    du, dv = 10, 4
    pivot = M - 256
    W, Y, E = _data(M, B, pivot)
    # ref：一半命中、一半随机，同时覆盖「相等」和「不等」两条路径
    ref1 = (
        torch.randint(0, 1 << du, (pivot, B), device=DEV, dtype=torch.int16)
        if pivot
        else torch.empty((0, B), device=DEV, dtype=torch.int16)
    )
    ref2 = torch.randint(0, 1 << dv, (M - pivot, B), device=DEV, dtype=torch.int16)
    cnt_ref = _ref(W, Y, E, B, pivot, du, dv, ref1, ref2)
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    assert fused_enc_gemm(W, Y, E, (ref1, ref2), cnt, B, du, dv) is True
    assert torch.equal(cnt, cnt_ref)

    # 全命中：计数应恒等于 M
    comp = torch.empty((M, B), device=DEV, dtype=torch.int32)
    acc = torch.mm(W, Y, out_dtype=torch.float32)
    x = acc.to(torch.int32) + E.to(torch.int32)
    x = x % Q
    x = torch.where(x < 0, x + Q, x)
    if pivot:
        comp[:pivot] = (((x[:pivot] << du) + 1664) // Q) & ((1 << du) - 1)
    comp[pivot:] = (((x[pivot:] << dv) + 1664) // Q) & ((1 << dv) - 1)
    del acc, x
    r1, r2 = comp[:pivot].to(torch.int16), comp[pivot:].to(torch.int16)
    cnt_ref2 = _ref(W, Y, E, B, pivot, du, dv, r1, r2)
    cnt2 = torch.zeros(B, device=DEV, dtype=torch.int32)
    assert fused_enc_gemm(W, Y, E, (r1, r2), cnt2, B, du, dv) is True
    assert torch.equal(cnt2, cnt_ref2)
    assert int(cnt2.min()) == M and int(cnt2.max()) == M


def test_accumulates_like_ops_cmp():
    """两次调用（du 段 / dv 段分开）累加到同一个 cnt，与 ops.cmp 的用法一致。"""
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm
    from mlkem_triton.pipeline import _cmp_kernel

    du, dv, M, B, pivot = 10, 4, 1024, 512, 768
    W, Y, E = _data(M, B, pivot)
    ref1 = torch.randint(0, 1 << du, (pivot, B), device=DEV, dtype=torch.int16)
    ref2 = torch.randint(0, 1 << dv, (M - pivot, B), device=DEV, dtype=torch.int16)
    acc2 = torch.mm(W, Y, out_dtype=torch.float32)
    c_ref = torch.zeros(B, device=DEV, dtype=torch.int32)
    _cmp_kernel[(6, triton.cdiv(B, 64))](acc2, E, ref1, c_ref, B, 0, pivot, D=du, BM=128, BN=64)
    _cmp_kernel[(2, triton.cdiv(B, 64))](
        acc2, E, ref2, c_ref, B, pivot, M - pivot, D=dv, BM=128, BN=64
    )

    c = torch.zeros(B, device=DEV, dtype=torch.int32)
    # v 段（pivot=0 表示整段都用 dv）先加，再把 u 段加上去
    assert fused_enc_gemm(W[pivot:], Y, E[pivot:], ref2, c, B, du, dv, pivot=0) is True
    assert fused_enc_gemm(W[:pivot], Y, E[:pivot], ref1, c, B, du, dv, pivot=pivot) is True
    assert torch.equal(c, c_ref)


def test_single_ref_tensor():
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm

    du, dv, M, B, pivot = 10, 4, 1024, 512, 768
    W, Y, E = _data(M, B, pivot)
    ref = torch.cat(
        [
            torch.randint(0, 1 << du, (pivot, B), device=DEV, dtype=torch.int16),
            torch.randint(0, 1 << dv, (M - pivot, B), device=DEV, dtype=torch.int16),
        ]
    )
    cnt_ref = _ref(W, Y, E, B, pivot, du, dv, ref[:pivot], ref[pivot:])
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    assert fused_enc_gemm(W, Y, E, ref, cnt, B, du, dv) is True
    assert torch.equal(cnt, cnt_ref)


def test_fallback_returns_false():
    """形状/dtype 不满足时必须返回 False，且不动 cnt。"""
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm

    du, dv, M, B, pivot = 10, 4, 1024, 512, 768
    W, Y, E = _data(M, B, pivot)
    ref1 = torch.zeros((pivot, B), device=DEV, dtype=torch.int16)
    ref2 = torch.zeros((M - pivot, B), device=DEV, dtype=torch.int16)
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    zero = lambda: torch.zeros(B, device=DEV, dtype=torch.int32)  # noqa: E731

    # K 不是 BK=32 的倍数
    assert (
        fused_enc_gemm(
            torch.zeros(M, 1000, device=DEV, dtype=torch.float16),
            torch.zeros(1000, B, device=DEV, dtype=torch.float16),
            E,
            (ref1, ref2),
            zero(),
            B,
            du,
            dv,
        )
        is False
    )
    # M 不是 BM 的倍数
    assert (
        fused_enc_gemm(
            torch.zeros(1000, K, device=DEV, dtype=torch.float16),
            Y,
            torch.zeros(1000, B, device=DEV, dtype=torch.int16),
            (torch.zeros(744, B, device=DEV, dtype=torch.int16), ref2),
            zero(),
            B,
            du,
            dv,
        )
        is False
    )
    # dtype 错
    assert fused_enc_gemm(W.float(), Y, E, (ref1, ref2), zero(), B, du, dv) is False
    # cnt 不是 int32
    assert (
        fused_enc_gemm(
            W, Y, E, (ref1, ref2), torch.zeros(B, device=DEV, dtype=torch.int64), B, du, dv
        )
        is False
    )
    # ref 形状错
    assert fused_enc_gemm(W, Y, E, (ref1, ref2[:, :16].contiguous()), zero(), B, du, dv) is False
    # 位宽非法
    assert fused_enc_gemm(W, Y, E, (ref1, ref2), zero(), B, 0, dv) is False
    # cnt 保持不动
    assert int(cnt.abs().sum()) == 0


# ---------------------------------------------------------------- DEC 融合
def _dec_data(M, B, seed=0):
    torch.manual_seed(seed)
    W = torch.randint(-1664, 1665, (M, K), device=DEV, dtype=torch.int16).to(torch.float16)
    U = torch.randint(-2, 3, (K, B), device=DEV, dtype=torch.int16).to(torch.float16)
    V = torch.randint(-2, 3, (M, B), device=DEV, dtype=torch.int16)
    return W, U, V


def _dec_ref(W, U, V, B):
    from mlkem_triton.kernels.glue import compress_pack
    from mlkem_triton.pipeline import _gemm_fp32

    return compress_pack(V, _gemm_fp32(W, U), B)


@pytest.mark.parametrize("B", [1, 37, 256, 4096, 1000])
def test_dec_bitwise_equal(B):
    """真实用例（M=256）：必须与「fp32 GEMM + glue.compress_pack」逐字节相等。"""
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm

    W, U, V = _dec_data(256, B)
    ref = _dec_ref(W, U, V, B)
    got = fused_dec_gemm(W, U, V, B)
    assert got is not None
    assert got.shape == (B, 32)
    assert got.dtype == torch.uint8
    assert torch.equal(got, ref)


@pytest.mark.parametrize("M", [256, 512, 1024, 2048])
@pytest.mark.parametrize("B", [64, 1000])
def test_dec_general_M(M, B):
    """M 不是 256 时也要对（`glue.compress_pack` 只支持 256，这里用定义式核对）。"""
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm
    from mlkem_triton.pipeline import _gemm_fp32

    W, U, V = _dec_data(M, B, seed=M)
    got = fused_dec_gemm(W, U, V, B)
    assert got is not None and got.shape == (B, M // 8)

    w = (V.to(torch.int32) - _gemm_fp32(W, U).to(torch.int32)) % Q
    comp = (((w << 1) + Q // 2) // Q) & 1
    bits = comp.T.reshape(B, M // 8, 8)
    want = (bits * (1 << torch.arange(8, device=DEV, dtype=torch.int32))).sum(-1)
    assert torch.equal(got.to(torch.int32), want)


@pytest.mark.parametrize("B", [1, 1000])
def test_dec_matches_operand_dtype(B):
    """U 是 fp16、V 是 int16，混着用也要对（回归：别把 V 当 fp16 读）。"""
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm

    W, U, V = _dec_data(256, B, seed=7)
    V = (V * 800).to(torch.int16)  # 让 (V - acc) 落在负区间
    assert torch.equal(fused_dec_gemm(W, U, V, B), _dec_ref(W, U, V, B))


def test_dec_falls_back_on_bad_shapes():
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm

    W, U, V = _dec_data(256, 64)
    assert fused_dec_gemm(W.double(), U, V, 64) is None
    assert fused_dec_gemm(W, U.double(), V, 64) is None
    assert fused_dec_gemm(W, U, V.float(), 64) is None
    assert fused_dec_gemm(W, U.T.contiguous(), V, 64) is None
    assert fused_dec_gemm(W, U, V[:, :32].contiguous(), 64) is None
    assert fused_dec_gemm(W[:100], U, V[:100], 64) is None  # M 不是 BM 的倍数


def test_dec_gemm_mod_q_matches_definition():
    """直接验语义：m_bytes 的第 b 行等于 compress_1((v - W@u) mod q) 的位打包。"""
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm
    from mlkem_triton.pipeline import _gemm_fp32

    M, B = 256, 128
    W, U, V = _dec_data(M, B, seed=3)
    got = fused_dec_gemm(W, U, V, B)

    acc = _gemm_fp32(W, U)
    w = (V.to(torch.int32) - acc.to(torch.int32)) % Q
    comp = (((w << 1) + Q // 2) // Q) & 1  # [M, B] ∈ {0,1}
    bits = comp.T.reshape(B, 32, 8)  # 第 j 字节 = 8j..8j+7 行
    want = (bits * (1 << torch.arange(8, device=DEV, dtype=torch.int32))).sum(-1)
    assert torch.equal(got.to(torch.int32), want)
