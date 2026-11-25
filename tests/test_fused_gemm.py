"""Acceptance tests for `kernels/fused_gemm.py`.

Both kernels must be bit-identical to the "unfused multi-step implementation":

- `fused_enc_gemm` ↔ generic fp32 GEMM + two `_cmp_kernel` launches
- `fused_dec_gemm` ↔ generic fp32 GEMM + `glue.compress_pack`
"""

from __future__ import annotations

import pytest
import torch
import triton
from _common import DEV, HAS_GPU

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")

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
    # ref: half hits, half random, covering both the "equal" and "not equal" paths
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

    # all hits: the count should always equal M
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
    """Two calls (the du segment and the dv segment separately) accumulate into the
    same cnt, as ops.cmp does.
    """
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
    # add the v segment first (pivot=0 means the whole segment uses dv), then add the u segment
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
    """Must return False when shape/dtype do not fit, and leave cnt untouched."""
    from mlkem_triton.kernels.fused_gemm import fused_enc_gemm

    du, dv, M, B, pivot = 10, 4, 1024, 512, 768
    W, Y, E = _data(M, B, pivot)
    ref1 = torch.zeros((pivot, B), device=DEV, dtype=torch.int16)
    ref2 = torch.zeros((M - pivot, B), device=DEV, dtype=torch.int16)
    cnt = torch.zeros(B, device=DEV, dtype=torch.int32)
    zero = lambda: torch.zeros(B, device=DEV, dtype=torch.int32)  # noqa: E731

    # K is not a multiple of BK=32
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
    # M is not a multiple of BM
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
    # wrong dtype
    assert fused_enc_gemm(W.float(), Y, E, (ref1, ref2), zero(), B, du, dv) is False
    # cnt is not int32
    assert (
        fused_enc_gemm(
            W, Y, E, (ref1, ref2), torch.zeros(B, device=DEV, dtype=torch.int64), B, du, dv
        )
        is False
    )
    # wrong ref shape
    assert fused_enc_gemm(W, Y, E, (ref1, ref2[:, :16].contiguous()), zero(), B, du, dv) is False
    # illegal bit width
    assert fused_enc_gemm(W, Y, E, (ref1, ref2), zero(), B, 0, dv) is False
    # cnt stays untouched
    assert int(cnt.abs().sum()) == 0


# ---------------------------------------------------------------- DEC fusion
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
    """Real case (M=256): must be byte-identical to "fp32 GEMM + glue.compress_pack"."""
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
    """Must also be correct when M is not 256 (`glue.compress_pack` only supports 256,
    so this checks against the definition).
    """
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
    """U is fp16, V is int16; mixing them must work (regression: do not read V as fp16)."""
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm

    W, U, V = _dec_data(256, B, seed=7)
    V = (V * 800).to(torch.int16)  # make (V - acc) land in the negative range
    assert torch.equal(fused_dec_gemm(W, U, V, B), _dec_ref(W, U, V, B))


def test_dec_falls_back_on_bad_shapes():
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm

    W, U, V = _dec_data(256, 64)
    assert fused_dec_gemm(W.double(), U, V, 64) is None
    assert fused_dec_gemm(W, U.double(), V, 64) is None
    assert fused_dec_gemm(W, U, V.float(), 64) is None
    assert fused_dec_gemm(W, U.T.contiguous(), V, 64) is None
    assert fused_dec_gemm(W, U, V[:, :32].contiguous(), 64) is None
    assert fused_dec_gemm(W[:100], U, V[:100], 64) is None  # M is not a multiple of BM


def test_dec_gemm_mod_q_matches_definition():
    """Direct semantic check: row b of m_bytes is the bit packing of
    compress_1((v - W@u) mod q).
    """
    from mlkem_triton.kernels.fused_gemm import fused_dec_gemm
    from mlkem_triton.pipeline import _gemm_fp32

    M, B = 256, 128
    W, U, V = _dec_data(M, B, seed=3)
    got = fused_dec_gemm(W, U, V, B)

    acc = _gemm_fp32(W, U)
    w = (V.to(torch.int32) - acc.to(torch.int32)) % Q
    comp = (((w << 1) + Q // 2) // Q) & 1  # [M, B] ∈ {0,1}
    bits = comp.T.reshape(B, 32, 8)  # byte j = rows 8j..8j+7
    want = (bits * (1 << torch.arange(8, device=DEV, dtype=torch.int32))).sum(-1)
    assert torch.equal(got.to(torch.int32), want)
