"""ntt_ref 与 FIPS 203 的自洽性测试，以及方案核心等式 T(a)@b == a*b 的验证。"""

from __future__ import annotations

import numpy as np
import pytest

from mlkem_triton.ntt_ref import (
    GAMMAS,
    ZETAS,
    N,
    Q,
    base_case_multiply,
    bitrev7,
    byte_decode,
    byte_encode,
    compress,
    decompress,
    intt,
    multiply_ntts,
    negacyclic_conv,
    ntt,
    toeplitz,
)


def _rnd(n=N):
    return np.random.randint(0, Q, size=n, dtype=np.int64)


# ---------------- 基础 ----------------


def test_bitrev7():
    for i in range(128):
        assert bitrev7(i) == int(format(i, "07b")[::-1], 2)


def test_zeta_is_primitive_256th_root():
    assert pow(17, 256, Q) == 1
    assert pow(17, 128, Q) == Q - 1  # 17^128 = -1
    assert pow(17, 64, Q) != 1


def test_zetas_shape():
    assert ZETAS.shape == (128,)
    assert GAMMAS.shape == (128,)
    assert int(ZETAS[0]) == 1
    # γ_i 全是奇数指数，且与 ZETAS 不同源
    for i in range(4):
        assert int(GAMMAS[i]) == pow(17, 2 * bitrev7(i) + 1, Q)


# ---------------- NTT / INTT ----------------


@pytest.mark.parametrize("seed", range(5))
def test_ntt_intt_roundtrip(seed):
    np.random.seed(seed)
    f = _rnd()
    assert np.array_equal(intt(ntt(f)), f % Q)


def test_ntt_is_linear():
    a, b = _rnd(), _rnd()
    lhs = ntt((a + b) % Q)
    rhs = (ntt(a) + ntt(b)) % Q
    assert np.array_equal(lhs, rhs)


# ---------------- 乘法 ----------------


@pytest.mark.parametrize("seed", range(3))
def test_multiply_ntts_matches_negacyclic(seed):
    np.random.seed(seed)
    a, b = _rnd(), _rnd()
    got = intt(multiply_ntts(ntt(a), ntt(b)))
    ref = negacyclic_conv(a, b)
    assert np.array_equal(got, ref), f"最大差 {np.abs(got - ref).max()}"


def test_base_case_multiply_consistency():
    # BaseCaseMultiply 与两个二次多项式直接在 Z_q[X]/(X^2-γ) 中相乘一致
    for _ in range(20):
        a0, a1, b0, b1, g = np.random.randint(0, Q, size=5)
        c0, c1 = base_case_multiply(a0, a1, b0, b1, g)
        assert c0 == (a0 * b0 + a1 * b1 * g) % Q
        assert c1 == (a0 * b1 + a1 * b0) % Q


# ---------------- 方案核心等式 ----------------


@pytest.mark.parametrize("seed", range(3))
def test_toeplitz_equals_negacyclic(seed):
    """T(a) @ b == a * b mod (X^256+1)。这是整个 GEMM 方案的正确性基础。"""
    np.random.seed(seed)
    a, b = _rnd(), _rnd()
    lhs = (toeplitz(a) @ b) % Q
    rhs = negacyclic_conv(a, b)
    assert np.array_equal(lhs, rhs), f"最大差 {np.abs(lhs - rhs).max()}"


def test_toeplitz_centered_equals_mod_q():
    """用中心化表示 [-1664,1664] 做整数矩阵乘，取模后与 mod-q 卷积一致。"""
    np.random.seed(0)
    a = np.random.randint(0, Q, size=N).astype(np.int64)
    b = np.random.randint(0, Q, size=N).astype(np.int64)
    a_c = np.where(a > (Q - 1) // 2, a - Q, a)
    b_c = np.where(b > (Q - 1) // 2, b - Q, b)
    lhs = (toeplitz(a_c) @ b_c) % Q
    rhs = negacyclic_conv(a, b)
    assert np.array_equal(lhs, rhs)


# ---------------- 编解码 / 压缩 ----------------


@pytest.mark.parametrize("d", [1, 4, 10, 11, 12])
def test_byte_codec_roundtrip(d):
    np.random.seed(d)
    vals = np.random.randint(0, 1 << d, size=N)
    enc = byte_encode(vals, d)
    assert len(enc) == 32 * d
    dec = byte_decode(enc, d)
    assert np.array_equal(dec, vals)


def _round_half_up(num: int, den: int) -> int:
    assert den > 0
    return (2 * num + den) // (2 * den)


@pytest.mark.parametrize("d", [1, 4, 10, 11, 12])
def test_compress_matches_definition(d):
    for x in range(Q):
        got = int(compress(x, d))
        want = _round_half_up(x * (1 << d), Q) % (1 << d)
        assert got == want, (d, x, got, want)


@pytest.mark.parametrize("d", [1, 4, 10, 11, 12])
def test_decompress_matches_definition(d):
    for y in range(1 << d):
        got = int(decompress(y, d))
        want = _round_half_up(y * Q, 1 << d)
        assert got == want, (d, y, got, want)


def test_decompress_1_is_1665():
    assert int(decompress(1, 1)) == 1665


# ---------------- 直接对拍 kyber-py ----------------


def test_against_kyberpy_ntt():
    """与 kyber-py 的 NTT 实现逐项对拍。

    注意：kyber-py 的 `to_ntt()` 会**原地改写**传入的 list（`_parse_coefficients`
    在 len == 256 时直接返回原对象，不拷贝），所以每次都要传新副本，否则会污染输入。
    """
    pytest.importorskip("kyber_py", reason="这个用例是对拍第三方实现，需要 kyber-py")
    from kyber_py.polynomials.polynomials import PolynomialRing

    R = PolynomialRing()
    np.random.seed(42)
    f0 = [int(x) for x in np.random.randint(0, Q, size=N)]
    g0 = [int(x) for x in np.random.randint(0, Q, size=N)]

    def their_ntt(v):
        return R(list(v)).to_ntt()

    assert np.array_equal(ntt(f0), their_ntt(f0).coeffs), "ntt 不一致"
    assert np.array_equal(intt(ntt(f0)), their_ntt(f0).from_ntt().coeffs), "intt 不一致"
    assert np.array_equal(
        multiply_ntts(ntt(f0), ntt(g0)),
        (their_ntt(f0) * their_ntt(g0)).coeffs,
    ), "multiply_ntts 不一致"
    assert np.array_equal(
        (their_ntt(f0) * their_ntt(g0)).from_ntt().coeffs,
        negacyclic_conv(f0, g0),
    ), "kyber-py 乘积 != 负循环卷积"
