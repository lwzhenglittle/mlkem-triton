"""Exact int8 GEMM: the arithmetic a finite-field GEMM is built on.

"Exact" is the property being tested, not an implementation detail: XMX int8 dot products
accumulate in int32 with no rounding, which is what lets 8 binary planes of a GF(2^8) matrix
be multiplied as 8 integer GEMMs and recombined with XOR. If this module ever rounds, the
bit-plane trick silently produces wrong field products.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.gemm import (
    gemm_int8_np,
    gemm_int8_torch,
    gf2_8_matmul_torch,
)


@pytest.mark.parametrize(
    ("m", "k", "n"),
    [
        (16, 16, 16),
        (32, 64, 32),
        (64, 96, 48),
        (17, 33, 19),
        (1, 1, 1),
        (128, 128, 128),
    ],
)
def test_matches_numpy_exactly(m, k, n):
    rng = np.random.default_rng(m * 1000 + k * 100 + n)
    a = rng.integers(-128, 128, (m, k), dtype=np.int8)
    b = rng.integers(-128, 128, (k, n), dtype=np.int8)
    got = gemm_int8_torch(a, b).cpu().numpy()
    assert np.array_equal(got.astype(np.int64), gemm_int8_np(a, b))


def test_extremes_do_not_round():
    """-128 x -128 is the largest magnitude: 16384 per term, and K = 1024 stays in int32."""
    a = np.full((16, 1024), -128, dtype=np.int8)
    b = np.full((1024, 16), -128, dtype=np.int8)
    got = gemm_int8_torch(a, b).cpu().numpy()
    assert np.array_equal(got, gemm_int8_np(a, b))
    assert int(got[0, 0]) == 1024 * 16384


def test_uint8_range_inputs():
    """Bit planes are 0/1, so the common case in a GF(2^k) GEMM is non-negative."""
    rng = np.random.default_rng(7)
    a = rng.integers(0, 2, (64, 64)).astype(np.int8)
    b = rng.integers(0, 2, (64, 64)).astype(np.int8)
    got = gemm_int8_torch(a, b).cpu().numpy()
    assert np.array_equal(got, gemm_int8_np(a, b))
    # for binary operands the int product counts the *intersection* of the two planes
    assert got[0].max() <= 64


def test_reference_checks_shape_and_range():
    with pytest.raises(ValueError, match="not a matrix product"):
        gemm_int8_np(np.zeros((4, 5), np.int8), np.zeros((6, 7), np.int8))
    with pytest.raises(ValueError, match="int8 operands"):
        gemm_int8_np(np.full((2, 2), 200, np.int64), np.ones((2, 2), np.int64))


def test_out_buffer_is_used():
    import torch

    from triton_crypto.devices import resolve_device

    rng = np.random.default_rng(11)
    a = rng.integers(-8, 8, (32, 32), dtype=np.int8)
    b = rng.integers(-8, 8, (32, 32), dtype=np.int8)
    out = torch.empty((32, 32), dtype=torch.int32, device=resolve_device())
    got = gemm_int8_torch(a, b, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert np.array_equal(got.cpu().numpy().astype(np.int64), gemm_int8_np(a, b))


def _gmul(x: int, y: int) -> int:
    """GF(2^8) product with the AES polynomial x^8 + x^4 + x^3 + x + 1 (0x11B)."""
    r = 0
    for _ in range(8):
        if y & 1:
            r ^= x
        y >>= 1
        x <<= 1
        if x & 0x100:
            x ^= 0x11B
    return r & 0xFF


def _reduce_mask(k: int) -> int:
    """`x^k mod 0x11B` as a bit mask over x^0..x^7, for 0 <= k <= 14."""
    m = 1 << k
    while m & ~0xFF:
        top = m.bit_length() - 1  # m = x^top
        m ^= 1 << top
        m ^= 0x1B << (top - 8)  # x^top = x^(top-8) * (x^4+x^3+x+1)
    return m


def test_bit_planes_compute_a_gf2_8_product():
    """Bit-plane construction: 64 tiny int8 GEMMs == one GF(2^8) product.

    For binary operands an int8 GEMM *is* a GF(2) matrix product, so the product of two 8x8
    GF(2^8) matrices can be assembled from per-bit-plane products XOR-ed through the
    polynomial's reduction masks. If the int8 GEMM rounded, this would be wrong in a way no
    single-plane test could see.
    """
    rng = np.random.default_rng(3)
    mat = rng.integers(0, 256, (8, 8), dtype=np.uint16)
    val = rng.integers(0, 256, (8, 8), dtype=np.uint16)

    ref = (
        np.array(
            [
                [_gmul(int(mat[i, j]), int(val[j, k])) for k in range(8)]
                for i in range(8)
                for j in range(8)
            ]
        )
        .reshape(8, 8, 8)
        .sum(axis=1)
        & 0
    )  # placeholder
    # reference: proper matrix product in GF(2^8)
    ref = np.zeros((8, 8), dtype=np.uint16)
    for i in range(8):
        for k in range(8):
            acc = 0
            for j in range(8):
                acc ^= _gmul(int(mat[i, j]), int(val[j, k]))
            ref[i, k] = acc

    got = np.zeros((8, 8), dtype=np.uint16)
    for bit_a in range(8):
        a_plane = ((mat >> bit_a) & 1).astype(np.int8)
        for bit_b in range(8):
            b_plane = ((val >> bit_b) & 1).astype(np.int8)
            acc = gemm_int8_torch(a_plane, b_plane).cpu().numpy().astype(np.uint16) & 1
            mask = _reduce_mask(bit_a + bit_b)
            for bit in range(8):
                if (mask >> bit) & 1:
                    got ^= acc << bit
    assert np.array_equal(got, ref)


def test_bit_planes_match_the_reference_elementwise():
    """Same thing without the field interpretation: every plane product is exact."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, 2, (16, 16)).astype(np.int8)
    b = rng.integers(0, 2, (16, 16)).astype(np.int8)
    assert np.array_equal(gemm_int8_torch(a, b).cpu().numpy(), gemm_int8_np(a, b))


def test_gf2_8_matmul_agrees_with_the_bit_plane_reference():
    """`gf2_8_matmul_torch` assembles 64 int8 GEMMs; this checks the assembly, not the GEMM.

    The reference here is the same construction done in numpy, which is independent of the device
    path (counts instead of XOR, one `@` per plane pair), and `test_bit_planes_compute_a_gf2_8_product`
    independently pins the construction itself against a scalar `_gmul` at 8x8.
    """
    n = 32
    rng = np.random.default_rng(11)
    a = rng.integers(0, 256, (n, n), dtype=np.uint16)
    b = rng.integers(0, 256, (n, n), dtype=np.uint16)
    want = np.zeros((n, n), dtype=np.uint16)
    for i in range(8):
        for j in range(8):
            prod = ((a >> i) & 1).astype(np.uint16) @ ((b >> j) & 1).astype(np.uint16)
            mask = _reduce_mask(i + j)
            for bit in range(8):
                if mask >> bit & 1:
                    want ^= ((prod & 1) << bit).astype(np.uint16)
    got = gf2_8_matmul_torch(a, b).cpu().numpy().astype(np.uint16)
    assert np.array_equal(got, want)


def test_gf2_8_matmul_rejects_a_bad_inner_dimension():
    with pytest.raises(ValueError, match="not a product"):
        gf2_8_matmul_torch(np.zeros((4, 5), np.uint16), np.zeros((6, 3), np.uint16))
