"""Prime-field arithmetic: the field object's contracts and the numpy Montgomery product.

Everything here is pure numpy, so it runs on a CPU-only machine. The device side of the same
module (calling `mont_mul` from another module's kernel) is `test_fields_gpu.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.fields import BABYBEAR, MONT_R, PrimeField, mont_mul_np

FIELDS = [
    PrimeField("ML-KEM", 3329),
    PrimeField("ML-DSA", 8380417),
    BABYBEAR,
]


@pytest.mark.parametrize("field", FIELDS)
def test_montgomery_constants(field):
    assert (field.qinv * field.q) % MONT_R == 1, "qinv must be q^-1 mod 2**32"
    assert field.mont_r == MONT_R % field.q
    assert field.bits == field.q.bit_length()


@pytest.mark.parametrize("field", FIELDS)
def test_montgomery_product_is_the_field_product(field):
    """`mont_mul(to_mont(a), b) == a*b`: the identity the NTT's twiddle representation uses."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, field.q, 64)
    b = rng.integers(0, field.q, 64)
    got = mont_mul_np(field.to_mont(a), b, field)
    assert np.array_equal(got, field.mul(a, b))


@pytest.mark.parametrize("field", FIELDS)
def test_montgomery_roundtrip(field):
    rng = np.random.default_rng(1)
    a = rng.integers(0, field.q, 64)
    assert np.array_equal(field.from_mont(field.to_mont(a)), a)


@pytest.mark.parametrize("field", FIELDS)
def test_numpy_ops_match_python_ints(field):
    rng = np.random.default_rng(2)
    a = rng.integers(0, field.q, 32).tolist()
    b = rng.integers(0, field.q, 32).tolist()
    q = field.q
    assert field.mul(a, b).tolist() == [(x * y) % q for x, y in zip(a, b, strict=True)]
    assert field.add(a, b).tolist() == [(x + y) % q for x, y in zip(a, b, strict=True)]
    assert field.sub(a, b).tolist() == [(x - y) % q for x, y in zip(a, b, strict=True)]
    assert field.pow(a, 7).tolist() == [pow(x, 7, q) for x in a]
    assert field.mul(a, field.inv(a)).tolist() == [1] * len(a)
    assert field.is_canonical(a)


def test_babybear_is_the_documented_modulus():
    assert BABYBEAR.q == 2**31 - 2**27 + 1
    # 2-adicity 27, which is what makes it NTT-friendly for ZK-sized transforms.
    q = BABYBEAR.q
    assert (q - 1) % 2**27 == 0 and (q - 1) % 2**28 != 0


@pytest.mark.parametrize(
    "q, reason",
    [
        (3328, "even"),
        (2, "too small"),
        (2**31, "needs a 128-bit product (Goldilocks class)"),
        (2**64 - 2**32 + 1, "needs a 128-bit product (Goldilocks class)"),
    ],
)
def test_rejects_moduli_the_32bit_path_cannot_handle(q, reason):
    with pytest.raises(ValueError):
        PrimeField("bad", q)


def test_goldilocks_error_names_the_reason():
    """A future Goldilocks user should learn *why* from the exception, not from a wrong sum."""
    with pytest.raises(ValueError, match="q < 2\\*\\*31"):
        PrimeField("Goldilocks", 2**64 - 2**32 + 1)
