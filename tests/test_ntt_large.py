"""Multi-pass NTT: the shape the in-register kernel cannot reach.

The reference here is the *same* algorithm in numpy (`LargeNTT.ntt`), which is in turn checked
against `NTTDomain` -- the in-register kernel's numpy twin -- at sizes both can do. That chain
matters: a multi-pass transform that agrees with its own numpy twin but not with the domain would
be a different transform, and pointwise products across the two kernels would silently break.

Sizes: 2^4 and 2^8 are small enough for the domain to cover them; 2^14 and 2^16 are the shapes a
Merkle-tree prover actually uses; 2^20 (the ZK target) is behind `-k slow`.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.ntt import NTTDomain
from triton_crypto.primitives.ntt.large import LargeNTT

Q = 2013265921  # BabyBear


def _reference(logn: int):
    t = LargeNTT(1 << logn, Q)
    dom = NTTDomain("babybear", q=Q, n=1 << logn, zeta=t.psi, layers=logn)
    return t, dom


@pytest.mark.parametrize("logn", [4, 8])
def test_forward_agrees_with_the_in_register_domain(logn):
    t, dom = _reference(logn)
    a = np.random.default_rng(logn).integers(0, Q, 1 << logn)
    assert np.array_equal(t.ntt(a), dom.ntt(a))


@pytest.mark.parametrize("logn", [4, 8, 10, 12])
def test_round_trip(logn):
    t = LargeNTT(1 << logn, Q)
    a = np.random.default_rng(logn + 100).integers(0, Q, 1 << logn)
    assert np.array_equal(t.intt(t.ntt(a)), a % Q)


def test_pointwise_product_is_the_negacyclic_convolution():
    """The property that makes an NTT useful: NTT(a) * NTT(b) inverse-transforms to a*b mod X^n+1."""
    logn = 8
    n = 1 << logn
    t = LargeNTT(n, Q)
    rng = np.random.default_rng(7)
    a = rng.integers(0, Q, n)
    b = rng.integers(0, Q, n)
    got = t.intt(t.ntt(a) * t.ntt(b) % Q)
    # schoolbook negacyclic product, reduced per term (the q^2-sized accumulator overflows int64)
    want = np.zeros(n, dtype=np.int64)
    for i in range(n):
        for j in range(n):
            k = i + j
            term = int(a[i]) * int(b[j]) % Q
            if k < n:
                want[k] = (want[k] + term) % Q
            else:
                want[k - n] = (want[k - n] - term) % Q
    assert np.array_equal(got % Q, want % Q)


@pytest.mark.parametrize("logn", [12, 14])
def test_device_forward_matches_the_numpy_path(logn):
    t = LargeNTT(1 << logn, Q)
    a = np.random.default_rng(logn + 200).integers(0, Q, 1 << logn)
    got = t.forward_torch(a).cpu().numpy().astype(np.int64)
    assert np.array_equal(got, t.ntt(a))


@pytest.mark.parametrize("logn", [12, 14])
def test_device_round_trip(logn):
    t = LargeNTT(1 << logn, Q)
    a = np.random.default_rng(logn + 300).integers(0, Q, 1 << logn)
    got = t.forward_torch(a)
    back = t.inverse_tensor_torch(got, got.device).cpu().numpy().astype(np.int64)
    assert np.array_equal(back, a % Q)


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        LargeNTT(1000, Q)


@pytest.mark.slow
def test_the_zk_target_size():
    """2^20: correct, and the number that matters is in docs/benchmarks.md §22."""
    logn = 20
    n = 1 << logn
    t = LargeNTT(n, Q)
    a = np.random.default_rng(20).integers(0, Q, n)
    got_t = t.forward_torch(a)
    got = got_t.cpu().numpy().astype(np.int64)
    assert np.array_equal(got, t.ntt(a))
    back = t.inverse_tensor_torch(got_t, got_t.device).cpu().numpy().astype(np.int64)
    assert np.array_equal(back, a % Q)
