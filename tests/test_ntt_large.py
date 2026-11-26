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


@pytest.mark.parametrize("logn", [10, 12, 15])
def test_radix4_agrees_with_radix2(logn):
    """The fused radix-4 pass is bit-identical to two radix-2 passes, including when log2(n) is
    odd (one radix-2 pass is left over). It is slower on this device, so it is opt-in."""
    a = np.random.default_rng(logn + 400).integers(0, Q, 1 << logn)
    t2 = LargeNTT(1 << logn, Q)
    t4 = LargeNTT(1 << logn, Q, radix=4)
    got4 = t4.forward_torch(a).cpu().numpy().astype(np.int64)
    assert np.array_equal(got4, t2.ntt(a))


def test_the_numpy_input_entry_points_agree_with_the_tensor_ones():
    """`forward_torch`/`inverse_torch` are the documented entry points for host values; both have
    to give the same answer as the device-tensor paths, since a caller will reach for whichever
    matches what it holds."""
    import torch

    logn = 12
    t = LargeNTT(1 << logn, Q)
    rng = np.random.default_rng(logn + 500)
    a = rng.integers(0, Q, 1 << logn)
    x = torch.as_tensor(a, dtype=torch.int32, device="xpu")
    assert np.array_equal(
        t.forward_torch(a).cpu().numpy().astype(np.int64),
        t.forward_tensor_torch(x, x.device).cpu().numpy().astype(np.int64),
    )
    have = rng.integers(0, Q, 1 << logn)
    assert np.array_equal(t.inverse_torch(have).cpu().numpy().astype(np.int64), t.intt(have))


@pytest.mark.parametrize("logn", [10, 14])
def test_degenerate_polynomials(logn):
    """Zero and constant polynomials.

    Zero maps to zero everywhere. A constant `c` does **not** map to `c` at every point, which is
    the tempting assumption and the wrong one: the transform evaluates at the n odd powers of a
    2n-th root, and the pre-twiddled sum runs over only n terms, so `F(c)_k = 2c / (1 - psi^(2k+1))`
    -- finite and different at every point. The test therefore checks the two things that are true:
    zero stays zero, and a constant agrees with the numpy path that the in-register domain also
    agrees with.
    """
    t = LargeNTT(1 << logn, Q)
    n = 1 << logn
    zero = t.forward_torch(np.zeros(n, dtype=np.int64)).cpu().numpy()
    assert not zero.any()
    c = 12345
    flat = t.forward_torch(np.full(n, c, dtype=np.int64)).cpu().numpy().astype(np.int64)
    assert np.array_equal(flat, t.ntt(np.full(n, c, dtype=np.int64)))
    assert not np.array_equal(flat, np.full(n, c))  # the assumption this test exists to kill
    a = np.random.default_rng(logn).integers(0, Q, n)
    assert not t.mul_torch(a, np.zeros(n, dtype=np.int64)).cpu().numpy().any()


@pytest.mark.parametrize("radix", [2, 4])
def test_the_input_tensor_is_not_modified(radix):
    """`forward_tensor_torch` promises "device tensor in, device tensor out".

    It lied: the ping-pong buffer started as `empty_like(x)` and was then swapped, so every pass
    after the first wrote into the caller's tensor. On a 19-pass transform the input ended up
    holding pass 18's output, and a second call transformed the wreckage of the first -- which is
    exactly the pattern a benchmark loop uses.
    """
    import torch

    t = LargeNTT(1 << 12, Q, radix=radix)
    a = np.random.default_rng(radix).integers(0, Q, 1 << 12)
    x = torch.as_tensor(a, dtype=torch.int32, device="xpu")
    before = x.cpu().numpy().copy()
    first = t.forward_tensor_torch(x, dev=x.device).cpu().numpy().astype(np.int64)
    second = t.forward_tensor_torch(x, dev=x.device).cpu().numpy().astype(np.int64)
    assert np.array_equal(before, x.cpu().numpy()), "the input tensor was modified"
    assert np.array_equal(first, second), "a second call on the same tensor differs"
    assert np.array_equal(first, t.ntt(a))


def test_rejects_wrong_shapes_before_touching_the_device():
    """A wrong length used to be a silent out-of-bounds read, not an error.

    The stage kernel loads `x[base + half + j]` without a mask past the buffer, so
    `mul_torch(zeros(64), zeros(32))` returned a result before these checks existed. Every entry
    point now refuses first, and names the operand.
    """
    import torch

    t = LargeNTT(64, Q)
    dev = torch.device("xpu")
    with pytest.raises(ValueError, match=r"b must be 1-D of length 64"):
        t.mul_torch(np.zeros(64, dtype=np.int64), np.zeros(32, dtype=np.int64))
    with pytest.raises(ValueError, match=r"a must be 1-D of length 64"):
        t.mul_torch(np.zeros((2, 64), dtype=np.int64), np.zeros(64, dtype=np.int64))
    with pytest.raises(ValueError, match=r"input must be 1-D of length 64"):
        t.forward_tensor_torch(torch.zeros(32, dtype=torch.int32, device=dev), dev)
    with pytest.raises(ValueError, match=r"input must be 1-D of length 64"):
        t.inverse_tensor_torch(torch.zeros((2, 64), dtype=torch.int32, device=dev), dev)
    with pytest.raises(TypeError, match="must be a torch tensor"):
        t.forward_tensor_torch(np.zeros(64, dtype=np.int64), dev)


def test_rejects_non_power_of_two():
    with pytest.raises(ValueError, match="power of two"):
        LargeNTT(1000, Q)


def test_device_product_matches_the_numpy_product():
    logn = 10
    t = LargeNTT(1 << logn, Q)
    rng = np.random.default_rng(11)
    a = rng.integers(0, Q, 1 << logn)
    b = rng.integers(0, Q, 1 << logn)
    got = t.mul_torch(a, b).cpu().numpy().astype(np.int64)
    assert np.array_equal(got, t.mul_np(a, b))


@pytest.mark.slow
def test_product_of_monomials_at_2_20():
    """A closed form at full scale: `X^i * X^j` is `X^(i+j)` or `-X^(i+j-n)`.

    There is no way to check a 2^20 negacyclic product against a schoolbook one, but monomials
    have an exact answer, and it exercises every piece: the transform, the pointwise multiply, the
    inverse, and the `X^n + 1` wrap.
    """
    logn = 20
    n = 1 << logn
    t = LargeNTT(n, Q)
    for i, j in ((1, 1), (0, n - 1), (n // 3, (2 * n) // 3 + 7), (n - 1, n - 1)):
        a = np.zeros(n, dtype=np.int64)
        b = np.zeros(n, dtype=np.int64)
        a[i] = 5
        b[j] = 7
        got = t.mul_torch(a, b).cpu().numpy().astype(np.int64)
        want = np.zeros(n, dtype=np.int64)
        k = i + j
        if k < n:
            want[k] = 35
        else:
            want[k - n] = Q - 35  # -35 mod q
        assert np.array_equal(got, want), f"X^{i} * X^{j}"


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
