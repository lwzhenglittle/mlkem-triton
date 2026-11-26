"""The shared Montgomery helper on the device, called from a *different* module.

This is the regression test for the rule `primitives/fields.py` documents: a `@triton.jit`
helper is shareable as long as the module that defines it imports `triton.language as tl` at
module level. Before this test existed, the same four lines were inlined three times inside
`primitives/ntt.py` because a helper defined in a factory (`import triton.language as tl` in a
function body) has no `tl` in its compiled globals and fails with `NameError: tl is not defined`.

If someone ever moves that import back inside a function, this file goes red.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import triton
import triton.language as tl
from _common import DEV

from triton_crypto.primitives.fields import BABYBEAR, PrimeField, mont_mul
from triton_crypto.primitives.ntt import (
    BABYBEAR_NTT,
    DILITHIUM,
    KYBER,
    intt_np,
    intt_torch,
    negacyclic_conv_np,
    ntt_np,
    ntt_torch,
)


@triton.jit
def _mont_kernel(A, B, OUT, N, Q: tl.constexpr, QINV: tl.constexpr):
    """Defined here, imported `mont_mul` from another module: that is the whole point."""
    i = tl.program_id(0) * 128 + tl.arange(0, 128)
    m = i < N
    a = tl.load(A + i, mask=m, other=0).to(tl.int64)
    b = tl.load(B + i, mask=m, other=0).to(tl.int64)
    tl.store(OUT + i, mont_mul(a, b, Q, QINV), mask=m)


FIELDS = [
    PrimeField("ML-KEM", 3329),
    PrimeField("ML-DSA", 8380417),
    BABYBEAR,
]


@pytest.mark.parametrize("field", FIELDS)
def test_mont_mul_is_shared_across_modules(field):
    rng = np.random.default_rng(3)
    a = rng.integers(0, field.q, 1000)
    b = rng.integers(0, field.q, 1000)
    dev = torch.device(DEV)
    at = torch.from_numpy(field.to_mont(a).astype(np.int64)).to(dev)
    bt = torch.from_numpy(b.astype(np.int64)).to(dev)
    out = torch.empty_like(at)
    _mont_kernel[(triton.cdiv(a.size, 128),)](at, bt, out, a.size, field.q, field.qinv)

    want = field.mul(a, b)  # mont(to_mont(a), b) == a*b
    assert np.array_equal(out.cpu().numpy(), want)


@pytest.mark.parametrize("domain", [KYBER, DILITHIUM, BABYBEAR_NTT])
def test_device_ntt_uses_the_shared_arithmetic(domain):
    """The refactor replaced three inlined Montgomery reductions; the transform must be
    bit-identical to the numpy reference for every domain, including BabyBear."""
    rng = np.random.default_rng(4)
    a = rng.integers(0, domain.q, (4, domain.n))
    ref = ntt_np(a, domain)
    got = ntt_torch(torch.from_numpy(a.astype(np.int32)).to(torch.device(DEV)), domain)
    assert np.array_equal(got.cpu().numpy().astype(np.int64), ref)

    back = intt_torch(got, domain)
    assert np.array_equal(back.cpu().numpy().astype(np.int64), a)


def test_babybear_ntt_is_a_ring_homomorphism():
    """Pointwise product in the transform domain == negacyclic product in the ring."""
    dom = BABYBEAR_NTT
    rng = np.random.default_rng(5)
    a = rng.integers(0, dom.q, (2, dom.n))
    b = rng.integers(0, dom.q, (2, dom.n))
    prod = intt_np(dom.mul(ntt_np(a, dom), ntt_np(b, dom)), dom)
    ref = np.stack([negacyclic_conv_np(a[i], b[i], dom) for i in range(2)])
    assert np.array_equal(prod, ref)
