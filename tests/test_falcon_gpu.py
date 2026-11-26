"""Falcon verification on the device: the batched core against the numpy reference.

What is checked here is the part that actually runs on the device -- `s2*h`, the subtraction of
the hashed point, the normalisation into [-q/2, q/2] and the L2 bound. The decoded signature
and the hashed point are computed on the host, so these tests feed the device the same inputs
the reference uses and compare the verdict *and* the squared norm, which is the only diagnostic
a rejection has.

Two traps this file exists to keep pinned, both found while writing the kernel:

* Triton's `%` on signed integers is C's truncated remainder, so `(prod - c) % q` leaves
  negative values negative -- numpy's floored `%` does not. That silently changes every norm.
* an `expand`ed tensor is not contiguous, so a kernel that assumes contiguity reads past the
  end of the first row; the pointwise kernel indexes the public key's transform modulo `n`.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_falcon import MSG, PK, SIG

from triton_crypto.falcon import FALCON_512, Verifier, comp_decode, hash_to_point, ref
from triton_crypto.falcon.kernels import check_np


def _prepare():
    p = FALCON_512
    s2 = comp_decode(SIG[41:], p)[0]
    c = hash_to_point(SIG[1:41], MSG, p)
    return p, s2, c


def test_one_signature_verdict_matches_the_reference():
    p, s2, c = _prepare()
    v = Verifier(PK)
    ok_ref, sqn_ref = ref.verify_core(c, s2, v.pk_poly, p)
    ok, sqn = v.verify_core_torch(c[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is ok_ref is True
    assert int(sqn.cpu().numpy()[0]) == sqn_ref


def test_batch_matches_the_reference_and_is_order_independent():
    p, s2, c = _prepare()
    v = Verifier(PK)
    rows = 37
    s2b = np.repeat(s2[None, :], rows, axis=0).astype(np.int32)
    s2b[1::2] = -s2b[1::2]  # different coefficient signs: different norms, same code path
    cb = np.repeat(c[None, :], rows, axis=0)
    _ok, sqn = v.verify_core_torch(cb, s2b)
    got = sqn.cpu().numpy()
    for i in range(rows):
        _, expect = ref.verify_core(cb[i], s2b[i].astype(np.int64), v.pk_poly, p)
        assert got[i] == expect, f"row {i}"


def test_a_different_message_is_rejected():
    p, s2, _ = _prepare()
    v = Verifier(PK)
    other = hash_to_point(SIG[1:41], b"a different message", p)
    ok, _ = v.verify_core_torch(other[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is False


def test_a_one_unit_perturbation_of_the_hash_point_can_still_verify():
    """Falcon's verdict is a norm bound, not an equality -- this is not a bug, but it is
    worth pinning, because it is the difference between "signature is valid" and "signature
    was produced by the signer". The embedded signature's norm is 30.08M against a bound of
    34.03M, so ~12% of slack remains and a +1 on one coefficient can *shrink* the norm."""
    p, s2, c = _prepare()
    v = Verifier(PK)
    c2 = c.copy()
    c2[0] = (c2[0] + 1) % p.q  # one coefficient, not all of them
    ok, _ = v.verify_core_torch(c2[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is True
    # and it is still the norm bound that decides: the reference agrees on this input
    ok_ref, _ = ref.verify_core(c2, s2, v.pk_poly, p)
    assert ok_ref is True


def test_check_kernel_matches_numpy_on_negative_differences():
    """The negative-modulo trap, isolated: half of `prod - c` is negative by construction."""
    q, n, bound = FALCON_512.q, FALCON_512.n, FALCON_512.l2bound
    rng = np.random.default_rng(5)
    prod = rng.integers(0, q, (6, n))
    c = rng.integers(0, q, (6, n))
    s2 = rng.integers(-400, 400, (6, n))
    ok_ref, sqn_ref = check_np(prod, c, s2, q, bound)
    import torch

    from triton_crypto.falcon.kernels import _check_kernel

    sqn = torch.empty(6, dtype=torch.int64, device="xpu")
    ok = torch.empty(6, dtype=torch.int8, device="xpu")
    _check_kernel[(6,)](
        torch.tensor(prod, dtype=torch.int32, device="xpu"),
        torch.tensor(c, dtype=torch.int32, device="xpu"),
        torch.tensor(s2, dtype=torch.int32, device="xpu"),
        sqn,
        ok,
        6,
        L2BOUND=bound,
        Q=q,
        N=n,
    )
    assert np.array_equal(sqn.cpu().numpy(), sqn_ref)
    assert np.array_equal(ok.cpu().numpy().astype(bool), ok_ref)


def test_public_api_verifies_the_embedded_signature():
    v = Verifier(PK)
    assert v.verify([MSG], [SIG]).tolist() == [True]
    bad = bytearray(SIG)
    bad[200] ^= 0x01
    assert v.verify([MSG], [bytes(bad)]).tolist() == [False]


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_public_api_batches(batch):
    v = Verifier(PK)
    ok = v.verify([MSG] * batch, [SIG] * batch)
    assert ok.shape == (batch,)
    assert ok.all()
