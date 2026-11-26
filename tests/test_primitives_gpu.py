"""GPU-side tests which need a device: sponge vs hashlib, NTT vs numpy, for both domains.

These are the "is the primitive actually the same primitive on the GPU" checks. They are
deliberately cross-domain: a bug that only shows up at q = 8380417 (int64 path, Montgomery
tables, 8 layers) is invisible to the ML-KEM cases, and vice versa.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from triton_crypto.primitives import (
    DILITHIUM,
    HASHES,
    KYBER,
    intt_np,
    intt_torch,
    ntt_np,
    ntt_torch,
    sponge_padded,
)
from triton_crypto.primitives.sponge import pad_batch, pad_torch

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")


def _np_to_dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


# ---------------- sponge ----------------


@pytest.mark.parametrize("name", sorted(HASHES))
@pytest.mark.parametrize("msglen", [0, 1, 5, 135, 136, 137, 300, 1000])
def test_sponge_matches_hashlib(name, msglen):
    hf = HASHES[name]
    n = 7
    rng = np.random.default_rng(msglen)
    msgs = rng.integers(0, 256, (n, msglen), dtype=np.uint8)
    outlen = (hf.outlen or 64) if name.startswith("SHA3") else 96
    if hf.outlen is not None and outlen > hf.outlen:
        outlen = hf.outlen
    got = sponge_padded(pad_torch(_np_to_dev(msgs), hf.rate, hf.domain), hf.rate, outlen)
    got = got.cpu().numpy()
    for i in range(n):
        h = hashlib.new(hf.ref_name, msgs[i].tobytes())
        want = h.digest(outlen) if hf.outlen is None else h.digest()
        assert got[i].tobytes() == want, (name, msglen, i)


@pytest.mark.parametrize("outlen", [32, 64, 136, 272, 408])
def test_sponge_multi_block_squeeze(outlen):
    rng = np.random.default_rng(outlen)
    msgs = rng.integers(0, 256, (3, 40), dtype=np.uint8)
    got = sponge_padded(pad_torch(_np_to_dev(msgs), 136, 0x1F), 136, outlen).cpu().numpy()
    for i in range(3):
        want = hashlib.shake_256(msgs[i].tobytes()).digest(outlen)
        assert got[i].tobytes() == want


def test_sponge_pad_batch_agrees_with_pad_torch():
    rng = np.random.default_rng(0)
    msgs = rng.integers(0, 256, (4, 200), dtype=np.uint8)
    a = pad_batch(msgs, 136, 0x1F)
    b = pad_torch(_np_to_dev(msgs), 136, 0x1F).cpu().numpy()
    assert np.array_equal(a, b)


# ---------------- NTT ----------------


@pytest.mark.parametrize("dom", [KYBER, DILITHIUM], ids=lambda d: d.name)
def test_gpu_ntt_matches_numpy(dom):
    x = np.random.default_rng(1).integers(0, dom.q, (37, 256), dtype=np.int64)
    got = ntt_torch(_np_to_dev(x.astype(np.int32)), dom).cpu().numpy().astype(np.int64)
    assert np.array_equal(got, ntt_np(x, dom))


@pytest.mark.parametrize("dom", [KYBER, DILITHIUM], ids=lambda d: d.name)
def test_gpu_intt_matches_numpy_and_roundtrips(dom):
    x = np.random.default_rng(2).integers(0, dom.q, (19, 256), dtype=np.int64)
    xh = ntt_np(x, dom)
    got = intt_torch(_np_to_dev(xh.astype(np.int32)), dom).cpu().numpy().astype(np.int64)
    assert np.array_equal(got, intt_np(xh, dom))
    assert np.array_equal(got, x)


def test_gpu_ntt_ring_identities():
    """The GPU transform composed with the *scheme's* product must be a convolution."""
    rng = np.random.default_rng(3)
    a = rng.integers(0, DILITHIUM.q, 256, dtype=np.int64)
    b = rng.integers(0, DILITHIUM.q, 256, dtype=np.int64)
    ah, bh = (
        ntt_torch(_np_to_dev(a.astype(np.int32)), DILITHIUM),
        ntt_torch(_np_to_dev(b.astype(np.int32)), DILITHIUM),
    )
    prod = (ah.to(torch.int64) * bh.to(torch.int64) % DILITHIUM.q).to(torch.int32)
    got = intt_torch(prod, DILITHIUM).cpu().numpy().astype(np.int64)
    assert np.array_equal(got, DILITHIUM.conv(a, b))


def test_mlkem_gpu_ntt_feeds_basecase_product():
    from triton_crypto import ntt_ref as kem_ref

    rng = np.random.default_rng(4)
    a = rng.integers(0, KYBER.q, 256, dtype=np.int64)
    b = rng.integers(0, KYBER.q, 256, dtype=np.int64)
    ah = ntt_torch(_np_to_dev(a.astype(np.int32)), KYBER).cpu().numpy().astype(np.int64)
    bh = ntt_torch(_np_to_dev(b.astype(np.int32)), KYBER).cpu().numpy().astype(np.int64)
    assert np.array_equal(ah, ntt_np(a, KYBER))
    got = intt_np(kem_ref.multiply_ntts(ah, bh), KYBER)
    assert np.array_equal(got, kem_ref.negacyclic_conv(a, b))


@pytest.mark.parametrize("br", [1, 2, 4])
def test_ntt_row_blocking_is_exact(br):
    x = np.random.default_rng(5).integers(0, DILITHIUM.q, (11, 256), dtype=np.int64)
    got = ntt_torch(_np_to_dev(x.astype(np.int32)), DILITHIUM, br=br).cpu().numpy()
    assert np.array_equal(got.astype(np.int64), ntt_np(x, DILITHIUM))


def test_keccak_f_kernel_matches_reference():
    """The standalone permutation is a real entry point, not just a constants module."""
    from triton_crypto.primitives.keccak import get_kernel, keccak_f_ref

    n = 5
    rng = np.random.default_rng(11)
    states = rng.integers(0, 2**64, (n, 25), dtype=np.uint64)
    x = torch.from_numpy(states.T.copy().view(np.int64)).to(DEV)  # [25, n] bit patterns
    out = torch.empty((25, n), dtype=torch.int64, device=DEV)
    get_kernel()[(1,)](x, out, n, BLOCK=32)
    got = out.cpu().numpy().view(np.uint64)  # [25, n]
    for i in range(n):
        want = keccak_f_ref(states[i])
        assert np.array_equal(got[:, i], want), i
