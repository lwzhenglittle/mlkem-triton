"""Tests for `triton_crypto.primitives.sm3` (GB/T 32905-2016 / ISO/IEC 10118-3)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
from _common import DEV

from triton_crypto.primitives.sm3 import (
    IV_SM3,
    sm3_compress_np,
    sm3_hash_np,
    sm3_hash_np_batch,
    sm3_hash_torch,
    sm3_pad_torch,
)

# Official GB/T 32905-2016 Appendix A test vectors + empty message
GB_VECTORS = [
    (b"", "1ab21d8355cfa17f8e61194831e81a8f22bec8c728fefb747ed035eb5082aa2b"),
    (b"abc", "66c7f0f462eeedd9d1f2d46bdc10e4e24167c4875cf2f7a2297da02b8f4ba8e0"),
    (b"abcd" * 16, "debe9ff92275b8a138604889c18e5a4d6fdb70e5387e5765293dcba39c0c5732"),
]


@pytest.mark.parametrize(("msg", "expected"), GB_VECTORS)
def test_sm3_ref_gb_vectors(msg: bytes, expected: str):
    assert sm3_hash_np(msg).hex() == expected
    if "sm3" in hashlib.algorithms_available:
        assert hashlib.new("sm3", msg).hexdigest() == expected


@pytest.mark.parametrize("mlen", [0, 1, 55, 56, 63, 64, 65, 119, 120, 128, 1024])
def test_sm3_ref_matches_hashlib(mlen: int):
    if "sm3" not in hashlib.algorithms_available:
        pytest.skip("OpenSSL/hashlib SM3 unavailable")
    rng = np.random.default_rng(1000 + mlen)
    msg = rng.integers(0, 256, size=mlen, dtype=np.uint8).tobytes()
    assert sm3_hash_np(msg) == hashlib.new("sm3", msg).digest()


def test_sm3_compress_validation():
    with pytest.raises(ValueError, match="multiple of 64"):
        sm3_compress_np(list(IV_SM3), b"\x00" * 63)


@pytest.mark.gpu
@pytest.mark.parametrize("mlen", [0, 3, 55, 56, 64, 65, 128, 1024])
def test_sm3_gpu_matches_ref_and_hashlib(mlen: int):
    import torch

    rng = np.random.default_rng(2000 + mlen)
    batch = 131  # non-multiple of BLOCK=128
    msgs = rng.integers(0, 256, size=(batch, mlen), dtype=np.uint8)
    expected = sm3_hash_np_batch(msgs)
    got = sm3_hash_torch(msgs, device=DEV).cpu().numpy()
    np.testing.assert_array_equal(got, expected)

    # Also verify pre-padded path
    t = torch.from_numpy(msgs).to(DEV)
    padded = sm3_pad_torch(t)
    got_pp = sm3_hash_torch(padded, device=DEV, pre_padded=True).cpu().numpy()
    np.testing.assert_array_equal(got_pp, expected)


@pytest.mark.gpu
def test_sm3_gpu_gb_vectors_and_empty_batch():
    import torch

    for msg, expected in GB_VECTORS:
        got = sm3_hash_torch([msg], device=DEV).cpu().numpy()[0]
        assert bytes(got).hex() == expected
    empty = sm3_hash_torch(torch.empty((0, 64), dtype=torch.uint8, device=DEV), device=DEV)
    assert empty.shape == (0, 32)
