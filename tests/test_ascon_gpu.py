"""Tests for `triton_crypto.primitives.ascon` (NIST SP 800-232 Ascon-Hash256 / AEAD128 & Ascon-128a v1.2)."""

from __future__ import annotations

import numpy as np
import pytest
from _common import DEV

from triton_crypto.primitives.ascon import (
    ascon128a_decrypt_np,
    ascon128a_encrypt_np,
    ascon128a_open_torch,
    ascon128a_torch,
    ascon_aead128_decrypt_np,
    ascon_aead128_encrypt_np,
    ascon_aead128_open_torch,
    ascon_aead128_torch,
    ascon_hash256_np,
    ascon_hash256_np_batch,
    ascon_hash256_torch,
)


def test_ascon_hash256_sp800_232_official_empty_vector():
    # Official NIST SP 800-232 Ascon-Hash256 Count=1 (empty message)
    assert (
        ascon_hash256_np(b"").hex()
        == "0b3be5850f2f6b98caf29f8fdea89b64a1fa70aa249b8f839bd53baa304d92b2"
    )


@pytest.mark.parametrize("pt_len", [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 65])
@pytest.mark.parametrize("ad_len", [0, 5, 16, 23])
@pytest.mark.parametrize("sp800", [True, False])
def test_ascon_aead_ref_roundtrip_and_tamper(pt_len: int, ad_len: int, sp800: bool):
    rng = np.random.default_rng(3000 + pt_len * 37 + ad_len * 7 + int(sp800))
    key = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
    nonce = rng.integers(0, 256, size=16, dtype=np.uint8).tobytes()
    pt = rng.integers(0, 256, size=pt_len, dtype=np.uint8).tobytes()
    ad = rng.integers(0, 256, size=ad_len, dtype=np.uint8).tobytes()
    enc = ascon_aead128_encrypt_np if sp800 else ascon128a_encrypt_np
    dec = ascon_aead128_decrypt_np if sp800 else ascon128a_decrypt_np

    ct, tag = enc(key, nonce, pt, ad)
    assert len(ct) == pt_len and len(tag) == 16
    got_pt, ok = dec(key, nonce, ct, tag, ad)
    assert ok and got_pt == pt

    bad_tag = bytes([tag[0] ^ 1]) + tag[1:]
    _, ok_bad = dec(key, nonce, ct, bad_tag, ad)
    assert not ok_bad


@pytest.mark.gpu
@pytest.mark.parametrize("mlen", [0, 1, 7, 8, 9, 15, 16, 64, 1024])
def test_ascon_hash256_gpu_matches_ref(mlen: int):
    rng = np.random.default_rng(4000 + mlen)
    batch = 131
    msgs = rng.integers(0, 256, size=(batch, mlen), dtype=np.uint8)
    expected = ascon_hash256_np_batch(msgs)
    got = ascon_hash256_torch(msgs, device=DEV).cpu().numpy()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.gpu
@pytest.mark.parametrize("pt_len", [0, 1, 7, 8, 9, 15, 16, 17, 1024])
@pytest.mark.parametrize("ad_len", [0, 11, 16])
@pytest.mark.parametrize("sp800", [True, False])
def test_ascon_aead_gpu_matches_ref_and_open(pt_len: int, ad_len: int, sp800: bool):
    rng = np.random.default_rng(5000 + pt_len * 19 + ad_len * 3 + int(sp800))
    batch = 67
    keys = rng.integers(0, 256, size=(batch, 16), dtype=np.uint8)
    nonces = rng.integers(0, 256, size=(batch, 16), dtype=np.uint8)
    pts = rng.integers(0, 256, size=(batch, pt_len), dtype=np.uint8)
    ads = rng.integers(0, 256, size=(batch, ad_len), dtype=np.uint8) if ad_len else None

    seal_gpu = ascon_aead128_torch if sp800 else ascon128a_torch
    open_gpu = ascon_aead128_open_torch if sp800 else ascon128a_open_torch
    seal_ref = ascon_aead128_encrypt_np if sp800 else ascon128a_encrypt_np

    ct_gpu, tag_gpu = seal_gpu(keys, nonces, pts, ad=ads, device=DEV)
    ct_np = ct_gpu.cpu().numpy()
    tag_np = tag_gpu.cpu().numpy()

    for i in (0, 1, batch // 2, batch - 1):
        exp_ct, exp_tag = seal_ref(
            bytes(keys[i]),
            bytes(nonces[i]),
            bytes(pts[i]),
            bytes(ads[i]) if ads is not None else b"",
        )
        assert bytes(ct_np[i]) == exp_ct
        assert bytes(tag_np[i]) == exp_tag

    rec_pt, ok = open_gpu(keys, nonces, ct_gpu, tag_gpu, ad=ads, device=DEV)
    assert bool(ok.all())
    np.testing.assert_array_equal(rec_pt.cpu().numpy(), pts)

    # Flip one bit in row 3's tag and verify only row 3 fails
    bad_tags = tag_gpu.clone()
    bad_tags[3, 0] ^= 0x01
    _, ok_bad = open_gpu(keys, nonces, ct_gpu, bad_tags, ad=ads, device=DEV)
    expected_ok = np.ones(batch, dtype=bool)
    expected_ok[3] = False
    np.testing.assert_array_equal(ok_bad.cpu().numpy(), expected_ok)
