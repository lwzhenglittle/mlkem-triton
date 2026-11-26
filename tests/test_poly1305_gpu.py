"""Poly1305 and ChaCha20-Poly1305 on the device, against the RFC and the reference.

The MAC vectors are RFC 8439's; the AEAD vector is §2.8.2 (with AAD, 114-byte plaintext), which
is the one that exercises pad16, the AAD prefix and the length trailer at once.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.poly1305 import (
    chacha20_poly1305_open_torch,
    chacha20_poly1305_torch,
    poly1305_np,
    poly1305_torch,
)

# RFC 8439 §2.8.2
AEAD_KEY = bytes.fromhex("808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f")
AEAD_NONCE = bytes.fromhex("070000004041424344454647")
AEAD_AAD = bytes.fromhex("50515253c0c1c2c3c4c5c6c7")
AEAD_PLAIN = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the "
    b"future, sunscreen would be it."
)
AEAD_CT = bytes.fromhex(
    "d31a8d34648e60db7b86afbc53ef7ec2a4aded51296e08fea9e2b5a736ee62d6"
    "3dbea45e8ca9671282fafb69da92728b1a71de0a9e060b2905d6a5b67ecd3b36"
    "92ddbd7f2d778b8c9803aee328091b58fab324e4fad675945585808b4831d7bc"
    "3ff4def08e4b7a9de576d26586cec64b6116"
)
AEAD_TAG = bytes.fromhex("1ae10b594f09e26a7e902ecbd0600691")


def _arr(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8)[None, :]


@pytest.mark.parametrize("nbytes", [16, 32, 48, 1024])
def test_mac_matches_the_reference(nbytes):
    rng = np.random.default_rng(nbytes)
    keys = rng.integers(0, 256, (16, 32), dtype=np.uint8)
    msgs = rng.integers(0, 256, (16, nbytes), dtype=np.uint8)
    got = poly1305_torch(keys, msgs).cpu().numpy()
    ref = np.stack(
        [
            np.frombuffer(poly1305_np(bytes(k), bytes(m)), dtype=np.uint8)
            for k, m in zip(keys, msgs, strict=True)
        ]
    )
    assert np.array_equal(got, ref)


def test_mac_rejects_partial_blocks():
    """The device path needs whole blocks: the last partial block's 0x01 byte *is* the high
    bit, so adding 2^128 on top of it (which the kernel always does) would be wrong."""
    keys = np.zeros((2, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="N\\*16"):
        poly1305_torch(keys, np.zeros((2, 17), dtype=np.uint8))


def test_aead_rfc_8439_vector():
    ct, tag = chacha20_poly1305_torch(
        _arr(AEAD_KEY), _arr(AEAD_NONCE), _arr(AEAD_PLAIN), _arr(AEAD_AAD)
    )
    assert ct.cpu().numpy()[0].tobytes() == AEAD_CT
    assert tag.cpu().numpy()[0].tobytes() == AEAD_TAG


def test_aead_open_round_trip_and_tamper():
    ct, tag = chacha20_poly1305_torch(
        _arr(AEAD_KEY), _arr(AEAD_NONCE), _arr(AEAD_PLAIN), _arr(AEAD_AAD)
    )
    pt, ok = chacha20_poly1305_open_torch(
        _arr(AEAD_KEY), _arr(AEAD_NONCE), ct.cpu().numpy(), tag.cpu().numpy(), _arr(AEAD_AAD)
    )
    assert bool(ok.cpu().numpy()[0]) is True
    assert pt.cpu().numpy()[0].tobytes() == AEAD_PLAIN

    bad = tag.cpu().numpy().copy()
    bad[0, 0] ^= 0x01
    _, ok_bad = chacha20_poly1305_open_torch(
        _arr(AEAD_KEY), _arr(AEAD_NONCE), ct.cpu().numpy(), bad, _arr(AEAD_AAD)
    )
    assert bool(ok_bad.cpu().numpy()[0]) is False


def test_aead_without_aad_differs_from_with_aad():
    """The AAD goes into the MAC, so the tag must change (the ciphertext must not)."""
    ct_a, tag_a = chacha20_poly1305_torch(
        _arr(AEAD_KEY), _arr(AEAD_NONCE), _arr(AEAD_PLAIN), _arr(AEAD_AAD)
    )
    ct_n, tag_n = chacha20_poly1305_torch(_arr(AEAD_KEY), _arr(AEAD_NONCE), _arr(AEAD_PLAIN))
    assert ct_a.cpu().numpy()[0].tobytes() == ct_n.cpu().numpy()[0].tobytes()
    assert tag_a.cpu().numpy()[0].tobytes() != tag_n.cpu().numpy()[0].tobytes()


@pytest.mark.parametrize("nbytes", [1, 16, 64, 1000])
def test_aead_arbitrary_lengths_round_trip(nbytes):
    rng = np.random.default_rng(nbytes)
    keys = rng.integers(0, 256, (3, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (3, 12), dtype=np.uint8)
    pt_in = rng.integers(0, 256, (3, nbytes), dtype=np.uint8)
    ct, tag = chacha20_poly1305_torch(keys, nonces, pt_in)
    pt, ok = chacha20_poly1305_open_torch(keys, nonces, ct.cpu().numpy(), tag.cpu().numpy())
    assert bool(ok.all().cpu().numpy()) is True
    assert np.array_equal(pt.cpu().numpy(), pt_in)
