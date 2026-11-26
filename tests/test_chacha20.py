"""ChaCha20: RFC 8439 test vectors against the numpy reference (CPU-only, no GPU needed).

The vectors are the RFC's, cross-checked against an independent implementation before being
written down (OpenSSL 3.5, `openssl enc -chacha20 -K <key> -iv 01000000||nonce`, which lays
the IV out as a little-endian 32-bit counter followed by the 96-bit nonce — the same
construction this module implements).
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.chacha20 import SIGMA, chacha20_keystream_np

# RFC 8439 §2.3.2 — the block function.
KEY_232 = bytes(range(32))
NONCE_232 = bytes.fromhex("000000090000004a00000000")
KEYSTREAM_232 = bytes.fromhex(
    "10f1e7e4d13b5915500fdd1fa32071c4"
    "c7d1f4c733c068030422aa9ac3d46c4e"
    "d2826446079faa0914c2d705d98b02a2"
    "b5129cd1de164eb9cbd083e8a2503c4e"
)

# RFC 8439 §2.4.2 — encryption of the "sunscreen" paragraph.
KEY_242 = bytes(range(0x80, 0xA0))
NONCE_242 = bytes.fromhex("070000004041424344454647")
PLAIN_242 = (
    b"Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the "
    b"future, sunscreen would be it."
)
CIPHER_242 = bytes.fromhex(
    "d31a8d34648e60db7b86afbc53ef7ec2"
    "a4aded51296e08fea9e2b5a736ee62d6"
    "3dbea45e8ca9671282fafb69da92728b"
    "1a71de0a9e060b2905d6a5b67ecd3b36"
    "92ddbd7f2d778b8c9803aee328091b58"
    "fab324e4fad675945585808b4831d7bc"
    "3ff4def08e4b7a9de576d26586cec64b6116"
)


def test_sigma_is_the_rfc_constant():
    assert SIGMA == (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)


def test_block_function_vector():
    got = chacha20_keystream_np(KEY_232, NONCE_232, 64, counters=1)
    assert got.shape == (1, 64)
    assert got[0].tobytes() == KEYSTREAM_232


def test_encryption_vector():
    ks = chacha20_keystream_np(KEY_242, NONCE_242, len(PLAIN_242), counters=1)[0]
    ct = bytes(a ^ b for a, b in zip(PLAIN_242, ks, strict=True))
    assert ct == CIPHER_242


def test_counter_selects_the_block():
    """Blocks are addressed by counter: block 2 starts where a longer stream's byte 64 does."""
    two = chacha20_keystream_np(KEY_232, NONCE_232, 128, counters=1)[0]
    second = chacha20_keystream_np(KEY_232, NONCE_232, 64, counters=2)[0]
    assert np.array_equal(two[64:], second)
    assert not np.array_equal(two[:64], second)


def test_batch_broadcasts_one_key_over_many_nonces():
    keys = np.frombuffer(KEY_232, dtype=np.uint8)
    nonces = np.stack(
        [
            np.frombuffer(NONCE_232, dtype=np.uint8),
            np.frombuffer(NONCE_242, dtype=np.uint8),
        ]
    )
    out = chacha20_keystream_np(keys, nonces, 64, counters=1)
    assert out.shape == (2, 64)
    assert out[0].tobytes() == KEYSTREAM_232  # broadcast key, so nonce 1 gives the vector


def test_truncates_without_changing_the_prefix():
    assert np.array_equal(
        chacha20_keystream_np(KEY_232, NONCE_232, 100, counters=1)[0][:64],
        np.frombuffer(KEYSTREAM_232, dtype=np.uint8),
    )
    assert chacha20_keystream_np(KEY_232, NONCE_232, 1, counters=1).shape == (1, 1)


@pytest.mark.parametrize(
    "key, nonce",
    [
        (bytes(31), NONCE_232),  # short key
        (KEY_232, bytes(11)),  # short nonce
        (KEY_232, bytes(13)),  # long nonce
    ],
)
def test_rejects_wrong_sizes(key, nonce):
    with pytest.raises(ValueError):
        chacha20_keystream_np(key, nonce, 64)


def test_rejects_mismatched_batches():
    keys = np.zeros((3, 32), dtype=np.uint8)
    nonces = np.zeros((2, 12), dtype=np.uint8)
    with pytest.raises(ValueError, match="batch mismatch"):
        chacha20_keystream_np(keys, nonces, 64)
