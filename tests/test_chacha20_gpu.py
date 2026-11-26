"""ChaCha20 on the device: the kernel against the RFC vectors and against the numpy reference.

The reference is what makes the batch cases trustworthy (arbitrary keys, nonces, counters,
sizes); the RFC vectors are what makes the reference trustworthy in the first place.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_chacha20 import (
    CIPHER_242,
    KEY_232,
    KEY_242,
    KEYSTREAM_232,
    NONCE_232,
    NONCE_242,
    PLAIN_242,
)

from triton_crypto.primitives.chacha20 import (
    chacha20_keystream,
    chacha20_keystream_np,
    chacha20_xor,
)

SIZES = [1, 63, 64, 65, 192, 1000]


def test_rfc_block_vector_on_device():
    got = chacha20_keystream(KEY_232, NONCE_232, 64, counters=1)
    assert got[0].tobytes() == KEYSTREAM_232


def test_rfc_encryption_on_device():
    ct = chacha20_xor(KEY_242, NONCE_242, PLAIN_242, counters=1)
    assert ct[0].tobytes() == CIPHER_242
    assert chacha20_xor(KEY_242, NONCE_242, ct, counters=1)[0].tobytes() == PLAIN_242


@pytest.mark.parametrize("nbytes", SIZES)
def test_matches_reference_across_sizes(nbytes):
    rng = np.random.default_rng(nbytes)
    keys = rng.integers(0, 256, (3, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (3, 12), dtype=np.uint8)
    ctrs = rng.integers(0, 2**31, (3,), dtype=np.uint32)
    assert np.array_equal(
        chacha20_keystream(keys, nonces, nbytes, ctrs),
        chacha20_keystream_np(keys, nonces, nbytes, ctrs),
    )


def test_batches_are_independent():
    """Different keys must give different streams — the classic bug is one shared state."""
    keys = np.zeros((4, 32), dtype=np.uint8)
    keys[1:] = np.arange(1, 4, dtype=np.uint8)[:, None]
    nonces = np.zeros((4, 12), dtype=np.uint8)
    out = chacha20_keystream(keys, nonces, 64)
    assert len({row.tobytes() for row in out}) == 4


def test_counter_continues_across_blocks():
    long = chacha20_keystream(KEY_232, NONCE_232, 192, counters=1)[0]
    for i in range(3):
        assert np.array_equal(
            long[i * 64 : (i + 1) * 64],
            chacha20_keystream(KEY_232, NONCE_232, 64, counters=1 + i)[0],
        )


def test_large_batch_crosses_the_row_tile():
    """The kernel handles 64 rows per program; 200 rows exercises the tail mask."""
    rng = np.random.default_rng(11)
    keys = rng.integers(0, 256, (200, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (200, 12), dtype=np.uint8)
    assert np.array_equal(
        chacha20_keystream(keys, nonces, 128, 7),
        chacha20_keystream_np(keys, nonces, 128, 7),
    )


def test_xor_xor_is_identity():
    rng = np.random.default_rng(12)
    data = rng.integers(0, 256, (2, 300), dtype=np.uint8)
    keys = rng.integers(0, 256, (2, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (2, 12), dtype=np.uint8)
    once = chacha20_xor(keys, nonces, data)
    assert np.array_equal(chacha20_xor(keys, nonces, once), data)
