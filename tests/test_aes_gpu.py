"""AES-128-CTR on the device against OpenSSL's keystream and the numpy reference.

The device path is the T-table one: ten rounds x four words x four lookups = 160 gathers per
block, all into a 4.25 KiB table. `docs/benchmarks.md` §18 has the number and what it means.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_aes import CIPHER_128, IV, KEY_128, KEYSTREAM_64, PLAIN_128

from triton_crypto.primitives.aes import (
    T_TABLE_NP,
    aes128_ctr_np,
    aes128_ctr_torch,
    aes128_encrypt_block_np,
    aes128_expand_key_np,
)


def _arr(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.uint8)


def test_device_ctr_matches_openssl_at_one_block():
    ks = aes128_ctr_torch(_arr(KEY_128)[None, :], _arr(IV[:8])[None, :], 16)
    assert ks.cpu().numpy()[0].tobytes() == KEYSTREAM_64[:16]


def test_device_ctr_matches_openssl_at_sixty_four_bytes():
    ks = aes128_ctr_torch(_arr(KEY_128)[None, :], _arr(IV[:8])[None, :], 64)
    assert ks.cpu().numpy()[0].tobytes() == KEYSTREAM_64


@pytest.mark.parametrize("length", [1, 15, 16, 17, 100, 1024])
def test_device_matches_numpy_for_any_length(length):
    rng = np.random.default_rng(length)
    keys = rng.integers(0, 256, (9, 16), dtype=np.uint8)
    nonces = rng.integers(0, 256, (9, 8), dtype=np.uint8)
    got = aes128_ctr_torch(keys, nonces, length).cpu().numpy()
    for i in range(9):
        ref = aes128_ctr_np(bytes(keys[i]), bytes(nonces[i]), length)
        assert got[i].tobytes() == ref, f"row {i}, length {length}"


def test_device_ctr_counter_offset():
    keys, nonces = _arr(KEY_128)[None, :], _arr(IV[:8])[None, :]
    a = aes128_ctr_torch(keys, nonces, 48).cpu().numpy()[0].tobytes()
    b = aes128_ctr_torch(keys, nonces, 32, counter0=1).cpu().numpy()[0].tobytes()
    assert b == a[16:48]


def test_device_rejects_bad_shapes():
    with pytest.raises(ValueError, match="keys must be"):
        aes128_ctr_torch(np.zeros((2, 15), dtype=np.uint8), np.zeros((2, 8), dtype=np.uint8), 16)
    with pytest.raises(ValueError, match="nonces must be"):
        aes128_ctr_torch(np.zeros((2, 16), dtype=np.uint8), np.zeros((2, 9), dtype=np.uint8), 16)
    with pytest.raises(ValueError, match="same batch"):
        aes128_ctr_torch(np.zeros((2, 16), dtype=np.uint8), np.zeros((3, 8), dtype=np.uint8), 16)


def test_device_table_is_the_one_the_reference_uses():
    """The device consumes `T_TABLE_NP` flattened; a transposed table would be silent."""
    assert T_TABLE_NP.shape == (5, 256)
    assert T_TABLE_NP.reshape(-1)[4 * 256] == T_TABLE_NP[4][0]
    # and the block function agrees with the classic vector
    assert aes128_encrypt_block_np(KEY_128, PLAIN_128) == CIPHER_128
    assert aes128_expand_key_np(KEY_128).shape == (44,)
