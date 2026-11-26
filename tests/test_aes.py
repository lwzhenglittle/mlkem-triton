"""AES-128: FIPS-197 for the block function, OpenSSL for the CTR stream.

Only the T-table CTR path exists, and it exists as a measurement (`docs/benchmarks.md` §18):
on a GPU a table-driven cipher is a gather benchmark, not a cipher. The vectors matter because
two things here are easy to get subtly wrong and neither crashes:

* the T-table column convention -- `Te0[x] = [2s, s, s, 3s]`, not `[s, 2s, 4s, 8s]`;
* `Rcon` is the round word's **top** byte, so it has to be shifted by 24 before XOR.

Both produced plausible-looking ciphertext for every input until a known vector was checked.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.aes import (
    SBOX,
    aes128_ctr_np,
    aes128_encrypt_block_np,
    aes128_expand_key_np,
    t_table,
)

# FIPS-197 appendix C.1
KEY_128 = bytes(range(16))
PLAIN_128 = bytes.fromhex("00112233445566778899aabbccddeeff")
CIPHER_128 = bytes.fromhex("69c4e0d86a7b0430d8cdb78070b4c55a")

# `openssl enc -aes-128-ctr -K <KEY_128> -iv 00010203040506070000000000000000` over zeros
IV = bytes.fromhex("00010203040506070000000000000000")
KEYSTREAM_64 = bytes.fromhex(
    "9dc28337d6d3b4be380dd72d8e5cc1a51019c78180555051d98aa1bf0d3dae53"
    "1c14926380b4c69f681eae440df9f3e04fff1074becedc95f00032469b24f1c5"
)


def test_sbox_is_the_aes_sbox():
    assert SBOX[0x00] == 0x63
    assert SBOX[0x53] == 0xED
    assert SBOX[0xFF] == 0x16
    assert len(np.unique(SBOX)) == 256  # a permutation


def test_t_table_shape_and_values():
    tbl = t_table()
    assert tbl.shape == (5, 256)
    # Te0[x] = [2s, s, s, 3s] with s = S[x]: for x = 0, s = 0x63 -> 0xc6 63 63 a5
    assert int(tbl[0][0]) == 0xC66363A5
    # Te1 is Te0 rotated right by 8, Te3 by 24
    assert int(tbl[1][0]) == 0xA5C66363
    assert int(tbl[3][0]) == 0x6363A5C6
    assert np.array_equal(tbl[4], SBOX)


def test_key_schedule_matches_fips_197_round_keys():
    rk = aes128_expand_key_np(KEY_128)
    assert len(rk) == 44
    assert [int(x) for x in rk[:4]] == [0x00010203, 0x04050607, 0x08090A0B, 0x0C0D0E0F]
    # the round-1 key from FIPS-197's key expansion table; the last round key is the one that,
    # together with the block function, reproduces FIPS-197 C.1 (checked below)
    assert [int(x) for x in rk[4:8]] == [0xD6AA74FD, 0xD2AF72FA, 0xDAA678F1, 0xD6AB76FE]
    assert [int(x) for x in rk[40:44]] == [0x13111D7F, 0xE3944A17, 0xF307A78B, 0x4D2B30C5]


def test_fips_197_block_vector():
    assert aes128_encrypt_block_np(KEY_128, PLAIN_128) == CIPHER_128


def test_zero_key_zero_block_vector():
    """A second, independent vector: the classic all-zero case."""
    assert aes128_encrypt_block_np(bytes(16), bytes(16)).hex() == (
        "66e94bd4ef8a2c3b884cfa59ca342b2e"
    )


def test_ctr_keystream_matches_openssl():
    assert aes128_ctr_np(KEY_128, IV[:8], 64) == KEYSTREAM_64


def test_ctr_partial_block_and_counter_offset():
    full = aes128_ctr_np(KEY_128, IV[:8], 48)
    assert aes128_ctr_np(KEY_128, IV[:8], 20) == full[:20]
    # a counter offset shifts the stream by exactly one block
    assert aes128_ctr_np(KEY_128, IV[:8], 16, counter0=1) == full[16:32]


def test_ctr_rejects_bad_shapes():
    with pytest.raises(ValueError, match="16 bytes"):
        aes128_expand_key_np(b"\x00" * 15)
    with pytest.raises(ValueError, match="16 bytes"):
        aes128_encrypt_block_np(KEY_128, b"\x00" * 15)
    with pytest.raises(ValueError, match="8 bytes"):
        aes128_ctr_np(KEY_128, b"\x00" * 7, 16)
