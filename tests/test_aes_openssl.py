"""AES-128 against the `openssl` CLI, which is a different code path from FIPS-197's paper vector.

The block function already has two known-answer vectors (`tests/test_aes.py`). What this adds is
coverage of the parts a single vector cannot reach: 40 random (key, plaintext) pairs through the
block function, and CTR streams whose counter crosses a 32-bit boundary -- the one arithmetic
detail the T-table kernel could plausibly get wrong without any KAT noticing.

Shells out to `openssl enc`, so it is skipped when the binary is missing (it is a test-time
dependency only; nothing in the package needs it).
"""

from __future__ import annotations

import random
import shutil
import subprocess

import pytest

from triton_crypto.primitives.aes import aes128_ctr_np, aes128_encrypt_block_np

OPENSSL = shutil.which("openssl")
pytestmark = pytest.mark.skipif(OPENSSL is None, reason="openssl CLI not available")


def _openssl(argv: list[str], data: bytes) -> bytes:
    out = subprocess.run([OPENSSL, "enc", *argv], input=data, capture_output=True, check=True)
    return out.stdout


def _ecb(key: bytes, block: bytes) -> bytes:
    return _openssl(["-aes-128-ecb", "-K", key.hex(), "-nopad"], block)


def _ctr(key: bytes, iv: bytes, data: bytes) -> bytes:
    return _openssl(["-aes-128-ctr", "-K", key.hex(), "-iv", iv.hex()], data)


def test_the_fips_vector_through_the_cli():
    """Sanity-check the helper before trusting it on random inputs."""
    key = bytes(range(16))
    pt = bytes.fromhex("00112233445566778899aabbccddeeff")
    assert _ecb(key, pt).hex() == "69c4e0d86a7b0430d8cdb78070b4c55a"


def test_random_blocks():
    rng = random.Random(20261126)
    for _ in range(40):
        key = bytes(rng.getrandbits(8) for _ in range(16))
        block = bytes(rng.getrandbits(8) for _ in range(16))
        assert aes128_encrypt_block_np(key, block) == _ecb(key, block)


@pytest.mark.parametrize("nbytes", [1, 16, 17, 64, 100, 1000])
def test_ctr_stream(nbytes):
    rng = random.Random(nbytes)
    key = bytes(rng.getrandbits(8) for _ in range(16))
    nonce = bytes(rng.getrandbits(8) for _ in range(8))
    iv = nonce + bytes(8)  # openssl's -iv is the whole 16-byte block: nonce || counter
    zeros = bytes(nbytes)
    mine = aes128_ctr_np(key, nonce, nbytes)
    assert mine == _ctr(key, iv, zeros), f"{nbytes} bytes"


def test_ctr_counter_wraps_the_low_32_bits_correctly():
    """The counter is 64 bits; only its low half moves for any realistic stream. A stream long
    enough to cross 2^32 would be 64 GiB, so this checks the *arithmetic* by starting the
    counter near the boundary instead."""
    rng = random.Random(1)
    key = bytes(rng.getrandbits(8) for _ in range(16))
    nonce = bytes(rng.getrandbits(8) for _ in range(8))
    start = 0xFFFFFFFF
    mine = aes128_ctr_np(key, nonce, 48, counter0=start)
    iv = nonce + start.to_bytes(8, "big")
    assert mine == _ctr(key, iv, bytes(48))


def test_ctr_key_stream_matches_across_implementations_for_a_long_stream():
    rng = random.Random(2)
    key = bytes(rng.getrandbits(8) for _ in range(16))
    nonce = bytes(rng.getrandbits(8) for _ in range(8))
    iv = nonce + bytes(8)
    assert aes128_ctr_np(key, nonce, 4096) == _ctr(key, iv, bytes(4096))
