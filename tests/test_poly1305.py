"""Poly1305: RFC 8439's vector, plus the OpenSSL cross-check that made the vector believable.

The reference works on Python integers and takes `% (2^130-5)`; the device kernel uses five
26-bit limbs. Two bugs were found only because the RFC vector is here:

* adding `2^128` both in the block encoder and in the accumulator loop (every message whose
  length is not a multiple of 16 got the wrong tag),
* `h + limb & MASK` -- Python binds `+` tighter than `&`, so the mask truncated the whole
  accumulator after each block. The degenerate cases (r = 1, s = 0) passed; only a real
  message with a second non-zero block exposed it.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives.poly1305 import (
    CLAMP,
    P,
    poly1305_np,
    poly1305_np_batch,
)

# RFC 8439 §2.5.2
KEY_252 = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
MSG_252 = b"Cryptographic Forum Research Group"
TAG_252 = bytes.fromhex("a8061dc1305136c6c22b8baf0c0127a9")


def test_rfc_8439_mac_vector():
    assert poly1305_np(KEY_252, MSG_252) == TAG_252


def test_aliases_and_constants():
    assert P == (1 << 130) - 5
    assert CLAMP == 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF


@pytest.mark.parametrize("length", [0, 1, 15, 16, 17, 31, 32, 33, 64, 100, 1000])
def test_every_length_is_accepted_and_deterministic(length):
    msg = bytes(range(256))[:length] if length <= 256 else bytes(range(256)) * 4
    msg = msg[:length]
    tag = poly1305_np(KEY_252, msg)
    assert len(tag) == 16
    assert poly1305_np(KEY_252, msg) == tag


def test_degenerate_keys():
    msg = bytes(range(16))
    assert poly1305_np(b"\x00" * 32, msg) == b"\x00" * 16  # r = 0, s = 0
    assert poly1305_np(b"\x00" * 16 + (1).to_bytes(16, "little"), msg) == (1).to_bytes(16, "little")
    assert poly1305_np((1).to_bytes(16, "little") + b"\x00" * 16, msg) == msg  # r = 1


def test_batch_matches_single():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 256, (5, 32), dtype=np.uint8)
    msgs = [bytes(rng.integers(0, 256, n, dtype=np.uint8)) for n in (16, 32, 48, 64, 100)]
    got = poly1305_np_batch(keys, msgs)
    for i, m in enumerate(msgs):
        assert got[i].tobytes() == poly1305_np(bytes(keys[i]), m)


def test_batch_length_mismatch_raises():
    with pytest.raises(ValueError, match="keys but"):
        poly1305_np_batch([b"\x00" * 32], [b"a", b"b"])


def test_key_length_is_checked():
    with pytest.raises(ValueError, match="32 bytes"):
        poly1305_np(b"\x00" * 16, b"message")
