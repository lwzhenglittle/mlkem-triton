"""Poly1305 against OpenSSL's `EVP_MAC`, for lengths the RFC vectors do not cover.

The RFC 8439 vector is 34 bytes; the interesting cases are the block boundaries (15/16/17,
31/32/33) and the long tail, and there is no reason to trust a hand-written reference on them.
OpenSSL 3.5's POLY1305 is an independent implementation with AES-NI/PCLMUL-era tuning, and the
`EVP_MAC` API is bound here with ctypes so the check adds no dependency and no build step.

Skipped when `libcrypto.so.3` is missing, like the liboqs cross-checks.
"""

from __future__ import annotations

import ctypes
import random

import pytest

from triton_crypto.primitives.poly1305 import CLAMP, poly1305_np


def _load():
    for name in ("libcrypto.so.3", "libcrypto.so", "libcrypto.dylib"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    pytest.skip("no libcrypto available for the EVP_MAC cross-check")


class EVPPoly1305:
    """`EVP_MAC_fetch("POLY1305")` + init/update/final, one context reused per key."""

    def __init__(self):
        lib = _load()
        lib.EVP_MAC_fetch.restype = ctypes.c_void_p
        lib.EVP_MAC_fetch.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.EVP_MAC_CTX_new.restype = ctypes.c_void_p
        lib.EVP_MAC_CTX_new.argtypes = [ctypes.c_void_p]
        lib.EVP_MAC_init.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_void_p,
        ]
        lib.EVP_MAC_update.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.EVP_MAC_final.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_size_t,
        ]
        self.lib = lib
        self.mac = lib.EVP_MAC_fetch(None, b"POLY1305", None)
        if not self.mac:
            pytest.skip("libcrypto has no POLY1305 MAC")
        self.ctx = lib.EVP_MAC_CTX_new(self.mac)
        if not self.ctx:
            pytest.skip("EVP_MAC_CTX_new failed")

    def __call__(self, key: bytes, msg: bytes) -> bytes:
        lib = self.lib
        kbuf = (ctypes.c_ubyte * len(key)).from_buffer_copy(key)
        mbuf = (ctypes.c_ubyte * max(len(msg), 1)).from_buffer_copy(msg or b"\x00")
        out = (ctypes.c_ubyte * 16)()
        outlen = ctypes.c_size_t(16)
        if lib.EVP_MAC_init(self.ctx, kbuf, len(key), None) != 1:
            raise RuntimeError("EVP_MAC_init failed")
        if msg and lib.EVP_MAC_update(self.ctx, mbuf, len(msg)) != 1:
            raise RuntimeError("EVP_MAC_update failed")
        if lib.EVP_MAC_final(self.ctx, out, ctypes.byref(outlen), 16) != 1:
            raise RuntimeError("EVP_MAC_final failed")
        return bytes(out[: outlen.value])


@pytest.fixture(scope="module")
def evp():
    return EVPPoly1305()


def test_the_rfc_vector_through_openssl(evp):
    """Sanity-check the binding itself before trusting it on random inputs."""
    key = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
    assert evp(key, b"Cryptographic Forum Research Group").hex() == (
        "a8061dc1305136c6c22b8baf0c0127a9"
    )


@pytest.mark.parametrize(
    "length",
    [
        0,
        1,
        15,
        16,
        17,
        31,
        32,
        33,
        47,
        48,
        49,
        63,
        64,
        65,
        127,
        128,
        129,
        255,
        256,
        1023,
        1024,
        1025,
        4096,
    ],
)
def test_every_block_boundary(evp, length):
    rng = random.Random(length)
    key = bytes(rng.getrandbits(8) for _ in range(32))
    msg = bytes(rng.getrandbits(8) for _ in range(length))
    assert poly1305_np(key, msg) == evp(key, msg), f"length {length}"


def test_random_keys_and_lengths(evp):
    rng = random.Random(20261126)
    for _ in range(40):
        key = bytes(rng.getrandbits(8) for _ in range(32))
        msg = bytes(rng.getrandbits(8) for _ in range(rng.randrange(0, 600)))
        assert poly1305_np(key, msg) == evp(key, msg)


@pytest.mark.parametrize(
    ("key_byte", "msg_byte", "length"),
    [
        (0x00, 0x00, 16),  # the degenerate all-zero case: r = 0, so the tag is s alone
        (0x00, 0x00, 1),
        (0xFF, 0xFF, 16),  # the largest clamped r, and a message with every bit set
        (0xFF, 0xFF, 17),  # ... crossing the block boundary, where 2^128 is added
        (0xFF, 0xFF, 32),
        (0x00, 0xFF, 33),
    ],
)
def test_degenerate_inputs_against_openssl(evp, key_byte, msg_byte, length):
    """The cases a random corpus reaches with probability ~0: `r = 0` makes the accumulator
    constant, and an all-ones message exercises every carry in the 130-bit reduction."""
    key = bytes([key_byte]) * 32
    msg = bytes([msg_byte]) * length
    assert poly1305_np(key, msg) == evp(key, msg), (key_byte, msg_byte, length)


def test_clamp_is_the_rfc_mask():
    """A key whose `r` is all-ones must behave as the clamped `r`, not the raw one."""
    rng = random.Random(7)
    key = (b"\xff" * 16) + bytes(rng.getrandbits(8) for _ in range(16))
    msg = bytes(rng.getrandbits(8) for _ in range(16))
    clamped = (int.from_bytes(b"\xff" * 16, "little") & CLAMP).to_bytes(16, "little")
    other = clamped + key[16:]
    assert poly1305_np(key, msg) == poly1305_np(other, msg)
