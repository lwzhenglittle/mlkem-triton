"""CPU-side FIPS 203 sampling and hashing (v1: hashlib, single-threaded,
correctness only)."""

from __future__ import annotations

import hashlib

import numpy as np


def H(b: bytes) -> bytes:
    return hashlib.sha3_256(b).digest()


def G(b: bytes):
    d = hashlib.sha3_512(b).digest()
    return d[:32], d[32:]


def J(b: bytes) -> bytes:
    return hashlib.shake_256(b).digest(32)


def prf(eta: int, s: bytes, b: int) -> bytes:
    """PRF_eta(s, b) = SHAKE256(s ‖ b, 64·eta)."""
    return hashlib.shake_256(s + bytes([b])).digest(64 * eta)


def cbd(buf: bytes, eta: int) -> np.ndarray:
    """SamplePolyCBD_eta (FIPS 203 Algorithm 8)."""
    bits = np.unpackbits(np.frombuffer(buf, dtype=np.uint8), bitorder="little")
    assert bits.size == 256 * 2 * eta
    bits = bits.reshape(256, 2 * eta).astype(np.int64)
    a = bits[:, :eta].sum(axis=1)
    b = bits[:, eta:].sum(axis=1)
    return a - b


__all__ = ["G", "H", "J", "cbd", "prf"]
