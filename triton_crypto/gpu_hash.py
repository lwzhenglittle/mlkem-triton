"""ML-KEM's batched G / J / PRF helpers, on top of the generic sponge primitive.

Byte-for-byte identical to the `hashlib` version, but it keeps the whole batch on the
device instead of crossing to the host and back once per message. The sponge kernel and
its input padding account for 16.5% of device time on GB10 and 24.5% on Intel — see
docs/benchmarks.md.

The sponge itself is scheme-agnostic and lives in `triton_crypto.primitives.sponge`; what
is left here is the ML-KEM call pattern (which rate/domain each FIPS 203 function uses)
and CBD, which is ML-KEM-only sampling.
"""

from __future__ import annotations

import numpy as np

from .primitives.sponge import SHA3_256, SHA3_512, SHAKE256, hash_np_batch, pad_batch

RATE_SHA3_512 = SHA3_512.rate
RATE_SHA3_256 = SHA3_256.rate
RATE_SHAKE = SHAKE256.rate


def sha3_512_batch(msgs: np.ndarray, device=None) -> np.ndarray:
    """G: [n, L] -> [n, 64]."""
    return hash_np_batch(msgs, SHA3_512, device=device)


def sha3_256_batch(msgs: np.ndarray, device=None) -> np.ndarray:
    """H: [n, L] -> [n, 32]."""
    return hash_np_batch(msgs, SHA3_256, device=device)


def shake256_batch(msgs: np.ndarray, outlen: int = 32, device=None) -> np.ndarray:
    """J / PRF: [n, L] -> [n, outlen]."""
    assert outlen % 8 == 0
    return hash_np_batch(msgs, SHAKE256, outlen, device=device)


def cbd_batch(bufs: np.ndarray, eta: int) -> np.ndarray:
    """[n, 64·eta] -> [n, 256] centered binomial sampling (FIPS 203 Alg. 8, vectorized)."""
    bufs = np.ascontiguousarray(bufs, dtype=np.uint8)
    n = bufs.shape[0]
    assert bufs.shape[1] == 64 * eta
    bits = np.unpackbits(bufs, axis=1, bitorder="little").reshape(n, 256, 2 * eta)
    bits = bits.astype(np.int64)
    a = bits[:, :, :eta].sum(axis=2)
    b = bits[:, :, eta:].sum(axis=2)
    return a - b


__all__ = [
    "cbd_batch",
    "pad_batch",
    "sha3_256_batch",
    "sha3_512_batch",
    "shake256_batch",
]
