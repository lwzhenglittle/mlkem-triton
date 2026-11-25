"""GPU batched hashing: one launch handles the whole batch for the FIPS 203 G / J / PRF.

Byte-for-byte identical to the `hashlib` version, but it keeps the whole batch on the device
instead of crossing to the host and back once per message. The sponge kernel and its input
padding account for 16.5% of device time on GB10 and 24.5% on Intel - see docs/benchmarks.md.
"""

from __future__ import annotations

import numpy as np
import torch
import triton

from .devices import resolve_device
from .kernels.shake import SHA3_DOMAIN, SHAKE_DOMAIN, get_sponge

RATE_SHA3_512 = 72
RATE_SHA3_256 = 136
RATE_SHAKE = 136
DEFAULT_BLOCK = 128
DEFAULT_WARPS = 4


def pad_batch(msgs: np.ndarray, rate: int, domain: int) -> np.ndarray:
    """Vectorized pad10*1 + domain separation. msgs [n, L] uint8 ->
    [n, ceil((L+1)/rate)*rate]."""
    msgs = np.ascontiguousarray(msgs, dtype=np.uint8)
    n, L = msgs.shape
    total = ((L + 1 + rate - 1) // rate) * rate
    out = np.zeros((n, total), dtype=np.uint8)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, -1] |= 0x80
    return out


def _sponge_batch(
    padded: np.ndarray,
    rate: int,
    outlen: int,
    block: int = DEFAULT_BLOCK,
    warps: int = DEFAULT_WARPS,
    device=None,
) -> np.ndarray:
    device = resolve_device(device)
    n, tot = padded.shape
    nblocks = tot // rate
    assert nblocks * rate == tot
    x = torch.from_numpy(padded.copy()).view(torch.int64).to(device)
    out = torch.empty((n, outlen // 8), dtype=torch.int64, device=device)
    kern = get_sponge(rate, outlen)
    grid = (triton.cdiv(n, block),)
    kern[grid](
        x, out, n, nblocks, nblocks * rate // 8, OUTW=outlen // 8, BLOCK=block, num_warps=warps
    )
    return out.cpu().numpy().view(np.uint8).reshape(n, outlen)


def sha3_512_batch(msgs: np.ndarray, device=None) -> np.ndarray:
    """G: [n, L] -> [n, 64]."""
    return _sponge_batch(
        pad_batch(msgs, RATE_SHA3_512, SHA3_DOMAIN), RATE_SHA3_512, 64, device=device
    )


def sha3_256_batch(msgs: np.ndarray, device=None) -> np.ndarray:
    """H: [n, L] -> [n, 32]."""
    return _sponge_batch(
        pad_batch(msgs, RATE_SHA3_256, SHA3_DOMAIN), RATE_SHA3_256, 32, device=device
    )


def shake256_batch(msgs: np.ndarray, outlen: int = 32, device=None) -> np.ndarray:
    """J / PRF: [n, L] -> [n, outlen], outlen <= 136 and a multiple of 8."""
    assert outlen % 8 == 0
    return _sponge_batch(
        pad_batch(msgs, RATE_SHAKE, SHAKE_DOMAIN), RATE_SHAKE, outlen, device=device
    )


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
