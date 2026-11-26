"""ChaCha20 (RFC 8439): the first non-PQC primitive in the library.

It is here because it is the cheapest complete ARX workload — 20 rounds of add / xor /
rotate on 32-bit words — and because a stream cipher is pure throughput, which is exactly
the measurement the benchmark harness needed a user for.

The 96-bit-nonce IETF construction is implemented (`counter(32) || nonce(96)`); the original
64-bit-nonce variant and XChaCha20's HChaCha20 step are not. Rotation is `(x << r) | (x >> (32-r))`
on `uint32`: on int32 the right shift is **arithmetic** and the result is silently wrong.
LLVM folds the idiom into a funnel shift, so NVIDIA gets one instruction; Xe has no rotate
and pays two to three (measured: rotate is 2.2x an add, see `docs/kernels.md`).

Layout: `keys` is `[B, 32]` or one 32-byte object, `nonces` is `[B, 12]` or one 12-byte
object, and a batch of 1 broadcasts. Output is `[B, nbytes] uint8`.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from ..devices import resolve_device, warps

__all__ = [
    "chacha20_keystream",
    "chacha20_keystream_np",
    "chacha20_keystream_torch",
    "chacha20_xor",
]

SIGMA = (0x61707865, 0x3320646E, 0x79622D32, 0x6B206574)  # "expand 32-byte k"
# Triton kernels may only read module globals that are `tl.constexpr`, hence the copies.
_C0 = tl.constexpr(SIGMA[0])
_C1 = tl.constexpr(SIGMA[1])
_C2 = tl.constexpr(SIGMA[2])
_C3 = tl.constexpr(SIGMA[3])
_BLOCK = 64
_KEY_WORDS = 8
_NONCE_WORDS = 3
_M32 = 0xFFFFFFFF


# --------------------------------------------------------------------------- kernels


@triton.jit
def _rotl(x, R: tl.constexpr):
    """Rotate-left of a 32-bit word. uint32 for the right shift: int32 would sign-extend."""
    u = x.to(tl.uint32)
    return ((u << R) | (u >> (32 - R))).to(tl.int32)


@triton.jit
def _qr(a, b, c, d):
    """The ChaCha quarter round (RFC 8439 §2.1)."""
    a += b
    d ^= a
    d = _rotl(d, 16)
    c += d
    b ^= c
    b = _rotl(b, 12)
    a += b
    d ^= a
    d = _rotl(d, 8)
    c += d
    b ^= c
    b = _rotl(b, 7)
    return a, b, c, d


@triton.jit
def chacha20_kernel(KEY, NON, CTR, OUT, B, NBLK, BR: tl.constexpr):
    """One program = `BR` streams x one 64-byte block; grid `(cdiv(B, BR), NBLK)`.

    | tensor | shape | dtype | meaning |
    |---|---|---|---|
    | `KEY` | `[B, 8]` | int32 | key words, little-endian |
    | `NON` | `[B, 3]` | int32 | 96-bit nonce words, little-endian |
    | `CTR` | `[B]` | int32 | counter of the stream's first block |
    | `OUT` | `[B, NBLK*16]` | int32 | keystream words (little-endian in memory) |
    """
    pid = tl.program_id(0)
    blk = tl.program_id(1)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < B

    # Sixteen scalars per lane, loaded one word at a time: a [BR, 16] tile cannot be sliced
    # by a constant index, so the round function works on separate vectors.
    x0 = tl.full((BR,), _C0, tl.int32)
    x1 = tl.full((BR,), _C1, tl.int32)
    x2 = tl.full((BR,), _C2, tl.int32)
    x3 = tl.full((BR,), _C3, tl.int32)
    x4 = tl.load(KEY + rows * 8 + 0, mask=m, other=0)
    x5 = tl.load(KEY + rows * 8 + 1, mask=m, other=0)
    x6 = tl.load(KEY + rows * 8 + 2, mask=m, other=0)
    x7 = tl.load(KEY + rows * 8 + 3, mask=m, other=0)
    x8 = tl.load(KEY + rows * 8 + 4, mask=m, other=0)
    x9 = tl.load(KEY + rows * 8 + 5, mask=m, other=0)
    x10 = tl.load(KEY + rows * 8 + 6, mask=m, other=0)
    x11 = tl.load(KEY + rows * 8 + 7, mask=m, other=0)
    x12 = tl.load(CTR + rows, mask=m, other=0) + blk
    x13 = tl.load(NON + rows * 3 + 0, mask=m, other=0)
    x14 = tl.load(NON + rows * 3 + 1, mask=m, other=0)
    x15 = tl.load(NON + rows * 3 + 2, mask=m, other=0)

    s0, s1, s2, s3 = x0, x1, x2, x3
    s4, s5, s6, s7 = x4, x5, x6, x7
    s8, s9, s10, s11 = x8, x9, x10, x11
    s12, s13, s14, s15 = x12, x13, x14, x15

    for _ in range(10):  # 20 rounds = 10 double rounds (RFC 8439 §2.3)
        x0, x4, x8, x12 = _qr(x0, x4, x8, x12)
        x1, x5, x9, x13 = _qr(x1, x5, x9, x13)
        x2, x6, x10, x14 = _qr(x2, x6, x10, x14)
        x3, x7, x11, x15 = _qr(x3, x7, x11, x15)
        x0, x5, x10, x15 = _qr(x0, x5, x10, x15)
        x1, x6, x11, x12 = _qr(x1, x6, x11, x12)
        x2, x7, x8, x13 = _qr(x2, x7, x8, x13)
        x3, x4, x9, x14 = _qr(x3, x4, x9, x14)

    x0 += s0
    x1 += s1
    x2 += s2
    x3 += s3
    x4 += s4
    x5 += s5
    x6 += s6
    x7 += s7
    x8 += s8
    x9 += s9
    x10 += s10
    x11 += s11
    x12 += s12
    x13 += s13
    x14 += s14
    x15 += s15

    # Four groups of four words, each assembled with two joins (see tests/test_chacha20.py:
    # `tl.join(tl.join(a, b), tl.join(c, d))` lays out as [a, b, c, d] when the offset uses
    # `2*last_axis + middle_axis`). Four 3-D stores beat sixteen strided ones.
    base = rows * (NBLK * 16) + blk * 16
    j = tl.arange(0, 2)
    off = base[:, None, None] + (j[None, None, :] * 2 + j[None, :, None])
    tl.store(
        OUT + off + 0,
        tl.join(tl.join(x0, x1), tl.join(x2, x3)),
        mask=m[:, None, None],
    )
    tl.store(
        OUT + off + 4,
        tl.join(tl.join(x4, x5), tl.join(x6, x7)),
        mask=m[:, None, None],
    )
    tl.store(
        OUT + off + 8,
        tl.join(tl.join(x8, x9), tl.join(x10, x11)),
        mask=m[:, None, None],
    )
    tl.store(
        OUT + off + 12,
        tl.join(tl.join(x12, x13), tl.join(x14, x15)),
        mask=m[:, None, None],
    )


@triton.jit
def _xor_kernel(A, B, OUT, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(OUT + i, tl.load(A + i, mask=m, other=0) ^ tl.load(B + i, mask=m, other=0), mask=m)


# --------------------------------------------------------------------------- numpy side


def _qr_py(a: int, b: int, c: int, d: int) -> tuple[int, int, int, int]:
    def rotl(v, r):
        v &= _M32
        return ((v << r) | (v >> (32 - r))) & _M32

    a = (a + b) & _M32
    d = rotl(d ^ a, 16)
    c = (c + d) & _M32
    b = rotl(b ^ c, 12)
    a = (a + b) & _M32
    d = rotl(d ^ a, 8)
    c = (c + d) & _M32
    b = rotl(b ^ c, 7)
    return a, b, c, d


def _block_py(key_words, nonce_words, counter: int) -> bytes:
    """One 64-byte keystream block, as bytes (little-endian words)."""
    x = list(SIGMA) + list(key_words) + [counter & _M32] + list(nonce_words)
    s = x.copy()
    for _ in range(10):
        for a, b, c, d in ((0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15)):
            x[a], x[b], x[c], x[d] = _qr_py(x[a], x[b], x[c], x[d])
        for a, b, c, d in ((0, 5, 10, 15), (1, 6, 11, 12), (2, 7, 8, 13), (3, 4, 9, 14)):
            x[a], x[b], x[c], x[d] = _qr_py(x[a], x[b], x[c], x[d])
    words = [(x[i] + s[i]) & _M32 for i in range(16)]
    return b"".join(w.to_bytes(4, "little") for w in words)


def _as_batch(x, size: int, name: str) -> np.ndarray:
    if isinstance(x, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(x), dtype=np.uint8).reshape(1, -1)
    else:
        arr = np.asarray(x, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != size:
        raise ValueError(f"{name} must be one {size}-byte value or a [B, {size}] array")
    return np.ascontiguousarray(arr)


def _batch_of(keys, nonces) -> int:
    k = keys.shape[0]
    n = nonces.shape[0]
    if k != n and k != 1 and n != 1:
        raise ValueError(f"batch mismatch: {k} keys vs {n} nonces")
    return max(k, n)


def _broadcast(arr: np.ndarray, batch: int) -> np.ndarray:
    return np.repeat(arr, batch, axis=0) if arr.shape[0] == 1 and batch > 1 else arr


def chacha20_keystream_np(keys, nonces, nbytes: int, counters=0) -> np.ndarray:
    """numpy reference: `[B, nbytes] uint8` keystream for each (key, nonce) pair."""
    keys = _as_batch(keys, 32, "key")
    nonces = _as_batch(nonces, 12, "nonce")
    batch = _batch_of(keys, nonces)
    keys, nonces = _broadcast(keys, batch), _broadcast(nonces, batch)
    counts = np.broadcast_to(np.asarray(counters, dtype=np.uint64), (batch,))
    blocks = -(-nbytes // _BLOCK)
    out = np.empty((batch, blocks * _BLOCK), dtype=np.uint8)
    for b in range(batch):
        kw = np.frombuffer(keys[b].tobytes(), dtype="<u4")
        nw = np.frombuffer(nonces[b].tobytes(), dtype="<u4")
        for i in range(blocks):
            block = _block_py(kw.tolist(), nw.tolist(), int(counts[b]) + i)
            out[b, i * _BLOCK : (i + 1) * _BLOCK] = np.frombuffer(block, dtype=np.uint8)
    return np.ascontiguousarray(out[:, :nbytes])


# --------------------------------------------------------------------------- device side


def _device_inputs(keys, nonces, counters, batch: int):
    """uint8 batches -> int32 word tensors (little-endian reinterpretation)."""
    kw = np.frombuffer(keys.tobytes(), dtype="<u4").reshape(batch, _KEY_WORDS)
    nw = np.frombuffer(nonces.tobytes(), dtype="<u4").reshape(batch, _NONCE_WORDS)
    cw = np.broadcast_to(np.asarray(counters, dtype=np.uint32), (batch,))
    return (
        np.ascontiguousarray(kw).astype(np.int32),
        np.ascontiguousarray(nw).astype(np.int32),
        np.ascontiguousarray(cw).astype(np.int32),
    )


def chacha20_keystream_torch(keys, nonces, nbytes: int, counters=0, device=None):
    """`[B, nbytes] uint8` keystream on the device (no host round trip).

    `keys` is 32 bytes (or `[B, 32]`), `nonces` 12 bytes (or `[B, 12]`); a batch of 1
    broadcasts against a longer one. `counters` is the first block counter of each stream.
    """
    import torch

    keys = _as_batch(keys, 32, "key")
    nonces = _as_batch(nonces, 12, "nonce")
    batch = _batch_of(keys, nonces)
    keys, nonces = _broadcast(keys, batch), _broadcast(nonces, batch)
    dev = resolve_device(device)
    if nbytes <= 0:
        return torch.empty((batch, 0), dtype=torch.uint8, device=dev)

    kw, nw, cw = _device_inputs(keys, nonces, counters, batch)
    blocks = -(-nbytes // _BLOCK)
    key_t = torch.from_numpy(kw).to(dev)
    non_t = torch.from_numpy(nw).to(dev)
    ctr_t = torch.from_numpy(cw).to(dev)
    out = torch.empty((batch, blocks * 16), dtype=torch.int32, device=dev)
    br = 64
    chacha20_kernel[(triton.cdiv(batch, br), blocks)](
        key_t, non_t, ctr_t, out, batch, blocks, br, num_warps=warps("chacha", 4)
    )
    raw = out.view(torch.uint8).view(batch, blocks * _BLOCK)
    if nbytes == blocks * _BLOCK:
        return raw
    return raw[:, :nbytes].contiguous()  # a truncated view would not be row-contiguous


def chacha20_keystream(keys, nonces, nbytes: int, counters=0, device=None) -> np.ndarray:
    """`[B, nbytes] uint8` keystream, computed on the device and copied back."""
    return chacha20_keystream_torch(keys, nonces, nbytes, counters, device).cpu().numpy()


def chacha20_xor(keys, nonces, data, counters=0, device=None) -> np.ndarray:
    """`data` (bytes, or `[B, L]` uint8) XORed with the keystream, on the device."""
    import torch

    if isinstance(data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(data), dtype=np.uint8)[None, :]
    else:
        arr = np.asarray(data, dtype=np.uint8)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError("data must be bytes or a [B, L] uint8 array")
    arr = np.ascontiguousarray(arr)

    keys_b = _as_batch(keys, 32, "key")
    nonces_b = _as_batch(nonces, 12, "nonce")
    batch = _batch_of(keys_b, nonces_b)
    if arr.shape[0] not in (1, batch):
        raise ValueError(f"data batch {arr.shape[0]} does not match {batch} stream(s)")
    keys_b = _broadcast(keys_b, batch)
    nonces_b = _broadcast(nonces_b, batch)
    arr = np.broadcast_to(arr, (batch, arr.shape[1]))

    ks = chacha20_keystream(keys_b, nonces_b, arr.shape[1], counters, device)
    dev = resolve_device(device)
    a = torch.from_numpy(np.array(arr, copy=True, order="C")).to(dev)
    b = torch.from_numpy(ks).to(dev)
    out = torch.empty_like(a)
    n = a.numel()
    _xor_kernel[(triton.cdiv(n, 4096),)](a, b, out, n, 4096)
    return out.cpu().numpy()
