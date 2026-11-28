"""SM3 cryptographic hash (GB/T 32905-2016 / ISO/IEC 10118-3): numpy reference and Triton GPU kernel.

SM3 is a 256-bit Merkle-Damgard hash with 64-byte blocks and 64 rounds per block. Its structure
mirrors SHA-256 (32-bit big-endian words, 8-word state, 16-word sliding message schedule), with
three differences that matter for GPU execution:

1. No widespread x86 hardware instruction (unlike SHA-NI for SHA-256), so a rolled 16-word GPU
   schedule wins decisively against 16 CPU processes.
2. Two distinct round regions (`j = 0..15` linear XOR `FF`/`GG`, `j = 16..63` majority/choose),
   implemented as two rolled loops with zero branches inside the round body.
3. Davies-Meyer feed-forward is XOR (`V_i ^ state`), not modular addition.

Validated against GB/T 32905-2016 standard test vectors and OpenSSL/hashlib (`hashlib.new("sm3")`).
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from .. import devices as _devices
from .._inputs import as_batch_bytes

__all__ = [
    "IV_SM3",
    "T_SM3",
    "sm3_compress_np",
    "sm3_hash_np",
    "sm3_hash_np_batch",
    "sm3_hash_torch",
    "sm3_pad_torch",
]

IV_SM3: tuple[int, ...] = (
    0x7380166F,
    0x4914B2B9,
    0x172442D7,
    0xDA8A0600,
    0xA96F30BC,
    0x163138AA,
    0xE38DEE4D,
    0xB0FB0E4E,
)

# Pre-rotated round constants T_j <<< (j mod 32) for j = 0..63.
T_SM3: tuple[int, ...] = tuple(
    (
        ((0x79CC4519 if j < 16 else 0x7A879D8A) << (j & 31))
        | ((0x79CC4519 if j < 16 else 0x7A879D8A) >> ((32 - (j & 31)) & 31))
    )
    & 0xFFFFFFFF
    for j in range(64)
)


def _rotl32(x: int, n: int) -> int:
    n &= 31
    if n == 0:
        return x & 0xFFFFFFFF
    return ((x << n) | (x >> (32 - n))) & 0xFFFFFFFF


def _p0(x: int) -> int:
    return x ^ _rotl32(x, 9) ^ _rotl32(x, 17)


def _p1(x: int) -> int:
    return x ^ _rotl32(x, 15) ^ _rotl32(x, 23)


def sm3_compress_np(state: list[int], blocks: bytes) -> list[int]:
    """Compress one or more 64-byte blocks with SM3 (plain Python oracle)."""
    if len(blocks) % 64 != 0:
        raise ValueError(f"blocks length {len(blocks)} is not a multiple of 64")
    v = [x & 0xFFFFFFFF for x in state]
    for off in range(0, len(blocks), 64):
        chunk = blocks[off : off + 64]
        w = [int.from_bytes(chunk[4 * i : 4 * (i + 1)], "big") for i in range(16)]
        for j in range(16, 68):
            w.append(
                _p1(w[j - 16] ^ w[j - 9] ^ _rotl32(w[j - 3], 15)) ^ _rotl32(w[j - 13], 7) ^ w[j - 6]
            )
        a, b, c, d, e, f, g, h = v
        for j in range(64):
            wp = w[j] ^ w[j + 4]
            a12 = _rotl32(a, 12)
            ss1 = _rotl32((a12 + e + T_SM3[j]) & 0xFFFFFFFF, 7)
            ss2 = ss1 ^ a12
            if j < 16:
                ff = a ^ b ^ c
                gg = e ^ f ^ g
            else:
                ff = (a & b) | (a & c) | (b & c)
                gg = (e & f) | ((~e & 0xFFFFFFFF) & g)
            tt1 = (ff + d + ss2 + wp) & 0xFFFFFFFF
            tt2 = (gg + h + ss1 + w[j]) & 0xFFFFFFFF
            d = c
            c = _rotl32(b, 9)
            b = a
            a = tt1
            h = g
            g = _rotl32(f, 19)
            f = e
            e = _p0(tt2)
        v = [x ^ y for x, y in zip(v, (a, b, c, d, e, f, g, h), strict=True)]
    return v


def sm3_hash_np(msg: bytes) -> bytes:
    """Single-message SM3 digest (GB/T 32905-2016)."""
    data = bytearray(msg)
    data.append(0x80)
    while len(data) % 64 != 56:
        data.append(0)
    data += (8 * len(msg)).to_bytes(8, "big")
    state = sm3_compress_np(list(IV_SM3), bytes(data))
    return b"".join(w.to_bytes(4, "big") for w in state)


def sm3_hash_np_batch(msgs) -> np.ndarray:
    """Batched SM3 reference: returns `[B, 32] uint8`."""
    arr = as_batch_bytes(msgs)
    b = arr.shape[0]
    out = np.empty((b, 32), dtype=np.uint8)
    for i in range(b):
        out[i] = np.frombuffer(sm3_hash_np(bytes(arr[i])), dtype=np.uint8)
    return out


# --------------------------------------------------------------------------- Triton GPU kernel


@triton.jit
def _bs32(x):
    x = x.to(tl.uint32)
    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | ((x >> 24) & 0xFF)


@triton.jit
def _rotl(x, n):
    x = x.to(tl.uint32)
    return (x << n) | (x >> (32 - n))


@triton.jit
def _sm3_kernel(
    KT,
    MSG,
    OUT,
    rows,
    NB,
    NW: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Rolled 16-word sliding-window SM3 compression kernel.

    `MSG` is `[rows, NW]` int32 (little-endian host view of pre-padded big-endian SM3 blocks,
    `NW = NB * 16`); `OUT` is `[rows, 8]` int32 (big-endian digest bytes viewed as int32).
    """
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows
    z = tl.zeros((BLOCK,), dtype=tl.uint32)

    h0 = z | tl.full((), 0x7380166F, tl.uint32)
    h1 = z | tl.full((), 0x4914B2B9, tl.uint32)
    h2 = z | tl.full((), 0x172442D7, tl.uint32)
    h3 = z | tl.full((), 0xDA8A0600, tl.uint32)
    h4 = z | tl.full((), 0xA96F30BC, tl.uint32)
    h5 = z | tl.full((), 0x163138AA, tl.uint32)
    h6 = z | tl.full((), 0xE38DEE4D, tl.uint32)
    h7 = z | tl.full((), 0xB0FB0E4E, tl.uint32)

    for blk in range(0, NB, 1):
        a = h0
        b = h1
        c = h2
        d = h3
        e = h4
        f = h5
        g = h6
        h = h7

        base = row * NW + blk * 16
        w0 = _bs32(tl.load(MSG + base + 0, mask=m, other=0).to(tl.uint32, bitcast=True))
        w1 = _bs32(tl.load(MSG + base + 1, mask=m, other=0).to(tl.uint32, bitcast=True))
        w2 = _bs32(tl.load(MSG + base + 2, mask=m, other=0).to(tl.uint32, bitcast=True))
        w3 = _bs32(tl.load(MSG + base + 3, mask=m, other=0).to(tl.uint32, bitcast=True))
        w4 = _bs32(tl.load(MSG + base + 4, mask=m, other=0).to(tl.uint32, bitcast=True))
        w5 = _bs32(tl.load(MSG + base + 5, mask=m, other=0).to(tl.uint32, bitcast=True))
        w6 = _bs32(tl.load(MSG + base + 6, mask=m, other=0).to(tl.uint32, bitcast=True))
        w7 = _bs32(tl.load(MSG + base + 7, mask=m, other=0).to(tl.uint32, bitcast=True))
        w8 = _bs32(tl.load(MSG + base + 8, mask=m, other=0).to(tl.uint32, bitcast=True))
        w9 = _bs32(tl.load(MSG + base + 9, mask=m, other=0).to(tl.uint32, bitcast=True))
        w10 = _bs32(tl.load(MSG + base + 10, mask=m, other=0).to(tl.uint32, bitcast=True))
        w11 = _bs32(tl.load(MSG + base + 11, mask=m, other=0).to(tl.uint32, bitcast=True))
        w12 = _bs32(tl.load(MSG + base + 12, mask=m, other=0).to(tl.uint32, bitcast=True))
        w13 = _bs32(tl.load(MSG + base + 13, mask=m, other=0).to(tl.uint32, bitcast=True))
        w14 = _bs32(tl.load(MSG + base + 14, mask=m, other=0).to(tl.uint32, bitcast=True))
        w15 = _bs32(tl.load(MSG + base + 15, mask=m, other=0).to(tl.uint32, bitcast=True))

        # Rounds 0..15: FF = a^b^c, GG = e^f^g
        for t in range(0, 16, 1):
            kt = tl.load(KT + t).to(tl.uint32, bitcast=True)
            a12 = _rotl(a, 12)
            ss1 = _rotl(a12 + e + kt, 7)
            ss2 = ss1 ^ a12
            tt1 = (a ^ b ^ c) + d + ss2 + (w0 ^ w4)
            tt2 = (e ^ f ^ g) + h + ss1 + w0
            d = c
            c = _rotl(b, 9)
            b = a
            a = tt1
            h = g
            g = _rotl(f, 19)
            f = e
            e = tt2 ^ _rotl(tt2, 9) ^ _rotl(tt2, 17)

            x = w0 ^ w7 ^ _rotl(w13, 15)
            nw = (x ^ _rotl(x, 15) ^ _rotl(x, 23)) ^ _rotl(w3, 7) ^ w10
            w0 = w1
            w1 = w2
            w2 = w3
            w3 = w4
            w4 = w5
            w5 = w6
            w6 = w7
            w7 = w8
            w8 = w9
            w9 = w10
            w10 = w11
            w11 = w12
            w12 = w13
            w13 = w14
            w14 = w15
            w15 = nw

        # Rounds 16..63: FF = maj(a,b,c), GG = ch(e,f,g)
        for t in range(16, 64, 1):
            kt = tl.load(KT + t).to(tl.uint32, bitcast=True)
            a12 = _rotl(a, 12)
            ss1 = _rotl(a12 + e + kt, 7)
            ss2 = ss1 ^ a12
            tt1 = ((a & b) | (a & c) | (b & c)) + d + ss2 + (w0 ^ w4)
            tt2 = ((e & f) | (~e & g)) + h + ss1 + w0
            d = c
            c = _rotl(b, 9)
            b = a
            a = tt1
            h = g
            g = _rotl(f, 19)
            f = e
            e = tt2 ^ _rotl(tt2, 9) ^ _rotl(tt2, 17)

            x = w0 ^ w7 ^ _rotl(w13, 15)
            nw = (x ^ _rotl(x, 15) ^ _rotl(x, 23)) ^ _rotl(w3, 7) ^ w10
            w0 = w1
            w1 = w2
            w2 = w3
            w3 = w4
            w4 = w5
            w5 = w6
            w6 = w7
            w7 = w8
            w8 = w9
            w9 = w10
            w10 = w11
            w11 = w12
            w12 = w13
            w13 = w14
            w14 = w15
            w15 = nw

        h0 = h0 ^ a
        h1 = h1 ^ b
        h2 = h2 ^ c
        h3 = h3 ^ d
        h4 = h4 ^ e
        h5 = h5 ^ f
        h6 = h6 ^ g
        h7 = h7 ^ h

    ob = row * 8
    tl.store(OUT + ob + 0, _bs32(h0).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 1, _bs32(h1).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 2, _bs32(h2).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 3, _bs32(h3).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 4, _bs32(h4).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 5, _bs32(h5).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 6, _bs32(h6).to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 7, _bs32(h7).to(tl.int32, bitcast=True), mask=m)


_T_CACHE: dict = {}


def _t_tensor(device):
    t = _T_CACHE.get(device)
    if t is None:
        import torch

        signed = [v - (1 << 32) if v >= (1 << 31) else v for v in T_SM3]
        t = torch.tensor(signed, dtype=torch.int32, device=device)
        _T_CACHE[device] = t
    return t


def sm3_pad_torch(msgs):
    """Pad `[B, L] uint8` device tensor to SM3 64-byte blocks `[B, NB * 64] uint8`."""
    import torch

    if msgs.ndim != 2 or msgs.dtype != torch.uint8:
        raise ValueError(
            f"msgs must be 2-D uint8, got shape={tuple(msgs.shape)} dtype={msgs.dtype}"
        )
    b, mlen = msgs.shape
    nblocks = (mlen + 9 + 63) // 64
    plen = nblocks * 64
    padded = torch.zeros((b, plen), dtype=torch.uint8, device=msgs.device)
    if b == 0:
        return padded
    if mlen > 0:
        padded[:, :mlen] = msgs
    padded[:, mlen] = 0x80
    bitlen = mlen * 8
    len_bytes = torch.tensor(list(bitlen.to_bytes(8, "big")), dtype=torch.uint8, device=msgs.device)
    padded[:, plen - 8 : plen] = len_bytes
    return padded


def sm3_hash_torch(msgs, device=None, pre_padded: bool = False, block: int = 128):
    """Batched SM3 on GPU: `[B, L] uint8` (or pre-padded `[B, NB*64] uint8`) -> `[B, 32] uint8`."""
    import torch

    dev = _devices.resolve_device(device)
    if isinstance(msgs, torch.Tensor):
        t = msgs.to(device=dev, dtype=torch.uint8).contiguous()
    else:
        arr = np.array(as_batch_bytes(msgs), dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).to(dev)
    if t.ndim != 2:
        raise ValueError(f"msgs must be 2-D, got shape {tuple(t.shape)}")
    b = t.shape[0]
    if b == 0:
        return torch.empty((0, 32), dtype=torch.uint8, device=dev)
    if pre_padded:
        if t.shape[1] == 0 or t.shape[1] % 64 != 0:
            raise ValueError(
                f"pre-padded length must be a positive multiple of 64, got {t.shape[1]}"
            )
        padded = t
    else:
        padded = sm3_pad_torch(t)
    nblocks = padded.shape[1] // 64
    msg_words = padded.view(torch.int32)
    out_words = torch.empty((b, 8), dtype=torch.int32, device=dev)
    _sm3_kernel[(triton.cdiv(b, block),)](
        _t_tensor(dev),
        msg_words,
        out_words,
        b,
        nblocks,
        NW=nblocks * 16,
        BLOCK=block,
        num_warps=_devices.warps("sm3", 4),
    )
    return out_words.view(torch.uint8)
