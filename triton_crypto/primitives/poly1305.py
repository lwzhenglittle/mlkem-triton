"""Poly1305 (RFC 8439 §2.5) and the ChaCha20-Poly1305 AEAD.

Poly1305 is the one small-integer primitive where the arithmetic itself is the story: 130-bit
modular multiplication by a secret 128-bit `r`, 16 bytes at a time. The device kernel uses the
canonical **5 x 26-bit limb** layout (poly1305-donna): a 26x26-bit product is 52 bits, five
such products summed stay below 2^58, and int64 carries the whole thing without a `umulhi`
pair per multiply. A 3 x 44-bit layout would use fewer multiplies but its 88-bit intermediate
does not fit an int64, which is why nobody ships it on GPUs.

Bit-width detail that matters: `2^130 = 5 (mod 2^130-5)`, so the fifth limb's carry wraps with
a factor of 5. The kernel does that explicitly; the numpy reference does not need it, because
it works on Python integers and just takes `% (2^130 - 5)`.

The AEAD side is a composition, not new arithmetic:

    poly key   = ChaCha20(block counter 0) bytes 0..31
    ciphertext = plaintext XOR ChaCha20(block counter 1...)
    tag        = Poly1305(poly key, ct || pad16 || le64(0) || le64(len(ct)))

Only the no-AAD case is implemented, and only lengths that are a multiple of 16 -- the second
one is a real limit, not a simplification to be hidden: `mac_data` needs a pad16 after the
ciphertext, and the bench sizes are all multiples of 16. `aes128`-style arbitrary lengths need
one more padding kernel.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

__all__ = [
    "POLY1305_TAG",
    "P",
    "chacha20_poly1305_open_torch",
    "chacha20_poly1305_torch",
    "poly1305_np",
    "poly1305_np_batch",
    "poly1305_torch",
]

P = (1 << 130) - 5
CLAMP = 0x0FFFFFFC0FFFFFFC0FFFFFFC0FFFFFFF  # r is clamped: top bits of each 32-bit word
BLOCK = 16
LIMBS = 5
MASK26 = (1 << 26) - 1
# Triton kernels may only read module globals that are constexpr; the plain ints above are
# for numpy and for tests, these two are what the kernel sees.
_MASK26 = tl.constexpr((1 << 26) - 1)
_CLAMP0 = tl.constexpr(0x0FFFFFFF)
_CLAMP = tl.constexpr(0x0FFFFFFC)
POLY1305_TAG = 16


# --------------------------------------------------------------------------- references


def _blocks(msg: bytes):
    """The message as Poly1305 integers: `m + 2^128` for a full block, and for a partial tail
    the 0x01 byte *is* the high bit -- `m || 0x01 || 0*` with no extra 2^128. Adding both is
    the classic off-by-2^128, and it is wrong for every message whose length is not a
    multiple of 16."""
    out = []
    for i in range(0, len(msg), BLOCK):
        block = msg[i : i + BLOCK]
        if len(block) == BLOCK:
            out.append(int.from_bytes(block, "little") + (1 << 128))
        else:
            padded = block + b"\x01" + b"\x00" * (BLOCK - len(block) - 1)
            out.append(int.from_bytes(padded, "little"))
    return out


def poly1305_np(key: bytes, msg: bytes) -> bytes:
    """One-shot MAC on Python integers: the definition, deliberately without limb tricks."""
    if len(key) != 32:
        raise ValueError(f"Poly1305 key must be 32 bytes, got {len(key)}")
    r = int.from_bytes(key[:16], "little") & CLAMP
    s = int.from_bytes(key[16:], "little")
    acc = 0
    for block in _blocks(msg):
        acc = (acc + block) * r % P  # `_blocks`already carries the 2^128 term
    return ((acc + s) & ((1 << 128) - 1)).to_bytes(16, "little")


def poly1305_np_batch(keys, msgs) -> np.ndarray:
    """`[B, 32]` keys x `B` messages -> `[B, 16]` tags."""
    if len(keys) != len(msgs):
        raise ValueError(f"{len(keys)} keys but {len(msgs)} messages")
    return (
        np.stack(
            [
                np.frombuffer(poly1305_np(bytes(k), m), dtype=np.uint8)
                for k, m in zip(keys, msgs, strict=True)
            ]
        )
        if msgs
        else np.zeros((0, 16), np.uint8)
    )


# --------------------------------------------------------------------------- device


@triton.jit
def _poly1305_kernel(KEY, MSG, TAG, B, NBLK: tl.constexpr, BLK: tl.constexpr):
    """One lane per (key, message). `MSG` is `[B, NBLK*16]` uint8, already padded.

    Five 26-bit accumulators and five 26-bit multipliers, updated once per 16-byte block.
    """
    pid = tl.program_id(0)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B

    r = ()
    for i in tl.static_range(4):
        w = tl.load(KEY + lane * 32 + i * 4, mask=m, other=0).to(tl.int64)
        w = w | (tl.load(KEY + lane * 32 + i * 4 + 1, mask=m, other=0).to(tl.int64) << 8)
        w = w | (tl.load(KEY + lane * 32 + i * 4 + 2, mask=m, other=0).to(tl.int64) << 16)
        w = w | (tl.load(KEY + lane * 32 + i * 4 + 3, mask=m, other=0).to(tl.int64) << 24)
        r = r + (w,)
    # clamp per 32-bit word, then split into 26-bit limbs
    c0 = r[0] & _CLAMP0
    c1 = r[1] & _CLAMP
    c2 = r[2] & _CLAMP
    c3 = r[3] & _CLAMP
    r0 = c0 & _MASK26
    r1 = ((c0 >> 26) | (c1 << 6)) & _MASK26
    r2 = ((c1 >> 20) | (c2 << 12)) & _MASK26
    r3 = ((c2 >> 14) | (c3 << 18)) & _MASK26
    r4 = (c3 >> 8) & _MASK26
    r1_5 = 5 * r1
    r2_5 = 5 * r2
    r3_5 = 5 * r3
    r4_5 = 5 * r4

    h0 = tl.zeros_like(lane).to(tl.int64)
    h1 = tl.zeros_like(lane).to(tl.int64)
    h2 = tl.zeros_like(lane).to(tl.int64)
    h3 = tl.zeros_like(lane).to(tl.int64)
    h4 = tl.zeros_like(lane).to(tl.int64)

    # A *dynamic* loop, not `static_range`: unrolling it made the kernel's IR grow with the
    # message length and a 1 KiB message took over ten minutes to compile.
    for blk_idx in range(NBLK):
        base = lane * (NBLK * 16) + blk_idx * 16
        w0 = tl.load(MSG + base + 0, mask=m, other=0).to(tl.int64)
        w1 = tl.load(MSG + base + 1, mask=m, other=0).to(tl.int64)
        w2 = tl.load(MSG + base + 2, mask=m, other=0).to(tl.int64)
        w3 = tl.load(MSG + base + 3, mask=m, other=0).to(tl.int64)
        w4 = tl.load(MSG + base + 4, mask=m, other=0).to(tl.int64)
        w5 = tl.load(MSG + base + 5, mask=m, other=0).to(tl.int64)
        w6 = tl.load(MSG + base + 6, mask=m, other=0).to(tl.int64)
        w7 = tl.load(MSG + base + 7, mask=m, other=0).to(tl.int64)
        w8 = tl.load(MSG + base + 8, mask=m, other=0).to(tl.int64)
        w9 = tl.load(MSG + base + 9, mask=m, other=0).to(tl.int64)
        w10 = tl.load(MSG + base + 10, mask=m, other=0).to(tl.int64)
        w11 = tl.load(MSG + base + 11, mask=m, other=0).to(tl.int64)
        w12 = tl.load(MSG + base + 12, mask=m, other=0).to(tl.int64)
        w13 = tl.load(MSG + base + 13, mask=m, other=0).to(tl.int64)
        w14 = tl.load(MSG + base + 14, mask=m, other=0).to(tl.int64)
        w15 = tl.load(MSG + base + 15, mask=m, other=0).to(tl.int64)
        # The block as five 26-bit limbs, then + 2^128 (the implicit high bit). Every mask is
        # parenthesised *inside* the addition: Python binds `+` tighter than `&`, so
        # `h + limb & MASK` means `(h + limb) & MASK` and silently truncates the accumulator
        # to 26 bits after every block. Batching a mask into an add is a trap.
        h0 = h0 + ((w0 | (w1 << 8) | (w2 << 16) | (w3 << 24)) & _MASK26)
        h1 = h1 + (((w3 >> 2) | (w4 << 6) | (w5 << 14) | (w6 << 22)) & _MASK26)
        h2 = h2 + (((w6 >> 4) | (w7 << 4) | (w8 << 12) | (w9 << 20)) & _MASK26)
        h3 = h3 + (((w9 >> 6) | (w10 << 2) | (w11 << 10) | (w12 << 18)) & _MASK26)
        h4 = h4 + ((((w12 >> 8) | (w13 << 0) | (w14 << 8) | (w15 << 16)) & _MASK26) + (1 << 24))

        d0 = h0 * r0 + h1 * r4_5 + h2 * r3_5 + h3 * r2_5 + h4 * r1_5
        d1 = h0 * r1 + h1 * r0 + h2 * r4_5 + h3 * r3_5 + h4 * r2_5
        d2 = h0 * r2 + h1 * r1 + h2 * r0 + h3 * r4_5 + h4 * r3_5
        d3 = h0 * r3 + h1 * r2 + h2 * r1 + h3 * r0 + h4 * r4_5
        d4 = h0 * r4 + h1 * r3 + h2 * r2 + h3 * r1 + h4 * r0

        c = d0 >> 26
        h0 = d0 & _MASK26
        d1 = d1 + c
        c = d1 >> 26
        h1 = d1 & _MASK26
        d2 = d2 + c
        c = d2 >> 26
        h2 = d2 & _MASK26
        d3 = d3 + c
        c = d3 >> 26
        h3 = d3 & _MASK26
        d4 = d4 + c
        c = d4 >> 26
        h4 = d4 & _MASK26
        h0 = h0 + c * 5
        c = h0 >> 26
        h0 = h0 & _MASK26
        h1 = h1 + c

    # final carry chain and the constant-time-ish "is h >= p" fold: p = 2^130-5, so
    # h - p = h + 5 - 2^130; the +5 propagates through the limb carries.
    c = h1 >> 26
    h1 = h1 & _MASK26
    h2 = h2 + c
    c = h2 >> 26
    h2 = h2 & _MASK26
    h3 = h3 + c
    c = h3 >> 26
    h3 = h3 & _MASK26
    h4 = h4 + c
    c = h4 >> 26
    h4 = h4 & _MASK26
    h0 = h0 + c * 5
    c = h0 >> 26
    h0 = h0 & _MASK26
    h1 = h1 + c

    g0 = h0 + 5
    c = g0 >> 26
    g0 = g0 & _MASK26
    g1 = h1 + c
    c = g1 >> 26
    g1 = g1 & _MASK26
    g2 = h2 + c
    c = g2 >> 26
    g2 = g2 & _MASK26
    g3 = h3 + c
    c = g3 >> 26
    g3 = g3 & _MASK26
    g4 = h4 + c - (1 << 26)
    take_g = g4 >= 0
    h0 = tl.where(take_g, g0, h0)
    h1 = tl.where(take_g, g1, h1)
    h2 = tl.where(take_g, g2, h2)
    h3 = tl.where(take_g, g3, h3)
    h4 = tl.where(take_g, g4, h4)

    # tag = (h + s) mod 2^128. `s` goes in as 26-bit limbs and is added *with carries*: adding
    # the four 32-bit words of `s` independently to the packed limbs loses every cross-word
    # carry, which is the kind of bug the degenerate cases (s = 0) do not catch.
    sl = ()
    for i in tl.static_range(4):
        w = tl.load(KEY + lane * 32 + 16 + i * 4 + 0, mask=m, other=0).to(tl.int64)
        w = w | (tl.load(KEY + lane * 32 + 16 + i * 4 + 1, mask=m, other=0).to(tl.int64) << 8)
        w = w | (tl.load(KEY + lane * 32 + 16 + i * 4 + 2, mask=m, other=0).to(tl.int64) << 16)
        w = w | (tl.load(KEY + lane * 32 + 16 + i * 4 + 3, mask=m, other=0).to(tl.int64) << 24)
        sl = sl + (w,)
    s0 = sl[0] & _MASK26
    s1 = ((sl[0] >> 26) | (sl[1] << 6)) & _MASK26
    s2 = ((sl[1] >> 20) | (sl[2] << 12)) & _MASK26
    s3 = ((sl[2] >> 14) | (sl[3] << 18)) & _MASK26
    s4 = (sl[3] >> 8) & _MASK26

    h0 = h0 + s0
    c = h0 >> 26
    h0 = h0 & _MASK26
    h1 = h1 + s1 + c
    c = h1 >> 26
    h1 = h1 & _MASK26
    h2 = h2 + s2 + c
    c = h2 >> 26
    h2 = h2 & _MASK26
    h3 = h3 + s3 + c
    c = h3 >> 26
    h3 = h3 & _MASK26
    h4 = h4 + s4 + c  # bits above 2^130 are dropped, which is the `mod 2^128` for the tag

    f = ()
    f = f + (h0 | (h1 << 26),)
    f = f + ((h1 >> 6) | (h2 << 20),)
    f = f + ((h2 >> 12) | (h3 << 14),)
    f = f + ((h3 >> 18) | (h4 << 8),)
    for i in tl.static_range(4):
        for j in tl.static_range(4):
            tl.store(TAG + lane * 16 + i * 4 + j, ((f[i] >> (8 * j)) & 0xFF).to(tl.uint8), mask=m)


def _to_u8(x, dev):
    """uint8 on `dev`, copying numpy input: `frombuffer` arrays are read-only and torch warns."""
    import torch

    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.uint8, device=dev)
    return torch.tensor(np.asarray(x, dtype=np.uint8), device=dev)


def poly1305_torch(keys, messages, device=None):
    """Batched MAC. `keys` `[B, 32]` uint8, `messages` `[B, N*16]` uint8 -> `[B, 16]` uint8."""
    import torch

    from .. import devices as _devices

    dev = _devices.resolve_device(device)
    k = _to_u8(keys, dev)
    m = _to_u8(messages, dev)
    if k.ndim != 2 or k.shape[1] != 32:
        raise ValueError("keys must be [B, 32]")
    if m.ndim != 2 or m.shape[0] != k.shape[0] or m.shape[1] % 16:
        raise ValueError("messages must be [B, N*16] with the same B as keys")
    # Each block carries the implicit +2^128, which is correct for a *full* 16-byte block and
    # wrong for a padded partial tail (there the 0x01 byte is the high bit). So the device
    # path takes lengths that are a multiple of 16; `poly1305_np` handles any length.
    if m.shape[1] == 0:
        raise ValueError("empty messages are not supported on the device path")
    batch = k.shape[0]
    tag = torch.empty((batch, 16), dtype=torch.uint8, device=dev)
    if batch == 0:
        return tag
    blk = 64
    grid = (triton.cdiv(batch, blk),)
    _poly1305_kernel[grid](k, m, tag, batch, NBLK=m.shape[1] // 16, BLK=blk)
    return tag


# --------------------------------------------------------------------------- AEAD


def _pad16(x):
    """Zero-pad the last axis to a multiple of 16 (the RFC's pad16; the MAC pads, the
    ciphertext does not)."""
    import torch

    rem = x.shape[1] % 16
    if rem == 0:
        return x
    pad = torch.zeros((x.shape[0], 16 - rem), dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=1)


def _mac_data(aad, ct):
    """`aad || pad16 || ct || pad16 || le64(len(aad)) || le64(len(ct))`, in bytes."""
    import torch

    parts = []
    if aad is not None and aad.shape[1]:
        parts.append(_pad16(aad))
    parts.append(_pad16(ct))
    trailer = np.zeros((ct.shape[0], 16), dtype=np.uint8)
    trailer[:, 0:8] = np.frombuffer(
        np.uint64(0 if aad is None else aad.shape[1]).tobytes(), dtype=np.uint8
    )
    trailer[:, 8:16] = np.frombuffer(np.uint64(ct.shape[1]).tobytes(), dtype=np.uint8)
    parts.append(torch.as_tensor(trailer, device=ct.device))
    return torch.cat(parts, dim=1)


def chacha20_poly1305_torch(keys, nonces, plaintext, aad=None, device=None):
    """Seal: returns `(ciphertext, tag)`. `plaintext` / `aad` are uint8 tensors `[B, n]`.

    Any length is allowed: the MAC pads to a multiple of 16 (RFC 8439's pad16), while the
    ciphertext keeps its length.
    """
    import torch

    from .. import devices as _devices
    from .chacha20 import chacha20_keystream_torch

    dev = _devices.resolve_device(device)
    pt = _to_u8(plaintext, dev)
    if pt.ndim != 2:
        raise ValueError("plaintext must be [B, nbytes]")
    _, nbytes = pt.shape
    aad_t = None if aad is None else _to_u8(aad, dev)

    poly_key = chacha20_keystream_torch(keys, nonces, 32, 0, device=dev)
    ks = chacha20_keystream_torch(keys, nonces, nbytes, 1, device=dev)
    ct = (pt ^ ks).to(torch.uint8)
    # keep both on the device: a `.cpu().numpy()` here costs a 4 MB round trip per call and
    # takes the AEAD from ~5 GB/s to 2 GB/s (measured).
    tag = poly1305_torch(poly_key, _mac_data(aad_t, ct), device=dev)
    return ct, tag


def chacha20_poly1305_open_torch(keys, nonces, ciphertext, tag, aad=None, device=None):
    """Open: returns `(plaintext, ok)`. `ok` is a full 16-byte tag comparison per row."""
    import torch

    from .. import devices as _devices
    from .chacha20 import chacha20_keystream_torch

    dev = _devices.resolve_device(device)
    ct = _to_u8(ciphertext, dev)
    if ct.ndim != 2:
        raise ValueError("ciphertext must be [B, nbytes]")
    aad_t = None if aad is None else _to_u8(aad, dev)
    poly_key = chacha20_keystream_torch(keys, nonces, 32, 0, device=dev)
    expect = poly1305_torch(poly_key, _mac_data(aad_t, ct), device=dev)
    got = _to_u8(tag, dev)
    ok = (got == expect).all(dim=1)
    ks = chacha20_keystream_torch(keys, nonces, ct.shape[1], 1, device=dev)
    return (ct ^ ks).to(torch.uint8), ok
