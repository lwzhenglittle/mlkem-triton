"""Ascon lightweight cryptography: Ascon-Hash256 and Ascon-AEAD128 (NIST SP 800-232) plus Ascon-128a (v1.2).

All three modes share the 320-bit Ascon-p permutation (`5 x uint64` state, bitsliced 5-bit S-box,
linear diffusion with 64-bit rotates). With only five 64-bit state words per lane (1/5 of Keccak's
state), the permutation fits entirely in registers with zero spills on every GPU backend.

Because x86 CPUs have no dedicated Ascon instruction and Ascon's 12/8-round serial dependency chain
limits per-core ILP, batched Ascon-Hash256 and Ascon-AEAD128/Ascon-128a are strong GPU workloads
under the three-factor rule (`docs/patterns.md` §0).

Standards implemented:
- **NIST SP 800-232 (August 2025, little-endian word order)**:
  - `ascon_hash256_np` / `ascon_hash256_np_batch` / `ascon_hash256_torch` (IV = `0x0000080100cc0002`, r = 8 B, p[12])
  - `ascon_aead128_encrypt_np` / `ascon_aead128_decrypt_np` / `ascon_aead128_torch` / `ascon_aead128_open_torch`
    (IV = `0x00001000808c0001`, r = 16 B, a = 12, b = 8, domain sep `s4 ^= 1 << 63`)
- **Ascon v1.2 (big-endian word order, NIST LWC submission)**:
  - `ascon128a_encrypt_np` / `ascon128a_decrypt_np` / `ascon128a_torch` / `ascon128a_open_torch`
    (IV = `0x80800c0800000000`, r = 16 B, a = 12, b = 8, domain sep `s4 ^= 1`)
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

from .. import devices as _devices
from .._inputs import as_batch_bytes

__all__ = [
    "ASCON128A_IV",
    "ASCON_AEAD128_IV",
    "ASCON_HASH256_IV",
    "ascon128a_decrypt_np",
    "ascon128a_encrypt_np",
    "ascon128a_open_torch",
    "ascon128a_torch",
    "ascon_aead128_decrypt_np",
    "ascon_aead128_encrypt_np",
    "ascon_aead128_open_torch",
    "ascon_aead128_torch",
    "ascon_hash256_np",
    "ascon_hash256_np_batch",
    "ascon_hash256_torch",
    "ascon_permute_np",
]

# NIST SP 800-232 (little-endian) IVs
ASCON_AEAD128_IV = 0x00001000808C0001
ASCON_HASH256_IV = 0x0000080100CC0002
# Ascon v1.2 (big-endian) IV
ASCON128A_IV = 0x80800C0800000000

_MASK64 = 0xFFFFFFFFFFFFFFFF
_RC = tuple(((0xF - i) << 4) | i for i in range(12))


def _rotr64(x: int, n: int) -> int:
    return ((x >> n) | (x << (64 - n))) & _MASK64


def ascon_permute_np(state: list[int], rounds: int = 12) -> list[int]:
    """320-bit Ascon-p[rounds] permutation on five 64-bit words (plain Python oracle)."""
    x0, x1, x2, x3, x4 = (w & _MASK64 for w in state)
    for r in range(12 - rounds, 12):
        x2 ^= _RC[r]
        # 5-bit bitsliced S-box
        x0 ^= x4
        x4 ^= x3
        x2 ^= x1
        t0 = (~x0 & _MASK64) & x1
        t1 = (~x1 & _MASK64) & x2
        t2 = (~x2 & _MASK64) & x3
        t3 = (~x3 & _MASK64) & x4
        t4 = (~x4 & _MASK64) & x0
        x0 ^= t1
        x1 ^= t2
        x2 ^= t3
        x3 ^= t4
        x4 ^= t0
        x1 ^= x0
        x0 ^= x4
        x3 ^= x2
        x2 ^= _MASK64
        # Linear diffusion
        x0 ^= _rotr64(x0, 19) ^ _rotr64(x0, 28)
        x1 ^= _rotr64(x1, 61) ^ _rotr64(x1, 39)
        x2 ^= _rotr64(x2, 1) ^ _rotr64(x2, 6)
        x3 ^= _rotr64(x3, 10) ^ _rotr64(x3, 17)
        x4 ^= _rotr64(x4, 7) ^ _rotr64(x4, 41)
    return [x0, x1, x2, x3, x4]


# Precomputed initial state of Ascon-Hash256 after Ascon-p[12]([IV, 0, 0, 0, 0])
_HASH256_INIT = tuple(ascon_permute_np([ASCON_HASH256_IV, 0, 0, 0, 0], 12))


def ascon_hash256_np(msg: bytes) -> bytes:
    """NIST SP 800-232 Ascon-Hash256 single-message oracle (32-byte digest)."""
    s = list(_HASH256_INIT)
    padded = bytearray(msg)
    padded.append(0x01)
    while len(padded) % 8 != 0:
        padded.append(0x00)
    for off in range(0, len(padded), 8):
        s[0] ^= int.from_bytes(padded[off : off + 8], "little")
        s = ascon_permute_np(s, 12)
    out = bytearray()
    for i in range(4):
        out += s[0].to_bytes(8, "little")
        if i < 3:
            s = ascon_permute_np(s, 12)
    return bytes(out)


def ascon_hash256_np_batch(msgs) -> np.ndarray:
    """Batched NIST SP 800-232 Ascon-Hash256 reference: returns `[B, 32] uint8`."""
    arr = as_batch_bytes(msgs)
    b = arr.shape[0]
    out = np.empty((b, 32), dtype=np.uint8)
    for i in range(b):
        out[i] = np.frombuffer(ascon_hash256_np(bytes(arr[i])), dtype=np.uint8)
    return out


def _aead_encrypt_core(
    key: bytes, nonce: bytes, pt: bytes, ad: bytes, *, sp800_232: bool
) -> tuple[bytes, bytes]:
    if len(key) != 16 or len(nonce) != 16:
        raise ValueError("Ascon AEAD requires 16-byte key and 16-byte nonce")
    order = "little" if sp800_232 else "big"
    iv = ASCON_AEAD128_IV if sp800_232 else ASCON128A_IV
    pad_byte = 0x01 if sp800_232 else 0x80
    dom_sep = (1 << 63) if sp800_232 else 1

    k0 = int.from_bytes(key[:8], order)
    k1 = int.from_bytes(key[8:], order)
    n0 = int.from_bytes(nonce[:8], order)
    n1 = int.from_bytes(nonce[8:], order)

    s = ascon_permute_np([iv, k0, k1, n0, n1], 12)
    s[3] ^= k0
    s[4] ^= k1

    if ad:
        ad_pad = bytearray(ad)
        ad_pad.append(pad_byte)
        while len(ad_pad) % 16 != 0:
            ad_pad.append(0)
        for off in range(0, len(ad_pad), 16):
            s[0] ^= int.from_bytes(ad_pad[off : off + 8], order)
            s[1] ^= int.from_bytes(ad_pad[off + 8 : off + 16], order)
            s = ascon_permute_np(s, 8)
    s[4] ^= dom_sep

    ct = bytearray()
    full_end = (len(pt) // 16) * 16
    for off in range(0, full_end, 16):
        s[0] ^= int.from_bytes(pt[off : off + 8], order)
        s[1] ^= int.from_bytes(pt[off + 8 : off + 16], order)
        ct += s[0].to_bytes(8, order) + s[1].to_bytes(8, order)
        s = ascon_permute_np(s, 8)

    rem = pt[full_end:]
    rem_len = len(rem)
    last_block = bytearray(16)
    last_block[:rem_len] = rem
    last_block[rem_len] = pad_byte
    s[0] ^= int.from_bytes(last_block[:8], order)
    s[1] ^= int.from_bytes(last_block[8:], order)
    if rem_len:
        ct += (s[0].to_bytes(8, order) + s[1].to_bytes(8, order))[:rem_len]

    s[2] ^= k0
    s[3] ^= k1
    s = ascon_permute_np(s, 12)
    t0 = s[3] ^ k0
    t1 = s[4] ^ k1
    tag = t0.to_bytes(8, order) + t1.to_bytes(8, order)
    return bytes(ct), tag


def _aead_decrypt_core(
    key: bytes, nonce: bytes, ct: bytes, tag: bytes, ad: bytes, *, sp800_232: bool
) -> tuple[bytes, bool]:
    if len(key) != 16 or len(nonce) != 16 or len(tag) != 16:
        raise ValueError("Ascon AEAD requires 16-byte key, nonce, and tag")
    order = "little" if sp800_232 else "big"
    iv = ASCON_AEAD128_IV if sp800_232 else ASCON128A_IV
    pad_byte = 0x01 if sp800_232 else 0x80
    dom_sep = (1 << 63) if sp800_232 else 1

    k0 = int.from_bytes(key[:8], order)
    k1 = int.from_bytes(key[8:], order)
    n0 = int.from_bytes(nonce[:8], order)
    n1 = int.from_bytes(nonce[8:], order)

    s = ascon_permute_np([iv, k0, k1, n0, n1], 12)
    s[3] ^= k0
    s[4] ^= k1

    if ad:
        ad_pad = bytearray(ad)
        ad_pad.append(pad_byte)
        while len(ad_pad) % 16 != 0:
            ad_pad.append(0)
        for off in range(0, len(ad_pad), 16):
            s[0] ^= int.from_bytes(ad_pad[off : off + 8], order)
            s[1] ^= int.from_bytes(ad_pad[off + 8 : off + 16], order)
            s = ascon_permute_np(s, 8)
    s[4] ^= dom_sep

    pt = bytearray()
    full_end = (len(ct) // 16) * 16
    for off in range(0, full_end, 16):
        c0 = int.from_bytes(ct[off : off + 8], order)
        c1 = int.from_bytes(ct[off + 8 : off + 16], order)
        pt += (s[0] ^ c0).to_bytes(8, order) + (s[1] ^ c1).to_bytes(8, order)
        s[0] = c0
        s[1] = c1
        s = ascon_permute_np(s, 8)

    rem = ct[full_end:]
    rem_len = len(rem)
    s_bytes = bytearray(s[0].to_bytes(8, order) + s[1].to_bytes(8, order))
    for i in range(rem_len):
        pt.append(s_bytes[i] ^ rem[i])
        s_bytes[i] = rem[i]
    s_bytes[rem_len] ^= pad_byte
    s[0] = int.from_bytes(s_bytes[:8], order)
    s[1] = int.from_bytes(s_bytes[8:], order)

    s[2] ^= k0
    s[3] ^= k1
    s = ascon_permute_np(s, 12)
    t0 = s[3] ^ k0
    t1 = s[4] ^ k1
    expected_tag = t0.to_bytes(8, order) + t1.to_bytes(8, order)
    return bytes(pt), expected_tag == tag


def ascon_aead128_encrypt_np(
    key: bytes, nonce: bytes, pt: bytes, ad: bytes = b""
) -> tuple[bytes, bytes]:
    """NIST SP 800-232 Ascon-AEAD128 seal oracle: returns `(ciphertext, 16B tag)`."""
    return _aead_encrypt_core(key, nonce, pt, ad, sp800_232=True)


def ascon_aead128_decrypt_np(
    key: bytes, nonce: bytes, ct: bytes, tag: bytes, ad: bytes = b""
) -> tuple[bytes, bool]:
    """NIST SP 800-232 Ascon-AEAD128 open oracle: returns `(plaintext, valid)`."""
    return _aead_decrypt_core(key, nonce, ct, tag, ad, sp800_232=True)


def ascon128a_encrypt_np(
    key: bytes, nonce: bytes, pt: bytes, ad: bytes = b""
) -> tuple[bytes, bytes]:
    """Ascon-128a (v1.2, big-endian) seal oracle: returns `(ciphertext, 16B tag)`."""
    return _aead_encrypt_core(key, nonce, pt, ad, sp800_232=False)


def ascon128a_decrypt_np(
    key: bytes, nonce: bytes, ct: bytes, tag: bytes, ad: bytes = b""
) -> tuple[bytes, bool]:
    """Ascon-128a (v1.2, big-endian) open oracle: returns `(plaintext, valid)`."""
    return _aead_decrypt_core(key, nonce, ct, tag, ad, sp800_232=False)


# --------------------------------------------------------------------------- Triton GPU kernels


@triton.jit
def _rotr(x, n):
    x = x.to(tl.uint64)
    return (x >> n) | (x << (64 - n))


@triton.jit
def _bs64(x):
    x = x.to(tl.uint64)
    x = ((x & 0x00FF00FF00FF00FF) << 8) | ((x >> 8) & 0x00FF00FF00FF00FF)
    x = ((x & 0x0000FFFF0000FFFF) << 16) | ((x >> 16) & 0x0000FFFF0000FFFF)
    return (x << 32) | (x >> 32)


@triton.jit
def _ascon_round(x0, x1, x2, x3, x4, rc):
    x2 = x2 ^ rc
    x0 = x0 ^ x4
    x4 = x4 ^ x3
    x2 = x2 ^ x1
    t0 = (~x0) & x1
    t1 = (~x1) & x2
    t2 = (~x2) & x3
    t3 = (~x3) & x4
    t4 = (~x4) & x0
    x0 = x0 ^ t1
    x1 = x1 ^ t2
    x2 = x2 ^ t3
    x3 = x3 ^ t4
    x4 = x4 ^ t0
    x1 = x1 ^ x0
    x0 = x0 ^ x4
    x3 = x3 ^ x2
    x2 = ~x2
    x0 = x0 ^ _rotr(x0, 19) ^ _rotr(x0, 28)
    x1 = x1 ^ _rotr(x1, 61) ^ _rotr(x1, 39)
    x2 = x2 ^ _rotr(x2, 1) ^ _rotr(x2, 6)
    x3 = x3 ^ _rotr(x3, 10) ^ _rotr(x3, 17)
    x4 = x4 ^ _rotr(x4, 7) ^ _rotr(x4, 41)
    return x0, x1, x2, x3, x4


@triton.jit
def _ascon_p12(x0, x1, x2, x3, x4):
    for r in range(0, 12, 1):
        rc = (((15 - r) << 4) | r).to(tl.uint64)
        x0, x1, x2, x3, x4 = _ascon_round(x0, x1, x2, x3, x4, rc)
    return x0, x1, x2, x3, x4


@triton.jit
def _ascon_p8(x0, x1, x2, x3, x4):
    for r in range(4, 12, 1):
        rc = (((15 - r) << 4) | r).to(tl.uint64)
        x0, x1, x2, x3, x4 = _ascon_round(x0, x1, x2, x3, x4, rc)
    return x0, x1, x2, x3, x4


@triton.jit
def _ascon_hash256_kernel(
    INIT,
    MSG,
    OUT,
    rows,
    NBLOCKS,
    BLOCK: tl.constexpr,
):
    """Batched NIST SP 800-232 Ascon-Hash256 over pre-padded `[rows, NBLOCKS]` int64 words."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows
    z = tl.zeros((BLOCK,), dtype=tl.uint64)

    x0 = z | tl.load(INIT + 0).to(tl.uint64, bitcast=True)
    x1 = z | tl.load(INIT + 1).to(tl.uint64, bitcast=True)
    x2 = z | tl.load(INIT + 2).to(tl.uint64, bitcast=True)
    x3 = z | tl.load(INIT + 3).to(tl.uint64, bitcast=True)
    x4 = z | tl.load(INIT + 4).to(tl.uint64, bitcast=True)

    for blk in range(0, NBLOCKS, 1):
        w = tl.load(MSG + row * NBLOCKS + blk, mask=m, other=0).to(tl.uint64, bitcast=True)
        x0 = x0 ^ w
        x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)

    tl.store(OUT + row * 4 + 0, x0.to(tl.int64, bitcast=True), mask=m)
    x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)
    tl.store(OUT + row * 4 + 1, x0.to(tl.int64, bitcast=True), mask=m)
    x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)
    tl.store(OUT + row * 4 + 2, x0.to(tl.int64, bitcast=True), mask=m)
    x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)
    tl.store(OUT + row * 4 + 3, x0.to(tl.int64, bitcast=True), mask=m)


@triton.jit
def _ascon_aead_kernel(
    KEYS,
    NONCES,
    AD,
    IN_BUF,
    OUT_BUF,
    TAGS,
    rows,
    N_AD_BLOCKS,
    N_FULL_BLOCKS,
    REM_BYTES,
    KEY_STRIDE,
    NONCE_STRIDE,
    AD_STRIDE,
    DATA_STRIDE,
    DECRYPT: tl.constexpr,
    SP800_232: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Batched Ascon-AEAD128 (SP 800-232) or Ascon-128a (v1.2) seal/open kernel.

    `IN_BUF` and `OUT_BUF` are padded to `DATA_STRIDE = 2 * (N_FULL_BLOCKS + 1)` int64 words per row
    (16-byte aligned); the caller slices `OUT_BUF` down to `len(pt)` bytes.
    """
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows
    z = tl.zeros((BLOCK,), dtype=tl.uint64)

    k_row = row * KEY_STRIDE
    n_row = row * NONCE_STRIDE
    k0 = tl.load(KEYS + k_row + 0, mask=m, other=0).to(tl.uint64, bitcast=True)
    k1 = tl.load(KEYS + k_row + 1, mask=m, other=0).to(tl.uint64, bitcast=True)
    n0 = tl.load(NONCES + n_row + 0, mask=m, other=0).to(tl.uint64, bitcast=True)
    n1 = tl.load(NONCES + n_row + 1, mask=m, other=0).to(tl.uint64, bitcast=True)
    if not SP800_232:
        k0 = _bs64(k0)
        k1 = _bs64(k1)
        n0 = _bs64(n0)
        n1 = _bs64(n1)

    if SP800_232:
        x0 = z | tl.full((), 0x00001000808C0001, tl.uint64)
    else:
        x0 = z | tl.full((), 0x80800C0800000000, tl.uint64)
    x1 = k0
    x2 = k1
    x3 = n0
    x4 = n1

    x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)
    x3 = x3 ^ k0
    x4 = x4 ^ k1

    # Associated data (already padded to N_AD_BLOCKS 16-byte blocks if non-empty)
    for blk in range(0, N_AD_BLOCKS, 1):
        a0 = tl.load(AD + row * AD_STRIDE + blk * 2 + 0, mask=m, other=0).to(
            tl.uint64, bitcast=True
        )
        a1 = tl.load(AD + row * AD_STRIDE + blk * 2 + 1, mask=m, other=0).to(
            tl.uint64, bitcast=True
        )
        if not SP800_232:
            a0 = _bs64(a0)
            a1 = _bs64(a1)
        x0 = x0 ^ a0
        x1 = x1 ^ a1
        x0, x1, x2, x3, x4 = _ascon_p8(x0, x1, x2, x3, x4)

    if SP800_232:
        x4 = x4 ^ tl.full((), 0x8000000000000000, tl.uint64)
    else:
        x4 = x4 ^ tl.full((), 1, tl.uint64)

    # Full 16-byte plaintext/ciphertext blocks
    for blk in range(0, N_FULL_BLOCKS, 1):
        w0 = tl.load(IN_BUF + row * DATA_STRIDE + blk * 2 + 0, mask=m, other=0).to(
            tl.uint64, bitcast=True
        )
        w1 = tl.load(IN_BUF + row * DATA_STRIDE + blk * 2 + 1, mask=m, other=0).to(
            tl.uint64, bitcast=True
        )
        if not SP800_232:
            w0 = _bs64(w0)
            w1 = _bs64(w1)
        o0 = x0 ^ w0
        o1 = x1 ^ w1
        if DECRYPT:
            x0 = w0
            x1 = w1
        else:
            x0 = o0
            x1 = o1
        if not SP800_232:
            o0 = _bs64(o0)
            o1 = _bs64(o1)
        tl.store(OUT_BUF + row * DATA_STRIDE + blk * 2 + 0, o0.to(tl.int64, bitcast=True), mask=m)
        tl.store(OUT_BUF + row * DATA_STRIDE + blk * 2 + 1, o1.to(tl.int64, bitcast=True), mask=m)
        x0, x1, x2, x3, x4 = _ascon_p8(x0, x1, x2, x3, x4)

    # Final partial block (0 <= REM_BYTES < 16).
    # Input tail bytes beyond REM_BYTES are guaranteed zero by the launcher.
    w0 = tl.load(IN_BUF + row * DATA_STRIDE + N_FULL_BLOCKS * 2 + 0, mask=m, other=0).to(
        tl.uint64, bitcast=True
    )
    w1 = tl.load(IN_BUF + row * DATA_STRIDE + N_FULL_BLOCKS * 2 + 1, mask=m, other=0).to(
        tl.uint64, bitcast=True
    )
    if not SP800_232:
        w0 = _bs64(w0)
        w1 = _bs64(w1)

    o0 = x0 ^ w0
    o1 = x1 ^ w1
    if not DECRYPT:
        x0 = o0
        x1 = o1
    else:
        # Mask keeps state bytes at >= REM_BYTES untouched, replaces < REM_BYTES with ciphertext
        if SP800_232:
            # LE: bytes 0..rem-1 occupy the lowest 8*rem bits
            if REM_BYTES == 0:
                m0 = tl.full((), 0, tl.uint64)
                m1 = tl.full((), 0, tl.uint64)
            elif REM_BYTES < 8:
                m0 = (tl.full((), 1, tl.uint64) << (REM_BYTES * 8)) - 1
                m1 = tl.full((), 0, tl.uint64)
            elif REM_BYTES == 8:
                m0 = tl.full((), 0xFFFFFFFFFFFFFFFF, tl.uint64)
                m1 = tl.full((), 0, tl.uint64)
            else:
                m0 = tl.full((), 0xFFFFFFFFFFFFFFFF, tl.uint64)
                m1 = (tl.full((), 1, tl.uint64) << ((REM_BYTES - 8) * 8)) - 1
        else:
            # BE: bytes 0..rem-1 occupy the highest 8*rem bits
            if REM_BYTES == 0:
                m0 = tl.full((), 0, tl.uint64)
                m1 = tl.full((), 0, tl.uint64)
            elif REM_BYTES < 8:
                m0 = ~((tl.full((), 1, tl.uint64) << ((8 - REM_BYTES) * 8)) - 1)
                m1 = tl.full((), 0, tl.uint64)
            elif REM_BYTES == 8:
                m0 = tl.full((), 0xFFFFFFFFFFFFFFFF, tl.uint64)
                m1 = tl.full((), 0, tl.uint64)
            else:
                m0 = tl.full((), 0xFFFFFFFFFFFFFFFF, tl.uint64)
                m1 = ~((tl.full((), 1, tl.uint64) << ((16 - REM_BYTES) * 8)) - 1)
        x0 = (x0 & (~m0)) | (w0 & m0)
        x1 = (x1 & (~m1)) | (w1 & m1)

    # Apply pad10* at byte position REM_BYTES
    if SP800_232:
        if REM_BYTES < 8:
            x0 = x0 ^ (tl.full((), 1, tl.uint64) << (REM_BYTES * 8))
        else:
            x1 = x1 ^ (tl.full((), 1, tl.uint64) << ((REM_BYTES - 8) * 8))
    else:
        if REM_BYTES < 8:
            x0 = x0 ^ (tl.full((), 0x80, tl.uint64) << ((7 - REM_BYTES) * 8))
        else:
            x1 = x1 ^ (tl.full((), 0x80, tl.uint64) << ((15 - REM_BYTES) * 8))

    if not SP800_232:
        o0 = _bs64(o0)
        o1 = _bs64(o1)
    tl.store(
        OUT_BUF + row * DATA_STRIDE + N_FULL_BLOCKS * 2 + 0,
        o0.to(tl.int64, bitcast=True),
        mask=m,
    )
    tl.store(
        OUT_BUF + row * DATA_STRIDE + N_FULL_BLOCKS * 2 + 1,
        o1.to(tl.int64, bitcast=True),
        mask=m,
    )

    # Finalization
    x2 = x2 ^ k0
    x3 = x3 ^ k1
    x0, x1, x2, x3, x4 = _ascon_p12(x0, x1, x2, x3, x4)
    t0 = x3 ^ k0
    t1 = x4 ^ k1
    if not SP800_232:
        t0 = _bs64(t0)
        t1 = _bs64(t1)
    tl.store(TAGS + row * 2 + 0, t0.to(tl.int64, bitcast=True), mask=m)
    tl.store(TAGS + row * 2 + 1, t1.to(tl.int64, bitcast=True), mask=m)


_HASH_INIT_CACHE: dict = {}


def _hash_init_tensor(device):
    t = _HASH_INIT_CACHE.get(device)
    if t is None:
        import torch

        signed = [v - (1 << 64) if v >= (1 << 63) else v for v in _HASH256_INIT]
        t = torch.tensor(signed, dtype=torch.int64, device=device)
        _HASH_INIT_CACHE[device] = t
    return t


def ascon_hash256_torch(msgs, device=None, pre_padded: bool = False, block: int = 128):
    """Batched NIST SP 800-232 Ascon-Hash256 on GPU: `[B, L] uint8` -> `[B, 32] uint8`."""
    import torch

    dev = _devices.resolve_device(device)
    if isinstance(msgs, torch.Tensor):
        t = msgs.to(device=dev, dtype=torch.uint8).contiguous()
    else:
        arr = np.array(as_batch_bytes(msgs), dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).to(dev)
    if t.ndim != 2:
        raise ValueError(f"msgs must be 2-D, got shape {tuple(t.shape)}")
    b, mlen = t.shape
    if b == 0:
        return torch.empty((0, 32), dtype=torch.uint8, device=dev)
    if pre_padded:
        if mlen == 0 or mlen % 8 != 0:
            raise ValueError(f"pre-padded length must be a positive multiple of 8, got {mlen}")
        padded = t
    else:
        nblocks = (mlen + 1 + 7) // 8
        padded = torch.zeros((b, nblocks * 8), dtype=torch.uint8, device=dev)
        if mlen > 0:
            padded[:, :mlen] = t
        padded[:, mlen] = 0x01
    nblocks = padded.shape[1] // 8
    out_words = torch.empty((b, 4), dtype=torch.int64, device=dev)
    _ascon_hash256_kernel[(triton.cdiv(b, block),)](
        _hash_init_tensor(dev),
        padded.view(torch.int64),
        out_words,
        b,
        nblocks,
        BLOCK=block,
        num_warps=_devices.warps("ascon", 4),
    )
    return out_words.view(torch.uint8)


def _to_u8_2d(x, expected_cols: int | None, dev, *, allow_1d_broadcast: bool = False):
    import torch

    if isinstance(x, torch.Tensor):
        t = x.to(device=dev, dtype=torch.uint8)
        if t.ndim == 1 and allow_1d_broadcast:
            t = t.unsqueeze(0)
        t = t.contiguous()
    else:
        arr = np.array(as_batch_bytes(x, expected_cols), dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).to(dev)
    if t.ndim != 2:
        raise ValueError(f"expected 2-D uint8 tensor, got shape {tuple(t.shape)}")
    if expected_cols is not None and t.shape[1] != expected_cols:
        raise ValueError(f"expected {expected_cols} bytes per row, got {t.shape[1]}")
    return t


def _ascon_aead_run(
    keys,
    nonces,
    data,
    ad=None,
    *,
    decrypt: bool,
    sp800_232: bool,
    device=None,
    block: int = 128,
):
    import torch

    dev = _devices.resolve_device(device)
    k = _to_u8_2d(keys, 16, dev, allow_1d_broadcast=True)
    n = _to_u8_2d(nonces, 16, dev, allow_1d_broadcast=True)
    d = _to_u8_2d(data, None, dev)
    b, dlen = d.shape
    if b == 0:
        return torch.empty((0, dlen), dtype=torch.uint8, device=dev), torch.empty(
            (0, 16), dtype=torch.uint8, device=dev
        )
    if k.shape[0] not in (1, b):
        raise ValueError(f"keys batch {k.shape[0]} incompatible with data batch {b}")
    if n.shape[0] not in (1, b):
        raise ValueError(f"nonces batch {n.shape[0]} incompatible with data batch {b}")

    pad_byte = 0x01 if sp800_232 else 0x80
    if ad is None:
        n_ad_blocks = 0
        ad_words = torch.empty((1, 2), dtype=torch.int64, device=dev)
        ad_stride = 0
    else:
        a = _to_u8_2d(ad, None, dev, allow_1d_broadcast=True)
        if a.shape[0] not in (1, b):
            raise ValueError(f"ad batch {a.shape[0]} incompatible with data batch {b}")
        alen = a.shape[1]
        if alen == 0:
            n_ad_blocks = 0
            ad_words = torch.empty((1, 2), dtype=torch.int64, device=dev)
            ad_stride = 0
        else:
            n_ad_blocks = (alen + 1 + 15) // 16
            ad_pad = torch.zeros((a.shape[0], n_ad_blocks * 16), dtype=torch.uint8, device=dev)
            ad_pad[:, :alen] = a
            ad_pad[:, alen] = pad_byte
            ad_words = ad_pad.view(torch.int64)
            ad_stride = 0 if a.shape[0] == 1 else n_ad_blocks * 2

    n_full = dlen // 16
    rem = dlen % 16
    data_words_per_row = (n_full + 1) * 2
    in_pad = torch.zeros((b, data_words_per_row * 8), dtype=torch.uint8, device=dev)
    if dlen > 0:
        in_pad[:, :dlen] = d
    out_pad = torch.empty_like(in_pad)
    tag_words = torch.empty((b, 2), dtype=torch.int64, device=dev)

    _ascon_aead_kernel[(triton.cdiv(b, block),)](
        k.view(torch.int64),
        n.view(torch.int64),
        ad_words,
        in_pad.view(torch.int64),
        out_pad.view(torch.int64),
        tag_words,
        b,
        n_ad_blocks,
        n_full,
        rem,
        0 if k.shape[0] == 1 else 2,
        0 if n.shape[0] == 1 else 2,
        ad_stride,
        data_words_per_row,
        DECRYPT=decrypt,
        SP800_232=sp800_232,
        BLOCK=block,
        num_warps=_devices.warps("ascon", 4),
    )
    out = out_pad[:, :dlen].contiguous()
    tags = tag_words.view(torch.uint8)
    return out, tags


def ascon_aead128_torch(keys, nonces, pt, ad=None, device=None, block: int = 128):
    """Batched NIST SP 800-232 Ascon-AEAD128 seal on GPU: returns `(ct [B, L], tag [B, 16])`."""
    return _ascon_aead_run(
        keys, nonces, pt, ad, decrypt=False, sp800_232=True, device=device, block=block
    )


def ascon_aead128_open_torch(keys, nonces, ct, tags, ad=None, device=None, block: int = 128):
    """Batched NIST SP 800-232 Ascon-AEAD128 open on GPU: returns `(pt [B, L], valid [B] bool)`."""
    dev = _devices.resolve_device(device)
    pt, expected = _ascon_aead_run(
        keys, nonces, ct, ad, decrypt=True, sp800_232=True, device=dev, block=block
    )
    tag_t = _to_u8_2d(tags, 16, dev)
    valid = (expected == tag_t).all(dim=1)
    return pt, valid


def ascon128a_torch(keys, nonces, pt, ad=None, device=None, block: int = 128):
    """Batched Ascon-128a (v1.2, big-endian) seal on GPU: returns `(ct [B, L], tag [B, 16])`."""
    return _ascon_aead_run(
        keys, nonces, pt, ad, decrypt=False, sp800_232=False, device=device, block=block
    )


def ascon128a_open_torch(keys, nonces, ct, tags, ad=None, device=None, block: int = 128):
    """Batched Ascon-128a (v1.2, big-endian) open on GPU: returns `(pt [B, L], valid [B] bool)`."""
    dev = _devices.resolve_device(device)
    pt, expected = _ascon_aead_run(
        keys, nonces, ct, ad, decrypt=True, sp800_232=False, device=dev, block=block
    )
    tag_t = _to_u8_2d(tags, 16, dev)
    valid = (expected == tag_t).all(dim=1)
    return pt, valid
