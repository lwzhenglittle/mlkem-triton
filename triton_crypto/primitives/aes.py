"""AES-128-CTR through a T-table: the lookup counter-example, not a fast path.

The point of this module is a measurement, not a cipher. On a CPU, AES is four T-table lookups
per column per round, and AES-NI beats it anyway; on a GPU there is no AES unit and no shared
memory, so the table lives in global memory and every lookup is a **gather**:

    10 rounds x 4 words x 4 lookups = 160 gathers per 16-byte block = 10 gathers per byte

Measured here: the gather costs 7-8.5 Gload/s on an Arc B390 (`probes/triton-primitive-probe.py`),
which puts a T-table AES at **~0.8 GB/s** -- against 13.8 GB/s for one core of AES-NI and
166 GB/s for sixteen processes. That is a 200x gap, and it is the clearest single number in
this repository for "do not port a table-driven cipher to a GPU".

The alternative (bitslicing: 113-gate S-box on 32-64 lanes of a register) is 2-3 weeks of work
and is *predicted* at 20-40 GB/s, i.e. still 0.12-0.23x of the 16-process CPU. Neither path is
worth it on this hardware; the negative result is the deliverable.

What is implemented:

    aes128_expand_key_np, aes128_encrypt_block_np, aes128_ctr_np   the reference
    aes128_ctr_torch                                               the T-table CTR on device
    T_TABLE_NP                                                     Te0..Te3 + S-box, 4.25 KiB

Correctness: FIPS-197 appendix C.1 for the block function, and `openssl enc -aes-128-ctr` for
the stream, both in `tests/test_aes.py`.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

__all__ = [
    "SBOX",
    "T_TABLE_NP",
    "aes128_ctr_np",
    "aes128_ctr_torch",
    "aes128_encrypt_block_np",
    "aes128_expand_key_np",
    "t_table",
]

SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    dtype=np.uint32,
)
RCON = np.array([0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1B, 0x36], dtype=np.uint32)

_M32 = tl.constexpr(0xFFFFFFFF)


def _xtime(x: np.ndarray) -> np.ndarray:
    return ((x << 1) ^ np.where(x & 0x80, 0x1B, 0)) & 0xFF


def t_table() -> np.ndarray:
    """`Te0..Te3 || S-box`, the 4.25 KiB table the kernel gathers from."""
    s = SBOX.astype(np.uint32)
    # Te0[x] = [2s, s, s, 3s] with s = S[x], 3s = 2s ^ s. (Building [s, 2s, 4s, 8s] instead is
    # an easy misreading of "the S-box column rotated with the xtime multiples".)
    te0 = (_xtime(s) << 24) | (s << 16) | (s << 8) | (_xtime(s) ^ s)
    ror = lambda x, r: ((x >> r) | (x << (32 - r))) & 0xFFFFFFFF  # noqa: E731
    te = [te0] + [ror(te0, 8 * i) for i in (1, 2, 3)]
    return np.concatenate([np.stack(te), s[None, :]]).astype(np.uint32)


T_TABLE_NP = t_table()  # [5, 256]: Te0..Te3 then the S-box


# --------------------------------------------------------------------------- reference


def _sub_word(w: np.ndarray) -> np.ndarray:
    return (
        (SBOX[(w >> 24) & 0xFF].astype(np.uint32) << 24)
        | (SBOX[(w >> 16) & 0xFF].astype(np.uint32) << 16)
        | (SBOX[(w >> 8) & 0xFF].astype(np.uint32) << 8)
        | SBOX[w & 0xFF].astype(np.uint32)
    )


def aes128_expand_key_np(key: bytes) -> np.ndarray:
    """44 round-key words, FIPS-197 §5.2."""
    if len(key) != 16:
        raise ValueError(f"AES-128 key must be 16 bytes, got {len(key)}")
    w = np.frombuffer(key, dtype=">u4").astype(np.uint32).copy()
    out = list(w)
    for i in range(4, 44):
        t = out[i - 1]
        if i % 4 == 0:
            t = _sub_word(np.array([((t << 8) | (t >> 24)) & 0xFFFFFFFF], dtype=np.uint32))[0]
            t ^= RCON[i // 4 - 1] << 24  # Rcon is the word's *top* byte
        out.append(out[i - 4] ^ t)
    return np.array(out, dtype=np.uint32)


def aes128_encrypt_block_np(key: bytes, block: bytes) -> bytes:
    """One block, word-oriented: the same T-table formulation the kernel uses.

    Deliberately not a byte-matrix implementation -- a second formulation would be a second
    thing to get wrong, and what matters here is that the *device* path is checked against
    something independent (OpenSSL) rather than against a twin.
    """
    if len(block) != 16:
        raise ValueError(f"AES block must be 16 bytes, got {len(block)}")
    rk = aes128_expand_key_np(key)
    s = np.frombuffer(block, dtype=">u4").astype(np.uint32)

    def sbox_word(w: np.ndarray) -> np.ndarray:
        return (
            (SBOX[(w >> 24) & 0xFF] << 24)
            | (SBOX[(w >> 16) & 0xFF] << 16)
            | (SBOX[(w >> 8) & 0xFF] << 8)
            | SBOX[w & 0xFF]
        )

    s = s ^ rk[0:4]
    for rnd in range(1, 10):
        t = np.zeros(4, dtype=np.uint32)
        for j in range(4):
            for i in range(4):
                t[j] ^= T_TABLE_NP[i][(s[(j + i) % 4] >> (24 - 8 * i)) & 0xFF]
            t[j] ^= rk[4 * rnd + j]
        s = t
    f = np.zeros(4, dtype=np.uint32)
    for j in range(4):
        f[j] = (
            (SBOX[(s[j] >> 24) & 0xFF] << 24)
            | (SBOX[(s[(j + 1) % 4] >> 16) & 0xFF] << 16)
            | (SBOX[(s[(j + 2) % 4] >> 8) & 0xFF] << 8)
            | SBOX[s[(j + 3) % 4] & 0xFF]
        )
        f[j] ^= rk[40 + j]
    return f.astype(">u4").tobytes()


def aes128_ctr_np(key: bytes, nonce: bytes, length: int, counter0: int = 0) -> bytes:
    """CTR keystream: `nonce(8) || counter(8)`, the whole 16-byte block big-endian, which is
    what `openssl enc -aes-128-ctr -iv <16 bytes>` takes as its IV."""
    if len(nonce) != 8:
        raise ValueError(f"CTR nonce must be 8 bytes, got {len(nonce)}")
    out = bytearray()
    for ctr in range((length + 15) // 16):
        block = nonce + ((counter0 + ctr) & 0xFFFFFFFFFFFFFFFF).to_bytes(8, "big")
        out += aes128_encrypt_block_np(key, block)
    return bytes(out[:length])


# --------------------------------------------------------------------------- device


@triton.jit
def _te(TBL, which: tl.constexpr, byte):
    return tl.load(TBL + which * 256 + byte).to(tl.int64)


@triton.jit
def _sb(TBL, byte):
    return tl.load(TBL + 4 * 256 + byte).to(tl.int64)


@triton.jit
def _aes_ctr_kernel(
    RK, NONCE, OUT, TBL, B, COUNTER0: tl.constexpr, NBLK: tl.constexpr, BLK: tl.constexpr
):
    """One lane per (key, nonce) stream; one iteration per 16-byte CTR block.

    Ten rounds x 4 words x 4 T-table lookups = 160 gathers per block, straight into global
    memory: Triton has no shared memory, so the table cannot be staged per workgroup. `RK` is
    the 44-word expanded key per lane (expanding on the device would cost more than it saves
    at these batch sizes, and the host expansion is 1 us for 4096 keys).
    """
    pid = tl.program_id(0)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B

    rk = ()
    for i in tl.static_range(44):
        rk = rk + (tl.load(RK + lane * 44 + i, mask=m, other=0).to(tl.int64),)

    for blk in range(NBLK):
        w0 = (
            (tl.load(NONCE + lane * 8 + 0, mask=m, other=0).to(tl.int64) << 24)
            | (tl.load(NONCE + lane * 8 + 1, mask=m, other=0).to(tl.int64) << 16)
            | (tl.load(NONCE + lane * 8 + 2, mask=m, other=0).to(tl.int64) << 8)
            | tl.load(NONCE + lane * 8 + 3, mask=m, other=0).to(tl.int64)
        )
        w1 = (
            (tl.load(NONCE + lane * 8 + 4, mask=m, other=0).to(tl.int64) << 24)
            | (tl.load(NONCE + lane * 8 + 5, mask=m, other=0).to(tl.int64) << 16)
            | (tl.load(NONCE + lane * 8 + 6, mask=m, other=0).to(tl.int64) << 8)
            | tl.load(NONCE + lane * 8 + 7, mask=m, other=0).to(tl.int64)
        )
        s0 = w0 ^ rk[0]
        s1 = w1 ^ rk[1]
        s2 = 0 ^ rk[2]
        s3 = (blk + COUNTER0) ^ rk[3]

        for rnd in tl.static_range(1, 10):
            t0 = (
                _te(TBL, 0, (s0 >> 24) & 0xFF)
                ^ _te(TBL, 1, (s1 >> 16) & 0xFF)
                ^ _te(TBL, 2, (s2 >> 8) & 0xFF)
                ^ _te(TBL, 3, s3 & 0xFF)
                ^ rk[4 * rnd]
            )
            t1 = (
                _te(TBL, 0, (s1 >> 24) & 0xFF)
                ^ _te(TBL, 1, (s2 >> 16) & 0xFF)
                ^ _te(TBL, 2, (s3 >> 8) & 0xFF)
                ^ _te(TBL, 3, s0 & 0xFF)
                ^ rk[4 * rnd + 1]
            )
            t2 = (
                _te(TBL, 0, (s2 >> 24) & 0xFF)
                ^ _te(TBL, 1, (s3 >> 16) & 0xFF)
                ^ _te(TBL, 2, (s0 >> 8) & 0xFF)
                ^ _te(TBL, 3, s1 & 0xFF)
                ^ rk[4 * rnd + 2]
            )
            t3 = (
                _te(TBL, 0, (s3 >> 24) & 0xFF)
                ^ _te(TBL, 1, (s0 >> 16) & 0xFF)
                ^ _te(TBL, 2, (s1 >> 8) & 0xFF)
                ^ _te(TBL, 3, s2 & 0xFF)
                ^ rk[4 * rnd + 3]
            )
            s0, s1, s2, s3 = t0, t1, t2, t3

        # final round: S-box and ShiftRows, no MixColumns, then the last round key
        f0 = (
            (_sb(TBL, (s0 >> 24) & 0xFF) << 24)
            | (_sb(TBL, (s1 >> 16) & 0xFF) << 16)
            | (_sb(TBL, (s2 >> 8) & 0xFF) << 8)
            | _sb(TBL, s3 & 0xFF)
        ) ^ rk[40]
        f1 = (
            (_sb(TBL, (s1 >> 24) & 0xFF) << 24)
            | (_sb(TBL, (s2 >> 16) & 0xFF) << 16)
            | (_sb(TBL, (s3 >> 8) & 0xFF) << 8)
            | _sb(TBL, s0 & 0xFF)
        ) ^ rk[41]
        f2 = (
            (_sb(TBL, (s2 >> 24) & 0xFF) << 24)
            | (_sb(TBL, (s3 >> 16) & 0xFF) << 16)
            | (_sb(TBL, (s0 >> 8) & 0xFF) << 8)
            | _sb(TBL, s1 & 0xFF)
        ) ^ rk[42]
        f3 = (
            (_sb(TBL, (s3 >> 24) & 0xFF) << 24)
            | (_sb(TBL, (s0 >> 16) & 0xFF) << 16)
            | (_sb(TBL, (s1 >> 8) & 0xFF) << 8)
            | _sb(TBL, s2 & 0xFF)
        ) ^ rk[43]

        base = lane * (NBLK * 16) + blk * 16
        for j in tl.static_range(4):
            f = f0 if j == 0 else (f1 if j == 1 else (f2 if j == 2 else f3))
            for k in tl.static_range(4):
                tl.store(OUT + base + j * 4 + k, ((f >> (24 - 8 * k)) & 0xFF).to(tl.uint8), mask=m)


def aes128_ctr_torch(keys, nonces, length: int, counter0: int = 0, device=None, rk=None):
    """CTR keystream on the device: `keys` `[B,16]`, `nonces` `[B,8]` -> `[B, length]` uint8.

    The block is `nonce(8) || counter(8)` big-endian, so `counter0` is the low 64 bits of the
    first block -- i.e. `openssl enc -aes-128-ctr -iv <nonce || counter0>` agrees with it.
    """
    import torch

    from .. import devices as _devices

    dev = _devices.resolve_device(device)
    k_np = np.asarray(keys, dtype=np.uint8)
    n_np = np.asarray(nonces, dtype=np.uint8)
    if k_np.ndim != 2 or k_np.shape[1] != 16:
        raise ValueError("keys must be [B, 16]")
    if n_np.ndim != 2 or n_np.shape[1] != 8:
        raise ValueError("nonces must be [B, 8]")
    if k_np.shape[0] != n_np.shape[0]:
        raise ValueError("keys and nonces must have the same batch")
    batch = k_np.shape[0]
    nblk = (length + 15) // 16
    if batch == 0:
        return torch.zeros((0, length), dtype=torch.uint8, device=dev)
    if rk is None:
        # The key schedule is host-side and costs ~80 us per key: for 4096 streams that is
        # 0.3 s, which dwarfs the kernel. Callers that time the device path pass `rk`.
        rk_np = np.stack([aes128_expand_key_np(bytes(row)) for row in k_np]).astype(np.int32)
        rk = torch.as_tensor(rk_np, device=dev)
    nonce = torch.as_tensor(np.array(n_np, copy=True), device=dev)
    out = torch.empty((batch, nblk * 16), dtype=torch.uint8, device=dev)
    tbl = torch.as_tensor(T_TABLE_NP.reshape(-1), dtype=torch.int32, device=dev)
    blk = 64
    _aes_ctr_kernel[(triton.cdiv(batch, blk),)](
        rk, nonce, out, tbl, batch, COUNTER0=counter0, NBLK=nblk, BLK=blk
    )
    return out[:, :length]
