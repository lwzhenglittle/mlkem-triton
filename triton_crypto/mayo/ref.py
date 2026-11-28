"""NumPy reference implementation of MAYO public-key expansion and verification."""

from __future__ import annotations

import hashlib

import numpy as np

from ..primitives.aes import aes128_ctr_np
from .params import BY_NAME, MayoParams

__all__ = [
    "GF16_MUL_TAB",
    "decode_nibbles",
    "encode_nibbles",
    "eval_public_map_np",
    "expand_pk_np",
    "verify_np",
]


def _gf16_mul_scalar(a: int, b: int) -> int:
    p = ((a & 1) * b) ^ ((a & 2) * b) ^ ((a & 4) * b) ^ ((a & 8) * b)
    top = p & 0xF0
    return (p ^ (top >> 4) ^ (top >> 3)) & 0xF


GF16_MUL_TAB = np.array(
    [[_gf16_mul_scalar(a, b) for b in range(16)] for a in range(16)], dtype=np.uint8
)


def decode_nibbles(data: bytes | np.ndarray, count: int) -> np.ndarray:
    """Decode `count` GF(16) nibbles (low nibble first) from bytes."""
    arr = (
        np.frombuffer(data, dtype=np.uint8)
        if isinstance(data, (bytes, bytearray, memoryview))
        else np.asarray(data, dtype=np.uint8)
    )
    out = np.empty(count, dtype=np.uint8)
    pairs = count // 2
    out[0 : 2 * pairs : 2] = arr[:pairs] & 0x0F
    out[1 : 2 * pairs : 2] = arr[:pairs] >> 4
    if count & 1:
        out[-1] = arr[pairs] & 0x0F
    return out


def encode_nibbles(nibbles: np.ndarray) -> bytes:
    """Encode GF(16) nibbles (low nibble first) into bytes."""
    nib = np.asarray(nibbles, dtype=np.uint8) & 0x0F
    count = nib.size
    pairs = count // 2
    out = np.empty((count + 1) // 2, dtype=np.uint8)
    out[:pairs] = nib[0 : 2 * pairs : 2] | (nib[1 : 2 * pairs : 2] << 4)
    if count & 1:
        out[pairs] = nib[-1]
    return out.tobytes()


def expand_pk_np(ps: MayoParams, cpk: bytes) -> np.ndarray:
    """Expand compact MAYO public key `cpk` into packed upper-triangular `[n, n, m_u32_words]` uint32.

    Each `uint32` word packs 8 GF(16) nibbles in little-endian nibble order (bits `4*i .. 4*i+3`).
    """
    if len(cpk) != ps.cpk_bytes:
        raise ValueError(f"{ps.name}: expected {ps.cpk_bytes}-byte cpk, got {len(cpk)}")
    p12_raw = aes128_ctr_np(
        cpk[: ps.pk_seed_bytes], b"\x00" * 8, ps.p1_bytes + ps.p2_bytes, counter0=0
    )
    p3_raw = cpk[ps.pk_seed_bytes :]

    n, v, o, m_bytes = ps.n, ps.v, ps.o, ps.m_bytes
    w = ps.m_u32_words
    p_bytes = np.zeros((n, n, w * 4), dtype=np.uint8)

    # P1: upper triangle v x v
    p1_vecs = np.frombuffer(p12_raw[: ps.p1_bytes], dtype=np.uint8).reshape(
        v * (v + 1) // 2, m_bytes
    )
    idx = 0
    for r in range(v):
        cnt = v - r
        p_bytes[r, r:v, :m_bytes] = p1_vecs[idx : idx + cnt]
        idx += cnt

    # P2: v x o block
    p2_vecs = np.frombuffer(p12_raw[ps.p1_bytes :], dtype=np.uint8).reshape(v, o, m_bytes)
    p_bytes[:v, v:n, :m_bytes] = p2_vecs

    # P3: upper triangle o x o
    p3_vecs = np.frombuffer(p3_raw, dtype=np.uint8).reshape(o * (o + 1) // 2, m_bytes)
    idx = 0
    for r in range(o):
        cnt = o - r
        p_bytes[v + r, v + r : n, :m_bytes] = p3_vecs[idx : idx + cnt]
        idx += cnt

    return p_bytes.view(np.uint32).reshape(n, n, w).copy()


def _gf16_mul_u32_np(a: np.ndarray, b: int) -> np.ndarray:
    a = a.astype(np.uint32, copy=False)
    r = a * np.uint32(b & 1)
    for shift in (1, 2, 3):
        msb = a & np.uint32(0x88888888)
        a = ((a ^ msb) << np.uint32(1)) ^ ((msb >> np.uint32(3)) * np.uint32(3))
        r ^= a * np.uint32((b >> shift) & 1)
    return r


def eval_public_map_np(ps: MayoParams, p_u32: np.ndarray, s_nibbles: np.ndarray) -> np.ndarray:
    """Evaluate MAYO public map P*(s) -> `[m]` GF(16) nibbles (oracle for GPU kernels)."""
    n, k, m, w = ps.n, ps.k, ps.m, ps.m_u32_words
    s = np.asarray(s_nibbles, dtype=np.uint8).reshape(k, n)
    ps_arr = np.zeros((n, k, w), dtype=np.uint32)
    for col in range(k):
        for r in range(n):
            acc = np.zeros(w, dtype=np.uint32)
            for c in range(r, n):
                sv = int(s[col, c])
                if sv:
                    acc ^= _gf16_mul_u32_np(p_u32[r, c], sv)
            ps_arr[r, col] = acc

    sps = np.zeros((k, k, w), dtype=np.uint32)
    for i in range(k):
        for col in range(k):
            acc = np.zeros(w, dtype=np.uint32)
            for r in range(n):
                sv = int(s[i, r])
                if sv:
                    acc ^= _gf16_mul_u32_np(ps_arr[r, col], sv)
            sps[i, col] = acc

    # Unpack to nibbles and fold mod f(z)
    sps_nib = np.empty((k, k, m), dtype=np.uint8)
    for i in range(k):
        for j in range(k):
            sps_nib[i, j] = decode_nibbles(sps[i, j].view(np.uint8), m)

    y = np.zeros(m, dtype=np.uint8)
    for i in range(k - 1, -1, -1):
        for j in range(i, k):
            top = int(y[m - 1])
            y[1:] = y[:-1]
            y[0] = 0
            for jj, ft in enumerate(ps.f_tail):
                y[jj] ^= GF16_MUL_TAB[top, ft]
            y ^= sps_nib[i, j]
            if i != j:
                y ^= sps_nib[j, i]
    return y


def verify_np(ps: MayoParams | str, cpk: bytes, msg: bytes, sig: bytes) -> bool:
    """Single-signature MAYO verification oracle."""
    if isinstance(ps, str):
        ps = BY_NAME[ps]
    if len(cpk) != ps.cpk_bytes or len(sig) != ps.sig_bytes:
        return False
    p_u32 = expand_pk_np(ps, cpk)
    md = hashlib.shake_256(msg).digest(ps.digest_bytes)
    salt = sig[ps.sig_bytes - ps.salt_bytes :]
    t_enc = hashlib.shake_256(md + salt).digest(ps.m_bytes)
    t = decode_nibbles(t_enc, ps.m)
    s = decode_nibbles(sig, ps.k * ps.n)
    y = eval_public_map_np(ps, p_u32, s)
    return bool(np.array_equal(y, t))
