"""NumPy reference implementation of UOV public-key expansion and verification."""

from __future__ import annotations

import hashlib

import numpy as np

from ..mayo.ref import decode_nibbles
from ..primitives.aes import aes128_ctr_np
from .params import BY_NAME, UOVParams

__all__ = [
    "eval_public_map_np",
    "expand_pk_np",
    "verify_np",
]


def _gf16_mul_u32_np(a: np.ndarray, b: int) -> np.ndarray:
    a = a.astype(np.uint32, copy=False)
    r = a * np.uint32(b & 1)
    for shift in (1, 2, 3):
        msb = a & np.uint32(0x88888888)
        a = ((a ^ msb) << np.uint32(1)) ^ ((msb >> np.uint32(3)) * np.uint32(3))
        r ^= a * np.uint32((b >> shift) & 1)
    return r


def _gf256_mul_u32_np(a: np.ndarray, b: int) -> np.ndarray:
    a = a.astype(np.uint32, copy=False)
    r = a * np.uint32(b & 1)
    for shift in range(1, 8):
        msb = a & np.uint32(0x80808080)
        a = ((a ^ msb) << np.uint32(1)) ^ ((msb >> np.uint32(7)) * np.uint32(0x1B))
        r ^= a * np.uint32((b >> shift) & 1)
    return r


def expand_pk_np(ps: UOVParams, pk: bytes) -> np.ndarray:
    """Expand classic or `-pkc` UOV public key into row-major upper-triangular `[T, m_u32_words]` uint32.

    In liboqs/PQOV wire format, `P1` (`0 <= i <= j < v`), `P2` (`0 <= i < v, v <= j < n`), and `P3`
    (`v <= i <= j < n`) are concatenated sequentially. Reordering them once into standard row-major
    upper-triangular order `0 <= i <= j < n` lets the GPU evaluator traverse `P` sequentially.
    """
    if len(pk) == ps.cpk_bytes:
        p12 = aes128_ctr_np(
            pk[: ps.pk_seed_bytes], b"\x00" * 8, ps.p1_bytes + ps.p2_bytes, counter0=0
        )
        full = p12 + pk[ps.pk_seed_bytes :]
    elif len(pk) == ps.pk_bytes:
        full = pk
    else:
        raise ValueError(
            f"{ps.name}: expected {ps.pk_bytes} (classic) or {ps.cpk_bytes} (pkc) bytes, got {len(pk)}"
        )

    v, o, mb, mw = ps.v, ps.o, ps.m_bytes, ps.m_u32_words
    p1 = np.frombuffer(full[: ps.p1_bytes], dtype=np.uint8).reshape(v * (v + 1) // 2, mb)
    p2 = np.frombuffer(full[ps.p1_bytes : ps.p1_bytes + ps.p2_bytes], dtype=np.uint8).reshape(
        v, o, mb
    )
    p3 = np.frombuffer(full[ps.p1_bytes + ps.p2_bytes :], dtype=np.uint8).reshape(
        o * (o + 1) // 2, mb
    )

    out = np.empty((ps.num_tri, mb), dtype=np.uint8)
    dst = 0
    src1 = 0
    for i in range(v):
        cnt1 = v - i
        out[dst : dst + cnt1] = p1[src1 : src1 + cnt1]
        dst += cnt1
        src1 += cnt1
        out[dst : dst + o] = p2[i]
        dst += o
    out[dst:] = p3
    return out.view(np.uint32).reshape(ps.num_tri, mw).copy()


def eval_public_map_np(ps: UOVParams, p_tri_u32: np.ndarray, sig_x_bytes: bytes) -> bytes:
    """Evaluate UOV public map P(x) -> `m_bytes` bytes (oracle for GPU kernels)."""
    x = (
        decode_nibbles(sig_x_bytes, ps.n)
        if ps.q == 16
        else np.frombuffer(sig_x_bytes, dtype=np.uint8)
    )
    mul = _gf16_mul_u32_np if ps.q == 16 else _gf256_mul_u32_np
    mw = ps.m_u32_words
    acc = np.zeros(mw, dtype=np.uint32)
    idx = 0
    for i in range(ps.n):
        xi = int(x[i])
        cnt = ps.n - i
        if xi:
            row_acc = np.zeros(mw, dtype=np.uint32)
            for j in range(i, ps.n):
                xj = int(x[j])
                if xj:
                    row_acc ^= mul(p_tri_u32[idx], xj)
                idx += 1
            acc ^= mul(row_acc, xi)
        else:
            idx += cnt
    return acc.view(np.uint8).tobytes()


def verify_np(ps: UOVParams | str, pk: bytes, msg: bytes, sig: bytes) -> bool:
    """Single-signature UOV verification oracle."""
    if isinstance(ps, str):
        ps = BY_NAME[ps]
    if len(pk) not in (ps.pk_bytes, ps.cpk_bytes) or len(sig) != ps.sig_bytes:
        return False
    p_tri = expand_pk_np(ps, pk)
    y = eval_public_map_np(ps, p_tri, sig[: ps.n_bytes])
    salt = sig[ps.n_bytes :]
    t = hashlib.shake_256(msg + salt).digest(ps.m_bytes)
    return y == t
