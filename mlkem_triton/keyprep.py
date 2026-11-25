"""Phase 2: private-key preprocessing — fold the polynomial operations of
decapsulation into two fixed matrices.

W_enc: [256(k+1), 256k], used for the FO re-encryption:
    [u'; v'] = W_enc @ y + [e1; e2 + Decompress_1(m')]
    row block i (i<k), column block j = T(A_coef[j][i])   (Aᵀ, note the transpose)
    row block k,       column block j = T(t_coef[j])

W_dec: [256, 256k], used for decryption:
    w = v - W_dec @ u
    column block j = T(s_coef[j])

where T(a)[i][j] = a[(i-j) mod 256] * (+1 if i >= j else -1),
satisfying T(a) @ b == a * b in Z_q[X]/(X^256+1) (see tests/test_ntt.py).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from .ntt_ref import byte_decode, intt, multiply_ntts, ntt, toeplitz
from .params import PARAMS, N, Q

HALF = (Q - 1) // 2  # 1664


# ---------------- Basics ----------------


def center(x) -> np.ndarray:
    """Z_q -> [-1664, 1664] centered representative."""
    x = np.asarray(x, dtype=np.int64) % Q
    return np.where(x > HALF, x - Q, x)


def sample_ntt(rho: bytes, j: int, i: int) -> np.ndarray:
    """Â[i][j] = SampleNTT(rho ‖ j ‖ i); returns NTT-domain coefficients
    (FIPS 203 Alg. 7 + 8)."""
    buf = hashlib.shake_128(rho + bytes([j, i])).digest(840)
    out = np.empty(N, dtype=np.int64)
    cnt = 0
    p = 0
    while cnt < N:
        d1 = buf[p] + 256 * (buf[p + 1] % 16)
        d2 = (buf[p + 1] // 16) + 16 * buf[p + 2]
        p += 3
        if d1 < Q:
            out[cnt] = d1
            cnt += 1
        if d2 < Q and cnt < N:
            out[cnt] = d2
            cnt += 1
    return out


def expand_matrix_hat(rho: bytes, k: int) -> np.ndarray:
    """Returns Â, shape [k][k][256], in the NTT domain."""
    A = np.empty((k, k, N), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            A[i, j] = sample_ntt(rho, j, i)
    return A


# ---------------- Parsing and construction ----------------


@dataclass
class KeyPrep:
    ps: str
    k: int
    eta1: int
    eta2: int
    rho: bytes
    A_hat: np.ndarray  # [k][k][N] NTT domain
    A_coef: np.ndarray  # [k][k][N] coefficient domain (mod q)
    t_hat: np.ndarray  # [k][N]
    t_coef: np.ndarray  # [k][N]
    s_hat: np.ndarray  # [k][N]
    s_coef: np.ndarray  # [k][N]
    W_enc: np.ndarray = field(repr=False)  # [256(k+1), 256k] centered integers
    W_dec: np.ndarray = field(repr=False)  # [256, 256k] centered integers

    # ---- Convenience properties ----
    @property
    def M(self) -> int:
        return 256 * (self.k + 1)

    @property
    def K(self) -> int:
        return 256 * self.k


def build_W_enc(A_coef: np.ndarray, t_coef: np.ndarray, k: int) -> np.ndarray:
    W = np.zeros((256 * (k + 1), 256 * k), dtype=np.int64)
    for i in range(k):
        for j in range(k):
            # u'_i = sum_j A[j][i] * y_j  → row block i, column block j uses A_coef[j][i]
            W[256 * i : 256 * (i + 1), 256 * j : 256 * (j + 1)] = toeplitz(center(A_coef[j][i]))
    for j in range(k):
        W[256 * k : 256 * (k + 1), 256 * j : 256 * (j + 1)] = toeplitz(center(t_coef[j]))
    return W


def build_W_dec(s_coef: np.ndarray, k: int) -> np.ndarray:
    W = np.zeros((256, 256 * k), dtype=np.int64)
    for j in range(k):
        W[:, 256 * j : 256 * (j + 1)] = toeplitz(center(s_coef[j]))
    return W


def parse_key(ps: str, dk: bytes, verify_hash: bool = True) -> KeyPrep:
    """Parse dk, expand Â, apply INTT, and build W_enc / W_dec.

    dk = dk_PKE(384k) ‖ ek(384k+32) ‖ H(ek)(32) ‖ z(32)
    ek = ByteEncode12(t̂) ‖ ρ
    """
    p = PARAMS[ps]
    k = p.k
    # Accept bytes-like or an array of uint8. Without this, `parse_key` on a
    # numpy array returns array views for `ek` / `h_ek`, and the hash check below
    # compares `bytes != ndarray`, which yields an array and makes `if` raise.
    if not isinstance(dk, (bytes, bytearray)):
        dk = np.asarray(dk, dtype=np.uint8).tobytes()
    if len(dk) != p.dk_size:
        raise ValueError(f"{ps}: dk length should be {p.dk_size}, got {len(dk)}")

    dk_pke = dk[: 384 * k]
    ek = dk[384 * k : 768 * k + 32]
    h_ek = dk[768 * k + 32 : 768 * k + 64]
    # z is in here too (the implicit-rejection key); `KeyPrep` does not use it,
    # `pipeline` takes it from dk directly.

    if verify_hash and hashlib.sha3_256(ek).digest() != h_ek:
        raise ValueError("dk hash check failed")

    rho = ek[384 * k :]
    if len(rho) != 32:
        raise ValueError("invalid rho length")

    t_hat = np.stack([byte_decode(ek[384 * j : 384 * (j + 1)], 12) for j in range(k)])
    s_hat = np.stack([byte_decode(dk_pke[384 * j : 384 * (j + 1)], 12) for j in range(k)])

    A_hat = expand_matrix_hat(rho, k)
    A_coef = np.stack([[intt(A_hat[i, j]) for j in range(k)] for i in range(k)])
    t_coef = np.stack([intt(t_hat[j]) for j in range(k)])
    s_coef = np.stack([intt(s_hat[j]) for j in range(k)])

    return KeyPrep(
        ps=ps,
        k=k,
        eta1=p.eta1,
        eta2=p.eta2,
        rho=rho,
        A_hat=A_hat,
        A_coef=A_coef,
        t_hat=t_hat,
        t_coef=t_coef,
        s_hat=s_hat,
        s_coef=s_coef,
        W_enc=build_W_enc(A_coef, t_coef, k),
        W_dec=build_W_dec(s_coef, k),
    )


@dataclass
class EncKeyPrep:
    """Everything derived from the public key `ek` alone — the half the
    encapsulation side needs."""

    ps: str
    k: int
    rho: bytes
    h_ek: bytes  # H(ek), 32 B
    t_hat: np.ndarray
    A_hat: np.ndarray
    W_enc: np.ndarray = field(repr=False)  # [256(k+1), 256k] centered integers


def parse_ek(ps: str, ek: bytes) -> EncKeyPrep:
    """Parse a public key, expand Â, and build the encapsulation `W_enc` (same
    formulas as the one in `parse_key`).

    `ek = ByteEncode12(t̂) ‖ ρ`, length `384k + 32`.
    Encapsulation only needs `W_enc`: `[u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]`.
    """
    p = PARAMS[ps]
    k = p.k
    # Accept bytes-like or an array of uint8; see the note in `parse_key`. Here the
    # array form would reach `sample_ntt`, where `rho` must support `bytes`
    # concatenation rather than numpy addition.
    if not isinstance(ek, (bytes, bytearray)):
        ek = np.asarray(ek, dtype=np.uint8).tobytes()
    if len(ek) != p.ek_size:
        raise ValueError(f"{ps}: ek length should be {p.ek_size}, got {len(ek)}")
    t_hat = np.stack([byte_decode(ek[384 * j : 384 * (j + 1)], 12) for j in range(k)])
    rho = ek[384 * k :]

    A_hat = expand_matrix_hat(rho, k)
    A_coef = np.stack([[intt(A_hat[i, j]) for j in range(k)] for i in range(k)])
    t_coef = np.stack([intt(t_hat[j]) for j in range(k)])

    return EncKeyPrep(
        ps=ps,
        k=k,
        rho=rho,
        h_ek=hashlib.sha3_256(ek).digest(),
        t_hat=t_hat,
        A_hat=A_hat,
        W_enc=build_W_enc(A_coef, t_coef, k),
    )


# ---------------- NTT reference path (for cross-checking) ----------------


def ref_reencrypt(kp: KeyPrep, y):
    """NTT path: u' = INTT(Âᵀ ∘ NTT(y)), v' = INTT(t̂ᵀ ∘ NTT(y))."""
    y = np.asarray(y, dtype=np.int64) % Q
    y_ntt = [ntt(y[j]) for j in range(kp.k)]

    u = np.empty((kp.k, N), dtype=np.int64)
    for i in range(kp.k):
        acc = np.zeros(N, dtype=np.int64)
        for j in range(kp.k):
            acc = (acc + multiply_ntts(kp.A_hat[j, i], y_ntt[j])) % Q
        u[i] = intt(acc)

    v_ntt = np.zeros(N, dtype=np.int64)
    for j in range(kp.k):
        v_ntt = (v_ntt + multiply_ntts(kp.t_hat[j], y_ntt[j])) % Q
    return u, intt(v_ntt)


def ref_decrypt_product(kp: KeyPrep, u):
    """NTT path: sᵀu = INTT(ŝᵀ ∘ NTT(u))."""
    u = np.asarray(u, dtype=np.int64) % Q
    acc = np.zeros(N, dtype=np.int64)
    for j in range(kp.k):
        acc = (acc + multiply_ntts(kp.s_hat[j], ntt(u[j]))) % Q
    return intt(acc)


# ---------------- Quantization helpers ----------------


def to_fp16(W: np.ndarray) -> np.ndarray:
    """Convert to fp16 and assert there is no precision loss (needs |W| <= 2048)."""
    import torch

    w = torch.from_numpy(np.ascontiguousarray(W))
    assert w.abs().max().item() <= 2048, f"|W|max={w.abs().max().item()} out of fp16 integer range"
    return w.to(torch.float16)


def split_limb(W: np.ndarray):
    """Split into int8 limbs: W = 64*hi + lo, hi ∈ [-26,26], lo ∈ [-32,31]."""
    W = np.asarray(W, dtype=np.int64)
    hi = np.floor_divide(W + 32, 64)
    lo = W - 64 * hi
    assert hi.min() >= -26 and hi.max() <= 26, (hi.min(), hi.max())
    assert lo.min() >= -32 and lo.max() <= 31, (lo.min(), lo.max())
    return hi.astype(np.int8), lo.astype(np.int8)


__all__ = [
    "EncKeyPrep",
    "KeyPrep",
    "build_W_dec",
    "build_W_enc",
    "center",
    "expand_matrix_hat",
    "parse_ek",
    "parse_key",
    "ref_decrypt_product",
    "ref_reencrypt",
    "sample_ntt",
    "split_limb",
    "to_fp16",
]
