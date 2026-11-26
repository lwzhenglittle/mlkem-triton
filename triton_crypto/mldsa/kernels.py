"""Triton kernels for ML-DSA signature verification (FIPS 204).

Everything here is data-layout-specific to ML-DSA and complements the two generic
primitives (`primitives.sponge` for the SHAKE calls, `primitives.ntt` for the transform
over q = 8380417):

- `unpack_z_kernel`      BitUnpack_{γ1−1} of a whole batch of z vectors
- `mac_kernel`           ŵ_i = Σ_j Â[i][j]·ẑ[j] − ĉ·t̂1_i, in the NTT domain
- `sample_in_ball_kernel` SampleInBall (Alg. 29): sequential, so one program per signature
- `hint_kernel`          HintBitUnpack (Alg. 20) into a dense 0/1 mask + validity bits
- `w1_pack_kernel`       Decompose + UseHint + BitPack_ω1 in one pass

The arithmetic is int64 on purpose: q = 8380417 is 23 bits, so a product of two reduced
coefficients is 46 bits — fp16/fp32 tensor cores cannot carry it exactly (this is exactly
why the project's Toeplitz-GEMM trick, which needs one *small* operand, does not transfer
from ML-KEM to ML-DSA; see docs/design.md §"ML-DSA").
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = [
    "hint_kernel",
    "mac_kernel",
    "pack_z_kernel",
    "pointwise_kernel",
    "sample_in_ball_kernel",
    "unpack_z_kernel",
    "w1_pack_kernel",
]

# q must be a tl.constexpr to be visible inside kernels
Q = tl.constexpr(8380417)


@triton.jit
def unpack_z_kernel(
    SIG,
    OUT,
    SIGSTRIDE,
    SIGROWS,
    ZOFF: tl.constexpr,
    POLYBYTES: tl.constexpr,
    D: tl.constexpr,
    GAMMA1: tl.constexpr,
    N: tl.constexpr,
    NL: tl.constexpr,
):
    """One program per (signature, polynomial): [B, SIG] uint8 -> [B*l, N] int32.

    z is stored as γ1 − z in `D`-bit little-endian fields, so unpacking is a subtract.
    For D ∈ {18, 20} the fields never straddle more than 3 bytes (18·k mod 8 ∈ {0,2,4,6}
    spans 24 bits), so three loads and a shift are enough.
    """
    pid = tl.program_id(0)
    i = tl.arange(0, N)
    bitpos = i * D
    poly = tl.program_id(1)
    boff = ZOFF + poly * POLYBYTES + bitpos // 8
    sh = (bitpos % 8).to(tl.int32)
    row = SIG + pid * SIGSTRIDE
    v0 = tl.load(row + boff).to(tl.int64)
    v1 = tl.load(row + boff + 1).to(tl.int64)
    v2 = tl.load(row + boff + 2).to(tl.int64)
    val = ((v0 | (v1 << 8) | (v2 << 16)) >> sh) & ((1 << D) - 1)
    tl.store(OUT + (pid * NL + poly) * N + i, (GAMMA1 - val).to(tl.int32))


@triton.jit
def mac_kernel(
    Z,
    A,
    CH,
    T1,
    OUT,
    B,
    K: tl.constexpr,
    L: tl.constexpr,
    N: tl.constexpr,
    Q: tl.constexpr,
    BN: tl.constexpr,
):
    """ŵ_i = Σ_j Â[i][j]·ẑ[j] − ĉ·t̂1_i (FIPS 204 Alg. 33/11), all mod q.

    Grid: (rows of the batch, i). Â and t̂1 are resident (they belong to the public key),
    so the only batch-sized streams are ẑ and the output. Products reach q² ≈ 7·10¹³,
    hence int64.
    """
    pid = tl.program_id(0)
    i = tl.program_id(1)
    rows = pid * BN + tl.arange(0, BN)
    m = rows < B
    n = tl.arange(0, N)

    acc = tl.zeros((BN, N), dtype=tl.int64)
    for j in tl.static_range(L):
        z = tl.load(Z + (rows * L + j)[:, None] * N + n[None, :], mask=m[:, None], other=0)
        a = tl.load(A + (i * L + j) * N + n)
        acc += z.to(tl.int64) * a.to(tl.int64)
    ch = tl.load(CH + rows[:, None] * N + n[None, :], mask=m[:, None], other=0).to(tl.int64)
    t1 = tl.load(T1 + i * N + n).to(tl.int64)
    acc -= ch * t1
    acc = acc % Q
    acc = tl.where(acc < 0, acc + Q, acc)
    tl.store(OUT + (rows * K + i)[:, None] * N + n[None, :], acc.to(tl.int32), mask=m[:, None])


@triton.jit
def sample_in_ball_kernel(BUF, OUT, N: tl.constexpr, TAU: tl.constexpr, RATE: tl.constexpr):
    """SampleInBall (FIPS 204 Alg. 29): one program per signature.

    The algorithm is inherently sequential — c[i] ← c[j] then c[j] ← ±1 with a
    rejection-sampled j — so each program keeps the 256 coefficients as one vector and
    updates them with masked selects instead of scatters. `while` handles the rejection:
    a fixed unroll would have to assume a bound that the standard does not give.
    """
    pid = tl.program_id(0)
    base = BUF + pid * RATE
    idx = tl.arange(0, N)
    # the first 8 bytes are the sign bits, little-endian
    s = tl.zeros((), dtype=tl.int64)
    for t in tl.static_range(8):
        s = s | (tl.load(base + t).to(tl.int64) << (8 * t))

    c = tl.zeros((N,), dtype=tl.int32)
    pos = 8
    for step in tl.static_range(TAU):
        i = N - TAU + step
        j = tl.load(base + pos).to(tl.int32)
        pos = pos + 1
        while j > i:
            j = tl.load(base + pos).to(tl.int32)
            pos = pos + 1
        old = tl.sum(tl.where(idx == j, c, 0), axis=0)
        c = tl.where(idx == i, old, c)
        sign = 1 - 2 * (s & 1).to(tl.int32)
        c = tl.where(idx == j, sign, c)
        s = s >> 1
    tl.store(OUT + pid * N + idx, c)


@triton.jit
def hint_kernel(
    SIG,
    H,
    OK,
    SIGSTRIDE,
    HOFF: tl.constexpr,
    OMEGA: tl.constexpr,
    OMEGA_P2: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """HintBitUnpack (FIPS 204 Alg. 20): one program per (signature, i).

    Writes a dense 0/1 mask for row i (H must be zero-initialised) and one validity bit
    per (signature, i): counts must be monotone and ≤ ω, indices strictly increasing, and
    the tail of the last row must be zero.
    """
    pid = tl.program_id(0)
    i = tl.program_id(1)
    row = SIG + pid * SIGSTRIDE + HOFF
    p = tl.arange(0, OMEGA_P2)
    live = p < OMEGA  # ω is 55/75/80, not a power of two
    cnt = tl.load(row + OMEGA + i).to(tl.int32)
    prev = tl.where(i == 0, 0, tl.load(row + OMEGA + i - 1).to(tl.int32))

    idx = tl.load(row + p, mask=live, other=0).to(tl.int32)
    prev_idx = tl.load(row + p - 1, mask=(p >= 1) & live, other=-1).to(tl.int32)
    used = live & (p >= prev) & (p < cnt)

    ok = (cnt >= prev) & (cnt <= OMEGA)
    ok = ok & tl.sum(((used & (p >= prev + 1) & (idx <= prev_idx)).to(tl.int32)), axis=0) == 0

    if i == K - 1:
        tail = tl.sum(tl.load(row + p, mask=live & (p >= cnt), other=0).to(tl.int32), axis=0)
        ok = ok & (tail == 0)

    tl.store(H + (pid * K + i) * N + idx, 1, mask=used)
    tl.store(OK + pid * K + i, ok.to(tl.int32))


@triton.jit
def _use_hint(
    w,
    h,
    Q: tl.constexpr,
    TWO_G2: tl.constexpr,
    G2: tl.constexpr,
    M: tl.constexpr,
):
    """Decompose_γ2 + UseHint_γ2 for one coefficient (FIPS 204 Alg. 36/37)."""
    r = w % Q
    r0 = r % TWO_G2
    r0 = tl.where(r0 > G2, r0 - TWO_G2, r0)
    r1 = (r - r0) // TWO_G2
    special = (r - r0) == Q - 1
    r1 = tl.where(special, 0, r1)
    r0 = tl.where(special, r0 - 1, r0)
    up = (r1 + 1) % M
    down = (r1 - 1) % M
    down = tl.where(down < 0, down + M, down)
    hint_val = tl.where(r0 > 0, up, down)
    return tl.where(h != 0, hint_val, r1)


@triton.jit
def w1_pack_kernel(
    W,
    H,
    OUT,
    B,
    K: tl.constexpr,
    N: tl.constexpr,
    MB: tl.constexpr,
    TWO_G2: tl.constexpr,
    G2: tl.constexpr,
    M: tl.constexpr,
    POLYB: tl.constexpr,
    BC: tl.constexpr,
):
    """Decompose → UseHint → BitPack (FIPS 204 Alg. 36/37/23) in one pass.

    One program packs `BC` bytes of one polynomial. A `MB`-bit field spans at most two
    coefficients (MB ∈ {4,6}), which is why two loads suffice.
    """
    pid = tl.program_id(0)
    i = tl.program_id(1)
    j = tl.program_id(2) * BC + tl.arange(0, BC)
    bitpos = j * 8
    c0 = bitpos // MB
    sh = (bitpos % MB).to(tl.int32)
    c1 = tl.minimum(c0 + 1, N - 1)

    base = (pid * K + i) * N
    live = j < POLYB  # polyw1_bytes is 128 or 192, i.e. not always a multiple of BC
    w0 = tl.load(W + base + c0, mask=live, other=0)
    w1 = tl.load(W + base + c1, mask=live, other=0)
    h0 = tl.load(H + base + c0, mask=live, other=0)
    h1 = tl.load(H + base + c1, mask=live, other=0)
    r0 = _use_hint(w0, h0, Q, TWO_G2, G2, M)
    r1 = _use_hint(w1, h1, Q, TWO_G2, G2, M)
    byte = ((r0 >> sh) | (r1 << (MB - sh))) & 0xFF
    tl.store(OUT + (pid * K + i) * POLYB + j, byte.to(tl.uint8), mask=live)


@triton.jit
def pack_z_kernel(
    Z,
    OUT,
    B,
    POLYB: tl.constexpr,
    D: tl.constexpr,
    GAMMA1: tl.constexpr,
    N: tl.constexpr,
    BC: tl.constexpr,
):
    """BitPack of z as γ1 − z in `D`-bit little-endian fields (FIPS 204 Alg. 22).

    One program per (signature, polynomial) and one program id per block of output bytes.
    A `D`-bit field spans at most two bytes (D ∈ {18, 20}), so two coefficient loads
    suffice — the same shape as `w1_pack_kernel`, but this one is on the signing path,
    where it replaces a numpy bit-pack that cost ~19 % of the total time.
    """
    pid = tl.program_id(0)
    j = tl.program_id(1) * BC + tl.arange(0, BC)
    live = (j < POLYB) & (pid < B)
    bitpos = j * 8
    c0 = bitpos // D
    sh = (bitpos % D).to(tl.int32)
    c1 = tl.minimum(c0 + 1, N - 1)
    base = pid * N
    v0 = tl.load(Z + base + c0, mask=live, other=0).to(tl.int64)
    v1 = tl.load(Z + base + c1, mask=live, other=0).to(tl.int64)
    a0 = (GAMMA1 - v0) & (2 * GAMMA1 - 1)
    a1 = (GAMMA1 - v1) & (2 * GAMMA1 - 1)
    byte = ((a0 >> sh) | (a1 << (D - sh))) & 0xFF
    tl.store(OUT + pid * POLYB + j, byte.to(tl.uint8), mask=live)


@triton.jit
def pointwise_kernel(
    A,
    B,
    OUT,
    ROWS,
    N: tl.constexpr,
    Q: tl.constexpr,
    BN: tl.constexpr,
):
    """ĉ ∘ ŝ mod q, one program over `BN` rows of 256 coefficients.

    The host version broadcasts an int64 product and then reduces — fine, but it
    materialises a 64-bit array of the whole batch three times per signing attempt.
    This keeps the product in registers and writes int32 back.
    """
    pid = tl.program_id(0)
    row = pid * BN + tl.arange(0, BN)
    col = tl.arange(0, N)
    m = row < ROWS
    a = tl.load(A + row[:, None] * N + col[None, :], mask=m[:, None], other=0)
    b = tl.load(B + row[:, None] * N + col[None, :], mask=m[:, None], other=0)
    v = (a.to(tl.int64) * b.to(tl.int64)) % Q
    v = tl.where(v < 0, v + Q, v)
    tl.store(OUT + row[:, None] * N + col[None, :], v.to(tl.int32), mask=m[:, None])
