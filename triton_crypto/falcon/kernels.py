"""Falcon verification kernels (Falcon-512 and Falcon-1024).

The transform reuses `primitives.ntt` (`q = 12289`, `N = 512` or `1024`, int32 -- no new NTT
kernel needed). The Falcon-specific device stages are:

    _pack_sig_words_kernel  pack variable-length signature bodies into transposed [W, B] uint32
    _comp_decode_kernel     register-buffered 64-bit prefix decoder (one lane per signature,
                            branchless 16-bit CLZ, strict + optional padded zero-tail check)
    _pad_nonce_msg_kernel   assemble and SHAKE256-pad `nonce || msg` on the device
    _hash_to_point_kernel   read SHAKE256 16-bit samples and compact survivors via prefix sum
    _mul_ntt_kernel         pointwise product in the NTT domain, `s2_hat * h_hat mod q`
    _check_kernel           `s1 = c - s2*h`, normalise into [-q/2, q/2], and the L2 bound

`_check_kernel` computes the squared norm in int64: with |s1| <= q/2 = 6144 and n <= 1024 the
squared norm is below 2^36, so the comparison against `l2bound` (34 034 726 for 512,
70 265 242 for 1024) is exact without saturation arithmetic.
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = [
    "_check_kernel",
    "_comp_decode_kernel",
    "_hash_to_point_kernel",
    "_mul_ntt_kernel",
    "_pack_sig_words_kernel",
    "_pad_nonce_msg_kernel",
    "check_np",
]


@triton.jit
def _pack_sig_words_kernel(
    FLAT_SIGS,
    SIG_OFFS,
    BODY_LENS,
    WORDS_T,
    B,
    BODY_SKIP: tl.constexpr,
    W: tl.constexpr,
    BLK_B: tl.constexpr,
    BLK_W: tl.constexpr,
):
    """Pack signature bodies from `FLAT_SIGS[SIG_OFFS[b] + BODY_SKIP : +BODY_LENS[b]]` into
    transposed `[W, B]` big-endian uint32 words (zero-padded beyond `BODY_LENS[b]`).

    Transposing to `[W, B]` makes the subsequent `_comp_decode_kernel` refills coalesced across
    lanes and guarantees trailing bytes past `BODY_LENS[b]` are zero.
    """
    pb = tl.program_id(0)
    pw = tl.program_id(1)
    b = pb * BLK_B + tl.arange(0, BLK_B)
    w = pw * BLK_W + tl.arange(0, BLK_W)
    mb = b < B
    mw = w < W
    off = tl.load(SIG_OFFS + b, mask=mb, other=0) + BODY_SKIP
    length = tl.load(BODY_LENS + b, mask=mb, other=0)

    byte_rel = w[:, None] * 4
    base = off[None, :] + byte_rel
    m = mw[:, None] & mb[None, :]
    b0 = tl.load(FLAT_SIGS + base + 0, mask=m & (byte_rel + 0 < length[None, :]), other=0).to(
        tl.uint32
    )
    b1 = tl.load(FLAT_SIGS + base + 1, mask=m & (byte_rel + 1 < length[None, :]), other=0).to(
        tl.uint32
    )
    b2 = tl.load(FLAT_SIGS + base + 2, mask=m & (byte_rel + 2 < length[None, :]), other=0).to(
        tl.uint32
    )
    b3 = tl.load(FLAT_SIGS + base + 3, mask=m & (byte_rel + 3 < length[None, :]), other=0).to(
        tl.uint32
    )
    word = (b0 << 24) | (b1 << 16) | (b2 << 8) | b3
    tl.store(WORDS_T + w[:, None] * B + b[None, :], word, mask=m)


@triton.jit
def _comp_decode_kernel(
    WORDS_T,
    LENS,
    OUT,
    OK,
    B,
    W: tl.constexpr,
    N: tl.constexpr,
    MAX_MAG: tl.constexpr,
    PADDED_BODY_LEN: tl.constexpr,
    ALLOW_PADDED: tl.constexpr,
    BLK: tl.constexpr,
):
    """Decode `B` Falcon signature bodies on device: one SIMD lane per signature.

    Each lane keeps a 64-bit bit reservoir (`acc`, `acc_len` in [0, 55]) and refills one
    big-endian 32-bit word from transposed `[W, B]` whenever `acc_len < 24` (~32% of steps).
    Unary run lengths are computed via a 4-stage branchless 16-bit CLZ (zero lookup tables).
    """
    pid = tl.program_id(0)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B

    length = tl.load(LENS + lane, mask=m, other=0)
    max_bits = length * 8

    acc = tl.zeros([BLK], dtype=tl.uint64)
    acc_len = tl.zeros([BLK], dtype=tl.int32)
    wpos = tl.zeros([BLK], dtype=tl.int32)
    bit_pos = tl.zeros([BLK], dtype=tl.int32)
    ok = m

    for u in range(N):
        need = acc_len < 24
        w = tl.load(WORDS_T + wpos * B + lane, mask=m & need & (wpos < W), other=0).to(tl.uint64)
        acc = tl.where(need, (acc << 32) | w, acc)
        acc_len = tl.where(need, acc_len + 32, acc_len)
        wpos = tl.where(need, wpos + 1, wpos)

        shift = (acc_len - 24).to(tl.uint64)
        top24 = ((acc >> shift) & 0xFFFFFF).to(tl.int32)
        head = (top24 >> 16) & 0xFF
        cont = top24 & 0xFFFF

        # Branchless 16-bit leading-zero count for the unary run (cont == 0 caught by `bad`).
        c = cont
        k = tl.zeros([BLK], dtype=tl.int32)
        s = (c >> 8) == 0
        k = tl.where(s, k + 8, k)
        c = tl.where(s, c << 8, c)
        s = (c >> 12) == 0
        k = tl.where(s, k + 4, k)
        c = tl.where(s, c << 4, c)
        s = (c >> 14) == 0
        k = tl.where(s, k + 2, k)
        c = tl.where(s, c << 2, c)
        s = (c >> 15) == 0
        k = tl.where(s, k + 1, k)

        sign = (head & 0x80) != 0
        mag = (head & 0x7F) + 128 * k
        used = 9 + k
        bit_pos = bit_pos + used
        acc_len = acc_len - used

        bad = (cont == 0) | (mag > MAX_MAG) | (sign & (mag == 0)) | (bit_pos > max_bits)
        ok = ok & (~bad)
        val = tl.where(sign, -mag, mag)
        tl.store(OUT + lane * N + u, val, mask=m)

    # Unused bits of the last consumed byte sit at the top of the remaining `acc_len` bits.
    tail_bits = (8 - (bit_pos & 7)) & 7
    rem_shift = tl.maximum(acc_len - tail_bits, 0).to(tl.uint64)
    tail_mask = ((1 << tail_bits) - 1).to(tl.uint64)
    tail_val = (acc >> rem_shift) & tail_mask
    ok = ok & ((tail_bits == 0) | (tail_val == 0))

    consumed_bytes = (bit_pos + 7) >> 3
    exact_len = consumed_bytes == length

    if ALLOW_PADDED:
        need_pad_check = (~exact_len) & ok & (length == PADDED_BODY_LEN)
        pad_ok = length == PADDED_BODY_LEN
        rem_full_bits = tl.maximum(acc_len - tail_bits, 0)
        rem_full_mask = (tl.full([BLK], 1, dtype=tl.uint64) << rem_full_bits.to(tl.uint64)) - 1
        pad_ok = pad_ok & ((acc & rem_full_mask) == 0)
        cur_w = wpos
        while tl.max((need_pad_check & (cur_w < W)).to(tl.int32), axis=0) > 0:
            rem_w = tl.load(
                WORDS_T + cur_w * B + lane,
                mask=need_pad_check & (cur_w < W),
                other=0,
            )
            pad_ok = pad_ok & (rem_w == 0)
            cur_w = cur_w + 1
        ok = ok & (exact_len | (need_pad_check & pad_ok))
    else:
        ok = ok & exact_len

    tl.store(OK + lane, ok.to(tl.int8), mask=m)


@triton.jit
def _pad_nonce_msg_kernel(
    FLAT_SIGS,
    SIG_OFFS,
    MSGS,
    OUT,
    B,
    NONCE_LEN: tl.constexpr,
    MSG_LEN: tl.constexpr,
    PADDED_LEN: tl.constexpr,
    DOMAIN_BYTE: tl.constexpr,
    BLK_B: tl.constexpr,
    BLK_C: tl.constexpr,
):
    """Build SHAKE256-padded `[B, PADDED_LEN]` on device from `FLAT_SIGS` (nonce at +1) and `MSGS`."""
    pb = tl.program_id(0)
    pc = tl.program_id(1)
    b = pb * BLK_B + tl.arange(0, BLK_B)
    c = pc * BLK_C + tl.arange(0, BLK_C)
    mb = b < B
    mc = c < PADDED_LEN
    m = mb[:, None] & mc[None, :]
    sig_off = tl.load(SIG_OFFS + b, mask=mb, other=0)[:, None]
    col = c[None, :]
    total = NONCE_LEN + MSG_LEN

    is_nonce = col < NONCE_LEN
    is_msg = (col >= NONCE_LEN) & (col < total)
    nv = tl.load(FLAT_SIGS + sig_off + 1 + col, mask=m & is_nonce, other=0).to(tl.uint8)
    mv = tl.load(
        MSGS + b[:, None] * MSG_LEN + (col - NONCE_LEN),
        mask=m & is_msg,
        other=0,
    ).to(tl.uint8)
    val = tl.where(is_nonce, nv, tl.where(is_msg, mv, tl.zeros([BLK_B, BLK_C], dtype=tl.uint8)))
    val = tl.where(col == total, (val | DOMAIN_BYTE).to(tl.uint8), val)
    val = tl.where(col == PADDED_LEN - 1, (val | 0x80).to(tl.uint8), val)
    tl.store(OUT + b[:, None] * PADDED_LEN + col, val, mask=m)


@triton.jit
def _mul_ntt_kernel(A, B, OUT, P, Q: tl.constexpr, N: tl.constexpr, BLK: tl.constexpr):
    """Elementwise `A * B mod Q` over `P` coefficients where `B` is one row of `N`.

    `B` is a *single* polynomial (the public key's transform), reused by every row: indexing it
    as `offs % N` avoids materialising `[P, N]` (an `expand`ed tensor is not contiguous, and a
    kernel that assumes contiguity would read past the end of the first row).
    """
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < P
    a = tl.load(A + offs, mask=m, other=0).to(tl.int64)
    b = tl.load(B + offs % N, mask=m, other=0).to(tl.int64)
    tl.store(OUT + offs, (a * b % Q).to(tl.int32), mask=m)


@triton.jit
def _check_kernel(PROD, C, S2, SQN, OK, P, L2BOUND: tl.constexpr, Q: tl.constexpr, N: tl.constexpr):
    """One program per signature: normalise `s1`, sum the squared norms, compare to the bound."""
    row = tl.program_id(0)
    offs = tl.arange(0, N)
    m = row < P
    prod = tl.load(PROD + row * N + offs, mask=m, other=0).to(tl.int64)
    c = tl.load(C + row * N + offs, mask=m, other=0).to(tl.int64)
    s2 = tl.load(S2 + row * N + offs, mask=m, other=0).to(tl.int64)

    # `prod - c` is negative for half the coefficients. Triton's `%` on signed integers is
    # C's truncated remainder (result keeps the dividend's sign), *not* numpy/Python's floored
    # modulo, so `(prod - c) % Q` leaves negatives negative. Normalise explicitly.
    t = prod - c
    t = tl.where(t < 0, t + Q, t)
    t = tl.where(t > Q // 2, t - Q, t)
    sqn = tl.sum(t * t, axis=0) + tl.sum(s2 * s2, axis=0)
    tl.store(SQN + row, sqn, mask=m)
    tl.store(OK + row, (sqn <= L2BOUND).to(tl.int8), mask=m)


def check_np(prod, c, s2, q: int = 12289, l2bound: int = 34034726):
    """numpy twin of `_check_kernel`, for the reference and for tests."""
    import numpy as np

    t = (np.asarray(prod, dtype=np.int64) - np.asarray(c, dtype=np.int64)) % q
    t = np.where(t < 0, t + q, t)  # numpy's % is floored; this is the same normalisation
    t = np.where(t > q // 2, t - q, t)
    sqn = (t * t).sum(axis=-1) + (np.asarray(s2, dtype=np.int64) ** 2).sum(axis=-1)
    return sqn <= l2bound, sqn


@triton.jit
def _hash_to_point_kernel(
    STREAM,
    OUT,
    OK,
    B,
    N: tl.constexpr,
    NSAMP: tl.constexpr,
    COLS: tl.constexpr,
    Q: tl.constexpr,
    BLK: tl.constexpr,
):
    """Rejection sampling of `HashToPoint`, per signature, reading directly from SHAKE256 bytes.

    From `NSAMP` candidate big-endian 16-bit samples (`n + oversample` rounded up to 4 samples),
    keep those below `5q`, reduce mod q, and take the first `n` **in order** via an exclusive
    prefix sum over the keep-mask. `COLS` is the power-of-two tile width (`1024` for Falcon-512,
    `2048` for Falcon-1024).
    """
    pid = tl.program_id(0)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B
    cols = tl.arange(0, COLS)
    col_mask = cols < NSAMP
    base = lane[:, None] * (NSAMP * 2) + cols[None, :] * 2
    hi = tl.load(STREAM + base, mask=m[:, None] & col_mask[None, :], other=0xFF).to(tl.int32)
    lo = tl.load(STREAM + base + 1, mask=m[:, None] & col_mask[None, :], other=0xFF).to(tl.int32)
    w = (hi << 8) | lo
    keep = (w < 5 * Q) & col_mask[None, :]
    kept = keep.to(tl.int32)
    pos = tl.cumsum(kept, axis=1) - kept
    tl.store(OUT + lane[:, None] * N + pos, w % Q, mask=m[:, None] & keep & (pos < N))
    tl.store(OK + lane, (tl.sum(kept, axis=1) >= N).to(tl.int8), mask=m)
