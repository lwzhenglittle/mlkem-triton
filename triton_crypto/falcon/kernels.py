"""Falcon verification kernels: the two launches that are not the transform itself.

The transform is `primitives.ntt` (q = 12289, N = 512, int32 -- no new kernel needed, which is
what `eval-triton-crypto-primitives.md` §4.5 found). What is Falcon-specific is:

    _mul_ntt_kernel   pointwise product in the NTT domain, `s2_hat * h_hat mod q`
    _check_kernel     `s1 = c - s2*h`, normalise into [-q/2, q/2], and the L2 bound

`_check_kernel` also does the norm in int64: with |s1| <= q/2 = 6144 and n = 512 the squared
norm is below 2^33, so the comparison against `floor(beta^2) = 34034726` is exact -- the
reference's saturation arithmetic exists only because it works in 32 bits.
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = ["_check_kernel", "_hash_to_point_kernel", "_mul_ntt_kernel", "check_np"]


@triton.jit
def _mul_ntt_kernel(A, B, OUT, P, Q: tl.constexpr, N: tl.constexpr, BLK: tl.constexpr):
    """Elementwise `A * B mod Q` over `P * N` coefficients where `B` is one row of `N`.

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
    SAMPLES, OUT, OK, B, N: tl.constexpr, NSAMP: tl.constexpr, Q: tl.constexpr, BLK: tl.constexpr
):
    """Rejection sampling of `HashToPoint`, per signature.

    From `n + 205` candidate 16-bit samples, keep those below `5q`, reduce mod q, and take the
    first `n` **in order** -- the reference's branch-free compaction keeps the order, so an
    exclusive prefix sum over the keep-mask gives each survivor its destination. One program
    handles `BLK` signatures; the tile is `[BLK, NSAMP]` with NSAMP <= 1024.

    Measured against the host filter (numpy over the same samples): 0.47 vs ~7 us/signature at
    B = 4096 -- the prefix-sum-plus-scatter shape is fast when the tile is contiguous, which is
    the opposite of the per-coefficient decoder (`comp_decode`) that lost by 35x.
    """
    pid = tl.program_id(0)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B
    cols = tl.arange(0, 1024)
    col_mask = cols < NSAMP
    w = tl.load(
        SAMPLES + lane[:, None] * NSAMP + cols[None, :],
        mask=m[:, None] & col_mask[None, :],
        other=0xFFFF,
    ).to(tl.int32)
    keep = (w < 5 * Q) & col_mask[None, :]
    kept = keep.to(tl.int32)
    pos = tl.cumsum(kept, axis=1) - kept
    tl.store(OUT + lane[:, None] * N + pos, w % Q, mask=m[:, None] & keep & (pos < N))
    tl.store(OK + lane, tl.sum(kept, axis=1) >= N, mask=m)
