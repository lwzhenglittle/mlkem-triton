"""Pure-Triton Curve25519 / X25519 and Ed25519 kernels using 10-limb radix-2^25.5 `int64` arithmetic.

Every field element in F_{2^255 - 19} is represented as ten signed `int64` limbs (`h0..h9`) with
alternating 26-bit and 25-bit radix widths (Bernstein's `ref10` layout). After `_fe_carry`,
`|h_{2i}| <= 2^25` and `|h_{2i+1}| <= 2^24`, so products of unreduced sums/differences stay
strictly below 2^62 — zero 64-bit overflow and zero `umulhi` / 128-bit integer dependency.

To keep IR size and register pressure low on both Intel XPU and NVIDIA CUDA:
- `_fe_inv` uses a rolled 255-step binary exponentiation loop for `p - 2 = 2^255 - 21` (2 `_fe_mul`
  instances in IR instead of 22 unrolled chain stages).
- Ed25519 verification splits the 253-step Shamir scalar multiplication (`_ed25519_msm_kernel`,
  using the complete unified twisted Edwards addition `add-2008-hwcd-3` for both doubling and
  table addition in a rolled 506-step loop — only 9 `_fe_mul` instances in IR) from the final
  inversion and byte-encoding check (`_ed25519_finalize_kernel`).
"""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = [
    "_ed25519_finalize_kernel",
    "_ed25519_msm_kernel",
    "_ed25519_reduce_sc_kernel",
    "_x25519_kernel",
]


# --------------------------------------------------------------------------- F_{2^255 - 19} helpers


@triton.jit
def _fe_add(
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
):
    return (
        f0 + g0,
        f1 + g1,
        f2 + g2,
        f3 + g3,
        f4 + g4,
        f5 + g5,
        f6 + g6,
        f7 + g7,
        f8 + g8,
        f9 + g9,
    )


@triton.jit
def _fe_sub(
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
):
    return (
        f0 - g0,
        f1 - g1,
        f2 - g2,
        f3 - g3,
        f4 - g4,
        f5 - g5,
        f6 - g6,
        f7 - g7,
        f8 - g8,
        f9 - g9,
    )


@triton.jit
def _fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9):
    c0 = (h0 + (1 << 25)) >> 26
    h1 += c0
    h0 -= c0 << 26
    c4 = (h4 + (1 << 25)) >> 26
    h5 += c4
    h4 -= c4 << 26
    c1 = (h1 + (1 << 24)) >> 25
    h2 += c1
    h1 -= c1 << 25
    c5 = (h5 + (1 << 24)) >> 25
    h6 += c5
    h5 -= c5 << 25
    c2 = (h2 + (1 << 25)) >> 26
    h3 += c2
    h2 -= c2 << 26
    c6 = (h6 + (1 << 25)) >> 26
    h7 += c6
    h6 -= c6 << 26
    c3 = (h3 + (1 << 24)) >> 25
    h4 += c3
    h3 -= c3 << 25
    c7 = (h7 + (1 << 24)) >> 25
    h8 += c7
    h7 -= c7 << 25
    c4 = (h4 + (1 << 25)) >> 26
    h5 += c4
    h4 -= c4 << 26
    c8 = (h8 + (1 << 25)) >> 26
    h9 += c8
    h8 -= c8 << 26
    c9 = (h9 + (1 << 24)) >> 25
    h0 += c9 * 19
    h9 -= c9 << 25
    c0 = (h0 + (1 << 25)) >> 26
    h1 += c0
    h0 -= c0 << 26
    return h0, h1, h2, h3, h4, h5, h6, h7, h8, h9


@triton.jit
def _mul5(a0, a1, a2, a3, a4, b0, b1, b2, b3, b4):
    """5-limb schoolbook product for radix weights [0, 26, 51, 77, 102] (25 int64 muls)."""
    a1_2 = 2 * a1
    a3_2 = 2 * a3
    return (
        a0 * b0,
        a0 * b1 + a1 * b0,
        a0 * b2 + a1_2 * b1 + a2 * b0,
        a0 * b3 + a1 * b2 + a2 * b1 + a3 * b0,
        a0 * b4 + a1_2 * b3 + a2 * b2 + a3_2 * b1 + a4 * b0,
        a1 * b4 + a2 * b3 + a3 * b2 + a4 * b1,
        a2 * b4 + a3_2 * b3 + a4 * b2,
        a3 * b4 + a4 * b3,
        a4 * b4,
    )


@triton.jit
def _sq5(a0, a1, a2, a3, a4):
    """5-limb symmetric squaring for radix weights [0, 26, 51, 77, 102] (15 int64 muls)."""
    a0_2 = 2 * a0
    a1_2 = 2 * a1
    a2_2 = 2 * a2
    a3_2 = 2 * a3
    return (
        a0 * a0,
        a0_2 * a1,
        a0_2 * a2 + a1_2 * a1,
        a0_2 * a3 + a1_2 * a2,
        a0_2 * a4 + 2 * a1_2 * a3 + a2 * a2,
        a1_2 * a4 + a2_2 * a3,
        a2_2 * a4 + a3_2 * a3,
        a3_2 * a4,
        a4 * a4,
    )


@triton.jit
def _fe_mul(
    f0,
    f1,
    f2,
    f3,
    f4,
    f5,
    f6,
    f7,
    f8,
    f9,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
):
    """1-level Karatsuba F_{2^255-19} multiply (75 int64 muls vs 100 schoolbook)."""
    b0 = 2 * f5
    b2 = 2 * f7
    b4 = 2 * f9
    d0 = 2 * g5
    d2 = 2 * g7
    d4 = 2 * g9

    l0, l1, l2, l3, l4, l5, l6, l7, l8 = _mul5(f0, f1, f2, f3, f4, g0, g1, g2, g3, g4)
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = _mul5(b0, f6, b2, f8, b4, d0, g6, d2, g8, d4)
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = _mul5(
        f0 + b0,
        f1 + f6,
        f2 + b2,
        f3 + f8,
        f4 + b4,
        g0 + d0,
        g1 + g6,
        g2 + d2,
        g3 + g8,
        g4 + d4,
    )
    x0 = m0 - l0 - p0
    x1 = m1 - l1 - p1
    x2 = m2 - l2 - p2
    x3 = m3 - l3 - p3
    x4 = m4 - l4 - p4
    x5 = m5 - l5 - p5
    x6 = m6 - l6 - p6
    x7 = m7 - l7 - p7
    x8 = m8 - l8 - p8

    h0 = l0 + 19 * (x5 + (p0 >> 1))
    h1 = l1 + 19 * ((x6 >> 1) + (p1 >> 1))
    h2 = l2 + 19 * (x7 + (p2 >> 1))
    h3 = l3 + 19 * ((x8 >> 1) + (p3 >> 1))
    h4 = l4 + 19 * (p4 >> 1)
    h5 = (x0 >> 1) + l5 + 19 * (p5 >> 1)
    h6 = x1 + l6 + 19 * (p6 >> 1)
    h7 = (x2 >> 1) + l7 + 19 * (p7 >> 1)
    h8 = x3 + l8 + 19 * (p8 >> 1)
    h9 = x4 >> 1
    return _fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)


@triton.jit
def _fe_sq(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
    """1-level Karatsuba F_{2^255-19} squaring (45 int64 muls vs 100 schoolbook)."""
    b0 = 2 * f5
    b2 = 2 * f7
    b4 = 2 * f9

    l0, l1, l2, l3, l4, l5, l6, l7, l8 = _sq5(f0, f1, f2, f3, f4)
    p0, p1, p2, p3, p4, p5, p6, p7, p8 = _sq5(b0, f6, b2, f8, b4)
    m0, m1, m2, m3, m4, m5, m6, m7, m8 = _sq5(f0 + b0, f1 + f6, f2 + b2, f3 + f8, f4 + b4)

    x0 = m0 - l0 - p0
    x1 = m1 - l1 - p1
    x2 = m2 - l2 - p2
    x3 = m3 - l3 - p3
    x4 = m4 - l4 - p4
    x5 = m5 - l5 - p5
    x6 = m6 - l6 - p6
    x7 = m7 - l7 - p7
    x8 = m8 - l8 - p8

    h0 = l0 + 19 * (x5 + (p0 >> 1))
    h1 = l1 + 19 * ((x6 >> 1) + (p1 >> 1))
    h2 = l2 + 19 * (x7 + (p2 >> 1))
    h3 = l3 + 19 * ((x8 >> 1) + (p3 >> 1))
    h4 = l4 + 19 * (p4 >> 1)
    h5 = (x0 >> 1) + l5 + 19 * (p5 >> 1)
    h6 = x1 + l6 + 19 * (p6 >> 1)
    h7 = (x2 >> 1) + l7 + 19 * (p7 >> 1)
    h8 = x3 + l8 + 19 * (p8 >> 1)
    h9 = x4 >> 1
    return _fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)


@triton.jit
def _fe_mul_121665(f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
    return _fe_carry(
        f0 * 121665,
        f1 * 121665,
        f2 * 121665,
        f3 * 121665,
        f4 * 121665,
        f5 * 121665,
        f6 * 121665,
        f7 * 121665,
        f8 * 121665,
        f9 * 121665,
    )


@triton.jit
def _fe_inv(z0, z1, z2, z3, z4, z5, z6, z7, z8, z9, BLOCK: tl.constexpr):
    """Rolled Fermat inversion z^(2^255 - 21) mod (2^255 - 19): only 2 `_fe_mul` instances in IR."""
    zero = tl.zeros((BLOCK,), dtype=tl.int64)
    r0, r1, r2, r3, r4, r5, r6, r7, r8, r9 = (
        zero + 1,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
        zero,
    )
    # p - 2 = 2^255 - 21: bits 254..5 are all 1, bits 4..0 are 01011 (bits 3, 1, 0 are 1)
    for i in range(0, 255, 1):
        r0, r1, r2, r3, r4, r5, r6, r7, r8, r9 = _fe_sq(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9)
        bit_pos = 254 - i
        if (bit_pos >= 5) | (bit_pos == 3) | (bit_pos == 1) | (bit_pos == 0):
            r0, r1, r2, r3, r4, r5, r6, r7, r8, r9 = _fe_mul(
                r0,
                r1,
                r2,
                r3,
                r4,
                r5,
                r6,
                r7,
                r8,
                r9,
                z0,
                z1,
                z2,
                z3,
                z4,
                z5,
                z6,
                z7,
                z8,
                z9,
            )
    return r0, r1, r2, r3, r4, r5, r6, r7, r8, r9


@triton.jit
def _u32_to_fe(w0, w1, w2, w3, w4, w5, w6, w7):
    """Unpack 8 little-endian uint32 words (top bit already cleared) to centered 10-limb int64."""
    u0 = w0.to(tl.uint64)
    u1 = w1.to(tl.uint64)
    u2 = w2.to(tl.uint64)
    u3 = w3.to(tl.uint64)
    u4 = w4.to(tl.uint64)
    u5 = w5.to(tl.uint64)
    u6 = w6.to(tl.uint64)
    u7 = w7.to(tl.uint64)

    m26 = tl.full((), (1 << 26) - 1, tl.uint64)
    m25 = tl.full((), (1 << 25) - 1, tl.uint64)

    h0 = (u0 & m26).to(tl.int64)
    h1 = (((u0 >> 26) | (u1 << 6)) & m25).to(tl.int64)
    h2 = (((u1 >> 19) | (u2 << 13)) & m26).to(tl.int64)
    h3 = (((u2 >> 13) | (u3 << 19)) & m25).to(tl.int64)
    h4 = ((u3 >> 6) & m26).to(tl.int64)
    h5 = (u4 & m25).to(tl.int64)
    h6 = (((u4 >> 25) | (u5 << 7)) & m26).to(tl.int64)
    h7 = (((u5 >> 19) | (u6 << 13)) & m25).to(tl.int64)
    h8 = (((u6 >> 12) | (u7 << 20)) & m26).to(tl.int64)
    h9 = ((u7 >> 6) & m25).to(tl.int64)
    return _fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)


@triton.jit
def _fe_to_u32(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9):
    """Canonical reduction mod (2^255 - 19) into 8 little-endian uint32 words."""
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9 = _fe_carry(h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)
    for _ in range(0, 2, 1):
        c = h0 >> 26
        h1 += c
        h0 -= c << 26
        c = h1 >> 25
        h2 += c
        h1 -= c << 25
        c = h2 >> 26
        h3 += c
        h2 -= c << 26
        c = h3 >> 25
        h4 += c
        h3 -= c << 25
        c = h4 >> 26
        h5 += c
        h4 -= c << 26
        c = h5 >> 25
        h6 += c
        h5 -= c << 25
        c = h6 >> 26
        h7 += c
        h6 -= c << 26
        c = h7 >> 25
        h8 += c
        h7 -= c << 25
        c = h8 >> 26
        h9 += c
        h8 -= c << 26
        c = h9 >> 25
        h0 += c * 19
        h9 -= c << 25

    q = (h0 + 19) >> 26
    q = (h1 + q) >> 25
    q = (h2 + q) >> 26
    q = (h3 + q) >> 25
    q = (h4 + q) >> 26
    q = (h5 + q) >> 25
    q = (h6 + q) >> 26
    q = (h7 + q) >> 25
    q = (h8 + q) >> 26
    q = (h9 + q) >> 25

    h0 += 19 * q
    c = h0 >> 26
    h1 += c
    h0 -= c << 26
    c = h1 >> 25
    h2 += c
    h1 -= c << 25
    c = h2 >> 26
    h3 += c
    h2 -= c << 26
    c = h3 >> 25
    h4 += c
    h3 -= c << 25
    c = h4 >> 26
    h5 += c
    h4 -= c << 26
    c = h5 >> 25
    h6 += c
    h5 -= c << 25
    c = h6 >> 26
    h7 += c
    h6 -= c << 26
    c = h7 >> 25
    h8 += c
    h7 -= c << 25
    c = h8 >> 26
    h9 += c
    h8 -= c << 26
    c = h9 >> 25
    h9 -= c << 25

    w0 = (h0 | (h1 << 26)).to(tl.uint32)
    w1 = ((h1 >> 6) | (h2 << 19)).to(tl.uint32)
    w2 = ((h2 >> 13) | (h3 << 13)).to(tl.uint32)
    w3 = ((h3 >> 19) | (h4 << 6)).to(tl.uint32)
    w4 = (h5 | (h6 << 25)).to(tl.uint32)
    w5 = ((h6 >> 7) | (h7 << 19)).to(tl.uint32)
    w6 = ((h7 >> 13) | (h8 << 12)).to(tl.uint32)
    w7 = ((h8 >> 20) | (h9 << 6)).to(tl.uint32)
    return w0, w1, w2, w3, w4, w5, w6, w7


# --------------------------------------------------------------------------- X25519 Montgomery ladder


@triton.jit
def _x25519_kernel(
    SCALARS,
    U_COORDS,
    OUT,
    rows,
    S_STRIDE,
    U_STRIDE,
    BLOCK: tl.constexpr,
):
    """Batched RFC 7748 X25519 Montgomery ladder: `[rows, 8]` uint32 -> `[rows, 8]` uint32."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows

    sb = row * S_STRIDE
    s0 = tl.load(SCALARS + sb + 0, mask=m, other=0).to(tl.uint32, bitcast=True) & 0xFFFFFFF8
    s1 = tl.load(SCALARS + sb + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    s2 = tl.load(SCALARS + sb + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    s3 = tl.load(SCALARS + sb + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    s4 = tl.load(SCALARS + sb + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    s5 = tl.load(SCALARS + sb + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    s6 = tl.load(SCALARS + sb + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    s7 = (
        tl.load(SCALARS + sb + 7, mask=m, other=0).to(tl.uint32, bitcast=True) & 0x7FFFFFFF
    ) | 0x40000000

    ub = row * U_STRIDE
    u0 = tl.load(U_COORDS + ub + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
    u1 = tl.load(U_COORDS + ub + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    u2 = tl.load(U_COORDS + ub + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    u3 = tl.load(U_COORDS + ub + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    u4 = tl.load(U_COORDS + ub + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    u5 = tl.load(U_COORDS + ub + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    u6 = tl.load(U_COORDS + ub + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    u7 = tl.load(U_COORDS + ub + 7, mask=m, other=0).to(tl.uint32, bitcast=True) & 0x7FFFFFFF

    x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7, x1_8, x1_9 = _u32_to_fe(
        u0, u1, u2, u3, u4, u5, u6, u7
    )

    z = tl.zeros((BLOCK,), dtype=tl.int64)
    one = z + 1
    x2_0, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8, x2_9 = one, z, z, z, z, z, z, z, z, z
    z2_0, z2_1, z2_2, z2_3, z2_4, z2_5, z2_6, z2_7, z2_8, z2_9 = z, z, z, z, z, z, z, z, z, z
    x3_0, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6, x3_7, x3_8, x3_9 = (
        x1_0,
        x1_1,
        x1_2,
        x1_3,
        x1_4,
        x1_5,
        x1_6,
        x1_7,
        x1_8,
        x1_9,
    )
    z3_0, z3_1, z3_2, z3_3, z3_4, z3_5, z3_6, z3_7, z3_8, z3_9 = one, z, z, z, z, z, z, z, z, z

    swap = tl.zeros((BLOCK,), dtype=tl.uint32)
    for t in range(254, -1, -1):
        w_idx = t >> 5
        bit = t & 31
        sw = tl.where(
            w_idx == 7,
            s7,
            tl.where(
                w_idx == 6,
                s6,
                tl.where(
                    w_idx == 5,
                    s5,
                    tl.where(
                        w_idx == 4,
                        s4,
                        tl.where(
                            w_idx == 3, s3, tl.where(w_idx == 2, s2, tl.where(w_idx == 1, s1, s0))
                        ),
                    ),
                ),
            ),
        )
        k_t = (sw >> bit) & 1
        do_swap = (swap ^ k_t) != 0
        swap = k_t

        tx0 = tl.where(do_swap, x3_0, x2_0)
        x3_0 = tl.where(do_swap, x2_0, x3_0)
        x2_0 = tx0
        tx1 = tl.where(do_swap, x3_1, x2_1)
        x3_1 = tl.where(do_swap, x2_1, x3_1)
        x2_1 = tx1
        tx2 = tl.where(do_swap, x3_2, x2_2)
        x3_2 = tl.where(do_swap, x2_2, x3_2)
        x2_2 = tx2
        tx3 = tl.where(do_swap, x3_3, x2_3)
        x3_3 = tl.where(do_swap, x2_3, x3_3)
        x2_3 = tx3
        tx4 = tl.where(do_swap, x3_4, x2_4)
        x3_4 = tl.where(do_swap, x2_4, x3_4)
        x2_4 = tx4
        tx5 = tl.where(do_swap, x3_5, x2_5)
        x3_5 = tl.where(do_swap, x2_5, x3_5)
        x2_5 = tx5
        tx6 = tl.where(do_swap, x3_6, x2_6)
        x3_6 = tl.where(do_swap, x2_6, x3_6)
        x2_6 = tx6
        tx7 = tl.where(do_swap, x3_7, x2_7)
        x3_7 = tl.where(do_swap, x2_7, x3_7)
        x2_7 = tx7
        tx8 = tl.where(do_swap, x3_8, x2_8)
        x3_8 = tl.where(do_swap, x2_8, x3_8)
        x2_8 = tx8
        tx9 = tl.where(do_swap, x3_9, x2_9)
        x3_9 = tl.where(do_swap, x2_9, x3_9)
        x2_9 = tx9

        tz0 = tl.where(do_swap, z3_0, z2_0)
        z3_0 = tl.where(do_swap, z2_0, z3_0)
        z2_0 = tz0
        tz1 = tl.where(do_swap, z3_1, z2_1)
        z3_1 = tl.where(do_swap, z2_1, z3_1)
        z2_1 = tz1
        tz2 = tl.where(do_swap, z3_2, z2_2)
        z3_2 = tl.where(do_swap, z2_2, z3_2)
        z2_2 = tz2
        tz3 = tl.where(do_swap, z3_3, z2_3)
        z3_3 = tl.where(do_swap, z2_3, z3_3)
        z2_3 = tz3
        tz4 = tl.where(do_swap, z3_4, z2_4)
        z3_4 = tl.where(do_swap, z2_4, z3_4)
        z2_4 = tz4
        tz5 = tl.where(do_swap, z3_5, z2_5)
        z3_5 = tl.where(do_swap, z2_5, z3_5)
        z2_5 = tz5
        tz6 = tl.where(do_swap, z3_6, z2_6)
        z3_6 = tl.where(do_swap, z2_6, z3_6)
        z2_6 = tz6
        tz7 = tl.where(do_swap, z3_7, z2_7)
        z3_7 = tl.where(do_swap, z2_7, z3_7)
        z2_7 = tz7
        tz8 = tl.where(do_swap, z3_8, z2_8)
        z3_8 = tl.where(do_swap, z2_8, z3_8)
        z2_8 = tz8
        tz9 = tl.where(do_swap, z3_9, z2_9)
        z3_9 = tl.where(do_swap, z2_9, z3_9)
        z2_9 = tz9

        a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _fe_add(
            x2_0,
            x2_1,
            x2_2,
            x2_3,
            x2_4,
            x2_5,
            x2_6,
            x2_7,
            x2_8,
            x2_9,
            z2_0,
            z2_1,
            z2_2,
            z2_3,
            z2_4,
            z2_5,
            z2_6,
            z2_7,
            z2_8,
            z2_9,
        )
        b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = _fe_sub(
            x2_0,
            x2_1,
            x2_2,
            x2_3,
            x2_4,
            x2_5,
            x2_6,
            x2_7,
            x2_8,
            x2_9,
            z2_0,
            z2_1,
            z2_2,
            z2_3,
            z2_4,
            z2_5,
            z2_6,
            z2_7,
            z2_8,
            z2_9,
        )
        c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = _fe_add(
            x3_0,
            x3_1,
            x3_2,
            x3_3,
            x3_4,
            x3_5,
            x3_6,
            x3_7,
            x3_8,
            x3_9,
            z3_0,
            z3_1,
            z3_2,
            z3_3,
            z3_4,
            z3_5,
            z3_6,
            z3_7,
            z3_8,
            z3_9,
        )
        d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = _fe_sub(
            x3_0,
            x3_1,
            x3_2,
            x3_3,
            x3_4,
            x3_5,
            x3_6,
            x3_7,
            x3_8,
            x3_9,
            z3_0,
            z3_1,
            z3_2,
            z3_3,
            z3_4,
            z3_5,
            z3_6,
            z3_7,
            z3_8,
            z3_9,
        )
        da0, da1, da2, da3, da4, da5, da6, da7, da8, da9 = _fe_mul(
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
        )
        cb0, cb1, cb2, cb3, cb4, cb5, cb6, cb7, cb8, cb9 = _fe_mul(
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9
        )
        aa0, aa1, aa2, aa3, aa4, aa5, aa6, aa7, aa8, aa9 = _fe_sq(
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
        )
        bb0, bb1, bb2, bb3, bb4, bb5, bb6, bb7, bb8, bb9 = _fe_sq(
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9
        )
        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9 = _fe_sub(
            aa0,
            aa1,
            aa2,
            aa3,
            aa4,
            aa5,
            aa6,
            aa7,
            aa8,
            aa9,
            bb0,
            bb1,
            bb2,
            bb3,
            bb4,
            bb5,
            bb6,
            bb7,
            bb8,
            bb9,
        )
        x2_0, x2_1, x2_2, x2_3, x2_4, x2_5, x2_6, x2_7, x2_8, x2_9 = _fe_mul(
            aa0,
            aa1,
            aa2,
            aa3,
            aa4,
            aa5,
            aa6,
            aa7,
            aa8,
            aa9,
            bb0,
            bb1,
            bb2,
            bb3,
            bb4,
            bb5,
            bb6,
            bb7,
            bb8,
            bb9,
        )
        ae0, ae1, ae2, ae3, ae4, ae5, ae6, ae7, ae8, ae9 = _fe_mul_121665(
            e0, e1, e2, e3, e4, e5, e6, e7, e8, e9
        )
        ae0, ae1, ae2, ae3, ae4, ae5, ae6, ae7, ae8, ae9 = _fe_add(
            aa0,
            aa1,
            aa2,
            aa3,
            aa4,
            aa5,
            aa6,
            aa7,
            aa8,
            aa9,
            ae0,
            ae1,
            ae2,
            ae3,
            ae4,
            ae5,
            ae6,
            ae7,
            ae8,
            ae9,
        )
        z2_0, z2_1, z2_2, z2_3, z2_4, z2_5, z2_6, z2_7, z2_8, z2_9 = _fe_mul(
            e0,
            e1,
            e2,
            e3,
            e4,
            e5,
            e6,
            e7,
            e8,
            e9,
            ae0,
            ae1,
            ae2,
            ae3,
            ae4,
            ae5,
            ae6,
            ae7,
            ae8,
            ae9,
        )
        s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9 = _fe_add(
            da0,
            da1,
            da2,
            da3,
            da4,
            da5,
            da6,
            da7,
            da8,
            da9,
            cb0,
            cb1,
            cb2,
            cb3,
            cb4,
            cb5,
            cb6,
            cb7,
            cb8,
            cb9,
        )
        x3_0, x3_1, x3_2, x3_3, x3_4, x3_5, x3_6, x3_7, x3_8, x3_9 = _fe_sq(
            s_0, s_1, s_2, s_3, s_4, s_5, s_6, s_7, s_8, s_9
        )
        m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9 = _fe_sub(
            da0,
            da1,
            da2,
            da3,
            da4,
            da5,
            da6,
            da7,
            da8,
            da9,
            cb0,
            cb1,
            cb2,
            cb3,
            cb4,
            cb5,
            cb6,
            cb7,
            cb8,
            cb9,
        )
        m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9 = _fe_sq(
            m_0, m_1, m_2, m_3, m_4, m_5, m_6, m_7, m_8, m_9
        )
        z3_0, z3_1, z3_2, z3_3, z3_4, z3_5, z3_6, z3_7, z3_8, z3_9 = _fe_mul(
            x1_0,
            x1_1,
            x1_2,
            x1_3,
            x1_4,
            x1_5,
            x1_6,
            x1_7,
            x1_8,
            x1_9,
            m_0,
            m_1,
            m_2,
            m_3,
            m_4,
            m_5,
            m_6,
            m_7,
            m_8,
            m_9,
        )

    do_swap = swap != 0
    x2_0 = tl.where(do_swap, x3_0, x2_0)
    x2_1 = tl.where(do_swap, x3_1, x2_1)
    x2_2 = tl.where(do_swap, x3_2, x2_2)
    x2_3 = tl.where(do_swap, x3_3, x2_3)
    x2_4 = tl.where(do_swap, x3_4, x2_4)
    x2_5 = tl.where(do_swap, x3_5, x2_5)
    x2_6 = tl.where(do_swap, x3_6, x2_6)
    x2_7 = tl.where(do_swap, x3_7, x2_7)
    x2_8 = tl.where(do_swap, x3_8, x2_8)
    x2_9 = tl.where(do_swap, x3_9, x2_9)

    z2_0 = tl.where(do_swap, z3_0, z2_0)
    z2_1 = tl.where(do_swap, z3_1, z2_1)
    z2_2 = tl.where(do_swap, z3_2, z2_2)
    z2_3 = tl.where(do_swap, z3_3, z2_3)
    z2_4 = tl.where(do_swap, z3_4, z2_4)
    z2_5 = tl.where(do_swap, z3_5, z2_5)
    z2_6 = tl.where(do_swap, z3_6, z2_6)
    z2_7 = tl.where(do_swap, z3_7, z2_7)
    z2_8 = tl.where(do_swap, z3_8, z2_8)
    z2_9 = tl.where(do_swap, z3_9, z2_9)

    zi0, zi1, zi2, zi3, zi4, zi5, zi6, zi7, zi8, zi9 = _fe_inv(
        z2_0, z2_1, z2_2, z2_3, z2_4, z2_5, z2_6, z2_7, z2_8, z2_9, BLOCK
    )
    r0, r1, r2, r3, r4, r5, r6, r7, r8, r9 = _fe_mul(
        x2_0,
        x2_1,
        x2_2,
        x2_3,
        x2_4,
        x2_5,
        x2_6,
        x2_7,
        x2_8,
        x2_9,
        zi0,
        zi1,
        zi2,
        zi3,
        zi4,
        zi5,
        zi6,
        zi7,
        zi8,
        zi9,
    )
    o0, o1, o2, o3, o4, o5, o6, o7 = _fe_to_u32(r0, r1, r2, r3, r4, r5, r6, r7, r8, r9)
    ob = row * 8
    tl.store(OUT + ob + 0, o0.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 1, o1.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 2, o2.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 3, o3.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 4, o4.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 5, o5.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 6, o6.to(tl.int32, bitcast=True), mask=m)
    tl.store(OUT + ob + 7, o7.to(tl.int32, bitcast=True), mask=m)


# --------------------------------------------------------------------------- Ed25519 scalar reduce & verify


@triton.jit
def _ed25519_reduce_sc_kernel(
    HASH64,
    SC_OUT,
    rows,
    BLOCK: tl.constexpr,
):
    """Reduce 64-byte SHA-512 digests `[rows, 16]` uint32 LE modulo group order L -> `[rows, 8]` uint32."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows
    base = row * 16

    w0 = tl.load(HASH64 + base + 0, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w1 = tl.load(HASH64 + base + 1, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w2 = tl.load(HASH64 + base + 2, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w3 = tl.load(HASH64 + base + 3, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w4 = tl.load(HASH64 + base + 4, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w5 = tl.load(HASH64 + base + 5, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w6 = tl.load(HASH64 + base + 6, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w7 = tl.load(HASH64 + base + 7, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w8 = tl.load(HASH64 + base + 8, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w9 = tl.load(HASH64 + base + 9, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w10 = tl.load(HASH64 + base + 10, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w11 = tl.load(HASH64 + base + 11, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w12 = tl.load(HASH64 + base + 12, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w13 = tl.load(HASH64 + base + 13, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w14 = tl.load(HASH64 + base + 14, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)
    w15 = tl.load(HASH64 + base + 15, mask=m, other=0).to(tl.uint32, bitcast=True).to(tl.int64)

    M21 = 2097151
    s0 = w0 & M21
    s1 = ((w0 >> 21) | (w1 << 11)) & M21
    s2 = ((w1 >> 10) | (w2 << 22)) & M21
    s3 = ((w1 >> 31) | (w2 << 1)) & M21
    s4 = ((w2 >> 20) | (w3 << 12)) & M21
    s5 = ((w3 >> 9) | (w4 << 23)) & M21
    s6 = ((w3 >> 30) | (w4 << 2)) & M21
    s7 = ((w4 >> 19) | (w5 << 13)) & M21
    s8 = ((w5 >> 8) | (w6 << 24)) & M21
    s9 = ((w5 >> 29) | (w6 << 3)) & M21
    s10 = ((w6 >> 18) | (w7 << 14)) & M21
    s11 = ((w7 >> 7) | (w8 << 25)) & M21
    s12 = ((w7 >> 28) | (w8 << 4)) & M21
    s13 = ((w8 >> 17) | (w9 << 15)) & M21
    s14 = ((w9 >> 6) | (w10 << 26)) & M21
    s15 = ((w9 >> 27) | (w10 << 5)) & M21
    s16 = ((w10 >> 16) | (w11 << 16)) & M21
    s17 = ((w11 >> 5) | (w12 << 27)) & M21
    s18 = ((w11 >> 26) | (w12 << 6)) & M21
    s19 = ((w12 >> 15) | (w13 << 17)) & M21
    s20 = ((w13 >> 4) | (w14 << 28)) & M21
    s21 = ((w13 >> 25) | (w14 << 7)) & M21
    s22 = ((w14 >> 14) | (w15 << 18)) & M21
    s23 = w15 >> 3

    s11 += s23 * 666643
    s12 += s23 * 470296
    s13 += s23 * 654183
    s14 -= s23 * 997805
    s15 += s23 * 136657
    s16 -= s23 * 683901
    s10 += s22 * 666643
    s11 += s22 * 470296
    s12 += s22 * 654183
    s13 -= s22 * 997805
    s14 += s22 * 136657
    s15 -= s22 * 683901
    s9 += s21 * 666643
    s10 += s21 * 470296
    s11 += s21 * 654183
    s12 -= s21 * 997805
    s13 += s21 * 136657
    s14 -= s21 * 683901
    s8 += s20 * 666643
    s9 += s20 * 470296
    s10 += s20 * 654183
    s11 -= s20 * 997805
    s12 += s20 * 136657
    s13 -= s20 * 683901
    s7 += s19 * 666643
    s8 += s19 * 470296
    s9 += s19 * 654183
    s10 -= s19 * 997805
    s11 += s19 * 136657
    s12 -= s19 * 683901
    s6 += s18 * 666643
    s7 += s18 * 470296
    s8 += s18 * 654183
    s9 -= s18 * 997805
    s10 += s18 * 136657
    s11 -= s18 * 683901

    c = (s6 + (1 << 20)) >> 21
    s7 += c
    s6 -= c << 21
    c = (s8 + (1 << 20)) >> 21
    s9 += c
    s8 -= c << 21
    c = (s10 + (1 << 20)) >> 21
    s11 += c
    s10 -= c << 21
    c = (s12 + (1 << 20)) >> 21
    s13 += c
    s12 -= c << 21
    c = (s14 + (1 << 20)) >> 21
    s15 += c
    s14 -= c << 21
    c = (s16 + (1 << 20)) >> 21
    s17 += c
    s16 -= c << 21
    c = (s7 + (1 << 20)) >> 21
    s8 += c
    s7 -= c << 21
    c = (s9 + (1 << 20)) >> 21
    s10 += c
    s9 -= c << 21
    c = (s11 + (1 << 20)) >> 21
    s12 += c
    s11 -= c << 21
    c = (s13 + (1 << 20)) >> 21
    s14 += c
    s13 -= c << 21
    c = (s15 + (1 << 20)) >> 21
    s16 += c
    s15 -= c << 21

    s5 += s17 * 666643
    s6 += s17 * 470296
    s7 += s17 * 654183
    s8 -= s17 * 997805
    s9 += s17 * 136657
    s10 -= s17 * 683901
    s4 += s16 * 666643
    s5 += s16 * 470296
    s6 += s16 * 654183
    s7 -= s16 * 997805
    s8 += s16 * 136657
    s9 -= s16 * 683901
    s3 += s15 * 666643
    s4 += s15 * 470296
    s5 += s15 * 654183
    s6 -= s15 * 997805
    s7 += s15 * 136657
    s8 -= s15 * 683901
    s2 += s14 * 666643
    s3 += s14 * 470296
    s4 += s14 * 654183
    s5 -= s14 * 997805
    s6 += s14 * 136657
    s7 -= s14 * 683901
    s1 += s13 * 666643
    s2 += s13 * 470296
    s3 += s13 * 654183
    s4 -= s13 * 997805
    s5 += s13 * 136657
    s6 -= s13 * 683901
    s0 += s12 * 666643
    s1 += s12 * 470296
    s2 += s12 * 654183
    s3 -= s12 * 997805
    s4 += s12 * 136657
    s5 -= s12 * 683901
    s12 = tl.zeros((BLOCK,), dtype=tl.int64)

    for _ in range(0, 2, 1):
        c = (s0 + (1 << 20)) >> 21
        s1 += c
        s0 -= c << 21
        c = (s1 + (1 << 20)) >> 21
        s2 += c
        s1 -= c << 21
        c = (s2 + (1 << 20)) >> 21
        s3 += c
        s2 -= c << 21
        c = (s3 + (1 << 20)) >> 21
        s4 += c
        s3 -= c << 21
        c = (s4 + (1 << 20)) >> 21
        s5 += c
        s4 -= c << 21
        c = (s5 + (1 << 20)) >> 21
        s6 += c
        s5 -= c << 21
        c = (s6 + (1 << 20)) >> 21
        s7 += c
        s6 -= c << 21
        c = (s7 + (1 << 20)) >> 21
        s8 += c
        s7 -= c << 21
        c = (s8 + (1 << 20)) >> 21
        s9 += c
        s8 -= c << 21
        c = (s9 + (1 << 20)) >> 21
        s10 += c
        s9 -= c << 21
        c = (s10 + (1 << 20)) >> 21
        s11 += c
        s10 -= c << 21
        c = (s11 + (1 << 20)) >> 21
        s12 += c
        s11 -= c << 21
        s0 += s12 * 666643
        s1 += s12 * 470296
        s2 += s12 * 654183
        s3 -= s12 * 997805
        s4 += s12 * 136657
        s5 -= s12 * 683901
        s12 = tl.zeros((BLOCK,), dtype=tl.int64)

    for _ in range(0, 2, 1):
        c = s0 >> 21
        s1 += c
        s0 -= c << 21
        c = s1 >> 21
        s2 += c
        s1 -= c << 21
        c = s2 >> 21
        s3 += c
        s2 -= c << 21
        c = s3 >> 21
        s4 += c
        s3 -= c << 21
        c = s4 >> 21
        s5 += c
        s4 -= c << 21
        c = s5 >> 21
        s6 += c
        s5 -= c << 21
        c = s6 >> 21
        s7 += c
        s6 -= c << 21
        c = s7 >> 21
        s8 += c
        s7 -= c << 21
        c = s8 >> 21
        s9 += c
        s8 -= c << 21
        c = s9 >> 21
        s10 += c
        s9 -= c << 21
        c = s10 >> 21
        s11 += c
        s10 -= c << 21
        c = s11 >> 21
        s12 += c
        s11 -= c << 21
        s0 += s12 * 666643
        s1 += s12 * 470296
        s2 += s12 * 654183
        s3 -= s12 * 997805
        s4 += s12 * 136657
        s5 -= s12 * 683901
        s12 = tl.zeros((BLOCK,), dtype=tl.int64)

    c = s0 >> 21
    s1 += c
    s0 -= c << 21
    c = s1 >> 21
    s2 += c
    s1 -= c << 21
    c = s2 >> 21
    s3 += c
    s2 -= c << 21
    c = s3 >> 21
    s4 += c
    s3 -= c << 21
    c = s4 >> 21
    s5 += c
    s4 -= c << 21
    c = s5 >> 21
    s6 += c
    s5 -= c << 21
    c = s6 >> 21
    s7 += c
    s6 -= c << 21
    c = s7 >> 21
    s8 += c
    s7 -= c << 21
    c = s8 >> 21
    s9 += c
    s8 -= c << 21
    c = s9 >> 21
    s10 += c
    s9 -= c << 21
    c = s10 >> 21
    s11 += c
    s10 -= c << 21
    c = s11 >> 21
    s12 += c
    s11 -= c << 21

    neg = s12 < 0
    s0 = tl.where(neg, s0 - 666643, s0)
    s1 = tl.where(neg, s1 - 470296, s1)
    s2 = tl.where(neg, s2 - 654183, s2)
    s3 = tl.where(neg, s3 + 997805, s3)
    s4 = tl.where(neg, s4 - 136657, s4)
    s5 = tl.where(neg, s5 + 683901, s5)
    s12 = tl.where(neg, s12 + 1, s12)

    c = s0 >> 21
    s1 += c
    s0 -= c << 21
    c = s1 >> 21
    s2 += c
    s1 -= c << 21
    c = s2 >> 21
    s3 += c
    s2 -= c << 21
    c = s3 >> 21
    s4 += c
    s3 -= c << 21
    c = s4 >> 21
    s5 += c
    s4 -= c << 21
    c = s5 >> 21
    s6 += c
    s5 -= c << 21
    c = s6 >> 21
    s7 += c
    s6 -= c << 21
    c = s7 >> 21
    s8 += c
    s7 -= c << 21
    c = s8 >> 21
    s9 += c
    s8 -= c << 21
    c = s9 >> 21
    s10 += c
    s9 -= c << 21
    c = s10 >> 21
    s11 += c
    s10 -= c << 21
    c = s11 >> 21
    s12 += c
    s11 -= c << 21

    t0 = s0 + 666643
    t1 = s1 + 470296
    t2 = s2 + 654183
    t3 = s3 - 997805
    t4 = s4 + 136657
    t5 = s5 - 683901
    t6 = s6
    t7 = s7
    t8 = s8
    t9 = s9
    t10 = s10
    t11 = s11
    t12 = s12 - 1
    c = t0 >> 21
    t1 += c
    t0 -= c << 21
    c = t1 >> 21
    t2 += c
    t1 -= c << 21
    c = t2 >> 21
    t3 += c
    t2 -= c << 21
    c = t3 >> 21
    t4 += c
    t3 -= c << 21
    c = t4 >> 21
    t5 += c
    t4 -= c << 21
    c = t5 >> 21
    t6 += c
    t5 -= c << 21
    c = t6 >> 21
    t7 += c
    t6 -= c << 21
    c = t7 >> 21
    t8 += c
    t7 -= c << 21
    c = t8 >> 21
    t9 += c
    t8 -= c << 21
    c = t9 >> 21
    t10 += c
    t9 -= c << 21
    c = t10 >> 21
    t11 += c
    t10 -= c << 21
    c = t11 >> 21
    t12 += c
    t11 -= c << 21
    ge_l = t12 >= 0
    s0 = tl.where(ge_l, t0, s0)
    s1 = tl.where(ge_l, t1, s1)
    s2 = tl.where(ge_l, t2, s2)
    s3 = tl.where(ge_l, t3, s3)
    s4 = tl.where(ge_l, t4, s4)
    s5 = tl.where(ge_l, t5, s5)
    s6 = tl.where(ge_l, t6, s6)
    s7 = tl.where(ge_l, t7, s7)
    s8 = tl.where(ge_l, t8, s8)
    s9 = tl.where(ge_l, t9, s9)
    s10 = tl.where(ge_l, t10, s10)
    s11 = tl.where(ge_l, t11, s11)
    s12 = tl.where(ge_l, t12, s12)

    o0 = (s0 | (s1 << 21)).to(tl.uint32)
    o1 = ((s1 >> 11) | (s2 << 10) | (s3 << 31)).to(tl.uint32)
    o2 = ((s3 >> 1) | (s4 << 20)).to(tl.uint32)
    o3 = ((s4 >> 12) | (s5 << 9) | (s6 << 30)).to(tl.uint32)
    o4 = ((s6 >> 2) | (s7 << 19)).to(tl.uint32)
    o5 = ((s7 >> 13) | (s8 << 8) | (s9 << 29)).to(tl.uint32)
    o6 = ((s9 >> 3) | (s10 << 18)).to(tl.uint32)
    o7 = ((s10 >> 14) | (s11 << 7) | (s12 << 28)).to(tl.uint32)

    ob = row * 8
    tl.store(SC_OUT + ob + 0, o0.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 1, o1.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 2, o2.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 3, o3.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 4, o4.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 5, o5.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 6, o6.to(tl.int32, bitcast=True), mask=m)
    tl.store(SC_OUT + ob + 7, o7.to(tl.int32, bitcast=True), mask=m)


@triton.jit
def _ed25519_msm_kernel(
    TABLES,
    D2_LIMBS,
    S_WORDS,
    K_WORDS,
    Q_OUT,
    rows,
    TBL_STRIDE,
    BLOCK: tl.constexpr,
):
    """253-step Shamir's trick (`s*B - k*A`) using unified complete `add-2008-hwcd-3` in 506 steps.

    Even steps compute `Q + Q` (doubling); odd steps compute `Q + T[b_s | (b_k << 1)]`. Sharing
    one complete addition body keeps only 9 `_fe_mul` instances in IR for fast compilation and low
    register pressure. Outputs `(X, Y, Z)` as `[rows, 30]` int64 limbs.
    """
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows

    sb = row * 8
    s0 = tl.load(S_WORDS + sb + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
    s1 = tl.load(S_WORDS + sb + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    s2 = tl.load(S_WORDS + sb + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    s3 = tl.load(S_WORDS + sb + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    s4 = tl.load(S_WORDS + sb + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    s5 = tl.load(S_WORDS + sb + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    s6 = tl.load(S_WORDS + sb + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    s7 = tl.load(S_WORDS + sb + 7, mask=m, other=0).to(tl.uint32, bitcast=True)

    k0 = tl.load(K_WORDS + sb + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
    k1 = tl.load(K_WORDS + sb + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    k2 = tl.load(K_WORDS + sb + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    k3 = tl.load(K_WORDS + sb + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    k4 = tl.load(K_WORDS + sb + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    k5 = tl.load(K_WORDS + sb + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    k6 = tl.load(K_WORDS + sb + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    k7 = tl.load(K_WORDS + sb + 7, mask=m, other=0).to(tl.uint32, bitcast=True)

    # Constant 2*d mod p (10 limbs, broadcast scalar)
    d2_0 = tl.load(D2_LIMBS + 0)
    d2_1 = tl.load(D2_LIMBS + 1)
    d2_2 = tl.load(D2_LIMBS + 2)
    d2_3 = tl.load(D2_LIMBS + 3)
    d2_4 = tl.load(D2_LIMBS + 4)
    d2_5 = tl.load(D2_LIMBS + 5)
    d2_6 = tl.load(D2_LIMBS + 6)
    d2_7 = tl.load(D2_LIMBS + 7)
    d2_8 = tl.load(D2_LIMBS + 8)
    d2_9 = tl.load(D2_LIMBS + 9)

    z = tl.zeros((BLOCK,), dtype=tl.int64)
    one = z + 1
    qx0, qx1, qx2, qx3, qx4, qx5, qx6, qx7, qx8, qx9 = z, z, z, z, z, z, z, z, z, z
    qy0, qy1, qy2, qy3, qy4, qy5, qy6, qy7, qy8, qy9 = one, z, z, z, z, z, z, z, z, z
    qz0, qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9 = one, z, z, z, z, z, z, z, z, z
    qt0, qt1, qt2, qt3, qt4, qt5, qt6, qt7, qt8, qt9 = z, z, z, z, z, z, z, z, z, z

    tbl_base = row * TBL_STRIDE

    for step in range(0, 506, 1):
        is_dbl = (step & 1) == 0
        bit_idx = 252 - (step >> 1)

        ymx0, ymx1, ymx2, ymx3, ymx4, ymx5, ymx6, ymx7, ymx8, ymx9 = _fe_sub(
            qy0,
            qy1,
            qy2,
            qy3,
            qy4,
            qy5,
            qy6,
            qy7,
            qy8,
            qy9,
            qx0,
            qx1,
            qx2,
            qx3,
            qx4,
            qx5,
            qx6,
            qx7,
            qx8,
            qx9,
        )
        ypx0, ypx1, ypx2, ypx3, ypx4, ypx5, ypx6, ypx7, ypx8, ypx9 = _fe_add(
            qy0,
            qy1,
            qy2,
            qy3,
            qy4,
            qy5,
            qy6,
            qy7,
            qy8,
            qy9,
            qx0,
            qx1,
            qx2,
            qx3,
            qx4,
            qx5,
            qx6,
            qx7,
            qx8,
            qx9,
        )

        if is_dbl:
            # Complete self-addition Q + Q using Q's own cached coordinates
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _fe_sq(
                ymx0, ymx1, ymx2, ymx3, ymx4, ymx5, ymx6, ymx7, ymx8, ymx9
            )
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = _fe_sq(
                ypx0, ypx1, ypx2, ypx3, ypx4, ypx5, ypx6, ypx7, ypx8, ypx9
            )
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = _fe_sq(
                qz0, qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9
            )
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = _fe_add(
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                d8,
                d9,
                d0,
                d1,
                d2,
                d3,
                d4,
                d5,
                d6,
                d7,
                d8,
                d9,
            )
            tt0, tt1, tt2, tt3, tt4, tt5, tt6, tt7, tt8, tt9 = _fe_sq(
                qt0, qt1, qt2, qt3, qt4, qt5, qt6, qt7, qt8, qt9
            )
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = _fe_mul(
                tt0,
                tt1,
                tt2,
                tt3,
                tt4,
                tt5,
                tt6,
                tt7,
                tt8,
                tt9,
                d2_0,
                d2_1,
                d2_2,
                d2_3,
                d2_4,
                d2_5,
                d2_6,
                d2_7,
                d2_8,
                d2_9,
            )
        else:
            w_idx = bit_idx >> 5
            bit = bit_idx & 31
            sw = tl.where(
                w_idx == 7,
                s7,
                tl.where(
                    w_idx == 6,
                    s6,
                    tl.where(
                        w_idx == 5,
                        s5,
                        tl.where(
                            w_idx == 4,
                            s4,
                            tl.where(
                                w_idx == 3,
                                s3,
                                tl.where(w_idx == 2, s2, tl.where(w_idx == 1, s1, s0)),
                            ),
                        ),
                    ),
                ),
            )
            kw = tl.where(
                w_idx == 7,
                k7,
                tl.where(
                    w_idx == 6,
                    k6,
                    tl.where(
                        w_idx == 5,
                        k5,
                        tl.where(
                            w_idx == 4,
                            k4,
                            tl.where(
                                w_idx == 3,
                                k3,
                                tl.where(w_idx == 2, k2, tl.where(w_idx == 1, k1, k0)),
                            ),
                        ),
                    ),
                ),
            )
            idx = (((sw >> bit) & 1) | (((kw >> bit) & 1) << 1)).to(tl.int64)
            pbase = tbl_base + idx * 40

            p0 = tl.load(TABLES + pbase + 0, mask=m, other=0)
            p1 = tl.load(TABLES + pbase + 1, mask=m, other=0)
            p2 = tl.load(TABLES + pbase + 2, mask=m, other=0)
            p3 = tl.load(TABLES + pbase + 3, mask=m, other=0)
            p4 = tl.load(TABLES + pbase + 4, mask=m, other=0)
            p5 = tl.load(TABLES + pbase + 5, mask=m, other=0)
            p6 = tl.load(TABLES + pbase + 6, mask=m, other=0)
            p7 = tl.load(TABLES + pbase + 7, mask=m, other=0)
            p8 = tl.load(TABLES + pbase + 8, mask=m, other=0)
            p9 = tl.load(TABLES + pbase + 9, mask=m, other=0)
            a0, a1, a2, a3, a4, a5, a6, a7, a8, a9 = _fe_mul(
                ymx0,
                ymx1,
                ymx2,
                ymx3,
                ymx4,
                ymx5,
                ymx6,
                ymx7,
                ymx8,
                ymx9,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
            )

            p0 = tl.load(TABLES + pbase + 10, mask=m, other=0)
            p1 = tl.load(TABLES + pbase + 11, mask=m, other=0)
            p2 = tl.load(TABLES + pbase + 12, mask=m, other=0)
            p3 = tl.load(TABLES + pbase + 13, mask=m, other=0)
            p4 = tl.load(TABLES + pbase + 14, mask=m, other=0)
            p5 = tl.load(TABLES + pbase + 15, mask=m, other=0)
            p6 = tl.load(TABLES + pbase + 16, mask=m, other=0)
            p7 = tl.load(TABLES + pbase + 17, mask=m, other=0)
            p8 = tl.load(TABLES + pbase + 18, mask=m, other=0)
            p9 = tl.load(TABLES + pbase + 19, mask=m, other=0)
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9 = _fe_mul(
                ypx0,
                ypx1,
                ypx2,
                ypx3,
                ypx4,
                ypx5,
                ypx6,
                ypx7,
                ypx8,
                ypx9,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
            )

            p0 = tl.load(TABLES + pbase + 20, mask=m, other=0)
            p1 = tl.load(TABLES + pbase + 21, mask=m, other=0)
            p2 = tl.load(TABLES + pbase + 22, mask=m, other=0)
            p3 = tl.load(TABLES + pbase + 23, mask=m, other=0)
            p4 = tl.load(TABLES + pbase + 24, mask=m, other=0)
            p5 = tl.load(TABLES + pbase + 25, mask=m, other=0)
            p6 = tl.load(TABLES + pbase + 26, mask=m, other=0)
            p7 = tl.load(TABLES + pbase + 27, mask=m, other=0)
            p8 = tl.load(TABLES + pbase + 28, mask=m, other=0)
            p9 = tl.load(TABLES + pbase + 29, mask=m, other=0)
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9 = _fe_mul(
                qz0,
                qz1,
                qz2,
                qz3,
                qz4,
                qz5,
                qz6,
                qz7,
                qz8,
                qz9,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
            )

            p0 = tl.load(TABLES + pbase + 30, mask=m, other=0)
            p1 = tl.load(TABLES + pbase + 31, mask=m, other=0)
            p2 = tl.load(TABLES + pbase + 32, mask=m, other=0)
            p3 = tl.load(TABLES + pbase + 33, mask=m, other=0)
            p4 = tl.load(TABLES + pbase + 34, mask=m, other=0)
            p5 = tl.load(TABLES + pbase + 35, mask=m, other=0)
            p6 = tl.load(TABLES + pbase + 36, mask=m, other=0)
            p7 = tl.load(TABLES + pbase + 37, mask=m, other=0)
            p8 = tl.load(TABLES + pbase + 38, mask=m, other=0)
            p9 = tl.load(TABLES + pbase + 39, mask=m, other=0)
            c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = _fe_mul(
                qt0,
                qt1,
                qt2,
                qt3,
                qt4,
                qt5,
                qt6,
                qt7,
                qt8,
                qt9,
                p0,
                p1,
                p2,
                p3,
                p4,
                p5,
                p6,
                p7,
                p8,
                p9,
            )

        e0, e1, e2, e3, e4, e5, e6, e7, e8, e9 = _fe_sub(
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
        )
        f0, f1, f2, f3, f4, f5, f6, f7, f8, f9 = _fe_sub(
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9
        )
        g0, g1, g2, g3, g4, g5, g6, g7, g8, g9 = _fe_add(
            d0, d1, d2, d3, d4, d5, d6, d7, d8, d9, c0, c1, c2, c3, c4, c5, c6, c7, c8, c9
        )
        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9 = _fe_add(
            b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9
        )

        qx0, qx1, qx2, qx3, qx4, qx5, qx6, qx7, qx8, qx9 = _fe_mul(
            e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9
        )
        qy0, qy1, qy2, qy3, qy4, qy5, qy6, qy7, qy8, qy9 = _fe_mul(
            g0, g1, g2, g3, g4, g5, g6, g7, g8, g9, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9
        )
        qt0, qt1, qt2, qt3, qt4, qt5, qt6, qt7, qt8, qt9 = _fe_mul(
            e0, e1, e2, e3, e4, e5, e6, e7, e8, e9, h0, h1, h2, h3, h4, h5, h6, h7, h8, h9
        )
        qz0, qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9 = _fe_mul(
            f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, g0, g1, g2, g3, g4, g5, g6, g7, g8, g9
        )

    qb = row * 30
    tl.store(Q_OUT + qb + 0, qx0, mask=m)
    tl.store(Q_OUT + qb + 1, qx1, mask=m)
    tl.store(Q_OUT + qb + 2, qx2, mask=m)
    tl.store(Q_OUT + qb + 3, qx3, mask=m)
    tl.store(Q_OUT + qb + 4, qx4, mask=m)
    tl.store(Q_OUT + qb + 5, qx5, mask=m)
    tl.store(Q_OUT + qb + 6, qx6, mask=m)
    tl.store(Q_OUT + qb + 7, qx7, mask=m)
    tl.store(Q_OUT + qb + 8, qx8, mask=m)
    tl.store(Q_OUT + qb + 9, qx9, mask=m)

    tl.store(Q_OUT + qb + 10, qy0, mask=m)
    tl.store(Q_OUT + qb + 11, qy1, mask=m)
    tl.store(Q_OUT + qb + 12, qy2, mask=m)
    tl.store(Q_OUT + qb + 13, qy3, mask=m)
    tl.store(Q_OUT + qb + 14, qy4, mask=m)
    tl.store(Q_OUT + qb + 15, qy5, mask=m)
    tl.store(Q_OUT + qb + 16, qy6, mask=m)
    tl.store(Q_OUT + qb + 17, qy7, mask=m)
    tl.store(Q_OUT + qb + 18, qy8, mask=m)
    tl.store(Q_OUT + qb + 19, qy9, mask=m)

    tl.store(Q_OUT + qb + 20, qz0, mask=m)
    tl.store(Q_OUT + qb + 21, qz1, mask=m)
    tl.store(Q_OUT + qb + 22, qz2, mask=m)
    tl.store(Q_OUT + qb + 23, qz3, mask=m)
    tl.store(Q_OUT + qb + 24, qz4, mask=m)
    tl.store(Q_OUT + qb + 25, qz5, mask=m)
    tl.store(Q_OUT + qb + 26, qz6, mask=m)
    tl.store(Q_OUT + qb + 27, qz7, mask=m)
    tl.store(Q_OUT + qb + 28, qz8, mask=m)
    tl.store(Q_OUT + qb + 29, qz9, mask=m)


@triton.jit
def _ed25519_finalize_kernel(
    Q_IN,
    R_WORDS,
    S_WORDS,
    PK_OK,
    OUT_OK,
    rows,
    PK_OK_STRIDE,
    BLOCK: tl.constexpr,
):
    """Invert Z, canonically encode `(X/Z, Y/Z)`, check `s < L` and equality with `R`."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows

    sb = row * 8
    s0 = tl.load(S_WORDS + sb + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
    s1 = tl.load(S_WORDS + sb + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    s2 = tl.load(S_WORDS + sb + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    s3 = tl.load(S_WORDS + sb + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    s4 = tl.load(S_WORDS + sb + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    s5 = tl.load(S_WORDS + sb + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    s6 = tl.load(S_WORDS + sb + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    s7 = tl.load(S_WORDS + sb + 7, mask=m, other=0).to(tl.uint32, bitcast=True)

    lt = s7 < 0x10000000
    eq = s7 == 0x10000000
    lt = lt | (eq & (s6 < 0))
    eq = eq & (s6 == 0)
    lt = lt | (eq & (s5 < 0))
    eq = eq & (s5 == 0)
    lt = lt | (eq & (s4 < 0))
    eq = eq & (s4 == 0)
    lt = lt | (eq & (s3 < 0x14DEF9DE))
    eq = eq & (s3 == 0x14DEF9DE)
    lt = lt | (eq & (s2 < 0xA2F79CD6))
    eq = eq & (s2 == 0xA2F79CD6)
    lt = lt | (eq & (s1 < 0x5812631A))
    eq = eq & (s1 == 0x5812631A)
    s_valid = lt | (eq & (s0 < 0x5CF5D3ED))

    qb = row * 30
    qz0 = tl.load(Q_IN + qb + 20, mask=m, other=1)
    qz1 = tl.load(Q_IN + qb + 21, mask=m, other=0)
    qz2 = tl.load(Q_IN + qb + 22, mask=m, other=0)
    qz3 = tl.load(Q_IN + qb + 23, mask=m, other=0)
    qz4 = tl.load(Q_IN + qb + 24, mask=m, other=0)
    qz5 = tl.load(Q_IN + qb + 25, mask=m, other=0)
    qz6 = tl.load(Q_IN + qb + 26, mask=m, other=0)
    qz7 = tl.load(Q_IN + qb + 27, mask=m, other=0)
    qz8 = tl.load(Q_IN + qb + 28, mask=m, other=0)
    qz9 = tl.load(Q_IN + qb + 29, mask=m, other=0)

    zi0, zi1, zi2, zi3, zi4, zi5, zi6, zi7, zi8, zi9 = _fe_inv(
        qz0, qz1, qz2, qz3, qz4, qz5, qz6, qz7, qz8, qz9, BLOCK
    )

    qx0 = tl.load(Q_IN + qb + 0, mask=m, other=0)
    qx1 = tl.load(Q_IN + qb + 1, mask=m, other=0)
    qx2 = tl.load(Q_IN + qb + 2, mask=m, other=0)
    qx3 = tl.load(Q_IN + qb + 3, mask=m, other=0)
    qx4 = tl.load(Q_IN + qb + 4, mask=m, other=0)
    qx5 = tl.load(Q_IN + qb + 5, mask=m, other=0)
    qx6 = tl.load(Q_IN + qb + 6, mask=m, other=0)
    qx7 = tl.load(Q_IN + qb + 7, mask=m, other=0)
    qx8 = tl.load(Q_IN + qb + 8, mask=m, other=0)
    qx9 = tl.load(Q_IN + qb + 9, mask=m, other=0)

    ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9 = _fe_mul(
        qx0,
        qx1,
        qx2,
        qx3,
        qx4,
        qx5,
        qx6,
        qx7,
        qx8,
        qx9,
        zi0,
        zi1,
        zi2,
        zi3,
        zi4,
        zi5,
        zi6,
        zi7,
        zi8,
        zi9,
    )
    xu0, _, _, _, _, _, _, _ = _fe_to_u32(ax0, ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9)

    qy0 = tl.load(Q_IN + qb + 10, mask=m, other=1)
    qy1 = tl.load(Q_IN + qb + 11, mask=m, other=0)
    qy2 = tl.load(Q_IN + qb + 12, mask=m, other=0)
    qy3 = tl.load(Q_IN + qb + 13, mask=m, other=0)
    qy4 = tl.load(Q_IN + qb + 14, mask=m, other=0)
    qy5 = tl.load(Q_IN + qb + 15, mask=m, other=0)
    qy6 = tl.load(Q_IN + qb + 16, mask=m, other=0)
    qy7 = tl.load(Q_IN + qb + 17, mask=m, other=0)
    qy8 = tl.load(Q_IN + qb + 18, mask=m, other=0)
    qy9 = tl.load(Q_IN + qb + 19, mask=m, other=0)

    ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8, ay9 = _fe_mul(
        qy0,
        qy1,
        qy2,
        qy3,
        qy4,
        qy5,
        qy6,
        qy7,
        qy8,
        qy9,
        zi0,
        zi1,
        zi2,
        zi3,
        zi4,
        zi5,
        zi6,
        zi7,
        zi8,
        zi9,
    )
    yu0, yu1, yu2, yu3, yu4, yu5, yu6, yu7 = _fe_to_u32(
        ay0, ay1, ay2, ay3, ay4, ay5, ay6, ay7, ay8, ay9
    )
    yu7 = yu7 | ((xu0 & 1) << 31)

    r0 = tl.load(R_WORDS + sb + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
    r1 = tl.load(R_WORDS + sb + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
    r2 = tl.load(R_WORDS + sb + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
    r3 = tl.load(R_WORDS + sb + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
    r4 = tl.load(R_WORDS + sb + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
    r5 = tl.load(R_WORDS + sb + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
    r6 = tl.load(R_WORDS + sb + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
    r7 = tl.load(R_WORDS + sb + 7, mask=m, other=0).to(tl.uint32, bitcast=True)

    pk_ok = tl.load(PK_OK + row * PK_OK_STRIDE, mask=m, other=0) != 0
    eq_r = (
        (yu0 == r0)
        & (yu1 == r1)
        & (yu2 == r2)
        & (yu3 == r3)
        & (yu4 == r4)
        & (yu5 == r5)
        & (yu6 == r6)
        & (yu7 == r7)
    )
    ok = (pk_ok & s_valid & eq_r).to(tl.uint8)
    tl.store(OUT_OK + row, ok, mask=m)
