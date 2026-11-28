"""Pure-Triton kernels for batched MAYO verification over GF(16) (packed 8 nibbles / uint32)."""

from __future__ import annotations

import triton
import triton.language as tl

__all__ = [
    "_gf16_mul_u32",
    "_mayo_fold_check_kernel",
    "_mayo_sps_kernel",
]


@triton.jit
def _gf16_mul_u32(a, b):
    """Multiply 8 packed GF(16) nibbles in `uint32` `a` by 4-bit scalar `b` mod (x^4 + x + 1)."""
    a = a.to(tl.uint32)
    b = b.to(tl.uint32)
    m8 = tl.full((), 0x88888888, tl.uint32)

    r = a * (b & 1)
    msb = a & m8
    a = ((a ^ msb) << 1) ^ ((msb >> 3) * 3)
    r = r ^ (a * ((b >> 1) & 1))

    msb = a & m8
    a = ((a ^ msb) << 1) ^ ((msb >> 3) * 3)
    r = r ^ (a * ((b >> 2) & 1))

    msb = a & m8
    a = ((a ^ msb) << 1) ^ ((msb >> 3) * 3)
    r = r ^ (a * ((b >> 3) & 1))
    return r


@triton.jit
def _mayo_sps_kernel(
    P_MAT,
    S_NIB_T,
    SPS_OUT,
    rows,
    N: tl.constexpr,
    K: tl.constexpr,
    M_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Fused `PS = P * S^T` and `SPS = S * PS` per `(row_tile, col, m_word)`.

    - `P_MAT`: `[N, N, M_WORDS]` int32 (bitcast uint32, 8 GF(16) nibbles/word), upper triangular.
    - `S_NIB_T`: `[K, N, rows]` uint8 (transposed so adjacent lanes read coalesced bytes!).
    - `SPS_OUT`: `[rows, K, K, M_WORDS]` int32 (bitcast uint32).

    Each thread computes ONE `(b, col, w)` slice across all `i in 0..K-1`, holding only `1` (`ps_r`)
    + `K` (`sps_0..sps_{K-1}`, K <= 12) `uint32` accumulators in registers with zero global `PS`
    scratch buffer. `P_MAT[r, c, w]` is scalar-broadcast across all `BLOCK_B` lanes.
    """
    pid_b = tl.program_id(0)
    col = tl.program_id(1)
    w = tl.program_id(2)

    row = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    m = row < rows

    z = tl.zeros((BLOCK_B,), dtype=tl.uint32)
    sps0 = z
    sps1 = z
    sps2 = z
    sps3 = z
    sps4 = z
    sps5 = z
    sps6 = z
    sps7 = z
    sps8 = z
    sps9 = z
    sps10 = z
    sps11 = z

    col_off = col * N * rows

    for r in range(0, N, 1):
        ps_r = tl.zeros((BLOCK_B,), dtype=tl.uint32)
        p_row_base = r * (N * M_WORDS) + w
        for c in range(r, N, 1):
            p_rc = tl.load(P_MAT + p_row_base + c * M_WORDS).to(tl.uint32, bitcast=True)
            s_jc = tl.load(S_NIB_T + col_off + c * rows + row, mask=m, other=0).to(tl.uint32)
            ps_r = ps_r ^ _gf16_mul_u32(p_rc, s_jc)

        # Accumulate ps_r * S[i, r, b] into sps_i for i = 0..K-1 (coalesced along b)
        sps0 = sps0 ^ _gf16_mul_u32(
            ps_r, tl.load(S_NIB_T + (0 * N + r) * rows + row, mask=m, other=0)
        )
        sps1 = sps1 ^ _gf16_mul_u32(
            ps_r, tl.load(S_NIB_T + (1 * N + r) * rows + row, mask=m, other=0)
        )
        sps2 = sps2 ^ _gf16_mul_u32(
            ps_r, tl.load(S_NIB_T + (2 * N + r) * rows + row, mask=m, other=0)
        )
        sps3 = sps3 ^ _gf16_mul_u32(
            ps_r, tl.load(S_NIB_T + (3 * N + r) * rows + row, mask=m, other=0)
        )
        if K > 4:
            sps4 = sps4 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (4 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 5:
            sps5 = sps5 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (5 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 6:
            sps6 = sps6 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (6 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 7:
            sps7 = sps7 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (7 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 8:
            sps8 = sps8 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (8 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 9:
            sps9 = sps9 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (9 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 10:
            sps10 = sps10 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (10 * N + r) * rows + row, mask=m, other=0)
            )
        if K > 11:
            sps11 = sps11 ^ _gf16_mul_u32(
                ps_r, tl.load(S_NIB_T + (11 * N + r) * rows + row, mask=m, other=0)
            )

    ob = row * (K * K * M_WORDS) + col * M_WORDS + w
    stride_i = K * M_WORDS
    tl.store(SPS_OUT + ob + 0 * stride_i, sps0.to(tl.int32, bitcast=True), mask=m)
    tl.store(SPS_OUT + ob + 1 * stride_i, sps1.to(tl.int32, bitcast=True), mask=m)
    tl.store(SPS_OUT + ob + 2 * stride_i, sps2.to(tl.int32, bitcast=True), mask=m)
    tl.store(SPS_OUT + ob + 3 * stride_i, sps3.to(tl.int32, bitcast=True), mask=m)
    if K > 4:
        tl.store(SPS_OUT + ob + 4 * stride_i, sps4.to(tl.int32, bitcast=True), mask=m)
    if K > 5:
        tl.store(SPS_OUT + ob + 5 * stride_i, sps5.to(tl.int32, bitcast=True), mask=m)
    if K > 6:
        tl.store(SPS_OUT + ob + 6 * stride_i, sps6.to(tl.int32, bitcast=True), mask=m)
    if K > 7:
        tl.store(SPS_OUT + ob + 7 * stride_i, sps7.to(tl.int32, bitcast=True), mask=m)
    if K > 8:
        tl.store(SPS_OUT + ob + 8 * stride_i, sps8.to(tl.int32, bitcast=True), mask=m)
    if K > 9:
        tl.store(SPS_OUT + ob + 9 * stride_i, sps9.to(tl.int32, bitcast=True), mask=m)
    if K > 10:
        tl.store(SPS_OUT + ob + 10 * stride_i, sps10.to(tl.int32, bitcast=True), mask=m)
    if K > 11:
        tl.store(SPS_OUT + ob + 11 * stride_i, sps11.to(tl.int32, bitcast=True), mask=m)


@triton.jit
def _mayo_fold_check_kernel(
    SPS_IN,
    T_WORDS,
    OUT_OK,
    rows,
    K: tl.constexpr,
    M: tl.constexpr,
    M_WORDS: tl.constexpr,
    F_TAIL_PACKED: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Fold `SPS[b, i, j]` mod `f(z)` in registers and compare against target `T_WORDS[b]`."""
    pid = tl.program_id(0)
    row = pid * BLOCK + tl.arange(0, BLOCK)
    m = row < rows

    z = tl.zeros((BLOCK,), dtype=tl.uint32)
    y0 = z
    y1 = z
    y2 = z
    y3 = z
    y4 = z
    y5 = z
    y6 = z
    y7 = z
    y8 = z
    y9 = z
    y10 = z
    y11 = z
    y12 = z
    y13 = z
    y14 = z
    y15 = z
    y16 = z
    y17 = z

    TOP_SHIFT: tl.constexpr = ((M - 1) % 8) * 4
    REM_NIB: tl.constexpr = M % 8
    LAST_MASK: tl.constexpr = 0xFFFFFFFF if REM_NIB == 0 else ((1 << (REM_NIB * 4)) - 1)
    f_tail_u32 = tl.full((), F_TAIL_PACKED, tl.uint32)

    sps_base = row * (K * K * M_WORDS)

    for i in range(K - 1, -1, -1):
        for j in range(i, K, 1):
            # Extract top nibble at index M - 1 (word M_WORDS - 1)
            if M_WORDS == 8:
                top = (y7 >> TOP_SHIFT) & 0xF
            elif M_WORDS == 10:
                top = (y9 >> TOP_SHIFT) & 0xF
            elif M_WORDS == 14:
                top = (y13 >> TOP_SHIFT) & 0xF
            else:
                top = (y17 >> TOP_SHIFT) & 0xF

            # Shift entire M-nibble vector left by 4 bits (multiply by z)
            if M_WORDS > 17:
                y17 = ((y17 << 4) | (y16 >> 28)) & LAST_MASK
            if M_WORDS > 16:
                y16 = (y16 << 4) | (y15 >> 28)
            if M_WORDS > 15:
                y15 = (y15 << 4) | (y14 >> 28)
            if M_WORDS > 14:
                y14 = (y14 << 4) | (y13 >> 28)
            if M_WORDS > 13:
                y13 = (y13 << 4) | (y12 >> 28)
                if M_WORDS == 14:
                    y13 = y13 & LAST_MASK
            if M_WORDS > 12:
                y12 = (y12 << 4) | (y11 >> 28)
            if M_WORDS > 11:
                y11 = (y11 << 4) | (y10 >> 28)
            if M_WORDS > 10:
                y10 = (y10 << 4) | (y9 >> 28)
            if M_WORDS > 9:
                y9 = (y9 << 4) | (y8 >> 28)
                if M_WORDS == 10:
                    y9 = y9 & LAST_MASK
            if M_WORDS > 8:
                y8 = (y8 << 4) | (y7 >> 28)
            y7 = (y7 << 4) | (y6 >> 28)
            if M_WORDS == 8:
                y7 = y7 & LAST_MASK
            y6 = (y6 << 4) | (y5 >> 28)
            y5 = (y5 << 4) | (y4 >> 28)
            y4 = (y4 << 4) | (y3 >> 28)
            y3 = (y3 << 4) | (y2 >> 28)
            y2 = (y2 << 4) | (y1 >> 28)
            y1 = (y1 << 4) | (y0 >> 28)
            y0 = y0 << 4

            # Reduce mod f(z): XOR top * f_tail[0..3] into bottom 4 nibbles of y0
            y0 = y0 ^ _gf16_mul_u32(f_tail_u32, top)

            # Add SPS[i, j] ^ ((i != j) * SPS[j, i])
            ij = sps_base + (i * K + j) * M_WORDS
            ji = sps_base + (j * K + i) * M_WORDS
            neq = i != j
            v0 = tl.load(SPS_IN + ij + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
            v1 = tl.load(SPS_IN + ij + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
            v2 = tl.load(SPS_IN + ij + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
            v3 = tl.load(SPS_IN + ij + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
            v4 = tl.load(SPS_IN + ij + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
            v5 = tl.load(SPS_IN + ij + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
            v6 = tl.load(SPS_IN + ij + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
            v7 = tl.load(SPS_IN + ij + 7, mask=m, other=0).to(tl.uint32, bitcast=True)
            if neq:
                v0 = v0 ^ tl.load(SPS_IN + ji + 0, mask=m, other=0).to(tl.uint32, bitcast=True)
                v1 = v1 ^ tl.load(SPS_IN + ji + 1, mask=m, other=0).to(tl.uint32, bitcast=True)
                v2 = v2 ^ tl.load(SPS_IN + ji + 2, mask=m, other=0).to(tl.uint32, bitcast=True)
                v3 = v3 ^ tl.load(SPS_IN + ji + 3, mask=m, other=0).to(tl.uint32, bitcast=True)
                v4 = v4 ^ tl.load(SPS_IN + ji + 4, mask=m, other=0).to(tl.uint32, bitcast=True)
                v5 = v5 ^ tl.load(SPS_IN + ji + 5, mask=m, other=0).to(tl.uint32, bitcast=True)
                v6 = v6 ^ tl.load(SPS_IN + ji + 6, mask=m, other=0).to(tl.uint32, bitcast=True)
                v7 = v7 ^ tl.load(SPS_IN + ji + 7, mask=m, other=0).to(tl.uint32, bitcast=True)
            y0 = y0 ^ v0
            y1 = y1 ^ v1
            y2 = y2 ^ v2
            y3 = y3 ^ v3
            y4 = y4 ^ v4
            y5 = y5 ^ v5
            y6 = y6 ^ v6
            y7 = y7 ^ v7

            if M_WORDS > 8:
                v8 = tl.load(SPS_IN + ij + 8, mask=m, other=0).to(tl.uint32, bitcast=True)
                v9 = tl.load(SPS_IN + ij + 9, mask=m, other=0).to(tl.uint32, bitcast=True)
                if neq:
                    v8 = v8 ^ tl.load(SPS_IN + ji + 8, mask=m, other=0).to(tl.uint32, bitcast=True)
                    v9 = v9 ^ tl.load(SPS_IN + ji + 9, mask=m, other=0).to(tl.uint32, bitcast=True)
                y8 = y8 ^ v8
                y9 = y9 ^ (v9 & LAST_MASK if M_WORDS == 10 else v9)

            if M_WORDS > 10:
                v10 = tl.load(SPS_IN + ij + 10, mask=m, other=0).to(tl.uint32, bitcast=True)
                v11 = tl.load(SPS_IN + ij + 11, mask=m, other=0).to(tl.uint32, bitcast=True)
                v12 = tl.load(SPS_IN + ij + 12, mask=m, other=0).to(tl.uint32, bitcast=True)
                v13 = tl.load(SPS_IN + ij + 13, mask=m, other=0).to(tl.uint32, bitcast=True)
                if neq:
                    v10 = v10 ^ tl.load(SPS_IN + ji + 10, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v11 = v11 ^ tl.load(SPS_IN + ji + 11, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v12 = v12 ^ tl.load(SPS_IN + ji + 12, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v13 = v13 ^ tl.load(SPS_IN + ji + 13, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                y10 = y10 ^ v10
                y11 = y11 ^ v11
                y12 = y12 ^ v12
                y13 = y13 ^ (v13 & LAST_MASK if M_WORDS == 14 else v13)

            if M_WORDS > 14:
                v14 = tl.load(SPS_IN + ij + 14, mask=m, other=0).to(tl.uint32, bitcast=True)
                v15 = tl.load(SPS_IN + ij + 15, mask=m, other=0).to(tl.uint32, bitcast=True)
                v16 = tl.load(SPS_IN + ij + 16, mask=m, other=0).to(tl.uint32, bitcast=True)
                v17 = tl.load(SPS_IN + ij + 17, mask=m, other=0).to(tl.uint32, bitcast=True)
                if neq:
                    v14 = v14 ^ tl.load(SPS_IN + ji + 14, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v15 = v15 ^ tl.load(SPS_IN + ji + 15, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v16 = v16 ^ tl.load(SPS_IN + ji + 16, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                    v17 = v17 ^ tl.load(SPS_IN + ji + 17, mask=m, other=0).to(
                        tl.uint32, bitcast=True
                    )
                y14 = y14 ^ v14
                y15 = y15 ^ v15
                y16 = y16 ^ v16
                y17 = y17 ^ (v17 & LAST_MASK)

    tb = row * M_WORDS
    diff = (
        (y0 ^ tl.load(T_WORDS + tb + 0, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y1 ^ tl.load(T_WORDS + tb + 1, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y2 ^ tl.load(T_WORDS + tb + 2, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y3 ^ tl.load(T_WORDS + tb + 3, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y4 ^ tl.load(T_WORDS + tb + 4, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y5 ^ tl.load(T_WORDS + tb + 5, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (y6 ^ tl.load(T_WORDS + tb + 6, mask=m, other=0).to(tl.uint32, bitcast=True))
        | (
            (y7 ^ tl.load(T_WORDS + tb + 7, mask=m, other=0).to(tl.uint32, bitcast=True))
            & (LAST_MASK if M_WORDS == 8 else 0xFFFFFFFF)
        )
    )
    if M_WORDS > 8:
        diff = diff | (y8 ^ tl.load(T_WORDS + tb + 8, mask=m, other=0).to(tl.uint32, bitcast=True))
        diff = diff | (
            (y9 ^ tl.load(T_WORDS + tb + 9, mask=m, other=0).to(tl.uint32, bitcast=True))
            & (LAST_MASK if M_WORDS == 10 else 0xFFFFFFFF)
        )
    if M_WORDS > 10:
        diff = diff | (
            y10 ^ tl.load(T_WORDS + tb + 10, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            y11 ^ tl.load(T_WORDS + tb + 11, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            y12 ^ tl.load(T_WORDS + tb + 12, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            (y13 ^ tl.load(T_WORDS + tb + 13, mask=m, other=0).to(tl.uint32, bitcast=True))
            & (LAST_MASK if M_WORDS == 14 else 0xFFFFFFFF)
        )
    if M_WORDS > 14:
        diff = diff | (
            y14 ^ tl.load(T_WORDS + tb + 14, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            y15 ^ tl.load(T_WORDS + tb + 15, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            y16 ^ tl.load(T_WORDS + tb + 16, mask=m, other=0).to(tl.uint32, bitcast=True)
        )
        diff = diff | (
            (y17 ^ tl.load(T_WORDS + tb + 17, mask=m, other=0).to(tl.uint32, bitcast=True))
            & LAST_MASK
        )

    tl.store(OUT_OK + row, (diff == 0).to(tl.uint8), mask=m)
