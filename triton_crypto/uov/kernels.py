"""Pure-Triton kernels for batched UOV quadratic public-map evaluation over GF(16) and GF(256)."""

from __future__ import annotations

import triton
import triton.language as tl

from ..mayo.kernels import _gf16_mul_u32

__all__ = [
    "_gf256_mul_u32",
    "_uov_eval_gf16_kernel",
    "_uov_eval_gf256_kernel",
]


@triton.jit
def _gf256_mul_u32(a, b):
    """Multiply 4 packed GF(256) bytes in `uint32` `a` by byte scalar `b` mod (x^8+x^4+x^3+x+1)."""
    a = a.to(tl.uint32)
    b = b.to(tl.uint32)
    m80 = tl.full((), 0x80808080, tl.uint32)

    r = a * (b & 1)
    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 1) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 2) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 3) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 4) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 5) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 6) & 1))

    msb = a & m80
    a = ((a ^ msb) << 1) ^ ((msb >> 7) * 0x1B)
    r = r ^ (a * ((b >> 7) & 1))
    return r


@triton.jit
def _uov_eval_gf16_kernel(
    P_TRI,
    X_ELEMS_T,
    OUT,
    rows,
    N: tl.constexpr,
    M_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Batched UOV public map over GF(16) (`OV-Is`): `[T, M_WORDS]` x `[N, rows]` -> `[rows, M_WORDS]`."""
    pid_b = tl.program_id(0)
    w = tl.program_id(1)
    row = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    m = row < rows

    acc = tl.zeros((BLOCK_B,), dtype=tl.uint32)

    for i in range(0, N, 1):
        row_acc = tl.zeros((BLOCK_B,), dtype=tl.uint32)
        row_off = i * N - (i * (i - 1)) // 2
        for j in range(i, N, 1):
            idx = row_off + (j - i)
            p_ij = tl.load(P_TRI + idx * M_WORDS + w).to(tl.uint32, bitcast=True)
            xj = tl.load(X_ELEMS_T + j * rows + row, mask=m, other=0)
            row_acc = row_acc ^ _gf16_mul_u32(p_ij, xj)
        xi = tl.load(X_ELEMS_T + i * rows + row, mask=m, other=0)
        acc = acc ^ _gf16_mul_u32(row_acc, xi)

    tl.store(OUT + row * M_WORDS + w, acc.to(tl.int32, bitcast=True), mask=m)


@triton.jit
def _uov_eval_gf256_kernel(
    P_TRI,
    X_ELEMS_T,
    OUT,
    rows,
    N: tl.constexpr,
    M_WORDS: tl.constexpr,
    BLOCK_B: tl.constexpr,
):
    """Batched UOV public map over GF(256) (`OV-Ip/III/V`): `[T, M_WORDS]` x `[N, rows]` -> `[rows, M_WORDS]`."""
    pid_b = tl.program_id(0)
    w = tl.program_id(1)
    row = pid_b * BLOCK_B + tl.arange(0, BLOCK_B)
    m = row < rows

    acc = tl.zeros((BLOCK_B,), dtype=tl.uint32)

    for i in range(0, N, 1):
        row_acc = tl.zeros((BLOCK_B,), dtype=tl.uint32)
        row_off = i * N - (i * (i - 1)) // 2
        for j in range(i, N, 1):
            idx = row_off + (j - i)
            p_ij = tl.load(P_TRI + idx * M_WORDS + w).to(tl.uint32, bitcast=True)
            xj = tl.load(X_ELEMS_T + j * rows + row, mask=m, other=0)
            row_acc = row_acc ^ _gf256_mul_u32(p_ij, xj)
        xi = tl.load(X_ELEMS_T + i * rows + row, mask=m, other=0)
        acc = acc ^ _gf256_mul_u32(row_acc, xi)

    tl.store(OUT + row * M_WORDS + w, acc.to(tl.int32, bitcast=True), mask=m)
