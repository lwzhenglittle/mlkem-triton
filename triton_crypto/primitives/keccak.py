"""Keccak-f[1600] in Triton (vector engine / SIMT style).

Mapping: **one thread = one Keccak instance**, each of the 25 lanes is a
`[BLOCK]` tensor. Each lane tensor is therefore contiguous along the instance
dimension and memory access coalesces well; in the unrolled code the θ/π lane
permutation shows up as register renaming, with no actual data movement.

Implementation notes:
- All 24 rounds are unrolled (code generation on the Python side), because
  Triton parses kernels as an AST and using Python lists to handle 25 variables
  inside a kernel is awkward.
- Watch out for Python 3.14 + Triton 3.5.0: `range(N)` with 1 argument does not
  work inside a kernel. Here we simply fully unroll, with no loop at all.
"""

from __future__ import annotations

import numpy as np

from .._codegen import compile_generated

# ---------------- Keccak constants ----------------

# r[x][y]: rotation amount for the ρ step
ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]


def _rotl_expr(var: str, r: int) -> str:
    if r == 0:
        return var
    return f"(({var} << {r}) | ({var} >> {64 - r}))"


def _emit_rounds(L: list[str], rnd: int, ind: str = "    ", rc_expr: str | None = None) -> None:
    """Emit one Keccak-f round on the Python-side variables a0..a24.

    This is the single round emitter: the standalone permutation kernel and the sponge
    kernel (which rolls the round loop and reads the constant from a tensor) both use it.
    """
    rc = rc_expr if rc_expr is not None else str(RC[rnd])
    for x in range(5):
        L.append(f"{ind}c{x} = a{x} ^ a{x + 5} ^ a{x + 10} ^ a{x + 15} ^ a{x + 20}")
    for x in range(5):
        cp, cn = (x + 1) % 5, (x + 4) % 5
        L.append(f"{ind}d{x} = c{cn} ^ {_rotl_expr(f'c{cp}', 1)}")
    for y in range(5):
        for x in range(5):
            i = x + 5 * y
            L.append(f"{ind}a{i} = a{i} ^ d{x}")
    for y in range(5):
        for x in range(5):
            src = x + 5 * y
            dst = y + 5 * ((2 * x + 3 * y) % 5)
            L.append(f"{ind}b{dst} = {_rotl_expr(f'a{src}', ROT[x][y])}")
    for y in range(5):
        for x in range(5):
            L.append(
                f"{ind}a{x + 5 * y} = b{x + 5 * y} ^ "
                f"((~b{(x + 1) % 5 + 5 * y}) & b{(x + 2) % 5 + 5 * y})"
            )
    L.append(f"{ind}a0 = a0 ^ {rc}")


def gen_source(n_rounds: int = 24) -> str:
    """Generate the full Triton kernel source (rounds unrolled)."""
    L = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def keccak_f_kernel(X, OUT, n, BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = offs < n",
    ]
    for i in range(25):
        L.append(
            f"    a{i} = tl.load(X + {i} * n + offs, mask=m, other=0).to(tl.uint64, bitcast=True)"
        )

    for rnd in range(n_rounds):
        _emit_rounds(L, rnd, ind="    ")

    for i in range(25):
        L.append(f"    tl.store(OUT + {i} * n + offs, a{i}.to(tl.int64, bitcast=True), mask=m)")
    return "\n".join(L) + "\n"


_KERNEL = None


def get_kernel():
    """Generate the source -> write a temp file -> import (Triton needs the source)."""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    mod = compile_generated(gen_source(), "keccak")
    _KERNEL = mod.keccak_f_kernel
    return _KERNEL


# ---------------- pure-Python reference implementation ----------------

M64 = (1 << 64) - 1


def _rotl(v: int, r: int) -> int:
    if r == 0:
        return v & M64
    return ((v << r) | (v >> (64 - r))) & M64


def keccak_f_ref(state: np.ndarray) -> np.ndarray:
    """state: [25] uint64. Runs 24 rounds on a copy and returns the new state."""
    a = [int(x) & M64 for x in state]
    for rnd in range(24):
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x + 4) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for y in range(5):
            for x in range(5):
                a[x + 5 * y] ^= d[x]
        b = [0] * 25
        for y in range(5):
            for x in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rotl(a[x + 5 * y], ROT[x][y])
        for y in range(5):
            for x in range(5):
                a[x + 5 * y] = (
                    b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & M64) & b[(x + 2) % 5 + 5 * y]
                )
        a[0] ^= RC[rnd]
    return np.array([x & M64 for x in a], dtype=np.uint64)


__all__ = ["RC", "ROT", "gen_source", "get_kernel", "keccak_f_ref"]
