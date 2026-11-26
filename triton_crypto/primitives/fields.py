"""Prime-field arithmetic that every modular primitive shares.

Before this module, "multiply modulo q" existed in exactly one place — inside
`primitives/ntt.py` — inlined three times, because a `@triton.jit` helper defined inside a
factory that did `import triton.language as tl` locally could not be shared: the compiled
helper has no `tl` in its globals and the backend fails with `NameError: tl is not defined`.
Every primitive that needs modular arithmetic (Poseidon2, a BabyBear NTT, X25519's field,
Falcon) would have paid for that by copying the same four lines again.

**The rule this module establishes: import `triton.language as tl` at module level in any
module that defines a `@triton.jit` helper meant to be called from elsewhere.** A module-level
import is what makes the helper's globals complete; nothing about the helper body is special.
`tests/test_fields.py` pins this down by calling `mont_mul` from a kernel in another module.

Arithmetic model: 32-bit Montgomery (REDC) with ``R = 2**32``, valid for odd ``q < 2**31``:

    prod = a*b                      # < q^2 < 2^62, fits int64
    u    = (prod * q^-1) mod 2^32
    out  = (prod - u*q) >> 32       # == a*b*R^-1 mod q; canonicalised to [0, q)

``a`` and ``b`` are int64 in ``[0, q)``; the two constraints are ``q`` odd and ``q < 2**31``
(the latter because ``u*q < 2**63``). ``q >= 2**31`` (Goldilocks) needs a 64x64->128 path and
is **not** implemented here — `PrimeField` rejects it rather than silently overflowing.

This stays a *field* module: no NTT tables, no twiddle factors, no scheme parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import triton
import triton.language as tl

__all__ = ["BABYBEAR", "MONT_R", "PrimeField", "mont_mul", "mont_mul_np"]

MONT_R = 1 << 32
_M32 = 0xFFFFFFFF
_MONT_LIMIT = 1 << 31


@triton.jit
def mont_mul(a, b, Q: tl.constexpr, QINV: tl.constexpr):
    """`a * b * R^-1 mod Q`, canonical in `[0, Q)`, for int64 operands in `[0, Q)`.

    Callable from any kernel in the package (see the module docstring): `tl` is imported at
    module level here, which is what keeps the compiled helper's globals complete.
    """
    prod = a * b
    u = ((prod & 0xFFFFFFFF) * QINV) & 0xFFFFFFFF
    # REDC's raw output is in (-q, q): `prod` can be smaller than `u*q`, so the shift of a
    # negative value floors. Canonicalise here so that every caller gets [0, q); the three
    # inlined copies this replaced all depended on a following `_norm` and a new caller should
    # not have to know that.
    r = (prod - u * Q) >> 32
    return tl.where(r < 0, r + Q, r)


def mont_mul_np(a, b, field: PrimeField) -> np.ndarray:
    """numpy twin of `mont_mul`, for references and tests."""
    prod = np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)
    # `(prod & 2**32-1) * qinv` overflows int64 deliberately: two's-complement multiplication
    # is exact mod 2**64, and only the low 32 bits are kept, which is the value wanted.
    u = ((prod & _M32) * field.qinv) & _M32
    r = (prod - u * field.q) >> 32
    return np.where(r < 0, r + field.q, r)


@dataclass(frozen=True)
class PrimeField:
    """An odd prime `q < 2**31` with its 32-bit Montgomery constants.

    Only the arithmetic lives here. A caller that needs a root of unity, a twiddle table or a
    modulus-specific reduction strategy keeps that in its own module (`primitives/ntt/`).
    """

    name: str
    q: int

    def __post_init__(self) -> None:
        if self.q < 3:
            raise ValueError(f"{self.name}: q must be >= 3, got {self.q}")
        if self.q % 2 == 0:
            raise ValueError(f"{self.name}: q must be odd for Montgomery arithmetic, got {self.q}")
        if self.q >= _MONT_LIMIT:
            raise ValueError(
                f"{self.name}: q = {self.q} >= 2**31; the 32-bit Montgomery path needs "
                "q < 2**31 (Goldilocks-class moduli need a 128-bit product and are not "
                "implemented)"
            )

    # ---- Montgomery constants -------------------------------------------------

    @cached_property
    def qinv(self) -> int:
        """`q^-1 mod 2**32` (exists because q is odd)."""
        return pow(self.q, -1, MONT_R)

    @cached_property
    def mont_r(self) -> int:
        """`R mod q`, the multiplicative identity in Montgomery form."""
        return MONT_R % self.q

    @cached_property
    def bits(self) -> int:
        return self.q.bit_length()

    def to_mont(self, a):
        """Standard representative -> Montgomery form (`a*R mod q`)."""
        return (np.asarray(a, dtype=np.int64) * self.mont_r) % self.q

    def from_mont(self, a):
        """Montgomery form -> standard representative."""
        return mont_mul_np(a, 1, self)

    # ---- numpy field operations ----------------------------------------------

    def reduce(self, a) -> np.ndarray:
        return np.asarray(a, dtype=np.int64) % self.q

    def add(self, a, b) -> np.ndarray:
        return (np.asarray(a, dtype=np.int64) + np.asarray(b, dtype=np.int64)) % self.q

    def sub(self, a, b) -> np.ndarray:
        return (np.asarray(a, dtype=np.int64) - np.asarray(b, dtype=np.int64)) % self.q

    def mul(self, a, b) -> np.ndarray:
        return (np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)) % self.q

    def pow(self, a, e: int) -> np.ndarray:
        """`a**e` in the field (square-and-multiply on Python ints; reference path only)."""
        a = np.asarray(a, dtype=np.int64)
        out = np.ones_like(a)
        base = a % self.q
        while e:
            if e & 1:
                out = (out * base) % self.q
            base = (base * base) % self.q
            e >>= 1
        return out

    def inv(self, a) -> np.ndarray:
        """`a**(q-2)`, i.e. the field inverse for prime q."""
        return self.pow(a, self.q - 2)

    def is_canonical(self, a) -> bool:
        a = np.asarray(a)
        return bool(((a >= 0) & (a < self.q)).all())

    # ---- kernel constants -----------------------------------------------------

    @property
    def kernel_consts(self) -> tuple[int, int]:
        """`(q, qinv)` in the order `mont_mul` expects."""
        return self.q, self.qinv


BABYBEAR = PrimeField("BabyBear", 2013265921)  # 2**31 - 2**27 + 1, 2-adicity 27
