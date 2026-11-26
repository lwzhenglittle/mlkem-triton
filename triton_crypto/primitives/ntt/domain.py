"""Negacyclic NTT domains: `Z_q[X]/(X^N+1)`, a root of unity, and the twiddle tables.

This module is data, not launch code: `kernel.py` owns the Triton side, `ref.py` the numpy
reference. Splitting it out is what lets the large-N transform (2^20 points, a different
algorithm with global-memory passes) arrive later without touching either.

Both standards this package implements use the same butterfly and the same twiddle
ordering (`ζ^BitRev7(k)`), and differ only in the modulus, in the order of ζ, and in
whether the transform runs to the last layer:

    ML-KEM (FIPS 203): q = 3329,    ζ = 17,   order N    -> 7 layers, int32
    ML-DSA (FIPS 204): q = 8380417, ζ = 1753, order 2N   -> 8 layers, int64
    BabyBear (ZK):     q = 2^31-2^27+1, ζ = 31^15, order 2N -> 8 layers, int64+Montgomery

ML-KEM stops one layer early on purpose: its 128 CRT slots are the quadratic extensions
Z_q[X]/(X²−γ), which is why FIPS 203 Alg. 12 needs a base-case multiply. ML-DSA splits
all the way into 256 linear slots, so its pointwise product is a plain Hadamard. The
butterfly is identical; what the products *mean* afterwards is the scheme's business, and
stays in the scheme's module (`triton_crypto.mlkem.ntt_ref.multiply_ntts`).

Everything here works in the **standard domain**: no Montgomery representation leaks into
the twiddle tables, so the output is directly comparable with the coefficients in
FIPS 203/204. (The reference implementations fold a factor of R⁻¹ into their tables,
which is exactly the kind of detail that makes cross-scheme reuse painful.)
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from ..fields import PrimeField

__all__ = [
    "BABYBEAR_NTT",
    "DILITHIUM",
    "KYBER",
    "NTTDomain",
    "bitrev",
    "mldsa_domain",
    "mlkem_domain",
]


def bitrev(k: int, bits: int) -> int:
    """`bits`-wide bit reversal (BitRev7 for ML-KEM, BitRev8 for ML-DSA)."""
    return int(format(k, f"0{bits}b")[::-1], 2)


def _multiplicative_order(a: int, q: int) -> int:
    """Order of `a` in (Z/q)ˣ, by removing prime factors of q-1 while the power stays 1."""
    order = q - 1
    rest = order
    p = 2
    while p * p <= rest:
        if rest % p == 0:
            while rest % p == 0:
                rest //= p
            while order % p == 0 and pow(a, order // p, q) == 1:
                order //= p
        p += 1
    if rest > 1:
        while order % rest == 0 and pow(a, order // rest, q) == 1:
            order //= rest
    return order


@dataclass(frozen=True)
class NTTDomain:
    """A negacyclic NTT domain: Z_q[X]/(X^N+1), a root ζ, and a layer count."""

    name: str
    q: int
    n: int
    zeta: int
    layers: int = 0  # 0 -> run all log2(N) layers

    def __post_init__(self):
        if self.n & (self.n - 1):
            raise ValueError("N must be a power of two")
        if self.layers == 0:
            object.__setattr__(self, "layers", self.n.bit_length() - 1)
        # The Montgomery path is 32-bit wide (R = 2**32), so the modulus has to fit.
        # Touching `field` here makes a Goldilocks-class q fail at construction rather
        # than in the middle of a kernel launch.
        _ = self.field  # validates q (raises for q >= 2**31)
        order = _multiplicative_order(self.zeta, self.q)
        if order != 2 ** (self.layers + 1):
            raise ValueError(
                f"ζ={self.zeta} has order {order} mod {self.q}, but {self.layers} layers "
                f"need order {2 ** (self.layers + 1)}"
            )

    @property
    def field(self) -> PrimeField:
        """The prime field this domain works over (`primitives.fields`)."""
        return PrimeField(self.name, self.q)

    @property
    def log_n(self) -> int:
        return self.n.bit_length() - 1

    @property
    def half(self) -> int:
        return self.n // 2

    @property
    def wide(self) -> bool:
        """Whether products of two reduced coefficients overflow int32."""
        return (self.q - 1) * (self.q - 1) > 2**31 - 1

    @cached_property
    def zetas(self) -> np.ndarray:
        """ζ^BitRev(k), k = 0 .. 2^layers-1 (the FIPS 203/204 twiddle order)."""
        return np.array(
            [pow(self.zeta, bitrev(k, self.layers), self.q) for k in range(2**self.layers)],
            dtype=np.int64,
        )

    @cached_property
    def scale(self) -> int:
        """2^-layers mod q: the normalisation the inverse needs in the standard domain.

        Not N⁻¹ in general: ML-KEM's transform stops after 7 layers, so it has 2^7 CRT
        slots and FIPS 203 Alg. 10 normalises by 128⁻¹ (= 3303), not 256⁻¹.
        """
        return pow(2**self.layers, -1, self.q)

    # layer geometry, shared by every implementation below
    def layer_len(self, s: int, inverse: bool = False) -> int:
        """Length `L` of the butterfly blocks in layer `s` (`0 <= s < layers`)."""
        if inverse:
            return 1 << (self.log_n - self.layers + s)
        return 1 << (self.log_n - 1 - s)

    @cached_property
    def qinv(self) -> int:
        """q⁻¹ mod 2³² — the Montgomery constant (only used on the wide path)."""
        return self.field.qinv

    @cached_property
    def mont_r(self) -> int:
        """R = 2³² mod q."""
        return self.field.mont_r

    @cached_property
    def tables(self) -> dict[str, np.ndarray]:
        """Per-position twiddles, `[layers*N]` each, ready for the kernel.

        Forward layer `s` uses zetas `2^s .. 2^(s+1)-1` ascending with the block index;
        the inverse walks the same blocks in reverse (FIPS 203 Alg. 10 / FIPS 204 Alg. 42),
        which is what makes one kernel serve both directions.
        """
        z = self.zetas
        n, layers = self.n, self.layers
        fwd = np.empty(layers * n, dtype=np.int64)
        inv = np.empty(layers * n, dtype=np.int64)
        for s in range(layers):
            for direction in ("fwd", "inv"):
                length = self.layer_len(s, inverse=(direction == "inv"))
                b = np.arange(n) // (2 * length)
                if direction == "fwd":
                    idx = (1 << s) + b
                    fwd[s * n : (s + 1) * n] = z[idx]
                else:
                    k0 = 1 << (self.layers - s)  # = 2 * blocks
                    inv[s * n : (s + 1) * n] = z[k0 - 1 - b]
        out = {"fwd": fwd, "inv": inv}
        if self.wide:
            # Montgomery form for the GPU path: the butterfly then computes
            # mont(zeta·R, b) = zeta·b, i.e. the values stay in the standard domain but the
            # reduction is a multiply-shift instead of a 64-bit division.
            r = self.mont_r
            out["fwd_mont"] = (fwd * r) % self.q
            out["inv_mont"] = (inv * r) % self.q
        return out

    # ---- numpy reference (same table layout as the kernel) ----

    def _reduce(self, x):
        return np.asarray(x, dtype=np.int64) % self.q

    def ntt(self, f) -> np.ndarray:
        """Forward transform: one vectorised gather per layer (same tables as the kernel)."""
        f = self._reduce(f).copy()
        q, n = self.q, self.n
        tab = self.tables["fwd"]
        for s in range(self.layers):
            length = self.layer_len(s)
            zetas = tab[s * n : (s + 1) * n : 2 * length]  # one twiddle per block
            x = f.reshape(n // (2 * length), 2 * length)
            a = x[:, :length].copy()
            b = x[:, length:]
            t = (zetas[:, None] * b) % q
            x[:, :length] = (a + t) % q
            x[:, length:] = (a - t) % q
        return f

    def intt(self, f) -> np.ndarray:
        """Inverse transform: layers reversed, then the 2^-layers normalisation."""
        f = self._reduce(f).copy()
        q, n = self.q, self.n
        tab = self.tables["inv"]
        for s in range(self.layers):
            length = self.layer_len(s, inverse=True)
            zetas = tab[s * n : (s + 1) * n : 2 * length]
            x = f.reshape(n // (2 * length), 2 * length)
            a = x[:, :length].copy()
            b = x[:, length:].copy()
            x[:, :length] = (a + b) % q
            x[:, length:] = (zetas[:, None] * (b - a)) % q
        return (f * self.scale) % q

    def mul(self, a, b) -> np.ndarray:
        """Pointwise (Hadamard) product mod q."""
        return (np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)) % self.q

    def conv(self, a, b) -> np.ndarray:
        """Schoolbook negacyclic convolution (O(N²)); for tests, not for use."""
        n = self.n
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        idx = np.arange(n)[:, None] + np.arange(n)[None, :]
        sign = np.where(idx < n, 1, -1)
        # Reduce every product *before* accumulating. The product itself is < q² < 2^62
        # (fine), but summing N of them overflowed int64 for q beyond ~2^24; after the
        # reduction the running sum is bounded by N·q < 2^39 for any q this module accepts.
        prod = (np.outer(a, b) % self.q) * sign
        c = np.zeros(n, dtype=np.int64)
        np.add.at(c, idx % n, prod)
        return c % self.q

    def toeplitz(self, a) -> np.ndarray:
        """`T(a)[i][j] = a[(i-j) mod N] · (+1 if i >= j else -1)`, so `T(a) @ b == a·b`."""
        a = np.asarray(a, dtype=np.int64)
        i = np.arange(self.n)[:, None]
        j = np.arange(self.n)[None, :]
        return a[(i - j) % self.n] * np.where(i >= j, 1, -1)


KYBER = NTTDomain("ML-KEM", q=3329, n=256, zeta=17, layers=7)
DILITHIUM = NTTDomain("ML-DSA", q=8380417, n=256, zeta=1753, layers=8)

# BabyBear, the ZK/Poseidon2 modulus. 31^((q-1)/512) has order 512 = 2N for N = 256; the same
# domain works for any N = 2^k with k <= 26 once the kernel supports that size (the
# in-register kernel is good to N <= 1024 — 2^20 is a different algorithm, not a bigger tile).
BABYBEAR_NTT = NTTDomain(
    "BabyBear", q=2013265921, n=256, zeta=pow(31, (2013265921 - 1) // 512, 2013265921), layers=8
)

# Backend-friendly aliases: the names match the schemes, the objects are the domains.
mlkem_domain = KYBER
mldsa_domain = DILITHIUM
