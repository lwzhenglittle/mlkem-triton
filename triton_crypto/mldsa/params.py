"""ML-DSA parameter sets (FIPS 204, Table 1)."""

from __future__ import annotations

from dataclasses import dataclass

Q = 8380417
N = 256
D = 13  # dropped bits: t = t1·2^d + t0


def _bits(x: int) -> int:
    """Bit length, i.e. ⌈log2(x+1)⌉."""
    return x.bit_length()


@dataclass(frozen=True)
class DSAParams:
    name: str
    k: int
    l: int
    eta: int
    tau: int
    beta: int
    gamma1: int
    gamma2: int
    omega: int
    ctilde_bytes: int

    # ---- derived sizes ----
    @property
    def z_bits(self) -> int:
        return 1 + _bits(self.gamma1 - 1)

    @property
    def w1_bits(self) -> int:
        return _bits((Q - 1) // (2 * self.gamma2) - 1)

    @property
    def polyz_bytes(self) -> int:
        return 32 * self.z_bits

    @property
    def polyw1_bytes(self) -> int:
        return 32 * self.w1_bits

    @property
    def polyeta_bytes(self) -> int:
        return 32 * _bits(2 * self.eta)

    @property
    def polyt1_bytes(self) -> int:
        return 320

    @property
    def polyt0_bytes(self) -> int:
        return 416

    # ---- byte lengths (FIPS 204 §5) ----
    @property
    def pk_size(self) -> int:
        return 32 + self.k * self.polyt1_bytes

    @property
    def sk_size(self) -> int:
        return (
            32 * 2
            + 64
            + self.l * self.polyeta_bytes
            + self.k * self.polyeta_bytes
            + self.k * self.polyt0_bytes
        )

    @property
    def sig_size(self) -> int:
        return self.ctilde_bytes + self.l * self.polyz_bytes + self.omega + self.k

    # ---- offsets inside a signature (c̃ ‖ z ‖ h) ----
    @property
    def z_offset(self) -> int:
        return self.ctilde_bytes

    @property
    def h_offset(self) -> int:
        return self.ctilde_bytes + self.l * self.polyz_bytes

    # ---- rounding constants ----
    @property
    def two_gamma2(self) -> int:
        return 2 * self.gamma2

    @property
    def m(self) -> int:
        """Number of HighBits buckets: (q-1)/(2γ2)."""
        return (Q - 1) // (2 * self.gamma2)

    @property
    def norm_bound(self) -> int:
        """γ1 − β: the infinity-norm bound on z."""
        return self.gamma1 - self.beta


PARAMS = {
    "ML-DSA-44": DSAParams(
        "ML-DSA-44",
        k=4,
        l=4,
        eta=2,
        tau=39,
        beta=78,
        gamma1=1 << 17,
        gamma2=(Q - 1) // 88,
        omega=80,
        ctilde_bytes=32,
    ),
    "ML-DSA-65": DSAParams(
        "ML-DSA-65",
        k=6,
        l=5,
        eta=4,
        tau=49,
        beta=196,
        gamma1=1 << 19,
        gamma2=(Q - 1) // 32,
        omega=55,
        ctilde_bytes=48,
    ),
    "ML-DSA-87": DSAParams(
        "ML-DSA-87",
        k=8,
        l=7,
        eta=2,
        tau=60,
        beta=120,
        gamma1=1 << 19,
        gamma2=(Q - 1) // 32,
        omega=75,
        ctilde_bytes=64,
    ),
}

__all__ = ["PARAMS", "D", "DSAParams", "N", "Q"]
