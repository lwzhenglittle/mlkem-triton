"""ML-KEM parameter sets (FIPS 203)."""

from __future__ import annotations

from dataclasses import dataclass

Q = 3329
N = 256


@dataclass(frozen=True)
class Params:
    name: str
    k: int
    eta1: int
    eta2: int
    du: int
    dv: int

    # ---- byte lengths (FIPS 203 §7) ----
    @property
    def ek_size(self) -> int:
        return 384 * self.k + 32

    @property
    def dk_size(self) -> int:
        return 768 * self.k + 96

    @property
    def ct_size(self) -> int:
        return 32 * (self.du * self.k + self.dv)

    @property
    def dk_pke_size(self) -> int:
        return 384 * self.k

    @property
    def ek_pke_size(self) -> int:
        return 384 * self.k


PARAMS = {
    "ML-KEM-512": Params("ML-KEM-512", k=2, eta1=3, eta2=2, du=10, dv=4),
    "ML-KEM-768": Params("ML-KEM-768", k=3, eta1=2, eta2=2, du=10, dv=4),
    "ML-KEM-1024": Params("ML-KEM-1024", k=4, eta1=2, eta2=2, du=11, dv=5),
}

__all__ = ["PARAMS", "N", "Params", "Q"]
