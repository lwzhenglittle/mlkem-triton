"""Official NIST Round 2 UOV parameter sets (OV-Is, OV-Ip, OV-III, OV-V)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ALL_SETS",
    "BY_NAME",
    "OV_III",
    "OV_IP",
    "OV_IS",
    "OV_V",
    "UOVParams",
    "from_pk_len",
]


@dataclass(frozen=True)
class UOVParams:
    name: str
    q: int
    n: int
    m: int
    salt_bytes: int = 16
    pk_seed_bytes: int = 16

    @property
    def o(self) -> int:
        return self.m

    @property
    def v(self) -> int:
        return self.n - self.m

    @property
    def m_bytes(self) -> int:
        return self.m // 2 if self.q == 16 else self.m

    @property
    def n_bytes(self) -> int:
        return self.n // 2 if self.q == 16 else self.n

    @property
    def sig_bytes(self) -> int:
        return self.n_bytes + self.salt_bytes

    @property
    def num_tri(self) -> int:
        return self.n * (self.n + 1) // 2

    @property
    def p1_bytes(self) -> int:
        return self.m_bytes * (self.v * (self.v + 1) // 2)

    @property
    def p2_bytes(self) -> int:
        return self.m_bytes * (self.v * self.o)

    @property
    def p3_bytes(self) -> int:
        return self.m_bytes * (self.o * (self.o + 1) // 2)

    @property
    def pk_bytes(self) -> int:
        return self.m_bytes * self.num_tri

    @property
    def cpk_bytes(self) -> int:
        return self.pk_seed_bytes + self.p3_bytes

    @property
    def m_u32_words(self) -> int:
        return self.m_bytes // 4


OV_IS = UOVParams(name="OV-Is", q=16, n=160, m=64)
OV_IP = UOVParams(name="OV-Ip", q=256, n=112, m=44)
OV_III = UOVParams(name="OV-III", q=256, n=184, m=72)
OV_V = UOVParams(name="OV-V", q=256, n=244, m=96)

ALL_SETS: tuple[UOVParams, ...] = (OV_IS, OV_IP, OV_III, OV_V)
BY_NAME: dict[str, UOVParams] = {}
for _p in ALL_SETS:
    BY_NAME[_p.name] = _p
    BY_NAME[_p.name.replace("-", "_")] = _p
    BY_NAME[f"{_p.name}-pkc"] = _p
    BY_NAME[f"{_p.name}-pkc-skc"] = _p


def from_pk_len(nbytes: int) -> UOVParams:
    for p in ALL_SETS:
        if nbytes in (p.pk_bytes, p.cpk_bytes):
            return p
    valid = ", ".join(f"{p.name}={p.pk_bytes}/{p.cpk_bytes}" for p in ALL_SETS)
    raise ValueError(f"unrecognised UOV public key length {nbytes} (expected {valid})")
