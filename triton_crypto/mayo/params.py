"""Official NIST Round 2 MAYO parameter sets (MAYO-1, MAYO-2, MAYO-3, MAYO-5)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "ALL_SETS",
    "BY_NAME",
    "MAYO_1",
    "MAYO_2",
    "MAYO_3",
    "MAYO_5",
    "MayoParams",
    "from_pk_len",
]


@dataclass(frozen=True)
class MayoParams:
    name: str
    n: int
    m: int
    o: int
    k: int
    f_tail: tuple[int, int, int, int]
    m_bytes: int
    p1_bytes: int
    p2_bytes: int
    p3_bytes: int
    cpk_bytes: int
    sig_bytes: int
    salt_bytes: int
    digest_bytes: int
    pk_seed_bytes: int = 16

    @property
    def v(self) -> int:
        return self.n - self.o

    @property
    def m_u32_words(self) -> int:
        return (self.m + 7) // 8

    @property
    def f_tail_packed(self) -> int:
        f0, f1, f2, f3 = self.f_tail
        return f0 | (f1 << 4) | (f2 << 8) | (f3 << 12)


MAYO_1 = MayoParams(
    name="MAYO-1",
    n=86,
    m=78,
    o=8,
    k=10,
    f_tail=(8, 1, 1, 0),
    m_bytes=39,
    p1_bytes=120159,
    p2_bytes=24336,
    p3_bytes=1404,
    cpk_bytes=1420,
    sig_bytes=454,
    salt_bytes=24,
    digest_bytes=32,
)

MAYO_2 = MayoParams(
    name="MAYO-2",
    n=81,
    m=64,
    o=17,
    k=4,
    f_tail=(8, 0, 2, 8),
    m_bytes=32,
    p1_bytes=66560,
    p2_bytes=34816,
    p3_bytes=4896,
    cpk_bytes=4912,
    sig_bytes=186,
    salt_bytes=24,
    digest_bytes=32,
)

MAYO_3 = MayoParams(
    name="MAYO-3",
    n=118,
    m=108,
    o=10,
    k=11,
    f_tail=(8, 0, 1, 7),
    m_bytes=54,
    p1_bytes=317844,
    p2_bytes=58320,
    p3_bytes=2970,
    cpk_bytes=2986,
    sig_bytes=681,
    salt_bytes=32,
    digest_bytes=48,
)

MAYO_5 = MayoParams(
    name="MAYO-5",
    n=154,
    m=142,
    o=12,
    k=12,
    f_tail=(4, 0, 8, 1),
    m_bytes=71,
    p1_bytes=720863,
    p2_bytes=120984,
    p3_bytes=5538,
    cpk_bytes=5554,
    sig_bytes=964,
    salt_bytes=40,
    digest_bytes=64,
)

ALL_SETS: tuple[MayoParams, ...] = (MAYO_1, MAYO_2, MAYO_3, MAYO_5)
BY_NAME: dict[str, MayoParams] = {
    **{p.name: p for p in ALL_SETS},
    **{p.name.replace("-", "_"): p for p in ALL_SETS},
}


def from_pk_len(nbytes: int) -> MayoParams:
    for p in ALL_SETS:
        if p.cpk_bytes == nbytes:
            return p
    valid = ", ".join(f"{p.name}={p.cpk_bytes}" for p in ALL_SETS)
    raise ValueError(f"unrecognised MAYO compact public key length {nbytes} (expected {valid})")
