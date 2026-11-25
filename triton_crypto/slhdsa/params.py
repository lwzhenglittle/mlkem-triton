"""FIPS 205 (SLH-DSA) parameter sets: 6 sizes x 2 hash families.

Only what verification needs is derived here, but the derivations are the ones in the standard,
so nothing is hard-coded that could be computed (and get out of sync with a table).

Notation follows FIPS 205 §7/§11: `n` hash size, `h` total hypertree height, `d` layers, `h' = h/d`
height of one XMSS tree, `a` FORS tree height, `k` FORS trees, `lg_w = 4` (w = 16) for every set.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["HASHES", "PARAMS", "SLHParams", "get"]


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class SLHParams:
    """One parameter set. All byte sizes are derived from the FIPS 205 formulas."""

    name: str
    family: str  # "SHAKE" or "SHA2"
    n: int  # security parameter: 16 / 24 / 32
    h: int  # total tree height
    d: int  # number of layers
    a: int  # FORS tree height
    k: int  # FORS tree count
    lg_w: int = 4  # w = 16 for all FIPS 205 sets

    # ---- derived ----------------------------------------------------------
    @property
    def hp(self) -> int:
        """h' — height of each XMSS tree."""
        return self.h // self.d

    @property
    def w(self) -> int:
        return 1 << self.lg_w

    @property
    def len1(self) -> int:
        """Number of base-w digits in one hashed message."""
        return _ceil_div(8 * self.n, self.lg_w)

    @property
    def len2(self) -> int:
        """Digits of the checksum."""
        return ((self.len1 * (self.w - 1)).bit_length() - 1) // self.lg_w + 1

    @property
    def wots_len(self) -> int:
        return self.len1 + self.len2

    @property
    def m(self) -> int:
        """Digest length: FORS indices, then tree index, then leaf index."""
        return (
            _ceil_div(self.k * self.a, 8) + _ceil_div(self.h - self.hp, 8) + _ceil_div(self.hp, 8)
        )

    @property
    def md_bytes(self) -> int:
        """Bytes of the digest that select the FORS indices."""
        return _ceil_div(self.k * self.a, 8)

    @property
    def tree_bytes(self) -> int:
        """Bytes of the digest that address the tree (h - h' bits)."""
        return _ceil_div(self.h - self.hp, 8)

    @property
    def leaf_bytes(self) -> int:
        """Bytes of the digest that address the leaf (h' bits)."""
        return _ceil_div(self.hp, 8)

    @property
    def pk_size(self) -> int:
        return 2 * self.n

    @property
    def sk_size(self) -> int:
        return 4 * self.n

    @property
    def fors_sig_size(self) -> int:
        return self.k * (self.a + 1) * self.n

    @property
    def xmss_sig_size(self) -> int:
        return (self.wots_len + self.hp) * self.n

    @property
    def sig_size(self) -> int:
        """R || SIG_FORS || SIG_HT"""
        return self.n + self.fors_sig_size + self.d * self.xmss_sig_size

    @property
    def is_shake(self) -> bool:
        return self.family == "SHAKE"

    @property
    def ph_name(self) -> str:
        """The one pre-hash function FIPS 205 Table 2 assigns to this set."""
        if self.family == "SHA2":
            return "SHA2-256" if self.n == 16 else "SHA2-512"
        return "SHAKE128" if self.n == 16 else "SHAKE256"


_SETS = {
    # name: (n, h, d, a, k)
    "128s": (16, 63, 7, 12, 14),
    "128f": (16, 66, 22, 6, 33),
    "192s": (24, 63, 7, 14, 17),
    "192f": (24, 66, 22, 8, 33),
    "256s": (32, 64, 8, 14, 22),
    "256f": (32, 68, 17, 9, 35),
}

PARAMS: dict[str, SLHParams] = {}
for _family in ("SHAKE", "SHA2"):
    for _size, (_n, _h, _d, _a, _k) in _SETS.items():
        _p = SLHParams(
            name=f"SLH-DSA-{_family}-{_size}",
            family=_family,
            n=_n,
            h=_h,
            d=_d,
            a=_a,
            k=_k,
        )
        PARAMS[_p.name] = _p

HASHES = tuple(PARAMS)


def get(name: str) -> SLHParams:
    """Look up a parameter set, accepting the FIPS 205 name or the bare size."""
    if name in PARAMS:
        return PARAMS[name]
    for p in PARAMS.values():
        if p.name.endswith("-" + name):
            return p
    raise KeyError(f"unknown SLH-DSA parameter set: {name!r}; known: {', '.join(HASHES)}")
