"""Falcon-512 parameters (the reference implementation's numbers, not the spec's prose)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FalconParams:
    """One Falcon degree. Only Falcon-512 ships; 1024 differs by `logn` and the bound."""

    name: str
    logn: int
    q: int = 12289
    nonce_len: int = 40
    pk_len: int = 897  # includes the 0x09 degree byte
    sk_len: int = 1281
    sig_max: int = 752  # an unpadded signature never exceeds this
    sig_padded: int = 666  # Falcon-padded-512: fixed length
    l2bound: int = 34034726  # floor(beta^2) for logn = 9, inclusive
    oversample: int = 205  # hash-to-point draws n + 205 samples

    @property
    def n(self) -> int:
        return 1 << self.logn

    @property
    def pk_header(self) -> int:
        return 0x00 + self.logn  # 0x09

    @property
    def sig_header(self) -> int:
        return 0x30 + self.logn  # 0x39

    @property
    def max_abs_coefficient(self) -> int:
        return 2047  # comp_decode rejects |s2| > 2047


FALCON_512 = FalconParams(name="Falcon-512", logn=9)
FALCON_PADDED_512 = FalconParams(name="Falcon-padded-512", logn=9)
