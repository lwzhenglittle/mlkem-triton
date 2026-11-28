"""Falcon-512 and Falcon-1024 parameters (from pqclean / FIPS 206 draft)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FalconParams:
    """One Falcon degree: Falcon-512 (`logn=9`) or Falcon-1024 (`logn=10`)."""

    name: str
    logn: int
    q: int = 12289
    nonce_len: int = 40
    pk_len: int = 897  # includes the degree header byte (0x09 or 0x0a)
    sk_len: int = 1281
    sig_max: int = 752  # an unpadded signature never exceeds this
    sig_padded: int = 666  # Falcon-padded: fixed length
    l2bound: int = 34034726  # floor(beta^2), inclusive
    oversample: int = 205  # hash-to-point draws n + oversample samples

    @property
    def n(self) -> int:
        return 1 << self.logn

    @property
    def pk_header(self) -> int:
        return 0x00 + self.logn  # 0x09 (512) or 0x0a (1024)

    @property
    def sig_header(self) -> int:
        return 0x30 + self.logn  # 0x39 (512) or 0x3a (1024)

    @property
    def max_abs_coefficient(self) -> int:
        return 2047  # comp_decode rejects |s2| > 2047


FALCON_512 = FalconParams(name="Falcon-512", logn=9)
FALCON_PADDED_512 = FalconParams(name="Falcon-padded-512", logn=9)
FALCON_1024 = FalconParams(
    name="Falcon-1024",
    logn=10,
    pk_len=1793,
    sk_len=2305,
    sig_max=1462,
    sig_padded=1280,
    l2bound=70265242,
    oversample=287,
)
FALCON_PADDED_1024 = FalconParams(
    name="Falcon-padded-1024",
    logn=10,
    pk_len=1793,
    sk_len=2305,
    sig_max=1462,
    sig_padded=1280,
    l2bound=70265242,
    oversample=287,
)
