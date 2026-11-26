"""LMS / HSS parameters (RFC 8554 §4.1, §5.1; SP 800-208 for the approved sets).

Every set is SHA-256 truncated to `n` bytes; what varies is the Winternitz factor `w` (which fixes
the number of chains `p` and the checksum shift `ls`) and the tree height `h`. Two families are
approved by SP 800-208 — SHA-256/256 (`n = m = 32`) and SHA-256/192 (`n = m = 24`) — and the
numeric type codes continue across them (1–4 and 5–9 for the original sets, 5–8 and 10–14 for the
192-bit ones).
"""

from __future__ import annotations

from dataclasses import dataclass

#: LM-OTS type code -> (n, w, p, ls).  RFC 8554 Table 1 and SP 800-208 Table 3.
OTS_TYPES: dict[int, tuple[int, int, int, int]] = {
    1: (32, 1, 265, 7),  # LMOTS_SHA256_N32_W1
    2: (32, 2, 133, 6),
    3: (32, 4, 67, 4),
    4: (32, 8, 34, 0),
    5: (24, 1, 200, 8),  # LMOTS_SHA256_N24_W1
    6: (24, 2, 101, 6),
    7: (24, 4, 51, 4),
    8: (24, 8, 26, 0),
}

#: LMS type code -> (m, h).  RFC 8554 Table 2 and SP 800-208 Table 4.
LMS_TYPES: dict[int, tuple[int, int]] = {
    5: (32, 5),  # LMS_SHA256_M32_H5
    6: (32, 10),
    7: (32, 15),
    8: (32, 20),
    9: (32, 25),
    10: (24, 5),  # LMS_SHA256_M24_H5
    11: (24, 10),
    12: (24, 15),
    13: (24, 20),
    14: (24, 25),
}

OTS_NAMES = {
    1: "LMOTS_SHA256_N32_W1",
    2: "LMOTS_SHA256_N32_W2",
    3: "LMOTS_SHA256_N32_W4",
    4: "LMOTS_SHA256_N32_W8",
    5: "LMOTS_SHA256_N24_W1",
    6: "LMOTS_SHA256_N24_W2",
    7: "LMOTS_SHA256_N24_W4",
    8: "LMOTS_SHA256_N24_W8",
}
LMS_NAMES = {
    5: "LMS_SHA256_M32_H5",
    6: "LMS_SHA256_M32_H10",
    7: "LMS_SHA256_M32_H15",
    8: "LMS_SHA256_M32_H20",
    9: "LMS_SHA256_M32_H25",
    10: "LMS_SHA256_M24_H5",
    11: "LMS_SHA256_M24_H10",
    12: "LMS_SHA256_M24_H15",
    13: "LMS_SHA256_M24_H20",
    14: "LMS_SHA256_M24_H25",
}


@dataclass(frozen=True)
class LMSSet:
    """One (tree height, Winternitz factor) combination."""

    lms_type: int
    ots_type: int

    @property
    def n(self) -> int:
        """Hash output length, in bytes."""
        return OTS_TYPES[self.ots_type][0]

    @property
    def m(self) -> int:
        """Tree node length, in bytes (equal to `n` for every approved set)."""
        return LMS_TYPES[self.lms_type][0]

    @property
    def h(self) -> int:
        return LMS_TYPES[self.lms_type][1]

    @property
    def w(self) -> int:
        return OTS_TYPES[self.ots_type][1]

    @property
    def p(self) -> int:
        return OTS_TYPES[self.ots_type][2]

    @property
    def ls(self) -> int:
        return OTS_TYPES[self.ots_type][3]

    @property
    def name(self) -> str:
        return f"{LMS_NAMES[self.lms_type]}/{OTS_NAMES[self.ots_type]}"

    @property
    def pk_size(self) -> int:
        """LMS public key: type, OTS type, I, T[1]."""
        return 24 + self.m

    @property
    def sig_size(self) -> int:
        """LMS signature: q, OTS type, C, y[p], LMS type, path[h]."""
        return 12 + self.n * (self.p + 1) + self.m * self.h

    @property
    def ots_sig_size(self) -> int:
        """LM-OTS signature as it appears inside an LMS signature (type, C, y[p])."""
        return 4 + self.n * (self.p + 1)


HSS_MAX_LEVELS = 8


def get(lms_type: int, ots_type: int) -> LMSSet:
    if lms_type not in LMS_TYPES:
        raise ValueError(f"unsupported LMS type code {lms_type}")
    if ots_type not in OTS_TYPES:
        raise ValueError(f"unsupported LM-OTS type code {ots_type}")
    if LMS_TYPES[lms_type][0] != OTS_TYPES[ots_type][0]:
        raise ValueError(
            f"{LMS_NAMES[lms_type]} needs a {LMS_TYPES[lms_type][0]}-byte LM-OTS set, got "
            f"{OTS_NAMES[ots_type]}"
        )
    return LMSSet(lms_type, ots_type)


def from_name(name: str) -> LMSSet:
    """`LMS_SHA256_M32_H10:LMOTS_SHA256_N32_W4`, or `H10W4`, or `M24H5W1`."""
    if ":" in name:
        lms, ots = name.split(":")
    else:
        head = name.upper()
        bits = "M24" if head.startswith("M24") else "M32"
        head = head.removeprefix("M24").removeprefix("M32")
        hpart, wpart = head.split("W")
        lms, ots = f"LMS_SHA256_{bits}_{hpart}", f"LMOTS_SHA256_N{bits[1:]}_W{wpart}"
    lms_type = next(k for k, v in LMS_NAMES.items() if v == lms)
    ots_type = next(k for k, v in OTS_NAMES.items() if v == ots)
    return get(lms_type, ots_type)


ALL_SETS = [
    LMSSet(lt, ot) for lt in LMS_TYPES for ot in OTS_TYPES if LMS_TYPES[lt][0] == OTS_TYPES[ot][0]
]
