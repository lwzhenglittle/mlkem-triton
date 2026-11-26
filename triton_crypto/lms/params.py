"""LMS / HSS parameters (RFC 8554 §4.1, §5.1; SP 800-208 for the approved subset).

Every parameter set is SHA-256 with a 32-byte digest; what varies is the Winternitz factor `w`
(which fixes the number of chains `p` and the checksum shift `ls`) and the tree height `h`.
The numeric type codes are the wire identifiers: an LMS public key carries both.
"""

from __future__ import annotations

from dataclasses import dataclass

N = 32  # bytes per hash output, and per chain element
M = 32  # bytes per tree node

#: LM-OTS type code -> (w, p, ls).  RFC 8554 Table 1.
OTS_TYPES: dict[int, tuple[int, int, int]] = {
    1: (1, 265, 7),  # LMOTS_SHA256_N32_W1
    2: (2, 133, 6),  # LMOTS_SHA256_N32_W2
    3: (4, 67, 4),  # LMOTS_SHA256_N32_W4
    4: (8, 34, 0),  # LMOTS_SHA256_N32_W8
}

#: LMS type code -> tree height.  RFC 8554 Table 2.
LMS_TYPES: dict[int, int] = {
    5: 5,  # LMS_SHA256_M32_H5
    6: 10,  # LMS_SHA256_M32_H10
    7: 15,  # LMS_SHA256_M32_H15
    8: 20,  # LMS_SHA256_M32_H20
    9: 25,  # LMS_SHA256_M32_H25
}

OTS_NAMES = {
    1: "LMOTS_SHA256_N32_W1",
    2: "LMOTS_SHA256_N32_W2",
    3: "LMOTS_SHA256_N32_W4",
    4: "LMOTS_SHA256_N32_W8",
}
LMS_NAMES = {
    5: "LMS_SHA256_M32_H5",
    6: "LMS_SHA256_M32_H10",
    7: "LMS_SHA256_M32_H15",
    8: "LMS_SHA256_M32_H20",
    9: "LMS_SHA256_M32_H25",
}


@dataclass(frozen=True)
class LMSSet:
    """One (tree height, Winternitz factor) combination."""

    lms_type: int
    ots_type: int

    @property
    def h(self) -> int:
        return LMS_TYPES[self.lms_type]

    @property
    def w(self) -> int:
        return OTS_TYPES[self.ots_type][0]

    @property
    def p(self) -> int:
        return OTS_TYPES[self.ots_type][1]

    @property
    def ls(self) -> int:
        return OTS_TYPES[self.ots_type][2]

    @property
    def name(self) -> str:
        return f"{LMS_NAMES[self.lms_type]}/{OTS_NAMES[self.ots_type]}"

    @property
    def pk_size(self) -> int:
        """LMS public key: type, OTS type, I, T[1]."""
        return 24 + M

    @property
    def sig_size(self) -> int:
        """LMS signature: q, OTS type, C, y[p], LMS type, path[h]."""
        return 12 + N * (self.p + 1) + M * self.h

    @property
    def ots_sig_size(self) -> int:
        """LM-OTS signature as it appears inside an LMS signature (type, C, y[p])."""
        return 4 + N * (self.p + 1)


HSS_MAX_LEVELS = 8


def get(lms_type: int, ots_type: int) -> LMSSet:
    if lms_type not in LMS_TYPES:
        raise ValueError(
            f"unsupported LMS type code {lms_type} (only the SHA-256 family is defined)"
        )
    if ots_type not in OTS_TYPES:
        raise ValueError(
            f"unsupported LM-OTS type code {ots_type} (only the SHA-256 family is defined)"
        )
    return LMSSet(lms_type, ots_type)


def from_name(name: str) -> LMSSet:
    """`LMS_SHA256_M32_H10:LMOTS_SHA256_N32_W4` or `H10W4`."""
    if ":" in name:
        lms, ots = name.split(":")
    else:
        hpart, wpart = name.upper().split("W")
        lms, ots = f"LMS_SHA256_M32_{hpart}", f"LMOTS_SHA256_N32_W{wpart}"
    lms_type = next(k for k, v in LMS_NAMES.items() if v == lms)
    ots_type = next(k for k, v in OTS_NAMES.items() if v == ots)
    return get(lms_type, ots_type)


ALL_SETS = [LMSSet(lt, ot) for lt in LMS_TYPES for ot in OTS_TYPES]
