"""The NIST ACVP LMS sigVer vectors (bundled under `data/acvp-lms/`).

`tools/fetch_acvp_lms.py` regenerates them from a sparse clone of `usnistgov/ACVP-Server`. ACVP
tests LMS at one level: the public key is a bare LMS public key and the signature a bare LMS
signature, so both get the HSS framing (a level count of 1 for the key, 0 for the signature) before
they reach `Verifier`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .params import LMS_NAMES, OTS_NAMES, get

DATA = Path(__file__).parent.parent.parent / "data/acvp-lms/sigVer.json"


@dataclass(frozen=True)
class Case:
    tg: int
    tc: int
    lms_type: int
    ots_type: int
    pk: bytes
    hss_pk: bytes
    message: bytes
    signature: bytes
    hss_sig: bytes
    expected: bool
    reason: str

    @property
    def label(self) -> str:
        return f"tg{self.tg}/tc{self.tc}"


def cases(path: Path | None = None) -> list[Case]:
    raw = json.loads((path or DATA).read_text())
    out = []
    for c in raw:
        lms_type = next(k for k, v in LMS_NAMES.items() if v == c["lms_mode"])
        ots_type = next(k for k, v in OTS_NAMES.items() if v == c["ots_mode"])
        get(lms_type, ots_type)  # validates the pair
        pk = bytes.fromhex(c["public_key"])
        sig = bytes.fromhex(c["signature"])
        out.append(
            Case(
                tg=c["tg"],
                tc=c["tc"],
                lms_type=lms_type,
                ots_type=ots_type,
                pk=pk,
                hss_pk=b"\x00\x00\x00\x01" + pk,
                message=bytes.fromhex(c["message"]) if c["message"] else b"",
                signature=sig,
                hss_sig=b"\x00\x00\x00\x00" + sig,
                expected=bool(c["expected"]),
                reason=c.get("reason", ""),
            )
        )
    return out
