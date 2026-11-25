"""Loader for the NIST ACVP SLH-DSA-sigVer vectors in `data/acvp-slhdsa/`.

The committed file is a subset of `gen-val/json-files/SLH-DSA-sigVer-FIPS205/prompt.json`
(4 of the 14 tests per group, 2 valid + 2 invalid where the group has both), because the full
file is 30 MB. Regenerate it with `tools/fetch_acvp_slhdsa.py`; set `ACVP_SLH_DSA_DIR` to point
at a different directory.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Case", "case_count", "cases", "group_summary", "load"]

DEFAULT_DIR = Path(
    os.environ.get(
        "ACVP_SLH_DSA_DIR",
        Path(__file__).resolve().parent.parent.parent / "data" / "acvp-slhdsa",
    )
)


@dataclass(frozen=True)
class Case:
    tg_id: int
    tc_id: int
    ps: str
    interface: str  # "external" or "internal"
    prehash: str | None  # None for pure / internal, else the FIPS 205 PH name
    ph: str | None  # pre-hash function for preHash tests ("SHA3-384", ...), as ACVP names it
    pk: bytes
    message: bytes
    context: bytes | None
    signature: bytes
    expected: bool

    @property
    def label(self) -> str:
        mode = self.interface if self.interface == "internal" else (self.prehash or "pure")
        return f"tg{self.tg_id}/{self.ps}/{mode}/tc{self.tc_id}"


def _norm_ph(name: str | None) -> str | None:
    """ACVP writes "SHAKE-128"; FIPS 205 Table 3 writes "SHAKE128"."""
    if name is None:
        return None
    return name.replace("SHAKE-", "SHAKE")


def load(path: str | os.PathLike | None = None) -> dict:
    p = Path(path) if path else DEFAULT_DIR / "sigVer.json"
    if not p.exists():
        raise FileNotFoundError(
            f"ACVP SLH-DSA vectors not found at {p}. Run tools/fetch_acvp_slhdsa.py, or set "
            "ACVP_SLH_DSA_DIR."
        )
    return json.loads(p.read_text())


def cases(*, ps: str | None = None, interface: str | None = None) -> list[Case]:
    out = []
    for g in load()["groups"]:
        if ps is not None and g["parameterSet"] != ps:
            continue
        if interface is not None and g["signatureInterface"] != interface:
            continue
        prehash = None if g["preHash"] in (None, "pure") else "PH"
        for t in g["tests"]:
            out.append(
                Case(
                    tg_id=g["tgId"],
                    tc_id=t["tcId"],
                    ps=g["parameterSet"],
                    interface=g["signatureInterface"],
                    prehash=prehash,
                    ph=_norm_ph(t.get("hashAlg")),
                    pk=base64.b64decode(t["pk"]),
                    message=base64.b64decode(t["message"]),
                    context=None if t["context"] is None else base64.b64decode(t["context"]),
                    signature=base64.b64decode(t["signature"]),
                    expected=t["expected"],
                )
            )
    return out


def case_count() -> int:
    return sum(len(g["tests"]) for g in load()["groups"])


def group_summary() -> list[str]:
    return [
        f"tg{g['tgId']:3d}  {g['parameterSet']:22s} {g['signatureInterface']:8s} "
        f"{g['preHash'] or '-'!s:8s} {len(g['tests'])} cases"
        for g in load()["groups"]
    ]
