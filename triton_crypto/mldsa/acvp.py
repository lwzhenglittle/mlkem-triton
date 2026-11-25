"""Parse the NIST ACVP-Server ML-DSA test vectors (FIPS 204).

Data source: https://github.com/usnistgov/ACVP-Server
Path: gen-val/json-files/ML-DSA-{keyGen,sigGen,sigVer}-FIPS204/{prompt,expectedResults}.json

Only the three files this package can use are expected in `data/acvp-mldsa/`:

    ML-DSA-keyGen-FIPS204-prompt.json / -expectedResults.json     (seed -> pk, sk)
    ML-DSA-sigGen-FIPS204-prompt.json / -expectedResults.json     (sk, M -> signature)
    ML-DSA-sigVer-FIPS204-prompt.json / -expectedResults.json     (pk, M, σ -> pass/fail)

Unlike ML-KEM's ACVP files, these are *not* flattened into one list by default: the test
group carries the interface (external/internal, pure/pre-hash, external mu), which is
exactly the thing under test, so `load()` keeps it.

Approximate sizes, because they explain the checkout's data footprint: keyGen 0.9 MB,
sigVer 3.0 MB, sigGen 7.3 MB (one secret key and message per case). `sigGen` is the slowest
gate — it re-runs the whole FIPS 204 signing loop, including rejections — which is why the
default test run only samples one case per interface and the full 360 are behind `-m slow`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DIR = Path(
    os.environ.get(
        "ACVP_MLDSA_DIR", Path(__file__).resolve().parent.parent.parent / "data" / "acvp-mldsa"
    )
)

KINDS = ("keyGen", "sigGen", "sigVer")


def _paths(kind: str, data_dir=None) -> tuple[Path, Path]:
    d = Path(data_dir) if data_dir else DEFAULT_DIR
    base = f"ML-DSA-{kind}-FIPS204"
    return d / f"{base}-prompt.json", d / f"{base}-expectedResults.json"


def load(kind: str = "sigVer", data_dir: Path | str | None = None) -> list[dict]:
    """Return one record per test case, with the group's interface fields attached.

    Record keys: ``tgId, parameterSet, tcId, interface, preHash, externalMu,
    deterministic, inputs, expected``.
    """
    if kind not in KINDS:
        raise KeyError(f"unknown vector kind {kind!r}; known: {KINDS}")
    prompt_p, exp_p = _paths(kind, data_dir)
    if not prompt_p.exists():
        raise FileNotFoundError(
            f"{prompt_p} not found. Download the ACVP vectors from\n"
            "  https://github.com/usnistgov/ACVP-Server/tree/master/gen-val/json-files\n"
            f"into {prompt_p.parent}/ , or set ACVP_MLDSA_DIR."
        )
    prompt = json.loads(prompt_p.read_text())
    expected = json.loads(exp_p.read_text())
    by_tg = {g["tgId"]: {t["tcId"]: t for t in g["tests"]} for g in expected["testGroups"]}

    out = []
    for g in prompt["testGroups"]:
        exp_tests = by_tg[g["tgId"]]
        for t in g["tests"]:
            e = exp_tests[t["tcId"]]
            out.append(
                {
                    "tgId": g["tgId"],
                    "parameterSet": g["parameterSet"],
                    "tcId": t["tcId"],
                    "interface": g.get("signatureInterface"),
                    "preHash": g.get("preHash"),
                    "externalMu": bool(g.get("externalMu")),
                    "deterministic": g.get("deterministic", True),
                    "inputs": {k: v for k, v in t.items() if k != "tcId"},
                    "expected": {k: v for k, v in e.items() if k != "tcId"},
                }
            )
    return out


def group_summary(cases=None, kind: str = "sigVer") -> dict:
    """Count cases per (parameter set, interface, pre-hash, external-mu)."""
    cases = cases if cases is not None else load(kind)
    out: dict[tuple, int] = {}
    for c in cases:
        key = (c["parameterSet"], c["interface"], c["preHash"], c["externalMu"])
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    for kind in KINDS:
        try:
            cs = load(kind)
        except FileNotFoundError as e:
            print(e)
            continue
        print(f"{kind}: {len(cs)} cases")
        for k, v in sorted(group_summary(cs, kind).items(), key=str):
            print(f"  {k[0]:<11} {k[1]:<9} preHash={k[2]!s:<9} extMu={k[3]!s:<5} {v:>3}")
