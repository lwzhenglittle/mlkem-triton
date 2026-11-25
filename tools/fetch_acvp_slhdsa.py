#!/usr/bin/env python3
"""Regenerate `data/acvp-slhdsa/` from a checkout of the NIST ACVP-Server repository.

The upstream `gen-val/json-files/SLH-DSA-sigVer-FIPS205/prompt.json` is 30 MB (504 tests), so the
committed subset keeps 4 tests per group — 2 valid, 2 invalid where the group has them — which is
enough to catch a broken verifier while staying small. `keyGen` is kept in full (120 tests, 44 KB).

    git clone --filter=blob:none --sparse --depth 1 \
        git@github.com:usnistgov/ACVP-Server.git
    cd ACVP-Server && git sparse-checkout set \
        gen-val/json-files/SLH-DSA-keyGen-FIPS205 \
        gen-val/json-files/SLH-DSA-sigVer-FIPS205
    python tools/fetch_acvp_slhdsa.py /path/to/ACVP-Server
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "data" / "acvp-slhdsa"
PER_GROUP = 4
b64 = lambda x: base64.b64encode(x).decode()  # noqa: E731


def _sigver(src: Path) -> dict:
    prompt = json.loads((src / "prompt.json").read_text())
    expected = json.loads((src / "expectedResults.json").read_text())
    exp = {g["tgId"]: {t["tcId"]: t for t in g["tests"]} for g in expected["testGroups"]}
    out = {
        "note": f"ACVP SLH-DSA-sigVer-FIPS205, {PER_GROUP} of 14 tests per group "
        "(2 valid + 2 invalid where present); byte fields are base64; 'hashAlg' is the "
        "pre-hash function for preHash tests",
        "vsId": prompt["vsId"],
        "revision": prompt["revision"],
        "groups": [],
    }
    for g in prompt["testGroups"]:
        valid = [t for t in g["tests"] if exp[g["tgId"]][t["tcId"]]["testPassed"]]
        invalid = [t for t in g["tests"] if not exp[g["tgId"]][t["tcId"]]["testPassed"]]
        tests = []
        for t in (valid[:2] + invalid[:2])[:PER_GROUP]:
            e = exp[g["tgId"]][t["tcId"]]
            tests.append(
                {
                    "tcId": t["tcId"],
                    "pk": b64(bytes.fromhex(t["pk"])),
                    "message": b64(bytes.fromhex(t["message"])),
                    "context": (
                        b64(bytes.fromhex(t["context"])) if t.get("context") is not None else None
                    ),
                    "hashAlg": t.get("hashAlg"),
                    "signature": b64(bytes.fromhex(t["signature"])),
                    "expected": bool(e["testPassed"]),
                }
            )
        out["groups"].append(
            {
                "tgId": g["tgId"],
                "parameterSet": g["parameterSet"],
                "signatureInterface": g["signatureInterface"],
                "preHash": g.get("preHash"),
                "tests": tests,
            }
        )
    return out


def _keygen(src: Path) -> dict:
    prompt = json.loads((src / "prompt.json").read_text())
    expected = json.loads((src / "expectedResults.json").read_text())
    exp = {g["tgId"]: {t["tcId"]: t for t in g["tests"]} for g in expected["testGroups"]}
    out = {"note": "ACVP SLH-DSA-keyGen-FIPS205, all tests; byte fields are base64", "groups": []}
    for g in prompt["testGroups"]:
        tests = []
        for t in g["tests"]:
            r = exp[g["tgId"]][t["tcId"]]
            seed = bytes.fromhex(t["skSeed"] + t["skPrf"] + t["pkSeed"])
            tests.append(
                {
                    "tcId": t["tcId"],
                    "seed": b64(seed),
                    "pk": b64(bytes.fromhex(r["pk"])),
                    "sk": b64(bytes.fromhex(r["sk"])),
                }
            )
        out["groups"].append({"tgId": g["tgId"], "parameterSet": g["parameterSet"], "tests": tests})
    return out


def main(argv: list[str]) -> int:
    root = Path(argv[1]) / "gen-val" / "json-files"
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in (("sigVer", _sigver), ("keyGen", _keygen)):
        data = fn(root / f"SLH-DSA-{name}-FIPS205")
        path = OUT / f"{name}.json"
        path.write_text(json.dumps(data, indent=1))
        n = sum(len(g["tests"]) for g in data["groups"])
        print(f"{path.name}: {n} tests, {path.stat().st_size // 1024} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
