"""Parse the NIST ACVP-Server ML-KEM encapDecap test vectors.

Data source: https://github.com/usnistgov/ACVP-Server
Path: gen-val/json-files/ML-KEM-encapDecap-FIPS203/{prompt,expectedResults}.json
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_DIR = Path(
    os.environ.get("ACVP_DIR", Path(__file__).resolve().parent.parent / "data" / "acvp")
)


def load(data_dir: Path | str | None = None):
    """Return the flattened list of test cases.

    Each record: {tgId, parameterSet, function, testType, tcId, inputs, expected}
    """
    d = Path(data_dir) if data_dir else DEFAULT_DIR
    prompt = json.loads((d / "prompt.json").read_text())
    exp = json.loads((d / "expectedResults.json").read_text())

    by_tg = {g["tgId"]: {t["tcId"]: t for t in g["tests"]} for g in exp["testGroups"]}

    cases = []
    for g in prompt["testGroups"]:
        exp_tests = by_tg[g["tgId"]]
        for t in g["tests"]:
            e = exp_tests[t["tcId"]]
            cases.append(
                {
                    "tgId": g["tgId"],
                    "parameterSet": g["parameterSet"],
                    "function": g["function"],
                    "testType": g["testType"],
                    "tcId": t["tcId"],
                    "inputs": {k: v for k, v in t.items() if k != "tcId"},
                    "expected": {k: v for k, v in e.items() if k != "tcId"},
                }
            )
    return cases


def group_summary(cases=None):
    cases = cases or load()
    out = {}
    for c in cases:
        key = (c["parameterSet"], c["function"], c["testType"])
        out[key] = out.get(key, 0) + 1
    return out


if __name__ == "__main__":
    cs = load()
    print(f"total {len(cs)} test cases")
    for k, v in sorted(group_summary(cs).items()):
        print(f"  {k[0]:<12} {k[1]:<22} {k[2]:<5} {v:>3}")
