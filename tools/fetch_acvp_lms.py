"""Regenerate `data/acvp-lms/sigVer.json` from a sparse clone of the NIST ACVP server.

The vectors are stateful-hash-signature test cases from SP 800-208 / RFC 8554; only the SHA-256
families are shipped, because the SP 800-208 SHAKE-256 sets still have TBD type codes.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

SOURCES = ("LMS-sigVer-1.0", "LMS-sigVer-SP800-208")
ROOT = Path(__file__).parent.parent / "data/acvp-lms/sigVer.json"


def main() -> int:
    with tempfile.TemporaryDirectory() as td:
        clone = Path(td) / "acvp"
        subprocess.run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--sparse",
                "--depth",
                "1",
                "https://github.com/usnistgov/ACVP-Server.git",
                str(clone),
            ],
            check=True,
            capture_output=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "gen-val/json-files"],
            cwd=clone,
            check=True,
            capture_output=True,
        )
        out = []
        for name in SOURCES:
            data = json.loads(
                (clone / "gen-val/json-files" / name / "internalProjection.json").read_text()
            )
            for g in data["testGroups"]:
                for t in g["tests"]:
                    out.append(
                        {
                            "source": name,
                            "tg": g["tgId"],
                            "tc": t["tcId"],
                            "lms_mode": g["lmsMode"],
                            "ots_mode": g["lmOtsMode"],
                            "public_key": g["publicKey"],
                            "message": t.get("message", ""),
                            "signature": t["signature"],
                            "expected": bool(t["testPassed"]),
                            "reason": t.get("reason", ""),
                        }
                    )
    ROOT.write_text(json.dumps(out, indent=1))
    print(f"wrote {len(out)} cases to {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
