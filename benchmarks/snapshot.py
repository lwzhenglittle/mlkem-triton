#!/usr/bin/env python
"""One window, one file: every measurement this repository has, in one snapshot.

The measurement discipline in `docs/benchmarks.md` §1 is "same process, same second, quote
ratios not absolutes" -- and the report that goes with it was assembled by hand, section by
section, from separate runs. This script makes the discipline the default:

    python benchmarks/snapshot.py                  # everything (~10 minutes)
    python benchmarks/snapshot.py --skip-cpu       # reuse benchmarks/cpu_baseline_barrier.json
    python benchmarks/snapshot.py --only primitive,falcon

It shells out to the existing benchmark scripts rather than importing them, because each one
sets up its own corpus (liboqs signatures, ACVP vectors) and prints its own copy-bandwidth
header; re-implementing that here would be a second set of conventions to keep in sync. What it
adds is the envelope: one timestamp, one git revision, one markdown file, and a JSON header
that says which window each section came from.

Output: `benchmarks/snapshot-<UTC stamp>.md` next to a `.json` with the metadata, unless
`--out` says otherwise.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "benchmarks"

# name -> (script, argv, what it answers)
SECTIONS: dict[str, tuple[str, list[str], str]] = {
    "cpu": (
        # The barrier harness, not `bench_cpu.py --merge`: the latter sums per-child minima, which
        # over-counts the CPU by 1.25x..4.25x (docs/benchmarks.md §15). A snapshot regenerated
        # with it would quietly restore the old column.
        "bench_16p_barrier.py",
        ["--all", "--reps", "2", "--json", "cpu_baseline_barrier.json"],
        "the rival: every primitive's CPU opponent, 1T and 16 processes, barrier-measured",
    ),
    "primitive": (
        "bench_primitive.py",
        ["--batch", "65536", "--md"],
        "the primitive layer against copy bandwidth and against the rival",
    ),
    "falcon": (
        "bench_falcon_verify.py",
        ["--ps", "falcon-512", "--batch", "512,4096"],
        "Falcon-512 verification: device core and end-to-end",
    ),
    "mldsa": (
        "bench_mldsa_verify.py",
        ["--ps", "ML-DSA-44", "--batch", "4096"],
        "ML-DSA-44 verification, with its own liboqs baseline",
    ),
    "mlkem": (
        "bench_encaps.py",
        ["--ps", "ML-KEM-768", "--batch", "4096"],
        "ML-KEM-768 encapsulation",
    ),
    "slhdsa": (
        "bench_slhdsa_verify.py",
        ["--ps", "SLH-DSA-SHAKE-128f", "--batch", "4096"],
        "SLH-DSA-SHAKE-128f verification",
    ),
}
DEFAULT_ORDER = ["cpu", "primitive", "falcon", "mldsa", "mlkem", "slhdsa"]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=20
        ).stdout.strip()
    except Exception:
        return "unknown"


def _run(name: str) -> dict:
    script, argv, _ = SECTIONS[name]
    cmd = [sys.executable, str(BENCH / script), *argv]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(BENCH), capture_output=True, text=True)
    dt = time.time() - t0
    out = proc.stdout
    # keep the measurement lines, drop the interpreter noise the XPU wheels emit
    lines = [
        ln
        for ln in out.splitlines()
        if "RuntimeWarning" not in ln
        and "frozen site" not in ln
        and "UserWarning" not in ln
        and not ln.startswith("  warn(")
    ]
    return {
        "section": name,
        "command": " ".join([script, *argv]),
        "rc": proc.returncode,
        "seconds": round(dt, 1),
        "text": "\n".join(lines).strip(),
        "stderr": proc.stderr.strip()[-2000:] if proc.returncode else "",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None, help="comma-separated section names")
    ap.add_argument(
        "--skip-cpu",
        action="store_true",
        help="reuse benchmarks/cpu_baseline_barrier.json (or cpu_baseline.json) instead of re-measuring rivals",
    )
    ap.add_argument(
        "--out", default=None, help="output path (default: benchmarks/snapshot-<stamp>.md)"
    )
    args = ap.parse_args()

    names = DEFAULT_ORDER if args.only is None else [n.strip() for n in args.only.split(",")]
    unknown = [n for n in names if n not in SECTIONS]
    if unknown:
        print(f"error: unknown section(s) {unknown}; known: {sorted(SECTIONS)}", file=sys.stderr)
        return 2
    if args.skip_cpu and "cpu" in names:
        names = [n for n in names if n != "cpu"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.out) if args.out else BENCH / f"snapshot-{stamp}.md"
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": platform.python_version(),
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cpu_baseline": "reused" if args.skip_cpu else "measured",
        "sections": [],
    }

    parts = []
    for name in names:
        _, _, what = SECTIONS[name]
        print(f"# {name}: {what} ...", flush=True)
        result = _run(name)
        meta["sections"].append({k: v for k, v in result.items() if k != "text"})
        status = "ok" if result["rc"] == 0 else f"FAILED rc={result['rc']}"
        print(f"#   {status} in {result['seconds']}s", flush=True)
        body = result["text"] or f"(no output)\n{result['stderr']}"
        parts.append(f"## {name} — {what}\n\n```\n{body}\n```\n")

    cpu_path = (
        BENCH / "cpu_baseline_barrier.json"
        if (not args.skip_cpu or (BENCH / "cpu_baseline_barrier.json").exists())
        else BENCH / "cpu_baseline.json"
    )
    cpu_method = "unknown"
    if cpu_path.exists():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            cpu_method = (
                json.loads(cpu_path.read_text()).get("meta", {}).get("method", "sum-of-minima")
            )
    meta["cpu_baseline_file"] = f"benchmarks/{cpu_path.name}"
    meta["cpu_baseline_method"] = cpu_method

    header = (
        f"# Measurement snapshot — {meta['timestamp_utc']}\n\n"
        f"Generated by `benchmarks/snapshot.py` from `{ROOT.name}` at `{meta['git_head']}`"
        f"{' (dirty tree)' if meta['git_dirty'] else ''}.\n\n"
        "Every section below was produced in this one window. Absolute numbers move 2-4x with\n"
        "the iGPU's clock state, so quote the ratios (`%copy`, `vs 1T`, `vs 16P`) and the\n"
        "header of each section rather than the raw rates.\n\n"
        f"- host: `{meta['host']}`, python {meta['python']}\n"
        f"- CPU rival: {meta['cpu_baseline']} (`{meta['cpu_baseline_file']}`, {cpu_method})\n"
        f"- sections: {', '.join(s['section'] for s in meta['sections'])}\n\n"
    )
    out.write_text(header + "\n".join(parts))
    json_out = out.with_suffix(".json")
    json_out.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nwrote {out}\nwrote {json_out}")
    failures = [s for s in meta["sections"] if s["rc"] != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
