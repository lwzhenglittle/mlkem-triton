"""Join a device run with the barrier CPU run and show what the old `16P` column cost.

Reads three files and prints one table, because the correction is easier to audit than to trust:

    benchmarks/bench_primitive.py --batch 65536 --json /tmp/dev.json --md
    benchmarks/bench_16p_barrier.py --all --reps 2 --json /tmp/cpu16.json
    benchmarks/join_16p_table.py --device /tmp/dev.json --cpu /tmp/cpu16.json

Every ratio is recomputed here from the raw rates: the device row's `bytes` or `ops` per second
against the rival's `one_thread` and `procs16`. Nothing is copied from a document, which is the
point -- the previous table's ratios were correct arithmetic over a denominator that was measured
two different ways in the same column.

Usage:
    python benchmarks/join_16p_table.py --device dev.json --cpu cpu16.json [--baseline PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench_primitive import CPU_RIVAL


def _fmt(rate: float, unit: str) -> str:
    if unit == "B/s":
        return f"{rate / 1e9:6.2f} GB/s"
    if rate >= 1e4:
        return f"{rate / 1e3:6.1f} k/s"
    return f"{rate:6.1f}/s"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=Path, required=True)
    ap.add_argument(
        "--cpu",
        type=Path,
        action="append",
        required=True,
        help="one or more bench_16p_barrier JSON files (repeatable, later wins)",
    )
    ap.add_argument("--baseline", type=Path, default=ROOT / "cpu_baseline.json")
    args = ap.parse_args()

    dev = json.loads(args.device.read_text())
    new = {}
    for path in args.cpu:
        new.update(json.loads(path.read_text())["rivals"])
    old = json.loads(args.baseline.read_text())["rivals"]

    print(
        f"device batch {dev['meta']['batch']}, size {dev['meta']['size']}, reps {dev['meta']['reps']}"
    )
    print()
    print(
        "| primitive | device | rival | 16P old (sum-of-minima) | 16P new (barrier) "
        "| ratio old | ratio new | change |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|")

    flips = []
    for row in dev["rows"]:
        name = row["name"]
        rival_name = CPU_RIVAL.get(name)
        if not rival_name:
            continue
        if rival_name not in new:
            print(
                f"| `{name}` | — | {rival_name} | not re-measured | not re-measured | — | — | — |"
            )
            continue
        unit = new[rival_name]["unit"]
        device_rate = row["bytes"] if unit == "B/s" else row["ops"]
        old16 = old[rival_name]["procs16"]
        new16 = new[rival_name]["procs16"]
        r_old = device_rate / old16
        r_new = device_rate / new16
        change = r_new / r_old
        mark = ""
        if (r_old < 1.0) != (r_new < 1.0):
            mark = " **FLIP**"
            flips.append(name)
        print(
            f"| `{name}` | {_fmt(device_rate, unit)} | {rival_name} | {_fmt(old16, unit)} "
            f"| {_fmt(new16, unit)} | {r_old:.2f}x | {r_new:.2f}x | {change:.2f}x{mark} |"
        )

    print()
    if flips:
        print(f"conclusion flips: {', '.join(flips)}")
    else:
        print("no row changed side of 1.0x")

    # The interval the correction is worth, so a later reader can bound it without re-running.
    changes = []
    for row in dev["rows"]:
        rival_name = CPU_RIVAL.get(row["name"])
        if not rival_name or rival_name not in new:
            continue
        unit = new[rival_name]["unit"]
        device_rate = row["bytes"] if unit == "B/s" else row["ops"]
        changes.append(
            (device_rate / new[rival_name]["procs16"]) / (device_rate / old[rival_name]["procs16"])
        )
    if changes:
        print(
            f"\nratio correction spans {min(changes):.2f}x .. {max(changes):.2f}x "
            f"(median {sorted(changes)[len(changes) // 2]:.2f}x): the old column made the device "
            f"look between {1 / max(changes):.1f}x and {1 / min(changes):.1f}x worse than it is"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
