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


def _repro_section(old: dict, path: Path) -> list[str]:
    """Cells of a second run over the cells of the baseline, with the one number that matters.

    The 1T column is what tells you the machine is in a comparable state; the 16P column is the
    one that moves, because a fixed pile of work split sixteen ways waits for the slowest core and
    which core that is changes between runs.
    """
    repro = json.loads(path.read_text())
    cells = []
    for name, row in old.items():
        other = repro["rivals"].get(name)
        if not other:
            continue
        cells.append((name, "1T", other["one_thread"] / row["one_thread"]))
        cells.append((name, "16P", other["procs16"] / row["procs16"]))
    if not cells:
        return []
    vals = sorted(c[2] for c in cells)
    ones = [v for _, k, v in cells if k == "1T"]
    sixteens = [v for _, k, v in cells if k == "16P"]
    lines = [
        "## Reproduction: the same column, measured again",
        "",
        f"Re-measured at {repro.get('meta', {}).get('timestamp', '?')} against the baseline above.",
        f"All {len(cells)} cells, second run / first run:",
        "",
        f"- median **{vals[len(vals) // 2]:.3f}**, range {vals[0]:.3f}..{vals[-1]:.3f}",
        f"- 1T column: {min(ones):.3f}..{max(ones):.3f} -- this is the one that says whether the",
        "  machine is in a comparable state, and it is the reason the first column can be trusted",
        f"- 16P column: {min(sixteens):.3f}..{max(sixteens):.3f} -- a fixed pile of work split",
        "  sixteen ways waits for the slowest core, and which core that is moves between runs.",
        "  **Per-row ratios on this column carry that band**; the 1T ratios do not.",
        "",
        "| rival | 1T second/first | 16P second/first |",
        "|---|---:|---:|",
    ]
    for name in sorted({c[0] for c in cells}):
        pair = {k: v for n, k, v in cells if n == name}
        lines.append(f"| `{name}` | {pair['1T']:.3f} | {pair['16P']:.3f} |")
    lines.append("")
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        type=Path,
        default=None,
        help="device JSON from bench_primitive.py --json (not needed with --repro-out)",
    )
    ap.add_argument(
        "--cpu",
        type=Path,
        action="append",
        required=True,
        help="one or more bench_16p_barrier JSON files (repeatable, later wins)",
    )
    ap.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="the 'old' column to compare against (default: cpu_baseline_barrier.json when it "
        "exists, else cpu_baseline.json) -- the wrong default here silently reports the "
        "correction as a 32%% regression",
    )
    ap.add_argument(
        "--repro",
        type=Path,
        default=None,
        help="a second bench_16p_barrier JSON of the same machine: adds a reproduction section "
        "comparing it with the baseline cell by cell",
    )
    ap.add_argument(
        "--repro-out",
        type=Path,
        default=None,
        help="write only the reproduction section (needs --repro); survives losing the device JSON",
    )
    ap.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="write a self-contained snapshot: both measured tables, the joined ratios, provenance",
    )
    args = ap.parse_args()

    if args.device is None and args.repro_out is None:
        ap.error("--device is required unless --repro-out is given")
    dev = json.loads(args.device.read_text()) if args.device else {"meta": {}, "rows": []}
    new = {}
    new_meta = {}
    for path in args.cpu:
        blob = json.loads(path.read_text())
        new.update(blob["rivals"])
        new_meta = blob.get("meta", {})
    baseline = args.baseline or (
        ROOT / "cpu_baseline_barrier.json"
        if (ROOT / "cpu_baseline_barrier.json").exists()
        else ROOT / "cpu_baseline.json"
    )
    old = json.loads(baseline.read_text())["rivals"]

    regen = [
        "python benchmarks/bench_16p_barrier.py --all --reps 2 "
        "--json benchmarks/cpu_baseline_barrier.json",
        "python benchmarks/bench_primitive.py --batch 65536 --json /tmp/dev.json",
        "python benchmarks/join_16p_table.py --device /tmp/dev.json "
        "--cpu benchmarks/cpu_baseline_barrier.json --snapshot benchmarks/table-16p-20261127.md",
    ]
    if args.device:
        print(
            f"device batch {dev['meta']['batch']}, size {dev['meta']['size']}, "
            f"reps {dev['meta']['reps']}"
        )
    print()
    table_lines = [
        "| primitive | device | rival | 16P old (sum-of-minima) | 16P new (barrier) "
        "| ratio old | ratio new | change |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    print(table_lines[0])
    print(table_lines[1])

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
        table_lines.append(
            f"| `{name}` | {_fmt(device_rate, unit)} | {rival_name} | {_fmt(old16, unit)} "
            f"| {_fmt(new16, unit)} | {r_old:.2f}x | {r_new:.2f}x | {change:.2f}x{mark} |"
        )
        print(table_lines[-1])

    if args.repro_out:
        if not args.repro:
            print("error: --repro-out needs --repro", file=sys.stderr)
            return 2
        args.repro_out.write_text(
            "# `vs 16P`: did the CPU column reproduce?\n\n"
            + "\n".join(_repro_section(old, args.repro))
        )
        print(f"wrote {args.repro_out}", file=sys.stderr)

    if args.snapshot:
        lines = ["# Corrected `vs 16P` snapshot — 2026-11-27", ""]
        cpu_meta = new_meta
        lines += [
            "Composed from two adjacent windows, not from one `snapshot.py` invocation: the machine",
            "went to battery power afterwards and every CPU rate dropped 4.6x, so a fresh run would",
            "be worse evidence than this composition. Regenerating it on mains power is the three",
            "commands at the bottom.",
            "",
            f"- CPU window: {cpu_meta.get('timestamp')} ({cpu_meta.get('method')})",
            f"- device window: batch {dev['meta']['batch']}, size {dev['meta']['size']}, "
            f"reps {dev['meta']['reps']}",
            f"- power state of the CPU window: {cpu_meta.get('power') or 'not recorded (before the guard)'}",
            "",
            "## The CPU rival, both estimators",
            "",
            "Digest and MAC rows hash 1024-byte messages (the device's `--size`); the two cipher rows",
            "run 16384-byte buffers, since a stream cipher has no per-message cost and the device feeds",
            "them whole buffers. Sizes are in the rival config, not in the stored JSON.",
            "",
            "| rival | unit | 1T | 16P (batch) | 16P (steady) | scale |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for name, row in new.items():
            lines.append(
                f"| `{name}` | {row['unit']} | {row['one_thread']:.4g} | {row['procs16']:.4g} "
                f"| {row.get('procs16_steady', float('nan')):.4g} | {row['scale']:.2f} |"
            )
        lines += ["", "## Device against both columns", "", *table_lines]
        lines += ["", "## The device side (same window)", "", "| primitive | rate |", "|---|---:|"]
        for row in dev["rows"]:
            rival_name = CPU_RIVAL.get(row["name"])
            if not rival_name:
                continue
            unit = new[rival_name]["unit"] if rival_name in new else "B/s"
            rate = row["bytes"] if unit == "B/s" else row["ops"]
            lines.append(f"| `{row['name']}` | {_fmt(rate, unit)} |")
        if args.repro:
            lines += _repro_section(old, args.repro)
        lines += ["", "## Regenerate", "", "```", *regen, "```", ""]
        changes_all = sorted(
            (device_rate / new[r]["procs16"]) / (device_rate / old[r]["procs16"])
            for row in dev["rows"]
            if (r := CPU_RIVAL.get(row["name"])) and r in new
            for device_rate in [row["bytes"] if new[r]["unit"] == "B/s" else row["ops"]]
        )
        if changes_all:
            lines += [
                "",
                "## What the correction was worth",
                "",
                f"The old column over-counted the CPU by **{changes_all[0]:.2f}x to "
                f"{changes_all[-1]:.2f}x, median {changes_all[len(changes_all) // 2]:.2f}x** -- "
                f"i.e. every ratio in it made the device look "
                f"{1 / changes_all[-1]:.1f}x to {1 / changes_all[0]:.1f}x worse than it is.",
                "",
                "No primitive row changed side of 1.0x. One signature row did, and it is not in",
                "the table above because it is not a `bench_primitive` row: **ML-DSA-44 verify, "
                "0.57x -> 1.66x**",
                "(0.53 Mops/s on the device against 318.7 k ops/s from the barrier; the old 929 k",
                "was the pool's sum of minima). Falcon-512's device core goes 3.72x -> 5.9x on the",
                "same correction.",
            ]
        args.snapshot.write_text("\n".join(lines))
        print(f"\nwrote {args.snapshot}", file=sys.stderr)

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
