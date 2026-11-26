#!/usr/bin/env python
"""Per-kernel time breakdown (answers "where does the time go").

    python benchmarks/bench_kernels.py --device cuda --batch 4096

Uses `torch.profiler` to get the self time of each device-side kernel. **Note: use
`min × call count` rather than `avg`** — this project's `_pre` kernel has 26x cold-start
spread (min 92 µs / avg 468 / max 2424); using the average would rank its share first,
completely reversing the ranking from the min-based calculation.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from triton_crypto import describe, resolve_device
from triton_crypto.mlkem import acvp
from triton_crypto.mlkem.pipeline import Decapsulator


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ps", default="ML-KEM-768")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=3)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"# device : {describe(dev)}")
    print(f"# ps     : {args.ps}   B = {args.batch}")

    cases = [
        c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == args.ps
    ]
    dk = bytes.fromhex(cases[0]["inputs"]["dk"])
    ct = bytes.fromhex(cases[0]["inputs"]["c"])
    dec = Decapsulator(args.ps, dk, device=dev)

    B = args.batch
    arr = np.frombuffer(b"".join([ct] * B), np.uint8).reshape(B, len(ct))
    tod = torch.from_numpy(arr.copy()).to(dev)

    for _ in range(3):
        dec.decaps_torch(tod, B)
    _sync(dev)

    act = torch.profiler.ProfilerActivity.CUDA
    if dev.type == "xpu":
        act = torch.profiler.ProfilerActivity.XPU
    with torch.profiler.profile(activities=[act]) as prof:
        for _ in range(args.iters):
            dec.decaps_torch(tod, B)
    _sync(dev)

    # per-kernel min single-call time × calls per invocation
    per_key = defaultdict(list)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CPU:
            continue
        t = getattr(ev, "device_time", 0.0)
        if t:
            per_key[ev.key].append(t)

    # device_time is in **microseconds**
    rows = []
    for key, ts in per_key.items():
        n_per_call = len(ts) / args.iters
        rows.append((min(ts) * n_per_call / 1000.0, key, n_per_call))
    rows.sort(reverse=True)

    total = sum(r[0] for r in rows)
    print(f"\n{'kernel':<52} {'ms/call':>9} {'x/call':>7} {'share':>7}")
    for ms, key, n in rows[:15]:
        print(f"{key[:52]:<52} {ms:>9.3f} {n:>7.1f} {ms / total * 100:>6.1f}%")
    print(f"{'TOTAL':<52} {total:>9.3f}")
    return 0


def _sync(dev):
    from triton_crypto.devices import sync

    sync(dev)


if __name__ == "__main__":
    raise SystemExit(main())
