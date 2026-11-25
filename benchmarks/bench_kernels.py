#!/usr/bin/env python
"""逐内核耗时拆分（回答「时间花在哪」）。

    python benchmarks/bench_kernels.py --device cuda --batch 4096

用 `torch.profiler` 拿设备侧每个 kernel 的 self time。**注意用 `min × 调用次数`
而不是 `avg`** —— 本项目的 `_pre` 内核有 26 倍的冷启动离散（min 92 µs / avg
468 / max 2424），用平均值会把它的占比算成第一，和按 min 算出来的排名完全相反。
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

from mlkem_triton import acvp, describe, resolve_device
from mlkem_triton.pipeline import Decapsulator


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

    # 每个 kernel 的 min 单次耗时 × 每次调用次数
    per_key = defaultdict(list)
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CPU:
            continue
        t = getattr(ev, "device_time", 0.0)
        if t:
            per_key[ev.key].append(t)

    # device_time 的单位是**微秒**
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
    from mlkem_triton.devices import sync

    sync(dev)


if __name__ == "__main__":
    raise SystemExit(main())
