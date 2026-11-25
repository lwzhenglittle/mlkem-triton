#!/usr/bin/env python
"""封装（encaps）吞吐，和 cuPQC 的同机实测数字对照。

python benchmarks/bench_encaps.py --batch 1024,4096,8192
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from mlkem_triton import acvp, describe, resolve_device
from mlkem_triton.devices import sync
from mlkem_triton.encaps import Encapsulator

# cuPQC 在 GB10 上的同机实测（BD=32 / SDK 默认 BD=128），纯 kernel
CUPQC_ENCAPS_NS = {"tuned": 199.7, "default": 243.5}


def _bench(fn, dev, n=15, warmup=3):
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ps", default="ML-KEM-768")
    ap.add_argument("--batch", default="1024,4096,8192")
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"# device : {describe(dev)}")
    print(f"# ps     : {args.ps}\n")

    cs = [
        x for x in acvp.load() if x["function"] == "encapsulation" and x["parameterSet"] == args.ps
    ]
    ek = bytes.fromhex(cs[0]["inputs"]["ek"])
    m0 = bytes.fromhex(cs[0]["inputs"]["m"])
    enc = Encapsulator(args.ps, ek, device=dev)

    c, K = enc.encapsulate([m0] * 8)
    assert all(c[i].tobytes() == bytes.fromhex(cs[0]["expected"]["c"]) for i in range(8))
    assert all(K[i].tobytes() == bytes.fromhex(cs[0]["expected"]["k"]) for i in range(8))
    print("ACVP 校验通过\n")

    # cuPQC 的数字是 GB10 上测的，只有 CUDA 机器上才可比
    same_hw = dev.type == "cuda"
    hdr = (f"{'vs cuPQC 调优':>14} {'vs cuPQC 默认':>14}") if same_hw else ""
    print(f"{'B':>7} {'e2e ms':>9} {'e2e µs/op':>10} {'dev µs/op':>10} {'Mops/s':>9} {hdr}")
    for B in [int(x) for x in args.batch.split(",")]:
        msgs = [m0] * B
        arr = np.frombuffer(b"".join(msgs), np.uint8).reshape(B, 32)
        mt = torch.from_numpy(arr).to(dev)
        e_min, _ = _bench(lambda B=B: enc.encapsulate([m0] * B), dev, n=args.iters)
        d_min, _ = _bench(lambda mt=mt, B=B: enc.encapsulate_torch(mt, B), dev, n=args.iters)
        us = e_min * 1e3 / B
        ns = d_min * 1e6 / B
        tail = (
            (f"{CUPQC_ENCAPS_NS['tuned'] / ns:>13.2f}x {CUPQC_ENCAPS_NS['default'] / ns:>13.2f}x")
            if same_hw
            else ""
        )
        print(
            f"{B:>7} {e_min:>9.3f} {us:>10.3f} {ns / 1e3:>10.3f} "
            f"{B / (d_min * 1e-3) / 1e6:>9.2f} {tail}"
        )
    if same_hw:
        print(
            f"\n# 对照 cuPQC 同机实测（GB10）：调优 {CUPQC_ENCAPS_NS['tuned']} ns/op，"
            f"SDK 默认 {CUPQC_ENCAPS_NS['default']} ns/op"
        )
        print("# 我们是设备侧（含 Python 入队），cuPQC 是纯 kernel —— 口径对我们略不利。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
