#!/usr/bin/env python
"""Encapsulation (encaps) throughput, compared against same-machine cuPQC numbers.

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

from triton_crypto import describe, resolve_device
from triton_crypto.devices import sync
from triton_crypto.mlkem import acvp
from triton_crypto.mlkem.encaps import Encapsulator

# cuPQC on GB10, same machine (BD=32 / SDK default BD=128), pure kernel
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
    print("ACVP check passed\n")

    # cuPQC numbers were measured on GB10; only comparable on CUDA machines
    same_hw = dev.type == "cuda"
    hdr = (f"{'vs cuPQC tuned':>14} {'vs cuPQC default':>16}") if same_hw else ""
    print(f"{'B':>7} {'e2e ms':>9} {'e2e µs/op':>10} {'dev µs/op':>10} {'Mops/s':>9} {hdr}")
    for B in [int(x) for x in args.batch.split(",")]:
        msgs = [m0] * B
        arr = np.frombuffer(b"".join(msgs), np.uint8).reshape(B, 32).copy()
        mt = torch.from_numpy(arr).to(dev)
        e_min, _ = _bench(lambda B=B: enc.encapsulate([m0] * B), dev, n=args.iters)
        d_min, _ = _bench(lambda mt=mt, B=B: enc.encapsulate_torch(mt, B), dev, n=args.iters)
        us = e_min * 1e3 / B
        ns = d_min * 1e6 / B
        tail = (
            (f"{CUPQC_ENCAPS_NS['tuned'] / ns:>13.2f}x {CUPQC_ENCAPS_NS['default'] / ns:>15.2f}x")
            if same_hw
            else ""
        )
        print(
            f"{B:>7} {e_min:>9.3f} {us:>10.3f} {ns / 1e3:>10.3f} "
            f"{B / (d_min * 1e-3) / 1e6:>9.2f} {tail}"
        )
    if same_hw:
        print(
            f"\n# against cuPQC same-machine measurements (GB10): tuned "
            f"{CUPQC_ENCAPS_NS['tuned']} ns/op, SDK default "
            f"{CUPQC_ENCAPS_NS['default']} ns/op"
        )
        print(
            "# Our number is device-side (including Python enqueue), cuPQC's is pure "
            "kernel — the comparison slightly disfavours us."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
