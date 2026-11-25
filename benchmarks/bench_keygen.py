#!/usr/bin/env python
"""Key generation throughput (FIPS 203 Alg. 16).

    python benchmarks/bench_keygen.py --batch 1024,4096,8192

keygen is **the only one of the three operations that must use the NTT** — the
multiplications in decaps/encaps are folded into fixed-matrix GEMMs, while keygen's
`s`/`e` are new for every key, with no reusable matrix.
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

from mlkem_triton import describe, resolve_device
from mlkem_triton.devices import sync
from mlkem_triton.keygen import KeyGenerator

# cuPQC on GB10, same machine (pure kernel): keygen 173.9 ns (BD=32) / 194.9 ns (default)
CUPQC_KEYGEN_NS = {"tuned": 173.9, "default": 194.9}
# liboqs 0.16.0 same-machine single-thread keygen 7.73 µs; for the aggregate over 16
# concurrent processes see docs/compare.md
LIBOQS_KEYGEN_US = 7.731


def _bench(fn, dev, n=12, warmup=3):
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
    kg = KeyGenerator(args.ps, device=dev)
    same_hw = dev.type == "cuda"

    print(
        f"{'B':>7} {'e2e ms':>9} {'e2e µs/op':>10} {'dev µs/op':>10} "
        f"{'Mops/s':>9} {'vs liboqs-1T':>13}"
        f"{'    vs cuPQC tuned' if same_hw else ''}"
    )
    for B in [int(x) for x in args.batch.split(",")]:
        d = np.random.default_rng(1).integers(0, 256, (B, 32), dtype=np.uint8)
        z = np.random.default_rng(2).integers(0, 256, (B, 32), dtype=np.uint8)
        dt = torch.from_numpy(d.copy()).to(dev)
        zt = torch.from_numpy(z.copy()).to(dev)
        e_min, _ = _bench(lambda B=B: kg.generate(B), dev, n=max(4, args.iters // 2))
        d_min, _ = _bench(lambda dt=dt, zt=zt: kg.generate_torch(dt, zt), dev, n=args.iters)
        us = e_min * 1e3 / B
        ns = d_min * 1e6 / B
        tail = f"{CUPQC_KEYGEN_NS['tuned'] / ns:>17.2f}x" if same_hw else ""
        print(
            f"{B:>7} {e_min:>9.3f} {us:>10.4f} {ns / 1e3:>10.4f} "
            f"{B / (d_min * 1e-3) / 1e6:>9.2f} {LIBOQS_KEYGEN_US / us:>12.2f}x{tail}"
        )
    if same_hw:
        print(
            f"\n# cuPQC on GB10, same machine: keygen {CUPQC_KEYGEN_NS['tuned']} ns/op "
            f"(BD=32) / {CUPQC_KEYGEN_NS['default']} ns (default)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
