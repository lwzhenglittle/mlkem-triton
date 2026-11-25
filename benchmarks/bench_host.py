#!/usr/bin/env python
"""Host-side overhead breakdown + comparison of three input forms.

    python benchmarks/bench_host.py --batch 4096

The end-to-end `decaps()` = host side (packing + H2D + D2H) + device side. No matter how
fast the device side is, you cannot reach the ceiling without removing that small
host-side cost. This script breaks it down.

**Warning: absolute values in this script are not trustworthy to within 10%.** On this
machine the GPU clock swings between 0.9 GHz (idle) and 2.2 GHz (fully loaded), and the
`xe` driver exposes no frequency-control interface
(`/sys/class/drm/*/gt_*_freq_mhz` does not exist), so two runs of the same script can
differ by 30%. Only look at **ratios within the same run**.

A few conclusions (all measured, each with numbers):

- **Pinned staging pays off on Intel but not on NVIDIA.** On this Arc, pageable→device
  reaches only ~14 GB/s (the driver goes through a bounce buffer), while pinned→device
  pure DMA reaches ~83 GB/s; even after the extra numpy→pinned memcpy (~178 GB/s) it
  still nets about 2x (0.50 → 0.28 ms). But on GB10 (Grace-Blackwell, coherent CPU/GPU
  interconnect) pageable direct copy already reaches 50 GB/s, and pinned is actually
  slower. So `pipeline._prefer_pinned()` measures both with a 4 MiB buffer on first
  call and decides from the result, rather than hardcoding it.
- **H2D cannot be hidden by two-stream overlap.** On this machine's driver, "two
  streams, zero dependencies" DMA ∥ compute measures 1.328 ms, while same-stream
  serial is 1.307 ms — no overlap at all; double-buffered pipelining is even slower
  (1.401 ms). The likely reason is that the pageable copy segment is a **CPU memcpy on
  the calling thread**, which cannot overlap with CPU-side enqueue in the first place.
- **Per-item length validation is pure waste.** `all(len(c) == n0 for c in lst)` costs
  ~0.13 ms at B=4096 (6% of end-to-end); changing it to "total joined length == n0 ×
  count" is O(1).

So what can actually be saved is the **calling convention**: make `decaps()` copy once
less, validate once less, and use pinned when it is worth it.
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
from mlkem_triton.pipeline import Decapsulator


def _bench(fn, dev, n=25, warmup=5, warm_seconds=0.0):
    for _ in range(warmup):
        fn()
    sync(dev)
    if warm_seconds:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < warm_seconds:
            for _ in range(5):
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


def _load(ps):
    c = next(x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == ps)
    return (
        bytes.fromhex(c["inputs"]["dk"]),
        bytes.fromhex(c["inputs"]["c"]),
        bytes.fromhex(c["expected"]["k"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ps", default="ML-KEM-768")
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--warm-seconds", type=float, default=0.0)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    dk, ct, want = _load(args.ps)
    dec = Decapsulator(args.ps, dk, device=dev)
    B = args.batch
    L = len(ct)
    lst = [ct] * B
    joined = b"".join(lst)
    arr = np.frombuffer(joined, np.uint8).reshape(B, L)
    arr_rw = arr.copy()

    print(f"# device : {describe(dev)}")
    print(f"# ps     : {args.ps}   B = {B}   ({B * L / 1e6:.2f} MB H2D)\n")

    ct_dev = torch.from_numpy(arr_rw).to(dev)
    rows = [
        ("torch.Tensor (already on device)", lambda: dec.decaps(ct_dev)),
        ("bytes (single block)", lambda: dec.decaps(joined)),
        ("np.ndarray [B,L]", lambda: dec.decaps(arr_rw)),
        ("list[bytes]", lambda: dec.decaps(lst)),
    ]
    warm_s = args.warm_seconds
    dev_only, _ = _bench(lambda: dec.decaps_torch(ct_dev, B), dev, warm_seconds=warm_s)
    print(f"{'Input form':<32} {'min':>8} {'med':>8} {'Host-side net overhead':>22}")
    print(f"{'(device-only baseline)':<32} {dev_only:>7.3f}m")
    for name, fn in rows:
        out = fn()
        assert out.shape == (B, 32), (name, out.shape)
        assert all(out[i].tobytes() == want for i in range(0, B, max(1, B // 8))), name
        mn, md = _bench(fn, dev, warm_seconds=warm_s)
        print(f"{name:<32} {mn:>7.3f}m {md:>7.3f}m {mn - dev_only:>+21.3f}m")

    print(f"\n# per-segment cost (B={B})")
    pin = torch.empty(B, L, dtype=torch.uint8, pin_memory=True)
    dbuf = torch.empty(B, L, dtype=torch.uint8, device=dev)
    seg = [
        ('b"".join(list)', lambda: b"".join(lst)),
        ("per-item length check all(len==L)", lambda: all(len(c) == L for c in lst)),
        ("H2D pageable -> device", lambda: dbuf.copy_(torch.from_numpy(arr_rw))),
        ("H2D pinned  -> device (pure DMA)", lambda: dbuf.copy_(pin, non_blocking=True)),
        ("memcpy bytes -> pinned", lambda: pin.view(-1).copy_(torch.from_numpy(arr.reshape(-1)))),
    ]
    for name, fn in seg:
        mn, md = _bench(fn, dev, n=15, warmup=3, warm_seconds=warm_s)
        print(f"  {name:<34} {mn:>7.3f}m ({B * L / mn / 1e6:5.1f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
