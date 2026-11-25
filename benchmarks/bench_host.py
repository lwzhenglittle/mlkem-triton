#!/usr/bin/env python
"""主机侧开销拆解 + 三种输入形态对比。

    python benchmarks/bench_host.py --batch 4096

`decaps()` 的端到端 = 主机侧（打包 + H2D + D2H）+ 设备侧。设备侧再快，
主机侧那几毛钱不省掉也到不了 ceil。这个脚本把它拆开。

**注意：这个脚本里的绝对值不能信到 10% 以内。** 本机 GPU 时钟在 0.9 GHz（空载）
到 2.2 GHz（压满）之间跳，而 `xe` 驱动没有暴露任何调频接口
（`/sys/class/drm/*/gt_*_freq_mhz` 不存在），所以同一个脚本跑两遍能差 30%。
只看**同一轮内的比值**。

几条结论（都是实测，逐条有数）：

- **pinned 中转在 Intel 上划算、在 NVIDIA 上不划算。** 这台 Arc 的 pageable→device
  只有 ~14 GB/s（驱动要过 bounce buffer），pinned→device 的纯 DMA 有 ~83 GB/s；
  多一次 numpy→pinned 的 memcpy（~178 GB/s）之后仍然净赚约 2 倍（0.50 → 0.28 ms）。
  但 GB10（Grace-Blackwell，CPU/GPU 一致性互连）的 pageable 直拷本来就有 50 GB/s，
  pinned 反而更慢。所以 `pipeline._prefer_pinned()` 第一次调用时用 4 MiB 缓冲
  两边各测一遍，按结果决定，不写死。
- **H2D 没法靠双流重叠藏掉。** 本机驱动上「两个流、零依赖」的 DMA ∥ 计算实测
  1.328 ms，而同流串行 1.307 ms —— 完全没有并行；双缓冲流水更慢（1.401 ms）。
  原因大概率是 pageable 拷贝那一段是**调用线程上的 CPU memcpy**，本来就没法和
  CPU 侧的入队并行。
- **逐条长度校验是纯浪费。** `all(len(c) == n0 for c in lst)` 在 B=4096 时要
  ~0.13 ms（占端到端 6%），改成「拼起来的总长 == n0 × 条数」是 O(1) 的。

所以真正能省的是**调用方式**：让 `decaps()` 少拷一次、少校验一次，并且在
值得的时候走 pinned。
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
        ("torch.Tensor（已在设备）", lambda: dec.decaps(ct_dev)),
        ("bytes 一整块", lambda: dec.decaps(joined)),
        ("np.ndarray [B,L]", lambda: dec.decaps(arr_rw)),
        ("list[bytes]", lambda: dec.decaps(lst)),
    ]
    warm_s = args.warm_seconds
    dev_only, _ = _bench(lambda: dec.decaps_torch(ct_dev, B), dev, warm_seconds=warm_s)
    print(f"{'输入形态':<28} {'min':>8} {'med':>8} {'主机侧净开销':>12}")
    print(f"{'（仅设备侧作为基准）':<28} {dev_only:>7.3f}m")
    for name, fn in rows:
        out = fn()
        assert out.shape == (B, 32), (name, out.shape)
        assert all(out[i].tobytes() == want for i in range(0, B, max(1, B // 8))), name
        mn, md = _bench(fn, dev, warm_seconds=warm_s)
        print(f"{name:<28} {mn:>7.3f}m {md:>7.3f}m {mn - dev_only:>+11.3f}m")

    print(f"\n# 单段成本（B={B}）")
    pin = torch.empty(B, L, dtype=torch.uint8, pin_memory=True)
    dbuf = torch.empty(B, L, dtype=torch.uint8, device=dev)
    seg = [
        ('b"".join(list)', lambda: b"".join(lst)),
        ("逐条长度校验 all(len==L)", lambda: all(len(c) == L for c in lst)),
        ("H2D pageable -> device", lambda: dbuf.copy_(torch.from_numpy(arr_rw))),
        ("H2D pinned  -> device (纯 DMA)", lambda: dbuf.copy_(pin, non_blocking=True)),
        ("memcpy bytes -> pinned", lambda: pin.view(-1).copy_(torch.from_numpy(arr.reshape(-1)))),
    ]
    for name, fn in seg:
        mn, md = _bench(fn, dev, n=15, warmup=3, warm_seconds=warm_s)
        print(f"  {name:<34} {mn:>7.3f}m ({B * L / mn / 1e6:5.1f} GB/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
