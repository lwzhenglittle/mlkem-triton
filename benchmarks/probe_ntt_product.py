"""The negacyclic product at 2^20, with the unit accounting written down once.

`LargeNTT.mul_torch(a, b)` is two forward transforms, a pointwise product and one inverse, so a
*product* costs three *transforms*. The published figures for it ("13x one core, 0.94x sixteen
processes") divided the device's **products per second** by the CPU's **transforms per second**.
That is a factor-of-three unit error in the CPU's favour, and it inverted the conclusion: with
both sides in products per second the device wins by ~2.8x, not 0.94x.

This script exists so the accounting cannot drift again: it measures the device product, takes the
CPU transform rate from the same barrier method `bench_16p_barrier.py` uses, and prints every
intermediate quantity (transforms/s, products/s, and both ratios) instead of one number that has
to be trusted.

Usage:
    python benchmarks/probe_ntt_product.py [--logn 20] [--reps 5] [--cpu-reps 2]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# One product = forward(a) + forward(b) + inverse + pointwise. The pointwise pass is elementwise
# and does not count as a transform, but the inverse does.
TRANSFORMS_PER_PRODUCT = 3

Q = 2013265921  # BabyBear


def device_product_ms(logn: int, reps: int, fused: bool = False) -> tuple[float, float]:
    import numpy as np
    import torch

    from triton_crypto.primitives.ntt.large import LargeNTT

    n = 1 << logn
    t = LargeNTT(n, Q)
    rng = np.random.default_rng(0)
    a = torch.as_tensor(rng.integers(0, Q, n, dtype=np.int64), dtype=torch.int32, device="xpu")
    b = torch.as_tensor(rng.integers(0, Q, n, dtype=np.int64), dtype=torch.int32, device="xpu")
    run = t.mul_fused_torch if fused else t.mul_torch
    run(a, b)  # warm: tables, JIT, allocator
    torch.xpu.synchronize()

    times = []
    for _ in range(reps):
        torch.xpu.synchronize()
        start = torch.xpu.Event(enable_timing=True)
        end = torch.xpu.Event(enable_timing=True)
        start.record()
        run(a, b)
        end.record()
        torch.xpu.synchronize()
        times.append(start.elapsed_time(end))
    return min(times), statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logn", type=int, default=20)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cpu-reps", type=int, default=2)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    from bench_16p_barrier import families, measure

    cfg = families(1024)["ntt-2^20"]
    cfg = dict(cfg, logn=args.logn, reps=6)
    cpu_one = max(measure("ntt-2^20", cfg, 1)["batch"] for _ in range(args.cpu_reps))
    cpu_many = max(measure("ntt-2^20", cfg, 16)["batch"] for _ in range(args.cpu_reps))

    pmin, pmed = device_product_ms(args.logn, args.reps)
    fmin, fmed = device_product_ms(args.logn, args.reps, fused=True)
    dev_per_s = 1000.0 / pmin
    dev_tx_per_s = dev_per_s * TRANSFORMS_PER_PRODUCT

    print(f"device product : {pmin:.3f} ms (min), {pmed:.3f} ms (median) of {args.reps}")
    print(
        f"device fused   : {fmin:.3f} ms (min), {fmed:.3f} ms (median)  "
        f"= {pmin / fmin:.2f}x the 42-pass version"
    )
    print(
        f"CPU 1 thread   : {cpu_one:9.2f} transforms/s  -> {cpu_one / TRANSFORMS_PER_PRODUCT:8.2f} products/s"
    )
    print(
        f"CPU 16 proc    : {cpu_many:9.2f} transforms/s  -> {cpu_many / TRANSFORMS_PER_PRODUCT:8.2f} products/s"
    )
    fused_per_s = 1000.0 / fmin
    print()
    print(
        f"device         : {dev_per_s:9.2f} products/s  = {dev_tx_per_s:8.2f} transform-equivalents/s"
    )
    print(f"device fused   : {fused_per_s:9.2f} products/s")
    print()
    print("ratios, same unit (products/s) -- the correct comparison:")
    print(f"  vs 1 thread  : {dev_per_s / (cpu_one / TRANSFORMS_PER_PRODUCT):6.2f}x")
    print(f"  vs 16 process: {dev_per_s / (cpu_many / TRANSFORMS_PER_PRODUCT):6.2f}x")
    print(f"  fused vs 16 p: {fused_per_s / (cpu_many / TRANSFORMS_PER_PRODUCT):6.2f}x")
    print()
    print("ratios mixing units (device products/s over CPU transforms/s) -- the old numbers:")
    print(f"  vs 1 thread  : {dev_per_s / cpu_one:6.2f}x")
    print(f"  vs 16 process: {dev_per_s / cpu_many:6.2f}x   <- published as 0.94x")

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "logn": args.logn,
                    "device_product_ms_min": pmin,
                    "device_product_ms_median": pmed,
                    "cpu_1t_transforms": cpu_one,
                    "cpu_16p_transforms": cpu_many,
                    "transforms_per_product": TRANSFORMS_PER_PRODUCT,
                    "ratio_vs_1t_products": dev_per_s / (cpu_one / TRANSFORMS_PER_PRODUCT),
                    "ratio_vs_16p_products": dev_per_s / (cpu_many / TRANSFORMS_PER_PRODUCT),
                    "ratio_vs_16p_mixed_units": dev_per_s / cpu_many,
                    "device_fused_product_ms_min": fmin,
                    "fused_speedup": pmin / fmin,
                    "ratio_fused_vs_16p_products": fused_per_s
                    / (cpu_many / TRANSFORMS_PER_PRODUCT),
                },
                indent=2,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
