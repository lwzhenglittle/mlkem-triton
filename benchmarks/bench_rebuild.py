#!/usr/bin/env python
"""Put "rebuild Â" back into the timed region and break down our 1.61× over cuPQC.

    python benchmarks/bench_rebuild.py --batch 8192

Background: cuPQC's decaps is a stateless API, so every ciphertext must rerun k² SampleNTT
(SHAKE-128, 840 B each) to obtain Â; our main path pre-expands Â and keeps it resident in
GPU memory, with zero SHAKE per op. So how much of "we are 1.61× faster than cuPQC" comes
from skipping Â and how much from Toeplitz-vs-NTT cannot be stated without separating
them.

This script reports three modes:

    A. pre-expanded (main path)     per-op = decaps/B
    B. rebuild once per batch       per-op = rebuild/B + decaps/B   (same information content)
    C. rebuild per op (cuPQC form)  per-op = rebuild + decaps/B     ← no deployment value,
                                                                    only an upper bound

B is the fair comparison for "holding the same key but allowed to batch"; C is the form
forced by a stateless API.
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
from mlkem_triton.kernels.rebuild import GpuKeyRebuilder
from mlkem_triton.pipeline import Decapsulator

# reference: NVIDIA cuPQC measured on GB10 (BD=32, batch 262144, pure kernel)
CUPQC_NS = {"GB10": 201.7, "H100(official)": 117.6}


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
    ap.add_argument("--batch", default="1024,4096,8192,16384")
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()

    dev = resolve_device(args.device)
    print(f"# device : {describe(dev)}")
    print(f"# ps     : {args.ps}\n")

    c = next(
        x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == args.ps
    )
    dk = bytes.fromhex(c["inputs"]["dk"])
    ct = bytes.fromhex(c["inputs"]["c"])
    want = bytes.fromhex(c["expected"]["k"])
    dec = Decapsulator(args.ps, dk, device=dev)
    rb = GpuKeyRebuilder(args.ps, device=dev)
    k = dec.p.k

    # ---- correctness: after rebuilding the weights the whole pipeline must still pass ----
    Wd, We = rb.rebuild(dk)
    dec.W_dec, dec.W_enc = Wd, We
    got = dec.decaps([ct] * 8)
    assert all(x.tobytes() == want for x in got), "decapsulation is incorrect with rebuilt weights"
    print("Decapsulation with rebuilt weights: correct\n")

    # ---- cost of each rebuild stage ----
    r_min, _ = _bench(lambda: rb.rebuild(dk), dev, n=args.iters)
    out64_w = 5 * 168 // 8
    from mlkem_triton.kernels.rebuild import _padded_msgs

    msgs = _padded_msgs(rb.rho_of(dk), k)
    if msgs is not None:
        mt = torch.from_numpy(msgs).to(dev)
        o64 = torch.empty((k * k, out64_w), dtype=torch.int64, device=dev)
        kern = rb._shake()
        shake_min, _ = _bench(
            lambda: kern[(1,)](
                mt.view(torch.int64), o64, k * k, 1, 21, OUTW=out64_w, BLOCK=128, num_warps=4
            ),
            dev,
            n=args.iters,
        )

    print(f"One rebuild (k={k}, {k * k} SampleNTT calls, {k * k * 840} B SHAKE-128 total):")
    print(f"  rebuild (full path)           {r_min:8.3f} ms  (min)")
    print(f"  of which the SHAKE-128 step   {shake_min:8.3f} ms  ← cuPQC pays this every op")
    print()

    print(
        f"{'B':>7} {'decaps ms':>10} {'A: pre-expanded':>15} "
        f"{'B: per-batch rebuild':>20} {'C: per-op rebuild':>17}"
    )
    results = {}
    for B in [int(x) for x in args.batch.split(",")]:
        arr = np.frombuffer(b"".join([ct] * B), np.uint8).reshape(B, len(ct))
        tod = torch.from_numpy(arr).to(dev)
        d_min, _ = _bench(
            lambda tod=tod, B=B: dec.decaps_torch(tod, B),
            dev,
            n=max(4, args.iters // 2),
        )
        a_ns = d_min * 1e6 / B
        b_ns = (r_min * 1e6 + d_min * 1e6) / B
        c_ns = r_min * 1e6 + a_ns
        results[B] = (a_ns, b_ns, c_ns)
        print(f"{B:>7} {d_min:>10.3f} {a_ns:>13.1f}ns {b_ns:>18.1f}ns {c_ns:>15.1f}ns")

    print(
        f"\nvs cuPQC (GB10 measured {CUPQC_NS['GB10']} ns/op, H100 official "
        f"{CUPQC_NS['H100(official)']} ns/op):"
    )
    for B, (a, b, _c) in results.items():
        print(
            f"  B={B:<6} A {a:6.1f}ns ({CUPQC_NS['GB10'] / a:.2f}x)   "
            f"B {b:6.1f}ns ({CUPQC_NS['GB10'] / b:.2f}x)"
        )

    print("\n# interpretation")
    worst = max(v[1] for v in results.values())
    if worst < CUPQC_NS["GB10"]:
        print(
            f"  Even with Â rebuild fully charged to the timing (mode B), we are still "
            f"{CUPQC_NS['GB10'] / worst:.2f}x faster than cuPQC."
        )
    else:
        print(
            f"  With Â rebuild charged to the timing (mode B), the worst case is "
            f"{worst:.1f} ns/op, "
            f"{worst / CUPQC_NS['GB10']:.2f}x slower than cuPQC's "
            f"{CUPQC_NS['GB10']} ns/op."
        )
    print(
        "  Mode C (rebuild every op) is not a viable deployment form; it only marks an upper bound."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
