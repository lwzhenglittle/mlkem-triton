#!/usr/bin/env python
"""把「Â 重新生成」塞回计时区间，拆开我们相对 cuPQC 的 1.61×。

    python benchmarks/bench_rebuild.py --batch 8192

背景：cuPQC 的 decaps 是无状态 API，每条密文都要重跑 k² 次 SampleNTT
（SHAKE-128，840 B/次）得到 Â；我们的主路径把 Â 预先展开常驻显存，每 op 零 SHAKE。
所以「我们比 cuPQC 快 1.61×」里有多少来自跳过 Â、多少来自 Toeplitz-vs-NTT，
不拆开就说不清。

本脚本报三档：

    A. 预展开（主路径）           per-op = decaps/B
    B. 每批重建一次（同信息量）    per-op = rebuild/B + decaps/B
    C. 每 op 重建（cuPQC 形态）    per-op = rebuild + decaps/B   ← 没部署价值，只作上界

B 是「同样掌握一把密钥、但允许批处理」的公平口径；C 是无状态 API 被迫的形态。
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

# 参照值：NVIDIA cuPQC 在 GB10 上实测（BD=32, batch 262144, 纯 kernel）
CUPQC_NS = {"GB10": 201.7, "H100(官方)": 117.6}


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

    # ---- 正确性：重建权重后整条流水线仍要对 ----
    Wd, We = rb.rebuild(dk)
    dec.W_dec, dec.W_enc = Wd, We
    got = dec.decaps([ct] * 8)
    assert all(x.tobytes() == want for x in got), "重建权重下解封装不正确"
    print("重建权重下的解封装：正确\n")

    # ---- 重建各段成本 ----
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

    print(f"重建一次（k={k}，{k * k} 次 SampleNTT，共 {k * k * 840} B SHAKE-128）：")
    print(f"  rebuild 全流程            {r_min:8.3f} ms  (min)")
    print(f"  其中 SHAKE-128 一步        {shake_min:8.3f} ms  ← cuPQC 每 op 都要付这个")
    print()

    print(f"{'B':>7} {'decaps ms':>10} {'A: 预展开':>11} {'B: 每批重建':>12} {'C: 每 op 重建':>13}")
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
        print(f"{B:>7} {d_min:>10.3f} {a_ns:>10.1f}ns {b_ns:>11.1f}ns {c_ns:>12.1f}ns")

    print(
        f"\n对照 cuPQC（GB10 实测 {CUPQC_NS['GB10']} ns/op，H100 官方 "
        f"{CUPQC_NS['H100(官方)']} ns/op）："
    )
    for B, (a, b, _c) in results.items():
        print(
            f"  B={B:<6} A {a:6.1f}ns ({CUPQC_NS['GB10'] / a:.2f}x)   "
            f"B {b:6.1f}ns ({CUPQC_NS['GB10'] / b:.2f}x)"
        )

    print("\n# 解读")
    worst = max(v[1] for v in results.values())
    if worst < CUPQC_NS["GB10"]:
        print(
            f"  即使把 Â 的重建完全算进计时（口径 B），我们仍然比 cuPQC 快 "
            f"{CUPQC_NS['GB10'] / worst:.2f}x。"
        )
    else:
        print(
            f"  把 Â 重建算进计时后（口径 B），最差一档是 {worst:.1f} ns/op，"
            f"比 cuPQC 的 {CUPQC_NS['GB10']} ns/op 慢 "
            f"{worst / CUPQC_NS['GB10']:.2f}x。"
        )
    print("  口径 C（每 op 重建一次）不是可行部署形态，只用来标出上界。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
