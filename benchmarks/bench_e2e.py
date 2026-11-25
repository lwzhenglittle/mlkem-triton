#!/usr/bin/env python
"""端到端批量解封装基准。

    python benchmarks/bench_e2e.py                       # ML-KEM-768, 自动选设备
    python benchmarks/bench_e2e.py --device cuda
    python benchmarks/bench_e2e.py --ps ML-KEM-512 --batch 1,256,1024,4096,8192

测的是**完整入口** `Decapsulator.decaps()`，包含 numpy 打包 / H2D / 设备侧
全流程 / D2H —— 也就是调用方真正看到的延迟。同时报告仅设备侧的耗时
（`decaps_torch`），方便区分「GPU 算得快不快」和「端到端快不快」。

密文用 NIST ACVP 官方向量（`data/acvp/`），不需要 kyber-py。
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

dev_override = None
from mlkem_triton.pipeline import Decapsulator


def _h2b(v):
    return bytes.fromhex(v) if isinstance(v, str) else v


def _load_cases(ps: str):
    """取**一对** (dk, ct, K)。

    注意：ACVP 的每个 decapsulation 向量都有**自己的 dk**（实测 10 条向量
    10 个不同 dk）。所以批量基准必须拿同一对 (dk, ct) 复制 B 份 —— 之前
    用 cases[0].dk 配所有 cases 的 ct 是错的（9/10 解密出垃圾，虽然不影响
    计时，但会让正确性检查看着莫名其妙地失败）。
    """
    cases = [c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    if not cases:
        raise SystemExit(f"data/acvp 里没有 {ps} 的 decapsulation 向量")
    c = cases[0]
    return _h2b(c["inputs"]["dk"]), _h2b(c["inputs"]["c"]), _h2b(c["expected"]["k"])


def _bench(fn, n: int, warmup: int, warm_seconds: float = 0.0):
    for _ in range(warmup):
        fn()
    torch_sync()
    if warm_seconds:
        # GPU 时钟会从 900 MHz 空载一路 boost 到 ~2.2 GHz。不先压满就这么测，
        # 前几轮会撞上低频，min/med 都能差一倍（本机实测）。
        import time as _t

        t0 = _t.perf_counter()
        while _t.perf_counter() - t0 < warm_seconds:
            for _ in range(5):
                fn()
            torch_sync()
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        torch_sync()
        ts.append((time.perf_counter() - t) * 1e3)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def torch_sync():
    sync(dev_override)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--ps", default="ML-KEM-768")
    ap.add_argument("--batch", default="1,256,1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument(
        "--warm-seconds",
        type=float,
        default=0.0,
        help="按墙钟时间预热，用来把 GPU 时钟顶到 boost（建议 2-3 秒）",
    )
    ap.add_argument(
        "--baseline", type=float, default=None, help="可选：单线程基线 µs/op，用来打印加速比"
    )
    args = ap.parse_args()

    global dev_override
    dev = resolve_device(args.device)
    dev_override = dev
    print(f"# device : {describe(dev)}")
    print(f"# ps     : {args.ps}")

    dk, ct1, k1 = _load_cases(args.ps)
    dec = Decapsulator(args.ps, dk, device=dev)

    print(
        f"\n{'B':>7} {'e2e min':>10} {'e2e med':>10} {'dev min':>10} "
        f"{'µs/op':>8} {'M ops/s':>9} {'exact':>6}"
    )
    for B in [int(x) for x in args.batch.split(",")]:
        batch = [ct1] * B
        got = dec.decaps(batch)
        ok = all(got[i].tobytes() == k1 for i in range(B))

        e_min, e_med = _bench(
            lambda batch=batch: dec.decaps(batch),
            args.iters,
            args.warmup,
            args.warm_seconds,
        )

        arr = np.frombuffer(b"".join(batch), np.uint8).reshape(B, len(batch[0]))
        ct = torch.from_numpy(arr).to(dev)
        d_min, _ = _bench(
            lambda ct=ct, B=B: dec.decaps_torch(ct, B),
            args.iters,
            args.warmup,
            args.warm_seconds,
        )

        print(
            f"{B:>7} {e_min:>9.3f}m {e_med:>9.3f}m {d_min:>9.3f}m "
            f"{e_min * 1e3 / B:>8.3f} {B / (e_min * 1e-3) / 1e6:>9.1f} {ok!s:>6}"
        )

    if args.baseline:
        print(f"\n# 单线程基线 {args.baseline:.3f} µs/op")

    _selfcheck(dec, ct1, dev)
    return 0


def _selfcheck(dec, ct1, dev):
    """确认融合内核真的生效，且与全回退路径**逐字节相同**。

    这是这个项目踩过的坑之一：融合内核因为一个 `is_contiguous()` 判定静默
    返回 False，整条流水线偷偷退回慢路径，数字看着正常但白算。所以基准
    脚本自己把回退路径也跑一遍做对照。
    """
    import mlkem_triton.pipeline as pl

    # 必须 patch `pipeline` 模块里的名字：`pipeline.py` 用的是
    # `from .kernels.fused_gemm import fused_dec_gemm, fused_enc_gemm`，
    # 导入时已经绑定，改 `kernels.fused_gemm` 模块里的属性**不会**生效
    # （我第一次就是这么干的，测出来的「收益」是 0.003 ms —— 全是假的）。
    B = 1024
    batch = [ct1] * B
    fused = dec.decaps(batch)

    saved = (pl.fused_dec_gemm, pl.fused_enc_gemm)
    try:
        pl.fused_dec_gemm = lambda *a, **k: None  # 强制回退
        pl.fused_enc_gemm = lambda *a, **k: False
        fallback = dec.decaps(batch)
        fb_min, _ = _bench(lambda: dec.decaps(batch), 8, 2)
    finally:
        pl.fused_dec_gemm, pl.fused_enc_gemm = saved

    ok = all(fused[i].tobytes() == fallback[i].tobytes() for i in range(B))
    full_min, _ = _bench(lambda: dec.decaps(batch), 8, 2)
    print("\n# 自检")
    print(f"  融合路径 vs 全回退路径输出一致 : {ok}")
    print(f"  端到端 融合 {full_min:.3f} ms vs 全回退 {fb_min:.3f} ms -> {fb_min / full_min:.2f}x")
    if not ok:
        raise SystemExit("融合路径与回退路径输出不一致！")


if __name__ == "__main__":
    raise SystemExit(main())
