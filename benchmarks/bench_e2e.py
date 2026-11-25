#!/usr/bin/env python
"""End-to-end batched decapsulation benchmark.

    python benchmarks/bench_e2e.py                       # ML-KEM-768, auto-select device
    python benchmarks/bench_e2e.py --device cuda
    python benchmarks/bench_e2e.py --ps ML-KEM-512 --batch 1,256,1024,4096,8192

Measures the **full entry point** `Decapsulator.decaps()`, including numpy packing / H2D /
the entire device-side flow / D2H — i.e. the latency the caller actually sees. It also
reports device-only time (`decaps_torch`), to separate "is the GPU fast" from "is
end-to-end fast".

Ciphertexts use official NIST ACVP vectors (`data/acvp/`), no kyber-py needed.
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
    """Take **one pair** (dk, ct, K).

    Note: every ACVP decapsulation vector has **its own dk** (10 vectors measured,
    10 different dks). So a batch benchmark must replicate the same (dk, ct) pair B
    times — using cases[0].dk with the ct of all cases was wrong (9/10 decrypted to
    garbage; it did not affect timing, but made the correctness check fail for no
    obvious reason).
    """
    cases = [c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    if not cases:
        raise SystemExit(f"no decapsulation vectors for {ps} in data/acvp")
    c = cases[0]
    return _h2b(c["inputs"]["dk"]), _h2b(c["inputs"]["c"]), _h2b(c["expected"]["k"])


def _bench(fn, n: int, warmup: int, warm_seconds: float = 0.0):
    for _ in range(warmup):
        fn()
    torch_sync()
    if warm_seconds:
        # GPU clocks boost from 900 MHz idle to ~2.2 GHz. Measuring without first
        # saturating it means the first few rounds hit low clocks, and min/med can
        # differ twofold (measured on this machine).
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
        help="warm up by wall-clock time to push GPU clocks to boost (2-3 s recommended)",
    )
    ap.add_argument(
        "--baseline",
        type=float,
        default=None,
        help="optional: single-thread baseline µs/op, used to print the speedup",
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
        print(f"\n# single-thread baseline {args.baseline:.3f} µs/op")

    _selfcheck(dec, ct1, dev)
    return 0


def _selfcheck(dec, ct1, dev):
    """Confirm the fused kernel really takes effect and is **byte-identical** to the
    full fallback path.

    This is one of the traps this project has hit: the fused kernel silently returns
    False because of one `is_contiguous()` check, the whole pipeline quietly falls back
    to the slow path, and the numbers look normal while being meaningless. So the
    benchmark script runs the fallback path itself as a control.
    """
    import mlkem_triton.pipeline as pl

    # Must patch the names in the `pipeline` module: `pipeline.py` does
    # `from .kernels.fused_gemm import fused_dec_gemm, fused_enc_gemm`,
    # already bound at import time; changing attributes in the `kernels.fused_gemm`
    # module does **not** take effect (my first attempt did exactly that, and the
    # measured "gain" was 0.003 ms — all fake).
    B = 1024
    batch = [ct1] * B
    fused = dec.decaps(batch)

    saved = (pl.fused_dec_gemm, pl.fused_enc_gemm)
    try:
        pl.fused_dec_gemm = lambda *a, **k: None  # force fallback
        pl.fused_enc_gemm = lambda *a, **k: False
        fallback = dec.decaps(batch)
        fb_min, _ = _bench(lambda: dec.decaps(batch), 8, 2)
    finally:
        pl.fused_dec_gemm, pl.fused_enc_gemm = saved

    ok = all(fused[i].tobytes() == fallback[i].tobytes() for i in range(B))
    full_min, _ = _bench(lambda: dec.decaps(batch), 8, 2)
    print("\n# self-check")
    print(f"  fused vs full-fallback outputs identical : {ok}")
    print(
        f"  end-to-end fused {full_min:.3f} ms vs full fallback {fb_min:.3f} ms"
        f" -> {fb_min / full_min:.2f}x"
    )
    if not ok:
        raise SystemExit("fused and fallback paths produced different outputs!")


if __name__ == "__main__":
    raise SystemExit(main())
