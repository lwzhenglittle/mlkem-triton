#!/usr/bin/env python
"""ML-DSA signature-verification throughput (FIPS 204 Alg. 3).

    python benchmarks/bench_mldsa_verify.py --batch 64,256,1024,4096,8192
    python benchmarks/bench_mldsa_verify.py --ps ML-DSA-87 --batch 4096 --profile

The corpus is produced by **liboqs** (mldsa-native C, a separate codebase) under one
public key, so this script is simultaneously a cross-implementation correctness check:
every signature the GPU accepts is one liboqs also accepts, and the microsecond figures
come from the same data.

Two timings are reported, as everywhere else in this project:

    device only   the whole pipeline, device-resident input, one synchronisation
                  at the end — the number to compare against other GPU kernels;
    end-to-end    numpy -> device, verify, one 1-byte-per-row result back, which is
                  what a caller that holds byte strings actually pays.

The CPU baseline is liboqs 0.16.0 with ML-DSA enabled, measured on the same machine:
single-thread (best of several passes) and aggregated over 16 concurrent *processes* —
threads do not scale here because the C code is called through ctypes in one process
and the machine is a hybrid 4P+8E+4LP part.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "tests"))

import numpy as np
import torch
from _liboqs import LibOQS

from triton_crypto.devices import describe, resolve_device, sync
from triton_crypto.mldsa.ref import MLDSA
from triton_crypto.mldsa.verify import Verifier


def _best(fn, dev, n=25, warmup=3) -> float:
    for _ in range(warmup):
        fn()
    sync(dev)
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def _liboqs_single(o: LibOQS, msgs, sigs, pk) -> float:
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        for m, s in zip(msgs, sigs, strict=True):
            assert o.verify(m, s, pk)
        best = min(best, (time.perf_counter() - t0) / len(msgs))
    return best


_WORK: tuple | None = None


def _proc_worker(_):
    """Verify `per` signatures in a child process (module level: must be picklable)."""
    ps, msgs, sigs, pk, per = _WORK
    impl = LibOQS(ps)
    for i in range(per):
        impl.verify(msgs[i], sigs[i], pk)
    return None


def _liboqs_16_procs(o: LibOQS, msgs, sigs, pk, n_proc=16) -> float:
    """Aggregate throughput over n_proc processes (the ML-KEM comparison's method)."""
    global _WORK
    import multiprocessing as mp

    per = max(1, len(msgs) // n_proc)
    _WORK = (o.ps, msgs[:per], sigs[:per], pk, per)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        pool.map(_proc_worker, range(n_proc))  # warm up (loads the .so in each child)
        t0 = time.perf_counter()
        pool.map(_proc_worker, range(n_proc))
        wall = time.perf_counter() - t0
    return wall / (per * n_proc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default="ML-DSA-65")
    ap.add_argument("--batch", default="64,256,1024,4096,8192")
    ap.add_argument("--msg-len", type=int, default=32)
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--profile", action="store_true", help="per-stage device breakdown")
    ap.add_argument("--no-cpu", action="store_true", help="skip the liboqs baseline")
    args = ap.parse_args()

    ps = args.ps
    dev = resolve_device()
    batches = [int(x) for x in args.batch.split(",") if x.strip()]
    n_max = max(batches)
    print(f"device : {describe(dev)}")
    print(f"scheme : {ps}  messages {args.msg_len} B  batch max {n_max}")
    print(f"torch  : {torch.__version__}   triton: {__import__('triton').__version__}")

    # ---------------- corpus from liboqs ----------------
    o = LibOQS(ps)
    pk, sk = o.keypair()
    msgs = [i.to_bytes(4, "little") * (args.msg_len // 4) for i in range(n_max)]
    t0 = time.perf_counter()
    sigs = [o.sign(m, sk) for m in msgs]
    gen = time.perf_counter() - t0
    print(
        f"corpus : {n_max} signatures from liboqs in {gen:.2f} s ({gen / n_max * 1e6:.0f} us/sig)"
    )

    t0 = time.perf_counter()
    cpu_ok = o.verify_many(msgs[: min(n_max, 256)], sigs[: min(n_max, 256)], pk)
    print(f"liboqs : {sum(cpu_ok)}/{len(cpu_ok)} valid on the first {len(cpu_ok)}")

    # a sample through our numpy reference (independent of the GPU kernels)
    ref = MLDSA(ps)
    sample = list(range(0, min(n_max, 8)))
    ref_ok = all(ref.verify(pk, msgs[i], sigs[i]) for i in sample)
    print(f"ref    : {sum(ref_ok for _ in sample)}/{len(sample)} valid (numpy FIPS 204)")

    # ---------------- GPU ----------------
    V = Verifier(ps, pk, device=dev)
    sig_all = np.stack([np.frombuffer(s, np.uint8) for s in sigs])
    print(
        f"\n{'B':>6} {'dev min':>10} {'dev/op':>10} {'Mops/s':>8} "
        f"{'e2e min':>10} {'e2e/op':>10} {'Mops/s':>8} {'valid':>7}"
    )

    for B in batches:
        msg_b = msgs[:B]
        sig_b = sig_all[:B]
        mu_np = np.stack(
            [np.frombuffer(ref.mu(pk, ref.format_message(m)), np.uint8) for m in msg_b]
        )
        mu_d = torch.from_numpy(mu_np.copy()).to(dev)
        sig_d = torch.from_numpy(sig_b.copy()).to(dev)

        t_dev = _best(lambda md=mu_d, sd=sig_d: V.verify_mu(md, sd), dev, n=args.reps)
        ok = V.verify(msg_b, sig_b)
        n_ok = int(ok.sum())
        # end-to-end includes the host-side H2D and the bool copy back
        t_e2e = _best(lambda mb=msg_b, sb=sig_b: V.verify(mb, sb), dev, n=args.reps)
        print(
            f"{B:>6} {t_dev * 1e3:>9.3f}ms {t_dev / B * 1e6:>9.1f}us "
            f"{B / t_dev / 1e6:>8.2f} {t_e2e * 1e3:>9.3f}ms {t_e2e / B * 1e6:>9.1f}us "
            f"{B / t_e2e / 1e6:>8.2f} {n_ok:>4}/{B}"
        )

    if args.profile:
        V.profile = True
        mu_np = np.stack(
            [np.frombuffer(ref.mu(pk, ref.format_message(m)), np.uint8) for m in msgs[:n_max]]
        )
        mu_d = torch.from_numpy(mu_np.copy()).to(dev)
        sig_d = torch.from_numpy(sig_all.copy()).to(dev)
        for _ in range(3):
            V.stage_times.clear()
            V.verify_mu(mu_d, sig_d)
        V.stage_times.clear()
        V.verify_mu(mu_d, sig_d)
        total = sum(V.stage_times.values())
        print(f"\nper stage at B={n_max} (device, synchronised; total {total * 1e3:.3f} ms):")
        for name, sec in sorted(V.stage_times.items(), key=lambda kv: -kv[1]):
            print(
                f"  {name:<12} {sec * 1e3:8.3f} ms  {sec / total * 100:5.1f}%  "
                f"{sec / n_max * 1e6:6.0f} ns/op"
            )
        V.profile = False

    # ---------------- CPU baseline ----------------
    if not args.no_cpu:
        nb = min(n_max, 2048)
        t1 = _liboqs_single(o, msgs[:nb], sigs[:nb], pk)
        print(f"\nliboqs single-thread : {t1 * 1e6:7.1f} us/op  {1 / t1 / 1e6:5.2f} Mops/s")
        if nb >= 1024:
            tp = _liboqs_16_procs(o, msgs[:nb], sigs[:nb], pk)
            print(f"liboqs 16 processes  : {tp * 1e6:7.2f} us/op  {1 / tp / 1e6:5.2f} Mops/s")
        else:
            print(
                "liboqs 16 processes  : skipped (needs >= 1024 signatures; process start-up "
                "would dominate)"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
