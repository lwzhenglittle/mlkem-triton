"""Measure a rival's sixteen-process rate the clean way, for comparison with `bench_cpu.py`.

Why this exists: `bench_cpu.py` sums per-child minima of a few reps. Because a child that finishes
early lets the others speed up, later reps are cheaper, and the sum over-counts. For liboqs
ML-DSA-44 the two harnesses disagreed by 4.8x, and this probe settled it:

    bench_cpu (sum of per-child minima)      949 k ops/s
    bench_mldsa_verify (pool wall / ops)     198 k ops/s
    this probe (barrier, verify only)        ~370 k ops/s

So every `16P` number produced by `bench_cpu.py` is an upper bound on the CPU, which makes every
`vs 16P` ratio a lower bound on the device's standing. `docs/benchmarks.md` §15 records that;
re-measuring the whole table with this method invalidates the snapshots' 16P columns together, so
it is a job for one session, not a half-done edit.

Usage:
    python benchmarks/probe_16p_method.py --tag ml_dsa_44 --n 256 [--procs 16]

It needs the liboqs build `bench_cpu.py` uses (`third_party/liboqs/build-all`), and it imports that
module's `LibOQS` rather than duplicating the ctypes bindings.
"""

from __future__ import annotations

import argparse
import contextlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_cpu import LIBOQS, LibOQS

# Each rival family is built into its own tree: `bench_cpu.py` uses `build-all` for ML-DSA/SLH-DSA
# and `build-falcon` for Falcon (a tag missing from a build makes the symbol lookup fail).
BUILD_FOR = {"falcon_512": "build-falcon", "falcon_padded_512": "build-falcon"}


def _worker(tag: str, so: str, n: int, barrier, out) -> None:
    """Generate and warm up outside the timed region, then verify one round in lockstep.

    A worker that raises must not take the barrier with it: the first version of this probe
    deadlocked instead of failing when the tag was missing from the build tree, because fifteen
    processes waited forever for a sixteenth that had already died.
    """
    try:
        impl = LibOQS(Path(so), tag)
        pk, corpus = impl.corpus(n)
        impl.verify_all(pk, corpus)  # warm-up, not timed
        barrier.wait(timeout=120)  # all processes start their timed round together
        t0 = time.perf_counter()
        ok = impl.verify_all(pk, corpus)
        out.put((ok, t0, time.perf_counter()))
    except Exception as exc:
        out.put(("error", type(exc).__name__, str(exc)[:120]))
        with contextlib.suppress(Exception):
            barrier.abort()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="ml_dsa_44", help="liboqs tag, e.g. ml_dsa_44, falcon_512")
    ap.add_argument("--n", type=int, default=256, help="signatures each process verifies")
    ap.add_argument("--procs", type=int, default=16)
    args = ap.parse_args()

    so = LIBOQS / BUILD_FOR.get(args.tag, "build-all") / "lib" / "liboqs.so"
    if not so.exists():
        print(f"error: {so} not built", file=sys.stderr)
        return 2

    ctx = mp.get_context("fork")
    queue, barrier = ctx.Queue(), ctx.Barrier(args.procs)
    procs = [
        ctx.Process(target=_worker, args=(args.tag, str(so), args.n, barrier, queue))
        for _ in range(args.procs)
    ]
    for p in procs:
        p.start()
    results = [queue.get() for _ in procs]
    for p in procs:
        p.join(timeout=10)

    errors = [r for r in results if r[0] == "error"]
    if errors:
        print(
            f"error: {len(errors)} worker(s) failed, first: {errors[0][1]}: {errors[0][2]}",
            file=sys.stderr,
        )
        return 2
    results = [r for r in results if r[0] != "error"]
    accepted = sum(ok for ok, _, _ in results)
    total = args.n * args.procs
    wall = max(end for _, _, end in results) - min(start for _, start, _ in results)
    print(f"{args.tag}: {args.procs} processes x {args.n} verifies, {accepted}/{total} accepted")
    print(f"  simultaneous wall : {wall * 1e3:7.2f} ms")
    print(f"  aggregate         : {total / wall:12.1f} ops/s")
    print(f"  per process       : {args.n / wall:12.1f} ops/s")
    return 0 if accepted == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
