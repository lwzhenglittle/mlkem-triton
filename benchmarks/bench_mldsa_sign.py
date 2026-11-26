"""ML-DSA signing throughput on the device.

Signing is not a fixed amount of work per signature: the rejection loop retries (usually 1–5
times), so the number of attempts — and therefore the device time — varies with the key and the
messages. The benchmark reports wall-clock per signature over a batch of independent messages,
which is the number that matters when a batch is signed together.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))

from _liboqs import LibOQS

from triton_crypto.mldsa.params import PARAMS
from triton_crypto.mldsa.ref import MLDSA
from triton_crypto.mldsa.sign import Signer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ps", nargs="+", default=["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--cpu", action="store_true", help="also time liboqs (1 thread and 16 processes)"
    )
    args = ap.parse_args()

    for ps in args.ps:
        if ps not in PARAMS:
            raise KeyError(ps)
        m = MLDSA(ps)
        _pk, sk = m.keygen(bytes(range(32)))
        signer = Signer(ps, sk)
        print(f"# {ps}: signature {m.p.sig_size} bytes")
        for b in args.batch:
            msgs = [b"message %d" % i for i in range(b)]
            signer.sign(msgs[:1])  # warm up (compiles the kernels)
            best = float("inf")
            for _ in range(args.reps):
                import time

                t0 = time.perf_counter()
                sigs = signer.sign(msgs)
                best = min(best, time.perf_counter() - t0)
                assert sigs.shape[0] == b
            line = f"  B={b:5d}  {best * 1e3:8.1f} ms  {best / b * 1e6:8.1f} µs/sig"
            if args.cpu:
                o = LibOQS(ps)
                _pk_c, sk_c = o.keypair()
                o.sign(msgs[0], sk_c)  # warm up
                n = min(b, 256)
                t0 = time.perf_counter()
                for i in range(n):
                    o.sign(msgs[i], sk_c)
                cpu1 = (time.perf_counter() - t0) / n * 1e6
                line += f"   liboqs 1T {cpu1:7.1f} µs"
            print(line)
    return 0


def _cpu_16(ps: str, msgs, sk: bytes, n_proc: int = 16) -> float:
    """Aggregate signing throughput over 16 concurrent processes."""
    import multiprocessing as mp

    global _WORK
    _WORK = (ps, msgs, sk)
    ctx = mp.get_context("fork")
    with ctx.Pool(n_proc) as pool:
        pool.map(_proc_worker, range(n_proc))
        t0 = time.perf_counter()
        pool.map(_proc_worker, range(n_proc))
        wall = time.perf_counter() - t0
    return wall / (len(msgs) * n_proc)


_WORK = None


def _proc_worker(_):
    ps, msgs, sk = _WORK
    o = LibOQS(ps)
    for m in msgs:
        o.sign(m, sk)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
