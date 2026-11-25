#!/usr/bin/env python
"""SLH-DSA (FIPS 205) verification throughput: device-only vs end-to-end.

    python benchmarks/bench_slhdsa_verify.py --ps SLH-DSA-SHAKE-128f --batch 256,1024,4096 --cpu

The corpus is the valid signatures from the ACVP sigVer vectors (the `internal` interface, so the
per-signature cost does not depend on context/pre-hash formatting), tiled up to the requested
batch. Tiling keeps the distribution of WOTS digit values — and therefore the work per signature —
representative, which is worth stating because a signature's cost does vary with its digits.

`--cpu` adds the liboqs single-thread baseline on the same machine and the same signatures, so the
printed speed-up means something.
"""

from __future__ import annotations

import argparse
import ctypes
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_crypto.slhdsa import acvp
from triton_crypto.slhdsa.verify import Verifier

LIBS = (
    Path(__file__).resolve().parent.parent.parent / "third_party/liboqs/build-multi/lib/liboqs.so",
    Path("/home/lwzheng/文档/crypto-accl-work/third_party/liboqs/build-multi/lib/liboqs.so"),
)


def _corpus(ps: str):
    cases = [c for c in acvp.cases(ps=ps, interface="internal") if c.expected]
    if not cases:
        raise SystemExit(f"no valid ACVP signatures for {ps}")
    # A Verifier holds one public key (the batch-verification deployment model), so the corpus
    # must be the signatures under the same key.
    pk = cases[0].pk
    cases = [c for c in cases if c.pk == pk]
    msgs = [c.message for c in cases]
    sigs = np.stack([np.frombuffer(c.signature, dtype=np.uint8) for c in cases])
    return cases[0].pk, msgs, sigs


def _liboqs(ps: str):
    lib_path = next((p for p in LIBS if p.exists()), None)
    if lib_path is None:
        return None
    lib = ctypes.CDLL(str(lib_path))
    base = ps.replace("SLH-DSA-", "").lower().replace("-", "_")
    sym = f"OQS_SIG_slh_dsa_pure_{base}_verify"
    if not hasattr(lib, sym):
        return None
    fn = getattr(lib, sym)
    P = ctypes.POINTER(ctypes.c_ubyte)
    fn.argtypes = [P, ctypes.c_size_t, P, ctypes.c_size_t, P]
    fn.restype = ctypes.c_int
    to = lambda b: (ctypes.c_ubyte * max(1, len(b))).from_buffer_copy(b or b"\0")  # noqa: E731

    def verify(msg: bytes, sig: bytes, pk: bytes) -> bool:
        return fn(to(msg), len(msg), to(sig), len(sig), to(pk)) == 0

    return verify


def _cpu_baseline(pk, msgs, sigs, reps: int = 3, tile: int = 64) -> float | None:
    """Median-of-`reps` over a tile of `tile` signatures (a 2-signature sample is not a baseline)."""
    verify = _liboqs(_PS)
    if verify is None:
        return None
    n = min(tile, len(msgs))
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for m, s in zip(msgs[:n], sigs[:n], strict=True):
            verify(m, s.tobytes(), pk)
        times.append((time.perf_counter() - t0) / n * 1e6)
    return min(times)


_PS = "SLH-DSA-SHAKE-128f"


def main() -> int:
    global _PS
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", default=_PS)
    ap.add_argument("--batch", default="64,256,1024,4096")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="add the liboqs single-thread baseline")
    args = ap.parse_args()
    _PS = args.ps

    pk, msgs, sigs = _corpus(args.ps)
    print(f"# scheme : {args.ps}   corpus: {len(msgs)} ACVP signatures, tiled")
    verifier = Verifier(args.ps, pk)
    verifier.verify_mprime(msgs[:1], sigs[:1])  # warm-up / compile

    cpu = _cpu_baseline(pk, msgs, sigs) if args.cpu else None
    if cpu:
        print(f"# liboqs : {cpu:.1f} µs/sig single-thread")

    header = f"\n{'B':>7} {'device ms':>10} {'µs/sig':>9} {'k sig/s':>9} {'valid':>8}"
    print(header + (f" {'vs cpu':>8}" if cpu else ""))
    for B in [int(x) for x in args.batch.split(",")]:
        batch_msgs = [msgs[i % len(msgs)] for i in range(B)]
        batch_sigs = np.stack([sigs[i % len(sigs)] for i in range(B)])
        times = []
        for _ in range(args.reps):
            t0 = time.perf_counter()
            out = verifier.verify_mprime(batch_msgs, batch_sigs)
            times.append(time.perf_counter() - t0)
        t = min(times)
        line = f"{B:7d} {t * 1e3:10.2f} {t / B * 1e6:9.1f} {B / t / 1e3:9.1f} {int(out.sum()):4d}/{B:<3d}"
        if cpu:
            line += f" {cpu / (t / B * 1e6):7.1f}x"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
