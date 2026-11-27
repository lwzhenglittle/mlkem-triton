#!/usr/bin/env python
"""SLH-DSA (FIPS 205) verification throughput: device-only vs end-to-end.

    python benchmarks/bench_slhdsa_verify.py --ps SLH-DSA-SHAKE-128f --batch 256,1024,4096 --cpu

The corpus is the valid signatures from the ACVP sigVer vectors (the `internal` interface, so the
per-signature cost does not depend on context/pre-hash formatting), tiled up to the requested
batch. Tiling keeps the distribution of WOTS digit values — and therefore the work per signature —
representative, which is worth stating because a signature's cost does vary with its digits.

`--cpu` adds the liboqs single-thread baseline on the same machine and the same signatures, so the
printed speed-up means something. The library is found through `SLHDSA_LIBOQS_SO` (the variable
`tests/_liboqs_slh.py` reads) or through a vendored `third_party/liboqs/build*/`; when neither
exists `--cpu` says so on stderr instead of printing a table with the column quietly missing.
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_crypto.slhdsa import acvp
from triton_crypto.slhdsa.verify import Verifier

_HERE = Path(__file__).resolve()
_REPO = _HERE.parents[1]  # mlkem-triton/
_WORKSPACE = _REPO.parent  # the directory that vendors third_party/


def _liboqs_candidates() -> tuple[Path, ...]:
    """Where a liboqs build with SLH-DSA might be, most specific first.

    The environment override wins. The two vendored layouts cover "third_party beside the
    checkout" and "third_party inside it", because the two live in different places on different
    machines. `build*` is globbed rather than named: the directory is a local choice
    (`build-multi`, `build-all`), and a build without the SLH-DSA symbols is rejected by the
    symbol check in `_liboqs`, which is a better question than the directory name.
    """
    override = os.environ.get("SLHDSA_LIBOQS_SO")
    found = [Path(override)] if override else []
    for base in (_WORKSPACE / "third_party/liboqs", _REPO / "third_party/liboqs"):
        found.extend(sorted(base.glob("build*/lib/liboqs.so")))
    return tuple(found)


LIBS = _liboqs_candidates()


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
    """The liboqs SLH-DSA verify for `ps`, or `None` if no candidate build can do it.

    Candidates are tried in order and the first one that actually exports the symbol wins.
    `third_party/liboqs` on this machine holds several builds and only some of them were compiled
    with SLH-DSA, so "the file exists" and "this build can verify SLH-DSA" are different
    questions -- the symbol is the one worth asking about.
    """
    base = ps.replace("SLH-DSA-", "").lower().replace("-", "_")
    sym = f"OQS_SIG_slh_dsa_pure_{base}_verify"
    pointer = ctypes.POINTER(ctypes.c_ubyte)

    def to(b: bytes):
        return (ctypes.c_ubyte * max(1, len(b))).from_buffer_copy(b or b"\0")

    for lib_path in LIBS:
        if not lib_path.exists():
            continue
        try:
            lib = ctypes.CDLL(str(lib_path))
        except OSError:
            continue
        fn = getattr(lib, sym, None)
        if fn is None:
            continue
        fn.argtypes = [pointer, ctypes.c_size_t, pointer, ctypes.c_size_t, pointer]
        fn.restype = ctypes.c_int

        def verify(msg: bytes, sig: bytes, pk: bytes, _fn=fn) -> bool:
            return _fn(to(msg), len(msg), to(sig), len(sig), to(pk)) == 0

        return verify
    return None


def _worker(args):
    ps, pk, msgs, sigs = args
    verify = _liboqs(ps)
    if verify is None:
        return 0
    t0 = time.perf_counter()
    for m, s in zip(msgs, sigs, strict=True):
        verify(m, s.tobytes(), pk)
    return time.perf_counter() - t0


def _cpu_processes(pk, msgs, sigs, procs: int) -> float | None:
    """Wall-clock µs/signature with `procs` processes, each verifying a slice of the batch."""
    import concurrent.futures as cf

    from triton_crypto.slhdsa.ref import SLH  # noqa: F401  (ensures the oracle imports)

    if _liboqs(_PS) is None:
        return None
    n = len(msgs)
    cuts = [(i * n) // procs for i in range(procs + 1)]
    jobs = [
        (_PS, pk, msgs[cuts[i] : cuts[i + 1]], sigs[cuts[i] : cuts[i + 1]]) for i in range(procs)
    ]
    t0 = time.perf_counter()
    with cf.ProcessPoolExecutor(max_workers=procs) as ex:
        list(ex.map(_worker, jobs))
    return (time.perf_counter() - t0) / n * 1e6


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
    ap.add_argument(
        "--cpu-processes", type=int, default=0, help="also measure liboqs over N processes"
    )
    args = ap.parse_args()
    _PS = args.ps

    pk, msgs, sigs = _corpus(args.ps)
    print(f"# scheme : {args.ps}   corpus: {len(msgs)} ACVP signatures, tiled")
    verifier = Verifier(args.ps, pk)
    verifier.verify_mprime(msgs[:1], sigs[:1])  # warm-up / compile

    cpu = _cpu_baseline(pk, msgs, sigs) if args.cpu else None
    if cpu:
        print(f"# liboqs : {cpu:.1f} µs/sig single-thread")
    elif args.cpu:
        print(
            f"# liboqs : not found (set SLHDSA_LIBOQS_SO, or build third_party/liboqs); "
            f"tried {', '.join(str(p) for p in LIBS) or 'nothing'}",
            file=sys.stderr,
        )
    cpu_n = None
    if args.cpu_processes:
        big_msgs = [msgs[i % len(msgs)] for i in range(4096)]
        big_sigs = np.stack([sigs[i % len(sigs)] for i in range(4096)])
        cpu_n = _cpu_processes(pk, big_msgs, big_sigs, args.cpu_processes)
        if cpu_n:
            print(f"# liboqs : {cpu_n:.1f} µs/sig over {args.cpu_processes} processes (B=4096)")

    header = f"\n{'B':>7} {'device ms':>10} {'µs/sig':>9} {'k sig/s':>9} {'valid':>8}"
    if cpu:
        header += f" {'vs cpu1T':>9}"
    if cpu_n:
        header += f" {'vs cpuNP':>9}"
    print(header)
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
        if cpu_n:
            line += f" {cpu_n / (t / B * 1e6):7.1f}x"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
