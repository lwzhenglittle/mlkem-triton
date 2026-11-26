#!/usr/bin/env python
"""CPU rivals for the primitive table: the denominator every "x faster" claim needs.

`bench_primitive.py` answers "how far from the hardware ceiling is this primitive". It cannot
answer "is this worth doing", because the rival is not the copy engine: it is the 16 x86 cores
in the same box. This script measures that rival, once, and writes it to
`benchmarks/cpu_baseline.json` with the metadata needed to know when the number is stale.

    python benchmarks/bench_cpu.py                 # everything available, 1T + 16P
    python benchmarks/bench_cpu.py --only chacha20,poseidon2
    python benchmarks/bench_cpu.py --list
    python benchmarks/bench_primitive.py --cpu      # joins the two tables

Three sources, chosen because each is the fastest thing that exists for its family:

    openssl     ChaCha20, ChaCha20-Poly1305, AES-128-CTR, SHA-256, SHA3-256/512
                (AES-NI, SHA-NI and the vectorised Keccak in OpenSSL 3.5)
    liboqs      ML-DSA / Falcon / SLH-DSA verification, via ctypes (the same C the GPU code
                is cross-checked against, so the two numbers are one corpus apart)
    cargo       Poseidon2 through Plonky3's packed monty-31x8 (`benchmarks/poseidon2_ref`)

Conventions, the same ones `docs/benchmarks.md` §1 states: same process, min-of-N, and **16P
is 16 concurrent processes**, not 16 threads in one process -- this CPU is a 4P+8E+4LP hybrid
and threads do not scale.

One structural caveat worth keeping in view: a primitive that has *hardware* support on the
CPU (AES-NI: 172 GB/s over 16P) is a different proposition from one that does not (Keccak has
no instruction, so 16P is 4.4 GB/s). Which family a primitive falls into decides whether a
GPU port can win far more reliably than any property of the GPU.
"""

from __future__ import annotations

import argparse
import contextlib
import ctypes
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "benchmarks" / "cpu_baseline.json"
LIBOQS = ROOT / "third_party" / "liboqs"
REPS = 5
NPROC = 16


# --------------------------------------------------------------------------- metadata


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _scaling() -> dict:
    """The clock state, which is the single largest source of cross-session error here."""
    state = {}
    base = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    for key in ("scaling_governor", "scaling_cur_freq", "cpuinfo_max_freq", "cpuinfo_min_freq"):
        with contextlib.suppress(OSError):
            state[key] = (base / key).read_text().strip()
    return state


def _openssl_version() -> str:
    try:
        return subprocess.run(
            ["openssl", "version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        return "unavailable"


def metadata() -> dict:
    return {
        "host": platform.node(),
        "cpu": _cpu_model(),
        "nproc": os.cpu_count(),
        "scaling": _scaling(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "openssl": _openssl_version(),
        "liboqs_builds": [
            str(p.relative_to(ROOT)) for p in sorted(LIBOQS.glob("build*/lib/liboqs.so"))
        ],
        "poseidon2_ref": str(
            (
                ROOT / "benchmarks" / "poseidon2_ref" / "target" / "release" / "poseidon2-ref"
            ).relative_to(ROOT)
        ),
    }


# --------------------------------------------------------------------------- openssl


def _openssl_rate(evp: str, seconds: int) -> float:
    """Bytes/s from `openssl speed`, at the largest block size it reports."""
    out = subprocess.run(
        ["openssl", "speed", "-evp", evp, "-seconds", str(seconds), "-bytes", "16384"],
        capture_output=True,
        text=True,
        timeout=seconds + 60,
    ).stdout
    rows = [ln for ln in out.splitlines() if ln.split() and ln.split()[0].lower() == evp.lower()]
    if not rows:
        raise RuntimeError(f"openssl speed {evp}: no result line")
    last = rows[-1]
    # "ChaCha20    627996.74k" -- thousands of bytes per second
    value = last.split()[-1]
    scale = 1e3 if value.endswith("k") else 1e6
    return float(value.rstrip("k")) * scale


def _parallel_rate(fn, args_list, nproc: int = NPROC) -> float:
    """Aggregate of NPROC children, each reporting its *own* measured rate.

    Summing per-child rates (rather than bytes divided by a wall clock that includes process
    start-up) is what makes 16P mean "sixteen processes each running at their own rate while
    the other fifteen are running too".
    """
    ctx = mp.get_context("fork")
    with ctx.Pool(nproc) as pool:
        rates = pool.map(fn, args_list)
    return float(sum(rates))


def _openssl_child(args) -> float:
    evp, seconds = args
    return min(_openssl_rate(evp, seconds) for _ in range(2))


def openssl_pair(evp: str, seconds: int = 1) -> tuple[float, float]:
    one = min(_openssl_rate(evp, seconds) for _ in range(REPS))
    many = _parallel_rate(_openssl_child, [(evp, seconds)] * NPROC)
    return one, many


# --------------------------------------------------------------------------- liboqs


U8P = ctypes.POINTER(ctypes.c_ubyte)


class LibOQS:
    """The four signatures, bound through ctypes, capacities read back from `sign`."""

    def __init__(self, so: Path, tag: str):
        self.lib = ctypes.CDLL(str(so))
        self.tag = tag
        self.keypair = self.lib[f"OQS_SIG_{tag}_keypair"]
        self.keypair.argtypes = [U8P, U8P]
        self.sign = self.lib[f"OQS_SIG_{tag}_sign"]
        self.sign.argtypes = [U8P, ctypes.POINTER(ctypes.c_size_t), U8P, ctypes.c_size_t, U8P]
        self.verify = self.lib[f"OQS_SIG_{tag}_verify"]
        self.verify.argtypes = [U8P, ctypes.c_size_t, U8P, ctypes.c_size_t, U8P]

    def corpus(self, n: int = 32):
        cap = 65536  # SLH-DSA-SHA2-128f signatures are 17 088 bytes
        pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
        if self.keypair(pk, sk) != 0:
            raise RuntimeError(f"{self.tag}: keypair failed")
        msgs = []
        for i in range(n):
            msg = b"k" * 28 + i.to_bytes(4, "little")
            mb = (ctypes.c_ubyte * len(msg)).from_buffer_copy(msg)
            sig = (ctypes.c_ubyte * cap)()
            siglen = ctypes.c_size_t(cap)
            if self.sign(sig, ctypes.byref(siglen), mb, len(msg), sk) != 0:
                raise RuntimeError(f"{self.tag}: sign failed")
            msgs.append(
                (mb, (ctypes.c_ubyte * siglen.value).from_buffer_copy(bytes(sig[: siglen.value])))
            )
        pkbuf = (ctypes.c_ubyte * cap).from_buffer_copy(bytes(pk))
        return pkbuf, msgs

    def verify_all(self, pk, corpus) -> int:
        ok = 0
        for mb, sb in corpus:
            ok += self.verify(mb, len(mb), sb, len(sb), pk) == 0
        return ok


def _lib_child(args) -> float:
    so, tag, n = args
    o = LibOQS(Path(so), tag)
    pk, corpus = o.corpus(n)
    assert o.verify_all(pk, corpus) == n
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        o.verify_all(pk, corpus)
        times.append(time.perf_counter() - t0)
    return n / min(times)


def _corpus_size(tag: str) -> int:
    """Enough messages to amortise the per-call ctypes overhead, few enough to set up fast."""
    if "slh_dsa" in tag:
        return 4
    if "falcon" in tag:
        return 64
    return 256


def liboqs_pair(so: Path, tag: str, n: int | None = None) -> tuple[float, float]:
    n = _corpus_size(tag) if n is None else n
    # Each measurement is its own address space. Measuring two Falcon parameter sets in one
    # process corrupted the heap (malloc(): invalid size) -- and a crash inside a rival must
    # not take the rest of the table with it.
    one = _parallel_rate(_lib_child, [(str(so), tag, n)], nproc=1)
    many = _parallel_rate(_lib_child, [(str(so), tag, n)] * NPROC)
    return one, many


# --------------------------------------------------------------------------- poseidon2


def _poseidon_exe() -> Path:
    exe = ROOT / "benchmarks" / "poseidon2_ref" / "target" / "release" / "poseidon2-ref"
    if not exe.exists():
        raise RuntimeError("benchmarks/poseidon2_ref is not built (see its Cargo.toml)")
    return exe


def _poseidon_rate(exe: str, n: int) -> float:
    best = 0.0
    for _ in range(3):
        out = subprocess.run([exe, "bench-packed", str(n)], capture_output=True, text=True)
        best = max(best, json.loads(out.stdout)["mperm_per_s"] * 1e6)
    return best


def _poseidon_child(args) -> float:
    exe, n = args
    return _poseidon_rate(exe, n)


def poseidon2_pair(n: int = 1_000_000) -> tuple[float, float]:
    exe = _poseidon_exe()
    one = _poseidon_rate(str(exe), n)
    many = _parallel_rate(_poseidon_child, [(str(exe), n // 2)] * NPROC)
    return one, many


# --------------------------------------------------------------------------- table


@dataclass
class Rival:
    """One CPU rival: the unit its rate is in, the runner, and which bench row it answers."""

    unit: str  # "B/s" or "op/s"
    runner: Callable[[], tuple[float, float]]
    note: str
    applies_to: tuple[str, ...]  # primitive names in bench_primitive.py


def _openssl_evp(evp: str):
    return lambda: openssl_pair(evp)


CASES: dict[str, Rival] = {
    "chacha20": Rival(
        "B/s", _openssl_evp("chacha20"), "openssl speed -evp chacha20", ("chacha20",)
    ),
    "chacha20-poly1305": Rival(
        "B/s", _openssl_evp("chacha20-poly1305"), "openssl speed -evp chacha20-poly1305", ()
    ),
    "aes-128-ctr": Rival("B/s", _openssl_evp("aes-128-ctr"), "openssl speed -evp aes-128-ctr", ()),
    "sha256": Rival("B/s", _openssl_evp("sha256"), "openssl speed -evp sha256", ("sha256",)),
    "sha512": Rival("B/s", _openssl_evp("sha512"), "openssl speed -evp sha512", ("sha512",)),
    "sha3-256": Rival(
        "B/s", _openssl_evp("sha3-256"), "openssl speed -evp sha3-256", ("sha3-256",)
    ),
    "sha3-512": Rival(
        "B/s", _openssl_evp("sha3-512"), "openssl speed -evp sha3-512", ("sha3-512",)
    ),
    "poseidon2": Rival(
        "op/s", poseidon2_pair, "Plonky3 0.8 packed monty-31x8 (cargo ref)", ("poseidon2",)
    ),
    "mldsa-44-verify": Rival(
        "op/s",
        lambda: liboqs_pair(LIBOQS / "build-all" / "lib" / "liboqs.so", "ml_dsa_44"),
        "liboqs 0.16.0 ML-DSA-44 verify",
        (),
    ),
    "falcon-512-verify": Rival(
        "op/s",
        lambda: liboqs_pair(LIBOQS / "build-falcon" / "lib" / "liboqs.so", "falcon_512"),
        "liboqs 0.16.0 Falcon-512 verify",
        (),
    ),
    "falcon-padded-512-verify": Rival(
        "op/s",
        lambda: liboqs_pair(LIBOQS / "build-falcon" / "lib" / "liboqs.so", "falcon_padded_512"),
        "liboqs 0.16.0 Falcon-padded-512 verify",
        (),
    ),
    "slhdsa-128f-verify": Rival(
        "op/s",
        lambda: liboqs_pair(LIBOQS / "build-all" / "lib" / "liboqs.so", "slh_dsa_pure_sha2_128f"),
        "liboqs 0.16.0 SLH-DSA-SHA2-128f verify",
        ("slhdsa-128f",),
    ),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma-separated rivals")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--seconds", type=int, default=1, help="openssl window per pass (default 1)")
    ap.add_argument(
        "--merge",
        action="store_true",
        help="update the rivals present in --out instead of replacing the file",
    )
    args = ap.parse_args()

    if args.list:
        for name, r in CASES.items():
            print(f"  {name:28s} {r.unit:5s} {r.note}")
        return 0

    names = list(CASES) if args.only is None else [n.strip() for n in args.only.split(",")]
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"error: unknown rival(s) {unknown}; try --list", file=sys.stderr)
        return 2

    out_path = Path(args.out)
    if args.merge and out_path.exists():
        results = json.loads(out_path.read_text())
        results["meta"] = metadata()  # the fresh window wins
    else:
        results = {"meta": metadata(), "rivals": {}}
    print(f"# CPU baseline: {results['meta']['cpu']}  nproc={results['meta']['nproc']}")
    print(f"# 16P = {NPROC} concurrent processes; min-of-{REPS}; {results['meta']['timestamp']}")
    print(f"\n{'rival':28s} {'unit':6s} {'1T':>12s} {'16P':>12s} {'scale':>7s}  note")
    for name in names:
        rival = CASES[name]
        try:
            one, many = rival.runner()
        except Exception as exc:
            print(f"{name:28s} SKIPPED {type(exc).__name__}: {str(exc)[:60]}")
            continue
        results["rivals"][name] = {
            "unit": rival.unit,
            "one_thread": one,
            "procs16": many,
            "note": rival.note,
        }
        print(
            f"{name:28s} {rival.unit:6s} {one:12.4g} {many:12.4g} {many / one:6.2f}x  {rival.note}"
        )

    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("fork")
    sys.exit(main())
