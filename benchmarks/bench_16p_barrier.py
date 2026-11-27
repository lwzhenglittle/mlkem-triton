"""Re-measure the CPU rivals with a barrier, one round each, at the *same shape* the device runs.

Two things are wrong with the `16P` column `bench_cpu.py` produces, and they are different bugs:

1. **The aggregation over-counts.** `_parallel_rate` sums each child's *minimum* over its own
   reps, from a pool whose children start and finish at different times. A child that finishes
   early frees a core and lets the survivors run faster, so their minima are taken under lighter
   load than they were actually measured at. Measured disagreement on liboqs ML-DSA-44: 949 k
   ops/s (pool, sum of minima) against ~370 k (this barrier). Every `vs 16P` ratio built on the
   old column is therefore a lower bound for the device.

2. **Some rows compare different message sizes.** The device hashes `--size`-byte messages
   (1024 by default, see `bench_primitive.py`), while the CPU rival was pointed at
   `openssl speed -bytes 16384` for the digests and at 1 MiB messages for BLAKE3 and Poly1305.
   Per-message init/final overhead is then charged to the device and not to the CPU. For SHA-256
   at 1 KiB the one-shot EVP path is 0.84 GB/s per core against roughly 2 GB/s at 16 KiB, so the
   size mismatch alone is worth about 2.4x.

`probe_16p_method.py` settled (1) for liboqs and left the rest of the table alone. This script
does every family, so the whole column can be replaced in one window instead of row by row.

Method, per family: build the payload **outside** the timed region, warm up, wait on a shared
barrier, then run exactly one round whose *unit count* is fixed by the caller (bytes, messages,
permutations, verifications). Aggregate = sum(units) / (last_end - first_start). Neither number
is a minimum over reps, and the window is measured around the work, so process start-up is not
charged to the CPU.

Usage:
    python benchmarks/bench_16p_barrier.py --all [--procs 16] [--reps 3] [--json out.json]
    python benchmarks/bench_16p_barrier.py --only sha256 --procs 16

The device side is not measured here; `bench_primitive.py` does that. `--compare` joins this
script's output with a `bench_primitive` JSON dump and prints the old and new ratios side by side.
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
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from bench_cpu import LIBOQS, LibOQS

U8P = ctypes.POINTER(ctypes.c_ubyte)


def _resolve_oqs_so(build: str = "build-all") -> Path:
    """Prefer `OQS_BENCH_SO`, then the unified `build-bench`, then the family-specific build."""
    override = os.environ.get("OQS_BENCH_SO")
    if override and Path(override).exists():
        return Path(override)
    for cand in (
        LIBOQS / "build-bench" / "lib" / "liboqs.so",
        LIBOQS / build / "lib" / "liboqs.so",
    ):
        if cand.exists():
            return cand
    return LIBOQS / build / "lib" / "liboqs.so"


class LibOQSKEM:
    """ML-KEM keypair / encaps / decaps bound through ctypes for barrier benchmarking."""

    def __init__(self, so: Path, tag: str):
        self.lib = ctypes.CDLL(str(so))
        self.tag = tag
        self.keypair = self.lib[f"OQS_KEM_{tag}_keypair"]
        self.keypair.argtypes = [U8P, U8P]
        self.keypair.restype = ctypes.c_int
        self.encaps = self.lib[f"OQS_KEM_{tag}_encaps"]
        self.encaps.argtypes = [U8P, U8P, U8P]
        self.encaps.restype = ctypes.c_int
        self.decaps = self.lib[f"OQS_KEM_{tag}_decaps"]
        self.decaps.argtypes = [U8P, U8P, U8P]
        self.decaps.restype = ctypes.c_int

    def prepare(self):
        cap = 4096
        pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
        ct, ss = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * 64)()
        if self.keypair(pk, sk) != 0:
            raise RuntimeError(f"{self.tag}: keypair failed")
        if self.encaps(ct, ss, pk) != 0:
            raise RuntimeError(f"{self.tag}: encaps failed")
        return pk, sk, ct, ss

    def run_keygen(self, n: int) -> int:
        cap = 4096
        pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
        bad = 0
        for _ in range(n):
            bad += self.keypair(pk, sk) != 0
        return bad

    def run_encaps(self, pk, n: int) -> int:
        cap = 4096
        ct, ss = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * 64)()
        bad = 0
        for _ in range(n):
            bad += self.encaps(ct, ss, pk) != 0
        return bad

    def run_decaps(self, sk, ct, ss_ref, n: int) -> int:
        ss = (ctypes.c_ubyte * 64)()
        ref32 = bytes(ss_ref[:32])
        bad = 0
        for _ in range(n):
            rc = self.decaps(ss, ct, sk)
            bad += rc != 0
        if bytes(ss[:32]) != ref32:
            bad += 1
        return bad


# Families whose rival is a compiled binary the repo already builds for `bench_cpu.py`.
BIN = {
    "evp": ROOT / "evp_cpu",
    "blake3": ROOT / "blake3_ref" / "target" / "release" / "blake3-ref",
    "poseidon2": ROOT / "poseidon2_ref" / "target" / "release" / "poseidon2-ref",
    "ntt": ROOT / "ntt_cpu",
}

# `bench_primitive.py` hashes 1024-byte messages by default, so the CPU rival must too. The
# ciphers are stream modes with no per-message cost, and the device feeds them `--size`-byte
# buffers in one launch, so 16 KiB there measures the same thing and keeps the AVX path honest.
MSG = 1024
CIPHER_MSG = 16384

# Byte targets are what make the round self-sizing: every child does the same *work*, not the
# same *duration*, so a slow child cannot finish early and leave the barrier group.
DIGEST_BYTES = 1 << 30
CIPHER_BYTES = 4 << 30
MAC_BYTES = 1 << 30


# --------------------------------------------------------------------------- rivals


def power_state() -> dict:
    """Battery, platform profile and clock, because on this box they multiply every number.

    Measured the hard way: a snapshot taken while the machine had dropped to battery power read
    every CPU rate 4.6x low (0.9 GHz against 3.4), and the `scale` column came out at 16.8x on
    sixteen threads -- physically impossible, which is what gave it away. The iGPU was being
    power-capped at the same time, which is also the most likely cause of the
    `UR_RESULT_ERROR_DEVICE_LOST` that killed a test run.
    """
    state: dict = {}
    for path in sorted(Path("/sys/class/power_supply").glob("*/status")):
        with contextlib.suppress(OSError):
            state["battery"] = path.read_text().strip()
    with contextlib.suppress(OSError):
        state["platform_profile"] = Path("/sys/firmware/acpi/platform_profile").read_text().strip()
    freqs = []
    for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_cur_freq"):
        with contextlib.suppress(OSError):
            freqs.append(int(path.read_text()))
    if freqs:
        freqs.sort()
        state["cur_freq_mhz_median"] = round(freqs[len(freqs) // 2] / 1000)
        state["cur_freq_mhz_max"] = round(freqs[-1] / 1000)
    return state


def state_warning(state: dict) -> str | None:
    """One line when the machine is not in a state where the numbers mean anything.

    The power *source* is deliberately not a trigger. Measured: this laptop reports
    `Discharging` at 99% with the charger unplugged, yet with `powerprofilesctl set performance`
    it holds 4.7 GHz and reproduces the stored baseline to within 3% -- so what matters is the
    clock and the profile, which are visible directly. Battery state stays in `meta` as context.
    """
    problems = []
    if state.get("profile") == "low-power":
        problems.append("ACPI profile is low-power")
    if state.get("platform_profile") == "low-power":
        problems.append("ACPI profile is low-power")
    if state.get("cur_freq_mhz_max", 10_000) < 2000:
        problems.append(f"CPU tops out at {state.get('cur_freq_mhz_max')} MHz")
    if not problems:
        return None
    return (
        "machine state is not comparable: "
        + ", ".join(problems)
        + " -- every rate here is low by 2-5x and `scale` can exceed the core count"
    )


def cpu_meta(procs: int, msgsize: int, reps: int) -> dict:
    """The provenance block `bench_primitive.load_cpu` checks before it will use a baseline.

    `method` is the whole point: a file that does not say how `procs16` was aggregated is how the
    sum-of-minima column survived as long as it did.
    """
    model = "unknown"
    with contextlib.suppress(OSError):
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                model = line.split(":", 1)[1].strip()
                break
    return {
        "host": os.uname().nodename,
        "cpu": model,
        "nproc": os.cpu_count(),
        "method": "barrier (one timed round per process, sum of units over the union window)",
        "procs": procs,
        "msgsize": msgsize,
        "reps": reps,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "power": power_state(),
    }


def _prepare(family: str, cfg: dict):
    """Everything expensive that is not the measured work: key generation and a signature corpus.

    Without this, `_round` for liboqs would generate its own corpus inside the timed window --
    twenty thousand ML-DSA signatures at ~80 us each, forty times the verify round it is supposed
    to be measuring.
    """
    kind = cfg["kind"]
    if kind == "oqs":
        so = _resolve_oqs_so(cfg.get("build", "build-all"))
        impl = LibOQS(so, cfg["tag"])
        pk, corpus = impl.corpus(cfg["n"])
        return (impl, pk, corpus)
    if kind == "oqs-sign":
        so = _resolve_oqs_so(cfg.get("build", "build-all"))
        impl = LibOQS(so, cfg["tag"])
        cap = 65536
        pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
        if impl.keypair(pk, sk) != 0:
            raise RuntimeError(f"{cfg['tag']}: keypair failed")
        msgs = []
        for i in range(cfg["n"]):
            msg = b"s" * 28 + i.to_bytes(4, "little")
            msgs.append((ctypes.c_ubyte * len(msg)).from_buffer_copy(msg))
        return (impl, sk, msgs, cap)
    if kind.startswith("oqs-kem-"):
        so = _resolve_oqs_so(cfg.get("build", "build-all"))
        impl = LibOQSKEM(so, cfg["tag"])
        return (impl, *impl.prepare())
    if kind == "torch-gemm-exact":
        import numpy as np
        import torch

        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)
        dim = cfg["dim"]
        rng = np.random.default_rng(0)
        a = torch.from_numpy(rng.integers(-128, 128, (dim, dim), dtype=np.int8).astype(np.float32))
        b = torch.from_numpy(rng.integers(-128, 128, (dim, dim), dtype=np.int8).astype(np.float32))
        return (torch, a, b)
    return None


def _round(family: str, cfg: dict, state=None) -> tuple[float, int]:
    """Do one round of `family` work and return `(units, ok)`; units in bytes, ops or messages."""

    kind = cfg["kind"]
    if kind == "evp":
        # C driver, not ctypes: three FFI calls per message cost more than the hash at 1 KiB.
        total = cfg.get("bytes", DIGEST_BYTES)
        out = subprocess.run(
            [str(BIN["evp"]), cfg["evp"], str(cfg["size"]), str(total)],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(out.stdout)
        return float(data["bytes"]), 0

    if kind == "oqs":
        impl, pk, corpus = state
        accepted = impl.verify_all(pk, corpus)
        # the second value is a *failure count*, not a success flag: a round that verified 19 999
        # of 20 000 must fail the run rather than pass on "nonzero"
        return float(cfg["n"]), int(cfg["n"]) - accepted

    if kind == "oqs-sign":
        impl, sk, msgs, cap = state
        sig = (ctypes.c_ubyte * cap)()
        bad = 0
        for mb in msgs:
            siglen = ctypes.c_size_t(cap)
            bad += impl.sign(sig, ctypes.byref(siglen), mb, len(mb), sk) != 0
        return float(len(msgs)), bad

    if kind == "oqs-kem-keygen":
        impl = state[0]
        return float(cfg["n"]), impl.run_keygen(cfg["n"])

    if kind == "oqs-kem-encaps":
        impl, pk = state[0], state[1]
        return float(cfg["n"]), impl.run_encaps(pk, cfg["n"])

    if kind == "oqs-kem-decaps":
        impl, _pk, sk, ct, ss_ref = state
        return float(cfg["n"]), impl.run_decaps(sk, ct, ss_ref, cfg["n"])

    if kind == "torch-gemm-exact":
        torch, a, b = state
        dim = cfg["dim"]
        reps = cfg["reps"]
        for _ in range(reps):
            _ = torch.mm(a, b)
        return float(reps * dim * dim * dim), 0

    if kind == "bin-ntt":
        exe = BIN["ntt"]
        out = subprocess.run(
            [str(exe), str(cfg["logn"]), str(cfg["reps"])],
            capture_output=True,
            text=True,
            env={**os.environ, "OMP_NUM_THREADS": "1"},
            check=True,
        )
        return float(cfg["reps"]), 0

    if kind == "bin-blake3":
        exe = BIN["blake3"]
        out = subprocess.run(
            [str(exe), "bench-total", str(cfg["size"]), str(cfg["reps"])],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(out.stdout)
        return float(data["units"]), 0

    if kind == "bin-poseidon":
        exe = BIN["poseidon2"]
        out = subprocess.run(
            [str(exe), "bench-packed-total", str(cfg["n"])],
            capture_output=True,
            text=True,
            check=True,
        )
        data = json.loads(out.stdout)
        if data["lanes"] < 4:
            raise RuntimeError(
                "poseidon rival is on the scalar path (rebuild with -C target-cpu=native)"
            )
        return float(data["units"]), 0

    raise ValueError(f"unknown family kind {kind!r}")


def _worker(family: str, cfg: dict, barrier, out) -> None:
    """Warm up outside the window, then one timed round; failures must not strand the barrier."""
    try:
        state = _prepare(family, cfg)
        _round(family, cfg, state)  # warm-up: page faults, JIT, allocation, frequency ramp
        barrier.wait(timeout=300)
        t0 = time.perf_counter()
        units, ok = _round(family, cfg, state)
        t1 = time.perf_counter()
        out.put((units, ok, t0, t1))
    except Exception as exc:
        out.put(("error", type(exc).__name__, str(exc)[:200]))
        with contextlib.suppress(Exception):
            barrier.abort()


def measure(family: str, cfg: dict, procs: int) -> dict:
    """One aggregate for `procs` simultaneous rounds, two ways.

    `batch` is total units over the union window: what the CPU finishes when a fixed pile of
    work is split `procs` ways and joined. `steady` sums each child's own unit rate: what the CPU
    sustains per second. They differ when the cores are heterogeneous (4P+8E+4LP here), and the
    device rows are batch throughputs, so `batch` is the like-for-like figure and the one
    quoted. `steady` is kept because the gap between them is itself the finding about this part.
    """
    ctx = mp.get_context("fork")
    queue, barrier = ctx.Queue(), ctx.Barrier(procs)
    ps = [ctx.Process(target=_worker, args=(family, cfg, barrier, queue)) for _ in range(procs)]
    for p in ps:
        p.start()
    results = [queue.get(timeout=600) for _ in ps]
    for p in ps:
        p.join(timeout=30)

    errors = [r for r in results if r[0] == "error"]
    if errors:
        raise RuntimeError(f"{len(errors)} worker(s) failed: {errors[0][1]}: {errors[0][2]}")
    units = sum(r[0] for r in results)
    bad = sum(r[1] for r in results)
    if bad:
        raise RuntimeError(f"{family}: {bad} wrong result(s) across {len(results)} round(s)")

    # The window is the union of the children's own timed regions, so neither process start-up
    # nor the warm-up lands in it. Using `max(end) - min(start)` (not the mean) is the
    # conservative choice: the aggregate is whichever the straggler allows.
    span = max(r[3] for r in results) - min(r[2] for r in results)
    steady = sum(r[0] / (r[3] - r[2]) for r in results)
    return {"batch": units / span, "steady": steady, "units": units, "span": span}


# --------------------------------------------------------------------------- families


def families(msgsize: int = MSG) -> dict[str, dict]:
    """Every rival with a `16P` column, at the message size the device row actually uses."""
    return {
        "sha3-256": {"kind": "evp", "evp": "sha3-256", "size": msgsize, "unit": "B/s"},
        "sha3-512": {"kind": "evp", "evp": "sha3-512", "size": msgsize, "unit": "B/s"},
        "shake128": {"kind": "evp", "evp": "shake128", "size": msgsize, "unit": "B/s"},
        "shake256": {"kind": "evp", "evp": "shake256", "size": msgsize, "unit": "B/s"},
        "sha256": {"kind": "evp", "evp": "sha256", "size": msgsize, "unit": "B/s"},
        "sha512": {"kind": "evp", "evp": "sha512", "size": msgsize, "unit": "B/s"},
        "aes-128-ctr": {
            "kind": "evp",
            "evp": "aes-128-ctr",
            "size": CIPHER_MSG,
            "bytes": CIPHER_BYTES,
            "unit": "B/s",
        },
        "chacha20": {
            "kind": "evp",
            "evp": "chacha20",
            "size": CIPHER_MSG,
            "bytes": CIPHER_BYTES,
            "unit": "B/s",
        },
        "poly1305": {
            "kind": "evp",
            "evp": "poly1305",
            "size": msgsize,
            "bytes": MAC_BYTES,
            "unit": "B/s",
        },
        "chacha20-poly1305": {
            "kind": "evp",
            "evp": "chacha20-poly1305",
            "size": msgsize,
            "bytes": MAC_BYTES,
            "unit": "B/s",
        },
        "blake3": {"kind": "bin-blake3", "size": msgsize, "reps": 650_000, "unit": "B/s"},
        "poseidon2": {"kind": "bin-poseidon", "n": 1_000_000, "unit": "op/s"},
        "ntt-2^20": {"kind": "bin-ntt", "logn": 20, "reps": 8, "unit": "op/s"},
        "gemm-int8": {
            "kind": "torch-gemm-exact",
            "dim": 1024,
            "reps": 24,
            "unit": "MAC/s",
        },
        # ML-KEM (FIPS 203): all three parameter sets x three operations
        "mlkem-512-keygen": {
            "kind": "oqs-kem-keygen",
            "tag": "ml_kem_512",
            "n": 40_000,
            "unit": "op/s",
        },
        "mlkem-512-encaps": {
            "kind": "oqs-kem-encaps",
            "tag": "ml_kem_512",
            "n": 40_000,
            "unit": "op/s",
        },
        "mlkem-512-decaps": {
            "kind": "oqs-kem-decaps",
            "tag": "ml_kem_512",
            "n": 40_000,
            "unit": "op/s",
        },
        "mlkem-768-keygen": {
            "kind": "oqs-kem-keygen",
            "tag": "ml_kem_768",
            "n": 30_000,
            "unit": "op/s",
        },
        "mlkem-768-encaps": {
            "kind": "oqs-kem-encaps",
            "tag": "ml_kem_768",
            "n": 30_000,
            "unit": "op/s",
        },
        "mlkem-768-decaps": {
            "kind": "oqs-kem-decaps",
            "tag": "ml_kem_768",
            "n": 30_000,
            "unit": "op/s",
        },
        "mlkem-1024-keygen": {
            "kind": "oqs-kem-keygen",
            "tag": "ml_kem_1024",
            "n": 20_000,
            "unit": "op/s",
        },
        "mlkem-1024-encaps": {
            "kind": "oqs-kem-encaps",
            "tag": "ml_kem_1024",
            "n": 20_000,
            "unit": "op/s",
        },
        "mlkem-1024-decaps": {
            "kind": "oqs-kem-decaps",
            "tag": "ml_kem_1024",
            "n": 20_000,
            "unit": "op/s",
        },
        # ML-DSA (FIPS 204): all three parameter sets x verify + sign
        "mldsa-44-verify": {"kind": "oqs", "tag": "ml_dsa_44", "n": 20_000, "unit": "op/s"},
        "mldsa-65-verify": {"kind": "oqs", "tag": "ml_dsa_65", "n": 12_000, "unit": "op/s"},
        "mldsa-87-verify": {"kind": "oqs", "tag": "ml_dsa_87", "n": 8_000, "unit": "op/s"},
        "mldsa-44-sign": {"kind": "oqs-sign", "tag": "ml_dsa_44", "n": 6_000, "unit": "op/s"},
        "mldsa-65-sign": {"kind": "oqs-sign", "tag": "ml_dsa_65", "n": 4_000, "unit": "op/s"},
        "mldsa-87-sign": {"kind": "oqs-sign", "tag": "ml_dsa_87", "n": 3_000, "unit": "op/s"},
        # Falcon (FIPS 206 draft)
        "falcon-512-verify": {
            "kind": "oqs",
            "tag": "falcon_512",
            "n": 8_000,
            "build": "build-falcon",
            "unit": "op/s",
        },
        "falcon-padded-512-verify": {
            "kind": "oqs",
            "tag": "falcon_padded_512",
            "n": 8_000,
            "build": "build-falcon",
            "unit": "op/s",
        },
        # SLH-DSA (FIPS 205): separate SHA2 and SHAKE families so GPU vs CPU stays like-for-like.
        # `slhdsa-128f-verify` is kept as the legacy SHA2-128f alias for older comparison scripts.
        "slhdsa-128f-verify": {
            "kind": "oqs",
            "tag": "slh_dsa_pure_sha2_128f",
            "n": 350,
            "unit": "op/s",
        },
        "slhdsa-sha2-128f-verify": {
            "kind": "oqs",
            "tag": "slh_dsa_pure_sha2_128f",
            "n": 350,
            "unit": "op/s",
        },
        "slhdsa-shake-128f-verify": {
            "kind": "oqs",
            "tag": "slh_dsa_pure_shake_128f",
            "n": 350,
            "unit": "op/s",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="every family")
    ap.add_argument("--only", action="append", default=[], help="family name, repeatable")
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--reps", type=int, default=3, help="rounds per cell; the best 16P is kept")
    ap.add_argument("--msgsize", type=int, default=MSG)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument(
        "--force",
        action="store_true",
        help="write --json even when the machine is on battery / in low-power mode",
    )
    args = ap.parse_args()

    table = families(args.msgsize)
    names = list(table) if args.all else args.only
    if not names:
        ap.error("give --all or --only FAMILY")
    unknown = [n for n in names if n not in table]
    if unknown:
        ap.error(f"unknown: {unknown}; have {sorted(table)}")

    state = power_state()
    warning = state_warning(state)
    if warning:
        print(f"# WARNING: {warning}", file=sys.stderr)

    rows = {}
    print(f"{'rival':26s} {'unit':5s} {'1T':>14s} {'16P':>14s} {'scale':>7s}  {'sum/bt':>6s}  msg")
    for name in names:
        cfg = table[name]
        unit = cfg["unit"]
        ones, manys = [], []
        for _ in range(args.reps):
            ones.append(measure(name, cfg, 1)["batch"])
            manys.append(measure(name, cfg, args.procs))
        one = max(ones)
        best = max(manys, key=lambda m: m["batch"])
        many = best["batch"]
        ones_sorted = sorted(ones)
        manys_sorted = sorted(m["batch"] for m in manys)
        msg = (
            f"{cfg.get('size', args.msgsize)} B/msg"
            if cfg["kind"] in ("evp", "bin-blake3")
            else "-"
        )
        rows[name] = {
            "unit": unit,
            "one_thread": one,
            "one_thread_median": ones_sorted[len(ones_sorted) // 2],
            "procs16": many,
            "procs16_median": manys_sorted[len(manys_sorted) // 2],
            "procs16_steady": best["steady"],
            "msgsize": cfg.get("size", args.msgsize),
            "scale": many / one,
            "procs": args.procs,
            "rounds": args.reps,
            "measured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        print(
            f"{name:26s} {unit:5s} {one:14.4g} {many:14.4g} {many / one:7.2f}  "
            f"{best['steady'] / many:5.2f}x  {msg}"
        )

    # A second machine-state check that does not depend on sysfs: the same binary and the same
    # workload measured twice must agree. If they do not, the two sessions are not comparable and
    # nothing that joins them is either.
    previous = ROOT / "cpu_baseline_barrier.json"
    if previous.exists() and (args.json is None or args.json.resolve() != previous.resolve()):
        with contextlib.suppress(OSError, json.JSONDecodeError):
            old = json.loads(previous.read_text())["rivals"]
            drift = {
                name: row["one_thread"] / old[name]["one_thread"]
                for name, row in rows.items()
                if name in old and old[name]["one_thread"]
            }
            worst = max(drift.items(), key=lambda kv: abs(kv[1] - 1.0), default=None)
            if worst and not 0.67 <= worst[1] <= 1.5:
                print(
                    f"# WARNING: 1T rate for {worst[0]} is {worst[1]:.2f}x the stored baseline "
                    f"-- this machine is in a different power/clock state, so these numbers "
                    f"cannot be joined with anything measured before",
                    file=sys.stderr,
                )

    over = {k: v["scale"] for k, v in rows.items() if v["scale"] > args.procs}
    if over:
        print(
            f"\n# WARNING: scale above {args.procs} is physically impossible; "
            f"re-run those rows: {over}",
            file=sys.stderr,
        )
    if args.json and warning and not args.force:
        # Writing here would replace a good baseline with a throttled one. It happened: a snapshot
        # taken on battery overwrote benchmarks/cpu_baseline_barrier.json with rates 4.6x low, and
        # the drift guard then agreed with itself because both sides were throttled.
        print(
            "error: refusing to write --json from a machine in this state (use --force to "
            "override); the existing baseline is the better artifact",
            file=sys.stderr,
        )
        return 3

    if args.json:
        args.json.write_text(
            json.dumps(
                {"meta": cpu_meta(args.procs, args.msgsize, args.reps), "rivals": rows}, indent=2
            )
            + "\n"
        )
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
