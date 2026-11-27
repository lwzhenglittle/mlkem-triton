#!/usr/bin/env python
"""Throughput of every scheme-agnostic primitive, one table.

    python benchmarks/bench_primitive.py                    # everything, 1 KiB messages
    python benchmarks/bench_primitive.py --list
    python benchmarks/bench_primitive.py --only chacha20,ntt-mldsa --batch 4096
    python benchmarks/bench_primitive.py --size 1024 --md   # markdown, for docs

Why this exists, separately from the scheme benchmarks: a scheme tells you whether an
operation is worth putting on a device, a primitive tells you **which shape of arithmetic the
device is good at**. `copy` is measured first and every byte-oriented primitive is printed
against it, because "GB/s" without a ceiling is not a number anyone can act on.

Measurement conventions are the same ones the rest of this directory uses (docs/benchmarks.md
§1): same process, min-of-N, device-only (an operation's H2D/D2H is not in the timing), and
nothing is compared across sessions.

Unit conventions:
    perm    one Keccak-f[1600] permutation
    p2      one Poseidon2-BabyBear/16 permutation (16 field elements in, 16 out)
    poly    one 256-coefficient polynomial transform (read + write of 1 KiB)
    msg     one message of `--size` bytes
    byte    a byte of keystream / ciphertext
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_crypto import devices
from triton_crypto.primitives import (
    aes,
    blake3,
    chacha20,
    gemm,
    keccak,
    poly1305,
    poseidon2,
    sha2,
    sponge,
)
from triton_crypto.primitives.ntt import (
    BABYBEAR_NTT,
    DILITHIUM,
    KYBER,
    ntt_torch,
)

REPS = 7


@dataclass
class Bench:
    """One measured call: `fn()` launches work, the harness syncs and times it."""

    fn: Callable[[], object]
    units: int  # how many `unit`s one call processes
    unit: str
    nbytes: int  # bytes read + written by one call, for the GB/s column
    note: str = ""


# --------------------------------------------------------------------------- builders


def _copy_kernel():
    """A plain copy, used as the bandwidth reference for every GB/s number below."""
    import triton
    import triton.language as tl

    @triton.jit
    def _copy(SRC, DST, n, BLOCK: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        tl.store(DST + i, tl.load(SRC + i, mask=m, other=0), mask=m)

    return _copy


_COPY = _copy_kernel()


def bench_copy(batch: int, size: int, dev) -> Bench:
    import torch
    import triton

    n = 1 << 26  # 256 MiB of int32
    src = torch.empty(n, dtype=torch.int32, device=dev)
    dst = torch.empty_like(src)
    return Bench(
        lambda: _COPY[(triton.cdiv(n, 4096),)](src, dst, n, 4096),
        units=n * 4,
        unit="B",
        nbytes=2 * n * 4,
        note="read+write",
    )


def bench_keccak(batch: int, size: int, dev) -> Bench:
    import torch
    import triton

    kern = keccak.get_kernel()
    # 4096 permutations is 0.8 MB: launch-bound. The documented rate is measured at >= 2^18.
    n = max(batch, 1 << 18)
    x = torch.zeros((25, n), dtype=torch.int64, device=dev)
    out = torch.empty_like(x)
    block = 128
    return Bench(
        lambda: kern[(triton.cdiv(n, block),)](x, out, n, block, num_warps=4),
        units=n,
        unit="perm",
        nbytes=2 * 25 * 8 * n,
        note="24 rounds, rolled",
    )


def _sponge_bench(hf, batch: int, size: int, dev, xof_out: int = 32) -> Bench:
    import torch

    rows = max(batch, 1)
    msgs = torch.zeros((rows, size), dtype=torch.uint8, device=dev)
    padded = sponge.pad_torch(msgs, hf.rate, hf.domain)
    rate = hf.rate
    outlen = hf.outlen or xof_out  # XOFs squeeze as much as you ask for; 32 B is the usual ask
    return Bench(
        lambda: sponge.sponge_padded(padded, rate, outlen),
        units=rows,
        unit="msg",
        nbytes=padded.numel() + rows * outlen,
        note=f"rate {rate}, {padded.shape[1] // rate} block(s)",
    )


def _sha_bench(bits: int, batch: int, size: int, dev) -> Bench:
    import torch

    rows = max(batch, 1)
    block_bytes = 64 if bits == 256 else 128
    dtype = torch.int32 if bits == 256 else torch.int64
    nblocks = max(1, size // block_bytes)
    msg = torch.zeros((rows, nblocks * 16), dtype=dtype, device=dev)
    out = torch.empty((rows, 8), dtype=dtype, device=dev)
    hin = sha2.iv_tensor(bits, dev)
    return Bench(
        lambda: sha2.sha_absorb(hin, msg, out, rows, bits, nblocks),
        units=rows,
        unit="msg",
        nbytes=rows * nblocks * block_bytes,
        note=f"{nblocks * block_bytes} B/msg, {bits}-bit",
    )


def _ntt_bench(domain, batch: int, size: int, dev) -> Bench:
    import torch

    rows = max(batch, 1)
    x = torch.zeros((rows, domain.n), dtype=torch.int32, device=dev)
    return Bench(
        lambda: ntt_torch(x, domain),
        units=rows,
        unit="poly",
        nbytes=2 * rows * domain.n * 4,
        note=f"q={domain.q}, {domain.layers} layers" + (" + Montgomery" if domain.wide else ""),
    )


def bench_poseidon2(batch: int, size: int, dev) -> Bench:
    """One permutation = 16 field elements in and out; `size` is unused."""
    import torch

    rows = max(batch, 1)
    x = torch.zeros((rows, 16), dtype=torch.int32, device=dev)
    return Bench(
        lambda: poseidon2.poseidon2_permute_torch(x, device=dev),
        units=rows,
        unit="p2",
        nbytes=2 * rows * 16 * 4,
        note="BabyBear width 16, 8 full + 13 partial",
    )


def bench_gemm_int8(batch: int, size: int, dev) -> Bench:
    """Exact int8 GEMM at 1024x1024x1024 (batch/size do not apply; MACs are the unit).

    256 is launch-bound (0.18 TMAC/s) and 2048 reaches 3.96; 1024 is a fair single number, and
    `docs/benchmarks.md` §19 has the scaling table.
    """
    import torch

    m = k = n = 1024
    rng = np.random.default_rng(0)
    a = torch.as_tensor(rng.integers(-128, 128, (m, k), dtype=np.int8), device=dev)
    b = torch.as_tensor(rng.integers(-128, 128, (k, n), dtype=np.int8), device=dev)
    return Bench(
        lambda: gemm.gemm_int8_torch(a, b, device=dev),
        units=m * n * k,
        unit="MAC",
        nbytes=(m * k + k * n + m * n * 4),
        note="int8 x int8 -> int32, exact",
    )


def bench_aes(batch: int, size: int, dev) -> Bench:
    """T-table AES-128-CTR. The host key schedule is hoisted out of the timing: it is ~80 us
    per key, i.e. 0.3 s for 4096 streams, which would swamp the kernel (see §18)."""
    import torch

    rows, size = max(batch, 1), max(size, 16)
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 256, (rows, 16), dtype=np.uint8)
    nonces = rng.integers(0, 256, (rows, 8), dtype=np.uint8)
    rk = torch.as_tensor(
        np.stack([aes.aes128_expand_key_np(bytes(k)) for k in keys]).astype(np.int32), device=dev
    )
    return Bench(
        lambda: aes.aes128_ctr_torch(keys, nonces, size, device=dev, rk=rk),
        units=rows * size,
        unit="B",
        nbytes=rows * size,
        note="AES-128-CTR, T-table (gather)",
    )


_LARGE_NTT = None


def bench_ntt_large(batch: int, size: int, dev) -> Bench:
    """The 2^20 multi-pass transform; `batch` and `size` do not apply."""
    import torch

    global _LARGE_NTT
    from triton_crypto.primitives.ntt import LargeNTT

    if _LARGE_NTT is None:
        _LARGE_NTT = LargeNTT(1 << 20, 2013265921)
    t = _LARGE_NTT
    x = torch.as_tensor(
        np.random.default_rng(0).integers(0, 2013265921, 1 << 20), dtype=torch.int32, device=dev
    )
    return Bench(
        lambda: t.forward_tensor_torch(x, dev),
        units=1,
        unit="p2^20",
        # 8 MB read + 8 MB write per pass, and the pass count depends on the radix -- hardcoding
        # 20 overstated the traffic (and therefore GB/s and %copy) by 2x once radix-4 became
        # the default.
        nbytes=(1 << 20) * 4 * 2 * t.n_passes,
        note=f"BabyBear 2^20, {t.n_passes} passes (radix={t.radix})",
    )


def bench_blake3(batch: int, size: int, dev) -> Bench:
    rows, size = max(batch, 1), max(size, 1)
    rng = np.random.default_rng(0)
    msgs = rng.integers(0, 256, (rows, size), dtype=np.uint8)
    return Bench(
        lambda: blake3.blake3_hash_torch(msgs, device=dev),
        units=rows * size,
        unit="B",
        nbytes=rows * size,
        note="BLAKE3, unkeyed, 32-byte output",
    )


def bench_poly1305(batch: int, size: int, dev) -> Bench:
    rows, size = max(batch, 1), max(size, 16)
    size -= size % 16  # the device path takes whole 16-byte blocks
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 256, (rows, 32), dtype=np.uint8)
    msgs = rng.integers(0, 256, (rows, size), dtype=np.uint8)
    return Bench(
        lambda: poly1305.poly1305_torch(keys, msgs, device=dev),
        units=rows * size,
        unit="B",
        nbytes=rows * size,
        note="RFC 8439, 5x26-bit limbs",
    )


def bench_aead(batch: int, size: int, dev) -> Bench:
    rows, size = max(batch, 1), max(size, 1)
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 256, (rows, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (rows, 12), dtype=np.uint8)
    pt = rng.integers(0, 256, (rows, size), dtype=np.uint8)
    return Bench(
        lambda: poly1305.chacha20_poly1305_torch(keys, nonces, pt, device=dev),
        units=rows * size,
        unit="B",
        nbytes=2 * rows * size,
        note="ChaCha20-Poly1305 seal, no AAD",
    )


def bench_chacha20(batch: int, size: int, dev) -> Bench:
    rng = np.random.default_rng(0)
    rows = max(batch, 1)
    keys = rng.integers(0, 256, (rows, 32), dtype=np.uint8)
    nonces = rng.integers(0, 256, (rows, 12), dtype=np.uint8)
    return Bench(
        lambda: chacha20.chacha20_keystream_torch(keys, nonces, size, 0, dev),
        units=rows * size,
        unit="B",
        nbytes=rows * size,
        note="RFC 8439, 20 rounds",
    )


CASES: dict[str, Callable[[int, int, object], Bench]] = {
    "copy": bench_copy,
    "keccak": bench_keccak,
    "sha3-256": lambda b, s, d: _sponge_bench(sponge.SHA3_256, b, s, d),
    "sha3-512": lambda b, s, d: _sponge_bench(sponge.SHA3_512, b, s, d),
    "shake128": lambda b, s, d: _sponge_bench(sponge.SHAKE128, b, s, d),
    "shake256": lambda b, s, d: _sponge_bench(sponge.SHAKE256, b, s, d),
    "sha256": lambda b, s, d: _sha_bench(256, b, s, d),
    "sha512": lambda b, s, d: _sha_bench(512, b, s, d),
    "ntt-mlkem": lambda b, s, d: _ntt_bench(KYBER, b, s, d),
    "ntt-mldsa": lambda b, s, d: _ntt_bench(DILITHIUM, b, s, d),
    "ntt-babybear": lambda b, s, d: _ntt_bench(BABYBEAR_NTT, b, s, d),
    "chacha20": bench_chacha20,
    "poseidon2": bench_poseidon2,
    "poly1305": bench_poly1305,
    "chacha20-poly1305": bench_aead,
    "aes128-ctr": bench_aes,
    "gemm-int8": bench_gemm_int8,
    "blake3": bench_blake3,
    "ntt-2^20": bench_ntt_large,
}

ORDER = list(CASES)

WHAT = {
    "copy": "bandwidth reference (read + write of a 256 MiB buffer)",
    "keccak": "Keccak-f[1600] permutation, one lane per thread",
    "sha3-256": "SHA3-256 over --size-byte messages",
    "sha3-512": "SHA3-512 over --size-byte messages",
    "shake128": "SHAKE128, 32-byte output",
    "shake256": "SHAKE256, 32-byte output",
    "sha256": "SHA-256 compression chain over --size-byte messages",
    "sha512": "SHA-512 compression chain over --size-byte messages",
    "ntt-mlkem": "negacyclic NTT, q = 3329, 7 layers",
    "ntt-mldsa": "negacyclic NTT, q = 8380417, 8 layers, Montgomery",
    "ntt-babybear": "negacyclic NTT, q = 2^31-2^27+1, 8 layers",
    "chacha20": "ChaCha20 keystream (RFC 8439) over --size-byte streams",
    "poseidon2": "Poseidon2-BabyBear/16 permutation (Plonky3 parameters)",
    "poly1305": "Poly1305 over --size-byte messages (multiple of 16)",
    "chacha20-poly1305": "ChaCha20-Poly1305 seal (encrypt + tag), no AAD",
    "aes128-ctr": "AES-128-CTR through a T-table (the gather counter-example)",
    "gemm-int8": "exact int8 GEMM at 1024^3 (the GF(2^8) product substrate; §19 has the scaling)",
    "blake3": "BLAKE3 (tree hash), --size-byte messages",
    "ntt-2^20": "BabyBear negacyclic NTT, 2^20 points (multi-pass, radix from LargeNTT)",
}


# --------------------------------------------------------------------------- measuring


def measure(bench: Bench, dev, reps: int = REPS) -> tuple[float, float]:
    """(min seconds, median seconds) for one call; includes the sync."""
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        bench.fn()
        devices.sync(dev)
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[0], times[len(times) // 2]


# Which CPU rival answers which row. Measured by `benchmarks/bench_cpu.py` (openssl, liboqs,
# Plonky3) into `benchmarks/cpu_baseline.json`; a row with no rival prints `n/a`, never a
# silently missing column. Rows whose CPU work is not measurable with a fast reference
# (Keccak's bare permutation, the NTTs -- numpy is 10-100x off) are deliberately absent.
CPU_RIVAL = {
    "chacha20": "chacha20",
    "sha256": "sha256",
    "sha512": "sha512",
    "sha3-256": "sha3-256",
    "sha3-512": "sha3-512",
    "poseidon2": "poseidon2",
    "chacha20-poly1305": "chacha20-poly1305",
    "aes128-ctr": "aes-128-ctr",
    "poly1305": "poly1305",
    "shake128": "shake128",
    "shake256": "shake256",
    "blake3": "blake3",
    "ntt-2^20": "ntt-2^20",
    "gemm-int8": "gemm-int8",
}

CPU_BASELINE = Path(__file__).resolve().parent / "cpu_baseline.json"
# The barrier measurement supersedes the sum-of-minima one: the latter over-counts the CPU by
# 1.25x to 4.25x depending on the row (docs/benchmarks.md §15), and every `vs 16P` ratio inherits
# that error. Preferred when present; `--baseline` picks one explicitly.
CPU_BASELINE_BARRIER = Path(__file__).resolve().parent / "cpu_baseline_barrier.json"


def load_cpu(path: Path | None = None) -> dict | None:
    """The CPU baseline, if it exists and was taken on this machine."""
    if path is None:
        path = CPU_BASELINE_BARRIER if CPU_BASELINE_BARRIER.exists() else CPU_BASELINE
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    meta = data.get("meta", {})
    if meta.get("host") != platform.node():
        print(
            f"# ignoring {path.name}: measured on {meta.get('host')!r}, this is "
            f"{platform.node()!r} -- rerun benchmarks/bench_cpu.py",
            file=sys.stderr,
        )
        return None
    return data


def _speedup(device: float, cpu: float) -> str:
    """`12.3x` / `0.40x` -- and no number at all when the ratio would be meaningless."""
    if not cpu or not device:
        return "n/a"
    r = device / cpu
    return f"{r:.1f}x" if r >= 10 else f"{r:.2f}x"


def _bytes_fmt(v: float | None) -> str:
    if v is None:
        return "n/a"
    for scale, suffix in ((1e6, "MB"), (1e3, "kB")):
        if v >= scale:
            return f"{v / scale:.1f}{suffix}"
    return f"{v:.0f}B"


def _rate(units: int, t: float) -> str:
    r = units / t
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if r >= scale:
            return f"{r / scale:.2f}{suffix}"
    return f"{r:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--batch",
        default="4096",
        help="items per launch, or a comma-separated list (default 4096). A list prints one table "
        "per batch, which is how a ratio between two batch sizes gets measured in one window -- "
        "the other benchmark scripts have taken lists from the start",
    )
    ap.add_argument("--size", type=int, default=1024, help="bytes per message (default 1024)")
    ap.add_argument("--reps", type=int, default=REPS, help=f"min-of-N (default {REPS})")
    ap.add_argument("--only", default=None, help="comma-separated primitive names")
    ap.add_argument("--list", action="store_true", help="list the primitives and exit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    ap.add_argument(
        "--baseline",
        default=None,
        help="CPU baseline JSON (default: cpu_baseline_barrier.json, falling back to "
        "cpu_baseline.json)",
    )
    ap.add_argument(
        "--json",
        default=None,
        help="also write every measured row as JSON (the join for bench_16p_barrier.py needs the "
        "numeric rate, not the formatted one)",
    )
    ap.add_argument(
        "--no-cpu",
        action="store_true",
        help="skip the vs-CPU columns (benchmarks/cpu_baseline.json)",
    )
    args = ap.parse_args()

    if args.list:
        for name in ORDER:
            print(f"  {name:14s} {WHAT.get(name, '')}")
        return 0

    dev = devices.resolve_device(args.device)
    names = [n.strip() for n in args.only.split(",")] if args.only else ORDER
    unknown = [n for n in names if n not in CASES]
    if unknown:
        print(f"error: unknown primitive(s) {unknown}; try --list", file=sys.stderr)
        return 2

    batches = [int(b) for b in str(args.batch).split(",")]
    print(
        f"# device={dev} torch={__import__('torch').__version__} "
        f"triton={__import__('triton').__version__} "
        f"batch={args.batch} size={args.size} reps={args.reps}"
    )

    return _table(names, batches, args, dev)


def _table(names: list[str], batches: list[int], args, dev) -> int:
    """One table per batch size in `batches`; the copy reference is measured per table."""
    # The bandwidth reference is measured even with --only, because every GB/s cell is a
    # fraction of it. It is not printed unless asked for.
    rc = 0
    for batch in batches:
        if len(batches) > 1:
            print(f"\n## batch={batch}")
        copy_t, _ = measure(bench_copy(batch, args.size, dev), dev, args.reps)
        copy_bw = 2 * (1 << 26) * 4 / copy_t / 1e9
        print(f"# copy bandwidth: {copy_bw:.1f} GB/s (read+write, measured in this process)")
        rc |= _one_table(names, batch, args, dev, copy_t, copy_bw)
    return rc


def _one_table(names, batch, args, dev, copy_t, copy_bw) -> int:
    rows = []
    for name in names:
        try:
            bench = CASES[name](batch, args.size, dev)
            # Reuse the bandwidth reference measurement instead of timing the same copy twice.
            t, med = (copy_t, copy_t) if name == "copy" else measure(bench, dev, args.reps)
        except Exception as e:  # a bench table should survive one broken case
            print(f"{name:14s} FAILED {type(e).__name__}: {str(e)[:120]}")
            continue
        gbs = bench.nbytes / t / 1e9
        share = 100.0 * gbs / copy_bw
        rows.append(
            {
                "name": name,
                "note": bench.note,
                "ms": t * 1e3,
                "spread": (med / t - 1) * 100 if t else 0.0,
                "rate": _rate(bench.units, t),
                "unit": bench.unit,
                "gbs": gbs,
                "share": share,
                "bytes": bench.nbytes / t,
                "ops": bench.units / t,
                # One *unit* of work, expressed as bytes of copy bandwidth. A byte-rate row
                # has no meaningful per-unit version, so it stays None.
                "copy_equiv": None if bench.unit == "B" else (t / bench.units) * copy_bw * 1e9,
            }
        )

    cpu = None if args.no_cpu else load_cpu(Path(args.baseline) if args.baseline else None)
    for r in rows:
        rival_name = CPU_RIVAL.get(r["name"])
        rival = (cpu or {}).get("rivals", {}).get(rival_name) if rival_name else None
        if rival is None:
            r["vs1t"] = r["vs16p"] = "n/a"
            continue
        unit = rival["unit"]
        device_rate = r["bytes"] if unit == "B/s" else r["ops"]
        r["vs1t"] = _speedup(device_rate, rival["one_thread"])
        r["vs16p"] = _speedup(device_rate, rival["procs16"])

    if args.json and str(batch) == str(args.batch).split(",")[0]:
        # only the first batch of a list: the file is meant to be joined, and a list would need a
        # schema the join does not have
        Path(args.json).write_text(
            json.dumps(
                {
                    "meta": {
                        "batch": batch,
                        "size": args.size,
                        "reps": args.reps,
                        "cpu_baseline": None if cpu is None else cpu.get("meta"),
                    },
                    "rows": rows,
                },
                indent=2,
            )
            + "\n"
        )

    if cpu:
        meta = cpu.get("meta", {})
        print(
            f"# CPU baseline: {meta.get('cpu')} from {meta.get('timestamp')} "
            f"({len(cpu['rivals'])} rivals, 16P = 16 processes, "
            f"{meta.get('method', 'method not recorded -- sum of per-child minima, an upper bound')})"
        )

    if args.md:
        print()
        print("| primitive | what | ms | rate | GB/s | % of copy BW | ≈B copy | vs 1T | vs 16P |")
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| `{r['name']}` | {r['note']} | {r['ms']:.3f} | "
                f"{r['rate']} {r['unit']}/s | {r['gbs']:.1f} | {r['share']:.0f}% | "
                f"{_bytes_fmt(r['copy_equiv'])} | {r['vs1t']} | {r['vs16p']} |"
            )
        return 0

    print()
    print(
        f"{'primitive':14s} {'what':30s} {'ms':>8s} {'rate':>10s} {'GB/s':>6s} {'%copy':>6s} "
        f"{'≈copy':>8s} {'vs 1T':>7s} {'vs 16P':>7s}"
    )
    for r in rows:
        print(
            f"{r['name']:14s} {r['note'][:30]:30s} {r['ms']:8.3f} "
            f"{r['rate'] + ' ' + r['unit'] + '/s':>10s} {r['gbs']:6.1f} {r['share']:5.0f}% "
            f"{_bytes_fmt(r['copy_equiv']):>8s} {r['vs1t']:>7s} {r['vs16p']:>7s}"
        )
    baseline_label = (
        f"benchmarks/{args.baseline.name}"
        if args.baseline
        else (
            "benchmarks/cpu_baseline_barrier.json"
            if CPU_BASELINE_BARRIER.exists()
            else "benchmarks/cpu_baseline.json"
        )
    )
    print(
        "\n# the GB/s column is traffic (read+write) over device time; for a hash it is the\n"
        "# absorbed message, not the digest. `%copy` is meaningless for perm/poly rows and is\n"
        "# printed anyway so the table has one shape.\n"
        "# `≈B copy` is the time one operation takes, expressed as bytes of copy bandwidth --\n"
        "# the only column here that does not move with the iGPU's clock state.\n"
        f"# `vs 1T`/`vs 16P` compare against {baseline_label} (16P = 16 processes);\n"
        "# `n/a` means no CPU reference exists for that row, not that the row is fast."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
