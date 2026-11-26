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
    poly    one 256-coefficient polynomial transform (read + write of 1 KiB)
    msg     one message of `--size` bytes
    byte    a byte of keystream / ciphertext
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from triton_crypto import devices
from triton_crypto.primitives import chacha20, keccak, sha2, sponge
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


def _rate(units: int, t: float) -> str:
    r = units / t
    for scale, suffix in ((1e9, "G"), (1e6, "M"), (1e3, "k")):
        if r >= scale:
            return f"{r / scale:.2f}{suffix}"
    return f"{r:.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--batch", type=int, default=4096, help="items per launch (default 4096)")
    ap.add_argument("--size", type=int, default=1024, help="bytes per message (default 1024)")
    ap.add_argument("--reps", type=int, default=REPS, help=f"min-of-N (default {REPS})")
    ap.add_argument("--only", default=None, help="comma-separated primitive names")
    ap.add_argument("--list", action="store_true", help="list the primitives and exit")
    ap.add_argument("--device", default=None)
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
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

    print(
        f"# device={dev} torch={__import__('torch').__version__} "
        f"triton={__import__('triton').__version__} "
        f"batch={args.batch} size={args.size} reps={args.reps}"
    )

    # The bandwidth reference is measured even with --only, because every GB/s cell is a
    # fraction of it. It is not printed unless asked for.
    copy_t, _ = measure(bench_copy(args.batch, args.size, dev), dev, args.reps)
    copy_bw = 2 * (1 << 26) * 4 / copy_t / 1e9
    print(f"# copy bandwidth: {copy_bw:.1f} GB/s (read+write, measured in this process)")

    rows = []
    for name in names:
        try:
            bench = CASES[name](args.batch, args.size, dev)
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
            }
        )

    if args.md:
        print()
        print("| primitive | what | ms | rate | GB/s | % of copy BW |")
        print("|---|---|---:|---:|---:|---:|")
        for r in rows:
            print(
                f"| `{r['name']}` | {r['note']} | {r['ms']:.3f} | "
                f"{r['rate']} {r['unit']}/s | {r['gbs']:.1f} | {r['share']:.0f}% |"
            )
        return 0

    print()
    print(f"{'primitive':14s} {'what':34s} {'ms':>9s} {'rate':>10s} {'GB/s':>7s} {'%copy':>6s}")
    for r in rows:
        print(
            f"{r['name']:14s} {r['note'][:34]:34s} {r['ms']:9.3f} "
            f"{r['rate'] + ' ' + r['unit'] + '/s':>10s} {r['gbs']:7.1f} {r['share']:5.0f}%"
        )
    print(
        "\n# the GB/s column is traffic (read+write) over device time; for a hash it is the\n"
        "# absorbed message, not the digest. `%copy` is meaningless for perm/poly rows and is\n"
        "# printed anyway so the table has one shape."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
