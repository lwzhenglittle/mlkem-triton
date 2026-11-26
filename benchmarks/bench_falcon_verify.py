#!/usr/bin/env python
"""Falcon-512 verification throughput: device core, end-to-end, and the liboqs rival.

    python benchmarks/bench_falcon_verify.py                 # Falcon-512, B = 64..4096
    python benchmarks/bench_falcon_verify.py --ps falcon-padded-512
    python benchmarks/bench_falcon_verify.py --profile       # where the device time goes

The corpus comes from liboqs 0.16.0 (`pqclean_falcon-512`), so the CPU baseline and the GPU
verifier consume exactly the same signatures.

Two numbers are printed for every batch, and they answer different questions:

    dev/op   the device core only: NTT(s2), the pointwise product, the inverse NTT and the
             norm check, on a corpus that was decoded and hashed already. This is the part
             that has a throughput story.
    e2e/op   `Verifier.verify`, which also pays for the signature decoder and SHAKE256 on the
             host -- 24 us and 9 us per signature at B = 4096. It is printed because hiding
             it would make the table lie about what a caller experiences.

The rival comes from `benchmarks/cpu_baseline.json` (run `benchmarks/bench_cpu.py` first).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triton_crypto import devices
from triton_crypto.falcon import (
    FALCON_512,
    FALCON_PADDED_512,
    Verifier,
    ref,
)
from triton_crypto.falcon.verify import comp_decode_batch

PARAMS = {"falcon-512": FALCON_512, "falcon-padded-512": FALCON_PADDED_512}
TAGS = {"falcon-512": "falcon_512", "falcon-padded-512": "falcon_padded_512"}
CPU_KEY = {"falcon-512": "falcon-512-verify", "falcon-padded-512": "falcon-padded-512-verify"}
LIBOQS = ROOT / "third_party" / "liboqs" / "build-falcon" / "lib" / "liboqs.so"
CPU_JSON = ROOT / "benchmarks" / "cpu_baseline.json"


def corpus(ps: str, n: int):
    """`n` signatures under one public key, from liboqs."""
    import ctypes

    tag = TAGS[ps]
    lib = ctypes.CDLL(str(LIBOQS))
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    keypair = lib[f"OQS_SIG_{tag}_keypair"]
    keypair.argtypes = [u8p, u8p]
    sign = lib[f"OQS_SIG_{tag}_sign"]
    sign.argtypes = [u8p, ctypes.POINTER(ctypes.c_size_t), u8p, ctypes.c_size_t, u8p]
    cap = 65536
    pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
    if keypair(pk, sk) != 0:
        raise RuntimeError("liboqs keygen failed")
    msgs, sigs = [], []
    for i in range(n):
        msg = b"bench-%06d" % i
        mbuf = (ctypes.c_ubyte * len(msg)).from_buffer_copy(msg)
        sig = (ctypes.c_ubyte * cap)()
        siglen = ctypes.c_size_t(cap)
        if sign(sig, ctypes.byref(siglen), mbuf, len(msg), sk) != 0:
            raise RuntimeError("liboqs sign failed")
        msgs.append(msg)
        sigs.append(bytes(sig[: siglen.value]))
    return bytes(pk[: FALCON_512.pk_len]), msgs, sigs


def best(fn, reps: int, dev=None) -> float:
    """min-of-N wall time, with a device sync: without it the number is launch overhead only
    (it read 0.00 us for a 4096-row NTT before this line existed)."""
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        devices.sync(dev)
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ps", default="falcon-512", choices=sorted(PARAMS))
    ap.add_argument("--batch", default="64,512,4096")
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    if not LIBOQS.exists():
        print(f"error: liboqs with Falcon not found at {LIBOQS}", file=sys.stderr)
        print("       see tests/test_falcon_liboqs.py for the cmake line", file=sys.stderr)
        return 2

    params = PARAMS[args.ps]
    batches = [int(b) for b in args.batch.split(",")]
    dev = devices.resolve_device(args.device)
    print(f"device : {devices.describe(dev)}")
    print(f"scheme : {args.ps}  (n={params.n}, q={params.q}, l2bound={params.l2bound})")
    print(f"corpus : from liboqs 0.16.0, largest batch {max(batches)}")

    pk, msgs, sigs = corpus(args.ps, max(batches))
    verifier = Verifier(pk, params, dev, padded=args.ps.endswith("padded-512"))

    cpu = None
    if CPU_JSON.exists():
        data = json.loads(CPU_JSON.read_text())
        cpu = data.get("rivals", {}).get(CPU_KEY[args.ps])

    print(
        f"\n{'B':>6} {'dev ms':>9} {'dev us/op':>10} {'dev Mo/s':>9} "
        f"{'e2e ms':>9} {'e2e us/op':>10}   valid"
    )
    for n in batches:
        m, s = msgs[:n], sigs[:n]
        s2, _keep = comp_decode_batch(
            [x[1 + params.nonce_len :] for x in s], params, padded=args.ps.endswith("padded-512")
        )
        c = ref.hash_to_point_batch([x[1 : 1 + params.nonce_len] for x in s], m, params)
        # warm-up (compiles the kernels) and a validity check before timing
        ok, _ = verifier.verify_core_torch(c, s2)
        valid = int(ok.cpu().numpy().sum())
        if valid != n:
            print(f"  WARNING: only {valid}/{n} accepted -- timing a wrong path is useless")
        t_dev = best(lambda c=c, s2=s2: verifier.verify_core_torch(c, s2), args.reps, dev)
        t_e2e = best(lambda m=m, s=s: verifier.verify(m, s), max(3, args.reps // 3), dev)
        print(
            f"{n:6d} {t_dev * 1e3:9.3f} {t_dev / n * 1e6:10.2f} {n / t_dev / 1e6:9.2f} "
            f"{t_e2e * 1e3:9.3f} {t_e2e / n * 1e6:10.2f}   {valid}/{n}"
        )

    n = batches[-1]
    s2, _ = comp_decode_batch(
        [x[1 + params.nonce_len :] for x in sigs[:n]], params, padded=args.ps.endswith("padded-512")
    )
    c = ref.hash_to_point_batch([x[1 : 1 + params.nonce_len] for x in sigs[:n]], msgs[:n], params)
    t_dev = best(lambda: verifier.verify_core_torch(c, s2), args.reps, dev) / n
    if cpu:
        print(f"\nliboqs CPU baseline ({cpu['one_thread']:.0f}/s 1T, {cpu['procs16']:.0f}/s 16P):")
        print(
            f"  device core {t_dev * 1e6:.2f} us/op = {1 / t_dev / cpu['one_thread']:.1f}x 1T, "
            f"{1 / t_dev / cpu['procs16']:.2f}x 16P"
        )
    else:
        print("\n(no benchmarks/cpu_baseline.json -- run benchmarks/bench_cpu.py for the rival)")

    if args.profile:
        import torch

        from triton_crypto.falcon.verify import _falcon_domain
        from triton_crypto.primitives.ntt import intt_torch, ntt_torch

        dom = _falcon_domain(params)
        s2_mod = torch.as_tensor(s2 % params.q, dtype=torch.int32, device=dev)
        t_ntt = best(lambda: ntt_torch(s2_mod, dom), args.reps, dev)
        hat = ntt_torch(s2_mod, dom)
        t_intt = best(lambda: intt_torch(hat, dom), args.reps, dev)
        t_all = best(lambda: verifier.verify_core_torch(c, s2), args.reps, dev)
        print(
            f"\nper-stage, B={n}: NTT {t_ntt / n * 1e6:.2f} us, INTT {t_intt / n * 1e6:.2f} us, "
            f"whole core {t_all / n * 1e6:.2f} us "
            f"(transforms are {(t_ntt + t_intt) / t_all * 100:.0f}% of it)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
