"""LMS / HSS verification throughput on the device.

There is no liboqs baseline here: liboqs does not implement LMS/HSS (it is a NIST SP 800-208
scheme, and the reference implementations are the hash-sigs C library and the RFC's Java/C++ ones).
`--cpu` therefore measures this repository's *Python* reference, which is a correctness oracle, not
a performance one — read the device numbers as absolute throughput, and the CPU numbers as "a
straightforward hashlib implementation would be this slow".
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from triton_crypto.lms import ref
from triton_crypto.lms.verify import Verifier

VECTORS = json.loads((Path(__file__).parent.parent / "data/lms-rfc8554/vectors.json").read_text())


def device_ms(pub: bytes, msg: bytes, sig: bytes, batch: int, reps: int = 3) -> float:
    v = Verifier(pub)
    messages = [msg] * batch
    sigs = np.stack([np.frombuffer(sig, dtype=np.uint8)] * batch)
    out = v.verify(messages, sigs)  # warm up: compiles the kernels
    assert out.all(), "the vector must verify before we time it"
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        got = v.verify(messages, sigs)
        best = min(best, time.perf_counter() - t0)
        assert got.all()
    return best * 1e3


def cpu_ms(pub: bytes, msg: bytes, sig: bytes, batch: int) -> float:
    t0 = time.perf_counter()
    for _ in range(batch):
        assert ref.hss_verify(pub, msg, sig)
    return (time.perf_counter() - t0) * 1e3


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", type=int, default=0, help="which RFC 8554 test vector to tile")
    ap.add_argument("--batch", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpu", action="store_true", help="also time the Python reference (slow)")
    args = ap.parse_args()

    case = VECTORS[args.case]
    pub, msg, sig = (
        bytes.fromhex(case["pub"]),
        bytes.fromhex(case["msg"]),
        bytes.fromhex(case["sig"]),
    )
    name = f"RFC 8554 test case {case['case']} ({len(sig)}-byte signature, {len(pub)}-byte pubkey)"
    print(f"# {name}")

    rows = []
    for b in args.batch:
        ms = device_ms(pub, msg, sig, b, args.reps)
        cpu = cpu_ms(pub, msg, sig, b) if args.cpu else float("nan")
        rows.append((b, ms, ms / b * 1e3, cpu, cpu / b * 1e3 if args.cpu else float("nan")))
        print(
            f"  B={b:5d}  device {ms:8.1f} ms  {ms / b * 1e3:8.1f} µs/sig"
            + (f"   python {cpu / b * 1e3:9.1f} µs/sig" if args.cpu else "")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
