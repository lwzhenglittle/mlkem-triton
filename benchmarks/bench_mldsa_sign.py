"""ML-DSA signing throughput on the device.

Signing is not a fixed amount of work per signature: the rejection loop retries (usually 1–5
times), so the number of attempts — and therefore the device time — varies with the key and the
messages. The benchmark reports wall-clock per signature over a batch of independent messages,
which is the number that matters when a batch is signed together.
"""

from __future__ import annotations

import argparse

from triton_crypto.mldsa.params import PARAMS
from triton_crypto.mldsa.ref import MLDSA
from triton_crypto.mldsa.sign import Signer


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ps", nargs="+", default=["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"])
    ap.add_argument("--batch", type=int, nargs="+", default=[64, 256, 1024])
    ap.add_argument("--reps", type=int, default=2)
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
            print(f"  B={b:5d}  {best * 1e3:8.1f} ms  {best / b * 1e6:8.1f} µs/sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
