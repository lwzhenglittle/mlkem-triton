#!/usr/bin/env python
"""One benchmark entry point for every scheme in the library.

    python benchmarks/bench.py --list
    python benchmarks/bench.py --scheme mldsa --ps ML-DSA-65 --batch 4096
    python benchmarks/bench.py --scheme mlkem --op e2e --ps ML-KEM-768 --batch 4096

This is a thin dispatcher, on purpose. Each scheme keeps its own measurement script (they have
different things to say: ML-KEM has three operations and a host-side baseline, ML-DSA has a
signature corpus and a per-stage profile), and this file only decides *which* one runs and with
which arguments. Reproducing a table in `docs/benchmarks.md` therefore means "run the same
script", not "run a re-implementation that drifted".

Measurement conventions live in `docs/benchmarks.md` §1 and are enforced inside the per-scheme
scripts: same-process A/B only, min-of-N (median also printed), device-only and end-to-end
reported separately, and a CPU baseline measured on the same machine.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# scheme -> operation -> script. `None` as the operation means "the scheme has a single script".
SCRIPTS = {
    "mlkem": {
        "e2e": "bench_e2e.py",
        "encaps": "bench_encaps.py",
        "keygen": "bench_keygen.py",
    },
    "mldsa": {
        None: "bench_mldsa_verify.py",
        "sign": "bench_mldsa_sign.py",
    },
    "slhdsa": {
        None: "bench_slhdsa_verify.py",
    },
    "lms": {
        None: "bench_lms_verify.py",
    },
}

SCHEME_INFO = {
    "mlkem": (
        "ML-KEM (FIPS 203) — keygen / encapsulate / decapsulate",
        "mlkem",
        ["e2e", "encaps", "keygen"],
    ),
    "mldsa": (
        "ML-DSA (FIPS 204) — signature verification",
        "mldsa",
        [],
    ),
    "lms": (
        "LMS / HSS (SP 800-208) — signature verification",
        "lms",
        [],
    ),
}


def _load(script: str):
    path = HERE / script
    spec = importlib.util.spec_from_file_location(f"bench_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _usage() -> int:
    print("schemes and operations:\n")
    for scheme, (blurb, _, ops) in SCHEME_INFO.items():
        print(f"  {scheme:6s} {blurb}")
        if ops:
            print(f"         --op {{{','.join(ops)}}}")
        else:
            print("         (no --op)")
    print("\nevery other flag is passed to the scheme script; run")
    print("  python benchmarks/bench.py --scheme mlkem --op e2e --help")
    print("for the full list.")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--scheme", choices=sorted(SCRIPTS))
    ap.add_argument("--op", default=None)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    known, rest = ap.parse_known_args(argv)

    if known.list or known.help or known.scheme is None:
        return _usage()

    ops = SCRIPTS[known.scheme]
    if known.op is not None and known.op not in ops:
        print(
            f"error: --op {known.op!r} is not valid for --scheme {known.scheme}; "
            f"choose from {', '.join(sorted(k for k in ops if k)) or '(none)'}",
            file=sys.stderr,
        )
        return 2
    script = ops[known.op] if known.op in ops else ops[None if None in ops else next(iter(ops))]

    print(
        f"# dispatch: --scheme {known.scheme}"
        f"{f' --op {known.op}' if known.op else ''} -> benchmarks/{script}"
    )
    module = _load(script)

    # The scheme scripts have their own argparse and read sys.argv directly; hand them exactly
    # the arguments that were not consumed above.
    saved = sys.argv
    sys.argv = [script, *rest]
    try:
        return module.main()
    finally:
        sys.argv = saved


if __name__ == "__main__":
    raise SystemExit(main())
