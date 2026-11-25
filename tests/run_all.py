#!/usr/bin/env python
"""Entry point for running the tests without pytest (remote / CI often has none).

    python tests/run_all.py                       # auto-select device
    python tests/run_all.py --device cuda         # specify device
    python tests/run_all.py --only test_glue      # run only one module
    python tests/run_all.py --list                # list cases only

If pytest is available in the environment, `pytest` is the right choice (this
runner covers only the subset of pytest this repo uses). With no pytest it
injects `tests/_pytest_shim.py`.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for p in (str(_ROOT), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)


def _resolve_fixtures(mod, fn, provided):
    """Inject the no-arg functions in the module marked `_is_fixture`, by param name.

    `provided` holds the parameter names already filled by parametrize; skip them.
    """
    import inspect

    kwargs = {}
    for name in inspect.signature(fn).parameters:
        if name in provided:
            continue
        f = getattr(mod, name, None)
        if f is None or not getattr(f, "_is_fixture", False):
            return None
        try:
            kwargs[name] = f()
        except TypeError:
            # fixture relies on pytest's `request` etc.; this runner does not emulate it, skip.
            return None
    return kwargs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="run the given module (`test_` prefix optional)")
    ap.add_argument("--list", action="store_true", help="list cases only, do not run")
    ap.add_argument("-x", action="store_true", help="stop at the first failure")
    args = ap.parse_args()
    if args.device:
        os.environ["MLKEM_DEVICE"] = args.device

    import _pytest_shim

    _pytest_shim.install()

    import pytest

    if getattr(pytest, "__file__", None) is not None:
        # real pytest is present, hand over to it (supports fixture / parametrize / -x etc.).
        argv = [str(_HERE)]
        if args.only:
            want = args.only if args.only.startswith("test_") else "test_" + args.only
            argv = [str(_HERE / (want + ".py"))]
        if args.list:
            argv.append("--collect-only")
        if args.x:
            argv.append("-x")
        print(f"# detected pytest {pytest.__version__}, handing over to it")
        return int(pytest.main(["-q", *argv]))

    import _common
    from _pytest_shim import Skipped

    is_shim = getattr(sys.modules.get("pytest"), "__file__", None) is None
    print(f"# device: {_common.dev_name()}")
    print(f"# runner: {'built-in shim (no pytest in the environment)' if is_shim else 'pytest'}\n")

    mods = sorted(p.stem for p in _HERE.glob("test_*.py"))
    if args.only:
        want = args.only if args.only.startswith("test_") else "test_" + args.only
        mods = [m for m in mods if m == want]
        if not mods:
            print(f"module not found: {args.only}", file=sys.stderr)
            return 2

    total_pass = total_fail = total_skip = 0
    failures = []
    for name in mods:
        try:
            mod = importlib.import_module(name)
        except Skipped as e:
            print(f"== {name} == SKIP: {e}")
            continue
        print(f"== {name} ==")
        fns = sorted(
            n
            for n in dir(mod)
            if n.startswith("test_")
            and callable(getattr(mod, n))
            and not getattr(getattr(mod, n), "_is_fixture", False)
        )
        for fn_name in fns:
            fn = getattr(mod, fn_name)
            cases = getattr(fn, "_param_cases", None) or [{}]
            for kwargs in cases:
                label = fn_name + (
                    f"[{', '.join(f'{k}={v!r}' for k, v in kwargs.items())}]" if kwargs else ""
                )
                if args.list:
                    print(f"  {label}")
                    continue
                fx = _resolve_fixtures(mod, fn, set(kwargs))
                if fx is None:
                    print(f"  SKIP {label}: cannot resolve fixture")
                    total_skip += 1
                    continue
                try:
                    fn(**fx, **kwargs)
                    total_pass += 1
                    print(f"  PASS {label}")
                except Skipped as e:
                    total_skip += 1
                    print(f"  SKIP {label}: {e}")
                except Exception as e:
                    total_fail += 1
                    failures.append((name, label, e))
                    print(f"  FAIL {label}: {type(e).__name__}: {e}")
                    if args.x:
                        break
            if args.x and failures:
                break
        if args.x and failures:
            break

    if args.list:
        return 0
    print(f"\n# {total_pass} passed, {total_skip} skipped, {total_fail} failed")
    for name, label, e in failures:
        print(f"\n--- {name}::{label} ---", file=sys.stderr)
        traceback.print_exception(type(e), e, e.__traceback__)
    return 1 if total_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
