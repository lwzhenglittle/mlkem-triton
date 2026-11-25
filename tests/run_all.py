#!/usr/bin/env python
"""不依赖 pytest 跑测试的入口（远程 / CI 上常常没有 pytest）。

    python tests/run_all.py                       # 自动选设备
    python tests/run_all.py --device cuda         # 指定设备
    python tests/run_all.py --only test_glue      # 只跑一个模块
    python tests/run_all.py --list                # 只列用例

如果环境里有 pytest，`pytest` 才是正确选择（这个运行器只覆盖本仓库用到的
pytest 子集）。没有 pytest 时它会注入 `tests/_pytest_shim.py`。
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
    """把模块里 `_is_fixture` 标了的无参函数按参数名注入。

    `provided` 是 parametrize 已经填掉的参数名，跳过它们。
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
            # fixture 依赖 pytest 的 `request` 之类；这个运行器不模拟，直接跳过。
            return None
    return kwargs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--only", default=None, help="只跑指定模块（可省略 test_ 前缀）")
    ap.add_argument("--list", action="store_true", help="只列出用例，不执行")
    ap.add_argument("-x", action="store_true", help="第一个失败就停")
    args = ap.parse_args()
    if args.device:
        os.environ["MLKEM_DEVICE"] = args.device

    import _pytest_shim

    _pytest_shim.install()

    import pytest

    if getattr(pytest, "__file__", None) is not None:
        # 环境里有真 pytest，直接交给它（支持 fixture / parametrize / -x 等）。
        argv = [str(_HERE)]
        if args.only:
            want = args.only if args.only.startswith("test_") else "test_" + args.only
            argv = [str(_HERE / (want + ".py"))]
        if args.list:
            argv.append("--collect-only")
        if args.x:
            argv.append("-x")
        print(f"# 检测到 pytest {pytest.__version__}，交给它跑")
        return int(pytest.main(["-q", *argv]))

    import _common
    from _pytest_shim import Skipped

    is_shim = getattr(sys.modules.get("pytest"), "__file__", None) is None
    print(f"# device: {_common.dev_name()}")
    print(f"# runner: {'内置 shim（环境里没有 pytest）' if is_shim else 'pytest'}\n")

    mods = sorted(p.stem for p in _HERE.glob("test_*.py"))
    if args.only:
        want = args.only if args.only.startswith("test_") else "test_" + args.only
        mods = [m for m in mods if m == want]
        if not mods:
            print(f"找不到模块 {args.only}", file=sys.stderr)
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
                    print(f"  SKIP {label}: 无法解析 fixture")
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
