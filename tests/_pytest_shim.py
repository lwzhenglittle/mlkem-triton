"""A minimal stand-in for pytest — injected by `run_all.py` only when pytest is
missing from the environment.

It covers the subset this repo actually uses: `mark.parametrize`, `fixture`,
`raises`, `skip`, `importorskip`, `mark.skipif`. It is **not** a replacement for
pytest; do not use it to run complex cases. When pytest is present, use pytest.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

__all__ = ["Skipped", "install"]


class Skipped(Exception):
    """Equivalent to `pytest.skip()`."""


def _split(argnames):
    if isinstance(argnames, str):
        return [n.strip() for n in argnames.split(",") if n.strip()]
    return list(argnames)


def _parametrize(argnames, argvalues, **kwargs):
    names = _split(argnames)

    def deco(fn):
        new = []
        for v in argvalues:
            vals = v if len(names) > 1 else (v,)
            new.append(dict(zip(names, vals, strict=False)))
        inner = getattr(fn, "_param_cases", None)
        if inner:
            # two stacked parametrize layers: Cartesian product
            fn._param_cases = [{**a, **b} for a in new for b in inner]
        else:
            fn._param_cases = new
        return fn

    return deco


def _skipif(cond, reason=""):
    def deco(obj):
        if cond:
            obj._skip_reason = reason or "skipped"
        return obj

    return deco


def _noop_mark(*a, **k):
    def deco(fn):
        return fn

    return deco


class _Mark:
    parametrize = staticmethod(_parametrize)
    skipif = staticmethod(_skipif)
    skip = staticmethod(lambda *a, **k: _noop_mark())
    slow = staticmethod(lambda *a, **k: _noop_mark())
    xfail = staticmethod(lambda *a, **k: _noop_mark())


def fixture(*args, **kwargs):
    def deco(fn):
        fn._is_fixture = True
        return fn

    return deco


@contextlib.contextmanager
def raises(exc, **kwargs):
    try:
        yield
    except exc:
        return
    raise AssertionError(f"expected {exc} to be raised, but nothing was")


def skip(reason="", **kwargs):
    raise Skipped(reason)


def fail(reason="", **kwargs):
    raise AssertionError(reason)


def importorskip(name, reason=None, **kwargs):
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise Skipped(reason or f"missing {name}") from e


def approx(v, rel=None, abs=None):
    """Minimal implementation that supports only `==` comparison."""

    class _Approx:
        def __eq__(self, other):
            tol = (abs if abs is not None else 1e-6) + (rel if rel is not None else 1e-6) * abs(v)
            return __builtins__["abs"](other - v) <= tol

    return _Approx()


def install() -> None:
    """Put the stand-in into `sys.modules["pytest"]` (if pytest is not installed)."""
    if "pytest" in sys.modules:
        return
    try:
        importlib.import_module("pytest")
        return
    except ImportError:
        pass
    mod = types.ModuleType("pytest")
    mod.mark = _Mark()
    mod.fixture = fixture
    mod.raises = raises
    mod.skip = skip
    mod.fail = fail
    mod.importorskip = importorskip
    mod.approx = approx
    mod.Skipped = Skipped
    sys.modules["pytest"] = mod
