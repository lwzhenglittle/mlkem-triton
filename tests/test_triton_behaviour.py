"""Triton behaviours this project's documents assert, checked against the installed backend.

Two of the traps recorded in `docs/patterns.md` are claims about the *toolchain*, not about this
code: how `%` treats negative integers, and whether `tl.sort` sorts. A claim like that can go stale
when the wheel changes, and nothing would notice -- so the claims that are cheap to check are tested
here, and a backend update that fixes (or breaks) one turns red instead of silently invalidating a
document.

The `tl.sort` entry is narrowed relative to the original note: 1-D and last-axis sorts are correct
on this build, and `axis=0` raises a `CompilationError` rather than returning a wrong answer.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _sort1d(X, OUT, N: tl.constexpr, DESC: tl.constexpr):
    v = tl.load(X + tl.arange(0, N))
    tl.store(OUT + tl.arange(0, N), tl.sort(v, descending=DESC))


@triton.jit
def _sort2d_last(X, OUT, B: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, B)
    cols = tl.arange(0, N)
    v = tl.load(X + rows[:, None] * N + cols[None, :])
    tl.store(OUT + rows[:, None] * N + cols[None, :], tl.sort(v, 1))


@triton.jit
def _sort2d_axis0(X, OUT, B: tl.constexpr, N: tl.constexpr):
    rows = tl.arange(0, B)
    cols = tl.arange(0, N)
    v = tl.load(X + rows[:, None] * N + cols[None, :])
    tl.store(OUT + rows[:, None] * N + cols[None, :], tl.sort(v, 0))


@pytest.mark.parametrize("descending", [False, True])
def test_one_dimensional_sort_is_correct(descending):
    dev = torch.device("xpu")
    data = np.random.default_rng(0).integers(0, 1000, 256).astype(np.int32)
    out = torch.zeros(256, dtype=torch.int32, device=dev)
    _sort1d[(1,)](torch.as_tensor(data, device=dev), out, N=256, DESC=descending)
    torch.accelerator.synchronize()
    want = np.sort(data)
    if descending:
        want = want[::-1]
    assert np.array_equal(out.cpu().numpy(), want)


def test_two_dimensional_sort_along_the_last_axis_is_correct():
    dev = torch.device("xpu")
    data = np.random.default_rng(1).integers(0, 1000, (8, 64)).astype(np.int32)
    out = torch.zeros((8, 64), dtype=torch.int32, device=dev)
    _sort2d_last[(1,)](torch.as_tensor(data, device=dev), out, B=8, N=64)
    torch.accelerator.synchronize()
    assert np.array_equal(out.cpu().numpy(), np.sort(data, axis=1))


def test_two_dimensional_sort_along_axis_zero_is_refused_loudly():
    """The original note said "silently wrong"; on this build it does not compile, which is the
    better failure mode and the one worth pinning."""
    dev = torch.device("xpu")
    data = np.zeros((8, 64), dtype=np.int32)
    out = torch.zeros((8, 64), dtype=torch.int32, device=dev)
    with pytest.raises(Exception):  # noqa: B017 - the point is that it raises at all
        _sort2d_axis0[(1,)](torch.as_tensor(data, device=dev), out, B=8, N=64)
        torch.accelerator.synchronize()


@triton.jit
def _mod_kernel(X, OUT, N: tl.constexpr, Q: tl.constexpr):
    off = tl.arange(0, N)
    v = tl.load(X + off)
    tl.store(OUT + off, v % Q)


def test_percent_on_a_negative_int_is_the_truncated_remainder():
    """`patterns.md` trap 1: Triton's `%` is C's truncated remainder, not numpy's floored modulo.

    This is why every modular subtraction in this repository is written as a conditional add
    rather than as `(a - b) % q`.
    """
    dev = torch.device("xpu")
    q = 12289
    data = np.array([-1, -q, -q - 1, 5, 0], dtype=np.int32)
    out = torch.zeros(len(data), dtype=torch.int32, device=dev)
    _mod_kernel[(1,)](torch.as_tensor(data, device=dev), out, N=8, Q=q)
    torch.accelerator.synchronize()
    got = out.cpu().numpy()[: len(data)]
    # the backend keeps the sign of the dividend (C semantics), numpy floors
    assert got.tolist() == [-1, 0, -1, 5, 0], got.tolist()
    assert (data % q).tolist() == [q - 1, 0, q - 1, 5, 0]
    assert got.tolist() != (data % q).tolist(), "if these agree the trap is gone; update the docs"
