"""Shared test helpers: device resolution.

The `MLKEM_DEVICE` environment variable (or `pytest --device`) decides which
card the tests run on; if neither is given, the first available accelerator is
selected automatically.
"""

from __future__ import annotations

from mlkem_triton.devices import describe, resolve_device

try:
    DEV = resolve_device()
    HAS_GPU = True
except RuntimeError:  # pure CPU machine: only run the cases that do not need a GPU
    DEV = None
    HAS_GPU = False


def dev_name() -> str:
    return describe(DEV) if HAS_GPU else "(no GPU)"
