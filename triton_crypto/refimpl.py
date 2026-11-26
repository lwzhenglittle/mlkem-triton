"""Deprecated alias for `triton_crypto.mlkem.reference.refimpl` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem.reference import refimpl as _impl

warnings.warn(
    "triton_crypto.refimpl moved to triton_crypto.mlkem.reference.refimpl; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
