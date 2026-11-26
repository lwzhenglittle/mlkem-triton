"""Deprecated alias for `triton_crypto.mlkem.ops` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem import ops as _impl

warnings.warn(
    "triton_crypto.ops moved to triton_crypto.mlkem.ops; the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
