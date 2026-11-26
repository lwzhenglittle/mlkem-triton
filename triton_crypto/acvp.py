"""Deprecated alias for `triton_crypto.mlkem.acvp` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem import acvp as _impl

warnings.warn(
    "triton_crypto.acvp moved to triton_crypto.mlkem.acvp; the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
