"""Deprecated alias for `triton_crypto.mlkem.pipeline` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem import pipeline as _impl

warnings.warn(
    "triton_crypto.pipeline moved to triton_crypto.mlkem.pipeline; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
