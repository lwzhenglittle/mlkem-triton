"""Deprecated alias for `triton_crypto.mlkem.gpu_hash` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem import gpu_hash as _impl

warnings.warn(
    "triton_crypto.gpu_hash moved to triton_crypto.mlkem.gpu_hash; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
