"""Deprecated alias for `triton_crypto.mlkem.ntt_ref` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem import ntt_ref as _impl

warnings.warn(
    "triton_crypto.ntt_ref moved to triton_crypto.mlkem.ntt_ref; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
