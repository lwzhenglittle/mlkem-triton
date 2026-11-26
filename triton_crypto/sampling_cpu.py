"""Deprecated alias for `triton_crypto.mlkem.reference.sampling_cpu` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem.reference import sampling_cpu as _impl

warnings.warn(
    "triton_crypto.sampling_cpu moved to triton_crypto.mlkem.reference.sampling_cpu; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
