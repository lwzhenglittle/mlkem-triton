"""Deprecated alias for `triton_crypto.mlkem.reference.pipeline_host_hash` (removed in 1.0)."""

from __future__ import annotations

import sys
import warnings

from .mlkem.reference import pipeline_host_hash as _impl

warnings.warn(
    "triton_crypto.pipeline_host_hash moved to triton_crypto.mlkem.reference.pipeline_host_hash; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)
sys.modules[__name__] = _impl
