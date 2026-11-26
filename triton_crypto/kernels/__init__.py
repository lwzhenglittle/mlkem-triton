"""Deprecated alias for `triton_crypto.mlkem.kernels` (removed in 1.0)."""

from __future__ import annotations

import importlib
import sys
import warnings

warnings.warn(
    "triton_crypto.kernels moved to triton_crypto.mlkem.kernels; "
    "the old path will be removed in 1.0",
    DeprecationWarning,
    stacklevel=2,
)

_SUBMODULES = (
    "elementwise",
    "encaps",
    "fused_gemm",
    "gemm_modq",
    "glue",
    "ntt",
    "padcat",
    "rebuild",
)
for _name in _SUBMODULES:
    _mod = importlib.import_module(f"triton_crypto.mlkem.kernels.{_name}")
    sys.modules[f"{__name__}.{_name}"] = _mod
    globals()[_name] = _mod
__all__ = list(_SUBMODULES)
