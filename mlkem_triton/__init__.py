"""mlkem-triton — ML-KEM batch decapsulation in pure Triton.

    from mlkem_triton import Decapsulator, Encapsulator
    dec = Decapsulator("ML-KEM-768", dk)
    keys = dec.decaps(list_of_ciphertexts)      # -> np.ndarray [B, 32] uint8
    enc = Encapsulator("ML-KEM-768", ek)
    ct, keys = enc.encapsulate(list_of_32B_msgs)

    eks, dks = generate_keypairs("ML-KEM-768", 4096)   # all three operations covered

Design notes: see README.md; performance data: see docs/design.md (in the original repository).
"""

from __future__ import annotations

from .devices import available_devices, backend_of, describe, gemm_cfg, resolve_device
from .params import PARAMS, N, Params, Q

__version__ = "0.1.0"

__all__ = [
    "PARAMS",
    "Decapsulator",
    "Encapsulator",
    "KeyGenerator",
    "N",
    "Params",
    "Q",
    "__version__",
    "available_devices",
    "backend_of",
    "decaps_batch",
    "describe",
    "encapsulate",
    "gemm_cfg",
    "generate_keypairs",
    "resolve_device",
]


_LAZY_MODULE = {
    "Decapsulator": "pipeline",
    "decaps_batch": "pipeline",
    "Encapsulator": "encaps",
    "encapsulate": "encaps",
    "KeyGenerator": "keygen",
    "generate_keypairs": "keygen",
}


def __getattr__(name):
    # Lazy import: `import mlkem_triton` should also work on a CPU-only machine
    # (if all you want is to look at the parameter sets / use the numpy reference
    # implementation), so don't blow up just because torch has no CUDA.
    import importlib

    # Only these three entry points — do **not** write it as "iterate over every
    # possible module"; that would trigger a recursive import on `pipeline.py`'s
    # own `from . import keyprep, ops`.
    mod = _LAZY_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)
