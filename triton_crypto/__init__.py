"""Portable post-quantum crypto kernels in pure Triton.

Two layers, deliberately separable:

`triton_crypto.primitives` — the mathematics, useful on its own
    keccak      Keccak-f[1600] permutation, plus a pure-Python reference
    sponge      pad10*1 and the named SHA3/SHAKE instances, batched
    ntt         negacyclic NTT over Z_q[X]/(X^n+1) for any q and 2N-th root of unity,
                forward and inverse, in the standard and the Montgomery domain

`triton_crypto` (ML-KEM, FIPS 203) and `triton_crypto.mldsa` (ML-DSA, FIPS 204) are thin
schemes on top; they keep only what is scheme-specific (ML-KEM's base-case multiply in the
quadratic extension, ML-DSA's SampleInBall / Decompose / hint encoding).

ML-KEM, all three operations:

    from triton_crypto import Decapsulator, Encapsulator, generate_keypairs
    dec = Decapsulator("ML-KEM-768", dk)
    keys = dec.decaps(list_of_ciphertexts)      # -> np.ndarray [B, 32] uint8
    enc = Encapsulator("ML-KEM-768", ek)
    ct, keys = enc.encapsulate(list_of_32B_msgs)
    eks, dks = generate_keypairs("ML-KEM-768", 4096)

ML-DSA verification:

    from triton_crypto.mldsa import Verifier
    ok = Verifier("ML-DSA-65", pk).verify(messages, signatures)

The `mlkem_triton` name still imports this same package (see that module's docstring).

Kernel contracts: docs/kernels.md. Design notes: README.md, docs/design.md.
Measurements, and how they were taken: docs/benchmarks.md.
"""

from __future__ import annotations

from .devices import available_devices, backend_of, describe, gemm_cfg, resolve_device
from .params import PARAMS, N, Params, Q

__version__ = "0.7.2"

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
    # Lazy import: `import triton_crypto` should also work on a CPU-only machine
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
