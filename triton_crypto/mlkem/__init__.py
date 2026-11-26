"""ML-KEM (FIPS 203): key generation, encapsulation and decapsulation.

The scheme is a thin layer over `triton_crypto.primitives`; everything here is
ML-KEM-specific: the Toeplitz matrices for the two GEMM passes, the base-case
multiply in the quadratic extension, ByteEncode/Compress and CBD.

    from triton_crypto.mlkem import Decapsulator, Encapsulator, generate_keypairs

    dec = Decapsulator("ML-KEM-768", dk)
    keys = dec.decaps(ciphertexts)
    cts, keys = Encapsulator("ML-KEM-768", ek).encapsulate(messages)

`triton_crypto` re-exports the same entry points for backwards compatibility.
"""

from __future__ import annotations

from .params import PARAMS, N, Params, Q

__all__ = [
    "PARAMS",
    "Decapsulator",
    "Encapsulator",
    "KeyGenerator",
    "N",
    "Params",
    "Q",
    "decaps_batch",
    "encapsulate",
    "generate_keypairs",
]

_LAZY = {
    "Decapsulator": ".pipeline",
    "decaps_batch": ".pipeline",
    "Encapsulator": ".encaps",
    "encapsulate": ".encaps",
    "KeyGenerator": ".keygen",
    "generate_keypairs": ".keygen",
}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(mod, __name__), name)
