"""Portable post-quantum crypto kernels in pure Triton.

Layout:

    primitives/   scheme-agnostic mathematics: Keccak-f[1600], the sponge
                  (SHA3/SHAKE, any rate/output, batched), SHA-2 compression and a
                  parameterised negacyclic NTT over Z_q[X]/(X^n+1)
    mlkem/        ML-KEM (FIPS 203): keygen, encapsulate, decapsulate
    mldsa/        ML-DSA (FIPS 204): sign, verify, plus a numpy reference
    slhdsa/       SLH-DSA (FIPS 205): verify, plus a numpy reference
    lms/          LMS/HSS (SP 800-208): verify, plus a numpy reference

Every scheme is a thin layer over `primitives`; it keeps only what is
scheme-specific (ML-KEM's Toeplitz GEMMs, ML-DSA's SampleInBall/Decompose,
FIPS 205's thash, LMS Winternitz chains).

ML-KEM, all three operations (also available from `triton_crypto.mlkem`):

    from triton_crypto import Decapsulator, Encapsulator, generate_keypairs
    dec = Decapsulator("ML-KEM-768", dk)
    keys = dec.decaps(list_of_ciphertexts)      # -> np.ndarray [B, 32] uint8
    enc = Encapsulator("ML-KEM-768", ek)
    ct, keys = enc.encapsulate(list_of_32B_msgs)
    eks, dks = generate_keypairs("ML-KEM-768", 4096)

    from triton_crypto.mldsa import Signer, Verifier
    from triton_crypto.slhdsa import Verifier
    from triton_crypto.lms import Verifier

The `mlkem_triton` name still imports this same package (deprecated until 1.0).

Kernel contracts: docs/kernels.md. Design notes: README.md, docs/design.md.
Measurements, and how they were taken: docs/benchmarks.md.
"""

from __future__ import annotations

from .devices import available_devices, backend_of, describe, gemm_cfg, resolve_device
from .mlkem.params import PARAMS, N, Params, Q

__version__ = "0.8.0"

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
    "Decapsulator": "mlkem.pipeline",
    "decaps_batch": "mlkem.pipeline",
    "Encapsulator": "mlkem.encaps",
    "encapsulate": "mlkem.encaps",
    "KeyGenerator": "mlkem.keygen",
    "generate_keypairs": "mlkem.keygen",
}


def __getattr__(name):
    # Lazy import: `import triton_crypto` should also work on a CPU-only machine
    # (if all you want is to look at the parameter sets / use the numpy reference
    # implementation), so don't blow up just because torch has no CUDA.
    import importlib

    mod = _LAZY_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)
