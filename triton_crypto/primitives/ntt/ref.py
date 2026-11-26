"""numpy entry points for the negacyclic NTT (the reference side of the transform)."""

from __future__ import annotations

import numpy as np

from .domain import KYBER, NTTDomain

__all__ = ["intt_np", "mul_np", "negacyclic_conv_np", "ntt_np"]


def ntt_np(x, domain: NTTDomain = KYBER) -> np.ndarray:
    """[..., N] integers -> NTT domain (int64)."""
    x = np.asarray(x, dtype=np.int64)
    if x.shape[-1] != domain.n:
        raise ValueError(f"expected last axis {domain.n}, got {x.shape}")
    flat = x.reshape(-1, domain.n)
    return np.stack([domain.ntt(row) for row in flat]).reshape(x.shape)


def intt_np(x, domain: NTTDomain = KYBER) -> np.ndarray:
    """[..., N] NTT-domain integers -> coefficients (int64, standard domain)."""
    x = np.asarray(x, dtype=np.int64)
    if x.shape[-1] != domain.n:
        raise ValueError(f"expected last axis {domain.n}, got {x.shape}")
    flat = x.reshape(-1, domain.n)
    return np.stack([domain.intt(row) for row in flat]).reshape(x.shape)


def mul_np(a, b, domain: NTTDomain = KYBER) -> np.ndarray:
    """Pointwise product in the NTT domain."""
    return domain.mul(a, b)


def negacyclic_conv_np(a, b, domain: NTTDomain = KYBER) -> np.ndarray:
    """Schoolbook product in Z_q[X]/(X^N+1), for tests."""
    return domain.conv(a, b)
