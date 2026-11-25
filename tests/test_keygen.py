"""Key generation (FIPS 203 Alg. 16) acceptance.

keygen is the only one of the three operations that uses the NTT, and the easiest to
get wrong (my first version wrote `G(d ‖ k)` as `G(d)`, so ρ was completely wrong).
So here we pin it down with a **byte-for-byte cross-check against kyber-py**.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from mlkem_triton.kernels.ntt import byte_encode12, forward_ntt, pow_mac
from mlkem_triton.keygen import KeyGenerator
from mlkem_triton.ntt_ref import byte_encode, multiply_ntts, ntt
from mlkem_triton.params import Q

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")

PS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


def test_forward_ntt_matches_reference():
    rng = np.random.default_rng(3)
    x = rng.integers(0, Q, (7, 256))
    got = forward_ntt(torch.from_numpy(x.astype(np.int32)).to(DEV)).cpu().numpy()
    assert np.array_equal(got, np.stack([ntt(x[i]) for i in range(7)]))


def test_pow_mac_matches_reference():
    k, B = 3, 3
    rng = np.random.default_rng(5)
    A = rng.integers(0, Q, (B * k * k, 256))
    S = rng.integers(0, Q, (B * k, 256))
    E = rng.integers(0, Q, (B * k, 256))
    f = lambda a: torch.from_numpy(a.astype(np.int32)).to(DEV)  # noqa: E731
    got = pow_mac(f(A), f(S), f(E), k).cpu().numpy()
    for b in range(B):
        for i in range(k):
            acc = E[b * k + i].astype(np.int64).copy()
            for j in range(k):
                acc = (acc + multiply_ntts(A[b * k * k + i * k + j], S[b * k + j])) % Q
            assert np.array_equal(got[b * k + i], acc), (b, i)


def test_byte_encode12_matches_reference():
    rng = np.random.default_rng(7)
    v = rng.integers(0, 4096, (4, 256))
    got = byte_encode12(torch.from_numpy(v.astype(np.int32)).to(DEV)).cpu().numpy()
    assert np.array_equal(
        got, np.stack([np.frombuffer(byte_encode(v[i], 12), np.uint8) for i in range(4)])
    )


@pytest.mark.parametrize("ps", PS)
def test_keygen_bit_exact_vs_kyberpy(ps):
    """The full 2400-byte dk must be byte-identical to kyber-py's (including the
    `‖ k` domain separation).
    """
    pytest.importorskip("kyber_py", reason="requires kyber-py for the byte-for-byte check")
    from mlkem_triton import refimpl

    d = bytes(range(32))
    z = bytes(range(32, 64))
    ek_ref, dk_ref = refimpl.ref_keygen(ps, d, z)
    kg = KeyGenerator(ps, device=DEV)
    ek, dk = kg.generate_torch(
        torch.from_numpy(np.frombuffer(d, np.uint8).copy()).to(DEV).reshape(1, 32),
        torch.from_numpy(np.frombuffer(z, np.uint8).copy()).to(DEV).reshape(1, 32),
    )
    assert ek.cpu().numpy()[0].tobytes() == ek_ref, "ek mismatch"
    assert dk.cpu().numpy()[0].tobytes() == dk_ref, "dk mismatch"


@pytest.mark.parametrize("ps", PS)
def test_keygen_batch_then_encaps_decaps(ps):
    """Batch generate → encaps with the first key → get the same K back with our own decaps."""
    pytest.importorskip("kyber_py", reason="requires kyber-py for cross-validation")
    from mlkem_triton import refimpl
    from mlkem_triton.encaps import Encapsulator
    from mlkem_triton.pipeline import Decapsulator

    eks, dks = KeyGenerator(ps, device=DEV).generate(5)
    ek, dk = eks[0].tobytes(), dks[0].tobytes()
    # kyber-py can encapsulate with our ek → so the ek is a valid FIPS 203 public key
    K, c = refimpl.ref_encaps(ps, ek)
    assert Decapsulator(ps, dk, device=DEV).decaps([c])[0].tobytes() == K
    # our encaps can also use the ek we generated
    c2, K2 = Encapsulator(ps, ek, device=DEV).encapsulate([bytes(32)])
    assert refimpl.ref_decaps(ps, dk, c2[0].tobytes()) == K2[0].tobytes()
