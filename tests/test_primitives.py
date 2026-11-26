"""The shared primitives, CPU-only half: NTT algebra, sponge padding, hash registry.

The GPU half (kernel vs hashlib / vs numpy) is in `test_primitives_gpu.py`; this file is
on the "runs without a graphics card" whitelist in `conftest.py`.

What is actually being protected here:

- the **two domains** (`KYBER` q=3329/ζ=17/7 layers, `DILITHIUM` q=8380417/ζ=1753/8
  layers) must both be ring isomorphisms — but with *different* products: ML-KEM's
  base-case multiply in Z_q[X]/(X²−γ) and ML-DSA's plain Hadamard. Getting those two
  confused is the classic way to "unify" the transform into something that is wrong for
  one of the schemes, so both are pinned down against a schoolbook convolution;
- the inverse scale is 2^-layers, not N⁻¹ (ML-KEM normalises by 128⁻¹ = 3303);
- the sponge's padding is standard pad10*1 + domain separation.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from triton_crypto import ntt_ref as kem_ref
from triton_crypto.primitives import (
    DILITHIUM,
    HASHES,
    KYBER,
    intt_np,
    ntt_np,
    pad10star1,
    pad_batch,
    padded_len,
)
from triton_crypto.primitives.ntt import bitrev


def _rnd(n, q, seed=0):
    return np.random.default_rng(seed).integers(0, q, n, dtype=np.int64)


# ---------------- domain description ----------------


def test_domain_parameters():
    assert (KYBER.q, KYBER.n, KYBER.layers, KYBER.scale) == (3329, 256, 7, 3303)
    assert (DILITHIUM.q, DILITHIUM.n, DILITHIUM.layers) == (8380417, 256, 8)
    assert KYBER.wide is False and DILITHIUM.wide is True
    # ML-KEM's ζ has order N, ML-DSA's has order 2N: that is why one stops a layer early
    assert pow(KYBER.zeta, KYBER.n, KYBER.q) == 1
    assert pow(DILITHIUM.zeta, DILITHIUM.n, DILITHIUM.q) == DILITHIUM.q - 1


def test_zetas_are_bit_reversed_powers():
    for dom in (KYBER, DILITHIUM):
        want = [pow(dom.zeta, bitrev(k, dom.layers), dom.q) for k in range(2**dom.layers)]
        assert np.array_equal(dom.zetas, np.array(want))


def test_mlkem_zetas_still_match_fips203_table():
    # the first 32 entries of the FIPS 203 ZETAS table
    assert KYBER.zetas[0] == 1
    assert list(KYBER.zetas[1:5]) == [1729, 2580, 3289, 2642]


def test_bad_domain_is_rejected():
    from triton_crypto.primitives.ntt import NTTDomain

    with pytest.raises(ValueError):
        NTTDomain("bogus", q=3329, n=256, zeta=3, layers=7)


# ---------------- transform algebra ----------------


@pytest.mark.parametrize("dom", [KYBER, DILITHIUM], ids=lambda d: d.name)
def test_ntt_roundtrip(dom):
    x = _rnd((3, 256), dom.q, seed=1)
    assert np.array_equal(intt_np(ntt_np(x, dom), dom), x % dom.q)
    # and with values that are not pre-reduced
    assert np.array_equal(intt_np(ntt_np(x + dom.q, dom), dom), x % dom.q)


def test_mlkem_ntt_matches_independent_transcription():
    """Compare against a literal transcription of FIPS 203 Alg. 9 (kept here on purpose)."""
    f0 = _rnd(256, KYBER.q, seed=2)
    f = f0.copy()
    i, length = 1, 128
    while length >= 2:
        for start in range(0, 256, 2 * length):
            zeta = int(KYBER.zetas[i])
            i += 1
            j = np.arange(start, start + length)
            t = (zeta * f[j + length]) % KYBER.q
            f[j + length] = (f[j] - t) % KYBER.q
            f[j] = (f[j] + t) % KYBER.q
        length //= 2
    assert np.array_equal(ntt_np(f0, KYBER), f)


def test_mldsa_ntt_matches_independent_transcription():
    """Same for FIPS 204 Alg. 41 (8 layers, zetas 1..255)."""
    q = DILITHIUM.q
    f0 = _rnd(256, q, seed=3)
    f = f0.copy()
    k, length = 0, 128
    while length > 0:
        for start in range(0, 256, 2 * length):
            k += 1
            zeta = int(DILITHIUM.zetas[k])
            j = np.arange(start, start + length)
            t = (zeta * f[j + length]) % q
            f[j + length] = (f[j] - t) % q
            f[j] = (f[j] + t) % q
        length //= 2
    assert np.array_equal(ntt_np(f0, DILITHIUM), f)


def test_mlkem_basecase_product_is_a_convolution():
    a, b = _rnd(256, KYBER.q, seed=4), _rnd(256, KYBER.q, seed=5)
    got = intt_np(kem_ref.multiply_ntts(ntt_np(a, KYBER), ntt_np(b, KYBER)), KYBER)
    assert np.array_equal(got, kem_ref.negacyclic_conv(a, b))


def test_mldsa_hadamard_product_is_a_convolution():
    a, b = _rnd(256, DILITHIUM.q, seed=6), _rnd(256, DILITHIUM.q, seed=7)
    got = intt_np(DILITHIUM.mul(ntt_np(a, DILITHIUM), ntt_np(b, DILITHIUM)), DILITHIUM)
    assert np.array_equal(got, DILITHIUM.conv(a, b))


def test_toeplitz_identity_still_holds_for_both_domains():
    """T(a) @ b == a·b — the identity the ML-KEM pipeline is built on."""
    for dom in (KYBER, DILITHIUM):
        a, b = _rnd(256, dom.q, seed=8), _rnd(256, dom.q, seed=9) % 7
        assert np.array_equal((dom.toeplitz(a) @ b) % dom.q, dom.conv(a, b))


# ---------------- sponge / padding ----------------


def test_pad10star1_lengths():
    for rate in (72, 136, 168):
        for n in (0, 1, rate - 2, rate - 1, rate, rate + 1, 3 * rate):
            out = pad10star1(b"\x00" * n, rate, 0x1F)
            assert len(out) == padded_len(n, rate)
            assert len(out) % rate == 0
            assert out[n] == 0x1F or (n == len(out) - 1 and out[n] == (0x1F | 0x80))
            assert out[-1] & 0x80


def test_pad_batch_matches_pad10star1():
    for rate in (72, 136, 168):
        msgs = np.random.default_rng(0).integers(0, 256, (5, 300), dtype=np.uint8)
        batch = pad_batch(msgs, rate, 0x1F)
        for i in range(5):
            assert batch[i].tobytes() == pad10star1(msgs[i].tobytes(), rate, 0x1F)


def test_hash_registry():
    assert HASHES["SHA3-224"].rate == 144
    assert HASHES["SHA3-256"].rate == 136
    assert HASHES["SHA3-384"].rate == 104
    assert HASHES["SHA3-512"].rate == 72
    assert HASHES["SHAKE128"].rate == 168
    assert HASHES["SHAKE256"].rate == 136
    for name, hf in HASHES.items():
        msg = b"abc"
        got = hf.digest_bytes(msg, 32) if hf.outlen is None else hf.digest_bytes(msg)
        fn = getattr(hashlib, hf.ref_name)
        want = fn(msg).digest(32) if hf.outlen is None else fn(msg).digest()
        assert got == want, name
