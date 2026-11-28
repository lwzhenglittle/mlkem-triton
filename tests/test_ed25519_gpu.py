"""Tests for `triton_crypto.ed25519` (RFC 7748 X25519 and RFC 8032 Ed25519)."""

from __future__ import annotations

import numpy as np
import pytest
from _common import DEV

from triton_crypto.ed25519 import (
    CURVE_L,
    Verifier,
    ed25519_public_key_np,
    ed25519_sign_np,
    ed25519_verify_np,
    verify_batch,
    x25519_base_np,
    x25519_base_torch,
    x25519_np,
    x25519_torch,
)

# Official RFC 7748 §5.2 & §6.1 test vectors
RFC7748_VECTORS = [
    (
        bytes.fromhex("a546e36bf0527c9d3b16154b82465edd62144c0ac1fc5a18506a2244ba449ac4"),
        bytes.fromhex("e6db6867583030db3594c1a424b15f7c726624ec26b3353b10a903a6d0ab1c4c"),
        bytes.fromhex("c3da55379de9c6908e94ea4df28d084f32eccf03491c71f754b4075577a28552"),
    ),
    (
        bytes.fromhex("4b66e9d4d1b4673c5ad22691957d6af5c11b6421e0ea01d42ca4169e7918ba0d"),
        bytes.fromhex("e5210f12786811d3f4b7959d0538ae2c31dbe7106fc03c3efc4cd549c715a493"),
        bytes.fromhex("95cbde9476e8907d7aade45cb4b873f88b595a68799fa152e6f8f7647aac7957"),
    ),
    # §6.1 Alice privkey * basepoint (9) = Alice pubkey
    (
        bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"),
        (9).to_bytes(32, "little"),
        bytes.fromhex("8520f0098930a754748b7ddcb43ef75a0dbf3a0d26381af4eba4a98eaa9b4e6a"),
    ),
    # §6.1 Bob privkey * basepoint (9) = Bob pubkey
    (
        bytes.fromhex("5dab087e624a8a4b79e17f8b83800ee66f3bb1292618b6fd1c2f8b27ff88e0eb"),
        (9).to_bytes(32, "little"),
        bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"),
    ),
    # §6.1 Shared secret
    (
        bytes.fromhex("77076d0a7318a57d3c16c17251b26645df4c2f87ebc0992ab177fba51db92c2a"),
        bytes.fromhex("de9edb7d7b7dc1b4d35b61c2ece435373f8343c85b78674dadfc7e146f882b4f"),
        bytes.fromhex("4a5d9d5ba4ce2de1728e3bf480350f25e07e21c947d19e3376f09b3c1e161742"),
    ),
]

# Official RFC 8032 §7.1 Test Vectors 1, 2, 3
RFC8032_VECTORS = [
    (
        bytes.fromhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60"),
        bytes.fromhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"),
        b"",
        bytes.fromhex(
            "e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
            "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"
        ),
    ),
    (
        bytes.fromhex("4ccd089b28ff96da9db6c346ec114e0f5b8a319f35aba624da8cf6ed4fb8a6fb"),
        bytes.fromhex("3d4017c3e843895a92b70aa74d1b7ebc9c982ccf2ec4968cc0cd55f12af4660c"),
        bytes.fromhex("72"),
        bytes.fromhex(
            "92a009a9f0d4cab8720e820b5f642540a2b27b5416503f8fb3762223ebdb69da"
            "085ac1e43e15996e458f3613d0f11d8c387b2eaeb4302aeeb00d291612bb0c00"
        ),
    ),
    (
        bytes.fromhex("c5aa8df43f9f837bedb7442f31dcb7b166d38535076f094b85ce3a2e0b4458f7"),
        bytes.fromhex("fc51cd8e6218a1a38da47ed00230f0580816ed13ba3303ac5deb911548908025"),
        bytes.fromhex("af82"),
        bytes.fromhex(
            "6291d657deec24024827e69c3abe01a30ce548a284743a445e3680d7db5ac3ac"
            "18ff9b538d16f290ae67f760984dc6594a7c15e9716ed28dc027beceea1ec40a"
        ),
    ),
]


@pytest.mark.parametrize(("scalar", "u", "expected"), RFC7748_VECTORS)
def test_x25519_ref_rfc7748_vectors(scalar: bytes, u: bytes, expected: bytes):
    assert x25519_np(scalar, u) == expected


@pytest.mark.parametrize(("seed", "pk", "msg", "sig"), RFC8032_VECTORS)
def test_ed25519_ref_rfc8032_vectors(seed: bytes, pk: bytes, msg: bytes, sig: bytes):
    assert ed25519_public_key_np(seed) == pk
    assert ed25519_sign_np(seed, msg) == sig
    assert ed25519_verify_np(pk, msg, sig)
    # Tamper message, R, or s
    assert not ed25519_verify_np(pk, msg + b"\x00", sig)
    assert not ed25519_verify_np(pk, msg, bytes([sig[0] ^ 1]) + sig[1:])
    assert not ed25519_verify_np(pk, msg, sig[:32] + CURVE_L.to_bytes(32, "little"))


@pytest.mark.gpu
def test_x25519_gpu_rfc7748_and_random_batch():
    scalars = [v[0] for v in RFC7748_VECTORS]
    us = [v[1] for v in RFC7748_VECTORS]
    expected = [v[2] for v in RFC7748_VECTORS]
    got = x25519_torch(scalars, us, device=DEV).cpu().numpy()
    for i, exp in enumerate(expected):
        assert bytes(got[i]) == exp

    # Base-point broadcast + non-multiple-of-64 batch
    rng = np.random.default_rng(7748)
    batch = 73
    sc = rng.integers(0, 256, size=(batch, 32), dtype=np.uint8)
    got_base = x25519_base_torch(sc, device=DEV).cpu().numpy()
    for i in (0, 1, 31, 63, 64, 72):
        assert bytes(got_base[i]) == x25519_base_np(bytes(sc[i]))


@pytest.mark.gpu
def test_ed25519_gpu_rfc8032_single_and_multi_key():
    # Single-key verifier on all 3 RFC 8032 vectors + tampered cases
    for _seed, pk, msg, sig in RFC8032_VECTORS:
        v = Verifier(pk, device=DEV)
        bad_r = bytes([sig[0] ^ 1]) + sig[1:]
        bad_s = sig[:32] + CURVE_L.to_bytes(32, "little")
        ok = v.verify([msg, msg, msg, msg + b"x"], [sig, bad_r, bad_s, sig])
        np.testing.assert_array_equal(ok, [True, False, False, False])

    # Multi-key batch across variable-length messages
    pks = [v[1] for v in RFC8032_VECTORS]
    msgs = [v[2] for v in RFC8032_VECTORS]
    sigs = [v[3] for v in RFC8032_VECTORS]
    ok_multi = verify_batch(pks, msgs, sigs, device=DEV)
    np.testing.assert_array_equal(ok_multi, [True, True, True])


@pytest.mark.gpu
def test_ed25519_gpu_uniform_batch_matches_ref():
    rng = np.random.default_rng(8032)
    seed = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
    pk = ed25519_public_key_np(seed)
    batch = 71  # non-multiple of BLOCK=64
    msgs = [rng.integers(0, 256, size=64, dtype=np.uint8).tobytes() for _ in range(batch)]
    sigs = [ed25519_sign_np(seed, m) for m in msgs]
    # Corrupt a few signatures (bad R, bad s, s >= L)
    expected = np.ones(batch, dtype=bool)
    sigs[5] = bytes([sigs[5][0] ^ 0x80]) + sigs[5][1:]
    expected[5] = False
    sigs[19] = sigs[19][:32] + bytes([sigs[19][32] ^ 1]) + sigs[19][33:]
    expected[19] = False
    sigs[64] = sigs[64][:32] + CURVE_L.to_bytes(32, "little")
    expected[64] = False

    v = Verifier(pk, device=DEV)
    got = v.verify(msgs, sigs)
    np.testing.assert_array_equal(got, expected)

    # Empty batch edge case
    assert v.verify([], []).shape == (0,)
