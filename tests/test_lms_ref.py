"""LMS / HSS reference: parameter sets, the RFC 8554 test vectors, and sign/verify round-trips."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from triton_crypto.lms import params, ref

VECTORS = json.loads((Path(__file__).parent.parent / "data/lms-rfc8554/vectors.json").read_text())


def test_parameter_sizes():
    assert params.get(5, 4).sig_size == 1292  # H5/W8
    assert params.get(6, 3).sig_size == 2508  # H10/W4
    for s in params.ALL_SETS:
        assert s.pk_size == 56
        assert s.ots_sig_size == 4 + 32 * (s.p + 1)
        assert s.sig_size == 12 + 32 * (s.p + 1) + 32 * s.h


def test_parameter_sets_cover_the_standard():
    assert sorted(params.LMS_TYPES) == [5, 6, 7, 8, 9]
    assert sorted(params.OTS_TYPES) == [1, 2, 3, 4]
    assert len(params.ALL_SETS) == 20
    assert params.from_name("H10W4").sig_size == 2508


@pytest.mark.parametrize("case", VECTORS, ids=lambda c: f"rfc{c['case']}")
def test_rfc8554_test_vectors(case):
    pub = bytes.fromhex(case["pub"])
    msg = bytes.fromhex(case["msg"])
    sig = bytes.fromhex(case["sig"])
    assert ref.hss_verify(pub, msg, sig)


@pytest.mark.parametrize("case", VECTORS, ids=lambda c: f"rfc{c['case']}")
def test_rfc8554_vectors_reject_tampering(case):
    pub = bytes.fromhex(case["pub"])
    msg = bytes.fromhex(case["msg"])
    sig = bytearray.fromhex(case["sig"])
    sig[len(sig) // 2] ^= 0x01
    assert not ref.hss_verify(pub, msg, bytes(sig))
    sig2 = bytearray.fromhex(case["sig"])
    assert not ref.hss_verify(pub, msg + b"x", bytes(sig2))
    sig3 = bytearray.fromhex(case["sig"])
    sig3[0] ^= 0x01  # wrong number of levels
    assert not ref.hss_verify(pub, msg, bytes(sig3))


@pytest.mark.parametrize("ots_type", [1, 2, 3, 4])
def test_round_trip_all_winternitz_factors(ots_type):
    s = params.get(5, ots_type)
    seed, ident = bytes(range(32)), bytes(range(16, 32))
    pub, _ = ref.keygen(seed, ident, s)
    msg = b"the powers not delegated"
    sig = ref.lms_sign(pub, seed, 3, msg, s, bytes([7]) * 32)
    assert ref.lms_verify(pub, msg, sig)
    assert not ref.lms_verify(pub, msg + b".", sig)
    assert not ref.lms_verify(pub, msg, sig[:-1])


def test_hss_two_levels_with_different_sets():
    s0, s1 = params.get(5, 3), params.get(5, 4)
    seeds = [bytes([1] * 32), bytes([2] * 32)]
    idents = [bytes([3] * 16), bytes([4] * 16)]
    pub = ref.hss_keygen(seeds, idents, [s0, s1])
    msg = b"hss"
    sig = ref.hss_sign(seeds, idents, [s0, s1], [1, 2], msg, [bytes([9]) * 32, bytes([8]) * 32])
    assert ref.hss_verify(pub, msg, sig)
    assert not ref.hss_verify(pub, b"other", sig)
    assert struct.unpack(">I", sig[:4])[0] == 1  # Nspk
    assert len(sig) == 4 + s0.sig_size + s0.pk_size + s1.sig_size
