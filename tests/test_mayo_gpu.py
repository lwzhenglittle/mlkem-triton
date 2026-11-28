"""Tests for `triton_crypto.mayo` (MAYO-1, MAYO-2, MAYO-3, MAYO-5) against offline liboqs KATs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from _common import DEV

from triton_crypto.mayo import ALL_SETS, BY_NAME, Verifier, verify_batch, verify_np

KAT_PATH = Path(__file__).resolve().parent.parent / "data" / "vectors" / "mayo" / "kat.json"
KATS = json.loads(KAT_PATH.read_text())["vectors"]


@pytest.mark.parametrize("entry", KATS, ids=lambda e: e["alg"])
def test_mayo_ref_matches_offline_liboqs_kats(entry):
    ps = BY_NAME[entry["alg"]]
    pk = bytes.fromhex(entry["pk"])
    case = entry["cases"][0]
    msg = bytes.fromhex(case["msg"])
    sig = bytes.fromhex(case["sig"])
    assert verify_np(ps, pk, msg, sig)
    # Tamper one bit of signature and one bit of message
    assert not verify_np(ps, pk, msg, bytes([sig[0] ^ 1]) + sig[1:])
    assert not verify_np(ps, pk, msg + b"\x01", sig)


@pytest.mark.gpu
@pytest.mark.parametrize("entry", KATS, ids=lambda e: e["alg"])
def test_mayo_gpu_matches_offline_liboqs_kats(entry):
    pk = bytes.fromhex(entry["pk"])
    v = Verifier(pk, device=DEV)
    assert v.ps == BY_NAME[entry["alg"]]

    # Variable-length batch (valid + tampered)
    msgs = [bytes.fromhex(c["msg"]) for c in entry["cases"]]
    sigs = [bytes.fromhex(c["sig"]) for c in entry["cases"]]
    bad_sig = bytes([sigs[0][0] ^ 0x01]) + sigs[0][1:]
    bad_salt = sigs[0][:-1] + bytes([sigs[0][-1] ^ 0x01])
    ok = v.verify([*msgs, msgs[0], msgs[0], msgs[0] + b"x"], [*sigs, bad_sig, bad_salt, sigs[0]])
    expected = [True] * len(msgs) + [False, False, False]
    np.testing.assert_array_equal(ok, expected)

    # Uniform-length batch (> BLOCK_B=128) exercising GPU SHAKE256 + GPU SPS + GPU fold
    m0, s0 = msgs[-1], sigs[-1]
    batch = 131
    u_msgs = [m0] * batch
    u_sigs = [s0] * batch
    u_sigs[7] = bytes([s0[0] ^ 0x10]) + s0[1:]
    u_sigs[129] = s0[:-1] + bytes([s0[-1] ^ 0x01])
    u_ok = verify_batch(pk, u_msgs, u_sigs, device=DEV)
    u_exp = np.ones(batch, dtype=bool)
    u_exp[7] = False
    u_exp[129] = False
    np.testing.assert_array_equal(u_ok, u_exp)

    # Empty batch edge case
    assert v.verify([], []).shape == (0,)


def test_mayo_param_sets_complete():
    assert [p.name for p in ALL_SETS] == ["MAYO-1", "MAYO-2", "MAYO-3", "MAYO-5"]
