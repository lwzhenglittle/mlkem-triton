"""Tests for `triton_crypto.uov` (OV-Is, OV-Ip, OV-III, OV-V) against offline & live liboqs KATs."""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
import pytest
from _common import DEV

from triton_crypto.uov import ALL_SETS, BY_NAME, Verifier, verify_batch, verify_np

KAT_PATH = Path(__file__).resolve().parent.parent / "data" / "vectors" / "uov" / "kat.json"
KATS = json.loads(KAT_PATH.read_text())["vectors"]
LIBOQS_MQ = (
    Path(__file__).resolve().parent.parent
    / "third_party"
    / "liboqs"
    / "build-mq"
    / "lib"
    / "liboqs.so"
)


@pytest.mark.parametrize("entry", KATS, ids=lambda e: e["alg"])
def test_uov_ref_matches_offline_liboqs_kats(entry):
    ps = BY_NAME[entry["alg"]]
    pk = bytes.fromhex(entry["pk"])
    case = entry["cases"][0]
    msg = bytes.fromhex(case["msg"])
    sig = bytes.fromhex(case["sig"])
    assert verify_np(ps, pk, msg, sig)
    assert not verify_np(ps, pk, msg, bytes([sig[0] ^ 1]) + sig[1:])
    assert not verify_np(ps, pk, msg + b"\x01", sig)


@pytest.mark.gpu
@pytest.mark.parametrize("entry", KATS, ids=lambda e: e["alg"])
def test_uov_gpu_matches_offline_liboqs_kats(entry):
    pk = bytes.fromhex(entry["pk"])
    v = Verifier(pk, device=DEV)
    assert v.ps == BY_NAME[entry["alg"]]

    msgs = [bytes.fromhex(c["msg"]) for c in entry["cases"]]
    sigs = [bytes.fromhex(c["sig"]) for c in entry["cases"]]
    bad_sig = bytes([sigs[0][0] ^ 0x01]) + sigs[0][1:]
    bad_salt = sigs[0][:-1] + bytes([sigs[0][-1] ^ 0x01])
    ok = v.verify([*msgs, msgs[0], msgs[0], msgs[0] + b"x"], [*sigs, bad_sig, bad_salt, sigs[0]])
    np.testing.assert_array_equal(ok, [True] * len(msgs) + [False, False, False])

    # Uniform-length batch (> BLOCK_B=128) exercising GPU SHAKE256 + GPU quadratic map
    m0, s0 = msgs[-1], sigs[-1]
    batch = 131
    u_msgs = [m0] * batch
    u_sigs = [s0] * batch
    u_sigs[5] = bytes([s0[0] ^ 0x10]) + s0[1:]
    u_sigs[128] = s0[:-1] + bytes([s0[-1] ^ 0x01])
    u_ok = verify_batch(pk, u_msgs, u_sigs, device=DEV)
    u_exp = np.ones(batch, dtype=bool)
    u_exp[5] = False
    u_exp[128] = False
    np.testing.assert_array_equal(u_ok, u_exp)

    assert v.verify([], []).shape == (0,)


@pytest.mark.gpu
@pytest.mark.parametrize("alg", ["OV-Is", "OV-Ip", "OV-III", "OV-V"])
def test_uov_gpu_classic_live_liboqs(alg: str):
    """Cross-check all four classic uncompressed UOV sets against live liboqs when built."""
    if not LIBOQS_MQ.exists():
        pytest.skip("optional liboqs build-mq not present")
    lib = ctypes.CDLL(str(LIBOQS_MQ))

    class _OQS_SIG(ctypes.Structure):
        _fields_ = [
            ("method_name", ctypes.c_char_p),
            ("alg_version", ctypes.c_char_p),
            ("claimed_nist_level", ctypes.c_uint8),
            ("euf_cma", ctypes.c_bool),
            ("suf_cma", ctypes.c_bool),
            ("sig_with_ctx_support", ctypes.c_bool),
            ("length_public_key", ctypes.c_size_t),
            ("length_secret_key", ctypes.c_size_t),
            ("length_signature", ctypes.c_size_t),
        ]

    lib.OQS_SIG_new.argtypes = [ctypes.c_char_p]
    lib.OQS_SIG_new.restype = ctypes.POINTER(_OQS_SIG)
    lib.OQS_SIG_keypair.argtypes = [ctypes.POINTER(_OQS_SIG), ctypes.c_char_p, ctypes.c_char_p]
    lib.OQS_SIG_sign.argtypes = [
        ctypes.POINTER(_OQS_SIG),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    lib.OQS_SIG_free.argtypes = [ctypes.POINTER(_OQS_SIG)]

    obj = lib.OQS_SIG_new(alg.encode())
    assert obj, f"liboqs missing {alg}"
    try:
        s = obj.contents
        pk = ctypes.create_string_buffer(s.length_public_key)
        sk = ctypes.create_string_buffer(s.length_secret_key)
        sig = ctypes.create_string_buffer(s.length_signature)
        siglen = ctypes.c_size_t(0)
        msg = b"live classic uov test"
        assert lib.OQS_SIG_keypair(obj, pk, sk) == 0
        assert lib.OQS_SIG_sign(obj, sig, ctypes.byref(siglen), msg, len(msg), sk) == 0
        sig_b = sig.raw[: siglen.value]
        v = Verifier(pk.raw, device=DEV)
        assert v.ps.name == alg
        bad = bytes([sig_b[0] ^ 1]) + sig_b[1:]
        np.testing.assert_array_equal(v.verify([msg, msg], [sig_b, bad]), [True, False])
    finally:
        lib.OQS_SIG_free(obj)


def test_uov_param_sets_complete():
    assert [p.name for p in ALL_SETS] == ["OV-Is", "OV-Ip", "OV-III", "OV-V"]
