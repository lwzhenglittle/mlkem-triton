"""LMS / HSS on the device: the RFC vectors, batching, and the usual reject paths."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from triton_crypto.lms import params, ref
from triton_crypto.lms.verify import Verifier

VECTORS = json.loads((Path(__file__).parent.parent / "data/lms-rfc8554/vectors.json").read_text())

needs_gpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()) and not torch.cuda.is_available(),
    reason="needs a GPU",
)


def _case(case):
    return bytes.fromhex(case["pub"]), bytes.fromhex(case["msg"]), bytes.fromhex(case["sig"])


@needs_gpu
@pytest.mark.parametrize("case", VECTORS, ids=lambda c: f"rfc{c['case']}")
def test_rfc8554_vectors_on_device(case):
    pub, msg, sig = _case(case)
    assert bool(Verifier(pub).verify([msg], [sig])[0])


@needs_gpu
@pytest.mark.parametrize("case", VECTORS, ids=lambda c: f"rfc{c['case']}")
def test_rfc8554_rejects_tampered(case):
    pub, msg, sig = _case(case)
    v = Verifier(pub)
    bad = bytearray(sig)
    bad[len(bad) // 2] ^= 0x01
    assert not bool(v.verify([msg], [bytes(bad)])[0])
    assert not bool(v.verify([msg + b"."], [sig])[0])


@needs_gpu
def test_batched_matches_reference():
    case = VECTORS[0]
    pub, msg, sig = _case(case)
    v = Verifier(pub)
    n = 64
    msgs = [msg] * n
    sigs = np.stack([np.frombuffer(sig, dtype=np.uint8)] * n)
    got = v.verify(msgs, sigs)
    assert got.all()
    # a batch with one bad signature in the middle
    sigs2 = sigs.copy()
    sigs2[n // 2, 100] ^= 0x01
    got2 = v.verify(msgs, sigs2)
    assert got2.sum() == n - 1 and not got2[n // 2]


@needs_gpu
def test_mixed_message_lengths():
    case = VECTORS[0]
    pub, msg, sig = _case(case)
    v = Verifier(pub)
    # the same signature must verify for its own message regardless of what else is in the batch
    other = b"a different message"
    got = v.verify([msg, other, msg], [sig, sig, sig])
    assert got.tolist() == [True, False, True]


@needs_gpu
@pytest.mark.parametrize("ots_type", [1, 4])
def test_fresh_key_round_trip(ots_type):
    s = params.get(5, ots_type)
    seed, ident = bytes([11] * 32), bytes([22] * 16)
    lms_pub, _ = ref.keygen(seed, ident, s)
    pub = b"\x00\x00\x00\x01" + lms_pub  # an HSS wrapper with a single level
    msg = b"fresh key"
    sig = b"\x00\x00\x00\x00" + ref.lms_sign(lms_pub, seed, 1, msg, s, bytes([5]) * 32)
    assert bool(Verifier(pub).verify([msg], [sig])[0])
    bad = bytearray(sig)
    bad[-1] ^= 0x80
    assert not bool(Verifier(pub).verify([msg], [bytes(bad)])[0])


@needs_gpu
def test_two_level_hss_on_device():
    s0, s1 = params.get(5, 3), params.get(5, 4)
    seeds = [bytes([1] * 32), bytes([2] * 32)]
    idents = [bytes([3] * 16), bytes([4] * 16)]
    pub = ref.hss_keygen(seeds, idents, [s0, s1])
    msg = b"two levels"
    sig = ref.hss_sign(seeds, idents, [s0, s1], [0, 1], msg, [bytes([9]) * 32, bytes([8]) * 32])
    assert bool(Verifier(pub).verify([msg], [sig])[0])
    assert not bool(Verifier(pub).verify([b"nope"], [sig])[0])


@needs_gpu
def test_empty_batch():
    pub, _, _ = _case(VECTORS[0])
    assert Verifier(pub).verify([], []).size == 0
