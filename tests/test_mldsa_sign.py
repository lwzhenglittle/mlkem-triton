"""ML-DSA signing on the device: byte-exact against the reference and the ACVP sigGen vectors."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import torch

from triton_crypto.mldsa.ref import MLDSA
from triton_crypto.mldsa.sign import Signer
from triton_crypto.mldsa.verify import Verifier

PS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
DIR = Path(__file__).parent.parent / "data/acvp-mldsa"

needs_gpu = pytest.mark.skipif(
    not (hasattr(torch, "xpu") and torch.xpu.is_available()) and not torch.cuda.is_available(),
    reason="needs a GPU",
)


def _acvp_siggen():
    prompt = json.loads((DIR / "ML-DSA-sigGen-FIPS204-prompt.json").read_text())
    exp = json.loads((DIR / "ML-DSA-sigGen-FIPS204-expectedResults.json").read_text())
    expmap = {(g["tgId"], t["tcId"]): t for g in exp["testGroups"] for t in g["tests"]}
    out = []
    for g in prompt["testGroups"]:
        for t in g["tests"]:
            out.append({**t, "group": g, "expected": expmap[(g["tgId"], t["tcId"])]["signature"]})
    return out


ACVP = _acvp_siggen()


@needs_gpu
@pytest.mark.parametrize("ps", PS)
def test_matches_reference_byte_for_byte(ps):
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    msg = b"deterministic signing must be reproducible"
    want = m.sign(sk, msg, rnd=None)
    got = Signer(ps, sk).sign([msg])[0].tobytes()
    assert got == want
    assert m.verify(pk, msg, got)


@needs_gpu
@pytest.mark.parametrize("ps", PS)
def test_hedged_signing_matches_reference(ps):
    m = MLDSA(ps)
    _pk, sk = m.keygen(bytes(range(32)))
    rnd = bytes(range(32))
    msg = b"hedged"
    want = m.sign(sk, msg, rnd=rnd)
    got = Signer(ps, sk).sign([msg], rnd=rnd)[0].tobytes()
    assert got == want


@needs_gpu
def test_acvp_siggen_first_of_each_group():
    seen = set()
    for case in ACVP:
        g = case["group"]
        key = (g["parameterSet"], g["signatureInterface"], g.get("preHash"), g["deterministic"])
        if key in seen:
            continue
        seen.add(key)
        _check_acvp(case)


@needs_gpu
@pytest.mark.slow
def test_acvp_siggen_all():
    for case in ACVP:
        _check_acvp(case)


def _check_acvp(case):
    g = case["group"]
    ps = g["parameterSet"]
    signer = Signer(ps, bytes.fromhex(case["sk"]))
    rnd = bytes.fromhex(case["rnd"]) if "rnd" in case else None
    if "mu" in case:
        got = signer.sign_mu([bytes.fromhex(case["mu"])], rnd=rnd)[0].tobytes()
    elif g["signatureInterface"] == "internal":
        got = signer.sign_mprime([bytes.fromhex(case["message"])], rnd=rnd)[0].tobytes()
    else:
        got = signer.sign(
            [bytes.fromhex(case["message"])],
            ctx=bytes.fromhex(case.get("context", "")),
            prehash=case.get("hashAlg") if g.get("preHash") == "preHash" else None,
            rnd=rnd,
        )[0].tobytes()
    assert got.hex().upper() == case["expected"].upper(), f"tcId={case['tcId']} {ps}"


@needs_gpu
def test_batch_with_rejections():
    """A batch big enough that some rows need several attempts (the compaction path)."""
    ps = "ML-DSA-44"
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    signer = Signer(ps, sk)
    n = 24
    msgs = [f"message {i:08d}".encode() for i in range(n)]
    t0 = time.perf_counter()
    sigs = signer.sign(msgs)
    assert sigs.shape == (n, len(m.sign(sk, msgs[0], rnd=None)))
    assert time.perf_counter() - t0 < 120
    v = Verifier(ps, pk)
    assert v.verify(msgs, sigs).all()
    # and every signature is the same one the reference produces
    for i in (0, 7, n - 1):
        assert sigs[i].tobytes() == m.sign(sk, msgs[i], rnd=None)


@needs_gpu
def test_context_and_prehash():
    ps = "ML-DSA-65"
    m = MLDSA(ps)
    _pk, sk = m.keygen(bytes(range(32)))
    signer = Signer(ps, sk)
    msg = b"with context"
    ctx = b"ctx"
    for prehash in (None, "SHA2-256"):
        want = m.sign(sk, msg, ctx=ctx, prehash=prehash, rnd=None)
        got = signer.sign([msg], ctx=ctx, prehash=prehash)[0].tobytes()
        assert got == want


@needs_gpu
def test_empty_batch():
    ps = "ML-DSA-44"
    _pk, sk = MLDSA(ps).keygen(bytes(range(32)))
    assert Signer(ps, sk).sign([]).size == 0
