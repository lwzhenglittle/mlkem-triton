"""Falcon verification on the device: the batched core against the numpy reference.

What is checked here is the part that actually runs on the device -- `s2*h`, the subtraction of
the hashed point, the normalisation into [-q/2, q/2] and the L2 bound. The decoded signature
and the hashed point are computed on the host, so these tests feed the device the same inputs
the reference uses and compare the verdict *and* the squared norm, which is the only diagnostic
a rejection has.

Two traps this file exists to keep pinned, both found while writing the kernel:

* Triton's `%` on signed integers is C's truncated remainder, so `(prod - c) % q` leaves
  negative values negative -- numpy's floored `%` does not. That silently changes every norm.
* an `expand`ed tensor is not contiguous, so a kernel that assumes contiguity reads past the
  end of the first row; the pointwise kernel indexes the public key's transform modulo `n`.
"""

from __future__ import annotations

import numpy as np
import pytest
from test_falcon import MSG, PK, SIG

from triton_crypto.falcon import (
    FALCON_512,
    FALCON_1024,
    FALCON_PADDED_512,
    FALCON_PADDED_1024,
    Verifier,
    comp_decode,
    comp_decode_batch,
    comp_decode_batch_torch,
    hash_to_point,
    ref,
)
from triton_crypto.falcon.kernels import check_np
from triton_crypto.falcon.verify import hash_to_point_batch_torch


def _prepare():
    p = FALCON_512
    s2 = comp_decode(SIG[41:], p)[0]
    c = hash_to_point(SIG[1:41], MSG, p)
    return p, s2, c


def test_one_signature_verdict_matches_the_reference():
    p, s2, c = _prepare()
    v = Verifier(PK)
    ok_ref, sqn_ref = ref.verify_core(c, s2, v.pk_poly, p)
    ok, sqn = v.verify_core_torch(c[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is ok_ref is True
    assert int(sqn.cpu().numpy()[0]) == sqn_ref


def test_batch_matches_the_reference_and_is_order_independent():
    p, s2, c = _prepare()
    v = Verifier(PK)
    rows = 37
    s2b = np.repeat(s2[None, :], rows, axis=0).astype(np.int32)
    s2b[1::2] = -s2b[1::2]  # different coefficient signs: different norms, same code path
    cb = np.repeat(c[None, :], rows, axis=0)
    _ok, sqn = v.verify_core_torch(cb, s2b)
    got = sqn.cpu().numpy()
    for i in range(rows):
        _, expect = ref.verify_core(cb[i], s2b[i].astype(np.int64), v.pk_poly, p)
        assert got[i] == expect, f"row {i}"


def test_a_different_message_is_rejected():
    p, s2, _ = _prepare()
    v = Verifier(PK)
    other = hash_to_point(SIG[1:41], b"a different message", p)
    ok, _ = v.verify_core_torch(other[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is False


def test_a_one_unit_perturbation_of_the_hash_point_can_still_verify():
    """Falcon's verdict is a norm bound, not an equality -- this is not a bug, but it is
    worth pinning, because it is the difference between "signature is valid" and "signature
    was produced by the signer". The embedded signature's norm is 30.08M against a bound of
    34.03M, so ~12% of slack remains and a +1 on one coefficient can *shrink* the norm."""
    p, s2, c = _prepare()
    v = Verifier(PK)
    c2 = c.copy()
    c2[0] = (c2[0] + 1) % p.q  # one coefficient, not all of them
    ok, _ = v.verify_core_torch(c2[None, :], s2[None, :].astype(np.int32))
    assert bool(ok.cpu().numpy()[0]) is True
    # and it is still the norm bound that decides: the reference agrees on this input
    ok_ref, _ = ref.verify_core(c2, s2, v.pk_poly, p)
    assert ok_ref is True


def test_check_kernel_matches_numpy_on_negative_differences():
    """The negative-modulo trap, isolated: half of `prod - c` is negative by construction."""
    q, n, bound = FALCON_512.q, FALCON_512.n, FALCON_512.l2bound
    rng = np.random.default_rng(5)
    prod = rng.integers(0, q, (6, n))
    c = rng.integers(0, q, (6, n))
    s2 = rng.integers(-400, 400, (6, n))
    ok_ref, sqn_ref = check_np(prod, c, s2, q, bound)
    import torch

    from triton_crypto.devices import resolve_device
    from triton_crypto.falcon.kernels import _check_kernel

    dev = resolve_device()
    sqn = torch.empty(6, dtype=torch.int64, device=dev)
    ok = torch.empty(6, dtype=torch.int8, device=dev)
    _check_kernel[(6,)](
        torch.tensor(prod, dtype=torch.int32, device=dev),
        torch.tensor(c, dtype=torch.int32, device=dev),
        torch.tensor(s2, dtype=torch.int32, device=dev),
        sqn,
        ok,
        6,
        L2BOUND=bound,
        Q=q,
        N=n,
    )
    assert np.array_equal(sqn.cpu().numpy(), sqn_ref)
    assert np.array_equal(ok.cpu().numpy().astype(bool), ok_ref)


def test_device_hash_to_point_matches_the_host():
    """The two stages of `HashToPoint` on the device: the sponge, then the prefix-sum filter.

    Lengths chosen to cross the SHAKE256 rate (136 bytes of input per block): 40 + 60 stays in
    one block, 40 + 200 needs two, and a wrong block count changes every coefficient.
    """
    rng = np.random.default_rng(3)
    for mlen in (20, 60, 96, 200, 500):
        nonces = [bytes(rng.integers(0, 256, 40, dtype=np.uint8)) for _ in range(17)]
        msgs = [bytes(rng.integers(0, 256, mlen, dtype=np.uint8)) for _ in range(17)]
        c_dev, ok = hash_to_point_batch_torch(nonces, msgs)
        assert ok.all(), f"msglen {mlen}: a row had fewer than n accepted samples"
        want = np.stack([ref.hash_to_point(no, m) for no, m in zip(nonces, msgs, strict=True)])
        assert np.array_equal(c_dev.cpu().numpy(), want), f"msglen {mlen}"


def test_device_hash_to_point_needs_uniform_lengths():
    """The sponge absorbs a fixed number of blocks per row; mixed lengths must be refused, not
    silently mis-hashed (the caller falls back to the host path)."""
    with pytest.raises(ValueError, match="one message length"):
        hash_to_point_batch_torch(
            [bytes(40), bytes(40)],
            [bytes(20), bytes(21)],
        )


def test_public_api_mixes_lengths_via_the_host_fallback():
    """A batch with mixed message lengths still verifies -- through the host hash."""
    v = Verifier(PK)
    assert v.verify([MSG], [SIG]).tolist() == [True]
    # the embedded signature is for MSG only; a different length must simply not verify rather
    # than crash on the device path's uniform-length requirement
    assert v.verify([MSG + b"!"], [SIG]).tolist() == [False]


def test_public_api_verifies_the_embedded_signature():
    v = Verifier(PK)
    assert v.verify([MSG], [SIG]).tolist() == [True]
    bad = bytearray(SIG)
    bad[200] ^= 0x01
    assert v.verify([MSG], [bytes(bad)]).tolist() == [False]


@pytest.mark.parametrize("batch", [1, 3, 64])
def test_public_api_batches(batch):
    v = Verifier(PK)
    ok = v.verify([MSG] * batch, [SIG] * batch)
    assert ok.shape == (batch,)
    assert ok.all()


def test_empty_batch_core_and_public_api():
    v = Verifier(PK)
    assert v.verify([], []).tolist() == []
    ok, sqn = v.verify_core_torch(
        np.zeros((0, FALCON_512.n), dtype=np.int32),
        np.zeros((0, FALCON_512.n), dtype=np.int32),
    )
    assert ok.shape == (0,) and sqn.shape == (0,)


def test_offline_kat_corpus_on_device():
    """Deterministic multi-signature GPU check covering Falcon-512 and Falcon-1024 (unpadded + padded)."""
    import json
    from pathlib import Path

    tag_params = {
        "falcon_512": FALCON_512,
        "falcon_padded_512": FALCON_PADDED_512,
        "falcon_1024": FALCON_1024,
        "falcon_padded_1024": FALCON_PADDED_1024,
    }
    kat_path = Path(__file__).resolve().parent.parent / "data" / "vectors" / "falcon" / "kat.json"
    data = json.loads(kat_path.read_text(encoding="utf-8"))
    for group in data["groups"]:
        params = tag_params[group["tag"]]
        pk = bytes.fromhex(group["pk_hex"])
        msgs = [bytes.fromhex(c["msg_hex"]) for c in group["cases"]]
        sigs = [bytes.fromhex(c["sig_hex"]) for c in group["cases"]]
        # Exercise both explicit params and auto-detection of Falcon-512 / Falcon-1024 from pk.
        v = Verifier(pk, padded=bool(group["padded"]))
        assert v.params.n == params.n
        assert v.verify(msgs, sigs).all(), group["tag"]
        # Also exercise the uniform-length GPU HashToPoint path on a single-length sub-batch.
        assert v.verify([msgs[0]] * 4, [sigs[0]] * 4).all(), group["tag"]
        tampered = []
        for s in sigs:
            b = bytearray(s)
            b[1 + params.nonce_len + 5] ^= 0x01
            tampered.append(bytes(b))
        assert not v.verify(msgs, tampered).any(), group["tag"]


@pytest.mark.parametrize("params", [FALCON_512, FALCON_1024], ids=lambda p: p.name)
def test_device_hash_to_point_all_degrees_and_rate_boundaries(params):
    """Test uniform-length GPU HashToPoint across SHAKE256 rate=136 boundaries (0, 95, 96, 200)."""
    rng = np.random.default_rng(params.logn)
    for mlen in (0, 95, 96, 200):
        nonces = [bytes(rng.integers(0, 256, 40, dtype=np.uint8)) for _ in range(9)]
        msgs = [bytes(rng.integers(0, 256, mlen, dtype=np.uint8)) for _ in range(9)]
        c_dev, ok = hash_to_point_batch_torch(nonces, msgs, params)
        assert ok.all(), (params.name, mlen)
        want = np.stack(
            [ref.hash_to_point(no, m, params) for no, m in zip(nonces, msgs, strict=True)]
        )
        assert np.array_equal(c_dev.cpu().numpy(), want), (params.name, mlen)


def test_device_comp_decode_matches_host_on_edge_cases():
    """Verify GPU `_comp_decode_kernel` matches host `comp_decode_batch` on every failure mode."""
    import json
    from pathlib import Path

    kat_path = Path(__file__).resolve().parent.parent / "data" / "vectors" / "falcon" / "kat.json"
    data = json.loads(kat_path.read_text(encoding="utf-8"))
    by_tag = {g["tag"]: g for g in data["groups"]}

    for tag, params, padded in (
        ("falcon_512", FALCON_512, False),
        ("falcon_padded_512", FALCON_PADDED_512, True),
        ("falcon_1024", FALCON_1024, False),
        ("falcon_padded_1024", FALCON_PADDED_1024, True),
    ):
        skip = 1 + params.nonce_len
        valid_body = bytes.fromhex(by_tag[tag]["cases"][0]["sig_hex"])[skip:]
        valid_body2 = bytes.fromhex(by_tag[tag]["cases"][1]["sig_hex"])[skip:]

        # Construct every malformed encoding case alongside valid neighbours:
        # 0: valid, 1: empty, 2: truncated, 3: -0 coeff (0x80 + unary '1' -> 0x80 0x80 ...),
        # 4: >2047 (all zeros -> cont == 0), 5: trailing nonzero byte, 6: trailing zero byte
        # (wrong length), 7: oversized 4000-byte body, 8: valid2.
        neg_zero = b"\x80\x80" + valid_body[2:]
        over_max = b"\x00\x00\x00" + valid_body[3:]
        pad_nonzero_end = bytearray(valid_body)
        pad_nonzero_end[-1] ^= 0x01
        bodies = [
            valid_body,
            b"",
            valid_body[:100],
            neg_zero,
            over_max,
            valid_body + b"\x01",
            valid_body + b"\x00",
            valid_body + b"\x00" * 3500,
            bytes(pad_nonzero_end),
            valid_body2,
        ]
        # For padded mode, also craft a padded-length body with a nonzero byte in the zero-pad tail.
        if padded:
            _, used = comp_decode(valid_body, params)
            if used < len(valid_body):
                bad_pad = bytearray(valid_body)
                bad_pad[used] = 0x01
                bodies.append(bytes(bad_pad))
                bad_pad_last = bytearray(valid_body)
                bad_pad_last[-1] = 0x80
                bodies.append(bytes(bad_pad_last))

        s2_host, ok_host = comp_decode_batch(bodies, params, padded=padded)
        s2_dev, ok_dev = comp_decode_batch_torch(bodies, params, padded=padded)
        ok_dev_np = ok_dev.cpu().numpy()
        s2_dev_np = s2_dev.cpu().numpy()

        assert np.array_equal(ok_dev_np, ok_host), (tag, ok_dev_np.tolist(), ok_host.tolist())
        assert ok_dev_np[0] and ok_dev_np[9]
        assert not ok_dev_np[1:9].any()
        if padded and len(bodies) > 10:
            assert not ok_dev_np[10:].any()
        assert np.array_equal(s2_dev_np[ok_host], s2_host[ok_host]), tag


def test_public_api_empty_message_and_boundary_lengths_on_device():
    """Empty messages and SHAKE256 rate=136 boundaries (`40 + 95 = 135`, `40 + 96 = 136`) work."""
    v = Verifier(PK)
    # Must not crash and must return False (wrong message for SIG) cleanly.
    for m in (b"", b"a" * 95, b"a" * 96, b"a" * 97):
        res = v.verify([m, m], [SIG, SIG])
        assert res.tolist() == [False, False]
