"""ML-DSA verification on the GPU: kernel-level checks and end-to-end acceptance.

Three layers, mirroring how the ML-KEM side is tested:

1. **kernels against numpy** — BitUnpack of z, HintBitUnpack, SampleInBall,
   Decompose/UseHint/BitPack, and the NTT-domain MAC are each compared with the FIPS 204
   reference on the same input, so a data-dependent bug cannot hide behind a correct
   final verdict;
2. **end-to-end against the reference** — random key, random messages, valid and mutated
   signatures; the GPU must agree with `mldsa.ref` case by case;
3. **official ACVP vectors** — all 180 sigVer cases (in `test_mldsa_ref.py` for the numpy
   path, here for the GPU), plus a cross-check against liboqs when it is built.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from mlkem_triton.mldsa import acvp
from mlkem_triton.mldsa.kernels import (
    hint_kernel,
    mac_kernel,
    sample_in_ball_kernel,
    unpack_z_kernel,
    w1_pack_kernel,
)
from mlkem_triton.mldsa.params import D
from mlkem_triton.mldsa.ref import MLDSA
from mlkem_triton.mldsa.verify import Verifier

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")

PS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]
RATE = 136


def _dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).to(DEV)


def _key_and_sigs(ps: str, n: int = 3, seed: bytes | None = None):
    m = MLDSA(ps)
    pk, sk = m.keygen(seed or bytes(range(32)))
    msgs = [bytes([i + 1]) * 16 for i in range(n)]  # uniform length: array API too
    sigs = [m.sign(sk, msg) for msg in msgs]
    return m, pk, sk, msgs, sigs


# ---------------------------------------------------------------- kernels vs reference


@pytest.mark.parametrize("ps", PS)
def test_unpack_z_kernel(ps):
    m = MLDSA(ps)
    p = m.p
    _, _, _, _, sigs = _key_and_sigs(ps, 2)
    sig = torch.from_numpy(np.stack([np.frombuffer(s, np.uint8) for s in sigs])).to(DEV)
    out = torch.empty((2 * p.l, 256), dtype=torch.int32, device=DEV)
    unpack_z_kernel[(2, p.l)](
        sig,
        out,
        p.sig_size,
        2,
        p.z_offset,
        p.polyz_bytes,
        p.z_bits,
        p.gamma1,
        256,
        p.l,
        num_warps=4,
    )
    for b, s in enumerate(sigs):
        want = np.stack(
            [
                m.unpack_z(s[p.z_offset + i * p.polyz_bytes : p.z_offset + (i + 1) * p.polyz_bytes])
                for i in range(p.l)
            ]
        ).reshape(-1, 256)
        assert np.array_equal(out[b * p.l : (b + 1) * p.l].cpu().numpy(), want.astype(np.int32))


@pytest.mark.parametrize("ps", PS)
def test_hint_kernel(ps):
    m = MLDSA(ps)
    p = m.p
    _, _, _, _, sigs = _key_and_sigs(ps, 3)
    sig = torch.from_numpy(np.stack([np.frombuffer(s, np.uint8) for s in sigs])).to(DEV)
    h = torch.zeros((3, p.k, 256), dtype=torch.uint8, device=DEV)
    ok = torch.empty((3, p.k), dtype=torch.int32, device=DEV)
    hint_kernel[(3, p.k)](
        sig,
        h,
        ok,
        p.sig_size,
        p.h_offset,
        p.omega,
        1 << (p.omega - 1).bit_length(),
        p.k,
        256,
        num_warps=4,
    )
    for b, s in enumerate(sigs):
        want = m.unpack_h(s[p.h_offset :])
        assert np.array_equal(h[b].cpu().numpy(), want.astype(np.uint8))
        assert bool(ok[b].all())

    # malformed hint blocks must be flagged, not crash: non-monotone counts, a duplicate
    # index, and a non-zero tail after the last row's count
    def mutate(fn):
        bad = bytearray(sigs[0])
        hb = p.h_offset
        raw = bytearray(bad[hb:])
        fn(raw)
        bad[hb:] = raw
        return bytes(bad)

    broken = [
        mutate(lambda r: r.__setitem__(p.omega, 0xFF)),  # count > ω
        mutate(lambda r: r.__setitem__(p.omega + 1, 0)),  # counts decrease
        mutate(lambda r: r.__setitem__(1, r[0]) if r[0] else None),  # duplicate index
        mutate(lambda r: r.__setitem__(p.omega - 1, 7)),  # tail after the last count
    ]
    bsig = torch.from_numpy(np.stack([np.frombuffer(s, np.uint8) for s in broken])).to(DEV)
    hb = torch.zeros((len(broken), p.k, 256), dtype=torch.uint8, device=DEV)
    okb = torch.empty((len(broken), p.k), dtype=torch.int32, device=DEV)
    hint_kernel[(len(broken), p.k)](
        bsig,
        hb,
        okb,
        p.sig_size,
        p.h_offset,
        p.omega,
        1 << (p.omega - 1).bit_length(),
        p.k,
        256,
        num_warps=4,
    )
    assert not bool(okb.all()), "malformed hints were not detected"
    for s in broken:
        with pytest.raises(ValueError):
            m.unpack_h(s[p.h_offset :])


@pytest.mark.parametrize("ps", PS)
def test_sample_in_ball_kernel(ps):
    m = MLDSA(ps)
    p = m.p
    ctilde = np.random.default_rng(0).integers(0, 256, (4, p.ctilde_bytes), dtype=np.uint8)
    padded = np.zeros((4, RATE), dtype=np.uint8)
    padded[:, : p.ctilde_bytes] = ctilde
    padded[:, p.ctilde_bytes] = 0x1F
    padded[:, -1] |= 0x80
    from mlkem_triton.primitives.sponge import sponge_padded

    buf = sponge_padded(_dev(padded), RATE, RATE)  # the XOF stream, not the padded input
    out = torch.empty((4, 256), dtype=torch.int32, device=DEV)
    sample_in_ball_kernel[(4,)](buf, out, 256, p.tau, RATE, num_warps=4)
    for i in range(4):
        want = m.sample_in_ball(ctilde[i].tobytes())
        got = out[i].cpu().numpy()
        assert np.array_equal(got, want.astype(np.int32))
        assert np.count_nonzero(got) == p.tau


@pytest.mark.parametrize("ps", PS)
def test_w1_pack_kernel(ps):
    """Decompose + UseHint + BitPack against the reference, on random w and hints."""
    m = MLDSA(ps)
    p = m.p
    rng = np.random.default_rng(1)
    b, k = 2, p.k
    w = rng.integers(0, 8380417, (b, k, 256), dtype=np.int64)
    h = (rng.random((b, k, 256)) < 0.02).astype(np.uint8)
    out = torch.empty((b, k * p.polyw1_bytes), dtype=torch.uint8, device=DEV)
    w1_pack_kernel[(b, k, (p.polyw1_bytes + 63) // 64)](
        _dev(w.astype(np.int32)),
        _dev(h),
        out,
        b,
        k,
        256,
        p.w1_bits,
        p.two_gamma2,
        p.gamma2,
        p.m,
        p.polyw1_bytes,
        64,
        num_warps=4,
    )
    for i in range(b):
        for j in range(k):
            want = m.use_hint(h[i, j], w[i, j])
            got = m.unpack_w1(
                out[i, j * p.polyw1_bytes : (j + 1) * p.polyw1_bytes].cpu().numpy().tobytes()
            )
            assert np.array_equal(got, want)


@pytest.mark.parametrize("ps", PS)
def test_mac_kernel(ps):
    """ŵ = Â·ẑ − ĉ·t̂1 against the reference, in the NTT domain."""
    from mlkem_triton.primitives.ntt import DILITHIUM, ntt_np

    m = MLDSA(ps)
    p = m.p
    _, pk, _, _, sigs = _key_and_sigs(ps, 1)
    V = Verifier(ps, pk)
    sig = np.frombuffer(sigs[0], np.uint8)
    rng = np.random.default_rng(2)
    z = rng.integers(0, 8380417, (1, p.l, 256), dtype=np.int64)
    ch = rng.integers(0, 8380417, (1, 256), dtype=np.int64)
    z_hat = _dev(ntt_np(z, DILITHIUM).astype(np.int32))
    ch_hat = _dev(ntt_np(ch, DILITHIUM).astype(np.int32))
    out = torch.empty((1, p.k, 256), dtype=torch.int32, device=DEV)
    mac_kernel[(1, p.k)](
        z_hat, V.A_hat, ch_hat, V.t1_hat, out, 1, p.k, p.l, 256, 8380417, 8, num_warps=4
    )
    A = m.expand_a(pk[:32])
    t1 = m.unpack_pk(pk)[1]
    want = (
        (A * ntt_np(z, DILITHIUM)[0][None]).sum(1)
        - ntt_np(ch, DILITHIUM)[0] * ntt_np((t1 * (1 << D)) % 8380417, DILITHIUM)
    ) % 8380417
    assert np.array_equal(out[0].cpu().numpy().astype(np.int64), want)
    del sig


# ---------------------------------------------------------------- end to end


@pytest.mark.parametrize("ps", PS)
def test_verify_accepts_reference_signatures(ps):
    m, pk, _, msgs, sigs = _key_and_sigs(ps, 4)
    V = Verifier(ps, pk)
    assert V.verify(msgs, sigs).all()
    # same messages given as an [B, L] array, and through the internal/external-mu APIs
    arr = np.stack([np.frombuffer(s, np.uint8) for s in msgs])
    assert V.verify(arr, sigs).all()
    mprimes = [m.format_message(msg) for msg in msgs]
    assert V.verify_internal(mprimes, sigs).all()
    mu = np.stack([np.frombuffer(m.mu(pk, mp), np.uint8) for mp in mprimes])
    assert V.verify_mu(mu, sigs).all()


@pytest.mark.parametrize("ps", PS)
def test_verify_rejects_mutations_like_reference(ps):
    m, pk, _, msgs, sigs = _key_and_sigs(ps, 2)
    V = Verifier(ps, pk)
    p = m.p
    rng = np.random.default_rng(7)
    cases = []
    for _ in range(12):
        bad = bytearray(sigs[0])
        pos = int(rng.integers(0, len(bad)))
        bad[pos] ^= 1 << int(rng.integers(0, 8))
        cases.append(bytes(bad))
    # truncation is an API error, not a verdict
    with pytest.raises(ValueError):
        V.verify([msgs[0]], [sigs[0][:-1]])
    # a message that does not match
    cases.append(sigs[0])
    got = V.verify([msgs[0]] * 1 + [b"other"] * (len(cases) - 1), cases)
    want = [
        m.verify(pk, msgs[0] if i < len(cases) - 1 else b"other", c) for i, c in enumerate(cases)
    ]
    assert list(got) == want, (list(got), want)
    # every mutation of the c̃ prefix must fail
    ct = p.ctilde_bytes
    for pos in (0, ct - 1):
        bad = bytearray(sigs[0])
        bad[pos] ^= 0x01
        assert not V.verify([msgs[0]], [bytes(bad)])[0]


@pytest.mark.parametrize("ps", PS)
def test_verify_with_context_and_prehash(ps):
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    ctx = b"acvp-context"
    msgs = [b"", b"hello"]
    sigs = [m.sign(sk, msg, ctx=ctx) for msg in msgs]
    V = Verifier(ps, pk)
    assert V.verify(msgs, sigs, ctx=ctx).all()
    assert not V.verify(msgs, sigs).any()  # wrong context must fail
    for alg in ("SHA2-256", "SHA3-512", "SHAKE-128"):
        ph = [m.sign(sk, msg, ctx=ctx, prehash=alg) for msg in msgs]
        got = V.verify(msgs, ph, ctx=ctx, prehash=alg)
        assert got.all(), alg
        assert not V.verify(msgs, ph, ctx=ctx).any()


def test_verify_groups_mixed_message_lengths():
    ps = "ML-DSA-65"
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    msgs = [b"", b"a", b"bb" * 200, b"ccc" * 3, b"d" * 500]
    sigs = [m.sign(sk, msg) for msg in msgs]
    V = Verifier(ps, pk)
    assert V.verify(msgs, sigs).all()
    want = [m.verify(pk, msg, sig) for msg, sig in zip(msgs, sigs, strict=True)]
    assert list(V.verify(msgs, sigs)) == want


def test_empty_batch():
    """An empty batch is a no-op, not a crash (and no zero-grid kernel launch)."""
    ps = "ML-DSA-65"
    m = MLDSA(ps)
    pk, _ = m.keygen(bytes(range(32)))
    V = Verifier(ps, pk)
    assert V.verify([], []).shape == (0,)
    assert V.verify_mu(
        np.zeros((0, 64), np.uint8), np.zeros((0, m.p.sig_size), np.uint8)
    ).shape == (0,)
    assert V.verify_internal([], []).shape == (0,)


def test_batch_does_not_mix_rows():
    """One valid signature, many messages: exactly the matching row may pass."""
    ps = "ML-DSA-65"
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    msg = b"the one"
    sig = m.sign(sk, msg)
    msgs = [b"other-%02d" % i for i in range(31)] + [msg]
    got = Verifier(ps, pk).verify(msgs, [sig] * len(msgs))
    assert list(got) == [False] * 31 + [True]


# ---------------------------------------------------------------- ACVP + liboqs


def test_acvp_sigver_all_180():
    """The NIST signature-verification vectors, through the public API.

    All four interfaces are covered: external pure / pre-hash (with a per-case context)
    and internal with either a raw M′ or an externally supplied μ.
    """
    cases = acvp.load("sigVer")
    assert len(cases) == 180
    bad = []
    for c in cases:
        pk = bytes.fromhex(c["inputs"]["pk"])
        sig = bytes.fromhex(c["inputs"]["signature"])
        V = Verifier(c["parameterSet"], pk, device=DEV)
        if c["externalMu"]:
            mu = np.frombuffer(bytes.fromhex(c["inputs"]["mu"]), np.uint8)[None]
            got = bool(V.verify_mu(mu, [sig])[0])
        elif c["interface"] == "internal":
            got = bool(V.verify_internal([bytes.fromhex(c["inputs"]["message"])], [sig])[0])
        else:
            got = bool(
                V.verify(
                    [bytes.fromhex(c["inputs"]["message"])],
                    [sig],
                    ctx=bytes.fromhex(c["inputs"].get("context", "")),
                    prehash=c["inputs"].get("hashAlg"),
                )[0]
            )
        if got != c["expected"]["testPassed"]:
            bad.append((c["tgId"], c["parameterSet"], c["tcId"], got))
    assert not bad, bad[:5]


@pytest.mark.parametrize("ps", PS)
def test_matches_liboqs(ps):
    """Cross-implementation check: liboqs signs, we verify, liboqs agrees."""
    liboqs = pytest.importorskip("_liboqs", reason="liboqs with ML-DSA not built")
    if not liboqs.LibOQS.available(ps):
        pytest.skip("liboqs with ML-DSA not built")
    o = liboqs.LibOQS(ps)
    pk, sk = o.keypair()
    msgs = [f"msg-{i}".encode() for i in range(24)]
    sigs = [o.sign(m, sk) for m in msgs]
    V = Verifier(ps, pk, device=DEV)
    got = V.verify(msgs, sigs)
    assert got.all(), f"{int((~got).sum())} valid liboqs signatures were rejected"
    assert all(o.verify(m, s, pk) for m, s in zip(msgs, sigs, strict=True))
    # and the same signatures with a flipped bit: both sides must agree
    bad = []
    for s in sigs:
        b = bytearray(s)
        b[len(b) // 2] ^= 0x04
        bad.append(bytes(b))
    ours = V.verify(msgs, bad)
    theirs = [o.verify(m, s, pk) for m, s in zip(msgs, bad, strict=True)]
    assert list(ours) == theirs
