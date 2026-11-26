"""Acceptance tests for the GPU-resident pipeline (`pipeline.Decapsulator`)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV

from triton_crypto import acvp, refimpl
from triton_crypto.pipeline import Decapsulator

pytest.importorskip("kyber_py", reason="refimpl needs the optional kyber-py")

pytest.importorskip("triton")
XPU = True  # device availability is skipped centrally by tests/conftest.py (HAS_GPU)
pytestmark = pytest.mark.skipif(not XPU, reason="requires GPU")

PARAM_SETS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_gpu_roundtrip(ps):
    from triton_crypto.pipeline import Decapsulator

    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    cts, expect = [], []
    for _ in range(4):
        K, c = refimpl.ref_encaps(ps, ek)
        cts.append(np.frombuffer(c, np.uint8))
        expect.append(K)
    got = dec.decaps(np.stack(cts))
    assert [got[i].tobytes() for i in range(len(expect))] == expect


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_gpu_official_acvp(ps):
    from triton_crypto.pipeline import Decapsulator

    cases = [c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    assert cases
    for c in cases:
        dk = bytes.fromhex(c["inputs"]["dk"])
        ct = bytes.fromhex(c["inputs"]["c"])
        got = Decapsulator(ps, dk).decaps(np.frombuffer(ct, np.uint8)[None, :])
        assert got[0].tobytes().hex().upper() == c["expected"]["k"].upper(), f"tcId={c['tcId']}"


def test_gpu_implicit_rejection():
    from triton_crypto.pipeline import Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    K, c = refimpl.ref_encaps(ps, ek)
    cts = []
    for pos in (0, len(c) // 2, len(c) - 1):
        bad = bytearray(c)
        bad[pos] ^= 1
        cts.append(np.frombuffer(bytes(bad), np.uint8))
    got = dec.decaps(np.stack(cts))
    for i, ct in enumerate(cts):
        ref = refimpl.ref_decaps(ps, dk, ct.tobytes())
        assert got[i].tobytes() == ref
        assert got[i].tobytes() != K


def test_gpu_batch_one():
    from triton_crypto.pipeline import Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    K, c = refimpl.ref_encaps(ps, ek)
    got = Decapsulator(ps, dk).decaps(np.frombuffer(c, np.uint8)[None, :])
    assert got[0].tobytes() == K


def test_unpack_matches_numpy():
    from triton_crypto.kernels.elementwise import _unpack_bits
    from triton_crypto.pipeline import unpack_bits_gpu

    rng = np.random.default_rng(0)
    for d, count in [(10, 768), (4, 256), (11, 1024), (12, 256)]:
        nb = count * d // 8
        data = rng.integers(0, 256, size=(5, nb), dtype=np.uint8)
        ref = _unpack_bits(data, d, count).T  # [5, count]
        got = unpack_bits_gpu(torch.from_numpy(data).to(DEV), d, count).cpu().numpy()
        assert np.array_equal(got, ref), (d, count)


def test_cbd_matches_reference():
    from triton_crypto.pipeline import cbd_gpu
    from triton_crypto.sampling_cpu import cbd as cbd_ref

    rng = np.random.default_rng(1)
    for eta in (2, 3):
        bufs = rng.integers(0, 256, size=(17, 64 * eta), dtype=np.uint8)
        ref = np.stack([cbd_ref(bufs[i].tobytes(), eta) for i in range(17)])
        got = cbd_gpu(torch.from_numpy(bufs).to(DEV), eta).cpu().numpy()
        assert np.array_equal(got, ref), eta


# ---------------------------------------------------------------- input forms
@pytest.mark.parametrize("ps", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_all_input_forms_agree(ps):
    """The four input forms must give the same outputs (bytes / ndarray / list /
    device tensor).
    """
    from triton_crypto import refimpl

    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    kl = dec.p.ct_size
    cts = []
    for _ in range(8):
        _, c = refimpl.ref_encaps(ps, ek)
        cts.append(c)
    joined = b"".join(cts)

    ref = dec.decaps(joined)  # bytes
    for name, arg in [
        ("np.ndarray", np.frombuffer(joined, np.uint8).reshape(len(cts), kl)),
        ("np.ndarray writable", np.frombuffer(joined, np.uint8).reshape(len(cts), kl).copy()),
        ("bytearray", bytearray(joined)),
        ("memoryview", memoryview(joined)),
        ("list[bytes]", list(cts)),
    ]:
        got = dec.decaps(arg)
        assert got.shape == ref.shape, name
        assert np.array_equal(got, ref), name

    dev = torch.from_numpy(np.frombuffer(joined, np.uint8).reshape(len(cts), kl).copy()).to(DEV)
    assert np.array_equal(dec.decaps(dev), ref), "torch.Tensor"


def test_bad_input_forms_raise():
    from triton_crypto import refimpl

    ek, dk = refimpl.ref_keygen("ML-KEM-768")
    dec = Decapsulator("ML-KEM-768", dk)
    kl = dec.p.ct_size
    _, c = refimpl.ref_encaps("ML-KEM-768", ek)

    with pytest.raises(ValueError):  # length is not a whole multiple
        dec.decaps(c + b"\x00")
    with pytest.raises(ValueError):  # last dim is not ct_size
        dec.decaps(np.zeros((4, kl - 1), np.uint8))
    with pytest.raises(TypeError):  # wrong dtype
        dec.decaps(torch.zeros(4, kl, dtype=torch.int16, device=DEV))


def test_pinned_staging_reuse_is_safe():
    """The staging buffer is reused: call twice with different inputs; the second
    result must be the second call's.
    """
    from triton_crypto import refimpl

    ek, dk = refimpl.ref_keygen("ML-KEM-768")
    dec = Decapsulator("ML-KEM-768", dk)
    K1, c1 = refimpl.ref_encaps("ML-KEM-768", ek)
    K2, c2 = refimpl.ref_encaps("ML-KEM-768", ek)
    B = 32
    a = dec.decaps([c1] * B)
    b = dec.decaps([c2] * B)
    assert all(x.tobytes() == K1 for x in a)
    assert all(x.tobytes() == K2 for x in b)


def test_eta2_fast_path_only_for_ukem768_1024():
    """Record a known limitation: the fused CBD supports only η=2, so ML-KEM-512 takes
    the slow path.

    It is functionally correct (the ACVP vectors pass), just ~3x slower. This test exists
    to make the limitation surface immediately if someone ever implements η=3 too.
    """
    from triton_crypto import refimpl
    from triton_crypto.kernels import glue

    for ps, expect_fast in (("ML-KEM-512", False), ("ML-KEM-768", True), ("ML-KEM-1024", True)):
        _ek, dk = refimpl.ref_keygen(ps)
        dec = Decapsulator(ps, dk)
        p = dec.p
        has_fast = p.eta1 == 2 and p.eta2 == 2
        assert has_fast is expect_fast, ps
        if expect_fast:
            # the fast-path predicate really does accept the real shapes
            B, k = 8, p.k
            y_buf = torch.zeros(B * k, 64 * p.eta1, dtype=torch.uint8, device=DEV)
            assert glue.can_cbd_y(y_buf, B, k, p.eta1)


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_empty_and_non_contiguous_ciphertexts(ps):
    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    empty = dec.decaps(np.empty((0, dec.p.ct_size), np.uint8))
    assert empty.shape == (0, 32)

    cts = []
    for _ in range(4):
        _k, c = refimpl.ref_encaps(ps, ek)
        cts.append(np.frombuffer(c, np.uint8))
    arr = np.stack(cts)
    want = dec.decaps(arr)

    # [B, 2*ct] with every other byte used: valid values, non-contiguous rows
    wide = torch.zeros((4, 2 * dec.p.ct_size), dtype=torch.uint8, device=DEV)
    view = wide[:, ::2]
    view.copy_(torch.from_numpy(arr).to(DEV))
    assert not view.is_contiguous()
    got = dec.decaps(view)
    assert np.array_equal(got, want)

    got_dev = dec.decaps_torch(view, 4).cpu().numpy()
    assert np.array_equal(got_dev, want)

    with pytest.raises(ValueError, match="ciphertext"):  # ct size per set
        dec.decaps([b"\x00" * (dec.p.ct_size - 1), b"\x00" * (dec.p.ct_size + 1)])
