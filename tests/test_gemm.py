"""Phase 3.1 acceptance: Triton GEMM matches the CPU int64 reference for many batch sizes."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from triton_crypto import keyprep, refimpl
from triton_crypto.params import Q

pytest.importorskip("triton")

from _common import DEV

from triton_crypto.kernels.gemm_modq import gemm_modq, gemm_modq_int8

pytest.importorskip("kyber_py", reason="refimpl needs the optional kyber-py")

XPU = True  # device availability is skipped centrally by tests/conftest.py (HAS_GPU)
pytestmark = pytest.mark.skipif(not XPU, reason="requires GPU")


def _f16(a):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a))).to(torch.float16).to(DEV)


def _i16(a):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int16))).to(DEV)


def _i8(a):
    return torch.from_numpy(np.ascontiguousarray(np.asarray(a, dtype=np.int8))).to(DEV)


@pytest.fixture(scope="module")
def kp():
    _, dk = refimpl.ref_keygen("ML-KEM-768")
    return keyprep.parse_key("ML-KEM-768", dk)


def _enc_inputs(kp, B, seed=0):
    rng = np.random.default_rng(seed)
    k = kp.k
    Y = rng.integers(-kp.eta1, kp.eta1 + 1, size=(kp.K, B))
    e1 = rng.integers(-kp.eta2, kp.eta2 + 1, size=(256 * k, B))
    e2 = rng.integers(-kp.eta2, kp.eta2 + 1, size=(256, B))
    m = rng.integers(0, 2, size=(256, B))
    E = np.vstack([e1, e2 + 1665 * m]).astype(np.int16)
    return Y, E


@pytest.mark.parametrize("B", [1, 7, 16, 37, 64, 128, 1000])
def test_enc_matches_cpu(kp, B):
    Y, E = _enc_inputs(kp, B)
    ref = (kp.W_enc @ Y + E) % Q

    out = gemm_modq(_f16(kp.W_enc), _f16(Y), _i16(E), mode="enc")
    got = out.cpu().numpy()
    assert got.dtype == np.int16
    assert np.array_equal(got.astype(np.int64), ref), (
        f"B={B}: mismatch {(got.astype(np.int64) != ref).sum()}"
    )


@pytest.mark.parametrize("B", [1, 7, 16, 37, 64, 128, 1000])
def test_dec_matches_cpu(kp, B):
    rng = np.random.default_rng(B + 100)
    U = rng.integers(0, Q, size=(kp.K, B))
    Uc = keyprep.center(U)
    V = rng.integers(0, Q, size=(256, B)).astype(np.int16)
    ref = (V - kp.W_dec @ Uc) % Q

    out = gemm_modq(_f16(kp.W_dec), _f16(Uc), _i16(V), mode="dec")
    got = out.cpu().numpy().astype(np.int64)
    assert np.array_equal(got, ref), f"B={B}: mismatch {(got != ref).sum()}"


def test_output_range(kp):
    Y, E = _enc_inputs(kp, 64)
    out = gemm_modq(_f16(kp.W_enc), _f16(Y), _i16(E), mode="enc").cpu()
    assert out.min().item() >= 0 and out.max().item() < Q


def test_zero_y_gives_only_addend(kp):
    B = 16
    Y = np.zeros((kp.K, B), dtype=np.int64)
    E = np.full((kp.M, B), 7, dtype=np.int16)
    out = gemm_modq(_f16(kp.W_enc), _f16(Y), _i16(E), mode="enc").cpu().numpy()
    assert np.array_equal(out, np.full((kp.M, B), 7, dtype=np.int16))


@pytest.mark.parametrize("B", [1, 37, 128])
def test_int8_limb_matches_cpu(kp, B):
    Y, E = _enc_inputs(kp, B, seed=7)
    hi, lo = keyprep.split_limb(kp.W_enc)
    ref = (kp.W_enc @ Y + E) % Q

    out = gemm_modq_int8(_i8(hi), _i8(lo), _i8(Y), _i16(E), mode="enc")
    got = out.cpu().numpy().astype(np.int64)
    assert np.array_equal(got, ref), f"B={B}: mismatch {(got != ref).sum()}"


def test_identical_across_runs(kp):
    """Determinism: the same input run twice must give identical results."""
    Y, E = _enc_inputs(kp, 64)
    a = gemm_modq(_f16(kp.W_enc), _f16(Y), _i16(E), mode="enc").cpu().numpy()
    b = gemm_modq(_f16(kp.W_enc), _f16(Y), _i16(E), mode="enc").cpu().numpy()
    assert np.array_equal(a, b)
