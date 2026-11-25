"""Phase 2 acceptance: W_enc / W_dec match the NTT path element by element."""

from __future__ import annotations

import numpy as np
import pytest

from mlkem_triton import keyprep, refimpl
from mlkem_triton.params import PARAMS, N, Q

pytest.importorskip("kyber_py", reason="refimpl needs the optional kyber-py")

PARAM_SETS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


@pytest.fixture(scope="module", params=PARAM_SETS)
def kp(request):
    ps = request.param
    _, dk = refimpl.ref_keygen(ps)
    return keyprep.parse_key(ps, dk)


# ---------------- shapes and bounds ----------------


def test_shapes(kp):
    p = PARAMS[kp.ps]
    assert kp.W_enc.shape == (256 * (p.k + 1), 256 * p.k)
    assert kp.W_dec.shape == (256, 256 * p.k)
    assert kp.A_hat.shape == (p.k, p.k, N)
    assert kp.t_hat.shape == (p.k, N)
    assert kp.s_hat.shape == (p.k, N)


def test_magnitude_bounds(kp):
    p = PARAMS[kp.ps]
    assert np.abs(kp.W_enc).max() <= 1664, np.abs(kp.W_enc).max()
    assert np.abs(kp.W_dec).max() <= p.eta1, np.abs(kp.W_dec).max()


def test_A_hat_roundtrip(kp):
    """Â should return to its original value after INTT then NTT."""
    from mlkem_triton.ntt_ref import intt, ntt

    for i in range(kp.k):
        for j in range(kp.k):
            assert np.array_equal(ntt(intt(kp.A_hat[i, j])), kp.A_hat[i, j])


def test_sample_ntt_matches_kyberpy(kp):
    from kyber_py.polynomials.polynomials import PolynomialRing

    R = PolynomialRing()
    for i in range(kp.k):
        for j in range(kp.k):
            ref = R.ntt_sample(
                __import__("hashlib").shake_128(kp.rho + bytes([j, i])).digest(840)
            ).coeffs
            assert np.array_equal(kp.A_hat[i, j], ref), f"A[{i}][{j}] mismatch"


# ---------------- core cross-check ----------------


def test_W_enc_matches_ntt_path(kp):
    """(W_enc @ y_centered) mod q == [INTT(Âᵀ∘NTT(y)); INTT(t̂ᵀ∘NTT(y))]."""
    rng = np.random.default_rng(0)
    for trial in range(3):
        # y's coefficient range is [-η1, η1]
        y_centered = rng.integers(-kp.eta1, kp.eta1 + 1, size=(kp.k, N))
        y_mod = y_centered % Q

        u_ref, v_ref = keyprep.ref_reencrypt(kp, y_mod)
        lhs = (kp.W_enc @ y_centered.reshape(-1)) % Q

        u_got = lhs[: 256 * kp.k].reshape(kp.k, N)
        v_got = lhs[256 * kp.k :]

        assert np.array_equal(u_got, u_ref), f"trial {trial}: u' mismatch"
        assert np.array_equal(v_got, v_ref), f"trial {trial}: v' mismatch"


def test_W_dec_matches_ntt_path(kp):
    """(W_dec @ u_centered) mod q == INTT(ŝᵀ∘NTT(u))."""
    rng = np.random.default_rng(1)
    for trial in range(3):
        u_mod = rng.integers(0, Q, size=(kp.k, N))
        u_centered = keyprep.center(u_mod)

        ref = keyprep.ref_decrypt_product(kp, u_mod)
        got = (kp.W_dec @ u_centered.reshape(-1)) % Q

        assert np.array_equal(got, ref), f"trial {trial}: mismatch"


def test_W_enc_zero_input():
    """With y=0 the output should be all 0."""
    ps = "ML-KEM-768"
    _, dk = refimpl.ref_keygen(ps)
    kp = keyprep.parse_key(ps, dk)
    assert not (kp.W_enc @ np.zeros(256 * 3, dtype=np.int64)).any()


# ---------------- quantization ----------------


def test_fp16_exact_roundtrip(kp):
    """Converting W to fp16 must lose no precision."""
    import torch

    for name in ("W_enc", "W_dec"):
        W = getattr(kp, name)
        t = keyprep.to_fp16(W)
        back = t.to(torch.float32).numpy().astype(np.int64)
        assert np.array_equal(back, W), f"{name} fp16 loses precision"


def test_int8_limb_split(kp):
    for name in ("W_enc", "W_dec"):
        W = getattr(kp, name)
        hi, lo = keyprep.split_limb(W)
        recon = 64 * hi.astype(np.int64) + lo.astype(np.int64)
        assert np.array_equal(recon, W), f"{name} limb reconstruction failed"


def test_centering_involution():
    x = np.array([0, 1664, 1665, 3328, 1, 3327])
    c = keyprep.center(x)
    assert np.array_equal(c % Q, x)
    assert np.array_equal(keyprep.center(c), c)


# ---------------- batch-size independence ----------------


@pytest.mark.parametrize("B", [1, 7, 37, 256])
def test_batch_shapes_consistent(B):
    """Shape derivation for W @ (a batch of k 256-vectors), confirming that B only
    affects the number of columns.
    """
    ps = "ML-KEM-768"
    _, dk = refimpl.ref_keygen(ps)
    kp = keyprep.parse_key(ps, dk)
    Y = np.random.default_rng(B).integers(-2, 3, size=(kp.K, B))
    out = kp.W_enc @ Y
    assert out.shape == (kp.M, B)
