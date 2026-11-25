"""Acceptance for key rebuild on the GPU (`kernels/rebuild.py`).

This module exists only for measurement (mirroring cuPQC's stateless API, which
recomputes Â per op), so the requirement is **element-wise equality with the CPU's
`keyprep.parse_key`** — if they differ, every "same-basis" conclusion is void.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from triton_crypto import acvp, keyprep
from triton_crypto.kernels.rebuild import GpuKeyRebuilder, intt_matrix
from triton_crypto.ntt_ref import intt
from triton_crypto.params import N, Q
from triton_crypto.pipeline import Decapsulator

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")


def _dk(ps: str) -> bytes:
    c = next(x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == ps)
    return bytes.fromhex(c["inputs"]["dk"])


def test_intt_matrix_is_exact():
    """INTT uses a 256×256 matmul instead of butterflies and must be bit-exact (2^53 is
    enough in fp64).
    """
    mt = intt_matrix(DEV)
    for seed in range(4):
        np.random.seed(seed)
        x = np.random.randint(0, Q, N)
        got = (
            (torch.matmul(torch.from_numpy(x.astype(np.float64)).to(DEV), mt).to(torch.int64) % Q)
            .cpu()
            .numpy()
        )
        assert np.array_equal(got, intt(x)), seed


@pytest.mark.parametrize("ps", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_rebuild_matches_keyprep(ps):
    """The rebuilt W_dec / W_enc must equal the CPU path element by element (in fp16)."""
    dk = _dk(ps)
    kp = keyprep.parse_key(ps, dk)
    Wd, We = GpuKeyRebuilder(ps, device=DEV).rebuild(dk)
    torch_sync = getattr(torch, DEV.type).synchronize
    torch_sync()

    Wd_ref = torch.from_numpy(kp.W_dec).to(torch.float16).cpu()
    We_ref = torch.from_numpy(kp.W_enc).to(torch.float16).cpu()
    assert tuple(Wd.shape) == tuple(Wd_ref.shape)
    assert tuple(We.shape) == tuple(We_ref.shape)
    assert torch.equal(Wd.cpu(), Wd_ref), "W_dec mismatch"
    assert torch.equal(We.cpu(), We_ref), "W_enc mismatch"


@pytest.mark.parametrize("ps", ["ML-KEM-768"])
def test_rebuilt_weights_give_correct_decaps(ps):
    """Run the whole pipeline with the rebuilt weights; the ACVP vectors must pass —
    only that proves the weight semantics are right.
    """

    c = next(x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == ps)
    dk = bytes.fromhex(c["inputs"]["dk"])
    ct = bytes.fromhex(c["inputs"]["c"])
    want = bytes.fromhex(c["expected"]["k"])

    dec = Decapsulator(ps, dk, device=DEV)
    assert dec.decaps([ct] * 4)[0].tobytes() == want  # pre-expanded path

    dec.W_dec, dec.W_enc = GpuKeyRebuilder(ps, device=DEV).rebuild(dk)
    got = dec.decaps([ct] * 4)
    assert all(x.tobytes() == want for x in got), "decapsulation with rebuilt weights is wrong"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
