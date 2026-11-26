"""Poseidon2 on the device: the Triton kernel against Plonky3's vectors and against numpy.

The vectors are a cross-implementation check (they come from `p3-baby-bear`, not from this
package); the numpy reference is what makes arbitrary batches trustworthy, including sizes
that are not a multiple of the kernel's block and sizes below it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triton_crypto.primitives.poseidon2 import (
    BABYBEAR_POSEIDON2_16,
    poseidon2_permute_np,
    poseidon2_permute_torch,
)

VECTORS = Path(__file__).resolve().parent.parent / "data" / "vectors" / "poseidon2" / "bb16.json"
BATCHES = [1, 2, 7, 255, 256, 257, 1000]


def test_plonky3_vectors_on_device():
    kat = json.loads(VECTORS.read_text())
    src = np.array([v["in"] for v in kat["vectors"]], dtype=np.int32)
    got = poseidon2_permute_torch(src).cpu().numpy()
    expected = np.array([v["out"] for v in kat["vectors"]], dtype=np.int64)
    assert np.array_equal(got.astype(np.int64), expected)


@pytest.mark.parametrize("batch", BATCHES)
def test_device_matches_numpy(batch):
    rng = np.random.default_rng(batch)
    x = rng.integers(0, BABYBEAR_POSEIDON2_16.q, (batch, 16))
    ref = poseidon2_permute_np(x)
    got = poseidon2_permute_torch(x).cpu().numpy()
    assert np.array_equal(got.astype(np.int64), ref)


def test_single_state_round_trip():
    rng = np.random.default_rng(7)
    x = rng.integers(0, BABYBEAR_POSEIDON2_16.q, 16)
    ref = poseidon2_permute_np(x)
    got = poseidon2_permute_torch(x).cpu().numpy()
    assert got.shape == (16,)
    assert np.array_equal(got.astype(np.int64), ref)


def test_out_buffer_is_used():
    import torch

    rng = np.random.default_rng(11)
    x = rng.integers(0, BABYBEAR_POSEIDON2_16.q, (64, 16))
    out = torch.empty((64, 16), dtype=torch.int32, device="xpu")
    got = poseidon2_permute_torch(x, out=out)
    assert got.data_ptr() == out.data_ptr()
    assert np.array_equal(got.cpu().numpy().astype(np.int64), poseidon2_permute_np(x))


def test_extremes_on_device():
    q = BABYBEAR_POSEIDON2_16.q
    x = np.stack([np.zeros(16, dtype=np.int64), np.full(16, q - 1, dtype=np.int64)])
    got = poseidon2_permute_torch(x).cpu().numpy().astype(np.int64)
    assert got.min() >= 0 and got.max() < q
    assert np.array_equal(got, poseidon2_permute_np(x))
