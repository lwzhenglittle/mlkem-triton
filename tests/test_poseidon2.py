"""Poseidon2-BabyBear/16 against Plonky3's own vectors (CPU-only, no GPU needed).

`data/vectors/poseidon2/bb16.json` is emitted by `benchmarks/poseidon2_ref` (cargo mode
`kat`), which calls `p3_baby_bear::default_babybear_poseidon2_16()` directly. That makes it a
real cross-implementation check rather than a self-consistency check: the constants, the
round order and the two linear layers all have to agree.

The vectors caught two bugs while this module was being written, which is why they are here:

* Plonky3 applies an external **linear layer before the first round**
  (`external_initial_permute_state` starts with `mds_light_permutation`); without it every
  output is wrong but nothing crashes.
* the S-box is x^7 = (x^2)^2 * x^2 * x, not (x^2)^2 * x * x, which is x^6.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triton_crypto.primitives.poseidon2 import (
    BABYBEAR_POSEIDON2_16,
    poseidon2_permute_np,
)

VECTORS = Path(__file__).resolve().parent.parent / "data" / "vectors" / "poseidon2" / "bb16.json"


def _load() -> dict:
    return json.loads(VECTORS.read_text())


def test_parameters():
    p = BABYBEAR_POSEIDON2_16
    assert (p.q, p.width, p.full_rounds, p.partial_rounds) == (2013265921, 16, 8, 13)
    assert p.half_rounds == 4
    assert len(p.rc_initial) == 4
    assert len(p.rc_final) == 4
    assert len(p.rc_internal) == 13
    assert len(p.diag) == 16
    assert p.diag[0] == p.q - 2  # the -2 that its AVX2 implementation wants
    # every M4 row is a permutation of 1,1,2,3-ish weights; the matrix is MDS and circulant
    assert sum(p.m4[0]) == 7 and sum(p.m4[3]) == 7


def test_plonky3_vectors():
    kat = _load()
    assert kat["q"] == BABYBEAR_POSEIDON2_16.q
    for i, v in enumerate(kat["vectors"]):
        got = poseidon2_permute_np(np.array(v["in"]))
        assert got.tolist() == v["out"], f"vector {i}"


def test_batch_shape_matches_single():
    kat = _load()
    batch = np.array([v["in"] for v in kat["vectors"]])
    many = poseidon2_permute_np(batch)
    for i, v in enumerate(kat["vectors"]):
        assert many[i].tolist() == v["out"]
    # 1-D input stays 1-D
    assert poseidon2_permute_np(batch[0]).shape == (16,)


def test_permutation_is_not_identity_and_is_deterministic():
    kat = _load()
    x = np.array(kat["vectors"][0]["in"])
    once = poseidon2_permute_np(x)
    assert once.tolist() != x.tolist()
    assert poseidon2_permute_np(x).tolist() == once.tolist()


def test_extremes_are_in_range():
    q = BABYBEAR_POSEIDON2_16.q
    for state in (np.zeros(16, dtype=np.int64), np.full(16, q - 1, dtype=np.int64)):
        out = poseidon2_permute_np(state)
        assert out.dtype == np.int64
        assert out.min() >= 0 and out.max() < q


def test_rejects_out_of_range_and_wrong_width():
    q = BABYBEAR_POSEIDON2_16.q
    with pytest.raises(ValueError, match="16 field elements"):
        poseidon2_permute_np(np.zeros(15, dtype=np.int64))
    with pytest.raises(ValueError, match="must be in"):
        poseidon2_permute_np(np.full(16, q, dtype=np.int64))
    with pytest.raises(ValueError, match="must be in"):
        poseidon2_permute_np(np.full(16, -1, dtype=np.int64))


def test_declared_kernel_contract():
    """The vector file must name its source, so a stale file is detectable."""
    kat = _load()
    assert "plonky3" in kat["source"].lower()
    assert kat["full_rounds"] == BABYBEAR_POSEIDON2_16.full_rounds
    assert kat["partial_rounds"] == BABYBEAR_POSEIDON2_16.partial_rounds
