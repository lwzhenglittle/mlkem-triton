"""BLAKE3 on the device: one launch per block index, one per tree level.

Two things this file exists to pin, both of which cost sessions:

* **`counter >> 32` on a 32-bit value is undefined behaviour.** The Xe backend accepted it,
  emitted a warning that a test log filters out, and folded the whole kernel's results to zero --
  with every input reading back correct from inside the same launch. The counter is now passed as
  an explicit `(low, high)` pair, and the first test here is the one that would have caught it:
  a *non-zero chunk counter* (any message over 1 KiB) has to change the digest.
* the empty message needs `n >= 1` on the device path (the kernel reads at least one byte), so
  it is answered by the numpy reference rather than by a special case in the kernel.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from triton_crypto.primitives.blake3 import blake3_hash_np, blake3_hash_torch

VECTORS = Path(__file__).resolve().parent.parent / "data" / "vectors" / "blake3" / "kat.json"


def _msg(n: int) -> bytes:
    return bytes((i % 251) for i in range(n))


def test_official_vectors_on_device():
    kat = json.loads(VECTORS.read_text())["vectors"]
    for v in kat:
        if v["len"] == 0:
            continue  # the device path needs at least one byte; the reference covers the empty case
        got = blake3_hash_torch(np.frombuffer(_msg(v["len"]), dtype=np.uint8)[None, :])
        assert got.cpu().numpy()[0].tobytes().hex() == v["hash"], f"length {v['len']}"


def test_multichunk_message_uses_the_chunk_counter():
    """4 chunks: if the counter were folded away, all chunks would compress identically."""
    n = 4096
    got = blake3_hash_torch(np.frombuffer(_msg(n), dtype=np.uint8)[None, :]).cpu().numpy()[0]
    assert got.tobytes().hex() == blake3_hash_np(_msg(n)).hex()


@pytest.mark.parametrize("n", [1, 64, 65, 1024, 1025, 5000])
def test_batches_are_independent(n):
    """Rows of equal length but different content must not influence each other."""
    base = np.frombuffer(_msg(n), dtype=np.uint8)
    msgs = np.stack([np.roll(base, i) for i in range(8)])
    got = blake3_hash_torch(msgs).cpu().numpy()
    for i in range(8):
        assert got[i].tobytes().hex() == blake3_hash_np(bytes(np.roll(base, i))).hex(), f"row {i}"


def test_ragged_batch_size_off_the_block_multiple():
    """The grid is `cdiv(batch, 64)`, so a batch of 37 has lanes past the end."""
    msgs = np.stack([np.frombuffer(_msg(200), dtype=np.uint8)] * 37)
    msgs[-1] = np.frombuffer(_msg(200), dtype=np.uint8)[::-1].copy()  # one row differs
    got = blake3_hash_torch(msgs).cpu().numpy()
    assert got[0].tobytes().hex() == blake3_hash_np(_msg(200)).hex()
    assert got[-1].tobytes().hex() != got[0].tobytes().hex()


def test_empty_message_is_refused_on_the_device_path():
    with pytest.raises(ValueError, match="at least one byte"):
        blake3_hash_torch(np.zeros((2, 0), dtype=np.uint8))
