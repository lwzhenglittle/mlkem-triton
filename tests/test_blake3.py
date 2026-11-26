"""BLAKE3 against the official vectors, chunk by chunk.

`data/vectors/blake3/kat.json` comes from the official Rust implementation
(`benchmarks/blake3_ref`, cargo mode `kat`) for 26 lengths from 0 to 64 KiB. The lengths are the
point: 0, 1, 63/64/65 and 127/128/129 cross the block and multi-block boundaries, 1023/1024/1025
cross the chunk boundary, and 5000 (5 chunks) and 31744 (31 chunks) are the non-power-of-two
chunk counts where the tree shape -- "pair adjacent, carry an odd node up unchanged" -- is the
thing under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from triton_crypto.primitives.blake3 import BLAKE3_IV, blake3_hash_np, compression_np

VECTORS = Path(__file__).resolve().parent.parent / "data" / "vectors" / "blake3" / "kat.json"


def _inputs():
    """The generator hashes `i % 251` bytes, so the vectors need no input data shipped."""
    return json.loads(VECTORS.read_text())


def test_parameters():
    assert len(BLAKE3_IV) == 8
    assert BLAKE3_IV[0] == 0x6A09E667  # SHA-256's IV, reused
    assert BLAKE3_IV[-1] == 0x5BE0CD19


def test_empty_message_is_the_published_constant():
    """BLAKE3's empty-input digest, quoted in the specification."""
    assert blake3_hash_np(b"").hex() == (
        "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
    )


def test_every_official_vector():
    kat = _inputs()
    assert len(kat["vectors"]) >= 26
    for v in kat["vectors"]:
        msg = bytes((i % 251) for i in range(v["len"]))
        assert blake3_hash_np(msg).hex() == v["hash"], f"length {v['len']}"


def test_chunk_boundaries_change_the_tree_shape():
    """1023/1024/1025 are three different trees (one partial chunk, one exact chunk, two)."""
    kat = {v["len"]: v["hash"] for v in _inputs()["vectors"]}
    assert kat[1023] != kat[1024] != kat[1025]
    for n in (1023, 1024, 1025):
        assert blake3_hash_np(bytes((i % 251) for i in range(n))).hex() == kat[n]


def test_xof_length_is_rejected_beyond_one_block():
    with pytest.raises(ValueError, match="out_len above 64"):
        blake3_hash_np(b"x", out_len=65)


def test_compression_is_a_function_of_all_sixteen_words():
    """A compression that ignored part of the message would still be deterministic, so this
    checks that changing *any* word changes the output."""
    key = list(BLAKE3_IV)
    base = list(range(16))
    ref, _ = compression_np(key, base, 0, 64, 0)
    for i in range(16):
        other = list(base)
        other[i] ^= 1
        got, _ = compression_np(key, other, 0, 64, 0)
        assert got != ref, f"word {i} does not affect the chaining value"


def test_the_counter_and_flags_matter():
    key, block = list(BLAKE3_IV), list(range(16))
    base, _ = compression_np(key, block, 0, 64, 0)
    assert compression_np(key, block, 1, 64, 0)[0] != base
    assert compression_np(key, block, 0, 63, 0)[0] != base
    assert compression_np(key, block, 0, 64, 8)[0] != base
