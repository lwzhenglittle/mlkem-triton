"""What each primitive refuses, and why the refusal matters.

Every case here used to *succeed* and return something: a float array truncated to bytes, an
integer 300 wrapped to 44, a wrong-length polynomial read past the end of its buffer. A silent
wrong answer is the worst failure mode a primitive library can have -- the digest is well-formed,
the product is well-formed, and nothing says otherwise -- so the refusals are part of the API and
are tested like any other behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.primitives import aes, blake3, gemm, poly1305, poseidon2


def test_blake3_refuses_floats():
    with pytest.raises(TypeError, match="integer bytes"):
        blake3.blake3_hash_torch(np.zeros((2, 64), np.float32))


def test_blake3_refuses_out_of_range_integers():
    """300 wraps to 44 and -1 to 255 in a `uint8` cast, so the digest would be of other bytes."""
    with pytest.raises(ValueError, match=r"byte values"):
        blake3.blake3_hash_torch(np.array([[300, -1]], dtype=np.int64))


def test_blake3_accepts_a_wider_integer_dtype_in_range():
    wide = np.array([[1, 2, 3]], dtype=np.int64)
    narrow = wide.astype(np.uint8)
    assert np.array_equal(
        blake3.blake3_hash_torch(wide).cpu().numpy(), blake3.blake3_hash_torch(narrow).cpu().numpy()
    )


def test_gemm_refuses_anything_but_int8():
    """The contract is *exact* int8 x int8 -> int32; a float operand would be truncated into it."""
    with pytest.raises(TypeError, match="must be int8"):
        gemm.gemm_int8_torch(np.zeros((4, 4), np.float32), np.zeros((4, 4), np.int8))
    with pytest.raises(TypeError, match="must be int8"):
        gemm.gemm_int8_torch(np.zeros((4, 4), np.int8), np.zeros((4, 4), np.int32))


def test_gemm_refuses_mismatched_inner_dimension():
    with pytest.raises(ValueError):
        gemm.gemm_int8_torch(np.zeros((4, 5), np.int8), np.zeros((6, 3), np.int8))


@pytest.mark.parametrize(
    ("pattern", "call"),
    [
        ("AES-128 key must be 16", lambda: aes.aes128_encrypt_block_np(b"k" * 8, bytes(16))),
        ("AES block must be 16", lambda: aes.aes128_encrypt_block_np(bytes(16), bytes(8))),
    ],
)
def test_aes_refuses_wrong_lengths(pattern, call):
    with pytest.raises(ValueError, match=pattern):
        call()


def test_aes_ctr_refuses_a_short_nonce():
    with pytest.raises(ValueError, match="nonce"):
        aes.aes128_ctr_np(bytes(16), bytes(4), 16)


def test_poseidon2_refuses_a_narrow_state():
    with pytest.raises(ValueError, match="16 field elements"):
        poseidon2.poseidon2_permute_torch(np.zeros(8, np.int32))


def test_poly1305_refuses_wrong_key_lengths():
    for n in (16, 64):
        with pytest.raises(ValueError, match="32 bytes"):
            poly1305.poly1305_np(b"k" * n, b"m")
