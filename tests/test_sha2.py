"""SHA-256 / SHA-512: the numpy primitives against hashlib, the device kernel against numpy."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from triton_crypto.primitives import sha2

BITS = (256, 512)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("msg_len", [0, 1, 55, 56, 63, 64, 65, 200, 1000])
def test_numpy_matches_hashlib(bits, msg_len):
    msg = bytes((i * 7 + 3) & 0xFF for i in range(msg_len))
    want = hashlib.new("sha256" if bits == 256 else "sha512", msg).digest()
    assert sha2.hash_np(msg, bits) == want


def _device_block(bits: int, rows: int, blocks: int):
    """A [rows, blocks*block] random byte buffer and its word view."""
    bsize = 64 if bits == 256 else 128
    raw = np.random.default_rng(0).integers(0, 256, (rows, blocks * bsize), dtype=np.uint8)
    t = torch.from_numpy(raw).to("xpu" if torch.xpu.is_available() else "cpu")
    dtype = torch.int32 if bits == 256 else torch.int64
    return raw, t.contiguous().view(dtype)


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("blocks", [1, 3])
def test_kernel_matches_numpy(bits, blocks):
    if not (hasattr(torch, "xpu") and torch.xpu.is_available()) and not torch.cuda.is_available():
        pytest.skip("no GPU")
    dev = "xpu" if torch.xpu.is_available() else "cuda"
    rows = 5
    raw, words = _device_block(bits, rows, blocks)
    out = torch.empty((rows, 8), dtype=torch.int32 if bits == 256 else torch.int64, device=dev)
    hin = sha2.iv_tensor(bits, dev)
    sha2.sha_absorb(hin, words, out, rows, bits, blocks)
    wsize = bits // 64
    got = out.view(torch.uint8).cpu().numpy()
    for r in range(rows):
        state = sha2.compress_np(
            list(sha2.IV256 if bits == 256 else sha2.IV512), raw[r].tobytes(), bits
        )
        want = b"".join(x.to_bytes(wsize, "big") for x in state)
        assert bytes(got[r]) == want
