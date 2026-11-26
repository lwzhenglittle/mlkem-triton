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


def test_scheme_verifiers_refuse_mismatched_batches():
    """A batch of 10 messages and 9 signatures must not verify 9 of them and misalign the rest.

    All four verifiers check this before any kernel runs; the assertion here is that the check
    exists, because a silent `zip` would produce a well-formed `[9]` verdict array that a caller
    would read as `[10]`.
    """
    import pathlib
    import re

    from triton_crypto.falcon import Verifier

    src = pathlib.Path(__file__).resolve().parent / "test_falcon.py"
    text = src.read_text()

    def grab(name):
        block = re.search(name + r" = \(\n(.*?)\n\)", text, re.S).group(1)
        return bytes.fromhex("".join(re.findall(r'"([0-9a-f]+)"', block)))

    pk, sig = grab("PK_HEX"), grab("SIG_HEX")
    msg = re.search(r'MSG = b"(.*?)"', text).group(1).encode()
    verifier = Verifier(pk)
    assert verifier.verify([msg], [sig]).tolist() == [True]
    with pytest.raises(ValueError, match=r"1 messages but 2 signatures"):
        verifier.verify([msg], [sig, sig])
    with pytest.raises(ValueError, match="2 messages"):
        verifier.verify([msg, msg], [sig])


def test_key_generators_refuse_a_negative_count():
    from triton_crypto import generate_keypairs

    with pytest.raises(ValueError, match="non-negative"):
        generate_keypairs("ML-KEM-768", -1)


def test_scheme_constructors_refuse_wrong_key_lengths():
    from triton_crypto import Decapsulator, Encapsulator

    with pytest.raises(ValueError, match="dk length"):
        Decapsulator("ML-KEM-768", bytes(100))
    with pytest.raises(ValueError, match="ek length"):
        Encapsulator("ML-KEM-768", bytes(100))


def test_device_entry_points_accept_device_tensors():
    """`gemm_int8_torch` is documented as running *on the device*.

    The dtype check added earlier routed both operands through `np.asarray`, so a device tensor
    raised "can't convert xpu:0 device type tensor to numpy" -- a confusing way to say the dtype
    is fine. Both a host array and a device tensor must work and agree.
    """
    import torch

    a = np.arange(64, dtype=np.int8).reshape(8, 8)
    b = np.ones((8, 8), np.int8)
    want = gemm.gemm_int8_np(a, b)
    assert np.array_equal(gemm.gemm_int8_torch(a, b).cpu().numpy(), want)
    dev = torch.device("xpu")
    got = (
        gemm.gemm_int8_torch(torch.as_tensor(a, device=dev), torch.as_tensor(b, device=dev))
        .cpu()
        .numpy()
    )
    assert np.array_equal(got, want)


@pytest.mark.parametrize("api", ["blake3", "chacha20", "poseidon2", "ntt"])
def test_device_entry_points_do_not_modify_their_inputs(api):
    """The NTT used to: its ping-pong buffer wrote pass 18 into the caller's tensor on a 19-pass
    transform, so a second call transformed the wreckage of the first. The invariant is worth
    pinning for every device entry point, not just that one.
    """
    import torch

    dev = torch.device("xpu")
    if api == "blake3":
        msgs = np.zeros((4, 64), np.uint8)
        before = msgs.copy()
        blake3.blake3_hash_torch(msgs)
        assert np.array_equal(before, msgs)
    elif api == "chacha20":
        from triton_crypto.primitives import chacha20

        keys = np.zeros((4, 32), np.uint8)
        nonces = np.zeros((4, 12), np.uint8)
        before = keys.copy()
        chacha20.chacha20_keystream(keys, nonces, 64)
        assert np.array_equal(before, keys)
    elif api == "poseidon2":
        state = torch.zeros(16, dtype=torch.int32, device=dev)
        snapshot = state.cpu().numpy().copy()
        poseidon2.poseidon2_permute_torch(state)
        assert np.array_equal(snapshot, state.cpu().numpy())
    else:
        from triton_crypto.primitives.ntt import LargeNTT

        t = LargeNTT(1 << 12, 2013265921)
        x = torch.ones(1 << 12, dtype=torch.int32, device=dev)
        snapshot = x.cpu().numpy().copy()
        t.forward_tensor_torch(x, dev)
        assert np.array_equal(snapshot, x.cpu().numpy())
        t.inverse_tensor_torch(x, dev)
        assert np.array_equal(snapshot, x.cpu().numpy())


def test_blake3_accepts_a_device_tensor():
    """A tensor is the input a caller most plausibly has on a GPU machine."""
    import torch

    rng = np.random.default_rng(9)
    msgs = rng.integers(0, 256, (3, 300), dtype=np.uint8)
    want = blake3.blake3_hash_torch(msgs).cpu().numpy()
    dev = torch.device("xpu")
    assert np.array_equal(
        blake3.blake3_hash_torch(torch.as_tensor(msgs, device=dev)).cpu().numpy(), want
    )
    assert np.array_equal(blake3.blake3_hash_torch(torch.as_tensor(msgs)).cpu().numpy(), want)
    with pytest.raises(TypeError, match="integer bytes"):
        blake3.blake3_hash_torch(torch.zeros((2, 8), dtype=torch.float32, device=dev))


def test_mul_torch_accepts_device_operands():
    """ "Both operands and the result stay on the device" is what the docstring promises."""
    import torch

    from triton_crypto.primitives.ntt import LargeNTT

    q = 2013265921
    t = LargeNTT(1 << 10, q)
    a = np.random.default_rng(10).integers(0, q, 1 << 10)
    dev = torch.device("xpu")
    ta = torch.as_tensor(a, dtype=torch.int32, device=dev)
    assert np.array_equal(t.mul_torch(ta, ta).cpu().numpy().astype(np.int64), t.mul_np(a, a))
    # and the host path still works
    assert np.array_equal(t.mul_torch(a, a).cpu().numpy().astype(np.int64), t.mul_np(a, a))


@pytest.mark.parametrize("api", ["chacha20_keystream", "chacha20_xor"])
def test_chacha20_accepts_device_tensors(api):
    """The same sweep that fixed the GEMM, `mul_torch` and BLAKE3, applied to the last module.

    A GPU caller has device tensors; `np.asarray` on one raises "can't convert xpu:0 device type
    tensor to numpy", which says nothing about what was actually wrong (nothing was).
    """
    import torch

    from triton_crypto.primitives import chacha20

    dev = torch.device("xpu")
    keys = np.zeros((2, 32), np.uint8)
    nonces = np.zeros((2, 12), np.uint8)
    if api == "chacha20_keystream":
        want = chacha20.chacha20_keystream(keys, nonces, 64)
        got = chacha20.chacha20_keystream(
            torch.as_tensor(keys, device=dev), torch.as_tensor(nonces, device=dev), 64
        )
    else:
        data = np.zeros((2, 64), np.uint8)
        want = chacha20.chacha20_xor(keys, nonces, data)
        got = chacha20.chacha20_xor(
            torch.as_tensor(keys, device=dev),
            torch.as_tensor(nonces, device=dev),
            torch.as_tensor(data, device=dev),
        )
    assert np.array_equal(got, want)
