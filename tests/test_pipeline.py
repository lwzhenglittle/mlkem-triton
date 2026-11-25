"""Phase 3.2/4 验收：elementwise 与端到端解封装。"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from mlkem_triton import acvp, refimpl
from mlkem_triton.params import Q

pytest.importorskip("triton")

from mlkem_triton.kernels import elementwise as ew

pytest.importorskip("kyber_py", reason="refimpl 需要可选的 kyber-py")

XPU = True  # 见 _common.DEV：能 import 就说明有可用 GPU
pytestmark = pytest.mark.skipif(not XPU, reason="需要 GPU")


# ---------------- elementwise ----------------


@pytest.mark.parametrize("d", [1, 4, 10, 11, 12])
def test_compress_decompress_torch_matches_numpy(d):
    rng = np.random.default_rng(d)
    x = rng.integers(0, 1 << d, size=256)
    assert np.array_equal(ew.decompress_t(torch.from_numpy(x), d).numpy(), ew.decompress_np(x, d))

    y = rng.integers(0, Q, size=256)
    assert np.array_equal(ew.compress_t(torch.from_numpy(y), d).numpy(), ew.compress_np(y, d))


@pytest.mark.parametrize("d", [1, 4, 10, 11, 12])
def test_pack_unpack_roundtrip(d):
    rng = np.random.default_rng(d + 50)
    count, B = 256 * 3, 5
    vals = rng.integers(0, 1 << d, size=(count, B))
    packed = ew.pack_bits(vals, d)
    assert packed.shape == (B, count * d // 8)
    back = ew._unpack_bits(packed, d, count)
    assert np.array_equal(back, vals)


def test_select_bytes_constant_shape():
    kp = np.array([[1, 2, 3], [4, 5, 6]], np.uint8)
    kb = np.array([[9, 9, 9], [8, 8, 8]], np.uint8)
    out = ew.select_bytes(kp, kb, np.array([True, False]))
    assert np.array_equal(out, np.array([[1, 2, 3], [8, 8, 8]]))


# ---------------- 端到端 ----------------

PARAM_SETS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_pipeline_roundtrip(ps):
    from mlkem_triton.pipeline_host_hash import ReferenceDecapsulator as Decapsulator

    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    cts, expect = [], []
    for _ in range(5):
        K, c = refimpl.ref_encaps(ps, ek)
        cts.append(np.frombuffer(c, np.uint8))
        expect.append(K)
    got = dec.decaps(np.stack(cts))
    for i, K in enumerate(expect):
        assert got[i].tobytes() == K, f"{ps} 第 {i} 条不符"


@pytest.mark.parametrize("B", [1, 2, 3])
def test_pipeline_batch_sizes(B):
    from mlkem_triton.pipeline_host_hash import ReferenceDecapsulator as Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    cts, expect = [], []
    for _ in range(B):
        K, c = refimpl.ref_encaps(ps, ek)
        cts.append(np.frombuffer(c, np.uint8))
        expect.append(K)
    got = dec.decaps(np.stack(cts))
    assert [got[i].tobytes() for i in range(B)] == expect


def test_pipeline_implicit_rejection():
    """篡改密文必须走隐式拒绝，且与参考实现一致。"""
    from mlkem_triton.pipeline_host_hash import ReferenceDecapsulator as Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    K, c = refimpl.ref_encaps(ps, ek)

    cts = []
    for pos in (0, len(c) // 2, len(c) - 1):
        bad = bytearray(c)
        bad[pos] ^= 1
        cts.append(np.frombuffer(bytes(bad), np.uint8))

    got = dec.decaps(np.stack(cts))
    for i, ct in enumerate(cts):
        ref = refimpl.ref_decaps(ps, dk, ct.tobytes())
        assert got[i].tobytes() == ref, f"pos 变体 {i} 不一致"
        assert got[i].tobytes() != K
        assert got[i].tobytes() == refimpl.implicit_reject_key(ps, dk, ct.tobytes())


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_pipeline_official_acvp_decaps(ps):
    """官方 ACVP 解封装向量（含隐式拒绝）跑通整条流水线。"""
    from mlkem_triton.pipeline_host_hash import ReferenceDecapsulator as Decapsulator

    cases = [c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    assert cases
    for c in cases:
        dk = bytes.fromhex(c["inputs"]["dk"])
        ct = bytes.fromhex(c["inputs"]["c"])
        dec = Decapsulator(ps, dk)
        got = dec.decaps(np.frombuffer(ct, np.uint8)[None, :])
        assert got[0].tobytes().hex().upper() == c["expected"]["k"].upper(), f"tcId={c['tcId']}"
