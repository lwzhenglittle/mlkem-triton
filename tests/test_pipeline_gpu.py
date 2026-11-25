"""GPU 常驻流水线（`pipeline.Decapsulator`）的验收测试。"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV

from mlkem_triton import acvp, refimpl
from mlkem_triton.pipeline import Decapsulator

pytest.importorskip("kyber_py", reason="refimpl 需要可选的 kyber-py")

pytest.importorskip("triton")
XPU = True  # 见 _common.DEV：能 import 就说明有可用 GPU
pytestmark = pytest.mark.skipif(not XPU, reason="需要 GPU")

PARAM_SETS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_gpu_roundtrip(ps):
    from mlkem_triton.pipeline import Decapsulator

    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    cts, expect = [], []
    for _ in range(4):
        K, c = refimpl.ref_encaps(ps, ek)
        cts.append(np.frombuffer(c, np.uint8))
        expect.append(K)
    got = dec.decaps(np.stack(cts))
    assert [got[i].tobytes() for i in range(len(expect))] == expect


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_gpu_official_acvp(ps):
    from mlkem_triton.pipeline import Decapsulator

    cases = [c for c in acvp.load() if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    assert cases
    for c in cases:
        dk = bytes.fromhex(c["inputs"]["dk"])
        ct = bytes.fromhex(c["inputs"]["c"])
        got = Decapsulator(ps, dk).decaps(np.frombuffer(ct, np.uint8)[None, :])
        assert got[0].tobytes().hex().upper() == c["expected"]["k"].upper(), f"tcId={c['tcId']}"


def test_gpu_implicit_rejection():
    from mlkem_triton.pipeline import Decapsulator

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
        assert got[i].tobytes() == ref
        assert got[i].tobytes() != K


def test_gpu_batch_one():
    from mlkem_triton.pipeline import Decapsulator

    ps = "ML-KEM-768"
    ek, dk = refimpl.ref_keygen(ps)
    K, c = refimpl.ref_encaps(ps, ek)
    got = Decapsulator(ps, dk).decaps(np.frombuffer(c, np.uint8)[None, :])
    assert got[0].tobytes() == K


def test_unpack_matches_numpy():
    from mlkem_triton.kernels.elementwise import _unpack_bits
    from mlkem_triton.pipeline import unpack_bits_gpu

    rng = np.random.default_rng(0)
    for d, count in [(10, 768), (4, 256), (11, 1024), (12, 256)]:
        nb = count * d // 8
        data = rng.integers(0, 256, size=(5, nb), dtype=np.uint8)
        ref = _unpack_bits(data, d, count).T  # [5, count]
        got = unpack_bits_gpu(torch.from_numpy(data).to(DEV), d, count).cpu().numpy()
        assert np.array_equal(got, ref), (d, count)


def test_cbd_matches_reference():
    from mlkem_triton.pipeline import cbd_gpu
    from mlkem_triton.sampling_cpu import cbd as cbd_ref

    rng = np.random.default_rng(1)
    for eta in (2, 3):
        bufs = rng.integers(0, 256, size=(17, 64 * eta), dtype=np.uint8)
        ref = np.stack([cbd_ref(bufs[i].tobytes(), eta) for i in range(17)])
        got = cbd_gpu(torch.from_numpy(bufs).to(DEV), eta).cpu().numpy()
        assert np.array_equal(got, ref), eta


# ---------------------------------------------------------------- 输入形态
@pytest.mark.parametrize("ps", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_all_input_forms_agree(ps):
    """四种输入形态必须给出同一批输出（bytes / ndarray / list / 设备张量）。"""
    from mlkem_triton import refimpl

    ek, dk = refimpl.ref_keygen(ps)
    dec = Decapsulator(ps, dk)
    kl = dec.p.ct_size
    cts = []
    for _ in range(8):
        _, c = refimpl.ref_encaps(ps, ek)
        cts.append(c)
    joined = b"".join(cts)

    ref = dec.decaps(joined)  # bytes
    for name, arg in [
        ("np.ndarray", np.frombuffer(joined, np.uint8).reshape(len(cts), kl)),
        ("np.ndarray 可写", np.frombuffer(joined, np.uint8).reshape(len(cts), kl).copy()),
        ("bytearray", bytearray(joined)),
        ("memoryview", memoryview(joined)),
        ("list[bytes]", list(cts)),
    ]:
        got = dec.decaps(arg)
        assert got.shape == ref.shape, name
        assert np.array_equal(got, ref), name

    dev = torch.from_numpy(np.frombuffer(joined, np.uint8).reshape(len(cts), kl).copy()).to(DEV)
    assert np.array_equal(dec.decaps(dev), ref), "torch.Tensor"


def test_bad_input_forms_raise():
    from mlkem_triton import refimpl

    ek, dk = refimpl.ref_keygen("ML-KEM-768")
    dec = Decapsulator("ML-KEM-768", dk)
    kl = dec.p.ct_size
    _, c = refimpl.ref_encaps("ML-KEM-768", ek)

    with pytest.raises(ValueError):  # 长度不是整数倍
        dec.decaps(c + b"\x00")
    with pytest.raises(ValueError):  # 最后维不是 ct_size
        dec.decaps(np.zeros((4, kl - 1), np.uint8))
    with pytest.raises(TypeError):  # dtype 不对
        dec.decaps(torch.zeros(4, kl, dtype=torch.int16, device=DEV))


def test_pinned_staging_reuse_is_safe():
    """暂存缓冲会被复用：连调两次、输入不同，第二次的结果必须是第二次的。"""
    from mlkem_triton import refimpl

    ek, dk = refimpl.ref_keygen("ML-KEM-768")
    dec = Decapsulator("ML-KEM-768", dk)
    K1, c1 = refimpl.ref_encaps("ML-KEM-768", ek)
    K2, c2 = refimpl.ref_encaps("ML-KEM-768", ek)
    B = 32
    a = dec.decaps([c1] * B)
    b = dec.decaps([c2] * B)
    assert all(x.tobytes() == K1 for x in a)
    assert all(x.tobytes() == K2 for x in b)


def test_eta2_fast_path_only_for_ukem768_1024():
    """记录一个已知限制：融合 CBD 只支持 η=2，所以 ML-KEM-512 走慢路径。

    功能是对的（ACVP 向量过），只是慢 ~3x。这个测试就是为了让这条限制在
    「哪天有人把 η=3 也实现了」时立刻暴露出来。
    """
    from mlkem_triton import refimpl
    from mlkem_triton.kernels import glue

    for ps, expect_fast in (("ML-KEM-512", False), ("ML-KEM-768", True), ("ML-KEM-1024", True)):
        _ek, dk = refimpl.ref_keygen(ps)
        dec = Decapsulator(ps, dk)
        p = dec.p
        has_fast = p.eta1 == 2 and p.eta2 == 2
        assert has_fast is expect_fast, ps
        if expect_fast:
            # 快路径的判定条件确实会接受真实形状
            B, k = 8, p.k
            y_buf = torch.zeros(B * k, 64 * p.eta1, dtype=torch.uint8, device=DEV)
            assert glue.can_cbd_y(y_buf, B, k, p.eta1)
