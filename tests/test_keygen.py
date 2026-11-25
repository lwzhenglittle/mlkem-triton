"""密钥生成（FIPS 203 Alg. 16）验收。

keygen 是三个操作里唯一用 NTT 的，也是最容易错的一个（我第一版把
`G(d ‖ k)` 写成了 `G(d)`，ρ 全错）。所以这里用**逐字节对拍 kyber-py** 来锁死。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from mlkem_triton.kernels.ntt import byte_encode12, forward_ntt, pow_mac
from mlkem_triton.keygen import KeyGenerator
from mlkem_triton.ntt_ref import byte_encode, multiply_ntts, ntt
from mlkem_triton.params import Q

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="需要 GPU")

PS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


def test_forward_ntt_matches_reference():
    rng = np.random.default_rng(3)
    x = rng.integers(0, Q, (7, 256))
    got = forward_ntt(torch.from_numpy(x.astype(np.int32)).to(DEV)).cpu().numpy()
    assert np.array_equal(got, np.stack([ntt(x[i]) for i in range(7)]))


def test_pow_mac_matches_reference():
    k, B = 3, 3
    rng = np.random.default_rng(5)
    A = rng.integers(0, Q, (B * k * k, 256))
    S = rng.integers(0, Q, (B * k, 256))
    E = rng.integers(0, Q, (B * k, 256))
    f = lambda a: torch.from_numpy(a.astype(np.int32)).to(DEV)  # noqa: E731
    got = pow_mac(f(A), f(S), f(E), k).cpu().numpy()
    for b in range(B):
        for i in range(k):
            acc = E[b * k + i].astype(np.int64).copy()
            for j in range(k):
                acc = (acc + multiply_ntts(A[b * k * k + i * k + j], S[b * k + j])) % Q
            assert np.array_equal(got[b * k + i], acc), (b, i)


def test_byte_encode12_matches_reference():
    rng = np.random.default_rng(7)
    v = rng.integers(0, 4096, (4, 256))
    got = byte_encode12(torch.from_numpy(v.astype(np.int32)).to(DEV)).cpu().numpy()
    assert np.array_equal(
        got, np.stack([np.frombuffer(byte_encode(v[i], 12), np.uint8) for i in range(4)])
    )


@pytest.mark.parametrize("ps", PS)
def test_keygen_bit_exact_vs_kyberpy(ps):
    """全长 2400 字节的 dk 必须与 kyber-py 逐字节相同（含那个 `‖ k` 域分隔）。"""
    pytest.importorskip("kyber_py", reason="需要 kyber-py 做逐字节对拍")
    from mlkem_triton import refimpl

    d = bytes(range(32))
    z = bytes(range(32, 64))
    ek_ref, dk_ref = refimpl.ref_keygen(ps, d, z)
    kg = KeyGenerator(ps, device=DEV)
    ek, dk = kg.generate_torch(
        torch.from_numpy(np.frombuffer(d, np.uint8).copy()).to(DEV).reshape(1, 32),
        torch.from_numpy(np.frombuffer(z, np.uint8).copy()).to(DEV).reshape(1, 32),
    )
    assert ek.cpu().numpy()[0].tobytes() == ek_ref, "ek 不一致"
    assert dk.cpu().numpy()[0].tobytes() == dk_ref, "dk 不一致"


@pytest.mark.parametrize("ps", PS)
def test_keygen_batch_then_encaps_decaps(ps):
    """批量生成 → 用第一把钥匙 encaps → 用自己的 decaps 拿回同一个 K。"""
    pytest.importorskip("kyber_py", reason="需要 kyber-py 交叉验证")
    from mlkem_triton import refimpl
    from mlkem_triton.encaps import Encapsulator
    from mlkem_triton.pipeline import Decapsulator

    eks, dks = KeyGenerator(ps, device=DEV).generate(5)
    ek, dk = eks[0].tobytes(), dks[0].tobytes()
    # kyber-py 能用我们生成的 ek 封装 → 说明 ek 是合法的 FIPS 203 公钥
    K, c = refimpl.ref_encaps(ps, ek)
    assert Decapsulator(ps, dk, device=DEV).decaps([c])[0].tobytes() == K
    # 我们的 encaps 也能用我们生成的 ek
    c2, K2 = Encapsulator(ps, ek, device=DEV).encapsulate([bytes(32)])
    assert refimpl.ref_decaps(ps, dk, c2[0].tobytes()) == K2[0].tobytes()
