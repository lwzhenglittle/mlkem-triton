"""GPU 上的密钥重建（`kernels/rebuild.py`）验收。

这个模块只为测量而存在（对照 cuPQC 无状态 API 每 op 重算 Â），所以要求是
**逐元素等于 CPU 的 `keyprep.parse_key`** —— 一旦不等，所有"同口径"结论都不成立。
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from mlkem_triton import acvp, keyprep
from mlkem_triton.kernels.rebuild import GpuKeyRebuilder, intt_matrix
from mlkem_triton.ntt_ref import intt
from mlkem_triton.params import N, Q
from mlkem_triton.pipeline import Decapsulator

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="需要 GPU")


def _dk(ps: str) -> bytes:
    c = next(x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == ps)
    return bytes.fromhex(c["inputs"]["dk"])


def test_intt_matrix_is_exact():
    """INTT 用 256×256 矩阵乘代替蝴蝶，必须逐位精确（fp64 下 2^53 够）。"""
    mt = intt_matrix(DEV)
    for seed in range(4):
        np.random.seed(seed)
        x = np.random.randint(0, Q, N)
        got = (
            (torch.matmul(torch.from_numpy(x.astype(np.float64)).to(DEV), mt).to(torch.int64) % Q)
            .cpu()
            .numpy()
        )
        assert np.array_equal(got, intt(x)), seed


@pytest.mark.parametrize("ps", ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"])
def test_rebuild_matches_keyprep(ps):
    """重建出的 W_dec / W_enc 必须与 CPU 路径逐元素相等（fp16 下）。"""
    dk = _dk(ps)
    kp = keyprep.parse_key(ps, dk)
    Wd, We = GpuKeyRebuilder(ps, device=DEV).rebuild(dk)
    torch_sync = getattr(torch, DEV.type).synchronize
    torch_sync()

    Wd_ref = torch.from_numpy(kp.W_dec).to(torch.float16).cpu()
    We_ref = torch.from_numpy(kp.W_enc).to(torch.float16).cpu()
    assert tuple(Wd.shape) == tuple(Wd_ref.shape)
    assert tuple(We.shape) == tuple(We_ref.shape)
    assert torch.equal(Wd.cpu(), Wd_ref), "W_dec 不一致"
    assert torch.equal(We.cpu(), We_ref), "W_enc 不一致"


@pytest.mark.parametrize("ps", ["ML-KEM-768"])
def test_rebuilt_weights_give_correct_decaps(ps):
    """用重建出来的权重跑完整流水线，ACVP 向量必须过 —— 这才证明权重语义对。"""

    c = next(x for x in acvp.load() if x["function"] == "decapsulation" and x["parameterSet"] == ps)
    dk = bytes.fromhex(c["inputs"]["dk"])
    ct = bytes.fromhex(c["inputs"]["c"])
    want = bytes.fromhex(c["expected"]["k"])

    dec = Decapsulator(ps, dk, device=DEV)
    assert dec.decaps([ct] * 4)[0].tobytes() == want  # 预展开路径

    dec.W_dec, dec.W_enc = GpuKeyRebuilder(ps, device=DEV).rebuild(dk)
    got = dec.decaps([ct] * 4)
    assert all(x.tobytes() == want for x in got), "重建权重下解封装不正确"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
