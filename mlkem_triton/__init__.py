"""mlkem-triton —— 纯 Triton 的 ML-KEM 批量解封装。

    from mlkem_triton import Decapsulator, Encapsulator
    dec = Decapsulator("ML-KEM-768", dk)
    keys = dec.decaps(list_of_ciphertexts)      # -> np.ndarray [B, 32] uint8
    enc = Encapsulator("ML-KEM-768", ek)
    ct, keys = enc.encapsulate(list_of_32B_msgs)

    eks, dks = generate_keypairs("ML-KEM-768", 4096)   # 三个操作齐了

设计要点见 README；性能数据见 results/OPTIMIZATION_LOG.md（原项目仓库）。
"""

from __future__ import annotations

from .devices import available_devices, backend_of, describe, gemm_cfg, resolve_device
from .params import PARAMS, N, Params, Q

__version__ = "0.1.0"

__all__ = [
    "PARAMS",
    "Decapsulator",
    "Encapsulator",
    "KeyGenerator",
    "N",
    "Params",
    "Q",
    "__version__",
    "available_devices",
    "backend_of",
    "decaps_batch",
    "describe",
    "encapsulate",
    "gemm_cfg",
    "generate_keypairs",
    "resolve_device",
]


_LAZY_MODULE = {
    "Decapsulator": "pipeline",
    "decaps_batch": "pipeline",
    "Encapsulator": "encaps",
    "encapsulate": "encaps",
    "KeyGenerator": "keygen",
    "generate_keypairs": "keygen",
}


def __getattr__(name):
    # 惰性导入：`import mlkem_triton` 在纯 CPU 机器上也应该能用（只是想看
    # 参数集 / 用 numpy 参考实现的话），不要因为 torch 缺 CUDA 就炸。
    import importlib

    # 只认这三个入口 —— **不要**写成「遍历所有可能模块」，那样会在
    # `pipeline.py` 自己的 `from . import keyprep, ops` 上触发递归导入。
    mod = _LAZY_MODULE.get(name)
    if mod is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f".{mod}", __name__), name)
