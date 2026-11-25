"""测试公用：设备解析。

`MLKEM_DEVICE` 环境变量（或 `pytest --device`）决定跑在哪块卡上；
都不给就自动选第一个可用加速设备。
"""

from __future__ import annotations

from mlkem_triton.devices import describe, resolve_device

try:
    DEV = resolve_device()
    HAS_GPU = True
except RuntimeError:  # 纯 CPU 机器：只跑不需要 GPU 的那些用例
    DEV = None
    HAS_GPU = False


def dev_name() -> str:
    return describe(DEV) if HAS_GPU else "(无 GPU)"
