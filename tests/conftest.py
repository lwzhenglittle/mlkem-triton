"""pytest 配置：`--device cuda` / `--device xpu` 选择设备。

设备在 **conftest 导入时**就写进 `MLKEM_DEVICE`，因为测试模块在收集阶段
就会 `import _common`（那里 `DEV = resolve_device()` 已经定了）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
for p in (str(_ROOT), str(_ROOT / "tests")):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption(
        "--device",
        action="store",
        default=None,
        help="cuda / xpu / cuda:1 ...（默认自动选第一个可用设备）",
    )


def pytest_configure(config):
    dev = config.getoption("--device")
    if dev:
        os.environ["MLKEM_DEVICE"] = dev


# 无 GPU 时：**集中**把需要显卡的模块整体 skip 掉。
# 在各测试文件里逐个写 skipif 容易漏（新增文件就会漏），所以放在这里兜底。
# 白名单里的几个模块是纯 numpy / 纯数学，不需要 GPU。
_NO_GPU_OK = {"test_ntt.py", "test_keyprep.py", "test_ref.py"}


def pytest_collection_modifyitems(config, items):
    import _common

    if _common.HAS_GPU:
        return
    import pytest

    skip = pytest.mark.skip(reason="无可用 GPU（只有数学/参考实现用例能在 CPU 上跑）")
    for item in items:
        if item.path.name not in _NO_GPU_OK:
            item.add_marker(skip)
