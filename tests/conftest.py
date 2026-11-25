"""pytest configuration: `--device cuda` / `--device xpu` selects the device.

The device is written into `MLKEM_DEVICE` at **conftest import time**, because
test modules import `_common` during collection (where
`DEV = resolve_device()` is already fixed by then).
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
        help="cuda / xpu / cuda:1 ... (default: auto-select the first available device)",
    )


def pytest_configure(config):
    dev = config.getoption("--device")
    if dev:
        os.environ["MLKEM_DEVICE"] = dev


# Without a GPU: **centrally** skip the modules that need a graphics card.
# Writing skipif in each test file is easy to get wrong (a newly added file
# would be missed), so the fallback lives here. The whitelisted modules are
# pure numpy / pure math and do not need a GPU.
_NO_GPU_OK = {
    "test_ntt.py",
    "test_keyprep.py",
    "test_ref.py",
    "test_primitives.py",
    "test_mldsa_ref.py",
    "test_compat.py",
}


def pytest_collection_modifyitems(config, items):
    import _common

    if _common.HAS_GPU:
        return
    import pytest

    skip = pytest.mark.skip(reason="no usable GPU (only the math/reference cases can run on CPU)")
    for item in items:
        if item.path.name not in _NO_GPU_OK:
            item.add_marker(skip)
