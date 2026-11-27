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
# Maintaining a positive whitelist of CPU-only modules drifted silently whenever a
# new pure-math, reference, or contract test was added (e.g. `test_primitives_contract.py`,
# `test_bench_harness.py`, `test_cpu_only.py`). Listing GPU modules instead -- plus
# auto-skipping `*_gpu.py` and anything marked `@pytest.mark.gpu` -- ensures every
# device-free test runs on CPU-only machines and hosted CI by default.
_NEEDS_GPU = {
    "test_dot_exact.py",
    "test_encaps.py",
    "test_falcon_liboqs.py",
    "test_fused_gemm.py",
    "test_gemm.py",
    "test_gemm_int8.py",
    "test_glue.py",
    "test_input_validation.py",
    "test_keygen.py",
    "test_lms.py",
    "test_mldsa.py",
    "test_mldsa_sign.py",
    "test_ntt_large.py",
    "test_pipeline.py",
    "test_rebuild.py",
    "test_slhdsa.py",
    "test_triton_behaviour.py",
}


def pytest_collection_modifyitems(config, items):
    import _common

    if _common.HAS_GPU:
        return
    import pytest

    skip = pytest.mark.skip(
        reason="no usable GPU (only the math/reference/contract cases run on CPU)"
    )
    for item in items:
        name = item.path.name
        if (
            name in _NEEDS_GPU
            or name.endswith("_gpu.py")
            or item.get_closest_marker("gpu") is not None
        ):
            item.add_marker(skip)
