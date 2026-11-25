"""后端探测 + 每后端调优参数表。

这个包的内核全部是标准 Triton（``tl.dot`` / ``tl.trans`` / ``tl.interleave`` /
``tl.atomic_add``），没有 CUDA / XPU 专属指令。需要按后端区分的只有 **tile 形状**
和 **num_stages**，因为不同硬件的寄存器文件和流水深度不一样。

已实测（不是猜）：
- ``xpu``（Intel Arc B390 / Xe3-LPG，256 GRF/线程）：
  ``BM*BN/(16*num_warps) <= 128`` —— 每线程 fp32 累加器超过 128 个就大量 spill。
  ``num_stages=2`` 最优（片上内存小）。
- ``cuda`` / ``hip``：没有 256 GRF 这个约束，``BK=64``、``num_stages=3`` 通常更好。
  **这两个配置没有在真机上验证过** —— 见 README「支持矩阵」。
"""

from __future__ import annotations

import contextlib
import os

import torch

__all__ = [
    "available_devices",
    "backend_of",
    "describe",
    "gemm_cfg",
    "resolve_device",
    "sync",
]


def _has(module: str, attr: str = "is_available") -> bool:
    mod = getattr(torch, module, None)
    if mod is None:
        return False
    try:
        return bool(getattr(mod, attr)())
    except Exception:
        return False


def available_devices() -> list[torch.device]:
    """按优先级返回可用的加速设备（cuda / xpu）。空列表表示只有 CPU。"""
    out: list[torch.device] = []
    if _has("cuda"):
        out.append(torch.device("cuda"))
    if _has("xpu"):
        out.append(torch.device("xpu"))
    return out


def resolve_device(name: str | torch.device | None = None) -> torch.device:
    """把 ``None`` 解析成第一个可用加速设备。

    优先级：显式传入 > 环境变量 ``MLKEM_DEVICE`` > cuda > xpu。
    都没有就抛 RuntimeError（本包没有 CPU 后端：Triton 没有可用的 CPU target）。
    """
    if name is not None:
        dev = torch.device(name)
        _check_available(dev)
        return dev
    env = os.environ.get("MLKEM_DEVICE")
    if env:
        dev = torch.device(env)
        _check_available(dev)
        return dev
    devs = available_devices()
    if not devs:
        raise RuntimeError(
            "找不到可用的 GPU。本包只支持 Triton 有后端的设备"
            "（NVIDIA CUDA / AMD ROCm / Intel XPU），不支持纯 CPU 运行。"
            "若要用 CPU 只做算法验证，请用 mlkem_triton.ntt_ref / keyprep 的 numpy 实现。"
        )
    return devs[0]


def _check_available(dev: torch.device) -> None:
    if dev.type == "cuda" and not _has("cuda"):
        raise RuntimeError("指定了 cuda 设备，但 torch.cuda.is_available() 为 False")
    if dev.type == "xpu" and not _has("xpu"):
        raise RuntimeError("指定了 xpu 设备，但 torch.xpu.is_available() 为 False")
    if dev.type == "cpu":
        raise RuntimeError("本包没有 CPU 后端；Triton 没有可用的 CPU target")


def sync(device: torch.device | None = None) -> None:
    """同步设备（``None`` = 当前解析出来的那块）。

    存在的意义：调用方不该写死 `torch.cuda.synchronize()` /
    `torch.xpu.synchronize()`，否则换后端就炸。
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        torch.xpu.synchronize(dev)


def backend_of(dev: torch.device) -> str:
    """归一化成 ``xpu`` / ``cuda`` / ``hip`` 三个键之一。"""
    if dev.type == "cuda":
        return "hip" if getattr(torch.version, "hip", None) else "cuda"
    return dev.type


# 未在表里的后端用这个保守配置（能跑，不是最优）
_FALLBACK_CFG = {"BM": 64, "BN": 64, "BK": 32, "GM": 8, "num_warps": 4, "num_stages": 2}

_GEMM_CFG = {
    # Intel Xe3-LPG 实测最优（B=4096，见 results/OPTIMIZATION_LOG.md）
    "xpu": {"BM": 128, "BN": 128, "BK": 32, "GM": 4, "num_warps": 8, "num_stages": 2},
    # 以下两个**未实测**，按公开经验给的稳妥值
    "cuda": {"BM": 128, "BN": 128, "BK": 64, "GM": 8, "num_warps": 8, "num_stages": 3},
    "hip": {"BM": 128, "BN": 128, "BK": 64, "GM": 8, "num_warps": 8, "num_stages": 2},
}


def gemm_cfg(dev: torch.device) -> dict:
    """返回该后端适用的 tile 配置（已拷贝，调用方可随意改）。"""
    return dict(_GEMM_CFG.get(backend_of(dev), _FALLBACK_CFG))


_WARPS_OVERRIDE: dict[str, int] | None = None


def warps(name: str, default: int) -> int:
    """某个小内核的 `num_warps`，支持环境变量覆盖。

    这些值**只在 Intel Xe3 上扫过**（见 `results/comparison_prior_work.md` 的内核
    占比表）。小内核（sponge / padcat / CBD / 打包）对 `num_warps` 不敏感，所以其他
    后端沿用同一组值不会有量级差异；但**没有任何 CUDA/ROCm 上的调优数据**，所以这里
    只提供一套默认值 + 一个覆盖入口，不编造「每后端最优」。

        MLKEM_WARPS="shake=8,ntt=4"  python benchmarks/bench_e2e.py

    注意 GEMM 的 tile 配置不在这里 —— 那是 `gemm_cfg()`，因为 tile 形状还必须满足
    `M % BM == 0` 之类的整除约束，不能随便改。
    """
    global _WARPS_OVERRIDE
    if _WARPS_OVERRIDE is None:
        _WARPS_OVERRIDE = {}
        raw = os.environ.get("MLKEM_WARPS", "")
        for item in raw.split(","):
            if "=" not in item:
                continue
            k, _, v = item.partition("=")
            with contextlib.suppress(ValueError):
                _WARPS_OVERRIDE[k.strip()] = int(v)
    return _WARPS_OVERRIDE.get(name, default)


def describe(dev: torch.device) -> str:
    """一行人类可读的设备描述，给 README / 日志用。"""
    be = backend_of(dev)
    try:
        if dev.type == "cuda":
            name = torch.cuda.get_device_name(dev)
            cap = torch.cuda.get_device_capability(dev)
            return f"{name} ({be}, capability {cap[0]}.{cap[1]})"
        if dev.type == "xpu":
            return f"{torch.xpu.get_device_name(dev)} ({be})"
    except Exception:
        pass
    return str(dev)
