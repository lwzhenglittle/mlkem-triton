"""Backend detection + per-backend tuning parameter tables.

Every kernel in this package is plain Triton (``tl.dot`` / ``tl.trans`` /
``tl.interleave`` / ``tl.atomic_add``); there are no CUDA / XPU specific
instructions. Only **tile shapes** and **num_stages** need to differ per backend,
because different hardware has different register files and pipeline depths.

Measured (not guessed):
- ``xpu`` (Intel Arc B390 / Xe3-LPG, 256 GRF/thread):
  ``BM*BN/(16*num_warps) <= 128`` — more than 128 fp32 accumulators per thread
  spills heavily. ``num_stages=2`` is best (small on-chip memory).
- ``cuda``: no 256 GRF constraint, so ``BK=64`` with ``num_stages=3`` is fine. The whole
  suite does pass on a GB10 (sm_121), but **these tile values were never swept there** — they
  are conservative values from public experience, not a measured optimum.
- ``hip``: same as ``cuda`` except ``num_stages=2``. **Not run on real AMD hardware at all.**
  See README.md section `Support matrix` and docs/support.md.
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
    """Return available accelerator devices in priority order (cuda / xpu).

    An empty list means CPU only.
    """
    out: list[torch.device] = []
    if _has("cuda"):
        out.append(torch.device("cuda"))
    if _has("xpu"):
        out.append(torch.device("xpu"))
    return out


def resolve_device(name: str | torch.device | None = None) -> torch.device:
    """Resolve ``None`` to the first available accelerator device.

    Priority: explicit argument > ``MLKEM_DEVICE`` environment variable > cuda > xpu.
    If none is available, raise RuntimeError (this package has no CPU backend:
    there is no usable CPU target in Triton).
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
            "No usable GPU found. This package only supports devices that Triton has "
            "a backend for (NVIDIA CUDA / AMD ROCm / Intel XPU) and does not support "
            "pure CPU execution. To use the CPU only for algorithm validation, use the "
            "numpy implementations in triton_crypto.mlkem.ntt_ref / keyprep."
        )
    return devs[0]


def _check_available(dev: torch.device) -> None:
    if dev.type == "cuda" and not _has("cuda"):
        raise RuntimeError("cuda device specified, but torch.cuda.is_available() is False")
    if dev.type == "xpu" and not _has("xpu"):
        raise RuntimeError("xpu device specified, but torch.xpu.is_available() is False")
    if dev.type == "cpu":
        raise RuntimeError("this package has no CPU backend; Triton has no usable CPU target")


def sync(device: torch.device | None = None) -> None:
    """Synchronize the device (``None`` = whichever one is currently resolved).

    Why this exists: callers should not hard-code `torch.cuda.synchronize()` /
    `torch.xpu.synchronize()`, or switching backends breaks.
    """
    dev = resolve_device(device)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    elif dev.type == "xpu":
        torch.xpu.synchronize(dev)


def backend_of(dev: torch.device) -> str:
    """Normalize to one of the three keys ``xpu`` / ``cuda`` / ``hip``."""
    if dev.type == "cuda":
        return "hip" if getattr(torch.version, "hip", None) else "cuda"
    return dev.type


# Backends not in the table use this conservative config (it runs, it is not optimal)
_FALLBACK_CFG = {"BM": 64, "BN": 64, "BK": 32, "GM": 8, "num_warps": 4, "num_stages": 2}

_GEMM_CFG = {
    # Measured optimum on Intel Xe3-LPG (B=4096; see docs/support.md)
    "xpu": {"BM": 128, "BN": 128, "BK": 32, "GM": 4, "num_warps": 8, "num_stages": 2},
    # The following two were never swept on their hardware; safe values from public
    # experience. `cuda` runs correctly on GB10, `hip` has not been run at all.
    "cuda": {"BM": 128, "BN": 128, "BK": 64, "GM": 8, "num_warps": 8, "num_stages": 3},
    "hip": {"BM": 128, "BN": 128, "BK": 64, "GM": 8, "num_warps": 8, "num_stages": 2},
}


def gemm_cfg(dev: torch.device) -> dict:
    """Return the tile config for this backend (already copied; callers may modify it)."""
    return dict(_GEMM_CFG.get(backend_of(dev), _FALLBACK_CFG))


_WARPS_OVERRIDE: dict[str, int] | None = None


def warps(name: str, default: int) -> int:
    """`num_warps` for a small kernel, with environment-variable override.

    These values were **only swept on Intel Xe3** (see the kernel share table in
    `docs/compare.md`). Small kernels (sponge / padcat / CBD / packing) are not
    sensitive to `num_warps`, so other backends reusing the same set of values
    will not differ by an order of magnitude; but there is **no tuning data for
    CUDA/ROCm at all**, so this provides only one default set plus an override
    hook, and does not invent a "per-backend optimum".

        MLKEM_WARPS="shake=8,ntt=4"  python benchmarks/bench_e2e.py

    Note that the GEMM tile config is not here — that is `gemm_cfg()`, because tile
    shapes must also satisfy divisibility constraints such as `M % BM == 0` and
    cannot be changed at will.
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
    """One-line human-readable device description, for README / logs."""
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
