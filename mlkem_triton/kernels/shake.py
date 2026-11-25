"""批量 SHAKE256 / SHA3-512 的 Triton sponge 实现（代码生成）。

映射：一个线程 = 一个 sponge 实例，25 个 lane 各一个 `[BLOCK]` uint64 张量。
输入在 host 侧预填充（pad10*1 + domain separation），GPU 侧只做整块的
absorb XOR + 24 轮 permute + 单块 squeeze（我们所有用途的输出都 <= rate）。

**24 轮走运行时 for 循环，轮常数从一张 24 元素的 uint64 表里读**（见下方
「为什么滚动」）。滚动后 kernel 的静态指令数从 15254 降到 790，
每线程 scratch 溢出（spill）从 3584 B 降到 0，吞吐在 rate=136/out=128 上
提升约 1.3x，编译时间从 ~25 s 降到 ~3 s。

为什么滚动（实测证据，见 results/keccak_optimization.md）：
- Xe 上一个 lane 的 uint64 占 8 GRF（SIMD32 下每 lane 8 字节），
  25 个 lane 的状态 = 200 GRF，而硬件只有 256 GRF/线程
  （`-cl-intel-256-GRF-per-thread`，512-GRF 在该平台不支持）。
- 全展开时 ρπ 的 25 个 b 还要再 200 GRF，峰值 ~400 GRF，
  IGC 只能把状态溢出到 scratch：zeinfo 里 `spill_size: 3584`，
  ISA 里 1202 条 `// spill to offset[..]` / `fill from ..`。
- 滚动后同一段代码的活跃区间缩短，spill_size = 0。

用法:
    K = get_sponge(rate=136, outlen=32)
    K(X, OUT, n, nblocks, nblocks*rate/8, OUTW=32//8, BLOCK=128, num_warps=4)
    X: [n, nblocks*rate/8] int64（已填充）
    OUT: [n, outlen/8] int64
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import torch

from .keccak import RC, ROT, _rotl_expr

# 轮常数表：作为 kernel 的运行时参数（uint64 视图用 int64 承载，逐位一致）。
#
# 按 device 缓存 —— 早先是模块级单例硬编码在 "xpu" 上，导致非 Intel 后端
# 第一次调用就炸。表只有 24 个 int64 = 192 B，按设备各存一份没有成本。
_RC_SIGNED = [c - (1 << 64) if c >= (1 << 63) else c for c in RC]
_RC_TENSORS: dict[torch.device, torch.Tensor] = {}


def _rc_tensor(device: torch.device) -> torch.Tensor:
    t = _RC_TENSORS.get(device)
    if t is None:
        t = torch.tensor(_RC_SIGNED, dtype=torch.int64, device=device)
        _RC_TENSORS[device] = t
    return t


def _emit_rounds(L: list[str], rnd: int, ind: str = "        ", rc_expr: str | None = None) -> None:
    rc = rc_expr if rc_expr is not None else str(RC[rnd])
    for x in range(5):
        L.append(f"{ind}c{x} = a{x} ^ a{x + 5} ^ a{x + 10} ^ a{x + 15} ^ a{x + 20}")
    for x in range(5):
        cp, cn = (x + 1) % 5, (x + 4) % 5
        L.append(f"{ind}d{x} = c{cn} ^ {_rotl_expr(f'c{cp}', 1)}")
    for y in range(5):
        for x in range(5):
            i = x + 5 * y
            L.append(f"{ind}a{i} = a{i} ^ d{x}")
    for y in range(5):
        for x in range(5):
            src = x + 5 * y
            dst = y + 5 * ((2 * x + 3 * y) % 5)
            L.append(f"{ind}b{dst} = {_rotl_expr(f'a{src}', ROT[x][y])}")
    for y in range(5):
        for x in range(5):
            L.append(
                f"{ind}a{x + 5 * y} = b{x + 5 * y} ^ "
                f"((~b{(x + 1) % 5 + 5 * y}) & b{(x + 2) % 5 + 5 * y})"
            )
    L.append(f"{ind}a0 = a0 ^ {rc}")


def gen_sponge_source(rate: int, outlen: int) -> str:
    nw = rate // 8
    assert rate % 8 == 0 and outlen % 8 == 0
    assert outlen <= 16 * rate, "squeeze 块数太多"

    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    # NB / ROW 用运行时参数：块数不再需要展开（24 轮已经滚成循环，
    # 静态指令数与 NB 无关）。RCP = 轮常数表（uint64 视图）。
    L.append("def sponge_kernel(X, OUT, RCP, n, NB, ROW, OUTW: tl.constexpr, BLOCK: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    offs = pid * BLOCK + tl.arange(0, BLOCK)")
    L.append("    m = offs < n")
    L.append("    base = offs * ROW")
    for i in range(25):
        L.append(f"    a{i} = tl.zeros((BLOCK,), dtype=tl.uint64)")
    L.append("    for _b in range(0, NB, 1):")
    for lane in range(nw):
        L.append(
            f"        a{lane} = a{lane} ^ tl.load(X + base + _b * {nw} + {lane}, "
            f"mask=m, other=0).to(tl.uint64, bitcast=True)"
        )
    # 24 轮 permute（滚动；必须在 `for _b` 内部，多块 absorb 才是正确 sponge）
    L.append("        for _r in range(0, 24, 1):")
    L.append("            rc = tl.load(RCP + _r)")
    _emit_rounds(L, 0, ind="            ", rc_expr="rc")

    # squeeze：可能是多块（例如 eta=3 时 PRF 要 192B > rate 136B）
    nfull = outlen // rate
    rem = outlen - nfull * rate
    blocks = [rate] * nfull + ([rem] if rem else [])
    for bi, blen in enumerate(blocks):
        for lane in range(blen // 8):
            L.append(
                f"    tl.store(OUT + offs * OUTW + {bi * nw + lane}, "
                f"a{lane}.to(tl.int64, bitcast=True), mask=m)"
            )
        if bi < len(blocks) - 1:
            L.append("    for _r in range(0, 24, 1):")
            L.append("        rc = tl.load(RCP + _r)")
            _emit_rounds(L, 0, ind="        ", rc_expr="rc")
    return "\n".join(L) + "\n"


class _SpongeLauncher:
    """`sponge_kernel[grid](X, OUT, ...)` 的薄包装：自动注入轮常数表。

    这样 `ops.py` / `gpu_hash.py` 里的调用点不需要改（接口保持原样），
    kernel 本身又能用一张运行时表驱动 24 轮循环。

    轮常数表在**启动时**按 `X.device` 取，不在 `get_sponge()` 时绑定 ——
    否则一个进程里换设备就错了。
    """

    __slots__ = ("_kern",)

    def __init__(self, kern):
        self._kern = kern

    def __getitem__(self, grid):
        kern = self._kern

        def launch(X, OUT, *args, **kwargs):
            return kern[grid](X, OUT, _rc_tensor(X.device), *args, **kwargs)

        return launch

    def __getattr__(self, name):
        return getattr(self._kern, name)


_CACHE: dict[tuple[int, int], object] = {}


def get_sponge(rate: int, outlen: int):
    key = (rate, outlen)
    if key in _CACHE:
        return _CACHE[key]
    src = gen_sponge_source(rate, outlen)
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_mlkem_sponge_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_mlkem_sponge_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[key] = _SpongeLauncher(mod.sponge_kernel)
    return _CACHE[key]


def pad10star1(msg: bytes, rate: int, domain: int) -> bytes:
    """pad10*1 + 域分隔，返回 len 为 rate 整数倍的字节串。"""
    p = bytearray(msg)
    p.append(domain)
    p.extend(b"\x00" * ((-len(p)) % rate))
    p[-1] |= 0x80
    return bytes(p)


SHAKE_DOMAIN = 0x1F
SHA3_DOMAIN = 0x06


__all__ = [
    "SHA3_DOMAIN",
    "SHAKE_DOMAIN",
    "gen_sponge_source",
    "get_sponge",
    "pad10star1",
]
