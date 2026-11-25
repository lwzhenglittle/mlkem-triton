"""把 pad10*1 + 域分隔（含段拼接/广播/单字节后缀）压成**一次 launch**。

为什么需要它
------------
`pad_gpu(cat([a, b]))` 在 XPU 上是 3 次 launch（`cat`、`torch.zeros`、`copy_`），
而且 `cat` 还要处理 `expand` 出来的 stride-0 源。B=4096 的 J 档
（`z(32B) || ct(1088B)`，pad 到 9 块 × 136 B）实测 **192 µs**，
而它服务的 sponge kernel 本身才 256 µs —— 也就是 4.5 MB 的输入准备
吃掉了 J 阶段的 43 %。融合成一次 launch、整行写完（含零填充）后 **28 µs**。

语义严格等价于（逐字节）：

    out = torch.zeros((n, total), uint8)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, total - 1] |= 0x80

三个变体
--------
- `padcat2`：消息 = `a || b`，`a`/`b` 各自可广播（J：z 广播 + ct 逐行；
  G：m' 逐行 + h_ek 广播）。
- `padcat_sfx`：消息 = `a[row//mod] || [row % mod + off]`（PRF：r' || [j]）。
- 没有 `padcat` 的普通单段情形继续用 `pad_gpu`（标量/小批量更划算）。

实现要点
--------
- **word（int64）粒度**：要求段长、总长都是 8 的倍数（J/G/PRF 都满足），
  指令数是字节粒度的 1/8。调用方不满足时自己回退。
- 零填充区显式**排除** domain word 和最后一个 word，这两个用定点标量写，
  避免同一次 kernel 里「先写 0 再写 domain」的跨 warp 写-写竞争。
- tile 取 `[BR=32, <=128]`：BR=128 时会因为每线程 128 个 int64（=128 GRF）
  而掉速 2.6×（实测 25 µs → 65 µs），所以 BR 不能大。
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import torch
import triton

from ..devices import warps

BR_DEFAULT = 32
WARPS_DEFAULT = 4
_MAX_CHUNK = 128  # 每个 tile 最多 128 个 word（8 的倍数、2 的幂）


def _sgn64(v: int) -> int:
    """把无符号 64 位常量写进 Triton 的 int64（避免 >= 2^63 报错）。"""
    return v - (1 << 64) if v >= (1 << 63) else v


def _chunks(n: int) -> list[tuple[int, int]]:
    """把 n 拆成递减的 2 的幂之和（136 -> [(0,128),(128,8)]）。"""
    out, off = [], 0
    while n > 0:
        p = min(1 << (n.bit_length() - 1), _MAX_CHUNK)
        out.append((off, p))
        off += p
        n -= p
    return out


def gen_padcat2_source(
    la_w: int, lb_w: int, total_w: int, domain: int, a_bcast: bool, b_bcast: bool
) -> str:
    assert la_w > 0 and lb_w >= 0 and total_w > la_w + lb_w
    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    L.append("def padcat2_kernel(A, B, OUT, n, BR: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    rows = pid * BR + tl.arange(0, BR)")
    L.append("    m = rows < n")
    L.append(f"    prow = OUT + rows[:, None] * {total_w}")
    # ---- A 段 ----
    for off, sz in _chunks(la_w):
        tail = off + sz > la_w
        L.append(f"    c = tl.arange(0, {sz})")
        if a_bcast:
            L.append(
                f"    v = tl.load(A + {off} + c"
                + (f", mask=c < {la_w - off}, other=0)" if tail else ")")
            )
            L.append(
                f"    tl.store(prow + {off} + c[None, :], "
                f"tl.broadcast_to(v[None, :], (BR, {sz})), mask=m[:, None])"
            )
        else:
            L.append(
                f"    v = tl.load(A + rows[:, None] * {la_w} + {off} + c[None, :], "
                + (
                    f"mask=m[:, None] & (c[None, :] < {la_w - off}), other=0)"
                    if tail
                    else "mask=m[:, None], other=0)"
                )
            )
            L.append(f"    tl.store(prow + {off} + c[None, :], v, mask=m[:, None])")
    # ---- B 段 ----
    for off, sz in _chunks(lb_w):
        tail = off + sz > lb_w
        o = la_w + off
        L.append(f"    c = tl.arange(0, {sz})")
        if b_bcast:
            L.append(
                f"    v = tl.load(B + {off} + c"
                + (f", mask=c < {lb_w - off}, other=0)" if tail else ")")
            )
            L.append(
                f"    tl.store(prow + {o} + c[None, :], "
                f"tl.broadcast_to(v[None, :], (BR, {sz})), mask=m[:, None])"
            )
        else:
            L.append(
                f"    v = tl.load(B + rows[:, None] * {lb_w} + {off} + c[None, :], "
                + (
                    f"mask=m[:, None] & (c[None, :] < {lb_w - off}), other=0)"
                    if tail
                    else "mask=m[:, None], other=0)"
                )
            )
            L.append(f"    tl.store(prow + {o} + c[None, :], v, mask=m[:, None])")
    # ---- 零区（排除 domain word 与最后一个 word）----
    z0, z1 = la_w + lb_w + 1, total_w - 1
    for off, sz in _chunks(max(0, z1 - z0)):
        o = z0 + off
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(f"    tl.store(prow + {o} + c[None, :], tl.zeros((BR, {sz}), dtype=tl.int64),")
        L.append(f"             mask=m[:, None] & (c[None, :] < {z1 - o}))")
    # ---- domain word + 最后一个 word ----
    dom_w, last_w = la_w + lb_w, total_w - 1
    dom_val = _sgn64(domain | (0x80 << 56)) if dom_w == last_w else _sgn64(domain)
    L.append(
        f"    tl.store(OUT + rows * {total_w} + {dom_w}, "
        f"tl.full((BR,), {dom_val}, dtype=tl.int64), mask=m)"
    )
    if dom_w != last_w:
        L.append(
            f"    tl.store(OUT + rows * {total_w} + {last_w}, "
            f"tl.full((BR,), {_sgn64(0x80 << 56)}, dtype=tl.int64), mask=m)"
        )
    return "\n".join(L) + "\n"


def gen_padcat_sfx_source(
    prefix_w: int,
    total_w: int,
    domain: int,
    mod: int,
    off: int,
    stride_w: int | None = None,
    base_w: int = 0,
) -> str:
    """消息 = a[row // mod] || bytes[row % mod + off] || pad10*1。

    `prefix_w` 个 word 来自 a（每个消息只有 `mod` 个不同前缀，按 row//mod 取），
    紧跟一个由后缀字节 + domain 组成的 word，然后零填充，最后一个 word 是 0x80。
    `stride_w`/`base_w` 让前缀可以直接从一个大张量（如 G 的 64 B 输出）里原地取，
    省掉一次 `contiguous()` 拷贝。
    """
    assert prefix_w > 0 and total_w > prefix_w + 1 and mod >= 1
    sw = prefix_w if stride_w is None else stride_w
    assert sw >= prefix_w + base_w
    dom_w, last_w = prefix_w, total_w - 1
    assert dom_w < last_w, "后缀+domain 不能和 0x80 落在同一个 word（本实现用不到）"
    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    L.append("def padcat_sfx_kernel(A, OUT, n, MOD: tl.constexpr, OFF: tl.constexpr,")
    L.append("                      BR: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    rows = pid * BR + tl.arange(0, BR)")
    L.append("    m = rows < n")
    L.append(f"    prow = OUT + rows[:, None] * {total_w}")
    arow = f"(rows[:, None] // MOD) * {sw} + {base_w}"
    for o, sz in _chunks(prefix_w):
        tail = o + sz > prefix_w
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(
            f"    v = tl.load(A + {arow} + {o} + c[None, :], "
            + (
                f"mask=m[:, None] & (c[None, :] < {prefix_w - o}), other=0)"
                if tail
                else "mask=m[:, None], other=0)"
            )
        )
        L.append(f"    tl.store(prow + {o} + c[None, :], v, mask=m[:, None])")
    # 后缀字节 + domain 共处一个 word（小端：后缀在最低字节）
    L.append("    sfx = (rows % MOD) + OFF")
    L.append(f"    w = (sfx.to(tl.int64) | {domain << 8})")
    L.append(f"    tl.store(OUT + rows * {total_w} + {dom_w}, w, mask=m)")
    z0, z1 = dom_w + 1, total_w - 1
    for o, sz in _chunks(max(0, z1 - z0)):
        oo = z0 + o
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(f"    tl.store(prow + {oo} + c[None, :], tl.zeros((BR, {sz}), dtype=tl.int64),")
        L.append(f"             mask=m[:, None] & (c[None, :] < {z1 - oo}))")
    L.append(
        f"    tl.store(OUT + rows * {total_w} + {last_w}, "
        f"tl.full((BR,), {_sgn64(0x80 << 56)}, dtype=tl.int64), mask=m)"
    )
    return "\n".join(L) + "\n"


_CACHE: dict[tuple, object] = {}


def _compile(src: str, fname: str):
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_mlkem_{fname}_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_mlkem_{fname}_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, fname)


def _launch(kern, args, n: int, br: int = BR_DEFAULT, nwarps: int = WARPS_DEFAULT):
    grid = (triton.cdiv(n, br),)
    kern[grid](*args, n=n, BR=br, num_warps=warps("padcat", nwarps))


def padcat2(
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
    la_w: int,
    lb_w: int,
    total_w: int,
    domain: int,
    a_bcast: bool = False,
    b_bcast: bool = False,
) -> torch.Tensor:
    """`pad_gpu(cat([a, b]))` 的一次 launch 版本；返回 [n, total_w*8] uint8。

    参数里的 `*_w` 都是 int64 word 数。a/b 需已 `.view(torch.int64)`（不可广播的
    那一段必须是 [n, *_w] 连续张量；可广播的那段是 [*_w]）。
    """
    key = ("c2", la_w, lb_w, total_w, domain, a_bcast, b_bcast)
    kern = _CACHE.get(key)
    if kern is None:
        kern = _compile(
            gen_padcat2_source(la_w, lb_w, total_w, domain, a_bcast, b_bcast), "padcat2_kernel"
        )
        _CACHE[key] = kern
    out = torch.empty((n, total_w), dtype=torch.int64, device=a.device)
    _launch(kern, (a, b, out), n)
    return out.view(torch.uint8).view(n, total_w * 8)


def padcat_sfx(
    a: torch.Tensor,
    n: int,
    prefix_w: int,
    total_w: int,
    domain: int,
    mod: int,
    off: int,
    stride_w: int | None = None,
    base_w: int = 0,
) -> torch.Tensor:
    """`pad_gpu(cat([a[row//mod], [row%mod+off]]))` 的一次 launch 版本。"""
    key = ("sfx", prefix_w, total_w, domain, mod, off, stride_w, base_w)
    kern = _CACHE.get(key)
    if kern is None:
        kern = _compile(
            gen_padcat_sfx_source(prefix_w, total_w, domain, mod, off, stride_w, base_w),
            "padcat_sfx_kernel",
        )
        _CACHE[key] = kern
    out = torch.empty((n, total_w), dtype=torch.int64, device=a.device)
    grid = (triton.cdiv(n, BR_DEFAULT),)
    kern[grid](a, out, n, MOD=mod, OFF=off, BR=BR_DEFAULT, num_warps=warps("padcat", WARPS_DEFAULT))
    return out.view(torch.uint8).view(n, total_w * 8)


__all__ = ["gen_padcat2_source", "gen_padcat_sfx_source", "padcat2", "padcat_sfx"]
