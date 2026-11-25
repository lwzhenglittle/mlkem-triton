"""Keccak-f[1600] 的 Triton 实现（向量单元 / SIMT 风格）。

映射方式：**一个线程 = 一个 Keccak 实例**，25 个 lane 各用一个 `[BLOCK]` 张量。
这样每个 lane 张量在实例维度上是连续的，访存合并良好；θ/π 的 lane 换位在
unrolled 代码里表现为寄存器重命名，没有实际数据搬运。

实现细节：
- 24 轮全部展开（Python 侧代码生成），因为 Triton 的 kernel 是 AST 解析的，
  不方便在 kernel 里用 Python list 操作 25 个变量。
- 注意 Python 3.14 + Triton 3.5.0 的坑：kernel 里不能用 1 参数 `range(N)`。
  这里干脆完全展开，不写循环。
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import numpy as np

# ---------------- Keccak 常量 ----------------

# r[x][y]：ρ 步骤的旋转量
ROT = [
    [0, 36, 3, 41, 18],
    [1, 44, 10, 45, 2],
    [62, 6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39, 8, 14],
]

RC = [
    0x0000000000000001,
    0x0000000000008082,
    0x800000000000808A,
    0x8000000080008000,
    0x000000000000808B,
    0x0000000080000001,
    0x8000000080008081,
    0x8000000000008009,
    0x000000000000008A,
    0x0000000000000088,
    0x0000000080008009,
    0x000000008000000A,
    0x000000008000808B,
    0x800000000000008B,
    0x8000000000008089,
    0x8000000000008003,
    0x8000000000008002,
    0x8000000000000080,
    0x000000000000800A,
    0x800000008000000A,
    0x8000000080008081,
    0x8000000000008080,
    0x0000000080000001,
    0x8000000080008008,
]


def _rotl_expr(var: str, r: int) -> str:
    if r == 0:
        return var
    return f"(({var} << {r}) | ({var} >> {64 - r}))"


def gen_source(n_rounds: int = 24) -> str:
    """生成完整的 Triton kernel 源码。"""
    L = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def keccak_f_kernel(X, OUT, n, BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    offs = pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = offs < n",
    ]
    for i in range(25):
        L.append(
            f"    a{i} = tl.load(X + {i} * n + offs, mask=m, other=0).to(tl.uint64, bitcast=True)"
        )

    for rnd in range(n_rounds):
        # θ
        for x in range(5):
            L.append(f"    c{x} = a{x} ^ a{x + 5} ^ a{x + 10} ^ a{x + 15} ^ a{x + 20}")
        for x in range(5):
            cp, cn = (x + 1) % 5, (x + 4) % 5
            L.append(f"    d{x} = c{cn} ^ {_rotl_expr(f'c{cp}', 1)}")
        for y in range(5):
            for x in range(5):
                i = x + 5 * y
                L.append(f"    a{i} = a{i} ^ d{x}")
        # ρ + π
        for y in range(5):
            for x in range(5):
                src = x + 5 * y
                dst = y + 5 * ((2 * x + 3 * y) % 5)
                L.append(f"    b{dst} = {_rotl_expr(f'a{src}', ROT[x][y])}")
        # χ
        for y in range(5):
            for x in range(5):
                L.append(
                    f"    a{x + 5 * y} = b{x + 5 * y} ^ "
                    f"((~b{(x + 1) % 5 + 5 * y}) & b{(x + 2) % 5 + 5 * y})"
                )
        # ι
        L.append(f"    a0 = a0 ^ {RC[rnd]}")

    for i in range(25):
        L.append(f"    tl.store(OUT + {i} * n + offs, a{i}.to(tl.int64, bitcast=True), mask=m)")
    return "\n".join(L) + "\n"


_KERNEL = None


def get_kernel():
    """生成源码 -> 写临时文件 -> import（Triton 需要能拿到源码）。"""
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    src = gen_source()
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_mlkem_keccak_{tag}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"_mlkem_keccak_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _KERNEL = mod.keccak_f_kernel
    return _KERNEL


# ---------------- 纯 Python 参考实现 ----------------

M64 = (1 << 64) - 1


def _rotl(v: int, r: int) -> int:
    if r == 0:
        return v & M64
    return ((v << r) | (v >> (64 - r))) & M64


def keccak_f_ref(state: np.ndarray) -> np.ndarray:
    """state: [25] uint64，原地做 24 轮（返回新数组）。"""
    a = [int(x) & M64 for x in state]
    for rnd in range(24):
        c = [a[x] ^ a[x + 5] ^ a[x + 10] ^ a[x + 15] ^ a[x + 20] for x in range(5)]
        d = [c[(x + 4) % 5] ^ _rotl(c[(x + 1) % 5], 1) for x in range(5)]
        for y in range(5):
            for x in range(5):
                a[x + 5 * y] ^= d[x]
        b = [0] * 25
        for y in range(5):
            for x in range(5):
                b[y + 5 * ((2 * x + 3 * y) % 5)] = _rotl(a[x + 5 * y], ROT[x][y])
        for y in range(5):
            for x in range(5):
                a[x + 5 * y] = (
                    b[x + 5 * y] ^ ((~b[(x + 1) % 5 + 5 * y]) & M64) & b[(x + 2) % 5 + 5 * y]
                )
        a[0] ^= RC[rnd]
    return np.array([x & M64 for x in a], dtype=np.uint64)


__all__ = ["RC", "ROT", "gen_source", "get_kernel", "keccak_f_ref"]
