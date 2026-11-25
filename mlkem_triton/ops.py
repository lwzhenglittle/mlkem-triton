"""把 decaps_gpu 里的手写 Triton 内核包装成 torch.library 自定义算子。

目的：Triton 内核是直接 Python 调用，Dynamo 遇到就 graph break，
两侧的 elementwise 融合被切断。包成 custom_op 后 Dynamo 把它们当黑盒节点放进图里，
Inductor 就能跨它们调度 / 融合其它算子，同时发射开销也降下来。

注意：这些算子内部是原地写输出张量的，所以用 mutates_args 声明（或干脆内部自己分配再返回），
fake 实现只描述形状/dtype，供 Dynamo 追踪用。
"""

from __future__ import annotations

import torch
import triton
from torch import Tensor
from torch.library import custom_op

from .devices import warps

LIB = "mlkem_triton"

# 与 decaps_gpu 保持一致的常量
BLOCK = 128
WARPS = 4


# ---------------------------------------------------------------- sponge (Keccak)
@custom_op(f"{LIB}::sponge", mutates_args=())
def sponge(padded: Tensor, rate: int, outlen: int) -> Tensor:
    # 纯 Triton 实现。开发原型里还有一个手写 SYCL 版本（Intel 专属，n=4096 上
    # 快 2.2–2.6×），因为要走子进程调 icpx 构建、彻底破坏可移植性，没有包含在
    # 这个包里 —— 见 README「限制与说明」。
    from .kernels.shake import get_sponge

    n, tot = padded.shape
    nblocks = tot // rate
    assert nblocks * rate == tot
    x = padded.contiguous().view(torch.int64)
    out = torch.empty((n, outlen // 8), dtype=torch.int64, device=padded.device)
    kern = get_sponge(rate, outlen)
    grid = (triton.cdiv(n, BLOCK),)
    kern[grid](
        x,
        out,
        n,
        nblocks,
        nblocks * rate // 8,
        OUTW=outlen // 8,
        BLOCK=BLOCK,
        num_warps=warps("shake", WARPS),
    )
    return out.view(torch.uint8).view(n, outlen)


@sponge.register_fake
def _(padded, rate, outlen):
    return padded.new_empty((padded.shape[0], outlen), dtype=torch.uint8)


# ---------------------------------------------------------------- CBD
@custom_op(f"{LIB}::cbd", mutates_args=())
def cbd(bufs: Tensor, eta: int) -> Tensor:
    from .pipeline import _cbd2_kernel

    n = bufs.shape[0]
    assert bufs.shape[1] == 64 * eta
    if eta != 2:
        raise NotImplementedError("custom op only covers the eta=2 fast path")
    out = torch.empty((n, 256), dtype=torch.int16, device=bufs.device)
    BL = 64
    _cbd2_kernel[(triton.cdiv(n, BL),)](bufs, out, n, BLOCK=BL)
    return out


@cbd.register_fake
def _(bufs, eta):
    return bufs.new_empty((bufs.shape[0], 256), dtype=torch.int16)


# ---------------------------------------------------------------- unpack + decompress
@custom_op(f"{LIB}::pre", mutates_args=())
def pre(
    ct: Tensor, B: int, CTLEN: int, BYTE0: int, COUNT: int, D: int, CENTER: int, BC: int, JT: int
) -> list[Tensor]:
    from .pipeline import _pre_kernel

    out = torch.empty((COUNT, B), dtype=torch.float16 if CENTER else torch.int16, device=ct.device)
    ref = torch.empty((COUNT, B), dtype=torch.int16, device=ct.device)
    grid = (triton.cdiv(B, BC),)
    _pre_kernel[grid](ct, out, ref, B, CTLEN, BYTE0, COUNT=COUNT, D=D, CENTER=CENTER, BC=BC, JT=JT)
    return [out, ref]


@pre.register_fake
def _(ct, B, CTLEN, BYTE0, COUNT, D, CENTER, BC, JT):
    return [
        ct.new_empty((COUNT, B), dtype=torch.float16 if CENTER else torch.int16),
        ct.new_empty((COUNT, B), dtype=torch.int16),
    ]


# ---------------------------------------------------------------- mod q + compress + pack m'
@custom_op(f"{LIB}::compress_pack", mutates_args=())
def compress_pack(acc: Tensor, V: Tensor, B: int) -> Tensor:
    from .kernels.glue import compress_pack as impl

    return impl(V, acc, B)


@compress_pack.register_fake
def _(acc, V, B):
    return acc.new_empty((B, 32), dtype=torch.uint8)


# ---------------------------------------------------------------- build Y / E
@custom_op(f"{LIB}::build_ye", mutates_args=())
def build_ye(y: Tensor, e: Tensor, mbytes: Tensor, B: int, K: int) -> list[Tensor]:
    from .kernels.glue import build_ye as impl

    return list(impl(y, e, mbytes, B, K))


@build_ye.register_fake
def _(y, e, mbytes, B, K):
    return [
        y.new_empty((K * 256, B), dtype=torch.float16),
        e.new_empty(((K + 1) * 256, B), dtype=torch.int16),
    ]


# ---------------------------------------------------------------- CBD 直接写 Y / E
@custom_op(f"{LIB}::cbd_y", mutates_args=())
def cbd_y(y_buf: Tensor, B: int, K: int) -> Tensor:
    from .kernels.glue import cbd_y as impl

    return impl(y_buf, B, K)


@cbd_y.register_fake
def _(y_buf, B, K):
    return y_buf.new_empty((K * 256, B), dtype=torch.float16)


@custom_op(f"{LIB}::cbd_e", mutates_args=())
def cbd_e(e_buf: Tensor, mbytes: Tensor, B: int, K: int) -> Tensor:
    from .kernels.glue import cbd_e as impl

    return impl(e_buf, mbytes, B, K)


@cbd_e.register_fake
def _(e_buf, mbytes, B, K):
    return e_buf.new_empty(((K + 1) * 256, B), dtype=torch.int16)


# ---------------------------------------------------------------- mod q + compress + cmp
@custom_op(f"{LIB}::cmp", mutates_args=("cnt",))
def cmp(
    acc2: Tensor,
    E: Tensor,
    ref: Tensor,
    cnt: Tensor,
    B: int,
    ROW0: int,
    ROWS: int,
    D: int,
    BM: int,
    BN: int,
) -> None:
    from .pipeline import _cmp_kernel

    grid = (ROWS // BM, triton.cdiv(B, BN))
    _cmp_kernel[grid](acc2, E, ref, cnt, B, ROW0, ROWS, D=D, BM=BM, BN=BN)


@cmp.register_fake
def _(acc2, E, ref, cnt, B, ROW0, ROWS, D, BM, BN):
    return None


__all__ = [
    "build_ye",
    "cbd",
    "cbd_e",
    "cbd_y",
    "cmp",
    "compress_pack",
    "pre",
    "sponge",
]
