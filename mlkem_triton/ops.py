"""Wrap the device-side Triton kernels as torch.library custom ops.

Rationale: the Triton kernels are called directly from Python, so Dynamo graph-breaks
on them and the elementwise fusion on both sides is cut. Once wrapped as custom_op,
Dynamo places them in the graph as black-box nodes, so Inductor can schedule / fuse
other ops across them, and launch overhead also goes down.

Note: these ops write the output tensors in place internally, so they are declared
with mutates_args (or they simply allocate internally and return); the fake
implementations only describe shapes/dtypes, for Dynamo to trace.
"""

from __future__ import annotations

import torch
import triton
from torch import Tensor
from torch.library import custom_op

from .devices import warps

LIB = "mlkem_triton"

# constants kept in sync with kernels/shake.py
BLOCK = 128
WARPS = 4


# ---------------------------------------------------------------- sponge (Keccak)
@custom_op(f"{LIB}::sponge", mutates_args=())
def sponge(padded: Tensor, rate: int, outlen: int) -> Tensor:
    # Pure Triton implementation. The development prototype also carried a
    # hand-written SYCL version (Intel-only, 2.2–2.6× faster at n=4096); it is not
    # included in this package because it has to shell out to icpx in a subprocess
    # to build, which completely destroys portability — see `README.md` section
    # `Limitations`.
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


# ---------------------------------------------------------------- CBD writes Y / E directly
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
