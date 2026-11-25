"""全设备常驻的批量解封装流水线（快路径）。

整条链路（位解包 → 解压 → 两个 GEMM → mod q → G/J/PRF → CBD → 压缩 →
比较 → 选择）全部留在 GPU 上，**整批只在最后同步一次**（只把 32 字节的
共享密钥拷回主机）。

数值基础
--------
所有乘法都在 **fp16 tensor core 上做、fp32 累加**：

    |a| <= 1664 < 2^11,  |b| <= 2,  K = 768
    -> 每个部分和的绝对值 < 768 * 3328 = 2.56e6 < 2^24
    -> fp32 尾数 24 bit，**任意累加顺序都逐元素精确**

所以这里不需要 NTT：`keyprep` 把每个多项式乘法折叠成一个固定的 Toeplitz
矩阵 `T(a)`，`T(a) @ b == a * b mod (X^256+1)`，直接喂给 GEMM（见
`tests/test_ntt.py`）。

为什么不用 `torch.mm(..., out_dtype=torch.float32)`
---------------------------------------------------
那个参数是 torch-xpu 的私有扩展（`aten::mm.dtype`），公开签名只有
`mm(input, mat2, *, out=None)`。fp16 输出也不行 —— `|sum|` 峰值约 1.8e5，
超过 fp16 上限 65504，直接变 inf。所以两趟 GEMM 都用本包自己的 Triton
内核（`kernels/fused_gemm.py`），fp32 累加 + 就地 epilogue，既精确又可移植。
"""

from __future__ import annotations

import os
import time
import warnings

import numpy as np
import torch
import triton
import triton.language as tl

from . import keyprep, ops
from .devices import resolve_device
from .devices import sync as _sync
from .kernels import glue
from .kernels.fused_gemm import fused_dec_gemm, fused_enc_gemm
from .kernels.padcat import padcat2, padcat_sfx
from .kernels.shake import SHA3_DOMAIN, SHAKE_DOMAIN
from .params import PARAMS, Q

RATE_SHA3_512 = 72
RATE_SHA3_256 = 136
RATE_SHAKE = 136
BLOCK = 128
WARPS = 4


# ---------------- 设备上的基础算子 ----------------


@triton.jit
def _pre_kernel(
    ct,
    out,
    ref,
    B,
    CTLEN,
    BYTE0,
    COUNT: tl.constexpr,
    D: tl.constexpr,
    CENTER: tl.constexpr,
    BC: tl.constexpr,
    JT: tl.constexpr,
):
    """unpack + decompress (+ center) + 转置，一次完成；顺带存一份 int16 参考值。

    COUNT（256/512/768/1024）不是 2 的幂，所以按 JT=256 分块。
    """
    pid = tl.program_id(0)
    rows = pid * BC + tl.arange(0, BC)
    mask = rows < B
    for c in range(0, COUNT, JT):
        j = c + tl.arange(0, JT)
        bitpos = j * D + BYTE0 * 8
        b0 = bitpos // 8
        sh = (bitpos % 8).to(tl.int32)
        off = rows[:, None] * CTLEN + b0[None, :]
        # 掩码到行边界内：D=10/4 时窗口可能越过行尾，靠掩码兜住，不再需要 padding
        ok = mask[:, None] & (b0[None, :] < CTLEN)
        v0 = tl.load(ct + off, mask=ok, other=0).to(tl.int32)
        v1 = tl.load(ct + off + 1, mask=ok & (b0[None, :] + 1 < CTLEN), other=0).to(tl.int32)
        v2 = tl.load(ct + off + 2, mask=ok & (b0[None, :] + 2 < CTLEN), other=0).to(tl.int32)
        val = ((v0 | (v1 << 8) | (v2 << 16)) >> sh[None, :]) & ((1 << D) - 1)

        # ref 用 [COUNT, B] 布局：写沿 rows 连续，正好也是 _cmp_kernel 沿 cols 读的方向
        tl.store(
            ref + j[:, None] * B + rows[None, :], tl.trans(val.to(tl.int16)), mask=mask[None, :]
        )

        dec = ((val * 3329) + (1 << (D - 1))) >> D
        if CENTER:
            dec = tl.where(dec > 1664, dec - 3329, dec)
            t = tl.trans(dec.to(tl.float16))
        else:
            t = tl.trans(dec.to(tl.int16))
        tl.store(out + j[:, None] * B + rows[None, :], t, mask=mask[None, :])


@triton.jit
def _cmp_kernel(
    acc2, E, ref, cnt, B, ROW0, ROWS, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr
):
    """mod q + compress + 与参考比较，按列累加相等计数。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    rows = ROW0 + pid_m * BM + tl.arange(0, BM)
    cols = pid_n * BN + tl.arange(0, BN)
    mask = cols[None, :] < B
    a = tl.load(acc2 + rows[:, None] * B + cols[None, :], mask=mask, other=0.0)
    e = tl.load(E + rows[:, None] * B + cols[None, :], mask=mask, other=0)
    x = a.to(tl.int32) + e.to(tl.int32)
    x = x % 3329
    x = tl.where(x < 0, x + 3329, x)
    comp = (((x << D) + 1664) // 3329) & ((1 << D) - 1)
    r = tl.load(ref + (rows - ROW0)[:, None] * B + cols[None, :], mask=mask, other=-1)
    s = tl.sum((comp == r).to(tl.int32), axis=0)
    tl.atomic_add(cnt + cols, s, mask=cols < B)


# 每个后端一次：pinned 中转到底值不值得
_PINNED_PREF: dict[str, bool] = {}
# 探测用 4 MiB：1 MiB 太小，固定开销会盖住真实差距（实测 1 MiB 只差 1.2x，
# 4.5 MiB 差到 1.9x，而这个函数的调用方本来就在 MB 级批量上）。
_PINNED_PROBE_BYTES = 1 << 22
_PINNED_MIN_BYTES = 1 << 20  # 小于 1 MiB 的批量不值得探测，直接用 pageable


def _prefer_pinned(dev: torch.device, nbytes: int = 0) -> bool:
    """是否用 pinned 暂存中转（每后端只判一次）。

    这件事**没有跨平台统一的答案**，实测两边结论相反：

    - Intel Arc B390：pageable 直拷 12–14 GB/s（驱动过 bounce buffer），
      pinned 纯 DMA 78 GB/s。多一次 memcpy（127 GB/s）之后净赚 ~2x。
    - NVIDIA GB10（Grace-Blackwell，CPU/GPU 一致性互连）：pageable 直拷已经有
      50 GB/s，pinned 反而更慢 —— 0.092 ms → memcpy 0.346 + DMA 0.084 ms。

    所以不写死，第一次调用时用 1 MiB 缓冲两边各测 3 次，只有快 10% 以上才用。
    `MLKEM_PINNED=1/0` 可以强制。
    """
    v = _PINNED_PREF.get(dev.type)
    if v is not None:
        return v
    env = os.environ.get("MLKEM_PINNED")
    if env is not None:
        v = env not in ("0", "false", "False", "")
        _PINNED_PREF[dev.type] = v
        return v
    if nbytes and nbytes < _PINNED_MIN_BYTES:
        return False  # 小批量：H2D 不是瓶颈，也别花时间探测
    v = _probe_pinned(dev)
    _PINNED_PREF[dev.type] = v
    return v


def _probe_pinned(dev: torch.device) -> bool:
    n = _PINNED_PROBE_BYTES
    try:
        pin = torch.empty(n, dtype=torch.uint8, pin_memory=True)
    except Exception:
        return False
    src = np.zeros(n, np.uint8)
    dbuf = torch.empty(n, dtype=torch.uint8, device=dev)
    src_t = _cpu_tensor(src)

    def best(fn, k=3):
        fn()
        _sync(dev)
        out = float("inf")
        for _ in range(k):
            t0 = time.perf_counter()
            fn()
            _sync(dev)
            out = min(out, time.perf_counter() - t0)
        return out

    direct = best(lambda: dbuf.copy_(src_t))
    staged = best(lambda: (pin.copy_(src_t), dbuf.copy_(pin, non_blocking=True)))
    return staged * 1.15 < direct


def total_is_uniform(buf, count: int, n0: int) -> bool:
    """`count` 条密文拼起来的总长是否正好 `count * n0`。

    这是 O(1) 的；逐条 `all(len(c) == n0 for c in lst)` 在 B=4096 时要 0.13 ms
    （占端到端 6%），不值得。误判的唯一可能是「混了长短不一的密文、而总长
    恰好等于 count*n0」—— 固定参数集的合法输入不会出现。
    """
    return n0 > 0 and len(buf) == count * n0


def _cpu_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy（可能是只读）→ CPU 张量，零拷贝。

    `np.frombuffer(b"...")` 出来的是只读数组，`torch.from_numpy` 会报一条
    UserWarning（"numpy array is not writable"）。那条警告针对的是「之后
    可能写这个张量」的场景，而我们这里只读、并且立刻拷走，所以静音。
    反过来用 `bytearray(...)` 消警告会多一次 4.4 MB 的 memcpy（+0.15 ms），
    不值得。
    """
    if arr.flags.writeable:
        return torch.from_numpy(arr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        return torch.from_numpy(arr)


def _to_device(arr: np.ndarray, dev: torch.device) -> torch.Tensor:
    """把 uint8 数组搬到设备上，**零拷贝**构造源张量。

    `np.frombuffer(b"...")` 出来的是只读数组，`torch.from_numpy` 会报一条
    UserWarning（"numpy array is not writable"）。那条警告针对的是「之后
    可能写这个张量」的场景，而我们这里只读、并且立刻 `.to(dev)` 拷走，
    所以直接静音。反过来用 `bytearray(...)` 消警告会多一次 4.4 MB 的
    memcpy（B=4096 时约 +0.4 ms），不值得。
    """
    return _cpu_tensor(arr).to(dev)


def _gemm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """通用兜底 GEMM：fp16 输入、fp32 输出的**精确**矩阵乘。

    为什么不能直接 `torch.mm(a, b)`：fp16 输出上限 65504，而这里的结果峰值
    约 1.8e5，会变 inf（实测）。`torch.mm(..., out_dtype=fp32)` 是 torch-xpu
    的私有扩展，公开 API 里没有，`out=` 又要求 dtype 一致。

    把两个操作数都升到 fp32 是最可移植的做法（任何后端都支持），代价是
    失去 fp16 tensor core，慢 1.7–3.7x。**只在 `fused_*_gemm` 返回 None
    时走这条路**，正常路径不会用到。
    """
    return torch.mm(a.to(torch.float32), b.to(torch.float32))


def pad_gpu(msgs: torch.Tensor, rate: int, domain: int) -> torch.Tensor:
    """pad10*1 + 域分隔，全在设备上。msgs [n,L] uint8 -> [n, 多倍 rate]。"""
    n, L = msgs.shape
    total = ((L + 1 + rate - 1) // rate) * rate
    out = torch.zeros((n, total), dtype=torch.uint8, device=msgs.device)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, total - 1] = out[:, total - 1] | 0x80
    return out


def _pad2_gpu(a: torch.Tensor, b: torch.Tensor, n: int, rate: int, domain: int) -> torch.Tensor:
    """`pad_gpu(cat([a, b]))` 的一次 launch 等价实现。

    a/b 中 1 维的按「广播到 n 行」处理（等价于 `.expand(n, L)`），2 维的按逐行。
    这就是 J（z 广播 + ct 逐行）和 G（m' 逐行 + h_ek 广播）的形状。
    老路径是 `cat` + `zeros` + `copy_` 三次 launch，J 档（B=4096）实测 192 µs；
    这里 28 µs。段长不是 8 的倍数时回退（当前三套参数都满足）。
    """
    la = a.shape[1] if a.dim() == 2 else a.numel()
    lb = b.shape[1] if b.dim() == 2 else b.numel()
    total = ((la + lb + 1 + rate - 1) // rate) * rate
    if la % 8 == 0 and lb % 8 == 0 and total % 8 == 0:
        return padcat2(
            a.view(torch.int64),
            b.view(torch.int64),
            n,
            la // 8,
            lb // 8,
            total // 8,
            domain,
            a_bcast=a.dim() == 1,
            b_bcast=b.dim() == 1,
        )
    ca = a if a.dim() == 2 else a.expand(n, la)
    cb = b if b.dim() == 2 else b.expand(n, lb)
    return pad_gpu(torch.cat([ca, cb], dim=1), rate, domain)


def _pad_prf_gpu(
    prefix: torch.Tensor, n: int, rate: int, domain: int, mod: int, off: int
) -> torch.Tensor:
    """`pad_gpu(cat([prefix[row // mod], [row % mod + off]]))` 的一次 launch 版本。

    PRF 的输入是 `r'`（32 B）接一个 1 字节的 j，即 33 B/条；`prefix` 可以是大张量
    里的一列切片（例如 G 的 64 B 输出里的后 32 B），只要最后一维连续。
    """
    total = ((prefix.shape[-1] + 1 + 1 + rate - 1) // rate) * rate
    last_ok = prefix.dim() == 2 and prefix.shape[-1] % 8 == 0 and prefix.stride(-1) == 1
    if last_ok and total % 8 == 0:
        v = prefix.view(torch.int64)
        stride_w = v.stride(0)
        return padcat_sfx(v, n, v.shape[-1], total // 8, domain, mod, off, stride_w=stride_w)
    rows = torch.arange(n, device=prefix.device)
    sfx = ((rows % mod) + off).to(torch.uint8)[:, None]
    return pad_gpu(torch.cat([prefix.index_select(0, rows // mod), sfx], dim=1), rate, domain)


def sponge_gpu(padded: torch.Tensor, rate: int, outlen: int) -> torch.Tensor:
    """已填充的 [n, 多倍 rate] uint8 -> [n, outlen] uint8，全程设备端。"""
    _n, tot = padded.shape
    nblocks = tot // rate
    assert nblocks * rate == tot
    return ops.sponge(padded, rate, outlen)


@triton.jit
def _unpack_kernel(buf, out, B, STRIDE, COUNT, D: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    pid_r = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_r * BR + tl.arange(0, BR)
    cols = pid_c * BC + tl.arange(0, BC)
    mask = (rows[:, None] < B) & (cols[None, :] < COUNT)

    bitpos = cols * D
    b0 = bitpos // 8
    sh = (bitpos % 8).to(tl.int32)
    off = rows[:, None] * STRIDE + b0[None, :]

    v0 = tl.load(buf + off, mask=mask, other=0).to(tl.int32)
    v1 = tl.load(buf + off + 1, mask=mask, other=0).to(tl.int32)
    v2 = tl.load(buf + off + 2, mask=mask, other=0).to(tl.int32)
    val = ((v0 | (v1 << 8) | (v2 << 16)) >> sh[None, :]) & ((1 << D) - 1)
    tl.store(out + rows[:, None] * COUNT + cols[None, :], val, mask=mask)


def unpack_bits_gpu(buf: torch.Tensor, d: int, count: int) -> torch.Tensor:
    """[B, nb] uint8 -> [B, count] int32，按小端位序取 d-bit 字段。

    用 Triton 逐字节读取（比 PyTorch 的 3 次高级索引 gather 快约 3 倍）。
    """
    B, nb = buf.shape
    padded = torch.zeros((B, nb + 2), dtype=torch.uint8, device=buf.device)
    padded[:, :nb] = buf
    out = torch.empty((B, count), dtype=torch.int32, device=buf.device)
    BR, BC = 32, 128
    grid = (triton.cdiv(B, BR), triton.cdiv(count, BC))
    _unpack_kernel[grid](padded, out, B, nb + 2, count, D=d, BR=BR, BC=BC)
    return out


def pack_bits_gpu(vals: torch.Tensor, d: int) -> torch.Tensor:
    """[count, B] 整数 -> [B, count*d/8] uint8（ByteEncode 的位打包）。"""
    count, B = vals.shape
    nbits = count * d
    assert nbits % 8 == 0
    shifts = torch.arange(d, device=vals.device, dtype=torch.int32)
    bits = (vals.to(torch.int32)[:, :, None] >> shifts) & 1  # [count,B,d]
    bits = bits.permute(0, 2, 1).reshape(nbits, B).T  # [B,nbits]
    bits = bits.reshape(B, nbits // 8, 8).to(torch.int32)
    w = 1 << torch.arange(8, device=vals.device, dtype=torch.int32)
    return (bits * w).sum(-1).to(torch.uint8)


def decompress_gpu(x: torch.Tensor, d: int) -> torch.Tensor:
    return ((x.to(torch.int32) * Q) + (1 << (d - 1))) >> d


def compress_gpu(x: torch.Tensor, d: int) -> torch.Tensor:
    x = x.to(torch.int32) % Q
    return (((x << d) + (Q // 2)) // Q) & ((1 << d) - 1)


@triton.jit
def _cbd2_kernel(buf, out, n, BLOCK: tl.constexpr):
    """eta=2 的 CBD：一个字节恰好两个系数，一次读入、一次写出。

    原 PyTorch 实现在 [n,128] 上串了 6 个中间张量（&、>>、popcount…），
    纯带宽就吃掉 1 ms；这里把整段压进一个 kernel。
    """
    pid = tl.program_id(0)
    rows = pid * BLOCK + tl.arange(0, BLOCK)
    cols = tl.arange(0, 128)
    mask = rows[:, None] < n
    b = tl.load(buf + rows[:, None] * 128 + cols[None, :], mask=mask, other=0).to(tl.int32)
    lo = b & 0x0F
    hi = (b >> 4) & 0x0F
    c0 = ((lo & 1) + ((lo >> 1) & 1)) - (((lo >> 2) & 1) + ((lo >> 3) & 1))
    c1 = ((hi & 1) + ((hi >> 1) & 1)) - (((hi >> 2) & 1) + ((hi >> 3) & 1))
    v = tl.interleave(c0, c1).to(tl.int16)
    ocols = tl.arange(0, 256)
    tl.store(out + rows[:, None] * 256 + ocols[None, :], v, mask=rows[:, None] < n)


def cbd_gpu(bufs: torch.Tensor, eta: int) -> torch.Tensor:
    """[n, 64·eta] uint8 -> [n, 256] int32。"""
    n = bufs.shape[0]
    assert bufs.shape[1] == 64 * eta

    if eta == 2:
        return ops.cbd(bufs, eta)

    # 通用路径（eta=3 等）
    b = bufs.to(torch.int32)
    shifts = torch.arange(8, device=bufs.device, dtype=torch.int32)
    bits = (b[:, :, None] >> shifts) & 1  # [n, 64·eta, 8] int32
    bits = bits.reshape(n, 256, 2 * eta)
    a = bits[:, :, :eta].sum(-1)
    c = bits[:, :, eta:].sum(-1)
    return a - c


def center_gpu(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int32) % Q
    return torch.where(x > (Q - 1) // 2, x - Q, x)


# ---------------- 主类 ----------------


class Decapsulator:
    """把一个 ML-KEM 私钥装到设备上，之后可以批量解封装任意多条密文。

    Args:
        ps: 参数集名，``"ML-KEM-512"`` / ``"ML-KEM-768"`` / ``"ML-KEM-1024"``。
        dk: 私钥字节串（长度 ``768k+96``）。
        device: ``None`` 表示自动选第一个可用加速设备；也可显式给
            ``"cuda"`` / ``"xpu"`` / ``"cuda:1"``。环境变量 ``MLKEM_DEVICE``
            优先级低于显式参数、高于自动探测。
    """

    def __init__(self, ps: str, dk: bytes, device=None):
        p = PARAMS[ps]
        device = resolve_device(device)
        self.ps, self.p, self.device = ps, p, device
        self.kp = keyprep.parse_key(ps, dk)
        k = p.k
        self.h_ek = torch.frombuffer(
            bytearray(dk[768 * k + 32 : 768 * k + 64]), dtype=torch.uint8
        ).to(device)
        self.z = torch.frombuffer(bytearray(dk[768 * k + 64 : 768 * k + 96]), dtype=torch.uint8).to(
            device
        )
        self.W_dec = (
            torch.from_numpy(np.ascontiguousarray(self.kp.W_dec)).to(torch.float16).to(device)
        )
        self.W_enc = (
            torch.from_numpy(np.ascontiguousarray(self.kp.W_enc)).to(torch.float16).to(device)
        )
        self._staging: dict[tuple[int, int], tuple] = {}

    # ---- 哈希批处理 ----
    def _G(self, m_bytes: torch.Tensor, B: int) -> torch.Tensor:
        # m' || h_ek，pad 到 1 块 72 B；一次 launch（原来是 cat+zeros+copy 三次）
        return sponge_gpu(
            _pad2_gpu(m_bytes, self.h_ek, B, RATE_SHA3_512, SHA3_DOMAIN), RATE_SHA3_512, 64
        )

    def _J(self, cts: torch.Tensor, B: int) -> torch.Tensor:
        # z || ct，pad 到 9 块 136 B；一次 launch（原来 192 µs，现在 28 µs）
        return sponge_gpu(_pad2_gpu(self.z, cts, B, RATE_SHAKE, SHAKE_DOMAIN), RATE_SHAKE, 32)

    def _prf_bufs(self, r_prime: torch.Tensor, B: int):
        """PRF：r' → 两个 sponge 输出（还没做 CBD），供融合的 CBD→Y/E 用。"""
        p, k = self.p, self.p.k
        # r' || [j]（33 B）→ pad 到 1 块 136 B。r_prime 通常是 G 输出的后 32 B
        # （[B,64] 的切片，最后一维连续），可以让 kernel 带 stride 原地读，不用拷。
        y_pad = _pad_prf_gpu(r_prime, B * k, RATE_SHAKE, SHAKE_DOMAIN, mod=k, off=0)
        e_pad = _pad_prf_gpu(r_prime, B * (k + 1), RATE_SHAKE, SHAKE_DOMAIN, mod=k + 1, off=k)
        return (
            sponge_gpu(y_pad, RATE_SHAKE, 64 * p.eta1),
            sponge_gpu(e_pad, RATE_SHAKE, 64 * p.eta2),
        )

    def _prf_cbd(self, r_prime: torch.Tensor, B: int):
        p, k = self.p, self.p.k
        y_buf, e_buf = self._prf_bufs(r_prime, B)
        y = cbd_gpu(y_buf, p.eta1).reshape(B, k, 256)
        e = cbd_gpu(e_buf, p.eta2).reshape(B, k + 1, 256)
        return y, e

    # ---- 主流程 ----
    def decaps(self, cts) -> np.ndarray:
        """批量解封装。

        `cts` 支持四种形态，换算成设备张量所需的**主机侧成本差很多**
        （B=4096 实测，见 benchmarks/bench_host.py）：

        ============================================  ===========
        形态                                            主机侧
        ============================================  ===========
        `torch.Tensor`（已在设备上）                      ~0 ms
        `bytes` / `bytearray` / `memoryview`（一整块）     ~0.11 ms
        `np.ndarray` [B, ct_size]                        ~0.14 ms
        `list[bytes]`（每条一条密文）                     ~0.36 ms
        ============================================  ===========

        差别全在 `b"".join` 和「每条长度是否一致」的校验上。服务端从网络收
        密文时，如果能直接收进一整块缓冲（`socket.recv_into` 或 pinned 内存），
        就走前两行；如果是别处解析出来的 `list[bytes]`，就付那 0.36 ms。
        """
        n0_expected = self.p.ct_size
        if isinstance(cts, torch.Tensor):
            self._check_shape(cts.shape[-2:] if cts.dim() > 1 else cts.shape, n0_expected)
            ct = cts.reshape(-1, n0_expected)
            if ct.dtype != torch.uint8:
                raise TypeError(f"cts 需要 uint8，收到 {ct.dtype}")
            if ct.device != self.device:
                ct = ct.to(self.device)
            return self.decaps_torch(ct, ct.shape[0]).cpu().numpy()

        arr = self._to_numpy2d(cts, n0_expected)
        B = arr.shape[0]

        # 唯一一次 H2D（走 pinned 暂存，见 _h2d）。
        ct = self._h2d(arr)

        # 唯一一次 D2H。
        return self.decaps_torch(ct, B).cpu().numpy()

    def _h2d(self, arr: np.ndarray) -> torch.Tensor:
        """主机 → 设备的单次拷贝，尽量走 **pinned 暂存**。

        为什么值得：这一台机器上 pageable → device 只有 12 GB/s（驱动要经过
        一个 bounce buffer），而 pinned → device 的纯 DMA 有 78 GB/s。
        多出来的那一次「numpy → pinned」memcpy 只要 0.035 ms（4.46 MB @127 GB/s），
        所以净赚：

            pageable 直拷        join 0.157 + 0.362 = 0.519 ms
            pinned 中转          join 0.157 + 0.035 + 0.057 = 0.249 ms

        （B=4096 实测，见 benchmarks/bench_host.py。）

        暂存缓冲按 (B, ct_size) 缓存并复用；`MLKEM_NO_PINNED=1` 可关掉。
        设备侧那块缓冲也是复用的 —— `decaps_torch` 只读它，而且返回前已经
        同步过，所以下一次调用覆盖它是安全的。
        """
        n, L = arr.shape
        key = (n, L)
        st = self._staging.get(key)
        if st is None:
            pin = None
            if _prefer_pinned(self.device, arr.nbytes):
                try:
                    pin = torch.empty(n, L, dtype=torch.uint8, pin_memory=True)
                except Exception:
                    pin = None
            st = (pin, torch.empty(n, L, dtype=torch.uint8, device=self.device))
            if len(self._staging) >= 4:
                self._staging.clear()
            self._staging[key] = st
        pin, dbuf = st
        if pin is None:
            return _to_device(arr, self.device)
        src = _cpu_tensor(arr).reshape(-1)
        pin.view(-1).copy_(src)
        dbuf.copy_(pin, non_blocking=True)
        return dbuf

    @staticmethod
    def _check_shape(shape, n0: int) -> None:
        if len(shape) >= 1 and shape[-1] != n0:
            raise ValueError(f"每条密文应为 {n0} 字节，收到 {tuple(shape)}")

    @staticmethod
    def _to_numpy2d(cts, n0: int) -> np.ndarray:
        """把各种输入形态归一成 [B, n0] uint8 的 CPU 数组（尽量零/单次拷贝）。"""
        if isinstance(cts, np.ndarray):
            if cts.dtype != np.uint8:
                cts = cts.astype(np.uint8)
            arr = np.ascontiguousarray(cts)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or (n0 > 0 and arr.shape[1] != n0):
                raise ValueError(f"期望 [B, {n0}] 的 uint8 数组，收到 {arr.shape}")
            return arr

        if isinstance(cts, (bytes, bytearray, memoryview)):
            n = len(cts)
            if n0 <= 0 or n % n0:
                raise ValueError(f"输入长度 {n} 不是密文长度 {n0} 的整数倍")
            return np.frombuffer(cts, np.uint8).reshape(n // n0, n0)

        lst = list(cts)
        if not lst:
            return np.empty((0, n0), np.uint8)
        try:
            buf = b"".join(lst)  # 只接受 bytes-like，否则 TypeError
        except TypeError:
            buf = None
        # O(1) 校验：总长正好是 n0 * len(lst) 就认为是等长的。
        # 之前逐条 `all(len(c) == n0 for c in lst)`，B=4096 时白花 0.13 ms
        # （占端到端 6%）。代价：如果调用方混了不同长度的密文、而它们的总长
        # 又恰好等于 n0*len，会被误判 —— 对固定参数集的合法输入不可能发生。
        if buf is not None and total_is_uniform(buf, len(lst), n0):
            return np.frombuffer(buf, np.uint8).reshape(len(lst), n0)
        # 慢路径：元素不是 bytes-like，或者长度确实不等
        return np.stack([np.frombuffer(bytes(c), np.uint8) for c in lst])

    def decaps_torch(self, ct: torch.Tensor, B: int) -> torch.Tensor:
        """纯 device 侧流程，输入 ct[B, ct_size] uint8，输出 K[B, 32] uint8。"""
        p, k, dev = self.p, self.p.k, self.device

        # 1. 融合：unpack + 解压 (+center) + 转置  →  U[256k,B] fp16, V[256,B] int16
        #    顺带产出 int16 参考值 c1r/c2r 供最后比较（避免再解包一次）
        KK = 256 * k
        U, c1r = ops.pre(ct, B, p.ct_size, 0, KK, p.du, 1, 16, 256)
        V, c2r = ops.pre(ct, B, p.ct_size, 32 * p.du * k, 256, p.dv, 0, 16, 256)

        # 2. w = v − W_dec @ u，再 mod q → compress(1 bit) → 打包成 m_bytes。
        #    GEMM 和整段 epilogue 融成一个内核（acc 不落显存，4.2 MB 的
        #    写出 + 读回全省掉）。实测比「mm + compress_pack」快 1.36x。
        m_bytes = fused_dec_gemm(self.W_dec, U, V, B)
        if m_bytes is None:  # 形状/后端不满足约束
            acc = _gemm_fp32(self.W_dec, U)
            if glue.can_compress_pack(V, acc, B):
                m_bytes = ops.compress_pack(acc, V, B)  # [B,32] uint8
            else:
                w = (V.to(torch.int32) - acc.to(torch.int32)) % Q
                m_bytes = pack_bits_gpu(compress_gpu(w, 1), 1)  # [B,32] uint8

        # 4. G / J
        gr = self._G(m_bytes, B)
        K_prime = gr[:, :32]
        r_prime = gr[:, 32:]
        K_bar = self._J(ct, B)

        # 5. PRF + CBD（+ 转置）。eta=2 时 CBD 直接写进 Y/E，不落 y/e 中间张量。
        y_buf, e_buf = self._prf_bufs(r_prime, B)
        YE = None
        if glue.can_cbd_y(y_buf, B, k, p.eta1) and glue.can_cbd_e(e_buf, m_bytes, B, k, p.eta2):
            YE = (ops.cbd_y(y_buf, B, k), ops.cbd_e(e_buf, m_bytes, B, k))
        if YE is None:
            y = cbd_gpu(y_buf, p.eta1).reshape(B, k, 256)
            e = cbd_gpu(e_buf, p.eta2).reshape(B, k + 1, 256)

        # 6. 重加密：y / e / m_bytes → Y [256k,B] fp16 + E [256(k+1),B] int16，一个 kernel
        #    Y 必须**连续**（fused_enc_gemm 的约束）。旧写法
        #    `permute(1,2,0).reshape(...).to(fp16)` 实际是 stride=(1,256k) 的转置视图，
        #    值一样但非连续，会让融合 GEMM 静默返回 False、整条流水线走 mm + cmp 回退。
        if YE is not None:
            Y, E = YE
        elif glue.can_build_ye(y, e, m_bytes, B, k):
            Y, E = ops.build_ye(y, e, m_bytes, B, k)
        else:
            Y = y.permute(1, 2, 0).reshape(k * 256, B).to(torch.float16).contiguous()
            # int16 足够：E1 ∈ [-2,2]，E2+1665*m_bits ∈ [-2, 1667]。
            E1 = e[:, :k].permute(1, 2, 0).reshape(k * 256, B).to(torch.int16)
            E2 = (e[:, k].T + 1665 * glue.unpack_mbits(m_bytes, B)).to(torch.int16)
            E = torch.cat([E1, E2], dim=0)  # [256(k+1),B]

        # 7. 融合：GEMM + mod q + compress + 比较（按列累加相等计数）
        #    fused_enc_gemm 返回 False 表示形状/dtype 不满足其约束且未写任何输出，
        #    此时回退到 mm + cmp 两步。
        cnt = torch.zeros(B, dtype=torch.int32, device=dev)
        if not fused_enc_gemm(self.W_enc, Y, E, (c1r, c2r), cnt, B, p.du, p.dv):
            acc2 = _gemm_fp32(self.W_enc, Y)
            BM, BN = 128, 64
            ops.cmp(acc2, E, c1r, cnt, B, 0, KK, p.du, BM, BN)
            ops.cmp(acc2, E, c2r, cnt, B, KK, 256, p.dv, BM, BN)
        eq = cnt == (KK + 256)

        # 8. 常数时间选择
        return torch.where(eq[:, None], K_prime, K_bar)  # [B,32] uint8


def decaps_batch(ps: str, dk: bytes, cts, device=None) -> np.ndarray:
    """一次性解封装一批密文（每次都重新预处理私钥，批量小的时候别用它）。"""
    return Decapsulator(ps, dk, device=device).decaps(cts)


__all__ = [
    "Decapsulator",
    "cbd_gpu",
    "center_gpu",
    "compress_gpu",
    "decaps_batch",
    "decompress_gpu",
    "pack_bits_gpu",
    "pad_gpu",
    "sponge_gpu",
    "unpack_bits_gpu",
]
