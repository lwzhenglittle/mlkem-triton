"""Fully device-resident batched decapsulation pipeline (fast path).

The whole chain (bit unpack → decompress → two GEMMs → mod q → G/J/PRF → CBD →
compress → compare → select) stays on the GPU, and **the whole batch is
synchronized only once at the end** (only the 32-byte shared key of each
ciphertext, i.e. 32*B bytes in total, is copied back to the host).

Numeric basis
-------------
All multiplications run on the **fp16 tensor cores with fp32 accumulation**:

    |a| <= 1664 < 2^11,  |b| <= 2,  K = 768
    -> the absolute value of every partial sum < 768 * 3328 = 2.56e6 < 2^24
    -> fp32 has a 24-bit mantissa, so **any accumulation order is exact elementwise**

So no NTT is needed here: `keyprep` folds each polynomial multiplication into a
fixed Toeplitz matrix `T(a)`, with `T(a) @ b == a * b mod (X^256+1)`, fed
straight into the GEMM (see `tests/test_ntt.py`).

Why not `torch.mm(..., out_dtype=torch.float32)`
------------------------------------------------
That argument is a torch-xpu private extension (`aten::mm.dtype`); the public
signature only has `mm(input, mat2, *, out=None)`. fp16 output does not work
either — the `|sum|` peak is about 1.8e5, above the fp16 maximum of 65504, so it
turns into inf. Both GEMM passes therefore use this package's own Triton
kernels (`kernels/fused_gemm.py`): fp32 accumulation plus an in-place epilogue,
exact and portable.
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
from .params import PARAMS, Q
from .primitives.sponge import DOMAIN_SHA3, DOMAIN_SHAKE, SHA3_256, SHA3_512, SHAKE256

RATE_SHA3_512 = SHA3_512.rate
RATE_SHA3_256 = SHA3_256.rate
RATE_SHAKE = SHAKE256.rate
BLOCK = 128
WARPS = 4


# ---------------- basic operators on device ----------------


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
    """unpack + decompress (+ center) + transpose, all in one pass; also keeps an
    int16 reference copy.

    COUNT (256/512/768/1024) is not a power of 2, so it is tiled by JT=256.
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
        # Mask within the row bounds: with D=10/4 the window may cross the row end;
        # the mask covers it, so padding is no longer needed
        ok = mask[:, None] & (b0[None, :] < CTLEN)
        v0 = tl.load(ct + off, mask=ok, other=0).to(tl.int32)
        v1 = tl.load(ct + off + 1, mask=ok & (b0[None, :] + 1 < CTLEN), other=0).to(tl.int32)
        v2 = tl.load(ct + off + 2, mask=ok & (b0[None, :] + 2 < CTLEN), other=0).to(tl.int32)
        val = ((v0 | (v1 << 8) | (v2 << 16)) >> sh[None, :]) & ((1 << D) - 1)

        # ref uses a [COUNT, B] layout: writes are contiguous along rows, which is
        # exactly the direction _cmp_kernel reads along cols
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
    """mod q + compress + compare against the reference, accumulating the
    equality count per column."""
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


# once per backend: whether pinned staging is worth it at all
_PINNED_PREF: dict[str, bool] = {}
# Probe with 4 MiB: 1 MiB is too small, fixed overhead would mask the real gap
# (measured only 1.2x at 1 MiB, up to 1.9x at 4.5 MiB, and this function's
# callers are on MB-scale batches anyway).
_PINNED_PROBE_BYTES = 1 << 22
_PINNED_MIN_BYTES = 1 << 20  # batches below 1 MiB: not worth probing, just use pageable


def _prefer_pinned(dev: torch.device, nbytes: int = 0) -> bool:
    """Whether to use pinned staging (decided once per backend).

    There is **no cross-platform uniform answer** here; measurements on the two
    sides reached opposite conclusions:

    - Intel Arc B390: direct pageable copy 12–14 GB/s (the driver goes through
      a bounce buffer), pinned pure DMA 78 GB/s. After one extra memcpy
      (127 GB/s) it nets ~2x.
    - NVIDIA GB10 (Grace-Blackwell, CPU/GPU coherent interconnect): direct
      pageable copy already reaches 50 GB/s, and pinned is actually slower —
      0.092 ms → memcpy 0.346 + DMA 0.084 ms.

    So it is not hardcoded: the first call times both sides 3 times on a 4 MiB
    buffer (1 MiB is too small -- fixed overhead masks the real gap), and pinned
    is used only if it is more than 15% faster. `MLKEM_PINNED=1/0` forces the
    choice.
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
        return False  # small batch: H2D is not the bottleneck; skip probing
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
    """Whether the concatenated length of `count` ciphertexts is exactly `count * n0`.

    This is O(1); the per-item `all(len(c) == n0 for c in lst)` costs 0.13 ms at
    B=4096 (6% of end-to-end), which is not worth it. The only way to misjudge
    is "ciphertexts of differing lengths mixed together whose total length
    happens to equal count*n0" — impossible for valid inputs of a fixed
    parameter set.
    """
    return n0 > 0 and len(buf) == count * n0


def _cpu_tensor(arr: np.ndarray) -> torch.Tensor:
    """numpy (possibly read-only) → CPU tensor, zero-copy.

    `np.frombuffer(b"...")` yields a read-only array, and `torch.from_numpy`
    emits a UserWarning ("numpy array is not writable"). That warning targets
    the case where the tensor may be written later, but here we only read it
    and copy it away immediately, so it is silenced. Suppressing the warning
    with `bytearray(...)` instead would add another 4.4 MB memcpy (+0.15 ms),
    which is not worth it.
    """
    if arr.flags.writeable:
        return torch.from_numpy(arr)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        return torch.from_numpy(arr)


def _to_device(arr: np.ndarray, dev: torch.device) -> torch.Tensor:
    """Move a uint8 array to the device, building the source tensor **zero-copy**.

    `np.frombuffer(b"...")` yields a read-only array, and `torch.from_numpy`
    emits a UserWarning ("numpy array is not writable"). That warning targets
    the case where the tensor may be written later, but here we only read it
    and copy it away immediately with `.to(dev)`, so it is silenced directly.
    Suppressing the warning with `bytearray(...)` instead would add another
    4.4 MB memcpy (about +0.4 ms at B=4096), which is not worth it.
    """
    return _cpu_tensor(arr).to(dev)


def _gemm_fp32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Generic fallback GEMM: an **exact** matrix multiplication with fp16 inputs
    and fp32 output.

    Why not just `torch.mm(a, b)`: the fp16 output ceiling is 65504, while the
    result peak here is about 1.8e5, which turns into inf (measured).
    `torch.mm(..., out_dtype=fp32)` is a torch-xpu private extension, absent
    from the public API, and `out=` requires a matching dtype.

    Upcasting both operands to fp32 is the most portable approach (every
    backend supports it); the cost is losing the fp16 tensor cores, 1.7–3.7x
    slower. **This path is taken only when `fused_*_gemm` returns None**; the
    normal path never uses it.
    """
    return torch.mm(a.to(torch.float32), b.to(torch.float32))


def pad_gpu(msgs: torch.Tensor, rate: int, domain: int) -> torch.Tensor:
    """pad10*1 + domain separation, all on device.

    msgs [n,L] uint8 -> [n, a multiple of rate].
    """
    n, L = msgs.shape
    total = ((L + 1 + rate - 1) // rate) * rate
    out = torch.zeros((n, total), dtype=torch.uint8, device=msgs.device)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, total - 1] = out[:, total - 1] | 0x80
    return out


def _pad2_gpu(a: torch.Tensor, b: torch.Tensor, n: int, rate: int, domain: int) -> torch.Tensor:
    """Single-launch equivalent of `pad_gpu(cat([a, b]))`.

    1-D `a`/`b` are treated as "broadcast to n rows" (equivalent to
    `.expand(n, L)`), 2-D ones row by row. These are exactly the shapes of J
    (z broadcast + ct row-wise) and G (m' row-wise + h_ek broadcast). The old
    path was three launches (`cat` + `zeros` + `copy_`), measured 192 µs for
    the J case (B=4096); here it is 28 µs. Falls back when a segment length is
    not a multiple of 8 (all three current parameter sets satisfy this).
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
    """Single-launch version of `pad_gpu(cat([prefix[row // mod], [row % mod + off]]))`.

    The PRF input is `r'` (32 B) followed by a 1-byte j, i.e. 33 B per item;
    `prefix` may be a column slice of a larger tensor (e.g. the last 32 B of
    the 64 B G output), as long as the last dimension is contiguous.
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
    """Padded [n, a multiple of rate] uint8 -> [n, outlen] uint8, all on device."""
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
    """[B, nb] uint8 -> [B, count] int32, reading d-bit fields in little-endian
    bit order.

    Reads bytes one by one in Triton (about 3 times faster than PyTorch's 3
    advanced-index gathers).
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
    """[count, B] integers -> [B, count*d/8] uint8 (ByteEncode bit packing)."""
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
    """CBD for eta=2: one byte holds exactly two coefficients, read once and
    written once.

    The original PyTorch implementation chained 6 intermediate tensors over
    [n,128] (&, >>, popcount...), and pure bandwidth alone ate 1 ms; here the
    whole thing is packed into one kernel.
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
    """[n, 64·eta] uint8 -> [n, 256] int32."""
    n = bufs.shape[0]
    assert bufs.shape[1] == 64 * eta

    if eta == 2:
        return ops.cbd(bufs, eta)

    # generic path (eta=3 etc.)
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


# ---------------- main class ----------------


class Decapsulator:
    """Load one ML-KEM private key onto the device, then decapsulate any number
    of ciphertexts in batches.

    Args:
        ps: parameter set name, ``"ML-KEM-512"`` / ``"ML-KEM-768"`` /
            ``"ML-KEM-1024"``.
        dk: private key byte string (length ``768k+96``).
        device: ``None`` selects the first available accelerator automatically;
            a value may also be given explicitly, e.g. ``"cuda"`` / ``"xpu"`` /
            ``"cuda:1"``. The ``MLKEM_DEVICE`` environment variable has lower
            priority than an explicit argument and higher than auto-detection.
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

    # ---- hash batching ----
    def _G(self, m_bytes: torch.Tensor, B: int) -> torch.Tensor:
        # m' || h_ek, padded to 1 block of 72 B; one launch (previously three: cat+zeros+copy)
        return sponge_gpu(
            _pad2_gpu(m_bytes, self.h_ek, B, RATE_SHA3_512, DOMAIN_SHA3), RATE_SHA3_512, 64
        )

    def _J(self, cts: torch.Tensor, B: int) -> torch.Tensor:
        # z || ct, padded to 9 blocks of 136 B; one launch (previously 192 µs, now 28 µs)
        return sponge_gpu(_pad2_gpu(self.z, cts, B, RATE_SHAKE, DOMAIN_SHAKE), RATE_SHAKE, 32)

    def _prf_bufs(self, r_prime: torch.Tensor, B: int):
        """PRF: r' → two sponge outputs (CBD not applied yet), for the fused CBD→Y/E path."""
        p, k = self.p, self.p.k
        # r' || [j] (33 B) → padded to 1 block of 136 B. r_prime is usually the
        # last 32 B of the G output (a slice of [B,64], contiguous in the last
        # dimension), so the kernel can read it in place with a stride, no copy.
        y_pad = _pad_prf_gpu(r_prime, B * k, RATE_SHAKE, DOMAIN_SHAKE, mod=k, off=0)
        e_pad = _pad_prf_gpu(r_prime, B * (k + 1), RATE_SHAKE, DOMAIN_SHAKE, mod=k + 1, off=k)
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

    # ---- main flow ----
    def decaps(self, cts) -> np.ndarray:
        """Batched decapsulation.

        `cts` accepts four forms, and the **host-side cost** of converting them
        into a device tensor differs a lot (measured at B=4096, see
        benchmarks/bench_host.py):

        ============================================  ===========
        form                                              host side
        ============================================  ===========
        `torch.Tensor` (already on device)                ~0 ms
        `bytes` / `bytearray` / `memoryview` (one block)  ~0.11 ms
        `np.ndarray` [B, ct_size]                         ~0.14 ms
        `list[bytes]` (one ciphertext each)               ~0.36 ms
        ============================================  ===========

        The difference is all in `b"".join` and the "all items have the same
        length" check. When a server receives ciphertexts from the network and
        can read them straight into one contiguous buffer (`socket.recv_into`
        or pinned memory), it takes the first two rows; with a `list[bytes]`
        parsed elsewhere, it pays the 0.36 ms.
        """
        n0_expected = self.p.ct_size
        if isinstance(cts, torch.Tensor):
            self._check_shape(cts.shape[-2:] if cts.dim() > 1 else cts.shape, n0_expected)
            ct = cts.reshape(-1, n0_expected)
            if ct.dtype != torch.uint8:
                raise TypeError(f"cts must be uint8, got {ct.dtype}")
            if ct.device != self.device:
                ct = ct.to(self.device)
            return self.decaps_torch(ct, ct.shape[0]).cpu().numpy()

        arr = self._to_numpy2d(cts, n0_expected)
        B = arr.shape[0]

        # the only H2D (via pinned staging, see _h2d).
        ct = self._h2d(arr)

        # the only D2H.
        return self.decaps_torch(ct, B).cpu().numpy()

    def _h2d(self, arr: np.ndarray) -> torch.Tensor:
        """Single host → device copy, using **pinned staging** where possible.

        Why it is worth it: on this machine pageable → device is only 12 GB/s
        (the driver goes through a bounce buffer), while pinned → device pure
        DMA is 78 GB/s. The extra "numpy → pinned" memcpy takes only 0.035 ms
        (4.46 MB @127 GB/s), so it nets:

            pageable direct      join 0.157 + 0.362 = 0.519 ms
            pinned staging       join 0.157 + 0.035 + 0.057 = 0.249 ms

        (measured at B=4096, see benchmarks/bench_host.py.)

        The staging buffer is cached and reused per (B, ct_size); set
        `MLKEM_PINNED=0` to disable it. The device-side buffer is reused
        too — `decaps_torch` only reads it and has already synchronized before
        returning, so overwriting it on the next call is safe.
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
            raise ValueError(f"each ciphertext must be {n0} bytes, got {tuple(shape)}")

    @staticmethod
    def _to_numpy2d(cts, n0: int) -> np.ndarray:
        """Normalize the various input forms into a [B, n0] uint8 CPU array
        (zero or single copy where possible)."""
        if isinstance(cts, np.ndarray):
            if cts.dtype != np.uint8:
                cts = cts.astype(np.uint8)
            arr = np.ascontiguousarray(cts)
            if arr.ndim == 1:
                arr = arr[None, :]
            if arr.ndim != 2 or (n0 > 0 and arr.shape[1] != n0):
                raise ValueError(f"expected a [B, {n0}] uint8 array, got {arr.shape}")
            return arr

        if isinstance(cts, (bytes, bytearray, memoryview)):
            n = len(cts)
            if n0 <= 0 or n % n0:
                raise ValueError(f"input length {n} is not a multiple of ciphertext length {n0}")
            return np.frombuffer(cts, np.uint8).reshape(n // n0, n0)

        lst = list(cts)
        if not lst:
            return np.empty((0, n0), np.uint8)
        try:
            buf = b"".join(lst)  # accepts bytes-like only, otherwise TypeError
        except TypeError:
            buf = None
        # O(1) check: if the total length is exactly n0 * len(lst), treat the
        # items as equal-length. The per-item `all(len(c) == n0 for c in lst)`
        # previously wasted 0.13 ms at B=4096 (6% of end-to-end). The cost: if
        # the caller mixes ciphertexts of differing lengths whose total length
        # happens to equal n0*len, this is misjudged — impossible for valid
        # inputs of a fixed parameter set.
        if buf is not None and total_is_uniform(buf, len(lst), n0):
            return np.frombuffer(buf, np.uint8).reshape(len(lst), n0)
        # slow path: elements are not bytes-like, or the lengths really differ
        return np.stack([np.frombuffer(bytes(c), np.uint8) for c in lst])

    def decaps_torch(self, ct: torch.Tensor, B: int) -> torch.Tensor:
        """Pure device-side flow: input ct[B, ct_size] uint8, output K[B, 32] uint8."""
        p, k, dev = self.p, self.p.k, self.device

        # 1. fused: unpack + decompress (+center) + transpose  →  U[256k,B] fp16, V[256,B] int16
        #    also emits int16 reference values c1r/c2r for the final comparison
        #    (avoids a second unpacking pass)
        KK = 256 * k
        U, c1r = ops.pre(ct, B, p.ct_size, 0, KK, p.du, 1, 16, 256)
        V, c2r = ops.pre(ct, B, p.ct_size, 32 * p.du * k, 256, p.dv, 0, 16, 256)

        # 2. w = v − W_dec @ u, then mod q → compress(1 bit) → pack into m_bytes.
        #    The GEMM and the whole epilogue are fused into one kernel (acc never
        #    hits VRAM, saving the 4.2 MB write + read-back). Measured 1.36x
        #    faster than "mm + compress_pack".
        m_bytes = fused_dec_gemm(self.W_dec, U, V, B)
        if m_bytes is None:  # shape/backend does not meet the constraints
            acc = _gemm_fp32(self.W_dec, U)
            if glue.can_compress_pack(V, acc, B):
                m_bytes = ops.compress_pack(acc, V, B)  # [B,32] uint8
            else:
                w = (V.to(torch.int32) - acc.to(torch.int32)) % Q
                m_bytes = pack_bits_gpu(compress_gpu(w, 1), 1)  # [B,32] uint8

        # 3. G / J
        gr = self._G(m_bytes, B)
        K_prime = gr[:, :32]
        r_prime = gr[:, 32:]
        K_bar = self._J(ct, B)

        # 4. PRF + CBD (+ transpose). With eta=2, CBD writes straight into Y/E,
        #    with no y/e intermediate tensors.
        y_buf, e_buf = self._prf_bufs(r_prime, B)
        YE = None
        if glue.can_cbd_y(y_buf, B, k, p.eta1) and glue.can_cbd_e(e_buf, m_bytes, B, k, p.eta2):
            YE = (ops.cbd_y(y_buf, B, k), ops.cbd_e(e_buf, m_bytes, B, k))
        if YE is None:
            y = cbd_gpu(y_buf, p.eta1).reshape(B, k, 256)
            e = cbd_gpu(e_buf, p.eta2).reshape(B, k + 1, 256)

        # 5. re-encryption: y / e / m_bytes → Y [256k,B] fp16 + E [256(k+1),B] int16, one kernel
        #    Y must be **contiguous** (a fused_enc_gemm constraint). The old
        #    `permute(1,2,0).reshape(...).to(fp16)` was actually a transposed view
        #    with stride=(1,256k): same values but non-contiguous, which made the
        #    fused GEMM silently return False and sent the whole pipeline down the
        #    mm + cmp fallback.
        if YE is not None:
            Y, E = YE
        elif glue.can_build_ye(y, e, m_bytes, B, k):
            Y, E = ops.build_ye(y, e, m_bytes, B, k)
        else:
            Y = y.permute(1, 2, 0).reshape(k * 256, B).to(torch.float16).contiguous()
            # int16 is enough: E1 ∈ [-2,2], E2+1665*m_bits ∈ [-2, 1667].
            E1 = e[:, :k].permute(1, 2, 0).reshape(k * 256, B).to(torch.int16)
            E2 = (e[:, k].T + 1665 * glue.unpack_mbits(m_bytes, B)).to(torch.int16)
            E = torch.cat([E1, E2], dim=0)  # [256(k+1),B]

        # 6. fused: GEMM + mod q + compress + compare (accumulates the equality count per column)
        #    fused_enc_gemm returning False means the shape/dtype does not
        #    satisfy its constraints and no output was written; in that case fall
        #    back to the two steps mm + cmp.
        cnt = torch.zeros(B, dtype=torch.int32, device=dev)
        if not fused_enc_gemm(self.W_enc, Y, E, (c1r, c2r), cnt, B, p.du, p.dv):
            acc2 = _gemm_fp32(self.W_enc, Y)
            BM, BN = 128, 64
            ops.cmp(acc2, E, c1r, cnt, B, 0, KK, p.du, BM, BN)
            ops.cmp(acc2, E, c2r, cnt, B, KK, 256, p.dv, BM, BN)
        eq = cnt == (KK + 256)

        # 7. constant-time selection
        return torch.where(eq[:, None], K_prime, K_bar)  # [B,32] uint8


def decaps_batch(ps: str, dk: bytes, cts, device=None) -> np.ndarray:
    """Decapsulate a batch of ciphertexts in one call (the private key is
    re-prepared every time; do not use it for small batches)."""
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
