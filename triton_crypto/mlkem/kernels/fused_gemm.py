"""Fused kernels for the two GEMM passes: fp32 accumulation + in-place epilogue.

Both passes are "a fixed weight matrix × per-batch variables", so they share one
skeleton and only the epilogue changes:

**ENC (re-encryption)** `fused_enc_gemm`

    Replaces these three steps:
        acc2 = torch.mm(W_enc, Y, out_dtype=fp32)     # [1024, B] fp32 = 16.8 MB
        cmp(acc2, E, c1r, ...)                        # du segment
        cmp(acc2, E, c2r, ...)                        # dv segment
    After fusion acc2 never lands in GPU memory; only a [M/BM, B] int32 partial
    buffer is written.

**DEC (decryption)** `fused_dec_gemm`

    Replaces these two steps:
        acc = torch.mm(W_dec, U, out_dtype=fp32)      # [256, B] fp32 = 4.2 MB
        m_bytes = compress_pack(V, acc, B)            # mod q → compress 1 bit → pack
    After fusion acc does not land either. Measured (B=4096, Intel Xe3) it is **1.36×**
    faster than the two steps (0.213 → 0.157 ms), with byte-identical output.

Why both epilogues need `x %= q` and an add-back for negatives
--------------------------------------------------------------
Triton / C `int % int` truncates toward zero (`-7 % 3 == -1`), but we want the
canonical representative in `[0, q)`, so we uniformly do
`x = x % q; if x < 0: x += q`.

About `out_dtype=torch.float32`
-------------------------------
`torch.mm` has no such public argument (that is torch-xpu's private
`aten::mm.dtype`). Also, an fp16 output is unusable here: the `|sum|` peak is about
1.8e5 > the fp16 maximum of 65504 → inf. So these two passes must go through our own
Triton kernels.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...devices import gemm_cfg

Q = 3329
QL = tl.constexpr(Q)
BLOCK_OUT = 128  # column block of the partial-combining kernel


# --------------------------------------------------------------- shared GEMM skeleton
@triton.jit
def _gemm_tile(
    W,
    Y,
    pid_m,
    pid_n,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """One (BM, BN) tile of fp16×fp16 → fp32 accumulation.

    M is only used to compute the row offsets, so it does not have to be passed in.
    """
    offm = pid_m * BM + tl.arange(0, BM)
    offn = pid_n * BN + tl.arange(0, BN)
    offk = tl.arange(0, BK)

    a_ptrs = W + offm[:, None] * K + offk[None, :]
    b_ptrs = Y + offk[:, None] * N + offn[None, :]
    mask_n = offn < N
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BK)):
        # B is padded to BN by the caller only sometimes (it is the batch size), so the
        # last column tile must be masked; unmasked loads read into the next row.
        acc = tl.dot(
            tl.load(a_ptrs),
            tl.load(b_ptrs, mask=mask_n[None, :], other=0.0),
            acc,
            out_dtype=tl.float32,
        )
        a_ptrs += BK
        b_ptrs += BK * N
    return acc


@triton.jit
def _tile_id(
    pid, M: tl.constexpr, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, GM: tl.constexpr
):
    """L2-friendly domino-style grouping (grouped ordering), returns (pid_m, pid_n)."""
    npm, npn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    ng = GM * npn
    gid = pid // ng
    gsz = min(npm - gid * GM, GM)
    pid_m = gid * GM + ((pid % ng) % gsz)
    pid_n = (pid % ng) // gsz
    return pid_m, pid_n


# --------------------------------------------------------------- ENC epilogue
@triton.jit
def _epi_enc(
    acc,
    E,
    REF,
    OUT,
    offm,
    rrow0,
    cbase,
    D: tl.constexpr,
    BN: tl.constexpr,
    N: tl.constexpr,
    MASK_N: tl.constexpr,
    BM: tl.constexpr,
    DSTOFF,
):
    """mod q → compress → compare with the reference → column reduce → write partial."""
    cols = cbase + tl.arange(0, BN)
    if MASK_N:
        m = cols[None, :] < N
        e = tl.load(E + offm[:, None] * N + cols[None, :], mask=m, other=0)
    else:
        e = tl.load(E + offm[:, None] * N + cols[None, :])

    x = acc.to(tl.int32) + e.to(tl.int32)
    x = x % QL
    x = tl.where(x < 0, x + QL, x)  # Python-semantics mod
    comp = (((x << D) + 1664) // QL) & ((1 << D) - 1)  # x >= 0, so // truncates toward zero

    if MASK_N:
        r = tl.load(REF + (offm - rrow0)[:, None] * N + cols[None, :], mask=m, other=-1)
    else:
        r = tl.load(REF + (offm - rrow0)[:, None] * N + cols[None, :])

    s = tl.sum((comp == r).to(tl.int32), axis=0)  # [BN] int32
    if MASK_N:
        tl.store(OUT + DSTOFF + cols, s, mask=cols < N)
    else:
        tl.store(OUT + DSTOFF + cols, s)


@triton.jit
def fused_enc_kernel(
    W,
    Y,
    E,
    ref1,
    ref2,
    parts,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    PIVOT: tl.constexpr,
    DU: tl.constexpr,
    DV: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _tile_id(pid, M, N, BM, BN, GM)
    acc = _gemm_tile(W, Y, pid_m, pid_n, K, N, BM, BN, BK)
    offm = pid_m * BM + tl.arange(0, BM)
    if pid_m * BM < PIVOT:
        _epi_enc(acc, E, ref1, parts, offm, 0, pid_n * BN, DU, BN, N, MASK_N, BM, pid_m * N)
    else:
        _epi_enc(acc, E, ref2, parts, offm, PIVOT, pid_n * BN, DV, BN, N, MASK_N, BM, pid_m * N)


@triton.jit
def _combine_parts(parts, cnt, NPM: tl.constexpr, N: tl.constexpr, BN: tl.constexpr):
    """[NPM, N] row-block partial sums → atomically add into cnt[N]. Only N*NPM integers."""
    cols = tl.program_id(0) * BN + tl.arange(0, BN)
    m = cols < N
    s = tl.zeros((BN,), dtype=tl.int32)
    for i in tl.static_range(NPM):
        s += tl.load(parts + i * N + cols, mask=m, other=0)
    tl.atomic_add(cnt + cols, s, mask=m)


# --------------------------------------------------------------- DEC epilogue
@triton.jit
def _epi_dec(
    acc,
    V,
    MB,
    offm,
    cbase,
    ccol0,
    NB: tl.constexpr,
    BN: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    """`(V - acc) mod q` → compress to 1 bit → pack 8 rows into one byte → write m_bytes.

    Bit t of m_bytes[b][j] comes from row `8j+t` — bit-for-bit identical to the
    semantics of `glue.compress_pack_kernel` (which loops over 8 rows; here a single
    reshape reduction does it).

    `tl.reshape([BM, BN] -> [BM//8, 8, BN])` is a row-major split, so element (i, j)
    lands at (i//8, i%8, j), which is exactly the grouping we want.
    """
    cols = cbase + tl.arange(0, BN)
    if MASK_N:
        m = cols[None, :] < N
        v = tl.load(V + offm[:, None] * N + cols[None, :], mask=m, other=0)
    else:
        v = tl.load(V + offm[:, None] * N + cols[None, :])

    x = v.to(tl.int32) - acc.to(tl.int32)
    x = x % QL
    x = tl.where(x < 0, x + QL, x)
    comp = (((x << 1) + 1664) // QL) & 1

    r = tl.reshape(comp, (BM // 8, 8, BN))
    w = (1 << tl.arange(0, 8))[None, :, None]
    byte = tl.sum(r * w, axis=1)  # [BM//8, BN]
    j = ccol0 + tl.arange(0, BM // 8)
    tl.store(MB + cols[None, :] * NB + j[:, None], byte.to(tl.uint8), mask=cols[None, :] < N)


@triton.jit
def fused_dec_kernel(
    W,
    U,
    V,
    MB,
    M: tl.constexpr,
    K: tl.constexpr,
    N: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    GM: tl.constexpr,
    MASK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    pid_m, pid_n = _tile_id(pid, M, N, BM, BN, GM)
    acc = _gemm_tile(W, U, pid_m, pid_n, K, N, BM, BN, BK)
    offm = pid_m * BM + tl.arange(0, BM)
    _epi_dec(acc, V, MB, offm, pid_n * BN, pid_m * (BM // 8), M // 8, BN, N, BM, MASK_N)


# --------------------------------------------------------------- config selection
def _pick_cfg(M: int, dev) -> dict:
    """Pick a tile.

    Start from the backend's baseline config (`devices.gemm_cfg`), then adjust for
    problem size: when M is small (the DEC pass has M=256; the ENC pass always has
    M = 256*(k+1) >= 768, so this only ever fires for DEC) there are too few
    128-wide CTAs, so switching to BN=64 measured 8~17% faster
    (B=512: 48.2 vs 57.1 µs; B=2048: 116 vs 133 µs).
    """
    cfg = gemm_cfg(dev)
    if 2 * cfg["BM"] >= M:
        cfg["BN"] = min(cfg["BN"], 64)
        cfg["GM"] = min(cfg["GM"], 4)
    return cfg


# --------------------------------------------------------------- ENC entry point
def fused_enc_gemm(W, Y, E, ref, cnt, B: int, du: int, dv: int, pivot: int | None = None) -> bool:
    """ENC re-encryption GEMM + mod q + compress + compare + column sum, one kernel.

    Parameters
    ----------
    W    : [M, K] fp16, re-encryption matrix (already flattened row by row)
    Y    : [K, B] fp16, the sampled y
    E    : [M, B] int16, E1 || E2 (E2 already has 1665*m_bits added)
    ref  : (c1r, c2r) two int16 tensors, or a single [M, B] int16
    cnt  : [B] int32, **accumulated into** (zero it before the call)
    du, dv : the two compression widths (ENC: du covers the first pivot rows, dv the
             remaining 256 rows)
    pivot  : number of rows in the du segment, defaults to M - 256

    Returns
    -------
    True if it ran; False if the shapes/dtype do not satisfy the constraints, in
    which case **no output was written** and the caller should fall back to
    "generic GEMM + element-wise compare".
    """
    if W.dim() != 2 or Y.dim() != 2 or E.dim() != 2:
        return False
    M, K = W.shape
    if M <= 0 or K <= 0 or B <= 0:
        return False
    if Y.shape[0] != K or Y.shape[1] != B:
        return False
    if E.shape != (M, B) or cnt.shape != (B,) or cnt.dtype != torch.int32:
        return False
    if W.dtype != torch.float16 or Y.dtype != torch.float16:
        return False
    if E.dtype != torch.int16 or not (
        W.is_contiguous() and Y.is_contiguous() and E.is_contiguous() and cnt.is_contiguous()
    ):
        return False
    if pivot is None:
        pivot = M - 256
    if not (0 <= pivot <= M):
        return False
    if isinstance(ref, (tuple, list)):
        if len(ref) != 2:
            return False
        ref1, ref2 = ref
        if ref1.dtype != torch.int16 or ref2.dtype != torch.int16:
            return False
        if not (ref1.is_contiguous() and ref2.is_contiguous()):
            return False
        if tuple(ref1.shape) != (pivot, B) or tuple(ref2.shape) != (M - pivot, B):
            return False
    else:
        if ref.dtype != torch.int16 or tuple(ref.shape) != (M, B) or not ref.is_contiguous():
            return False
        ref1, ref2 = ref[:pivot], ref[pivot:]

    cfg = _pick_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM or pivot % BM:
        return False
    if not (1 <= du <= 15 and 1 <= dv <= 15):
        return False
    if not (W.device == E.device == Y.device == cnt.device):
        return False

    npm = M // BM
    grid = (npm * triton.cdiv(B, BN),)
    mask_n = (B % BN) != 0
    parts = torch.empty((npm, B), dtype=torch.int32, device=cnt.device)

    fused_enc_kernel[grid](
        W,
        Y,
        E,
        ref1,
        ref2,
        parts,
        M,
        K,
        B,
        pivot,
        du,
        dv,
        BM,
        BN,
        BK,
        cfg["GM"],
        mask_n,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    _combine_parts[(triton.cdiv(B, BLOCK_OUT),)](parts, cnt, npm, B, BLOCK_OUT, num_warps=1)
    return True


# --------------------------------------------------------------- DEC entry point
def fused_dec_gemm(W, U, V, B: int):
    """DEC decryption GEMM + `(V - W@U) mod q` + compress(1 bit) + packing.

    Args:
        W: [M, K] fp16, decryption matrix (`W_dec` produced by `keyprep`)
        U: [K, B] fp16, decompressed u
        V: [M, B] int16, decompressed v

    Returns:
        m_bytes [B, M/8] uint8, or **None** (shapes/dtype/layout do not satisfy the
        constraints, in which case no output was written and the caller should fall
        back).
    """
    if W.dim() != 2 or U.dim() != 2 or V.dim() != 2:
        return None
    M, K = W.shape
    if M <= 0 or K <= 0 or B <= 0 or M % 8:
        return None
    if U.shape != (K, B) or V.shape != (M, B):
        return None
    if W.dtype != torch.float16 or U.dtype != torch.float16 or V.dtype != torch.int16:
        return None
    if not (W.is_contiguous() and U.is_contiguous() and V.is_contiguous()):
        return None
    if not (W.device == U.device == V.device):
        return None

    cfg = _pick_cfg(M, W.device)
    BM, BN, BK = cfg["BM"], cfg["BN"], cfg["BK"]
    if K % BK or M % BM:
        return None

    nb = M // 8
    out = torch.empty((B, nb), dtype=torch.uint8, device=V.device)
    grid = (triton.cdiv(M, BM) * triton.cdiv(B, BN),)
    fused_dec_kernel[grid](
        W,
        U,
        V,
        out,
        M,
        K,
        B,
        BM,
        BN,
        BK,
        cfg["GM"],
        (B % BN) != 0,
        num_warps=cfg["num_warps"],
        num_stages=cfg["num_stages"],
    )
    return out


__all__ = [
    "fused_dec_gemm",
    "fused_dec_kernel",
    "fused_enc_gemm",
    "fused_enc_kernel",
]
