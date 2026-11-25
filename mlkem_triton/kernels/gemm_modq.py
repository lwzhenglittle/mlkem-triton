"""Phase 3.1: Triton GEMM + fused epilogue.

    out = (W @ X + E) mod q      (MODE=0, re-encryption)
    out = (E - W @ X) mod q      (MODE=1, decryption)

- W [M,K] fp16, X [K,B] fp16, E [M,B] int16, out [M,B] int16 ∈ [0, q)
- FP32 accumulation: verified **element-wise exact** over this project's data range
  (see tests/test_dot_exact.py)
- Batch size B need not be a multiple of 16: the tile has a fixed BN>=16, and a mask
  guards out-of-range columns
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

Q = 3329
INV_Q = 1.0 / Q


@triton.jit
def _gemm_modq_kernel(
    W,
    X,
    E,
    OUT,
    M,
    N,
    K,
    sw_m,
    sw_k,
    sx_k,
    sx_n,
    se_m,
    se_n,
    so_m,
    so_n,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc = tl.zeros((BM, BN), dtype=tl.float32)
    w_ptrs = W + offs_m[:, None] * sw_m + offs_k[None, :] * sw_k
    x_ptrs = X + offs_k[:, None] * sx_k + offs_n[None, :] * sx_n

    for k in range(0, K, BK):
        w = tl.load(
            w_ptrs,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K),
            other=0.0,
        )
        x = tl.load(
            x_ptrs,
            mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N),
            other=0.0,
        )
        acc = tl.dot(w, x, acc)
        w_ptrs += BK * sw_k
        x_ptrs += BK * sx_k

    # ---- epilogue ----
    e_ptrs = E + offs_m[:, None] * se_m + offs_n[None, :] * se_n
    e = tl.load(e_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0)

    acc_i = acc.to(tl.int32)
    if MODE == 0:
        r = acc_i + e.to(tl.int32)
    else:
        r = e.to(tl.int32) - acc_i

    # Modulo: estimate the quotient in float, then two corrections. For |r| < 2^24
    # fp32 is exact, so floor is off by at most 1.
    q_est = tl.math.floor(r.to(tl.float32) * (1.0 / 3329.0)).to(tl.int32)
    r = r - q_est * 3329
    r = tl.where(r < 0, r + 3329, r)
    r = tl.where(r >= 3329, r - 3329, r)

    o_ptrs = OUT + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    tl.store(
        o_ptrs,
        r.to(tl.int16),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def _pick_tile(M: int, B: int, BM=None, BN=None, BK=None):
    BM = BM or (64 if M % 64 == 0 else 32)
    BN = BN or (64 if B >= 64 else 16)
    BK = BK or 64
    return BM, BN, BK


def _launch(W, X, E, mode: str, tile=None) -> torch.Tensor:
    M, K = W.shape
    _, B = X.shape
    out = torch.empty((M, B), device=X.device, dtype=torch.int16)
    BM, BN, BK = _pick_tile(M, B, *(tile or (None, None, None)))
    grid = (triton.cdiv(M, BM), triton.cdiv(B, BN))
    _gemm_modq_kernel[grid](
        W,
        X,
        E,
        out,
        M,
        B,
        K,
        W.stride(0),
        W.stride(1),
        X.stride(0),
        X.stride(1),
        E.stride(0),
        E.stride(1),
        out.stride(0),
        out.stride(1),
        MODE=0 if mode == "enc" else 1,
        BM=BM,
        BN=BN,
        BK=BK,
    )
    return out


_MIN_B = 16


def _pad_if_degenerate(W, X, E, mode: str, launcher) -> torch.Tensor:
    """At B == 1 the strides of `[K,1]` degenerate to (1,1) and Triton's block
    loads/stores compute the wrong addresses.

    So we temporarily pad to 16 columns and slice back to 1 column afterwards.
    For B >= 2 the strides are normal, so the direct path is used.
    """
    M, K = W.shape
    B = X.shape[1]
    if B != 1:
        return launcher(W, X, E, mode)
    Bp = _MIN_B
    Xp = torch.zeros((K, Bp), dtype=X.dtype, device=X.device)
    Xp[:, :1] = X
    Ep = torch.zeros((M, Bp), dtype=E.dtype, device=E.device)
    Ep[:, :1] = E
    out = launcher(W, Xp, Ep, mode)
    return out[:, :1].contiguous()


def gemm_modq(
    W: torch.Tensor, X: torch.Tensor, E: torch.Tensor, mode: str = "enc", tile=None, num_warps=None
) -> torch.Tensor:
    """W [M,K] fp16, X [K,B] fp16, E [M,B] int16 -> [M,B] int16."""
    assert W.dtype == torch.float16 and X.dtype == torch.float16
    assert E.dtype == torch.int16
    M, K = W.shape
    K2, B = X.shape
    assert K2 == K and E.shape == (M, B)
    if tile is not None or num_warps is not None:
        # Benchmarking: pass tile/num_warps explicitly
        if X.shape[1] == 1:
            return _pad_if_degenerate(W, X, E, mode, lambda w, x, e, m: _launch(w, x, e, m, tile))
        out = torch.empty((W.shape[0], X.shape[1]), device=X.device, dtype=torch.int16)
        M, K = W.shape
        B = X.shape[1]
        BM, BN, BK = _pick_tile(M, B, *(tile or (None, None, None)))
        grid = (triton.cdiv(M, BM), triton.cdiv(B, BN))
        kw = {} if num_warps is None else {"num_warps": num_warps}
        _gemm_modq_kernel[grid](
            W,
            X,
            E,
            out,
            M,
            B,
            K,
            W.stride(0),
            W.stride(1),
            X.stride(0),
            X.stride(1),
            E.stride(0),
            E.stride(1),
            out.stride(0),
            out.stride(1),
            MODE=0 if mode == "enc" else 1,
            BM=BM,
            BN=BN,
            BK=BK,
            **kw,
        )
        return out
    return _pad_if_degenerate(W, X, E, mode, _launch)


# ---------------- INT8 limb version (reference/fallback) ----------------


@triton.jit
def _gemm_modq_i8_kernel(
    HI,
    LO,
    X,
    E,
    OUT,
    M,
    N,
    K,
    sh_m,
    sh_k,
    sl_m,
    sl_k,
    sx_k,
    sx_n,
    se_m,
    se_n,
    so_m,
    so_n,
    MODE: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BM + tl.arange(0, BM)
    offs_n = pid_n * BN + tl.arange(0, BN)
    offs_k = tl.arange(0, BK)

    acc_hi = tl.zeros((BM, BN), dtype=tl.int32)
    acc_lo = tl.zeros((BM, BN), dtype=tl.int32)
    hi_ptrs = HI + offs_m[:, None] * sh_m + offs_k[None, :] * sh_k
    lo_ptrs = LO + offs_m[:, None] * sl_m + offs_k[None, :] * sl_k
    x_ptrs = X + offs_k[:, None] * sx_k + offs_n[None, :] * sx_n

    for k in range(0, K, BK):
        mh = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mx = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        h = tl.load(hi_ptrs, mask=mh, other=0)
        lo_v = tl.load(lo_ptrs, mask=mh, other=0)
        x = tl.load(x_ptrs, mask=mx, other=0)
        acc_hi = tl.dot(h, x, acc_hi)
        acc_lo = tl.dot(lo_v, x, acc_lo)
        hi_ptrs += BK * sh_k
        lo_ptrs += BK * sl_k
        x_ptrs += BK * sx_k

    acc = 64 * acc_hi + acc_lo
    e_ptrs = E + offs_m[:, None] * se_m + offs_n[None, :] * se_n
    e = tl.load(e_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0)
    if MODE == 0:
        r = acc + e.to(tl.int32)
    else:
        r = e.to(tl.int32) - acc

    q_est = tl.math.floor(r.to(tl.float32) * (1.0 / 3329.0)).to(tl.int32)
    r = r - q_est * 3329
    r = tl.where(r < 0, r + 3329, r)
    r = tl.where(r >= 3329, r - 3329, r)

    o_ptrs = OUT + offs_m[:, None] * so_m + offs_n[None, :] * so_n
    tl.store(o_ptrs, r.to(tl.int16), mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _launch_i8(hi, lo, X, E, mode: str) -> torch.Tensor:
    M, K = hi.shape
    _, B = X.shape
    out = torch.empty((M, B), device=X.device, dtype=torch.int16)
    BM, BN, BK = _pick_tile(M, B)
    grid = (triton.cdiv(M, BM), triton.cdiv(B, BN))
    _gemm_modq_i8_kernel[grid](
        hi,
        lo,
        X,
        E,
        out,
        M,
        B,
        K,
        hi.stride(0),
        hi.stride(1),
        lo.stride(0),
        lo.stride(1),
        X.stride(0),
        X.stride(1),
        E.stride(0),
        E.stride(1),
        out.stride(0),
        out.stride(1),
        MODE=0 if mode == "enc" else 1,
        BM=BM,
        BN=BN,
        BK=BK,
    )
    return out


def gemm_modq_int8(
    hi: torch.Tensor, lo: torch.Tensor, X: torch.Tensor, E: torch.Tensor, mode: str = "enc"
) -> torch.Tensor:
    """W = 64*hi + lo, composed from two int8 dots."""
    M, K = hi.shape
    K2, B = X.shape
    assert X.dtype == torch.int8 and hi.dtype == torch.int8 and lo.dtype == torch.int8
    assert lo.shape == hi.shape and E.shape == (M, B) and K2 == K

    if B != 1:
        return _launch_i8(hi, lo, X, E, mode)

    Bp = _MIN_B
    Xp = torch.zeros((K, Bp), dtype=X.dtype, device=X.device)
    Xp[:, :1] = X
    Ep = torch.zeros((M, Bp), dtype=E.dtype, device=E.device)
    Ep[:, :1] = E
    out = _launch_i8(hi, lo, Xp, Ep, mode)
    return out[:, :1].contiguous()
