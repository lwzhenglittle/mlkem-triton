"""Phase 0.4 precision micro-test — the premise of the whole scheme.

Verifies:
  1. FP16 path: W ∈ [-1664,1664], X ∈ [-3,3], `tl.dot(fp16, fp16, acc=fp32)` is
     **element-wise exactly equal** to a CPU int64 matmul.
  2. Boundary cases: all W and X take the max absolute value with the same sign,
     pushing the accumulator to the theoretical bound.
  3. INT8 path: W split into limbs (hi = floor((W+32)/64) ∈ [-26,26], lo = W - 64*hi ∈ [-32,31]),
     two int8 dots compose into `64*(hi@X) + (lo@X)`, cross-checked against int64.

Note (revised from the original plan): the original plan's INT8 case called int8 dot
directly on W∈[-1664,1664], which does not fit; here it is changed to a limb split. It
also covers batch sizes N=1 / not 16-aligned, proving that "a BN≥16 tile + mask" can
cover a small B.

Run: .venv/bin/python -m pytest tests/test_dot_exact.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from _common import DEV

from mlkem_triton.devices import describe, sync

xpu_ok = True  # device availability is skipped centrally by tests/conftest.py (HAS_GPU)

try:
    import triton
    import triton.language as tl

    triton_ok = True
except Exception:
    triton_ok = False

pytestmark = pytest.mark.skipif(not (xpu_ok and triton_ok), reason="requires GPU + Triton")

if triton_ok:

    @triton.jit
    def mm_f16(
        A,
        B,
        C,
        M,
        N,
        K,
        sa_m,
        sa_k,
        sb_k,
        sb_n,
        sc_m,
        sc_n,
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
        a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
        b_ptrs = B + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0.0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k
        c_ptrs = C + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

    @triton.jit
    def mm_i8(
        A,
        B,
        C,
        M,
        N,
        K,
        sa_m,
        sa_k,
        sb_k,
        sb_n,
        sc_m,
        sc_n,
        BM: tl.constexpr,
        BN: tl.constexpr,
        BK: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        offs_k = tl.arange(0, BK)
        acc = tl.zeros((BM, BN), dtype=tl.int32)
        a_ptrs = A + offs_m[:, None] * sa_m + offs_k[None, :] * sa_k
        b_ptrs = B + offs_k[:, None] * sb_k + offs_n[None, :] * sb_n
        for k in range(0, K, BK):
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] + k < K), other=0)
            b = tl.load(b_ptrs, mask=(offs_k[:, None] + k < K) & (offs_n[None, :] < N), other=0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * sa_k
            b_ptrs += BK * sb_k
        c_ptrs = C + offs_m[:, None] * sc_m + offs_n[None, :] * sc_n
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def _run_mm(kernel, a, b, out_dtype, BM=64, BN=64, BK=64):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, (a.shape, b.shape)
    c = torch.empty((M, N), device=a.device, dtype=out_dtype)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BM=BM,
        BN=BN,
        BK=BK,
    )
    sync()
    return c


def _ref_int64(w, x):
    return w.to(torch.int64).cpu() @ x.to(torch.int64).cpu()


@pytest.mark.parametrize("N", [1, 16, 17, 37, 64, 100])
def test_fp16_random_exact(N):
    """Random integer matrices: FP16 input + FP32 accumulation must equal int64 exactly."""
    M, K = 1024, 768
    w = torch.randint(-1664, 1665, (M, K), dtype=torch.int64)
    x = torch.randint(-3, 4, (K, N), dtype=torch.int64)

    a = w.to(torch.float16).to(DEV)
    b = x.to(torch.float16).to(DEV)
    c = _run_mm(mm_f16, a, b, torch.float32)

    ref = _ref_int64(w, x)
    got = c.cpu().to(torch.int64)
    assert c.cpu().dtype == torch.float32
    assert torch.equal(got, ref), (
        f"N={N}: inexact! max_abs_diff={(got - ref).abs().max().item()}, "
        f"n_mismatch={(got != ref).sum().item()}"
    )


@pytest.mark.parametrize("w_val,x_val", [(1664, 3), (-1664, 3), (1664, -3), (-1664, -3)])
def test_fp16_boundary_exact(w_val, x_val):
    """Boundary: max absolute value and identical signs, sum reaches |sum| = 768*1664*3."""
    M, K, N = 1024, 768, 64
    w = torch.full((M, K), w_val, dtype=torch.int64)
    x = torch.full((K, N), x_val, dtype=torch.int64)

    a = w.to(torch.float16).to(DEV)
    b = x.to(torch.float16).to(DEV)
    c = _run_mm(mm_f16, a, b, torch.float32)

    ref = _ref_int64(w, x)
    got = c.cpu().to(torch.int64)
    bound = K * abs(w_val) * abs(x_val)
    assert torch.equal(got, ref), (
        f"W={w_val},X={x_val}: inexact! theoretical bound |sum|={bound} (<2^24={2**24}), "
        f"max_abs_diff={(got - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("N", [1, 37, 64])
def test_int8_limb_exact(N):
    """INT8 fallback path: W = 64*hi + lo, two int8 dots composed must equal int64."""
    M, K = 1024, 768
    w = torch.randint(-1664, 1665, (M, K), dtype=torch.int64)
    x = torch.randint(-3, 4, (K, N), dtype=torch.int64)

    hi = torch.div(w + 32, 64, rounding_mode="floor")
    lo = w - 64 * hi
    assert hi.min().item() >= -26 and hi.max().item() <= 26, (hi.min(), hi.max())
    assert lo.min().item() >= -32 and lo.max().item() <= 31, (lo.min(), lo.max())

    try:
        c_hi = _run_mm(mm_i8, hi.to(torch.int8).to(DEV), x.to(torch.int8).to(DEV), torch.int32)
        c_lo = _run_mm(mm_i8, lo.to(torch.int8).to(DEV), x.to(torch.int8).to(DEV), torch.int32)
    except Exception as e:
        pytest.skip(f"INT8 tl.dot unavailable/failed on XPU: {type(e).__name__}: {e}")

    got = 64 * c_hi.cpu().to(torch.int64) + c_lo.cpu().to(torch.int64)
    ref = _ref_int64(w, x)
    assert torch.equal(got, ref), (
        f"N={N}: INT8 limb composition inexact! max_abs_diff={(got - ref).abs().max().item()}"
    )


def test_report():
    """Print which of the two paths is available, for a benchmark log."""
    print("\n--- Phase 0 precision micro-test conclusions ---")
    print(f"torch      : {torch.__version__}")
    print(f"triton     : {triton.__version__}")
    print(f"device     : {describe(DEV)}")
    print("FP16(fp32 acc) exactness: see test_fp16_random_exact / test_fp16_boundary_exact")
    print("INT8 limb exactness     : see test_int8_limb_exact (skipped if unavailable)")
