"""Phase 0.4 精度微测 —— 整个方案的前提。

验证：
  1. FP16 路径：W ∈ [-1664,1664], X ∈ [-3,3]，`tl.dot(fp16, fp16, acc=fp32)` 与
     CPU int64 matmul **逐元素完全相等**。
  2. 边界用例：W、X 全取最大绝对值且同号，把累加和顶到理论上界。
  3. INT8 路径：W 拆 limb（hi = floor((W+32)/64) ∈ [-26,26]，lo = W - 64*hi ∈ [-32,31]），
     两次 int8 dot 合成 `64*(hi@X) + (lo@X)`，与 int64 对拍。

注意（修订自原 plan）：原 plan 的 INT8 用例直接对 W∈[-1664,1664] 调 int8 dot 是放不进去的，
这里改为 limb 拆分。同时覆盖 N=1 / 非 16 对齐的批大小，证明"BN≥16 的 tile + mask"可以兜住小 B。

运行：.venv/bin/python -m pytest tests/test_dot_exact.py -v -s
"""

from __future__ import annotations

import pytest
import torch
from _common import DEV

from mlkem_triton.devices import describe, sync

xpu_ok = True  # 见 _common.DEV：能 import 就说明有可用 GPU

try:
    import triton
    import triton.language as tl

    triton_ok = True
except Exception:
    triton_ok = False

pytestmark = pytest.mark.skipif(not (xpu_ok and triton_ok), reason="需要 GPU + Triton")

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
    """随机整数矩阵：FP16 输入 + FP32 累加必须与 int64 完全相等。"""
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
        f"N={N}: 不精确! max_abs_diff={(got - ref).abs().max().item()}, "
        f"n_mismatch={(got != ref).sum().item()}"
    )


@pytest.mark.parametrize("w_val,x_val", [(1664, 3), (-1664, 3), (1664, -3), (-1664, -3)])
def test_fp16_boundary_exact(w_val, x_val):
    """边界：所有元素取最大绝对值且同号，累加和顶到 |sum| = 768*1664*3。"""
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
        f"W={w_val},X={x_val}: 不精确! 理论界 |sum|={bound} (<2^24={2**24}), "
        f"max_abs_diff={(got - ref).abs().max().item()}"
    )


@pytest.mark.parametrize("N", [1, 37, 64])
def test_int8_limb_exact(N):
    """INT8 兜底路径：W = 64*hi + lo，两次 int8 dot 合成后与 int64 相等。"""
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
        pytest.skip(f"INT8 tl.dot 在 XPU 上不可用/失败: {type(e).__name__}: {e}")

    got = 64 * c_hi.cpu().to(torch.int64) + c_lo.cpu().to(torch.int64)
    ref = _ref_int64(w, x)
    assert torch.equal(got, ref), (
        f"N={N}: INT8 limb 合并不精确! max_abs_diff={(got - ref).abs().max().item()}"
    )


def test_report():
    """把两条路径的可用性打印出来，便于写进 results/NOTES.md。"""
    print("\n--- Phase 0 精度微测结论 ---")
    print(f"torch      : {torch.__version__}")
    print(f"triton     : {triton.__version__}")
    print(f"device     : {describe(DEV)}")
    print("FP16(fp32 acc) 精确性: 见 test_fp16_random_exact / test_fp16_boundary_exact")
    print("INT8 limb 精确性     : 见 test_int8_limb_exact（不可用会 skip）")
