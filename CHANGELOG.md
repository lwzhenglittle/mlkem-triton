# 变更记录

## 0.1.0

首个可运行版本。**纯 Triton，无内联汇编、无厂商专属 intrinsic**。

### 三个操作

| 操作 | 实现 | 正确性 | GB10 每 op |
|---|---|---|---:|
| decaps | Toeplitz GEMM（无 NTT）| NIST ACVP 3×10 条 | 125 ns |
| encaps | Toeplitz GEMM（无 NTT）| NIST ACVP 3×25 条 | 88 ns |
| keygen | NTT（必须）| 与 kyber-py 逐字节对拍 | 469 ns |

### 已在真机验证

- Intel Arc B390（Xe3-LPG）/ triton-xpu 3.9 / torch 2.15
- NVIDIA GB10（sm_121）/ triton 3.8 / torch 2.14

264 个测试在两台机器上全绿（无 GPU 时自动降级为 90 passed / 174 skipped）。
ROCm **未测**（只有代码级推断）。

### 已知限制

- keygen 比 cuPQC 慢约 2.7 倍（NTT 是它的结构性弱项，见
  `results/comparison_prior_work.md` §0.1d）
- 每把私钥 7.9 MB 显存 + 13.5 ms CPU 预处理（GPU 重建 0.158 ms）
- ML-KEM-512 的 η₁=3 走通用 CBD 路径，慢约 3 倍
- 承重接口需要大批量：Intel 核显上 B ≥ 1024 才追平 liboqs-16 线程
- 没有形式化的常数时间验证（未做 dudect/ctgrind）

### 测量方法学

`results/comparison_prior_work.md` 记录了完整的对照数据、与 cuPQC 的同机对比、
以及本项目里**被证伪的假设**（例如「pinned 中转不划算」其实是测错了 memcpy 的方式）。
