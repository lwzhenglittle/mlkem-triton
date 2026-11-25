# 与已有 GPU 上 Kyber/ML-KEM 实现的系统对比

> 我方 = `mlkem-triton`（本仓库）。对比对象：NVIDIA cuPQC、HI-Kyber（南京邮电大学，2023）、
> ConvKyber（TCHES 2024）、以及 `dspace.hansung.ac.kr` 那篇 OAK 论文。
>
> **本文件里所有我方数字都是实测**（机器、命令、口径在 §1 写清楚）。
> 对方的数字全部标注来源。查不到的一律写「查不到」，不估。

---

## 0. 先说结论

### 0.1 同机同操作对照：我们比 cuPQC 快 1.61 倍

cuPQC 0.6.0 被真的在 **GB10 / sm_121** 上编译跑通了（官方支持列表里没有 sm_121，
但 `-dlto` bitcode 链接期生成 SASS，实测正确且性能正常）。于是有了一个**同一块卡、
同一个操作、同样是批量吞吐**的干净对照：

| ML-KEM-768 decaps | 硬件 | SM | 吞吐 | 每 op | 每 SM | 批大小 | 计时 |
|---|---|---:|---:|---:|---:|---:|---|
| **cuPQC-PK 0.6.0**（BD=32，我调过）| **GB10** | 48 | **4.96 M/s** | 202 ns | 103 k/SM | 262,144 | 纯 kernel（cudaEvent）|
| **我们** | **GB10** | 48 | **8.0 M/s** | 125 ns | **167 k/SM** | 8,192 | 设备侧（含 13 次入队）|
| | | | **1.61×** | | **1.61×** | | |
| cuPQC-PK 0.6.0（**SDK 默认 BD=128**）| GB10 | 48 | 4.27 M/s | 234 ns | 89 k/SM | 1,048,576 | 纯 kernel |
| **→ 开箱即用比值** | | | | | | | **1.86×** |
| cuPQC-PK 官方 | H100 | 132 | 8.5 M/s | 118 ns | 64 k/SM | 未说 | 未说 |

**这是本次调研最重要的一行：在同一块 GB10 上、同样的完整 decaps，我们的吞吐是 cuPQC 的 1.61 倍。**
而且我们的口径更吃亏一些（含 13 次 Python 侧 Triton 入队；载荷在 B=8192 时被设备工作掩盖，
但仍然算了进去）。

跨代对照：cuPQC 在 GB10 上只跑到官方 H100 数字的 58%；我们在 GB10 上跑到 8.0 M/s，
已经超过 cuPQC 在 H100 上的 8.5 M/s 的 94%。

### 0.1a 三轮交替复核（时钟对齐，跨进程）

cuPQC 是 C++ 二进制、我们是 Python，没法同进程交错 A/B。退而求其次：两个进程
交替跑 3 轮，每轮前后采样 `clocks.sm`。结果比预想的稳：

| 轮次 | 时钟（cuPQC 段 / 我方段）| cuPQC decaps（SDK 默认 BD=128）| 我们（B=8192）|
|---|---|---|---|
| 1 | 2398 MHz / 2398–2411 MHz | 4.260 M/s（234.6 ns）| 7.976 M/s（125.4 ns）|
| 2 | 2398 / 2398 | 4.267 M/s（234.4 ns）| 7.937 M/s（126.0 ns）|
| 3 | 2398 / 2398–2411 | 4.279 M/s（233.7 ns）| 7.914 M/s（126.4 ns）|

**三轮比值：1.86× / 1.86× / 1.85× —— 抖动 < 1%，两台进程的时钟完全对齐
（2398–2411 MHz）。** 这个差异远大于测量噪声，方向是可靠的。

注意仓库里那个 `bench768` 二进制是 **SDK 默认 BlockDim=128** 编的；BD=32 是调研时
单独编的。所以：
- **开箱即用**（都用默认设置）：我们 **1.86×**
- **双方都调过**（cuPQC 调到 BD=32、我们扫过 tile）：我们 **1.60×** ← 保守口径

### 0.1b 把这个 1.61× 拆开：86% 来自内核，14% 来自预展开

cuPQC 是无状态 API，**每条密文**都要重跑 k² 次 SampleNTT（SHAKE-128，840 B/次）
得到 `Â`。我们把 `Â` 预先展开成 Toeplitz 矩阵常驻显存、每 op 零 SHAKE。
所以 1.61× 里有多少来自「跳过 `Â` 生成」原本分不清。

为此写了 `kernels/rebuild.py`：**在 GPU 上从 `dk` 完整重建 `W_dec` / `W_enc`**
（SHAKE-128 → 拒绝采样 → INTT → 中心化 → Toeplitz 展开），并验证它与 CPU 的
`keyprep.parse_key` **逐元素相等**（三个参数集都是，`tests/test_rebuild.py` 锁住）。
把重建塞回计时区间，就分得清了。

GB10 实测（ML-KEM-768，k=3）：

| 口径 | 每批 B=8192 | 每 op | vs cuPQC 201.7 ns |
|---|---:|---:|---:|
| **A：预展开**（主路径）| decaps 1.033 ms | **126.0 ns** | **1.60×** |
| **B：每批重建一次**（同信息量的公平口径）| decaps 1.033 + rebuild 0.158 ms | **145.4 ns** | **1.39×** |
| C：每 op 重建一次（cuPQC 被迫的形态，无部署价值）| — | 158,413 ns | 0.001× |

后续 O2 收尾时又试了两轮 keygen 优化（预展开轮常数表消掉每级整数除法；
把 `pow_mac` 里 16 次 `tl.gather` 换成 stride=2 的纯 load/store）：
**Intel 上 1.50 → 1.46 µs/key（−3%），GB10 上 469 → 468 ns/key（无变化）**。
换 gather 那一版没有收益，因为 `pow_mac` 本来就是访存受限 —— 见上面的分布表。

重建的成本构成（GB10，rebuild 一次 0.158 ms）：

| 步骤 | 耗时 |
|---|---:|
| SHAKE-128（9 × 840 B）| 0.034 ms |
| 其余（拒绝采样 + INTT + **Toeplitz 展开 7.9 MB 写入**）| 0.124 ms |

**结论：把 `Â` 重建完全算进计时后，我们仍然比 cuPQC 快 1.39 倍**（145.4 vs 201.7 ns）。

而且这个 1.39× 是**下界**，因为我们的重建比 cuPQC 需要的更贵：他们要的只是
7.5 KB 的 NTT 域 `Â`，我们要写出 **7.86 MB 的 Toeplitz 矩阵**（0.124 ms 里绝大部分）。
cuPQC 不用付这笔 Toeplitz 展开的钱。

所以 1.61× 的分解是：

```
1.60×  =  1.39×（Toeplitz GEMM 内核 vs NTT 内核）  ×  1.15×（预展开 vs 每 op 重建）
```

即 **87% 来自内核本身，13% 来自预计算。**

盈亏平衡：口径 B 在 B ≥ 4096 时赢（1.19×），B=1024 时输（0.47×）；
B=16384 时 1.54×。**这个设计需要大批量，和前面所有结论一致。**

### 0.1c 封装（encaps）：快 2.27 倍，而且这次是**没有争议**的

`encaps.py` 已实现并通过 **全部 75 条 ACVP 封装向量**（3 个参数集 × 25 条，
密文 `c` 与共享密钥 `K` 都逐字节相等）。

GB10 实测（ML-KEM-768，设备侧含 Python 入队）：

| B | 设备侧 ns/op | 吞吐 | vs cuPQC 调优 199.7 ns | vs cuPQC 默认 243.5 ns |
|---:|---:|---:|---:|---:|
| 1024 | 195 | 5.14 M/s | 1.03× | 1.25× |
| 4096 | 94 | 10.67 M/s | 2.13× | 2.60× |
| 8192 | **88** | **11.38 M/s** | **2.27×** | **2.77×** |
| 16384 | 90 | 11.08 M/s | 2.21× | 2.70× |

**比 decaps 的优势更大**，原因也清楚：封装同样要 `Â`，而我们把
`[u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]` 直接做成一次 GEMM
（复用 enc 那趟的融合内核，只换 epilogue：从「比较计数」换成「压缩 + 位打包」）。
`r`、`e1`、`e2` 都是小量，精度界限照旧成立。

扣掉预展开：`W_enc` 的重建成本与 dk 那套同量级（约 0.14 ms，因为少 2k 行），
摊到 B=8192 约 17 ns/op → **约 105 ns/op，仍是 1.90×**。

### 0.1d keygen：**我们输 2.7 倍** —— 这是唯一输的一项

三个操作现在齐了（`keygen.py`），keygen 用**逐字节对拍 kyber-py** 锁死：
三个参数集的 `ek`（800/1184/1568 B）和 `dk`（1632/2400/3168 B）**完全一致**。

GB10 实测：

| B | 设备侧 ns/key | 吞吐 | vs liboqs-1T (7.73 µs) | **vs cuPQC 173.9 ns** |
|---:|---:|---:|---:|---:|
| 1024 | 561 | 1.78 M/s | 10.2× | 0.31× |
| 4096 | 467 | 2.14 M/s | 13.3× | 0.37× |
| 8192 | 469 | 2.13 M/s | 13.5× | **0.37×** |

**cuPQC 的 keygen 比我们快 2.7 倍。** 原因就是预期中的那个：

- decaps / encaps 的乘法被折成「跨操作固定的矩阵」的 GEMM，Toeplitz 技巧成立
- keygen 的 `s`/`e` **每把密钥都是新的**，没有任何可复用的矩阵，必须老老实实做 NTT
- 而 `ek` 里必须放 `t̂` 的 NTT 域表示，所以 NTT 躲不掉

分段耗时（B=4096，Intel）：NTT 372 ns、SampleNTT 448 ns、SHAKE-128 ×9 269 ns、
逐点乘 223 ns、ByteEncode12 ×2 141 ns、CBD ×2 44 ns。

**GB10 上 keygen 的真实分布**（B=4096，分段单独计时）：

| 阶段 | ns/key | 占比 |
|---|---:|---:|
| `pow_mac`（逐点乘 + 扩域，要读 37.7 MB 的 Â）| 152 | **32.7%** |
| NTT（2k 条多项式）| 78 | 16.7% |
| SampleNTT | 76 | 16.4% |
| SHAKE-128 ×9 | 75 | 16.1% |
| ByteEncode12 ×2 | 45 | 9.8% |
| CBD ×2 | 11 | 2.3% |
| 其余（msgs 构造 / pad / H(ek)）| 25 | 5.5% |
| **合计** | **466** | |

**这张表推翻了「NTT 是 keygen 瓶颈」的假设。** NTT 只占 17%；最大的一块是
`pow_mac`，而它 0.62 ms 对应的访存量约 75 MB（Â 37.7 + Ŝ 12.6 + Ê 12.6 + 写 12.6），
按 GB10 的 273 GB/s 算理论下界 ~0.28 ms —— **它是访存受限，不是算力受限**。

所以真正的差距在于**我们让 Â 在 kernel 之间落了显存**（B=4096 时 37.7 MB），
而 cuPQC 把整个 keygen 融在一两个 kernel 里。要真正追上，需要把
SampleNTT → NTT → pow_mac 融成一条流水，而不是继续抠单个 kernel。

**NTT 内核本身调了两次**才可用：

| 版本 | 做法 | P=24576 时耗时 |
|---|---|---:|
| v1 | 两个分支各算一遍（2 乘 4 mod）+ `tl.gather` + BR=16 | **15.2 ms** |
| v2 | 用 `where` 把 a/b 归一到两个半区（**1 乘 2 mod**）+ BR=2 | **1.5 ms（10×）** |

`tl.gather` 本身不慢（7 次只有 1.2 ms），算术也不慢（0.9 ms）—— 但**两者放在同一个
kernel 里、配大 tile（BR≥16）就爆炸**（16.7 ms）。降到 BR=2 就好了。这是 Triton
在这个形状上的一个病态点，值得记下来。

### 0.2 但有一个必须讲清的前提

**没有任何一篇论文在"同一块卡上"比较过「Toeplitz + 张量核」和「NTT」。**
所以我们**不能**声称我们的方法比 NTT 快 —— 只能说在同样报批量吞吐的三个实现里，
我们的绝对值领先。ConvKyber 的 12.48× 是「它的张量核 NTT vs 蝶形 NTT」，
TensorCrypto 的 2–5× 是「张量核卷积 vs CUDA core INT32」，都不是和我们比。

### 0.3 我们对 TensorCrypto 那条判据的正面回答

TensorCrypto（IEEE Access 2022，就是用户给的那个 Hansung 链接，只不过它是 NTRU/LAC
不是 Kyber）明确写了为什么不做 Kyber：

> "schemes such as KYBER and NewHope already use NTT-based multiplications, the polynomial
> convolution is already efficient, so it would be more advantageous to use NTT...
> it is also difficult to use tensor cores to accelerate the NTT computations, since the
> size of modulus (q) is most likely to exceed 11-bit. Therefore, we restrict the scope of
> our paper to the schemes where NTT is slow or not applicable."

它给的理由是「Kyber 的 NTT 已经够快，卷积没优势」。**我们的实测是对这条判断的经验性反驳**
（在 GB10 上 8.0 M/s、Arc 核显上 3.3 M/s，远超 CPU 基线）。

但要注意：**这不是严格证明。** 我们没有任何同卡同口径的 NTT 实现做对照，所以我们的优势
可能来自「用了张量核」而不是「用了 Toeplitz」。要真正回答这个问题，需要把
ConvKyber 的整数张量核 NTT 和我们的 fp16 Toeplitz GEMM 放在同一块卡上跑同一批
密文 —— 这件事**没人做过，我们也没做**。

### 0.4 精度路线：三家都做到了精确，但走的路不同

| | 做法 | 精确性来源 |
|---|---|---|
| ConvKyber | INT8/INT16 张量核 + INT32 累加 | 整数，天然精确；12-bit 系数拆成两个 6-bit 半，多次累加只做一次 Montgomery 归约 |
| TensorCrypto | FP16 输入 + FP32 累加 | 位宽论证：q < 2^11、一方三值 → 单积 ≤11 bit → 累加上界 N×2^11 ≤ 2^21，FP32 的 24-bit 够 |
| **我们** | **FP16 输入 + FP32 累加** | **一方的取值范围不是 q/2 而是 η（≤3）或 CBD 采样范围（≤2）** |

**我们的关键观察（也是和 TensorCrypto 的实质差别）**：Kyber 的每一次多项式乘法里，
**总有一个操作数是很小的**，从来不会出现「两个满量程系数相乘」：

| GEMM | 权重侧 | 变量侧 | 单积上界 | K | 累加上界 | 距 2^24 |
|---|---|---|---|---|---|---|
| DEC `W_dec·U` | `T(s)`，实测 max **2** | `u` ∈ [−1664,1664] | 3328 | 768 | 2.56e6 | **6.6×** |
| ENC `W_enc·Y` | `T(Â)/T(t)`，实测 max **1664** | `y` ∈ [−2,2] | 3328 | 768 | 2.56e6 | **6.6×** |
| DEC/ENC（ML-KEM-1024）| 同上 | 同上 | 3328 | 1024 | 3.41e6 | **4.9×** |

`max|W_dec| = 2`（= η₁）、`max|W_enc| = 1664` 都是**在真实密钥上实测**的，不是推的。
TensorCrypto 的位宽论证假设两个操作数都是满量程（1664×1664 = 2.77e6，×256 = 7.1e8 > 2^24，
确实不够），而 Kyber 的实际算子里这种情形根本不出现。

### 0.5 我们真正独有的两点

1. **fp16 精确性可证明**（4.9–6.6× 余量），不需要整型拆分、不需要 Montgomery、不需要误差补偿。
2. **同一份纯 Triton 代码在两个厂商的硬件上都跑通并通过全部 237 个测试**（Intel Xe3 + NVIDIA sm_121）——
   对比的四家全是 NVIDIA-only（TensorCrypto 开源但只有 CUDA）。

### 0.6 我们最实质的短板

**只有 decaps，且每把私钥要 7.9 MB + 13.5 ms 预处理。** 其他三家都能做完整的
keygen/encaps/decaps，`Â` 每次由 ρ 现算。我们的做法只在「一把长期私钥 + 海量密文」
下成立 —— 这相当于**把一部分工作挪出了计时区间**，比吞吐时必须承认。

---

## 1. 我方数据：测什么、怎么测

### 1.1 两台机器

| | Intel Arc B390 | NVIDIA GB10 |
|---|---|---|
| 形态 | **核显**（`is_integrated_gpu=1`），B390 / Xe3-LPG | DGX Spark 单芯片，`sm_121` |
| 计算单元 | 96 EU / 12 subslice | 48 SM |
| 末级缓存 | 16 MB | 24 MB L2 |
| 显存 | 29 GB（共享系统内存） | 124 GB 统一内存 |
| 时钟 | 空载 0.9 GHz / 满载 ~2.2 GHz | 2.418 GHz |
| 软件栈 | torch 2.15.0+xpu / triton-xpu 3.9.0 | torch 2.14.0+cu130 / triton 3.8.0 |

### 1.2 口径

- **设备侧**：只算 `decaps_torch()` 的墙钟（含 Python 侧 13 次 kernel 入队 + 设备执行），
  不含 H2D/D2H。论文一般用 `cudaEvent` 只圈 kernel，和这一栏最接近。
- **端到端**：`decaps()` 全流程，含 numpy 打包、H2D、D2H。
- **min-of-N**：N=10~25，每次前后 `synchronize()`。GPU 时钟在 0.9–2.2 GHz 之间漂且
  `xe` 驱动没有调频接口，所以**绝对值有 ±20% 抖动**；表里给的是多次运行里最好的值，
  同一轮内配对比值才是可靠的。
- 密文全部来自 NIST ACVP 官方向量，输出逐字节与 expectedResults 比对通过。

### 1.3 吞吐（ML-KEM-768，B=4096，单流）

| | 设备侧 | | | 端到端 | | |
|---|---:|---:|---:|---:|---:|---:|
| | ms | **µs/op** | Mops/s | ms | **µs/op** | Mops/s |
| Intel Arc B390 | 1.256 | **0.307** | **3.26** | 1.905 | **0.465** | 2.15 |
| NVIDIA GB10 | 0.540 | **0.132** | **7.59** | 0.757 | **0.185** | 5.41 |

B 扫描（设备侧 µs/op，找渐近值）：

| B | 1024 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---:|---:|---:|---:|---:|---:|
| Intel | 0.668 | **0.311** | 0.395 | 0.408 | 0.393 | 0.380 |
| GB10 | 0.272 | 0.130 | **0.125** | 0.125 | 0.128 | 0.130 |

- Intel 在 B=4096 最优（工作集 ≈16 MB，正好是它的末级缓存），再大就掉。
- GB10 从 B=8192 起就平坦，一直到 65536 不掉。
- **渐近吞吐：Intel 3.2 Mops/s，GB10 8.0 Mops/s。**

### 1.4 三个参数集（B=4096）

| ps | Intel 设备侧 µs/op | GB10 设备侧 µs/op | 备注 |
|---|---:|---:|---|
| ML-KEM-512 | 0.932 | 0.760 | **慢路径**（见下）|
| ML-KEM-768 | 0.307 | 0.132 | 快路径 |
| ML-KEM-1024 | 0.411 | 0.173 | 快路径 |

ML-KEM-512 的 `η1 = 3`，而融合 CBD 内核只实现了 `η = 2` 的位打包，所以 512 回退到通用
CBD 路径，比 768 还慢 3 倍。功能正确（ACVP 通过），只是没优化。

### 1.5 单条延迟（这是我们的弱项）

| | B=1 端到端 | B=8 端到端 |
|---|---:|---:|
| Intel Arc B390 | **945 µs** | 1086 µs（136 µs/op）|
| GB10 | **282 µs** | — |

单条 282–945 µs，比任何单条优化的 GPU 实现都慢 1~2 个数量级。原因很直接：整条流水线
13 次 kernel launch，每次固定开销 ~35–40 µs，单条时没有任何东西可以摊薄。
**这个设计是吞吐导向的，不是延迟导向的。**

### 1.6 算术量与实际利用率

ML-KEM-768 每条解封装的 GEMM 算术量（Toeplitz 展开后）：

```
DEC  W_dec[256, 768] × U[768, B]   = 196,608 MAC
ENC  W_enc[1024,768] × Y[768, B]   = 786,432 MAC
                                     983,040 MAC = 1.97 MFLOP / op
```

| | 吞吐 | 折算 FLOPS | 我们自己的 Triton GEMM 峰值 | 占峰值 |
|---|---:|---:|---:|---:|
| Intel Arc B390 | 3.26 Mops/s | 6.4 TFLOPS | 26–29 TFLOPS | ~24% |
| GB10 | 7.59 Mops/s | 15.0 TFLOPS | 53.1 TFLOPS | ~28% |

「我们自己的 GEMM 峰值」是同一形状（M=1024,K=768,B=4096）的纯 `tl.dot` 循环实测值，
所以这个百分比说明**还有 3–4 倍的理论空间**（主要卡在 Keccak、CBD、`_pre` 这些非
GEMM 阶段和 epilogue 上，见 §4）。

### 1.7 相对 CPU 基线（liboqs 0.16.0，同机实测，同一批 ACVP 密文）

| B | Intel 端到端 µs/op | vs liboqs-16T (1.45) | vs liboqs-1T (10.26) | GB10 端到端 µs/op | vs liboqs-16T | vs liboqs-1T |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 966 | 0.00× | 0.01× | 282 | 0.01× | 0.04× |
| 64 | 11.8 | 0.12× | 0.87× | 4.65 | 0.31× | 2.21× |
| 128 | 7.29 | 0.20× | 1.41× | 2.35 | 0.62× | 4.37× |
| 256 | 2.86 | 0.51× | 3.59× | 1.21 | **1.20×** | 8.47× |
| 512 | 1.83 | 0.79× | 5.62× | 0.644 | 2.25× | 15.9× |
| 1024 | 0.98 | **1.48×** | 10.5× | 0.349 | 4.15× | 29.4× |
| 4096 | 0.44 | 3.27× | 23.1× | 0.236 | 6.15× | 43.5× |
| 8192 | 0.41 | 3.58× | 25.3× | 0.227 | 6.38× | 45.2× |

**盈亏平衡点**：Intel 要 B ≥ 1024 才追平 liboqs-16T（B=512 还慢 21%）；GB10 在 B ≥ 256 就超过。
小批量下我们不如一颗 16 线程的 CPU —— 这是必须讲清楚的。

### 1.7b encaps 相对 CPU 基线（同机 liboqs 0.16.0）

`liboqs` 的 `speed_kem` 是**单线程**程序（`OMP_NUM_THREADS` 无效），所以多核基线是用
**16 个并发进程**跑出来的聚合吞吐。

| | ML-KEM-768 encaps | | ML-KEM-768 decaps | |
|---|---:|---:|---:|---:|
| liboqs，单线程 | 7.82 µs/op | 127.9 k/s | 9.52 µs/op | 105.1 k/s |
| liboqs，16 并发进程 | **1.109 µs/op** | 902 k/s | **1.375 µs/op** | 727 k/s |

我方（Arc B390，端到端含 H2D/D2H）：

| B | encaps 端到端 µs/op | vs liboqs-16 | vs liboqs-1T | encaps 设备侧 µs/op |
|---:|---:|---:|---:|---:|
| 256 | 2.248 | 0.49× | 3.48× | 1.467 |
| 512 | 1.211 | 0.92× | 6.46× | 0.838 |
| 1024 | 0.695 | **1.60×** | 11.3× | 0.491 |
| 4096 | 0.381 | 2.91× | 20.5× | 0.218 |
| 8192 | 0.379 | 2.93× | 20.6× | 0.184 |
| 16384 | **0.310** | **3.58×** | **25.2×** | **0.175** |

盈亏平衡点：对 liboqs-16 并发约 **B ≈ 700**（decaps 要 B ≥ 1024），对 liboqs 单线程
B = 256 就已经快 3.5 倍。

**可比性**：两边都是完整的 FIPS 203 encaps，都包含把 1088 字节密文写到主机内存
（liboqs 直接写在缓存里；我们是 D2H，B=16384 时 17.8 MB 传输占端到端的 43%）。
所以端到端那一列是可比的；设备侧那一列**不含**传输，对 GPU 有利，不应拿来和
liboqs 比。

### 1.8 一次性成本 / 资源占用（对比时经常被忽略，但很关键）

| | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|---|---:|---:|---:|
| `keyprep` 预处理 | 6.8 ms | **13.5 ms** | 19.8 ms |
| 常驻显存（`W_dec`+`W_enc`，fp16）| 4.2 MB | **7.9 MB** | 12.6 MB |
| 标准实现里 `dk` 的大小 | 1632 B | 2400 B | 3168 B |

**这是本方案最实质的架构代价**：标准实现每次由 ρ 现算 `Â`，只存 2400 字节；
我们把 `Â`/`t`/`s` 全部展开成 Toeplitz 矩阵常驻显存，一把钥匙 7.9 MB + 13.5 ms 建表。
只在「一把长期私钥、海量密文」下成立。

---

## 2. 对比对象

### 2.1 NVIDIA cuPQC（cuPQC-PK）

- 出处：<https://developer.nvidia.com/cupqc>（NVIDIA 官方产品页，抓取于 2026-11-25）
- 性质：**闭源商业 SDK**（"Download cuPQC SDK"，需申请），不是公开仓库
- 组成：cuPQC-BigInt / cuPQC-NTT / cuPQC-Hash / **cuPQC-PK**
- cuPQC-PK 支持 ML-KEM-512/768/1024 与 ML-DSA-44/65/87
- 路线：**NTT**（cuPQC-NTT 是独立组件）+ 哈希库；用 link-time optimization 按配置选 kernel
- 官方原话：「optimized for both low-latency single operations and high-throughput batch processing」

**官方公布的数字**（ML-KEM-768，**NVIDIA H100**，相对单核 AMD EPYC 7313P）：

| 操作 | 吞吐 | 相对单核 CPU |
|---|---:|---:|
| keygen | **13.5 M/s** | 175× |
| encaps | **9.0 M/s** | 120× |
| **decaps** | **8.5 M/s** | 135× |

换算：decap = **0.1176 µs/op**。反推他们的 CPU 基线 ≈ 63 k decap/s（15.9 µs/op），
和我们在同一台机器上用 liboqs 单线程测到的 10.26 µs/op 是同一量级，可信。

> 这是我方唯一一个可以**直接对比吞吐**的对象 —— 因为它也是「大批量吞吐」口径。

**在本机 GB10 上实测成功**（详见 §3.4）：cuPQC 0.6.0（2026-08-27 构建，tar sha256
`f797d65f…8fe992`），`nvcc -dlto -arch=sm_121 -O3` 一次编译通过，无报错。
keygen→encaps→decaps 回环 32/32 一致，尺寸与 FIPS 203 一致（pk 1184 / sk 2400 /
ct 1088 / ss 32 B）。**SDK 里没有任何 KAT/ACVP 程序**（grep 全包无命中），所以它的
FIPS 203 合规性无法用它自己的向量背书。

GB10 实测（2.4 GHz boost，cudaEvent 背靠背，纯 kernel）：

| 配置 | batch | keygen | encaps | **decaps** |
|---|---:|---:|---:|---:|
| **BD=32**（扫描出的最优；默认是 128）| 262,144 | 5.75 M/s | 5.01 M/s | **4.96 M/s**（202 ns）|
| BD=128（SDK 默认）| 1,048,576 | 5.12 M/s | 4.03 M/s | 4.27 M/s（234 ns）|

其他实测事实：
- **多流零收益**（1/2/4/8 流：4.955 / 4.947 / 4.937 / 4.929 M/s）。和我们在 Intel 上
  测到的「H2D 无法与计算重叠」是同一类现象：这个平台不做并发。
- **batch=1 单条延迟 57.4 µs（kernel）/ 61.4 µs（端到端），其中约 53 µs 是 launch 固定开销**
  —— 这块卡的 launch 开销极大，所以「单条走 GPU」在 GB10 上对谁都没意义。
- decaps 实际带宽只有 ~17 GB/s（3488 B/op × 4.96 M/s），远低于 LPDDR5X —— **算力受限，不是带宽受限**。
- 许可证是 **"NVIDIA Software and Model Evaluation License"（评估许可，非商用）**，
  闭源二进制。要商用得单独谈。
- 官方支持架构列表是 70/75/80/86/87/89/90，**不含 sm_121**，但实测能跑。

### 2.2 HI-Kyber（南京邮电大学，2023）

- **是 GPU 论文，不是 FPGA。**
- IACR ePrint **2023/1194**（2023-08），期刊版 **IEEE TPDS 2024**，
  DOI [10.1109/TPDS.2024.3379734](https://doi.org/10.1109/TPDS.2024.3379734)
- 作者：Xinyi Ji, Jiankuo Dong, Pinchang Zhang, Deng Tonggui, Fu Xiao（**NJUPT**）；
  Hua Jiafeng（西电）
- **路线：NTT，CUDA core**（明确把 tensor-core 实现当对手）。做了 SLM / SDFS-NTT /
  EDFS-NTT 三种改写，分别 +7.5% / +28.5% / +41.6%
- **没有用卷积/Toeplitz 替代 NTT**

**能拿到的数字（全部来自摘要页，因为全文闭源）**：

| 指标 | 数值 |
|---|---:|
| key exchange 吞吐 | **1,664 kops/s** |
| vs 同指令集 GPU 实现 | 3.52× |
| vs "SOTA AI-accelerated tensor core" 实现（未点名）| 1.78× |

**查不到的**：GPU 型号、keygen/encaps/decaps 各自数字、批大小、计时方式、是否多流。
原因：TPDS 闭源（OpenAlex `oa_status: closed`），ePrint 的 PDF 被 Cloudflare 403 拦。

→ **无法与我们比较。** 而且它的 "key exchange kops/s" 没给公式，和 ConvKyber 的
`ab/(a+b)` 口径也不能互相比。

### 2.3 ConvKyber（TCHES 2024）

- "ConvKyber: Unleashing the Power of AI Accelerators for Faster Kyber with Novel
  Iteration-based Approaches"，**TCHES 2024(2): 25–63**，
  [DOI](https://doi.org/10.46586/tches.v2024.i2.25-63)，
  [PDF](https://tches.iacr.org/index.php/TCHES/article/download/11420/10923)（CC-BY）
- 作者：Tian Zhou, Wenxu Tang, Jingqiang Lin（USTC）；Fangyu Zheng, Lipeng Wan, Yi Bian（UCAS）；
  Guang Fan, Yixuan Song（蚂蚁）
- 平台：**Jetson AGX Orin**（15/30/50/60W，SM 6/8/16）+ **RTX 3080**（68 SM，1.8 GHz，
  100 KB smem/SM）
- 算法：**Kyber round-3（v3）**，不是 FIPS 203
- **路线澄清：它没有用卷积替代 NTT。** 原文自述
  "we name this framework convKyber because we convert the core NTT operations into
  operations **resembling** convolutions" —— 它是保留 NTT、把蝶形重排成迭代小块矩阵乘。
  Toeplitz 卷积做多项式乘是它引用的**前人工作**（Wan et al. 用于 LAC/NTRU）。
- 精度：**INT8/INT16 张量核 + INT32 累加**，12-bit 系数拆两个 6-bit 半（`SHL 12`/`SHL 6` 回拼），
  可配 Karatsuba（4 乘 → 3 乘），多次累加后只做一次 Montgomery 归约。**不用 fp16。**
- 开源：无

**Kyber-1024 吞吐（Table 8/9，kops/s）**：

| 平台 | KeyGen | Enc | Dec | Decaps† | KX |
|---|---:|---:|---:|---:|---:|
| **RTX 3080（本文）** | 2,074 | 1,587 | **8,458** | **1,336†** | 1,336 |
| Wan et al. tensor core, R3080 | 1,250 | 1,299 | 2,381 | — | 820 |
| Gupta et al.，R3080 复测 | / | / | / | — | 473 |

† **Decaps 是公式推算**（表注：`ab/(a+b)`，a=Enc、b=Dec），不是实测。

- 批大小：R3080 KeyGen/Enc **17,408** 并发（grid 136 × block 128），Dec 8,704
- 全流程延迟 1.02–5.68 ms（整批，不是单条）；单条 `polyvec_ntt` 0.259 µs
- 计时：500 次迭代平均，**未提 cudaEvent、未说明是否含 host/H2D**
- **Kyber-768 没有表格数字**，只在 Figure 18 柱状图里

### 2.4 Hansung OAK_1445 → 实际是 **TensorCrypto（IEEE Access 2022）**

**用户给的链接是坏的**：`.../2024.oak/1883/2/OAK_1445.pdf` → HTTP 404 "Invalid Identifier"。
同 handle 下的真实文件是
`.../1883/2/AR_E1A1A4_TensorCrypto- High Throughput Acceleration of Lattice-Based Cryptography Using Tensor Core on GPU.pdf`，
条目页 <https://dspace.hansung.ac.kr/handle/2024.oak/1883>。

- **"TensorCrypto: High Throughput Acceleration of Lattice-Based Cryptography Using
  Tensor Core on GPU"**，**IEEE Access 10:20616–20632, 2022**，
  [DOI](https://doi.org/10.1109/ACCESS.2022.3152217)
- 作者：Wai-Kong Lee（Gachon）, **Hwajeong Seo（Hansung）**, Zhenfei Zhang（EF）,
  Seong Oun Hwang（Gachon）
- 平台：RTX 2060 / RTX 3080，1.71 GHz；CPU 对照 i7-9700F
- **算法：NTRU（hps2048509/677）、LAC、FrodoKEM。完全没有 Kyber/ML-KEM。**
- **路线：就是「Toeplitz/循环矩阵乘 + 张量核替代 NTT」**，FP16 输入 + FP32 累加。
  这是和我们最接近的先例。
- 开源：<https://github.com/benlwk/Tensorcrypto>（public domain）

**数字（NTRU，不是 Kyber）**：TensorTRU 在 RTX 2060 上封装 **793,651/s**、解封装
**505,051/s**（512 blocks）；卷积本身 **0.33 µs/次**（N=512）；vs CUDA core INT32 GPU
最高 3.41×（2060）/ 5.77×（3080）。

**它自己写的、为什么不做 Kyber**（这段是对我们方案最直接的相关表态）：

> "schemes such as KYBER and NewHope already use NTT-based multiplications, the polynomial
> convolution is already efficient, so it would be more advantageous to use NTT...
> it is also difficult to use tensor cores to accelerate the NTT computations, since the
> size of modulus (q) is most likely to exceed 11-bit. Therefore, we restrict the scope of
> our paper to the schemes where NTT is slow or not applicable."

即：**它认为 Kyber 的 NTT 已经够快，卷积没有优势，所以不做。** 我们的做法是对这条
判断的经验性反驳（见 §0.3、§0.4）—— 但严格说没有同卡对照，只能算经验证据。

---

## 3. 横向表格

### 3.1 全部对齐到「完整 decaps 吞吐」

| 实现 | 硬件 | 计算单元 | 算法 | 是否 NTT | 张量核 | 完整 decaps | 每 SM | 批大小 |
|---|---|---:|---|---|---|---:|---:|---:|
| **cuPQC-PK（我实测）** | **GB10** | **48 SM** | **ML-KEM**（FIPS 203）| 是 | 是（未披露）| **4.96 M/s** | 103 k | 262,144 |
| **我们** | **GB10** | **48 SM** | **ML-KEM**（FIPS 203）| **否** | fp16+fp32 | **8.0 M/s** | **167 k** | 8,192 |
| **→ 同机比值** | | | | | | **我们 1.61×** | **1.61×** | |
| cuPQC-PK（官方） | H100 | 132 SM | **ML-KEM** | 是 | 是 | **8.5 M/s** | 64 k | 未说 |
| **我们**（1024）| GB10 | 48 SM | ML-KEM | 否 | 是 | **5.78 M/s** | **120 k** | 4,096 |
| ConvKyber（1024）| RTX 3080 | 68 SM | Kyber v3 | 是（重排成类卷积）| 是（int8/16+int32）| **1.34 M/s†** | 19.6 k | 17,408 |
| **我们**（核显）| Arc B390 | 96 EU | ML-KEM | 否 | 是 | **3.26 M/s** | — | 4,096 |
| HI-Kyber | 查不到 | 查不到 | Kyber v3 | 是 | 否 | **查不到** | — | 查不到 |
| TensorCrypto | RTX 2060/3080 | — | **NTRU/LAC，无 Kyber** | **否**（卷积）| 是（fp16+fp32）| **不适用** | — | 512 |

† 推算值，非实测。

### 3.2 相对 CPU 基线（各文自报，**基线不同，不可横比**）

| 实现 | 基线 | decaps 加速比 |
|---|---|---:|
| cuPQC | 单核 EPYC 7313P | 135× |
| **我们** | liboqs 0.16.0 16 线程（同机） | **3.6×（Arc）/ 6.4×（GB10）** |
| **我们** | liboqs 0.16.0 单线程（同机） | **25×（Arc）/ 45×（GB10）** |
| ConvKyber | AVX2 单核 | 慢 136×（单条延迟）；按 17,408 并发算才是赢 |
| TensorCrypto | AVX2 | 2×+（NTRU）|

注意我们用的基线是 **liboqs 的 16 线程**（1.45 µs/op），比 cuPQC 用的单核（15.9 µs/op）
强约 11 倍。所以我们的 3.6–6.4× 和 cuPQC 的 135× **完全不能比。**

### 3.3 单条延迟

**同机对照（GB10，都是 batch=1，端到端含 launch）：**

| | 单条延迟 |
|---|---:|
| **cuPQC（GB10）** | **61.4 µs**（其中 ~53 µs 是 launch 固定开销）|
| **我们（GB10）** | **282 µs** |
| **→ 我们慢 4.6×** | |
| **我们（Arc B390，核显）** | 966 µs |

**修正一个我之前说错的说法**：我先前写"慢 3 个数量级"，那是拿 ConvKyber 的
**摊销值**（整批 17,408 条的总时间 ÷ 17,408 = 0.06–0.33 µs）当单条延迟来比 —— 不成立。
同机、同口径（batch=1、含 launch）比下来我们是 **4.6× 慢**，不是 3 个数量级。

原因是这两台平台的 launch 固定开销都极大（GB10 上 cuPQC 自己也要吃 53 µs，
Intel 核显上我们 13 次 launch 要吃约 900 µs）。B=1 时谁都没法摊薄。

**用在「一次一条」的场景我们仍然很差**，只是没差到 3 个数量级。

### 3.4 cuPQC 同机实测的完整数据（GB10）

| 指标 | keygen | encaps | **decaps** |
|---|---:|---:|---:|
| 纯 kernel（BD=32, batch 262144）| 5.75 M/s (174 ns) | 5.01 M/s (200 ns) | **4.96 M/s (202 ns)** |
| 纯 kernel（BD=128 默认, batch 1048576）| 5.12 M/s (195 ns) | 4.03 M/s (248 ns) | 4.27 M/s (234 ns) |
| batch=1 kernel | 53.4 µs | 55.4 µs | 57.4 µs |
| batch=1 端到端 | 57.3 µs | 59.4 µs | 61.4 µs |

BlockDim 扫描（decaps，Mops/s）：

| BD | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| batch 262144 | **4.958** | 3.882 | 4.275 | 1.654 |

**默认 BD=128 需要调到 32 才快 15%** —— 和我们在 Intel 上"必须自己扫 tile 配置"
是同一回事。

多流（decaps，BD=32，batch 262144）：1/2/4/8 流 → 4.955 / 4.947 / 4.937 / 4.929 M/s，
**零收益甚至略降**。

批大小收敛：512 → 4096 → 32768 → 262144 → 1048576 吞吐单调收敛，2.6e5 之后饱和。

---

## 4. 我们的优点

### 4.1 没有 NTT —— 整条路径是稠密的

NTT 版的瓶颈不在算术量，在**访问模式**：蝴蝶运算的步长是 128/64/…/1，全局内存上是
规整的 stride 但每个 stage 都要重排，而且每次点乘后都要立刻约简到 `[0,q)`，靠的是
逐元素取模（一条除法或乘加序列）。这类 kernel 的算术强度低、容易被 shuffle 和
整数除法卡住。

我们的做法是把乘法折叠成固定的 256×256 分块 Toeplitz 矩阵，直接把
`[M, 256k] × [256k, B]` 丢给 fp16 张量核。结果是**纯稠密 GEMM**，张量核利用率
可以量化：Intel 上 6.4 TFLOPS、GB10 上 15.0 TFLOPS，分别是本机同形状纯 GEMM 峰值
（26–29 / 53 TFLOPS）的 **24% / 28%**。

### 4.2 fp16 精确性是**可证明**的，不是调出来的

```
|W| <= 1664 < 2^11,  |y| <= 2,  K = 256k <= 1024
=> |部分和| < 1024 * 3328 = 3.4e6 < 2^24
```

fp32 尾数 24 bit，所以**只要累加发生在 fp32 里，任意累加顺序都逐元素精确** ——
与 GPU 的归约树形状无关、与后端无关。`tests/test_dot_exact.py` 在两种硬件上都验证了
边界值（±1664 × ±3）逐元素相等。

这条比看起来重要：flash-attention 之外绝大多数 fp16 GEMM 应用都在容忍误差，
而密码学里一个 bit 都不能错。很多 GPU Kyber 实现因此转用 int16/int32 或做误差补偿；
我们这条路是免费的。

### 4.3 唯一一个**已经实测过两个厂商**的实现

同一份代码（纯 Triton，没有内联汇编、没有厂商专属 intrinsic）：
- Intel Arc B390 / Xe3-LPG / triton-xpu 3.9 → **237 个测试全绿**
- NVIDIA GB10 / sm_121 / triton 3.8 → **237 个测试全绿**

一行内核都没改，只有 tile 配置换了（`devices.gemm_cfg`）。

### 4.4 报的是端到端，而且主机侧真的优化过

论文常用 `cudaEvent` 只圈 kernel，主机侧的 numpy 打包 / H2D / D2H 全被排除。我们两种
都报，而且把主机侧当成一等公民：

- `decaps()` 支持 `torch.Tensor`（零主机拷贝）/ `bytes` / `np.ndarray` / `list[bytes]`，
  主机侧成本从 0.89 ms 降到 0.57 ms（B=4096，Intel）
- H2D 走 pinned 暂存是**自适应**的：第一次调用用 4 MiB 实测两边，只有快 15% 以上才用
  （Intel 上值得：0.50 → 0.28 ms；GB10 上不值得，因为它 pageable 本来就有 50 GB/s）
- 逐条长度校验从 O(B) 改成 O(1)（省 0.13 ms）
- 顺带证明了**这台 Intel 驱动的 H2D 无法和计算重叠**（双流零依赖实测 1.328 ms
  vs 同流串行 1.307 ms），所以「流式双缓冲」这条路直接排除

### 4.5 融合深度

整条流水线 **13 次 kernel launch**（B=4096，ML-KEM-768）。两趟 GEMM 的 epilogue
全部融进 GEMM 内核：

- ENC：`GEMM + mod q + compress(du/dv) + 比较 + 按列归约` → 不落 16.8 MB 的 `acc2`
- DEC：`GEMM + (V−acc) mod q + compress(1bit) + 8 行打包` → 不落 4.2 MB 的 `acc`

同进程配对比值：ENC 融合让设备侧从 2.75 → 1.62 ms，DEC 融合再 1.62 → 1.22 ms
（Intel，B=4096；`fallback/fused` 中位 1.89×）。另外 sponge 的输入准备
（pad10\*1 + 段拼接 + 广播）也压成一次 launch（J 那档 192 µs → 28 µs）。

### 4.6 正确性是可复现地验证过的，而且防了「静默回退」

- NIST ACVP 官方向量逐字节比对（两台机器都过）
- 每个融合内核都有一个「必须与未融合的多步实现**逐字节相等**」的测试 —— 这层防护是
  必需的：融合内核在不满足约束时返回 `False`/`None` 并且不写输出，调用方悄悄回退。
  我们在开发中真的踩过一次（`Y` 是个 stride=(1,768) 的转置视图，导致融合 GEMM 一直
  没执行）。
- `benchmarks/bench_e2e.py` 自带融合-vs-全回退的对照

### 4.7 测量方法学是公开的

- GPU 时钟 0.9 ↔ 2.2 GHz 漂移、`xe` 驱动无调频接口 → 只信同轮配对比值
- 一个被证伪的假设（pinned 中转「不划算」→ 其实划算，因为我把 memcpy 用 numpy 赋值
  测成了 0.25 ms，换成 `copy_` 只有 0.035 ms）
- 一个测量陷阱（patch 错了命名空间，把融合收益测成 0.003 ms）

---

## 5. 我们的缺点

按严重程度排：

### 5.1 只做了 decaps，没有 keygen / encaps

对比对象都是完整的 KEM 实现。我们只覆盖服务端解封装。封装侧（客户端）和密钥生成
直接借用 kyber-py（纯 Python），性能上不具可比性。

### 5.2 单条延迟比 cuPQC 慢 4.6 倍

| | B=1 端到端 |
|---|---:|
| cuPQC（GB10，同机）| **61.4 µs** |
| 我们（GB10） | **282 µs** |
| 我们（Arc B390 核显） | 966 µs |

同机同口径是 **4.6×**。（之前我写"差 3 个数量级"是拿别人的**摊销值**当单条延迟比，
不成立，已更正。）原因是 13 次 kernel launch、每次固定开销 35–40 µs，B=1 时摊不开；
而 cuPQC 在 GB10 上自己也要吃 53 µs 的 launch 开销。

**用在「一次一条」的场景我们明显更差**，这是吞吐导向设计的固有代价。

### 5.3 每把私钥 7.9 MB 显存 + 13.5 ms 预处理

> 补充（同口径实验后）：把重建搬到 GPU 上（`kernels/rebuild.py`）只要 **0.158 ms**
> （GB10），比 CPU 的 13.45 ms 快 85 倍。所以「建表成本压垮它」这个说法要收窄：
> **GPU 上重建只要 0.158 ms，摊到 B=8192 是 19.3 ns/op** —— 完全可以接受。
> 真正的约束变成「每把钥匙 7.9 MB 常驻显存」，而不是时间。

| | 我方 | 标准实现 |
|---|---:|---:|
| `dk` 大小 | 2400 B（不变）| 2400 B |
| 额外常驻显存 | **7.86 MB** | 0 |
| 装载/预处理 | **13.45 ms**（CPU numpy）| ~0 |
| `Â` 的来源 | 预处理时展开成 Toeplitz 矩阵 | 每次由 ρ 经 SHAKE 现算 |

标准的 ML-KEM 实现不存 `Â`，只用 32 字节的 ρ 在需要时 `SampleNTT` 出来。我们把
`Â`、`t`、`s` 全展开常驻。**这直接限定了适用场景**：一把长期私钥 + 海量密文（TLS
服务端、KEM 解封装服务）可以；每连接一把临时密钥就会直接被 13 ms 压死。

### 5.4 ML-KEM-512 走慢路径

`η1 = 3`，而融合 CBD 内核只实现了 `η = 2` 的打包，所以 512 回退到通用 CBD：

| ps | Intel 设备侧 µs/op | GB10 设备侧 µs/op |
|---|---:|---:|
| ML-KEM-512 | 0.932 | 0.760 |
| ML-KEM-768 | **0.307** | **0.132** |

512 比 768 还慢 3 倍。功能是对的（ACVP 过），但没优化。已加测试锁住这条限制。

**逐内核证据**（`bench_kernels.py --ps ML-KEM-512 --batch 4096`，设备侧合计 3.118 ms）：

| kernel | ms/call | 占比 |
|---|---:|---:|
| `at::native::xpu::ReduceKernel` | 0.933 | 29.9% |
| `at::native::xpu::ElementwiseGroupRangeKernel` | 0.542 | 17.4% |
| `fused_enc_kernel` | 0.188 | 6.0% |
| `at::native::xpu::CatArrayBatchedCopy` | 0.171 | 5.5% |
| `sponge_kernel` | 0.154 | 4.9% |
| `_pre_kernel` | 0.123 | 3.9% |
| `fused_dec_kernel` | 0.059 | 1.9% |

前两名（47%）加上 `CatArrayBatchedCopy`、几支 `VectorizedElementwiseKernel` 一共
约 **2.2 ms 全是 PyTorch 原生逐元素/归约算子** —— 也就是通用 CBD 回退路径
（`bits.reshape(n,256,2η)` + 两段 sum + 相减 + `torch.cat`）。两只融合内核
（`fused_enc` + `fused_dec`）加起来只有 0.25 ms，占 8%。所以 512 慢的原因**确认**是
η=3 的 CBD 走了回退，不是 GEMM 或别的。

### 5.5 依赖 fp16 张量核

没有张量核（或被迫关掉）的设备会退化到 `mm(fp32, fp32)` 回退路径，实测慢 2–4×
（DEC 0.20 → 0.44 ms，ENC 0.31 → 1.13 ms）。这不是「能在任何 GPU 上跑」那么强的
可移植性 —— 是「任何有 fp16 张量核的 GPU」。

### 5.6 没有形式化的常数时间保证

- 时序不依赖密文内容（GEMM + `torch.where` 选择，都是数据无关的）
- 但**没有跑过 dudect / ctgrind / Valgrind-based 泄漏测试**，也没做过「缓存侧信道」
  分析。多把私钥同时常驻时，7.9 MB/把的权重在 L2 里的争用是否会泄漏密钥相关性，
  完全没研究过。
- 批量大小会影响时间（这不是我们特有的，但也没做恒定化）

### 5.7 主机侧仍占 29%（Intel）

B=4096 端到端 1.99 ms 里主机侧 0.57 ms：`b"".join` 0.25 + H2D 0.28 + D2H 0.03。
已经压到「`list[bytes]` 接口」形态的下限了；要再降只能让调用方直接给整块缓冲
（可降到 ~0.09 ms，但那是改 API 而不是改实现）。

### 5.8 没在数据中心卡上跑过，也没测过 AMD

对比对象用的都是 A100 / RTX 3090 / H100 这个级别。我们手上只有：
- 一块**核显**（96 EU，29 GB 共享内存）
- 一块 **GB10**（48 SM 的消费级模组）

所以本文所有跨实现的绝对值比较都带着这个不公平。

### 5.9 没有同卡同口径的 NTT 对照 —— 所以「我们的方法更快」这句话说不得

我们能证明的只有：
- 在 GB10 上 8.0 M decaps/s、Arc 核显上 3.26 M/s（实测，ACVP 校验过）
- 这超过了同机 liboqs 16 线程 3.6–6.4 倍

**不能证明的**：我们比一个同样用张量核的 NTT 实现快。ConvKyber 在 RTX 3080 上的
完整 decaps（推算 1.34 M/s）比我们 GB10 的 8.0 M/s 低 6 倍，但那张卡更小、而且它的数
是推算的、算法是 Kyber v3。三个变量同时不同，结论不成立。

要真正回答这个问题，唯一干净的做法是：**把 ConvKyber 的整数张量核 NTT 和我们的
fp16 Toeplitz GEMM 放在同一块卡上、跑同一批 ACVP 密文、同一套计时**。
这件事没人做过 —— 包括我们。

### 5.10 GEMM 只到峰值的 24–28%

非 GEMM 阶段吃掉了大头：Keccak 16%、`_pre` 17%、CBD→Y/E 16%、padcat 8%。
VTune 早就指出 `XVE Array Stalled/Idle 75.1%`、occupancy 28.5%。

---

## 6. 可比性警告（读任何对比表之前先看这一节）

1. **硬件不同。** 我们是核显 96 EU 和 48-SM 的 GB10；论文是 A100/H100/RTX 3090。
   跨硬件的绝对值比较没有意义，只能比「同一硬件上相对 CPU 基线的倍数」或「占张量核
   峰值的百分比」。
2. **批大小完全不同。** 他们报单条延迟（batch=1），我们报 B=4096~8192 的吞吐。
   同一份代码在 B=1 和 B=8192 上能差 **3000 倍**（Intel：966 → 0.31 µs/op）。把我们的
   0.31 µs/op 和他们单条的 10 µs 比是**误导**。
3. **计时口径不同。** 他们通常 `cudaEvent` 只圈 kernel；我们区分「设备侧」和「端到端」。
   我们的设备侧口径更接近他们，但仍然包含 Python 侧 13 次入队（在 B 大时被设备工作掩盖）。
4. **算法阶段不同。** 只有 decaps。完整的 KEM 还有 keygen 和 encaps。
5. **Kyber768 vs ML-KEM-768。** 核心数学一样，但 FIPS 203 的解封装多一步隐式拒绝
   哈希（`J(z‖c)`）和常数时间选择。我们实现的是 ML-KEM。
6. **我们的绝对值有 ±20% 抖动。** GPU 时钟 0.9 ↔ 2.2 GHz 漂且无法锁定；只信同轮比值。
7. **「加速比」比的是什么基线？** 论文常拿 AVX2/AVX-512 单核实现做基线，我们用的是
   liboqs 0.16.0 的 1 线程（10.26 µs）和 16 线程（1.45 µs）。不同基线不同结论。
8. **他们有 NTT，我们没有。** 这不是「谁优化得好」，是**两种不同的算法组织方式**。
   他们的实现可以直接用于 keygen/encap（ρ 每次变），我们不能。
