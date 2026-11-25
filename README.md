# mlkem-triton

**用 Toeplitz 矩阵乘替代 NTT 的 ML-KEM（FIPS 203）实现，纯 Triton。**

批量解封装 / 封装 / 密钥生成。没有内联汇编、没有厂商专属 intrinsic：多项式乘法
折叠成固定矩阵，直接喂给 fp16 张量核 —— 除了 keygen（它必须用 NTT）。

> ### 关于命名：这里实现的是 ML-KEM，不是 Kyber round-3
>
> **ML-KEM（FIPS 203）** 是 NIST 在 2024 年标准化的方案，源自 CRYSTALS-Kyber。
> 两者有两处**不兼容**的差别：FIPS 203 去掉了 encaps 末尾的 KDF（`K` 直接是
> `G(m‖H(ek))` 的前 32 字节），keygen 的种子扩展用 `G(d ‖ k)` 而不是 `G(d)`。
>
> 本项目的全部正确性证据都来自 ML-KEM 的 NIST ACVP 向量，**没有任何一条 Kyber
> KAT**。仓库名叫 `mlkem-triton` 只是历史原因，不代表与 Kyber round-3 互通。

```python
from mlkem_triton import Decapsulator, Encapsulator, generate_keypairs

dec = Decapsulator("ML-KEM-768", dk)  # 私钥装到 GPU 上一次
keys = dec.decaps(ciphertexts)  # list[bytes] / np.ndarray -> [B, 32] uint8

enc = Encapsulator("ML-KEM-768", ek)  # 公钥装到 GPU 上一次
cts, keys = enc.encapsulate(msgs)  # 每条 32 字节随机消息 -> [B, 1088] / [B, 32]
```

三个操作都有了，全设备常驻：

| 操作 | 正确性 | 每 op（GB10，设备侧）|
|---|---|---:|
| `decaps` | ACVP 3×10 条，密文/密钥逐字节 | **125 ns** |
| `encaps` | ACVP 3×25 条，密文/密钥逐字节 | **88 ns** |
| `keygen` | 与 kyber-py **逐字节对拍**（ek+dk）| **469 ns** |

**keygen 比另外两个慢 4~5 倍，而且比 cuPQC 慢 2.7 倍** —— 见文末「限制」。

在 Intel Arc B390 上，768 参数集、B=4096：**约 0.45 µs/次**，比 liboqs 单线程快约 23×，比 liboqs 16 并发进程快约 3.3×

---

## 它为什么快

ML-KEM 的绝大部分算力花在多项式乘法上，常规做法是 NTT。这个项目的做法是反过来：

**对固定矩阵做一次预处理，然后让每次解封装的乘法变成一次 GEMM。**

对 `a ∈ Z_q[X]/(X^256+1)` 定义 Toeplitz 矩阵

```
T(a)[i][j] = a[(i - j) mod 256] · (+1 若 i >= j 否则 -1)
```

则对任意 `b`：

```
T(a) @ b  ==  a · b   在 Z_q[X]/(X^256+1) 中
```

（证明与验证见 `tests/test_ntt.py`。）`keyprep` 在装载私钥时把 `A`、`t`、`s` 全部展开成这样的矩阵，之后每次解封装只需要两趟 GEMM：

| 步骤 | 形状 |
|---|---|
| 解密 `w = v − W_dec·u` | `[256,768] × [768,B]` |
| 重加密 `[u';v'] = W_enc·y + [e1; e2+1665m']` | `[1024,768] × [768,B]` |

代价是权重矩阵比多项式大 256 倍（约 1.5 MB，一次装好常驻），换来的是一次访存友好的稠密 GEMM。

### 为什么 fp16 tensor core 也能精确

```
|W| <= 1664 < 2^11,   |y| <= 2,   K = 768
=>  每个部分和  <  768 × 3328 = 2.56e6  <  2^24
```

fp32 的尾数是 24 bit，所以只要**累加发生在 fp32 里，任意累加顺序都逐元素精确** —— 和 GPU 的归约顺序无关，跨后端成立。`tl.dot(..., out_dtype=tl.float32)` 保证这一点。

反过来说，**输出不能用 fp16**：`|sum|` 峰值约 1.8e5，超过 fp16 上限 65504，会直接变 inf。这也是为什么两趟 GEMM 都用本项目自己的 Triton 内核，而不是 `torch.mm`。

---

## 安装

```bash
pip install mlkem-triton
```

依赖：`torch>=2.4`、`triton>=3.0`、`numpy`。不需要编译任何 C/C++。

想要「随机生成密钥对」这类测试辅助（`mlkem_triton.refimpl`）：

```bash
pip install "mlkem-triton[ref]"     # 装 kyber-py
```

正确性主 gate 用 `data/acvp/` 里的 NIST 官方测试向量，**不需要** kyber-py。

---

## 用法

```python
import numpy as np
from mlkem_triton import Decapsulator

dec = Decapsulator("ML-KEM-768", dk)  # device=None -> 自动选 cuda / xpu

# 一批密文（每条 1088 字节）
keys = dec.decaps(ciphertexts)  # -> np.ndarray [B, 32] uint8

# 也可以给一个 [B, 1088] uint8 数组
keys = dec.decaps(arr)

# 指定设备
dec = Decapsulator("ML-KEM-768", dk, device="cuda:1")
```

`decaps()` 内部做：numpy 打包 → 一次 H2D → 设备侧全流程 → 一次 D2H。
只有 32 字节 × B 从设备回到主机。

如果调用方自己管理设备张量，用 `dec.decaps_torch(ct[B, ct_size], B)`，输入输出都是 `torch.Tensor`，中间不会同步。

**批量越大越划算。** 单条 32 字节的输出、加上一次 kernel launch 的固定开销，在 B < 64 时主导总时间；B >= 1024 才能吃满。

---

## 性能

ML-KEM-768，min-of-25，**含 numpy 打包、H2D、D2H**。

**NVIDIA GB10（DGX Spark，sm_121）/ torch 2.14 / triton 3.8**

| B | 端到端 min | 端到端 med | 仅设备侧 | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.390 ms | 0.275 ms | 0.374 | 2.67 | 3.9× |
| 2048 | 0.493 ms | 0.501 ms | 0.315 ms | 0.241 | 4.15 | 6.0× |
| 4096 | 0.869 ms | 0.876 ms | 0.533 ms | 0.212 | 4.71 | 6.8× |
| 8192 | 1.665 ms | 1.678 ms | 1.018 ms | 0.203 | 4.92 | 7.1× |

**Intel Arc B390（Xe3-LPG，96 EU）/ torch 2.15 / triton-xpu 3.9**

| B | 端到端 min | 端到端 med | 仅设备侧 | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.919 ms | 1.090 ms | 0.665 ms | 0.898 | 1.11 | 1.61× |
| 2048 | 1.242 ms | 1.431 ms | 0.769 ms | 0.607 | 1.65 | 2.39× |
| 4096 | 1.943 ms | 2.196 ms | 1.237 ms | 0.474 | 2.11 | 3.06× |
| 8192 | 3.674 ms | 4.277 ms | 2.227 ms | 0.449 | 2.23 | 3.23× |

对照基线（本机实测，同一批密文，liboqs 0.16.0，ML-KEM-768）：

| | µs/op | 相对本库 Intel（B=8192） | 相对本库 GB10（B=8192） |
|---|---:|---:|---:|
| liboqs 单线程 | 10.26 | 22.9× | 50.5× |
| liboqs 16 线程 | 1.45 | 3.23× | 7.14× |

（liboqs 的数字取自同一台机器、同一批 ACVP 密文的基准。）

> 注意：GPU 时钟在不同会话之间会漂 2–3 倍，所以**只有同进程内交错的 A/B 才有意义**。
> 这里的 `vs liboqs` 是跨进程的参考值，看量级就好；同一份代码在不同会话复测，绝对值
> 可能差 10–20%。

### 设备侧时间花在哪（B=4096）

`python benchmarks/bench_kernels.py --batch 4096`：

| kernel | 次数 | Intel ms | 占比 | GB10 ms | 占比 |
|---|---:|---:|---:|---:|---:|
| `fused_enc_kernel` | 1 | 0.290 | 29.9% | 0.116 | 36.4% |
| `cbd_ye_kernel` | 2 | 0.159 | 16.4% | 0.065 | 20.4% |
| `_pre_kernel` | 2 | 0.170 | 17.5% | 0.047 | 14.9% |
| `sponge_kernel` | 4 | 0.160 | 16.5% | 0.035 | 10.9% |
| `fused_dec_kernel` | 1 | 0.093 | 9.5% | 0.030 | 9.5% |
| `padcat_sfx_kernel` | 2 | 0.054 | 5.6% | 0.012 | 3.9% |
| `padcat2_kernel` | 2 | 0.023 | 2.4% | 0.005 | 1.7% |
| 其它 | — | 0.023 | 2.4% | 0.008 | 2.6% |
| **合计** | | **0.972** | | **0.318** | |

整条流水线一共 **13 次 kernel launch**。

### 融合带来的收益（同进程交错 A/B）

| 改动 | 效果 |
|---|---|
| DEC 那趟：`GEMM + mod q + compress + 打包` 融成一个内核 | 设备侧 1.616 → 1.222 ms（**1.32×**）|
| ENC 那趟：`GEMM + mod q + compress + 比较 + 归约` 融成一个内核 | 再省 1.14 ms |
| 两个融合都开 vs 都不开（端到端） | **1.55×** |

---

## 支持矩阵

| 后端 | 状态 | 备注 |
|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **实测通过** | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9，231 个测试全绿 |
| NVIDIA CUDA | ✅ **实测通过** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8，231 个测试全绿 |
| AMD ROCm | ⚠️ **未实测** | 同上 |
| Apple MPS | ❌ | 没有 Triton 后端 |
| CPU | ❌ | 没有可用的 Triton CPU target |

CUDA 那栏是**真的在这台机器上跑过**的：同一份代码、同一批 ACVP 向量、231 个测试全绿，
不需要改一行内核。ROCm 仍然只是代码级推断。

已经验证过的跨后端细节：

- `tl.dot` fp16 输入 / fp32 累加在两种硬件上都**逐元素精确**（`test_dot_exact.py`）。
- sponge 内核每个线程持有一个完整 Keccak 状态（25 × uint64）。Intel 上编译出 96 寄存器 / 0 spill，CUDA 上 0 spill。
- 64 位旋转、`tl.interleave`、`tl.trans`、`tl.atomic_add` 的行为一致。
- 两个后端的 tile 配置不同（`devices.gemm_cfg`）：Intel 受 256 GRF 约束必须压小累加器，CUDA 可以用 `BK=64` + `num_stages=3`。

---

## 调优参数

GEMM 的 tile 配置在 `devices.gemm_cfg()`（按后端选，且必须满足 `M % BM == 0`
这类整除约束）。小内核的 `num_warps` 在各自模块顶部，可以用环境变量临时覆盖：

```bash
MLKEM_WARPS="shake=8,ntt=4" python benchmarks/bench_e2e.py
```

**只有 Intel Xe3 上扫过这些值**，CUDA/ROCm 上沿用同一组（小内核对此不敏感），
没有做过逐后端调优 —— 请不要把文档读成「每后端都调过」。

## 测试

```bash
pytest                          # 最推荐；无 GPU 时 GPU 用例自动 skip
pytest --device cuda            # 指定设备
ruff check . && ruff format --check .
```

**没有 GPU 时 `pytest tests` 会自己 skip 掉需要显卡的用例**（只跑数学/参考实现那
部分，目前是 90 passed / 174 skipped），所以纯 CPU 机器上也能做冒烟测试。
贡献流程与**测量纪律**见 `CONTRIBUTING.md`。

测试分三层：

1. **纯数学**（`test_ntt.py`、`test_keyprep.py`、`test_dot_exact.py`）—— Toeplitz 等式、NTT 自洽、fp16 累加的精确性边界。不需要 GPU 也能跑一部分。
2. **内核逐位对齐**（`test_fused_gemm.py`、`test_glue.py`）—— 每个融合内核必须和「未融合的多步实现」**逐字节相等**。这是防「静默回退」的关键：融合内核在不满足约束时会返回 `False`/`None` 并且**不写任何输出**，调用方回退；如果哪天判定条件写错，这一层会立刻红。
3. **端到端**（`test_pipeline*.py`）—— NIST ACVP 官方向量 + 隐式拒绝 + 各批量大小。

---

## 项目结构

```
mlkem_triton/
├── devices.py            后端探测 + 每后端 tile 配置
├── params.py             FIPS 203 参数集
├── keyprep.py            私钥预处理 -> W_dec / W_enc（numpy，纯 CPU）
├── ntt_ref.py            NTT / INTT / Toeplitz 的 numpy 参考实现
├── pipeline.py           主流水线（全设备常驻）
├── pipeline_host_hash.py 对照实现（哈希在 CPU 上做，用来做基准比较）
├── ops.py                把 Triton 内核包成 torch.library 自定义算子
├── gpu_hash.py           批量 sponge 的 numpy 薄封装
├── refimpl.py            kyber-py 参考实现封装（可选依赖）
├── acvp.py               NIST ACVP 向量解析
└── kernels/
    ├── keccak.py         Keccak 常量表
    ├── shake.py          sponge 内核（代码生成）
    ├── glue.py           压缩/打包、Y/E 组装、CBD 直写
    ├── padcat.py         pad10*1 + 段拼接（一次 launch）
    ├── fused_gemm.py     两趟融合 GEMM（ENC / DEC）
    ├── gemm_modq.py      通用 Triton GEMM（对照路径）
    └── elementwise.py    逐元素算子
```

`pipeline.py` 的流程：

```
ct ──▶ _pre (unpack+decompress+transpose) ──▶ U, V + 参考值
                                              │
V, U ──▶ fused_dec_gemm ──▶ m_bytes ──▶ padcat2 ──▶ sponge(G) ──▶ K', r'
                                     │
ct ──▶ padcat2 ──▶ sponge(J) ──▶ K̄
                                     │
r' ──▶ padcat_sfx ×2 ──▶ sponge(PRF) ×2 ──▶ cbd_ye ×2 ──▶ Y, E
                                                          │
                              fused_enc_gemm ◀────────────┘
                                     │
                              cnt == 768+256 ? ──▶ K' 或 K̄
```

---

## 限制与说明

- **keygen 是唯一输给 cuPQC 的一项。** decaps/encaps 的乘法被折成「跨操作固定的
  矩阵」的 GEMM，Toeplitz 技巧成立；但 keygen 的 `s`/`e` 每把密钥都是新的，没有可复用
  的矩阵，而且 `ek` 里必须放 `t̂` 的 NTT 域表示 —— 所以 NTT 躲不掉。
  GB10 上 469 ns/key，cuPQC 是 174 ns（**我们慢 2.7 倍**）。
  `kernels/ntt.py` 里那个 NTT 调过两轮才从 15.2 ms 降到 1.5 ms（见对比文档 §0.1d）。
- **ML-KEM-512 走的是慢路径。** 它的 `η1 = 3`，而 `cbd_ye` 那个融合 CBD 内核只实现了
  `η = 2` 的位打包（一个字节恰好两个系数）。512 会回退到通用 CBD 路径，B=4096 上
  设备侧是 **0.93 µs/op（Intel）/ 0.76 µs/op（GB10）**，而 768 是 0.31 / 0.13。
  功能正确（ACVP 向量通过），只是没优化。768 和 1024 都在快路径上。
- **每个私钥要 7.9 MB 显存和 13 ms 预处理。** `keyprep` 把 `Â`、`t`、`s` 全部展开成
  Toeplitz 矩阵常驻显存（768：`W_dec` 256×768 + `W_enc` 1024×768，fp16 共 7.86 MB），
  建一次 13.45 ms（CPU 上 numpy）。**标准实现只存 2400 字节的 `dk`，`Â` 每次由 ρ 现算。**
  所以这个设计只在「一把长期私钥、海量密文」下成立；如果每连接一把临时密钥，
  13 ms 的建表成本会直接压垮它。
- **批量要大。** 见上面「用法」。
- **Toeplitz 替代 NTT 不是本项目的发明。** 把格密码的乘法写成矩阵乘、丢给 tensor core 是已有文献里的做法。本项目的贡献在工程侧：fp16 精确性论证、两趟 GEMM 的 epilogue 融合、sponge/padcat/CBD 的融合，以及在真实硬件上把这条路径调通并量化。
- **常数时间**只保证到「批量形状固定」这一层。批量大小会影响时间，单条密文的内容不会。
- **没有手写 SYCL 内核。** 开发这套东西的原型在 Intel 上还带一个手写的 SYCL Keccak
  （只覆盖 sponge，比 Triton 版快 2.2–2.6×）。那个内核是 Intel 专属的、要走子进程
  调 `icpx` 构建，让整个库失去可移植性，所以在开源版里被去掉了。代价实测：sponge
  占比从 10.9% 涨到 16.5%，整机端到端慢约 20%。如果你只跑 Intel 且在意这 20%，
  那个内核在原仓库里。
- **`kernels/rebuild.py` 是测量工具，不是生产路径。** 它可以在 GPU 上从 `dk` 完整重建
  `W_dec`/`W_enc`（SHAKE-128 → 拒绝采样 → INTT → Toeplitz 展开，0.158 ms @ GB10），
  用来和 NVIDIA cuPQC 那种「无状态、每 op 重算 Â」的 API 做同口径对比。
  主路径用的是 `keyprep` 的预展开版本。
- **`torch.mm` 的 `out_dtype=torch.float32` 不能用。** 它是 torch-xpu 的私有重载
  （`aten::mm.dtype`），公开签名只有 `out=None`。而 fp16 输出在这里会溢出成 inf。
  所以两趟 GEMM 都是本项目自己的 Triton 内核 —— 顺带还比 `torch.mm` 快（DEC 那趟
  `0.200 + 0.070` → `0.111` ms）。

## 许可

Apache-2.0。ML-KEM 本身是 NIST 公开标准；ACVP 测试向量来自 [usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server)（美国政府作品，公有领域）。
