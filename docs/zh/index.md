# 中文原始文档（附加内容 / supplementary）

这个目录保存的是本项目长文档的**中文原始版本**，作为附加材料保留。

**英文版是权威版本**，中文版不保证与英文版同步：

| 中文原文 | 英文权威版 |
|---|---|
| [readme.md](readme.md) | [`/README.md`](../../README.md) |
| [contributing.md](contributing.md) | [`/CONTRIBUTING.md`](../../CONTRIBUTING.md) |
| [changelog.md](changelog.md) | [`/CHANGELOG.md`](../../CHANGELOG.md) |
| [comparison.md](comparison.md) | [`/docs/compare.md`](../compare.md) |

英文版在改写时做了**删减**（去掉了重复叙述，把过长的表格下沉到 `docs/benchmarks.md`），
所以两边篇幅不同，中文版更全、英文版更准。

---

## 已知与英文版冲突的地方

下面几处中文原文的陈述**已经被后续工作推翻**，英文版已更正，中文版为了保持"原始记录"
的性质没有改动。读到这些段落时请以英文版为准：

| 位置 | 中文原文的说法 | 实际情况 |
|---|---|---|
| `readme.md` 支持矩阵 | 「231 个测试全绿」 | 现在是 **812** 个测试（0.7.2） |
| `comparison.md` §4.3 | 「237 个测试全绿」 | 现在是 **812** 个测试（0.7.2） |
| `comparison.md` §5.1 | 「只做了 decaps，没有 keygen / encaps」 | **三个操作都已实现并通过 ACVP / 逐字节对拍**；真正的短板是 keygen 比 cuPQC 慢 2.7 倍（见英文版 §5.1） |
| `comparison.md` §0.1d 结尾 | 「需要把 SampleNTT → NTT → pow_mac 融成一条流水」 | 融合的收益上界已实测为 **81 ns/key（1.21×）**，做完也只能到 cuPQC 的 0.45×，因此**决定不做**（见英文版 §0.4 与 `CHANGELOG.md`） |

另外中文原文里指向 `results/OPTIMIZATION_LOG.md`、`results/keccak_gpu_feasibility.md`
的引用已失效 —— `results/` 目录在拆分英文文档时被并入 `docs/`。

---

## 为什么保留中文版

- 它是这个项目的**原始实验记录**，包含若干英文版删掉的细节（例如 ML-KEM-512 的 Intel 逐
  内核分解表）。
- 里面记录的**被证伪的假设**（17× Keccak 的跨会话比较、"Triton 天花板 8.3 TF"、pinned
  中转"不划算"、单条延迟"差 3 个数量级"）是这份文档诚实性的一部分，删掉可惜。

如果中英文不一致，**以英文版为准**。
