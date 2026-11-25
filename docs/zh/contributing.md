> **附加内容 / supplementary material.** 这是中文原始版本，**英文版是权威版本**：`/CONTRIBUTING.md`
> 已知冲突与失效引用见 [index.md](index.md)。

---

# 贡献指南

## 环境

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,ref]"
```

`[ref]` 会装 `kyber-py`，只有一部分测试要用它（随机生成密钥对、做逐字节对拍）。

硬件要求：

- **跑全部测试**：需要一块有 Triton 后端的 GPU（NVIDIA / AMD / Intel）。
  没有 GPU 时 `pytest tests` 仍然能跑（90 passed / 174 skipped），GPU 用例由
  `tests/conftest.py` 集中 skip 掉。**这个项目没有 GPU CI** —— 测试都是在真机上
  手动跑的，所以 PR 里请写明你跑过哪些硬件。
- **只是看看代码 / 跑参考实现**：纯 CPU 就够，`mlkem_triton.ntt_ref` / `keyprep`
  是纯 numpy。

## 测试

```bash
pytest                      # 全部
pytest --device cuda        # 指定后端
pytest tests/test_glue.py   # 单文件
ruff check . && ruff format --check .
```

三层：

1. **纯数学**（`test_ntt.py`、`test_keyprep.py`、`test_dot_exact.py`）——
   Toeplitz 等式、NTT 自洽、fp16 累加的精确性边界。
2. **内核逐位对齐**（`test_fused_gemm.py`、`test_glue.py`、`test_encaps.py`、
   `test_keygen.py`、`test_rebuild.py`）—— 每个融合内核必须和「未融合的多步实现
   逐字节相等」。**这一层是防静默回退的关键**：融合内核在不满足约束时会返回
   `False`/`None` 且不写任何输出，调用方回退；判定条件写错时这一层立刻红。
   （开发过程中真的踩过一次：`Y` 是个转置视图导致融合 GEMM 一直没执行。）
3. **端到端**（`test_pipeline*.py`）—— NIST ACVP 官方向量 + 隐式拒绝 + 各批量。

## 加一个新后端

只改 `mlkem_triton/devices.py`：

1. `available_devices()` / `_check_available()` 里加上该设备类型；
2. `_GEMM_CFG` 里给一组 tile 配置（必须同时满足 `M % BM == 0`、`K % BK == 0`
   这类整除约束，见 `fused_gemm._pick_cfg`）；
3. `sync()` 里加同步分支；
4. 小内核的 `num_warps` 默认值在各自模块顶部，可以用
   `MLKEM_WARPS="shake=8,ntt=4"` 临时覆盖，扫完再写回默认值。

**请在 PR 里写明实际测过的硬件型号**，不要提交「理论上应该更快」的配置。

## 测量纪律（这个项目吃过亏，务必遵守）

- **只信同进程交错 A/B 的比值**。GPU 时钟在不同会话之间能差 2~4 倍
  （本机 Intel 核显空载 0.9 GHz、满载 2.2 GHz，而 `xe` 驱动不暴露调频接口）。
- 报 **min-of-N**，不要报平均值。本项目的 `_pre` 内核有 26 倍冷启动离散，
  用平均值算出来的占比排名和按 min 算的**完全相反**。
- **先确认优化路径真的执行了**。融合内核会静默回退，跨进程比较会把时钟噪声
  当成收益。
- 用 `benchmarks/` 里的脚本，它们自带「融合 vs 全回退」的一致性自检。
- 基准数字要附上：硬件型号、软件版本、批大小、计时口径（纯 kernel / 含 host）。

## 提交

- 跑 `ruff format .`
- 它会让 diff 变大，但请单独一个 commit
- 性能改动请附上改动前后的配对比值，以及跑在什么硬件上
