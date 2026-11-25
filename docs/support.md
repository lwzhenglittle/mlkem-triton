# Supported backends, tuning knobs and layout

---

## 1. Support matrix

| backend | status | notes |
|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, 264 tests green |
| NVIDIA CUDA | ✅ **measured** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8, 264 tests green |
| AMD ROCm | ⚠️ **untested** | code-level reasoning only |
| Apple MPS | ❌ | no Triton backend |
| CPU | ❌ | no usable Triton CPU target |

The CUDA row was **actually run on that machine**: same code, same ACVP vectors, 264 tests
green, without changing a single kernel. Only the tile configuration differs. ROCm remains an
inference from the code and is not claimed as working.

With no GPU at all, `pytest tests` skips the device-dependent cases (currently 90 passed /
174 skipped), so a CPU-only machine can still smoke-test the maths and reference paths.

### Cross-backend details that were verified, not assumed

- `tl.dot` with fp16 inputs and fp32 accumulation is **element-wise exact** on both backends
  (`test_dot_exact.py`), which is what makes the numerical argument in
  [docs/design.md](design.md) §2 portable rather than device-specific.
- The sponge kernel holds a whole Keccak state per thread (25 × uint64). On Intel that compiles
  to 96 registers and 0 spill; on CUDA 0 spill.
- 64-bit rotates, `tl.interleave`, `tl.trans` and `tl.atomic_add` behave identically.
- The tile configurations differ between backends (`devices.gemm_cfg`): Intel is constrained by
  256 GRF per thread and must keep the accumulator small; CUDA can afford `BK=64` with
  `num_stages=3`.

---

## 2. Tuning knobs

### GEMM tiles — `devices.gemm_cfg(dev)`

These are per-backend and must satisfy divisibility constraints (`M % BM == 0` and friends), so
they are not exposed as free-form environment variables.

| backend | BM | BN | BK | GM | num_warps | num_stages |
|---|---:|---:|---:|---:|---:|---:|
| `xpu` (measured optimum at B=4096) | 128 | 128 | 32 | 4 | 8 | 2 |
| `cuda` | 128 | 128 | 64 | 8 | 8 | 3 |
| `hip` | 128 | 128 | 64 | 8 | 8 | 2 |
| fallback (unlisted backend) | 64 | 64 | 32 | 8 | 4 | 2 |

**Only the `xpu` row was swept on real hardware.** The `cuda` and `hip` rows are conservative
values from public experience: they run correctly and the test suite passes on CUDA, but their
values were not tuned there. The Intel constraint is `BM*BN/(16*num_warps) <= 128` — beyond 128
fp32 accumulators per thread the kernel spills heavily. `num_stages=2` is optimal on Intel
because on-chip memory is small.

### Small kernels — `devices.warps(name, default)`

`num_warps` for the sponge, padcat, CBD and packing kernels lives at the top of each module.
These **were only swept on Intel Xe3**; the other backends reuse the same values, which is
defensible because small kernels are not very sensitive to `num_warps`, but there is **no
CUDA or ROCm tuning data** and this document does not pretend otherwise. A single default plus
an override hook is the honest arrangement.

```bash
MLKEM_WARPS="shake=8,ntt=4" python benchmarks/bench_e2e.py
```

Note that the NTT wants **2** warps, not 4 — 2 is 1.8× better (222 vs 397 ns/key, measured as a
paired same-process ratio).

### Device selection

`MLKEM_DEVICE` picks a backend when none is passed explicitly; the priority is explicit
argument > `MLKEM_DEVICE` > `cuda` > `xpu`. There is no CPU backend, because Triton has no
usable CPU target, so with no accelerator present `resolve_device` raises.

---

## 3. Project structure

```
mlkem_triton/
├── devices.py             backend detection, per-backend tile config, num_warps override
├── params.py              FIPS 203 parameter sets
├── keyprep.py             private key -> W_dec / W_enc (numpy, host side)
├── ntt_ref.py             numpy reference for NTT / INTT / Toeplitz
├── pipeline.py            the main pipeline (fully device-resident)
├── pipeline_host_hash.py  comparison implementation (hashing on CPU), for baselines
├── ops.py                 Triton kernels wrapped as torch.library custom ops
├── gpu_hash.py            thin numpy wrapper over the batch sponge
├── refimpl.py             kyber-py reference wrapper (optional dependency)
├── acvp.py                NIST ACVP vector parsing
└── kernels/
    ├── keccak.py          Keccak constant tables
    ├── shake.py           sponge kernel (code-generated)
    ├── glue.py            compress/pack, Y/E assembly, CBD direct write
    ├── padcat.py          pad10*1 + segment concatenation (one launch)
    ├── fused_gemm.py      the two fused GEMM passes (ENC / DEC)
    ├── gemm_modq.py       generic Triton GEMM (comparison path)
    ├── ntt.py             forward NTT, pointwise multiply, ByteEncode12
    ├── rebuild.py         on-GPU key-material rebuild (measurement tool)
    ├── encaps.py          encaps-specific kernels and packing
    └── elementwise.py     elementwise operators
```

The decapsulation dataflow is diagrammed in [docs/design.md](design.md) §3.

---

## 4. Adding a backend

1. Add a row to `_GEMM_CFG` in `devices.py`, or rely on `_FALLBACK_CFG` first to see whether the
   thing runs at all.
2. Run `pytest tests/ --device <dev>`. The suite is written to be backend-agnostic: every fused
   kernel is checked against its unfused counterpart byte-for-byte, and `test_dot_exact.py`
   verifies the fp16 exactness bound on the actual hardware rather than trusting it.
3. Run `python benchmarks/bench_kernels.py --batch 4096` to see which kernel is out of line.
   A backend whose `fused_enc_kernel` share differs wildly from the numbers in
   [docs/benchmarks.md](benchmarks.md) §5 usually means a tile configuration problem.
4. Only then sweep tiles. Report both the device-side and end-to-end figures, and state the
   clock conditions — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the measurement discipline
   this project requires.

Do not add CI: GitHub Actions has no GPU runners, so a workflow would burn quota to test
nothing. The test suite skips cleanly without a GPU, which is the useful property.
