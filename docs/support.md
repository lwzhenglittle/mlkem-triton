# Supported backends, tuning knobs and layout

---

## 1. Support matrix

### Schemes

| backend | ML-KEM (FIPS 203) | ML-DSA verify + sign (FIPS 204) | SLH-DSA verify (FIPS 205) | LMS/HSS verify (SP 800-208) | Falcon verify (FIPS 206 draft) | notes |
|---|---|---|---|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | ✅ **measured** | ✅ **measured** (all 12) | ✅ **measured** (both SHA-256 families, ACVP) | ✅ **measured** (Falcon-512 & 1024, unpadded + padded) | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, **1235 tests** (cold cache: 1232 passed in 1.1.0 baseline + 3 compat/harness checks; warm cache skips 1 compile-cache test) |
| NVIDIA CUDA (`sm_75+`) | ✅ **measured** | ✅ **measured** | ✅ **measured** (all 12) | ✅ **measured** (both SHA-256 families, ACVP) | ✅ **measured** (Falcon-512 & 1024, unpadded + padded) | DGX Spark GB10 / `sm_121` / torch 2.14+cu130 / triton 3.8, **1222 passed + 10 skipped** (9 optional full-SLH-DSA liboqs + 1 compile-cache), 0 failed; representative 16-workload suite also passes on A100 (`sm_80`), RTX 4090 (`sm_89`), and H100 (`sm_90`) on torch 2.15 + triton 3.9 |
| AMD ROCm | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | code-level reasoning only |
| Apple MPS | ❌ | ❌ | ❌ | ❌ | ❌ | no Triton backend |
| CPU | ❌ | ❌ | ❌ | ❌ | ❌ | no usable Triton CPU target |

### Scheme-agnostic primitives

Same convention: ✅ means *measured on that backend*. All 11 primitive families are now verified
and benchmarked on both Intel XPU and NVIDIA CUDA (`benchmarks/local-gpu-vs-cpu-20261128.md` and
`benchmarks/cuda-gb10-vs-local-cpu-20261128.md`).

| primitive | what | XPU | CUDA | verified against |
|---|---|---|---|---|
| `primitives.sponge` / `keccak` | SHA3-224/256/384/512, SHAKE128/256, any rate | ✅ | ✅ | NIST CAVP, numpy reference |
| `primitives.sha2` | SHA-256 / SHA-512 compression | ✅ | ✅ | ACVP via the SLH-DSA/LMS suites |
| `primitives.ntt` | negacyclic NTT, `(q, N, ζ, layers)`, N<=1024 | ✅ | ✅ | per-domain numpy, schoolbook convolution |
| `primitives.ntt.large` | multi-pass NTT up to 2^20 (45% copy BW on B390; 0.20 ms / 181% DRAM copy BW in L2 on GB10) | ✅ | ✅ | `NTTDomain` at sizes both can do, round trip, negacyclic product |
| `primitives.fields` | 32-bit Montgomery product, prime-field | ✅ | ✅ | numpy twin, cross-module call test |
| `primitives.chacha20` | ChaCha20 (RFC 8439) | ✅ | ✅ | RFC 8439 §2.3.2 / §2.4.2, OpenSSL |
| `primitives.poly1305` | Poly1305 + ChaCha20-Poly1305 AEAD | ✅ | ✅ | RFC 8439 §2.5.2 / §2.8.2, OpenSSL EVP_MAC |
| `primitives.poseidon2` | Poseidon2-BabyBear/16 | ✅ | ✅ | Plonky3 0.8 (its own `kat` output) |
| `primitives.aes` | AES-128-CTR, T-table (a measurement, not a fast path) | ✅ | ✅ | FIPS-197 C.1, `openssl enc -aes-128-ctr` |
| `primitives.gemm` | exact `int8 x int8 -> int32` | ✅ | ✅ | numpy, and a bit-plane GF(2^8) product |
| `primitives.blake3` | BLAKE3 (tree hash) | ✅ | ✅ | the official Rust implementation's 26 KAT vectors |

Both the Intel XPU (`Arc B390`) and NVIDIA CUDA (`DGX Spark GB10 / sm_121`) rows were **actually
run on real hardware**: same pure-Triton kernels, same ACVP / RFC / Plonky3 / BLAKE3 / Falcon KAT
vectors, zero failures on either backend. The only kernel-level portability fix required by CUDA
(`triton 3.8` NVPTX) was moving 32-bit endian byte-swapping out of `lms_chain_kernel` into its
PyTorch launcher (`lms/kernels.py`) to avoid an LLVM instruction-selection crash
(`Cannot select: v2i32 = bswap`). ROCm remains an inference from the code and is not claimed as
working.

**Minimum toolchain and architecture floor (`triton>=3.3`, CUDA `sm_75+`):**
Measured on Tesla V100S (`sm_70`, CUDA 12.4, PyTorch 2.6.0, Triton 3.2.0):
- PyTorch 2.4–2.6 requires `typing.List[Tensor]` return annotations on `torch.library.custom_op`
  wrappers (`mlkem/ops.py` uses `typing.List[Tensor]` for `pre` and `build_ye`).
- Triton < 3.3 lacks `tl.gather` (breaking NTT, ML-KEM keygen, ML-DSA verify, and Falcon verify)
  and rejects `tl.static_range` tuple accumulation in BLAKE3; SLH-DSA `thash_kernel` also fails to
  compile on Triton 3.2.
- Volta `sm_70` lacks int8 Tensor Cores (`gemm-int8` fails lowering) and produced an ACVP
  `AssertionError` in `mlkem-768-decaps` under Triton 3.2 (while `mlkem-768-encaps` passed).
  Volta `sm_70` and Triton < 3.3 are therefore unsupported.

Every measurement in this project is paired with a CPU baseline taken **on the same machine**
(`benchmarks/bench_16p_barrier.py`, `vs 1T` / `vs 16P` columns, barrier-measured at matched 1 KiB
message sizes). On this machine the honest summary is:
**eight primitive rows beat sixteen x86 cores device-side** — exact int8 GEMM (8.3–9.3×), the
sponge/SHA-2 family (2.4–7.8×) and the 2^20 NTT (3.3–6.1× across clock states) — while the
ALU-bound primitives stay at 0.09–0.64× (ChaCha20-Poly1305 near parity at 0.92–0.96×). At the
scheme level, **15 scheme operations beat sixteen CPU processes end-to-end**: ML-KEM-512/768/1024
encaps & decaps (1.5–8.4× dev, 1.1–3.8× e2e), ML-DSA-44/65/87 verify (1.5–2.0× dev, 1.0–1.4× e2e),
SLH-DSA SHAKE/SHA2-128f verify (3.3× / 3.9×), and **all four Falcon variants** — Falcon-512 &
Falcon-padded-512 (8.3× dev, **2.9× e2e**) and Falcon-1024 & Falcon-padded-1024 (4.4–4.6× dev,
**2.3–2.4× e2e**). `docs/benchmarks.md` §1, `benchmarks/local-gpu-vs-cpu-20261128.md`, and
`benchmarks/snapshot-20261128T063642Z.md` have the full tables.

With no GPU at all, `pytest tests` skips the device-dependent cases centrally, so a CPU-only
machine can still smoke-test the maths and reference paths.

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
fp32 accumulators per thread the kernel spills heavily. `num_stages=2` is optimal on Intel for
generic GEMM because on-chip memory is small; for the three fused ML-KEM GEMM+epilogue kernels
(`fused_enc_kernel`, `fused_dec_kernel`, `encaps_kernel`), `_fused_cfg` narrows `num_stages=1` on
`xpu` only to work around the `triton-xpu` VA `0x0` pipeliner bug (`docs/kernels.md`), while
`cuda` and `hip` keep their table values above.

### Small kernels — `devices.warps(name, default)`

`num_warps` for the sponge, padcat, CBD, packing and ChaCha20 kernels lives at the top of each
module and can be overridden per kernel with `MLKEM_WARPS="shake=8,chacha=8"`. It is not in
`devices.__all__`: it is an implementation knob, not part of the public API.
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

## 3. Adding a backend

1. Add a row to `_GEMM_CFG` in `devices.py`, or rely on `_FALLBACK_CFG` first to see whether the
   thing runs at all.
2. Run `pytest tests/ --device <dev>`. The suite is written to be backend-agnostic: every fused
   kernel is checked against its unfused counterpart byte-for-byte, and `test_dot_exact.py`
   verifies the fp16 exactness bound on the actual hardware rather than trusting it.
3. Run `python benchmarks/bench_kernels.py --batch 4096` to see which kernel is out of line.
   A backend whose `fused_enc_kernel` share differs wildly from the numbers in
   [docs/benchmarks.md](benchmarks.md) §3.1 usually means a tile configuration problem.
4. Only then sweep tiles. Report both the device-side and end-to-end figures, and state the
   clock conditions — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the measurement discipline
   this project requires.

CI keeps the checks that do not need a GPU on hosted runners (`ruff check`, wheel layout) and the
device suite on a self-hosted GPU runner. The test suite also skips cleanly without a GPU, so a
plain `pytest` on a CPU-only machine is still a useful smoke test.
