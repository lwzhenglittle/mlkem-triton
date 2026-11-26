# Supported backends, tuning knobs and layout

---

## 1. Support matrix

### Schemes

| backend | ML-KEM (FIPS 203) | ML-DSA verify + sign (FIPS 204) | SLH-DSA verify (FIPS 205) | LMS/HSS verify (SP 800-208) | Falcon verify (FIPS 206 draft) | notes |
|---|---|---|---|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | ✅ **measured** | ✅ **measured** (all 12) | ✅ **measured** (both SHA-256 families, ACVP) | ✅ **measured** (Falcon-512 + padded) | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, **1142 tests green** |
| NVIDIA CUDA | ✅ **measured** | ⚠️ **not run** | ⚠️ **not run** | ⚠️ **not run** | ⚠️ **not run** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8, ML-KEM suite green (264 tests at the time) |
| AMD ROCm | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | code-level reasoning only |
| Apple MPS | ❌ | ❌ | ❌ | ❌ | ❌ | no Triton backend |
| CPU | ❌ | ❌ | ❌ | ❌ | ❌ | no usable Triton CPU target |

### Scheme-agnostic primitives

Same convention: ✅ means *measured on that backend*, ⚠️ means written in plain Triton with the
same idioms but **not run** there. Nothing in this column is claimed beyond that.

| primitive | what | XPU | CUDA | verified against |
|---|---|---|---|---|
| `primitives.sponge` / `keccak` | SHA3-224/256/384/512, SHAKE128/256, any rate | ✅ | ⚠️ | NIST CAVP, numpy reference |
| `primitives.sha2` | SHA-256 / SHA-512 compression | ✅ | ⚠️ | ACVP via the SLH-DSA/LMS suites |
| `primitives.ntt` | negacyclic NTT, `(q, N, ζ, layers)`, N<=1024 | ✅ | ⚠️ | per-domain numpy, schoolbook convolution |
| `primitives.ntt.large` | multi-pass NTT up to 2^20 (bandwidth-bound) | ✅ | ⚠️ | `NTTDomain` at sizes both can do, round trip, negacyclic product |
| `primitives.fields` | 32-bit Montgomery product, prime-field | ✅ | ⚠️ | numpy twin, cross-module call test |
| `primitives.chacha20` | ChaCha20 (RFC 8439) | ✅ | ⚠️ | RFC 8439 §2.3.2 / §2.4.2, OpenSSL |
| `primitives.poly1305` | Poly1305 + ChaCha20-Poly1305 AEAD | ✅ | ⚠️ | RFC 8439 §2.5.2 / §2.8.2, OpenSSL EVP_MAC |
| `primitives.poseidon2` | Poseidon2-BabyBear/16 | ✅ | ⚠️ | Plonky3 0.8 (its own `kat` output) |
| `primitives.aes` | AES-128-CTR, T-table (a measurement, not a fast path) | ✅ | ⚠️ | FIPS-197 C.1, `openssl enc -aes-128-ctr` |
| `primitives.gemm` | exact `int8 x int8 -> int32` | ✅ | ⚠️ | numpy, and a bit-plane GF(2^8) product |
| `primitives.blake3` | BLAKE3 (tree hash) | ✅ | ⚠️ | the official Rust implementation's 26 KAT vectors |

The ML-KEM CUDA row was **actually run on that machine**: same code, same ACVP vectors, 264
tests green, without changing a single kernel. Only the tile configuration differs. The other
CUDA cells are reasoning, not measurement, and are marked as such. ROCm remains an inference
from the code and is not claimed as working.

Every measurement in this project is paired with a CPU baseline taken **in the same window**
(`benchmarks/bench_cpu.py`, `vs 1T` / `vs 16P` columns); on this machine the honest summary is
that the sponge family wins 1.6-4x over sixteen x86 cores, the ALU-bound primitives lose
0.06-0.5x, and Falcon verification wins 3.2x. `docs/benchmarks.md` §15.1 and §15-§19 have the
tables.

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
fp32 accumulators per thread the kernel spills heavily. `num_stages=2` is optimal on Intel
because on-chip memory is small.

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
   [docs/benchmarks.md](benchmarks.md) §5 usually means a tile configuration problem.
4. Only then sweep tiles. Report both the device-side and end-to-end figures, and state the
   clock conditions — see [CONTRIBUTING.md](../CONTRIBUTING.md) for the measurement discipline
   this project requires.

CI keeps the checks that do not need a GPU on hosted runners (`ruff check`, wheel layout) and the
device suite on a self-hosted GPU runner. The test suite also skips cleanly without a GPU, so a
plain `pytest` on a CPU-only machine is still a useful smoke test.
