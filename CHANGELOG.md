# Changelog

## 0.1.0

First working release. **Pure Triton — no inline assembly, no vendor-specific intrinsics.**

### The three operations

| operation | implementation | correctness | per op on GB10 |
|---|---|---|---:|
| decaps | Toeplitz GEMM (no NTT) | NIST ACVP 3×10 | 125 ns |
| encaps | Toeplitz GEMM (no NTT) | NIST ACVP 3×25 | 88 ns |
| keygen | NTT (unavoidable) | byte-exact against kyber-py | 469 ns |

### Verified on real hardware

- Intel Arc B390 (Xe3-LPG) / triton-xpu 3.9 / torch 2.15
- NVIDIA GB10 (sm_121) / triton 3.8 / torch 2.14

264 tests green on both machines (degrading to 90 passed / 174 skipped with no GPU).
ROCm is **untested** (code-level reasoning only).

### Known limitations

- keygen is about 2.7× slower than cuPQC — the NTT is its structural weakness, see
  [docs/compare.md](docs/compare.md) §0.4
- 7.9 MB of VRAM and 13.5 ms of CPU pre-processing per private key (on-GPU rebuild: 0.158 ms)
- ML-KEM-512 takes the generic CBD path because `η1 = 3`, about 3× slower
- the load-bearing interface needs large batches: on the Intel iGPU B >= 1024 is required to
  match 16-threaded liboqs
- no formal constant-time validation (no dudect / ctgrind)

### Measured, then not done

- **Fusing the keygen pipeline** (SampleNTT → NTT → `pow_mac`) was measured before being built.
  It would remove one 37.7 MB write (0.200 ms) and one read (0.132 ms) of `Â` at B=4096, i.e.
  **81 ns/key**. That is an upper bound of **1.21×** (466 → best case 385 ns/key), still 0.45×
  of cuPQC's 174 ns. The work was declined on that basis; the measurement is recorded in
  [docs/compare.md](docs/compare.md) §0.4 so nobody re-opens it on the assumption that fusion
  closes the gap.
- **Streaming double-buffered H2D** was ruled out by measurement rather than by argument: on
  this Intel driver H2D cannot overlap compute (dual-stream zero-dependency 1.328 ms vs serial
  1.307 ms).

### Measurement methodology

[docs/compare.md](docs/compare.md) records the full comparison data, the same-machine comparison
against cuPQC, and — deliberately — **the hypotheses this project falsified**, for example that
pinned staging was "not worth it" (it was: the memcpy had been measured with numpy assignment
instead of `copy_`).

### Documentation

The long-form documents were rewritten in English and split into
[design](docs/design.md), [benchmarks](docs/benchmarks.md), [support](docs/support.md) and
[comparison](docs/compare.md). The original Chinese versions are kept as supplementary material
in [docs/zh/](docs/zh/); the English versions are authoritative. The unused
`data/acvp/internalProjection.json` (1.4 MB, never read by any code path) was dropped.
