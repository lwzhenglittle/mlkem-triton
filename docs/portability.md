# Cross-backend portability: Intel XPU, NVIDIA CUDA, and Hygon HCU

Date: 2026-11-28  
Source under evaluation: `e82515f` (`triton-crypto-kernels` 1.1.0)

## 1. Executive summary

This project demonstrates **strong source-level correctness portability**, but not automatic
performance portability.

The same pure-Triton implementation has run on Intel XPU, NVIDIA CUDA, and Hygon HCU. On the
Hygon BW1000, the evaluated suite reached **1210 passed, 0 failed, and 7 skipped**, with 15
BLAKE3-related device cases excluded from that count. No production algorithm or kernel change was
needed for the passing Hygon workloads. The required work was selecting a compatible Hygon Triton
release and fixing a small number of test and benchmark assumptions.

Performance is workload-specific. BW1000 has much higher copy bandwidth and is particularly strong
at NTTs, ML-DSA verification, Falcon verification, and exact int8 GEMM. B390 remains faster on the
ML-KEM Toeplitz GEMM path. Host transfers can reverse a device-side advantage on any backend.

The defensible claim is therefore:

> Most core operators can be reused across Triton backends when the backend supports the required
> language features and its compiler is stable. Basic tile, warp, and host-path adaptation may still
> be necessary. Operators whose computation and memory-access patterns fit the device can achieve
> competitive performance, but correctness on a backend does not imply a useful speed-up on it.

## 2. Systems and measurement conventions

| Platform | Accelerator | Software | Host notes |
|---|---|---|---|
| Intel XPU | Arc B390, Xe3-LPG, 96 EU | torch 2.15.0+xpu, triton-xpu 3.9.0 | Core Ultra X7 358H, 16 logical CPUs |
| NVIDIA CUDA | GB10, `sm_121`, 48 SM | torch 2.14.0+cu130, Triton 3.8.0 | DGX Spark, Grace ARM64 |
| Hygon HCU | BW1000, `gfx936`, 80 CU, warp size 64, 64 GiB | torch 2.7.1, HIP 6.3, Hygon Triton 3.3.0 | Hygon C86; 128 visible CPUs, but a 15-core cgroup quota |

Unless stated otherwise:

- scheme measurements use batch 4096; ML-DSA signing uses batch 1024;
- primitive measurements use batch 65536 and 1 KiB messages, except GEMM and the 2^20 NTT;
- **device** includes the operation's Python-side launches but excludes host-to-device and
  device-to-host transfers;
- **e2e** includes the NumPy-to-device-to-NumPy path;
- reported values are warmed-up min-of-N measurements, not cold-start compilation time.

The three platforms were not measured in one physical session. B390 absolute timings can move
substantially with its clock state. Cross-platform ratios in this document therefore indicate broad
performance classes, not statistically controlled hardware rankings.

CPU comparisons in the existing GB10 matrix use the stored Intel x86 16-process baseline, not the
Grace CPU in the GB10 host. The Hygon CPU comparisons use a native Hygon barrier baseline. The
15-core quota means that 16 processes already saturate the Hygon allocation; a 128-process run did
not improve throughput and caused throttling.

## 3. Correctness evidence

| Backend | Recorded result | Qualification |
|---|---|---|
| Intel XPU | 1232 passed in the cold-cache record | Full suite |
| NVIDIA CUDA | 1222 passed, 10 skipped | Nine optional full-SLH-DSA liboqs checks and one compile-cache check |
| Hygon HCU | **1210 passed, 0 failed, 7 skipped** | Another 15 BLAKE3-related device cases were excluded; seven skips are optional full-SLH-DSA liboqs checks |

The Hygon result includes ML-KEM, ML-DSA, SLH-DSA, LMS/HSS, Falcon, AES, ChaCha20, Poly1305,
Poseidon2, NTT, fused GEMM, and exact int8 GEMM. Independent checks include NIST ACVP vectors,
liboqs, OpenSSL, Plonky3, and NumPy references.

Important Hygon examples include:

- 14/14 FP16-input/FP32-accumulation exactness cases;
- 20/20 ML-KEM encapsulation cases;
- 18/18 Falcon GPU cases and 12/12 Falcon liboqs cross-checks;
- 37/37 fused-GEMM cases;
- 14/14 exact int8 GEMM cases;
- 39/39 large-NTT cases;
- 11/11 Poseidon2 GPU cases.

This is evidence for mathematical and implementation portability. It is not a formal constant-time
or side-channel certification.

## 4. What transferred unchanged, and what did not

### Core implementation

The backend abstraction in `triton_crypto/devices.py` was sufficient for Hygon. PyTorch presents the
HCU through its HIP-backed `cuda` device API, and `backend_of()` maps it to `hip`. Device selection,
synchronization, and the generic HIP tile configuration therefore work without changing production
modules.

The following constructs were exercised successfully on the selected Hygon toolchain:

- FP16 `tl.dot` with exact FP32 accumulation;
- int8 `tl.dot` with int32 accumulation;
- Keccak and SHA-2 compression;
- in-register and multi-pass NTT;
- fused ML-KEM GEMM epilogues;
- Falcon prefix decoding, HashToPoint, NTT multiplication, and norm checking;
- Poseidon2, AES T-tables, ChaCha20, and Poly1305.

### Test-only adaptations on Hygon

Two test assumptions were generalized in the remote evaluation copy:

1. `tests/test_fused_gemm.py` used the private `torch.mm(..., out_dtype=torch.float32)` extension in
   two reference calculations. The portable replacement upcasts the operands and calls public FP32
   `torch.mm`.
2. `tests/test_cpu_only.py` hard-coded `num_stages=3` for a `cuda` device. On a HIP-backed device the
   configured value is legitimately 2.

These changes affect test references and expectations, not the implementation being tested.

### Benchmark-only adaptations on Hygon

1. `benchmarks/bench_primitive.py` needed explicit module-level `triton` and `triton.language`
   imports so that the generated `_copy` kernel could resolve `tl` on the Hygon frontend.
2. `benchmarks/bench_local_full.py` omitted BLAKE3 explicitly rather than presenting an unmeasured
   row as a successful result.

The Hygon benchmark harness also emitted a misleading clock warning by comparing its new 1T GEMM
rate with the stored Intel baseline. A drift check should first establish that the old and new
measurements describe the same CPU and host.

## 5. Scheme performance across the three platforms

All cells are microseconds per operation, written as **device / e2e**. Lower is better.

| Workload | Arc B390 | NVIDIA GB10 | Hygon BW1000 |
|---|---:|---:|---:|
| ML-KEM-512 keygen | 1.947 / 2.275 | 1.523 / 1.665 | **1.158** / 2.374 |
| ML-KEM-512 encaps | **0.673** / **0.944** | 0.680 / 0.921 | 0.587 / 1.221 |
| ML-KEM-512 decaps | 0.782 / **1.094** | **0.763** / 1.020 | 0.772 / 1.386 |
| ML-KEM-768 keygen | 1.458 / 1.805 | **0.480** / **0.640** | 0.642 / 2.138 |
| ML-KEM-768 encaps | 0.194 / 0.471 | **0.093** / **0.343** | 0.324 / 1.008 |
| ML-KEM-768 decaps | 0.273 / 0.563 | **0.132** / **0.404** | 0.457 / 1.097 |
| ML-KEM-1024 keygen | 2.218 / 2.654 | **0.737** / **0.916** | 0.843 / 2.609 |
| ML-KEM-1024 encaps | 0.246 / 0.551 | **0.121** / **0.382** | 0.323 / 1.077 |
| ML-KEM-1024 decaps | 0.363 / 0.694 | **0.175** / **0.486** | 0.467 / 1.186 |
| ML-DSA-44 verify | 1.966 / 2.827 | **0.314** / **0.856** | 0.425 / 1.964 |
| ML-DSA-65 verify | 2.614 / 3.692 | **0.459** / **1.209** | 0.477 / 2.336 |
| ML-DSA-87 verify | 3.756 / 5.118 | 0.721 / **1.806** | **0.611** / 3.186 |
| ML-DSA-44 sign | 37.6 / 37.5 | **15.0** / **16.1** | 46.0 / 49.4 |
| ML-DSA-65 sign | **51.2** / **52.5** | 20.0 / 21.3 | 57.3 / 61.9 |
| ML-DSA-87 sign | **51.4** / **53.1** | 21.5 / 22.8 | 52.7 / 57.1 |
| SLH-DSA-SHAKE-128f verify | 51.7 / 51.7 | **35.8** / **35.8** | 91.4 / 91.4 |
| SLH-DSA-SHA2-128f verify | 29.4 / 29.4 | **28.3** / **28.3** | 72.5 / 72.5 |
| Falcon-512 verify | 0.318 / 0.915 | **0.069** / **0.435** | 0.089 / 0.805 |
| Falcon-1024 verify | 1.194 / 2.215 | 0.169 / **0.786** | **0.160** / 1.293 |
| LMS/HSS (case 0) verify | 10.6 / 10.6 | **5.8** / **5.8** | 17.3 / 17.3 |

The table illustrates why "portable" must not be read as "uniformly fast":

- GB10 is fastest on the ML-KEM-768 Toeplitz path, but BW1000 is faster than B390 on ML-DSA and
  Falcon device-side verification.
- BW1000 ML-KEM-768 decapsulation is about **0.60x B390 device-side** and **0.51x e2e**.
- BW1000 ML-DSA verification is **4.6–6.1x B390 device-side**, and remains **1.44–1.61x e2e**.
- BW1000 Falcon-512 verification is **3.57x B390 device-side**, but only **1.14x e2e**.
- BW1000 Falcon-1024 verification is **7.47x B390 device-side**, and **1.71x e2e**.
- ML-DSA signing remains structurally difficult on all three platforms when compared with an
  aggregate multi-process CPU baseline.

## 6. Primitive performance across the three platforms

The unit is specific to each row. For hash and cipher rows, GB/s is the benchmark's logical traffic
metric. Higher rates and lower NTT/Poseidon2 times are better. The last column is the BW1000/B390
ratio computed from device time, not from rounded displayed rates.

| Primitive | Unit | Arc B390 | NVIDIA GB10 | Hygon BW1000 | BW/B390 device |
|---|---|---:|---:|---:|---:|
| Copy | GB/s, read+write | 219.0 | 232.4 | **1051.9** | **4.80x** |
| Keccak-f | GB/s, equivalent traffic | 83.9 | 221.3 | 154.7 | **1.84x** |
| SHA3-256 | GB/s | 31.5 | 114.3 | 53.7 | **1.70x** |
| SHA3-512 | GB/s | 17.4 | 94.5 | 30.8 | **1.78x** |
| SHAKE128 | GB/s | 37.6 | 109.9 | 65.4 | **1.74x** |
| SHAKE256 | GB/s | 31.6 | 113.4 | 53.9 | **1.70x** |
| SHA-256 | GB/s | 80.9 | 124.9 | 65.5 | 0.81x |
| SHA-512 | GB/s | 46.0 | 123.9 | 46.4 | 1.01x |
| ML-KEM NTT | GB/s | 48.8 | 230.0 | **239.5** | **4.91x** |
| ML-DSA NTT | GB/s | 14.1 | 232.1 | **178.3** | **12.63x** |
| BabyBear NTT | GB/s | 13.9 | 230.8 | **177.7** | **12.80x** |
| ChaCha20 | GB/s | 12.8 | 105.7 | 41.9 | **3.27x** |
| Poseidon2 | ns/permutation | 40.0 | **4.15** | 17.6 | **2.27x** |
| Poly1305 | GB/s | 11.9 | 27.8 | 5.9 | 0.50x |
| ChaCha20-Poly1305 | GB/s | 8.9 | 31.1 | 8.7 | 0.98x |
| AES-128-CTR | GB/s | 8.7 | 16.6 | 5.7 | 0.65x |
| Exact int8 GEMM, 1024^3 | TMAC/s | 3.80 | 20.4 | **7.35** | **1.93x** |
| BabyBear NTT, 2^20 | ms/transform | 1.575 | **0.199** | 0.623 | **2.53x** |
| BLAKE3 | GB/s | 10.0 | 25.7 | not validated | n/a |

The especially large BW1000 result for the wide Montgomery NTTs is consistent with the platform's
memory bandwidth, but it is not a general rule for every arithmetic workload. Poly1305 and AES are
counterexamples.

## 7. Device-side gains do not automatically survive e2e

The following ratios compare BW1000 with B390 on the same e2e workload. Values above 1 favor BW1000.

| Primitive | BW/B390 device | BW/B390 e2e |
|---|---:|---:|
| SHA3-256 | 1.70x | 0.75x |
| SHA3-512 | 1.78x | 0.85x |
| SHAKE128 | 1.74x | 0.78x |
| SHAKE256 | 1.70x | 0.77x |
| SHA-256 | 0.81x | 0.74x |
| SHA-512 | 1.01x | 0.78x |
| ChaCha20 | 3.27x | 0.64x |
| Poseidon2 | 2.27x | **1.54x** |
| Poly1305 | 0.50x | 0.50x |
| ChaCha20-Poly1305 | 0.98x | 0.77x |
| AES-128-CTR | 0.65x | 0.81x |
| Exact int8 GEMM | 1.93x | 0.67x |
| 2^20 NTT | 2.53x | 0.055x; anomalous host path, see below |

On BW1000, SHA-256 drops from **5.01x the native 16P CPU device-side** to **0.89x e2e**. ChaCha20
drops from 0.91x to 0.07x. These are deployment properties, not compiler curiosities: a library that
keeps a pipeline resident on the device can preserve value that a NumPy-round-trip API cannot.

### The large-NTT e2e anomaly

The BW1000 2^20 NTT takes 0.623 ms device-side, but the measured NumPy-to-NumPy path took 103.2 ms.
Separate measurements gave approximately:

- 9 ms for host int64 conversion and modular reduction;
- 1 ms for H2D after preparation;
- 0.58 ms for the device transform;
- 0.7–0.8 ms for D2H.

The composed operation nevertheless remained around 96–100 ms. Fine-grained tracing showed
approximately 90 ms stalls moving between host preparation and H2D. The cause is unresolved; it may
involve allocation, synchronization, or DTK host-path interactions. It should not be presented as a
normal PCIe bandwidth result.

## 8. Batch-size portability

The batch at which a backend beats the CPU is part of the operator's contract.

| Workload | B390 device / e2e | BW1000 device / e2e |
|---|---:|---:|
| ML-KEM-768 decaps vs 16P | 1024 / 1024 | 1024 / 4096 |
| ML-DSA-44 verify vs 16P | 1024 / 1024 | 1024 / 1024 |
| Falcon-512 verify vs 16P | 256 / 1024 | 256 / 1024 |
| SHA3-256 vs 16P | 1024 / 4096 | 1024 / 4096 |
| ChaCha20 vs 16P | 16384 / not reached | not reached / not reached |

For BW1000 ML-KEM-768 decapsulation, B=1024 is already 1.35x device-side but 0.95x e2e; the e2e
crossover occurs only at B=4096. A single asymptotic throughput figure would hide that distinction.

## 9. Hygon toolchain findings

The Hygon evaluation is also a reminder that "supports Triton" is not a sufficiently precise
compatibility statement.

### Triton 3.1.0

The initial full run reported 130 failures. The main causes were missing or incompatible frontend
behaviour:

- no `tl.gather`, affecting NTT and dependent scheme paths;
- tuple accumulation failures in `tl.static_range`, affecting AES and other primitives;
- constexpr handling differences in generated SLH-DSA kernels.

These were compilation failures, not observed standard-vector arithmetic errors.

### Triton 3.5.1

This release fixed AES and other frontend issues, but ChaCha20 triggered a native segmentation fault
in the AMD-compatible `make_amdgcn` compiler path. It was therefore not selected for the full
evaluation.

### Triton 3.3.0

This was the best of the three tested releases: AES, ChaCha20, NTT, and the other non-BLAKE3
workloads passed. It is the version used for all Hygon performance numbers in this document.

### BLAKE3 remains unresolved

- Triton 3.1 failed on tuple accumulation.
- Triton 3.3 rejected subscripting the constexpr-wrapped message-permutation table.
- Triton 3.5 spent more than ten minutes compiling the first device test.
- An experimental named-scalar runtime-loop version compiled quickly but failed official vectors and
  exposed int32/uint32 loop-carried type errors in tree reduction. It was reverted.

No BLAKE3 performance claim is made for Hygon. Excluding it is a limitation of the Hygon result, not
evidence that its reference implementation or the other backends fail.

## 10. Practical portability rules

A new backend should be evaluated in this order:

1. **Pin the exact Triton build.** A generic `triton>=3.0` requirement does not capture the frontend
   and compiler differences observed here.
2. **Run the numerical prerequisites first.** `test_dot_exact.py`, `test_fields_gpu.py`, and
   `test_primitives_gpu.py` establish the assumptions on which larger pipelines depend.
3. **Run modules in separate processes during bring-up.** A compiler segmentation fault should not
   discard the results of every unrelated module.
4. **Check fused kernels against unfused references.** Compilation alone proves nothing.
5. **Keep the stock configuration and the tuned configuration distinct.** The existing HIP tile
   values were not swept on BW1000, so its results are a portable baseline, not a tuning ceiling.
6. **Measure device and e2e separately.** Include batch size, input layout, host conversion, and
   transfer costs.
7. **Measure a native CPU rival.** Check container CPU quotas and throttling before interpreting
   multi-process scaling.
8. **Do not generalize from one primitive.** NTT, tensor-core GEMM, serial hashes, table lookups,
   and multi-limb arithmetic exercise different hardware properties.

## 11. Scope and data provenance

- B390 values come from `benchmarks/local-gpu-vs-cpu-20261128.{md,json}` and the referenced
  same-window snapshot.
- GB10 values come from `benchmarks/cuda-gb10-vs-local-cpu-20261128.{md,json}` and
  `benchmarks/cuda-gb10-snapshot-20261128.{md,json}`.
- BW1000 values come from an external evaluation of source `e82515f` on 2026-11-28. Its remote raw
  outputs were `bw1000-full.{md,json}` and `bw1000-cpu16.json`; those remote files are not
  repository paths. The summary is reproduced here to keep this document self-contained.
- Hygon is an AMD-derived HIP target, but these results do not claim that AMD ROCm hardware has been
  tested. The support matrix should continue to distinguish them.

The result supports **strong but conditional source portability**. It does not support a blanket
promise that every operator will perform well on every Triton-capable device.
