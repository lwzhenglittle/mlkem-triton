# Measured results

Everything here was measured on real hardware on the two machines in §2. Where a number is derived
rather than measured, it says so. Cross-implementation comparisons against NVIDIA cuPQC, HI-Kyber,
ConvKyber and TensorCrypto live in [docs/compare.md](compare.md); the cross-pattern decision rule
lives in [docs/patterns.md](patterns.md).

---

## Contents

- [1. Executive summary (local Arc B390 vs Core Ultra X7 358H)](#1-executive-summary-local-arc-b390-vs-core-ultra-x7-358h)
  - [1.1 All five schemes × operations (Dev + E2E vs 1T / 16P)](#11-all-five-schemes--operations-dev--e2e-vs-1t--16p)
  - [1.2 All 11 scheme-agnostic primitives (Dev + E2E vs 1T / 16P)](#12-all-11-scheme-agnostic-primitives-dev--e2e-vs-1t--16p)
  - [1.3 Batch-size breakeven curves (when GPU beats CPU)](#13-batch-size-breakeven-curves-when-gpu-beats-cpu)
  - [1.4 NVIDIA GB10 (CUDA sm_121) vs Intel Arc B390 side-by-side](#14-nvidia-gb10-cuda-sm_121-vs-intel-arc-b390-side-by-side)
- [2. Machines, methodology and reproducing](#2-machines-methodology-and-reproducing)
- [3. Post-quantum schemes](#3-post-quantum-schemes)
  - [3.1 ML-KEM (FIPS 203): decaps, encaps, keygen, fusion A/B and GB10 CUDA](#31-ml-kem-fips-203-decaps-encaps-keygen-fusion-ab-and-gb10-cuda)
  - [3.2 ML-DSA (FIPS 204): verification and signing](#32-ml-dsa-fips-204-verification-and-signing)
  - [3.3 SLH-DSA (FIPS 205): SHAKE and SHA-2 verification](#33-slh-dsa-fips-205-shake-and-sha-2-verification)
  - [3.4 LMS / HSS (SP 800-208): verification](#34-lms--hss-sp-800-208-verification)
  - [3.5 Falcon (FIPS 206 draft): verification core vs host decode](#35-falcon-fips-206-draft-verification-core-vs-host-decode)
- [4. Scheme-agnostic primitives](#4-scheme-agnostic-primitives)
  - [4.1 Hashes and sponges (Keccak / SHA-3 / SHA-2 / BLAKE3 / Poseidon2)](#41-hashes-and-sponges-keccak--sha-3--sha-2--blake3--poseidon2)
  - [4.2 Symmetric ciphers and MAC (ChaCha20, Poly1305, AEAD, AES-128-CTR)](#42-symmetric-ciphers-and-mac-chacha20-poly1305-aead-aes-128-ctr)
  - [4.3 Large-N multi-pass NTT (2^20 BabyBear) and negacyclic product](#43-large-n-multi-pass-ntt-220-babybear-and-negacyclic-product)
  - [4.4 Exact int8 GEMM and bit-plane GF(2^8) matrix multiply](#44-exact-int8-gemm-and-bit-plane-gf28-matrix-multiply)
- [5. Appendix: measurement corrections and historical lessons](#5-appendix-measurement-corrections-and-historical-lessons)

---

## 1. Executive summary (local Arc B390 vs Core Ultra X7 358H)

All three tables below come from a single continuous window on `lwzheng-TB14-AOSC`
(`Intel Core Ultra X7 358H`, 16 logical CPUs, `performance` profile, `4.7 GHz` peak vs
`Intel Arc B390 GPU`, `torch 2.15.0+xpu`, `triton 3.9.0`, measured copy bandwidth **219.0 GB/s**):

- Full unified local matrix: [benchmarks/local-gpu-vs-cpu-20261128.md](../benchmarks/local-gpu-vs-cpu-20261128.md) (`.json` sibling)
- Standard snapshot: [benchmarks/snapshot-20261128T063642Z.md](../benchmarks/snapshot-20261128T063642Z.md)
- CPU rival baseline (36 rivals, barrier-measured 1T and 16P): [benchmarks/cpu_baseline_barrier.json](../benchmarks/cpu_baseline_barrier.json)

Both **GPU dev** (device-resident input/output, synced) and **GPU e2e** (Host NumPy → H2D → kernels → D2H → Host NumPy) are shown side-by-side against the same CPU baseline so callers can distinguish raw accelerator throughput from actual end-to-end gain.

### 1.1 All five schemes × operations (Dev + E2E vs 1T / 16P)

| Scheme | Op | B | GPU dev (µs/op) | GPU e2e (µs/op) | CPU 1T (µs/op) | CPU 16P (µs/op) | dev vs 1T | dev vs 16P | e2e vs 1T | e2e vs 16P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ML-KEM-512 | keygen | 4096 | 1.947 | 2.275 | 5.393 | 0.895 | 2.77× | 0.46× | 2.37× | 0.39× |
| ML-KEM-512 | encaps | 4096 | 0.673 | 0.944 | 6.011 | 1.006 | 8.94× | **1.49×** | 6.37× | **1.07×** |
| ML-KEM-512 | decaps | 4096 | 0.782 | 1.094 | 7.165 | 1.195 | 9.16× | **1.53×** | 6.55× | **1.09×** |
| ML-KEM-768 | keygen | 4096 | 1.458 | 1.805 | 8.159 | 1.382 | 5.59× | 0.95× | 4.52× | 0.77× |
| ML-KEM-768 | encaps | 4096 | **0.194** | **0.471** | 8.688 | 1.521 | 44.7× | **7.83×** | 18.4× | **3.23×** |
| ML-KEM-768 | decaps | 4096 | **0.273** | **0.563** | 10.43 | 1.827 | 38.2× | **6.68×** | 18.5× | **3.25×** |
| ML-KEM-1024 | keygen | 4096 | 2.218 | 2.654 | 11.07 | 1.855 | 4.99× | 0.84× | 4.17× | 0.70× |
| ML-KEM-1024 | encaps | 4096 | **0.246** | **0.551** | 11.95 | 2.068 | 48.5× | **8.39×** | 21.7× | **3.76×** |
| ML-KEM-1024 | decaps | 4096 | **0.363** | **0.694** | 14.35 | 2.489 | 39.6× | **6.86×** | 20.7× | **3.58×** |
| ML-DSA-44 | verify | 4096 | 1.966 | 2.827 | 15.65 | 2.904 | 7.96× | **1.48×** | 5.54× | **1.03×** |
| ML-DSA-44 | sign | 1024 | 37.57 | 37.54 | 45.01 | 8.370 | 1.20× | 0.22× | 1.20× | 0.22× |
| ML-DSA-65 | verify | 4096 | 2.614 | 3.692 | 25.44 | 4.853 | 9.73× | **1.86×** | 6.89× | **1.31×** |
| ML-DSA-65 | sign | 1024 | 51.22 | 52.49 | 71.85 | 13.24 | 1.40× | 0.26× | 1.37× | 0.25× |
| ML-DSA-87 | verify | 4096 | 3.756 | 5.118 | 38.74 | 7.365 | 10.3× | **1.96×** | 7.57× | **1.44×** |
| ML-DSA-87 | sign | 1024 | 51.36 | 53.11 | 85.44 | 16.09 | 1.66× | 0.31× | 1.61× | 0.30× |
| SLH-DSA-SHAKE-128f | verify | 4096 | 51.67 | 51.67 | 1216.1 | 168.3 | 23.5× | **3.26×** | 23.5× | **3.26×** |
| SLH-DSA-SHA2-128f | verify | 4096 | 29.41 | 29.41 | 934.8 | 113.4 | 31.8× | **3.86×** | 31.8× | **3.86×** |
| Falcon-512 | verify | 4096 | **0.318** | **0.915** | 22.79 | 2.634 | 71.7× | **8.29×** | 24.9× | **2.88×** |
| Falcon-padded-512 | verify | 4096 | **0.321** | **0.913** | 22.74 | 2.652 | 70.9× | **8.26×** | 24.9× | **2.90×** |
| Falcon-1024 | verify | 4096 | **1.194** | **2.215** | 43.78 | 5.192 | 36.7× | **4.35×** | 19.8× | **2.34×** |
| Falcon-padded-1024 | verify | 4096 | **1.135** | **2.183** | 43.87 | 5.197 | 38.6× | **4.58×** | 20.1× | **2.38×** |
| LMS/HSS (RFC 8554 case 0) | verify | 4096 | 10.57 | 10.57 | n/a | n/a | n/a | n/a | n/a | n/a |

LMS/HSS has no liboqs C rival (the in-repo Python `hashlib` reference is `3 866 µs/op`).

**Key takeaways across schemes:**
- **End-to-end winners against 16 CPU cores (15 scheme operations)**: ML-KEM-512/768/1024 encaps & decaps (`1.07×–3.76×` e2e, `1.49×–8.39×` dev), ML-DSA-44/65/87 verify (`1.03×–1.44×` e2e, `1.48×–1.96×` dev), SLH-DSA SHAKE-128f / SHA2-128f verify (`3.26× / 3.86×`), and **all four Falcon variants** — Falcon-512 / Falcon-padded-512 (`2.88×–2.90×` e2e, `8.26×–8.29×` dev) and Falcon-1024 / Falcon-padded-1024 (`2.34×–2.38×` e2e, `4.35×–4.58×` dev).
- **CPU wins against 16P**: ML-KEM keygen (`0.46×–0.95×` dev, `0.39×–0.77×` e2e) and ML-DSA sign (`0.22×–0.31×` vs 16P, though still `1.2×–1.7×` faster than 1 CPU thread).

### 1.2 All 11 scheme-agnostic primitives (Dev + E2E vs 1T / 16P)

`B = 65536`, `size = 1024 B` (except `copy` 256 MiB, `keccak` 2¹⁸ states, `gemm-int8` 1024³, `ntt-2^20` single 2²⁰ transform). Copy bandwidth in the same window: **219.0 GB/s** (`local-gpu-vs-cpu-20261128.md`).

| Primitive | What | dev ms | e2e ms | GB/s | %copy | dev vs 1T | dev vs 16P | e2e vs 1T | e2e vs 16P |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `copy` | read+write | 2.452 | n/a | 219.0 | 100% | n/a | n/a | n/a | n/a |
| `keccak` | 24 rounds, rolled | 1.250 | n/a | 83.9 | 38% | n/a | n/a | n/a | n/a |
| `sha3-256` | rate 136, 8 blocks | 2.328 | 8.159 | 31.5 | 14% | 50.7× | **4.99×** | 14.5× | **1.42×** |
| `sha3-512` | rate 72, 15 blocks | 4.319 | 10.332 | 17.4 | 8% | 51.8× | **5.06×** | 21.7× | **2.12×** |
| `shake128` | rate 168, 7 blocks | 2.103 | 8.340 | 37.6 | 17% | 53.3× | **7.77×** | 13.4× | **1.96×** |
| `shake256` | rate 136, 8 blocks | 2.320 | 8.332 | 31.6 | 14% | 51.2× | **7.39×** | 14.3× | **2.06×** |
| `sha256` | 1024 B/msg, 256-bit | 0.830 | 4.249 | 80.9 | 37% | 25.8× | **2.50×** | 5.04× | 0.49× |
| `sha512` | 1024 B/msg, 512-bit | 1.460 | 4.942 | 46.0 | 21% | 51.0× | **6.60×** | 15.1× | **1.95×** |
| `ntt-mlkem` | q=3329, 7 layers | 2.751 | n/a | 48.8 | 22% | n/a | n/a | n/a | n/a |
| `ntt-mldsa` | q=8380417, 8 layers + Mont. | 9.506 | n/a | 14.1 | 6% | n/a | n/a | n/a | n/a |
| `ntt-babybear` | q=2^31−2^27+1, 8 layers + Mont. | 9.669 | n/a | 13.9 | 6% | n/a | n/a | n/a | n/a |
| `chacha20` | RFC 8439, 20 rounds | 5.243 | 12.519 | 12.8 | 6% | 2.99× | 0.55× | 1.25× | 0.23× |
| `poseidon2` | BabyBear/16, 8 full + 13 partial | 2.619 | 3.402 | 3.2 | 1% | 2.69× | 0.54× | 2.07× | 0.42× |
| `poly1305` | RFC 8439, 5×26-bit limbs | 5.643 | 5.918 | 11.9 | 5% | 2.01× | 0.30× | 1.91× | 0.29× |
| `chacha20-poly1305` | AEAD seal, no AAD | 15.081 | 23.054 | 8.9 | 4% | 5.64× | 0.92× | 3.69× | 0.60× |
| `aes128-ctr` | AES-128-CTR, T-table (gather) | 7.678 | 15.300 | 8.7 | 4% | 0.74× | 0.09× | 0.37× | 0.04× |
| `gemm-int8` | int8 × int8 → int32, exact (1024³) | 0.283 | 0.714 | 22.2 | 10% | 57.1× | **9.28×** | 22.6× | **3.68×** |
| `blake3` | BLAKE3, unkeyed, 32-byte out | 6.726 | 7.129 | 10.0 | 5% | 7.41× | 0.63× | 6.99× | 0.59× |
| `ntt-2^20` | BabyBear 2^20, 10 passes (radix-4) | 1.575 | 5.638 | 53.3 | 24% | 33.2× | **3.37×** | 9.28× | 0.94× |

**What the Dev vs E2E split reveals:**
- **Eight primitives win device-side against 16P CPU**: `gemm-int8` (**9.28×**), `shake128` (**7.77×**), `shake256` (**7.39×**), `sha512` (**6.60×**), `sha3-512` (**5.06×**), `sha3-256` (**4.99×**), `ntt-2^20` (**3.37×**; up to **6.05×** dev / **1.16×** e2e at higher iGPU clocks), and `sha256` (**2.50×**).
- **Six of those eight win end-to-end consistently** (`1.42×–3.68×` vs 16P), and `ntt-2^20` sits right around parity (`0.94×–1.16×` e2e depending on clock state).
- **`sha256` flips from 2.50× (dev) to 0.49× (e2e)**: transferring 64 MiB over H2D takes ~3.4 ms, dwarfing the 0.83 ms kernel. SHA-256 on GPU pays off only when data is already device-resident (e.g. SLH-DSA/LMS chains or multi-stage GPU pipelines).

### 1.3 Batch-size breakeven curves (when GPU beats CPU)

From `benchmarks/bench_local_full.py` Part C (Arc B390 vs X7 358H barrier 1T / 16P):

| Workload | Breakeven `dev ≥ 1T` | Breakeven `dev ≥ 16P` | Breakeven `e2e ≥ 1T` | Breakeven `e2e ≥ 16P` | Peak `e2e vs 16P` |
|---|---:|---:|---:|---:|---:|
| ML-KEM-768 decaps | **B = 64** (1.36×) | **B = 1024** (3.06×) | **B = 64** (1.12×) | **B = 1024** (1.91×) | **3.25×** (B=4096) |
| ML-DSA-44 verify | **B = 64** (1.91×) | **B = 1024** (1.40×) | **B = 64** (1.40×) | **B = 1024** (1.03×) | **1.03×** (B=4096) |
| Falcon-512 verify | **B = 16** (1.76×) | **B = 256** (3.55×) | **B = 64** (1.08×) | **B = 1024** (2.05×) | **2.93×** (B=4096) |
| SHA3-256 (1 KiB) | **B = 256** (2.63×) | **B = 1024** (1.02×) | **B = 256** (1.80×) | **B = 4096** (1.05×) | **1.39×** (B=65536) |
| ChaCha20 (1 KiB) | **B = 1024** (2.49×) | **B = 16384**\* (1.47×) | **B = 1024** (1.34×) | never (max 0.49×) | 0.49× (B=16384) |

\* At `B = 16384` (16 MiB keystream), ChaCha20 fits inside Arc B390's 16 MiB LLC and briefly peaks at `34.1 GB/s` (`1.47×` 16P dev-only) before spilling LLC at `B = 65536` (`12.8 GB/s`, `0.55×`); end-to-end never exceeds `0.49×`.

Detailed ML-KEM-768 decaps, Falcon-512 verify, and SHA3-256 sweep points (full 5-workload tables in [benchmarks/local-gpu-vs-cpu-20261128.md](../benchmarks/local-gpu-vs-cpu-20261128.md)):

| B | ML-KEM-768 dev/e2e (µs) | ML-KEM-768 e2e vs 16P | Falcon-512 dev/e2e (µs) | Falcon-512 e2e vs 16P | SHA3-256 dev/e2e (µs) | SHA3-256 e2e vs 16P |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 38.11 / 42.68 | 0.04× | 12.95 / 94.59 | 0.03× | 11.07 / 14.40 | 0.01× |
| 64 | 7.672 / 9.319 | 0.20× | 3.454 / 21.08 | 0.12× | 2.741 / 3.829 | 0.05× |
| 256 | 2.001 / 2.956 | 0.62× | 0.742 / 3.375 | 0.78× | 0.685 / 1.000 | 0.18× |
| 1024 | 0.598 / 0.957 | **1.91×** | 0.449 / 1.284 | **2.05×** | 0.174 / 0.321 | 0.55× |
| 4096 | 0.273 / 0.563 | **3.25×** | 0.319 / 0.898 | **2.93×** | 0.054 / 0.168 | **1.05×** |

### 1.4 NVIDIA GB10 (CUDA sm_121) vs Intel Arc B390 side-by-side

The exact same pure-Triton kernels were verified (`1222 passed, 0 failed`) and benchmarked on
`spark-ceb3` (`NVIDIA GB10`, `sm_121`, `torch 2.14.0+cu130`, `triton 3.8.0`, measured copy
bandwidth **232.4 GB/s**), compared against the same 36-rival local x86 CPU baseline
(`Intel Core Ultra X7 358H`, 16P barrier in `benchmarks/cpu_baseline_barrier.json`):

- Full GB10 CUDA matrix (Dev + E2E + 5 breakeven curves): [benchmarks/cuda-gb10-vs-local-cpu-20261128.md](../benchmarks/cuda-gb10-vs-local-cpu-20261128.md) (`.json` sibling)
- GB10 CUDA standard snapshot: [benchmarks/cuda-gb10-snapshot-20261128.md](../benchmarks/cuda-gb10-snapshot-20261128.md)

**Schemes on NVIDIA GB10 vs Intel Arc B390 (B = 4096; ML-DSA sign B = 1024; ratios vs local 16P x86 CPU):**

| Scheme | Op | Arc B390 dev / e2e (µs) | GB10 CUDA dev / e2e (µs) | GB10 dev vs 16P | GB10 e2e vs 16P | GB10 vs B390 (dev) |
|---|---|---:|---:|---:|---:|---:|
| ML-KEM-512 | keygen / encaps / decaps | 1.95 / 0.67 / 0.78 (dev) | 1.52 / 0.68 / 0.76 (dev) | 0.59× / 1.48× / 1.57× | 0.54× / 1.09× / 1.17× | ~1.0–1.3× |
| ML-KEM-768 | keygen | 1.458 / 1.805 | **0.480 / 0.640** | **2.88×** | **2.16×** | **3.0×** |
| ML-KEM-768 | encaps | 0.194 / 0.471 | **0.093 / 0.343** | **16.3×** | **4.43×** | **2.1×** |
| ML-KEM-768 | decaps | 0.273 / 0.563 | **0.132 / 0.404** | **13.8×** | **4.52×** | **2.1×** |
| ML-KEM-1024 | keygen | 2.218 / 2.654 | **0.737 / 0.916** | **2.52×** | **2.02×** | **3.0×** |
| ML-KEM-1024 | encaps | 0.246 / 0.551 | **0.121 / 0.382** | **17.1×** | **5.41×** | **2.0×** |
| ML-KEM-1024 | decaps | 0.363 / 0.694 | **0.175 / 0.486** | **14.2×** | **5.12×** | **2.1×** |
| ML-DSA-44 | verify | 1.966 / 2.827 | **0.314 / 0.856** | **9.25×** | **3.39×** | **6.3×** |
| ML-DSA-65 | verify | 2.614 / 3.692 | **0.459 / 1.209** | **10.6×** | **4.01×** | **5.7×** |
| ML-DSA-87 | verify | 3.756 / 5.118 | **0.721 / 1.806** | **10.2×** | **4.08×** | **5.2×** |
| ML-DSA-44/65/87 | sign (B=1024) | 37.6 / 51.2 / 51.4 (dev) | **15.0 / 20.0 / 21.5** (dev) | 0.56× / 0.66× / 0.75× | 0.52× / 0.62× / 0.71× | **2.4–2.6×** |
| SLH-DSA-SHAKE-128f | verify | 51.67 | **35.76** | **4.71×** | **4.71×** | **1.4×** |
| SLH-DSA-SHA2-128f | verify | 29.41 | **28.34** | **4.00×** | **4.00×** | 1.04× |
| Falcon-512 | verify | 0.318 / 0.915 | **0.069 / 0.435** | **38.4×** | **6.05×** | **4.6×** |
| Falcon-padded-512 | verify | 0.321 / 0.913 | **0.069 / 0.442** | **38.5×** | **6.01×** | **4.7×** |
| Falcon-1024 | verify | 1.194 / 2.215 | **0.169 / 0.786** | **30.6×** | **6.60×** | **7.1×** |
| Falcon-padded-1024 | verify | 1.135 / 2.183 | **0.170 / 0.776** | **30.5×** | **6.70×** | **6.7×** |
| LMS/HSS (case 0) | verify | 10.57 | **5.785** | n/a | n/a | **1.8×** |

**Primitives on NVIDIA GB10 vs Intel Arc B390 (`B = 65536, 1024 B`; ratios vs local 16P x86 CPU):**

| Primitive | Arc B390 dev ms (%copy) | GB10 CUDA dev ms (%copy) | GB10 dev vs 16P | GB10 e2e vs 16P | GB10 vs B390 (dev) |
|---|---:|---:|---:|---:|---:|
| `copy` (256 MiB) | 2.452 (219.0 GB/s) | **2.310 (232.4 GB/s)** | n/a | n/a | 1.06× |
| `keccak` | 1.250 (38%) | **0.474 (95%)** | n/a | n/a | **2.6×** |
| `sha3-256` / `sha3-512` | 2.328 / 4.319 | **0.642 / 0.793** | **18.1× / 27.5×** | **3.89× / 6.97×** | **3.6–5.4×** |
| `shake128` / `shake256` | 2.103 / 2.320 | **0.720 / 0.647** | **22.7× / 26.5×** | **5.25× / 5.78×** | **2.9–3.6×** |
| `sha256` / `sha512` | 0.830 / 1.460 | **0.537 / 0.542** | **3.87× / 17.8×** | **1.21× / 5.37×** | **1.5–2.7×** |
| `ntt-mlkem` / `mldsa` / `babybear` | 2.75 / 9.51 / 9.67 | **0.584 / 0.578 / 0.582** (**99–100%**) | n/a | n/a | **4.7–16.6×** |
| `chacha20` | 5.243 (6%) | **0.635 (45%)** | **4.53×** | **1.63×** | **8.3×** |
| `poseidon2` | 2.619 (1%) | **0.272 (13%)** | **5.24×** | **3.25×** | **9.6×** |
| `poly1305` | 5.643 (5%) | **2.417 (12%)** | 0.70× | 0.70× | **2.3×** |
| `chacha20-poly1305` | 15.081 (4%) | **4.322 (13%)** | **3.22×** | **2.53×** | **3.5×** |
| `aes128-ctr` | 7.678 (4%) | **4.054 (7%)** | 0.16× | 0.13× | **1.9×** |
| `gemm-int8` (`1024³`) | 0.283 | **0.053** (**20.4 TMAC/s**) | **50.0×** | **13.1×** | **5.3×** |
| `blake3` | 6.726 (5%) | **2.609 (11%)** | **1.62×** | **1.59×** | **2.6×** |
| `ntt-2^20` (radix-4) | 1.575 (24%) | **0.199 (181%, L2-resident)** | **26.6×** | **1.64×** | **7.9×** |

**Key cross-backend insights (GB10 CUDA vs Arc B390 XPU):**
- **17 of 21 rival-paired scheme operations and 13 of 15 rival-paired primitives beat 16P x86 CPU end-to-end on GB10** (including ML-KEM-768/1024 keygen, SHA-256, ChaCha20, Poseidon2, ChaCha20-Poly1305, and BLAKE3, which lost on Arc B390's iGPU ALUs).
- **Int64 / Montgomery NTT saturates DRAM on GB10**: all three in-register NTT domains (`ntt-mlkem`, `ntt-mldsa`, `ntt-babybear`) run at **230–232 GB/s (99–100% of copy BW)** — a **16.6×** speedup over Arc B390 on 64-bit Montgomery NTTs (`ntt-mldsa` / `ntt-babybear`), driving ML-DSA verify (**5.2–6.3×** faster) and Falcon-512/1024 verify (**4.6–7.1×** faster, `14.3 Mops/s` Falcon-512 core).
- **Breakeven shifts 4× lower on GB10**: ML-KEM-768 decaps, ML-DSA-44 verify, and Falcon-512 verify all beat 16P x86 CPU **end-to-end by `B = 256`** (`1.19×`, `1.61×`, `1.60×`; Falcon-512 dev core already beats 16P at `B = 64`). Only `poly1305` (`0.70×` vs PCLMUL) and `aes128-ctr` (`0.16×` vs AES-NI/VAES) remain behind 16P x86 CPU.

---

## 2. Machines, methodology and reproducing

| | Intel Arc B390 | NVIDIA GB10 |
|---|---|---|
| form factor | integrated GPU (`is_integrated_gpu=1`), B390 / Xe3-LPG | DGX Spark single chip, `sm_121` |
| host CPU | Intel Core Ultra X7 358H (4P + 8E + 4LP, 16 logical CPUs) | Grace / ARM64 (DGX Spark) |
| compute units | 96 EU / 12 subslice | 48 SM |
| last-level cache | 16 MB | 24 MB L2 |
| memory | 29 GB shared system memory | 124 GB unified |
| clock | 0.9 GHz idle / ~2.2 GHz loaded (`xe` exposes no lock) | 2.418 GHz |
| software | torch 2.15.0+xpu / triton-xpu 3.9.0 | torch 2.14.0+cu130 / triton 3.8.0 |

**Measurement conventions:**
- **Device-only (`dev`)**: wall clock of `*_torch` / `verify_mu` / `verify_core_torch` only, with `devices.sync(dev)` before and after each iteration. Excludes H2D/D2H.
- **End-to-end (`e2e`)**: full public API from host bytes / NumPy arrays through H2D, device kernels, and D2H back to host arrays.
- **CPU 16P barrier method (`benchmarks/bench_16p_barrier.py`)**: 16 forked processes warm up outside the timed region, synchronize on a shared `Barrier`, then execute one fixed-work round at the exact same message/operation shape the GPU runs. Aggregate = `sum(units) / (max(end) - min(start))`.
- **Power-state guard**: `bench_16p_barrier.py` records ACPI `platform_profile` and CPU clock in metadata and refuses to overwrite a baseline if the CPU is throttled (`< 2000 MHz` or `low-power`).
- **Ratios travel; raw rates drift**: Arc B390's iGPU clock moves up to 2× between windows, so always pair device numbers with copy bandwidth and CPU baselines from the same window.

**Reproducing:**

```bash
# 1. Full 34-rival CPU barrier baseline (requires OpenSSL, Rust rivals, liboqs build-bench)
python benchmarks/bench_16p_barrier.py --all --reps 2 --json benchmarks/cpu_baseline_barrier.json

# 2. Unified local GPU vs CPU matrix (Schemes Dev+E2E + Primitives Dev+E2E + Breakeven)
python benchmarks/bench_local_full.py --out benchmarks/local-gpu-vs-cpu.md

# 3. Compact standard snapshot (reusing cpu_baseline_barrier.json)
python benchmarks/snapshot.py --skip-cpu

# 4. Per-scheme / diagnostic scripts
python benchmarks/bench_e2e.py --batch 8192
python benchmarks/bench_kernels.py --batch 4096
python benchmarks/bench_host.py --batch 4096
python benchmarks/bench_encaps.py
python benchmarks/bench_keygen.py
python benchmarks/bench_mldsa_verify.py --ps ML-DSA-65 --batch 4096 --profile
```

---

## 3. Post-quantum schemes

### 3.1 ML-KEM (FIPS 203): decaps, encaps, keygen, fusion A/B and GB10 CUDA

#### Cross-backend throughput (Arc B390 XPU vs NVIDIA GB10 CUDA)

**Decapsulation (ML-KEM-768):**

| B | GB10 e2e min | GB10 dev min | GB10 dev µs/op | B390 e2e min | B390 dev min | B390 dev µs/op |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.275 ms | 0.269 | 0.919 ms | 0.585–0.665 ms | 0.57–0.65 |
| 2048 | 0.493 ms | 0.315 ms | 0.154 | 1.242 ms | 0.680–0.769 ms | 0.33–0.38 |
| 4096 | 0.869 ms | 0.533 ms | 0.130 | 1.943 ms | 1.124–1.237 ms | **0.274** |
| 8192 | 1.665 ms | 1.018 ms | **0.125** | 3.674 ms | 2.006–2.227 ms | 0.245–0.272 |

Asymptotic device-side decaps: **GB10 8.0 Mops/s (`125 ns/op`, flat B=8192..65536)**; **Arc B390 3.6 Mops/s (`274 ns/op` at B=4096, where the ~16 MB working set fits LLC)**. Single-operation latency (`B=1` e2e): **282 µs** on GB10, **945 µs** on Arc B390 (13 kernel launches at ~35–40 µs fixed cost each).

**Encapsulation & Key generation (GB10 CUDA, ML-KEM-768, device-side):**
- Encapsulation: `195 ns/op` (B=1024), `94 ns/op` (B=4096), **`88 ns/op` (`11.38 Mops/s`, B=8192)**, `90 ns/op` (B=16384).
- Key generation: `561 ns/key` (B=1024), **`467 ns/key` (`2.14 Mops/s`, B=4096)**, `469 ns/key` (B=8192).

**All three parameter sets at B=4096 (device-side µs/op):**

| Parameter set | B390 decaps | GB10 decaps | B390 encaps | B390 keygen | Note |
|---|---:|---:|---:|---:|---|
| ML-KEM-512 | 0.775 | 0.760 | 0.659 | 1.855 | slow path (`η1 = 3` generic CBD fallback) |
| ML-KEM-768 | **0.274** | **0.132** | **0.191** | 1.382 | fast path |
| ML-KEM-1024 | **0.359** | **0.173** | **0.241** | 2.123 | fast path |

#### `num_stages=1` XPU VA `0x0` guard cost

`_fused_cfg` narrows `num_stages=1` on Intel `xpu` only (CUDA keeps `num_stages=3`) for `fused_enc_kernel`, `fused_dec_kernel`, and `encaps_kernel` (`docs/kernels.md`). Re-measured on Arc B390 (`B=4096`, ML-KEM-768): `bench_kernels.py` total is **`0.916 ms`** (`fused_enc_kernel` `0.309 ms`, `fused_dec_kernel` `0.084 ms`) vs `0.972 ms` (`0.290` / `0.093 ms`) previously, `bench_e2e.py` device-side `1.124–1.129 ms` (`0.274–0.276 µs/op`) vs `1.237 ms`, and `bench_encaps.py` `0.191 µs/op` — **no measurable penalty on Xe3**.

#### Per-kernel breakdown (ML-KEM-768 decaps, B=4096, 13 launches)

| Kernel | Calls | B390 ms (pre-guard / post-guard) | Share | GB10 ms | Share |
|---|---:|---:|---:|---:|---:|
| `fused_enc_kernel` | 1 | 0.290 / 0.309 | ~32% | 0.116 | 36.4% |
| `cbd_ye_kernel` | 2 | 0.159 | 16.4% | 0.065 | 20.4% |
| `_pre_kernel` | 2 | 0.170 | 17.5% | 0.047 | 14.9% |
| `sponge_kernel` | 4 | 0.160 | 16.5% | 0.035 | 10.9% |
| `fused_dec_kernel` | 1 | 0.093 / 0.084 | ~9.3% | 0.030 | 9.5% |
| `padcat_sfx_kernel` | 2 | 0.054 | 5.6% | 0.012 | 3.9% |
| `padcat2_kernel` | 2 | 0.023 | 2.4% | 0.005 | 1.7% |
| other | — | 0.023 | 2.4% | 0.008 | 2.6% |
| **total** | **13** | **0.972 / 0.916** | | **0.318** | |

Arithmetic utilisation: each ML-KEM-768 decaps executes `983,040 MAC` (`1.97 MFLOP/op`). Achieved effective rate is ~6.4–7.2 TFLOPS on Arc B390 (~24% of same-shape pure `tl.dot` peak 26–29 TFLOPS) and 15.0 TFLOPS on GB10 (~28% of 53.1 TFLOPS).

For keygen (GB10, B=4096, 466 ns/key): `pow_mac` is **32.7%** (152 ns, memory-bound reading 37.7 MB of `Â`), NTT 16.7% (78 ns), SampleNTT 16.4% (76 ns), SHAKE-128 ×9 16.1% (75 ns), ByteEncode12 ×2 9.8% (45 ns), CBD ×2 2.3% (11 ns). Fusing SampleNTT → NTT → `pow_mac` can save at most the 81 ns/key `Â` round-trip (`385 ns/key` ceiling; see [docs/compare.md](compare.md) §0.4).

#### Paired fusion A/B and host-side decomposition

| Change | Measured effect (paired, same process) |
|---|---|
| DEC pass: `GEMM + mod q + compress + pack` fused | device-side `1.616 → 1.222 ms` (**1.32×**) |
| Both DEC + ENC fusions on vs both off (e2e) | **1.55×** |
| Sponge `pad10*1 + concat + broadcast` fused (`padcat`) | J stage `192 → 28 µs` (Intel) |
| Accept `torch.Tensor` (zero host copy) | host cost `0.89 → 0.57 ms` (B=4096, Intel) |
| Adaptive pinned staging for H2D | Intel H2D `0.50 → 0.28 ms` (unnecessary on GB10) |
| Dual-stream H2D + compute overlap (Intel XPU) | `1.328 ms` vs `1.307 ms` serial — **does not overlap on `xe`** |

One-time `keyprep` cost & resident VRAM: ML-KEM-512 `6.8 ms / 4.2 MB`, ML-KEM-768 **`13.5 ms / 7.9 MB`**, ML-KEM-1024 `19.8 ms / 12.6 MB`. On-GPU rebuild (`mlkem/kernels/rebuild.py`) takes `0.158 ms` (`19.3 ns/op` at B=8192).

### 3.2 ML-DSA (FIPS 204): verification and signing

All three parameter sets measured against `liboqs 0.16.0` (`mldsa-native` C) in the same window (Table 1.1):
- **Verification (B=4096, Arc B390)**:
  - `ML-DSA-44`: **`1.86 µs`** dev (`8.42×` 1T, **`1.56×` 16P**), **`2.68 µs`** e2e (**`1.08×` 16P**).
  - `ML-DSA-65`: **`2.48 µs`** dev (`10.3×` 1T, **`1.96×` 16P**), **`3.52 µs`** e2e (**`1.38×` 16P**).
  - `ML-DSA-87`: **`3.71 µs`** dev (`10.5×` 1T, **`1.99×` 16P**), **`4.98 µs`** e2e (**`1.48×` 16P**).
  - Device breakdown (ML-DSA-65, B=4096): NTT + MAC + INTT is 77% (`sib + NTT(z) + NTT(c)` 39%, `MAC + INTT` 38%). Vectorized numpy message formatting cut the e2e gap from `6.0 → 3.5–4.1 µs/op`.
- **Signing (B=1024, Arc B390)**:
  - `ML-DSA-44`: `34.2 µs` dev / `35.4 µs` e2e (**`1.32×` 1T**, **`0.24×` 16P**).
  - `ML-DSA-65`: `49.0 µs` dev / `50.1 µs` e2e (**`1.47×` 1T**, **`0.27×` 16P**).
  - `ML-DSA-87`: `48.0 µs` dev / `49.3 µs` e2e (**`1.78×` 1T**, **`0.34×` 16P**).
  - Speculation (evaluating multiple `κ + k·ℓ` candidates per pass) plus the fused `sign_check_kernel` and device-side `pack_z_kernel` brought signing from ~245 µs down to **34–49 µs/sig** (`1.3–1.8×` faster than 1 CPU thread), but the ~22% per-attempt acceptance tail leaves ~18 kernel launches per round latency-bound, keeping signing behind 16 CPU processes (`0.24×–0.34×`). Three tail-mitigation attempts (`XPUGraph` replay `229 µs`, slot/queue `196–320 µs`, fused 3× pointwise `186 µs`) were measured and rejected.

### 3.3 SLH-DSA (FIPS 205): SHAKE and SHA-2 verification

In the same window (`B = 4096`, `internal` interface, matched same-family liboqs 16P barrier rivals):
- **`SLH-DSA-SHAKE-128f`**: **`49.9 µs/sig`** vs liboqs SHAKE-128f `1216 µs` 1T / `168.3 µs` 16P → **`24.4×` 1T, `3.37×` 16P**.
- **`SLH-DSA-SHA2-128f`**: **`28.7 µs/sig`** vs liboqs SHA2-128f `935 µs` 1T / `113.4 µs` 16P → **`32.6×` 1T, `3.95×` 16P**.

All 12 parameter sets (`B = 4096`, Arc B390 vs same-process liboqs 1T):

| SHAKE set | dev µs/sig | vs liboqs 1T | SHA-2 set | dev µs/sig | vs liboqs 1T |
|---|---:|---:|---|---:|---:|
| SHAKE-128s | 18.7 | 24.5× | SHA2-128s | 13.1 | 29.6× |
| SHAKE-128f | 47.8–50.1 | 24.4–27.9× | SHA2-128f | 27.8–28.7 | 32.6–42.0× |
| SHAKE-192s | 26.5 | 24.6× | SHA2-192s | 17.5 | 32.8× |
| SHAKE-192f | 67.4 | 29.2× | SHA2-192f | 37.9 | 42.0× |
| SHAKE-256s | 37.8 | 27.3× | SHA2-256s | 26.0 | 31.4× |
| SHAKE-256f | 71.2 | 28.7× | SHA2-256f | 44.1 | 38.7× |

Two architectural optimisations drove these numbers: (1) fusing `PK.seed ‖ ADRS ‖ data` into `thash_kernel` cut 128f from `172 → 71 µs/sig`, and (2) sorting WOTS+ chains by digit per layer (chain compaction) cut 128f from `71 → 48–50 µs/sig`. Crossover vs 1T CPU occurs at `B ≈ 32` (`B = 1` is 37 ms due to ~400 kernel launches).

### 3.4 LMS / HSS (SP 800-208): verification

`benchmarks/bench_lms_verify.py` (Arc B390, RFC 8554 test vectors tiled across the batch; liboqs does not implement LMS/HSS, so Python `hashlib` reference numbers are shown only as context):

| Vector | B = 256 | B = 1024 | B = 4096 | Python ref |
|---|---:|---:|---:|---:|
| RFC 8554 case 0 (H5/W8, 2 levels) | 29.7 µs | 12.1 µs | **8.5–10.0 µs** | 2 300–3 795 µs |
| RFC 8554 case 1 (H10/W4 + H5/W8, 2 levels) | — | — | **7.1 µs** | 2 530 µs |

Each LM-OTS chain runs up to 255 sequential SHA-256 compressions; sorting rows by starting step and exiting per block achieves ~1 Ghash/s of 32-bit SHA-256 compression (~3× SLH-DSA's 64-bit Keccak rate).

### 3.5 Falcon (FIPS 206 draft): Falcon-512 & Falcon-1024 end-to-end on GPU

In the same window (`B = 4096`, liboqs `pqclean_falcon-{512,1024}` corpus, barrier 16P CPU baseline):
- **`Falcon-512`**: device core **`0.318 µs/op`** (**`71.7×` 1T, `8.29×` 16P**); end-to-end **`0.90–0.92 µs/op`** (**`24.9–25.4×` 1T, `2.88–2.94×` 16P**).
- **`Falcon-padded-512`**: device core **`0.321 µs/op`** (**`70.9×` 1T, `8.26×` 16P**); end-to-end **`0.913 µs/op`** (**`24.9×` 1T, `2.90×` 16P**).
- **`Falcon-1024`**: device core **`1.194 µs/op`** (**`36.7×` 1T, `4.35×` 16P**); end-to-end **`2.20–2.22 µs/op`** (**`19.8–19.9×` 1T, `2.34–2.36×` 16P**).
- **`Falcon-padded-1024`**: device core **`1.135 µs/op`** (**`38.6×` 1T, `4.58×` 16P**); end-to-end **`2.183 µs/op`** (**`20.1×` 1T, `2.38×` 16P**).

End-to-end stage breakdown at `B = 4096` (measured with `benchmarks/bench_falcon_verify.py --profile`):

| Stage | Where | Falcon-512 (µs/sig) | Falcon-1024 (µs/sig) |
|---|---|---:|---:|
| Prefix signature decode (`_pack_sig_words_kernel` + `_comp_decode_kernel`, incl. flat pack & H2D) | GPU | **0.20** (0.086 dev kernel) | **0.35** (0.115 dev kernel) |
| `HashToPoint` (`_pad_nonce_msg_kernel` + rolled `sponge` + `_hash_to_point_kernel`) | GPU (uniform length) | **~0.35** | **~0.65** |
| Device core (`NTT(s2)` + pointwise + `INTT` + L2 norm check) | GPU | **0.31–0.32** (NTT 0.14, INTT 0.11) | **1.19** (NTT 0.64, INTT 0.34) |
| Full end-to-end (`Verifier.verify`) | GPU + flat H2D | **0.90–0.92** (**2.9× 16P**) | **2.20–2.22** (**2.35× 16P**) |

**How `comp_decode` flipped from a 35× loss to a 168×–232× win:** an early GPU `comp_decode` attempt ran 3-byte uncoalesced gathers plus global CLZ lookup tables per coefficient and measured **2.6 ms/sig** (~190× slower than the 13.5 µs host NumPy scan). Replacing it with (1) a single flat `b"".join(signatures)` upload, (2) `_pack_sig_words_kernel` transposing bodies into `[W, B]` big-endian `uint32` words, (3) a **64-bit register bit reservoir** (`acc`, `acc_len`) that refills one coalesced `uint32` word only ~32% of steps, (4) a 4-stage **branchless 16-bit CLZ** with zero lookup tables, and (5) keeping `c_t` and `s2_t` device-resident through `verify_core_torch` cut Falcon-512 end-to-end verification from **15.35 µs → 0.90 µs** (**17× faster**, flipping `0.22×` → **`2.88–2.94×` vs 16P CPU**). For Falcon-1024, rolling multi-block SHAKE256 squeezes in `primitives/sponge.py` enables the 20-block (`2624 B`) `HashToPoint` stream cleanly with one copy of the 24-round loop.

---

## 4. Scheme-agnostic primitives

### 4.1 Hashes and sponges (Keccak / SHA-3 / SHA-2 / BLAKE3 / Poseidon2)

- **Keccak / SHA-3 / SHAKE**: `keccak` hits `210 Mperm/s` (`84.1 GB/s`, 38–39% of copy BW). All four sponge modes beat 16P CPU by **`4.95×–7.77×` device-only** and **`1.41×–2.08×` end-to-end**. Keeping the 24-round loop *rolled* avoids register spilling (`spill_size` `3584 → 0`, 30% faster).
- **SHA-256 / SHA-512**: `sha256` reaches **`78.8–81.0 GB/s`** (`2.44–2.51×` 16P dev, despite x86 SHA-NI) and `sha512` **`45.0–45.1 GB/s`** (`6.47–6.48×` 16P dev, **`1.93×` e2e**). As Table 1.2 shows, `sha256` drops to `0.48×` e2e when paying 64 MiB H2D for a single hash.
- **BLAKE3**: **`9.7–10.1 GB/s`** (`7.39–7.53×` 1T, `0.63–0.64×` 16P dev; `0.59×` e2e), flat across 1 KiB (1 chunk) and 16 KiB (16 chunks, 5 tree levels). Loses to 16P CPU because the official Rust AVX-512 implementation hashes 16 blocks concurrently. (Development note: shifting a `uint32` chunk counter by 32 triggered silent zero-folding UB on `triton-xpu`; fixed via explicit `(low, high)` counter pair and pinned by multi-chunk tests.)
- **Poseidon2-BabyBear/16**: **`38.5–39.6 ns/perm`** (`25.2–26.0 M perm/s`, `2.71–2.79×` 1T, `0.55–0.57×` 16P against Plonky3 0.8 AVX2 packed `monty-31x8`; `0.44×` e2e). Needs `B ≥ 4096` to saturate (`BLK = 128`).

### 4.2 Symmetric ciphers and MAC (ChaCha20, Poly1305, AEAD, AES-128-CTR)

- **ChaCha20**: `14.1–14.3 GB/s` at B=65536 (`3.29–3.34×` 1T, `0.60–0.61×` 16P dev; `0.23×` e2e). At B=16384 (16 MiB LLC fit) device-only peaks at `34.1 GB/s` (`1.48×` 16P), but e2e tops out at `0.53×`.
- **Poly1305 & ChaCha20-Poly1305 AEAD**: `poly1305` runs at `11.7–12.0 GB/s` (`1.97–2.02×` 1T, `0.30×` 16P); `chacha20-poly1305` seal reaches `4.64 GB/s` plaintext (`9.1–9.3 GB/s` total traffic, `5.77–5.88×` 1T, `0.94–0.96×` 16P dev, `0.60×` e2e). Keeping the Poly1305 one-time key on-device (eliminating a `cpu().numpy()` round trip) doubled AEAD throughput (`2.10 → 4.04+ GB/s`).
- **AES-128-CTR (T-table)**: **`9.8–10.9 GB/s`** (`0.83–0.93×` 1T AES-NI, `0.10–0.11×` 16P dev, `0.05×` e2e; host key schedule hoisted). This was **~12× faster** than a naive cold-gather extrapolation (`0.7–0.85 GB/s`) because the 4.25 KiB T-table stays L1-resident across 64 consecutive blocks per lane. Even so, AES-NI / VAES across 16 cores hits `101 GB/s`, so porting AES for throughput is not worthwhile.

### 4.3 Large-N multi-pass NTT (2^20 BabyBear) and negacyclic product

`LargeNTT(1 << 20, 2013265921, radix=4)` (the default: 10 fused radix-4 passes, 84 MB total traffic):
- **Single forward transform (same window)**: **`0.877–0.899 ms`** (**`93.3–95.6 GB/s`**, 42–44% of copy BW, **`58.2–59.6×` 1T**, **`5.91–6.05×` 16P** dev; **`4.581 ms`**, **`1.16×` 16P** e2e). Clock variation across sessions spans `0.85–1.64 ms` (`3.8–7.3×` 16P).
- **Optimisation path (`10.8 ms → 0.85–0.90 ms`)**: (1) cache twiddle tables per device (`1686 → 10.8 ms`), (2) replace runtime int64 `// half` with shift/mask, (3) tune to `BLK=1024, 8 warps`, (4) fold negacyclic pre/post twiddles into first/last pass, and (5) fuse radix-2 stages pairwise into radix-4 (`10 passes` instead of `20 passes` at radix-2; paired A/B: `1.50 → 1.34 ms` min, `1.67 → 1.36 ms` median, 11–19% faster).
- **What did not pay off (measured)**: Barrett BabyBear fold (`1.02×`, inside noise because LLVM already strength-reduces `% Q` for constexpr `Q`); collapsing tail stages `m <= 64` into one register pass (blocked by Triton lacking a DIF half-split without an extra bit-reversal pass); radix-8 (projected 4–6%, below clock noise).
- **Negacyclic product (`a · b mod X^n + 1`, what ZK provers call)**:
  - `mul_torch` (42 passes: 10+10 forward + 1 pointwise + 19 radix-2 inverse + twiddles): **`3.36 ms`** (`3.96×` 16P in like-for-like products/s).
  - `mul_fused_torch` (32 passes via `_inv2x2_kernel` + fused pointwise epilogue): **`2.41 ms`** (**`1.40×` faster**, **`5.53×` 16P**).

### 4.4 Exact int8 GEMM and bit-plane GF(2^8) matrix multiply

- **`gemm_int8_torch` (exact `int8 × int8 → int32`, Arc B390)**:

| Size (square) | ms | GMAC/s | TMAC/s | vs 16P CPU (409 GMAC/s at 1024³) |
|---:|---:|---:|---:|---:|
| 256³ | 0.093 | 180 | 0.18 | — |
| 512³ | 0.161 | 834 | 0.83 | — |
| **1024³ (same window)** | **0.281–0.291** | **3 689–3 816** | **3.69–3.82** | **9.03–9.33× dev / 3.81× e2e** |
| 2048³ | 2.167 | 3 965 | 3.96 | — |

The CPU rival at `1024³` is `torch.mm` on `fp32` inputs cast from `int8` (`1024 × 128² = 2²⁴`, bit-exact in IEEE 754 float32, utilizing CPU AVX2/FMA SGEMM: `66.5 GMAC/s` 1T, `409 GMAC/s` 16P barrier).

- **`gf2_8_matmul_torch` (64 bit-plane int8 GEMMs XORed via AES polynomial)**:
  - At `512³`: `13.7–15.3 GMAC/s` (`54–56×` below one int8 GEMM).
  - At `1024³`: **`33.5–33.8 GMAC/s`** (`60–61×` below one int8 GEMM at `2 040 GMAC/s` in that window).
  - Decomposition overhead is `~1/57` of the int8 rate, giving an honest empirical ceiling for bit-plane matrix-unit finite-field GEMMs before any field-specific tiling or VNNI rival comparison.

---

## 5. Appendix: measurement corrections and historical lessons

Six measurement traps were caught and fixed during development; they are kept here so future benchmarking does not repeat them:

1. **Sum-of-minima 16P over-count (`bench_cpu.py` → `bench_16p_barrier.py`)**: summing each child's minimum over unaligned runs over-counted the 16P CPU by **1.25×–4.25×** (median `1.94×`; e.g., ML-DSA-44 verify read `949k/s` instead of `344–370k/s` and flipped from `0.57×` to `1.56–1.66×`). The legacy `benchmarks/cpu_baseline.json` and `benchmarks/archive/snapshot-20261126T*.md` preserve the old record; [benchmarks/table-16p-20261127.md](../benchmarks/table-16p-20261127.md) documents the before/after diff.
2. **Message-size mismatch**: comparing GPU `1 KiB` digests against CPU `16 KiB` / `1 MiB` charged per-message init/final overhead to the GPU only (~2.4× bias on SHA-256). `bench_16p_barrier.py` now matches `1024 B/msg`.
3. **SLH-DSA family mismatch (fixed 2026-11-27)**: `bench_16p_barrier.py` previously measured `slh_dsa_pure_sha2_128f` (`8.82k/s` 16P) while `snapshot.py` timed `SLH-DSA-SHAKE-128f` (`5.94k/s` 16P). Both families are now measured and paired separately (`3.37×` SHAKE, `3.95×` SHA2).
4. **Silent rival degradation**: rebuilding `benchmarks/poseidon2_ref` without `RUSTFLAGS="-C target-cpu=native"` dropped Plonky3 to scalar (`7×` too slow) and made Poseidon2 look `1.25×` faster than 16P CPU. The harness now checks `lanes >= 4` and aborts on scalar builds.
5. **Power-state throttling (`4.6×` swing)**: running on battery under `low-power` capped CPU clocks at `0.9 GHz` and produced impossible `scale > 16`. `bench_16p_barrier.py` now refuses to write baselines in throttled states unless `--force` is given; reproducibility across reboots under `performance` landed at median `0.997` ([benchmarks/table-16p-repro.md](../benchmarks/table-16p-repro.md)).
6. **Mixed units & stale traffic formulas**: dividing GPU *products/s* by CPU *transforms/s* understated the 2²⁰ NTT product ratio by `3×` (`0.94×` vs `3.96×`), and hardcoding radix-2's `20 passes` after switching to radix-4 overstated `ntt-2^20` GB/s and `%copy` by `2×`. Both now derive directly from `LargeNTT.n_passes` and matched units.
