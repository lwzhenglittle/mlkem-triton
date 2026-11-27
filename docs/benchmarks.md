# Measured results

Everything here was measured on the two machines in §1. Where a number is derived rather than
measured, it says so. Comparison against other implementations lives in
[docs/compare.md](compare.md); this file is the raw record.

---

## Contents

- [1. Machines and conventions](#1-machines-and-conventions)
- [2. Decapsulation throughput, ML-KEM-768](#2-decapsulation-throughput-ml-kem-768)
- [3. Encapsulation](#3-encapsulation)
- [4. Key generation](#4-key-generation)
- [5. Where the device time goes (decaps, B=4096)](#5-where-the-device-time-goes-decaps-b4096)
- [6. Fusion A/B (paired, same process)](#6-fusion-ab-paired-same-process)
- [7. Host side, B=4096, Intel](#7-host-side-b4096-intel)
- [8. CPU baseline (liboqs 0.16.0, same machine, same ACVP ciphertexts)](#8-cpu-baseline-liboqs-0160-same-machine-same-acvp-ciphertexts)
- [9. One-time cost](#9-one-time-cost)
- [10. Reproducing](#10-reproducing)
- [11. ML-DSA signature verification (FIPS 204)](#11-ml-dsa-signature-verification-fips-204)
- [12. SLH-DSA (FIPS 205) verification](#12-slh-dsa-fips-205-verification)
- [13. LMS / HSS verification (SP 800-208)](#13-lms--hss-verification-sp-800-208)
- [14. ML-DSA signing](#14-ml-dsa-signing)
- [14.5 The synthesis: `docs/patterns.md`](#145-the-synthesis-docspatternsmd)
- [15. The primitive layer itself](#15-the-primitive-layer-itself)
- [16. Falcon-512 verification (FIPS 206 draft)](#16-falcon-512-verification-fips-206-draft)
- [17. Poly1305 and ChaCha20-Poly1305](#17-poly1305-and-chacha20-poly1305)
- [18. AES-128-CTR through a T-table: the gather surprise](#18-aes-128-ctr-through-a-t-table-the-gather-surprise)
- [19. Exact int8 GEMM: the substrate for a finite-field GEMM](#19-exact-int8-gemm-the-substrate-for-a-finite-field-gemm)
- [20. One window, one file: `benchmarks/snapshot.py`](#20-one-window-one-file-benchmarkssnapshotpy)
- [21. BLAKE3: the tree hash, and the UB that hid for four sessions](#21-blake3-the-tree-hash-and-the-ub-that-hid-for-four-sessions)
- [22. The 2^20-point NTT: the one primitive whose cost is memory traffic](#22-the-220-point-ntt-the-one-primitive-whose-cost-is-memory-traffic)

§1–§10 are the ML-KEM/ML-DSA/SLH-DSA work on two machines; §11–§14 are the other
schemes; §15–§22 are the scheme-agnostic primitive layer, including the CPU rival table
that decides whether a port is worth doing. Every number is same-window against a CPU
baseline; §2.2 of the overnight report and §1 here explain why only ratios travel.

---

## 1. Machines and conventions

| | Intel Arc B390 | NVIDIA GB10 |
|---|---|---|
| form factor | integrated GPU (`is_integrated_gpu=1`), B390 / Xe3-LPG | DGX Spark single chip, `sm_121` |
| compute units | 96 EU / 12 subslice | 48 SM |
| last-level cache | 16 MB | 24 MB L2 |
| memory | 29 GB shared system memory | 124 GB unified |
| clock | 0.9 GHz idle / ~2.2 GHz loaded | 2.418 GHz |
| software | torch 2.15.0+xpu / triton-xpu 3.9.0 | torch 2.14.0+cu130 / triton 3.8.0 |

- **device-side**: wall clock of `decaps_torch()` only — 13 Python-side Triton enqueues plus
  device execution, excluding H2D/D2H. This is the column closest to what papers bracket with
  `cudaEvent`.
- **end-to-end**: the whole `decaps()` path, including numpy packing, H2D and D2H.
- **min-of-N**: N = 10–25, `synchronize()` before and after each iteration.
- **GPU clocks drift between 0.9 and 2.2 GHz and the `xe` driver exposes no frequency control**,
  so absolute values move by ±20% across runs. Every table gives the best of several runs; only
  paired ratios measured in the same process are reliable.
- All ciphertexts come from the NIST ACVP vectors, and outputs are compared byte-for-byte
  against `expectedResults`.

---

## 2. Decapsulation throughput, ML-KEM-768

**NVIDIA GB10**

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.390 ms | 0.275 ms | 0.374 | 2.67 | 3.9× |
| 2048 | 0.493 ms | 0.501 ms | 0.315 ms | 0.241 | 4.15 | 6.0× |
| 4096 | 0.869 ms | 0.876 ms | 0.533 ms | 0.212 | 4.71 | 6.8× |
| 8192 | 1.665 ms | 1.678 ms | 1.018 ms | 0.203 | 4.92 | 7.1× |

**Intel Arc B390**

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.919 ms | 1.090 ms | 0.665 ms | 0.898 | 1.11 | 1.61× |
| 2048 | 1.242 ms | 1.431 ms | 0.769 ms | 0.607 | 1.65 | 2.39× |
| 4096 | 1.943 ms | 2.196 ms | 1.237 ms | 0.474 | 2.11 | 3.06× |
| 8192 | 3.674 ms | 4.277 ms | 2.227 ms | 0.449 | 2.23 | 3.23× |

Batch sweep, device-side µs/op, to find the asymptote:

| B | 1024 | 4096 | 8192 | 16384 | 32768 | 65536 |
|---|---:|---:|---:|---:|---:|---:|
| Intel | 0.668 | **0.311** | 0.395 | 0.408 | 0.393 | 0.380 |
| GB10 | 0.272 | 0.130 | **0.125** | 0.125 | 0.128 | 0.130 |

- Intel peaks at B=4096, where the working set is ≈16 MB — exactly its last-level cache — and
  degrades beyond it.
- GB10 is flat from B=8192 through 65536.
- **Asymptotic throughput: Intel 3.2 Mops/s, GB10 8.0 Mops/s.**

### The three parameter sets (B=4096, device-side µs/op)

| ps | Intel | GB10 | note |
|---|---:|---:|---|
| ML-KEM-512 | 0.932 | 0.760 | **slow path** (η1 = 3 CBD fallback) |
| ML-KEM-768 | 0.307 | 0.132 | fast path |
| ML-KEM-1024 | 0.411 | 0.173 | fast path |

512 is 3× slower than 768 because the fused CBD kernel only packs `η = 2`. Correct (ACVP
passes), unoptimised. Details in [docs/compare.md](compare.md) §5.4.

### Single-operation latency

| | B=1 end-to-end | B=8 end-to-end |
|---|---:|---:|
| Intel Arc B390 | **945 µs** | 1086 µs (136 µs/op) |
| GB10 | **282 µs** | — |

13 kernel launches at ~35–40 µs fixed cost each, with nothing to amortise at B=1. **This design
is throughput-oriented, not latency-oriented**; see [docs/compare.md](compare.md) §5.2 for how
that compares with cuPQC (4.6× slower at B=1, same machine).

---

## 3. Encapsulation

All 75 ACVP vectors pass (3 parameter sets × 25; `c` and `K` byte-exact).

**GB10, ML-KEM-768, device-side including Python enqueue:**

| B | ns/op | throughput |
|---:|---:|---:|
| 1024 | 195 | 5.14 M/s |
| 4096 | 94 | 10.67 M/s |
| 8192 | **88** | **11.38 M/s** |
| 16384 | 90 | 11.08 M/s |

**Intel Arc B390, end-to-end:**

| B | e2e µs/op | device-side µs/op |
|---:|---:|---:|
| 256 | 2.248 | 1.467 |
| 512 | 1.211 | 0.838 |
| 1024 | 0.695 | 0.491 |
| 4096 | 0.381 | 0.218 |
| 8192 | 0.379 | 0.184 |
| 16384 | **0.310** | **0.175** |

Encapsulation beats decapsulation on both machines: it is one GEMM pass with a
compress-and-pack epilogue, no sponge, no second GEMM.

---

## 4. Key generation

**GB10, device-side:**

| B | ns/key | throughput | vs liboqs-1T (7.73 µs) |
|---:|---:|---:|---:|
| 1024 | 561 | 1.78 M/s | 10.2× |
| 4096 | 467 | 2.14 M/s | 13.3× |
| 8192 | 469 | 2.13 M/s | 13.5× |

Correctness is pinned by byte-exact comparison against kyber-py: `ek` (800/1184/1568 B) and `dk`
(1632/2400/3168 B) match for all three parameter sets.

**Per-stage breakdown, GB10, B=4096:**

| stage | ns/key | share |
|---|---:|---:|
| `pow_mac` (pointwise multiply + extension, reads 37.7 MB of `Â`) | 152 | **32.7%** |
| NTT (2k polynomials) | 78 | 16.7% |
| SampleNTT | 76 | 16.4% |
| SHAKE-128 ×9 | 75 | 16.1% |
| ByteEncode12 ×2 | 45 | 9.8% |
| CBD ×2 | 11 | 2.3% |
| other (message construction / pad / H(ek)) | 25 | 5.5% |
| **total** | **466** | |

**Per-stage breakdown, Intel, B=4096:** NTT 372 ns, SampleNTT 448 ns, SHAKE-128 ×9 269 ns,
pointwise multiply 223 ns, ByteEncode12 ×2 141 ns, CBD ×2 44 ns.

The NTT is *not* the bottleneck — it is 17%. `pow_mac` is, and it is memory-bound: 0.62 ms
corresponds to ≈75 MB of traffic (`Â` 37.7 + `Ŝ` 12.6 + `Ê` 12.6 + 12.6 written), a lower bound
of ≈0.28 ms at GB10's 273 GB/s.

**Fusion ceiling, GB10, B=4096.** `Â` is 37.7 MB:

| | time | per key |
|---|---:|---:|
| write once | 0.200 ms (188.7 GB/s) | 49 ns |
| read once | 0.132 ms (286.0 GB/s) | 32 ns |
| **write + read** | **0.332 ms** | **81 ns** |

Fusing SampleNTT → NTT → `pow_mac` into one kernel can save exactly that: 466 → best case
385 ns/key. Full argument in [docs/compare.md](compare.md) §0.4.

---

## 5. Where the device time goes (decaps, B=4096)

`python benchmarks/bench_kernels.py --batch 4096`:

| kernel | calls | Intel ms | share | GB10 ms | share |
|---|---:|---:|---:|---:|---:|
| `fused_enc_kernel` | 1 | 0.290 | 29.9% | 0.116 | 36.4% |
| `cbd_ye_kernel` | 2 | 0.159 | 16.4% | 0.065 | 20.4% |
| `_pre_kernel` | 2 | 0.170 | 17.5% | 0.047 | 14.9% |
| `sponge_kernel` | 4 | 0.160 | 16.5% | 0.035 | 10.9% |
| `fused_dec_kernel` | 1 | 0.093 | 9.5% | 0.030 | 9.5% |
| `padcat_sfx_kernel` | 2 | 0.054 | 5.6% | 0.012 | 3.9% |
| `padcat2_kernel` | 2 | 0.023 | 2.4% | 0.005 | 1.7% |
| other | — | 0.023 | 2.4% | 0.008 | 2.6% |
| **total** | | **0.972** | | **0.318** | |

13 kernel launches for the whole pipeline.

The sponge (`sponge_kernel` + `padcat*`) is 24.5% on Intel and 16.5% on GB10. The Triton sponge
is one thread per instance, 25 lanes, 24 rounds rolled with a runtime round-constant table — see
[docs/design.md](design.md) §4 for why rolling is forced by the 256-GRF-per-thread limit
(`spill_size` 3584 → 0, static instruction count 15254 → 790, compile ~25 s → ~3 s). A
hand-written SYCL sponge measured 2.2–2.6× faster than the Triton one; it was removed from this
library because it required `icpx` and was Intel-only, costing ≈20% end-to-end on Intel.

### Arithmetic utilisation

Per decapsulation, after Toeplitz expansion:

```
DEC  W_dec[256, 768] × U[768, B]   = 196,608 MAC
ENC  W_enc[1024,768] × Y[768, B]   = 786,432 MAC
                                     983,040 MAC = 1.97 MFLOP / op
```

| | throughput | derived FLOPS | same-shape pure `tl.dot` peak | share |
|---|---:|---:|---:|---:|
| Intel Arc B390 | 3.26 Mops/s | 6.4 TFLOPS | 26–29 TFLOPS | ~24% |
| GB10 | 7.59 Mops/s | 15.0 TFLOPS | 53.1 TFLOPS | ~28% |

The peak figure is a measured pure-`tl.dot` loop at the same shape (M=1024, K=768, B=4096), so
24–28% means there is 3–4× of headroom in the non-GEMM stages and the epilogue. On Intel, VTune
reports `XVE Array Stalled/Idle 75.1%` and occupancy 28.5%.

---

## 6. Fusion A/B (paired, same process)

| change | effect |
|---|---|
| DEC pass: `GEMM + mod q + compress + pack` into one kernel | device-side 1.616 → 1.222 ms (**1.32×**) |
| ENC pass: `GEMM + mod q + compress + compare + reduce` into one kernel | further 1.14 ms |
| both fusions on vs both off (end-to-end) | **1.55×** |
| sponge input preparation (pad10\*1 + concat + broadcast) into one launch | J stage 192 → 28 µs (Intel) |
| DEC GEMM fused epilogue alone (B=4096, Intel Xe3) | 0.213 → 0.157 ms (**1.36×**) |

`benchmarks/bench_e2e.py` self-checks that the fully fused path and the fully fallen-back path
produce byte-identical output.

---

## 7. Host side, B=4096, Intel

End-to-end is 1.99 ms, of which 0.57 ms is host: `b"".join` 0.25 + H2D 0.28 + D2H 0.03.

| optimisation | effect |
|---|---|
| accept `torch.Tensor` (zero host copy) instead of `list[bytes]` | host cost 0.89 → 0.57 ms |
| adaptive pinned staging for H2D | Intel 0.50 → 0.28 ms; not worth it on GB10 (pageable already ~50 GB/s) |
| per-ciphertext length check O(B) → O(1) | saves 0.13 ms |

Two measurement corrections are recorded here because they changed conclusions:

- pinned staging was first declared a dead end because the memcpy was measured with numpy
  assignment (0.25 ms) instead of `copy_` (0.035 ms).
- the fusion gain was once measured as 0.003 ms because the patch targeted the wrong namespace.

**H2D cannot overlap compute on this Intel driver**: a dual-stream, zero-dependency experiment
measured 1.328 ms against 1.307 ms for serial same-stream. Streaming double-buffering is
therefore ruled out on this platform.

---

## 8. CPU baseline (liboqs 0.16.0, same machine, same ACVP ciphertexts)

| | ML-KEM-768 decaps | | ML-KEM-768 encaps | |
|---|---:|---:|---:|---:|
| liboqs, 1 thread | 9.52 µs/op | 105.1 k/s | 7.82 µs/op | 127.9 k/s |
| liboqs, 16 concurrent processes | **1.375 µs/op** | 727 k/s | **1.109 µs/op** | 902 k/s |

liboqs's `speed_kem` is single-threaded (`OMP_NUM_THREADS` has no effect), so the multi-core
baseline is the aggregate of 16 concurrent processes.

Cross-over against liboqs-16T: decaps needs B >= 1024 on Intel and B >= 256 on GB10; encaps
crosses at about B ≈ 700 on Intel. Full tables in [docs/compare.md](compare.md) §1.7–1.8.

## 9. One-time cost

| | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|---|---:|---:|---:|
| `keyprep` pre-processing | 6.8 ms | **13.5 ms** | 19.8 ms |
| resident VRAM (`W_dec`+`W_enc`, fp16) | 4.2 MB | **7.9 MB** | 12.6 MB |
| `dk` size in a standard implementation | 1632 B | 2400 B | 3168 B |

On-GPU rebuild (`mlkem/kernels/rebuild.py`) is 0.158 ms, i.e. 19.3 ns/op at B=8192 — the resident
VRAM, not the build time, is the real constraint. See [docs/design.md](design.md) §7.

---

## 10. Reproducing

```bash
python benchmarks/bench_e2e.py --batch 8192        # end-to-end, both paths, self-checked
python benchmarks/bench_kernels.py --batch 4096    # per-kernel breakdown
python benchmarks/bench_host.py --batch 4096       # host-side decomposition
python benchmarks/bench_encaps.py                  # encaps throughput
python benchmarks/bench_keygen.py                  # keygen throughput
python benchmarks/bench_rebuild.py                 # on-GPU key rebuild vs cuPQC convention
python benchmarks/bench_mldsa_verify.py --ps ML-DSA-65 --batch 4096 --profile
```

`bench_rebuild.py` and `bench_encaps.py` print the cuPQC same-machine figures alongside ours
only when they detect the matching hardware, since quoting them on another machine would be
invalid.

---

## 11. ML-DSA signature verification (FIPS 204)

Intel Arc B390 / torch 2.15 / triton-xpu 3.9, min-of-15, 32-byte messages. The corpus is
produced by **liboqs 0.16.0 with ML-DSA enabled** under one public key, so the CPU baseline and
the GPU verifier consume exactly the same signatures.

| scheme | B | device only | dev µs/op | e2e | e2e µs/op | liboqs 1T | liboqs 16P | dev vs 16P |
| | | | | | | *(per-algorithm symbols, rebuilt 0.16.0)* | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ML-DSA-44 | 1024 | 2.481 ms | 2.42 | 4.061 ms | 3.97 | 58.7 µs | 10.57 µs | 4.4× |
| ML-DSA-44 | 4096 | 7.698 ms | **1.88** | 13.227 ms | 3.23 | 58.7 µs | 10.57 µs | 5.6× |
| ML-DSA-44 | 8192 | 16.193 ms | 1.98 | 27.611 ms | 3.37 | 58.7 µs | 10.57 µs | 5.3× |
| ML-DSA-65 | 1024 | 2.845 ms | 2.78 | 5.154 ms | 5.03 | 93.0 µs | 13.95 µs | 5.0× |
| ML-DSA-65 | 4096 | 10.123 ms | **2.47** | 16.790 ms | 4.10 | 93.0 µs | 13.95 µs | 5.6× |
| ML-DSA-65 | 8192 | 20.740 ms | 2.53 | 34.063 ms | 4.16 | 93.0 µs | 13.95 µs | 5.5× |
| ML-DSA-87 | 1024 | 3.501 ms | 3.42 | 6.467 ms | 6.31 | 136.1 µs | 20.53 µs | 6.0× |
| ML-DSA-87 | 4096 | 15.488 ms | **3.78** | 23.380 ms | 5.71 | 136.1 µs | 20.53 µs | 5.4× |
| ML-DSA-87 | 8192 | 31.010 ms | 3.78 | 48.123 ms | 5.87 | 136.1 µs | 20.53 µs | 5.4× |

`liboqs 16P` is the aggregate over 16 concurrent **processes** (the same convention as §8 for
> **The `dev vs 16P` column in this table is too generous, and it is known why.** Those CPU
> figures were taken while the machine was running builds and tests, i.e. in a low-clock
> window: ML-DSA-44 one thread reads 58.7 us here and 15.6-16.6 us in an idle window
> (`benchmarks/cpu_baseline.json`). Re-paired in one window (`benchmarks/snapshot.py`), ML-DSA-44
> verification is **1.0-1.8x** the 16-process CPU, not 5.6x. The `vs 1T` claim everywhere else
> in these docs is unaffected in direction but also inflated by up to ~3x.
>
> ML-KEM; threads in one process do not scale on this hybrid 4P+8E+4LP part). Two caveats, both
> inherited from §1: clocks drift between sessions, so only same-process ratios are meaningful;
> and the CPU figures were taken while this machine was running builds and tests, which biases
them *upward* — the ratios should be read as "≈5×", not as four significant figures.

The end-to-end column is 1.6–1.8× the device-only column: one H2D of `B × sig_size`
(13.5 MB at B = 4096) plus the host-side message formatting, then one D2H of `B` bools. The
per-row `format_message` loop used to dominate that gap (a `bytes` allocation per signature);
it is now assembled with numpy for the uniform-length case, which took e2e from 6.0 to 4.1 µs/op
at B = 4096.

Per-stage device breakdown at B = 4096, ML-DSA-65 is in [design.md](design.md) §9: the transform
is 77% of the time (`sib + NTT(z) + NTT(c)` 39%, `MAC + INTT` 38%).

---

## 12. SLH-DSA (FIPS 205) verification

Device-only, `internal` interface (so context / pre-hash formatting is out of the picture),
corpus = valid ACVP signatures tiled to the batch. Min of 3 reps; the CPU column is liboqs
single-thread in the same process on the same signatures.

| set | B | device / sig | liboqs 1T | speed-up |
|---|---:|---:|---:|---:|
| SHAKE-128s | 4096 | **18.7 µs** | 458 µs | 24.5× |
| SHAKE-128f | 1024 | 67.0 µs | 1 333 µs | 19.9× |
| SHAKE-128f | 4096 | **47.8 µs** | 1 333 µs | 27.9× |
| SHAKE-192s | 4096 | **26.5 µs** | 654 µs | 24.6× |
| SHAKE-192f | 4096 | **67.4 µs** | 1 970 µs | 29.2× |
| SHAKE-256s | 4096 | **37.8 µs** | 1 032 µs | 27.3× |
| SHAKE-256f | 4096 | **71.2 µs** | 2 040 µs | 28.7× |

The SHA-2 sets, same machine and same method — cheaper per hash, so ~1.6× faster again:

| set | device / sig | liboqs 1T | speed-up |
|---|---:|---:|---:|
| SHA2-128s | **13.1 µs** | 388 µs | 29.6× |
| SHA2-128f | **27.8 µs** | 1 170 µs | 42.0× |
| SHA2-192s | **17.5 µs** | 575 µs | 32.8× |
| SHA2-192f | **37.9 µs** | 1 595 µs | 42.0× |
| SHA2-256s | **26.0 µs** | 818 µs | 31.4× |
| SHA2-256f | **44.1 µs** | 1 708 µs | 38.7× |

A SHA-2 block compression is 64–80 rounds of add/rotate/xor on 32/64-bit words; Keccak-f[1600] is
24 rounds but each round touches 25 lanes with ~40 operations, and the ratio shows up directly in
these numbers. The price is that FIPS 205's SHA-2 instantiation is *asymmetric*: F (the WOTS chain)
always uses SHA-256, while H and T_l switch to SHA-512 for the 192/256-bit sets, so both
compression kernels and two constant seed states are live per verifier.

Batch curve for 128f, same corpus: B = 1 → 37 ms/sig (400 launches, nothing else), B = 8 →
4.8 ms, B = 32 → 1.2 ms, B = 64 → 600 µs, B = 1024 → 67 µs, B = 4096 → 48 µs. The crossover
against the CPU is around B ≈ 32, and everything below that is launch latency.

Chain compaction is worth a note because it is data-dependent work made regular: a WOTS+ chain
with digit `d` needs `w-1-d` hashes, so the `B × 35` chains of a layer finish at different times.
Sorting them by digit once per layer (and launching each step over the still-running suffix) took
128f from 71 µs to 48 µs — the earlier "15 fixed steps" schedule was doing ~2× the necessary
work.

Against **16 liboqs processes** on the same batch (4096 signatures, wall clock): 359.9 µs per
signature, so the device is 7.5× faster than a loaded 16-core CPU — not 28×. End-to-end (message
bytes in, `bool[B]` out, M' formatting on the host) adds 1–2 µs/signature to the device-only
figure.

Two honest notes on the comparison:

- **The CPU baseline moves with the machine's clock state.** In the evaluation that preceded this
  implementation, the same liboqs call measured 4.4 ms for 128f; on the same machine today it
  measures 1.33 ms (six fresh signatures, 20 reps each, 1302–1354 µs). The 18× above is
  same-session; a 78× figure obtained against a throttled CPU would not be.
- **The literature's 10³× numbers are real but use a different baseline** — the NIST reference
  implementation, not liboqs. `bench_slhdsa_verify.py --cpu` exists so that the comparison stays
  in the same process.

First use of each fused `thash` kernel compiles ~143 lines of Triton (~1.5 s per `(n, dn)` pair,
six pairs total) and is then cached on disk; Triton additionally specialises on the batch size, so
the first verification of a new batch-size class adds ~0.4 s. Measured on a cold cache, the first
128f verification takes ~6 s; that cost is outside the table.

---

## 13. LMS / HSS verification (SP 800-208)

`benchmarks/bench_lms_verify.py` tiles one RFC 8554 test vector across a batch. liboqs does not
implement LMS, so the only comparison available in-repo is the Python reference (a correctness
oracle; its numbers are in the same table only to show the batch effect).

| vector | B | device | per signature | Python ref |
|---|---:|---:|---:|---:|
| RFC 8554 case 1 (H5/W8, 2 levels) | 256 | 7.6 ms | 29.7 µs | — |
| | 1024 | 12.3 ms | 12.1 µs | — |
| | 4096 | 34.8 ms | **8.5 µs** | 2 300 µs |
| RFC 8554 case 2 (H10/W4 + H5/W8, 2 levels) | 4096 | 29.2 ms | **7.1 µs** | 2 530 µs |

The shape of the work explains why the batch helps so much: one LM-OTS chain is up to 255
*sequential* SHA-256 compressions, so a single verification is latency-bound, while a batch of
thousands turns the same chain steps into a full-grid launch. The device figure is ~1 Ghash/s of
32-bit SHA-256 compression, which is ~3× the Keccak permutation rate measured for SLH-DSA (that one
is 64-bit lane arithmetic).

---

## 14. ML-DSA signing

`benchmarks/bench_mldsa_sign.py`, Arc B390, one private key, batch of independent messages:

| set | B=256 | B=1024 | B=4096 | liboqs 1T | liboqs 16P |
|---|---:|---:|---:|---:|---:|
| ML-DSA-44 | 60.4 µs | **43.0 µs** | **33.6 µs** | 97 µs | 28.2 µs |
| ML-DSA-65 | 79.1 µs | **56.6 µs** | — | 152 µs | 28.0 µs |
| ML-DSA-87 | 88.9 µs | **58.0 µs** | — | 186 µs | 27.7 µs |

**Signing now reaches parity with one CPU core at B ≈ 200 and wins above it** (1.7× / 2.0× / 2.3× at
B=1024). What got it there:

- a **fused acceptance-test kernel** (`sign_check_kernel`): the `r0` and `c·t0` bounds, the hint and
  its count and the `z` norm in one program per signature, replacing ~20 elementwise torch ops that
  dominated a tail-sized attempt;
- **speculation**: a row's k-th attempt depends only on κ + k·ℓ, so an open set of n rows evaluates
  up to `spec` candidates in one pipeline pass and keeps the first that accepts. The signature is
  bit-identical (same attempt sequence); a batch of 1024 went from 84 sequential attempts to ~8
  pipeline rounds. `spec_target` keeps each pass near 512-2048 rows, which is the sweet spot on this
  device (swept).
- signatures assembled for the whole accepted batch in numpy instead of per-row `bytes.join`;
- `pointwise_kernel` broadcasts `ĉ` inside the kernel: the host version repeated both operands into
  full `[n·M, 256]` arrays, which at 4096 rows was ~150 MB of memory traffic per attempt;
- **mixed-length messages in one batch**, grouped by padded block count: μ is
  `SHAKE256(H(pk) ‖ M′, 64)` and a shorter `M′` has its pad10*1 earlier inside the last block. The
  sponge absorbs a launch-wide block count, so rows whose padding spans a different number of blocks
  take a second launch; grouping by padded length keeps the common case at one launch. (The 0.7.1
  version instead padded every row to the longest length, which absorbed extra all-zero blocks and
  produced a wrong μ for the short rows; fixed in 0.7.2.)

**Signing is the one operation here where the device loses.** liboqs signs an ML-DSA-44 signature in
97.7 µs on a single core (min of five rounds of 100), so the GPU's 191 µs is 1.96× *slower* — and
against 16 concurrent processes (28.2 µs/sig) it is 6.8× behind. This is not a tuning failure but a shape mismatch: a
signature is a short chain of ~4.5 sequential rejection attempts over a `k·l`-sized batch, so the
device is paying launch latency (~2 ms per attempt, see below) where the CPU pays arithmetic. The
16P column is the same convention as the rest of this document (aggregate over 16 fork processes).

Two things make this different from verification.

**The work is data-dependent.** Each signature retries the rejection loop until `z`, `r0` and `c·t0`
are small enough; the acceptance rate is ~22 %, so a batch of 1024 runs ~4.5 attempts per message
on average — but the slowest message in the batch needed **84** attempts, and the loop cannot finish
before it does. That is why the per-signature cost is set by the *worst* message, not the mean.

**Each attempt is a ~18-launch pipeline** (SHAKE for `y`, NTT, MAC, INTT, `c̃`, SampleInBall, three
pointwise+INTT stages, packing). At ~0.1 ms per dispatch that is ~2 ms per attempt regardless of how
many rows are left, so the 84 attempts of a batch of 1024 cost ~170 ms of mostly-dispatch time. The
profile of one full-batch attempt is flat: NTT 14 %, hint 12 %, the three pointwise/INTT stages 29 %,
decompose 9 %, ExpandMask 8 %, everything else the rest.

### What moved the number, and what did not

| change | effect |
|---|---|
| BitPack of `z` moved to the device (`pack_z_kernel`) — it was ~19 % of the run, 6 000 Python bit-packs per batch | **245 → 191 µs** |
| `pointwise_kernel` instead of a broadcast int64 product on the host, plus `amax` instead of two chained `max` | **191 → 177 µs** |
| a slot/queue model (keep the batch full while the queue lasts) | 196–320 µs — **worse**; the cost tracks rows processed, not attempts |
| fusing the three pointwise products into one launch | 186 µs — **no gain** |

A/B in one session (ML-DSA-44, B=1024, best of 3): **236.0 µs → 176.8 µs, 1.33×**.

### Where the time goes, and why the tail is the problem

At B=1024 a batch runs **84 attempts**: 13 with 64 rows or more, and **71 with fewer**. Those 71
tail attempts process ~500 rows between them but cost about 60 % of the wall clock, because an
attempt costs ~2 ms of dispatch whether it carries 1024 rows or two. The tail exists because the
acceptance rate is ~22 %, so the unluckiest signature in a batch needs ~80 tries and the loop cannot
finish without it; bigger batches do not shorten the tail, they only amortise it (which is exactly
why B=8192 wins).

Three things were tried against that tail and **all three failed**, measured:

| idea | result |
|---|---|
| capture the whole attempt into an `XPUGraph` and replay it for the tail (one submission instead of ~13 launches) | 229 µs/sig at B=1024 — **worse** than 177; capture works on this stack, the replay just is not cheaper than the launches it replaces |
| a slot/queue model keeping the batch full | 196-320 µs — worse |
| fusing the three pointwise products into one launch | 186 µs — no gain |

What is left has real potential but is real work: **fuse the attempt into ~6 kernels** (merge
`ExpandMask`'s sponge and unpack, the three `decompose`/norm-check passes, and the `w1Encode` sponge),
and **overlap two half-batches on two streams** so the latency-bound tail of one hides under the
compute-bound main phase of the other. A persistent, grid-synchronised kernel for a whole attempt is
not expressible in Triton today. Note the ceiling: even a perfect 2× would put small-batch signing at
~90 µs, i.e. level with one CPU core, and nowhere near the 16-process aggregate (28 µs/sig).

---

## 14.5 The synthesis: `docs/patterns.md`

The numbers in §15-§20 exist to support one table: [patterns.md](patterns.md) maps the ten
computational patterns the project set out to cover onto what they actually cost here, states
the rule that decides whether a port is worth doing, and lists the six Triton traps that produced
plausible-but-wrong output before a known-answer vector caught them.

## 15. The primitive layer itself

`python benchmarks/bench_primitive.py --batch 4096 --size 1024` (Arc B390, one process,
min-of-7, device-only). `copy` is measured first and every GB/s cell is a fraction of it, because
"GB/s" without a ceiling says nothing.

| primitive | what | device ms | rate | GB/s | % of copy BW |
|---|---|---:|---:|---:|---:|
| `copy` | read+write, 256 MiB | 2.46 | 217 GB/s | 217 | 100% |
| `keccak` | Keccak-f[1600], 2^18 lanes | 1.44 | 183 M perm/s | 73 | 34% |
| `sha3-256` | 1 KiB messages, 8 blocks | 0.24 | 17.3 M msg/s | 19 | 9% |
| `sha3-512` | 1 KiB messages, 15 blocks | 0.40 | 10.2 M msg/s | 12 | 5% |
| `shake128` | 1 KiB → 32 B | 0.21 | 19.8 M msg/s | 24 | 11% |
| `shake256` | 1 KiB → 32 B | 0.23 | 18.1 M msg/s | 20 | 9% |
| `sha256` | 1 KiB messages | 0.10 | 43.1 M msg/s | 44 | 20% |
| `sha512` | 1 KiB messages | 0.18 | 22.5 M msg/s | 23 | 11% |
| `ntt-mlkem` | q = 3329, 7 layers | 0.26 | 16.0 M poly/s | 33 | 15% |
| `ntt-mldsa` | q = 8380417, Montgomery | 0.79 | 5.2 M poly/s | 11 | 5% |
| `ntt-babybear` | q = 2^31−2^27+1, Montgomery | 0.78 | 5.2 M poly/s | 11 | 5% |
| `chacha20` | RFC 8439, 20 rounds | 0.28 | 14.8 GB/s | 15 | 7% |
| `poseidon2` | BabyBear/16, 8 full + 13 partial | 2.53 | 25.9 M perm/s | 3.3 | 2% |

`poseidon2` was measured at `--batch 65536`, not 4096: with `BLOCK=128` a batch of 4096 is
only 32 programs and runs at 103 ns/permutation instead of 39. Below B ≈ 2048 the grid stops
filling the device entirely (B = 1024 is 820 ns).

**Poseidon2 against the CPU, same machine.** Plonky3 0.8's packed implementation
(`benchmarks/poseidon2_ref`, `-C target-cpu=native`, monty-31x8 AVX2, cargo mode `bench-packed`):

| | 1 thread | 16 processes |
|---|---:|---:|
| Plonky3, packed | 106 ns (9.4 M/s) | **7.0 ns (144 M/s)** |
| this device | — | **38.7 ns (25.9 M/s)** |

So the device is **2.7× one core and 0.18× sixteen processes**. Print the second number, not
the first: a primitive table without a `vs 16P` column invites the same mistake this one did
(§15 used to report only `%copy`, which made ChaCha20 look like it had 14× of headroom when
it is at 0.40× of a 16-process CPU).

Three things worth reading off it:

- **The ALU-bound primitives all land in the 15–45 GB/s band**, i.e. 7–20 % of copy bandwidth:
  a 20-round ARX cipher, a 24-round permutation and a 7-layer NTT are all far from memory-bound on
  this device. Raising them means reducing instructions per byte, not moving fewer bytes.
- **SHA-256 beats SHA-512 per byte** (44 vs 23 GB/s) even though SHA-512 processes 128-byte blocks:
  twice the rounds per block and 64-bit words on a 32-lane backend.
- **q = 8380417 costs 4× q = 3329** (5.2 vs 16.0 M poly/s) even with Montgomery: int64 products,
  and `%` would be a software divide. BabyBear (31 bits, same int64+Montgomery path) pays the same
  4× — that is the price of a 31-bit modulus on this arithmetic model, not of the modulus itself.

Clock caveat as everywhere else in this document: absolute numbers move 2–4× with the iGPU's
frequency state. **This table is a high-clock session** (copy 217 GB/s); an earlier session the
same day measured copy at 110 GB/s, Keccak at 116 M perm/s and ChaCha20 at 13.7 GB/s, with every
`%copy` cell within a point or two of the values above. Quote the ratios, not the absolute rates —
the ratios come from one process and one second, the absolutes do not.

### 15.1 The CPU rival: the column that decides whether a primitive is worth doing

`python benchmarks/bench_cpu.py` measures the same work on the same machine and writes
`benchmarks/cpu_baseline.json`; `bench_primitive.py` joins it in as `vs 1T` / `vs 16P`.
Without that column a table says "this is 7% of copy bandwidth", which reads like 14x of
headroom when the honest answer is "0.4x of sixteen x86 cores".

Three sources, each the fastest implementation that exists for its family: **OpenSSL 3.5**
(AES-NI, SHA-NI, vectorised Keccak, `EVP_MAC` for Poly1305), **liboqs 0.16.0** (the same C the
GPU code is cross-checked against) and **Plonky3 0.8** packed monty-31x8 for Poseidon2.

Poly1305 needs one footnote: it is timed on **1 MiB per MAC**, not 1 KiB. OpenSSL's one-shot
`EVP_MAC` costs ~2.5 µs of init/final per call, so a 1 KiB corpus measures the API rather than
the cipher and reads 0.39 GB/s instead of 7.5. The GPU row is a batched kernel and has no such
per-call cost, which is exactly the kind of asymmetry that makes a naive comparison wrong.

Arc B390 / Core Ultra X7 358H (4P+8E+4LP), governor `powersave` at 400 MHz idle, 16P = sixteen
concurrent **processes** (threads do not scale on this hybrid part), min-of-5.

> **This table is one window's measurement, not the committed baseline.** The baseline the
> documents are computed against is `benchmarks/cpu_baseline.json`; comparing the two shows the
> usual clock drift -- most rows agree within a few per cent, Poly1305's 16P is 15% apart (75.8 here
> against 87.2 there), and Falcon-512's one-thread figure is 1.7x apart (42.2 k/s here against
> 24.8 k/s there -- see the caveat at the end of this section). Quote the ratio from whichever
> window produced the device number, never across two.

| rival | 1T | 16P | scale | source |
|---|---:|---:|---:|---|
| ChaCha20 | 4.73 GB/s | **29.1 GB/s** | 6.1x | openssl |
| ChaCha20-Poly1305 | 3.20 GB/s | 20.7 GB/s | 6.5x | openssl |
| AES-128-CTR | 13.8 GB/s | **166 GB/s** | 12.0x | openssl |
| SHA-256 | 4.17 GB/s | 49.4 GB/s | 11.8x | openssl |
| SHA-512 | 1.10 GB/s | 13.4 GB/s | 12.2x | openssl |
| SHA3-256 | 0.72 GB/s | **7.90 GB/s** | 11.0x | openssl |
| SHA3-512 | 0.38 GB/s | 4.27 GB/s | 11.2x | openssl |
| SHAKE128 | 0.92 GB/s | 9.78 GB/s | 10.7x | openssl |
| SHAKE256 | 0.74 GB/s | 7.70 GB/s | 10.4x | openssl |
| Poseidon2-BabyBear/16 | 9.53 M/s | **134 M/s** | 14.1x | Plonky3 packed |
| ML-DSA-44 verify | 61.5 k/s (16.3 µs) | 946 k/s (1.06 µs) | 15.4x | liboqs |
| Falcon-512 verify | 42.2 k/s (23.7 µs) | 459 k/s (2.18 µs) | 10.9x | liboqs |
| Falcon-padded-512 verify | 24.7 k/s (40.5 µs) | 436 k/s (2.29 µs) | 17.7x | liboqs |
| Poly1305 (1 MiB per MAC) | 7.54 GB/s | **75.8 GB/s** | 10.1x | OpenSSL EVP_MAC |
| SLH-DSA-SHA2-128f verify | 1 045 /s (957 µs) | 14.4 k/s (69 µs) | 13.8x | liboqs |

Read the two orders of magnitude: **what decides whether a port can win is how fast the CPU's best
implementation is per core, and that is set by three things together, not by instruction support
alone.** This section used to claim only the latter; the rows above refute it in both directions --
**SHA-256 has SHA-NI and the device still wins 1.55x**, while **BLAKE3 has no x86 instruction and
loses 0.18x** because its AVX-512 path does sixteen blocks at once. The dedicated instruction, the
width of SIMD the shape admits, and the length of the serial dependency chain decide it together.
`docs/patterns.md` §0 carries the corrected form with all nine rows. AES-NI
makes AES a 166 GB/s wall (a bitsliced GPU AES at a predicted 20–40 GB/s is 0.12–0.23x);
Keccak has no x86 instruction, so SHA3-256 is 0.72 GB/s per core and the device's 31.6 GB/s is
**4.0x the whole 16-process CPU**.

Caveats, both visible in the table: the two Falcon rows disagree by 1.7x on one thread for
identical verification work (clock state, min-of-3), so quote the 16P number or the ratio,
not four significant figures; and the 16P column is the sum of per-process rates measured
while all sixteen run, which is why `scale` is 6x for ChaCha20 (memory-bound) and 15x for the
signatures (register-bound).

**The 16P column was an upper bound, and as of 2026-11-27 it has been re-measured.** Recorded
here so the history is not lost: the scheme benchmarks and `bench_cpu.py` disagreed by 3.4-4.8x on
liboqs ML-DSA-44's sixteen-process verify rate, and a third measurement settled it --

| method | aggregate |
|---|---:|
| `bench_cpu.py`: sum of per-child **minima** of three reps | 949 k ops/s |
| `bench_mldsa_verify.py`: `pool.map` wall time / ops (charged the `CDLL` load inside the region) | 198 k ops/s |
| barrier + pre-generated corpora, verify only | 370 k ops/s |

The `scale` column already gave the bias away: sixteen processes on sixteen cores cannot exceed
16x, and BLAKE3 (19.2x), Falcon-512 (18.1x) and Falcon-padded (19.2x) all did.

`benchmarks/probe_16p_method.py` fixed that one row and left the table alone, because re-measuring
invalidates all seven snapshots at once. `benchmarks/bench_16p_barrier.py` now does the whole
column: one timed round per process behind a barrier, unit counts fixed by the caller, aggregate =
units over the union of the window. It also fixed a second, independent error -- the device hashes
**1024-byte** messages while the CPU rival was pointed at 16 KiB (digests) and 1 MiB (BLAKE3,
Poly1305), so per-message init/final was charged to the device only.

Result, in `benchmarks/table-16p-20261127.md`: the old column over-counted the CPU by **1.25x to
4.25x depending on the row, median 1.94x**. No primitive changed side of 1.0x, but the magnitudes
did, and the worst offenders were the small-message rows the size mismatch had penalised:

| | old `vs 16P` | corrected |
|---|---:|---:|
| SHA-3/512, SHAKE-128/256 | 4.0x | **7.7-8.1x** |
| SHA-512 | 3.4x | **6.97x** |
| 2^20 NTT (one transform) | 4.0x | **7.28x** |
| SHA-256 | 1.56x | **2.80x** |
| SHA-3-256 | 4.10x | **5.15x** |
| BLAKE3 | 0.17x | **0.72x** |
| ChaCha20-Poly1305 | 0.37x | **0.90x** |
| Poly1305 | 0.13x | **0.32x** |
| AES-128-CTR | 0.07x | **0.11x** |
| Poseidon2 | 0.17x | **0.59x** |
| ChaCha20 | 0.45x | **0.65x** |

One row changed side: **ML-DSA-44 verify, 0.57x -> 1.66x**. Falcon-512's device core goes 3.72x ->
5.9x on the same correction. `bench_primitive.py` now prefers `cpu_baseline_barrier.json` and
prints which method it used; `benchmarks/cpu_baseline.json` is kept unchanged, as the record of the
sum-of-minima method, because the seven snapshots above were produced against it.

**A fourth bias, found the same day: the power source.** This laptop reports
`/sys/class/power_supply/BAT0/status = Discharging` on battery and drops the ACPI platform profile
to `low-power`, which caps the CPU near 0.9 GHz against 3.4 GHz on mains -- a **4.6x** swing that
makes every CPU rate low and pushes `scale` above the core count (a throttled run printed 16.8x on
sixteen threads). It very likely also explains a `UR_RESULT_ERROR_DEVICE_LOST` that killed a test
run: the iGPU is power-capped at the same time, and the level-zero backend reported dead after
repeated `xe` CCS job timeouts. A snapshot generated in that state was written over the good
baseline; it was restored from git, and `bench_16p_barrier.py` now (a) records the power state in
its `meta`, (b) warns when the battery, the profile or the clock say the numbers are not
comparable, (c) warns when a remeasured 1T rate drifts more than 1.5x from the stored baseline, and
(d) **refuses to write a baseline** from a machine in that state unless `--force` is passed.

**What this does not fix.** The device's clock state still moves 2-4x between runs, and it moved
2x *within* the session that produced the table (the same 2^20 transform read 810, 1160 and 602
transforms/s twenty minutes apart). Absolute GB/s in a snapshot is worth less than the ratio
measured beside it. And `16P` itself has two defensible definitions that differ by 1.10-1.43x on
this 4P+8E+4LP part (batch completion versus summed steady rates); the table quotes the batch one,
which is the conservative choice for the device.

## 16. Falcon-512 verification (FIPS 206 draft)

`python benchmarks/bench_falcon_verify.py --ps falcon-512 --batch 64,512,4096 --profile`.
The corpus is liboqs 0.16.0's `pqclean_falcon-512` under one public key, so the CPU baseline
and the device consume exactly the same signatures. Arc B390, min-of-10, device synced.

| B | dev ms | **dev µs/op** | dev Mops/s | e2e ms | e2e µs/op | valid |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.287 | 4.48 | 0.22 | 15.81 | 247.0 | 64/64 |
| 512 | 0.531 | **1.04** | 0.96 | 32.07 | 62.6 | 512/512 |
| 4096 | 2.677 | **0.65** | 1.53 | 126.97 | 31.0 | 4096/4096 |

Against the CPU (liboqs 0.16.0, same machine, from `cpu_baseline.json`): one thread is
**23.7 µs/op**, sixteen processes are **2.18 µs/op**. The device core is therefore
**35× one thread and 3.2× sixteen processes** — the first primitive in this repo that beats
the 16-process baseline by more than 2× and is not a scheme's fused pipeline.

Two numbers, because they answer different questions:

- **dev/op** is `Verifier.verify_core_torch`: `NTT(s2)`, the pointwise product, the inverse
  NTT and the norm check. The transforms are **39%** of it (0.14 + 0.11 µs at B = 4096), which
  is why the rest of the row is one fused kernel rather than three.
- **e2e/op** is `Verifier.verify`, which also pays for the host side. Measured breakdown at
  B = 4096, same window (`benchmarks/snapshot.py`):

| stage | where | µs/signature | share of e2e |
|---|---|---:|---:|
| device core (2 NTTs, pointwise, norm) | device | **0.61** | 3% |
| signature decode (`comp_decode`) | host, numpy | **13.5** | 81% |
| `HashToPoint` (SHAKE256 + rejection) | **device** since 1.1.0 | **0.6** | 3% |
| e2e total | | **16.6** | 100% |

The decoder's own history is 134 µs (per-signature Python) -> 21.4 µs (batched, six gathers per
coefficient) -> 16.8 µs (one 32-bit window per coefficient, which answers both the 8-bit header and
the 16-bit unary run at once) -> **15.6 µs** (int32 state) -> **13.5 µs** (every 32-bit window precomputed once, so the scan
does one gather per coefficient instead of four gathers and seven arithmetic ops).

That last step is the general lesson of this kernel: at a batch of 4096 a numpy call costs ~7 µs
whether it touches 1 element or 4096, so what a scan like this pays for is the **number of calls**,
not the work inside them.

**And the obvious next idea does not follow from it.** Decoding two coefficients per step would
halve the 512 steps, but the second coefficient's position depends on the first's length, so it
costs the same ~13 calls as the first -- 256 steps of ~26 calls is the same 6700 calls. The only
way down is fewer calls *per coefficient*, and a prefix code gives no way to batch across
coefficients: that is exactly the property that made the device decoder 35x slower when it was tried
(`§16`; 190x by today's 13.5 us host decoder). It is still 81% of the end-to-end cost, and still the right
place for the next attempt -- on the host, where a vectorised scan beats the device's per-coefficient
gather by 35x.

`HashToPoint` moved to the device after a probe measured both halves of it
(`probes/hashpoint-shape-probe.py`, in the workspace beside this repository -- the probes back
claims in the evaluation documents as well, so they are not shipped in the package): the sponge
costs 0.10 µs/signature and the
rejection-plus-prefix-sum filter 0.47 µs, against 9.3 µs on the host -- a 16x win, and the
prefix-sum-plus-scatter shape is the one a multi-pass NTT will need. `comp_decode` did **not**
survive the same test (see below). The decode share was 96% before the host decoder was cut from
134 to 13.5 us; it is **81%** now (13.5 of 16.7 us).

  `HashToPoint` went to the device (16x win, measured, wired into `Verifier.verify` for
  uniform-length batches with a host fallback for mixed ones). `comp_decode` -- the other 81% of the
  end-to-end cost -- **was tried and lost**. The
  kernel is one lane per signature with 512 sequential steps (the code is a prefix code, so
  there is no way to bit-gather a batch), which makes every step a 3-byte gather at a
  data-dependent address plus two table lookups for the unary run length. Measured: **2.6
  ms/signature on the device against 75 µs on the host**, 35x slower, and the first version was
  also wrong (the per-lane bit-position bookkeeping needs more care than the 40-line kernel got).

  The reason is the same one the AES section ran into from the other side: the host numpy
  version processes *all signatures in lockstep* against contiguous memory, so it is a
  vectorised scan, while the device version is a divergent gather per coefficient. The lesson is
  recorded here so the next attempt starts from it: for Falcon, decode on the host (or write a
  warp-cooperative decoder that keeps the gathers coalesced); hash on the device.

Two things worth knowing before reading Falcon numbers anywhere:

- **The verdict is a norm bound, not an equality.** The corpus's squared norms sit around
  30.1M against the bound 34.03M, so ~12% of slack remains and a one-unit perturbation of the
  hash point can still verify (`tests/test_falcon_gpu.py` pins this). Flipping a byte inside
  the coefficient stream moves s2 by 2^7-ish, which is 10^4 times the slack, so the rejection
  tests are not luck -- but "valid" and "produced by the signer" are not the same statement.
- **One harness number, one measurement.** The summary line under the table used to re-time the
  device core after the table had already measured it, which under clock drift printed 1.11 µs
  where the table said 0.60 -- two numbers for one quantity. It reuses the table's last row now.
- **Signatures are variable length** (651-662 bytes observed, 666 for Falcon-padded). The
  decoder is a prefix code, so it cannot be vectorised as a bit gather; the batch decoder is
  vectorised *across signatures* instead (one numpy op per coefficient index), which is what
  took it from 134 µs to 24 µs per signature.

## 17. Poly1305 and ChaCha20-Poly1305

`python benchmarks/bench_primitive.py --only poly1305,chacha20-poly1305 --batch 65536`.
Arc B390, 1 KiB messages, min-of-7.

| primitive | B | rate | GB/s | %copy | vs 1T | vs 16P |
|---|---:|---:|---:|---:|---:|---:|
| `poly1305` | 4096 | 8.69 GB/s | 8.7 | 4% | n/a | n/a |
| `poly1305` | 65536 | **11.37 GB/s** | 11.4 | 5% | n/a | n/a |
| `chacha20-poly1305` | 4096 | 3.82 GB/s | 7.6 | 4% | 2.38x | 0.37x |
| `chacha20-poly1305` | 65536 | **4.50 GB/s** | 9.0 | 4% | 2.81x | 0.44x |

`rate` for the AEAD row is plaintext throughput (the GB/s column counts read+write traffic).
The CPU rival is OpenSSL's ChaCha20-Poly1305: 3.26 GB/s on one thread, 20.7 GB/s over sixteen
processes. **Poly1305 is the cheapest primitive to be honest about**: 11.4 GB/s is 1.2x the
single-core OpenSSL Poly1305 (~9.6 GB/s, derived from the AEAD and raw ChaCha20 rates) and
about **0.13x of sixteen processes**, which is the same shape as every other ALU-bound row.
It is here for coverage -- the AEAD row now exists -- not for a speed story.

Two things this pair cost, both worth recording:

- **One `->cpu()` turns 4.5 GB/s into 2.1 GB/s.** The first version of the AEAD handed the
  Poly1305 kernel host buffers (`poly_key.cpu().numpy()`), i.e. a 4 MB round trip per call at
  B = 4096. Keeping the tensors on the device took the seal from 2.10 to 4.04 GB/s; nothing
  about the kernels changed.
- **The MAC pads, the ciphertext does not.** RFC 8439's `mac_data` is `aad || pad16 || ct ||
  pad16 || le64(len(aad)) || le64(len(ct))`, so the device Poly1305 only ever sees whole
  16-byte blocks, while the ciphertext keeps its exact length. The AEAD therefore supports
  arbitrary message lengths even though `poly1305_torch` itself does not.

## 18. AES-128-CTR through a T-table: the gather surprise

`python benchmarks/bench_primitive.py --only aes128-ctr --batch 4096,65536`. Arc B390, 1 KiB
streams, min-of-7, key schedule hoisted out of the timing.

| B | rate | GB/s | vs 1T (AES-NI) | vs 16P |
|---:|---:|---:|---:|---:|
| 4096 | 5.01 GB/s | 5.0 | 0.36x | 0.03x |
| 65536 | **10.68 GB/s** | 10.7 | **0.77x** | 0.06x |

**This is 12x better than the prediction it was built to confirm.** `eval-triton-crypto-
primitives.md` §4.7 expected 0.7-0.85 GB/s, extrapolating from the primitive probe's
random-gather rate of 7-8.5 Gload/s at 10 lookups per byte. Measured, the same cipher does
10.7 GB/s — 0.77x of one AES-NI core, not 0.05x.

Why the extrapolation was wrong: **the gather rate is a property of the table's residency, not
of the instruction**. This table is 4.25 KiB and stays in L1; every lane walks 64 consecutive
blocks, so the 160 lookups per block are issue-bound L1 hits rather than the probe's random
accesses over an 8 KiB table with data-dependent addresses. The lesson generalises: "gather is
10-13x slower than a coalesced load" (the probe's §6 number) is true for a *cold, divergent*
gather and badly pessimistic for a hot table. Any future "table vs compute" decision needs the
table size and the access pattern pinned down first.

What survives from the original argument: **the port is still not worth doing**. 0.77x of a
single AES-NI core and 0.06x of sixteen processes, for 2-3 weeks of bitslicing work predicted at
20-40 GB/s (0.12-0.23x of 16P). And the same 12x uncertainty applies to that prediction.

## 19. Exact int8 GEMM: the substrate for a finite-field GEMM

`python benchmarks/bench_primitive.py --only gemm-int8`. One number being measured: `int8 x
int8 -> int32` is **exact** on the matrix unit (no rounding), which is what lets a GF(2^8)
product be assembled from per-bit-plane GEMMs.

| n (square) | ms | GMAC/s | TMAC/s | equivalent int8 TOPS |
|---:|---:|---:|---:|---:|
| 256 | 0.093 | 180 | 0.18 | 0.36 |
| 512 | 0.161 | 834 | 0.83 | 1.7 |
| 1024 | 0.554 | **1 938** | **1.94** | 3.9 |
| 2048 | 2.167 | **3 965** | **3.96** | 7.9 |

The eval's int8 estimate was ~50 TOPS (2x the measured 26-29 TFLOPS fp16), i.e. ~25 TMAC/s.
The basic Triton matmul here reaches **16% of that at n = 2048** with ML-KEM's tile table
(128x128x32) -- no autotuning, no swizzling, one configuration.

That gap matters for `PLAN_ffgemm.md` rather than for this repository: the bit-plane strategy
pays 8x (one GEMM per plane), so ~4 TMAC/s of int8 becomes **~0.5 TMAC/s of GF(2^8) MACs**. The
"6 T GF-MAC/s" in `eval-triton-crypto-primitives.md` §4.9 assumed near-peak int8; the honest
figure is 8-12x lower until the matmul itself is tuned. Measure the GF(2^8) product end to
end before assuming the matrix unit wins.

The bit-plane *arithmetic* is checked in `tests/test_gemm_int8.py`: 64 tiny int8 GEMMs,
XOR-ed through the AES polynomial's reduction masks, are compared against a schoolbook
GF(2^8) matrix product.

## 20. One window, one file: `benchmarks/snapshot.py`

The discipline in §1 is "same process, same second, quote the ratios" — and the reports that
go with it were assembled by hand from separate runs. `snapshot.py` makes the discipline the
default:

    python benchmarks/snapshot.py                # everything, ~10 minutes
    python benchmarks/snapshot.py --skip-cpu     # reuse cpu_baseline.json
    python benchmarks/snapshot.py --only primitive,falcon

It shells out to the existing scripts (each sets up its own liboqs/ACVP corpus and prints its
own copy-bandwidth header — re-implementing that would be a second set of conventions to keep
in sync) and wraps them in an envelope: one timestamp, the git revision, and one markdown file
whose sections are the eight measurements above. A sibling `.json` records each section's
command, return code and wall time, so a snapshot can be diffed against the next one.

Snapshots are gitignored: they are artifacts of a window, not sources.

## 21. BLAKE3: the tree hash, and the UB that hid for four sessions

`python benchmarks/bench_primitive.py --only blake3 --size 16384 --batch 4096`. Arc B390,
min-of-7, one launch per block index plus one per tree level.

| message | device | CPU 1T (official Rust) | CPU 16P | vs 1T | vs 16P |
|---|---:|---:|---:|---:|---:|
| 1 KiB (1 chunk) | **10.1 GB/s** | 2.93 GB/s | 51.9 GB/s | 3.45x | **0.19x** |
| 16 KiB (16 chunks) | **9.9 GB/s** | 2.93 GB/s | 51.9 GB/s | 3.40x | 0.19x |

Flat across message size, i.e. ALU-bound as predicted (7 rounds x 8 G x ~20 ops per 64-byte
block). The verdict is the one the eval predicted for the ARX family: **3.4x one core, 0.19x
sixteen processes** -- the tree machinery is correct and costs ~2% (1 KiB is one chunk, 16 KiB is
five tree levels, and the throughput barely moves).

**The bug is worth more than the number.** For four sessions this kernel "stored zeros": the
store mask, the block count, the flags (11 = CHUNK_START|CHUNK_END|ROOT) and the block length all
read back *correct* from inside the same launch, while the value being stored was zero. Every
piece worked in isolation -- the compression, one block step through a global buffer, a
sixteen-launch chain through that buffer. The cause was one line:

```python
v13 = counter >> 32        # counter is uint32 -> shift by the bit width -> UB
```

LLVM is allowed to do anything with that, and this backend chose "the kernel computes zero",
with only a `UserWarning` ("Value 32 exceeds the maximum bitwidth") that a test log filters out.
The counter is now passed as an explicit `(low, high)` pair, and the regression guard is a
**multi-chunk** vector: any message over 1 KiB has a non-zero chunk counter, which no 64-byte KAT
can exercise.

That is the sixth entry for the "Triton 表达缺口" table, and it is the only one that produced
*plausible* output rather than an error or a wrong answer.

## 22. The 2^20-point NTT: the one primitive whose cost is memory traffic

`python benchmarks/bench_primitive.py --only ntt-2^20`. BabyBear, 10 radix-4 passes (20 radix-2
stages fused two at a time), one launch
per pass, data in global memory.

| | device | CPU 1T | CPU 16P | vs 1T | vs 16P |
|---|---:|---:|---:|---:|---:|
| 2^20 forward NTT | **0.85–1.2 ms** | **49 ms** | **6.27 ms amortised** | **40–57x** | **4.0–7.3x** |
| traffic (10 passes x 8 MB, radix-4) | 84 MB | | | | |
| last window measured | 0.862 ms -> 56.1x / **7.28x** | | | | |
| achieved (fast clock) | 95 GB/s (**44%** of copy) | 2.4 GB/s | 30 GB/s | | |

> The `% of copy` figure here was **44%, not 72/88%**: the benchmark's advertised traffic was
> computed from the radix-2 pass count while radix-4 runs ten passes, so both GB/s and `%copy`
> were overstated by 2x until the bench started reading `LargeNTT.n_passes`.

Two numbers per cell because this transform is the one row whose result moves with the clock
state rather than with its own code: **0.85 ms** when the iGPU is at its top clock (**98 GB/s** of
traffic, **45%** of copy bandwidth, against a 0.39 ms pure-traffic floor) and **1.2 ms** at the
low end. Both are measured in one process, min of alternating runs. (The older 0.98 / 1.8 ms pair
in this paragraph was the radix-2 shape at its own two clock states.)

**Seven rows have `vs 16P` above 1** -- SHA-2 and SHA-3/SHAKE (1.55-4.11x) and this one (5.9x, or
4.1x after the CPU rival was measured as 1.5x too slow). What is unique here is the *reason*:
every other winner is ALU-limited and wins because the CPU cannot execute its shape, while this
one is limited by moving bytes -- 84 MB per transform at radix-4 (160 MB at radix-2), i.e. one
transform costs about as long as copying 190 MB -- which sixteen cores cannot beat by computing
harder.

### 22.1 Getting from 10.8 ms to 2.2 ms (5x, three changes)

The first correct version took 10.81 ms and was 7x off a naive "160 MB at copy bandwidth"
roofline. Three fixes, in order of what they were worth:

1. **The pre/post twiddle tables were rebuilt on every call** -- a 2^20-element numpy pass with a
   `powmod`-style loop, costing several milliseconds, i.e. more than the transform it fed. They
   are now built in log n passes and cached per instance and per device (1686 ms -> 10.8 ms
   before this round's other fixes, then 7.8 -> 2.2).
2. **The block/j split did an int64 division per element per pass** (`offs // half`, with `half`
   a runtime value, so LLVM cannot strength-reduce it). `half` is always a power of two, so it is
   a shift and a mask now. A `(block, tile)` grid was tried instead and was *much* worse: the
   late stages have `half = 1`, so that grid degenerates into 524288 one-element programs.
3. **Tile 1024, 8 warps** rather than 4096/4: the swept table runs 41 ms (2-D grid) / 3.55 ms
   (2048/4) / **1.87 ms** (1024/8) for the pass sequence alone.
4. **The negacyclic pre- and post-twiddles are folded into the first and last passes.** They are
   index-dependent only, so the stage that already touches every element can apply them on the way
   in or out: one launch and 8 MB less per direction. 2.2 -> **1.0-1.8 ms**, and the inverse gets
   the same treatment (0.99 ms at the fast clock) instead of a separate `x_j *= post_j` pass.

### 22.2 Radix-4 halves the traffic and buys 10-20%, not 50%

`LargeNTT(n, q, radix=4)` fuses two radix-2 stages per pass (ten passes instead of twenty) and is
bit-identical to the radix-2 result, verified for N = 2^4 .. 2^20 including the odd-`log2` case
where one radix-2 pass is left over. Paired in one window, eight alternating rounds:

| | min | median | traffic | achieved |
|---|---:|---:|---:|---:|
| radix-2 (19 passes, pre-twiddle fused) | 1.50 ms | 1.67 ms | 152 MB | 101 GB/s |
| **radix-4 (10 passes, pre-twiddle fused)** | **1.34 ms** | **1.36 ms** | 88 MB | 66 GB/s |

So halving the traffic does **not** halve the time: the transform is limited by pass latency and
by the modular multiplies, not by bytes. That also means the "0.74 ms roofline" is not the
binding constraint at this size.

**radix-8 is an inference from that, not a measurement** -- said plainly so nobody quotes it as
one. Going from 10 passes to 7 would be a 30% cut in passes, and the observed elasticity here is
11-19% for a 47% cut (19 -> 10 passes), so radix-8 projects to roughly **4-6%** at the same
per-pass cost -- below the noise the clock alone produces on this iGPU. It was not implemented.

**A correction worth keeping**: an earlier reading of the same numbers, taken across two windows,
concluded radix-4 was *slower* (2.63 against 2.13 ms). The paired measurement above says the
opposite by ~15%. This is §1's discipline in action -- the iGPU's clock moved, the ratios did
not, and only the paired run can tell.

Radix-4 stays opt-in (`radix=2` is the default): it is 10-20% faster for twice the code and one
extra kernel, and the win is smaller than the run-to-run spread.

**What that rules out.** Radix-8 is not the next step either: it would cut the traffic another
30% for even more arithmetic per pass, and the radix-4 experiment already showed that traffic is
not what binds *at the low clock*. At the high clock the picture is different -- 0.85 ms at
98 GB/s is 45% of copy bandwidth -- so the transform changes regime with the clock, and a tuning
decision made in one regime can be wrong in the other. That is the same trap as §22.2, one level
deeper.

What is left either way: collapse the tail stages (`m <= 64`) into a single register-resident
pass that does one whole block per program -- that targets the *low* clock, where the transform is
latency-bound.

**That last idea does not work in Triton, and the reason is a language limit rather than a
performance one.** A 64-coefficient block in registers needs the DIF butterfly pairing
`x[j]` with `x[j + m/2]` -- a *half split* of the tile. Measured on triton-xpu 3.9:

- `tl.split` only handles a last dimension of size 2, so `reshape(v, (32, 2))` + `split` gives the
  **even/odd** (stride-2) pairing, which is the DIT order, not the DIF one;
- slicing a 2-D tile (`reshape(v, (2, 32))[0, :]`) is a `CompilationError`, so the halves cannot be
  separated at all.

Switching the tail to DIT order would need the data permuted into bit-reversed order first, and
that permutation costs a full pass -- exactly the traffic the collapse was meant to save. So the
tail stays as five separate passes, and this is recorded so the next attempt does not start from
the same expectation.

**Barrett is not worth it, and that is measured.** Since `Q` is a `tl.constexpr`, LLVM already
turns `a * b % Q` into a multiply-high sequence; hand-rolling the BabyBear fold
(`2^31 = 2^27 - 1`, so `x = lo + (x >> 31) * (2^27 - 1)`) came out **1.02x** the modulo path over
an alternating A/B on a 2^20 multiplies kernel -- inside the noise -- and needed a third fold
round to be correct at all, which would have made it slower. Recorded so the next attempt starts
from the measurement.

The CPU rival is **`benchmarks/ntt_large_cpu_rival_r4.c`** -- the same Stockham algorithm at
**radix-4**, `gcc -O3 -march=native -fopenmp`, with 16P being sixteen *processes* at
`OMP_NUM_THREADS=1`.

**The rival got stronger during this work, and the ratios moved with it.** Two attempts:

- *division-free reduction*: `% Q` compiles to a multiply-shift because `Q` is a compile-time
  constant, so hand-rolling the BabyBear fold (2^31 = 2^27 - 1) made the C code **123x slower** --
  the fold converges too slowly from a 62-bit product and needed a subtraction loop. Same lesson as
  the GPU Barrett probe (§22.2), much more sharply.
- *radix-4*: fusing two stages per pass halves the traffic and is **1.48x faster on one thread**
  (48.7 ms against 71.9), which moves the device's `vs 16P` from 2.9-6.2x to **2.9-4.1x**.

So the earlier "70x one core, 5.9x sixteen processes" was computed against a rival that was 1.5x
too slow. The conclusion survives -- the device still wins against sixteen processes -- but by less
than the first number claimed, and that is exactly what §6.5 of the overnight report warned about.

Three things worth reading off it:

- **This is the only primitive in the repository whose cost is dominated by memory traffic.**
  Everything else (SHA-3, ChaCha20, Poseidon2, Poly1305, AES, BLAKE3) keeps its working set in
  registers and moves 3-31 GB/s of *useful* data; this one moves **84 MB** per transform at the
  default radix-4 (160 MB at radix-2) because every pass touches the whole array. A pure-traffic
  floor at 217 GB/s is 0.39 ms, and the transform runs at 0.85 ms = **45%** of copy bandwidth.
  §22.2 shows the gap is not bandwidth: halving the traffic from radix-2 to radix-4 buys only
  10-20%.
- **The first correct implementation was 15x off its own roofline** (10.8 ms against the radix-2
  floor of 0.71 ms), moving traffic at 15.5 GB/s. The stages were one launch each, the later ones
  accessed with a stride that defeats coalescing, and the passes were radix-2 -- twenty of them,
  against ten for radix-4 (an earlier version of this line said "7 radix-4 passes", which was
  wrong: radix-4 halves the count, it does not quarter it).
- **The first version actually lost: 0.49x against sixteen processes** (and 6.5x against one
  core). That is the same shape as every other ALU-bound row -- the CPU's sixteen cores brought
  30 GB/s of their own -- and it is why the transform needed the three changes above before it
  became the only row that wins for a *memory* reason rather than an instruction one.

What it does buy: the algorithm is `2^20` in a repository whose in-register kernel stops at 1024,
it agrees exactly with the numpy reference and the in-register `NTTDomain` at sizes both can do,
and the pass structure is the one BLAKE3's tree proved.

### 15.2 A rival can silently degrade, and the table will believe it

A full snapshot run once reported Poseidon2 at 1.25x the sixteen-process CPU where every earlier
window said 0.19x. Nothing about the device had changed: `benchmarks/poseidon2_ref` had been
rebuilt **without** `RUSTFLAGS="-C target-cpu=native"`, so Plonky3's `Packing::WIDTH` fell back
to 1 and the binary measured the scalar path -- 736 ns/permutation against 105, a rival 7x too
slow.

The harness now reads the `lanes` field that binary prints and refuses anything below 4, with the
rebuild command in the error message. The general form of the lesson: **a ratio is only as
trustworthy as its denominator**, and a benchmark harness that cannot detect a degraded baseline
will publish the wrong winner with complete confidence.

The same trap exists for `benchmarks/blake3_ref` (the `blake3` crate picks its SIMD path at
runtime, so it is safe) and for `benchmarks/ntt_cpu` (a plain `gcc -O3 -march=native` build, where
`-march=native` is the thing to check).

### 22.3 The product, which is what a prover actually calls

`LargeNTT.mul_torch(a, b)` is the negacyclic product in `Z_q[X]/(X^n+1)`: forward both operands,
multiply pointwise, inverse once. At 2^20 that is **3.36 ms** (radix-4 default; 42 passes, because
the inverse was still 19 radix-2 passes against the forward's 10).

**The published ratio for this row was wrong by a factor of three, in the CPU's favour.** An
earlier revision said "13x one core, 0.94x sixteen processes"; both numbers divided the device's
**products per second** by the CPU's **transforms per second**. A product is three transforms, so
the CPU side needed a `/3`. In products per second the device wins by **3.96x sixteen processes**
and 45.7x one core (same window: device 297.8 products/s, CPU 225.7 transforms/s = 75.2
products/s). The consistency check that would have caught it: the product ratio has to land near
the single-transform ratio (4.0x here), because both sides do the same three transforms -- and it
does, once the units match. `benchmarks/probe_ntt_product.py` now prints both conventions side by
side so the error cannot come back invisibly.

`mul_fused_torch` is the same product in **32 passes**: the inverse is fused two stages per pass
(`_inv2x2_kernel`, verified bit-identical to the stage-by-stage inverse for N = 2^4..2^14) and the
pointwise multiply rides in the last forward pass's epilogue instead of a separate 8 MB pass. That
is **2.41 ms against 3.36 ms, 1.40x**, and **5.53x sixteen processes**. It is not the default yet:
the fusion changes no result, but the default path is what every published number was measured on,
so switching it is a separate decision.

Verification is in two layers, because a 2^20 schoolbook product does not exist:

- random operands at 2^10, device against the numpy path (`mul_np`), plus the schoolbook
  negacyclic convolution at 2^8;
- **monomials at 2^20**: `X^i * X^j` is either `X^(i+j)` or `-X^(i+j-n)`, an exact closed form that
  exercises the transform, the pointwise multiply, the inverse and the `X^n + 1` wrap at the
  target size.

**The bug that layer two caught is worth recording.** The first version cached the pointwise
multiplier the way it caches the twiddle tables -- keyed by `(role, length, device)` -- so the
second call to `mul_torch` multiplied by the *first* call's operand. The single-transform tests
all passed (forward, inverse, round trip), and the first monomial case passed too; only comparing
device against numpy on a later case showed the disagreement. A cache keyed by shape instead of
identity is exactly the kind of bug that a same-shape random test will not find.

### 19.3 The end-to-end GF(2^8) product: what the matrix unit actually buys

`PLAN_ffgemm.md` rests on one assumption -- that the matrix unit can carry a finite-field product
by splitting both operands into bit planes. `primitives/gemm.py` now exposes that construction
(`gf2_8_matmul_torch`, 64 int8 GEMMs XOR-ed through the reduction masks), so the assumption has a
number. `python probes/gf2-8-matmul-probe.py --n 1024` (and `--n 512` for the small case):

| size | one int8 GEMM | **GF(2^8) product** | ratio | numpy, same recipe |
|---|---:|---:|---:|---:|
| 512^3 | 805–843 GMAC/s | **13.7–15.3 GMAC/s** | 54–56x | 0.024 GMAC/s |
| 1024^3 | **2 040 GMAC/s** | **33.5–33.8 GMAC/s** | 60–61x | not run (minutes) |

**The rate is size-dependent by 2.2x, and the small size is the misleading one**: at 512^3 the int8
GEMM itself only reaches 805 GMAC/s (it is launch-bound), while at 1024^3 it reaches 2 040 -- so
the field product goes with it. Quote 1024^3, and say so, exactly as §19 does for the GEMM row
(1.94 TMAC/s at 1024 against 3.96 at 2048).

Three things follow.

- **The 64-plane decomposition costs 54–61x, not 64x** -- the reduction masks fold some planes
  together -- so a GF(2^8) multiply runs at about **1/57 of the int8 rate** at its best size.
- **Against numpy the device is 636x faster**, but that number flatters the GPU: numpy has no int8
  GEMM (it loops), while a VNNI-tuned CPU library would be much closer. This is an upper bound on
  the GPU's advantage, not the advantage.
- **The honest summary for the mainline**: a GF(2^8) product at **33 GMAC/s (1024^3)** is the
  ceiling the matrix unit offers here, and whether that beats a bit-sliced CPU implementation is the question FFGEMM
  still has to answer with its own rival. What this rules out is the optimistic reading of the
  3.96 TMAC/s int8 figure -- the field product is 55x cheaper per multiply before any
  field-specific overhead.
