# Measured results

Everything here was measured on the two machines in §1. Where a number is derived rather than
measured, it says so. Comparison against other implementations lives in
[docs/compare.md](compare.md); this file is the raw record.

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

On-GPU rebuild (`kernels/rebuild.py`) is 0.158 ms, i.e. 19.3 ns/op at B=8192 — the resident
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
ML-KEM; threads in one process do not scale on this hybrid 4P+8E+4LP part). Two caveats, both
inherited from §1: clocks drift between sessions, so only same-process ratios are meaningful;
and the CPU figures were taken while this machine was running builds and tests, which biases
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
