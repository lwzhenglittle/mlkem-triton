# Systematic comparison with existing GPU Kyber / ML-KEM implementations

> "Ours" = `mlkem-triton`. Objects: NVIDIA cuPQC, HI-Kyber (Nanjing University of Posts and
> Telecommunications, 2023), ConvKyber (TCHES 2024), and the OAK paper from
> `dspace.hansung.ac.kr`. **Every number for our implementation is measured**; machines,
> commands and full tables are in [docs/benchmarks.md](benchmarks.md). Every number for the
> others is sourced, and where none could be found this document says "unavailable" rather than
> guessing.

---

## 0. Bottom line

### 0.1 Same machine, same operation: 1.61× faster than cuPQC

cuPQC 0.6.0 was compiled and run on a **GB10 / sm_121** — not in NVIDIA's official support list,
but `-dlto` generates the SASS at link time and the result is correct and fast. That gives a
clean comparison: same card, same operation, both reporting batched throughput.

| ML-KEM-768 decaps | hardware | SMs | throughput | per op | per SM | batch | timing |
|---|---|---:|---:|---:|---:|---:|---|
| **cuPQC-PK 0.6.0** (BD=32, tuned by me) | **GB10** | 48 | **4.96 M/s** | 202 ns | 103 k/SM | 262,144 | pure kernel (cudaEvent) |
| **ours** | **GB10** | 48 | **8.0 M/s** | 125 ns | **167 k/SM** | 8,192 | device-side (13 launches included) |
| | | | **1.61×** | | **1.61×** | | |
| cuPQC-PK 0.6.0 (**SDK default BD=128**) | GB10 | 48 | 4.27 M/s | 234 ns | 89 k/SM | 1,048,576 | pure kernel |
| **→ out-of-the-box ratio** | | | | | | | **1.86×** |
| cuPQC-PK, vendor figure | H100 | 132 | 8.5 M/s | 118 ns | 64 k/SM | not stated | not stated |

**On the same GB10, on the same complete decaps, our throughput is 1.61× cuPQC's** — and our
convention is the less favourable one, since it counts 13 Python-side Triton launches. Cross-
generation: cuPQC reaches only 58% of its own H100 figure on GB10, while we reach 8.0 M/s on
GB10, i.e. 94% of cuPQC's 8.5 M/s on an H100.

A three-round alternating re-check in separate processes (they cannot share a process), sampling
`clocks.sm` around each round, gave **1.86× / 1.86× / 1.85×** with both sides at 2398–2411 MHz.
Note that `bench768` in the cuPQC build tree is the **SDK default BlockDim=128** build; BD=32 was
compiled separately. Hence two honest ratios: **1.86× out of the box**, **1.60–1.61× tuned**.

### 0.2 Decomposing the 1.61×: 86% kernel, 14% pre-expansion

cuPQC's API is stateless, so **every ciphertext** re-runs k² SampleNTT calls (SHAKE-128, 840 B
each) to obtain `Â`; we expand `Â` into a resident Toeplitz matrix once. `kernels/rebuild.py`
closes the gap by rebuilding `W_dec`/`W_enc` from `dk` on the GPU: 0.158 ms, **19.3 ns/op at
B=8192**, taking us from 125 to 144 ns/op — **still 1.39× cuPQC**. So ~86% of the advantage is the
kernel and 14% is not re-expanding `Â`.

### 0.3 encaps: 2.27× faster, and uncontroversial

All 75 ACVP encapsulation vectors pass (3 parameter sets × 25; `c` and `K` byte-exact).

| B | device-side ns/op | throughput | vs cuPQC tuned (199.7 ns) | vs cuPQC default (243.5 ns) |
|---:|---:|---:|---:|---:|
| 1024 | 195 | 5.14 M/s | 1.03× | 1.25× |
| 4096 | 94 | 10.67 M/s | 2.13× | 2.60× |
| 8192 | **88** | **11.38 M/s** | **2.27×** | **2.77×** |
| 16384 | 90 | 11.08 M/s | 2.21× | 2.70× |

Encapsulation also needs `Â`, and `[u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]` is a single
GEMM reusing the ENC fused kernel with a different epilogue. Net of rebuilding `W_enc` (~0.14 ms,
17 ns/op at B=8192): **about 105 ns/op, still 1.90×**.

### 0.4 keygen: **we lose by 2.7×**, the only operation we lose

All three operations exist, and keygen is pinned byte-exact against kyber-py for all three
parameter sets (`ek` 800/1184/1568 B, `dk` 1632/2400/3168 B).

| B | device-side ns/key | throughput | vs liboqs-1T (7.73 µs) | **vs cuPQC 173.9 ns** |
|---:|---:|---:|---:|---:|
| 1024 | 561 | 1.78 M/s | 10.2× | 0.31× |
| 4096 | 467 | 2.14 M/s | 13.3× | 0.37× |
| 8192 | 469 | 2.13 M/s | 13.5× | **0.37×** |

For exactly the expected reason: in decaps/encaps the multiplication folds into a GEMM over a
matrix fixed **across operations**, so the Toeplitz trick applies; in keygen `s`/`e` are **new for
every key**, there is no reusable matrix, and `ek` must contain `t̂` in the NTT domain — so the
NTT cannot be dodged.

Per-stage cost (GB10, B=4096, details in [docs/benchmarks.md](benchmarks.md) §4) puts `pow_mac`
at 32.7% and the NTT at only 16.7%: **`pow_mac` is memory-bound, not compute-bound.** It reads
37.7 MB of `Â`, whose 0.62 ms corresponds to ~75 MB of traffic, a ~0.28 ms lower bound at GB10's
273 GB/s. The real gap is that we spill `Â` to VRAM between kernels while cuPQC keeps keygen
inside one or two kernels.

**The fusion ceiling, measured before building it** (GB10, B=4096, `Â` = 37.7 MB): write once
0.200 ms (188.7 GB/s), read once 0.132 ms (286.0 GB/s), together **0.332 ms = 81 ns/key**. Fusing
SampleNTT → NTT → `pow_mac` can save exactly that: **466 → best case 385 ns/key, 0.45× of
cuPQC's 174 ns.** So the upper bound of fusion is **1.21×, not 2.7×**. Of the remaining 385 ns,
NTT 78, SHAKE 75, ByteEncode12 45 and CBD 11 are each doing irreducible work; cuPQC reaching
174 ns means every one of its kernels is more efficient, not that its pipeline has a better
shape. **This gap cannot be closed by fusion**, and the work was therefore declined — see
[CHANGELOG.md](../CHANGELOG.md).

### 0.5 A premise that must be stated

**No paper has compared "Toeplitz + tensor cores" against "NTT" on the same card.** We therefore
**cannot** claim our method is faster than the NTT — only that among the three implementations
reporting batched throughput, our absolute number leads. ConvKyber's 12.48× is "its tensor-core
NTT vs butterfly NTT"; TensorCrypto's 2–5× is "tensor-core convolution vs CUDA-core INT32".
Neither is a comparison against us.

### 0.6 Our direct answer to TensorCrypto's criterion

TensorCrypto (IEEE Access 2022 — the Hansung link, except it covers NTRU/LAC, not Kyber) states
why it does not do Kyber:

> "schemes such as KYBER and NewHope already use NTT-based multiplications, the polynomial
> convolution is already efficient, so it would be more advantageous to use NTT... it is also
> difficult to use tensor cores to accelerate the NTT computations, since the size of modulus (q)
> is most likely to exceed 11-bit. Therefore, we restrict the scope of our paper to the schemes
> where NTT is slow or not applicable."

Its reason is "Kyber's NTT is fast enough, convolution offers no advantage". **Our measurements
are an empirical rebuttal** (8.0 M/s on GB10, 3.26 M/s on an Arc iGPU, far above the CPU
baseline). But **this is not a proof**: we have no same-card NTT implementation to compare
against, so the advantage might come from "using tensor cores" rather than "using Toeplitz".

### 0.7 Precision: all three are exact, by different routes

| | approach | source of exactness |
|---|---|---|
| ConvKyber | INT8/INT16 tensor cores + INT32 accumulate | integers, exact by construction; 12-bit coefficients split into two 6-bit halves |
| TensorCrypto | FP16 + FP32 accumulate | bit-width argument: q < 2^11, one operand ternary → product ≤ 11 bit → sum ≤ 2^21 |
| **ours** | **FP16 + FP32 accumulate** | **the other operand is bounded by η (≤3) or the CBD range (≤2), not by q/2** |

**Our key observation, and the substantive difference from TensorCrypto**: in every polynomial
multiplication of Kyber, **one operand is always small**. Full-range × full-range never occurs:

| GEMM | weight operand | variable operand | product bound | K | accumulation | headroom to 2^24 |
|---|---|---|---|---|---|---|
| DEC `W_dec·U` | `T(s)`, measured max **2** | `u` ∈ [−1664,1664] | 3328 | 768 | 2.56e6 | **6.6×** |
| ENC `W_enc·Y` | `T(Â)/T(t)`, max **1664** | `y` ∈ [−2,2] | 3328 | 768 | 2.56e6 | **6.6×** |

For ML-KEM-1024 (K = 1024) the bound is 3.41e6, still 4.9× below 2^24. That is why
TensorCrypto's "q > 11 bit, so skip tensor cores" reasoning does not apply here: it assumes
full-range × full-range, which this scheme never performs.

---

## 1. Our data

Both machines, the timing conventions and the complete measured tables are in
[docs/benchmarks.md](benchmarks.md). In short: an **Intel Arc B390** integrated GPU (96 EU,
Xe3-LPG, 16 MB last-level cache, 0.9 GHz idle to ~2.2 GHz) and an **NVIDIA GB10** (48 SM,
`sm_121`, 2.418 GHz). This document uses **device-side** for the wall clock of `decaps_torch()`
including the 13 Python-side enqueues but excluding H2D/D2H, and **end-to-end** when packing, H2D
and D2H are included. Values are min-of-N with N = 10–25, byte-checked against the ACVP vectors;
**absolute numbers jitter by ±20%** because the clocks drift and cannot be locked, so only
same-process paired ratios are trusted here.

The figures the rest of this comparison leans on: asymptotic device-side decaps **3.2 Mops/s**
(Arc) and **8.0 Mops/s** (GB10), i.e. 0.41 and 0.227 µs/op end-to-end at B=8192; single-op
latency at B=1 of **966 µs** and **282 µs**; GEMM at **~24%** and **~28%** of the same-shape
`tl.dot` peak; and ML-KEM-512 **3.0×** / **5.8×** slower than 768.

CPU baseline on the same machines (liboqs 0.16.0, same ACVP ciphertexts): **9.52 µs/op**
single-threaded decaps and **1.375 µs/op** across 16 concurrent processes; encaps is 7.82 and
1.109 µs/op. liboqs's `speed_kem` is single-threaded (`OMP_NUM_THREADS` has no effect), so the
multi-core figure aggregates 16 processes. Cross-over against liboqs-16T: decaps needs B >= 1024
on Intel and B >= 256 on GB10; encaps crosses at about B ≈ 700. **At small batches we lose to a
16-thread CPU**, which is why the batch-size warning in §6 item 2 exists.

One-time cost — routinely omitted from comparisons, and this design's real architectural price:

| | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|---|---:|---:|---:|
| `keyprep` pre-processing | 6.8 ms | **13.5 ms** | 19.8 ms |
| resident VRAM (`W_dec`+`W_enc`, fp16) | 4.2 MB | **7.9 MB** | 12.6 MB |
| `dk` size in a standard implementation | 1632 B | 2400 B | 3168 B |

A standard implementation stores 2400 bytes and recomputes `Â` from ρ; we expand `Â`/`t`/`s` into
resident Toeplitz matrices. That works for one long-lived key with a flood of ciphertexts, and
only for that — see §5.3.

---

## 2. The comparison objects

| | cuPQC | HI-Kyber | ConvKyber | TensorCrypto |
|---|---|---|---|---|
| venue | NVIDIA product | ePrint 2023/1194, IEEE TPDS 2024 | TCHES 2024(2):25–63 | IEEE Access 10:20616–20632, 2022 |
| source | closed SDK | full text unavailable | CC-BY PDF | open (public domain) |
| algorithm | ML-KEM (FIPS 203) | Kyber v3 | Kyber v3 | NTRU / LAC / FrodoKEM |
| NTT? | yes | yes | yes, rearranged into convolution-*like* matmuls | no (convolution) |
| tensor cores | yes (undisclosed) | no (CUDA core) | yes, int8/16 + int32 | yes, fp16 + fp32 |
| hardware | H100 / GB10 | unavailable | Jetson AGX Orin, RTX 3080 | RTX 2060 / 3080 |

### 2.1 NVIDIA cuPQC (cuPQC-PK)

Commercial closed-source SDK (<https://developer.nvidia.com/cupqc>, fetched 2026-11-25), with
cuPQC-BigInt / NTT / Hash / **PK** components; ML-KEM-512/768/1024 and ML-DSA-44/65/87. NTT
based, with link-time optimisation selecting kernels per configuration. Vendor figures on an
**H100** against a single-core EPYC 7313P: keygen **13.5 M/s** (175×), encaps **9.0 M/s** (120×),
decaps **8.5 M/s** (135×) = **0.1176 µs/op**. Inverting that implies a CPU baseline of ~63 k
decap/s (15.9 µs/op), consistent with the 10.26 µs/op we measured — credible. **This is the only
object whose throughput we can compare directly**, because it also reports large-batch throughput.

Measured on our GB10 (0.6.0, built 2026-08-27, tar sha256 `f797d65f…8fe992`,
`nvcc -dlto -arch=sm_121 -O3`, clean first try; keygen→encaps→decaps round trip 32/32, sizes
matching FIPS 203: pk 1184 / sk 2400 / ct 1088 / ss 32 B):

| configuration | batch | keygen | encaps | **decaps** |
|---|---:|---:|---:|---:|
| **BD=32** (swept optimum; default 128) | 262,144 | 5.75 M/s | 5.01 M/s | **4.96 M/s** (202 ns) |
| BD=128 (SDK default) | 1,048,576 | 5.12 M/s | 4.03 M/s | 4.27 M/s (234 ns) |

Other measured facts: **multi-stream gives zero gain** (4.955 / 4.947 / 4.937 / 4.929 M/s at
1/2/4/8 streams), the same class of finding as our Intel H2D result; batch=1 is 57.4 µs kernel /
61.4 µs end-to-end, ~53 µs of it fixed launch cost; decaps achieves only ~17 GB/s of real
bandwidth, so it is **compute-bound**; and **the SDK contains no KAT or ACVP program at all**, so
its FIPS 203 compliance cannot be vouched for by its own vectors. The licence is evaluation-only
("NVIDIA Software and Model Evaluation License").

### 2.2 HI-Kyber (NJUPT, 2023)

**A GPU paper, not FPGA.** IACR ePrint 2023/1194, journal version IEEE TPDS 2024
([DOI](https://doi.org/10.1109/TPDS.2024.3379734)). Approach: **NTT on CUDA cores**, explicitly
treating tensor-core implementations as the competition, with SLM / SDFS-NTT / EDFS-NTT rewrites
for +7.5% / +28.5% / +41.6%. **It does not replace the NTT.** Available numbers, all from the
abstract page: key exchange **1,664 kops/s**, 3.52× vs a same-instruction-set GPU implementation
and 1.78× vs an unnamed "SOTA AI-accelerated tensor core" implementation.

**Unavailable**: GPU model, per-operation numbers, batch size, timing method, multi-stream — TPDS
is closed access and the ePrint PDF is behind a Cloudflare 403. **Not comparable with us**, and
its "key exchange kops/s" comes with no formula, so it cannot be reconciled with ConvKyber's
convention either.

### 2.3 ConvKyber (TCHES 2024)

Jetson AGX Orin and **RTX 3080** (68 SM, 1.8 GHz), on **Kyber round-3**, not FIPS 203.
**Approach clarified: it does not replace the NTT with convolution.** In the paper's words, "we
name this framework convKyber because we convert the core NTT operations into operations
**resembling** convolutions" — the NTT stays; butterflies are rearranged into iterative small
block matmuls. Toeplitz convolution is **prior work it cites**. Precision is INT8/INT16 tensor
cores with INT32 accumulate, 12-bit coefficients split into 6-bit halves, optional Karatsuba.

Kyber-1024, kops/s: KeyGen 2,074 / Enc 1,587 / Dec **8,458** / Decaps **1,336†** / KX 1,336, at
17,408 concurrent ops (Dec 8,704). † **Decaps is derived, not measured** — the table note gives
`ab/(a+b)` with a=Enc, b=Dec. Timing is an average of 500 iterations with no statement about
cudaEvent or host transfer, and **Kyber-768 appears only in a bar chart**.

### 2.4 Hansung OAK_1445 → actually **TensorCrypto (IEEE Access 2022)**

**The link supplied was broken**: `.../2024.oak/1883/2/OAK_1445.pdf` returns HTTP 404 "Invalid
Identifier". The real file under the same handle is
`.../1883/2/AR_E1A1A4_TensorCrypto- High Throughput Acceleration of Lattice-Based Cryptography Using Tensor Core on GPU.pdf`
(<https://dspace.hansung.ac.kr/handle/2024.oak/1883>).

RTX 2060 / RTX 3080 at 1.71 GHz, on **NTRU (hps2048509/677), LAC and FrodoKEM — no Kyber at
all**. Its approach is exactly ours: **Toeplitz/cyclic matmul on tensor cores instead of an NTT**,
fp16 in and fp32 accumulate. **This is the closest precedent to our work**, open source at
<https://github.com/benlwk/Tensorcrypto>. Numbers (NTRU, not Kyber): 793,651 encaps/s and
505,051 decaps/s on RTX 2060 at 512 blocks, convolution itself **0.33 µs** (N=512), up to 3.41×
(2060) / 5.77× (3080) over CUDA-core INT32. Its reason for skipping Kyber is quoted in §0.6.

---

## 3. Cross-implementation tables

### 3.1 Everything aligned to complete decaps throughput

| implementation | hardware | units | algorithm | NTT? | tensor cores | complete decaps | per SM | batch |
|---|---|---:|---|---|---|---:|---:|---:|
| **cuPQC-PK (I measured)** | **GB10** | **48 SM** | **ML-KEM** | yes | yes | **4.96 M/s** | 103 k | 262,144 |
| **ours** | **GB10** | **48 SM** | **ML-KEM** | **no** | fp16+fp32 | **8.0 M/s** | **167 k** | 8,192 |
| **→ same-machine ratio** | | | | | | **ours 1.61×** | **1.61×** | |
| cuPQC-PK (vendor) | H100 | 132 SM | ML-KEM | yes | yes | 8.5 M/s | 64 k | not stated |
| **ours** (1024) | GB10 | 48 SM | ML-KEM | no | yes | **5.78 M/s** | **120 k** | 4,096 |
| ConvKyber (1024) | RTX 3080 | 68 SM | Kyber v3 | yes (conv-like) | int8/16+int32 | **1.34 M/s†** | 19.6 k | 17,408 |
| **ours** (iGPU) | Arc B390 | 96 EU | ML-KEM | no | yes | **3.26 M/s** | — | 4,096 |
| HI-Kyber | unavailable | — | Kyber v3 | yes | no | **unavailable** | — | unavailable |
| TensorCrypto | RTX 2060/3080 | — | **NTRU/LAC** | **no** | fp16+fp32 | **not applicable** | — | 512 |

† derived, not measured.

### 3.2 Against CPU baselines (self-reported; **different baselines, not cross-comparable**)

| implementation | baseline | decaps speedup |
|---|---|---:|
| cuPQC | single-core EPYC 7313P | 135× |
| **ours** | liboqs 0.16.0, 16 threads (same machine) | **3.6× (Arc) / 6.4× (GB10)** |
| **ours** | liboqs 0.16.0, 1 thread (same machine) | **25× (Arc) / 45× (GB10)** |
| ConvKyber | AVX2 single core | 136× slower per op; a win only at 17,408 concurrency |
| TensorCrypto | AVX2 | 2×+ (NTRU) |

Our baseline is **liboqs with 16 threads** (1.45 µs/op), about 11× stronger than the single-core
baseline cuPQC uses (15.9 µs/op). So our 3.6–6.4× and cuPQC's 135× **cannot be compared at all.**

### 3.3 Single-operation latency

Same machine, both at batch=1, end-to-end including launch: **cuPQC 61.4 µs** (of which ~53 µs is
fixed launch cost) against **ours 282 µs** on GB10 and **966 µs** on the Arc iGPU —
**we are 4.6× slower**.

**Correcting an earlier error**: this document previously said "three orders of magnitude
slower". That compared ConvKyber's **amortized** figure (total time for a 17,408-op batch divided
by 17,408 = 0.06–0.33 µs) against single-op latency, which is invalid. Same machine, same
convention, the gap is **4.6×**. **For one-op-at-a-time use we are still much worse**, just not by
three orders of magnitude.

### 3.4 cuPQC's own tuning behaviour

BlockDim sweep (decaps, Mops/s): BD 32 **4.958** / 64 3.882 / 128 4.275 / 256 1.654 — **the
default BD=128 must be tuned to 32 to gain 15%**, the same story as "sweep your own tiles" on
Intel. Batch convergence saturates after 2.6e5.

---

## 4. Our advantages

- **No NTT, so the whole path is dense.** The NTT's bottleneck is not arithmetic volume but
  access pattern: butterfly strides force a permutation at every stage, and each pointwise
  product must be reduced into `[0,q)` elementwise. Folding the multiplication into fixed 256×256
  blocked Toeplitz matrices gives a pure dense GEMM whose utilisation is quantifiable: **24% /
  28%** of the same-shape pure-`tl.dot` peak on Intel and GB10.
- **fp16 exactness is provable, not tuned.** Any accumulation order is element-wise exact
  independent of the GPU's reduction tree and of the backend, and `test_dot_exact.py` checks the
  boundary values on both machines rather than trusting the argument. Outside flash-attention
  almost all fp16 GEMM work tolerates error; cryptography cannot lose a bit, which is why many
  GPU Kyber implementations switch to int16/int32. This route is free.
- **The only implementation measured on two vendors.** Same code, no inline assembly and no
  vendor intrinsics: **264 tests green** on Intel Arc B390 / triton-xpu 3.9 and on NVIDIA GB10 /
  triton 3.8, without changing one kernel line.
- **End-to-end reporting, with the host side actually optimised.** Four accepted input forms
  (host cost 0.89 → 0.57 ms at B=4096 on Intel), adaptive pinned staging (probed on 4 MiB, used
  only if >15% faster: worth 0.50 → 0.28 ms on Intel, not on GB10), O(1) instead of O(B) length
  validation (0.13 ms). It also proved **H2D cannot overlap compute on this Intel driver**
  (dual-stream zero-dependency 1.328 ms vs serial 1.307 ms), ruling out streaming double-buffering.
- **Deep fusion.** 13 launches for the whole decapsulation, both GEMM epilogues inside the GEMM
  kernel: ENC never materialises the 16.8 MB `acc2`, DEC never materialises the 4.2 MB `acc`.
  Paired same-process device-side: 2.75 → 1.62 → 1.22 ms (Intel, B=4096), and sponge input
  preparation 192 → 28 µs.
- **Verifiable correctness with silent fallback guarded.** Byte-exact ACVP on both machines plus
  a byte-equality test per fused kernel against the unfused path — necessary, because a fused
  kernel returns `False`/`None` and writes nothing when its constraints are unmet. We hit exactly
  that during development (`Y` was a transposed view, so the fused GEMM never ran). The
  measurement methodology is public too, including the hypotheses it falsified.

## 5. Our disadvantages

**5.1 keygen loses to cuPQC by 2.7×, and fusion cannot close it.** 469 ns/key against 173.9 ns.
The NTT is unavoidable (§0.4) and the measured fusion ceiling is 81 ns/key, leaving us at 0.45×.
Closing the gap means matching hand-written CUDA kernel by kernel.

**5.2 Single-operation latency is 4.6× worse** — 282 µs against 61.4 µs at B=1 on GB10. Thirteen
launches at 35–40 µs fixed cost each, with nothing to amortise; cuPQC itself eats 53 µs of launch
overhead. This is the inherent cost of a throughput-oriented design.

**5.3 7.9 MB and 13.5 ms per private key.** `dk` stays 2400 B, but the resident Toeplitz matrices
cost 7.86 MB and the CPU pre-processing 13.45 ms. After the equal-convention experiment this has
to be stated precisely: **on-GPU rebuild is 0.158 ms (19.3 ns/op at B=8192)**, so the real
constraint is the resident VRAM, not the time. It still limits the design to one long-lived key
with a flood of ciphertexts.

**5.4 ML-KEM-512 takes the slow path.** `η1 = 3` while the fused CBD kernel only packs `η = 2`, so
512 falls back to the generic path: **0.932 µs/op on Intel and 0.760 on GB10**, against 0.307 /
0.132 for 768. Correct (ACVP passes), unoptimised, pinned by a test. Per-kernel evidence confirms
the cause: ~2.2 ms of the 3.118 ms Intel total is PyTorch native elementwise/reduction operators
from the generic CBD fallback; the two fused kernels together are 0.25 ms.

**5.5 Depends on fp16 tensor cores.** Without them it degrades to `mm(fp32, fp32)`, measured 2–4×
slower (DEC 0.20 → 0.44 ms, ENC 0.31 → 1.13 ms). Not "any GPU" — "any GPU with fp16 tensor cores".

**5.6 No formal constant-time guarantee.** Timing does not depend on ciphertext content (GEMM
plus `torch.where`, both data-independent), but **no dudect / ctgrind / Valgrind leak test and no
cache-side-channel analysis has been run**. Whether L2 contention among several resident 7.9 MB
weight sets leaks key correlation is unstudied, and batch size affects timing.

**5.7 The host side is still 29% on Intel** — of 1.99 ms end-to-end at B=4096, 0.57 ms is
`b"".join` 0.25 + H2D 0.28 + D2H 0.03, which is the floor for a `list[bytes]` API.

**5.8 Never run on a datacentre GPU, never measured on AMD.** The comparison objects use A100 /
RTX 3090 / H100-class hardware; we have an **integrated GPU** (96 EU) and a 48-SM module. Every
cross-implementation absolute comparison carries that unfairness.

**5.9 No same-card, same-convention NTT comparison.** We can prove 8.0 M decaps/s on GB10 and
3.26 M/s on an Arc iGPU, beating same-machine liboqs with 16 threads by 3.6–6.4×. We **cannot**
prove we are faster than an NTT implementation that also uses tensor cores: ConvKyber's derived
1.34 M/s is on a smaller card, is derived, and runs Kyber v3 — three variables at once. The only
clean way is ConvKyber's integer tensor-core NTT against our fp16 Toeplitz GEMM, same card, same
ciphertexts, same timing. Nobody has done that, including us.

**5.10 GEMM reaches only 24–28% of peak.** The non-GEMM stages dominate: Keccak 16%, `_pre` 17%,
CBD→Y/E 16%, padcat 8%. VTune reports `XVE Array Stalled/Idle 75.1%` and occupancy 28.5%.

---

## 6. Comparability warnings (read before any table above)

1. **Different hardware.** A 96-EU iGPU and a 48-SM GB10 against A100/H100/RTX 3090. Only "speedup
   over a same-hardware CPU baseline" and "share of tensor-core peak" can travel.
2. **Completely different batch sizes.** They report batch=1 latency; we report B=4096–8192
   throughput. The same code differs by **3000×** between them on Intel (966 → 0.31 µs/op), so
   comparing our 0.31 µs/op against their per-op 10 µs is **misleading**.
3. **Different timing conventions.** They bracket the kernel with `cudaEvent`; we separate
   device-side from end-to-end. Ours is closer to theirs but still counts 13 Python enqueues.
4. **Different algorithm coverage.** All three operations exist here; the papers vary, some
   reporting only decaps or only key exchange.
5. **Kyber v3 vs ML-KEM.** Core maths is identical, but FIPS 203 decapsulation adds an implicit-
   rejection hash (`J(z‖c)`) and a constant-time selection. We implement ML-KEM.
6. **Our absolute values jitter by ±20%**; clocks drift 0.9 ↔ 2.2 GHz and cannot be locked.
7. **Which baseline?** Papers often use AVX2/AVX-512 single-core; we use liboqs 0.16.0 at 1 thread
   (10.26 µs) and 16 threads (1.45 µs). Different baselines, different conclusions.
8. **They have an NTT, we do not** — not "who optimised better" but **two different ways of
   organising the algorithm**. Theirs works directly for keygen/encaps (ρ changes every time).
