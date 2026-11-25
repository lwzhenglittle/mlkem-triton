# Systematic comparison with existing GPU Kyber / ML-KEM implementations

> "Ours" = `mlkem-triton` (this repository). The comparison objects are NVIDIA cuPQC,
> HI-Kyber (Nanjing University of Posts and Telecommunications, 2023), ConvKyber
> (TCHES 2024), and the OAK paper from `dspace.hansung.ac.kr`.
>
> **Every number for our own implementation is measured** (machines, commands and conventions
> are spelled out in §1). Every number for the other implementations is sourced. Where a
> number could not be found, this document says "unavailable" rather than guessing.

---

## 0. Bottom line

### 0.1 Same machine, same operation: 1.61× faster than cuPQC

cuPQC 0.6.0 was really compiled and run on a **GB10 / sm_121**. sm_121 is not in NVIDIA's
official support list, but `-dlto` bitcode generates the SASS at link time, and the result is
both correct and fast. That gives a clean comparison: **same card, same operation, both
reporting batched throughput.**

| ML-KEM-768 decaps | hardware | SMs | throughput | per op | per SM | batch | timing |
|---|---|---:|---:|---:|---:|---:|---|
| **cuPQC-PK 0.6.0** (BD=32, tuned by me) | **GB10** | 48 | **4.96 M/s** | 202 ns | 103 k/SM | 262,144 | pure kernel (cudaEvent) |
| **ours** | **GB10** | 48 | **8.0 M/s** | 125 ns | **167 k/SM** | 8,192 | device-side (13 launches included) |
| | | | **1.61×** | | **1.61×** | | |
| cuPQC-PK 0.6.0 (**SDK default BD=128**) | GB10 | 48 | 4.27 M/s | 234 ns | 89 k/SM | 1,048,576 | pure kernel |
| **→ out-of-the-box ratio** | | | | | | | **1.86×** |
| cuPQC-PK, vendor figure | H100 | 132 | 8.5 M/s | 118 ns | 64 k/SM | not stated | not stated |

**This is the most important row in the document: on the same GB10, on the same complete
decaps, our throughput is 1.61× cuPQC's.** And our convention is the less favourable one — it
includes 13 Python-side Triton launches (their cost is hidden by device work at B=8192, but it
is still counted).

Cross-generation: cuPQC reaches only 58% of its own H100 figure on GB10; we reach 8.0 M/s on
GB10, which is 94% of cuPQC's 8.5 M/s on an H100.

A three-round alternating re-check (the two binaries cannot be interleaved in one process, so
each ran in its own process, three rounds, sampling `clocks.sm` before and after each) gave
**1.86× / 1.86× / 1.85×** with both processes clocked at 2398–2411 MHz — drift under 1%, far
below the effect size. Note that the `bench768` binary in the cuPQC build tree is compiled
with **the SDK default BlockDim=128**; BD=32 was compiled separately for this study. Hence
two honest ratios: **1.86× out of the box**, **1.60–1.61× with both sides tuned**.

### 0.2 Decomposing the 1.61×: 86% from the kernel, 14% from pre-expansion

cuPQC's API is stateless, so **every ciphertext** re-runs k² SampleNTT calls (SHAKE-128,
840 B each) to obtain `Â`. We expand `Â` into a resident Toeplitz matrix once and pay no
SHAKE per operation. That makes the attribution ambiguous, so `kernels/rebuild.py` was written
to rebuild `W_dec`/`W_enc` from `dk` entirely on the GPU, closing the gap.

Putting key rebuild back into the timed region: rebuild 0.158 ms @ GB10, i.e. **19.3 ns/op at
B=8192**, taking us from 125 to 144 ns/op — **still 1.39× cuPQC** (202 ns). So roughly 86% of
the advantage is the kernel and 14% is not re-expanding `Â`.

### 0.3 encaps: 2.27× faster, and this one is uncontroversial

`encaps.py` passes **all 75 ACVP encapsulation vectors** (3 parameter sets × 25; ciphertext
`c` and shared secret `K` both byte-exact).

GB10, ML-KEM-768, device-side including Python enqueue:

| B | device-side ns/op | throughput | vs cuPQC tuned (199.7 ns) | vs cuPQC default (243.5 ns) |
|---:|---:|---:|---:|---:|
| 1024 | 195 | 5.14 M/s | 1.03× | 1.25× |
| 4096 | 94 | 10.67 M/s | 2.13× | 2.60× |
| 8192 | **88** | **11.38 M/s** | **2.27×** | **2.77×** |
| 16384 | 90 | 11.08 M/s | 2.21× | 2.70× |

The advantage over cuPQC is larger than for decaps, and the reason is clear: encapsulation
also needs `Â`, and we turn `[u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]` into a single
GEMM, reusing the ENC fused kernel and only swapping the epilogue (compare-and-count becomes
compress-and-bitpack). `r`, `e1`, `e2` are all small, so the same precision bound holds.

Net of pre-expansion: rebuilding `W_enc` costs the same order as the `dk` case (about
0.14 ms, it is 2k rows shorter), i.e. 17 ns/op at B=8192 → **about 105 ns/op, still 1.90×**.

### 0.4 keygen: **we lose by 2.7×**, the only operation we lose

All three operations now exist (`keygen.py`), and keygen is pinned by **byte-exact comparison
against kyber-py**: for all three parameter sets `ek` (800/1184/1568 B) and `dk`
(1632/2400/3168 B) match exactly.

GB10:

| B | device-side ns/key | throughput | vs liboqs-1T (7.73 µs) | **vs cuPQC 173.9 ns** |
|---:|---:|---:|---:|---:|
| 1024 | 561 | 1.78 M/s | 10.2× | 0.31× |
| 4096 | 467 | 2.14 M/s | 13.3× | 0.37× |
| 8192 | 469 | 2.13 M/s | 13.5× | **0.37×** |

cuPQC's keygen is 2.7× faster than ours, for exactly the expected reason:

- in decaps / encaps the multiplication folds into a GEMM over a matrix fixed **across
  operations**, so the Toeplitz trick applies;
- in keygen `s`/`e` are **new for every key** — there is no reusable matrix, so the NTT has
  to be done properly;
- and `ek` must contain `t̂` in the NTT domain, so the NTT cannot be dodged.

Per-stage cost on GB10 at B=4096, timed in isolation:

| stage | ns/key | share |
|---|---:|---:|
| `pow_mac` (pointwise multiply + extension; reads 37.7 MB of `Â`) | 152 | **32.7%** |
| NTT (2k polynomials) | 78 | 16.7% |
| SampleNTT | 76 | 16.4% |
| SHAKE-128 ×9 | 75 | 16.1% |
| ByteEncode12 ×2 | 45 | 9.8% |
| CBD ×2 | 11 | 2.3% |
| other (message construction / pad / H(ek)) | 25 | 5.5% |
| **total** | **466** | |

For reference, the same breakdown on Intel at B=4096: NTT 372 ns, SampleNTT 448 ns,
SHAKE-128 ×9 269 ns, pointwise multiply 223 ns, ByteEncode12 ×2 141 ns, CBD ×2 44 ns.

**This table refutes the assumption that the NTT is keygen's bottleneck.** It is only 17%.
The largest single piece is `pow_mac`, whose 0.62 ms corresponds to about 75 MB of traffic
(`Â` 37.7 + `Ŝ` 12.6 + `Ê` 12.6 + 12.6 written) — a lower bound of ~0.28 ms at GB10's
273 GB/s. **It is memory-bound, not compute-bound.**

So the real gap is that **we spill `Â` to VRAM between kernels** (37.7 MB at B=4096), while
cuPQC keeps the whole keygen inside one or two kernels. Catching up would need SampleNTT →
NTT → `pow_mac` fused into one pipeline rather than more per-kernel tuning.

#### The upper bound on that fusion (measured, GB10 / B=4096)

Before doing it, the payoff was measured. `Â` is 37.7 MB at B=4096, k=3:

| | |
|---|---:|
| write once | 0.200 ms (188.7 GB/s) |
| read once | 0.132 ms (286.0 GB/s) |
| **write + read** | **0.332 ms = 81 ns/key** |

`pow_mac` and `SampleNTT` each pay one of those today. Fusing everything into one kernel can
save exactly those 81 ns: **466 → best case 385 ns/key, i.e. 0.45× of cuPQC's 174 ns.**

In other words **the upper bound of fusion is 1.21×, not 2.7×.** Of the remaining 385 ns,
NTT 78, SHAKE 75, the rest of SampleNTT, ByteEncode12 45 and CBD 11 are each doing their own
irreducible work. cuPQC reaching 174 ns means **every one of its kernels is more efficient**
(hand-written CUDA, years of tuning), not that its pipeline has a better shape.

Conclusion: **this gap cannot be closed by fusion.** Getting below 385 ns under the
pure-Triton constraint means approaching hand-written CUDA efficiency kernel by kernel — a
different order of engineering effort. This was measured and then declined; see `CHANGELOG.md`.

#### The NTT kernel took two iterations to become usable

| version | approach | time at P=24576 |
|---|---|---:|
| v1 | both branches computed separately (2 mul + 4 mod) + `tl.gather` + BR=16 | **15.2 ms** |
| v2 | `where` normalises a/b into two half-ranges (**1 mul + 2 mod**) + BR=2 | **1.5 ms (10×)** |

`tl.gather` alone is not slow (7 of them cost 1.2 ms) and the arithmetic alone is not slow
(0.9 ms) — but **the two together in one kernel with a large tile (BR >= 16) explode**
(16.7 ms). Dropping to BR=2 fixes it. That is a pathological corner of Triton at this shape,
worth recording.

### 0.5 A premise that must be stated

**No paper has ever compared "Toeplitz + tensor cores" against "NTT" on the same card.** So
we **cannot** claim our method is faster than the NTT — only that among the three
implementations that report batched throughput, our absolute number leads. ConvKyber's 12.48×
is "its tensor-core NTT vs butterfly NTT", and TensorCrypto's 2–5× is "tensor-core convolution
vs CUDA-core INT32". Neither is a comparison against us.

### 0.6 Our direct answer to TensorCrypto's criterion

TensorCrypto (IEEE Access 2022 — the Hansung link, except it covers NTRU/LAC, not Kyber)
states explicitly why it does not do Kyber:

> "schemes such as KYBER and NewHope already use NTT-based multiplications, the polynomial
> convolution is already efficient, so it would be more advantageous to use NTT...
> it is also difficult to use tensor cores to accelerate the NTT computations, since the
> size of modulus (q) is most likely to exceed 11-bit. Therefore, we restrict the scope of
> our paper to the schemes where NTT is slow or not applicable."

Its stated reason is "Kyber's NTT is already fast enough, convolution offers no advantage".
**Our measurements are an empirical rebuttal** (8.0 M/s on GB10, 3.3 M/s on an Arc iGPU,
far above the CPU baseline).

But be careful: **this is not a proof.** We have no same-card, same-convention NTT
implementation to compare against, so our advantage might come from "using tensor cores"
rather than from "using Toeplitz". Answering the question properly requires running
ConvKyber's integer tensor-core NTT and our fp16 Toeplitz GEMM on the same card over the same
ciphertexts. **Nobody has done that, including us.**

### 0.7 Precision: all three approaches are exact, by different routes

| | approach | source of exactness |
|---|---|---|
| ConvKyber | INT8/INT16 tensor cores + INT32 accumulate | integers, exact by construction; 12-bit coefficients split into two 6-bit halves, one Montgomery reduction for many accumulations |
| TensorCrypto | FP16 inputs + FP32 accumulate | bit-width argument: q < 2^11, one operand ternary → product ≤ 11 bit → accumulation ≤ N × 2^11 ≤ 2^21, fits FP32's 24-bit mantissa |
| **ours** | **FP16 inputs + FP32 accumulate** | **the other operand is bounded not by q/2 but by η (≤3) or the CBD sampling range (≤2)** |

**Our key observation, and the substantive difference from TensorCrypto**: in every
polynomial multiplication of Kyber, **one operand is always small**. Two full-range
coefficients are never multiplied together:

| GEMM | weight side | variable side | single product bound | K | accumulation bound | headroom to 2^24 |
|---|---|---|---|---|---|---|
| DEC `W_dec·U` | `T(s)`, measured max **2** | `u` ∈ [−1664,1664] | 3328 | 768 | 2.56e6 | **6.6×** |
| ENC `W_enc·Y` | `T(Â)/T(t)`, measured max **1664** | `y` ∈ [−2,2] | 3328 | 768 | 2.56e6 | **6.6×** |

For ML-KEM-1024 (K = 1024) the bound is 3.41e6, still 4.9× below 2^24. This is what makes
TensorCrypto's "q > 11 bit, so skip tensor cores" reasoning inapplicable: that argument
assumes full-range × full-range, which never occurs here.

---

## 1. Our data

Machines, timing conventions and the complete measured tables are in
[docs/benchmarks.md](benchmarks.md). This document uses **device-side** and **end-to-end**
exactly as defined there: device-side is the wall clock of `decaps_torch()` including the 13
Python-side enqueues but excluding H2D/D2H; end-to-end includes numpy packing, H2D and D2H.
Values are min-of-N with N = 10-25, validated byte-for-byte against the NIST ACVP vectors, and
**absolute numbers jitter by +-20% because the GPU clocks drift and cannot be locked** - only
same-process paired ratios are trusted anywhere in this document.

| | Intel Arc B390 | NVIDIA GB10 |
|---|---|---|
| form factor | integrated GPU, B390 / Xe3-LPG | DGX Spark single chip, `sm_121` |
| compute units | 96 EU / 12 subslice | 48 SM |
| clock | 0.9 GHz idle / ~2.2 GHz loaded | 2.418 GHz |
| software | torch 2.15.0+xpu / triton-xpu 3.9.0 | torch 2.14.0+cu130 / triton 3.8.0 |

The figures the rest of this comparison leans on:

| | Intel Arc B390 | NVIDIA GB10 |
|---|---:|---:|
| asymptotic decaps, device-side | 3.2 Mops/s | 8.0 Mops/s |
| decaps, end-to-end, B=8192 | 0.41 µs/op | 0.227 µs/op |
| single-op latency, B=1 | 966 µs | 282 µs |
| GEMM share of same-shape `tl.dot` peak | ~24% | ~28% |
| ML-KEM-512 slower than 768, device-side | 3.0× | 5.8× |

CPU baseline on the same machines (liboqs 0.16.0, same ACVP ciphertexts): **9.52 µs/op**
single-threaded decaps and **1.375 µs/op** at 16 concurrent processes, against **7.82** and
**1.109 µs/op** for encaps. liboqs's `speed_kem` is single-threaded (`OMP_NUM_THREADS` has no
effect), so the multi-core figure is the aggregate of 16 processes.

Cross-over against liboqs-16T: decaps needs B >= 1024 on Intel and B >= 256 on GB10; encaps
crosses at about B ≈ 700 on Intel. **At small batches we lose to a 16-thread CPU**, which has to
be said plainly and is the reason for the batch-size warning in §6.2.

One-time cost - routinely omitted from comparisons, and this design's real architectural price:

| | ML-KEM-512 | ML-KEM-768 | ML-KEM-1024 |
|---|---:|---:|---:|
| `keyprep` pre-processing | 6.8 ms | **13.5 ms** | 19.8 ms |
| resident VRAM (`W_dec`+`W_enc`, fp16) | 4.2 MB | **7.9 MB** | 12.6 MB |
| `dk` size in a standard implementation | 1632 B | 2400 B | 3168 B |

A standard implementation recomputes `Â` from ρ on demand and stores 2400 bytes; we expand
`Â`/`t`/`s` into resident Toeplitz matrices. That works for one long-lived private key with a
flood of ciphertexts, and only for that - see §5.3.

---

## 2. The comparison objects

### 2.1 NVIDIA cuPQC (cuPQC-PK)

- Source: <https://developer.nvidia.com/cupqc> (NVIDIA product page, fetched 2026-11-25)
- Nature: a **closed-source commercial SDK** ("Download cuPQC SDK", application required),
  not a public repository
- Components: cuPQC-BigInt / cuPQC-NTT / cuPQC-Hash / **cuPQC-PK**
- cuPQC-PK supports ML-KEM-512/768/1024 and ML-DSA-44/65/87
- Approach: **NTT** (cuPQC-NTT is a separate component) plus a hash library; link-time
  optimisation selects kernels per configuration
- Vendor claim: "optimized for both low-latency single operations and high-throughput batch
  processing"

**Vendor figures** (ML-KEM-768, **NVIDIA H100**, relative to a single-core AMD EPYC 7313P):

| operation | throughput | vs single-core CPU |
|---|---:|---:|
| keygen | **13.5 M/s** | 175× |
| encaps | **9.0 M/s** | 120× |
| **decaps** | **8.5 M/s** | 135× |

Converting: decaps = **0.1176 µs/op**. Inverting their ratio implies a CPU baseline of about
63 k decap/s (15.9 µs/op), the same order as the 10.26 µs/op we measured for single-threaded
liboqs — credible.

> This is the only object we can compare throughput **directly** against, because it also
> reports large-batch throughput.

**Measured successfully on our GB10** (details in §3.4): cuPQC 0.6.0 (built 2026-08-27, tar
sha256 `f797d65f…8fe992`), `nvcc -dlto -arch=sm_121 -O3`, compiled clean on the first try.
keygen → encaps → decaps round trip agreed 32/32, and sizes match FIPS 203 (pk 1184 / sk 2400
/ ct 1088 / ss 32 B). **The SDK contains no KAT or ACVP program at all** (grep across the
whole package found nothing), so its FIPS 203 compliance cannot be vouched for by its own
vectors.

GB10 measurements (2.4 GHz boost, back-to-back cudaEvent, pure kernel):

| configuration | batch | keygen | encaps | **decaps** |
|---|---:|---:|---:|---:|
| **BD=32** (swept optimum; default is 128) | 262,144 | 5.75 M/s | 5.01 M/s | **4.96 M/s** (202 ns) |
| BD=128 (SDK default) | 1,048,576 | 5.12 M/s | 4.03 M/s | 4.27 M/s (234 ns) |

Other measured facts:

- **Multi-stream gives zero gain** (1/2/4/8 streams: 4.955 / 4.947 / 4.937 / 4.929 M/s). This
  is the same class of phenomenon as our Intel finding that H2D cannot overlap compute: the
  platform does not do concurrency.
- **batch=1 latency is 57.4 µs (kernel) / 61.4 µs (end-to-end), of which about 53 µs is fixed
  launch cost** — launch overhead on this card is enormous, so "one op at a time on the GPU"
  makes no sense for anybody here.
- decaps achieves only ~17 GB/s of actual bandwidth (3488 B/op × 4.96 M/s), far below
  LPDDR5X — **compute-bound, not bandwidth-bound**.
- The licence is the **"NVIDIA Software and Model Evaluation License"** (evaluation,
  non-commercial), closed-source binaries. Commercial use needs a separate agreement.
- The officially supported architecture list is 70/75/80/86/87/89/90, **not including
  sm_121**, but it runs.

### 2.2 HI-Kyber (Nanjing University of Posts and Telecommunications, 2023)

- **It is a GPU paper, not FPGA.**
- IACR ePrint **2023/1194** (2023-08); journal version **IEEE TPDS 2024**, DOI
  [10.1109/TPDS.2024.3379734](https://doi.org/10.1109/TPDS.2024.3379734)
- Authors: Xinyi Ji, Jiankuo Dong, Pinchang Zhang, Deng Tonggui, Fu Xiao (**NJUPT**);
  Hua Jiafeng (Xidian)
- **Approach: NTT, CUDA cores** (it explicitly treats tensor-core implementations as the
  competition). It rewrites the NTT three ways — SLM / SDFS-NTT / EDFS-NTT — for +7.5% /
  +28.5% / +41.6% respectively.
- **It does not replace the NTT with convolution/Toeplitz.**

**Numbers available (all from the abstract page; the full text is closed):**

| metric | value |
|---|---:|
| key exchange throughput | **1,664 kops/s** |
| vs same-instruction-set GPU implementation | 3.52× |
| vs "SOTA AI-accelerated tensor core" implementation (unnamed) | 1.78× |

**Unavailable**: GPU model, per-operation keygen/encaps/decaps numbers, batch size, timing
method, whether multi-stream. Reason: TPDS is closed access (OpenAlex `oa_status: closed`) and
the ePrint PDF is behind a Cloudflare 403.

→ **Not comparable with us.** Its "key exchange kops/s" also comes with no formula, so it
cannot be reconciled with ConvKyber's `ab/(a+b)` convention either.

### 2.3 ConvKyber (TCHES 2024)

- "ConvKyber: Unleashing the Power of AI Accelerators for Faster Kyber with Novel
  Iteration-based Approaches", **TCHES 2024(2): 25–63**,
  [DOI](https://doi.org/10.46586/tches.v2024.i2.25-63),
  [PDF](https://tches.iacr.org/index.php/TCHES/article/download/11420/10923) (CC-BY)
- Authors: Tian Zhou, Wenxu Tang, Jingqiang Lin (USTC); Fangyu Zheng, Lipeng Wan, Yi Bian
  (UCAS); Guang Fan, Yixuan Song (Ant Group)
- Platforms: **Jetson AGX Orin** (15/30/50/60 W, SM 6/8/16) and **RTX 3080** (68 SM, 1.8 GHz,
  100 KB smem/SM)
- Algorithm: **Kyber round-3 (v3)**, not FIPS 203
- **Approach clarified: it does not replace the NTT with convolution.** In the paper's own
  words, "we name this framework convKyber because we convert the core NTT operations into
  operations **resembling** convolutions" — the NTT stays; the butterflies are rearranged into
  iterative small block matmuls. Toeplitz convolution for polynomial multiplication is
  **prior work it cites** (Wan et al., used for LAC/NTRU).
- Precision: **INT8/INT16 tensor cores + INT32 accumulate**; 12-bit coefficients split into two
  6-bit halves (recombined with `SHL 12`/`SHL 6`); optional Karatsuba (4 mults → 3); one
  Montgomery reduction after many accumulations. **No fp16.**
- Open source: no

**Kyber-1024 throughput (Tables 8/9, kops/s):**

| platform | KeyGen | Enc | Dec | Decaps† | KX |
|---|---:|---:|---:|---:|---:|
| **RTX 3080 (this paper)** | 2,074 | 1,587 | **8,458** | **1,336†** | 1,336 |
| Wan et al., tensor core, R3080 | 1,250 | 1,299 | 2,381 | — | 820 |
| Gupta et al., re-measured on R3080 | / | / | / | — | 473 |

† **Decaps is derived, not measured** (table note: `ab/(a+b)`, a=Enc, b=Dec).

- Batch size: R3080 KeyGen/Enc **17,408** concurrent (grid 136 × block 128), Dec 8,704
- Whole-pipeline latency 1.02–5.68 ms (for the entire batch, not per op); single `polyvec_ntt`
  0.259 µs
- Timing: average of 500 iterations, **no mention of cudaEvent, no statement about whether
  host/H2D is included**
- **No tabulated numbers for Kyber-768**, only a bar chart in Figure 18

### 2.4 Hansung OAK_1445 → actually **TensorCrypto (IEEE Access 2022)**

**The link supplied was broken**: `.../2024.oak/1883/2/OAK_1445.pdf` returns HTTP 404 "Invalid
Identifier". The real file under the same handle is
`.../1883/2/AR_E1A1A4_TensorCrypto- High Throughput Acceleration of Lattice-Based Cryptography Using Tensor Core on GPU.pdf`,
item page <https://dspace.hansung.ac.kr/handle/2024.oak/1883>.

- **"TensorCrypto: High Throughput Acceleration of Lattice-Based Cryptography Using Tensor
  Core on GPU"**, **IEEE Access 10:20616–20632, 2022**,
  [DOI](https://doi.org/10.1109/ACCESS.2022.3152217)
- Authors: Wai-Kong Lee (Gachon), **Hwajeong Seo (Hansung)**, Zhenfei Zhang (Electric Coin
  Company), Seong Oun Hwang (Gachon)
- Platforms: RTX 2060 / RTX 3080 at 1.71 GHz; CPU baseline i7-9700F
- **Algorithms: NTRU (hps2048509/677), LAC, FrodoKEM. No Kyber/ML-KEM at all.**
- **Approach: exactly "Toeplitz/cyclic matmul + tensor cores instead of NTT"**, FP16 inputs +
  FP32 accumulate. This is the closest precedent to our work.
- Open source: <https://github.com/benlwk/Tensorcrypto> (public domain)

**Numbers (NTRU, not Kyber)**: TensorTRU on RTX 2060 does 793,651 encaps/s and 505,051
decaps/s (512 blocks); the convolution itself is **0.33 µs** (N=512); up to 3.41× (2060) /
5.77× (3080) over CUDA-core INT32 GPU.

**Its own explanation for skipping Kyber** is quoted in §0.6. In short: **it judged Kyber's NTT
fast enough that convolution offers no advantage, so it did not attempt it.** Our work is an
empirical rebuttal (§0.5, §0.7) — though strictly there is no same-card comparison, so it
counts only as empirical evidence.

---

## 3. Cross-implementation tables

### 3.1 Everything aligned to complete decaps throughput

| implementation | hardware | compute units | algorithm | NTT? | tensor cores | complete decaps | per SM | batch |
|---|---|---:|---|---|---|---:|---:|---:|
| **cuPQC-PK (I measured)** | **GB10** | **48 SM** | **ML-KEM** (FIPS 203) | yes | yes (undisclosed) | **4.96 M/s** | 103 k | 262,144 |
| **ours** | **GB10** | **48 SM** | **ML-KEM** (FIPS 203) | **no** | fp16+fp32 | **8.0 M/s** | **167 k** | 8,192 |
| **→ same-machine ratio** | | | | | | **ours 1.61×** | **1.61×** | |
| cuPQC-PK (vendor) | H100 | 132 SM | **ML-KEM** | yes | yes | **8.5 M/s** | 64 k | not stated |
| **ours** (1024) | GB10 | 48 SM | ML-KEM | no | yes | **5.78 M/s** | **120 k** | 4,096 |
| ConvKyber (1024) | RTX 3080 | 68 SM | Kyber v3 | yes (rearranged into convolution-like form) | yes (int8/16 + int32) | **1.34 M/s†** | 19.6 k | 17,408 |
| **ours** (integrated GPU) | Arc B390 | 96 EU | ML-KEM | no | yes | **3.26 M/s** | — | 4,096 |
| HI-Kyber | unavailable | unavailable | Kyber v3 | yes | no | **unavailable** | — | unavailable |
| TensorCrypto | RTX 2060/3080 | — | **NTRU/LAC, no Kyber** | **no** (convolution) | yes (fp16+fp32) | **not applicable** | — | 512 |

† derived value, not measured.

### 3.2 Against CPU baselines (self-reported, **different baselines, not cross-comparable**)

| implementation | baseline | decaps speedup |
|---|---|---:|
| cuPQC | single-core EPYC 7313P | 135× |
| **ours** | liboqs 0.16.0, 16 threads (same machine) | **3.6× (Arc) / 6.4× (GB10)** |
| **ours** | liboqs 0.16.0, 1 thread (same machine) | **25× (Arc) / 45× (GB10)** |
| ConvKyber | AVX2 single core | 136× slower on single-op latency; a win only at 17,408 concurrency |
| TensorCrypto | AVX2 | 2×+ (NTRU) |

Note our baseline is **liboqs with 16 threads** (1.45 µs/op), about 11× stronger than the
single-core baseline cuPQC uses (15.9 µs/op). So our 3.6–6.4× and cuPQC's 135× **cannot be
compared at all.**

### 3.3 Single-operation latency

**Same machine (GB10, both at batch=1, end-to-end including launch):**

| | single-op latency |
|---|---:|
| **cuPQC (GB10)** | **61.4 µs** (of which ~53 µs is fixed launch cost) |
| **ours (GB10)** | **282 µs** |
| **→ we are 4.6× slower** | |
| **ours (Arc B390, integrated GPU)** | 966 µs |

**Correcting an earlier error**: this document previously said "three orders of magnitude
slower". That compared ConvKyber's **amortized** figure (total time for a 17,408-op batch
divided by 17,408 = 0.06–0.33 µs) against single-op latency — invalid. Same machine, same
convention (batch=1, launch included) the gap is **4.6×**, not three orders of magnitude.

The reason is that launch overhead is enormous on both platforms (cuPQC itself eats 53 µs on
GB10; our 13 launches eat about 900 µs on the Intel iGPU). At B=1 nothing can be amortised.
**For one-op-at-a-time use we are still much worse**, just not by three orders of magnitude.

### 3.4 Complete cuPQC same-machine data (GB10)

| metric | keygen | encaps | **decaps** |
|---|---:|---:|---:|
| pure kernel (BD=32, batch 262144) | 5.75 M/s (174 ns) | 5.01 M/s (200 ns) | **4.96 M/s (202 ns)** |
| pure kernel (BD=128 default, batch 1048576) | 5.12 M/s (195 ns) | 4.03 M/s (248 ns) | 4.27 M/s (234 ns) |
| batch=1 kernel | 53.4 µs | 55.4 µs | 57.4 µs |
| batch=1 end-to-end | 57.3 µs | 59.4 µs | 61.4 µs |

BlockDim sweep (decaps, Mops/s):

| BD | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|
| batch 262144 | **4.958** | 3.882 | 4.275 | 1.654 |

**The default BD=128 has to be tuned to 32 to gain 15%** — the same story as "you must sweep
your own tile configuration" on Intel.

Multi-stream (decaps, BD=32, batch 262144): 1/2/4/8 streams → 4.955 / 4.947 / 4.937 / 4.929
M/s, **zero gain, slightly negative**.

Batch convergence: 512 → 4096 → 32768 → 262144 → 1048576 converges monotonically, saturating
after 2.6e5.

---

## 4. Our advantages

### 4.1 No NTT — the whole path is dense

The NTT's bottleneck is not arithmetic volume but **access pattern**: butterfly strides of
128/64/…/1 mean every stage has to be permuted, and every pointwise product must immediately
be reduced into `[0,q)` via elementwise modular arithmetic (a division or a multiply-add
sequence). Such kernels have low arithmetic intensity and are easily stalled by shuffles and
integer division.

We fold the multiplication into fixed 256×256 blocked Toeplitz matrices and hand
`[M, 256k] × [256k, B]` straight to fp16 tensor cores. The result is a **pure dense GEMM**,
and tensor-core utilisation can be quantified: 6.4 TFLOPS on Intel and 15.0 TFLOPS on GB10,
which is **24% / 28%** of the same-shape pure-GEMM peak on each machine (26–29 / 53 TFLOPS).

### 4.2 fp16 exactness is **provable**, not tuned

```
|W| <= 1664 < 2^11,  |y| <= 2,  K = 256k <= 1024
=> |partial sum| < 1024 * 3328 = 3.4e6 < 2^24
```

fp32 has a 24-bit mantissa, so **provided the accumulation happens in fp32, every accumulation
order is element-wise exact** — independent of the GPU's reduction tree and of the backend.
`tests/test_dot_exact.py` verifies the boundary values (±1664 × ±3) for element-wise equality
on both machines.

This matters more than it looks: outside flash-attention, almost all fp16 GEMM applications
tolerate error, whereas cryptography cannot be off by a single bit. Many GPU Kyber
implementations therefore switch to int16/int32 or add error compensation; this route is free.

### 4.3 The only implementation measured on two vendors

The same code (pure Triton — no inline assembly, no vendor-specific intrinsics):

- Intel Arc B390 / Xe3-LPG / triton-xpu 3.9 → **264 tests green**
- NVIDIA GB10 / sm_121 / triton 3.8 → **264 tests green**

Not one kernel line changed; only the tile configuration differs (`devices.gemm_cfg`).

### 4.4 We report end-to-end, and the host side is genuinely optimised

Papers typically bracket just the kernel with `cudaEvent`, excluding numpy packing / H2D / D2H
entirely. We report both, and treat the host side as first-class:

- `decaps()` accepts `torch.Tensor` (zero host copy) / `bytes` / `np.ndarray` / `list[bytes]`;
  host cost drops from 0.89 ms to 0.57 ms (B=4096, Intel)
- H2D via pinned staging is **adaptive**: the first call measures both paths on 4 MiB and only
  uses pinned if it is more than 15% faster (worth it on Intel: 0.50 → 0.28 ms; not worth it
  on GB10, whose pageable path already does 50 GB/s)
- per-ciphertext length validation went from O(B) to O(1) (saving 0.13 ms)
- and it incidentally proved that **H2D cannot overlap compute on this Intel driver**
  (dual-stream zero-dependency 1.328 ms vs same-stream serial 1.307 ms), which rules out
  streaming double-buffering entirely

### 4.5 Depth of fusion

The whole pipeline is **13 kernel launches** (B=4096, ML-KEM-768), with both GEMM epilogues
fused into the GEMM kernel:

- ENC: `GEMM + mod q + compress(du/dv) + compare + column reduction` → never materialises the
  16.8 MB `acc2`
- DEC: `GEMM + (V−acc) mod q + compress(1 bit) + 8-row packing` → never materialises the
  4.2 MB `acc`

Paired same-process ratios: ENC fusion takes device-side from 2.75 → 1.62 ms and DEC fusion
from 1.62 → 1.22 ms (Intel, B=4096; `fallback/fused` median 1.89×). Sponge input preparation
(pad10\*1 + segment concatenation + broadcast) is also compressed into one launch (the J stage
goes from 192 µs to 28 µs).

### 4.6 Correctness is reproducibly verified, and silent fallback is guarded against

- byte-exact against the official NIST ACVP vectors (passing on both machines)
- every fused kernel has a test asserting it must be **byte-equal to the unfused multi-step
  implementation**. This layer is necessary: a fused kernel returns `False`/`None` and writes
  no output when its constraints are unmet, and the caller falls back silently. We actually hit
  this during development — `Y` was a transposed view with stride (1,768), so the fused GEMM
  was never executing.
- `benchmarks/bench_e2e.py` self-checks fused against the fully-fallback path

### 4.7 The measurement methodology is public

- GPU clocks drift 0.9 ↔ 2.2 GHz and the `xe` driver exposes no frequency control → only
  same-round paired ratios are trusted
- one falsified hypothesis (pinned staging "not worth it" → actually worth it, because I had
  measured the memcpy with numpy assignment at 0.25 ms instead of `copy_` at 0.035 ms)
- one measurement trap (patched the wrong namespace and measured the fusion gain as 0.003 ms)

---

## 5. Our disadvantages

In order of severity.

### 5.1 keygen loses to cuPQC by 2.7×, and fusion cannot close it

469 ns/key against cuPQC's 173.9 ns on GB10. The NTT cannot be avoided in keygen (§0.4), and
the measured upper bound on pipeline fusion is 81 ns/key (§0.4), which would leave us at
0.45×. Closing this gap means matching hand-written CUDA kernel by kernel.

### 5.2 Single-operation latency is 4.6× worse than cuPQC

| | B=1 end-to-end |
|---|---:|
| cuPQC (GB10, same machine) | **61.4 µs** |
| ours (GB10) | **282 µs** |
| ours (Arc B390 integrated GPU) | 966 µs |

Same machine, same convention: **4.6×**. The cause is 13 kernel launches at 35–40 µs fixed
cost each, with nothing to amortise at B=1 — and cuPQC itself eats 53 µs of launch overhead on
GB10. **For one-op-at-a-time use we are clearly worse**; this is the inherent cost of a
throughput-oriented design.

### 5.3 7.9 MB of VRAM and 13.5 ms of pre-processing per private key

> Addendum after the equal-convention experiment: moving the rebuild onto the GPU
> (`kernels/rebuild.py`) takes only **0.158 ms** on GB10, 85× faster than the 13.45 ms CPU
> path. So the claim "build cost kills it" has to be narrowed: **on-GPU rebuild is 0.158 ms,
> i.e. 19.3 ns/op at B=8192** — perfectly acceptable. The real constraint becomes "7.9 MB
> resident per key", not time.

| | ours | standard implementation |
|---|---:|---:|
| `dk` size | 2400 B (unchanged) | 2400 B |
| extra resident VRAM | **7.86 MB** | 0 |
| load / pre-process | **13.45 ms** (CPU numpy) | ~0 |
| source of `Â` | expanded to a Toeplitz matrix during pre-processing | recomputed from ρ via SHAKE each time |

A standard ML-KEM implementation does not store `Â`; it keeps 32 bytes of ρ and calls
`SampleNTT` when needed. We expand `Â`, `t` and `s` and keep them resident. **That directly
limits the use case**: one long-lived private key plus a flood of ciphertexts (a TLS server, a
KEM decapsulation service) is fine; one ephemeral key per connection would be crushed by the
13 ms.

### 5.4 ML-KEM-512 takes the slow path

`η1 = 3`, while the fused CBD kernel only implements packing for `η = 2`, so 512 falls back to
the generic CBD:

| ps | Intel device-side µs/op | GB10 device-side µs/op |
|---|---:|---:|
| ML-KEM-512 | 0.932 | 0.760 |
| ML-KEM-768 | **0.307** | **0.132** |

512 is 3× slower than 768. Functionally correct (ACVP passes), just unoptimised. A test pins
this limitation.

The per-kernel evidence (`bench_kernels.py --ps ML-KEM-512 --batch 4096`, 3.118 ms device-side
total on Intel) shows that the top two entries alone — `at::native::xpu::ReduceKernel` (0.933
ms, 29.9%) and `at::native::xpu::ElementwiseGroupRangeKernel` (0.542 ms, 17.4%) — plus
`CatArrayBatchedCopy` and several `VectorizedElementwiseKernel`s account for about 2.2 ms, all
of them PyTorch native elementwise/reduction operators, i.e. the generic CBD fallback
(`bits.reshape(n,256,2η)` + two sums + subtract + `torch.cat`). The two fused kernels together
are only 0.25 ms, 8% of the total. So 512's slowness is **confirmed** to be the η=3 CBD
fallback, not the GEMMs.

### 5.5 Depends on fp16 tensor cores

A device without tensor cores (or with them disabled) degrades to the `mm(fp32, fp32)`
fallback, measured 2–4× slower (DEC 0.20 → 0.44 ms, ENC 0.31 → 1.13 ms). This is not the
portability promise "runs on any GPU" — it is "any GPU with fp16 tensor cores".

### 5.6 No formal constant-time guarantee

- timing does not depend on ciphertext content (GEMM plus `torch.where` selection, both
  data-independent)
- but **no dudect / ctgrind / Valgrind-based leak test has been run**, and no cache-side-channel
  analysis has been done. Whether L2 contention among the 7.9 MB-per-key weights leaks
  key correlation when several keys are resident at once is completely unstudied.
- batch size affects timing (not unique to us, but nothing has been done to constant-time it)

### 5.7 The host side is still 29% (Intel)

Of the 1.99 ms end-to-end at B=4096, 0.57 ms is host: `b"".join` 0.25 + H2D 0.28 + D2H 0.03.
That is already the floor for the `list[bytes]` interface; going lower requires the caller to
hand over one contiguous buffer (down to ~0.09 ms, but that changes the API, not the
implementation).

### 5.8 Never run on a datacentre GPU, never measured on AMD

The comparison objects use A100 / RTX 3090 / H100-class hardware. We only have:

- an **integrated GPU** (96 EU, 29 GB shared memory)
- a **GB10** (a 48-SM consumer module)

So every cross-implementation absolute comparison in this document carries that unfairness.

### 5.9 No same-card, same-convention NTT comparison — so "our method is faster" cannot be said

What we can prove:

- 8.0 M decaps/s on GB10 and 3.26 M/s on an Arc iGPU (measured, ACVP-validated)
- which beats same-machine liboqs with 16 threads by 3.6–6.4×

**What we cannot prove**: that we are faster than an NTT implementation that also uses tensor
cores. ConvKyber's complete decaps on RTX 3080 (derived 1.34 M/s) is 6× below our 8.0 M/s on
GB10, but that card is smaller, its number is derived, and its algorithm is Kyber v3. Three
variables differ at once, so no conclusion holds.

The only clean way to answer the question is **ConvKyber's integer tensor-core NTT against our
fp16 Toeplitz GEMM, on the same card, over the same ACVP ciphertexts, with the same timing**.
Nobody has done that — including us.

### 5.10 GEMM only reaches 24–28% of peak

The non-GEMM stages take the majority: Keccak 16%, `_pre` 17%, CBD→Y/E 16%, padcat 8%. VTune
has long reported `XVE Array Stalled/Idle 75.1%` and occupancy 28.5%.

---

## 6. Comparability warnings (read this before any comparison table)

1. **Different hardware.** We run on a 96-EU integrated GPU and a 48-SM GB10; the papers run
   on A100/H100/RTX 3090. Cross-hardware absolute comparisons are meaningless — only "speedup
   over a CPU baseline on the same hardware" or "percentage of tensor-core peak" can be
   compared.
2. **Completely different batch sizes.** They report single-op latency (batch=1); we report
   throughput at B=4096–8192. The same code can differ by **3000×** between B=1 and B=8192
   (Intel: 966 → 0.31 µs/op). Comparing our 0.31 µs/op against their per-op 10 µs is
   **misleading**.
3. **Different timing conventions.** They usually bracket the kernel with `cudaEvent`; we
   distinguish "device-side" from "end-to-end". Our device-side convention is closer to theirs
   but still includes 13 Python-side enqueues (hidden by device work at large B).
4. **Different algorithm coverage.** All three operations now exist here; the papers vary — some
   report only decaps, some only key exchange.
5. **Kyber768 vs ML-KEM-768.** The core maths is identical, but FIPS 203 decapsulation adds an
   implicit-rejection hash (`J(z‖c)`) and a constant-time selection. We implement ML-KEM.
6. **Our absolute values jitter by ±20%.** GPU clocks drift 0.9 ↔ 2.2 GHz and cannot be locked;
   only same-round ratios are trusted.
7. **What baseline is a "speedup" measured against?** Papers often use AVX2/AVX-512
   single-core; we use liboqs 0.16.0 at 1 thread (10.26 µs) and 16 threads (1.45 µs). Different
   baselines, different conclusions.
8. **They have an NTT, we do not.** This is not "who optimised better", it is **two different
   ways of organising the algorithm**. Their implementation can be used directly for
   keygen/encaps (ρ changes every time); ours cannot.
