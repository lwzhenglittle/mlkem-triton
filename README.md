# mlkem-triton

**ML-KEM (FIPS 203) implemented in pure Triton, with the NTT replaced by Toeplitz matrix
multiplication.**

Batched decapsulation, encapsulation and key generation. No inline assembly, no
vendor-specific intrinsics: the polynomial multiplication is folded into a fixed matrix and
handed straight to the fp16 tensor cores — except in keygen, which genuinely needs an NTT.

> ### Naming: this is ML-KEM, not Kyber round-3
>
> **ML-KEM (FIPS 203)** is the scheme NIST standardised in 2024, derived from CRYSTALS-Kyber.
> The two are **not interoperable**: FIPS 203 dropped the KDF at the end of encaps (`K` is
> simply the first 32 bytes of `G(m‖H(ek))`), and keygen expands the seed with `G(d ‖ k)`
> rather than `G(d)`. Every correctness claim here rests on the NIST ACVP vectors for ML-KEM;
> there is **not a single Kyber KAT** in this repo.

```python
from mlkem_triton import Decapsulator, Encapsulator, generate_keypairs

dec = Decapsulator("ML-KEM-768", dk)  # dk uploaded to the GPU once
keys = dec.decaps(ciphertexts)  # list[bytes] / np.ndarray -> [B, 32] uint8

enc = Encapsulator("ML-KEM-768", ek)  # ek uploaded once
cts, keys = enc.encapsulate(msgs)  # 32 random bytes each -> [B, 1088] / [B, 32]
```

| op | correctness | per op (GB10, device-side) |
|---|---|---:|
| `decaps` | ACVP 3×10 vectors, ciphertext and key byte-exact | **125 ns** |
| `encaps` | ACVP 3×25 vectors, ciphertext and key byte-exact | **88 ns** |
| `keygen` | byte-exact against kyber-py (ek + dk) | **469 ns** |

**keygen is the weak operation** — 4–5× slower than the other two, and 2.7× slower than
NVIDIA cuPQC. The other two are 1.61× and 2.27× *faster* than cuPQC on the same card. See
[docs/compare.md](docs/compare.md).

---

## Why it is fast

Almost all of ML-KEM's arithmetic is polynomial multiplication, conventionally done with an
NTT. This project inverts the arrangement: **pre-process a fixed matrix once, then make each
operation's multiplications a single GEMM.** For `a ∈ Z_q[X]/(X^256+1)`, define
`T(a)[i][j] = a[(i - j) mod 256] · (+1 if i >= j else -1)`; then `T(a) @ b == a · b` in
`Z_q[X]/(X^256+1)`. Decapsulation becomes two GEMMs, encapsulation is the second one.

fp16 tensor cores are still exact, because one operand is always small: `|W| <= 1664 < 2^11`
and `|y| <= 2`, so with `K <= 1024` every partial sum stays below `3.4e6 < 2^24` and any
accumulation order is element-wise exact, on any backend. The output cannot be fp16 (`|sum|`
peaks near 1.8e5, above fp16's 65504), which is why both GEMMs are this project's own Triton
kernels. Full argument, fusion catalogue and Triton pitfalls: [docs/design.md](docs/design.md).


## Install

**Not on PyPI yet** — the name is unregistered, so `pip install mlkem-triton` fails with
"No matching distribution found". Install from GitHub until it is published:

```bash
pip install "mlkem-triton @ git+https://github.com/lwzhenglittle/mlkem-triton.git"
```

Dependencies: `torch>=2.4`, `triton>=3.0`, `numpy`. Nothing needs compiling, and the wheel is
pure Python (`py3-none-any`). Add `[ref]` to also pull in kyber-py for random key-pair helpers.

The NIST ACVP vectors in `data/acvp/` and the `tests/` tree are **not** shipped in the built
distribution, so the correctness gate needs a source checkout:

```bash
git clone https://github.com/lwzhenglittle/mlkem-triton && cd mlkem-triton
pip install -e ".[test,ref]" && pytest
```

## Usage

```python
from mlkem_triton import Decapsulator

dec = Decapsulator("ML-KEM-768", dk)  # device=None -> picks cuda / xpu automatically
keys = dec.decaps(ciphertexts)  # -> np.ndarray [B, 32] uint8
keys = dec.decaps(arr)  # or pass a [B, 1088] uint8 array
dec = Decapsulator("ML-KEM-768", dk, device="cuda:1")
```

`decaps()` does numpy packing → one H2D → the whole pipeline on device → one D2H, so only
32 bytes × B travel back; `dec.decaps_torch(ct, B)` is the same path with `torch.Tensor` in and
out, without synchronising in between. **Bigger batches pay off**: launch overhead dominates
below B = 64, and you need B >= 1024 to saturate.

---

## Performance

ML-KEM-768, min-of-25, **including numpy packing, H2D and D2H**. Both machines use the same
columns, so a batch can be read across them.

**NVIDIA GB10** (DGX Spark, `sm_121`, 48 SM) / torch 2.14 / triton 3.8

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.390 ms | 0.275 ms | 0.374 | 2.67 | 3.9× |
| 2048 | 0.493 ms | 0.501 ms | 0.315 ms | 0.241 | 4.15 | 6.0× |
| 4096 | 0.869 ms | 0.876 ms | 0.533 ms | 0.212 | 4.71 | 6.8× |
| 8192 | 1.665 ms | 1.678 ms | 1.018 ms | 0.203 | 4.92 | 7.1× |

**Intel Arc B390** (Xe3-LPG, 96 EU, integrated) / torch 2.15 / triton-xpu 3.9

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.919 ms | 1.090 ms | 0.665 ms | 0.898 | 1.11 | 1.61× |
| 2048 | 1.242 ms | 1.431 ms | 0.769 ms | 0.607 | 1.65 | 2.39× |
| 4096 | 1.943 ms | 2.196 ms | 1.237 ms | 0.474 | 2.11 | 3.06× |
| 8192 | 3.674 ms | 4.277 ms | 2.227 ms | 0.449 | 2.23 | 3.23× |

The two differ in more than speed. **GB10** is **2.2×** faster per op at B=8192, and its batch
curve stays flat out to B=65536 (0.125–0.130 µs/op, device-side). **B390** oscillates between
0.380 and 0.408 µs/op from B=8192 up. The min-vs-med spread separates them too: under 1% on GB10
against **16%** on B390 — that is the ±20% clock drift, and the reason only same-process ratios
are trusted.

Baseline, same machine over the same ACVP ciphertexts (liboqs 0.16.0, ML-KEM-768): **10.26
µs/op** single-threaded, **1.45 µs/op** across 16 concurrent processes.

> GPU clocks drift by 2–3× between sessions with no frequency control exposed, so **only
> same-process interleaved A/B is meaningful**; `vs liboqs` is a cross-process reference value.

Where the device time goes, the per-kernel breakdown, the fusion A/B, encaps and keygen
tables, and the comparison against cuPQC: [docs/benchmarks.md](docs/benchmarks.md).

---

## Support matrix

| backend | status | notes |
|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, 264 tests green |
| NVIDIA CUDA | ✅ **measured** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8, 264 tests green |
| AMD ROCm | ⚠️ **untested** | code-level reasoning only |
| Apple MPS / CPU | ❌ | no Triton backend / no usable Triton CPU target |

The CUDA column was actually run on that machine: same code, same vectors, 264 tests green,
not one kernel line changed — only the tile configuration differs. Detail and tuning knobs:
[docs/support.md](docs/support.md).

## Testing

`pytest` (GPU cases skip themselves with no GPU), plus `ruff check . && ruff format --check .`.
Three layers: pure maths (Toeplitz identity, fp16 exactness bound), kernel bit-alignment
(every fused kernel must be byte-equal to the unfused version — this is the guard against
silent fallback), and end-to-end ACVP vectors. Without a GPU: 90 passed / 174 skipped.
Contribution process and **measurement discipline**: [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

Read these first; detail in [docs/compare.md](docs/compare.md) §5.

- **One long-lived private key, many ciphertexts** is the only shape that fits: 7.9 MB of VRAM
  and 13.5 ms of pre-processing per key (on-GPU rebuild: 0.158 ms), against 2400 bytes of `dk`.
- **Throughput, not latency.** 13 launches at ~35–40 µs fixed cost make B=1 take 282 µs on GB10,
  4.6× slower than cuPQC; below B ≈ 1024 on Intel a 16-thread CPU wins.
- **keygen loses to cuPQC by 2.7×** and fusion cannot fix it — `ek` needs `t̂` in the NTT domain,
  and the fusion ceiling is 81 ns/key (1.21×).
- **ML-KEM-512 takes a slow path** (η1 = 3 falls back to generic CBD, ~3× slower); 768 and 1024
  are fast.
- **Needs fp16 tensor cores**: otherwise an `mm(fp32, fp32)` fallback, 2–4× slower. "Any GPU
  with fp16 tensor cores", not "any GPU".
- **No formal constant-time validation** (no dudect/ctgrind, no cache-side-channel analysis).
  Timing is ciphertext-independent; batch size is not.
- **No same-card NTT comparison exists**, so we cannot claim "our method beats NTT", only that we
  lead among the three implementations reporting batched throughput. Toeplitz-on-tensor-cores is
  prior art (TensorCrypto, 2022); the contribution is the exactness argument, the epilogue
  fusion, and two vendors' hardware.

## Documentation

- [docs/design.md](docs/design.md) — why Toeplitz, the fp16 exactness proof, fusion catalogue,
  Triton pitfalls
- [docs/benchmarks.md](docs/benchmarks.md) — every measured table: throughput, per-kernel,
  fusion A/B, host side
- [docs/support.md](docs/support.md) — backend status in detail, tuning knobs, project structure
- [docs/compare.md](docs/compare.md) — cuPQC, HI-Kyber, ConvKyber, TensorCrypto
- [CONTRIBUTING.md](CONTRIBUTING.md) — environment, tests, a new backend, measurement discipline

The Chinese originals of the long-form documents are supplementary material in
[docs/zh/](docs/zh/); the English versions above are authoritative.

## License

Apache-2.0. ML-KEM itself is a public NIST standard; the ACVP test vectors come from
[usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server) (a US Government work, in the
public domain).
