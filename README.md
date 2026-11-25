# mlkem-triton

**ML-KEM (FIPS 203) implemented in pure Triton, with the NTT replaced by Toeplitz matrix
multiplication.**

Batched decapsulation, encapsulation and key generation. No inline assembly, no
vendor-specific intrinsics: the polynomial multiplication is folded into a fixed matrix and
handed straight to the fp16 tensor cores — except in keygen, which genuinely needs an NTT.

> ### Naming: this is ML-KEM, not Kyber round-3
>
> **ML-KEM (FIPS 203)** is the scheme NIST standardised in 2024, derived from CRYSTALS-Kyber.
> The two differ in ways that are **not interoperable**: FIPS 203 dropped the KDF at the end
> of encaps (`K` is simply the first 32 bytes of `G(m‖H(ek))`), and keygen expands the seed
> with `G(d ‖ k)` rather than `G(d)`.
>
> Every correctness claim in this project rests on the NIST ACVP vectors for ML-KEM. There is
> **not a single Kyber KAT** in this repo. The name `mlkem-triton` is what it is; it does not
> mean the library interoperates with Kyber round-3.

```python
from mlkem_triton import Decapsulator, Encapsulator, generate_keypairs

dec = Decapsulator("ML-KEM-768", dk)  # dk uploaded to the GPU once
keys = dec.decaps(ciphertexts)  # list[bytes] / np.ndarray -> [B, 32] uint8

enc = Encapsulator("ML-KEM-768", ek)  # ek uploaded once
cts, keys = enc.encapsulate(msgs)  # 32 random bytes each -> [B, 1088] / [B, 32]
```

All three operations exist, and all three keep their key material resident on the device:

| op | correctness | per op (GB10, device-side) |
|---|---|---:|
| `decaps` | ACVP 3×10 vectors, ciphertext and key byte-exact | **125 ns** |
| `encaps` | ACVP 3×25 vectors, ciphertext and key byte-exact | **88 ns** |
| `keygen` | byte-exact against kyber-py (ek + dk) | **469 ns** |

**keygen is 4–5× slower than the other two, and 2.7× slower than cuPQC** — see
[Limitations](#limitations).

On an Intel Arc B390, ML-KEM-768 at B=4096: **≈0.45 µs/op**, about 23× faster than
single-threaded liboqs and 3.3× faster than 16 concurrent liboqs processes.

---

## Why it is fast

Almost all of ML-KEM's arithmetic is polynomial multiplication, and the conventional way to
do that is an NTT. This project inverts the arrangement:

**pre-process a fixed matrix once, then make each decapsulation's multiplications a single
GEMM.**

For `a ∈ Z_q[X]/(X^256+1)` define the Toeplitz matrix

```
T(a)[i][j] = a[(i - j) mod 256] · (+1 if i >= j else -1)
```

Then for any `b`:

```
T(a) @ b  ==  a · b   in Z_q[X]/(X^256+1)
```

(The proof, and a test for it, are in `tests/test_ntt.py`.) `keyprep` expands `A`, `t` and
`s` into such matrices when the key is loaded; after that each decapsulation is two GEMMs:

| step | shape |
|---|---|
| decrypt `w = v − W_dec·u` | `[256,768] × [768,B]` |
| re-encrypt `[u';v'] = W_enc·y + [e1; e2+1665m']` | `[1024,768] × [768,B]` |

The price is a weight matrix 256× larger than the polynomial (about 1.5 MB, loaded once and
kept resident), in exchange for a dense, cache-friendly GEMM. See
[docs/design.md](docs/design.md) for the full argument.

### Why fp16 tensor cores are still exact

```
|W| <= 1664 < 2^11,   |y| <= 2,   K = 768
=>  every partial sum  <  768 × 3328 = 2.56e6  <  2^24
```

fp32 has a 24-bit mantissa, so as long as **the accumulation happens in fp32, every
accumulation order is element-wise exact** — independent of the GPU's reduction order, and
therefore across backends. `tl.dot(..., out_dtype=tl.float32)` guarantees this.

Conversely, **the output cannot be fp16**: `|sum|` peaks around 1.8e5, above fp16's 65504
limit, and becomes inf. That is also why both GEMMs use this project's own Triton kernels
rather than `torch.mm`.

---

## Install

```bash
pip install mlkem-triton
```

Dependencies: `torch>=2.4`, `triton>=3.0`, `numpy`. Nothing needs to be compiled.

For test helpers that need to generate key pairs at random (`mlkem_triton.refimpl`):

```bash
pip install "mlkem-triton[ref]"     # pulls in kyber-py
```

The primary correctness gate is the official NIST test vectors in `data/acvp/`, which
**do not** require kyber-py.

---

## Usage

```python
from mlkem_triton import Decapsulator

dec = Decapsulator("ML-KEM-768", dk)  # device=None -> picks cuda / xpu automatically

keys = dec.decaps(ciphertexts)  # -> np.ndarray [B, 32] uint8
keys = dec.decaps(arr)  # or pass a [B, 1088] uint8 array
dec = Decapsulator("ML-KEM-768", dk, device="cuda:1")
```

`decaps()` does: numpy packing → one H2D → the whole pipeline on device → one D2H. Only
32 bytes × B ever travel back.

If the caller already manages device tensors, use `dec.decaps_torch(ct[B, ct_size], B)`:
both sides are `torch.Tensor` and nothing synchronises in between.

**Bigger batches pay off.** A single 32-byte output plus one kernel launch's fixed cost
dominates below B = 64; you need B >= 1024 to saturate the device.

---

## Performance

ML-KEM-768, min-of-25, **including numpy packing, H2D and D2H**.

**NVIDIA GB10 (DGX Spark, sm_121) / torch 2.14 / triton 3.8**

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.390 ms | 0.275 ms | 0.374 | 2.67 | 3.9× |
| 2048 | 0.493 ms | 0.501 ms | 0.315 ms | 0.241 | 4.15 | 6.0× |
| 4096 | 0.869 ms | 0.876 ms | 0.533 ms | 0.212 | 4.71 | 6.8× |
| 8192 | 1.665 ms | 1.678 ms | 1.018 ms | 0.203 | 4.92 | 7.1× |

**Intel Arc B390 (Xe3-LPG, 96 EU) / torch 2.15 / triton-xpu 3.9**

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.919 ms | 1.090 ms | 0.665 ms | 0.898 | 1.11 | 1.61× |
| 2048 | 1.242 ms | 1.431 ms | 0.769 ms | 0.607 | 1.65 | 2.39× |
| 4096 | 1.943 ms | 2.196 ms | 1.237 ms | 0.474 | 2.11 | 3.06× |
| 8192 | 3.674 ms | 4.277 ms | 2.227 ms | 0.449 | 2.23 | 3.23× |

Baseline, measured on the same machine over the same ACVP ciphertexts (liboqs 0.16.0,
ML-KEM-768):

| | µs/op | vs this library, Intel (B=8192) | vs this library, GB10 (B=8192) |
|---|---:|---:|---:|
| liboqs, 1 thread | 10.26 | 22.9× | 50.5× |
| liboqs, 16 threads | 1.45 | 3.23× | 7.14× |

> Caveat: GPU clocks drift by 2–3× between sessions, so **only same-process interleaved
> A/B is meaningful**. The `vs liboqs` column is a cross-process reference value — read it
> for order of magnitude. Re-measuring the same code in a different session can move the
> absolute numbers by 10–20%.

Where the device-side time goes, the per-kernel breakdown, the fusion A/B and the comparison
against cuPQC live in [docs/benchmarks.md](docs/benchmarks.md).

---

## Support matrix

| backend | status | notes |
|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, 264 tests green |
| NVIDIA CUDA | ✅ **measured** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8, 264 tests green |
| AMD ROCm | ⚠️ **untested** | code-level reasoning only |
| Apple MPS | ❌ | no Triton backend |
| CPU | ❌ | no usable Triton CPU target |

The CUDA column was **actually run on that machine**: same code, same ACVP vectors, 264
tests green, without touching a single kernel. ROCm remains an inference from the code.

---

## Testing

```bash
pytest                          # recommended; GPU cases skip themselves with no GPU
pytest --device cuda            # pin a device
ruff check . && ruff format --check .
```

**With no GPU, `pytest tests` skips the cases that need one** (only the maths / reference
half runs — currently 90 passed / 174 skipped), so a CPU-only machine can still smoke-test.
Contribution process and the **measurement discipline** are in [CONTRIBUTING.md](CONTRIBUTING.md).

The suite has three layers:

1. **Pure maths** (`test_ntt.py`, `test_keyprep.py`, `test_dot_exact.py`) — the Toeplitz
   identity, NTT self-consistency, the exactness bound of fp16 accumulation. Partly runs
   without a GPU.
2. **Kernel bit-alignment** (`test_fused_gemm.py`, `test_glue.py`) — every fused kernel must
   be **byte-equal** to the unfused multi-step version. This is the guard against silent
   fallback: a fused kernel returns `False`/`None` and **writes nothing** when its
   constraints are not met, and the caller falls back. If the guard condition is ever
   written wrong, this layer goes red immediately.
3. **End-to-end** (`test_pipeline*.py`) — official NIST ACVP vectors, implicit rejection,
   and every batch size.

---

## Limitations

Read these before reaching for this library.

- **keygen is the one operation that loses to cuPQC.** For decaps/encaps the multiplication
  folds into a GEMM over a matrix that is fixed **across operations**, so the Toeplitz trick
  applies; but keygen's `s`/`e` are new for every key, there is no reusable matrix, and `ek`
  must contain `t̂` in the NTT domain — so the NTT cannot be avoided. On GB10: 469 ns/key
  versus cuPQC's 174 ns (**we are 2.7× slower**). See [docs/compare.md](docs/compare.md).
- **ML-KEM-512 takes the slow path.** Its `η1 = 3`, while the fused `cbd_ye` kernel only
  implements bit-packing for `η = 2` (exactly two coefficients per byte). 512 falls back to
  the generic CBD path: at B=4096, **0.93 µs/op on Intel / 0.76 µs/op on GB10**, against
  0.31 / 0.13 for 768. Correct (the ACVP vectors pass), just unoptimised. 768 and 1024 are
  both on the fast path.
- **Each private key costs 7.9 MB of VRAM and 13 ms of pre-processing.** `keyprep` expands
  `Â`, `t` and `s` into resident Toeplitz matrices (768: `W_dec` 256×768 + `W_enc` 1024×768,
  7.86 MB in fp16), which takes 13.45 ms once (numpy, on CPU). **A standard implementation
  stores 2400 bytes of `dk` and recomputes `Â` from ρ on demand.** So this design only works
  for *one long-lived private key and a flood of ciphertexts*; one ephemeral key per
  connection would be destroyed by the 13 ms.
- **Batches must be large.** See [Usage](#usage).
- **Replacing the NTT with Toeplitz matrices is not this project's invention.** Writing
  lattice-crypto multiplication as a matmul and handing it to tensor cores appears in the
  prior literature. The contribution here is on the engineering side: the fp16 exactness
  argument, the two-GEMM epilogue fusion, the sponge/padcat/CBD fusion, and getting the path
  working and quantified on real hardware.
- **Constant time** is only guaranteed to the level of "the batch shape is fixed". Batch size
  affects timing; the contents of an individual ciphertext do not.
- **There are no hand-written SYCL kernels.** The original prototype of this work also
  carried a hand-written SYCL Keccak on Intel (sponge only, 2.2–2.6× faster than the Triton
  version). That kernel was Intel-only and required shelling out to `icpx` to build, which
  cost the library its portability, so it was removed from the open-source version. Measured
  cost: sponge goes from 10.9% to 16.5% of device time, and end-to-end is about 20% slower.
  It still lives in the original repository if you only ever run Intel and care about that
  20%.
- **`kernels/rebuild.py` is a measurement tool, not a production path.** It rebuilds
  `W_dec`/`W_enc` from `dk` entirely on the GPU (SHAKE-128 → rejection sampling → INTT →
  Toeplitz expansion, 0.158 ms on GB10), which exists so that this library can be compared
  on equal terms with a stateless API like NVIDIA cuPQC that recomputes `Â` per operation.
  The main path uses the pre-expanded matrices from `keyprep`.
- **`torch.mm`'s `out_dtype=torch.float32` is unusable.** It is a private torch-xpu overload
  (`aten::mm.dtype`); the public signature only has `out=None`. And an fp16 output overflows
  to inf here. So both GEMMs are this project's own Triton kernels — which incidentally are
  also faster than `torch.mm` (the DEC pass goes from `0.200 + 0.070` to `0.111` ms).

---

## Documentation

| document | contents |
|---|---|
| [docs/design.md](docs/design.md) | why Toeplitz, the fp16 exactness proof, the fusion design |
| [docs/benchmarks.md](docs/benchmarks.md) | every measured table: throughput, per-kernel breakdown, fusion A/B |
| [docs/support.md](docs/support.md) | backend support detail, tuning knobs, project structure |
| [docs/compare.md](docs/compare.md) | systematic comparison with cuPQC, HI-Kyber, ConvKyber, TensorCrypto |
| [CONTRIBUTING.md](CONTRIBUTING.md) | environment, tests, adding a backend, measurement discipline |

The Chinese originals of the long-form documents are kept as supplementary material in
[docs/zh/](docs/zh/); the English versions above are the authoritative ones.

## License

Apache-2.0. ML-KEM itself is a public NIST standard; the ACVP test vectors come from
[usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server) (a US Government work, in
the public domain).
