# Changelog

## 0.2.0 — ML-DSA verification, and Keccak/NTT as shared primitives

Two things, and they are related: **ML-DSA verification was never the problem; the Toeplitz
trick was.** The reason ML-DSA "cannot" be done like ML-KEM is that q = 8380417 is 23 bits, so
fp16 partial sums stop being exact, and there is nothing to fold anyway: `Â` arrives already in
the NTT domain. So ML-DSA needs an NTT and a sponge — which is exactly what ML-KEM already had,
buried in `kernels/`. Hence the second half of this release: those two blocks are now
scheme-independent primitives, and both schemes are thin layers on top.

### Added — ML-DSA (FIPS 204) signature verification

| | ML-DSA-44 | ML-DSA-65 | ML-DSA-87 |
|---|---:|---:|---:|
| device-side per op, B=4096 (B390) | 1.9 µs | 2.5 µs | 3.8 µs |
| end-to-end (numpy → GPU → bool) | 3.2 µs | 4.1 µs | 5.7 µs |
| vs liboqs 0.16.0, 16 processes | 5.6× | 5.6× | 5.4× |

- `mlkem_triton.mldsa.Verifier`: one long-lived public key, batch verification through the
  external interface (with `ctx`), the pre-hash interface (HashML-DSA), the internal interface,
  or an externally supplied `μ`.
- `mlkem_triton.mldsa.ref`: a complete numpy FIPS 204 implementation (keygen, sign, verify),
  byte-exact against ACVP keyGen (75 vectors) and sigGen (360, deterministic *and* hedged) and
  180/180 on sigVer. It is the oracle the GPU kernels are tested against, and the reason the
  tests need no third-party package.
- Kernels: BitUnpack of `z` (18/20-bit fields), HintBitUnpack with full validity checking,
  SampleInBall (the one sequential step: one program per signature, `while` for the rejection),
  Decompose/UseHint/BitPack of `w1` in one pass, and the NTT-domain `MAC`.
- `benchmarks/bench_mldsa_verify.py`, with a liboqs ctypes binding (`tests/_liboqs.py`) that
  doubles as the CPU baseline and as a cross-implementation check.
- Official vectors in `data/acvp-mldsa/` (sigVer 180, keyGen 75, sigGen 360 cases). ~12 MB of
  JSON, against 0.8 MB for the ML-KEM set: the sigGen prompt carries a secret key per case, and
  the sigVer prompt carries ~7 KB messages. `mldsa/acvp.py` explains where each file comes from;
  the tests skip cleanly if a file is missing.

### Changed — `primitives/` replaces `kernels/keccak.py`, `kernels/shake.py`, `kernels/ntt.py`

| new module | was | now |
|---|---|---|
| `primitives/keccak.py` | `kernels/keccak.py` | unchanged, moved |
| `primitives/sponge.py` | `kernels/shake.py` | generic sponge + `HashFn` registry (`SHA3-256/512`, `SHAKE128/256`), `pad_torch`, `sponge_padded`, `hash_np_batch` |
| `primitives/ntt.py` | `kernels/ntt.py` (ML-KEM only) | `NTTDomain(q, n, ζ, layers)`, forward **and inverse**, numpy + Triton, two instantiations: `KYBER` (int32) and `DILITHIUM` (int64) |

- The NTT's per-layer geometry, twiddle tables and inverse scale are now derived from the domain
  instead of hard-coded: ML-KEM runs **7** layers and normalises by 128⁻¹, ML-DSA runs 8 and
  normalises by 256⁻¹. Getting that wrong still passes a roundtrip test and still fails every
  ACVP vector, so it is asserted explicitly in `tests/test_primitives.py`.
- The 23-bit modulus switches to Montgomery reduction (no 64-bit `%`): 6.97 → 3.63 ms on 24 576
  rows on the B390, bit-identical output.
- The int32 path got faster as a side effect of writing the butterfly once, cleanly (select
  instead of a `±1` multiply, one reduction per output): **1.995 → 1.198 ms** on the same 24 576
  rows, bit-identical — measured against the pre-refactor kernel from git, not assumed. ML-KEM
  results are unchanged either way.
- `mlkem_triton/kernels/ntt.py` keeps only what is ML-KEM's: the base-case multiply
  (`pow_mac`), `ByteEncode_12`, and a `forward_ntt` that delegates to the primitive.
- No ML-KEM behaviour change: the 264 existing tests (including byte-exact keygen and the ACVP
  vectors) pass unmodified, and the kernel body/tables/launch config are the same.

### Testing

375 tests (was 264); 123 passed / 252 skipped without a GPU. New layers: NTT algebra for both
domains against a schoolbook convolution, sponge against `hashlib` for every rate and message
length, ML-DSA kernels against the numpy reference, and ACVP sigVer through the public API.

### Documentation

README gains "ML-DSA (FIPS 204) signature verification", "Shared primitives" and a table of what
is deliberately *not* unified between the two schemes; [docs/design.md](docs/design.md) §8–9
records the layer-count trap, the Montgomery measurement and the Toeplitz-vs-NTT arithmetic;
[docs/benchmarks.md](docs/benchmarks.md) §11 has the full ML-DSA tables.

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
