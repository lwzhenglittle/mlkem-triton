# Changelog

## [0.6.1] — signing is 1.33× faster, and the ceiling is dispatch, not arithmetic

Optimising the signing path, measured A/B in one session (ML-DSA-44, B=1024, best of 3):
**236.0 µs → 176.8 µs per signature**.

Kept:

- `pack_z_kernel` — BitPack of `z` on the device. It was ~19 % of the run: the numpy bit-packer
  extracted 18–20 bits coefficient-by-coefficient, ~6 000 times per batch. Now one launch, and the
  accepted rows are copied to the host once per attempt instead of once per signature.
- `pointwise_kernel` — `ĉ ∘ ŝ` in the NTT domain, replacing a broadcast int64 product that
  materialised three 64-bit arrays of the whole batch per attempt.
- `amax(dim=(1, 2))` instead of two chained `max` for the three norm checks.

Tried and reverted, with numbers:

- a slot/queue model that keeps the batch full while the queue lasts (so stragglers do not shrink
  it): **196–320 µs, worse**. The cost tracks rows processed, not attempts.
- fusing the three pointwise products into one launch: **186 µs, no gain**.

The profile says why there is little left at this level: one attempt is ~18 launches and ~2 ms of
dispatch, a batch of 1024 needs ~4.5 attempts per message on average but up to 84 for the slowest,
and the per-attempt stage costs are flat (NTT 14 %, hint 12 %, pointwise/INTT 29 %, decompose 9 %,
ExpandMask 8 %). Further gains need fewer, larger kernels, not tighter host code.

## [0.6.0] — ML-DSA signing on the device, and the ACVP LMS vectors

### Added — ML-DSA signing (FIPS 204 Alg. 7)

The repository was verification-only for ML-DSA; it now signs too, batched, with one long-lived
private key:

- `triton_crypto/mldsa/sign.py` — `Signer`: ExpandMask (one SHAKE per polynomial, 19-bit fields),
  `Â·ŷ` through the verifier's `mac_kernel`, `c̃ = H(μ ‖ w1Encode(w))`, SampleInBall, `c·s1`,
  `w − c·s2`, `c·t0`, MakeHint, and the acceptance checks. The rejection loop compacts the batch
  after every attempt, and messages of different lengths are signed in groups (the μ hash needs a
  rectangular buffer).
- Three entry points, mirroring the standard: `sign` (external interface with context and
  pre-hash), `sign_mprime` (internal, M′ given), `sign_mu` (external μ).
- The two encoders (`pack_z`, `pack_h`) run on the host with the ACVP-checked reference bit packers.

**360/360 ACVP sigGen cases byte-exact on device** (all three sets × internal / external pure /
pre-hash / external-μ × deterministic and hedged), plus fresh-key byte-exactness and batches large
enough to exercise the compaction. Throughput at B = 1024: 245 / 279 / 321 µs per signature
(44/65/87) versus 5.4 ms for the Python reference. No liboqs baseline: the build was not present in
the workspace for this session, and a number is not quoted rather than guessed.

### Added — the SHA-256/192 LMS family and the ACVP LMS vectors

Covered in the previous entry; the ACVP `LMS-sigVer` vectors pass 16/16 on device and in the
reference.

## [0.5.0] — LMS/HSS verification, and a measured "no" to fusing the WOTS chains

### Added — LMS / HSS (NIST SP 800-208, RFC 8554)

A new stateful-hash-based scheme next to SLH-DSA, verification only:

- `triton_crypto/lms/params.py` — the 20 SHA-256 parameter sets (H5–H25 × W1/W2/W4/W8) and their
  wire type codes. The SP 800-208 SHAKE-256 variants are deliberately absent: their numeric
  identifiers are still TBD in the standard.
- `triton_crypto/lms/ref.py` — LM-OTS / LMS / HSS verification, key generation from a seed
  (Appendix A) and a signer, checked against **both RFC 8554 test vectors** (one of which mixes
  H5/W8 with H10/W4).
- `triton_crypto/lms/kernels.py` — `lms_chain_kernel`: every LM-OTS chain of the whole batch in one
  launch, one single-block SHA-256 compression per step, per-block exit once a block's rows are
  done (the step count is data-dependent, 0–255).
- `triton_crypto/lms/verify.py` — batched HSS verification: `6 + h` launches per level for the
  whole batch, message-length bucketing, mixed parameter sets per level.

Measured at batch 4096: **8.5 µs/signature** for RFC case 1 (2 levels, H5/W8) and **7.1 µs** for
case 2, roughly 1 Ghash/s of SHA-256 compression.

### Added — the SHA-256/192 family and the ACVP LMS vectors

`params.py` now carries the second SP 800-208 family: `n = m = 24` (SHA-256 truncated to 192 bits),
type codes 5–8 for LM-OTS and 10–14 for LMS, `p = 200/101/51/26` and `ls = 8/6/4/0`. The reference,
the verifier and the chain kernel are parameterised by the set instead of assuming 32-byte values
(the kernel assembles a 47-byte message block instead of 55).

The NIST ACVP `LMS-sigVer` vectors (both the 1.0 and the SP 800-208 files) are bundled under
`data/acvp-lms/` and pass **16/16 on the device and in the reference**. They include the
"modify signature header" cases, which is what forced explicit validation of the two type words and
of `q < 2^h` in the device path — those checks are now part of `_header_ok`.

### Tried and rejected — fusing the WOTS+ chain into one kernel

The obvious next optimisation was to fuse SLH-DSA's 15 chain steps into one kernel (one launch per
layer instead of ~8). Measured on the same 128f workload at B=4096: **56.5 µs/signature versus
48 µs** unfused — slower. Keeping the 25-word Keccak state plus the ADRS words and the value live
across an inner loop spills registers, and the per-step schedule is already at 76–83 % of the
permutation roofline, so there is no launch overhead left to win. Fusing the XMSS *authentication
path* (h' steps) was neutral (50.8 µs) and broke one ACVP case under batching, so both were
reverted; the numbers are recorded here so the experiment is not repeated.

## 0.4.0 — SLH-DSA (FIPS 205) verification, and a fused `thash` kernel

The first scheme in the box that needs no arithmetic at all: SLH-DSA is hash-based, so it runs
entirely on the sponge this library already had. What it needed instead was a *scheduler* — the
verification is a dependency chain of thousands of hashes — and a hash kernel cheap enough that
400 launches per signature do not dominate.

### Added — SLH-DSA (FIPS 205) verification

- **The six SHAKE parameter sets**, verification only, batched, one long-lived public key. The
  external interface (pure and pre-hash, with context strings) and the internal one (`M'` given)
  are both supported, in `triton_crypto/slhdsa/`.
- **A numpy reference** (`slhdsa/ref.py`) implementing Alg. 1–20 for *all twelve* sets, both
  instantiations, plus `keygen_internal`. It is the oracle the device path is checked against, and
  it is itself checked against the NIST vectors: 144/144 ACVP sigVer cases (SHA-2 included) and
  120/120 keyGen cases with pk *and* sk byte-exact.
- **`thash_kernel`** (`slhdsa/kernels.py`): `PK.seed ‖ ADRS ‖ data` assembled in registers and
  permuted in one launch, specialised per `(n, dn)` so the padding is a constant. This is what
  took 128f from 172 µs to **71 µs** per signature at B = 4096.
- `benchmarks/bench_slhdsa_verify.py` and a `--scheme slhdsa` route in `benchmarks/bench.py`.
- **In-place chain steps and chain compaction.** The fused kernel reads its own output buffer
  (per-row read-then-write), so a chain step needs no `torch.where` and no copy; and because a
  chain with digit `d` needs `w-1-d` hashes, each layer sorts its rows by digit and every step
  launches only over the still-running suffix. Together: 71 µs → 48 µs for 128f at B = 4096, and
  17× → ~25× over liboqs.

Measured at batch 4096 against liboqs single-thread on the same machine: 18.7 µs (128s),
47.8 µs (128f), 67.4 µs (192f), 71.2 µs (256f) — 24–29×. The crossover against the CPU is
around B ≈ 32; below that the device is launch-bound.

### Added — the six SHA-2 parameter sets (12 of 12)

The SHA-2 instantiation hashes with SHA-256/512 instead of Keccak, and FIPS 205 makes it
asymmetric: PRF/F are always SHA-256-based, while H and T_l switch to SHA-512 above 128 bits.
Both compression kernels live in `primitives/sha2.py`, and `slhdsa/verify.py` dispatches its
`thash` to them (`_thash_sha` for one block, `_thash_sha_multi` for the ltree); the chain /
hypertree / FORS schedule is untouched.

The SHA-2 sets are also *faster* — 13.1/27.8/17.5/37.9/26.0/44.1 µs at B = 4096 for
128s/128f/192s/192f/256s/256f, i.e. 30–42× over liboqs — because one SHA-2 block compression is
cheaper on the device than 24 Keccak rounds.

Two traps cost real time here, both documented in the code: the chain kernel cannot write in place
(the value is built from the same bytes, and Triton assumes the buffers do not alias), so the
SHA-2 chain ping-pongs between two packed buffers; and the F case must write back only the n-byte
value (`n/ws` words) rather than the full 8-word state, or the packed output stride stops matching
the caller's buffer from the second row on.

### Notes

(Nothing here: all twelve sets run on device as of this release.)
- Two spec traps are documented where they bit: `WOTS_PK` / `FORS_ROOTS` addresses keep the key
  pair field, and keygen builds the tree at layer `d-1` while verification walks layers up from 0.
- `docs/kernels.md` §SLH-DSA has the kernel contract; `docs/benchmarks.md` §12 has the numbers and
  the two caveats about CPU baselines.

## 0.3.0 — renamed to `triton-crypto-kernels`, and every kernel got a contract

The package outgrew its name. 0.2.0 already had two schemes sitting on one primitive layer, but
it was still shipped as "mlkem-triton", and the kernels that both schemes actually reuse were
undocumented internals. This release is about making the library part explicit — no algorithmic
behaviour changed, which is why the whole test suite passes untouched.

### Changed

- **Package renamed `mlkem_triton` → `triton_crypto`**, distribution renamed
  `mlkem-triton` → `triton-crypto-kernels`, version bumped to 0.3.0. The old import name still
  works: `mlkem_triton/__init__.py` is a 20-line shim that maps every submodule onto the new
  package (same module objects, so `isinstance` and caches behave identically). It will be
  removed in 0.4.0.
- `README.md` now opens with what the library *is* (primitives + schemes + kernel catalogue)
  rather than with one scheme's story; the per-scheme sections are unchanged.
- `pyproject.toml`: new distribution name, description, keywords; `mlkem_triton*` added to the
  packaged modules so the shim ships.

### Added

- **`docs/kernels.md` — the kernel catalogue.** All 27 launched kernels with shape, dtype, grid,
  tiling knobs and the constraint that makes each one legal (e.g. the fp16 exactness bound that
  `gemm_modq` depends on, the three-byte `dv=5` packing case, why `pow_mac` needs no `tl.gather`).
  Plus the 10 inline `@triton.jit` helpers and the reason one of them cannot be factored out.
- **`benchmarks/bench.py`** — one entry point: `--scheme {mlkem,mldsa} [--op …]`, dispatching to
  the per-scheme measurement scripts so that reproducing a table means running the same code that
  produced it, not a re-implementation. `--list` shows what is available.

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

- `triton_crypto.mldsa.Verifier`: one long-lived public key, batch verification through the
  external interface (with `ctx`), the pre-hash interface (HashML-DSA), the internal interface,
  or an externally supplied `μ`.
- `triton_crypto.mldsa.ref`: a complete numpy FIPS 204 implementation (keygen, sign, verify),
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
- `triton_crypto/kernels/ntt.py` keeps only what is ML-KEM's: the base-case multiply
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
