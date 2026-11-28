# Changelog

## [1.1.0] — six primitive modules, Falcon verification, barrier 16P baselines, and CI hardening

### Added

- **Six new scheme-agnostic primitive modules** under `triton_crypto.primitives`:
  - `primitives/poseidon2.py`: Poseidon2-BabyBear/16 (Plonky3 parameters, 8 full + 13 partial rounds), verified against `data/vectors/poseidon2/bb16.json` (`38.5–39.6 ns/perm` on Arc B390).
  - `primitives/poly1305.py`: Poly1305 (5×26-bit limbs) and `chacha20_poly1305_torch` / `chacha20_poly1305_open_torch` AEAD, cross-checked against OpenSSL `EVP_MAC`.
  - `primitives/aes.py`: AES-128-CTR through a 4.25 KiB T-table (`aes128_ctr_torch`), cross-checked against `openssl enc` (`9.8–10.9 GB/s`, an empirical L1-gather measurement rather than a fast path against AES-NI).
  - `primitives/gemm.py`: exact `int8 × int8 → int32` GEMM (`gemm_int8_torch`, `3.7–4.0 TMAC/s`, `9.0–9.3×` vs 16P CPU at `1024³`) and bit-plane `GF(2^8)` matrix multiply (`gf2_8_matmul_torch`, `33.5 GMAC/s` at `1024³`).
  - `primitives/blake3.py`: multi-chunk BLAKE3 tree hash (`blake3_hash_torch`, `9.7–10.1 GB/s`), verified against official Rust BLAKE3 KAT vectors (`data/vectors/blake3/kat.json`).
  - `primitives/ntt/large.py`: multi-pass `LargeNTT` up to `2^20` points over BabyBear with fused radix-4 forward (`_stage4_kernel`), fused 2×2 inverse (`_inv2x2_kernel`), and `mul_fused_torch` (`0.85–1.6 ms` forward, `2.41 ms` negacyclic product).
- **Falcon-512 and Falcon-1024 verification (`triton_crypto.falcon`, FIPS 206 draft)**: `Verifier` (auto-detecting Falcon-512 / Falcon-1024, supporting both unpadded and padded formats), fully on-device end-to-end pipeline (`_pack_sig_words_kernel` → 64-bit bit-reservoir + branchless-CLZ `_comp_decode_kernel`, `_pad_nonce_msg_kernel` + rolled multi-block `sponge` + `_hash_to_point_kernel`, reusing `primitives.ntt` for `N = 512 / 1024`). On Arc B390 (`B = 4096`), Falcon-512 reaches **0.32 µs/op** dev core (`8.3×` 16P) / **0.90–0.92 µs/op** e2e (**`2.9×` 16P**, down from `15.35 µs`); Falcon-1024 reaches **1.14–1.19 µs/op** dev core (`4.4–4.6×` 16P) / **2.18–2.22 µs/op** e2e (**`2.3–2.4×` 16P**).
- **Offline deterministic Falcon KAT corpus**: `data/vectors/falcon/kat.json` (72 liboqs-generated vectors across Falcon-512, Falcon-padded-512, Falcon-1024, and Falcon-padded-1024) exercised on both CPU and GPU without requiring a local `liboqs` build.
- **Unified barrier benchmarking & local GPU vs CPU matrix**:
  - `benchmarks/bench_16p_barrier.py`: barrier-synchronized 1T / 16P CPU baselines (36 rivals covering ML-KEM 512/768/1024, ML-DSA 44/65/87 verify+sign, SLH-DSA SHAKE & SHA2, Falcon 512/1024 unpadded+padded, all symmetric/hash/ZK primitives, and exact `gemm-int8`), with power/clock-state refusal guards.
  - `benchmarks/bench_local_full.py` & `benchmarks/local-gpu-vs-cpu-20261128.{md,json}`: single-window Arc B390 vs Core Ultra X7 358H matrix reporting **Device-only + End-to-End** ratios and 5-workload batch-size breakeven curves (including Falcon-512).
  - `benchmarks/snapshot.py`: single-window snapshot runner with JSON metadata; legacy 2026-11-26 snapshots archived under `benchmarks/archive/`.
- **Executable contracts & packaging**:
  - `tests/test_primitives_contract.py`, `tests/test_docs_api.py`, `tests/test_triton_behaviour.py`, `tests/test_cpu_only.py`, and `tests/test_bench_harness.py`.
  - PEP 561 `triton_crypto/py.typed` marker shipped in wheels.
  - Hosted GitHub Actions `cpu-test` job (Python 3.10 & 3.12 matrix), `ruff format --check .` in `lint`, and wheel import + reverse pre-0.8 path leak checks in `package`.
- **Documentation**: `docs/patterns.md` (10 computational patterns, corrected 3-factor porting rule, and Triton backend traps); restructured `docs/benchmarks.md` and package-aligned `docs/kernels.md`.

### Changed

- `benchmarks/bench_primitive.py`, `bench_falcon_verify.py`, `snapshot.py`, and `join_16p_table.py` now default to `benchmarks/cpu_baseline_barrier.json` and print the active baseline file and aggregation method dynamically.
- `tests/conftest.py` replaced the drifting `_NO_GPU_OK` allowlist with a reverse `_NEEDS_GPU` + `*_gpu.py` + `@pytest.mark.gpu` gate so new CPU tests run automatically without a GPU.
- `pyproject.toml` updated with complete `keywords`, Python 3.10–3.13 / ROCm `classifiers`, and `py.typed` package data.
- `triton_crypto/__init__.py` docstring updated to list `falcon/` and all v1.1 primitives.

### Fixed

- **Intel XPU VA `0x0` GPU hang (`UR_RESULT_ERROR_DEVICE_LOST`)**: `_fused_cfg` disables software pipelining (`num_stages=1`) on `xpu` only (preserving `num_stages=3` on CUDA) and enforces `MASK_N=True` across `fused_enc_kernel`, `fused_dec_kernel`, and `encaps_kernel`, with no measurable penalty on Xe3 (`0.274 µs/op` ML-KEM-768 decaps, `0.191 µs/op` encaps at B=4096).
- **BLAKE3 multi-chunk UB**: replaced `uint32 >> 32` (which silently folded kernel outputs to zero on `triton-xpu`) with explicit `(low, high)` counter words.
- **Benchmarking correctness**:
  - Replaced sum-of-minima 16P aggregation (1.25–4.25× CPU over-count) and mismatched 16 KiB / 1 MiB message sizes with barrier-synchronized 1 KiB measurements (`benchmarks/table-16p-20261127.md`).
  - Split SLH-DSA CPU rivals into matched `slhdsa-shake-128f-verify` and `slhdsa-sha2-128f-verify` entries.
  - Added scalar-path guard (`lanes >= 4`) for Plonky3 Poseidon2 rival, fixed radix-4 pass-count traffic reporting, and fixed 2^20 NTT product-vs-transform unit mismatch (`3.96×` / `5.53×` fused).
  - Removed tracked Cargo build artifacts from `benchmarks/blake3_ref/target/` and switched SLH-DSA benchmark liboqs discovery to environment/relative lookup.
- **API edges & warnings**: eliminated the last non-writable NumPy `UserWarning` in `primitives/aes.py`, fixed padded-mode error masking in host `falcon.comp_decode_batch`, rolled multi-block `sponge` squeeze loops in `primitives/sponge.py` (supporting 20-block Falcon-1024 XOF streams), guarded empty batches (`B == 0`) in `falcon.verify_core_torch` / `mldsa.Verifier` / `mldsa.Signer` / `lms.Verifier`, and removed hardcoded `device="xpu"` in `tests/test_falcon_gpu.py`.

### Not adopted (measured)

- **Barrett reduction for BabyBear NTT**: 1.02× (inside noise; LLVM already strength-reduces constexpr `% Q`).
- **Bitsliced GPU AES**: AES-NI / VAES reaches 101 GB/s across 16P CPU; a predicted 20–40 GB/s GPU port still loses by 2.5–5×.

## [1.0.0] — the pre-0.8 layout is gone (breaking)

0.9.0 gave the primitive layer a floor. This is the removal the deprecation shims had been
promising since 0.8.0: everything that existed only to keep the old `mlkem_triton` name and the
old root module paths alive is deleted, and the package is 1.0.

**Migration** — the old path on the left now raises `ModuleNotFoundError`:

| before 1.0 | now |
|---|---|
| `import mlkem_triton` | `import triton_crypto` |
| `triton_crypto.pipeline` | `triton_crypto.mlkem.pipeline` |
| `triton_crypto.params` | `triton_crypto.mlkem.params` |
| `triton_crypto.keygen` / `.encaps` / `.keyprep` / `.ops` / `.ntt_ref` / `.acvp` / `.gpu_hash` | `triton_crypto.mlkem.*` |
| `triton_crypto.refimpl` / `.sampling_cpu` / `.pipeline_host_hash` | `triton_crypto.mlkem.reference.*` |
| `triton_crypto.kernels.*` | `triton_crypto.mlkem.kernels.*` |
| `triton_crypto.primitives.sha2.sha_thash` | `triton_crypto.slhdsa.thash_sha2.sha_thash` |

Unchanged: `from triton_crypto import Decapsulator, Encapsulator, KeyGenerator,
generate_keypairs, encapsulate, decaps_batch` (the ML-KEM entry points), every scheme subpackage
and every `primitives` import.

**Removing the shims found three imports that had been resolving through them**, which is the
honest argument for doing this at all: `mlkem/reference/pipeline_host_hash.py` imported
`...keyprep`, `...kernels` and `...params` — the deprecated root aliases — rather than its own
sibling modules. They now resolve inside `mlkem`, so a future shuffle of that package cannot be
masked the same way.

**Enforcement, not just deletion.** `tests/test_removed_api.py` asserts every removed path stays
removed (18 module paths, the `sha_thash` attribute, and that `find_packages()` no longer returns
`mlkem_triton`), and the CI wheel check fails if any of them reappears in a wheel. The removal
tests replace `tests/test_compat.py`, which used to assert the opposite.

`pyproject.toml` now ships only `triton_crypto*`, and gained the `[project.urls]` block it had
never had. **891 tests** (the 4 compat assertions were replaced by 36), `ruff check .` and
`ruff format --check .` clean.

The GitHub repository was renamed `mlkem-triton` → `triton-crypto-kernels` to match the
distribution, so every name in the project now agrees: distribution, repository, README title and
package root. GitHub redirects the old URLs, but clone/fetch URLs should be updated.

## [0.9.0] — the primitive layer gets a floor, and the library stops being PQC-only

The 0.8.0 refactor made the four schemes uniform. This release does the same job one layer
down, where the next work actually goes: it gives modular arithmetic one home, gives the
primitive layer its own benchmark table, and adds the first non-PQC primitive.

**`primitives/fields.py` — one definition of "multiply modulo q".** The 32-bit Montgomery
product used to be inlined three times inside `primitives/ntt.py`, and `docs/kernels.md`
recorded why: a `@triton.jit` helper defined inside a factory that imports `tl` locally has no
`tl` in its compiled globals, so no other kernel could call it. The fix is the import, not the
helper: `fields.py` imports `triton.language as tl` at module level, and `mont_mul` is callable
from anywhere. `tests/test_fields_gpu.py` calls it from a *different* module's kernel so the
rule stays pinned. `PrimeField` carries `q`, `qinv`, `R`, `to_mont` / `from_mont`, numpy
`mul`/`pow`/`inv`, and rejects `q >= 2**31` with a reason (Goldilocks-class moduli need a
128-bit product; that path does not exist yet). `mont_mul` returns a canonical `[0, q)` value: REDC's raw output can be negative, the
inlined copies all relied on a following `_norm` to fix the sign, and a new caller (Poseidon2,
X25519) should not have to know that.

**`primitives/ntt/` is a package** (`domain.py` / `kernel.py` / `ref.py`), because a
2^20-point transform is a different algorithm (multi-pass, global memory) that will live
beside the in-register one, not inside it. `NTTDomain` now validates the modulus at
construction. `BABYBEAR_NTT` (q = 2^31−2^27+1) needed **no new kernel**: the `(q, N, ζ, layers)`
parameterisation plus the int64+Montgomery path was already general, and
`tests/test_fields_gpu.py` checks it against a schoolbook convolution. `negacyclic_conv_np`
no longer overflows int64 for q beyond ~2^24 (it reduced per term instead of accumulating
q²-sized products; the old version was wrong for any 31-bit modulus).

**`primitives/chacha20.py`** — RFC 8439, batched, `[B, 32]` keys × `[B, 12]` nonces, counters,
keystream and XOR. Rotation needs `.to(tl.uint32)`; on int32 the right shift is arithmetic and
the result is silently wrong. Vectors: RFC 8439 §2.3.2 and §2.4.2, cross-checked against
OpenSSL 3.5 before being written into the test. It is the first primitive in the library that
is not post-quantum, which is the point: `primitives` is a crypto layer, not a PQC layer.

**`benchmarks/bench_primitive.py`** answers a question the scheme benchmarks cannot: which
*shape* of arithmetic this device is good at. One table, a measured copy bandwidth as the
ceiling, and `bench.py --scheme primitives` dispatches to it. First numbers on the Arc B390
(1 KiB messages, B = 4096): ChaCha20 **14.8 GB/s**, Keccak **183 Mperm/s**, SHA-256
**43 M msg/s**, ML-KEM NTT **16.0 M poly/s**, ML-DSA NTT **5.2 M poly/s** — every ALU-bound
primitive lands at 5–20 % of copy bandwidth.

Also: the package description, README opening and `__init__` docstring no longer say
"post-quantum only"; `docs/kernels.md` gained the ChaCha20 kernel contract and the module-level
`tl` rule; `docs/benchmarks.md` §15 is the new table.

**860 tests** (up from 813), `ruff check .` and `ruff format --check .` clean.

## [0.8.0] — package layout: one subpackage per scheme

No algorithm changed. This is the structural cleanup the four-scheme package needed:
ML-KEM finally has the same shape as ML-DSA, SLH-DSA and LMS/HSS.

- **`triton_crypto/mlkem/`** now holds `params`, `keygen`, `encaps`, `pipeline`, `keyprep`,
  `ops`, `gpu_hash`, `acvp`, `ntt_ref`, the scheme's `kernels/`, and a `reference/` subpackage
  for the kyber-py wrapper, the CPU sampling reference and the host-hashing baseline.
  `triton_crypto.mlkem` exposes `Decapsulator`, `Encapsulator`, `KeyGenerator`,
  `generate_keypairs`, `encapsulate` and `decaps_batch`.
- **Old root modules keep working for one release** (`triton_crypto.pipeline`,
  `.kernels.*`, `.refimpl`, ...) as `DeprecationWarning` aliases pointing at the new modules;
  they will be removed in 1.0. `from triton_crypto import Decapsulator` is unchanged.
- **`sha_thash` moved out of `primitives/sha2.py`** into `slhdsa/thash_sha2.py`: FIPS 205's
  construction is not a generic SHA-2 primitive. `primitives.sha2.sha_thash` remains as a
  deprecated alias.
- `kernels/` is now explicitly ML-KEM's; the scheme-agnostic layer is `primitives/`.
- Dropped `docs/support.md`'s hand-maintained project tree (it had already drifted), and fixed
  the root `__init__` / `mlkem_triton` / `kernels` docstrings.

813 tests, `ruff check .` and `ruff format --check .` clean.

## [0.7.2] — API, packaging and input-contract hardening

Fixes the P0/P1 findings of the internal primitives-quality review.
No algorithm changed; the device math is the same code, now with the contracts it claimed.

**Packaging and API**

- Version metadata (0.3.0) finally matches the changelog; `pyproject.toml` and
  `triton_crypto.__version__` are both 0.7.2.
- `triton_crypto/lms/` now has an `__init__.py` — without it `setuptools.find_packages()` silently
  omitted the whole subpackage from wheels. Exports `Verifier`, `LMSSet`, `get`, `from_name`,
  `ALL_SETS`, `verify_batch`.
- `triton_crypto.mldsa` exports `Signer` / `sign_batch`; `triton_crypto.slhdsa` exports `Verifier` /
  `verify_batch`. The README examples that were broken now run, and a test executes every README
  import line.
- SHA3-224 and SHA3-384 added (`HashFn.hash` and 1-D `HashFn.hash_torch` too).

**ML-KEM**

- `decaps_torch` / `encapsulate_torch` copy non-contiguous inputs and validate dtype, shape,
  device and `B`. A transposed view used to return a wrong key without an error.
- Empty batches return empty arrays instead of raising `RuntimeError: stride(-1)`.
- FIPS 203 §7.2: `Encapsulator` rejects a non-canonical `ek` (any 12-bit coefficient ≥ q);
  `validate=False` is the explicit escape hatch for formally-invalid vectors.
- `KeyGenerator.generate` uses `secrets.token_bytes` (not NumPy PCG64) and accepts `rng=` /
  `seeds=` for deterministic tests.

**Input normalisation**

- One shared `_inputs.as_bytes_list` / `as_batch_bytes`: one `bytes` is one item, 1-D arrays are one
  item, lists are length-checked per item. This fixes SLH-DSA's single-`bytes` messages (they
  returned 1952 `False` values), 1-D signature arrays, LMS single-`bytes` messages (`IndexError`),
  and ML-KEM's `[31, 33]` list being accepted as two 32-byte messages.

**Safety / robustness**

- Generated-source cache is per-user `0700` with `0600` files and `O_EXCL` writes instead of
  predictable names in `/tmp`.
- GEMM column loads are masked (B is not padded); the `_zero_ch.expand` out-of-bounds read in
  signing is gone (`mac_kernel(..., ZERO_CH)` constexpr branch).
- `NTTDomain` validates q (prime, odd, int64-safe), N, layers and ζ; `ntt_torch` refuses values
  that do not fit int32 instead of truncating; `intt_torch` exposes `mont`.
- SHA-2's block count is a runtime kernel argument (one compile per word size) and `compress_np`
  validates `bits` / block length.
- Public `assert`s became `ValueError`/`TypeError`; LMS/SLH malformed signatures return `False`,
  batch-size mismatches raise, and HSS's `N_spk` and embedded public-key lengths are validated.

812 tests, ruff clean. The full suite and `ruff check .` were run on the Arc B390; the wheel now
contains `triton_crypto/lms/` (checked by the `package` CI job).

**Docs**

- Removed the stale Chinese translations in `docs/zh/`: they duplicated README, CHANGELOG,
  CONTRIBUTING and docs/compare, quoted 231-264 tests against today's 812, and linked to files
  that no longer exist. Git history retains them.
- Corrected the README's 375-test count and its "ML-DSA verification only" limitation, refreshed
  the support matrix and the project tree, and applied `ruff format`.

## [0.7.1] — signing to 43 µs (2.3× one core), and mixed-length batches

Building on the speculation work: ML-DSA-44 signing at B=1024 went 56.3 µs → **43.0 µs** (liboqs one
core: 97-100 µs), B=4096 → **33.6 µs**, B=256 → 60.4 µs.

- **Mixed-length messages are signed in one batch when their padding lands in the same number of
  SHAKE blocks.** μ hashes `H(pk) ‖ M′` per row; rows that cross a block boundary need a second
  launch. *(Corrected in 0.7.2: the original version padded every row to the longest block count,
  which absorbed extra all-zero blocks for the shorter rows and produced a wrong μ. Lengths within
  one block — e.g. 10 and 20 — happened to pass, which is why the tests missed it.)*
- **`pointwise_kernel` no longer materialises its operands.** The host repeated `ĉ` and ŝ into full
  `[n·M, 256]` arrays — at 4096 rows that is ~150 MB of traffic per attempt. The kernel now
  broadcasts inside: `i = row // M`, `j = row % M`.
- Signature assembly for the accepted batch happens once in numpy rather than per row.

Also fixed along the way: the attempt pipeline had grown a *duplicate* `hint_kernel`/`_use_hint`
definition when the pointwise kernel was moved (Python silently kept the last one).

360/360 ACVP sigGen byte-exact; 771 tests, ruff clean.

## [0.7.0] — signing reaches parity with one CPU core, by speculating on the rejection loop

ML-DSA-44 signing at B=1024 went from 193 µs to **56.3 µs** per signature (liboqs on one core:
90 µs), i.e. from 1.96× slower to 1.7× faster. At B=4096 it is 37.0 µs (2.5×); 65 and 87 land at
64.0 and 74.8 µs against 131 and 169. Parity is at B ≈ 200.

Two changes did it, and the second is the interesting one:

- `sign_check_kernel` fuses the whole acceptance test — both norm bounds, the hint, its count and
  the `z` bound — into one program per signature. It replaced ~20 elementwise torch ops, which are
  what a tail-sized attempt actually spends its time on. (While writing it, the `z` bound was
  initially left out; ML-DSA-65/87 still verified by luck, ML-DSA-44 caught it immediately.)
- **Speculation.** The k-th attempt of a row depends only on κ + k·ℓ, so several candidate attempts
  can be evaluated in one pipeline pass and resolved afterwards — the row takes the first candidate
  that accepts, and the signature is bit-identical. A batch no longer needs 84 sequential rounds
  (the unluckiest signature's tail); it needs ~8, at the cost of running the pipeline on `spec × n`
  rows, which is free while the batch is launch-bound. `spec=8`, `spec_target=512` was the best of
  the swept settings.

Signatures are now also assembled for the whole accepted batch in numpy instead of one
`bytes.join` per signature.

360/360 ACVP sigGen still byte-exact; 771 tests, ruff clean.

## [0.6.3] — where signing actually wins: the batch size, and a measured dead end

Signing's cost is dominated by a batch's *tail*: at B=1024 the loop runs 84 attempts, 71 of them on
fewer than 64 rows, and those 71 cost ~60 % of the wall clock because an attempt costs ~2 ms of
dispatch regardless of how many rows it carries. The tail exists because the acceptance rate is
~22 %, so the unluckiest signature needs ~80 tries and the loop cannot finish without it.

That reframes the comparison. Bigger batches amortise the tail without changing the API:
ML-DSA-44 goes 466 µs/signature at B=256 → 193 µs at 1024 → **94.6 µs at 4096 → 81.0 µs at 8192**,
against 97.7 µs for liboqs on one core. **The crossover is at B ≈ 3 000-4 000**; above it the device
is 1.1-1.3× faster.

Tried against the tail and rejected, all measured:

- **XPU-graph replay** of the whole attempt (capture verified working on this stack): 229 µs/sig at
  B=1024 versus 177 — the replay is not cheaper than the ~13 launches it replaces.
- slot/queue model: 196-320 µs. Fusing the three pointwise products: 186 µs.

Still open, with real work attached: fuse an attempt into ~6 kernels (ExpandMask's sponge+unpack,
the three decompose/norm passes, w1Encode's sponge), and overlap two half-batches on two streams so
a latency-bound tail hides under a compute-bound main phase. The ceiling is worth stating: even 2×
would only draw level with one CPU core, and the 16-process aggregate (28 µs) is out of reach for
this workload shape.

## [0.6.2] — liboqs rebuilt, and the honest comparison it produces

The liboqs build that earlier baselines were taken against was no longer in the workspace, so it was
rebuilt from 0.16.0 (minimal build: ML-DSA-44/65/87 and the six SHAKE SLH-DSA sets) and every
ML-DSA number re-measured in one session.

**Verification got better, not worse.** Calling the per-algorithm entry points
(`OQS_SIG_ml_dsa_44_verify`) instead of the generic `OQS_SIG_verify` dispatch is ~3× faster on the
CPU side: 19.6 / 31.6 / 47.3 µs instead of 58.7 / 93.0 / 136.1 µs for 44/65/87. Against the freshly
measured device numbers (1.8 / 2.4 / 3.5 µs at B=4096) the speed-up is **10.9× / 13.2× / 13.5× over
one thread**, and 4.4× / 4.4× / 2.3× over 16 concurrent processes. The earlier table quoted the
generic-API baseline; the README and docs/benchmarks.md now say which one they use and why the two
differ.

**Signing is a loss, and it is now stated as one.** liboqs signs ML-DSA-44 in 97.7 µs on one core
(65: 137.6 µs, 87: 181.3 µs, min of five rounds of 100), against 191 / 199 / 209 µs on the device at
B=1024 — i.e. 1.96× / 1.45× / 1.15× *slower*, and 6.8× / 7.1× / 7.5× behind the 16-process aggregate (28.2 / 28.0 / 27.7 µs/sig).
The rejection loop makes a signature a chain of ~4.5 short sequential attempts, which is the shape
the GPU is worst at. The previous entry's "≈28× over the Python reference" was true and misleading:
the reference is an oracle, not a competitor.

The SLH-DSA baselines were checked against the same fresh build and reproduce within ~10 % — the
fresh numbers are 1 473 / 501 / 2 080 / 2 151 µs for 128f / 128s / 192f / 256f against the documented
1 333 / 458 / 1 970 / 2 040, so with the device's 47.8 / 18.7 / 67.4 / 71.2 µs the speed-up is
**30.8× / 26.8× / 30.9× / 30.2×** (slightly better than the 24-29× those tables quote).

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
[comparison](docs/compare.md). The original Chinese versions were later removed from the tree
(Git history retains them); the English versions are authoritative. The unused
`data/acvp/internalProjection.json` (1.4 MB, never read by any code path) was dropped.
