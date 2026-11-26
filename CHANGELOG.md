# Changelog

## [1.1.0] — four primitives, one scheme, and the column that decides whether to port

What the primitive layer learned overnight: the CPU rival matters more than the device. Every
row below is measured in the same window against `benchmarks/cpu_baseline.json`, and the
`vs 16P` column is now part of the table.


### Poseidon2-BabyBear/16, and the first `vs CPU` column in the primitive table

The primitive layer gains the operator ZK systems actually use, and the benchmark table gains
the column that was missing: what the same work costs on the CPU **in the same window**.

- **`primitives/poseidon2.py`** — Poseidon2 over BabyBear, width 16, 8 full + 13 partial
  rounds, Plonky3's `default_babybear_poseidon2_16` parameters verbatim (round constants from
  the Grain LFSR, `MDSMat4`, internal diagonal `[-2, 1, 2, 1/2, 3, 4, -1/2, -3, -4, 1/2^8,
  1/4, 1/8, 1/2^27, -1/2^8, -1/16, -1/2^27]`). numpy reference plus one Triton kernel;
  `poseidon2_permute_np` / `poseidon2_permute_torch`.
- **`benchmarks/poseidon2_ref/`** — a small cargo project on `p3-baby-bear` 0.8 that emits the
  known-answer vectors (`kat`) and runs the *tuned* CPU rival (`bench-packed`, monty-31x8
  AVX2). `data/vectors/poseidon2/bb16.json` is its output, so the Triton kernel is checked
  against a different implementation rather than against itself.
- **Measured** (Arc B390, copy 200–213 GB/s): **38.7 ns/permutation at B = 65536**
  (`BLOCK=128`); 103 ns at B = 4096, where only 32 programs exist. Against Plonky3 on the
  same machine: **2.7× one core (106 ns), 0.18× sixteen processes (7.0 ns)**. The second
  number is the one that belongs in the table, and `docs/benchmarks.md` §15 now says so.
- Two traps are recorded in `docs/kernels.md` because both were hit while writing it: Plonky3
  applies an **external linear layer before the first round**, and the S-box is
  `(x^2)^2 * x^2 * x`, not `(x^2)^2 * x * x`.
- `% Q` on int64 is enough for 31-bit operands (`a*b < 2^62`), so the kernel carries no
  Montgomery representation; `mont_mul` would cost two extra instructions per multiply.
- 18 new tests (Plonky3 vectors, numpy↔device at batches around the block size, singles, an
  `out=` buffer, the extremes).

### Falcon-512 verification, the first primitive that beats 16 processes by more than 2x

`triton_crypto/falcon/` is a scheme subpackage in the shape the 0.8.0 refactor established
(`params`, `ref`, `kernels`, `verify`). Verification only: signing needs the floating-point FFT
and Gaussian sampler that §5.3 of the eval explicitly rules out for a cross-backend code base.

- **The transform needed no new kernel.** `q = 12289, N = 512` runs through the existing
  `NTTDomain` parameterisation; the only new arithmetic is the pointwise product and the
  norm check, two launches in `falcon/kernels.py`.
- **Measured: 0.65 µs/op at B = 4096** (device core), against liboqs 0.16.0 on the same
  machine at 23.7 µs on one thread and 2.18 µs over sixteen processes -- **35x and 3.2x**.
  The transforms are 39% of the core; the rest is one fused kernel.
- **End to end is host-bound at 31 µs/op**: the variable-length signature decoder (24 µs) and
  SHAKE256 + rejection sampling (9 µs) still run on the host. `bench_falcon_verify.py` prints
  both numbers rather than hiding one inside the other, and the two remaining kernels are the
  obvious next step.
- **Two traps, both recorded in `docs/kernels.md`**: Triton's `%` on signed integers is C's
  truncated remainder (not numpy's floored modulo), so `(prod - c) % q` leaves negatives
  negative; and an `expand`ed tensor is not contiguous, so a kernel that assumes contiguity
  reads past the first row.
- **Tests without liboqs** use one embedded signature from liboqs plus the numpy reference;
  with liboqs built (`build-falcon`), 64 signatures and 64 tampered variants are cross-checked
  against `pqclean`'s own decoder, including a mixed batch where one bad row must not disturb
  its neighbours.

### `primitives/gemm.py`: the generic half of the GEMM, and 16% of peak

The 0.8.0 review (P1-2) pointed out that `docs/kernels.md` called `_gemm_modq_kernel` "generic"
while it lived in ML-KEM's package. The generic piece is now extracted rather than moved:
`primitives/gemm.py` is a bare exact `int8 x int8 -> int32` matmul (no modulus, no epilogue),
and `mlkem/kernels/gemm_modq.py` stays where its `Q = 3329` semantics belong.

Measured 1.94 TMAC/s at n = 1024 and **3.96 TMAC/s at n = 2048** (256 is launch-bound at 0.18).
That is **16% of the ~25 TMAC/s int8 peak** with ML-KEM's untuned tile table, which matters for
`PLAN_ffgemm.md`: the bit-plane strategy pays 8x, so ~4 TMAC/s of int8 becomes ~0.5 TMAC/s of
GF(2^8) MACs, not the 6 T MAC/s the eval assumed. `tests/test_gemm_int8.py` checks the
arithmetic end to end -- 64 tiny int8 GEMMs XOR-ed through the AES reduction masks against a
schoolbook GF(2^8) product.

### AES-128-CTR through a T-table, and a prediction that came out 12x wrong

`primitives/aes.py` is the repository's measurement of what a table-driven cipher costs on a
GPU: ten rounds x four words x four lookups = 160 gathers per block, all into a 4.25 KiB table.

- **Measured 10.68 GB/s** at B = 65536, i.e. **0.77x of one AES-NI core** and 0.06x of sixteen
  processes. `eval-triton-crypto-primitives.md` §4.7 predicted 0.7-0.85 GB/s by extrapolating
  the probe's random-gather rate; the same cipher is **12x faster** than that. The gather rate
  turned out to be a property of the table's residency and access pattern, not of the
  instruction: a hot 4.25 KiB table with 64 blocks per lane is issue-bound L1 traffic, not the
  probe's cold divergent gather.
- The conclusion survives anyway -- 0.77x of one core is still not worth 2-3 weeks of
  bitslicing, and the same 12x uncertainty applies to that prediction -- but the *reason* has
  changed, and `docs/benchmarks.md` §18 says so.
- Two conventions cost real time and are recorded in `docs/kernels.md`: `Te0[x] = [2s, s, s,
  3s]` (not `[s, 2s, 4s, 8s]`), and `Rcon` is the round word's **top** byte (`rcon << 24`).
  Both produce plausible ciphertext until a known vector is checked.
- The host key schedule is hoisted out of the benchmark: 4096 numpy key expansions cost 340 ms,
  50x the kernel for a 4 MiB batch. `aes128_ctr_torch` takes a precomputed `rk` for that reason.
- FIPS-197 C.1, the all-zero vector, and `openssl enc -aes-128-ctr` are the three oracles; all
  three are in `tests/test_aes*.py`.

### A 7x-wrong CPU rival, caught by the snapshot, and a guard so it cannot recur

A full `benchmarks/snapshot.py` run reported Poseidon2 at **1.25x sixteen CPU processes** where
every earlier window said 0.19x. The device had not changed; the *rival* had. `benchmarks/
poseidon2_ref` had been rebuilt without `RUSTFLAGS="-C target-cpu=native"`, so Plonky3's
`Packing::WIDTH` fell back to 1 and the binary measured the **scalar** path: 736 ns per
permutation instead of 105, a rival 7x too slow, which the table then reported as a device win.

`bench_cpu.py` now reads the `lanes` field that binary already prints and refuses to report a
number below 4, with the rebuild command in the error. The corrected rival puts the row back at
0.18x.

Two lessons, both already in the docs and both worth repeating: a ratio is only as good as both
of its sides, and a benchmark harness that cannot detect a degraded *baseline* will confidently
publish the wrong winner.

### The 2^20 NTT reaches the memory roofline: 10.8 -> 1.0 ms in four steps

Folding the negacyclic pre- and post-twiddles into the first and last passes (they are
index-dependent only, so the stage that already touches every element can apply them) removed one
launch and 8 MB of traffic per direction, and removed the inverse's separate `x_j *= post_j` pass.

**2^20 forward: 0.98 ms at the top clock, 1.8 ms at the bottom**, twelve alternating runs in one
process. Against the CPU rival: **70x one core and 5.3x sixteen processes** at the fast end, 39x
and 2.9x at the slow one. Traffic is 152 MB, so 0.98 ms is 155 GB/s -- **72% of copy bandwidth**,
i.e. essentially the 0.71 ms memory roofline. The inverse is the same (0.99 ms).

The full arc is 10.81 -> 1.0 ms in four changes, each recorded in `docs/benchmarks.md` §22.1: stop
rebuilding the twiddle tables per call (-150x), replace an int64 division in the index math with a
shift (-30%), pick tile 1024 / 8 warps (-2x), and fold the twiddles into the end passes (-2x).

Note the regime change: at the low clock the transform is latency-bound and halving traffic
(radix-4) buys 10-20%; at the high clock it is bandwidth-bound and at the roofline. A tuning
decision made in one regime can be wrong in the other.

### Radix-4 for the 2^20 NTT: half the traffic, 10-20% of the time

`LargeNTT(n, q, radix=4)` fuses two radix-2 stages per pass. Bit-identical to the radix-2 result
(verified 2^4..2^20, including the odd-`log2` case with a leftover radix-2 pass), and paired in
one window: 1.74 ms against 1.88 (min), 2.64 against 3.18 (median), for 80 MB of traffic instead
of 160.

**Halving the traffic buys 10-20%, not 50%** -- which corrects the framing of the previous
entry: the transform's cost is *dominated* by traffic but it is not *bandwidth-saturated* (76 of
216 GB/s), and the limit is pass latency plus the modular multiplies. It also corrects a reading
taken across two windows that concluded radix-4 was slower; the paired run says otherwise by
~15%. Radix-4 stays opt-in: `radix=2` is the default because the win is smaller than the
run-to-run spread.

### The 2^20-point NTT gets 5x faster, and becomes the one row that beats sixteen processes

The first correct version took 10.81 ms. It now takes **2.22 ms**: **31.5x one CPU core and
2.37x sixteen processes** -- the only `vs 16P` above 1 in the whole primitive table.

Three fixes, each worth knowing:

- **The pre/post twiddle tables were rebuilt on every call.** A 2^20-element numpy pass costing
  several milliseconds -- more than the transform it fed. Built in log n passes and cached per
  instance and per device.
- **The block/j split did an int64 division per element per pass** (`offs // half`, `half` a
  runtime value, so no strength reduction). `half` is a power of two: it is a shift and a mask
  now. A `(block, tile)` grid was tried first and was *much* worse -- the late stages have
  `half = 1`, so it degenerates into 524288 one-element programs.
- **Tile 1024 with 8 warps**, swept: 3.55 ms at 2048/4, 1.87 ms at 1024/8 for the pass sequence.

The result is the answer to the project's opening question. ALU-bound primitives lose to sixteen
cores (0.06-0.5x); bandwidth-bound ones do not, because sixteen cores cannot match 160 MB of
traffic per useful operation.

### The 2^20-point NTT: the first bandwidth-bound primitive

`primitives/ntt/large.py` — a multi-pass Stockham transform, one launch per stage, for sizes the
in-register kernel cannot hold. 20 passes over 2^20 coefficients, pre-twiddled for the
negacyclic wrap, DIF forward and DIT inverse.

- **10.81 ms** for a 2^20 forward transform, against 69.9 ms for one CPU core and 5.25 ms for
  sixteen concurrent processes: **6.46x and 0.49x**.
- The CPU rival is a C Stockham implementation (`benchmarks/ntt_large_cpu_rival.c`, 16P = sixteen
  processes with `OMP_NUM_THREADS=1`). The first version of it charged the CPU a `powmod` per
  coefficient for the pre-twiddle; the table is now built once, like the device's.
- Correctness chain: agrees with `NTTDomain` (the in-register kernel's numpy twin) at 2^4 and
  2^8, round trip exact at 2^4..2^20, and the pointwise product reproduces a schoolbook
  negacyclic convolution.
- **It is 7x off its own roofline** (160 MB of traffic, 0.74 ms at copy bandwidth, 15.5 GB/s
  achieved). The stages are radix-2 and the later ones have a stride that defeats coalescing;
  radix-4/8 plus a blocked layout is the next 2-3 weeks, and it is the strongest candidate for a
  result that beats sixteen processes.
- Also here: the pre/post twiddle tables are built in log n numpy passes and cached per device.
  A million-iteration Python loop was costing 1686 ms per call -- 150x the transform.

### BLAKE3 lands on the fourth attempt, and the bug was one line of undefined behaviour

`primitives/blake3.py`: 7-round compression, 1 KiB chunks, a level-by-level tree — one launch per
block index plus one per level, with the chaining value in a `[B * chunks, 8]` buffer.

**Measured 10.1 GB/s** at 1 KiB messages (flat to 16 KiB), against the official Rust
implementation at 2.93 GB/s on one thread and 51.9 GB/s over sixteen processes: **3.45x one
core, 0.19x sixteen processes**.

The interesting part is why it took four sessions. The kernel "stored zeros": the store mask,
block count, flags (11 = CHUNK_START|CHUNK_END|ROOT) and block length all read back correct from
inside the same launch, and every piece worked in isolation — the compression, a single block
step through a global buffer, a sixteen-launch chain through that buffer. The cause was

    v13 = counter >> 32      # uint32, shifted by its own width: UB

LLVM may do anything with that; this backend chose "the kernel computes zero" and printed a
`UserWarning` that a test log filters out. The counter is now an explicit `(low, high)` pair, and
the regression guard is a multi-chunk vector — any message over 1 KiB has a non-zero chunk
counter, which no 64-byte known-answer test can see. It is the sixth entry in the Triton gap
table and the only one that produced plausible output instead of an error.

Also: `benchmarks/blake3_ref` supplies both the 26 official vectors and the CPU rival;
`benchmarks/bench_cpu.py` grew a BLAKE3 row (2.93 GB/s 1T, 51.9 GB/s 16P, min-of-3 — `max`
biased the aggregate by 30%).

### The wheel is verified end to end, not just built

`pip wheel` succeeding proves nothing about whether the installed package works. A smoke run from
a foreign working directory, against the *installed* 1.1.0 wheel, now checks: every subpackage
imports (four schemes, nine primitives), Poseidon2 reproduces a Plonky3 vector, Poly1305 matches
RFC 8439, AES matches FIPS-197, the int8 GEMM is exact, and Falcon verifies the embedded signature
and rejects a one-bit-tampered copy — all from `site-packages`, not the checkout. The wheel was
uninstalled afterwards; nothing about the venv's previous state changed.

The check is not in CI (it needs a GPU for the device paths) but it is the reason the 1.1.0 wheel
is claimed to work rather than merely to build.

### `HashToPoint` moves to the device: Falcon's end-to-end drops 31.3 -> 24.4 us

`comp_decode` and `HashToPoint` were the two host stages; a probe measured both halves of the
second one before any kernel was written (`probes/hashpoint-shape-probe.py`): the sponge costs
0.10 us/signature and the rejection-plus-prefix-sum filter 0.47 us, against 9.3 us for
`hashlib` + numpy on the host. **16x**, so it was written.

- `falcon/kernels.py` gains `_hash_to_point_kernel`: a contiguous `[BLK, 720]` tile of 16-bit
  samples, an exclusive prefix sum over the keep-mask to give each survivor its destination, and
  a scatter of the first `n` in order. Measured 0.47 us per signature at B = 4096.
- `falcon/verify.py` gains `hash_to_point_batch_torch` and uses it from `Verifier.verify` when
  the batch has one message length (the sponge absorbs a fixed number of blocks per row); mixed
  lengths fall back to the host, and a test pins that the fallback is taken rather than
  mis-hashing.
- End to end: **24.4 us/op** at B = 4096, of which 96% is now `comp_decode` on the host. The
  device core is unchanged at 0.60 us.

The contrast with `comp_decode` -- which lost by 35x on the same data -- is now recorded in
`docs/patterns.md`: a contiguous tile with a prefix sum and a scatter is fast; the same number of
*gathered bytes* at data-dependent addresses is not. Same problem size, opposite verdict.

### The Poseidon2 vectors grew from 9 to 64

The Plonky3 `kat` mode emitted nine cases: the 0..15 ramp, seven offsets chosen to stress the
reduction paths, and all-`q-1`. Nine vectors can all pass while a bug sits in the middle of the
field -- they exercise the edges by design. Fifty-five deterministic pseudo-random states (an
LCG in the generator, so the JSON stays reproducible) now sit alongside them, and
`tests/test_poseidon2.py` checks every one. Regenerate with
`cargo run --release -- kat > data/vectors/poseidon2/bb16.json`; the README in that directory
also records the `RUSTFLAGS="-C target-cpu=native"` requirement, without which the "packed" CPU
rival silently measures the scalar path.

### Two more independent oracles: Poly1305 and AES against OpenSSL

The known-answer vectors each primitive already had were one point each. Two new test files
extend the cross-check to the inputs those vectors cannot reach, against implementations that
share no code with this one:

- `tests/test_poly1305_openssl.py` binds OpenSSL's `EVP_MAC("POLY1305")` with ctypes and checks
  23 block-boundary lengths (0, 1, 15/16/17, 31/32/33, ..., 1024/1025, 4096) plus 40 random
  (key, length) pairs, plus the clamp property (an all-ones `r` must behave as the RFC mask).
- `tests/test_aes_openssl.py` shells out to `openssl enc`: 40 random blocks through ECB, CTR
  streams at six lengths, and a stream started at `counter0 = 2^32-1` so the 64-bit counter's low
  half crosses the boundary -- the one arithmetic detail the T-table kernel could get wrong
  without any single KAT noticing.

Both skip when the tool is missing; neither adds a runtime dependency. **1084 tests.**

### Verification pass: a fresh clone, isolated test files, a hundred repeated transforms

Three checks that the tree is self-contained and stable, all run before freezing it overnight:

- **a fresh `git clone` runs the new tests** (103 passed) and, checked explicitly, imports the
  clone's own `triton_crypto`, `primitives/blake3.py` and `ntt.large` — nothing needed by the test
  suite lives outside the repository;
- **every device test file passes on its own** (blake3, ntt_large, falcon, poly1305, aes,
  poseidon2, gemm), so none of them depends on another test having warmed a kernel first;
- **a hundred repeated 2^16 transforms** stay bit-identical with device memory flat at 1.1 MB
  allocated / 2.1 MB reserved from the first iteration to the last.

None of this changed a line of code; it is the evidence that the 1125-test tree is what a
reviewer would get, not just what happens to work in this session's state.

### `docs/patterns.md`: the table the project was actually for

The roadmap asked for ten *computational patterns*, not ten algorithms, on the theory that the
by-product would be a table saying which patterns are easy and which are hard. That table now
exists as a document: every pattern with its representative primitive, its status, its measured
device throughput against the 16-process CPU baseline, what it cost in Triton, and what stopped
the ones that did not land.

Its first section is the rule that the whole batch of measurements converged on: **whether a
pattern is worth porting is decided by whether the CPU has a hardware instruction for it.** The
device's ALU-bound throughput barely moves between rows (10-31 GB/s); the 100x spread in the
`vs 16P` column comes entirely from the CPU side -- SHA3 4.0x (no Keccak instruction), SHA-256
1.58x, ChaCha20 0.49x, Poly1305 0.15x, AES 0.06x (AES-NI).

Also collects the six Triton traps (signed `%` is truncated, `h + x & M` masks the sum,
`expand` is not contiguous, a `tl.constexpr` cannot take a loop variable, a runtime loop cannot
carry a tuple, and `tl.sort` is silently wrong on XPU 3.9) and the two shapes that did not land,
with what stopped each.

### `benchmarks/snapshot.py`: the measurement discipline as the default

Every number in these docs is supposed to come from one window (§1: "same process, same second,
quote ratios not absolutes"), and the reports were still assembled by hand from separate runs.
`snapshot.py` runs the CPU rivals, the primitive table and the four scheme benchmarks in one
window and writes one timestamped markdown plus a JSON header (git revision, per-section
command, return code, wall time, whether the CPU baseline was reused or measured).

It shells out to the existing scripts on purpose: each one builds its own liboqs/ACVP corpus
and prints its own copy-bandwidth header, and duplicating that would mean a second set of
conventions to keep in sync. Snapshots are gitignored — artifacts of a window, not sources.

### The CPU rival table is complete: Poly1305 and SHAKE

`benchmarks/bench_cpu.py` grew three rivals, so the primitive table now has a `vs 1T` / `vs 16P`
column for every row except the bare Keccak permutation and the NTTs (no OpenSSL equivalent;
numpy would be 10-100x off, and a wrong rival is worse than a missing one).

- **Poly1305** through OpenSSL's `EVP_MAC` (the CLI has no `speed` entry for a MAC):
  **7.54 GB/s one thread, 75.8 GB/s over 16 processes**. It is timed on **1 MiB per MAC** on
  purpose -- the one-shot API costs ~2.5 us of init/final per call, so a 1 KiB corpus measured
  0.39 GB/s, i.e. the API and not the cipher. Against 75.8 GB/s the device's 11.55 GB/s is
  **0.15x**, which is the honest number the docs quote.
- **SHAKE128 / SHAKE256**: 0.92 / 0.74 GB/s one thread. The device's 37.8 / 31.5 GB/s is
  **3.9x** sixteen processes -- the sponge family keeps winning, for the same reason SHA-3
  does: x86 has no Keccak instruction.

### Poly1305 and ChaCha20-Poly1305

The primitive layer's last cheap pattern: a 130-bit modular MAC, and the AEAD that composes it
with ChaCha20.

- **`primitives/poly1305.py`** — Poly1305 in the canonical 5 x 26-bit limb layout (int64
  carries every product; no Montgomery form, no `umulhi` pairs), a Python-integer reference
  that handles any length, and `chacha20_poly1305_torch` / `_open_torch` for the AEAD with AAD
  and arbitrary message lengths.
- **RFC 8439 both ways**: the §2.5.2 MAC vector and the §2.8.2 AEAD vector (114-byte
  plaintext, 12-byte AAD) are reproduced byte for byte, and `openssl mac ... POLY1305` was
  used to confirm the reference before the vector was trusted.
- **Measured**: Poly1305 **11.37 GB/s** and the AEAD **4.50 GB/s** at B = 65536 (1 KiB
  messages) — 2.8x one core, **0.44x of sixteen processes**. Coverage, not a speed story, and
  the docs say so.
- **Three bugs the vectors caught**, all recorded in `docs/kernels.md`: `2^128` added twice for
  messages whose length is not a multiple of 16; `h + limb & MASK` masking the whole
  accumulator because Python binds `+` tighter than `&`; and a key stride of 16 instead of 32,
  which works for lane 0 and for a batch of one.
- **One host round trip cost half the throughput**: the first AEAD version handed the MAC host
  buffers (4 MB per call at B = 4096), 2.10 GB/s against 4.04 GB/s with the tensors left on the
  device.

### The CPU rival, as a script and as two columns

`benchmarks/bench_cpu.py` measures the work the primitive table is compared against, on the
same machine and in the same session, and writes `benchmarks/cpu_baseline.json` with the
metadata needed to know when the number is stale (host, governor, current/max frequency,
liboqs builds, OpenSSL version). Twelve rivals: OpenSSL for ChaCha20 / AEAD / AES-CTR /
SHA-2 / SHA-3, liboqs for ML-DSA / Falcon / SLH-DSA verification, and Plonky3 for Poseidon2.

`bench_primitive.py` reads it and adds **`≈copy`** (one operation expressed as bytes of copy
bandwidth — the only column that does not move with the iGPU clock state), **`vs 1T`** and
**`vs 16P`**. Rows without a CPU reference print `n/a` rather than vanishing. Two conventions
are enforced in code: 16P is sixteen concurrent *processes*, and each child reports its own
measured rate (summing rates, not bytes over a wall clock that includes process start-up —
the first version of this over-reported by 2x).

The numbers change what "worth doing" means. AES-128-CTR is **166 GB/s over 16P** because of
AES-NI, so a bitsliced GPU AES at a predicted 20–40 GB/s is 0.12–0.23x; SHA3-256 is
**0.72 GB/s per core** because x86 has no Keccak instruction, so the device's 31.6 GB/s is
**4.0x the whole 16-process CPU**. Whether the CPU has hardware support decides the outcome
far more reliably than any property of the device.

### A contract test for the primitive layer

`tests/test_primitives_contract.py` (owed since the 0.8.0 review, P1-1) turns "what counts as
a finished primitive" into a red light: exported from `primitives` and listed in `__all__`,
a numpy reference, a device path (or a declared reason in `HOST_ONLY`), a test file or shared
coverage, a kernel contract in `docs/kernels.md`, and a case in `bench_primitive.py`. Adding a
module without one of those fails.

**946 tests** (up from 891), `ruff check .` and `ruff format --check .` clean.

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

Fixes the P0/P1 findings of the internal primitives-quality review (`eval-primitives-quality.md`).
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
