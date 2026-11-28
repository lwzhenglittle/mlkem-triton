# What this GPU is good at, pattern by pattern

The roadmap this project started from (`Triton 密码学原语：候选算法与路线.md`) asked for ten
*computational patterns* rather than ten algorithms, on the theory that each one probes a
different capability of Triton and of the hardware — bit operations, table lookups, multi-limb
carries, modular arithmetic, the matrix unit, irregular memory access — and that the by-product
would be a table saying which patterns are easy and which are hard.

This is that table, as of 1.1.0. Every device number is from `benchmarks/snapshot.py` (one
window, `benchmarks/snapshot-20261128T063642Z.md`) and `benchmarks/local-gpu-vs-cpu-20261128.md`;
every CPU number is from `benchmarks/cpu_baseline_barrier.json` (OpenSSL / liboqs / Plonky3 /
exact CPU GEMM, same machine, 16 concurrent processes measured with a barrier at matched 1 KiB
message sizes). Sections in `docs/benchmarks.md` are cited where the detail lives.

## 0. The rule, corrected: it is three factors, not one

The first version of this table said **"whether a pattern is worth porting is decided by whether
the CPU has hardware support for it"**. That is wrong, and this project's own numbers refute it in
both directions:

- **SHA-256 has SHA-NI and the device still wins 2.80x.** A dedicated instruction does not by
  itself beat 96 EUs of width.
- **BLAKE3 has no CPU instruction at all and the device loses 0.74x.** No instruction does not
  mean the CPU is slow -- its AVX-512 implementation does 16 blocks at once.

The rule that does describe all nine measured rows: **the CPU's best implementation decides it, and
that is set by three factors together.**

1. **A dedicated instruction** (AES-NI, PCLMUL, SHA-NI). Necessary to lose, not sufficient.
2. **How wide a SIMD the shape admits** -- BLAKE3 16 ways, ChaCha20 8-16, Poly1305's 26-bit limbs
   vectorise cleanly, versus Keccak's 4 usable lanes.
3. **How long the serial dependency chain is** -- Keccak's 24 rounds and SHA-2's chain leave a core
   latency-bound, so the CPU can only scale by adding cores; the GPU hides the latency with threads.

| primitive | device | CPU 1T | CPU 16P (barrier) | vs 16P | instruction | SIMD width | chain |
|---|---:|---:|---:|---:|---|---|---|
| SHA3-256 | 30.9 GB/s | 0.62 GB/s | 6.32 GB/s | **4.89x** | none | 4 lanes | 24 rounds |
| SHA-256 | 78.6 GB/s | 3.14 GB/s | 32.3 GB/s | **2.43x** | **SHA-NI** | serial | 64 rounds |
| SHA-512 | 44.1 GB/s | 0.90 GB/s | 6.96 GB/s | **6.34x** | none here | serial | 80 rounds |
| ChaCha20 | 13.0 GB/s | 4.28 GB/s | 23.3 GB/s | 0.56x | none | 8-16 lanes | short |
| Poly1305 | 11.8 GB/s | 5.92 GB/s | 39.4 GB/s | 0.30x | **PCLMUL** | 26-bit limbs | short |
| AES-128-CTR | 9.4 GB/s | 11.8 GB/s | 101 GB/s | 0.09x | **AES-NI** | VAES 8-16 | short |
| BLAKE3 | 9.9 GB/s | 1.35 GB/s | 15.9 GB/s | 0.62x | none | **16 lanes** | short |
| Poseidon2 | 3.2 GB/s | 9.31M/s | 46.0M/s | 0.54x | none | Plonky3 asm | long |
| 2^20 NTT | 1.61 ms (0.85-1.64 ms, radix-4) | 19.1/s | 188/s | **3.30x** | none | memory-bound | n/a |
| exact int8 GEMM | 3.38 TMAC/s (1024^3) | 66.5 GMAC/s | 409 GMAC/s | **8.28x** | FMA/AVX2 | matrix unit | n/a |

(Device and CPU columns are from `benchmarks/snapshot-20261128T063642Z.md` and
`benchmarks/cpu_baseline_barrier.json`; across clock states, the 2^20 NTT spans `3.3–7.3x`.
Poseidon2, the NTT and GEMM are counted in permutations, transforms and MACs, not bytes.
See `docs/benchmarks.md` §1.2 for the companion Host→Device→Host E2E column — notably `sha256`
flips from `2.51x` device-only to `0.48x` end-to-end when paying a single 64 MiB H2D copy.)

**The device's ALU-bound throughput barely varies** (10-31 GB/s across every row). The 100x spread
in the last column is entirely the CPU side -- which is why "can the GPU do this pattern" and "is
the GPU worth it" have different answers.

Two corollaries that still hold:

- **A single iGPU core-equivalent is not the competition.** Against one CPU thread every port looks
  3-10x better. A CPU baseline taken during a build (or under a throttled `powersave` governor at
  400–900 MHz) reads 3–4.6× slower than the same code measured idle under `performance`, which is
  why `bench_16p_barrier.py` now refuses to write baselines from a throttled power state
  (`docs/benchmarks.md` §5).
- **The interesting GPU shapes are the ones the CPU cannot vectorise**: batched independent small
  objects (signature verification), hot small tables, and the matrix unit.

## 1. The patterns

`✅` device path measured; `⚠️` reference only; `❌` not attempted. Numbers are device vs the
16-process CPU baseline unless stated.

| # | pattern | representative | status | device | vs 1T | vs 16P | what it cost in Triton |
|---|---|---|---|---|---|---|---|
| 1 | ARX (add-rotate-xor) | ChaCha20 | ✅ | 14.2 GB/s | 3.36x | 0.65x | easy; the only trap is that `>>` on int32 is arithmetic, so rotate needs `.to(tl.uint32)` |
| 2 | tree hash | BLAKE3 | ✅ | 9.7 GB/s | 7.44x | 0.74x | one launch per block index + one per level; the chaining value lives in a buffer because a runtime loop cannot carry a tuple. See §2 for the four sessions it took |
| 3 | sponge | SHA3/SHAKE | ✅ | 31.4 GB/s | 51.1x | **5.15x** | easy and a winner: one lane holds 25 uint64 and the round loop must be *rolled*, not unrolled (unrolled: `spill_size=3584`, rolled: 0 and 30% faster) |
| 4 | lookup vs bitslice | AES-CTR T-table | ✅ | 10.6 GB/s | 0.90x | 0.11x | 160 gathers/block into a 4.25 KiB hot table; bitslice (2-3 weeks) not attempted |
| 5 | multi-limb big integer | Poly1305 | ✅ | 11.9 GB/s | 2.11x | 0.32x | 5x26-bit limbs fit int64 exactly; no `umulhi` pairs needed. X25519 not attempted |
| 6 | small-prime NTT | ML-KEM/ML-DSA/BabyBear NTT | ✅ | 23.8 / 7.2 / 7.2 M poly/s (N<=1024); **2^20 in 0.85-1.64 ms** | **32-57x** | **3.8-7.3x** | the `(q, N, ζ, layers)` parameterisation covered BabyBear and Falcon with no new kernel; 2^20 needed a multi-pass one (`ntt/large.py`) and is the only bandwidth-bound primitive here |
| 7 | algebraic hash | Poseidon2-BabyBear/16 | ✅ | 38.7 ns/perm | 2.78x | 0.58x | 16 int64 state words per lane, one program per 128 permutations; state as int32 + widen per multiply is *slower* |
| 8 | finite-field GEMM | exact int8 GEMM (UOV/RS substrate) | ✅ | 1.94 TMAC/s (n=1024), 3.96 (n=2048) | — | — | 16% of the int8 peak with an untuned tile table; the GF(2^8) product itself is verified via bit planes |
| 9 | floating-point FFT | Falcon *signing* | ❌ | — | — | — | deliberately not attempted: Triton does not expose FMA fusion, so a cross-backend bit-exact signature is not claimable. fp64 itself works and is bit-exact elementwise |
| 10 | scatter + atomics | MSM / Pippenger | ❌ | — | — | — | 32/64-bit atomics are correct; the gather penalty and the 381-bit arithmetic (900 ops per modular multiply) make it a 1-2 month item that ICICLE already owns |
| — | memory-hard | Argon2id | ❌ | — | — | — | the only load that would show an iGPU-vs-CPU memory-subsystem difference; not attempted |

### Where the device does win, and by how much

| result | number | why it wins |
|---|---:|---|
| SHA3-256 / SHAKE128 / SHAKE256 | **5.15-8.10x** | no Keccak instruction, 4 usable lanes, 24 dependent rounds |
| SHA-256 | **2.80x** | SHA-NI is real but slower than the device's rolled schedule |
| Falcon-512 / Falcon-1024 verification | **8.3x / 4.4x dev (2.9x / 2.3x e2e)** | batched NTT + transposed bit-reservoir decoder + GPU `HashToPoint` beats 16P CPU end-to-end |
| SHA3-512 / SHA-512 | **7.73x / 6.94x** | same as SHA3-256 (wide lanes) |
| ML-DSA-44 verification | **1.6x** | batched NTT + sponge beats 16 barrier-measured liboqs processes |

Everything else is parity at best. That is the honest summary of porting a primitive library to
an integrated GPU with a 16-core hybrid CPU beside it.

## 2. The patterns that did not land, and what stopped them

- **Tree hash (BLAKE3) — landed on the fourth attempt.** The shape is one launch per block
  index plus one per tree level. What blocked it was not the tree: `counter >> 32` on a uint32 is
  undefined behaviour, the Xe backend folded the whole kernel to zero, and every input read back
  correct from inside the same launch. Four sessions of bisection: the compression, one block
  step through a global buffer and a sixteen-launch chain all worked in isolation; the one line
  that differed was the shift. See `docs/benchmarks.md` §21.
- **Falcon signature decoding on the device — failed first, then landed 168×–232× faster than host.**
  The first attempt did 3-byte uncoalesced gathers plus global CLZ lookup tables per coefficient
  and measured **2.6 ms/signature** (35× slower than the early 75 µs host decoder). Redesigning
  the access pattern fixed it completely: `_pack_sig_words_kernel` transposes variable-length
  bodies into `[W, B]` big-endian `uint32` words, and `_comp_decode_kernel` keeps a 64-bit
  register bit reservoir per lane (refilling one coalesced word only ~32% of steps) plus a
  4-stage branchless 16-bit CLZ with zero lookup tables. Result: **0.086–0.142 µs/signature** on
  device, flipping Falcon-512 end-to-end from `15.35 µs` (`0.22×` 16P) to **`0.90–0.92 µs`**
  (**`2.9×` 16P**) and Falcon-1024 to **`2.18–2.22 µs`** (**`2.3–2.4×` 16P**; see
  `docs/benchmarks.md` §3.5). Same algorithm, opposite verdict — the access pattern decides
  everything.
- **Large-N NTT (2^20) — landed and tuned, and it is the one row that wins.** `ntt/large.py` is
  correct (agrees with `NTTDomain` where both fit, round trip exact) and takes **1.0-1.8 ms:
  70x one CPU core and 5.3x sixteen processes** at the fast clock. Four fixes took it from 10.8 ms
  (a per-call table rebuild, an int64 division in the index math, the tile/warp choice, and folding
  the negacyclic twiddles into the first and last passes -- `docs/benchmarks.md` §22.1). At the fast
  clock it runs at **45%** of copy bandwidth (98 GB/s of 217), not at the memory roofline -- the
  roofline is 0.39 ms and the transform takes 0.85 ms, and §22.2 shows the remaining gap is pass
  latency plus the modular multiplies.

  This is the answer to the question the project started with. Every ALU-bound primitive loses
  to sixteen cores (0.11-0.74x, with ChaCha20-Poly1305 near parity at 0.90-1.00x); the sponge/SHA-2
  family wins by 2.8-8.1x because **Keccak is bad on all three CPU-side factors** (no instruction,
  four usable lanes, 24 dependent rounds) -- not because of the instruction alone (SHA-256 has
  SHA-NI and still wins 2.80x). The operator that wins for a memory reason is the one whose
  **working set cannot fit in registers**: 84 MB of traffic per useful call is something sixteen
  cores cannot match, because each of them has to move the same bytes through the same memory system.

## 3. Five Triton traps found the hard way

All of them are recorded at the site where it bit. The four that are claims about the *toolchain*
rather than about this code -- the truncated `%`, the non-contiguous `expand`, the tuple in a
runtime loop, and the 32-bit shift by its own width -- are now checked against the installed backend
by `tests/test_triton_behaviour.py`, because a claim like that goes stale when the wheel changes and
nothing would notice. (The `tl.sort` entry in that file is narrowed: 1-D and last-axis sorts are
correct on this build, and `axis=0` raises a `CompilationError` instead of returning a permutation.
The 32-bit shift test skips itself when the kernel comes from the compile cache, since the warning
that is the only evidence for it is emitted at compile time and never again.)

1. **Signed `%` is C's truncated remainder**, not numpy's floored modulo. `(a - b) % q` leaves
   negative differences negative; Falcon's norm check was wrong for half the coefficients.
2. **`h + limb & MASK` masks the sum, not the limb.** Python binds `+` tighter than `&`, so a
   Poly1305 accumulator was truncated to 26 bits after every block. The degenerate keys
   (`r=1, s=0`) passed either way.
3. **`expand`ed tensors are not contiguous.** A kernel that indexes them linearly reads past the
   first row (Falcon's pointwise product).
4. **Triton rejects starred unpacking** (`(*t, v)` is "unsupported AST node type: Starred"), so
   tuple accumulation in a jit body must be `t = t + (v,)` and only works under `tl.static_range`.
   The companion claim -- that a `tl.constexpr` cannot receive a loop variable -- was recorded here
   and **does not reproduce**: `j: tl.constexpr = i` inside a runtime loop produced the correct
   result, identical to passing a runtime argument. The kernel keeps the runtime argument because
   it is unambiguously correct, not because the alternative is known to fail.
5. **A runtime loop cannot carry a tuple assignment.** Any state that has to survive a loop
   needs named scalars — and unrolling to keep the tuple is not free (trap 6).
6. **"It compiles" is not evidence.** An earlier session recorded `tl.sort` as *silently wrong* on
   XPU 3.9 (a permutation rather than the sorted order). **That does not reproduce on this build**:
   1-D ascending and descending both sort correctly, and so does the **last axis** of a 2-D tile,
   while `axis=0` raises a `CompilationError` -- a loud refusal, not a silent wrong answer. The
   claim is narrowed to that, and `tests/test_triton_behaviour.py` pins the current behaviour so a
   backend change turns red. The rule it stood for survives: every kernel here is checked against a
   KAT or an independent implementation, because compiling proves nothing.
7. **Shifting a 32-bit value by 32 is undefined behaviour, and the backend turns the whole
   kernel's results into zeros** — with a `UserWarning` that a test log filters out. It cost four
   sessions of bisection on BLAKE3; the tell was that the *inputs* read back correct from inside
   the launch while the stored value was zero.

## 4. If you are deciding what to port next

1. Find the fastest CPU implementation that exists (OpenSSL, liboqs, the reference crate) and
   measure **1 thread and 16 processes on this machine, in the same window** as the device
   number. `benchmarks/bench_16p_barrier.py` + `benchmarks/snapshot.py` do exactly that.
2. **Score three things, not one.** The old checklist here said "if the CPU has a hardware
   instruction, do not port; if not, the work is worth it", and this project's own numbers refute
   both halves -- SHA-256 *has* SHA-NI and wins 2.80x, BLAKE3 has *no* instruction and loses 0.74x.
   Score instead:
   - **dedicated instruction?** AES-NI / VAES / PCLMUL / SHA-NI -> the CPU side is at 13-95 GB/s
     and you lose. Necessary to lose, not sufficient.
   - **how wide a SIMD does the shape admit?** 16 lanes (BLAKE3) or 8-16 (ChaCha20) and the CPU
     wins by 1.3-3x; 4 lanes (Keccak) or a serial chain and it does not.
   - **how long is the serial dependency chain?** 24-80 rounds of dependent work per block leaves a
     core latency-bound, which is where the GPU's parallelism actually pays.
   All three favourable -> 0.11-0.65x. All three unfavourable -> 5.1-8.1x. Mixed -> measure, do not
   guess: SHA-256 is the mixed case and it came out at 2.80x.
3. If it is a *shape* the CPU cannot batch (thousands of independent small transforms), the
   GPU wins by 3-60x regardless of instruction counts — that is what Falcon verification and
   the SLH-DSA suite demonstrate.
4. If it needs cross-warp communication (sort, scan, irregular reduction), check the gather and
   atomics costs first; `docs/benchmarks.md` §15 and §16 have the numbers.
