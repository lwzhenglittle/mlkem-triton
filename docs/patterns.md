# What this GPU is good at, pattern by pattern

The roadmap this project started from (`Triton 密码学原语：候选算法与路线.md`) asked for ten
*computational patterns* rather than ten algorithms, on the theory that each one probes a
different capability of Triton and of the hardware — bit operations, table lookups, multi-limb
carries, modular arithmetic, the matrix unit, irregular memory access — and that the by-product
would be a table saying which patterns are easy and which are hard.

This is that table, as of 1.1.0. Every device number is from `benchmarks/snapshot.py` (one
window, list at the top of each section below); every CPU number is from
`benchmarks/cpu_baseline.json` (OpenSSL / liboqs / Plonky3, same machine, 16 concurrent
processes). Sections in `docs/benchmarks.md` are cited where the detail lives.

## 0. The one rule that survives everything else

**Whether a pattern is worth porting is decided by whether the CPU has hardware support for
it, far more than by anything about the GPU.** Same iGPU, same measurement method:

| primitive | device | CPU 1T | CPU 16P | vs 16P | why |
|---|---:|---:|---:|---:|---|
| SHA3-256 | 31.5 GB/s | 0.72 GB/s | 7.9 GB/s | **4.00x** | x86 has no Keccak instruction |
| SHA-256 | 78.0 GB/s | 4.17 GB/s | 49.4 GB/s | 1.58x | SHA-NI exists but is 2-3x slower than the device |
| ChaCha20 | 14.1 GB/s | 4.73 GB/s | 29.1 GB/s | 0.49x | AVX-512 is enough |
| AES-128-CTR (T-table) | 10.6 GB/s | 13.8 GB/s | 166 GB/s | 0.06x | AES-NI/VAES is unbeatable |
| Poly1305 | 11.6 GB/s | 7.54 GB/s | 75.8 GB/s | 0.15x | AVX-512 vectorises the 26-bit limbs |

The absolute device throughput barely varies across those rows (10-31 GB/s for everything that
is ALU-bound). The 100x spread in the last column comes entirely from the CPU side.

Two corollaries this project ran into:

- **A single iGPU core-equivalent is not the competition.** Comparing against one CPU thread
  flatters every port by 3-10x. `eval-next-steps-1.0.0.md` §2 has the numbers: a CPU baseline
  taken during a build (or on a `powersave` governor at 400 MHz) is 3x slower than the same
  code measured idle, which is how "5.6x vs CPU" became "1.0-1.8x" without touching the device
  code (`docs/benchmarks.md` §11 now warns about it).
- **The interesting GPU shapes are the ones the CPU cannot vectorise**: batched independent
  small objects (signature verification), hot small tables, and the matrix unit.

## 1. The patterns

`✅` device path measured; `⚠️` reference only; `❌` not attempted. Numbers are device vs the
16-process CPU baseline unless stated.

| # | pattern | representative | status | device | vs 1T | vs 16P | what it cost in Triton |
|---|---|---|---|---|---|---|---|
| 1 | ARX (add-rotate-xor) | ChaCha20 | ✅ | 14.1 GB/s | 2.99x | 0.49x | easy; the only trap is that `>>` on int32 is arithmetic, so rotate needs `.to(tl.uint32)` |
| 2 | tree hash | BLAKE3 | ✅ | 10.1 GB/s | 3.45x | 0.19x | one launch per block index + one per level; the chaining value lives in a buffer because a runtime loop cannot carry a tuple. See §2 for the four sessions it took |
| 3 | sponge | SHA3/SHAKE | ✅ | 31.5 GB/s | 43.9x | **4.00x** | easy and a winner: one lane holds 25 uint64 and the round loop must be *rolled*, not unrolled (unrolled: `spill_size=3584`, rolled: 0 and 30% faster) |
| 4 | lookup vs bitslice | AES-CTR T-table | ✅ | 10.6 GB/s | 0.77x | 0.06x | 160 gathers/block into a 4.25 KiB hot table; bitslice (2-3 weeks) not attempted |
| 5 | multi-limb big integer | Poly1305 | ✅ | 11.6 GB/s | 1.53x | 0.15x | 5x26-bit limbs fit int64 exactly; no `umulhi` pairs needed. X25519 not attempted |
| 6 | small-prime NTT | ML-KEM/ML-DSA/BabyBear NTT | ✅ | 24.9 / 6.9 / 6.9 M poly/s (N<=1024); **2^20 in 2.22 ms** | **31.5x** | **2.37x** | the `(q, N, ζ, layers)` parameterisation covered BabyBear and Falcon with no new kernel; 2^20 needed a multi-pass one (`ntt/large.py`) and is the only bandwidth-bound primitive here |
| 7 | algebraic hash | Poseidon2-BabyBear/16 | ✅ | 38.7 ns/perm | 2.72x | 0.19x | 16 int64 state words per lane, one program per 128 permutations; state as int32 + widen per multiply is *slower* |
| 8 | finite-field GEMM | exact int8 GEMM (UOV/RS substrate) | ✅ | 1.94 TMAC/s (n=1024), 3.96 (n=2048) | — | — | 16% of the int8 peak with an untuned tile table; the GF(2^8) product itself is verified via bit planes |
| 9 | floating-point FFT | Falcon *signing* | ❌ | — | — | — | deliberately not attempted: Triton does not expose FMA fusion, so a cross-backend bit-exact signature is not claimable. fp64 itself works and is bit-exact elementwise |
| 10 | scatter + atomics | MSM / Pippenger | ❌ | — | — | — | 32/64-bit atomics are correct; the gather penalty and the 381-bit arithmetic (900 ops per modular multiply) make it a 1-2 month item that ICICLE already owns |
| — | memory-hard | Argon2id | ❌ | — | — | — | the only load that would show an iGPU-vs-CPU memory-subsystem difference; not attempted |

### Where the device does win, and by how much

| result | number | why it wins |
|---|---:|---|
| SHA3-256 / SHAKE128 / SHAKE256 | **4.0x** | no Keccak instruction on x86 |
| SHA-256 | 1.58x | SHA-NI is real but slower than the device's rolled schedule |
| Falcon-512 verification | **3.58x** | the CPU has no way to batch thousands of independent 512-point transforms |
| SHA3-512 / SHA-512 | 4.07x / 3.57x | same as SHA3-256 (wide lanes) |

Everything else is parity at best. That is the honest summary of porting a primitive library to
an integrated GPU with a 16-core hybrid CPU beside it.

## 2. The patterns that did not land, and what stopped them

- **Tree hash (BLAKE3) — landed on the fourth attempt.** The shape is one launch per block
  index plus one per tree level. What blocked it was not the tree: `counter >> 32` on a uint32 is
  undefined behaviour, the Xe backend folded the whole kernel to zero, and every input read back
  correct from inside the same launch. Four sessions of bisection: the compression, one block
  step through a global buffer and a sixteen-launch chain all worked in isolation; the one line
  that differed was the shift. See `docs/benchmarks.md` §21.
- **Falcon signature decoding on the device.** Tried because `comp_decode` is 96% of Falcon's
  end-to-end cost. One lane per signature, 512 sequential steps (a prefix code cannot be
  bit-gathered across a batch). **Measured 2.6 ms/signature against the host's 75 us — 35x
  slower.** The host numpy version processes all signatures in lockstep against contiguous
  memory; the device version turns every coefficient into a data-dependent gather. Reverted;
  `docs/benchmarks.md` §16 records it.

  The contrast with `HashToPoint`, which *did* move to the device (9.3 -> 0.6 us/signature), is
  the useful part: a contiguous `[BLK, 720]` tile with a prefix-sum and a scatter is fast
  (0.47 us per row), while the same number of gathered *bytes* at data-dependent addresses is
  35x slower. Same problem size, opposite verdict — the access pattern, not the arithmetic.
- **Large-N NTT (2^20) — landed and tuned, and it is the one row that wins.** `ntt/large.py` is
  correct (agrees with `NTTDomain` where both fit, round trip exact) and takes **2.22 ms:
  31.5x one CPU core and 2.37x sixteen processes.** Three fixes took it from 10.8 to 2.2 ms (a
  per-call table rebuild, an int64 division in the index math, and the tile/warp choice --
  `docs/benchmarks.md` §22.1). Radix-4/8 is the remaining 2-3x to the traffic roofline.

  This is the answer to the question the project started with. Every ALU-bound primitive loses
  to sixteen cores (0.06-0.5x) and the sponge family wins by 1.6-4x only because x86 has no
  Keccak instruction. The operators that win by more are the **bandwidth-bound** ones: 160 MB of
  traffic per useful operation is something sixteen cores cannot beat without 160 MB/shot of
  their own bandwidth.

## 3. Five Triton traps found the hard way

All of them are recorded at the site where they bit, and every one of them produced *plausible*
output before a known-answer vector caught it.

1. **Signed `%` is C's truncated remainder**, not numpy's floored modulo. `(a - b) % q` leaves
   negative differences negative; Falcon's norm check was wrong for half the coefficients.
2. **`h + limb & MASK` masks the sum, not the limb.** Python binds `+` tighter than `&`, so a
   Poly1305 accumulator was truncated to 26 bits after every block. The degenerate keys
   (`r=1, s=0`) passed either way.
3. **`expand`ed tensors are not contiguous.** A kernel that indexes them linearly reads past the
   first row (Falcon's pointwise product).
4. **A `tl.constexpr` parameter cannot receive a loop variable**, and Triton rejects starred
   unpacking (`(*t, v)` is "unsupported AST node type: Starred"), so tuple accumulation in a jit
   body must be `t = t + (v,)` and only works under `tl.static_range`.
5. **A runtime loop cannot carry a tuple assignment.** Any state that has to survive a loop
   needs named scalars — and unrolling to keep the tuple is not free (trap 6).
6. **`tl.sort` compiles and is silently wrong on XPU 3.9** (returns a permutation of the input,
   not the sorted order). "It compiles" is not evidence; every kernel here is checked against a
   KAT or an independent implementation.
7. **Shifting a 32-bit value by 32 is undefined behaviour, and the backend turns the whole
   kernel's results into zeros** — with a `UserWarning` that a test log filters out. It cost four
   sessions of bisection on BLAKE3; the tell was that the *inputs* read back correct from inside
   the launch while the stored value was zero.

## 4. If you are deciding what to port next

1. Find the fastest CPU implementation that exists (OpenSSL, liboqs, the reference crate) and
   measure **1 thread and 16 processes on this machine, in the same window** as the device
   number. `benchmarks/bench_cpu.py` + `benchmark/snapshot.py` do exactly that.
2. If the CPU has a hardware instruction for it (AES-NI, SHA-NI, PCLMUL), expect 0.06-0.5x and
   do not port it for throughput.
3. If it does not (Keccak, algebraic hashes, batched small transforms), expect 1.6-4x and the
   work is worth it.
4. If it is a *shape* the CPU cannot batch (thousands of independent small transforms), the
   GPU wins by 3-60x regardless of instruction counts — that is what Falcon verification and
   the SLH-DSA suite demonstrate.
5. If it needs cross-warp communication (sort, scan, irregular reduction), check the gather and
   atomics costs first; `docs/benchmarks.md` §15 and §16 have the numbers.
