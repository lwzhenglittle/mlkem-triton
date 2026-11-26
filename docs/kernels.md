# Kernel catalogue

Every kernel the library launches, with its contract. This is the part of the repository that
is meant to be reused: the schemes are thin, the kernels are not.

Conventions used below:

- `[rows, cols]` is the **logical** shape; kernels take whatever strides the launcher passes, so
  a transposed view is fine as long as the launcher computes the strides (the ML-KEM kernels take
  explicit `sw_*`/`sx_*`/`se_*`/`so_*` stride arguments for exactly this reason).
- `int16` / `int32` hold **standard-domain** residues in `[0, q)` unless a comment says
  "centered" (then `(−q/2, q/2]`).
- `uint64` / `int64` state words are bit-cast views of ropes of bytes, not numbers.
- `cdiv` is `triton.cdiv`; `B` is the batch dimension throughout.
- "grid" is written the way the launcher writes it, so it can be copied.

Several kernels are emitted as **source strings** and compiled at import time
(`primitives/keccak.py`, `primitives/sponge.py`, `primitives/sha2.py`, `mlkem/kernels/padcat.py`,
`slhdsa/kernels.py`, `lms/kernels.py`). Triton needs a real file to import, and these are
specialised per shape (round count, `rate`/`outlen`, message segment lengths, SHA-2 word size),
so generating their source is cheaper than carrying a `tl.static_range` over every variant.
`_codegen.compile_generated` owns the write/import step (per-user 0700 directory, 0600 files).

## Summary

| kernel | file | layer | grid | launched by |
|---|---|---|---|---|
| `keccak_f_kernel` | `primitives/keccak.py` | primitive | `(cdiv(n, BLOCK),)` | direct tests; `sponge` shares the round emitter |
| `sponge_kernel` | `primitives/sponge.py` | primitive | `(cdiv(n, BLOCK),)` | `sponge_padded`, `get_sponge`, `padcat` users |
| `ntt_kernel` | `primitives/ntt/kernel.py` | primitive | `(cdiv(P, BR),)` | `ntt_torch`, `intt_torch` |
| `sha_kernel` | `primitives/sha2.py` | primitive | `(cdiv(rows, BLOCK),)` | `sha2.sha_absorb` |
| `chacha20_kernel` | `primitives/chacha20.py` | primitive | `(cdiv(B, 64), NBLK)` | `chacha20_keystream*` |
| `_xor_kernel` | `primitives/chacha20.py` | primitive | `(cdiv(n, 4096),)` | `chacha20_xor` |
| `_poseidon2_kernel` | `primitives/poseidon2.py` | primitive | `(cdiv(B, 128),)` | `poseidon2_permute_torch` |
| `_gemm_int8_kernel` | `primitives/gemm.py` | primitive | `(cdiv(M,BM), cdiv(N,BN))` | `gemm_int8_torch` |
| `_aes_ctr_kernel` | `primitives/aes.py` | primitive | `(cdiv(B, 64),)` | `aes128_ctr_torch` |
| `_stage_kernel` | `primitives/ntt/large.py` | primitive | `(cdiv(N/2, 1024),)` | `LargeNTT`, one launch per radix-2 pass |
| `_stage4_kernel` | `primitives/ntt/large.py` | primitive | `(cdiv(N/4, 1024),)` | `LargeNTT(radix=4)`, two stages per pass |
| `_mul_kernel` | `primitives/ntt/large.py` | primitive | `(cdiv(N, 65536),)` | the negacyclic pre/post twiddle |
| `_block_kernel` | `primitives/blake3.py` | primitive | `(chunks, cdiv(B, 64))` | `blake3_hash_torch`, per block index |
| `_pair_kernel` | `primitives/blake3.py` | primitive | `(pairs, cdiv(B, 64))` | `blake3_hash_torch`, per tree level |
| `_poly1305_kernel` | `primitives/poly1305.py` | primitive | `(cdiv(B, 64),)` | `poly1305_torch`, AEAD |
| `_hash_to_point_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B, 32),)` | `hash_to_point_batch_torch` |
| `_mul_ntt_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B*512, 4096),)` | `Verifier.verify_core_torch` |
| `_check_kernel` | `falcon/kernels.py` | Falcon | `(B,)` | `Verifier.verify_core_torch` |
| `lms_chain_kernel` | `lms/kernels.py` | LMS / HSS | `(cdiv(rows, BLOCK),)` | `lms.verify.Verifier` |
| `_gemm_modq_kernel` | `mlkem/kernels/gemm_modq.py` | ML-KEM | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq` |
| `_gemm_modq_i8_kernel` | `mlkem/kernels/gemm_modq.py` | ML-KEM | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq_int8` |
| `padcat2_kernel` | `mlkem/kernels/padcat.py` | generic | `(cdiv(n, BR),)` | `padcat2` |
| `padcat_sfx_kernel` | `mlkem/kernels/padcat.py` | generic | `(cdiv(n, BR),)` | `padcat_sfx` |
| `encaps_kernel` | `mlkem/kernels/encaps.py` | ML-KEM | `(cdiv(M,BM)*cdiv(B,BN),)` | `fused_encaps` |
| `pack_seg_kernel` | `mlkem/kernels/encaps.py` | ML-KEM | 2-D, `(nseg, cdiv(B,BLK))` | `pack_ciphertext` |
| `fused_enc_kernel` | `mlkem/kernels/fused_gemm.py` | ML-KEM | `(npm*cdiv(B,BN),)` | `fused_enc_gemm` |
| `fused_dec_kernel` | `mlkem/kernels/fused_gemm.py` | ML-KEM | `(cdiv(M,BM)*cdiv(B,BN),)` | `fused_dec_gemm` |
| `compress_pack_kernel` | `mlkem/kernels/glue.py` | ML-KEM | `(cdiv(B,BN), 32)` | `compress_pack` |
| `build_ye_kernel` | `mlkem/kernels/glue.py` | ML-KEM | `(cdiv(B,BN), K+1)` | `build_ye` |
| `cbd_ye_kernel` | `mlkem/kernels/glue.py` | ML-KEM | `(cdiv(B,BN), K)` / `(…, K+1)` | `cbd_y`, `cbd_e` |
| `pow_mac_kernel` | `mlkem/kernels/ntt.py` | ML-KEM | `(cdiv(P, br),)` | `pow_mac` |
| `encode12_kernel` | `mlkem/kernels/ntt.py` | ML-KEM | `(P, 3)` | `byte_encode12` |
| `_sample_ntt_kernel` | `mlkem/kernels/rebuild.py` | ML-KEM | `(n,)` | `rebuild`, `keygen` |
| `_toeplitz_kernel` | `mlkem/kernels/rebuild.py` | ML-KEM | `(n_blocks*(N/BR),)` | `rebuild` |
| `_unpack_kernel` | `mlkem/pipeline.py` | ML-KEM | `(cdiv(B,BR), cdiv(count,BC))` | `ops.unpack_ck` |
| `_pre_kernel` | `mlkem/pipeline.py` | ML-KEM | `(cdiv(B,BC),)` | `ops.pre_ck` |
| `_cmp_kernel` | `mlkem/pipeline.py` | ML-KEM | `(ROWS/BM, cdiv(B,BN))` | `ops.cmp_ck` |
| `_cbd2_kernel` | `mlkem/pipeline.py` | ML-KEM | `(cdiv(n,BLOCK),)` | `ops.cbd2_ck` |
| `unpack_z_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, l)` | `Verifier._verify_device` |
| `mac_kernel` | `mldsa/kernels.py` | ML-DSA | `(cdiv(b,MAC_BN), k)` | `Verifier._verify_device` |
| `sample_in_ball_kernel` | `mldsa/kernels.py` | ML-DSA | `(b,)` | `Verifier._verify_device` |
| `hint_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, k)` | `Verifier._verify_device` |
| `thash_kernel` | `slhdsa/kernels.py` | SLH-DSA | `(cdiv(rows, BLOCK),)` | `slhdsa/verify.py` |
| `w1_pack_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, k, cdiv(polyw1_bytes, BC))` | `Verifier._verify_device` |

---

## The exported surface of `triton_crypto.primitives`

Everything the package exports, in one place. A test asserts that every name here appears in this
document, so an export cannot be added without a contract; the sections below give the arguments
and the traps for the ones with a kernel behind them.

| export | what it is |
|---|---|
| `LargeNTT` / `NTTDomain` / `BABYBEAR_NTT` / `KYBER` / `DILITHIUM` / `BABYBEAR` | the multi-pass NTT class and the prime/NTT parameter sets it takes |
| `PrimeField` / `MONT_R` | Montgomery arithmetic helpers used by the wide domains |
| `Poseidon2` / `BABYBEAR_POSEIDON2_16` | the permutation class and its BabyBear width-16 parameters |
| `SHA3_224` / `SHA3_256` / `SHA3_384` / `SHA3_512` / `SHAKE128` / `SHAKE256` | sponge configurations (rate/domain constants) |
| `HASHES` / `HashFn` | the registry `hash_np` / `hash_np_batch` look names up in |
| `DOMAIN_SHA3` / `DOMAIN_SHAKE` | the pad bytes those configurations use |
| `IV256` / `K256` / `IV512` / `K512` | SHA-2 initial values and round constants |
| `annotations` | the `__future__` feature object itself, which the module import accidentally binds into its namespace. Not an API; listed because it is genuinely reachable as `triton_crypto.primitives.annotations` |
| `keccak_f_ref` | scalar Keccak-f[1600], the oracle for the kernel |
| `hash_np` / `hash_np_batch` / `pad_torch` / `pad_batch` / `pad10star1` / `padded_len` | the sponge reference and its padding helpers |
| `ntt_np` / `intt_np` / `mul_np` / `negacyclic_conv_np` / `bitrev` | the NTT reference path (`ntt/ref.py`, `fields.py`) |
| `large_ntt_np` / `large_ntt_torch` | the 2^k multi-pass path, numpy and device |
| `poseidon2_permute_np` | the Plonky3-shaped reference permutation |
| `gemm_int8_np` / `gemm_int8_torch` | exact int8 GEMM (the matrix-unit substrate) |
| `gf2_8_matmul_torch` | **GF(2^8) product from 64 int8 GEMMs** — see its own entry below |
| `aes128_expand_key_np` / `aes128_encrypt_block_np` / `aes128_ctr_np` / `aes128_ctr_torch` / `t_table` | AES reference and the T-table device path |
| `chacha20_keystream_np` / `chacha20_keystream_torch` | ChaCha20 keystream, reference and device |
| `poly1305_np_batch` / `poly1305_torch` / `chacha20_poly1305_torch` / `chacha20_poly1305_open_torch` | Poly1305 and the AEAD, reference and device |
| `blake3_hash_np` / `blake3_hash_torch` | BLAKE3 reference and the tree-hash device path |
| `get_sponge` / `sponge_padded` / `sha_absorb` / `sha2` | the sponge/sha2 entry points the schemes call |
| `intt_torch` / `mont_mul` / `poseidon2_permute_torch` | the device inverse NTT, the Montgomery helper and the device permutation |
| `chacha20_xor` | ChaCha20 XOR (the AEAD's data path), one `_xor_kernel` launch |
| `compress_np` | the 7-round BLAKE3 compression primitive `compression_np` wraps |
| `compression_np` | one BLAKE3 compression, the oracle `blake3_hash_np` is built on |

## Primitives

### `keccak_f_kernel(X, OUT, n, BLOCK)` — `primitives/keccak.py`

Keccak-f[1600] on `n` independent states.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[25, n]` | `int64` (bit-cast `uint64`) | state-major: word `i` of permutation `j` is `X[i, j]` |
| `OUT` | `[25, n]` | `int64` | permuted states, same layout |

One lane holds one full state (25 words) in registers and runs all 24 rounds as straight-line
code; `BLOCK` lanes per program. The source is generated with each round unrolled into θ/ρ/π/χ/ι
statements, so there is no `tl.static_range` and no per-round indexing. `__rotl` is expressed as
`(v << r) | (v >> (64-r))`, which is what makes this 64-bit and not 32-bit work — on Xe the 64-bit
ALU is half rate, and that is the single largest cost in the library.

### `sponge_kernel(X, OUT, RCP, n, NB, ROW, OUTW, BLOCK)` — `primitives/sponge.py`

The sponge around the permutation: `NB`-block absorb, optional multi-block squeeze.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[n, ROW]` | `int64` words | `ROW = NB*rate//8`; row `i` is one padded message |
| `RCP` | `[24]` | `int64` | round constants (injected by the launcher) |
| `OUT` | `[n, OUTW]` | `int64` words | `OUTW = outlen//8`; `outlen` must be a multiple of 8 |

Specialised per `(rate, outlen)` at compile time (`OUTW` is a `constexpr`, the squeeze loop is
unrolled). Multi-block squeeze re-permutes between output blocks, which is how `SHAKE256(·, 64)`
and `SHAKE128(·, 168)` are produced. `n` rows per `BLOCK` lanes; each lane keeps its own 25-word
state.

**Measured**: 218 Mperm/s at ≥2¹⁸ rows on the Arc B390 (4.6 ns/permutation), 137 Mperm/s at 2¹⁶.
See `docs/benchmarks.md`; this is the single number that decides whether a hash-based scheme is
worth putting on the device.

### `ntt_kernel(X, OUT, ZT, P, N, Q, SCALE, LOG_N, LAYERS, BR, INVERSE, WIDE, MONT, QINV)` — `primitives/ntt/kernel.py`

Negacyclic NTT over `Z_q[X]/(X^N+1)`, forward or inverse, standard or NTT domain.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[P, N]` | `int32` (`int64` when `WIDE`) | input coefficients, rows independent |
| `ZT` | `[LAYERS * N]` | same | per-layer twiddle factors (layer `s` at `s*N`), pre-multiplied by `SCALE` on the inverse side |
| `OUT` | `[P, N]` | same | transformed rows |

`N`, `Q`, `SCALE`, `LOG_N`, `LAYERS`, `BR`, `INVERSE`, `WIDE`, `MONT`, `QINV` are all `constexpr`,
so every (domain, direction, arithmetic-width) combination is a separate specialisation that is
compiled on first use. `LOG_N` layers are a runtime loop inside the kernel; within a layer the
butterfly partner is `idx ^ L` (one `tl.gather`, not a shuffle pattern), and twiddles are indexed
by `(iteration, row)`.

`WIDE=True` is the ML-DSA path (q ≈ 2²³, products do not fit int32): twiddles are stored in
Montgomery form and the reduction is `primitives.fields.mont_mul` (`t = (a*b * QINV) mod 2³²;
r = (a*b - t*Q) >> 32`, canonicalised to `[0, q)`), which is bit-identical to the reference and
1.92× faster than the naive int64 modulo. `MONT=True` switches the same reduction on for a narrow
`q` (1.66× on ML-KEM's q = 3329). The twiddle tables are built in `primitives/ntt/domain.py`;
`NTTDomain` is generic in `(q, N, ζ, layers)`, so `BABYBEAR_NTT` (q = 2³¹−2²⁷+1) reuses the same
kernel with no new code.

### `sha_kernel(HIN, KT, MSG, OUT, rows, NB, NW, NOUT, BLOCK)` — `primitives/sha2.py`

SHA-256 / SHA-512 compression, absorbing `NB` fixed-size blocks from a given input state.
Generated once per `bits`; `NB` is a runtime argument, so a new message length no longer compiles a
new kernel. The round loop is rolled (one body, `KT` from a tensor) so the
source stays short.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `HIN` | `[8]` | `int32`/`int64` | state to start from (`sha2.iv_tensor` for a fresh hash, or the state after a constant prefix) |
| `KT` | `[64]`/`[80]` | same | round constants |
| `MSG` | `[rows, nblocks*16]` | same | pre-padded message, little-endian words (swapped on load) |
| `OUT` | `[rows, 8]` | same | final state, byte-swapped back so a `uint8` view reads big-endian |

The point of "start from a state" is that the first block of every SLH-DSA SHA-2 hash is
`blockpad(PK.seed)`, constant per parameter set: its state is computed once on the host
(`sha2.compress_np`) and the kernel only absorbs what changes. Same trick as the fused `thash`,
different hash function.

`sha_thash` (now `slhdsa/thash_sha2.py`; `primitives.sha2.sha_thash` remains as a deprecated
alias) is the same idea specialised to SLH-DSA's shape: one block whose first 22 bytes are the
compressed ADRS, built in registers, with the input value copied through for rows whose WOTS chain
has finished. Both are used by the SHA-2 parameter sets (`slhdsa/verify.py`), and both are checked
against numpy for every `(n, dn)` the scheme can produce (`tests/test_sha2.py`).

### `chacha20_kernel(KEY, NON, CTR, OUT, B, NBLK, BR)` — `primitives/chacha20.py`

ChaCha20 keystream (RFC 8439, 96-bit nonce): one program is `BR` streams × one 64-byte block.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `KEY` | `[B, 8]` | `int32` | key words, little-endian |
| `NON` | `[B, 3]` | `int32` | nonce words, little-endian |
| `CTR` | `[B]` | `int32` | first block counter of each stream |
| `OUT` | `[B, NBLK*16]` | `int32` | keystream words (little-endian in memory) |

The state is sixteen scalars per lane, not a `[BR, 16]` tile: a tile cannot be sliced by a
constant index, and the quarter round needs `x0, x4, x8, x12` as separate vectors. Output is
serialised as four 3-D stores built from `tl.join(tl.join(a, b), tl.join(c, d))` — that lays out
as `[a, b, c, d]` when the offset uses `2*last_axis + middle_axis`, which `tests/test_chacha20_gpu.py`
pins.

Rotation is `(x << r) | (x >> (32-r))` on **uint32**. On int32 the right shift is arithmetic and
the result is silently wrong; LLVM folds the uint32 idiom into a funnel shift (one `shf` on
NVIDIA, two to three instructions on Xe, where rotate is 2.2× the cost of an add).

### `_poseidon2_kernel(X, OUT, P, RC_INIT, RC_FINAL, RC_INT, DIAG, Q, BLK)` — `primitives/poseidon2.py`

Poseidon2 over BabyBear, width 16, 8 full + 13 partial rounds. One program is `BLK`
permutations; the 16 state words are 16 separate int64 vectors per lane (a `[BLK, 16]` tile
cannot be sliced by a constant index, and every layer needs `state[i]` as its own vector).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[P, 16]` | `int32` | canonical field elements |
| `OUT` | `[P, 16]` | `int32` | permuted state |
| `RC_INIT`, `RC_FINAL` | `[4, 16]` | `int64` | external round constants |
| `RC_INT` | `[13]` | `int64` | partial-round constants |
| `DIAG` | `[16]` | `int64` | internal matrix diagonal, loaded once per program |

Three things this kernel exists to record:

- **The external layer is applied before the first round.** Plonky3's
  `external_initial_permute_state` starts with `mds_light_permutation`; leaving it out gives
  wrong output for every input and looks like a constants bug. `data/vectors/poseidon2/bb16.json`
  is Plonky3's own output and catches it.
- **`% Q` is enough; no Montgomery form.** Operands are 31-bit, so `a*b < 2^62` fits int64
  exactly. `mont_mul` would add a multiply-and-mask pair per field multiply; the kernel uses
  plain `(a*b) % Q` with `Q` a constexpr, which LLVM strength-reduces to a multiply-high.
- **`BLK = 128`.** Swept on an Arc B390 at B = 65536: 77.8 ns/permutation at 64, **41.5 at 128**,
  48.5 at 256 (256 spills). Below B ≈ 2048 the grid stops filling the device (B = 1024 is
  820 ns/permutation); the fix is a persistent grid, which is not written.

### `_mul_ntt_kernel(A, B, OUT, P, Q, N, BLK)` and `_check_kernel(PROD, C, S2, SQN, OK, P, L2BOUND, Q, N)` — `falcon/kernels.py`

Falcon-512 verification, the two launches that are not the transform. `ntt_torch`/`intt_torch`
handle `q = 12289, N = 512` through the existing parameterisation; these do the rest.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `A` | `[B, N]` | `int32` | `NTT(s2)` |
| `B` | `[N]` | `int32` | `NTT(h)`, the public key's transform, one row reused by all |
| `PROD` | `[B, N]` | `int32` | `INTT(NTT(s2) * NTT(h))` |
| `C` | `[B, N]` | `int32` | `HashToPoint(nonce ‖ m)` |
| `S2` | `[B, N]` | `int32` | decoded signature coefficients, signed |
| `SQN`, `OK` | `[B]` | `int64`, `int8` | squared norm and the verdict |

Two properties are load-bearing and both were found the hard way:

- **`%` on signed integers in Triton is C's truncated remainder**, not numpy's floored modulo.
  `(prod - c) % Q` leaves negative differences negative, which changes every norm -- the kernel
  normalises with an explicit `where(t < 0, t + Q, t)` before folding into `[-Q/2, Q/2]`.
- **An `expand`ed tensor is not contiguous.** `h.expand(B, N)` has stride 0 on the first axis,
  so a kernel that indexes `offs` linearly reads past the end of the first row; `B` is indexed
  as `offs % N` instead, which also avoids materialising `[B, N]`.

The norm is computed in int64 and compared against `floor(beta^2) = 34034726`. With
`|s1| <= q/2 = 6144` and `n = 512` the sum is below `2^33`, so the comparison is exact; the
reference's saturation arithmetic exists only because it works in 32 bits.

### `_poly1305_kernel(KEY, MSG, TAG, B, NBLK, BLK)` — `primitives/poly1305.py`

Poly1305 in the canonical 5 x 26-bit limb layout (poly1305-donna), one lane per key.
`MSG` is `[B, NBLK*16]` uint8 and must be whole 16-byte blocks: every block carries the
implicit `+2^128`, which is correct for a full block and wrong for a padded partial tail
(there the `0x01` byte *is* the high bit). `poly1305_np` handles any length; the AEAD pads
with zeros per RFC 8439's pad16 before calling this.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `KEY` | `[B, 32]` | `uint8` | `r || s` |
| `MSG` | `[B, NBLK*16]` | `uint8` | message blocks |
| `TAG` | `[B, 16]` | `uint8` | the tag |

Three things this kernel pins down:

- **A 26x26-bit product is 52 bits**, and five of them summed stay under 2^58: int64 carries
  the whole computation, so no `umulhi` pair per multiply and no Montgomery form.
- **`h + limb & MASK` is not what it looks like.** Python binds `+` tighter than `&`, so the
  mask truncates the accumulator to 26 bits after every block. It has to be
  `h + (limb & MASK)`. The degenerate keys (r = 1, s = 0) pass either way; only a real
  multi-block message catches it.
- **The key stride is 32, not 16.** Reading `lane * 16` works for lane 0 and for a batch of 1,
  which is exactly the shape of a hand-written smoke test.

### `_aes_ctr_kernel(RK, NONCE, OUT, TBL, B, COUNTER0, NBLK, BLK)` — `primitives/aes.py`

AES-128-CTR through a T-table, and the repository's measurement of what a lookup-driven cipher
costs on a GPU.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `RK` | `[B, 44]` | `int32` | expanded round keys (host-side, 44 words per key) |
| `NONCE` | `[B, 8]` | `uint8` | the nonce half of `nonce || counter(8)` |
| `OUT` | `[B, NBLK*16]` | `uint8` | keystream |
| `TBL` | `[5*256]` | `int32` | `Te0..Te3` then the S-box, 4.25 KiB, gathered from global memory |

Two conventions this kernel is a trap for, both caught only by a known-answer vector:

- **`Te0[x] = [2s, s, s, 3s]`** with `s = S[x]`. Building `[s, 2s, 4s, 8s]` is an easy
  misreading of "the S-box column rotated through the xtime multiples", and it produces
  plausible ciphertext for every input.
- **`Rcon` is the round word's top byte**, so it is XORed as `rcon << 24`. XORing it low gives
  a key schedule that is wrong in every round and still looks like a key schedule.

The host key schedule is *not* in the kernel: expanding 4096 keys with numpy costs ~340 ms,
which is 50x the kernel for a 4 MiB batch. `aes128_ctr_torch` therefore takes a precomputed
`rk` and `bench_primitive.py` hoists it out of the timing.

### `_gemm_int8_kernel(A, B, OUT, M, N, K, sa_m, sa_k, sb_k, sb_n, so_m, so_n, BM, BN, BK)` — `primitives/gemm.py`

Exact `int8 x int8 -> int32` matrix multiply, and nothing else: no modulus, no epilogue. It is
the generic half of `mlkem/kernels/gemm_modq.py` (which is ML-KEM's, with `Q = 3329`, two
epilogues and a float-assisted reduction). The 0.8.0 review pointed out that the docs called
that one "generic" while it lived in ML-KEM's package; the generic piece now lives here, and
`gemm_modq` stays where its semantics belong.

Why a bare int8 dot is a primitive: XMX int8 accumulators are int32 and exact, which is what
lets a GF(2^8) matrix product be assembled from per-bit-plane integer products XOR-ed through
the polynomial's reduction masks (`tests/test_gemm_int8.py` does exactly that and checks it
against a schoolbook GF(2^8) product). Whether that beats a bitsliced CUDA-core version is the
question `PLAN_ffgemm.md` exists to answer; this kernel only guarantees the arithmetic.

### `_block_kernel(...)` and `_pair_kernel(...)` — `primitives/blake3.py`

BLAKE3's tree, one launch per block index and one per level. A lane holds one
(message, chunk); the 8-word chaining value lives in a `[B * chunks, 8]` buffer between
launches.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `MSG` | `[B, nbytes]` | `uint8` | the messages |
| `CVS` | `[B * chunks, 8]` | `int32` | chaining values, seeded with `BLAKE3_IV` |
| `ROOT_CV`, `ROOT_XOF` | `[B, 8]` | `int32` | the digest halves when the root is reached |

Four constraints shaped this, and the fourth cost four sessions:

- a **runtime loop cannot carry a tuple** in Triton, so the chaining value has to live in memory
  between block indices rather than in registers;
- a `tl.constexpr` parameter **cannot receive a loop variable**, so the block index is a runtime
  argument (passing a loop variable to a constexpr silently loads the wrong block);
- the CV store is masked by `block < nblocks`: the last chunk of a message is usually partial and
  the extra iterations would overwrite it with a compression of zero padding;
- **never shift a 32-bit value by 32.** `counter >> 32` is undefined behaviour. The Xe backend
  accepts it, prints a warning that any test log filters out, and folds the **entire kernel's
  results to zero** -- with every input reading back correct from inside the same launch. The
  counter is now an explicit `(low, high)` pair and the guard is a test: any message over 1 KiB
  has a non-zero chunk counter, which is exactly the case a 64-byte KAT cannot see.

### `_hash_to_point_kernel(SAMPLES, OUT, OK, B, N, NSAMP, Q, BLK)` — `falcon/kernels.py`

Falcon's `HashToPoint` tail: from `n + 205` candidate 16-bit samples, keep those below `5q`,
reduce mod q, and take the first `n` **in order** — the reference's compaction is branch-free and
order-preserving, so an exclusive prefix sum over the keep-mask gives every survivor its
destination index.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `SAMPLES` | `[B, NSAMP]` | `int32` | big-endian 16-bit candidates, assembled from the sponge stream |
| `OUT` | `[B, n]` | `int32` | the first `n` accepted coefficients, in order |
| `OK` | `[B]` | `bool` | false when a row had fewer than `n` survivors |

0.47 µs per signature at B = 4096, against ~7 µs for the same filter in numpy — the tile is a
contiguous `[BLK, NSAMP]` block with a prefix sum and a scatter, and that is why this shape is
fast while the per-coefficient decoder (`comp_decode`) lost by 35x on the same data
(`docs/benchmarks.md` §16).

### `_stage_kernel(X, Y, M, TOTAL, LOG_HALF, W, Q, FORWARD, BLK)` — `primitives/ntt/large.py`

One Stockham pass of a 2^20-point transform, i.e. one launch per stage with the data in global
memory. `M` is the stage's block length: it halves for the forward DIF pass (N, N/2, ..., 2) and
doubles for the inverse DIT pass (2, 4, ..., N). `W` holds that stage's `M/2` twiddles.

Two things it has to get right, and both are the same two that bit other kernels here:

- **`%` on a negative int64 is C's truncated remainder**, so `u - v` is normalised with an
  explicit `tl.where(t < 0, t + Q, t)` rather than a modulo;
- the twiddle table and the negacyclic pre/post tables are **built once and cached per device**.
  The first version rebuilt them per call with a million-iteration Python loop, which cost more
  than the transform it fed (1686 ms against 10.8 ms).

Memory traffic is 20 passes x 8 MB = 160 MB for N = 2^20: this is the only primitive here whose
cost is dominated by traffic, at 76 GB/s of the 216 GB/s copy bandwidth. It is *not*
bandwidth-saturated -- `_stage4_kernel` halves the traffic and buys 10-20% -- so the remaining
limit is pass latency plus the modular multiplies (`docs/benchmarks.md` §22.1-§22.2).

### `_stage4_kernel(X, Y, M, TOTAL, LOG_Q, W1A, W1B, W2, Q, BLK)` — `primitives/ntt/large.py`

Two radix-2 stages fused into one radix-4 pass, so ten passes instead of twenty and half the
traffic. Bit-identical to the radix-2 sequence (verified 2^4..2^20, including the odd-`log2` case
where one radix-2 pass is left over), which also means the output ordering -- and therefore
pointwise compatibility with the in-register kernel -- is unchanged. It needs the first stage's
twiddles at positions `j` and `j + M/4` plus the second stage's at `j`, hence three tables.

Opt-in (`LargeNTT(..., radix=4)`): faster by 10-20% when both are measured in the same window, and
*slower* by 20% if the two are compared across windows -- the clock-drift trap in
`docs/benchmarks.md` §22.2.

### `lms_chain_kernel(VALUE, IW, QW, CIDX, START, KT, IV, rows, STEPS, BLOCK)` — `lms/kernels.py`

Every LM-OTS chain of every signature in one launch. A row is one (signature, chain element); the
kernel walks the chain in place, one *single-block* SHA-256 compression per step — the hashed string
is `I(16) || u32(q) || u16(i) || u8(j) || value(32)`, 55 bytes, so the round function starts from
the IV and no prefix state is needed.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `VALUE` | `[rows, 8]` | `int32` | the LM-OTS `y[i]`, the eight big-endian words of the 32-byte value (updated in place) |
| `IW` | `[rows, 4]` | `int32` | the key identifier `I`, four big-endian words |
| `QW` / `CIDX` / `START` | `[rows]` | `int32` | leaf number `q`, chain index `i`, first step index `a` |
| `STEPS` | scalar | `int64` | `2^w - 1` (255 for `w = 8`) |

Two details make it fast rather than merely correct. The step count is data-dependent (`a` comes
from the message digits), so a fixed 255-step loop would be ~2× the average work for `w = 8`; the
rows are sorted by `START` on the host and the kernel exits per block once its rows are done. And
the value starts at byte 23, i.e. three bytes into a word, so the block words are assembled with
`(v[k] << 8) | (v[k+1] >> 24)` shifts — cheaper than loading the value as bytes.

### `gf2_8_matmul_torch(a, b, device=None, planes=None)` — `primitives/gemm.py`

A product in GF(2^8) with the AES polynomial `x^8 + x^4 + x^3 + x + 1`, assembled from int8 GEMMs.
This is the construction `PLAN_ffgemm.md` rests on, exposed so the assumption has an end-to-end
number (`benchmarks/../probes/gf2-8-matmul-probe.py`, `docs/benchmarks.md` §19.3).

| argument | shape | dtype | meaning |
|---|---|---|---|
| `a` | `[M, K]` | anything `np.asarray` reads as `uint16` | byte values |
| `b` | `[K, N]` | same | byte values |
| `device` | -- | -- | resolved the same way as every launcher here |
| `planes` | -- | -- | `8` (both operands full bytes) or `1` for binary operands |

Returns an `int32` tensor on the device. It is *not* a `@triton.jit` kernel: it issues
`64` (or `planes**2`) `gemm_int8_torch` calls and XORs the reduction-masked results together, so its
cost is that many GEMMs -- measured at **1/55-1/61 of the int8 rate** depending on size, which is
the number that decides whether the matrix unit is worth it for a finite field.

Raises `ValueError` when the inner dimensions disagree.

## Generic (scheme-agnostic)

### `_gemm_modq_kernel(W, X, E, OUT, M, N, K, sw_m, sw_k, sx_k, sx_n, se_m, se_n, so_m, so_n, MODE, BM, BN, BK)` — `mlkem/kernels/gemm_modq.py`

`OUT = mod_q(±W·X + E)` with fp16 tensor cores and fp32 accumulation.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `W` | `[M, K]` | `fp16` | fixed matrix (small entries: the caller guarantees exactness) |
| `X` | `[K, N]` | `fp16` | per-batch operand (`N` is the batch dimension) |
| `E` | `[M, N]` | `int16` | addend (`MODE=0`, "enc") or subtrahend (`MODE=1`, "dec") |
| `OUT` | `[M, N]` | `int16` | centred residue in `[0, q)` |

All six strides are runtime arguments, so `W` can be a transposed or interleaved view. `MODE` is a
`constexpr` branch, not a runtime one.

Exactness constraint (this is what makes it legal to do modular arithmetic in fp16): entries are
bounded so that every partial sum is `< 2²⁴` and therefore exactly representable in the fp32
accumulator. `docs/design.md` §2 proves the bound for ML-KEM's use; any new user of this kernel
must reproduce that argument.

### `_gemm_modq_i8_kernel(HI, LO, X, E, OUT, …)` — `mlkem/kernels/gemm_modq.py`

Same contract, but `W = 64*HI + LO` is split into two `int8` matrices so the int8 tensor cores can
be used, and the two accumulators are recombined as `64*acc_hi + acc_lo`. Used when the fp16
exactness bound does not hold for a given tile.

### `padcat2_kernel(A, B, OUT, n, BR)` — `mlkem/kernels/padcat.py`

`OUT[row] = pad10*1(A[row] ‖ B[row], domain)`, one launch. `A`, `B`, `OUT` are byte views of
`int64` words; the segment lengths, `domain` and the two broadcast flags are baked into the
generated source, so each distinct message shape is a separate compilation. Used by ML-KEM where
two byte strings must be absorbed as one sponge input (e.g. `G(m ‖ H(ek))`).

### `padcat_sfx_kernel(A, OUT, n, MOD, OFF, BR)` — `mlkem/kernels/padcat.py`

`OUT[row] = pad10*1(A[row // MOD] ‖ bytes[row % MOD + OFF], domain)` — a prefix shared by `MOD`
rows plus a per-row suffix. This is the ML-KEM rejection-sampling pattern (`rho ‖ j ‖ i`), where
`A` is the 32-byte `rho` and the suffix is two bytes of indices.

## ML-KEM (FIPS 203)

These kernels hard-code `q = 3329` and the ML-KEM block sizes; for another negacyclic ring use
`primitives/ntt/`, which is parameterised by `(q, N, ζ, layers)`.

### `encaps_kernel(W, Y, E, OUT, M, K, N, PIVOT, DU, DV, BM, BN, BK, GM, MASK_N)` — `mlkem/kernels/encaps.py`

`[u; v] = W @ Y + E`, then `Compress_du`/`Compress_dv` in the epilogue; writes `comp [M, B] int16`.
`W` is the Toeplitz-expanded re-encryption matrix (`[M, K]` fp16), `Y [K, B]` fp16, `E [M, B] int16`.
`PIVOT` selects where the `du` segment ends and the `dv` segment starts; `MASK_N` keeps the tail of
a `BN` tile from reading past `B`. Bit packing is deliberately **not** fused here (a byte can span
three coefficients at `dv=5`, which needs a reorder across tile rows) — see `pack_seg_kernel`.

### `pack_seg_kernel(COMP, OUT, B, CTLEN, OUTOFF, NSEG, D, BLK, OBLK)` — `mlkem/kernels/encaps.py`

`COMP [NSEG*256, B] int16` → `OUT [B, CTLEN]` uint8, `D*32` bytes per polynomial. A byte may span
**three** coefficients (`D=5`); the kernel always unrolls three terms and masks by the shift.

### `fused_enc_kernel(…)` — `mlkem/kernels/fused_gemm.py`

The decapsulation re-encryption pass with its epilogue fused: GEMM → `mod q` → `Compress` →
compare against the reference ciphertext → column-reduce into `parts [npm, B] int32`. The
`[1024, B]` fp32 intermediate never reaches memory; `_combine_parts` (an inline helper) atomically
sums the partials afterwards. `npm = M/BM`.

### `fused_dec_kernel(W, U, V, MB, M, K, N, BM, BN, BK, GM, MASK_N)` — `mlkem/kernels/fused_gemm.py`

`mod q (V − W·U)` → `Compress_1` → pack 8 rows into a byte → `MB [B, 32] uint8`. Same skeleton as
`fused_enc_kernel`, different epilogue.

### `compress_pack_kernel(acc, V, mbytes, B, BN)` — `mlkem/kernels/glue.py`

Fallback for the DEC epilogue when the fused kernel's constraints are not met: `acc [256, B] fp32`,
`V [256, B] int16` → `mbytes [B, 32] uint8`, with eight rows bit-OR'ed into one byte per column
(no `m_bits` intermediate, no transpose).

### `build_ye_kernel(y, e, mbytes, Y, E, B, K, BN, CHUNK)` — `mlkem/kernels/glue.py`

The `Y`/`E` layout conversion for the next GEMM: `y [B,K,256]`, `e [B,K+1,256]`, `mbytes [B,32]` →
`Y [K*256, B] fp16`, `E [(K+1)*256, B] int16`, with `E2 += 1665 * bit` folded in. Replaces a
`permute + reshape + cat` chain that cost more than the GEMM.

### `cbd_ye_kernel(bufs, mbytes, OUT, B, K, RSTRIDE, BN, ETA, F16, E2)` — `mlkem/kernels/glue.py`

Centred binomial distribution, straight from bytes into the fp16/int16 GEMM operand layout. Two
personalities: `F16=True` produces `Y [K*256, B] fp16` from `y_buf`; `F16=False, E2=True` produces
`E [(K+1)*256, B] int16` from `e_buf`, with the `1665·bit` term added on the `pid_r == K` segment.
For `eta=2` one byte holds exactly two coefficients, so it is a single read and a single write.

### `pow_mac_kernel(A, S, E, OUT, GAM, P, K, N, HALF, BR)` — `mlkem/kernels/ntt.py`

`t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]` (FIPS 203 Alg. 11/12). `∘` is the **base-case** multiply in
`Z_q[X]/(X²−γ)`, so even and odd coefficients form two independent channels; the kernel loads them
with `stride=2` instead of gathering, which is why there is no `tl.gather` here at all. `GAM` is the
128-entry `γ` table. Rows are independent; `BR` rows per program.

### `encode12_kernel(V, OUT, N, BYTES, OBLK)` — `mlkem/kernels/ntt.py`

`ByteEncode_12`: `[P, 256] int32` in `[0, 4096)` → `[P, 384] uint8`. `D=12` means a byte spans at
most two coefficients, so two loads suffice (contrast `pack_seg_kernel`, where `D=5` spans three).

### `_sample_ntt_kernel(BUF, OUT, NB, N, QL, NCAND)` — `mlkem/kernels/rebuild.py`

FIPS 203 Alg. 7 (rejection sampling of `Â`) from a sponge output: one program per message,
`tl.cumsum` over the acceptance mask to find prefix positions, then a scatter that keeps the first
256 legal values. 840-byte input → 256 NTT-domain coefficients.

### `_toeplitz_kernel(A, META, OUT, K, N, BR)` — `mlkem/kernels/rebuild.py`

Expands a centred polynomial `a` into a 256×256 negacyclic Toeplitz block
(`T(a)[i][j] = a[(i-j) mod N] · (±1)`), writing into the fp16 GEMM operand `W`. `META [n_blocks, 3]`
holds `(source row of the coefficient, W row block, W column block)`. Triton's `%` truncates toward
zero, so the index is normalised explicitly.

### `_unpack_kernel(buf, out, B, STRIDE, COUNT, D, BR, BC)` — `mlkem/pipeline.py`

Ciphertext bytes → `int16` coefficients for one `D`-bit segment, `COUNT` values per row.

### `_pre_kernel(ct, out, ref, B, CTLEN, BYTE0, COUNT, D, CENTER, BC, JT)` — `mlkem/pipeline.py`

Unpack + decompress (+ centre) + transpose in one pass, while keeping an `int16` reference copy for
the compare kernel. `COUNT ∈ {256, 512, 768, 1024}` is not a power of two, so it is tiled by
`JT=256`.

### `_cmp_kernel(acc2, E, ref, cnt, B, ROW0, ROWS, D, BM, BN)` — `mlkem/pipeline.py`

`mod q` → compress → compare against the reference → accumulate per-column equality counts. This is
the fallback path for `fused_enc_kernel` (used when the fused kernel's tiling does not apply).

### `_cbd2_kernel(buf, out, n, BLOCK)` — `mlkem/pipeline.py`

CBD for `eta=2` over `[n, 128]` bytes; replaces six chained PyTorch ops.

## ML-DSA (FIPS 204)

### `unpack_z_kernel(SIG, OUT, SIGSTRIDE, SIGROWS, ZOFF, POLYBYTES, D, GAMMA1, N, NL)` — `mldsa/kernels.py`

`[B, SIG] uint8` → `[B*l, N] int32`. `z` is stored as `γ1 − z` in `D`-bit little-endian fields
(`D ∈ {18, 20}`), so unpacking is a subtract from `γ1`; a field never straddles more than three
bytes, so three loads per field suffice. One program per (signature, polynomial).

### `mac_kernel(Z, A, CH, T1, OUT, B, K, L, N, Q, BN, ZERO_CH)` — `mldsa/kernels.py`

`ŵ_i = Σ_j Â[i][j]·ẑ[j] − ĉ·t̂1_i` (FIPS 204 Alg. 33/11). `Â` and `t̂1` are resident (they are part
of the public key), so only `ẑ` and the output stream through. Products reach `q² ≈ 7·10¹³` and are
computed in `int64`; `BN` batch rows per program, one program per `(row block, i)`.
`ZERO_CH=True` skips the `ĉ·t̂1` load entirely (signing's first pass has `ĉ = 0`); passing a
one-row tensor with `expand()` instead made the kernel read past that row.

### `sample_in_ball_kernel(BUF, OUT, N, TAU, RATE)` — `mldsa/kernels.py`

SampleInBall (FIPS 204 Alg. 29), one program per signature. The algorithm is inherently sequential
(`c[i] ← c[j]`, then `c[j] ← ±1` with a rejection-sampled `j`), so the kernel keeps the 256
coefficients as one vector and uses masked selects instead of scatters, and a `while` loop for the
rejection (a fixed unroll would have to assume a bound the standard does not give).

### `hint_kernel(SIG, H, OK, SIGSTRIDE, HOFF, OMEGA, OMEGA_P2, K, N)` — `mldsa/kernels.py`

HintBitUnpack (FIPS 204 Alg. 20), one program per `(signature, i)`: writes a dense 0/1 row and one
validity bit per row (counts monotone and ≤ ω, indices strictly increasing, tail of the last row
zero). `OMEGA` (55/75/80) is not a power of two, hence the padded `OMEGA_P2` and the store mask.

### `w1_pack_kernel(W, H, OUT, B, K, N, MB, TWO_G2, G2, M, POLYB, BC)` — `mldsa/kernels.py`

`Decompose → UseHint → BitPack` (FIPS 204 Alg. 36/37/23) in one pass, `BC` output bytes per
program. An `MB`-bit field (`MB ∈ {4, 6}`) spans at most two coefficients, so two loads suffice.
`POLYB` masks the last, partially filled byte block.

## SLH-DSA (FIPS 205)

### `thash_kernel(SEED, OUT, DATA, RCP, KP, CHAIN, INDEX, TREE, rows, LAYER, TYPE, BLOCK)` — `slhdsa/kernels.py`

`T_l` / `F` / `H` of FIPS 205 — that is, `SHAKE256(PK.seed ‖ ADRS ‖ data)` — with the absorption
and the permutation in one launch. Generated per `(n, dn)`: for `n ∈ {16,24,32}` and
`dn ∈ {n, 2n}` there are six kernels, each specialised on both sizes, so the padding offset is a
compile-time constant.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `SEED` | `[n/8]` | `int64` | words of `PK.seed` (bit pattern, not numeric) |
| `DATA` | `[rows, dn/8]` | `int64` | words of the message part, little-endian lane order |
| `OUT` | `[rows, n/8]` | `int64` | the first `n` bytes of the digest |
| `KP`, `CHAIN`, `INDEX`, `TREE` | `[rows]` | `int64` | the four variable ADRS fields |
| `LAYER`, `TYPE` | scalar | `int64` | constant within a call |

The ADRS is 32 bytes and every piece of the message starts on an 8-byte boundary (`n` is a
multiple of 8, the ADRS is 32), so the 17 state words can be assembled arithmetically instead of
building a byte buffer: `PK.seed ‖ ADRS ‖ data` is words `0..n/8-1`, `n/8..n/8+3`, `n/8+4..`. Each
32-bit ADRS field is written big-endian by the scheme and read little-endian by the sponge, hence
the byte swaps. `pad10*1` lands at a constant word/shift (`0x1F` after the message, `0x80` in byte
135), so it is an OR against a constant.

Why this exists: composing the buffer on the host and calling the sponge kernel costs several
tensor writes per hash, and a 128f verification makes ~400 of them; the fused kernel took the
128f cost from 172 µs to 71 µs per signature at B = 4096.

`TREE` is passed as the int64 *bit pattern* (`torch` has no uint64): 256f's tree index is a full
64 bits, and a signed shift would sign-extend it.

Messages longer than one block (`T_l` over `len·n` bytes, at the top of each WOTS+ layer) go
through `verify._thash_multi`, which composes the buffer and calls `sponge_padded` — a handful of
launches per signature, not worth a second kernel.

---

## Inline helpers (not launched)

These are `@triton.jit` functions called from inside other kernels; they exist to keep the
epilogues readable and are inlined by the compiler. They have no grid and no contract of their own.

| helper | file | used by |
|---|---|---|
| `_gemm_tile`, `_tile_id` | `mlkem/kernels/fused_gemm.py` | `fused_enc_kernel`, `fused_dec_kernel` |
| `_epi_enc`, `_epi_dec`, `_combine_parts` | `mlkem/kernels/fused_gemm.py` | the two fused kernels' epilogues |
| `_compress_epi` | `mlkem/kernels/encaps.py` | `encaps_kernel` |
| `_modq` | `mlkem/kernels/ntt.py` | `pow_mac_kernel` |
| `_mod`, `_norm` | `primitives/ntt/kernel.py` | `ntt_kernel` |
| `mont_mul` | `primitives/fields.py` | `ntt_kernel`, every future modular primitive |
| `_rotl`, `_qr` | `primitives/chacha20.py` | `chacha20_kernel` |
| `_use_hint` | `mldsa/kernels.py` | `w1_pack_kernel` |

One constraint worth repeating because it has bitten: **a `@triton.jit` helper whose body never
mentions `tl` loses the closure** and fails at compile time with `NameError: tl is not defined`.
That is why the Montgomery reduction is written out at each of its three call sites instead of being
factored into a helper.
