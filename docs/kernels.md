# Kernel catalogue

Every kernel the library launches, with its contract. This is the part of the repository that
is meant to be reused: the schemes are thin, the kernels are not. The layout below mirrors
`triton_crypto/` package-for-package (`primitives/`, `mlkem/`, `mldsa/`, `slhdsa/`, `lms/`,
`falcon/`).

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
| `_inv2x2_kernel` | `primitives/ntt/large.py` | primitive | `(cdiv(N/4, 1024),)` | inverse transform, two stages per pass (`mul_fused_torch`) |
| `_mul_kernel` | `primitives/ntt/large.py` | primitive | `(cdiv(N, 65536),)` | the negacyclic pre/post twiddle |
| `_block_kernel` | `primitives/blake3.py` | primitive | `(chunks, cdiv(B, 64))` | `blake3_hash_torch`, per block index |
| `_pair_kernel` | `primitives/blake3.py` | primitive | `(pairs, cdiv(B, 64))` | `blake3_hash_torch`, per tree level |
| `_poly1305_kernel` | `primitives/poly1305.py` | primitive | `(cdiv(B, 64),)` | `poly1305_torch`, AEAD |
| `_gemm_modq_kernel` | `mlkem/kernels/gemm_modq.py` | ML-KEM / generic | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq` |
| `_gemm_modq_i8_kernel` | `mlkem/kernels/gemm_modq.py` | ML-KEM / generic | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq_int8` |
| `padcat2_kernel` | `mlkem/kernels/padcat.py` | ML-KEM / generic | `(cdiv(n, BR),)` | `padcat2` |
| `padcat_sfx_kernel` | `mlkem/kernels/padcat.py` | ML-KEM / generic | `(cdiv(n, BR),)` | `padcat_sfx` |
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
| `w1_pack_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, k, cdiv(polyw1_bytes, BC))` | `Verifier._verify_device` |
| `thash_kernel` | `slhdsa/kernels.py` | SLH-DSA | `(cdiv(rows, BLOCK),)` | `slhdsa/verify.py` |
| `lms_chain_kernel` | `lms/kernels.py` | LMS / HSS | `(cdiv(rows, BLOCK),)` | `lms.verify.Verifier` |
| `_pack_sig_words_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B, 64), cdiv(W, 16))` | `Verifier.verify`, `comp_decode_batch_torch` |
| `_comp_decode_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B, 128),)` | `Verifier.verify`, `comp_decode_batch_torch` |
| `_pad_nonce_msg_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B, 32), cdiv(PADDED_LEN, 64))` | `Verifier.verify` (uniform message lengths) |
| `_hash_to_point_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B, BLK),)` | `hash_to_point_batch_torch`, `Verifier.verify` |
| `_mul_ntt_kernel` | `falcon/kernels.py` | Falcon | `(cdiv(B*N, 4096),)` | `Verifier.verify_core_torch` |
| `_check_kernel` | `falcon/kernels.py` | Falcon | `(B,)` | `Verifier.verify_core_torch` |

---

## The exported surface of `triton_crypto.primitives`

Everything the package exports, in one place. `tests/test_primitives_contract.py` asserts that
every name here appears in this document, so an export cannot be added without a contract; the
sections below give the arguments and the traps for the ones with a kernel behind them.

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
| `primitives.keccak.keccak_f_ref` | scalar Keccak-f[1600], the oracle for the kernel (submodule, not re-exported) |
| `hash_np` / `hash_np_batch` / `pad_torch` / `pad_batch` / `pad10star1` / `padded_len` | the sponge reference and its padding helpers |
| `ntt_np` / `intt_np` / `mul_np` / `negacyclic_conv_np` | the NTT reference path (`ntt/ref.py`, `fields.py`) |
| `primitives.ntt.domain.bitrev` | the bit-reversal helper (submodule, not re-exported) |
| `primitives.ntt.large.large_ntt_np` / `primitives.ntt.large.large_ntt_torch` | the 2^k multi-pass path, numpy and device (submodule; the class `LargeNTT` is the top-level entry) |
| `poseidon2_permute_np` | the Plonky3-shaped reference permutation |
| `gemm_int8_np` / `gemm_int8_torch` | exact int8 GEMM (the matrix-unit substrate) |
| `gf2_8_matmul_torch` | **GF(2^8) product from 64 int8 GEMMs** — see Composite primitive wrappers below |
| `aes128_expand_key_np` / `aes128_encrypt_block_np` / `aes128_ctr_np` / `aes128_ctr_torch` | AES reference and the T-table device path |
| `primitives.aes.t_table` | the generated T-table itself (submodule) |
| `chacha20_keystream_np` / `chacha20_keystream_torch` | ChaCha20 keystream, reference and device |
| `poly1305_np_batch` / `poly1305_torch` / `chacha20_poly1305_torch` / `chacha20_poly1305_open_torch` | Poly1305 and the AEAD, reference and device |
| `blake3_hash_np` / `blake3_hash_torch` | BLAKE3 reference and the tree-hash device path |
| `get_sponge` / `sponge_padded` / `sha_absorb` / `sha2` | the sponge/sha2 entry points the schemes call |
| `intt_torch` / `mont_mul` / `poseidon2_permute_torch` | the device inverse NTT, the Montgomery helper and the device permutation |
| `chacha20_xor` | ChaCha20 XOR (the AEAD's data path), one `_xor_kernel` launch |
| `compress_np` | the 7-round BLAKE3 compression primitive, exported at the top level |
| `primitives.blake3.compression_np` | one BLAKE3 compression, the oracle `blake3_hash_np` is built on (submodule) |

---

## Primitives (`triton_crypto/primitives/`)

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

**Measured**: 210–218 Mperm/s at ≥2¹⁸ rows on the Arc B390 (~4.6–4.8 ns/permutation). See
`docs/benchmarks.md` §4.1.

### `ntt_kernel(X, OUT, ZT, P, N, Q, SCALE, LOG_N, LAYERS, BR, INVERSE, WIDE, MONT, QINV)` — `primitives/ntt/kernel.py`

Negacyclic NTT over `Z_q[X]/(X^N+1)`, forward or inverse, standard or NTT domain.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[P, N]` | `int32` (`int64` when `WIDE`) | input coefficients, rows independent |
| `ZT` | `[LAYERS * N]` | same | per-layer twiddle factors (layer `s` at `s*N`), pre-multiplied by `SCALE` on the inverse side |
| `OUT` | `[P, N]` | same | transformed rows |

`N`, `Q`, `SCALE`, `LOG_N`, `LAYERS`, `BR`, `INVERSE`, `WIDE`, `MONT`, `QINV` are all `constexpr`,
so every (domain, direction, arithmetic-width) combination is a separate specialisation compiled on
first use. `LOG_N` layers are a runtime loop inside the kernel; within a layer the butterfly
partner is `idx ^ L` (one `tl.gather`, not a shuffle pattern), and twiddles are indexed by
`(iteration, row)`.

`WIDE=True` is the ML-DSA path (q ≈ 2²³, products do not fit int32): twiddles are stored in
Montgomery form and the reduction is `primitives.fields.mont_mul` (`t = (a*b * QINV) mod 2³²;
r = (a*b - t*Q) >> 32`, canonicalised to `[0, q)`), which is bit-identical to the reference and
1.92× faster than the naive int64 modulo. `MONT=True` switches the same reduction on for a narrow
`q` (1.66× on ML-KEM's q = 3329). The twiddle tables are built in `primitives/ntt/domain.py`;
`NTTDomain` is generic in `(q, N, ζ, layers)`, so `BABYBEAR_NTT` (q = 2³¹−2²⁷+1) and Falcon
(q = 12289, N = 512) reuse the same kernel with no new code.

### `_stage_kernel(X, Y, M, TOTAL, LOG_HALF, W, Q, FORWARD, BLK)` — `primitives/ntt/large.py`

One Stockham pass of a 2^20-point transform, i.e. one launch per stage with the data in global
memory. `M` is the stage's block length: it halves for the forward DIF pass (N, N/2, ..., 2) and
doubles for the inverse DIT pass (2, 4, ..., N). `W` holds that stage's `M/2` twiddles.

- **`%` on a negative int64 is C's truncated remainder**, so `u - v` is normalised with an
  explicit `tl.where(t < 0, t + Q, t)` rather than a modulo;
- the twiddle table and the negacyclic pre/post tables are **built once and cached per device**
  (avoiding a per-call 2^20 Python loop).

Memory traffic is 8 MB per pass: **84 MB** per transform at the default radix-4 (ten passes),
160 MB at radix-2 (`docs/benchmarks.md` §4.3).

### `_stage4_kernel(X, Y, M, TOTAL, LOG_Q, W1A, W1B, W2, PRE, MUL, Q, FUSE_PRE, FUSE_MUL, BLK)` — `primitives/ntt/large.py`

Two radix-2 stages fused into one radix-4 pass, so ten passes instead of twenty and half the
traffic. Bit-identical to the radix-2 sequence (verified 2^4..2^20, including the odd-`log2` case
where one radix-2 pass is left over). **This is the default** (`LargeNTT(n, q, radix=4)`; pass
`radix=2` for the older path).

`FUSE_PRE` applies the negacyclic pre-twiddle on the way into the first pass. `FUSE_MUL` multiplies
each store by a caller-supplied vector, which is where the negacyclic product's pointwise step
happens at zero extra pass cost.

### `_inv2x2_kernel(X, Y, M, TOTAL, LOG_H, WA, WB, POST, Q, FUSE_POST, BLK)` — `primitives/ntt/large.py`

Two inverse DIT stages fused into one pass, the mirror of `_stage4_kernel` on the way back.
Together with `FUSE_MUL`, this cuts a 2^20 negacyclic product from 42 passes (`mul_torch`, 3.36 ms)
to 32 passes (`mul_fused_torch`, 2.41 ms, 1.40× faster; `docs/benchmarks.md` §4.3). Bit-identical
to the stage-by-stage inverse for N = 2^4..2^14 (`tests/test_ntt_large.py`).

### `sha_kernel(HIN, KT, MSG, OUT, rows, NB, NW, NOUT, BLOCK)` — `primitives/sha2.py`

SHA-256 / SHA-512 compression, absorbing `NB` fixed-size blocks from a given input state.
Generated once per `bits`; `NB` is a runtime argument, so a new message length no longer compiles a
new kernel. The round loop is rolled (one body, `KT` from a tensor).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `HIN` | `[8]` | `int32`/`int64` | state to start from (`sha2.iv_tensor` for a fresh hash, or the state after a constant prefix) |
| `KT` | `[64]`/`[80]` | same | round constants |
| `MSG` | `[rows, nblocks*16]` | same | pre-padded message, little-endian words (swapped on load) |
| `OUT` | `[rows, 8]` | same | final state, byte-swapped back so a `uint8` view reads big-endian |

Starting from a pre-absorbed state lets SLH-DSA SHA-2 absorb `blockpad(PK.seed)` once on the host
(`primitives.sha2.compress_np`) and only run the variable blocks on the device (`slhdsa/thash_sha2.py`).

### `chacha20_kernel(KEY, NON, CTR, OUT, B, NBLK, BR)` — `primitives/chacha20.py`

ChaCha20 keystream (RFC 8439, 96-bit nonce): one program is `BR` streams × one 64-byte block.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `KEY` | `[B, 8]` | `int32` | key words, little-endian |
| `NON` | `[B, 3]` | `int32` | nonce words, little-endian |
| `CTR` | `[B]` | `int32` | first block counter of each stream |
| `OUT` | `[B, NBLK*16]` | `int32` | keystream words (little-endian in memory) |

The state is sixteen scalars per lane, serialised via `tl.join(tl.join(a, b), tl.join(c, d))`.
Rotation is `(x << r) | (x >> (32-r))` on **uint32** (on int32 the right shift is arithmetic and
silently wrong).

### `_poseidon2_kernel(X, OUT, P, RC_INIT, RC_FINAL, RC_INT, DIAG, Q, BLK)` — `primitives/poseidon2.py`

Poseidon2 over BabyBear, width 16, 8 full + 13 partial rounds. One program is `BLK` (128)
permutations; the 16 state words are 16 separate int64 vectors per lane.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[P, 16]` | `int32` | canonical field elements |
| `OUT` | `[P, 16]` | `int32` | permuted state |
| `RC_INIT`, `RC_FINAL` | `[4, 16]` | `int64` | external round constants |
| `RC_INT` | `[13]` | `int64` | partial-round constants |
| `DIAG` | `[16]` | `int64` | internal matrix diagonal, loaded once per program |

- **Initial external MDS layer precedes round 0** (matches Plonky3's
  `external_initial_permute_state`, verified against `data/vectors/poseidon2/bb16.json`).
- **`% Q` with constexpr `Q` suffices** (31-bit operands fit int64 products `< 2^62`; LLVM
  strength-reduces to multiply-high).

### `_poly1305_kernel(KEY, MSG, TAG, B, NBLK, BLK)` — `primitives/poly1305.py`

Poly1305 in the canonical 5 × 26-bit limb layout (poly1305-donna), one lane per key. `MSG` is
`[B, NBLK*16]` uint8 and must be whole 16-byte blocks (every block carries the implicit `+2^128`).
`chacha20_poly1305_torch` handles arbitrary lengths via RFC 8439 `pad16`.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `KEY` | `[B, 32]` | `uint8` | `r \|\| s` |
| `MSG` | `[B, NBLK*16]` | `uint8` | message blocks |
| `TAG` | `[B, 16]` | `uint8` | the tag |

Key invariants: 26×26-bit products sum under 2⁵⁸ in int64; write `h + (limb & MASK)` (not
`h + limb & MASK`, because `+` binds tighter than `&`); key stride is 32, not 16.

### `_aes_ctr_kernel(RK, NONCE, OUT, TBL, B, COUNTER0, NBLK, BLK)` — `primitives/aes.py`

AES-128-CTR through a T-table (a gather measurement rather than a fast path against AES-NI).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `RK` | `[B, 44]` | `int32` | expanded round keys (host-side, 44 words per key) |
| `NONCE` | `[B, 8]` | `uint8` | the nonce half of `nonce \|\| counter(8)` |
| `OUT` | `[B, NBLK*16]` | `uint8` | keystream |
| `TBL` | `[5*256]` | `int32` | `Te0..Te3` then the S-box, 4.25 KiB, gathered from global memory |

`Te0[x] = [2s, s, s, 3s]` and `Rcon` is XORed as `rcon << 24`. Host key expansion is hoisted out of
the device path (`rk` argument).

### `_gemm_int8_kernel(A, B, OUT, M, N, K, sa_m, sa_k, sb_k, sb_n, so_m, so_n, BM, BN, BK)` — `primitives/gemm.py`

Exact `int8 × int8 → int32` matrix multiply with no modulus and no epilogue. Because XMX / Tensor
Core int8 accumulators are exact int32, binary bit-plane GEMMs compute exact `GF(2)` products that
compose into `GF(2^8)` via XOR reduction (`tests/test_gemm_int8.py`).

### `_block_kernel(...)` and `_pair_kernel(...)` — `primitives/blake3.py`

BLAKE3's tree, one launch per block index and one per tree level. A lane holds one
`(message, chunk)`; the 8-word chaining value lives in `[B * chunks, 8]` between launches.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `MSG` | `[B, nbytes]` | `uint8` | the messages |
| `CVS` | `[B * chunks, 8]` | `int32` | chaining values, seeded with `BLAKE3_IV` |
| `ROOT_CV`, `ROOT_XOF` | `[B, 8]` | `int32` | the digest halves when the root is reached |

Key constraints: (1) chaining state lives in memory because Triton runtime loops cannot carry
tuples; (2) block index is passed as a runtime argument; (3) CV stores are masked by
`block < nblocks`; (4) the 64-bit chunk counter is split into `(low, high)` int32 arguments to
avoid `uint32 >> 32` UB (which caused the Xe backend to fold kernel stores to zero).

---

## Composite primitive wrappers

### `gf2_8_matmul_torch(a, b, device=None, planes=None)` — `primitives/gemm.py`

A product in `GF(2^8)` with the AES polynomial `x^8 + x^4 + x^3 + x + 1`, assembled from bit-plane
int8 GEMMs (`docs/benchmarks.md` §4.4).

| argument | shape | dtype | meaning |
|---|---|---|---|
| `a` | `[M, K]` | anything `np.asarray` reads as `uint16` | byte values |
| `b` | `[K, N]` | same | byte values |
| `device` | — | — | resolved via `devices.resolve_device` |
| `planes` | — | — | `8` (full bytes) or `1` for binary operands |

Returns an `int32` tensor on the device. It is not a `@triton.jit` kernel: it dispatches
`planes²` (64 for full bytes) `gemm_int8_torch` calls and XORs the reduction-masked results
together (~1/57 of the int8 GEMM rate at `1024³`). Raises `ValueError` when inner dimensions
disagree.

---

## ML-KEM (FIPS 203) (`triton_crypto/mlkem/`)

These kernels hard-code `q = 3329` and the ML-KEM block sizes (plus the generic `gemm_modq` /
`padcat` helpers housed under `mlkem/kernels/`); for another negacyclic ring use `primitives/ntt/`.

### `_gemm_modq_kernel(W, X, E, OUT, M, N, K, sw_m, sw_k, sx_k, sx_n, se_m, se_n, so_m, so_n, MODE, BM, BN, BK)` — `mlkem/kernels/gemm_modq.py`

`OUT = mod_q(±W·X + E)` with fp16 tensor cores and fp32 accumulation.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `W` | `[M, K]` | `fp16` | fixed matrix (small entries: the caller guarantees exactness) |
| `X` | `[K, N]` | `fp16` | per-batch operand (`N` is the batch dimension) |
| `E` | `[M, N]` | `int16` | addend (`MODE=0`, "enc") or subtrahend (`MODE=1`, "dec") |
| `OUT` | `[M, N]` | `int16` | centred residue in `[0, q)` |

All six strides are runtime arguments; `MODE` is a `constexpr` branch. Entries are bounded so every
partial sum stays `< 2²⁴` and is exact in fp32 (`docs/design.md` §2).

### `_gemm_modq_i8_kernel(HI, LO, X, E, OUT, …)` — `mlkem/kernels/gemm_modq.py`

Same contract, splitting `W = 64*HI + LO` into two `int8` matrices and recombining
`64*acc_hi + acc_lo` when fp16 exactness bounds do not hold.

### `padcat2_kernel(A, B, OUT, n, BR)` — `mlkem/kernels/padcat.py`

`OUT[row] = pad10*1(A[row] ‖ B[row], domain)`, one launch. Generated per `(lenA, lenB, domain,
broadcast)` shape for ML-KEM multi-segment sponge inputs (e.g. `G(m ‖ H(ek))`).

### `padcat_sfx_kernel(A, OUT, n, MOD, OFF, BR)` — `mlkem/kernels/padcat.py`

`OUT[row] = pad10*1(A[row // MOD] ‖ bytes[row % MOD + OFF], domain)` — shared prefix plus per-row
suffix (`rho ‖ j ‖ i` rejection-sampling pattern).

### `encaps_kernel(W, Y, E, OUT, M, K, N, PIVOT, DU, DV, BM, BN, BK, GM, MASK_N)` — `mlkem/kernels/encaps.py`

`[u; v] = W @ Y + E`, then `Compress_du`/`Compress_dv` in the epilogue; writes `comp [M, B] int16`.
Uses `_fused_cfg` (`num_stages=1` on Intel `xpu`) and `MASK_N=True` (see `fused_enc_kernel` below).
Bit packing is kept in `pack_seg_kernel` because `dv=5` spans three coefficients per byte.

### `pack_seg_kernel(COMP, OUT, B, CTLEN, OUTOFF, NSEG, D, BLK, OBLK)` — `mlkem/kernels/encaps.py`

`COMP [NSEG*256, B] int16` → `OUT [B, CTLEN]` uint8, `D*32` bytes per polynomial. Unrolls three
terms and masks by shift for `D=5`.

### `fused_enc_kernel(…)` — `mlkem/kernels/fused_gemm.py`

Decapsulation re-encryption pass with fused epilogue: GEMM → `mod q` → `Compress` → compare
against reference ciphertext → column-reduce into `parts [npm, B] int32` (summed by
`_combine_parts`).

**This kernel (along with `fused_dec_kernel` and `encaps_kernel`) runs with `num_stages=1` on Intel
`xpu` and a masked epilogue (`MASK_N=True`).** With `num_stages >= 2` on `triton-xpu 3.9` / Xe3,
pipelined epilogue loads dereference VA `0x0`, triggering an *Engine memory CAT error* and
`UR_RESULT_ERROR_DEVICE_LOST`:

| configuration | M=1024 B=512 | M=1280 B=256 |
|---|---|---|
| `num_stages=2`, unmasked epilogue | 5/5 fault | fault |
| `num_stages=2`, masked epilogue | 0/5 | **still faults** |
| `num_stages=1` | clean | clean |

`_fused_cfg` forces `num_stages=1` on `xpu` only (CUDA/ROCm keep `_pick_cfg`'s stages) with zero
measurable slowdown on Xe3 (`docs/benchmarks.md` §3.1). Regressions:
`tests/test_fused_gemm.py::test_the_epilogue_mask_keeps_the_pipelined_kernel_safe` and
`tests/test_cpu_only.py::test_fused_cfg_disables_pipelining_only_on_xpu`.

### `fused_dec_kernel(W, U, V, MB, M, K, N, BM, BN, BK, GM, MASK_N)` — `mlkem/kernels/fused_gemm.py`

`mod q (V − W·U)` → `Compress_1` → pack 8 rows into a byte → `MB [B, 32] uint8`.

### `compress_pack_kernel(acc, V, mbytes, B, BN)` — `mlkem/kernels/glue.py`

Fallback DEC epilogue when fused GEMM constraints are unmet: `acc [256, B] fp32`, `V [256, B] int16`
→ `mbytes [B, 32] uint8`.

### `build_ye_kernel(y, e, mbytes, Y, E, B, K, BN, CHUNK)` — `mlkem/kernels/glue.py`

Layout conversion `y [B,K,256]`, `e [B,K+1,256]`, `mbytes [B,32]` → `Y [K*256, B] fp16`,
`E [(K+1)*256, B] int16`, folding `E2 += 1665 * bit`.

### `cbd_ye_kernel(bufs, mbytes, OUT, B, K, RSTRIDE, BN, ETA, F16, E2)` — `mlkem/kernels/glue.py`

Centred binomial distribution straight from bytes into fp16 `Y` or int16 `E` (with `1665·bit` on
the `pid_r == K` segment).

### `pow_mac_kernel(A, S, E, OUT, GAM, P, K, N, HALF, BR)` — `mlkem/kernels/ntt.py`

`t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]` (FIPS 203 Alg. 11/12) in `Z_q[X]/(X²−γ)` using `stride=2` loads
(no `tl.gather`).

### `encode12_kernel(V, OUT, N, BYTES, OBLK)` — `mlkem/kernels/ntt.py`

`ByteEncode_12`: `[P, 256] int32` in `[0, 4096)` → `[P, 384] uint8`.

### `_sample_ntt_kernel(BUF, OUT, NB, N, QL, NCAND)` — `mlkem/kernels/rebuild.py`

FIPS 203 Alg. 7 rejection sampling of `Â` via `tl.cumsum` prefix positions + scatter: 840-byte
input → 256 NTT-domain coefficients.

### `_toeplitz_kernel(A, META, OUT, K, N, BR)` — `mlkem/kernels/rebuild.py`

Expands centred polynomial `a` into a 256×256 negacyclic Toeplitz block (`T(a)[i][j] = a[(i-j) mod
N] · (±1)`) in fp16 `W`. Normalises negative indices explicitly against Triton's truncated `%`.

### `_unpack_kernel(buf, out, B, STRIDE, COUNT, D, BR, BC)` — `mlkem/pipeline.py`

Ciphertext bytes → `int16` coefficients for one `D`-bit segment.

### `_pre_kernel(ct, out, ref, B, CTLEN, BYTE0, COUNT, D, CENTER, BC, JT)` — `mlkem/pipeline.py`

Unpack + decompress (+ centre) + transpose in one pass, keeping an `int16` reference copy for the
compare kernel (`JT=256`).

### `_cmp_kernel(acc2, E, ref, cnt, B, ROW0, ROWS, D, BM, BN)` — `mlkem/pipeline.py`

Fallback for `fused_enc_kernel`: `mod q` → compress → compare → accumulate per-column counts.

### `_cbd2_kernel(buf, out, n, BLOCK)` — `mlkem/pipeline.py`

CBD for `eta=2` over `[n, 128]` bytes.

---

## ML-DSA (FIPS 204) (`triton_crypto/mldsa/`)

### `unpack_z_kernel(SIG, OUT, SIGSTRIDE, SIGROWS, ZOFF, POLYBYTES, D, GAMMA1, N, NL)` — `mldsa/kernels.py`

`[B, SIG] uint8` → `[B*l, N] int32`. Unpacks `γ1 − z` from `D`-bit little-endian fields
(`D ∈ {18, 20}`), three byte loads per field.

### `mac_kernel(Z, A, CH, T1, OUT, B, K, L, N, Q, BN, ZERO_CH)` — `mldsa/kernels.py`

`ŵ_i = Σ_j Â[i][j]·ẑ[j] − ĉ·t̂1_i` (FIPS 204 Alg. 33/11) in `int64`. `ZERO_CH=True` skips the
`ĉ·t̂1` load during signing's first pass (avoiding non-contiguous `expand()` reads).

### `sample_in_ball_kernel(BUF, OUT, N, TAU, RATE)` — `mldsa/kernels.py`

SampleInBall (FIPS 204 Alg. 29), one program per signature: 256 coefficients in one register
vector, masked selects + `while` rejection loop.

### `hint_kernel(SIG, H, OK, SIGSTRIDE, HOFF, OMEGA, OMEGA_P2, K, N)` — `mldsa/kernels.py`

HintBitUnpack (FIPS 204 Alg. 20), one program per `(signature, i)`: writes a dense 0/1 row and a
validity bit (monotone counts ≤ ω, strictly increasing indices, zero tail).

### `w1_pack_kernel(W, H, OUT, B, K, N, MB, TWO_G2, G2, M, POLYB, BC)` — `mldsa/kernels.py`

`Decompose → UseHint → BitPack` (FIPS 204 Alg. 36/37/23) in one pass (`MB ∈ {4, 6}`).

---

## SLH-DSA (FIPS 205) (`triton_crypto/slhdsa/`)

### `thash_kernel(SEED, OUT, DATA, RCP, KP, CHAIN, INDEX, TREE, rows, LAYER, TYPE, BLOCK)` — `slhdsa/kernels.py`

`T_l` / `F` / `H` of FIPS 205 (`SHAKE256(PK.seed ‖ ADRS ‖ data)`) with absorption and permutation in
one launch. Generated per `(n, dn)` for `n ∈ {16, 24, 32}` and `dn ∈ {n, 2n}` (six kernels).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `SEED` | `[n/8]` | `int64` | words of `PK.seed` (bit pattern, not numeric) |
| `DATA` | `[rows, dn/8]` | `int64` | words of the message part, little-endian lane order |
| `OUT` | `[rows, n/8]` | `int64` | the first `n` bytes of the digest |
| `KP`, `CHAIN`, `INDEX`, `TREE` | `[rows]` | `int64` | the four variable ADRS fields |
| `LAYER`, `TYPE` | scalar | `int64` | constant within a call |

Assembles the 17 state words arithmetically with big-to-little endian swaps on ADRS and constant
`pad10*1` ORs. Cut 128f verification from `172 → 71 µs/sig` (before chain compaction took it to
`48–50 µs/sig`). Multi-block `T_l` calls go through `verify._thash_multi` + `sponge_padded`.

---

## LMS / HSS (SP 800-208) (`triton_crypto/lms/`)

### `lms_chain_kernel(VALUE, IW, QW, CIDX, START, KT, IV, rows, STEPS, BLOCK)` — `lms/kernels.py`

Walks every LM-OTS chain of every signature in one launch, one single-block SHA-256 compression
per step over `I(16) ‖ u32(q) ‖ u16(i) ‖ u8(j) ‖ value(32)` (55 bytes, fresh IV).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `VALUE` | `[rows, 8]` | `int32` | LM-OTS `y[i]`, eight big-endian words (updated in place) |
| `IW` | `[rows, 4]` | `int32` | key identifier `I`, four big-endian words |
| `QW` / `CIDX` / `START` | `[rows]` | `int32` | leaf number `q`, chain index `i`, first step index `a` |
| `STEPS` | scalar | `int64` | `2^w - 1` (255 for `w = 8`) |

Rows are sorted by `START` on the host so blocks exit early once their chains finish (~2× less
work for `w = 8`), and the 3-byte-unaligned value words are assembled with word shifts
`(v[k] << 8) | (v[k+1] >> 24)`.

---

## Falcon (FIPS 206 draft) (`triton_crypto/falcon/`)

`primitives/ntt` handles the `q = 12289, N = 512` (Falcon-512) and `N = 1024` (Falcon-1024) forward
and inverse transforms with no new NTT kernel; all six Falcon-specific launches live in
`falcon/kernels.py` so the entire verification pipeline (prefix decode, `HashToPoint`, NTT product,
and L2 norm check) stays on the device.

### `_pack_sig_words_kernel(FLAT_SIGS, SIG_OFFS, BODY_LENS, WORDS_T, B, BODY_SKIP, W, BLK_B, BLK_W)` — `falcon/kernels.py`

Packs variable-length signature bodies from a single flat byte upload into transposed `[W, B]`
big-endian `uint32` words (zero-padded past `BODY_LENS[b]`). Transposing to `[W, B]` makes the
subsequent bit-reservoir refills in `_comp_decode_kernel` coalesced across lanes.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `FLAT_SIGS` | `[total_bytes]` | `uint8` | concatenated signatures (`b"".join(signatures)`) |
| `SIG_OFFS`, `BODY_LENS` | `[B]` | `int32` | per-signature start offset and body length |
| `WORDS_T` | `[W, B]` | `uint32` | transposed big-endian word buffer (`W <= ceil(3*N/4)`) |

### `_comp_decode_kernel(WORDS_T, LENS, OUT, OK, B, W, N, MAX_MAG, PADDED_BODY_LEN, ALLOW_PADDED, BLK)` — `falcon/kernels.py`

Device-side Falcon prefix-code decoder (`BLK = 128`, one SIMD lane per signature). Each lane keeps
a 64-bit register bit reservoir (`acc`, `acc_len` in `[0, 55]`), refilling one `uint32` word from
`WORDS_T` whenever `acc_len < 24` (~32% of steps) and computing unary run lengths with a 4-stage
branchless 16-bit CLZ (zero lookup tables). Enforces `-0`, `|s2| <= 2047`, trailing unused bit
zeros, and exact consumption (or zero-tail validation when `ALLOW_PADDED=True`). Runs in
`0.086–0.142 µs/sig` on Arc B390 (`130–230×` faster than host NumPy).

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `WORDS_T` | `[W, B]` | `uint32` | transposed big-endian signature body words |
| `LENS` | `[B]` | `int32` | body length in bytes |
| `OUT` | `[B, N]` | `int32` | decoded signed `s2` coefficients |
| `OK` | `[B]` | `int8` | 1 if the encoding and tail constraints are valid |

### `_pad_nonce_msg_kernel(FLAT_SIGS, SIG_OFFS, MSGS, OUT, B, NONCE_LEN, MSG_LEN, PADDED_LEN, DOMAIN_BYTE, BLK_B, BLK_C)` — `falcon/kernels.py`

Assembles and SHAKE256-pads (`pad10*1` + `0x1F`) `nonce || msg` directly on the device from
`FLAT_SIGS` (nonce at `sig_off + 1`) and uniform-length `MSGS`, eliminating host row-copy loops.

### `_hash_to_point_kernel(STREAM, OUT, OK, B, N, NSAMP, COLS, Q, BLK)` — `falcon/kernels.py`

Falcon's `HashToPoint` tail: reads big-endian 16-bit candidates directly from the SHAKE256 byte
stream (`n + 205` for Falcon-512 with `COLS = 1024`; `n + 287` for Falcon-1024 with `COLS = 2048`),
keeps values `< 5q`, reduces mod `q`, and compacts the first `n` survivors **in order** via an
exclusive prefix sum.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `STREAM` | `[B, NSAMP * 2]` | `uint8` | raw SHAKE256 output bytes |
| `OUT` | `[B, N]` | `int32` | first `N` accepted coefficients, in order |
| `OK` | `[B]` | `int8` | 1 if the row produced at least `N` survivors |

### `_mul_ntt_kernel(A, B, OUT, P, Q, N, BLK)` and `_check_kernel(PROD, C, S2, SQN, OK, P, L2BOUND, Q, N)` — `falcon/kernels.py`

Pointwise product `s2_hat * h_hat mod q` and L2 norm check `‖(c − s2·h, s2)‖² ≤ floor(β²)`.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `A` | `[B, N]` | `int32` | `NTT(s2)` |
| `B` | `[N]` | `int32` | `NTT(h)`, single public-key row indexed as `offs % N` (avoids non-contiguous `expand`) |
| `PROD` | `[B, N]` | `int32` | `INTT(NTT(s2) * NTT(h))` |
| `C` | `[B, N]` | `int32` | `HashToPoint(nonce ‖ m)` |
| `S2` | `[B, N]` | `int32` | decoded signature coefficients, signed |
| `SQN`, `OK` | `[B]` | `int64`, `int8` | squared norm (exact in int64, `< 2^36`) and verdict |

Normalises `(prod - c) % Q` with explicit `where(t < 0, t + Q, t)` because Triton's signed `%`
truncates toward zero.

---

## Inline helpers (not launched)

These are `@triton.jit` functions called from inside other kernels; they have no grid and no
contract of their own.

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

Constraint: **a `@triton.jit` helper whose body never mentions `tl` loses the closure** and fails
at compile time with `NameError: tl is not defined`.
