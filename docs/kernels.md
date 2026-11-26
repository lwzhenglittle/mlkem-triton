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

Three kernels are emitted as **source strings** and compiled at import time
(`primitives/keccak.py`, `primitives/sponge.py`, `kernels/padcat.py`). Triton needs a real file to
import, and these three are specialised per shape (round count, `rate`/`outlen`, message segment
lengths), so generating their source is cheaper than carrying a `tl.static_range` over every
variant.

## Summary

| kernel | file | layer | grid | launched by |
|---|---|---|---|---|
| `keccak_f_kernel` | `primitives/keccak.py` | primitive | `(cdiv(n, BLOCK),)` | `keccak.py` only (tests, `sponge`) |
| `sponge_kernel` | `primitives/sponge.py` | primitive | `(cdiv(n, BLOCK),)` | `sponge_padded`, `get_sponge`, `padcat` users |
| `ntt_kernel` | `primitives/ntt.py` | primitive | `(cdiv(P, BR),)` | `ntt_torch`, `intt_torch` |
| `sha_kernel` | `primitives/sha2.py` | primitive | `(cdiv(rows, BLOCK),)` | `sha2.sha_absorb` |
| `lms_chain_kernel` | `lms/kernels.py` | LMS / HSS | `(cdiv(rows, BLOCK),)` | `lms.verify.Verifier` |
| `_gemm_modq_kernel` | `kernels/gemm_modq.py` | generic | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq` |
| `_gemm_modq_i8_kernel` | `kernels/gemm_modq.py` | generic | `(cdiv(M,BM), cdiv(B,BN))` | `gemm_modq_int8` |
| `padcat2_kernel` | `kernels/padcat.py` | generic | `(cdiv(n, BR),)` | `padcat2` |
| `padcat_sfx_kernel` | `kernels/padcat.py` | generic | `(cdiv(n, BR),)` | `padcat_sfx` |
| `encaps_kernel` | `kernels/encaps.py` | ML-KEM | `(cdiv(M,BM)*cdiv(B,BN),)` | `fused_encaps` |
| `pack_seg_kernel` | `kernels/encaps.py` | ML-KEM | 2-D, `(nseg, cdiv(B,BLK))` | `pack_ciphertext` |
| `fused_enc_kernel` | `kernels/fused_gemm.py` | ML-KEM | `(npm*cdiv(B,BN),)` | `fused_enc_gemm` |
| `fused_dec_kernel` | `kernels/fused_gemm.py` | ML-KEM | `(cdiv(M,BM)*cdiv(B,BN),)` | `fused_dec_gemm` |
| `compress_pack_kernel` | `kernels/glue.py` | ML-KEM | `(cdiv(B,BN), 32)` | `compress_pack` |
| `build_ye_kernel` | `kernels/glue.py` | ML-KEM | `(cdiv(B,BN), K+1)` | `build_ye` |
| `cbd_ye_kernel` | `kernels/glue.py` | ML-KEM | `(cdiv(B,BN), K)` / `(…, K+1)` | `cbd_y`, `cbd_e` |
| `pow_mac_kernel` | `kernels/ntt.py` | ML-KEM | `(cdiv(P, br),)` | `pow_mac` |
| `encode12_kernel` | `kernels/ntt.py` | ML-KEM | `(P, 3)` | `byte_encode12` |
| `_sample_ntt_kernel` | `kernels/rebuild.py` | ML-KEM | `(n,)` | `rebuild`, `keygen` |
| `_toeplitz_kernel` | `kernels/rebuild.py` | ML-KEM | `(n_blocks*(N/BR),)` | `rebuild` |
| `_unpack_kernel` | `pipeline.py` | ML-KEM | `(cdiv(B,BR), cdiv(count,BC))` | `ops.unpack_ck` |
| `_pre_kernel` | `pipeline.py` | ML-KEM | `(cdiv(B,BC),)` | `ops.pre_ck` |
| `_cmp_kernel` | `pipeline.py` | ML-KEM | `(ROWS/BM, cdiv(B,BN))` | `ops.cmp_ck` |
| `_cbd2_kernel` | `pipeline.py` | ML-KEM | `(cdiv(n,BLOCK),)` | `ops.cbd2_ck` |
| `unpack_z_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, l)` | `Verifier._verify_device` |
| `mac_kernel` | `mldsa/kernels.py` | ML-DSA | `(cdiv(b,MAC_BN), k)` | `Verifier._verify_device` |
| `sample_in_ball_kernel` | `mldsa/kernels.py` | ML-DSA | `(b,)` | `Verifier._verify_device` |
| `hint_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, k)` | `Verifier._verify_device` |
| `thash_kernel` | `slhdsa/kernels.py` | SLH-DSA | `(cdiv(rows, BLOCK),)` | `slhdsa/verify.py` |
| `w1_pack_kernel` | `mldsa/kernels.py` | ML-DSA | `(b, k, cdiv(polyw1_bytes, BC))` | `Verifier._verify_device` |

---

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

### `ntt_kernel(X, OUT, ZT, P, N, Q, SCALE, LOG_N, LAYERS, BR, INVERSE, WIDE, MONT, QINV)` — `primitives/ntt.py`

Negacyclic NTT over `Z_q[X]/(X^N+1)`, forward or inverse, standard or NTT domain.

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `X` | `[P, N]` | `int32` (`int64` when `WIDE`) | input coefficients, rows independent |
| `ZT` | `[LAYERS, N/2]` | same | per-layer twiddle factors, pre-multiplied by `SCALE` on the inverse side |
| `OUT` | `[P, N]` | same | transformed rows |

`N`, `Q`, `SCALE`, `LOG_N`, `LAYERS`, `BR`, `INVERSE`, `WIDE`, `MONT`, `QINV` are all `constexpr`,
so every (domain, direction, arithmetic-width) combination is a separate specialisation that is
compiled on first use. `LOG_N` layers are a runtime loop inside the kernel; within a layer the
butterfly partner is `idx ^ L` (one `tl.gather`, not a shuffle pattern), and twiddles are indexed
by `(iteration, row)`.

`WIDE=True` is the ML-DSA path (q ≈ 2²³, products do not fit int32): twiddles are stored in
Montgomery form and the reduction is `t = (a*b * QINV) mod 2³²; r = (a*b - t*Q) >> 32`, which is
bit-identical to the reference and 1.92× faster than the naive int64 modulo. `MONT=True` switches
the same reduction on for a narrow `q` (1.66× on ML-KEM's q = 3329).

### `sha_kernel(HIN, KT, MSG, OUT, rows, NW, NOUT, BLOCK)` — `primitives/sha2.py`

SHA-256 / SHA-512 compression, absorbing `nblocks` fixed-size blocks from a given input state.
Generated per `(bits, nblocks)`; the round loop is rolled (one body, `KT` from a tensor) so the
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

`sha_thash` is the same idea specialised to SLH-DSA's shape: one block whose first 22 bytes are the
compressed ADRS, built in registers, with the input value copied through for rows whose WOTS chain
has finished. Both are used by the SHA-2 parameter sets (`slhdsa/verify.py`), and both are checked
against numpy for every `(n, dn)` the scheme can produce (`tests/test_sha2.py`).

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

## Generic (scheme-agnostic)

### `_gemm_modq_kernel(W, X, E, OUT, M, N, K, sw_m, sw_k, sx_k, sx_n, se_m, se_n, so_m, so_n, MODE, BM, BN, BK)` — `kernels/gemm_modq.py`

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

### `_gemm_modq_i8_kernel(HI, LO, X, E, OUT, …)` — `kernels/gemm_modq.py`

Same contract, but `W = 64*HI + LO` is split into two `int8` matrices so the int8 tensor cores can
be used, and the two accumulators are recombined as `64*acc_hi + acc_lo`. Used when the fp16
exactness bound does not hold for a given tile.

### `padcat2_kernel(A, B, OUT, n, BR)` — `kernels/padcat.py`

`OUT[row] = pad10*1(A[row] ‖ B[row], domain)`, one launch. `A`, `B`, `OUT` are byte views of
`int64` words; the segment lengths, `domain` and the two broadcast flags are baked into the
generated source, so each distinct message shape is a separate compilation. Used by ML-KEM where
two byte strings must be absorbed as one sponge input (e.g. `G(m ‖ H(ek))`).

### `padcat_sfx_kernel(A, OUT, n, MOD, OFF, BR)` — `kernels/padcat.py`

`OUT[row] = pad10*1(A[row // MOD] ‖ bytes[row % MOD + OFF], domain)` — a prefix shared by `MOD`
rows plus a per-row suffix. This is the ML-KEM rejection-sampling pattern (`rho ‖ j ‖ i`), where
`A` is the 32-byte `rho` and the suffix is two bytes of indices.

## ML-KEM (FIPS 203)

### `encaps_kernel(W, Y, E, OUT, M, K, N, PIVOT, DU, DV, BM, BN, BK, GM, MASK_N)` — `kernels/encaps.py`

`[u; v] = W @ Y + E`, then `Compress_du`/`Compress_dv` in the epilogue; writes `comp [M, B] int16`.
`W` is the Toeplitz-expanded re-encryption matrix (`[M, K]` fp16), `Y [K, B]` fp16, `E [M, B] int16`.
`PIVOT` selects where the `du` segment ends and the `dv` segment starts; `MASK_N` keeps the tail of
a `BN` tile from reading past `B`. Bit packing is deliberately **not** fused here (a byte can span
three coefficients at `dv=5`, which needs a reorder across tile rows) — see `pack_seg_kernel`.

### `pack_seg_kernel(COMP, OUT, B, CTLEN, OUTOFF, NSEG, D, BLK, OBLK)` — `kernels/encaps.py`

`COMP [NSEG*256, B] int16` → `OUT [B, CTLEN]` uint8, `D*32` bytes per polynomial. A byte may span
**three** coefficients (`D=5`); the kernel always unrolls three terms and masks by the shift.

### `fused_enc_kernel(…)` — `kernels/fused_gemm.py`

The decapsulation re-encryption pass with its epilogue fused: GEMM → `mod q` → `Compress` →
compare against the reference ciphertext → column-reduce into `parts [npm, B] int32`. The
`[1024, B]` fp32 intermediate never reaches memory; `_combine_parts` (an inline helper) atomically
sums the partials afterwards. `npm = M/BM`.

### `fused_dec_kernel(W, U, V, MB, M, K, N, BM, BN, BK, GM, MASK_N)` — `kernels/fused_gemm.py`

`mod q (V − W·U)` → `Compress_1` → pack 8 rows into a byte → `MB [B, 32] uint8`. Same skeleton as
`fused_enc_kernel`, different epilogue.

### `compress_pack_kernel(acc, V, mbytes, B, BN)` — `kernels/glue.py`

Fallback for the DEC epilogue when the fused kernel's constraints are not met: `acc [256, B] fp32`,
`V [256, B] int16` → `mbytes [B, 32] uint8`, with eight rows bit-OR'ed into one byte per column
(no `m_bits` intermediate, no transpose).

### `build_ye_kernel(y, e, mbytes, Y, E, B, K, BN, CHUNK)` — `kernels/glue.py`

The `Y`/`E` layout conversion for the next GEMM: `y [B,K,256]`, `e [B,K+1,256]`, `mbytes [B,32]` →
`Y [K*256, B] fp16`, `E [(K+1)*256, B] int16`, with `E2 += 1665 * bit` folded in. Replaces a
`permute + reshape + cat` chain that cost more than the GEMM.

### `cbd_ye_kernel(bufs, mbytes, OUT, B, K, RSTRIDE, BN, ETA, F16, E2)` — `kernels/glue.py`

Centred binomial distribution, straight from bytes into the fp16/int16 GEMM operand layout. Two
personalities: `F16=True` produces `Y [K*256, B] fp16` from `y_buf`; `F16=False, E2=True` produces
`E [(K+1)*256, B] int16` from `e_buf`, with the `1665·bit` term added on the `pid_r == K` segment.
For `eta=2` one byte holds exactly two coefficients, so it is a single read and a single write.

### `pow_mac_kernel(A, S, E, OUT, GAM, P, K, N, HALF, BR)` — `kernels/ntt.py`

`t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]` (FIPS 203 Alg. 11/12). `∘` is the **base-case** multiply in
`Z_q[X]/(X²−γ)`, so even and odd coefficients form two independent channels; the kernel loads them
with `stride=2` instead of gathering, which is why there is no `tl.gather` here at all. `GAM` is the
128-entry `γ` table. Rows are independent; `BR` rows per program.

### `encode12_kernel(V, OUT, N, BYTES, OBLK)` — `kernels/ntt.py`

`ByteEncode_12`: `[P, 256] int32` in `[0, 4096)` → `[P, 384] uint8`. `D=12` means a byte spans at
most two coefficients, so two loads suffice (contrast `pack_seg_kernel`, where `D=5` spans three).

### `_sample_ntt_kernel(BUF, OUT, NB, N, QL, NCAND)` — `kernels/rebuild.py`

FIPS 203 Alg. 7 (rejection sampling of `Â`) from a sponge output: one program per message,
`tl.cumsum` over the acceptance mask to find prefix positions, then a scatter that keeps the first
256 legal values. 840-byte input → 256 NTT-domain coefficients.

### `_toeplitz_kernel(A, META, OUT, K, N, BR)` — `kernels/rebuild.py`

Expands a centred polynomial `a` into a 256×256 negacyclic Toeplitz block
(`T(a)[i][j] = a[(i-j) mod N] · (±1)`), writing into the fp16 GEMM operand `W`. `META [n_blocks, 3]`
holds `(source row of the coefficient, W row block, W column block)`. Triton's `%` truncates toward
zero, so the index is normalised explicitly.

### `_unpack_kernel(buf, out, B, STRIDE, COUNT, D, BR, BC)` — `pipeline.py`

Ciphertext bytes → `int16` coefficients for one `D`-bit segment, `COUNT` values per row.

### `_pre_kernel(ct, out, ref, B, CTLEN, BYTE0, COUNT, D, CENTER, BC, JT)` — `pipeline.py`

Unpack + decompress (+ centre) + transpose in one pass, while keeping an `int16` reference copy for
the compare kernel. `COUNT ∈ {256, 512, 768, 1024}` is not a power of two, so it is tiled by
`JT=256`.

### `_cmp_kernel(acc2, E, ref, cnt, B, ROW0, ROWS, D, BM, BN)` — `pipeline.py`

`mod q` → compress → compare against the reference → accumulate per-column equality counts. This is
the fallback path for `fused_enc_kernel` (used when the fused kernel's tiling does not apply).

### `_cbd2_kernel(buf, out, n, BLOCK)` — `pipeline.py`

CBD for `eta=2` over `[n, 128]` bytes; replaces six chained PyTorch ops.

## ML-DSA (FIPS 204)

### `unpack_z_kernel(SIG, OUT, SIGSTRIDE, SIGROWS, ZOFF, POLYBYTES, D, GAMMA1, N, NL)` — `mldsa/kernels.py`

`[B, SIG] uint8` → `[B*l, N] int32`. `z` is stored as `γ1 − z` in `D`-bit little-endian fields
(`D ∈ {18, 20}`), so unpacking is a subtract from `γ1`; a field never straddles more than three
bytes, so three loads per field suffice. One program per (signature, polynomial).

### `mac_kernel(Z, A, CH, T1, OUT, B, K, L, N, Q, BN)` — `mldsa/kernels.py`

`ŵ_i = Σ_j Â[i][j]·ẑ[j] − ĉ·t̂1_i` (FIPS 204 Alg. 33/11). `Â` and `t̂1` are resident (they are part
of the public key), so only `ẑ` and the output stream through. Products reach `q² ≈ 7·10¹³` and are
computed in `int64`; `BN` batch rows per program, one program per `(row block, i)`.

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
| `_gemm_tile`, `_tile_id` | `kernels/fused_gemm.py` | `fused_enc_kernel`, `fused_dec_kernel` |
| `_epi_enc`, `_epi_dec`, `_combine_parts` | `kernels/fused_gemm.py` | the two fused kernels' epilogues |
| `_compress_epi` | `kernels/encaps.py` | `encaps_kernel` |
| `_modq` | `kernels/ntt.py` | `pow_mac_kernel` |
| `_mod`, `_norm` | `primitives/ntt.py` | `ntt_kernel` |
| `_use_hint` | `mldsa/kernels.py` | `w1_pack_kernel` |

One constraint worth repeating because it has bitten: **a `@triton.jit` helper whose body never
mentions `tl` loses the closure** and fails at compile time with `NameError: tl is not defined`.
That is why the Montgomery reduction is written out at each of its three call sites instead of being
factored into a helper.
