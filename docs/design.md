# Design notes

Why the multiplication is folded into a matrix, why fp16 is still exact, and what each fused
kernel actually removes. Every number below is measured; the measurement conventions are in
[docs/benchmarks.md](benchmarks.md).

---

## 1. Toeplitz instead of NTT

In `Z_q[X]/(X^n+1)` with `n = 256`, multiplying by a fixed `a` is a linear map, so it has a
matrix. Concretely, define

```
T(a)[i][j] = a[(i - j) mod n] · (+1 if i >= j else -1)
```

then for any `b`

```
T(a) @ b  ==  a · b   in Z_q[X]/(X^n+1)
```

The sign flip is what makes this a *negacyclic* rather than a cyclic convolution: wrapping
around the modulus `X^n + 1` negates each wrapped coefficient. `tests/test_ntt.py` verifies the
identity against the reference implementation.

The matrix is 256× larger than the polynomial. That sounds fatal until you notice what a GEMM
gives you: a dense, sequential, cache-friendly access pattern handed to hardware that is
extremely good at exactly that. Two full GEMMs per operation, in fact:

| step | shape |
|---|---|
| decrypt `w = v − W_dec·u` | `[256,768] × [768,B]` |
| re-encrypt `[u';v'] = W_enc·y + [e1; e2+1665m']` | `[1024,768] × [768,B]` |

Encapsulation reuses the second GEMM with a different epilogue: `[u; v] = W_enc @ r + [e1; e2 +
Decompress_1(m)]`.

**The matrices are fixed across operations, and that is the entire trick.** For a given private
key, `T(Â)`, `T(t)` and `T(s)` do not change from one ciphertext to the next, so the expansion
cost is paid once. This is also precisely why **keygen cannot use it**: keygen's `s` and `e` are
new for every key, so there is no matrix worth expanding (§6).

### What it is not

Folding lattice multiplication into a Toeplitz matmul on tensor cores is prior art —
TensorCrypto (IEEE Access 2022) does exactly this for NTRU/LAC/FrodoKEM. The contribution here
is the fp16 exactness argument (§2), the epilogue fusion (§4), and getting the path working and
quantified on two vendors' hardware. See [docs/compare.md](compare.md) §0.7.

---

## 2. Why fp16 tensor cores are exact here

The textbook objection to fp16 for Kyber is that `q = 3329 > 2^11`, so coefficients do not fit
in an 11-bit mantissa. That objection assumes **full-range × full-range** multiplication, which
never happens in ML-KEM.

In every polynomial multiplication of the scheme, **one operand is small**:

| GEMM | weight operand | variable operand | single-product bound | K | accumulation bound | headroom to 2^24 |
|---|---|---|---|---|---|---|
| DEC `W_dec·U` | `T(s)`, measured max **2** | `u` ∈ [−1664,1664] | 3328 | 768 | 2.56e6 | **6.6×** |
| ENC `W_enc·Y` | `T(Â)/T(t)`, measured max **1664** | `y` ∈ [−2,2] | 3328 | 768 | 2.56e6 | **6.6×** |

`|W_dec|` maxes at 2 because its entries are `±s_i` with `s` from CBD with `η1 = 2`. `|y|` maxes
at 2 for the same reason. The 1664 comes from `q/2 = 1664.5` — the range of a centred
coefficient.

For ML-KEM-1024, `K = 1024` and the bound is 3.41e6, still 4.9× below 2^24. So:

```
fp32 has a 24-bit mantissa
partial sums stay below 2^24
=> every accumulation order is element-wise exact
```

Independent of the GPU's reduction tree, and therefore **backend-independent** — which is what
makes the same kernels correct on Intel and NVIDIA. `tl.dot(a, b, out_dtype=tl.float32)`
guarantees the accumulate happens in fp32.

`tests/test_dot_exact.py` checks the boundary values (±1664 × ±3) for element-wise equality on
both machines rather than trusting the argument.

### The output cannot be fp16

`|sum|` peaks around 1.8e5, well above fp16's 65504, and becomes inf. This has a practical
consequence: `torch.mm(W, X, out_dtype=torch.float32)` does not exist in the public API (it is a
private torch-xpu overload, `aten::mm.dtype`), and fp16 output is unusable, so both GEMMs are
this project's own Triton kernels. Which is fine — they are also faster than `torch.mm`: the DEC
pass goes from `0.200 + 0.070` to `0.111` ms.

---

## 3. The three pipelines

### Decapsulation (13 kernel launches)

```
ct ──▶ _pre (unpack + decompress + transpose) ──▶ U, V + reference value
                                                    │
V, U ──▶ fused_dec_gemm ──▶ m_bytes ──▶ padcat2 ──▶ sponge(G) ──▶ K', r'
                                    │
ct ──▶ padcat2 ──▶ sponge(J) ──▶ K̄
                                    │
r' ──▶ padcat_sfx ×2 ──▶ sponge(PRF) ×2 ──▶ cbd_ye ×2 ──▶ Y, E
                                                            │
                                    fused_enc_gemm ◀────────┘
                                            │
                                    cnt == 768+256 ? ──▶ K' or K̄
```

The last step is FIPS 203's implicit rejection expressed as a value comparison rather than a
branch, which keeps the timing independent of ciphertext contents (to the extent described in
[docs/compare.md](compare.md) §5.6).

### Encapsulation

`[u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]`, i.e. the ENC GEMM with a compress-and-pack
epilogue instead of compare-and-count.

### Key generation

See §6. Keygen is the one place where an NTT is unavoidable.

---

## 4. Fusion catalogue

Each entry lists what the fused kernel replaces, and why it was worth it.

### `fused_dec_gemm` — GEMM + mod q + 1-bit compress + packing

Replaces:

```python
acc      = torch.mm(W_dec, U, out_dtype=fp32)   # [256, B] fp32 = 4.2 MB
m_bytes  = compress_pack(V, acc, B)             # mod q -> compress to 1 bit -> pack
```

`acc` is never materialised. Measured 1.36× faster at B=4096 on Intel Xe3 (0.213 → 0.157 ms),
output byte-identical.

### `fused_enc_gemm` — GEMM + mod q + compress(du/dv) + compare + column reduction

Replaces `torch.mm(W_enc, Y)` plus two `_cmp_kernel` launches. The 16.8 MB fp32 intermediate is
never written; only an `[M/BM, B]` int32 partial buffer is.

Both epilogues need `x %= q` with negative-value correction, because **Triton/C `int % int`
truncates toward zero** (`-7 % 3 == -1`) whereas we need the canonical representative in
`[0, q)`. The uniform fix is `x = x % q; if x < 0: x += q`.

### `compress_pack` — mod q + 1-bit compress + pack, one launch

One program owns "BN columns × 8 rows", i.e. exactly one byte per column. The eight rows are
OR-ed into the byte by shift, so there is no reshape, no transpose, and no intermediate `m_bits`
tensor — the downstream `build_ye` reads `m_bytes` directly.

### `build_ye` — transpose + scale + concatenate, one launch

Replaces:

```python
Y  = y.permute(1,2,0).reshape(k*256, B).to(fp16)
E1 = e[:, :k].permute(1,2,0).reshape(k*256, B).to(int16)
E2 = (e[:, k].T + 1665 * m_bits).to(int16)
E  = torch.cat([E1, E2], dim=0)
```

The grid is (column block, k+1): the first k programs transpose one polynomial each, the last
computes `e[:,k] + 1665·bit` reading the bit out of `m_bytes` in place, broadcasting each byte to
eight rows.

This one is worth a war story. The old expression `y.permute(1,2,0).reshape(...).to(fp16)`
returns a **transposed view with stride (1, 256k)** — same values, non-contiguous, because
`reshape` is a view and `.to(fp16)` preserves memory format. `fused_enc_gemm` requires `Y`
contiguous, so the fused ENC GEMM was silently never executing and the whole pipeline was
falling back to `mm + _cmp_kernel`. That is why every fused kernel returns `None`/`False` and
writes nothing when its constraints are unmet, and why there is a test asserting the fused and
unfused paths are byte-equal.

### `padcat` / `padcat2` / `padcat_sfx` — pad10\*1 + segment concat in one launch

`pad_gpu(cat([a, b]))` is three launches on XPU (`cat`, `torch.zeros`, `copy_`), and `cat` has
to deal with stride-0 sources produced by `expand`. For the J stage at B=4096 (`z(32B) || ct
(1088B)`, padded to 9 blocks × 136 B) that measured **192 µs**, while the sponge kernel it feeds
is only 256 µs — i.e. 4.5 MB of input preparation was eating 43% of the J stage. Fusing it into
one launch that writes whole rows (zero padding included) took it to **28 µs**.

Implementation notes that mattered:

- **word (int64) granularity**: requires segment lengths and the total length to be multiples of
  8, which J/G/PRF all satisfy. Instruction count is 1/8 of a byte-granular version.
- the zero-fill region explicitly **excludes** the domain word and the last word, which are
  written with fixed scalar stores — otherwise a cross-warp write-write race appears between
  "write 0" and "write domain" in the same kernel.
- tile is `[BR=32, <=128]`; at BR=128 the 128 int64 per thread (= 128 GRF) costs 2.6×
  (measured 25 µs → 65 µs).

### `sponge` — one thread per sponge instance

A thread holds one whole Keccak state: 25 lanes, each a `[BLOCK]` uint64 tensor. Input is
pre-padded on the host (pad10\*1 + domain separation), so the device only does whole-block
absorb XOR + 24-round permutation + single-block squeeze (every use here squeezes at most one
rate).

**The 24 rounds run as a runtime `for` loop, with round constants read from a 24-entry uint64
table.** This is not premature tuning, it is forced by the register file:

- On Xe, one lane's uint64 occupies 8 GRF (SIMD32, 8 bytes per lane), so a 25-lane state is
  **200 GRF**, and the hardware allows only **256 GRF per thread**
  (`-cl-intel-256-GRF-per-thread`; 512-GRF is unsupported on this platform).
- Fully unrolled, the 25 `b` values of ρπ need another 200 GRF, peaking around 400. IGC then
  spills state to scratch: `zeinfo` reports `spill_size: 3584`, and the ISA contains 1202
  `// spill to offset[..]` / `fill from ..` instructions.
- Rolling the loop shortens every live range. `spill_size: 0`, static instruction count
  **15254 → 790**, throughput **1.3×** better at rate=136/out=128, and compile time
  **~25 s → ~3 s**.

Round constants are passed as a runtime kernel argument (uint64 carried as int64, bit-identical)
and cached **per device** — an earlier version hard-coded the table as a module-level singleton
on `"xpu"` and blew up on the first call on any other backend. 24 int64 is 192 B, so one copy
per device costs nothing.

There used to be a hand-written SYCL Keccak as well, 2.2–2.6× faster than this Triton kernel.
It is not in the open-source version: it was Intel-only and required shelling out to `icpx`,
which cost the library its portability. Measured cost of dropping it: sponge goes from 10.9% to
16.5% of device time, end-to-end about 20% slower. See [docs/compare.md](compare.md) §5.

### `cbd_ye` — CBD sampling straight into the packed Y/E layout

Samples `y` and `e` from the PRF output and writes them already transposed and scaled, so
`build_ye` does not need a separate transpose for them. Only implements `η = 2` bit-packing
(two coefficients per byte), which is why ML-KEM-512 (`η1 = 3`) falls back to the generic path
— see [docs/compare.md](compare.md) §5.4.

---

## 5. Triton and Intel-GPU pathologies worth recording

These cost real debugging time; they are not obvious from the docs.

- **`(row - col) % n` is C truncation, not floor.** For a Toeplitz index it produces negative
  values and silently wrong matrices. Use `& (n-1)` for power-of-two `n`.
- **`tl.gather` and arithmetic do not compose at large tile sizes.** Gathers alone: 1.2 ms for
  seven. Arithmetic alone: 0.9 ms. The two in one kernel at BR >= 16: **16.7 ms**. At BR = 2 it
  is fine. This single finding took the NTT from 15.2 ms to 1.5 ms.
- **`tl.arange` needs a power of two.** `tl.arange(0, 640)` fails; use 1024 and mask.
- **`tl.reshape` shape arguments must be plain Python ints**, derived before the call — passing a
  `tl.constexpr`-typed tensor triggers `Shape element 1 must have type constexpr[int]`.
- **`num_warps` matters far more on Intel than expected.** For the NTT, 2 warps beats 4 by 1.8×
  (222 vs 397 ns/key), measured as a paired same-process ratio. The default was 4 for a while
  and it was simply wrong.
- **The Intel `xe` driver exposes no frequency control** (`gt_*_freq_mhz` absent), and the
  iGPU idles at 0.9 GHz and boosts to 2.2 GHz, so absolute timings vary 2–4× between sessions.
  Only same-process paired ratios are trustworthy.
- **H2D cannot overlap compute on this Intel driver.** A dual-stream, zero-dependency
  experiment measured 1.328 ms against 1.307 ms for the serial same-stream version, which rules
  out streaming double-buffering as an optimisation.

---

## 6. Keygen: why the NTT cannot be avoided

Keygen needs no explanation beyond one sentence: **`ek` must contain `t̂` in the NTT domain**,
and `s`/`e` are fresh for every key so there is no reusable matrix to expand. Nothing about the
Toeplitz trick survives here.

`kernels/ntt.py` implements the forward NTT with the standard 7 stages over `idx ^ L` and a flat
`[7*256]` zeta table, plus `pow_mac` for the pointwise multiply in the quadratic extension.

`pow_mac` deliberately contains **no gathers**: the even/odd channels of the quadratic extension
are loaded with stride-2 accesses instead. Gathering was the other half of the 16.7 ms pathology
above.

Per-stage cost and the fusion upper bound are in [docs/compare.md](compare.md) §0.4. In short:
keygen is 466–469 ns/key on GB10 against cuPQC's 173.9 ns, `pow_mac` is the largest stage at
32.7% because it is memory-bound on a 37.7 MB `Â`, and fusing SampleNTT → NTT → `pow_mac` has a
measured ceiling of 81 ns/key (1.21×) — not enough to change the conclusion.

## 7. Rebuilding key material on the GPU

`kernels/rebuild.py` is a **measurement tool, not a production path**. It exists to answer one
question: cuPQC's API is stateless and re-runs k² SampleNTT calls (SHAKE-128, 840 B each) per
ciphertext to obtain `Â`, while we expand `Â` once. So how much of the 1.61× comes from skipping
`Â` generation?

Pipeline, entirely on device (only 1.5 KB crosses from the host):

```
dk ──┬─ rho(32B) ──▶ SHAKE-128(rate 168) × k² ──▶ rejection sample ──▶ Â_hat
     ├─ t_hat   ──▶ ByteDecode12 ──┐
     └─ s_hat   ──▶ ByteDecode12 ──┤
                                   ▼
                          INTT (256×256 matmul)
                                   ▼
                         centre ──▶ Toeplitz expand ──▶ W_dec / W_enc (fp16)
```

The INTT uses a precomputed 256×256 linear map in **fp64**, which is exact here: matrix entries
and inputs are ≤ 3329, so a single term is ≤ 3329² = 1.1e7 and a 256-term sum is
≤ 2.84e9 < 2^53. An 8-stage butterfly would produce identical numbers with much more code; what
this module needed was the **cost**, and 15 256-point matmuls are negligible.

`tests/test_rebuild.py` requires the rebuilt `W_dec`/`W_enc` to be element-wise equal to
`keyprep.parse_key`'s CPU-numpy result.

Result: 0.158 ms on GB10, 19.3 ns/op at B=8192, so 86% of the 1.61× is the kernel and 14% is
not re-expanding `Â`. Full numbers in [docs/compare.md](compare.md) §0.2.
