# triton-crypto-kernels

**Post-quantum cryptography kernels in pure Triton.** No inline assembly, no vendor-specific
intrinsics: one code base that runs on Intel, NVIDIA and AMD GPUs.

| layer | contents | contracts |
|---|---|---|
| `triton_crypto.primitives` | Keccak-f[1600]; the sponge (SHA3 / SHAKE, any rate and output length, batched); a parameterised negacyclic NTT over `Z_q[X]/(X^n+1)` for any `q` and 2N-th root of unity, forward and inverse | [docs/kernels.md](docs/kernels.md) |
| schemes | ML-KEM (FIPS 203) — keygen, encapsulate, decapsulate · ML-DSA (FIPS 204) — verify and sign · SLH-DSA (FIPS 205) — verify · LMS/HSS (SP 800-208) — verify | below |

Every kernel the library launches is documented with its shape / dtype / grid contract in
[docs/kernels.md](docs/kernels.md). That catalogue, not the list of schemes, is the part meant to
be reused: the schemes are thin, the kernels are not.

*Numbers at a glance*, device-side at batch ≥ 4096: ML-KEM-768 `decaps` **125 ns/op** (GB10),
ML-DSA-44 `verify` **1.9 µs/op** (Arc B390). Every figure here is compared against an optimised
baseline measured **on the same machine** — liboqs single-thread, liboqs over 16 processes, and
NVIDIA cuPQC — never against the NIST reference implementation; that is where the 1000× numbers
in the literature come from. Tables, baselines and the measurement conventions:
[docs/benchmarks.md](docs/benchmarks.md).

## Usage

ML-KEM, all three operations:

> ### Naming: this is ML-KEM, not Kyber round-3
>
> **ML-KEM (FIPS 203)** is the scheme NIST standardised in 2024, derived from CRYSTALS-Kyber.
> The two are **not interoperable**: FIPS 203 dropped the KDF at the end of encaps (`K` is
> simply the first 32 bytes of `G(m‖H(ek))`), and keygen expands the seed with `G(d ‖ k)`
> rather than `G(d)`. Every correctness claim here rests on the NIST ACVP vectors for ML-KEM;
> there is **not a single Kyber KAT** in this repo.

```python
from triton_crypto import Decapsulator, Encapsulator, generate_keypairs

dec = Decapsulator("ML-KEM-768", dk)  # dk uploaded to the GPU once
keys = dec.decaps(ciphertexts)  # list[bytes] / np.ndarray -> [B, 32] uint8

enc = Encapsulator("ML-KEM-768", ek)  # ek uploaded once
cts, keys = enc.encapsulate(msgs)  # 32 random bytes each -> [B, 1088] / [B, 32]
```

| op | correctness | per op (GB10, device-side) |
|---|---|---:|
| `decaps` | ACVP 3×10 vectors, ciphertext and key byte-exact | **125 ns** |
| `encaps` | ACVP 3×25 vectors, ciphertext and key byte-exact | **88 ns** |
| `keygen` | byte-exact against kyber-py (ek + dk) | **469 ns** |

**keygen is the weak operation** — 4–5× slower than the other two, and 2.7× slower than
NVIDIA cuPQC. The other two are 1.61× and 2.27× *faster* than cuPQC on the same card. See
[docs/compare.md](docs/compare.md).

ML-DSA, batch verification and signing:

```python
from triton_crypto.mldsa import Verifier

ok = Verifier("ML-DSA-65", pk).verify(messages, signatures)  # -> [B] bool
```

The pre-0.3 import name `mlkem_triton` still works — a shim maps it onto this package.

---

## Why it is fast

Almost all of ML-KEM's arithmetic is polynomial multiplication, conventionally done with an
NTT. This project inverts the arrangement: **pre-process a fixed matrix once, then make each
operation's multiplications a single GEMM.** For `a ∈ Z_q[X]/(X^256+1)`, define
`T(a)[i][j] = a[(i - j) mod 256] · (+1 if i >= j else -1)`; then `T(a) @ b == a · b` in
`Z_q[X]/(X^256+1)`. Decapsulation becomes two GEMMs, encapsulation is the second one.

fp16 tensor cores are still exact, because one operand is always small: `|W| <= 1664 < 2^11`
and `|y| <= 2`, so with `K <= 1024` every partial sum stays below `3.4e6 < 2^24` and any
accumulation order is element-wise exact, on any backend. The output cannot be fp16 (`|sum|`
peaks near 1.8e5, above fp16's 65504), which is why both GEMMs are this project's own Triton
kernels. Full argument, fusion catalogue and Triton pitfalls: [docs/design.md](docs/design.md).


## Install

**Not on PyPI yet** — the distribution name `triton-crypto-kernels` is unregistered, so
`pip install triton-crypto-kernels` fails with "No matching distribution found" until the first
release. Install from GitHub meanwhile (the GitHub repository itself is still called
`mlkem-triton` and will be renamed at the next release):

```bash
pip install "triton-crypto-kernels @ git+https://github.com/lwzhenglittle/mlkem-triton.git"
```

Dependencies: `torch>=2.4`, `triton>=3.0`, `numpy`. Nothing needs compiling, and the wheel is
pure Python (`py3-none-any`). Add `[ref]` to also pull in kyber-py for random key-pair helpers.

The NIST ACVP vectors in `data/acvp/` (ML-KEM) and `data/acvp-mldsa/` (ML-DSA) plus the
`tests/` tree are **not** shipped in the built distribution, so the correctness gate needs a
source checkout:

```bash
git clone https://github.com/lwzhenglittle/mlkem-triton && cd mlkem-triton
pip install -e ".[test,ref]" && pytest
```

## Usage details

```python
from triton_crypto import Decapsulator

dec = Decapsulator("ML-KEM-768", dk)  # device=None -> picks cuda / xpu automatically
keys = dec.decaps(ciphertexts)  # -> np.ndarray [B, 32] uint8
keys = dec.decaps(arr)  # or pass a [B, 1088] uint8 array
dec = Decapsulator("ML-KEM-768", dk, device="cuda:1")
```

`decaps()` does numpy packing → one H2D → the whole pipeline on device → one D2H, so only
32 bytes × B travel back; `dec.decaps_torch(ct, B)` is the same path with `torch.Tensor` in and
out, without synchronising in between. **Bigger batches pay off**: launch overhead dominates
below B = 64, and you need B >= 1024 to saturate.

---

## Performance

ML-KEM-768, min-of-25, **including numpy packing, H2D and D2H**. Both machines use the same
columns, so a batch can be read across them.

**NVIDIA GB10** (DGX Spark, `sm_121`, 48 SM) / torch 2.14 / triton 3.8

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.383 ms | 0.390 ms | 0.275 ms | 0.374 | 2.67 | 3.9× |
| 2048 | 0.493 ms | 0.501 ms | 0.315 ms | 0.241 | 4.15 | 6.0× |
| 4096 | 0.869 ms | 0.876 ms | 0.533 ms | 0.212 | 4.71 | 6.8× |
| 8192 | 1.665 ms | 1.678 ms | 1.018 ms | 0.203 | 4.92 | 7.1× |

**Intel Arc B390** (Xe3-LPG, 96 EU, integrated) / torch 2.15 / triton-xpu 3.9

| B | e2e min | e2e med | device only | µs/op | M ops/s | vs liboqs-16T |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 0.919 ms | 1.090 ms | 0.665 ms | 0.898 | 1.11 | 1.61× |
| 2048 | 1.242 ms | 1.431 ms | 0.769 ms | 0.607 | 1.65 | 2.39× |
| 4096 | 1.943 ms | 2.196 ms | 1.237 ms | 0.474 | 2.11 | 3.06× |
| 8192 | 3.674 ms | 4.277 ms | 2.227 ms | 0.449 | 2.23 | 3.23× |

The two differ in more than speed. **GB10** is **2.2×** faster per op at B=8192, and its batch
curve stays flat out to B=65536 (0.125–0.130 µs/op, device-side). **B390** oscillates between
0.380 and 0.408 µs/op from B=8192 up. The min-vs-med spread separates them too: under 1% on GB10
against **16%** on B390 — that is the ±20% clock drift, and the reason only same-process ratios
are trusted.

Baseline, same machine over the same ACVP ciphertexts (liboqs 0.16.0, ML-KEM-768): **10.26
µs/op** single-threaded, **1.45 µs/op** across 16 concurrent processes.

> GPU clocks drift by 2–3× between sessions with no frequency control exposed, so **only
> same-process interleaved A/B is meaningful**; `vs liboqs` is a cross-process reference value.

Where the device time goes, the per-kernel breakdown, the fusion A/B, encaps and keygen
tables, and the comparison against cuPQC: [docs/benchmarks.md](docs/benchmarks.md).

---

## ML-DSA (FIPS 204) verification and signing

Verification is the batchable half of ML-DSA and has exactly the shape this project optimises for:
**one long-lived public key, very many signatures**. Signing is the other half, and it is here too
(one long-lived *private* key, many messages): the rejection loop is compacted across the batch —
accepted signatures leave, the rest keep trying — while the heavy stages are the verifier's own
kernels (`ntt_torch`, `mac_kernel`, `sample_in_ball_kernel`, `unpack_z_kernel`, `w1_pack_kernel`).
 `Â = ExpandA(ρ)` and `t̂1` are expanded once when the `Verifier`
is built; a signature then costs one BitUnpack, `l + k + 1` transforms, `k·l` pointwise
products and three SHAKE calls.

```python
from triton_crypto.mldsa import Verifier

v = Verifier("ML-DSA-65", pk)  # pk preprocessing, once
ok = v.verify(messages, signatures)  # -> np.ndarray[bool], any batch size
ok = v.verify(messages, sigs, ctx=b"")  # context-bound (FIPS 204 §5.2)
ok = v.verify(messages, sigs, prehash="SHA2-256")  # HashML-DSA
ok = v.verify_mu(mu, sigs)  # external-μ interface
```

| scheme | device-side, B=4096 | e2e (numpy→GPU→bool) | liboqs 1T | liboqs 16P | speed-up dev / vs 16P |
|---|---:|---:|---:|---:|---:|
| ML-DSA-44 | **1.8 µs** | 2.2 µs | 19.6 µs | 7.9 µs | 10.9× / 4.4× |
| ML-DSA-65 | **2.4 µs** | 2.9 µs | 31.6 µs | 10.6 µs | 13.2× / 4.4× |
| ML-DSA-87 | **3.5 µs** | 4.2 µs | 47.3 µs | 8.2 µs | 13.5× / 2.3× |

The CPU column was **re-measured** for this release against a freshly built liboqs 0.16.0, calling
the per-algorithm entry points (`OQS_SIG_ml_dsa_44_verify`) rather than the generic `OQS_SIG_verify`
dispatch: the generic path costs ~3× more (58.7 µs for ML-DSA-44) and was what an earlier revision
of this table quoted. The SLH-DSA baselines below were taken through the per-algorithm symbols from
the start, and re-measuring them against the new build reproduced them within 10 % (1 473 µs vs
1 333 µs for 128f).

Intel Arc B390, torch 2.15 / triton-xpu 3.9, 32-byte messages; the corpus is **signed by
liboqs** 0.16.0 (ML-DSA enabled) so the timing and the cross-implementation check share their
data. `liboqs 16P` = aggregate over 16 concurrent processes, min-of-15:
`python benchmarks/bench_mldsa_verify.py --ps ML-DSA-65 --batch 4096 --profile`.

**Correctness gates**, all NIST ACVP or a second implementation:

| layer | gate |
|---|---|
| GPU verify | 180/180 ACVP **sigVer** cases (44/65/87 × external pure / pre-hash / internal / external-μ, 144 of them invalid) |
| numpy reference | byte-exact ACVP **keyGen** (75) and **sigGen** (360, deterministic *and* hedged), 180/180 sigVer |
| kernels | BitUnpack(z), HintBitUnpack, SampleInBall, Decompose/UseHint/BitPack, NTT-domain MAC each compared with `mldsa/ref.py` |
| cross-implementation | signatures produced by liboqs verify here; both reject the same corrupted ones |

### Signing (FIPS 204 Alg. 7)

```python
from triton_crypto.mldsa import Signer

s = Signer("ML-DSA-65", sk)  # expanded secret key, preprocessing once
sigs = s.sign(messages)  # deterministic (rnd = 0), any batch size
sigs = s.sign(messages, ctx=b"", prehash="SHA2-256", rnd=os.urandom(32))  # hedged
sigs = s.sign_mu(mus)  # external-μ interface
```

| gate | result |
|---|---|
| NIST ACVP **sigGen** on device, all 360 cases (44/65/87 × internal / external pure / pre-hash / external-μ × deterministic and hedged) | **360/360 byte-exact** |
| fresh keys, all three sets, byte-exact against the reference | pass |
| batches large enough to exercise the rejection compaction, then verified by `Verifier` | pass |

Throughput (Arc B390), against liboqs on one CPU thread (≈ 90 / 131 / 169 µs for 44 / 65 / 87):

| set | B=256 | B=1024 | B=4096 |
|---|---:|---:|---:|
| ML-DSA-44 | 60.4 µs | **43.0 µs** | **33.6 µs** |
| ML-DSA-65 | 79.1 µs | **56.6 µs** | — |
| ML-DSA-87 | 88.9 µs | **58.0 µs** | — |

**Signing is 2.3× / 2.7× / 3.2× faster than one CPU core at B=1024** (liboqs: 97 / 152 / 186 µs
for 44 / 65 / 87) and reaches parity at B ≈ 150. Three things got it there: a fused acceptance-test kernel, *speculation* (a row's k-th
attempt depends only on κ + k·ℓ, so a pass evaluates 8 candidates and keeps the first that accepts —
the signature is bit-identical), and signing messages of **mixed lengths in one batch** when their
padding lands in the same number of SHAKE blocks (rows that cross a block boundary take a second
launch; the per-block-count grouping is what keeps μ byte-exact). Against 16 CPU processes the device is behind at small batches
and roughly level at B=1024. Signing is the one operation in this repository the GPU does not accelerate — the reason is
in the shape of the work: a signature is ~4.5 sequential rejection attempts, each a small
`k·l`-sized batch, so the device pays launch latency where the CPU pays arithmetic. Verification
(10-13×) is the opposite case. Writing a signature is
*not* the same kind of problem as verifying one: the rejection loop leaves the number of attempts
data-dependent, so a batch of 1024 runs ~4.5 attempts per message on average and up to ~80 for the
slowest, and each attempt is a ~18-launch pipeline that costs ~2 ms to dispatch no matter how many
rows are left. That dispatch is the current ceiling: see `docs/benchmarks.md` §14 for the profile
and for the two optimisations that paid (device-side `z` packing, a pointwise kernel) and the two
that did not (a slot/queue model, fusing the three pointwise products). **No liboqs baseline is quoted here**: the liboqs build was not present in
the workspace for this session, so the only measured comparison is against our own reference.

### Why there is no Toeplitz GEMM here

The ML-KEM trick needs one operand to be small: `|W| <= 1664`, `|y| <= 2`, so every fp16
partial sum stays below 2²⁴ and the tensor-core result is *exactly* the integer result. ML-DSA
cannot buy that. `q = 8380417` is 23 bits, so two reduced coefficients multiply to 46 bits
before reduction; keeping that exact on int8 tensor cores takes 9 limb-GEMMs (~2.4 million
MACs per signature), a hundred times the arithmetic of the NTT it would replace. And there is
nothing to fold: verification's products are `ŵ = Â·ẑ`, already in the NTT domain, and
`c·t̂1` with a `c` that has τ ≤ 60 non-zeros.

So ML-DSA verification is genuinely an **NTT + sponge** workload — which is precisely why the
next section exists.

---

## LMS / HSS (NIST SP 800-208) verification

Stateful hash-based signatures: LM-OTS Winternitz chains over a Merkle tree, and HSS as a chain of
LMS trees where each level signs the next level's public key. Verification has no secrets and no
rewind hazard — the statefulness that makes these schemes awkward to *sign* with does not touch the
verifier.

Implemented: **both SHA-256 families from SP 800-208** — SHA-256/256 (`n = m = 32`) and
SHA-256/192 (`n = m = 24`), heights H5–H25 × Winternitz factors W1/W2/W4/W8, 40 combinations — at
any HSS depth up to 8, including hierarchies that mix parameter sets. The
SHAKE-256 variants from SP 800-208 are **not** implemented: the standard still lists their numeric
type codes as TBD, so there is nothing interoperable to encode.

| gate | result |
|---|---|
| RFC 8554 test vectors (two HSS cases, one of them mixed H5/W8 + H10/W4) | **2/2**, device and reference |
| NIST ACVP `LMS-sigVer` vectors (SHA-256/192 sets, valid and header-tampered) | **16/16**, device and reference |
| reference sign → device verify round-trips, all four Winternitz factors | **pass** |
| device vs reference on fresh keys, two-level HSS, tamper and wrong-message cases | **pass** |

Per signature at batch 4096 (one launch per chain group, not per chain):

| vector | device | Python reference |
|---|---:|---:|
| RFC 8554 case 1 (2 levels, H5/W8, 2 644-byte signature) | **8.5 µs** | 2 300 µs |
| RFC 8554 case 2 (2 levels, H10/W4 + H5/W8, 3 860-byte signature) | **7.1 µs** | 2 530 µs |

**There is no liboqs baseline**: liboqs does not implement LMS/HSS, so the honest comparison is
against this repository's own Python reference (an oracle, not a tuned CPU implementation) and the
numbers are otherwise absolute. The interesting workload is W8, where p = 34 chains of up to 255
SHA-256 compressions run per level — 4 300 hashes per signature per level on average, which is
where a batch of thousands of independent verifications pays off.

## SLH-DSA (FIPS 205) signature verification

Hash-based, so it needs no arithmetic of its own: every operation is a single-block SHAKE256
permutation, and the scheme sits directly on `primitives.sponge`. Verification recomputes the
hypertree root from the signature and compares it with the public key — no secrets, no early exit,
no data-dependent branches beyond the WOTS+ digits.

Implemented: **all twelve parameter sets** (six SHAKE, six SHA-2), verification only. The SHA-2
sets use `primitives.sha2` (SHA-256/512 compression) instead of the Keccak sponge — and they are
*faster*, because a SHA-2 block compression is cheaper on the device than 24 Keccak rounds.
Signing is out of scope, as with ML-DSA.

| gate | result |
|---|---|
| ACVP sigVer on device, all three interfaces, all twelve sets | **144/144** |
| ACVP sigVer through the numpy reference (SHA-2 sets included) | **144/144** |
| ACVP keyGen through the reference, pk *and* sk byte-exact | **120/120** |
| device vs liboqs cross-implementation | agrees on all six SHAKE sets |

Per signature, device-only at batch 4096 (Arc B390), against liboqs single-thread on the same
machine and the same signatures:

| set | device | liboqs 1T | speed-up |
|---|---:|---:|---:|
| SLH-DSA-SHAKE-128s | **18.7 µs** | 458 µs | 24.5× |
| SLH-DSA-SHAKE-128f | **47.8 µs** | 1 333 µs | 27.9× |
| SLH-DSA-SHAKE-192s | **26.5 µs** | 654 µs | 24.6× |
| SLH-DSA-SHAKE-192f | **67.4 µs** | 1 970 µs | 29.2× |
| SLH-DSA-SHAKE-256s | **37.8 µs** | 1 032 µs | 27.3× |
| SLH-DSA-SHAKE-256f | **71.2 µs** | 2 040 µs | 28.7× |

The SHA-2 sets, same conditions — cheaper per hash, so ~1.6× faster again:

| set | device | liboqs 1T | speed-up |
|---|---:|---:|---:|
| SLH-DSA-SHA2-128s | **13.1 µs** | 388 µs | 29.6× |
| SLH-DSA-SHA2-128f | **27.8 µs** | 1 170 µs | 42.0× |
| SLH-DSA-SHA2-192s | **17.5 µs** | 575 µs | 32.8× |
| SLH-DSA-SHA2-192f | **37.9 µs** | 1 595 µs | 42.0× |
| SLH-DSA-SHA2-256s | **26.0 µs** | 818 µs | 31.4× |
| SLH-DSA-SHA2-256f | **44.1 µs** | 1 708 µs | 38.7× |

**The speed-up is real but it is not the 10³× the literature reports**, and the reason is worth
stating: those figures compare against the NIST reference implementation, not against an
optimised CPU. Measured against liboqs on the same machine, ~25× is what a batch buys — and 7.5× against
16 liboqs processes, which is the number that survives once the CPU is allowed to use all its
cores.

Why the batch is needed: one verification is 2k–12k hashes in a dependency chain, and the batch
turns it into `B × 35` independent chains per layer. The schedule is one launch per chain step —
`d × 15` of them for 128f — and each hash is a fused kernel (`thash_kernel` in
[docs/kernels.md](docs/kernels.md)) that builds `PK.seed ‖ ADRS ‖ data` in registers and permutes
it, so no intermediate buffer is written. Chains also finish early: a chain whose digit is `d`
needs `w-1-d` hashes, so the rows are sorted once per layer and each step launches only over the
still-running suffix — that alone halved the chain work and moved the table from ~18× to ~25×. Small batches are pure launch overhead: at B = 1 a
signature costs 37 ms, the crossover against the CPU is around **B ≈ 32**, and B ≥ 1024 is where
the table above lives.

```python
from triton_crypto.slhdsa import Verifier

v = Verifier("SLH-DSA-SHAKE-128f", pk)  # one long-lived public key, like ML-DSA
ok = v.verify(messages, signatures)  # -> np.ndarray[B] bool; ctx=..., prehash=...
ok = v.verify_mprime(mprimes, signatures)  # the "internal" interface: M' given directly
```

## Shared primitives: Keccak and NTT are not scheme-specific

`triton_crypto/primitives/` is the layer both standards sit on. Neither module knows which
scheme is calling it.

| module | what it is | consumers |
|---|---|---|
| `primitives.keccak` | Keccak-f[1600]: generated Triton kernel (one lane per thread, 24 unrolled rounds), the round emitter shared with the sponge, and a numpy reference | tests, sponge (emitter) |
| `primitives.sponge` | pad10\*1, multi-block absorb, multi-block squeeze, and the named `SHA3-256` / `SHA3-512` / `SHAKE128` / `SHAKE256` instances | both |
| `primitives.ntt` | negacyclic NTT over `Z_q[X]/(X^N+1)`, parameterised by `(q, N, ζ, layers)`, forward **and** inverse | both |

```python
>>> from triton_crypto.primitives import KYBER, DILITHIUM
>>> KYBER.q, KYBER.zeta, KYBER.layers, KYBER.wide      # ML-KEM: 7 layers, order-N ζ
(3329, 17, 7, False)
>>> DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.layers, DILITHIUM.wide   # ML-DSA: 8 layers, order-2N ζ
(8380417, 1753, 8, True)
```

Same butterfly, two instantiations. Two differences are *not* unified, on purpose:

1. **arithmetic width** — `q = 3329` fits int32 and `%` by a constant is cheap; `q = 8380417`
   needs int64 products, and there `%` is a 64-bit software divide, so the wide path uses
   Montgomery reduction (measured 1.9× faster on the transform, and bit-identical).
2. **what a product means afterwards** — ML-KEM's ζ has order `N`, so the transform stops
   after 7 layers and its 128 CRT slots are the quadratic extensions `Z_q[X]/(X²−γ)`: the
   pointwise step is FIPS 203 Alg. 12's base-case multiply (`triton_crypto.ntt_ref`). ML-DSA's
   ζ has order `2N`, the transform splits all 256 slots, and the pointwise step is a plain
   Hadamard product (`DILITHIUM.mul`). Unifying *that* would be wrong for one of the schemes,
   so it stays in the scheme layer.

What each scheme keeps for itself: ML-KEM's base-case multiply, ByteEncode/Compress, CBD;
ML-DSA's SampleInBall, Decompose/UseHint, hint encoding, and the bit-packing of `z`, `w1` and
`t1`. The NTT kernel's ML-KEM instantiation is unchanged by the refactor — the ML-KEM test
suite, including the byte-exact keygen and the ACVP vectors, still pass bit-for-bit.

---

## Support matrix

| backend | ML-KEM (FIPS 203) | ML-DSA (FIPS 204) | SLH-DSA (FIPS 205) | LMS/HSS (SP 800-208) | notes |
|---|---|---|---|---|---|
| Intel XPU (`triton-xpu`) | ✅ **measured** | ✅ **sign + verify** | ✅ **measured** (all 12) | ✅ **measured** | Arc B390 / Xe3-LPG / torch 2.15 / triton-xpu 3.9, **812 tests green** |
| NVIDIA CUDA | ✅ **measured** | ⚠️ **not run** | ⚠️ **not run** | ⚠️ **not run** | DGX Spark GB10 / sm_121 / torch 2.14 / triton 3.8, ML-KEM suite green |
| AMD ROCm | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | ⚠️ **untested** | code-level reasoning only |
| Apple MPS / CPU | ❌ | ❌ | ❌ | ❌ | no Triton backend / no usable Triton CPU target |

The ML-KEM CUDA column was actually run on that machine: same code, same vectors, the ML-KEM
suite green, not one kernel line changed — only the tile configuration differs. ML-DSA, SLH-DSA
and LMS have only been run on the Intel device so far; there is no NVIDIA hardware in this
checkout's development environment. Detail and tuning knobs: [docs/support.md](docs/support.md).

## Testing

`pytest` (GPU cases skip themselves with no GPU), plus `ruff check . && ruff format --check .`.
**812 tests**, five layers:

| layer | what it protects against |
|---|---|
| pure maths | Toeplitz identity, fp16 exactness bound, NTT algebra for both `(q, N, ζ)` domains against a schoolbook convolution |
| kernels vs numpy | every fused kernel must be byte-equal to its unfused/numpy equivalent — the guard against silent fallback, and the only practical way to debug a data-dependent verdict |
| ACVP vectors | ML-KEM encap/decap, ML-DSA sigVer/keyGen/sigGen, SLH-DSA keyGen/sigVer, LMS sigVer |
| cross-implementation | byte-exact ML-KEM keygen against kyber-py; ML-DSA against liboqs-signed batches when liboqs is built |
| packaging/API | version metadata, every README import line, `find_packages()` (LMS was missing), public `__all__`, input normalization |

Without a GPU, the device-dependent cases skip centrally and the maths/reference paths still
run. Contribution process and **measurement discipline**: [CONTRIBUTING.md](CONTRIBUTING.md).

## Limitations

Read these first; detail in [docs/compare.md](docs/compare.md) §5.

- **One long-lived private key, many ciphertexts** is the only shape that fits: 7.9 MB of VRAM
  and 13.5 ms of pre-processing per key (on-GPU rebuild: 0.158 ms), against 2400 bytes of `dk`.
- **Throughput, not latency.** 13 launches at ~35–40 µs fixed cost make B=1 take 282 µs on GB10,
  4.6× slower than cuPQC; below B ≈ 1024 on Intel a 16-thread CPU wins.
- **keygen loses to cuPQC by 2.7×** and fusion cannot fix it — `ek` needs `t̂` in the NTT domain,
  and the fusion ceiling is 81 ns/key (1.21×).
- **ML-KEM-512 takes a slow path** (η1 = 3 falls back to generic CBD, ~3× slower); 768 and 1024
  are fast.
- **Needs fp16 tensor cores**: otherwise an `mm(fp32, fp32)` fallback, 2–4× slower. "Any GPU
  with fp16 tensor cores", not "any GPU".
- **No formal constant-time validation** (no dudect/ctgrind, no cache-side-channel analysis).
  Timing is ciphertext-independent; batch size is not.
- **No same-card NTT comparison exists**, so we cannot claim "our method beats NTT", only that we
  lead among the three implementations reporting batched throughput. Toeplitz-on-tensor-cores is
  prior art (TensorCrypto, 2022); the contribution is the exactness argument, the epilogue
  fusion, and two vendors' hardware.

ML-DSA specific:

- **Signing is batch-bound, not latency-bound.** `Signer` is GPU-resident and byte-exact against
  all 360 ACVP sigGen cases, at ~43–45 µs/signature at B=1024 on the B390 (liboqs one core:
  ~97 µs). At B=1 the rejection loop and launch overhead dominate; many signatures in one batch
  is the shape it optimises for.
- **Same one-key shape.** Â and t̂1 are expanded once per `Verifier`: **4.3 ms** of host-side
  work on the B390 (2.5 ms of which is `k·l` SHAKE128 calls with rejection sampling, 1.8 ms the
  `k` transforms), plus the first device upload when the device is cold. Many keys × few
  signatures each is not this API's shape.

SLH-DSA specific:

- **Verification only.** SLH-DSA signing is not implemented (ML-DSA signing is); the numpy
  reference in `slhdsa/ref.py` can still produce signatures for test vectors.
- **Two hash backends, two schedules.** SHAKE sets hash with Keccak; SHA-2 sets with SHA-256/512
  (and FIPS 205's asymmetric instantiation: F is always SHA-256, H/T_l switch to SHA-512 above
  128 bits). The chain loop is the same; only the `thash` differs.
- **Batch-bound.** A signature is ~400 launches, so at B = 1 it costs 37 ms — 28× *slower* than
  the CPU. The crossover is around B ≈ 32; B ≥ 1024 is where the table in the section above lives.
- **First call compiles six kernels.** Each `(n, dn)` pair of the fused `thash` is its own kernel
  (~1.5 s each, then cached on disk by Triton), so a cold machine pays a few seconds before the
  first verification.
- **The Toeplitz trick does not apply** (see above): 23-bit products cannot be exact in fp16,
  and verification's products are already in the NTT domain. The speed here comes from batching
  and from the shared primitives, not from tensor cores — the kernel is int64 scalar work.
- **ML-DSA is ~2× heavier per operation than ML-KEM-768** and scales with `k·l`; the measured
  5.4–5.6× over 16-threaded liboqs is a batch-throughput claim, not a latency claim (B=1 is
  dominated by launch overhead, ~13 launches).

## Documentation

- [docs/design.md](docs/design.md) — why Toeplitz, the fp16 exactness proof, fusion catalogue,
  Triton pitfalls, the primitives layer, and why ML-DSA uses the NTT instead
- [docs/benchmarks.md](docs/benchmarks.md) — every measured table: throughput, per-kernel,
  fusion A/B, host side
- [docs/support.md](docs/support.md) — backend status in detail, tuning knobs, project structure
- [docs/compare.md](docs/compare.md) — cuPQC, HI-Kyber, ConvKyber, TensorCrypto
- [CONTRIBUTING.md](CONTRIBUTING.md) — environment, tests, a new backend, measurement discipline


## License

Apache-2.0. ML-KEM and ML-DSA are public NIST standards; the ACVP test vectors come from
[usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server) (a US Government work, in the
public domain).
