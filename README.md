# triton-crypto-kernels

**Cryptographic kernels in pure Triton** — no inline assembly, no vendor-specific intrinsics.
One code base, measured on Intel XPU and NVIDIA CUDA, written for any Triton backend.

Post-quantum schemes:

| scheme | operations | standard |
|---|---|---|
| ML-KEM | keygen, encapsulate, decapsulate | FIPS 203 |
| ML-DSA | sign, verify | FIPS 204 |
| SLH-DSA | verify (all 12 parameter sets) | FIPS 205 |
| LMS / HSS | verify (SHA-256/256 and /192) | SP 800-208 |
| Falcon | verify (Falcon-512 / Falcon-1024, unpadded & padded) | FIPS 206 draft |

And the scheme-agnostic primitives underneath them, usable on their own:

| primitive | what |
|---|---|
| `primitives.sponge` | Keccak-f[1600] and the sponge: SHA3-224/256/384/512, SHAKE128/256, any rate, batched |
| `primitives.sha2` | SHA-256 / SHA-512 compression |
| `primitives.ntt` | negacyclic NTT over `Z_q[X]/(X^N+1)`, parameterised by `(q, N, ζ, layers)`; in-register to N = 1024, multi-pass radix-4 to 2^20 (**0.85–1.6 ms**, 3.8–7.3× sixteen CPU processes) |
| `primitives.fields` | prime-field arithmetic: the 32-bit Montgomery product, shared by every modular primitive |
| `primitives.chacha20` | ChaCha20 (RFC 8439) — the first of the non-PQC primitives |
| `primitives.poseidon2` | Poseidon2-BabyBear/16 (Plonky3 parameters, 8 full + 13 partial rounds) |
| `primitives.poly1305` | Poly1305 (RFC 8439) and the ChaCha20-Poly1305 AEAD |
| `primitives.aes` | AES-128-CTR through a T-table — a gather measurement, not a fast path |
| `primitives.blake3` | BLAKE3 — the tree hash, one launch per block index and per level |
| `primitives.gemm` | exact int8 GEMM (3.8–4.0 TMAC/s, **9.0–9.3×** vs 16P CPU at n = 1024), the GF(2^k) GEMM substrate |

**1232 tests** (`pytest -q`; the number is in this line because prose cannot run it) —
NIST ACVP vectors, RFC 8439 vectors, Plonky3 Poseidon2 vectors, offline Falcon KATs, liboqs
cross-checks and independent numpy references. Each scheme has its own subpackage (`triton_crypto.mlkem`, `.mldsa`, `.slhdsa`,
`.lms`, `.falcon`) and so does the primitive layer; the top level re-exports the ML-KEM entry points for
backwards compatibility.

> **1.0 removed the pre-0.8 import paths.** `import mlkem_triton` and the old root modules
> (`triton_crypto.pipeline`, `.params`, `.kernels.*`, ...) are gone; the migration table is in
> [CHANGELOG.md](CHANGELOG.md). `from triton_crypto import Decapsulator` is unchanged.

## Install

Not on PyPI yet. From GitHub:

```bash
pip install "triton-crypto-kernels @ git+https://github.com/lwzhenglittle/triton-crypto-kernels.git"
```

Dependencies: `torch>=2.4`, `triton>=3.0`, `numpy`. The wheel is pure Python; nothing compiles.
For the test suite (ACVP vectors are not shipped in the wheel):

```bash
git clone https://github.com/lwzhenglittle/triton-crypto-kernels && cd triton-crypto-kernels
pip install -e ".[test,ref]" && pytest
```

## Quick start

### ML-KEM

```python
from triton_crypto import Decapsulator, Encapsulator, generate_keypairs

dec = Decapsulator("ML-KEM-768", dk)  # dk uploaded to the device once
keys = dec.decaps(ciphertexts)  # -> [B, 32] uint8

enc = Encapsulator("ML-KEM-768", ek)  # ek uploaded once
cts, keys = enc.encapsulate(messages)  # 32 bytes each -> [B, 1088] / [B, 32]

eks, dks = generate_keypairs("ML-KEM-768", 1024)
```

`decaps` / `encapsulate` accept `bytes`, `list[bytes]`, a 2-D numpy array or a device tensor;
`*_torch` variants keep the whole call on the device. Bigger batches pay off — launch overhead
dominates below B ≈ 64, and ML-KEM saturates around B ≈ 1024.

> **ML-KEM, not Kyber round-3.** FIPS 203 dropped encaps's KDF and keygen uses `G(d ‖ k)` rather
> than `G(d)`. Every correctness claim rests on the NIST ML-KEM ACVP vectors; there is no Kyber KAT.

### ML-DSA

```python
from triton_crypto.mldsa import Signer, Verifier

v = Verifier("ML-DSA-65", pk)  # pk preprocessing once
ok = v.verify(messages, signatures)  # -> [B] bool

s = Signer("ML-DSA-65", sk)  # expanded secret key
sigs = s.sign(messages)  # deterministic; any batch size
sigs = s.sign(messages, ctx=b"", prehash="SHA2-256", rnd=os.urandom(32))  # hedged
```

Verification is the batch-friendly half (one long-lived public key, many signatures). Signing is
also on-device: the rejection loop is compacted across the batch, and messages of different
lengths share a launch when their padding lands in the same number of SHAKE blocks.

### SLH-DSA and LMS/HSS

```python
from triton_crypto.slhdsa import Verifier

v = Verifier("SLH-DSA-SHAKE-128f", pk)
ok = v.verify(messages, signatures)  # ctx=..., prehash=... supported
```

```python
from triton_crypto.lms import Verifier

v = Verifier(hss_public_key)
ok = v.verify(messages, signatures)  # any HSS depth up to 8
```

### Falcon

```python
from triton_crypto.falcon import Verifier

v = Verifier(pk)  # auto-detects Falcon-512 (897 B) or Falcon-1024 (1793 B)
ok = v.verify(messages, signatures)  # -> [B] bool (padded=True accepts padded + unpadded)
```

Falcon verification is integer-only, so it is bit-exact by construction; signing needs the
floating-point FFT and Gaussian sampler, which this code base deliberately does not port.
`Verifier.verify` runs the entire pipeline on the device — prefix decoding (`_comp_decode_kernel`),
`HashToPoint` (for uniform-length batches), NTT multiplication, and the L2 norm check — reaching
**0.90–0.92 µs/op** end-to-end for Falcon-512 (**2.9×** vs 16P CPU) and **2.18–2.22 µs/op** for
Falcon-1024 (**2.3–2.4×** vs 16P CPU); `verify_core_torch` is the device-only entry point
(**0.32 µs/op** / **1.14–1.19 µs/op**).

SLH-DSA verification covers all twelve sets (SHAKE and SHA-2); signing is not implemented. LMS/HSS
verification covers both approved SHA-256 families; the SP 800-208 SHAKE-256 sets are not
implemented because their type codes are still TBD.

### Shared primitives

```python
from triton_crypto.primitives import KYBER, DILITHIUM, BABYBEAR_NTT

KYBER.q, KYBER.zeta, KYBER.layers, KYBER.wide  # (3329, 17, 7, False)
DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.layers, DILITHIUM.wide  # (8380417, 1753, 8, True)

from triton_crypto.primitives import chacha20_keystream, SHA3_256, hash_np_batch

ks = chacha20_keystream(keys, nonces, nbytes)  # [B, nbytes] uint8, RFC 8439
# keys: 32 bytes or [B, 32]; nonces: 12 bytes or [B, 12]; a batch of 1 broadcasts
digests = hash_np_batch(messages, SHA3_256)  # [B, 32] uint8
```

The same negacyclic NTT, parameterised by `(q, N, ζ, layers)`, serves both schemes — and a third
domain (BabyBear, for ZK work) needs no new kernel; the sponge provides SHA3/SHAKE at any rate and
output length, batched. **[docs/patterns.md](docs/patterns.md)** is the synthesis: the ten computational patterns the
project set out to cover, what each cost in Triton, and the three CPU-side factors that decide
whether a port is worth doing (dedicated instruction, SIMD width, and serial dependency chain).

`benchmarks/snapshot.py` runs everything — CPU rivals, the primitive table and the scheme
benchmarks — in one window and writes one timestamped markdown file.
`benchmarks/bench_primitive.py` prints one throughput table for this whole
layer against a measured copy bandwidth, and joins in `benchmarks/cpu_baseline_barrier.json` (from
`benchmarks/bench_16p_barrier.py`) as `vs 1T` / `vs 16P` columns — the rival that decides whether a
port is worth doing. `benchmarks/poseidon2_ref/` is a small cargo project on Plonky3 that both
emits the Poseidon2 known-answer vectors and supplies that CPU number.

## Performance

Representative device-side numbers; the baseline is measured on the same machine.

| workload | batch | this library | baseline |
|---|---:|---:|---|
| ML-KEM-768 decaps | 4096 | **132 ns/op** (GB10), **273 ns/op** (B390) | **13.8× dev / 4.5× e2e** (GB10); **6.7× dev / 3.3× e2e** (B390) vs 16P |
| ML-KEM-768 encaps | 4096 | **93 ns/op** (GB10), **189–194 ns/op** (B390) | 2.3× faster than cuPQC, **16.3× dev / 4.4× e2e** (GB10); **7.8× dev / 3.2× e2e** (B390) |
| ML-KEM-768 keygen | 4096 | **480 ns/op** (GB10), 1.46 µs/op (B390) | **2.9× dev / 2.2× e2e** (GB10); 0.95× dev / 0.77× e2e (B390) vs 16P |
| ML-DSA-44 verify | 4096 | **0.31 µs/op** (GB10), **2.0 µs/op** (B390) | **9.3× dev / 3.4× e2e** (GB10); **1.5× dev / 1.03× e2e** (B390) vs 16P |
| ML-DSA-44 sign | 1024 | **15.0 µs/sig** (GB10), **37.6 µs/sig** (B390) | 3.0× 1T / 0.56× 16P (GB10); 1.2× 1T / 0.22× 16P (B390) |
| SLH-DSA-SHAKE-128f verify | 4096 | **35.8 µs/op** (GB10), **51.7 µs/op** (B390) | **4.7×** (GB10), **3.3×** (B390) vs liboqs SHAKE-128f 16P (SHA2-128f: **4.0× / 3.9×**) |
| Falcon-512 verify | 4096 | **0.07 µs** dev / **0.42–0.44 µs** e2e (GB10); **0.32 µs/op** (B390) dev core, **0.90 µs** e2e | **38.4× dev / 6.1× e2e** (GB10); **8.3×** dev / **2.9×** e2e vs liboqs 16P, 71.7× dev vs 1T (B390) |
| Falcon-1024 verify | 4096 | **0.17 µs** dev / **0.74–0.79 µs** e2e (GB10); **1.19 µs/op** (B390) dev core, **2.22 µs** e2e | **30.6× dev / 6.6× e2e** (GB10); **4.4×** dev / **2.3×** e2e vs liboqs 16P, 36.7× dev vs 1T (B390) |
| BabyBear NTT, 2^20 points | 1 | **0.20 ms** (GB10), **1.58–1.61 ms** (B390; 0.85–1.6 ms range) | **26.6× dev / 1.6× e2e** (GB10); **3.3–3.4× dev** (up to 6.1×) vs 16P (B390) |
| BLAKE3, 1 KiB messages | 65536 | **25.7 GB/s** (GB10), **9.9 GB/s** (B390) | **1.6×** 16P (GB10); 7.4× vs 1T, 0.62× vs 16 processes (B390) |
| LMS/HSS (RFC 8554 case 0) | 4096 | **5.8 µs/op** (GB10), **10.6 µs/op** (B390; case 2: 7.1 µs) | no liboqs baseline |

GB10 = DGX Spark `sm_121`; B390 = Intel Arc B390 (Xe3-LPG). Device-side excludes H2D/D2H.
Every B390 row is from `benchmarks/snapshot-20261128T063642Z.md` and the full same-window
local GPU vs CPU matrix [benchmarks/local-gpu-vs-cpu-20261128.md](benchmarks/local-gpu-vs-cpu-20261128.md);
every GB10 CUDA row is from [benchmarks/cuda-gb10-vs-local-cpu-20261128.md](benchmarks/cuda-gb10-vs-local-cpu-20261128.md)
compared against the same 36-rival local 16P barrier x86 CPU baseline in `benchmarks/cpu_baseline_barrier.json`.
The B390 iGPU clock moves by up to 2× between sessions, so treat the ratios as the durable part, not the absolutes.

On the barrier 16P baseline (matched 1 KiB message sizes, one timed round per process behind a
barrier), **eight primitive rows** beat sixteen CPU processes device-side: exact int8 GEMM
**8.3–9.3×** (**3.7×** e2e), SHAKE-128 **7.8×** (**2.0×** e2e), SHAKE-256 **7.4×** (**2.1×** e2e),
SHA-512 **6.3–6.6×** (**2.0×** e2e), SHA3-512 **5.0–5.1×** (**2.1×** e2e), SHA3-256 **4.9–5.0×**
(**1.4×** e2e), NTT 2^20 **3.3–3.4×** (up to 6.1× across clock states), and SHA-256 **2.4–2.5×**
(0.49× e2e due to H2D). At the scheme level, **15 scheme operations beat sixteen CPU processes
end-to-end**: **ML-KEM-512/768/1024 encaps & decaps (1.5–8.4× dev, 1.1–3.8× e2e)**,
**ML-DSA-44/65/87 verify (1.5–2.0× dev, 1.0–1.4× e2e)**, **SLH-DSA SHAKE/SHA2-128f verify
(3.3× / 3.9×)**, and **all four Falcon variants — Falcon-512/padded-512 (8.3× dev, 2.9× e2e) and
Falcon-1024/padded-1024 (4.4–4.6× dev, 2.3–2.4× e2e)**.
Everything else stays at or below 1.0× (ChaCha20-Poly1305 **0.92×**, BLAKE3 **0.62×**, ChaCha20
**0.56×**, Poseidon2 **0.54×**, Poly1305 **0.30×**, AES-128-CTR **0.09×**). The before/after
comparison against the superseded sum-of-minima baseline (1.25–4.25× CPU over-count, median 1.94×)
is in [benchmarks/table-16p-20261127.md](benchmarks/table-16p-20261127.md). Full tables (including
Dev vs E2E splits, batch-size breakeven curves, and per-kernel breakdowns):
[docs/benchmarks.md](docs/benchmarks.md).

### Why ML-KEM is fast

ML-KEM's multiplication is folded into a fixed matrix `T(a)` with `T(a) @ b == a · b` in
`Z_q[X]/(X^256+1)`, so decapsulation is two GEMMs and encapsulation one. fp16 tensor cores stay
exact because one operand is small (`|W| ≤ 1664`, `|y| ≤ 2`), so every fp32 partial sum is below
2²⁴ and any accumulation order gives the integer result. ML-DSA cannot use this: q is 23 bits,
so products need 46 bits and the NTT is cheaper. Details: [docs/design.md](docs/design.md).

## Correctness and testing

**1232 tests**, six layers:

- **pure maths** — Toeplitz identity, fp16 exactness bound, NTT algebra for both domains;
- **kernels vs numpy** — every fused kernel byte-equal to its unfused equivalent;
- **ACVP** — ML-KEM, ML-DSA sigVer/keyGen/sigGen, SLH-DSA keyGen/sigVer, LMS sigVer;
- **independent oracles** — OpenSSL's `EVP_MAC` for Poly1305, `openssl enc` for AES,
  Plonky3 for Poseidon2, the official Rust BLAKE3, liboqs for Falcon;
- **cross-implementation** — byte-exact ML-KEM keygen against kyber-py, ML-DSA against liboqs;
- **packaging/API** — version, README imports, `find_packages()`, public exports, input handling.

```bash
pytest                                  # GPU cases skip centrally without a device
ruff check . && ruff format --check .
```

## Backends

| backend | status |
|---|---|
| Intel XPU (`triton-xpu`) | all five schemes and all 11 primitive families measured — Arc B390 |
| NVIDIA CUDA | all five schemes and all 11 primitive families measured — DGX Spark GB10 (`sm_121`) |
| AMD ROCm | untested, code-level reasoning only |
| Apple MPS / CPU | unsupported (no usable Triton backend) |

Detail, tile configurations and tuning knobs: [docs/support.md](docs/support.md).

## Limitations

- **Batch-first.** At B = 1, launch overhead dominates; below B ≈ 1024 on Intel a 16-thread CPU
  wins for ML-KEM. One long-lived key per object is the intended shape.
- **ML-KEM-512 has a slow path** (η1 = 3 uses generic CBD, ~3× slower); 768 and 1024 are fast.
- **ML-KEM needs fp16 tensor cores** for the fast path (fp32 fallback is 2–4× slower). ML-DSA,
  SLH-DSA and LMS do not — they are NTT/sponge workloads.
- **No formal constant-time validation** (no dudect/ctgrind, no cache analysis). Timing is
  ciphertext-independent; batch size is not.
- **SLH-DSA signing is not implemented** (the numpy reference covers keygen and verify only), and
  LMS/HSS is verification-only. SP 800-208 SHAKE-256
  LMS sets are not implemented.
- **ML-DSA signing is batch-bound**: ~43–45 µs/signature at B=1024, versus ~97 µs for one liboqs
  core; many signatures in one batch is the shape it optimises for.

More, including the keygen gap against cuPQC: [docs/compare.md](docs/compare.md) §5.

## Documentation

- [docs/kernels.md](docs/kernels.md) — every kernel's shape / dtype / grid contract
- [docs/design.md](docs/design.md) — Toeplitz argument, fp16 exactness proof, fusion catalogue
- [docs/benchmarks.md](docs/benchmarks.md) — all measured tables and how they were taken
- [docs/support.md](docs/support.md) — backend status and tuning knobs
- [docs/compare.md](docs/compare.md) — cuPQC, HI-Kyber, ConvKyber, TensorCrypto
- [CHANGELOG.md](CHANGELOG.md) — release notes
- [CONTRIBUTING.md](CONTRIBUTING.md) — setup, tests, a new backend, measurement discipline

## License

Apache-2.0. ML-KEM, ML-DSA and SLH-DSA are public NIST standards; the ACVP vectors come from
[usnistgov/ACVP-Server](https://github.com/usnistgov/ACVP-Server) (a US Government work, in the
public domain).
