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
| Falcon | verify (Falcon-512 and Falcon-padded-512) | FIPS 206 draft |

And the scheme-agnostic primitives underneath them, usable on their own:

| primitive | what |
|---|---|
| `primitives.sponge` | Keccak-f[1600] and the sponge: SHA3-224/256/384/512, SHAKE128/256, any rate, batched |
| `primitives.sha2` | SHA-256 / SHA-512 compression |
| `primitives.ntt` | negacyclic NTT over `Z_q[X]/(X^N+1)`, parameterised by `(q, N, ζ, layers)` |
| `primitives.fields` | prime-field arithmetic: the 32-bit Montgomery product, shared by every modular primitive |
| `primitives.chacha20` | ChaCha20 (RFC 8439) — the first of the non-PQC primitives |
| `primitives.poseidon2` | Poseidon2-BabyBear/16 (Plonky3 parameters, 8 full + 13 partial rounds) |
| `primitives.poly1305` | Poly1305 (RFC 8439) and the ChaCha20-Poly1305 AEAD |
| `primitives.aes` | AES-128-CTR through a T-table — a gather measurement, not a fast path |

**1031 tests** — NIST ACVP vectors, RFC 8439 vectors, Plonky3 Poseidon2 vectors, liboqs
cross-checks and independent numpy references. Each scheme has its own subpackage (`triton_crypto.mlkem`, `.mldsa`, `.slhdsa`,
`.lms`) and so does the primitive layer; the top level re-exports the ML-KEM entry points for
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

Falcon verification is integer-only, so it is bit-exact by construction; signing needs the
floating-point FFT and Gaussian sampler, which this code base deliberately does not port.
`Verifier.verify` decodes signatures and hashes on the host today, so its end-to-end number is
host-bound (31 µs/op); `verify_core_torch` is the device-only entry point (0.65 µs/op).

SLH-DSA verification covers all twelve sets (SHAKE and SHA-2); signing is not implemented. LMS/HSS
verification covers both approved SHA-256 families; the SP 800-208 SHAKE-256 sets are not
implemented because their type codes are still TBD.

### Shared primitives

```python
from triton_crypto.primitives import KYBER, DILITHIUM, BABYBEAR_NTT

KYBER.q, KYBER.zeta, KYBER.layers, KYBER.wide  # (3329, 17, 7, False)
DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.layers, DILITHIUM.wide  # (8380417, 1753, 8, True)

from triton_crypto.primitives import chacha20_keystream, SHA3_256, hash_np_batch

ks = chacha20_keystream(key, nonce, nbytes)  # [B, nbytes] uint8, RFC 8439
digests = hash_np_batch(messages, SHA3_256)  # [B, 32] uint8
```

The same negacyclic NTT, parameterised by `(q, N, ζ, layers)`, serves both schemes — and a third
domain (BabyBear, for ZK work) needs no new kernel; the sponge provides SHA3/SHAKE at any rate and
output length, batched. `benchmarks/bench_primitive.py` prints one throughput table for this whole
layer against a measured copy bandwidth, and joins in `benchmarks/cpu_baseline.json` (from
`benchmarks/bench_cpu.py`) as `vs 1T` / `vs 16P` columns — the rival that decides whether a
port is worth doing. `benchmarks/poseidon2_ref/` is a small cargo project on Plonky3 that both
emits the Poseidon2 known-answer vectors and supplies that CPU number.

## Performance

Representative device-side numbers; the baseline is measured on the same machine.

| workload | batch | this library | baseline |
|---|---:|---:|---|
| ML-KEM-768 decaps | 4096 | **125 ns/op** (GB10) | 6.8× vs liboqs 16T |
| ML-KEM-768 encaps | 4096 | **88 ns/op** (GB10) | 2.3× faster than cuPQC |
| ML-KEM-768 keygen | 4096 | 469 ns/op (GB10) | 2.7× slower than cuPQC |
| ML-DSA-44 verify | 4096 | **1.8 µs/op** (B390) | 10.9× vs liboqs 1T |
| ML-DSA-44 sign | 1024 | **43 µs/sig** (B390) | 2.3× vs liboqs 1T |
| SLH-DSA-SHAKE-128f verify | 4096 | **47.8 µs/op** (B390) | 27.9× vs liboqs 1T |
| Falcon-512 verify | 4096 | **0.65 µs/op** (B390) | 3.2× vs liboqs 16P, 35× vs 1T |
| LMS/HSS (RFC 8554 case 2) | 4096 | **7.1 µs/op** (B390) | no liboqs baseline |

GB10 = DGX Spark `sm_121`; B390 = Intel Arc B390 (Xe3-LPG). Device-side excludes H2D/D2H.
GPU clocks drift 2–3× between sessions, so only same-process A/B ratios are trusted. Full tables,
per-kernel breakdowns and measurement conventions: [docs/benchmarks.md](docs/benchmarks.md).

### Why ML-KEM is fast

ML-KEM's multiplication is folded into a fixed matrix `T(a)` with `T(a) @ b == a · b` in
`Z_q[X]/(X^256+1)`, so decapsulation is two GEMMs and encapsulation one. fp16 tensor cores stay
exact because one operand is small (`|W| ≤ 1664`, `|y| ≤ 2`), so every fp32 partial sum is below
2²⁴ and any accumulation order gives the integer result. ML-DSA cannot use this: q is 23 bits,
so products need 46 bits and the NTT is cheaper. Details: [docs/design.md](docs/design.md).

## Correctness and testing

**891 tests**, five layers:

- **pure maths** — Toeplitz identity, fp16 exactness bound, NTT algebra for both domains;
- **kernels vs numpy** — every fused kernel byte-equal to its unfused equivalent;
- **ACVP** — ML-KEM, ML-DSA sigVer/keyGen/sigGen, SLH-DSA keyGen/sigVer, LMS sigVer;
- **cross-implementation** — byte-exact ML-KEM keygen against kyber-py, ML-DSA against liboqs;
- **packaging/API** — version, README imports, `find_packages()`, public exports, input handling.

```bash
pytest                                  # GPU cases skip centrally without a device
ruff check . && ruff format --check .
```

## Backends

| backend | status |
|---|---|
| Intel XPU (`triton-xpu`) | all four schemes measured — Arc B390 |
| NVIDIA CUDA | ML-KEM measured (DGX Spark GB10); other schemes not run |
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
