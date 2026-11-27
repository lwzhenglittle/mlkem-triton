# Contributing

## Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,ref]"
```

`[ref]` pulls in `kyber-py`; only part of the suite needs it (random key-pair generation and
byte-exact cross-checking of ML-KEM keygen).

ML-DSA's cross-implementation test (`tests/test_mldsa.py::test_matches_liboqs`) needs **liboqs
with ML-DSA enabled**, which is not a pip dependency. It skips cleanly when absent; to enable it:

```bash
cd third_party/liboqs            # any liboqs 0.16+ checkout works
cmake -B build-mldsa -GNinja -DCMAKE_BUILD_TYPE=Release \
      -DOQS_MINIMAL_BUILD="SIG_ml_dsa_44;SIG_ml_dsa_65;SIG_ml_dsa_87" \
      -DOQS_BUILD_ONLY_LIB=ON -DBUILD_SHARED_LIBS=ON
ninja -C build-mldsa             # -> lib/liboqs.so
# point the tests elsewhere with MLDSA_LIBOQS_SO=/path/to/liboqs.so
```

That library is also the CPU baseline in `benchmarks/bench_mldsa_verify.py`, and it is what makes
the ML-DSA numbers in `docs/benchmarks.md` §11 cross-implementation rather than self-referential.

SLH-DSA has the same arrangement under a different variable: `SLHDSA_LIBOQS_SO` points at a
liboqs build compiled with SLH-DSA, and both `tests/_liboqs_slh.py` and
`benchmarks/bench_slhdsa_verify.py --cpu` read it. With nothing set, the benchmark falls back to
whichever `third_party/liboqs/build*/lib/liboqs.so` exports the symbol it needs (a build without
SLH-DSA is skipped rather than used), and it says so on stderr when no candidate qualifies.

Hardware:

- **To run the whole suite**: a GPU with a Triton backend (NVIDIA / AMD / Intel). Without a GPU
  `pytest tests` still runs because `tests/conftest.py` skips the device-dependent cases
  centrally. GPU CI is self-hosted rather than on hosted runners — please also state in your PR
  which hardware you ran.
- **Just reading the code or running the reference implementation**: CPU is enough;
  `triton_crypto.mlkem.ntt_ref` and `triton_crypto.mlkem.keyprep` are pure numpy.

## Tests

```bash
pytest                      # everything
pytest --device cuda        # pin a backend
pytest tests/test_glue.py   # one file
ruff check . && ruff format --check .
```

Five layers:

1. **Pure maths** (`test_ntt.py`, `test_keyprep.py`, `test_dot_exact.py`, `test_primitives.py`) —
   the Toeplitz identity, NTT algebra for both `(q, N, ζ)` domains against a schoolbook
   convolution, the exactness bound of fp16 accumulation, sponge padding.
2. **Kernel bit-alignment** (`test_fused_gemm.py`, `test_glue.py`, `test_encaps.py`,
   `test_keygen.py`, `test_rebuild.py`) — every fused kernel must be **byte-equal to the unfused
   multi-step implementation**. **This is the guard against silent fallback**: a fused kernel
   returns `False`/`None` and writes no output when its constraints are unmet, and the caller
   falls back silently. If the guard condition is ever written wrong, this layer goes red
   immediately. (We did hit this during development: `Y` was a transposed view, so the fused
   GEMM was never executing.)
3. **Reference-implementation agreement** (`test_mldsa_ref.py`) — the numpy FIPS 204
   implementation against ACVP keyGen (byte-exact), sigGen (byte-exact, deterministic *and*
   hedged) and sigVer (180/180), plus the codec edge cases.
4. **Hash-based schemes** (`test_slhdsa*.py`, `test_lms*.py`) — all twelve SLH-DSA sets and both
   SHA-256 LMS families against ACVP, including header-tampering cases.
5. **End-to-end and packaging** (`test_pipeline*.py`, `test_mldsa*.py`, `test_api_contract.py`) —
   official NIST ACVP vectors, implicit rejection, every batch size, tampered signatures and
   hint blocks; version metadata, README imports and `find_packages()`.

## Adding a backend

Only `triton_crypto/devices.py` needs to change:

1. add the device type to `available_devices()` / `_check_available()`;
2. add a tile configuration to `_GEMM_CFG`. It must satisfy the divisibility constraints
   (`M % BM == 0`, `K % BK == 0`, …) checked by `fused_gemm._pick_cfg`;
3. add a synchronisation branch to `sync()`;
4. small-kernel `num_warps` defaults live at the top of each module and can be overridden
   temporarily with `MLKEM_WARPS="shake=8,ntt=4"`. Sweep first, then write the value back as the
   default.

**State the exact hardware model you measured in the PR.** Do not submit configurations that
are only "supposed to be faster".

## Packaging

```bash
python -m build --wheel          # the release path: builds from the sdist, in isolation
```

**Do not use `pip wheel .` or `setup.py bdist_wheel` for a release unless you have removed
`build/` first.** `build_py` copies the package tree into `build/lib/` and does not delete files
that have disappeared from the source, so a wheel built on top of a stale `build/` ships modules
that no longer exist — this actually happened when 1.0.0 removed the pre-0.8 layout: the wheel
still contained `mlkem_triton/`, `triton_crypto/params.py` and `triton_crypto/kernels/`.
The `package` job in CI asserts both directions (every required module present, no removed path
present) precisely because a dirty build directory is invisible in the source tree.

## Measurement discipline (this project has been burned; follow it)

- **Only same-process interleaved A/B ratios are trustworthy.** GPU clocks differ by 2–4×
  between sessions (this machine's Intel iGPU idles at 0.9 GHz and loads at 2.2 GHz, and the
  `xe` driver exposes no frequency control).
- Report **min-of-N**, never the mean. The `_pre` kernel here has 26× cold-start spread, and the
  ranking by share computed from means is **the exact opposite** of the one computed from mins.
- **Confirm the optimised path actually ran.** Fused kernels fall back silently, and
  cross-process comparisons turn clock noise into apparent gains.
- Use the scripts in `benchmarks/`; they self-check fused against the fully-fallback path.
- Every benchmark number must be accompanied by: hardware model, software versions, batch size,
  and timing convention (pure kernel vs host-included).
- **Pair the device number with a CPU baseline measured in the same window.** `bench_cpu.py`
  writes `benchmarks/cpu_baseline.json` (one thread and sixteen concurrent processes per rival) and
  `bench_primitive.py` turns it into the `vs 1T` / `vs 16P` columns. A ratio whose two sides came
  from different windows is not a ratio.
- **`benchmarks/snapshot.py` runs everything in one window** (CPU rivals, the primitive table, the
  four scheme benchmarks; about two minutes) and writes a timestamped markdown plus a JSON header.
  Prefer one snapshot over a set of hand-collected numbers.
- **Quote the traffic a row actually moves.** One number here was overstated by 2x for a while
  because the benchmark derived it from the radix-2 pass count after the transform had moved to
  radix-4; the traffic, the GB/s and the `%copy` were all wrong together. If a figure is computed,
  take its inputs from the implementation (`LargeNTT.n_passes`, not a literal).
- **A checker must contain a case that fails.** `check-docs.py` in the workspace next to this
  repository went two rounds reporting PASS while one of its parsers matched nothing; it now has
  `--selftest`. The same applies to tests: a loop over a loaded corpus passes vacuously when the
  corpus is empty, which is why `tests/test_primitives_contract.py` asserts the corpora are
  populated.

## Where the decisions are written down

- `docs/patterns.md` — which computational patterns are worth porting, with the measured
  `vs 16P` column and the rule that decides it (the CPU's instruction set, not the GPU's speed).
- `docs/kernels.md` — one contract per kernel: tensors, dtypes, launch shape, and the Triton traps
  each one hit. A kernel that is not named there fails the suite.
- `docs/benchmarks.md` — the measured record, including the ideas that lost (`§22` has two of them).
- `docs/support.md` — what runs on which backend, with "not run" marked as such.

## Commits

- run `ruff format .`
- it makes the diff large, so keep it in its own commit
- performance changes must include paired before/after ratios and the hardware they ran on
