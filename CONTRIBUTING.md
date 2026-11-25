# Contributing

## Environment

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[test,ref]"
```

`[ref]` pulls in `kyber-py`; only part of the suite needs it (random key-pair generation and
byte-exact cross-checking).

Hardware:

- **To run the whole suite**: a GPU with a Triton backend (NVIDIA / AMD / Intel). Without a GPU
  `pytest tests` still runs (90 passed / 174 skipped) because `tests/conftest.py` skips the
  device-dependent cases centrally. **This project has no GPU CI** — the tests are run by hand
  on real machines, so please state in your PR which hardware you ran.
- **Just reading the code or running the reference implementation**: CPU is enough;
  `mlkem_triton.ntt_ref` and `keyprep` are pure numpy.

## Tests

```bash
pytest                      # everything
pytest --device cuda        # pin a backend
pytest tests/test_glue.py   # one file
ruff check . && ruff format --check .
```

Three layers:

1. **Pure maths** (`test_ntt.py`, `test_keyprep.py`, `test_dot_exact.py`) — the Toeplitz
   identity, NTT self-consistency, the exactness bound of fp16 accumulation.
2. **Kernel bit-alignment** (`test_fused_gemm.py`, `test_glue.py`, `test_encaps.py`,
   `test_keygen.py`, `test_rebuild.py`) — every fused kernel must be **byte-equal to the unfused
   multi-step implementation**. **This is the guard against silent fallback**: a fused kernel
   returns `False`/`None` and writes no output when its constraints are unmet, and the caller
   falls back silently. If the guard condition is ever written wrong, this layer goes red
   immediately. (We did hit this during development: `Y` was a transposed view, so the fused
   GEMM was never executing.)
3. **End-to-end** (`test_pipeline*.py`) — official NIST ACVP vectors, implicit rejection, every
   batch size.

## Adding a backend

Only `mlkem_triton/devices.py` needs to change:

1. add the device type to `available_devices()` / `_check_available()`;
2. add a tile configuration to `_GEMM_CFG`. It must satisfy the divisibility constraints
   (`M % BM == 0`, `K % BK == 0`, …) checked by `fused_gemm._pick_cfg`;
3. add a synchronisation branch to `sync()`;
4. small-kernel `num_warps` defaults live at the top of each module and can be overridden
   temporarily with `MLKEM_WARPS="shake=8,ntt=4"`. Sweep first, then write the value back as the
   default.

**State the exact hardware model you measured in the PR.** Do not submit configurations that
are only "supposed to be faster".

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

## Commits

- run `ruff format .`
- it makes the diff large, so keep it in its own commit
- performance changes must include paired before/after ratios and the hardware they ran on
