# `vs 16P`: did the CPU column reproduce?

## Reproduction: the same column, measured again

Re-measured at 2026-11-27T17:24:50+0800 against the baseline above.
All 34 cells, second run / first run:

- median **0.997**, range 0.868..1.518
- 1T column: 0.868..1.036 -- this is the one that says whether the
  machine is in a comparable state, and it is the reason the first column can be trusted
- 16P column: 0.909..1.518 -- a fixed pile of work split
  sixteen ways waits for the slowest core, and which core that is moves between runs.
  **Per-row ratios on this column carry that band**; the 1T ratios do not.

| rival | 1T second/first | 16P second/first |
|---|---:|---:|
| `aes-128-ctr` | 0.998 | 0.928 |
| `blake3` | 0.996 | 0.989 |
| `chacha20` | 0.996 | 0.989 |
| `chacha20-poly1305` | 0.997 | 0.986 |
| `falcon-512-verify` | 1.003 | 0.909 |
| `falcon-padded-512-verify` | 1.036 | 1.061 |
| `mldsa-44-verify` | 0.999 | 1.022 |
| `ntt-2^20` | 0.999 | 0.917 |
| `poly1305` | 0.868 | 0.992 |
| `poseidon2` | 0.986 | 0.967 |
| `sha256` | 0.999 | 0.990 |
| `sha3-256` | 1.006 | 1.036 |
| `sha3-512` | 0.983 | 1.518 |
| `sha512` | 1.007 | 0.930 |
| `shake128` | 0.978 | 1.012 |
| `shake256` | 0.989 | 1.026 |
| `slhdsa-128f-verify` | 1.027 | 1.070 |
