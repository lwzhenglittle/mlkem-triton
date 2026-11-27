# Archived snapshots (sum-of-minima CPU baseline era)

These seven snapshots (`2026-11-26T14:09` through `2026-11-26T17:16`) were produced
against `benchmarks/cpu_baseline.json`, whose 16-process column summed per-child minima
and over-counted the CPU by 1.25x–4.25x (median 1.94x). Their `ntt-2^20` rows also
predate the `LargeNTT.n_passes` traffic fix (`20 passes` instead of radix-4 `10 passes`).

They are preserved here so historical figures in `report-2026-11-27.md` §2 and §2.2
remain verifiable. For current barrier-corrected measurements, use the latest
`benchmarks/snapshot-*.md` in the parent directory.
