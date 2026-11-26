# poseidon2_ref — Plonky3's Poseidon2-BabyBear/16, for the vectors and the CPU rival

`cargo run --release -- kat > ../../../data/vectors/poseidon2/bb16.json` emits **64** known-answer
vectors from Plonky3 0.8's `default_babybear_poseidon2_16()`: the structured edge cases (the
0..15 ramp, seven offsets that stress the reduction paths, all-`q-1`) plus 55 deterministic
pseudo-random states from an LCG. The random states were added after the first nine-vector file
passed while a block-boundary bug could still have existed — the nine cases only exercise the
edges of the field, not the middle.

`cargo run --release -- bench-packed` is the CPU rival in `benchmarks/cpu_baseline.json`
(AVX2 monty-31x8: 106-121 ns/permutation one thread, ~7 ns over sixteen processes). Use
`RUSTFLAGS="-C target-cpu=native"` when building, otherwise `Packing::WIDTH` degrades to 1 and the
"packed" benchmark silently measures the scalar path.

The cargo build output is gitignored; the crate is fetched from crates.io, so no Plonky3
checkout is needed.
