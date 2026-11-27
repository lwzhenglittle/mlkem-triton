#!/bin/sh
# Build every CPU rival binary the benchmarks need, from the source each one is meant to come
# from, and print one line of proof from each.
#
# Why this exists: `benchmarks/ntt_cpu` is built from `ntt_large_cpu_rival_r4.c`, not from
# `ntt_large_cpu_rival.c`. Rebuilding it from the radix-2 source is a silent 1.48x slowdown of the
# rival -- 48.5 ms becomes 71.5 ms at 2^20 -- which moves every `vs 16P` NTT ratio in the opposite
# direction. That happened once, during a benchmark run, and nothing flagged it: the binary has the
# same name and the same output format. One script, one source per binary, checked by eye.
set -eu
cd "$(dirname "$0")"

CC=${CC:-gcc}
FLAGS="-O3 -march=native"

echo "== ntt_cpu (radix-4 rival: the one the 16P column was measured with)"
$CC $FLAGS -fopenmp -o ntt_cpu ntt_large_cpu_rival_r4.c
OMP_NUM_THREADS=1 ./ntt_cpu 20

echo "== ntt_cpu_radix2 (kept for the radix-2/4 comparison only, not the denominator)"
$CC $FLAGS -fopenmp -o ntt_cpu_radix2 ntt_large_cpu_rival.c
OMP_NUM_THREADS=1 ./ntt_cpu_radix2 20

echo "== evp_cpu (digests, ciphers and Poly1305 through libcrypto, in C)"
$CC $FLAGS -o evp_cpu evp_cpu_rival.c -lcrypto
./evp_cpu sha256 1024 134217728

echo "== rust refs (message shape the device uses: 1 KiB)"
( cd blake3_ref && RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet )
( cd poseidon2_ref && RUSTFLAGS="-C target-cpu=native" cargo build --release --quiet )
./blake3_ref/target/release/blake3-ref bench-total 1024 200000
./poseidon2_ref/target/release/poseidon2-ref bench-packed-total 2000000
