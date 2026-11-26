# blake3_ref — the BLAKE3 reference, kept for the device path that is not written yet

`cargo run --release -- kat` writes `data/vectors/blake3/kat.json`: the official Rust
implementation's digests for 26 message lengths from 0 to 64 KiB, including the
non-power-of-two chunk counts where the tree shape is the thing being tested.
`bench` times it (2.93 GB/s on this machine, one thread) for the CPU column.

The **numpy reference** was written and is correct against all 26 vectors (the tree rule is
"pair adjacent chaining values, carry an odd node up unchanged"). The **Triton path is not
written**: the chunk loop has to carry eight chaining words across up to sixteen block
compressions, and Triton does not carry tuple assignments across a runtime loop — the rewrite
that carries them as named scalars compiled but returned the initial chaining value, and it was
dropped rather than shipped unproven. Next session: make `_compress` return into named scalars
and store the block index as a *runtime* argument (`_load_block` currently takes it as
`tl.constexpr`, which cannot receive a loop variable).

Everything here is unused by the test suite on purpose: it is the hard part of the next attempt
(the vectors), not a half-finished module.
