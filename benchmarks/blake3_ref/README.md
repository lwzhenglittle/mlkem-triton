# blake3_ref — the BLAKE3 reference, kept for the device path that is not written yet

`cargo run --release -- kat > ../../../data/vectors/blake3/kat.json` writes the official Rust
implementation's digests for 26 message lengths from 0 to 64 KiB, including the
non-power-of-two chunk counts where the tree shape is the thing being tested.
`bench` times it (2.93 GB/s on this machine, one thread) for the CPU column.

## Where the device path stands (2026-11-26)

The **numpy reference is correct against all 26 vectors**, and the tree rule is established:
pair adjacent chaining values, carry an odd node up unchanged.

The **Triton path is not written**, and three attempts narrowed it down to a specific, odd place.
Each piece works in isolation:

1. the compression function is bit-exact against the reference (`206 tests` worth of confidence:
   every one of the 26 vectors passes through it in numpy, and a device kernel computing one
   compression matched the reference CV word for word);
2. a **single block step through a global chaining-value buffer** matched the reference exactly
   (kernel: load CV, load 16 block words, compress, store CV; verified for a 64-byte chunk);
3. a **multi-launch chain through a global buffer** works (a toy kernel adding 1 to a buffer
   sixteen times read back +16).

But when those exact pieces are assembled into the module's `_block_kernel`, the kernel stores
zeros: the store mask, block count, flags (11 = CHUNK_START|CHUNK_END|ROOT) and block length (64)
all read back correct from inside the same launch, while the value the kernel stores -- and the
root buffer -- come out zero.

A bisect then ruled out the most suspicious feature. The probe kernel was extended one step at a
time:

| variant | result |
|---|---|
| A: probe kernel, one destination | MATCH |
| B: + root output buffer, `ROOT_MODE=False` | MATCH |
| B: + `ROOT_MODE=True` (flags gain ROOT, two destinations, `r[8+i]` as well) | produces the *correct* value for those flags |

So the dual-destination store and the constexpr branch are both innocent, and the difference is
somewhere in the module's block-address / `nblocks` / `blen` plumbing. The next step is a
variant C that is the module's kernel verbatim, then deleting one of those pieces per run --
half an hour of mechanical work, not a redesign, and it needs its own session rather than the
tail of a long one. The module was deleted rather than shipped unproven, three times.

What is *not* worth re-deriving next time: the three Triton constraints that shaped the design.
A runtime loop cannot carry a tuple assignment (so the chaining value has to live in a global
buffer); a `tl.constexpr` parameter cannot receive a loop variable (so the block index must be a
runtime argument); and the `tl.static_range(16)` variant compiles the IR for more than ten
minutes on the Xe backend, which is why the design is one launch per block index.

Everything here is unused by the test suite on purpose: it is the hard part of the next attempt
(the vectors), not a half-finished module.
