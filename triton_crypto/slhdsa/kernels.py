"""A fused `thash` kernel: absorb `PK.seed || ADRS || data` and permute, in one launch.

Every hash in SLH-DSA verification is `SHAKE256(PK.seed || ADRS || data)` with all three pieces
shorter than one 136-byte block. Composing that buffer on the host and then calling the sponge
kernel costs several tensor writes per call — ~330 calls per 128f signature — and that overhead is
most of the runtime at small batches. This kernel builds the 17 state words directly from the
inputs (every field starts on an 8-byte boundary: n ∈ {16,24,32}, ADRS is 32 bytes) and runs the
24 rounds in registers, so one launch is the whole hash.

Contract:

    SEED   [n/8]        int64 words  (little-endian lane load, so pass the bytes as int64)
    DATA   [rows, dn/8] int64 words
    OUT    [rows, n/8]  int64 words
    KP     [rows]       int64   key pair address        (ADRS bytes 20..24)
    CHAIN  [rows]       int64   chain address / height  (ADRS bytes 24..28)
    INDEX  [rows]       int64   hash address / idx      (ADRS bytes 28..32)
    LAYER, TREE, TYPE   scalars; TREE is the int64 bit pattern of the 64-bit tree address.

The ADRS fields are written big-endian in the address (FIPS 205 §11.2), while the sponge reads
little-endian lanes, so each 32-bit field is byte-swapped into its lane.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

from ..primitives.sponge import _emit_rounds, _rc_tensor

__all__ = ["thash"]

_CACHE: dict[tuple[int, int], object] = {}


def gen_source(n: int, dn: int) -> str:
    """One kernel per (hash size, data size): both are constexpr, so the padding is static."""
    assert n % 8 == 0 and dn % 8 == 0
    nw_seed, nw_data, nw_out = n // 8, dn // 8, n // 8
    length = n + 32 + dn  # bytes absorbed before pad10*1
    assert length < 136, "one block only"
    pad_word, pad_shift = length // 8, (length % 8) * 8

    L = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def _bs32(x):",
        "    x = x.to(tl.uint64)",
        "    return (((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00)"
        " | ((x >> 24) & 0xFF))",
        "",
        "@triton.jit",
        "def _bs64(x):",
        "    return _bs32(x & 0xFFFFFFFF) << 32 | _bs32((x >> 32) & 0xFFFFFFFF)",
        "",
        "@triton.jit",
        "def thash_kernel(SEED, OUT, DATA, RCP, KP, CHAIN, INDEX, TREE, rows,",
        "                 LAYER: tl.int64, TYPE: tl.int64, T: tl.int64, W: tl.int64, ROW0: tl.int64,",
        "                 BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    row = ROW0 + pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = row < rows",
    ]
    # ADRS words (bytes 0..32), each written big-endian by the scheme and read little-endian here
    # layout: seed (n bytes) || ADRS (32 bytes) || data (dn bytes), all 8-byte aligned
    L.append("    zero_v = tl.zeros((BLOCK,), dtype=tl.uint64)")  # broadcast scalars to vectors
    for i in range(nw_seed):
        L.append(f"    a{i} = tl.load(SEED + {i}).to(tl.uint64, bitcast=True) | zero_v")
    L.append("    w0 = _bs32(LAYER)")
    L.append("    w1 = _bs64(tl.load(TREE + row, mask=m, other=0))")
    L.append("    w2 = _bs32(tl.load(KP + row, mask=m, other=0)) << 32")
    L.append("    w2 = w2 | _bs32(TYPE)")
    L.append("    addr = tl.load(INDEX + row, mask=m, other=0) + T")
    L.append("    active = addr < W")
    L.append("    w3 = _bs32(tl.where(active, addr, 0)) << 32")
    L.append("    w3 = w3 | _bs32(tl.load(CHAIN + row, mask=m, other=0))")
    L.append(f"    a{nw_seed} = w0 | zero_v")
    L.append(f"    a{nw_seed + 1} = w1 | zero_v")
    L.append(f"    a{nw_seed + 2} = w2 | zero_v")
    L.append(f"    a{nw_seed + 3} = w3 | zero_v")
    for i in range(nw_data):
        L.append(
            f"    m{i} = tl.load(DATA + row * {nw_data} + {i}, mask=m,"
            f" other=0).to(tl.uint64, bitcast=True)"
        )
        L.append(f"    a{nw_seed + 4 + i} = m{i}")

    used = nw_seed + 4 + nw_data  # words filled so far
    for i in range(used, 17):
        L.append(f"    a{i} = tl.zeros((BLOCK,), dtype=tl.uint64)")
    for i in range(17, 25):
        L.append(f"    a{i} = tl.zeros((BLOCK,), dtype=tl.uint64)")

    # pad10*1: 0x1F right after the message, 0x80 in the last byte of the block
    assert pad_word < 17
    L.append(f"    a{pad_word} = a{pad_word} | (tl.full((BLOCK,), 0x1F, tl.uint64) << {pad_shift})")
    L.append("    a16 = a16 | (tl.full((BLOCK,), 0x80, tl.uint64) << 56)")

    # One rolled loop over the 24 rounds, like the sponge kernel: the round body is identical
    # except for the constant, so inlining 24 copies would only make the source ~2000 lines and
    # the compile slow. Constants come from a tensor because some do not fit a signed int64.
    L.append("    for _r in range(0, 24, 1):")
    L.append("        rc = tl.load(RCP + _r)")
    _emit_rounds(L, 0, ind="        ", rc_expr="rc")

    # inactive rows keep their input (this also makes OUT == DATA safe: per-row read-then-write)
    for i in range(nw_out):
        if i < nw_data:
            L.append(f"    a{i} = tl.where(active, a{i}, m{i})")
        L.append(
            f"    tl.store(OUT + row * {nw_out} + {i}, a{i}.to(tl.int64, bitcast=True), mask=m)"
        )
    return "\n".join(L) + "\n"


def _compile(n: int, dn: int):
    key = (n, dn)
    kern = _CACHE.get(key)
    if kern is not None:
        return kern
    src = gen_source(n, dn)
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_slhdsa_thash_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_slhdsa_thash_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[key] = mod.thash_kernel
    return mod.thash_kernel


def thash(
    seed_words,
    data,
    out_words,
    kp,
    chain,
    index,
    tree,
    layer,
    typ,
    n,
    t=0,
    w=1 << 40,
    row0=0,
    block=128,
):
    """Launch the fused thash over `rows = data.shape[0]` rows; `out_words` is [rows, n/8]."""
    import triton

    rows = data.shape[0]
    kern = _compile(n, data.shape[1] * 8)  # data is an int64 view
    # `row0` lets the chain loop launch only over the rows whose chain is still running
    kern[(max(1, triton.cdiv(rows - row0, block)),)](
        seed_words,
        out_words,
        data,
        _rc_tensor(out_words.device),
        kp,
        chain,
        index,
        tree,
        rows,
        layer,
        typ,
        t,
        w,
        row0,
        BLOCK=block,
        num_warps=4,
    )
    return out_words
