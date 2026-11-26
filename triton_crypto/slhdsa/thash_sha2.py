"""SHA-2 `thash` for SLH-DSA (FIPS 205 §11.1), on the device.

`thash(PK.seed, ADRS, M) = SHA-2(blockpad(PK.seed) || ADRSc(ADRS) || M)` for one block.
The seed block is constant per parameter set and is absorbed on the host; this kernel
builds the 22-byte compressed address and the message in registers and runs the
compression loop. It lives in the scheme package because it is FIPS 205's construction,
not a generic SHA-2 primitive; `primitives/sha2.py` keeps the compression kernel.

"""

from __future__ import annotations

from .._codegen import compile_generated
from ..primitives.sha2 import _k_tensor

__all__ = ["gen_thash_source", "sha_thash"]


def gen_thash_source(bits: int, dn: int, nout_words: int, passthrough: bool) -> str:
    """A one-block SHA-2 kernel whose block is `ADRSc(22) || data(dn) || pad`, built in registers.

    SLH-DSA's SHA-2 instantiation hashes `blockpad(PK.seed) || ADRSc || M` with a 22-byte
    compressed address. The seed block is absorbed on the host (constant), so this kernel absorbs
    exactly one block whose first 22 bytes are the address — byte-shifted relative to any word
    boundary, which is why the composition is generated per `(bits, dn)` instead of done with
    copies.
    """
    W = bits // 8  # word bits (32 or 64)
    T = f"tl.uint{W}"
    rounds = 64 if bits == 256 else 80
    nw = 16  # words per block, both sizes
    bsize = nw * (W // 8)
    tail = 22 + dn
    total_msg = bsize + tail
    lenbytes = 8 if bits == 256 else 16
    if tail + 1 + lenbytes > bsize:
        raise ValueError(f"dn={dn} needs more than one SHA-{bits} block")
    if bits == 256:
        (s0a, s0b, s0s), (s1a, s1b, s1s) = (7, 18, 3), (17, 19, 10)
        (S0a, S0b, S0c), (S1a, S1b, S1c) = (2, 13, 22), (6, 11, 25)
        bs = "_bs32"
        it = "tl.int32"
    else:
        (s0a, s0b, s0s), (s1a, s1b, s1s) = (1, 8, 7), (19, 61, 6)
        (S0a, S0b, S0c), (S1a, S1b, S1c) = (28, 34, 39), (14, 18, 41)
        bs = "_bs64"
        it = "tl.int64"
    length = total_msg * 8
    # Whether the caller wants the n-byte value written back in place of the full state, and how
    # many words that is: the F case (dn == n) wants it, H and T_l want the whole 8-word state.
    n_bytes = nout_words * (W // 8) if passthrough else None

    def addr_byte(j: int) -> str:
        if j == 0:
            e = "(LAYER & 0xFF)"
        elif j <= 8:
            e = f"((TREE >> {8 * (8 - j)}) & 0xFF)"
        elif j == 9:
            e = "(TYPE & 0xFF)"
        elif j <= 13:
            e = f"((KP >> {8 * (13 - j)}) & 0xFF)"
        elif j <= 17:
            e = f"((CHAIN >> {8 * (17 - j)}) & 0xFF)"
        else:
            e = f"((INDEX >> {8 * (21 - j)}) & 0xFF)"
        return f"({e}).to({T})"

    def byte_expr(p: int) -> str:
        """Byte `p` of the block, as an expression over the kernel inputs."""
        if p < 22:
            return addr_byte(p)
        k = p - 22
        if k < dn:
            return f"tl.load(DATA + row * {dn} + {k}, mask=m, other=0).to({T})"
        if k == dn:
            return f"tl.full((BLOCK,), 0x80, {T})"
        if p >= bsize - lenbytes:
            shift = 8 * (bsize - 1 - p)
            return f"tl.full((BLOCK,), {(length >> shift) & 0xFF}, {T})"
        return f"tl.zeros((BLOCK,), dtype={T})"

    L = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def _bs32(x):",
        "    x = x.to(tl.uint32)",
        "    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | ((x >> 24) & 0xFF)",
        "",
        "@triton.jit",
        "def _bs64(x):",
        "    hi = _bs32((x >> 32) & 0xFFFFFFFF).to(tl.uint64)",
        "    lo = _bs32(x & 0xFFFFFFFF).to(tl.uint64)",
        "    return (lo << 32) | hi",
        "",
        "@triton.jit",
        "def _r(x, r):",
        f"    x = x.to({T})",
        f"    return (x >> r) | (x << ({W} - r))",
        "",
        "@triton.jit",
        "def sha_thash_kernel(HIN, KT, DATA, DWORDS, OUT, KP, CHAIN, INDEX, TREE, rows,",
        "                     LAYER: tl.int64, TYPE: tl.int64, T: tl.int64, W: tl.int64,",
        "                     N: tl.constexpr, BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    row = pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = row < rows",
        "    KP = tl.load(KP + row, mask=m, other=0).to(tl.int64)",
        "    CHAIN = tl.load(CHAIN + row, mask=m, other=0).to(tl.int64)",
        "    INDEX = tl.load(INDEX + row, mask=m, other=0).to(tl.int64) + T",
        "    active = INDEX < W",
        "    TREE = tl.load(TREE + row, mask=m, other=0).to(tl.int64)",
    ]
    L.append(f"    zero_v = tl.zeros((BLOCK,), dtype={T})")
    for i in range(8):
        L.append(f"    h{i} = tl.load(HIN + {i}).to({T}, bitcast=True) | zero_v")
    L.append("    a0, a1, a2, a3, a4, a5, a6, a7 = h0, h1, h2, h3, h4, h5, h6, h7")
    # message words, big-endian from byte expressions
    for w in range(nw):
        parts = [byte_expr(4 * w + k) if W == 32 else byte_expr(8 * w + k) for k in range(W // 8)]
        expr = " | ".join(
            f"(({p_expr}) << {8 * (W // 8 - 1 - i)})" for i, p_expr in enumerate(parts)
        )
        L.append(f"    w{w} = {expr}")
    L.append(f"    for _t in range(0, {rounds}, 1):")
    L.append(f"        kt = tl.load(KT + _t).to({T})")
    L.append(f"        S1 = _r(a4, {S1a}) ^ _r(a4, {S1b}) ^ _r(a4, {S1c})")
    L.append("        ch = (a4 & a5) ^ (~a4 & a6)")
    L.append("        t1 = a7 + S1 + ch + kt + w0")
    L.append(f"        S0 = _r(a0, {S0a}) ^ _r(a0, {S0b}) ^ _r(a0, {S0c})")
    L.append("        maj = (a0 & a1) ^ (a0 & a2) ^ (a1 & a2)")
    L.append("        t2 = S0 + maj")
    for i in range(7, 0, -1):
        L.append(f"        a{i} = a{i - 1}" if i != 4 else "        a4 = a3 + t1")
    L.append("        a0 = t1 + t2")
    L.append(
        f"        nw_ = w0 + (_r(w1, {s0a}) ^ _r(w1, {s0b}) ^ (w1 >> {s0s}))"
        f" + w9 + (_r(w14, {s1a}) ^ _r(w14, {s1b}) ^ (w14 >> {s1s}))"
    )
    for i in range(15):
        L.append(f"        w{i} = w{i + 1}")
    L.append("        w15 = nw_")
    for i in range(8):
        L.append(f"    h{i} = h{i} + a{i}")
    passthrough = n_bytes is not None
    # The F case writes only the n-byte value (n/ws words): that is all the chain needs, it keeps
    # OUT the same packed shape as DATA (so two buffers can ping-pong), and finished rows keep
    # their input — OUT must not alias DATA, since Triton would reorder loads after stores.
    nout = nout_words
    for i in range(nout):
        if passthrough:
            L.append(f"    keep = tl.load(DWORDS + row * {dn // (W // 8)} + {i}, mask=m, other=0)")
            L.append(
                f"    tl.store(OUT + row * {nout} + {i},"
                f" tl.where(active, {bs}(h{i}).to({it}, bitcast=True), keep), mask=m)"
            )
        else:
            L.append(
                f"    tl.store(OUT + row * 8 + {i}, {bs}(h{i}).to({it}, bitcast=True),"
                f" mask=m & active)"
            )
    return "\n".join(L) + "\n"


_THASH_CACHE: dict[tuple[int, int], object] = {}


def _compile_thash(bits: int, dn: int, nout_words: int, passthrough: bool):
    key = (bits, dn, nout_words, passthrough)
    kern = _THASH_CACHE.get(key)
    if kern is not None:
        return kern
    mod = compile_generated(gen_thash_source(bits, dn, nout_words, passthrough), "sha2_thash")
    _THASH_CACHE[key] = mod.sha_thash_kernel
    return mod.sha_thash_kernel


def sha_thash(
    hin,
    data,
    out_words,
    kp,
    chain,
    index,
    tree,
    layer: int,
    typ: int,
    bits: int,
    nout_words: int,
    passthrough: bool,
    t: int = 0,
    w: int = 1 << 40,
    block: int = 128,
):
    """Absorb one block `ADRSc || data || pad` starting from `hin`; `data` is [rows, dn] uint8."""
    import torch
    import triton

    rows, dn = data.shape
    kern = _compile_thash(bits, dn, nout_words, passthrough)
    dtype = torch.int32 if bits == 256 else torch.int64
    kern[(triton.cdiv(rows, block),)](
        hin,
        _k_tensor(bits, out_words.device),
        data,
        data.view(dtype),
        out_words,
        kp,
        chain,
        index,
        tree,
        rows,
        layer,
        typ,
        t,
        w,
        N=nout_words,
        BLOCK=block,
        num_warps=4,
    )
