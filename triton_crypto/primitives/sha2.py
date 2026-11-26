"""SHA-256 / SHA-512 compression: a device kernel that absorbs pre-padded blocks, plus the numpy
primitives used to check it and to precompute constant states.

These are what the **SHA-2 parameter sets of FIPS 205** use on device (`slhdsa/verify.py`
dispatches to `sha_thash` / `sha_absorb` when the set is not SHAKE). Validated against numpy and
`hashlib` in `tests/test_sha2.py`, and end-to-end against the ACVP vectors for all six SHA-2 sets.

SLH-DSA's SHA-2 parameter sets need these. Their hashes are `SHA(PK.seed || 0…0 || ADRSc || M)`
and `MGF1-SHA`; both are *fixed* prefixes, so the state after the seed block is computed once on
the host and the kernel only absorbs what changes.

The kernel takes the message as a pre-padded byte buffer viewed as little-endian words; it swaps
each word to big-endian on load (SHA-2 is big-endian) and swaps back on store. That keeps the
caller's job to "copy the bytes in the right order", which torch is good at.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

__all__ = ["IV256", "IV512", "K256", "K512", "compress_np", "sha_absorb", "sha_thash"]

IV256 = [
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
]
IV512 = [
    0x6A09E667F3BCC908,
    0xBB67AE8584CAA73B,
    0x3C6EF372FE94F82B,
    0xA54FF53A5F1D36F1,
    0x510E527FADE682D1,
    0x9B05688C2B3E6C1F,
    0x1F83D9ABFB41BD6B,
    0x5BE0CD19137E2179,
]

K256 = [
    0x428A2F98,
    0x71374491,
    0xB5C0FBCF,
    0xE9B5DBA5,
    0x3956C25B,
    0x59F111F1,
    0x923F82A4,
    0xAB1C5ED5,
    0xD807AA98,
    0x12835B01,
    0x243185BE,
    0x550C7DC3,
    0x72BE5D74,
    0x80DEB1FE,
    0x9BDC06A7,
    0xC19BF174,
    0xE49B69C1,
    0xEFBE4786,
    0x0FC19DC6,
    0x240CA1CC,
    0x2DE92C6F,
    0x4A7484AA,
    0x5CB0A9DC,
    0x76F988DA,
    0x983E5152,
    0xA831C66D,
    0xB00327C8,
    0xBF597FC7,
    0xC6E00BF3,
    0xD5A79147,
    0x06CA6351,
    0x14292967,
    0x27B70A85,
    0x2E1B2138,
    0x4D2C6DFC,
    0x53380D13,
    0x650A7354,
    0x766A0ABB,
    0x81C2C92E,
    0x92722C85,
    0xA2BFE8A1,
    0xA81A664B,
    0xC24B8B70,
    0xC76C51A3,
    0xD192E819,
    0xD6990624,
    0xF40E3585,
    0x106AA070,
    0x19A4C116,
    0x1E376C08,
    0x2748774C,
    0x34B0BCB5,
    0x391C0CB3,
    0x4ED8AA4A,
    0x5B9CCA4F,
    0x682E6FF3,
    0x748F82EE,
    0x78A5636F,
    0x84C87814,
    0x8CC70208,
    0x90BEFFFA,
    0xA4506CEB,
    0xBEF9A3F7,
    0xC67178F2,
]
K512 = [
    0x428A2F98D728AE22,
    0x7137449123EF65CD,
    0xB5C0FBCFEC4D3B2F,
    0xE9B5DBA58189DBBC,
    0x3956C25BF348B538,
    0x59F111F1B605D019,
    0x923F82A4AF194F9B,
    0xAB1C5ED5DA6D8118,
    0xD807AA98A3030242,
    0x12835B0145706FBE,
    0x243185BE4EE4B28C,
    0x550C7DC3D5FFB4E2,
    0x72BE5D74F27B896F,
    0x80DEB1FE3B1696B1,
    0x9BDC06A725C71235,
    0xC19BF174CF692694,
    0xE49B69C19EF14AD2,
    0xEFBE4786384F25E3,
    0x0FC19DC68B8CD5B5,
    0x240CA1CC77AC9C65,
    0x2DE92C6F592B0275,
    0x4A7484AA6EA6E483,
    0x5CB0A9DCBD41FBD4,
    0x76F988DA831153B5,
    0x983E5152EE66DFAB,
    0xA831C66D2DB43210,
    0xB00327C898FB213F,
    0xBF597FC7BEEF0EE4,
    0xC6E00BF33DA88FC2,
    0xD5A79147930AA725,
    0x06CA6351E003826F,
    0x142929670A0E6E70,
    0x27B70A8546D22FFC,
    0x2E1B21385C26C926,
    0x4D2C6DFC5AC42AED,
    0x53380D139D95B3DF,
    0x650A73548BAF63DE,
    0x766A0ABB3C77B2A8,
    0x81C2C92E47EDAEE6,
    0x92722C851482353B,
    0xA2BFE8A14CF10364,
    0xA81A664BBC423001,
    0xC24B8B70D0F89791,
    0xC76C51A30654BE30,
    0xD192E819D6EF5218,
    0xD69906245565A910,
    0xF40E35855771202A,
    0x106AA07032BBD1B8,
    0x19A4C116B8D2D0C8,
    0x1E376C085141AB53,
    0x2748774CDF8EEB99,
    0x34B0BCB5E19B48A8,
    0x391C0CB3C5C95A63,
    0x4ED8AA4AE3418ACB,
    0x5B9CCA4F7763E373,
    0x682E6FF3D6B2B8A3,
    0x748F82EE5DEFB2FC,
    0x78A5636F43172F60,
    0x84C87814A1F0AB72,
    0x8CC702081A6439EC,
    0x90BEFFFA23631E28,
    0xA4506CEBDE82BDE9,
    0xBEF9A3F7B2C67915,
    0xC67178F2E372532B,
    0xCA273ECEEA26619C,
    0xD186B8C721C0C207,
    0xEADA7DD6CDE0EB1E,
    0xF57D4F7FEE6ED178,
    0x06F067AA72176FBA,
    0x0A637DC5A2C898A6,
    0x113F9804BEF90DAE,
    0x1B710B35131C471B,
    0x28DB77F523047D84,
    0x32CAAB7B40C72493,
    0x3C9EBE0A15C9BEBC,
    0x431D67C49C100D4C,
    0x4CC5D4BECB3E42B6,
    0x597F299CFC657E2A,
    0x5FCB6FAB3AD6FAEC,
    0x6C44198C4A475817,
]


def _rotr(x: int, r: int, wbits: int) -> int:
    return ((x >> r) | (x << (wbits - r))) & ((1 << wbits) - 1)


def compress_np(state: list[int], blocks: bytes, bits: int = 256) -> list[int]:
    """SHA-256/512 compression of `blocks` (a multiple of the block size), plain Python.

    `bits` is the digest size; the *word* size is `bits // 8` bits (32 for SHA-256, 64 for
    SHA-512), which is the one that matters for masking and rotation.
    """
    nw, rounds = 16, (64 if bits == 256 else 80)
    wbits = bits // 8
    wsize = wbits // 8
    bsize = nw * wsize
    mask = (1 << wbits) - 1
    K = K256 if bits == 256 else K512
    if bits == 256:
        (r_s0a, r_s0b, s0sh), (r_s1a, r_s1b, s1sh) = (7, 18, 3), (17, 19, 10)
        (r_S0a, r_S0b, r_S0c), (r_S1a, r_S1b, r_S1c) = (2, 13, 22), (6, 11, 25)
    else:
        (r_s0a, r_s0b, s0sh), (r_s1a, r_s1b, s1sh) = (1, 8, 7), (19, 61, 6)
        (r_S0a, r_S0b, r_S0c), (r_S1a, r_S1b, r_S1c) = (28, 34, 39), (14, 18, 41)
    out = list(state)
    for off in range(0, len(blocks), bsize):
        chunk = blocks[off : off + bsize]
        w = [int.from_bytes(chunk[i * wsize : (i + 1) * wsize], "big") for i in range(nw)]
        for t in range(nw, rounds):
            s0 = (
                _rotr(w[t - 15], r_s0a, wbits)
                ^ _rotr(w[t - 15], r_s0b, wbits)
                ^ (w[t - 15] >> s0sh)
            )
            s1 = _rotr(w[t - 2], r_s1a, wbits) ^ _rotr(w[t - 2], r_s1b, wbits) ^ (w[t - 2] >> s1sh)
            w.append((w[t - 16] + s0 + w[t - 7] + s1) & mask)
        a, b, c, d, e, f, g, h = out
        for t in range(rounds):
            s1 = _rotr(e, r_S1a, wbits) ^ _rotr(e, r_S1b, wbits) ^ _rotr(e, r_S1c, wbits)
            ch = (e & f) ^ ((~e) & g)
            t1 = (h + s1 + ch + K[t] + w[t]) & mask
            s0 = _rotr(a, r_S0a, wbits) ^ _rotr(a, r_S0b, wbits) ^ _rotr(a, r_S0c, wbits)
            maj = (a & b) ^ (a & c) ^ (b & c)
            h = g
            g = f
            f = e
            e = (d + t1) & mask
            d = c
            c = b
            b = a
            a = (t1 + s0 + maj) & mask
        out = [(x + y) & mask for x, y in zip((a, b, c, d, e, f, g, h), out, strict=True)]
    return out


def hash_np(msg: bytes, bits: int = 256) -> bytes:
    """Plain SHA-256/512 with the standard padding, for tests."""
    wbits = bits // 8
    wsize = wbits // 8
    bsize = 16 * wsize
    lenbytes = 8 if bits == 256 else 16
    state = list(IV256 if bits == 256 else IV512)
    data = bytearray(msg)
    data.append(0x80)
    while len(data) % bsize != bsize - lenbytes:
        data.append(0)
    data += (8 * len(msg)).to_bytes(lenbytes, "big")
    state = compress_np(state, bytes(data), bits)
    return b"".join(x.to_bytes(wsize, "big") for x in state)


_CACHE: dict[tuple[int, int], object] = {}


def gen_source(bits: int, nblocks: int) -> str:
    """A kernel that absorbs `nblocks` pre-padded blocks and writes the 8-word state."""
    W = bits // 8  # word bits
    T = f"tl.uint{W}"
    rounds = 64 if bits == 256 else 80
    if bits == 256:
        (s0a, s0b, s0s), (s1a, s1b, s1s) = (7, 18, 3), (17, 19, 10)
        (S0a, S0b, S0c), (S1a, S1b, S1c) = (2, 13, 22), (6, 11, 25)
        bs = "_bs32"
    else:
        (s0a, s0b, s0s), (s1a, s1b, s1s) = (1, 8, 7), (19, 61, 6)
        (S0a, S0b, S0c), (S1a, S1b, S1c) = (28, 34, 39), (14, 18, 41)
        bs = "_bs64"
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
        "def sha_kernel(HIN, KT, MSG, OUT, rows, NW: tl.constexpr, NOUT: tl.constexpr,",
        "               BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    row = pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = row < rows",
    ]
    L.append(f"    zero_v = tl.zeros((BLOCK,), dtype={T})")
    for i in range(8):
        L.append(f"    h{i} = tl.load(HIN + {i}).to({T}, bitcast=True) | zero_v")
    L.append(f"    for _b in range(0, {nblocks}, 1):")
    for i in range(8):
        L.append(f"        a{i} = h{i}")
    for i in range(16):
        L.append(
            f"        w{i} = {bs}(tl.load(MSG + row * NW + _b * 16 + {i}, mask=m,"
            f" other=0).to({T}, bitcast=True))"
        )
    L.append(f"        for _t in range(0, {rounds}, 1):")
    L.append(f"            kt = tl.load(KT + _t).to({T})")
    L.append(f"            S1 = _r(a4, {S1a}) ^ _r(a4, {S1b}) ^ _r(a4, {S1c})")
    L.append("            ch = (a4 & a5) ^ (~a4 & a6)")
    L.append("            t1 = a7 + S1 + ch + kt + w0")
    L.append(f"            S0 = _r(a0, {S0a}) ^ _r(a0, {S0b}) ^ _r(a0, {S0c})")
    L.append("            maj = (a0 & a1) ^ (a0 & a2) ^ (a1 & a2)")
    L.append("            t2 = S0 + maj")
    for i in range(7, 0, -1):
        L.append(f"            a{i} = a{i - 1}" if i != 4 else "            a4 = a3 + t1")
    L.append("            a0 = t1 + t2")
    L.append(
        f"            nw_ = w0 + (_r(w1, {s0a}) ^ _r(w1, {s0b}) ^ (w1 >> {s0s}))"
        f" + w9 + (_r(w14, {s1a}) ^ _r(w14, {s1b}) ^ (w14 >> {s1s}))"
    )
    for i in range(15):
        L.append(f"            w{i} = w{i + 1}")
    L.append("            w15 = nw_")
    for i in range(8):
        L.append(f"        h{i} = h{i} + a{i}")
    it = "tl.int32" if bits == 256 else "tl.int64"
    for i in range(8):
        L.append(f"    tl.store(OUT + row * NOUT + {i}, {bs}(h{i}).to({it}, bitcast=True), mask=m)")
    return "\n".join(L) + "\n"


def _compile(bits: int, nblocks: int):
    key = (bits, nblocks)
    kern = _CACHE.get(key)
    if kern is not None:
        return kern
    src = gen_source(bits, nblocks)
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_sha2_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_sha2_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[key] = mod.sha_kernel
    return mod.sha_kernel


def sha_absorb(hin, msg_words, out_words, rows: int, bits: int, nblocks: int, block: int = 128):
    """Absorb `nblocks` blocks; `msg_words` is [rows, nblocks*nw] and `out_words` [rows, 8]."""
    import triton

    kern = _compile(bits, nblocks)
    kern[(triton.cdiv(rows, block),)](
        hin,
        _k_tensor(bits, out_words.device),
        msg_words,
        out_words,
        rows,
        NW=nblocks * (16),
        NOUT=8,
        BLOCK=block,
        num_warps=4,
    )


def _k_tensor(bits: int, device):
    cache = _K_TENSORS.get((bits, device))
    if cache is None:
        vals = K256 if bits == 256 else K512
        cache = word_tensor(vals, bits, device)
        _K_TENSORS[(bits, device)] = cache
    return cache


_K_TENSORS: dict = {}


def word_tensor(values: list[int], bits: int, device):
    """A state / constant table as a signed tensor of the right width (bit patterns)."""
    import torch

    wbits = bits // 8
    signed = [v - (1 << wbits) if v >= (1 << (wbits - 1)) else v for v in values]
    dtype = torch.int32 if bits == 256 else torch.int64
    return torch.tensor(signed, dtype=dtype, device=device)


def iv_tensor(bits: int, device):
    return word_tensor(IV256 if bits == 256 else IV512, bits, device)


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
    assert tail + 1 + lenbytes <= bsize, "one block only"
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
    src = gen_thash_source(bits, dn, nout_words, passthrough)
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_sha2_thash_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_sha2_thash_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
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
