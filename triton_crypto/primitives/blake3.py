"""BLAKE3 (unkeyed): the tree hash, and the pattern a 2^20-point NTT will need.

The compression is a 7-round ARX permutation; the difficulty is the **tree**. 1 KiB chunks are
compressed independently, their chaining values paired level by level, and the root is the last
compression with the `ROOT` flag. That "one launch per level" shape is what a multi-pass NTT
needs, which is why this module exists before the large-N transform.

Kernel shape, and the four constraints that decide it (the first three cost a rewrite each, the
fourth cost four sessions):

* **one launch per block index**, covering every (message, chunk) at once, with the 8-word
  chaining value in a `[B * chunks, 8]` buffer. A runtime loop inside one kernel cannot carry the
  CV -- Triton does not carry tuple assignments across a runtime loop -- and unrolling the
  sixteen steps with `tl.static_range` made the Xe backend spend more than ten minutes on the IR;
* the block index is a **runtime** argument: as a `tl.constexpr` it cannot receive a loop
  variable, and passing one silently loads the wrong block;
* the chaining-value store is masked by `block < nblocks`, because the last chunk of a message is
  usually partial;
* **never shift a 32-bit value by 32.** `counter >> 32` on a uint32 is undefined behaviour; the
  Xe backend accepts it, emits a warning that is easy to filter out of a test log, and folds the
  entire kernel's results to **zero**. That is what made this module "store zeros" for four
  attempts: every input read back correct from inside the same launch, the stored value did not.
  The counter is now passed as an explicit `(low, high)` pair.

Vectors: `data/vectors/blake3/kat.json` from the official Rust implementation
(`benchmarks/blake3_ref`, cargo mode `kat`), 26 lengths from 0 to 64 KiB including the
non-power-of-two chunk counts where the tree shape is the thing being tested.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

__all__ = ["BLAKE3_IV", "blake3_hash_np", "blake3_hash_torch", "compression_np"]

BLAKE3_IV = (
    0x6A09E667,
    0xBB67AE85,
    0x3C6EF372,
    0xA54FF53A,
    0x510E527F,
    0x9B05688C,
    0x1F83D9AB,
    0x5BE0CD19,
)
MSG_PERM = (2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8)

CHUNK_START = 1
CHUNK_END = 2
PARENT = 4
ROOT = 8
BLOCK_LEN = 64
CHUNK_LEN = 1024
BLOCKS_PER_CHUNK = CHUNK_LEN // BLOCK_LEN
M32 = 0xFFFFFFFF

PERM_C = tl.constexpr(MSG_PERM)
_CS = tl.constexpr(CHUNK_START)
_CE = tl.constexpr(CHUNK_END)
_PARENT = tl.constexpr(PARENT)
_ROOT = tl.constexpr(ROOT)


# --------------------------------------------------------------------------- reference


def _rotr(x: int, n: int) -> int:
    x &= M32
    return ((x >> n) | (x << (32 - n))) & M32


def _g(v, a, b, c, d, mx, my):
    v[a] = (v[a] + v[b] + mx) & M32
    v[d] = _rotr(v[d] ^ v[a], 16)
    v[c] = (v[c] + v[d]) & M32
    v[b] = _rotr(v[b] ^ v[c], 12)
    v[a] = (v[a] + v[b] + my) & M32
    v[d] = _rotr(v[d] ^ v[a], 8)
    v[c] = (v[c] + v[d]) & M32
    v[b] = _rotr(v[b] ^ v[c], 7)


def compression_np(key, block_words, counter: int, block_len: int, flags: int):
    """One compression: `(cv[8], 64-byte output)` -- CV words then XOF words."""
    v = (
        [int(x) for x in key]
        + [int(x) for x in BLAKE3_IV[:4]]
        + [
            counter & M32,
            (counter >> 32) & M32,
            block_len,
            flags,
        ]
    )
    m = [int(x) for x in block_words]
    for _ in range(7):
        _g(v, 0, 4, 8, 12, m[0], m[1])
        _g(v, 1, 5, 9, 13, m[2], m[3])
        _g(v, 2, 6, 10, 14, m[4], m[5])
        _g(v, 3, 7, 11, 15, m[6], m[7])
        _g(v, 0, 5, 10, 15, m[8], m[9])
        _g(v, 1, 6, 11, 12, m[10], m[11])
        _g(v, 2, 7, 8, 13, m[12], m[13])
        _g(v, 3, 4, 9, 14, m[14], m[15])
        m = [m[i] for i in MSG_PERM]
    cv = [v[i] ^ v[i + 8] for i in range(8)]
    xof = [v[i + 8] ^ int(key[i]) for i in range(8)]
    return cv, b"".join(w.to_bytes(4, "little") for w in cv + xof)


def _block_words(block: bytes):
    return [int(x) for x in np.frombuffer(block + b"\x00" * (BLOCK_LEN - len(block)), "<u4")]


def _chunk_cv(chunk: bytes, counter: int):
    nblocks = max(1, (len(chunk) + BLOCK_LEN - 1) // BLOCK_LEN)
    cv = list(BLAKE3_IV)
    last = None
    for i in range(nblocks):
        block = chunk[i * BLOCK_LEN : (i + 1) * BLOCK_LEN]
        flags = (CHUNK_START if i == 0 else 0) | (CHUNK_END if i == nblocks - 1 else 0)
        if i == nblocks - 1:
            last = (list(cv), block, counter, len(block), flags)
        cv, _ = compression_np(cv, _block_words(block), counter, len(block), flags)
    return cv, last


def blake3_hash_np(msg: bytes, out_len: int = 32) -> bytes:
    if out_len > 64:
        raise ValueError("out_len above 64 needs a second XOF block")
    nchunks = max(1, (len(msg) + CHUNK_LEN - 1) // CHUNK_LEN)
    level = []
    for i in range(nchunks):
        cv, last = _chunk_cv(msg[i * CHUNK_LEN : (i + 1) * CHUNK_LEN], i)
        if nchunks == 1:
            key, block, counter, block_len, flags = last
            _, out = compression_np(key, _block_words(block), counter, block_len, flags | ROOT)
            return out[:out_len]
        level.append(cv)
    while True:
        if len(level) == 2:
            _, out = compression_np(
                list(BLAKE3_IV), list(level[0]) + list(level[1]), 0, BLOCK_LEN, PARENT | ROOT
            )
            return out[:out_len]
        nxt = []
        i = 0
        while i + 1 < len(level):
            cv, _ = compression_np(
                list(BLAKE3_IV), list(level[i]) + list(level[i + 1]), 0, BLOCK_LEN, PARENT
            )
            nxt.append(cv)
            i += 2
        if i < len(level):
            nxt.append(level[i])
        level = nxt


# --------------------------------------------------------------------------- device


@triton.jit
def _rotr32(x, n: tl.constexpr):
    return (x >> n) | (x << (32 - n))


@triton.jit
def _g4(a, b, c, d, mx, my):
    a = a + b + mx
    d = _rotr32(d ^ a, 16)
    c = c + d
    b = _rotr32(b ^ c, 12)
    a = a + b + my
    d = _rotr32(d ^ a, 8)
    c = c + d
    b = _rotr32(b ^ c, 7)
    return a, b, c, d


@triton.jit
def _permute(m):
    out = ()
    for i in tl.static_range(16):
        out = out + (m[PERM_C[i]],)
    return out


@triton.jit
def _compress(k0, k1, k2, k3, k4, k5, k6, k7, m, ctr_lo, ctr_hi, blen, flags, IVP):
    """7 rounds. The counter arrives as `(low, high)`: shifting a 32-bit value by 32 is UB."""
    v0, v1, v2, v3, v4, v5, v6, v7 = k0, k1, k2, k3, k4, k5, k6, k7
    v8 = tl.load(IVP + 0) + ctr_lo * 0
    v9 = tl.load(IVP + 1) + ctr_lo * 0
    v10 = tl.load(IVP + 2) + ctr_lo * 0
    v11 = tl.load(IVP + 3) + ctr_lo * 0
    v12, v13, v14, v15 = ctr_lo, ctr_hi, blen, flags
    for _ in tl.static_range(7):
        v0, v4, v8, v12 = _g4(v0, v4, v8, v12, m[0], m[1])
        v1, v5, v9, v13 = _g4(v1, v5, v9, v13, m[2], m[3])
        v2, v6, v10, v14 = _g4(v2, v6, v10, v14, m[4], m[5])
        v3, v7, v11, v15 = _g4(v3, v7, v11, v15, m[6], m[7])
        v0, v5, v10, v15 = _g4(v0, v5, v10, v15, m[8], m[9])
        v1, v6, v11, v12 = _g4(v1, v6, v11, v12, m[10], m[11])
        v2, v7, v8, v13 = _g4(v2, v7, v8, v13, m[12], m[13])
        v3, v4, v9, v14 = _g4(v3, v4, v9, v14, m[14], m[15])
        m = _permute(m)
    return (
        v0 ^ v8,
        v1 ^ v9,
        v2 ^ v10,
        v3 ^ v11,
        v4 ^ v12,
        v5 ^ v13,
        v6 ^ v14,
        v7 ^ v15,
        v8 ^ k0,
        v9 ^ k1,
        v10 ^ k2,
        v11 ^ k3,
        v12 ^ k4,
        v13 ^ k5,
        v14 ^ k6,
        v15 ^ k7,
    )


@triton.jit
def _block_kernel(
    MSG,
    IVP,
    CVS,
    ROOT_CV,
    ROOT_XOF,
    B,
    NBYTES,
    NCHUNKS,
    BLKIDX,
    ROOT_MODE: tl.constexpr,
    BLK: tl.constexpr,
):
    """One launch = one block index, for every (message, chunk) pair."""
    chunk = tl.program_id(0)
    pid = tl.program_id(1)
    lane = pid * BLK + tl.arange(0, BLK)
    m = lane < B
    start = chunk * 1024
    avail = tl.minimum(tl.maximum(NBYTES - start, 1), 1024)
    nblocks = (avail + 63) // 64
    row = lane * NCHUNKS + chunk

    key = ()
    for i in tl.static_range(8):
        key = key + (tl.load(CVS + row * 8 + i, mask=m, other=0).to(tl.uint32),)
    words = ()
    for w in tl.static_range(16):
        v = tl.zeros_like(lane).to(tl.uint32)
        for b in tl.static_range(4):
            off = BLKIDX * 64 + w * 4 + b
            keep = m & (nblocks > BLKIDX) & (off < avail)
            v = v | (
                tl.load(MSG + lane * NBYTES + start + off, mask=keep, other=0).to(tl.uint32)
                << (8 * b)
            )
        words = words + (v,)
    flags = tl.where(BLKIDX == 0, _CS, 0) + tl.zeros_like(lane)
    flags = flags | (tl.where(nblocks - 1 == BLKIDX, _CE, 0) + tl.zeros_like(lane))
    is_root = tl.zeros_like(lane) > 1
    if ROOT_MODE:
        is_root = (chunk == NCHUNKS - 1) & (nblocks - 1 == BLKIDX) & m
        flags = flags | (tl.where(is_root, _ROOT, 0) + tl.zeros_like(lane))
    blen = tl.where(nblocks > BLKIDX, tl.minimum(avail - BLKIDX * 64, 64), 0) + tl.zeros_like(lane)
    zero = tl.zeros_like(lane)
    r = _compress(
        key[0],
        key[1],
        key[2],
        key[3],
        key[4],
        key[5],
        key[6],
        key[7],
        words,
        (chunk + zero).to(tl.uint32),
        zero.to(tl.uint32),
        blen.to(tl.uint32),
        flags.to(tl.uint32),
        IVP,
    )
    store = m & (nblocks > BLKIDX)
    for i in tl.static_range(8):
        tl.store(CVS + row * 8 + i, r[i].to(tl.int32), mask=store)
        if ROOT_MODE:
            tl.store(ROOT_CV + lane * 8 + i, r[i].to(tl.int32), mask=is_root)
            tl.store(ROOT_XOF + lane * 8 + i, r[8 + i].to(tl.int32), mask=is_root)


@triton.jit
def _pair_kernel(
    CVS,
    NEXT,
    ROOT_CV,
    ROOT_XOF,
    B,
    STRIDE_IN,
    STRIDE_OUT,
    IVP,
    ROOT_MODE: tl.constexpr,
    BLK: tl.constexpr,
):
    """Combine adjacent chaining values; `ROOT_MODE` writes the digest instead."""
    pair = tl.program_id(0)
    pid = tl.program_id(1)
    lane = pid * BLK + tl.arange(0, BLK)
    mask = lane < B
    zero = tl.zeros_like(lane)
    m = ()
    for i in tl.static_range(8):
        m = m + (
            tl.load(CVS + (lane * STRIDE_IN + 2 * pair) * 8 + i, mask=mask, other=0).to(tl.uint32),
        )
    for i in tl.static_range(8):
        m = m + (
            tl.load(CVS + (lane * STRIDE_IN + 2 * pair + 1) * 8 + i, mask=mask, other=0).to(
                tl.uint32
            ),
        )
    flags = (_PARENT | (_ROOT if ROOT_MODE else 0)) + zero
    r = _compress(
        tl.load(IVP + 0) + zero,
        tl.load(IVP + 1) + zero,
        tl.load(IVP + 2) + zero,
        tl.load(IVP + 3) + zero,
        tl.load(IVP + 4) + zero,
        tl.load(IVP + 5) + zero,
        tl.load(IVP + 6) + zero,
        tl.load(IVP + 7) + zero,
        m,
        zero.to(tl.uint32),
        zero.to(tl.uint32),
        tl.full([BLK], 64, tl.int32),
        flags,
        IVP,
    )
    if ROOT_MODE:
        for i in tl.static_range(8):
            tl.store(ROOT_CV + lane * 8 + i, r[i].to(tl.int32), mask=mask)
            tl.store(ROOT_XOF + lane * 8 + i, r[8 + i].to(tl.int32), mask=mask)
    else:
        for i in tl.static_range(8):
            tl.store(NEXT + (lane * STRIDE_OUT + pair) * 8 + i, r[i].to(tl.int32), mask=mask)


def _iv_tensor(dev):
    import torch

    return torch.tensor(BLAKE3_IV, dtype=torch.uint32, device=dev).view(torch.int32)


def blake3_hash_torch(messages, device=None):
    """`[B, n]` uint8 -> `[B, 32]` uint8. One launch per block index, then one per tree level."""
    import torch

    from .. import devices as _devices

    dev = _devices.resolve_device(device)
    arr = np.asarray(messages)
    if not np.issubdtype(arr.dtype, np.integer):
        # a float array would be truncated to bytes silently, and a digest of the wrong bytes is
        # indistinguishable from a digest of the right ones
        raise TypeError(f"messages must be integer bytes, got dtype {arr.dtype}")
    if arr.dtype != np.uint8:
        if arr.size and (arr.min() < 0 or arr.max() > 255):
            raise ValueError(
                f"messages must hold byte values (0..255), got [{arr.min()}, {arr.max()}]"
            )
        arr = arr.astype(np.uint8)
    if not arr.flags.writeable:
        # a read-only view (np.frombuffer, or a sliced bytes literal) makes torch warn on the
        # zero-copy conversion; nothing here writes to the input, but the copy is cheap compared
        # to a warning in every caller's log and an undefined-behaviour footnote
        arr = arr.copy()
    msg = torch.as_tensor(arr, device=dev)
    if msg.ndim != 2:
        raise ValueError("messages must be [B, nbytes]")
    batch, nbytes = msg.shape
    if batch == 0:
        return torch.zeros((0, 32), dtype=torch.uint8, device=dev)
    if nbytes == 0:
        raise ValueError("pass at least one byte per message")

    nchunks = max(1, (nbytes + CHUNK_LEN - 1) // CHUNK_LEN)
    blk = 64
    iv = _iv_tensor(dev)
    root_cv = torch.zeros((batch, 8), dtype=torch.int32, device=dev)
    root_xof = torch.zeros((batch, 8), dtype=torch.int32, device=dev)
    level = iv.repeat(batch * nchunks, 1).contiguous()
    max_blocks = min(BLOCKS_PER_CHUNK, (nbytes + BLOCK_LEN - 1) // BLOCK_LEN)
    for blkidx in range(max_blocks):
        _block_kernel[(nchunks, triton.cdiv(batch, blk))](
            msg,
            iv,
            level,
            root_cv,
            root_xof,
            batch,
            nbytes,
            nchunks,
            blkidx,
            ROOT_MODE=(nchunks == 1),
            BLK=blk,
        )

    count = nchunks
    while count > 1:
        pairs = count // 2
        if count == 2:
            _pair_kernel[(pairs, triton.cdiv(batch, blk))](
                level, level, root_cv, root_xof, batch, count, 1, iv, ROOT_MODE=True, BLK=blk
            )
            break
        nxt_count = (count + 1) // 2
        nxt = torch.zeros((batch * nxt_count, 8), dtype=torch.int32, device=dev)
        _pair_kernel[(pairs, triton.cdiv(batch, blk))](
            level, nxt, root_cv, root_xof, batch, count, nxt_count, iv, ROOT_MODE=False, BLK=blk
        )
        if count % 2:
            odd = level.reshape(batch, count, 8)[:, -1, :]
            nxt.reshape(batch, nxt_count, 8)[:, -1, :] = odd
        level, count = nxt, nxt_count

    words = torch.cat([root_cv, root_xof], dim=1)
    return words.view(torch.uint8).reshape(batch, 64)[:, :32].contiguous()
