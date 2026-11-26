"""Fold pad10*1 + domain separation (including segment concatenation,
broadcast / single-byte suffix) into **one launch**.

Why it is needed
----------------
`pad_gpu(cat([a, b]))` is 3 launches on XPU (`cat`, `torch.zeros`, `copy_`), and
`cat` also has to handle stride-0 sources produced by `expand`. The J case at
B=4096 (`z(32B) || ct(1088B)`, padded to 9 blocks × 136 B) measures **192 µs**,
while the sponge kernel it serves is only 256 µs — i.e. preparing 4.5 MB of
input eats 43 % of the J stage. Fused into one launch, with the whole row
written (zero padding included), it is **28 µs**.

Strictly equivalent to (byte for byte):

    out = torch.zeros((n, total), uint8)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, total - 1] |= 0x80

Three variants
--------------
- `padcat2`: message = `a || b`, with `a`/`b` each independently broadcastable
  (J: z broadcast + ct per row; G: m' per row + h_ek broadcast).
- `padcat_sfx`: message = `a[row//mod] || [row % mod + off]` (PRF: r' || [j]).
- The plain single-segment case without `padcat` keeps using `pad_gpu` (cheaper
  for scalars / small batches).

Implementation notes
--------------------
- **word (int64) granularity**: segment lengths and the total length must both
  be multiples of 8 (J/G/PRF all satisfy this), so the instruction count is 1/8
  of the byte-granular one. Callers that do not satisfy this fall back on their
  own.
- The zero-fill region explicitly **excludes** the domain word and the last
  word; those two are written with fixed-point scalar stores, avoiding a
  cross-warp write-write race from "write 0 then write domain" within the same
  kernel.
- The tile is `[BR=32, <=128]`: at BR=128, holding 128 int64 per thread (=128
  GRF), throughput drops 2.6× (measured 25 µs → 65 µs), so BR cannot be large.
"""

from __future__ import annotations

import torch
import triton

from ..._codegen import compile_generated
from ...devices import warps

BR_DEFAULT = 32
WARPS_DEFAULT = 4
_MAX_CHUNK = 128  # at most 128 words per tile (a multiple of 8 and a power of 2)


def _sgn64(v: int) -> int:
    """Write an unsigned 64-bit constant into a Triton int64 (avoids an error at >= 2^63)."""
    return v - (1 << 64) if v >= (1 << 63) else v


def _chunks(n: int) -> list[tuple[int, int]]:
    """Split n into a descending sum of powers of 2 (136 -> [(0,128),(128,8)])."""
    out, off = [], 0
    while n > 0:
        p = min(1 << (n.bit_length() - 1), _MAX_CHUNK)
        out.append((off, p))
        off += p
        n -= p
    return out


def gen_padcat2_source(
    la_w: int, lb_w: int, total_w: int, domain: int, a_bcast: bool, b_bcast: bool
) -> str:
    if la_w <= 0 or lb_w < 0 or total_w <= la_w + lb_w:
        raise ValueError(f"invalid padcat2 word layout: {la_w}, {lb_w}, {total_w}")
    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    L.append("def padcat2_kernel(A, B, OUT, n, BR: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    rows = pid * BR + tl.arange(0, BR)")
    L.append("    m = rows < n")
    L.append(f"    prow = OUT + rows[:, None] * {total_w}")
    # ---- segment A ----
    for off, sz in _chunks(la_w):
        tail = off + sz > la_w
        L.append(f"    c = tl.arange(0, {sz})")
        if a_bcast:
            L.append(
                f"    v = tl.load(A + {off} + c"
                + (f", mask=c < {la_w - off}, other=0)" if tail else ")")
            )
            L.append(
                f"    tl.store(prow + {off} + c[None, :], "
                f"tl.broadcast_to(v[None, :], (BR, {sz})), mask=m[:, None])"
            )
        else:
            L.append(
                f"    v = tl.load(A + rows[:, None] * {la_w} + {off} + c[None, :], "
                + (
                    f"mask=m[:, None] & (c[None, :] < {la_w - off}), other=0)"
                    if tail
                    else "mask=m[:, None], other=0)"
                )
            )
            L.append(f"    tl.store(prow + {off} + c[None, :], v, mask=m[:, None])")
    # ---- segment B ----
    for off, sz in _chunks(lb_w):
        tail = off + sz > lb_w
        o = la_w + off
        L.append(f"    c = tl.arange(0, {sz})")
        if b_bcast:
            L.append(
                f"    v = tl.load(B + {off} + c"
                + (f", mask=c < {lb_w - off}, other=0)" if tail else ")")
            )
            L.append(
                f"    tl.store(prow + {o} + c[None, :], "
                f"tl.broadcast_to(v[None, :], (BR, {sz})), mask=m[:, None])"
            )
        else:
            L.append(
                f"    v = tl.load(B + rows[:, None] * {lb_w} + {off} + c[None, :], "
                + (
                    f"mask=m[:, None] & (c[None, :] < {lb_w - off}), other=0)"
                    if tail
                    else "mask=m[:, None], other=0)"
                )
            )
            L.append(f"    tl.store(prow + {o} + c[None, :], v, mask=m[:, None])")
    # ---- zero region (excludes domain word and last word) ----
    z0, z1 = la_w + lb_w + 1, total_w - 1
    for off, sz in _chunks(max(0, z1 - z0)):
        o = z0 + off
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(f"    tl.store(prow + {o} + c[None, :], tl.zeros((BR, {sz}), dtype=tl.int64),")
        L.append(f"             mask=m[:, None] & (c[None, :] < {z1 - o}))")
    # ---- domain word + last word ----
    dom_w, last_w = la_w + lb_w, total_w - 1
    dom_val = _sgn64(domain | (0x80 << 56)) if dom_w == last_w else _sgn64(domain)
    L.append(
        f"    tl.store(OUT + rows * {total_w} + {dom_w}, "
        f"tl.full((BR,), {dom_val}, dtype=tl.int64), mask=m)"
    )
    if dom_w != last_w:
        L.append(
            f"    tl.store(OUT + rows * {total_w} + {last_w}, "
            f"tl.full((BR,), {_sgn64(0x80 << 56)}, dtype=tl.int64), mask=m)"
        )
    return "\n".join(L) + "\n"


def gen_padcat_sfx_source(
    prefix_w: int,
    total_w: int,
    domain: int,
    mod: int,
    off: int,
    stride_w: int | None = None,
    base_w: int = 0,
) -> str:
    """message = a[row // mod] || bytes[row % mod + off] || pad10*1.

    The first `prefix_w` words come from a (each message has only `mod` distinct
    prefixes, taken by row//mod), followed by one word made up of the suffix byte
    + domain, then zero fill; the last word is 0x80. `stride_w`/`base_w` let the
    prefix be read in place directly from a large tensor (e.g. the 64 B output of
    G), saving one `contiguous()` copy.
    """
    if prefix_w <= 0 or total_w <= prefix_w + 1 or mod < 1:
        raise ValueError(f"invalid padcat_sfx layout: {prefix_w}, {total_w}, {mod}")
    sw = prefix_w if stride_w is None else stride_w
    if sw < prefix_w + base_w:
        raise ValueError(f"stride {sw} is smaller than {prefix_w + base_w}")
    dom_w, last_w = prefix_w, total_w - 1
    if dom_w >= last_w:
        raise ValueError("suffix and domain byte would share a word with 0x80")
    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    L.append("def padcat_sfx_kernel(A, OUT, n, MOD: tl.constexpr, OFF: tl.constexpr,")
    L.append("                      BR: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    rows = pid * BR + tl.arange(0, BR)")
    L.append("    m = rows < n")
    L.append(f"    prow = OUT + rows[:, None] * {total_w}")
    arow = f"(rows[:, None] // MOD) * {sw} + {base_w}"
    for o, sz in _chunks(prefix_w):
        tail = o + sz > prefix_w
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(
            f"    v = tl.load(A + {arow} + {o} + c[None, :], "
            + (
                f"mask=m[:, None] & (c[None, :] < {prefix_w - o}), other=0)"
                if tail
                else "mask=m[:, None], other=0)"
            )
        )
        L.append(f"    tl.store(prow + {o} + c[None, :], v, mask=m[:, None])")
    # suffix byte + domain share one word (little-endian: suffix in the lowest byte)
    L.append("    sfx = (rows % MOD) + OFF")
    L.append(f"    w = (sfx.to(tl.int64) | {domain << 8})")
    L.append(f"    tl.store(OUT + rows * {total_w} + {dom_w}, w, mask=m)")
    z0, z1 = dom_w + 1, total_w - 1
    for o, sz in _chunks(max(0, z1 - z0)):
        oo = z0 + o
        L.append(f"    c = tl.arange(0, {sz})")
        L.append(f"    tl.store(prow + {oo} + c[None, :], tl.zeros((BR, {sz}), dtype=tl.int64),")
        L.append(f"             mask=m[:, None] & (c[None, :] < {z1 - oo}))")
    L.append(
        f"    tl.store(OUT + rows * {total_w} + {last_w}, "
        f"tl.full((BR,), {_sgn64(0x80 << 56)}, dtype=tl.int64), mask=m)"
    )
    return "\n".join(L) + "\n"


_CACHE: dict[tuple, object] = {}


def _compile(src: str, fname: str):
    return getattr(compile_generated(src, fname), fname)


def _launch(kern, args, n: int, br: int = BR_DEFAULT, nwarps: int = WARPS_DEFAULT):
    grid = (triton.cdiv(n, br),)
    kern[grid](*args, n=n, BR=br, num_warps=warps("padcat", nwarps))


def padcat2(
    a: torch.Tensor,
    b: torch.Tensor,
    n: int,
    la_w: int,
    lb_w: int,
    total_w: int,
    domain: int,
    a_bcast: bool = False,
    b_bcast: bool = False,
) -> torch.Tensor:
    """One-launch version of `pad_gpu(cat([a, b]))`; returns [n, total_w*8] uint8.

    The `*_w` arguments are all int64 word counts. a/b must already be
    `.view(torch.int64)` (the non-broadcast segment must be a contiguous
    [n, *_w] tensor; the broadcast one is [*_w]).
    """
    key = ("c2", la_w, lb_w, total_w, domain, a_bcast, b_bcast)
    kern = _CACHE.get(key)
    if kern is None:
        kern = _compile(
            gen_padcat2_source(la_w, lb_w, total_w, domain, a_bcast, b_bcast), "padcat2_kernel"
        )
        _CACHE[key] = kern
    out = torch.empty((n, total_w), dtype=torch.int64, device=a.device)
    _launch(kern, (a, b, out), n)
    return out.view(torch.uint8).view(n, total_w * 8)


def padcat_sfx(
    a: torch.Tensor,
    n: int,
    prefix_w: int,
    total_w: int,
    domain: int,
    mod: int,
    off: int,
    stride_w: int | None = None,
    base_w: int = 0,
) -> torch.Tensor:
    """One-launch version of `pad_gpu(cat([a[row//mod], [row%mod+off]]))`."""
    key = ("sfx", prefix_w, total_w, domain, mod, off, stride_w, base_w)
    kern = _CACHE.get(key)
    if kern is None:
        kern = _compile(
            gen_padcat_sfx_source(prefix_w, total_w, domain, mod, off, stride_w, base_w),
            "padcat_sfx_kernel",
        )
        _CACHE[key] = kern
    out = torch.empty((n, total_w), dtype=torch.int64, device=a.device)
    grid = (triton.cdiv(n, BR_DEFAULT),)
    kern[grid](a, out, n, MOD=mod, OFF=off, BR=BR_DEFAULT, num_warps=warps("padcat", WARPS_DEFAULT))
    return out.view(torch.uint8).view(n, total_w * 8)


__all__ = ["gen_padcat2_source", "gen_padcat_sfx_source", "padcat2", "padcat_sfx"]
