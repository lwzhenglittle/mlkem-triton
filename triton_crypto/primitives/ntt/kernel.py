"""The Triton kernel for the in-register negacyclic NTT ([P, N] with N <= 1024).

One program handles `BR` polynomials and one butterfly layer per iteration: `tl.gather`
swaps the two halves of each block (index `idx ^ L`), then one multiply and two reductions
produce the low and high outputs — folding a/b into the halves with `where` costs one
multiply instead of two.

The kernel is defined **at module level** (not inside a factory) because `tl` is imported at
module level here; that is the arrangement `primitives.fields` documents, and it is what lets
`mont_mul` be shared instead of inlined at each call site.
"""

from __future__ import annotations

import triton
import triton.language as tl

from ..fields import mont_mul
from .domain import KYBER, NTTDomain

__all__ = ["intt_torch", "ntt_torch"]


@triton.jit
def _mod(v, Q: tl.constexpr):
    # Triton's integer `%` is C semantics (truncated), hence the fix-up.
    v = v % Q
    return tl.where(v < 0, v + Q, v)


@triton.jit
def _norm(v, Q: tl.constexpr):
    """Bring a value with |v| < 2q into [0, q) with two selects (no division)."""
    v = tl.where(v >= Q, v - Q, v)
    return tl.where(v < 0, v + Q, v)


@triton.jit
def ntt_kernel(
    X,
    OUT,
    ZT,
    P,
    N: tl.constexpr,
    Q: tl.constexpr,
    SCALE: tl.constexpr,
    LOG_N: tl.constexpr,
    LAYERS: tl.constexpr,
    BR: tl.constexpr,
    INVERSE: tl.constexpr,
    WIDE: tl.constexpr,
    MONT: tl.constexpr,
    QINV: tl.constexpr,
):
    """[P, N] coefficients -> NTT domain, or the inverse; one program per BR rows."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < P
    idx = tl.arange(0, N)
    dt = tl.int64 if WIDE else tl.int32
    x = tl.load(X + rows[:, None] * N + idx[None, :], mask=m[:, None], other=0).to(dt)
    x = _norm(x, Q) if MONT else _mod(x, Q)

    for s in tl.static_range(LAYERS):
        # `s` is a compile-time constant here, so L folds; a plain assignment is used
        # because a constexpr annotated name cannot be reassigned across iterations.
        L = 1 << ((LOG_N - LAYERS + s) if INVERSE else (LOG_N - 1 - s))
        pv = tl.gather(x, (idx ^ L)[None, :].broadcast_to((BR, N)), axis=1)
        z = tl.load(ZT + s * N + idx).to(dt)[None, :]
        lo = ((idx & L) == 0)[None, :]
        a = tl.where(lo, x, pv)
        b = tl.where(lo, pv, x)
        if INVERSE:
            # low: a + b; high: ζ·(b − a)
            if MONT:
                # The twiddle is stored as ζ·R, so one Montgomery product is enough:
                # mont_mul(ζ·R, b − a) = ζ·(b − a) in the standard domain.
                hi = mont_mul(z, _norm(b - a, Q), Q, QINV)
            else:
                hi = _mod(z * _mod(b - a, Q), Q)
            x = tl.where(lo, a + b, hi)
        else:
            if MONT:
                t = mont_mul(z, b, Q, QINV)
            else:
                t = _mod(z * b, Q)
            x = tl.where(lo, a + t, a - t)
        if MONT:
            x = _norm(x, Q)
        else:
            x = _mod(x, Q)

    if INVERSE:
        if MONT:
            x = _norm(mont_mul(x, SCALE, Q, QINV), Q)
        else:
            x = _mod(x * SCALE, Q)
    else:
        if MONT:
            x = _norm(x, Q)
    tl.store(OUT + rows[:, None] * N + idx[None, :], x.to(tl.int32), mask=m[:, None])


_TABLES: dict[tuple, tuple] = {}


def _table_tensor(domain: NTTDomain, direction: str, device, mont: bool):
    import torch

    key = (domain.name, domain.q, domain.n, domain.zeta, domain.layers, direction, mont)
    cached = _TABLES.get(key)
    if cached is not None and cached[0] == device:
        return cached[1]
    name = f"{direction}_mont" if mont else direction
    t = torch.from_numpy(domain.tables[name]).to(device)
    if not mont:
        t = t.to(torch.int32)
    _TABLES[key] = (device, t)
    return t


def ntt_torch(
    x,
    domain: NTTDomain = KYBER,
    inverse: bool = False,
    br: int = 2,
    warps: int = 2,
    mont: bool | None = None,
):
    """[P, N] int32 coefficients -> [P, N] int32, on the device of `x`.

    `inverse=True` computes INTT and returns the standard representatives in [0, q).
    Layout: row-major, one polynomial per row.
    """
    import torch

    squeeze = x.dim() == 1
    if squeeze:
        x = x[None, :]
    if x.dim() != 2 or x.shape[1] != domain.n:
        raise ValueError(f"expected [P, {domain.n}] input, got {tuple(x.shape)}")
    p = x.shape[0]
    if x.dtype != torch.int32:
        x = x.to(torch.int32)
    x = x.contiguous()
    out = torch.empty_like(x)
    # int32 moduli (ML-KEM) keep the `%` path: division by a small constant is cheap there.
    # 23-bit moduli (ML-DSA) switch to Montgomery, where `%` means a 64-bit software divide.
    use_mont = domain.wide if mont is None else mont
    table = _table_tensor(domain, "inv" if inverse else "fwd", x.device, use_mont)
    scale = (domain.scale * domain.mont_r) % domain.q if use_mont else domain.scale
    ntt_kernel[(triton.cdiv(p, br),)](
        x,
        out,
        table,
        p,
        domain.n,
        domain.q,
        scale,
        domain.log_n,
        domain.layers,
        br,
        inverse,
        domain.wide,
        use_mont,
        domain.qinv,
        num_warps=warps,
    )
    return out[0] if squeeze else out


def intt_torch(x, domain: NTTDomain = KYBER, br: int = 2, warps: int = 2):
    """[P, N] int32 NTT-domain values -> [P, N] int32 coefficients in [0, q)."""
    return ntt_torch(x, domain, inverse=True, br=br, warps=warps)
