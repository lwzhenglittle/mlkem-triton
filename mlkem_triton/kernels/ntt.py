"""Forward Kyber/ML-KEM NTT on Triton (FIPS 203 Alg. 9).

**This is the only component in this repo unrelated to the Toeplitz trick.**
Neither decaps nor encaps needs an NTT (polynomial multiplication is folded into
a GEMM with fixed matrices), but keygen does: `ek` must carry the NTT-domain
representation of `t̂`, and `s`/`e` are new for every key, so no
"matrix fixed across operations" is available. So here it is just one plain
butterfly.

Mapping: one program handles `BR` polynomials (256 int32 each).
7 butterfly stages (length = 128, 64, …, 2), each stage uses `tl.gather` to swap
pairs by `idx ^ L`.

The pairing index `idx ^ L` is equivalent: inside a block aligned to `2L`,
`idx ^ L` swaps the two halves exactly. `ZETAS` has `2^s` entries per stage, and
the offset is `2^s` too (i accumulates 1,2,4,… starting from 1).

Measured (GB10, B=8192 equivalent scale): see `benchmarks/bench_keygen.py`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ..devices import warps
from ..ntt_ref import GAMMAS, ZETAS

Q = 3329
QL = tl.constexpr(Q)
N = 256


_GM: dict = {}


_ZT_FLAT: dict = {}


def _zt_flat(dev):
    """ζ for each stage × each element position, flattened to [7*256], so a
    single load suffices.

    Previously every element at every stage had to compute `idx // (2*L)` and
    then gather the constant table -- 7 stages meant 7 integer divisions + 7
    small table gathers. Integer division is 20+ instructions on the GPU.
    """
    t = _ZT_FLAT.get(dev)
    if t is None:
        import numpy as _np

        z = _np.asarray([int(v) for v in ZETAS], dtype=_np.int32)
        flat = _np.empty(7 * 256, dtype=_np.int32)
        L = 128
        off = 1
        for s_i in range(7):
            idx = _np.arange(256)
            flat[s_i * 256 : (s_i + 1) * 256] = z[off + idx // (2 * L)]
            L //= 2
            off *= 2
        t = torch.from_numpy(flat).to(dev)
        _ZT_FLAT[dev] = t
    return t


def _gammas(dev):
    t = _GM.get(dev)
    if t is None:
        t = torch.tensor([int(g) for g in GAMMAS[:128]], dtype=torch.int32, device=dev)
        _GM[dev] = t
    return t


@triton.jit
def _modq(v):
    v = v % QL
    return tl.where(v < 0, v + QL, v)


@triton.jit
def ntt_kernel(X, OUT, ZT, P, N: tl.constexpr, BR: tl.constexpr):
    """[P, 256] coefficient domain int32 → NTT domain int32. In-place is
    unnecessary; input and output may be the same buffer."""
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < P
    idx = tl.arange(0, N)
    x = tl.load(X + rows[:, None] * N + idx[None, :], mask=m[:, None], other=0).to(tl.int32)
    x = _modq(x)

    L = N // 2
    for _s in tl.static_range(7):
        pv = tl.gather(x, (idx ^ L)[None, :].broadcast_to((BR, N)), axis=1)
        z = tl.load(ZT + _s * N + idx).to(tl.int32)[None, :]
        lo = ((idx & L) == 0)[None, :]
        # low half out = a + z·b, high half out = a − z·b (a is the j-th element
        # of the block, b is the (j+L)-th). Folding a/b into the two halves with
        # where needs only **one** multiply + two modq, instead of "compute both
        # branches" (which is two multiplies + four modq).
        # Measured: this change + BR going from 16 to 2 took the NTT from
        # 15.2 ms to 1.5 ms (P=24576).
        a = tl.where(lo, x, pv)
        b = tl.where(lo, pv, x)
        sg = tl.where(lo, 1, -1)
        x = _modq(a + sg * _modq(z * b))
        L = L // 2

    tl.store(OUT + rows[:, None] * N + idx[None, :], x, mask=m[:, None])


@triton.jit
def pow_mac_kernel(
    A, S, E, OUT, GAM, P, K: tl.constexpr, N: tl.constexpr, HALF: tl.constexpr, BR: tl.constexpr
):
    """`t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]` (FIPS 203 Alg. 11/12).

    `multiply_ntts` is 128 **quadratic-extension-field** multiplications:

        c0 = a0·b0 + (a1·b1 mod q)·γ
        c1 = a0·b1 + a1·b0

    So even/odd coefficients form two independent channels. **Loading them
    directly with stride=2** removes any need for a gather -- this version
    turned 16 `tl.gather` calls into plain strided load/store. (On GB10 this
    step used to be 33% of keygen, its single largest part.)
    """
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    m = rows < P
    h = tl.arange(0, HALF)
    b = rows // K
    i = rows - b * K
    arow = b * (K * K) + i * K
    srow = b * K

    e_base = E + rows[:, None] * N
    acc0 = tl.load(e_base + 2 * h[None, :], mask=m[:, None], other=0).to(tl.int32)
    acc1 = tl.load(e_base + 2 * h[None, :] + 1, mask=m[:, None], other=0).to(tl.int32)
    g = tl.load(GAM + h).to(tl.int32)[None, :]

    for j in tl.static_range(4):  # K ∈ {2,3,4}, always run the full 4
        if j < K:
            ab = A + (arow + j)[:, None] * N + 2 * h[None, :]
            sb = S + (srow + j)[:, None] * N + 2 * h[None, :]
            a0 = tl.load(ab, mask=m[:, None], other=0).to(tl.int32)
            a1 = tl.load(ab + 1, mask=m[:, None], other=0).to(tl.int32)
            b0 = tl.load(sb, mask=m[:, None], other=0).to(tl.int32)
            b1 = tl.load(sb + 1, mask=m[:, None], other=0).to(tl.int32)
            acc0 = _modq(acc0 + _modq(a0 * b0 + _modq(_modq(a1 * b1) * g)))
            acc1 = _modq(acc1 + _modq(a0 * b1 + a1 * b0))

    o_base = OUT + rows[:, None] * N
    tl.store(o_base + 2 * h[None, :], acc0, mask=m[:, None])
    tl.store(o_base + 2 * h[None, :] + 1, acc1, mask=m[:, None])


def forward_ntt(x: torch.Tensor, br: int = 2) -> torch.Tensor:
    """[P, 256] int32 (coefficient domain, [0,q)) → [P, 256] int32 (NTT domain)."""
    P = x.shape[0]
    out = torch.empty_like(x, dtype=torch.int32)
    ntt_kernel[(triton.cdiv(P, br),)](
        x, out, _zt_flat(x.device), P, N, br, num_warps=warps("ntt", 2)
    )
    return out


def pow_mac(a, s, e, k: int, br: int = 16) -> torch.Tensor:
    """[B*k*k,256] × [B*k,256] → t̂ [B*k,256]; inputs must already be NTT'ed."""
    P = s.shape[0]
    out = torch.empty_like(s, dtype=torch.int32)
    pow_mac_kernel[(triton.cdiv(P, br),)](
        a,
        s,
        e,
        out,
        _gammas(s.device),
        P,
        k,
        N,
        N // 2,
        br,
        num_warps=warps("pow_mac", 4),
    )
    return out


__all__ = ["forward_ntt", "ntt_kernel", "pow_mac", "pow_mac_kernel"]


# --------------------------------------------------------------- ByteEncode_12
@triton.jit
def encode12_kernel(V, OUT, N: tl.constexpr, BYTES: tl.constexpr, OBLK: tl.constexpr):
    """`ByteEncode_12`: the 256 coefficients in [0,4096) of one polynomial →
    384 bytes.

    D=12 > 8, so a byte spans at most **2** coefficients (unlike dv=5 on the
    encaps side, which spans 3).
    """
    p = tl.program_id(0)
    o = tl.program_id(1) * OBLK + tl.arange(0, OBLK)
    m = o < BYTES
    bit0 = o * 8
    c0 = bit0 // 12
    s0 = bit0 - c0 * 12
    c1 = tl.minimum(c0 + 1, N - 1)
    v0 = tl.load(V + p * N + c0, mask=m, other=0).to(tl.int32)
    v1 = tl.load(V + p * N + c1, mask=m, other=0).to(tl.int32)
    byte = ((v0 >> s0) | (v1 << (12 - s0))) & 0xFF
    tl.store(OUT + p * BYTES + o, byte.to(tl.uint8), mask=m)


def byte_encode12(vals: torch.Tensor) -> torch.Tensor:
    """[P, 256] int32 ([0,4096)) → [P, 384] uint8."""
    P = vals.shape[0]
    out = torch.empty((P, 384), dtype=torch.uint8, device=vals.device)
    encode12_kernel[(P, 3)](vals, out, N, 384, 128, num_warps=warps("encode12", 4))
    return out


__all__ += ["byte_encode12", "encode12_kernel"]
