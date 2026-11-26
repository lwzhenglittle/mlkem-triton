"""ML-KEM-specific NTT-domain kernels: pointwise multiply-accumulate and ByteEncode_12.

The transform itself is **not** here any more: `triton_crypto.primitives.ntt` owns the
negacyclic NTT, parameterised by the domain (q, N, ζ), and ML-KEM is just the
`q = 3329, ζ = 17, int32` instance of it. What stays in this module is what genuinely
depends on ML-KEM's layout:

- `pow_mac`: `t̂[i] = Σ_j Â[i][j] ∘ ŝ[j] + ê[i]`, where `∘` is the *base-case*
  multiplication in Z_q[X]/(X²−γ) (FIPS 203 Alg. 11/12) — even/odd coefficients form
  two independent channels, which is a property of ML-KEM's NTT layout only.
- `byte_encode12`: ByteEncode_12 for `ek`/`dk`.

ML-KEM still needs an NTT at all only in keygen: decaps and encaps fold their
multiplications into Toeplitz GEMMs. `ek` must carry `t̂`, and `s`/`e` are new for every
key, so no matrix can be pre-expanded — so keygen calls the generic primitive.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ...devices import warps
from ...primitives.ntt import KYBER, ntt_torch
from ..ntt_ref import GAMMAS

Q = 3329
QL = tl.constexpr(Q)
N = 256


_GM: dict = {}


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


def forward_ntt(x: torch.Tensor, br: int = 2) -> torch.Tensor:
    """[P, 256] int32 (coefficient domain, [0,q)) → [P, 256] int32 (NTT domain)."""
    return ntt_torch(x, KYBER, br=br, warps=warps("ntt", 2))


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


__all__ = ["forward_ntt", "pow_mac", "pow_mac_kernel"]


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
