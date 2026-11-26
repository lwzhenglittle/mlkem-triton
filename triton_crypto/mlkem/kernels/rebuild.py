"""Rebuild `W_dec` / `W_enc` from `dk` on the GPU (mirrors cuPQC's "stateless,
per-ciphertext Â" approach).

**This module is not on the production path.** It exists to answer a
measurement question:

    cuPQC's decaps is a stateless API; every ciphertext re-runs k² SampleNTT
    calls (SHAKE-128, 840 B each) to obtain Â. Our main path pre-expands Â into
    Toeplitz matrices that stay resident in device memory, so each ciphertext
    costs zero SHAKE. So the "we are 1.61× faster than cuPQC" figure does not
    tell how much of it comes from "skipping Â generation" and how much from
    "Toeplitz GEMM instead of NTT".

Moving the rebuild into the timed region as well makes the two separable.

Flow (entirely on device; only 1.5 KB of input comes from the host):

    dk ──┬─ rho(32B) ──▶ SHAKE-128(rate 168) × k² ──▶ rejection sampling ──▶ Â_hat
         ├─ t_hat   ──▶ ByteDecode12 ──┐
         └─ s_hat   ──▶ ByteDecode12 ──┤
                                       ▼
                             INTT(256×256 matmul)
                                       ▼
           centering ──▶ Toeplitz expansion ──▶ W_dec / W_enc (fp16)

The INTT uses a precomputed 256×256 linear-transform matrix and is exact in
**fp64**: matrix entries and inputs are all ≤ 3329, a single product is
≤ 3329² = 1.1e7, and a 256-term sum ≤ 2.84e9 < 2^53. (One could also write an
8-stage butterfly on the GPU, but that yields the same numbers with more code;
what we need here is the **cost**, and 15 matmuls of 256 points are negligible.)

Verification: `tests/test_rebuild.py` requires the rebuilt `W_dec` / `W_enc` to
be **element-wise equal** to `keyprep.parse_key` (the CPU numpy path).
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ...devices import resolve_device, warps
from ..ntt_ref import intt
from ..params import PARAMS, N, Q

RATE_SHAKE128 = 168
DOMAIN = 0x1F

# ---------------------------------------------------------------- INTT matrix
_INTT_M: dict[torch.device, torch.Tensor] = {}


def intt_matrix(dev) -> torch.Tensor:
    """Return the **transposed** INTT matrix `MT`, so that `x @ MT == intt(x)`.

    `intt` is linear over Z_q, so `intt(x)[i] = Σ_j M[i,j]·x[j]`, where
    `M[:,j] = intt(e_j)`. For batched work `x @ Mᵀ` is handier (rows are
    samples), so we cache `Mᵀ` directly.
    """
    t = _INTT_M.get(dev)
    if t is None:
        m = np.empty((N, N), dtype=np.float64)
        for j in range(N):
            e = np.zeros(N, dtype=np.int64)
            e[j] = 1
            m[:, j] = intt(e)  # column j = intt(e_j)
        t = torch.from_numpy(np.ascontiguousarray(m.T)).to(dev)
        _INTT_M[dev] = t
    return t


# -------------------------------------------------------------- SHAKE-128 input
def _padded_msgs(rho: bytes, k: int) -> np.ndarray:
    """k² pad10*1 inputs of `rho ‖ j ‖ i`, [k², 168] uint8 (host side, 1.5 KB)."""
    msgs = np.zeros((k * k, RATE_SHAKE128), dtype=np.uint8)
    n = 0
    for i in range(k):
        for j in range(k):
            msgs[n, :32] = np.frombuffer(rho, np.uint8)
            msgs[n, 32] = j
            msgs[n, 33] = i
            msgs[n, 34] = DOMAIN
            msgs[n, RATE_SHAKE128 - 1] |= 0x80
            n += 1
    return msgs


# ------------------------------------------------------------- rejection sampling
@triton.jit
def _sample_ntt_kernel(
    BUF, OUT, NB: tl.constexpr, N: tl.constexpr, QL: tl.constexpr, NCAND: tl.constexpr
):
    """840 B → 256 NTT-domain coefficients in [0,q) (FIPS 203 Alg. 7).

    One program handles one message: expand 280 triples into 560 candidate
    values, take prefix positions via `tl.cumsum`, then scatter by position and
    keep only the first 256 legal values.
    """
    pid = tl.program_id(0)
    base = BUF + pid * NB
    c = tl.arange(0, NCAND)  # candidate index 0..NCAND-1; the first 560 are usable
    tri = c // 2  # triple index
    valid = c < 2 * (NB // 3)
    b0 = tl.load(base + 3 * tri, mask=valid, other=0).to(tl.int32)
    b1 = tl.load(base + 3 * tri + 1, mask=valid, other=0).to(tl.int32)
    b2 = tl.load(base + 3 * tri + 2, mask=valid, other=0).to(tl.int32)
    d1 = b0 + 256 * (b1 & 0x0F)
    d2 = (b1 >> 4) + 16 * b2
    val = tl.where((c % 2) == 0, d1, d2)
    ok = valid & (val < QL)
    pos = tl.cumsum(ok.to(tl.int32), axis=0) - 1  # 0-based position
    tl.store(OUT + pid * N + pos, val, mask=ok & (pos >= 0) & (pos < N))


# ------------------------------------------------------------ Toeplitz expansion
@triton.jit
def _toeplitz_kernel(A, META, OUT, K, N: tl.constexpr, BR: tl.constexpr):
    """Write one 256×256 Toeplitz block into W.

    `T(a)[i][j] = a[(i-j) mod N] * (+1 if i >= j else -1)`, `a` is already
    centered. A: [n_blocks, N] int32 (centered coefficients); META: [n_blocks, 2]
    = (row block, column block).
    """
    pid = tl.program_id(0)
    nrow = tl.cdiv(N, BR)
    blk = pid // nrow  # which Toeplitz block
    r0 = (pid % nrow) * BR
    src = tl.load(META + blk * 3 + 0)  # which row of coef
    row0 = tl.load(META + blk * 3 + 1)  # which row block of W to write
    col0 = tl.load(META + blk * 3 + 2)  # which column block of W
    row = r0 + tl.arange(0, BR)[:, None]
    col = tl.arange(0, N)[None, :]
    # Note: Triton's int32 `%` is C semantics (truncation toward zero), so
    # `(row-col) % 256` gives a negative index for negative values. N is a
    # power of 2, so use bitwise and instead. numpy does floor-mod there, so
    # the two are not equivalent here -- we were bitten by this once.
    idx = (row - col) & (N - 1)
    a = tl.load(A + src * N + idx)
    sgn = tl.where(row >= col, 1, -1)
    dst = OUT + (row0 * N + row) * K + (col0 * N + col)
    tl.store(dst, (a * sgn).to(tl.float16))


# ------------------------------------------------------------------ main entry
class GpuKeyRebuilder:
    """Rebuild `W_dec` / `W_enc` from `dk` on the GPU; same interface as
    `keyprep.parse_key`."""

    def __init__(self, ps: str, device=None):
        self.ps = ps
        self.p = PARAMS[ps]
        self.k = self.p.k
        self.device = resolve_device(device)
        self.INTT = intt_matrix(self.device)
        # Metadata triples: (coef row, W row block, W column block). Computed
        # once and kept resident.
        # coef row order = [Â(0..k²-1) row-major, t̂(k²..k²+k-1), ŝ(k²+k..k²+2k-1)]
        k = self.k
        enc, dec = [], []
        for i in range(k):  # u' = Âᵀ y  →  row block i, column block j use Â[j][i]
            for j in range(k):
                enc.append((j * k + i, i, j))  # note A[j][i], not A[i][j]
        for j in range(k):  # v' = t̂ᵀ y
            enc.append((k * k + j, k, j))
        for j in range(k):  # w = -ŝᵀ u
            dec.append((k * k + k + j, 0, j))
        self.meta_enc = torch.tensor(enc, dtype=torch.int32, device=self.device)
        self.meta_dec = torch.tensor(dec, dtype=torch.int32, device=self.device)
        self._sha = None

    # -- SHAKE-128 kernel (compiled on first call) --
    def _shake(self):
        if self._sha is None:
            from ...primitives.sponge import get_sponge

            self._sha = get_sponge(RATE_SHAKE128, 5 * RATE_SHAKE128)
        return self._sha

    @staticmethod
    def rho_of(dk: bytes) -> bytes:
        """Extract ρ from `dk` (the last 32 bytes of ek)."""
        k = (len(dk) - 96) // 768
        return dk[768 * k - 32 : 768 * k]

    def _intt_batch(self, x_hat: torch.Tensor) -> torch.Tensor:
        """[n, 256] NTT domain → coefficient domain, then center to [-1664, 1664] (int32)."""
        c = torch.matmul(x_hat.to(torch.float64), self.INTT)  # x @ Mᵀ == intt(x)
        c = c.to(torch.int64) % Q
        c = torch.where(c > (Q - 1) // 2, c - Q, c)
        return c.to(torch.int32)

    def rebuild(self, dk: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """→ (W_dec [256, 256k] fp16, W_enc [256(k+1), 256k] fp16), both centered."""
        k, dev = self.k, self.device
        dk_pke = dk[: 384 * k]
        ek = dk[384 * k : 768 * k + 32]
        rho = ek[384 * k :]
        if len(rho) != 32:
            raise ValueError(f"rho must be 32 bytes, got {len(rho)}")

        # 1) Â_hat: SHAKE-128 × k² + rejection sampling
        n = k * k
        outw = 5 * RATE_SHAKE128 // 8  # 840/8 = 105
        msgs = torch.from_numpy(_padded_msgs(rho, k)).to(dev)
        out64 = torch.empty((n, outw), dtype=torch.int64, device=dev)
        self._shake()[(triton.cdiv(n, 128),)](
            msgs.view(torch.int64),
            out64,
            n,
            1,
            RATE_SHAKE128 // 8,
            OUTW=outw,
            BLOCK=128,
            num_warps=warps("shake", 4),
        )
        buf = out64.view(torch.uint8).view(n, 5 * RATE_SHAKE128)

        a_hat = torch.empty((n, N), dtype=torch.int32, device=dev)
        _sample_ntt_kernel[(n,)](
            buf, a_hat, 5 * RATE_SHAKE128, N, Q, 1024, num_warps=warps("sample_ntt", 4)
        )

        # 2) t_hat / s_hat: the 12-bit fields in dk
        from ..pipeline import unpack_bits_gpu

        # row order must match coef: [t̂(k of them), ŝ(k of them)]
        ts = torch.cat(
            [
                torch.from_numpy(np.frombuffer(ek[: 384 * k], np.uint8).reshape(k, 384).copy()).to(
                    dev
                ),
                torch.from_numpy(np.frombuffer(dk_pke, np.uint8).reshape(k, 384).copy()).to(dev),
            ]
        )
        ts_hat = unpack_bits_gpu(ts, 12, N).to(torch.int32)  # [2k, 256]

        # 3) INTT everything to the coefficient domain and center
        coef = self._intt_batch(torch.cat([a_hat, ts_hat]))  # [k²+2k, 256]

        # 4) Toeplitz expansion
        M, K = 256 * (k + 1), 256 * k
        W_enc = torch.empty((M, K), dtype=torch.float16, device=dev)
        W_dec = torch.empty((256, K), dtype=torch.float16, device=dev)
        BR = 32
        _toeplitz_kernel[(self.meta_enc.numel() // 3 * (N // BR),)](
            coef, self.meta_enc, W_enc, K, N, BR, num_warps=warps("toeplitz", 4)
        )
        _toeplitz_kernel[(self.meta_dec.numel() // 3 * (N // BR),)](
            coef, self.meta_dec, W_dec, K, N, BR, num_warps=warps("toeplitz", 4)
        )
        return W_dec, W_enc


__all__ = ["GpuKeyRebuilder", "intt_matrix"]
