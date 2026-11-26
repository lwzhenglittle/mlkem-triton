"""Multi-pass NTT for N up to 2^20: one launch per stage, data in global memory.

The in-register kernel (`kernel.py`) holds all N coefficients in registers, which is why it tops
out around N = 1024. A 2^20-point transform is a different algorithm, not a bigger tile: the data
does not fit anywhere but global memory, so every stage reads and writes the whole array. At
N = 2^20 that is 20 stages x 8 MB = 160 MB of traffic, i.e. ~1.4 ms at this device's 112 GB/s --
**bandwidth-bound, not ALU-bound**, which is the opposite of every other primitive here and the
reason this item exists (a CPU needs 15-30 ms for the same transform on one core).

Shape, the same one `primitives/blake3.py` uses for its tree and which took four sessions there:
state lives in a buffer, one launch per pass.

    forward   DIF Stockham, stage block length halves:  N, N/2, ..., 2
    inverse   DIT Stockham, block length doubles:       2, 4, ..., N

The forward's output is in the same (bit-reversed) order `NTTDomain.ntt` uses, so pointwise
multiplication against the in-register kernel's results works, and the DIT pass is its exact
inverse -- verified on N = 2^4 .. 2^12 against `NTTDomain`, and by round trip.

The negacyclic wrap is the usual pre-twiddle: multiply `a[j]` by `psi^j` (psi of order 2N) before
the cyclic transform and by `psi^-j * N^-1` after the inverse.
"""

from __future__ import annotations

import numpy as np
import triton
import triton.language as tl

__all__ = ["LargeNTT", "large_ntt_np", "large_ntt_torch"]

STAGE_BLK = 1024
STAGE_WARPS = 8


@triton.jit
def _stage_kernel(
    X, Y, M, TOTAL, LOG_HALF, W, Q: tl.constexpr, FORWARD: tl.constexpr, BLK: tl.constexpr
):
    """One Stockham pass over a flat butterfly grid.

    `half` is always a power of two, so the block/j split is a shift and a mask rather than the
    `offs // half` int64 division the first version did -- 2^19 divisions per pass, 20 passes,
    with `half` a runtime value (LLVM cannot strength-reduce that). A `(block, tile)` grid was
    tried instead: correct and much slower, because the late stages have half = 1 and the grid
    becomes 524288 one-element programs.
    """
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < TOTAL
    blk = offs >> LOG_HALF
    j = offs & ((1 << LOG_HALF) - 1)
    half = 1 << LOG_HALF
    base = blk * M
    u = tl.load(X + base + j, mask=mask, other=0).to(tl.int64)
    v = tl.load(X + base + half + j, mask=mask, other=0).to(tl.int64)
    if FORWARD:
        w = tl.load(W + j, mask=mask, other=0).to(tl.int64)
        s = u + v
        s = tl.where(s >= Q, s - Q, s)
        d = u - v
        d = tl.where(d < 0, d + Q, d)  # Triton's `%` is C-truncated: negatives stay negative
        tl.store(Y + base + j, s, mask=mask)
        tl.store(Y + base + half + j, d * w % Q, mask=mask)
    else:
        w = tl.load(W + j, mask=mask, other=0).to(tl.int64)
        vw = v * w % Q
        s = u + vw
        s = tl.where(s >= Q, s - Q, s)
        d = u - vw
        d = tl.where(d < 0, d + Q, d)
        tl.store(Y + base + j, s, mask=mask)
        tl.store(Y + base + half + j, d, mask=mask)


class LargeNTT:
    """A negacyclic NTT of size `n`, multi-pass, for n beyond the in-register kernel's range."""

    def __init__(self, n: int, q: int):
        if n & (n - 1):
            raise ValueError(f"n must be a power of two, got {n}")
        self.n = n
        self.q = q
        self.logn = n.bit_length() - 1
        self.psi = None
        for g in range(2, 256):
            z = pow(g, (q - 1) // (2 * n), q)
            if pow(z, n, q) == q - 1:
                self.psi = z
                break
        if self.psi is None:
            raise ValueError(f"no element of order {2 * n} mod {q}")

    # ------------------------------------------------------------------ tables

    def _table(self, half: int, root: int) -> np.ndarray:
        n, q = self.n, self.q
        m = 2 * half
        step = n // m
        return np.array([pow(root, step * j, q) for j in range(half)], dtype=np.int64)

    def fwd_tables(self) -> list[np.ndarray]:
        omega = self.psi * self.psi % self.q
        m = self.n
        out = []
        while m >= 2:
            out.append(self._table(m // 2, omega))
            m //= 2
        return out

    def inv_tables(self) -> list[np.ndarray]:
        omega = self.psi * self.psi % self.q
        inv_root = pow(omega, -1, self.q)
        out = []
        half = 1
        while half < self.n:
            out.append(self._table(half, inv_root))
            half *= 2
        return out

    @staticmethod
    def _powers(g: int, n: int, q: int) -> np.ndarray:
        """`[g^0, g^1, ..., g^(n-1)]` in log n numpy passes.

        The obvious `for j in range(n)` is a million Python multiplications at n = 2^20 -- over a
        second, which is three orders of magnitude more than the transform it feeds.
        """
        pw = np.ones(1, dtype=np.int64)
        while pw.size < n:
            step = pow(g, pw.size, q)
            pw = np.concatenate([pw, pw * step % q])
        return pw[:n]

    def pre(self) -> np.ndarray:
        """Cached: rebuilding it costs a 2^20-element numpy pass, several times the transform."""
        cache = self.__dict__.setdefault("_pre_cache", {})
        if "pre" not in cache:
            cache["pre"] = self._powers(self.psi, self.n, self.q)
        return cache["pre"]

    def post(self) -> np.ndarray:
        cache = self.__dict__.setdefault("_pre_cache", {})
        if "post" not in cache:
            inv_psi = pow(self.psi, -1, self.q)
            cache["post"] = self._powers(inv_psi, self.n, self.q) * pow(self.n, -1, self.q) % self.q
        return cache["post"]

    # ------------------------------------------------------------------ numpy

    def ntt(self, a) -> np.ndarray:
        x = np.asarray(a, dtype=np.int64) % self.q
        y = x * self.pre() % self.q
        buf = np.empty_like(y)
        m = self.n
        for tw in self.fwd_tables():
            half = m // 2
            blocks = y.reshape(-1, m)
            out = buf.reshape(-1, m)
            u = blocks[:, :half]
            v = blocks[:, half:]
            out[:, :half] = (u + v) % self.q
            out[:, half:] = (u - v) % self.q * tw % self.q
            y, buf = buf, y
            m = half
        return y % self.q

    def intt(self, x) -> np.ndarray:
        a = np.asarray(x, dtype=np.int64) % self.q
        buf = np.empty_like(a)
        half = 1
        for tw in self.inv_tables():
            m = 2 * half
            blocks = a.reshape(-1, m)
            out = buf.reshape(-1, m)
            vw = blocks[:, half:] * tw % self.q
            out[:, :half] = (blocks[:, :half] + vw) % self.q
            out[:, half:] = (blocks[:, :half] - vw) % self.q
            a, buf = buf, a
            half = m
        return a * self.post() % self.q

    # ------------------------------------------------------------------ device

    def _run(self, x, direction: str, dev):
        import torch

        q = self.q
        buf = torch.empty_like(x)
        tables = self.fwd_tables() if direction == "fwd" else self.inv_tables()
        for tw in tables:
            half = tw.size
            log_half = half.bit_length() - 1
            w = torch.as_tensor(tw, dtype=torch.int32, device=dev)
            _stage_kernel[(triton.cdiv(self.n // 2, STAGE_BLK),)](
                x,
                buf,
                2 * half,
                self.n // 2,
                log_half,
                w,
                Q=q,
                FORWARD=(direction == "fwd"),
                BLK=STAGE_BLK,
                num_warps=STAGE_WARPS,
            )
            x, buf = buf, x
        return x

    def _device_tables(self, direction: str, dev):
        key = (direction, str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            import torch

            tables = self.fwd_tables() if direction == "fwd" else self.inv_tables()
            cache[key] = [torch.as_tensor(t, dtype=torch.int32, device=dev) for t in tables]
        return cache[key]

    def _run_cached(self, x, direction: str, dev):
        import torch

        q = self.q
        buf = torch.empty_like(x)
        for w in self._device_tables(direction, dev):
            half = w.numel()
            log_half = half.bit_length() - 1
            _stage_kernel[(triton.cdiv(self.n // 2, STAGE_BLK),)](
                x,
                buf,
                2 * half,
                self.n // 2,
                log_half,
                w,
                Q=q,
                FORWARD=(direction == "fwd"),
                BLK=STAGE_BLK,
                num_warps=STAGE_WARPS,
            )
            x, buf = buf, x
        return x

    def _mul_dev(self, x, p_np, dev, q, role: str):
        import torch

        key = ("mul", role, p_np.shape[0], str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            cache[key] = torch.as_tensor(p_np, dtype=torch.int32, device=dev)
        p = cache[key]
        out = torch.empty_like(x)
        _mul_kernel[(triton.cdiv(x.numel(), 65536),)](x, p, out, x.numel(), Q=q, BLK=65536)
        return out

    def forward_torch(self, a, device=None):
        import torch

        from ... import devices as _devices

        dev = _devices.resolve_device(device)
        x = torch.as_tensor(np.asarray(a, dtype=np.int64) % self.q, dtype=torch.int32, device=dev)
        return self.forward_tensor_torch(x, dev)

    def forward_tensor_torch(self, x, dev):
        """Device tensor in, device tensor out, tables cached: the timing entry point."""
        x = self._mul_dev(x, self.pre(), dev, self.q, "pre")
        return self._run_cached(x, "fwd", dev)

    def inverse_torch(self, x, device=None):
        import torch

        from ... import devices as _devices

        dev = _devices.resolve_device(device)
        t = torch.as_tensor(np.asarray(x, dtype=np.int64) % self.q, dtype=torch.int32, device=dev)
        return self.inverse_tensor_torch(t, dev)

    def inverse_tensor_torch(self, t, dev):
        y = self._run_cached(t, "inv", dev)
        return self._mul_dev(y, self.post(), dev, self.q, "post")

    @staticmethod
    def _mul(x, p, dev, q):
        import torch

        from ... import devices as _devices

        out = torch.empty_like(x)
        _mul_kernel[(triton.cdiv(x.numel(), 65536),)](x, p, out, x.numel(), Q=q, BLK=65536)
        _devices.sync(dev)
        return out


@triton.jit
def _mul_kernel(X, P, OUT, N, Q: tl.constexpr, BLK: tl.constexpr):
    offs = tl.program_id(0) * BLK + tl.arange(0, BLK)
    mask = offs < N
    a = tl.load(X + offs, mask=mask, other=0).to(tl.int64)
    b = tl.load(P + offs, mask=mask, other=0).to(tl.int64)
    tl.store(OUT + offs, a * b % Q, mask=mask)


def large_ntt_np(a, n: int, q: int) -> np.ndarray:
    return LargeNTT(n, q).ntt(a)


def large_ntt_torch(a, n: int, q: int, device=None):
    return LargeNTT(n, q).forward_torch(a, device)
