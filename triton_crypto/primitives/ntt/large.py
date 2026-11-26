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
    X,
    Y,
    M,
    TOTAL,
    LOG_HALF,
    W,
    PRE,
    Q: tl.constexpr,
    FORWARD: tl.constexpr,
    FUSE_PRE: tl.constexpr,
    FUSE_POST: tl.constexpr,
    BLK: tl.constexpr,
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
    if FUSE_PRE:
        # the negacyclic pre-twiddle is index-dependent only, so the first pass can apply it on
        # the way in -- saving a whole launch and 16 MB of the transform's 160 MB
        u = u * tl.load(PRE + base + j, mask=mask, other=1).to(tl.int64) % Q
        v = v * tl.load(PRE + base + half + j, mask=mask, other=1).to(tl.int64) % Q
    if FORWARD:
        w = tl.load(W + j, mask=mask, other=0).to(tl.int64)
        s = u + v
        s = tl.where(s >= Q, s - Q, s)
        d = u - v
        d = tl.where(d < 0, d + Q, d)  # Triton's `%` is C-truncated: negatives stay negative
        if FUSE_POST:
            s = s * tl.load(PRE + base + j, mask=mask, other=1).to(tl.int64) % Q
            v = d * w % Q
            v = v * tl.load(PRE + base + half + j, mask=mask, other=1).to(tl.int64) % Q
        else:
            v = d * w % Q
        tl.store(Y + base + j, s, mask=mask)
        tl.store(Y + base + half + j, v, mask=mask)
    else:
        w = tl.load(W + j, mask=mask, other=0).to(tl.int64)
        vw = v * w % Q
        s = u + vw
        s = tl.where(s >= Q, s - Q, s)
        d = u - vw
        d = tl.where(d < 0, d + Q, d)
        if FUSE_POST:
            s = s * tl.load(PRE + base + j, mask=mask, other=1).to(tl.int64) % Q
            d = d * tl.load(PRE + base + half + j, mask=mask, other=1).to(tl.int64) % Q
        tl.store(Y + base + j, s, mask=mask)
        tl.store(Y + base + half + j, d, mask=mask)


class LargeNTT:
    """A negacyclic NTT of size `n`, multi-pass, for n beyond the in-register kernel's range."""

    def __init__(self, n: int, q: int, radix: int = 4):
        if n & (n - 1):
            raise ValueError(f"n must be a power of two, got {n}")
        if radix not in (2, 4):
            raise ValueError(f"radix must be 2 or 4, got {radix}")
        self.radix = radix
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

    @property
    def n_passes(self) -> int:
        """Passes per transform, for whoever wants to reason about traffic.

        `radix=4` does `log2(n) / 2` passes (plus one radix-2 pass when `log2(n)` is odd);
        `radix=2` does `log2(n)`.
        """
        halves, odd = divmod(self.logn, 2)
        return halves + (1 if odd else 0) if self.radix == 4 else self.logn

    def fwd_passes4(self) -> list[tuple]:
        """Forward passes as `(quarter_length, w1a, w1b, w2)` for radix-4, plus a final
        `(quarter_length, w)` radix-2 pass when `log2(n)` is odd."""
        n, q = self.n, self.q
        omega = self.psi * self.psi % q
        out = []
        m = n
        while m >= 4:
            quarter = m // 4
            step = pow(omega, n // m, q)
            step2 = pow(omega, 2 * n // m, q)
            idx = range(quarter)
            w1a = np.array([pow(step, j, q) for j in idx], dtype=np.int64)
            w1b = np.array([pow(step, j + quarter, q) for j in idx], dtype=np.int64)
            w2 = np.array([pow(step2, j, q) for j in idx], dtype=np.int64)
            out.append((4, m, w1a, w1b, w2))
            m //= 4
        if m == 2:
            out.append((2, 2, self._table(1, omega)))
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

    def _device_tables(self, direction: str, dev):
        key = (direction, str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            import torch

            tables = self.fwd_tables() if direction == "fwd" else self.inv_tables()
            cache[key] = [torch.as_tensor(t, dtype=torch.int32, device=dev) for t in tables]
        return cache[key]

    def _passes4_cached(self, dev):
        key = ("pass4", str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            import torch

            passes = []
            for p in self.fwd_passes4():
                if p[0] == 4:
                    _, m, w1a, w1b, w2 = p
                    passes.append(
                        (
                            4,
                            m,
                            *[
                                torch.as_tensor(w, dtype=torch.int32, device=dev)
                                for w in (w1a, w1b, w2)
                            ],
                        )
                    )
                else:
                    _, m, w = p
                    passes.append((2, m, torch.as_tensor(w, dtype=torch.int32, device=dev)))
            cache[key] = passes
        return cache[key]

    def _run4(self, x, dev):
        import torch

        q = self.q
        # Two scratch buffers and a read-only input: the ping-pong must never write into the
        # caller's tensor, which is what a single `buf` did (see `_run_cached` below).
        buf = torch.empty_like(x)
        alt = torch.empty_like(x)
        for i, p in enumerate(self._passes4_cached(dev)):
            if p[0] == 4:
                _, m, w1a, w1b, w2 = p
                quarter = w1a.numel()
                log_q = quarter.bit_length() - 1
                _stage4_kernel[(triton.cdiv(self.n // 4, STAGE_BLK),)](
                    x,
                    buf,
                    m,
                    self.n // 4,
                    log_q,
                    w1a,
                    w1b,
                    w2,
                    self._device_pre(dev),
                    Q=q,
                    FUSE_PRE=(i == 0),
                    BLK=STAGE_BLK,
                    num_warps=STAGE_WARPS,
                )
            else:
                _, m, w = p
                log_half = (m // 2).bit_length() - 1
                _stage_kernel[(triton.cdiv(self.n // 2, STAGE_BLK),)](
                    x,
                    buf,
                    m,
                    self.n // 2,
                    log_half,
                    w,
                    self._device_pre(dev),
                    Q=q,
                    FORWARD=True,
                    FUSE_PRE=False,
                    FUSE_POST=False,
                    BLK=STAGE_BLK,
                    num_warps=STAGE_WARPS,
                )
            x, buf, alt = buf, alt, buf
        return x

    def _run_cached(self, x, direction: str, dev):
        import torch

        q = self.q
        # The single-buffer version wrote pass 18's output into the caller's tensor on a 19-pass
        # transform, so a second `forward_tensor_torch(x)` transformed the wreckage of the first.
        buf = torch.empty_like(x)
        alt = torch.empty_like(x)
        tables = self._device_tables(direction, dev)
        pre = self._device_pre(dev)
        post = self._device_post(dev)
        for i, w in enumerate(tables):
            half = w.numel()
            log_half = half.bit_length() - 1
            fwd = direction == "fwd"
            _stage_kernel[(triton.cdiv(self.n // 2, STAGE_BLK),)](
                x,
                buf,
                2 * half,
                self.n // 2,
                log_half,
                w,
                pre if fwd or i != len(tables) - 1 else post,
                Q=q,
                FORWARD=fwd,
                FUSE_PRE=(i == 0 and fwd),
                FUSE_POST=(i == len(tables) - 1 and not fwd),
                BLK=STAGE_BLK,
                num_warps=STAGE_WARPS,
            )
            x, buf, alt = buf, alt, buf
        return x

    def _device_post(self, dev):
        import torch

        key = ("post_table", str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            cache[key] = torch.as_tensor(self.post(), dtype=torch.int32, device=dev)
        return cache[key]

    def _device_pre(self, dev):
        import torch

        key = ("pre_table", str(dev))
        cache = self.__dict__.setdefault("_tbl_cache", {})
        if key not in cache:
            cache[key] = torch.as_tensor(self.pre(), dtype=torch.int32, device=dev)
        return cache[key]

    def _mul_dev(self, x, p_np, dev, q, role: str):
        """Elementwise `x * p mod q`. Only the twiddle tables are cached.

        `role="pointwise"` is data, not a table: caching it handed a second product the first
        product's operand, which is the kind of bug that only a comparison against a different
        path (here `mul_np`) catches.
        """
        import torch

        if role in ("pre", "post"):
            key = ("mul", role, p_np.shape[0], str(dev))
            cache = self.__dict__.setdefault("_tbl_cache", {})
            if key not in cache:
                cache[key] = torch.as_tensor(p_np, dtype=torch.int32, device=dev)
            p = cache[key]
        else:
            p = (
                p_np
                if isinstance(p_np, torch.Tensor)
                else torch.as_tensor(p_np, dtype=torch.int32, device=dev)
            )
        out = torch.empty_like(x)
        _mul_kernel[(triton.cdiv(x.numel(), 65536),)](x, p, out, x.numel(), Q=q, BLK=65536)
        return out

    def forward_torch(self, a, device=None):
        import torch

        from ... import devices as _devices

        dev = _devices.resolve_device(device)
        x = torch.as_tensor(np.asarray(a, dtype=np.int64) % self.q, dtype=torch.int32, device=dev)
        return self.forward_tensor_torch(x, dev)

    def _check_tensor(self, x, what: str) -> None:
        """Refuse a tensor this transform cannot handle, before a kernel reads off the end.

        The stage kernel loads `x[base + j]` and `x[base + half + j]` with no mask past the
        buffer, so a wrong length is a silent out-of-bounds read rather than a crash. Measured
        before this check existed: `mul_torch(zeros(64), zeros(32))` returned a result.
        """
        import torch

        if not isinstance(x, torch.Tensor):
            raise TypeError(f"{what} must be a torch tensor, got {type(x).__name__}")
        if x.dim() != 1 or x.numel() != self.n:
            raise ValueError(f"{what} must be 1-D of length {self.n}, got shape {tuple(x.shape)}")

    def forward_tensor_torch(self, x, dev):
        """Device tensor in, device tensor out, tables cached: the timing entry point.

        `radix=4` (the default) fuses two stages per pass, halving the traffic: ten passes of 8 MB
        instead of nineteen. Paired against radix-2 in one window over ten alternating runs at
        2^20 -- both with the pre-twiddle fused -- it is **faster and steadier**: 1.34/1.36 ms
        (min/median) against 1.50/1.67. An earlier comparison said the opposite; that one had
        radix-4 paying for a separate pre-twiddle pass the radix-2 path no longer needed.
        """
        self._check_tensor(x, "input")
        if self.radix == 4:
            return self._run4(x, dev)
        return self._run_cached(x, "fwd", dev)

    def inverse_torch(self, x, device=None):
        import torch

        from ... import devices as _devices

        dev = _devices.resolve_device(device)
        t = torch.as_tensor(np.asarray(x, dtype=np.int64) % self.q, dtype=torch.int32, device=dev)
        return self.inverse_tensor_torch(t, dev)

    def inverse_tensor_torch(self, t, dev):
        self._check_tensor(t, "input")
        return self._run_cached(t, "inv", dev)

    # ------------------------------------------------------------------ products

    def mul_torch(self, a, b, device=None):
        """Negacyclic product in `Z_q[X]/(X^n + 1)`, all on the device: NTT, pointwise, INTT.

        This is the operation a ZK prover actually calls at 2^20, and the reason the multi-pass
        transform exists. Both operands and the result stay on the device.
        """
        import torch

        from ... import devices as _devices

        dev = _devices.resolve_device(device)
        for operand, name in ((a, "a"), (b, "b")):
            # A device tensor is the *documented* way to call this ("both operands and the result
            # stay on the device"), so the shape check must read it where it lives -- routing it
            # through numpy raises "can't convert xpu:0 device type tensor to numpy".
            if hasattr(operand, "shape") and hasattr(operand, "device"):
                shape = tuple(operand.shape)
            else:
                shape = np.asarray(operand).shape
            if len(shape) != 1 or shape[0] != self.n:
                raise ValueError(f"{name} must be 1-D of length {self.n}, got {shape}")

        def _as_device_poly(v):
            if hasattr(v, "to") and hasattr(v, "device"):
                return v.to(device=dev, dtype=torch.int32).contiguous()
            return torch.as_tensor(
                np.asarray(v, dtype=np.int64) % self.q, dtype=torch.int32, device=dev
            )

        fa = self.forward_tensor_torch(_as_device_poly(a), dev)
        fb = self.forward_tensor_torch(_as_device_poly(b), dev)
        prod = self._mul_dev(fa, fb, dev, self.q, "pointwise")
        return self.inverse_tensor_torch(prod, dev)

    def mul_np(self, a, b) -> np.ndarray:
        """The same product through the numpy path, as the oracle."""
        fa = self.ntt(a)
        fb = self.ntt(b)
        return self.intt(fa * fb % self.q)


@triton.jit
def _stage4_kernel(
    X,
    Y,
    M,
    TOTAL,
    LOG_Q,
    W1A,
    W1B,
    W2,
    PRE,
    Q: tl.constexpr,
    FUSE_PRE: tl.constexpr,
    BLK: tl.constexpr,
):
    """Two radix-2 DIF stages fused into one radix-4 pass: ten passes instead of twenty.

    Fusion is exact, not approximate: the second radix-2 stage runs on blocks of `M/2`, so its
    twiddle step is `root^(2N/M)` and it needs the twiddles at positions `j` and `j + M/4` of the
    first stage. Verified against the two-pass sequence in numpy for N = 2^4 .. 2^10, where the
    results are bit-identical -- which also means the output ordering (and therefore pointwise
    compatibility with the in-register kernel) is unchanged.
    """
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    mask = offs < TOTAL
    quarter = 1 << LOG_Q
    blk = offs >> LOG_Q
    j = offs & (quarter - 1)
    base = blk * M
    a0 = tl.load(X + base + j, mask=mask, other=0).to(tl.int64)
    a1 = tl.load(X + base + quarter + j, mask=mask, other=0).to(tl.int64)
    a2 = tl.load(X + base + 2 * quarter + j, mask=mask, other=0).to(tl.int64)
    a3 = tl.load(X + base + 3 * quarter + j, mask=mask, other=0).to(tl.int64)
    if FUSE_PRE:
        # the negacyclic pre-twiddle is index-dependent only: apply it on the way into the first
        # pass instead of spending a launch and 16 MB on a separate pass
        a0 = a0 * tl.load(PRE + base + j, mask=mask, other=1).to(tl.int64) % Q
        a1 = a1 * tl.load(PRE + base + quarter + j, mask=mask, other=1).to(tl.int64) % Q
        a2 = a2 * tl.load(PRE + base + 2 * quarter + j, mask=mask, other=1).to(tl.int64) % Q
        a3 = a3 * tl.load(PRE + base + 3 * quarter + j, mask=mask, other=1).to(tl.int64) % Q
    w1a = tl.load(W1A + j, mask=mask, other=0).to(tl.int64)
    w1b = tl.load(W1B + j, mask=mask, other=0).to(tl.int64)
    w2 = tl.load(W2 + j, mask=mask, other=0).to(tl.int64)
    u = a0 + a2
    u = tl.where(u >= Q, u - Q, u)
    d = a0 - a2
    d = tl.where(d < 0, d + Q, d)
    d = d * w1a % Q
    u2 = a1 + a3
    u2 = tl.where(u2 >= Q, u2 - Q, u2)
    d2 = a1 - a3
    d2 = tl.where(d2 < 0, d2 + Q, d2)
    d2 = d2 * w1b % Q
    s0 = u + u2
    s0 = tl.where(s0 >= Q, s0 - Q, s0)
    s1 = u - u2
    s1 = tl.where(s1 < 0, s1 + Q, s1)
    s2 = d + d2
    s2 = tl.where(s2 >= Q, s2 - Q, s2)
    s3 = d - d2
    s3 = tl.where(s3 < 0, s3 + Q, s3)
    tl.store(Y + base + j, s0, mask=mask)
    tl.store(Y + base + quarter + j, s1 * w2 % Q, mask=mask)
    tl.store(Y + base + 2 * quarter + j, s2, mask=mask)
    tl.store(Y + base + 3 * quarter + j, s3 * w2 % Q, mask=mask)


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
