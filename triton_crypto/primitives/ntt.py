"""Generic negacyclic NTT over Z_q[X]/(X^N+1), for N = 256 and any NTT-friendly q.

Both standards this package implements use the same butterfly and the same twiddle
ordering (`ζ^BitRev7(k)`), and differ only in the modulus, in the order of ζ, and in
whether the transform runs to the last layer:

    ML-KEM (FIPS 203): q = 3329,    ζ = 17,   order N    -> 7 layers, int32
    ML-DSA (FIPS 204): q = 8380417, ζ = 1753, order 2N   -> 8 layers, int64

ML-KEM stops one layer early on purpose: its 128 CRT slots are the quadratic extensions
Z_q[X]/(X²−γ), which is why FIPS 203 Alg. 12 needs a base-case multiply. ML-DSA splits
all the way into 256 linear slots, so its pointwise product is a plain Hadamard. The
butterfly is identical; what the products *mean* afterwards is the scheme's business, and
stays in the scheme's module (`triton_crypto.ntt_ref.multiply_ntts`).

Everything here works in the **standard domain**: no Montgomery representation leaks into
the twiddle tables, so the output is directly comparable with the coefficients in
FIPS 203/204. (The reference implementations fold a factor of R⁻¹ into their tables,
which is exactly the kind of detail that makes cross-scheme reuse painful.)

Primitives:
    NTTDomain      q, N, ζ, layers  (+ cached twiddle tables, both directions)
    ntt_np/intt_np numpy reference (same table layout as the kernel)
    ntt_torch      batched Triton kernel: [P, N] int32 -> [P, N] int32
    mul_np         pointwise (Hadamard) product mod q
    negacyclic_conv_np  schoolbook O(N²) product, used to test the transform itself

The Triton kernel maps one program to `BR` polynomials and one butterfly layer to a
single fused vector op: `tl.gather` swaps the two halves of each block (index `idx ^ L`),
then one multiply and two reductions produce the low and high outputs — folding a/b into
the halves with `where` costs one multiply instead of two.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

__all__ = [
    "DILITHIUM",
    "KYBER",
    "NTTDomain",
    "bitrev",
    "intt_np",
    "intt_torch",
    "mldsa_domain",
    "mlkem_domain",
    "mul_np",
    "negacyclic_conv_np",
    "ntt_np",
    "ntt_torch",
]


def bitrev(k: int, bits: int) -> int:
    """`bits`-wide bit reversal (BitRev7 for ML-KEM, BitRev8 for ML-DSA)."""
    return int(format(k, f"0{bits}b")[::-1], 2)


def _multiplicative_order(a: int, q: int) -> int:
    """Order of `a` in (Z/q)ˣ, by removing prime factors of q-1 while the power stays 1."""
    order = q - 1
    rest = order
    p = 2
    while p * p <= rest:
        if rest % p == 0:
            while rest % p == 0:
                rest //= p
            while order % p == 0 and pow(a, order // p, q) == 1:
                order //= p
        p += 1
    if rest > 1:
        while order % rest == 0 and pow(a, order // rest, q) == 1:
            order //= rest
    return order


@dataclass(frozen=True)
class NTTDomain:
    """A negacyclic NTT domain: Z_q[X]/(X^N+1), a root ζ, and a layer count."""

    name: str
    q: int
    n: int
    zeta: int
    layers: int = 0  # 0 -> run all log2(N) layers

    def __post_init__(self):
        if self.n & (self.n - 1):
            raise ValueError("N must be a power of two")
        if self.layers == 0:
            object.__setattr__(self, "layers", self.n.bit_length() - 1)
        order = _multiplicative_order(self.zeta, self.q)
        if order != 2 ** (self.layers + 1):
            raise ValueError(
                f"ζ={self.zeta} has order {order} mod {self.q}, but {self.layers} layers "
                f"need order {2 ** (self.layers + 1)}"
            )

    @property
    def log_n(self) -> int:
        return self.n.bit_length() - 1

    @property
    def half(self) -> int:
        return self.n // 2

    @property
    def wide(self) -> bool:
        """Whether products of two reduced coefficients overflow int32."""
        return (self.q - 1) * (self.q - 1) > 2**31 - 1

    @cached_property
    def zetas(self) -> np.ndarray:
        """ζ^BitRev(k), k = 0 .. 2^layers-1 (the FIPS 203/204 twiddle order)."""
        return np.array(
            [pow(self.zeta, bitrev(k, self.layers), self.q) for k in range(2**self.layers)],
            dtype=np.int64,
        )

    @cached_property
    def scale(self) -> int:
        """2^-layers mod q: the normalisation the inverse needs in the standard domain.

        Not N⁻¹ in general: ML-KEM's transform stops after 7 layers, so it has 2^7 CRT
        slots and FIPS 203 Alg. 10 normalises by 128⁻¹ (= 3303), not 256⁻¹.
        """
        return pow(2**self.layers, -1, self.q)

    # layer geometry, shared by every implementation below
    def layer_len(self, s: int, inverse: bool = False) -> int:
        """Length `L` of the butterfly blocks in layer `s` (`0 <= s < layers`)."""
        if inverse:
            return 1 << (self.log_n - self.layers + s)
        return 1 << (self.log_n - 1 - s)

    @cached_property
    def qinv(self) -> int:
        """q⁻¹ mod 2³² — the Montgomery constant (only used on the wide path)."""
        return pow(self.q, -1, 2**32)

    @cached_property
    def mont_r(self) -> int:
        """R = 2³² mod q."""
        return pow(2, 32, self.q)

    @cached_property
    def tables(self) -> dict[str, np.ndarray]:
        """Per-position twiddles, `[layers*N]` each, ready for the kernel.

        Forward layer `s` uses zetas `2^s .. 2^(s+1)-1` ascending with the block index;
        the inverse walks the same blocks in reverse (FIPS 203 Alg. 10 / FIPS 204 Alg. 42),
        which is what makes one kernel serve both directions.
        """
        z = self.zetas
        n, layers = self.n, self.layers
        fwd = np.empty(layers * n, dtype=np.int64)
        inv = np.empty(layers * n, dtype=np.int64)
        for s in range(layers):
            for direction in ("fwd", "inv"):
                length = self.layer_len(s, inverse=(direction == "inv"))
                b = np.arange(n) // (2 * length)
                if direction == "fwd":
                    idx = (1 << s) + b
                    fwd[s * n : (s + 1) * n] = z[idx]
                else:
                    k0 = 1 << (self.layers - s)  # = 2 * blocks
                    inv[s * n : (s + 1) * n] = z[k0 - 1 - b]
        out = {"fwd": fwd, "inv": inv}
        if self.wide:
            # Montgomery form for the GPU path: the butterfly then computes
            # mont(zeta·R, b) = zeta·b, i.e. the values stay in the standard domain but the
            # reduction is a multiply-shift instead of a 64-bit division.
            r = self.mont_r
            out["fwd_mont"] = (fwd * r) % self.q
            out["inv_mont"] = (inv * r) % self.q
        return out

    # ---- numpy reference (same table layout as the kernel) ----

    def _reduce(self, x):
        return np.asarray(x, dtype=np.int64) % self.q

    def ntt(self, f) -> np.ndarray:
        """Forward transform: one vectorised gather per layer (same tables as the kernel)."""
        f = self._reduce(f).copy()
        q, n = self.q, self.n
        tab = self.tables["fwd"]
        for s in range(self.layers):
            length = self.layer_len(s)
            zetas = tab[s * n : (s + 1) * n : 2 * length]  # one twiddle per block
            x = f.reshape(n // (2 * length), 2 * length)
            a = x[:, :length].copy()
            b = x[:, length:]
            t = (zetas[:, None] * b) % q
            x[:, :length] = (a + t) % q
            x[:, length:] = (a - t) % q
        return f

    def intt(self, f) -> np.ndarray:
        """Inverse transform: layers reversed, then the 2^-layers normalisation."""
        f = self._reduce(f).copy()
        q, n = self.q, self.n
        tab = self.tables["inv"]
        for s in range(self.layers):
            length = self.layer_len(s, inverse=True)
            zetas = tab[s * n : (s + 1) * n : 2 * length]
            x = f.reshape(n // (2 * length), 2 * length)
            a = x[:, :length].copy()
            b = x[:, length:].copy()
            x[:, :length] = (a + b) % q
            x[:, length:] = (zetas[:, None] * (b - a)) % q
        return (f * self.scale) % q

    def mul(self, a, b) -> np.ndarray:
        """Pointwise (Hadamard) product mod q."""
        return (np.asarray(a, dtype=np.int64) * np.asarray(b, dtype=np.int64)) % self.q

    def conv(self, a, b) -> np.ndarray:
        """Schoolbook negacyclic convolution (O(N²)); for tests, not for use."""
        n = self.n
        a = np.asarray(a, dtype=np.int64)
        b = np.asarray(b, dtype=np.int64)
        idx = np.arange(n)[:, None] + np.arange(n)[None, :]
        sign = np.where(idx < n, 1, -1)
        c = np.zeros(n, dtype=np.int64)
        np.add.at(c, idx % n, np.outer(a, b) * sign)
        return c % self.q

    def toeplitz(self, a) -> np.ndarray:
        """`T(a)[i][j] = a[(i-j) mod N] · (+1 if i >= j else -1)`, so `T(a) @ b == a·b`."""
        a = np.asarray(a, dtype=np.int64)
        i = np.arange(self.n)[:, None]
        j = np.arange(self.n)[None, :]
        return a[(i - j) % self.n] * np.where(i >= j, 1, -1)


KYBER = NTTDomain("ML-KEM", q=3329, n=256, zeta=17, layers=7)
DILITHIUM = NTTDomain("ML-DSA", q=8380417, n=256, zeta=1753, layers=8)

# Backend-friendly aliases: the names match the schemes, the objects are the domains.
mlkem_domain = KYBER
mldsa_domain = DILITHIUM


# ---------------- numpy entry points (module level, scheme-agnostic) ----------------


def ntt_np(x, domain: NTTDomain = KYBER) -> np.ndarray:
    """[..., N] integers -> NTT domain (int64)."""
    x = np.asarray(x, dtype=np.int64)
    if x.shape[-1] != domain.n:
        raise ValueError(f"expected last axis {domain.n}, got {x.shape}")
    flat = x.reshape(-1, domain.n)
    return np.stack([domain.ntt(row) for row in flat]).reshape(x.shape)


def intt_np(x, domain: NTTDomain = KYBER) -> np.ndarray:
    """[..., N] NTT-domain integers -> coefficients (int64, standard domain)."""
    x = np.asarray(x, dtype=np.int64)
    if x.shape[-1] != domain.n:
        raise ValueError(f"expected last axis {domain.n}, got {x.shape}")
    flat = x.reshape(-1, domain.n)
    return np.stack([domain.intt(row) for row in flat]).reshape(x.shape)


def mul_np(a, b, domain: NTTDomain = KYBER) -> np.ndarray:
    """Pointwise product in the NTT domain."""
    return domain.mul(a, b)


def negacyclic_conv_np(a, b, domain: NTTDomain = KYBER) -> np.ndarray:
    """Schoolbook product in Z_q[X]/(X^N+1), for tests."""
    return domain.conv(a, b)


# ---------------- Triton kernel ----------------


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


def _make_kernel():
    import triton
    import triton.language as tl

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
                    d = _norm(b - a, Q)
                    prod = z * d
                    u = ((prod & 0xFFFFFFFF) * QINV) & 0xFFFFFFFF
                    hi = (prod - u * Q) >> 32
                else:
                    hi = _mod(z * _mod(b - a, Q), Q)
                x = tl.where(lo, a + b, hi)
            else:
                if MONT:
                    prod = z * b
                    u = ((prod & 0xFFFFFFFF) * QINV) & 0xFFFFFFFF
                    t = (prod - u * Q) >> 32
                else:
                    t = _mod(z * b, Q)
                x = tl.where(lo, a + t, a - t)
            if MONT:
                x = _norm(x, Q)
            else:
                x = _mod(x, Q)

        if INVERSE:
            if MONT:
                prod = x * SCALE
                u = ((prod & 0xFFFFFFFF) * QINV) & 0xFFFFFFFF
                x = _norm((prod - u * Q) >> 32, Q)
            else:
                x = _mod(x * SCALE, Q)
        else:
            if MONT:
                x = _norm(x, Q)
        tl.store(OUT + rows[:, None] * N + idx[None, :], x.to(tl.int32), mask=m[:, None])

    return ntt_kernel


_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _make_kernel()
    return _KERNEL


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
    import triton

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
    _kernel()[(triton.cdiv(p, br),)](
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
