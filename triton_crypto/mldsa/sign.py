"""Batched ML-DSA signing on the device (FIPS 204 Alg. 7).

Signing is verification plus a rejection loop: each attempt samples `y` (a SHAKE call per
polynomial), computes `w = A·y`, derives `c̃` from μ and `w1Encode(w)`, and then either accepts —
when `z`, `r0` and `c·t0` are all small enough — or bumps κ and tries again. The loop is the only
part of ML-DSA that is not a straight-line circuit, so the batch is compacted after every attempt:
accepted signatures leave, the rest continue, and the work shrinks as they succeed.

Everything heavy is the verifier's machinery: `ntt_torch` / `intt_torch`, `mac_kernel` for
`Â·ŷ`, `sample_in_ball_kernel` for `c`, `unpack_z_kernel` for `ExpandMask`, and
`w1_pack_kernel` for `w1Encode`. What signing adds is the control flow, `MakeHint`, and the
packing of `z` and the hint — the last two on the host with the (ACVP-checked) reference bit
packers.
"""

from __future__ import annotations

import numpy as np
import torch
import triton

from ..devices import resolve_device
from ..primitives.ntt import DILITHIUM, intt_torch, ntt_torch
from ..primitives.sponge import DOMAIN_SHAKE, SHAKE256, padded_len, sponge_padded
from .kernels import (
    mac_kernel,
    pack_z_kernel,
    pointwise_kernel,
    sample_in_ball_kernel,
    unpack_z_kernel,
    w1_pack_kernel,
)
from .params import PARAMS, DSAParams
from .params import Q as Q_VALUE
from .ref import MLDSA

RATE_SHAKE = SHAKE256.rate
MAC_BN = 8
MAC_WARPS = 4
SIB_WARPS = 4
BC = 64


def _decompose(w: torch.Tensor, p: DSAParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose_γ2 (FIPS 204 Alg. 36) on int64 tensors."""
    r = w % Q_VALUE
    two = 2 * p.gamma2
    r0 = r % two
    r0 = torch.where(r0 > p.gamma2, r0 - two, r0)
    r1 = (r - r0) // two
    special = (r - r0) == Q_VALUE - 1
    r1 = torch.where(special, torch.zeros_like(r1), r1)
    r0 = torch.where(special, r0 - 1, r0)
    return r1, r0


def _center(x: torch.Tensor) -> torch.Tensor:
    return ((x + Q_VALUE // 2) % Q_VALUE) - Q_VALUE // 2


class Signer:
    """Sign any number of messages under one ML-DSA private key."""

    def __init__(self, ps: str, sk: bytes, device=None):
        if ps not in PARAMS:
            raise KeyError(f"unknown parameter set {ps}; known: {sorted(PARAMS)}")
        self.ps = ps
        self.p: DSAParams = PARAMS[ps]
        self.device = resolve_device(device)
        self.sk = bytes(sk)
        self.ref = MLDSA(ps)
        self.rho, self.key, self.tr, s1, s2, t0 = self.ref.unpack_sk(self.sk)
        dev = self.device
        self.A_hat = torch.from_numpy(
            self.ref.expand_a(self.rho).reshape(-1, 256).astype(np.int32)
        ).to(dev)
        self.s1_hat = torch.from_numpy(ntt_np_int32_cols(s1, self)).to(dev)
        self.s2_hat = torch.from_numpy(ntt_np_int32_cols(s2, self)).to(dev)
        self.t0_hat = torch.from_numpy(ntt_np_int32_cols(t0, self)).to(dev)
        self._zero_ch = torch.zeros((1, 256), dtype=torch.int32, device=dev)
        self._zero_t1 = torch.zeros((self.p.k, 256), dtype=torch.int32, device=dev)

    # ---------------------------------------------------------------- sampling
    def _expand_mask(self, rhoprime: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
        """y = ExpandMask(ρ′, κ): ℓ SHAKE calls per signature, then γ1-bounded unpacking."""
        p = self.p
        b = rhoprime.shape[0]
        poly_bytes = p.polyz_bytes
        rows = b * p.l
        seed = rhoprime.repeat_interleave(p.l, dim=0)  # [b*l, 64]
        offs = (kappa.repeat_interleave(p.l) + torch.arange(p.l, device=self.device).repeat(b)).to(
            torch.int64
        )
        msg = torch.zeros((rows, 64 + 2), dtype=torch.uint8, device=self.device)
        msg[:, :64] = seed
        # FIPS 204 Alg. 34: the counter is little-endian
        msg[:, 64] = (offs & 0xFF).to(torch.uint8)
        msg[:, 65] = (offs >> 8).to(torch.uint8)
        total = padded_len(66, RATE_SHAKE)
        buf = torch.zeros((rows, total), dtype=torch.uint8, device=self.device)
        buf[:, :66] = msg
        buf[:, 66] = DOMAIN_SHAKE
        buf[:, total - 1] = buf[:, total - 1] | 0x80
        stream = sponge_padded(buf, RATE_SHAKE, poly_bytes)
        y = torch.empty((rows, 256), dtype=torch.int32, device=self.device)
        unpack_z_kernel[(rows,)](
            stream,
            y,
            poly_bytes,
            rows,
            0,
            poly_bytes,
            p.z_bits,
            p.gamma1,
            256,
            1,
        )
        return y.view(b, p.l, 256)

    # ---------------------------------------------------------------- signing
    def sign(self, messages, ctx: bytes = b"", prehash: str | None = None, rnd=None) -> np.ndarray:
        """The external interface: M′ = 0x00 ‖ len(ctx) ‖ ctx ‖ M (or the pre-hash form)."""
        rows = [
            bytes(m)
            for m in (messages if not isinstance(messages, (bytes, bytearray)) else [messages])
        ]
        if not rows:
            return np.empty((0, 0), dtype=np.uint8)
        return self.sign_mprime([self.ref.format_message(m, ctx, prehash) for m in rows], rnd)

    def sign_mprime(self, mprimes, rnd=None) -> np.ndarray:
        """The internal interface: each message is already M′ (FIPS 204 Alg. 7's input)."""
        mprime = [bytes(m) for m in mprimes]
        if not mprime:
            return np.empty((0, 0), dtype=np.uint8)
        lengths = {len(m) for m in mprime}
        if len(lengths) > 1:
            # the SHAKE over μ needs a rectangular buffer, so lengths are signed in groups
            out = np.empty((len(mprime), self.p.sig_size), dtype=np.uint8)
            for length in lengths:
                idx = [i for i, m in enumerate(mprime) if len(m) == length]
                out[idx] = self.sign_mprime([mprime[i] for i in idx], rnd)
            return out
        mprimes_np = np.stack([np.frombuffer(m, dtype=np.uint8) for m in mprime])
        return self.sign_mu_rows(self._mu(mprimes_np, len(mprime[0])), rnd)

    def sign_mu(self, mus, rnd=None) -> np.ndarray:
        """The external-μ interface: each row is already μ (64 bytes)."""
        arr = np.ascontiguousarray(
            np.stack([np.frombuffer(bytes(m), dtype=np.uint8) for m in mus])
            if not isinstance(mus, np.ndarray)
            else mus,
            dtype=np.uint8,
        )
        mu = torch.from_numpy(arr).to(self.device)
        return self.sign_mu_rows(mu, rnd)

    def sign_mu_rows(self, mu: torch.Tensor, rnd=None) -> np.ndarray:
        """Sign from precomputed μ: [B, 64] uint8 on the device."""
        p = self.p
        b = mu.shape[0]
        dev = self.device
        rnd_b = b"\x00" * 32 if rnd is None else bytes(rnd)
        if len(rnd_b) != 32:
            raise ValueError("rnd must be 32 bytes")
        rho_prime = self._shake32(
            torch.cat(
                [
                    torch.frombuffer(bytearray(self.key), dtype=torch.uint8)
                    .to(dev)
                    .reshape(1, 32)
                    .repeat(b, 1),
                    torch.frombuffer(bytearray(rnd_b), dtype=torch.uint8)
                    .to(dev)
                    .reshape(1, 32)
                    .repeat(b, 1),
                    mu,
                ],
                dim=1,
            ),
            64,
        )

        remaining = torch.arange(b, device=dev)
        kappa = torch.zeros(b, dtype=torch.int64, device=dev)
        sigs: list[bytes | None] = [None] * b
        for _attempt in range(1000):
            if remaining.numel() == 0:
                break
            mu_a = mu[remaining]
            rp_a = rho_prime[remaining]
            kap_a = kappa[remaining]
            y = self._expand_mask(rp_a, kap_a)  # [n, l, 256]
            n = y.shape[0]
            y_hat = ntt_torch(y.reshape(n * p.l, 256), DILITHIUM).reshape(n, p.l, 256)
            w_hat = torch.empty((n, p.k, 256), dtype=torch.int32, device=dev)
            mac_kernel[(triton.cdiv(n, MAC_BN), p.k)](
                y_hat,
                self.A_hat,
                self._zero_ch.expand(n, 256),
                self._zero_t1,
                w_hat,
                n,
                p.k,
                p.l,
                256,
                Q_VALUE,
                MAC_BN,
                num_warps=MAC_WARPS,
            )
            w = intt_torch(w_hat.reshape(n * p.k, 256), DILITHIUM).reshape(n, p.k, 256)
            c_tilde = self._ctilde(mu_a, w)
            c = self._sample_in_ball(c_tilde)
            c_hat = ntt_torch(c, DILITHIUM)
            cs1 = intt_torch(
                (_pointwise(c_hat, self.s1_hat)).reshape(n * p.l, 256), DILITHIUM
            ).reshape(n, p.l, 256)
            z = y + cs1
            z = _center(z)
            z_ok = z.abs().amax(dim=(1, 2)) < p.norm_bound
            cs2 = intt_torch(
                (_pointwise(c_hat, self.s2_hat)).reshape(n * p.k, 256), DILITHIUM
            ).reshape(n, p.k, 256)
            w_minus = (w - cs2) % Q_VALUE
            _, r0 = _decompose(w_minus, p)
            r0_ok = r0.abs().amax(dim=(1, 2)) < p.gamma2 - p.beta
            ct0 = _center(
                intt_torch(
                    (_pointwise(c_hat, self.t0_hat)).reshape(n * p.k, 256), DILITHIUM
                ).reshape(n, p.k, 256)
            )
            ct0_ok = ct0.abs().amax(dim=(1, 2)) < p.gamma2
            h = (self._high_bits(w_minus + ct0, p)[0] != self._high_bits(w_minus, p)[0]).to(
                torch.int32
            )
            h_ok = h.sum(dim=(1, 2)) <= p.omega
            accept = z_ok & r0_ok & ct0_ok & h_ok
            if accept.any():
                idx = remaining[accept]
                z_bytes = self._pack_z(z[accept].contiguous())  # on device, one launch
                h_np = h[accept].cpu().numpy().astype(np.int64)
                ct_np = c_tilde[accept].cpu().numpy()
                z_np = z_bytes.cpu().numpy()
                for r in range(idx.numel()):
                    sigs[int(idx[r])] = b"".join(
                        [ct_np[r].tobytes(), z_np[r].tobytes(), self.ref.pack_h(h_np[r])]
                    )
            keep = ~accept
            remaining = remaining[keep]
            kappa[remaining] = kappa[remaining] + p.l
        if any(s is None for s in sigs):
            raise RuntimeError("signing did not converge")
        out = np.empty((b, len(sigs[0]) if sigs[0] is not None else 0), dtype=np.uint8)
        for i, s in enumerate(sigs):
            out[i] = np.frombuffer(s, dtype=np.uint8)
        return out

    # ---------------------------------------------------------------- helpers
    def _mu(self, mprimes: np.ndarray, length: int) -> torch.Tensor:
        b = mprimes.shape[0]
        total = padded_len(64 + length, RATE_SHAKE)
        buf = torch.zeros((b, total), dtype=torch.uint8, device=self.device)
        buf[:, :64] = torch.frombuffer(bytearray(self.tr), dtype=torch.uint8).to(self.device)
        buf[:, 64 : 64 + length] = torch.from_numpy(np.ascontiguousarray(mprimes)).to(self.device)
        buf[:, 64 + length] = DOMAIN_SHAKE
        buf[:, total - 1] = buf[:, total - 1] | 0x80
        return sponge_padded(buf, RATE_SHAKE, 64)

    def _shake32(self, data: torch.Tensor, out: int) -> torch.Tensor:
        b, ln = data.shape
        total = padded_len(ln, RATE_SHAKE)
        buf = torch.zeros((b, total), dtype=torch.uint8, device=self.device)
        buf[:, :ln] = data
        buf[:, ln] = DOMAIN_SHAKE
        buf[:, total - 1] = buf[:, total - 1] | 0x80
        return sponge_padded(buf, RATE_SHAKE, out)

    def _ctilde(self, mu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """c̃ = H(μ ‖ w1Encode(w)); the kernel decomposes, so it takes `w`, not `w1`."""
        p = self.p
        n = w.shape[0]
        packed = torch.zeros((n, p.k, p.polyw1_bytes), dtype=torch.uint8, device=self.device)
        w1_pack_kernel[(n, p.k, triton.cdiv(p.polyw1_bytes, BC))](
            w.reshape(n * p.k, 256),
            torch.zeros((n * p.k, 256), dtype=torch.int32, device=self.device),
            packed,
            n,
            p.k,
            256,
            p.w1_bits,
            p.two_gamma2,
            p.gamma2,
            p.m,
            p.polyw1_bytes,
            BC,
        )
        return self._shake32(torch.cat([mu, packed.reshape(n, -1)], dim=1), p.ctilde_bytes)

    def _sample_in_ball(self, c_tilde: torch.Tensor) -> torch.Tensor:
        p = self.p
        n = c_tilde.shape[0]
        buf = torch.zeros((n, RATE_SHAKE), dtype=torch.uint8, device=self.device)
        buf[:, : p.ctilde_bytes] = c_tilde
        buf[:, p.ctilde_bytes] = DOMAIN_SHAKE
        buf[:, RATE_SHAKE - 1] = buf[:, RATE_SHAKE - 1] | 0x80
        c_buf = sponge_padded(buf, RATE_SHAKE, RATE_SHAKE)
        c = torch.empty((n, 256), dtype=torch.int32, device=self.device)
        sample_in_ball_kernel[(n,)](c_buf, c, 256, p.tau, RATE_SHAKE, num_warps=SIB_WARPS)
        return c

    def _high_bits(self, r: torch.Tensor, p: DSAParams):
        return _decompose(r, p)

    def _pack_z(self, z: torch.Tensor) -> torch.Tensor:
        """BitPack of z for a batch of accepted rows: [n, l, polyz_bytes] uint8 on device."""
        p = self.p
        n = z.shape[0]
        out = torch.empty((n, p.l, p.polyz_bytes), dtype=torch.uint8, device=self.device)
        pack_z_kernel[(n * p.l, triton.cdiv(p.polyz_bytes, BC))](
            z.reshape(n * p.l, 256),
            out,
            n * p.l,
            p.polyz_bytes,
            p.z_bits,
            p.gamma1,
            256,
            BC,
        )
        return out


def _pointwise(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """ĉ ∘ ŝ mod q for [n, 256] × [m, 256] -> [n, m, 256], in one kernel.

    The host version broadcast an int64 product and then reduced, materialising a 64-bit array
    of the whole batch three times per attempt; this keeps the product in registers.
    """
    n, m = a.shape[0], b.shape[0]
    a_rep = a.repeat_interleave(m, dim=0).contiguous()
    b_rep = b.repeat(n, 1).contiguous()
    out = torch.empty((n * m, 256), dtype=torch.int32, device=a.device)
    pointwise_kernel[(triton.cdiv(n * m, 16),)](a_rep, b_rep, out, n * m, 256, Q_VALUE, 16)
    return out.view(n, m, 256)


def ntt_np_int32_cols(x: np.ndarray, signer: Signer) -> np.ndarray:
    """The reference NTT, as int32 rows (the device kernels take int32)."""
    from ..primitives.ntt import ntt_np

    arr = np.asarray(x, dtype=np.int64).reshape(-1, 256)
    return ntt_np(arr, DILITHIUM).astype(np.int32)


def sign_batch(
    ps: str, sk: bytes, messages, ctx: bytes = b"", prehash=None, rnd=None, device=None
) -> np.ndarray:
    return Signer(ps, sk, device=device).sign(messages, ctx, prehash, rnd)
