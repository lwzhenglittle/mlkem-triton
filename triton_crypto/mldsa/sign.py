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

from .._inputs import as_bytes_list
from ..devices import resolve_device
from ..primitives.ntt import DILITHIUM, intt_torch, ntt_torch
from ..primitives.sponge import DOMAIN_SHAKE, SHAKE256, padded_len, sponge_padded
from .kernels import (
    mac_kernel,
    pack_z_kernel,
    pointwise_kernel,
    sample_in_ball_kernel,
    sign_check_kernel,
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

    #: attempts evaluated per pipeline pass (speculation), and the row count the pass aims for
    spec = 8
    spec_target = 512

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
        if len(ctx) > 255:
            raise ValueError("context must be at most 255 bytes")
        rows = as_bytes_list(messages, "message")
        if not rows:
            return np.empty((0, 0), dtype=np.uint8)
        return self.sign_mprime([self.ref.format_message(m, ctx, prehash) for m in rows], rnd)

    def sign_mprime(self, mprimes, rnd=None) -> np.ndarray:
        """The internal interface: each message is already M′ (FIPS 204 Alg. 7's input)."""
        mprime = as_bytes_list(mprimes, "message")
        if not mprime:
            return np.empty((0, 0), dtype=np.uint8)
        return self.sign_mu_rows(self._mu(mprime), rnd)

    def sign_mu(self, mus, rnd=None) -> np.ndarray:
        """The external-μ interface: each row is already μ (64 bytes)."""
        if isinstance(mus, np.ndarray):
            arr = np.ascontiguousarray(mus, dtype=np.uint8)
        else:
            rows = as_bytes_list(mus, "mu")
            for i, m in enumerate(rows):
                if len(m) != 64:
                    raise ValueError(f"mu[{i}] must be 64 bytes, got {len(m)}")
            arr = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(-1, 64)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2 or arr.shape[1] != 64:
            raise ValueError(f"mu must be [B,64] uint8, got {arr.shape}")
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
        parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for _round in range(100000):
            open_n = int(remaining.numel())
            if open_n == 0:
                break
            # Speculation: the k-th attempt of a row only depends on κ + k·ℓ, so several attempts
            # can go through one pipeline pass and be resolved afterwards. The signature is
            # unchanged (a row takes the first accepting candidate); the tail of a batch — where
            # the pipeline is launch-bound, not compute-bound — needs far fewer rounds.
            spec = 1
            if self.spec > 1:
                spec = max(1, min(self.spec, self.spec_target // max(open_n, 1)))
            if spec > 1:
                mu_a = mu[remaining].repeat_interleave(spec, dim=0)
                rp_a = rho_prime[remaining].repeat_interleave(spec, dim=0)
                kap_a = kappa[remaining].repeat_interleave(spec) + (
                    torch.arange(spec, device=dev).repeat(open_n) * p.l
                )
            else:
                mu_a = mu[remaining]
                rp_a = rho_prime[remaining]
                kap_a = kappa[remaining]
            accept, z, h, c_tilde = self._attempt(mu_a, rp_a, kap_a)
            if spec > 1:
                acc2 = accept.view(open_n, spec)
                got = acc2.any(dim=1)
                first = acc2.to(torch.int64).argmax(dim=1)
                pick = first + torch.arange(open_n, device=dev) * spec
                pick = pick[got]
            else:
                got = accept
                pick = accept.nonzero(as_tuple=False).reshape(-1)
            if bool(got.any()):
                idx = remaining[got].cpu().numpy()
                z_bytes = self._pack_z(z[pick].contiguous()).cpu().numpy()
                h_packed = self._pack_h_batch(h[pick].cpu().numpy())
                ct_np = c_tilde[pick].cpu().numpy()
                parts.append((idx, ct_np, z_bytes, h_packed))
            remaining = remaining[~got]
            kappa[remaining] = kappa[remaining] + p.l * spec

        if sum(len(p[0]) for p in parts) != b:
            raise RuntimeError("signing did not converge")
        out = np.empty((b, self.p.sig_size), dtype=np.uint8)
        for idx, ct, zz, hh in parts:
            out[idx] = np.concatenate([ct, zz.reshape(len(idx), -1), hh], axis=1)
        return out

    # ---------------------------------------------------------------- one attempt
    def _attempt(self, mu_a, rp_a, kap_a):
        """One signing attempt for `n` rows: (accept, z, h, c̃), with no host synchronisation."""
        p = self.p
        dev = self.device
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
            True,
            num_warps=MAC_WARPS,
        )
        w = intt_torch(w_hat.reshape(n * p.k, 256), DILITHIUM).reshape(n, p.k, 256)
        c_tilde = self._ctilde(mu_a, w)
        c = self._sample_in_ball(c_tilde)
        c_hat = ntt_torch(c, DILITHIUM)
        # one pointwise product and one INTT for ĉ·ŝ1, ĉ·ŝ2 and ĉ·t̂0 together
        cs = intt_torch(
            _pointwise(c_hat, self.s1_hat, self.s2_hat, self.t0_hat).reshape(
                n * (p.l + 2 * p.k), 256
            ),
            DILITHIUM,
        ).reshape(n, p.l + 2 * p.k, 256)
        cs1 = cs[:, : p.l]
        cs2 = cs[:, p.l : p.l + p.k]
        ct0 = _center(cs[:, p.l + p.k :])
        z = _center(y + cs1)
        w_minus = (w - cs2) % Q_VALUE
        # the whole acceptance test — r0 bound, c·t0 bound, hints and their count — in one kernel
        h = torch.empty((n, p.k, 256), dtype=torch.int32, device=dev)
        accept = torch.empty(n, dtype=torch.int32, device=dev)
        sign_check_kernel[(n,)](
            w_minus.reshape(n * p.k, 256),
            ct0.reshape(n * p.k, 256),
            z.reshape(n * p.l, 256),
            h.reshape(n * p.k, 256),
            accept,
            p.k,
            p.l,
            256,
            Q_VALUE,
            p.two_gamma2,
            p.gamma2,
            p.beta,
            p.omega,
            p.norm_bound,
            num_warps=4,
        )
        return accept.to(torch.bool), z, h, c_tilde

    # ---------------------------------------------------------------- helpers
    def _mu(self, mprimes: list[bytes]) -> torch.Tensor:
        """μ = SHAKE256(H(pk) ‖ M′, 64) for a batch of M′.

        Rows are grouped by *padded* length. pad10*1 ends at a different offset inside the final
        block for different message lengths, and the sponge absorbs a launch-wide block count, so
        rows whose padding spans a different number of blocks cannot share one launch. Grouping by
        `padded_len` (rather than exact length) keeps the usual batch to one launch while staying
        byte-exact; the previous "pad everything to the longest block count" version absorbed extra
        all-zero blocks for short rows and produced a wrong μ.
        """
        b = len(mprimes)
        totals = [padded_len(64 + len(m), RATE_SHAKE) for m in mprimes]
        tr = torch.frombuffer(bytearray(self.tr), dtype=torch.uint8).to(self.device)
        out = torch.empty((b, 64), dtype=torch.uint8, device=self.device)
        for total in sorted(set(totals)):
            rows = [i for i, t in enumerate(totals) if t == total]
            buf = torch.zeros((len(rows), total), dtype=torch.uint8, device=self.device)
            buf[:, :64] = tr
            host = buf.cpu().numpy()
            for j, i in enumerate(rows):
                m = mprimes[i]
                host[j, 64 : 64 + len(m)] = np.frombuffer(m, dtype=np.uint8)
                host[j, 64 + len(m)] = DOMAIN_SHAKE
            buf = torch.from_numpy(host).to(self.device)
            buf[:, total - 1] = buf[:, total - 1] | 0x80
            out[torch.tensor(rows, device=self.device)] = sponge_padded(buf, RATE_SHAKE, 64)
        return out

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

    def _pack_h_batch(self, h: np.ndarray) -> np.ndarray:
        """HintBitPack (FIPS 204 Alg. 21) for a batch: [n, k, 256] -> [n, omega + k] uint8.

        Per signature this is a Python loop over k polynomials with `flatnonzero`; at a thousand
        signatures that is ~30 ms of pure host time, so it is done for the whole accepted batch at
        once instead.
        """
        p = self.p
        n = h.shape[0]
        out = np.zeros((n, p.omega + p.k), dtype=np.uint8)
        counts = h.sum(axis=2)  # [n, k]
        starts = np.zeros_like(counts)
        starts[:, 1:] = np.cumsum(counts, axis=1)[:, :-1]
        # rank of each hint inside its own polynomial, so it lands at `starts + rank`
        rank = np.cumsum(h, axis=2) - 1
        rows, polys, coeffs = np.nonzero(h)
        pos = starts[rows, polys] + rank[rows, polys, coeffs]
        out[rows, pos] = coeffs.astype(np.uint8)
        out[:, p.omega :] = np.cumsum(counts, axis=1).astype(np.uint8)
        return out

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


def _pointwise(a: torch.Tensor, *bs: torch.Tensor) -> torch.Tensor:
    """ĉ ∘ ŝᵢ mod q for [n, 256] × several [mᵢ, 256] -> [n, Σmᵢ, 256], in one kernel.

    The host version broadcast an int64 product and then reduced, materialising a 64-bit array
    of the whole batch three times per attempt; this keeps the product in registers.
    """
    n, m = a.shape[0], sum(b.shape[0] for b in bs)
    stacked = torch.cat(list(bs), dim=0).contiguous()  # [m, 256], broadcast inside the kernel
    out = torch.empty((n * m, 256), dtype=torch.int32, device=a.device)
    pointwise_kernel[(triton.cdiv(n * m, 16),)](a, stacked, out, n * m, m, 256, Q_VALUE, 16)
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
