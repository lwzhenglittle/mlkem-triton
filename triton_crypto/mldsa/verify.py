"""Batched ML-DSA signature verification on the GPU (FIPS 204 Alg. 3).

The shape this optimises for is the same one ML-KEM decapsulation optimises for:
**one long-lived public key, very many signatures**. Everything that depends only on
`pk` — Â = ExpandA(ρ) and t̂1 — is expanded once at construction and stays resident, so a
verification costs:

    unpack z (l polys) ──▶ NTT ×l ──┐
    SampleInBall(c̃) ────▶ NTT ×1 ──┼──▶ MAC (k·l pointwise) ──▶ INTT ×k ──▶ UseHint
    hints (dense mask) ─────────────┘                                          │
                                                                               ▼
                                    verdict ◀── compare c̃ ◀── SHAKE256(mu ‖ w1Encode)

There is **no Toeplitz GEMM** here, and that is not an omission. The ML-KEM trick needs
one operand to be small enough that fp16 partial sums stay below 2²⁴; ML-DSA's modulus is
q = 8380417 (23 bits), so products are 46-bit before reduction and would have to be
limb-split across 9 int8 GEMMs to stay exact — more arithmetic than the NTT itself. The
NTT is the right primitive for this scheme; the shared primitives layer is what makes it
reusable.

Measured throughput is in docs/benchmarks.md; the honest summary is that the SHAKE calls
(μ, c̃, and the final hash over w1Encode) are ~2/3 of the device time, as in ML-KEM.
"""

from __future__ import annotations

import warnings

import numpy as np
import torch
import triton

from ..devices import resolve_device
from ..primitives.ntt import DILITHIUM, intt_torch, ntt_torch
from ..primitives.sponge import DOMAIN_SHAKE, SHAKE256, padded_len, sponge_padded
from .kernels import (
    hint_kernel,
    mac_kernel,
    sample_in_ball_kernel,
    unpack_z_kernel,
    w1_pack_kernel,
)
from .params import PARAMS, D, DSAParams
from .params import Q as Q_VALUE
from .ref import MLDSA

RATE_SHAKE = SHAKE256.rate  # 136
MAC_BN = 8
MAC_WARPS = 4
SIB_WARPS = 4
BC = 64


def _to_device(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Copy a (possibly read-only) numpy array to the device without a spurious warning."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*not writable.*")
        return torch.from_numpy(arr).to(device)


def _as_device_uint8(x, device: torch.device) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.uint8:
            raise TypeError(f"expected uint8, got {x.dtype}")
        return x if x.device == device else x.to(device)
    if isinstance(x, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(bytes(x), np.uint8)
    else:
        arr = np.ascontiguousarray(x, dtype=np.uint8)
    return _to_device(arr, device)


class Verifier:
    """Verify any number of ML-DSA signatures under one public key.

    Args:
        ps: parameter set, e.g. ``"ML-DSA-65"``.
        pk: public key bytes (``32 + k*320`` bytes).
        device: ``None`` selects the first available accelerator.
    """

    def __init__(self, ps: str, pk: bytes, device=None):
        if ps not in PARAMS:
            raise KeyError(f"unknown parameter set {ps}; known: {sorted(PARAMS)}")
        self.ps = ps
        self.p: DSAParams = PARAMS[ps]
        self.device = resolve_device(device)
        self.pk = bytes(pk)
        self.ref = MLDSA(ps)
        #: set to True to fill `stage_times` (device-synchronising; benchmarking only)
        self.profile = False
        self.stage_times: dict[str, float] = {}

        rho, t1 = self.ref.unpack_pk(self.pk)  # raises on a wrong length
        # One-time public-key preprocessing, on the host: Â has k·l SHAKE128 calls with
        # rejection sampling (a few ms), t̂1 is one transform per polynomial. Both are
        # then resident for the lifetime of the object.
        self.A_hat = torch.from_numpy(self.ref.expand_a(rho).reshape(-1, 256).astype(np.int32)).to(
            self.device
        )
        t1_shift = (t1 * (1 << D)) % Q_VALUE  # t1·2^d before the transform (Alg. 3 step 8)
        self.t1_hat = torch.from_numpy(ntt_np_int32(t1_shift)).to(self.device)
        self.hpk = SHAKE256.digest_bytes(self.pk, 64)  # H(pk), 64 B, per key

    # ---------------------------------------------------------------- host side

    def _mu_torch(self, mprime: torch.Tensor) -> torch.Tensor:
        """μ = SHAKE256(H(pk) ‖ M′, 64) for a batch of equal-length M′."""
        b, lm = mprime.shape
        if b == 0:
            return torch.empty((0, 64), dtype=torch.uint8, device=mprime.device)
        total = padded_len(64 + lm, RATE_SHAKE)
        buf = torch.zeros((b, total), dtype=torch.uint8, device=mprime.device)
        buf[:, :64] = torch.frombuffer(bytearray(self.hpk), dtype=torch.uint8).to(mprime.device)
        buf[:, 64 : 64 + lm] = mprime
        buf[:, 64 + lm] = DOMAIN_SHAKE
        buf[:, total - 1] = buf[:, total - 1] | 0x80
        return sponge_padded(buf, RATE_SHAKE, 64)

    def _format_messages(self, msgs, ctx: bytes, prehash: str | None):
        """M′ = 0x00 ‖ len(ctx) ‖ ctx ‖ M (or the pre-hash form), as [B, L] uint8.

        The per-row `format_message` loop is the single largest host-side cost in the
        end-to-end path (2 bytes more than a `bytes` object per row), so the uniform
        case — which is the normal one — is assembled with numpy instead.
        """
        if prehash is not None:
            rows = [self.ref.format_message(m, ctx, prehash) for m in _split_rows(msgs, self.p)]
            return _stack_or_list(rows)
        if isinstance(msgs, np.ndarray):
            arr = np.ascontiguousarray(msgs, dtype=np.uint8)
            if arr.ndim == 1:
                arr = arr[None, :]
        elif isinstance(msgs, (bytes, bytearray, memoryview)):
            arr = np.frombuffer(bytes(msgs), np.uint8)[None, :]
        else:
            lst = [bytes(m) for m in msgs]
            if not lst:
                return np.empty((0, 0), dtype=np.uint8)
            n0 = len(lst[0])
            if any(len(m) != n0 for m in lst):
                # varying lengths: fall back to one row at a time (grouped downstream)
                return _stack_or_list([self.ref.format_message(m, ctx, None) for m in lst])
            arr = np.frombuffer(b"".join(lst), np.uint8).reshape(len(lst), n0)
        prefix = np.frombuffer(bytes([0, len(ctx)]) + ctx, np.uint8)
        out = np.empty((arr.shape[0], prefix.size + arr.shape[1]), dtype=np.uint8)
        out[:, : prefix.size] = prefix
        out[:, prefix.size :] = arr
        return out

    def _sigs_device(self, sigs) -> torch.Tensor:
        """Normalise '[B, sig_size] array / list[bytes] / bytes / torch' to [B, sig_size]."""
        n0 = self.p.sig_size
        if isinstance(sigs, torch.Tensor):
            t = sigs.reshape(-1, n0)
            if t.dtype != torch.uint8:
                raise TypeError(f"signatures must be uint8, got {t.dtype}")
            return t if t.device == self.device else t.to(self.device)
        if isinstance(sigs, np.ndarray):
            arr = np.ascontiguousarray(sigs, dtype=np.uint8).reshape(-1, n0)
        elif isinstance(sigs, (bytes, bytearray, memoryview)):
            buf = bytes(sigs)
            if len(buf) % n0:
                raise ValueError(f"input length {len(buf)} is not a multiple of {n0}")
            arr = np.frombuffer(buf, np.uint8).reshape(-1, n0)
        else:
            lst = list(sigs)
            arr = np.frombuffer(b"".join(lst), np.uint8).reshape(len(lst), n0)
        return _to_device(arr, self.device)

    # ---------------------------------------------------------------- device side

    def _mark(self, name: str, t0: float) -> float:
        if not self.profile:
            return t0
        from ..devices import sync

        sync(self.device)
        import time as _time

        now = _time.perf_counter()
        self.stage_times[name] = self.stage_times.get(name, 0.0) + (now - t0)
        return now

    def _verify_device(self, mu: torch.Tensor, sig: torch.Tensor) -> torch.Tensor:
        """mu [B,64] uint8, sig [B,SIG] uint8 -> bool [B] on device."""
        import time

        p, k, l = self.p, self.p.k, self.p.l
        b = sig.shape[0]
        dev = self.device
        if b == 0:  # nothing to launch; keeping it explicit avoids zero-grid kernels
            return torch.zeros(0, dtype=torch.bool, device=dev)
        _t = time.perf_counter()

        # 1. z: BitUnpack_{γ1−1}, [B*l, 256] int32 (signed, γ1 − field)
        z = torch.empty((b * l, 256), dtype=torch.int32, device=dev)
        unpack_z_kernel[(b, l)](
            sig,
            z,
            p.sig_size,
            b,
            p.z_offset,
            p.polyz_bytes,
            p.z_bits,
            p.gamma1,
            256,
            l,
            num_warps=4,
        )
        z3 = z.view(b, l, 256)
        norm_ok = (z3.abs() < p.norm_bound).all(dim=(1, 2))
        _t = self._mark("unpack_z", _t)

        # 2. ẑ and ĉ
        z_hat = ntt_torch(z, DILITHIUM)
        c_buf = sponge_padded(_pad_ctilde(sig[:, : p.ctilde_bytes]), RATE_SHAKE, RATE_SHAKE)
        c = torch.empty((b, 256), dtype=torch.int32, device=dev)
        sample_in_ball_kernel[(b,)](c_buf, c, 256, p.tau, RATE_SHAKE, num_warps=SIB_WARPS)
        c_hat = ntt_torch(c, DILITHIUM)
        _t = self._mark("sib+ntt", _t)

        # 3. ŵ = Â·ẑ − ĉ·t̂1, then INTT
        w_hat = torch.empty((b, k, 256), dtype=torch.int32, device=dev)
        mac_kernel[(triton.cdiv(b, MAC_BN), k)](
            z_hat,
            self.A_hat,
            c_hat,
            self.t1_hat,
            w_hat,
            b,
            k,
            l,
            256,
            Q_VALUE,
            MAC_BN,
            num_warps=MAC_WARPS,
        )
        w = intt_torch(w_hat.view(b * k, 256), DILITHIUM).view(b, k, 256)
        _t = self._mark("mac+intt", _t)

        # 4. hints (dense) + validity
        h = torch.zeros((b, k, 256), dtype=torch.uint8, device=dev)
        h_ok = torch.empty((b, k), dtype=torch.int32, device=dev)
        hint_kernel[(b, k)](
            sig,
            h,
            h_ok,
            p.sig_size,
            p.h_offset,
            p.omega,
            1 << (p.omega - 1).bit_length(),
            k,
            256,
            num_warps=4,
        )
        hint_ok = h_ok.all(dim=1)
        _t = self._mark("hints", _t)

        # 5. UseHint + BitPack_ω1
        w1 = torch.empty((b, k * p.polyw1_bytes), dtype=torch.uint8, device=dev)
        w1_pack_kernel[(b, k, triton.cdiv(p.polyw1_bytes, BC))](
            w,
            h,
            w1,
            b,
            k,
            256,
            p.w1_bits,
            p.two_gamma2,
            p.gamma2,
            p.m,
            p.polyw1_bytes,
            BC,
            num_warps=4,
        )
        _t = self._mark("w1pack", _t)

        # 6. c̃′ = SHAKE256(μ ‖ w1Encode(w′1)), compared with the c̃ in the signature
        lm = k * p.polyw1_bytes
        total = padded_len(64 + lm, RATE_SHAKE)
        buf = torch.zeros((b, total), dtype=torch.uint8, device=dev)
        buf[:, :64] = mu
        buf[:, 64 : 64 + lm] = w1
        buf[:, 64 + lm] = DOMAIN_SHAKE
        buf[:, total - 1] = buf[:, total - 1] | 0x80
        c2 = sponge_padded(buf, RATE_SHAKE, p.ctilde_bytes)
        digest_ok = (c2 == sig[:, : p.ctilde_bytes]).all(dim=1)
        self._mark("final_hash", _t)
        return norm_ok & hint_ok & digest_ok

    # ---------------------------------------------------------------- public API

    def verify_mu(self, mu, sigs) -> np.ndarray:
        """Verify with a precomputed μ (the external-mu interface of FIPS 204).

        Args:
            mu: ``[B, 64]`` uint8 (np.ndarray or torch.Tensor).
            sigs: ``[B, sig_size]`` uint8, a list of ``bytes``, or one bytes blob.

        Returns:
            bool ndarray ``[B]``.
        """
        sig = self._sigs_device(sigs)
        mu_t = _as_device_uint8(mu, self.device).reshape(-1, 64)
        if mu_t.shape[0] != sig.shape[0]:
            raise ValueError(f"mu has {mu_t.shape[0]} rows but there are {sig.shape[0]} signatures")
        return self._verify_device(mu_t, sig).cpu().numpy()

    def verify_internal(self, mprime, sigs) -> np.ndarray:
        """Verify a batch of already-formatted messages M′ (Alg. 3, internal interface)."""
        if isinstance(mprime, np.ndarray):
            arr = np.ascontiguousarray(mprime, dtype=np.uint8)
            arr = arr[None, :] if arr.ndim == 1 else arr
            return self._verify_mprime(arr, sigs)
        return self._verify_mprime(_split_rows(mprime, self.p), sigs)

    def verify(self, msgs, sigs, ctx: bytes = b"", prehash: str | None = None) -> np.ndarray:
        """Verify messages under ML-DSA (or HashML-DSA when `prehash` is given).

        `msgs` may be a ``[B, L]`` uint8 array, a list of ``bytes`` of possibly different
        lengths, or one concatenated ``bytes`` blob. Different lengths are handled by
        grouping: one SHAKE launch per distinct length (9 blocks best case).
        """
        return self._verify_mprime(self._format_messages(msgs, ctx, prehash), sigs)

    # ---- internals ----

    def _verify_mprime(self, rows, sigs) -> np.ndarray:
        """`rows`: [B, L] uint8 (uniform) or list[bytes] (grouped by length)."""
        sig = self._sigs_device(sigs)
        if isinstance(rows, np.ndarray):
            if rows.shape[0] != sig.shape[0]:
                raise ValueError(f"{rows.shape[0]} messages but {sig.shape[0]} signatures")
            mu = self._mu_torch(_to_device(rows, self.device))
            return self._verify_device(mu, sig).cpu().numpy()
        if len(rows) != sig.shape[0]:
            raise ValueError(f"{len(rows)} messages but {sig.shape[0]} signatures")
        out = np.zeros(len(rows), dtype=bool)
        by_len: dict[int, list[int]] = {}
        for i, m in enumerate(rows):
            by_len.setdefault(len(m), []).append(i)
        for lm, who in by_len.items():
            mp = _to_device(
                np.frombuffer(b"".join(rows[i] for i in who), np.uint8).reshape(-1, lm),
                self.device,
            )
            mu = self._mu_torch(mp)
            out[who] = self._verify_device(mu, sig[who]).cpu().numpy()
        return out


def _pad_ctilde(c: torch.Tensor) -> torch.Tensor:
    """c̃ [B,λ/4] -> [B,136] pad10*1 with the SHAKE domain byte (one block is enough)."""
    b, n = c.shape
    out = torch.zeros((b, RATE_SHAKE), dtype=torch.uint8, device=c.device)
    out[:, :n] = c
    out[:, n] = DOMAIN_SHAKE
    out[:, RATE_SHAKE - 1] = out[:, RATE_SHAKE - 1] | 0x80
    return out


def _stack_or_list(rows: list[bytes]):
    """[B, L] uint8 if all rows have the same length, else the plain list."""
    n0 = {len(r) for r in rows}
    if len(n0) == 1:
        return np.frombuffer(b"".join(rows), np.uint8).reshape(len(rows), n0.pop())
    return rows


def _split_rows(x, p: DSAParams) -> list[bytes]:
    """Normalise '[B, L] array / list[bytes] / bytes' into a list of byte strings."""
    if isinstance(x, np.ndarray):
        arr = np.ascontiguousarray(x, dtype=np.uint8)
        if arr.ndim == 1:
            return [arr.tobytes()]
        return [row.tobytes() for row in arr]
    if isinstance(x, (bytes, bytearray, memoryview)):
        return [bytes(x)]
    return [bytes(m) for m in x]


def ntt_np_int32(x: np.ndarray) -> np.ndarray:
    """Host-side transform helper (uses the numpy reference, int64 → int32)."""
    from ..primitives.ntt import ntt_np

    return ntt_np(x.astype(np.int64), DILITHIUM).astype(np.int32)


def verify_batch(ps: str, pk: bytes, msgs, sigs, ctx: bytes = b"", device=None) -> np.ndarray:
    """One-shot batch verification (rebuilds the public-key preprocessing every call)."""
    return Verifier(ps, pk, device=device).verify(msgs, sigs, ctx=ctx)


__all__ = ["Verifier", "verify_batch"]
