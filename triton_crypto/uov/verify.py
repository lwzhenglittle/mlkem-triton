"""Batched GPU verifier for UOV (`OV-Is`, `OV-Ip`, `OV-III`, `OV-V`, classic and `-pkc`)."""

from __future__ import annotations

import hashlib

import numpy as np
import triton

from .. import devices as _devices
from .._inputs import as_bytes_list
from ..primitives import sponge as _sponge
from .kernels import _uov_eval_gf16_kernel, _uov_eval_gf256_kernel
from .params import BY_NAME, UOVParams, from_pk_len
from .ref import expand_pk_np

__all__ = ["Verifier", "verify_batch"]


class Verifier:
    """Single-key batched UOV verifier.

    Expands `-pkc` (if compressed) and reorders `(P1, P2, P3)` into row-major upper-triangular
    `[T, m_u32_words]` `int32` once on construction. Uniform-length message batches compute
    `SHAKE256(M || salt)` and the quadratic evaluation entirely on the device.
    """

    def __init__(self, pk_or_ps, pk: bytes | None = None, device=None):
        import torch

        if pk is None:
            pk_bytes = bytes(pk_or_ps)
            self.ps = from_pk_len(len(pk_bytes))
        else:
            self.ps = BY_NAME[pk_or_ps] if isinstance(pk_or_ps, str) else pk_or_ps
            pk_bytes = bytes(pk)
            if len(pk_bytes) not in (self.ps.pk_bytes, self.ps.cpk_bytes):
                raise ValueError(
                    f"{self.ps.name}: expected {self.ps.pk_bytes} or {self.ps.cpk_bytes} bytes, got {len(pk_bytes)}"
                )
        self.device = _devices.resolve_device(device)
        self.pk = pk_bytes
        p_tri = expand_pk_np(self.ps, pk_bytes)
        self._p_tri = torch.from_numpy(p_tri.view(np.int32).copy()).to(self.device).contiguous()

    def eval_core_torch(self, x_elems, block: int = 128):
        """Device-only UOV quadratic map: `x_elems [B, N]` uint8 -> `[B, m_bytes]` uint8."""
        import torch

        b = x_elems.shape[0]
        ps = self.ps
        if b == 0:
            return torch.empty((0, ps.m_bytes), dtype=torch.uint8, device=self.device)
        mw = ps.m_u32_words
        out = torch.empty((b, mw), dtype=torch.int32, device=self.device)
        kern = _uov_eval_gf16_kernel if ps.q == 16 else _uov_eval_gf256_kernel
        x_t = x_elems.transpose(0, 1).contiguous()
        kern[(triton.cdiv(b, block), mw)](
            self._p_tri,
            x_t,
            out,
            b,
            N=ps.n,
            M_WORDS=mw,
            BLOCK_B=block,
            num_warps=_devices.warps("uov", 4),
        )
        return out.view(torch.uint8)

    def _unpack_x_torch(self, sigs_u8):
        import torch

        ps = self.ps
        raw = sigs_u8[:, : ps.n_bytes]
        if ps.q == 256:
            return raw.contiguous()
        lo = raw & 0x0F
        hi = raw >> 4
        return torch.stack((lo, hi), dim=-1).reshape(sigs_u8.shape[0], ps.n).contiguous()

    def verify_torch(self, msgs_u8, sigs_u8, block: int = 128):
        """Fully on-device verification for uniform-length `[B, L]` msgs and `[B, sig_bytes]` sigs."""
        import torch

        b = msgs_u8.shape[0]
        if b == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        ps = self.ps
        if sigs_u8.shape != (b, ps.sig_bytes):
            raise ValueError(
                f"expected signatures ({b}, {ps.sig_bytes}), got {tuple(sigs_u8.shape)}"
            )

        salt = sigs_u8[:, ps.n_bytes :]
        ms = torch.cat((msgs_u8, salt), dim=1) if msgs_u8.shape[1] > 0 else salt.contiguous()
        ms_pad = _sponge.pad_torch(ms, _sponge.SHAKE256.rate, _sponge.SHAKE256.domain)
        t = _sponge.sponge_padded(ms_pad, _sponge.SHAKE256.rate, ps.m_bytes)

        x_elems = self._unpack_x_torch(sigs_u8)
        y = self.eval_core_torch(x_elems, block=block)
        return (y == t).all(dim=1)

    def verify(self, messages, signatures) -> np.ndarray:
        """Verify a batch of `(messages, signatures)` -> `[B]` bool numpy array."""
        import torch

        if isinstance(messages, torch.Tensor) and isinstance(signatures, torch.Tensor):
            return (
                self.verify_torch(
                    messages.to(device=self.device, dtype=torch.uint8).contiguous(),
                    signatures.to(device=self.device, dtype=torch.uint8).contiguous(),
                )
                .cpu()
                .numpy()
            )

        msg_list = as_bytes_list(messages)
        sig_list = as_bytes_list(signatures)
        if len(msg_list) != len(sig_list):
            raise ValueError(f"{len(msg_list)} messages vs {len(sig_list)} signatures")
        b = len(msg_list)
        if b == 0:
            return np.zeros((0,), dtype=bool)

        ps = self.ps
        bad_len = np.array([len(s) != ps.sig_bytes for s in sig_list], dtype=bool)
        if bad_len.all():
            return np.zeros((b,), dtype=bool)
        sig_arr = np.zeros((b, ps.sig_bytes), dtype=np.uint8)
        for i, s in enumerate(sig_list):
            if len(s) == ps.sig_bytes:
                sig_arr[i] = np.frombuffer(s, dtype=np.uint8)

        mlens = {len(m) for m in msg_list}
        if len(mlens) == 1:
            mlen = next(iter(mlens))
            msg_arr = (
                np.frombuffer(b"".join(msg_list), dtype=np.uint8).reshape(b, mlen).copy()
                if mlen > 0
                else np.empty((b, 0), dtype=np.uint8)
            )
            ok = (
                self.verify_torch(
                    torch.from_numpy(msg_arr).to(self.device),
                    torch.from_numpy(sig_arr).to(self.device),
                )
                .cpu()
                .numpy()
            )
        else:
            t_arr = np.zeros((b, ps.m_bytes), dtype=np.uint8)
            for i, (m, s) in enumerate(zip(msg_list, sig_list, strict=True)):
                if len(s) == ps.sig_bytes:
                    t_enc = hashlib.shake_256(m + s[ps.n_bytes :]).digest(ps.m_bytes)
                    t_arr[i] = np.frombuffer(t_enc, dtype=np.uint8)
            sig_t = torch.from_numpy(sig_arr).to(self.device)
            x_elems = self._unpack_x_torch(sig_t)
            y = self.eval_core_torch(x_elems)
            t_t = torch.from_numpy(t_arr).to(self.device)
            ok = (y == t_t).all(dim=1).cpu().numpy()

        if bad_len.any():
            ok = ok & (~bad_len)
        return ok


def verify_batch(
    pk: bytes, messages, signatures, ps: UOVParams | str | None = None, device=None
) -> np.ndarray:
    """Convenience one-shot batch verification for UOV."""
    v = Verifier(pk, device=device) if ps is None else Verifier(ps, pk, device=device)
    return v.verify(messages, signatures)
