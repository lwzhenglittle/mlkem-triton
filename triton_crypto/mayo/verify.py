"""Batched GPU verifier for MAYO-1 / MAYO-2 / MAYO-3 / MAYO-5."""

from __future__ import annotations

import hashlib

import numpy as np
import triton

from .. import devices as _devices
from .._inputs import as_bytes_list
from ..primitives import sponge as _sponge
from .kernels import _mayo_fold_check_kernel, _mayo_sps_kernel
from .params import BY_NAME, MayoParams, from_pk_len
from .ref import expand_pk_np

__all__ = ["Verifier", "verify_batch"]


class Verifier:
    """Single-key batched MAYO verifier.

    Expands the compact public key `cpk` once on construction (`AES-128-CTR` of `pk_seed` + `P3`)
    and uploads the packed `[n, n, m_u32_words]` table to the device. Uniform-length message
    batches execute the entire pipeline — `SHAKE256(M)`, `SHAKE256(md || salt)`, nibble unpacking,
    `SPS = S * P * S^T`, polynomial folding mod `f(z)`, and digest check — on the device.
    """

    def __init__(self, cpk_or_ps, cpk: bytes | None = None, device=None):
        import torch

        if cpk is None:
            cpk_bytes = bytes(cpk_or_ps)
            self.ps = from_pk_len(len(cpk_bytes))
        else:
            self.ps = BY_NAME[cpk_or_ps] if isinstance(cpk_or_ps, str) else cpk_or_ps
            cpk_bytes = bytes(cpk)
            if len(cpk_bytes) != self.ps.cpk_bytes:
                raise ValueError(
                    f"{self.ps.name}: expected {self.ps.cpk_bytes}-byte cpk, got {len(cpk_bytes)}"
                )
        self.device = _devices.resolve_device(device)
        self.cpk = cpk_bytes
        p_u32 = expand_pk_np(self.ps, cpk_bytes)
        self._p_mat = torch.from_numpy(p_u32.view(np.int32).copy()).to(self.device).contiguous()

    def verify_core_torch(self, s_nibbles, t_words, block: int = 128):
        """Device-only MAYO core: `s_nibbles [B, K*N]` uint8, `t_words [B, M_WORDS]` int32 -> `[B]` bool."""
        import torch

        b = s_nibbles.shape[0]
        if b == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        ps = self.ps
        mw = ps.m_u32_words
        sps_buf = torch.empty((b, ps.k, ps.k, mw), dtype=torch.int32, device=self.device)
        out_ok = torch.empty((b,), dtype=torch.uint8, device=self.device)
        nw = _devices.warps("mayo", 4)
        s_nib_t = s_nibbles.reshape(b, ps.k, ps.n).permute(1, 2, 0).contiguous()
        _mayo_sps_kernel[(triton.cdiv(b, block), ps.k, mw)](
            self._p_mat,
            s_nib_t,
            sps_buf,
            b,
            N=ps.n,
            K=ps.k,
            M_WORDS=mw,
            BLOCK_B=block,
            num_warps=nw,
        )
        _mayo_fold_check_kernel[(triton.cdiv(b, block),)](
            sps_buf,
            t_words.contiguous(),
            out_ok,
            b,
            K=ps.k,
            M=ps.m,
            M_WORDS=mw,
            F_TAIL_PACKED=ps.f_tail_packed,
            BLOCK=block,
            num_warps=nw,
        )
        return out_ok.to(torch.bool)

    def _unpack_sig_nibbles_torch(self, sigs_u8):
        """Unpack `[B, sig_bytes]` uint8 into `[B, K*N]` uint8 nibbles on device."""
        import torch

        ps = self.ps
        kn = ps.k * ps.n
        nbytes = (kn + 1) // 2
        raw = sigs_u8[:, :nbytes]
        lo = raw & 0x0F
        hi = raw >> 4
        nib = torch.stack((lo, hi), dim=-1).reshape(sigs_u8.shape[0], nbytes * 2)
        return nib[:, :kn].contiguous()

    def _pack_t_words_torch(self, t_u8):
        import torch

        b = t_u8.shape[0]
        mw = self.ps.m_u32_words
        padded = torch.zeros((b, mw * 4), dtype=torch.uint8, device=t_u8.device)
        padded[:, : self.ps.m_bytes] = t_u8
        return padded.view(torch.int32)

    def verify_torch(self, msgs_u8, sigs_u8, block: int = 128):
        """Fully on-device verification for uniform-length `[B, L]` uint8 msgs and `[B, sig_bytes]` sigs."""
        import torch

        b = msgs_u8.shape[0]
        if b == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        ps = self.ps
        if sigs_u8.shape != (b, ps.sig_bytes):
            raise ValueError(
                f"expected signatures ({b}, {ps.sig_bytes}), got {tuple(sigs_u8.shape)}"
            )

        # 1. md = SHAKE256(msg, digest_bytes) on GPU
        m_pad = _sponge.pad_torch(msgs_u8, _sponge.SHAKE256.rate, _sponge.SHAKE256.domain)
        md = _sponge.sponge_padded(m_pad, _sponge.SHAKE256.rate, ps.digest_bytes)

        # 2. t_enc = SHAKE256(md || salt, m_bytes) on GPU
        salt = sigs_u8[:, ps.sig_bytes - ps.salt_bytes :]
        ms = torch.cat((md, salt), dim=1)
        ms_pad = _sponge.pad_torch(ms, _sponge.SHAKE256.rate, _sponge.SHAKE256.domain)
        t_enc = _sponge.sponge_padded(ms_pad, _sponge.SHAKE256.rate, ps.m_bytes)

        s_nib = self._unpack_sig_nibbles_torch(sigs_u8)
        t_words = self._pack_t_words_torch(t_enc)
        return self.verify_core_torch(s_nib, t_words, block=block)

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
                    md = hashlib.shake_256(m).digest(ps.digest_bytes)
                    t_enc = hashlib.shake_256(md + s[ps.sig_bytes - ps.salt_bytes :]).digest(
                        ps.m_bytes
                    )
                    t_arr[i] = np.frombuffer(t_enc, dtype=np.uint8)
            sig_t = torch.from_numpy(sig_arr).to(self.device)
            s_nib = self._unpack_sig_nibbles_torch(sig_t)
            t_words = self._pack_t_words_torch(torch.from_numpy(t_arr).to(self.device))
            ok = self.verify_core_torch(s_nib, t_words).cpu().numpy()

        if bad_len.any():
            ok = ok & (~bad_len)
        return ok


def verify_batch(
    cpk: bytes, messages, signatures, ps: MayoParams | str | None = None, device=None
) -> np.ndarray:
    """Convenience one-shot batch verification for MAYO."""
    v = Verifier(cpk, device=device) if ps is None else Verifier(ps, cpk, device=device)
    return v.verify(messages, signatures)
