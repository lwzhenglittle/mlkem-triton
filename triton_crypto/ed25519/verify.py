"""Device wrappers for Ed25519 batched verification and X25519 scalar multiplication."""

from __future__ import annotations

import hashlib

import numpy as np
import triton

from .. import devices as _devices
from .._inputs import as_batch_bytes, as_bytes_list
from ..primitives import sha2 as _sha2
from .kernels import (
    _ed25519_finalize_kernel,
    _ed25519_msm_kernel,
    _ed25519_reduce_sc_kernel,
    _x25519_kernel,
)
from .ref import (
    CURVE_D,
    CURVE_L,
    CURVE_P,
    ED25519_B,
    ED25519_O,
    fe_to_limbs10,
    point_add,
    point_decode,
    point_neg,
    point_to_cached,
)

_D2_CACHE: dict = {}


def _d2_tensor(device):
    t = _D2_CACHE.get(device)
    if t is None:
        import torch

        t = torch.tensor(fe_to_limbs10((2 * CURVE_D) % CURVE_P), dtype=torch.int64, device=device)
        _D2_CACHE[device] = t
    return t


__all__ = [
    "Verifier",
    "verify_batch",
    "x25519_base_torch",
    "x25519_torch",
]


def _build_shamir_table(pk: bytes) -> tuple[np.ndarray, bool]:
    """Precompute `[4, 4, 10]` int64 Shamir table for `{O, B, -A, B - A}`."""
    a_pt = point_decode(pk)
    if a_pt is None:
        return np.zeros((4, 4, 10), dtype=np.int64), False
    neg_a = point_neg(a_pt)
    b_minus_a = point_add(ED25519_B, neg_a)
    pts = (ED25519_O, ED25519_B, neg_a, b_minus_a)
    tbl = np.array([point_to_cached(p) for p in pts], dtype=np.int64)
    return tbl, True


def _sha512_pad_gpu(r_bytes, pk_bytes, msgs_u8):
    """Pad `R || pk || msg` on GPU into SHA-512 128-byte blocks `[B, NB * 16]` int64."""
    import torch

    b, mlen = msgs_u8.shape
    total_len = 64 + mlen
    # SHA-512 padding: 1 byte 0x80 + zeros + 16-byte big-endian bit length
    nblocks = (total_len + 17 + 127) // 128
    plen = nblocks * 128
    buf = torch.zeros((b, plen), dtype=torch.uint8, device=msgs_u8.device)
    buf[:, :32] = r_bytes
    buf[:, 32:64] = pk_bytes
    if mlen > 0:
        buf[:, 64 : 64 + mlen] = msgs_u8
    buf[:, total_len] = 0x80
    bitlen = total_len * 8
    len_bytes = torch.tensor(
        list(bitlen.to_bytes(16, "big")), dtype=torch.uint8, device=msgs_u8.device
    )
    buf[:, plen - 16 : plen] = len_bytes
    return buf.view(torch.int64), nblocks


def _reduce_sha512_to_k_words(hash_words, b: int):
    import torch

    k_words = torch.empty((b, 8), dtype=torch.int32, device=hash_words.device)
    block = 128
    _ed25519_reduce_sc_kernel[(triton.cdiv(b, block),)](
        hash_words.view(torch.int32),
        k_words,
        b,
        BLOCK=block,
        num_warps=4,
    )
    return k_words


class Verifier:
    """Single-key batched Ed25519 verifier (RFC 8032).

    Precomputes `{O, B, -A, B - A}` once on construction and keeps the 1.28 KiB table resident
    on the device. Uniform-length message batches compute `SHA-512(R || A || M)` and modular
    reduction mod L entirely on the device.
    """

    def __init__(self, pk: bytes, device=None):
        import torch

        pk_bytes = bytes(pk)
        if len(pk_bytes) != 32:
            raise ValueError(f"Ed25519 public key must be 32 bytes, got {len(pk_bytes)}")
        self.device = _devices.resolve_device(device)
        self.pk = pk_bytes
        tbl_np, self._pk_valid = _build_shamir_table(pk_bytes)
        self._table = torch.from_numpy(tbl_np).to(self.device).contiguous()
        self._pk_ok = torch.tensor(
            [1 if self._pk_valid else 0], dtype=torch.uint8, device=self.device
        )
        self._pk_u8 = torch.tensor(list(pk_bytes), dtype=torch.uint8, device=self.device).unsqueeze(
            0
        )

    def verify_core_torch(self, r_words, s_words, k_words, block: int = 64):
        """Device-only Ed25519 core: `r_words`, `s_words`, `k_words` are `[B, 8]` int32 LE -> `[B]` bool."""
        import torch

        b = r_words.shape[0]
        if b == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        q_buf = torch.empty((b, 30), dtype=torch.int64, device=self.device)
        out_ok = torch.empty((b,), dtype=torch.uint8, device=self.device)
        grid = (triton.cdiv(b, block),)
        nw = _devices.warps("ed25519", 4)
        _ed25519_msm_kernel[grid](
            self._table,
            _d2_tensor(self.device),
            s_words.contiguous(),
            k_words.contiguous(),
            q_buf,
            b,
            0,
            BLOCK=block,
            num_warps=nw,
        )
        _ed25519_finalize_kernel[grid](
            q_buf,
            r_words.contiguous(),
            s_words.contiguous(),
            self._pk_ok,
            out_ok,
            b,
            0,
            BLOCK=block,
            num_warps=nw,
        )
        return out_ok.to(torch.bool)

    def verify_torch(self, msgs_u8, sigs_u8, block: int = 64):
        """Fully on-device verification for uniform-length `[B, L]` uint8 messages and `[B, 64]` uint8 sigs."""
        import torch

        b = msgs_u8.shape[0]
        if b == 0:
            return torch.empty((0,), dtype=torch.bool, device=self.device)
        if sigs_u8.shape != (b, 64):
            raise ValueError(f"expected signatures shape ({b}, 64), got {tuple(sigs_u8.shape)}")
        r_u8 = sigs_u8[:, :32].contiguous()
        s_u8 = sigs_u8[:, 32:64].contiguous()
        pk_exp = self._pk_u8.expand(b, 32)
        padded, nblocks = _sha512_pad_gpu(r_u8, pk_exp, msgs_u8)
        digest = torch.empty((b, 8), dtype=torch.int64, device=self.device)
        _sha2.sha_absorb(_sha2.iv_tensor(512, self.device), padded, digest, b, 512, nblocks)
        k_words = _reduce_sha512_to_k_words(digest, b)
        return self.verify_core_torch(
            r_u8.view(torch.int32), s_u8.view(torch.int32), k_words, block=block
        )

    def verify(self, messages, signatures) -> np.ndarray:
        """Verify a batch of `(messages, signatures)` against this public key -> `[B]` bool numpy array."""
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

        sig_list = as_bytes_list(signatures)
        msg_list = as_bytes_list(messages)
        if len(sig_list) != len(msg_list):
            raise ValueError(f"{len(msg_list)} messages vs {len(sig_list)} signatures")
        b = len(msg_list)
        if b == 0:
            return np.zeros((0,), dtype=bool)
        if not self._pk_valid:
            return np.zeros((b,), dtype=bool)

        bad_len = np.array([len(s) != 64 for s in sig_list], dtype=bool)
        if bad_len.all():
            return np.zeros((b,), dtype=bool)
        sig_arr = np.zeros((b, 64), dtype=np.uint8)
        for i, s in enumerate(sig_list):
            if len(s) == 64:
                sig_arr[i] = np.frombuffer(s, dtype=np.uint8)

        # Uniform message length -> run SHA-512 + sc_reduce on GPU
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
            # Variable-length messages: host SHA-512 + mod L, GPU elliptic-curve check
            k_arr = np.empty((b, 32), dtype=np.uint8)
            for i, (m, s) in enumerate(zip(msg_list, sig_list, strict=True)):
                if len(s) == 64:
                    k_int = (
                        int.from_bytes(hashlib.sha512(s[:32] + self.pk + m).digest(), "little")
                        % CURVE_L
                    )
                    k_arr[i] = np.frombuffer(k_int.to_bytes(32, "little"), dtype=np.uint8)
                else:
                    k_arr[i] = 0
            r_t = torch.from_numpy(sig_arr[:, :32].copy()).to(self.device).view(torch.int32)
            s_t = torch.from_numpy(sig_arr[:, 32:64].copy()).to(self.device).view(torch.int32)
            k_t = torch.from_numpy(k_arr).to(self.device).view(torch.int32)
            ok = self.verify_core_torch(r_t, s_t, k_t).cpu().numpy()

        if bad_len.any():
            ok = ok & (~bad_len)
        return ok


def verify_batch(pks, messages, signatures, device=None, block: int = 64) -> np.ndarray:
    """Verify Ed25519 signatures with either one shared public key or one public key per signature."""
    import torch

    if isinstance(pks, (bytes, bytearray, memoryview)):
        return Verifier(bytes(pks), device=device).verify(messages, signatures)

    pk_list = as_bytes_list(pks)
    msg_list = as_bytes_list(messages)
    sig_list = as_bytes_list(signatures)
    if len(pk_list) == 1 and len(msg_list) >= 1:
        return Verifier(pk_list[0], device=device).verify(msg_list, sig_list)
    if not (len(pk_list) == len(msg_list) == len(sig_list)):
        raise ValueError(
            f"batch length mismatch: {len(pk_list)} pks, {len(msg_list)} msgs, {len(sig_list)} sigs"
        )
    b = len(msg_list)
    if b == 0:
        return np.zeros((0,), dtype=bool)

    dev = _devices.resolve_device(device)
    tables = np.zeros((b, 4, 4, 10), dtype=np.int64)
    pk_ok = np.zeros((b,), dtype=np.uint8)
    sig_arr = np.zeros((b, 64), dtype=np.uint8)
    pk_arr = np.zeros((b, 32), dtype=np.uint8)
    bad = np.zeros((b,), dtype=bool)
    cache: dict[bytes, tuple[np.ndarray, bool]] = {}

    for i, (pk, s) in enumerate(zip(pk_list, sig_list, strict=True)):
        if len(pk) != 32 or len(s) != 64:
            bad[i] = True
            continue
        cached = cache.get(pk)
        if cached is None:
            cached = _build_shamir_table(pk)
            cache[pk] = cached
        tbl, valid = cached
        tables[i] = tbl
        pk_ok[i] = 1 if valid else 0
        pk_arr[i] = np.frombuffer(pk, dtype=np.uint8)
        sig_arr[i] = np.frombuffer(s, dtype=np.uint8)

    mlens = {len(m) for m in msg_list}
    if len(mlens) == 1:
        mlen = next(iter(mlens))
        msg_arr = (
            np.frombuffer(b"".join(msg_list), dtype=np.uint8).reshape(b, mlen).copy()
            if mlen > 0
            else np.empty((b, 0), dtype=np.uint8)
        )
        r_u8 = torch.from_numpy(sig_arr[:, :32].copy()).to(dev)
        s_u8 = torch.from_numpy(sig_arr[:, 32:64].copy()).to(dev)
        pk_u8 = torch.from_numpy(pk_arr).to(dev)
        msg_u8 = torch.from_numpy(msg_arr).to(dev)
        padded, nblocks = _sha512_pad_gpu(r_u8, pk_u8, msg_u8)
        digest = torch.empty((b, 8), dtype=torch.int64, device=dev)
        _sha2.sha_absorb(_sha2.iv_tensor(512, dev), padded, digest, b, 512, nblocks)
        k_words = _reduce_sha512_to_k_words(digest, b)
        r_words = r_u8.view(torch.int32)
        s_words = s_u8.view(torch.int32)
    else:
        k_arr = np.zeros((b, 32), dtype=np.uint8)
        for i, (pk, m, s) in enumerate(zip(pk_list, msg_list, sig_list, strict=True)):
            if not bad[i]:
                k_int = int.from_bytes(hashlib.sha512(s[:32] + pk + m).digest(), "little") % CURVE_L
                k_arr[i] = np.frombuffer(k_int.to_bytes(32, "little"), dtype=np.uint8)
        r_words = torch.from_numpy(sig_arr[:, :32].copy()).to(dev).view(torch.int32)
        s_words = torch.from_numpy(sig_arr[:, 32:64].copy()).to(dev).view(torch.int32)
        k_words = torch.from_numpy(k_arr).to(dev).view(torch.int32)

    tbl_t = torch.from_numpy(tables).to(dev).contiguous()
    pk_ok_t = torch.from_numpy(pk_ok).to(dev)
    q_buf = torch.empty((b, 30), dtype=torch.int64, device=dev)
    out_ok = torch.empty((b,), dtype=torch.uint8, device=dev)
    grid = (triton.cdiv(b, block),)
    nw = _devices.warps("ed25519", 4)
    _ed25519_msm_kernel[grid](
        tbl_t,
        _d2_tensor(dev),
        s_words,
        k_words,
        q_buf,
        b,
        160,
        BLOCK=block,
        num_warps=nw,
    )
    _ed25519_finalize_kernel[grid](
        q_buf,
        r_words,
        s_words,
        pk_ok_t,
        out_ok,
        b,
        1,
        BLOCK=block,
        num_warps=nw,
    )
    ok = out_ok.to(torch.bool).cpu().numpy()
    if bad.any():
        ok = ok & (~bad)
    return ok


def _to_32b_2d(x, dev):
    import torch

    if isinstance(x, torch.Tensor):
        t = x.to(device=dev, dtype=torch.uint8)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        t = t.contiguous()
    else:
        arr = np.array(as_batch_bytes(x, 32), dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).to(dev)
    if t.ndim != 2 or t.shape[1] != 32:
        raise ValueError(f"expected [B, 32] uint8, got shape {tuple(t.shape)}")
    return t


def x25519_torch(scalars, u_coords, device=None, block: int = 64):
    """Batched RFC 7748 X25519 on GPU: `[B, 32] uint8` x `[B, 32] uint8` -> `[B, 32] uint8`.

    Either operand may have batch size 1 to broadcast against the other.
    """
    import torch

    dev = _devices.resolve_device(device)
    s = _to_32b_2d(scalars, dev)
    u = _to_32b_2d(u_coords, dev)
    b = max(s.shape[0], u.shape[0])
    if s.shape[0] == 0 or u.shape[0] == 0:
        return torch.empty((0, 32), dtype=torch.uint8, device=dev)
    if s.shape[0] not in (1, b) or u.shape[0] not in (1, b):
        raise ValueError(f"incompatible batch sizes: scalars={s.shape[0]}, u_coords={u.shape[0]}")

    out_words = torch.empty((b, 8), dtype=torch.int32, device=dev)
    _x25519_kernel[(triton.cdiv(b, block),)](
        s.view(torch.int32),
        u.view(torch.int32),
        out_words,
        b,
        0 if s.shape[0] == 1 else 8,
        0 if u.shape[0] == 1 else 8,
        BLOCK=block,
        num_warps=_devices.warps("x25519", 4),
    )
    return out_words.view(torch.uint8)


def x25519_base_torch(scalars, device=None, block: int = 64):
    """Batched RFC 7748 X25519 base-point scalar multiplication (`u = 9`)."""
    base = (9).to_bytes(32, "little")
    return x25519_torch(scalars, base, device=device, block=block)
