"""Falcon-512 and Falcon-1024 verification, batched end-to-end on the device.

Every stage of verification runs on the GPU for uniform-length batches:

| stage | where | notes |
|---|---|---|
| public-key decode (`modq_decode`) | host, once per verifier | 14-bit stream, `n` coeffs, negligible |
| signature decode (`comp_decode`) | **device** | transposed `[W, B]` uint32 + 64-bit bit reservoir, branchless CLZ |
| `HashToPoint` (SHAKE256 + rejection) | **device** (uniform lengths) / host (mixed) | `_pad_nonce_msg_kernel` + `sponge` + `_hash_to_point_kernel` |
| `s2_hat = NTT(s2)`, `s2_hat * h_hat`, `INTT` | **device** | three batched launches on `primitives.ntt` |
| `s1 = c - s2*h`, normalise, L2 bound | **device** | `_check_kernel`, one program per signature |

`verify_core_torch` remains the device-only entry point for pre-decoded `(c, s2)` tensors so
core throughput and full end-to-end throughput can both be measured cleanly.
"""

from __future__ import annotations

import numpy as np
import triton

from .. import devices as _devices
from ..primitives.ntt import intt_torch, ntt_torch
from . import ref
from .kernels import (
    _check_kernel,
    _comp_decode_kernel,
    _hash_to_point_kernel,
    _mul_ntt_kernel,
    _pack_sig_words_kernel,
    _pad_nonce_msg_kernel,
)
from .params import FALCON_512, FALCON_1024, FalconParams

__all__ = [
    "Verifier",
    "comp_decode_batch",
    "comp_decode_batch_torch",
    "hash_to_point_batch_torch",
]


def comp_decode_batch(
    bodies: list[bytes], params: FalconParams = FALCON_512, padded: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Host NumPy reference batch decoder: `[B, n]` signed coefficients + validity mask.

    A malformed signature is *invalid*, not an exception: a verifier's job is to say no.
    The mask is carried instead of returning zeros, because a zero row would sail through the
    norm check as "short" and turn garbage into an accepted signature.
    """
    x, ok, _used = ref.comp_decode_batch_np(bodies, params, padded=padded)
    return x.astype(np.int32), ok


def _comp_decode_from_flat_torch(
    flat_sigs_t,
    sig_offs_t,
    body_lens_t,
    max_body: int,
    params: FalconParams,
    padded: bool,
):
    """Device-side `comp_decode` directly from a packed flat signature buffer on GPU."""
    import torch

    b = int(body_lens_t.shape[0])
    dev = flat_sigs_t.device
    n = params.n
    skip = 1 + params.nonce_len
    padded_body_len = params.sig_padded - skip
    if b == 0:
        return (
            torch.empty((0, n), dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.int8, device=dev),
        )

    # Every coefficient consumes at most 24 bits (3 bytes), so a valid body is at most 3*n bytes.
    w_words = max(1, min((max_body + 3) // 4, (3 * n + 3) // 4))
    words_t = torch.empty((w_words, b), dtype=torch.uint32, device=dev)
    _pack_sig_words_kernel[(triton.cdiv(b, 64), triton.cdiv(w_words, 16))](
        flat_sigs_t,
        sig_offs_t,
        body_lens_t,
        words_t,
        b,
        BODY_SKIP=skip,
        W=w_words,
        BLK_B=64,
        BLK_W=16,
    )
    s2_t = torch.empty((b, n), dtype=torch.int32, device=dev)
    ok_t = torch.empty(b, dtype=torch.int8, device=dev)
    _comp_decode_kernel[(triton.cdiv(b, 128),)](
        words_t,
        body_lens_t,
        s2_t,
        ok_t,
        b,
        W=w_words,
        N=n,
        MAX_MAG=params.max_abs_coefficient,
        PADDED_BODY_LEN=padded_body_len,
        ALLOW_PADDED=padded,
        BLK=128,
    )
    return s2_t, ok_t


def comp_decode_batch_torch(
    bodies: list[bytes],
    params: FalconParams = FALCON_512,
    padded: bool = True,
    device=None,
):
    """Device-side `comp_decode` over raw signature bodies (header + nonce already stripped).

    Returns `(s2_t [B, n] int32, ok_t [B] bool)` on `device`.
    """
    import torch

    dev = _devices.resolve_device(device)
    b = len(bodies)
    if b == 0:
        return (
            torch.empty((0, params.n), dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.bool, device=dev),
        )
    lens = np.fromiter((len(x) for x in bodies), dtype=np.int32, count=b)
    offs = np.empty(b, dtype=np.int32)
    offs[0] = 0
    if b > 1:
        np.cumsum(lens[:-1], out=offs[1:])
    joined = b"".join(bodies)
    flat_np = (
        np.frombuffer(joined, dtype=np.uint8).copy() if joined else np.zeros(1, dtype=np.uint8)
    )
    flat_t = torch.from_numpy(flat_np).to(dev)
    offs_t = torch.from_numpy(offs).to(dev)
    lens_t = torch.from_numpy(lens).to(dev)

    n = params.n
    max_body = int(lens.max())
    w_words = max(1, min((max_body + 3) // 4, (3 * n + 3) // 4))
    words_t = torch.empty((w_words, b), dtype=torch.uint32, device=dev)
    _pack_sig_words_kernel[(triton.cdiv(b, 64), triton.cdiv(w_words, 16))](
        flat_t,
        offs_t,
        lens_t,
        words_t,
        b,
        BODY_SKIP=0,
        W=w_words,
        BLK_B=64,
        BLK_W=16,
    )
    s2_t = torch.empty((b, n), dtype=torch.int32, device=dev)
    ok_t = torch.empty(b, dtype=torch.int8, device=dev)
    padded_body_len = params.sig_padded - 1 - params.nonce_len
    _comp_decode_kernel[(triton.cdiv(b, 128),)](
        words_t,
        lens_t,
        s2_t,
        ok_t,
        b,
        W=w_words,
        N=n,
        MAX_MAG=params.max_abs_coefficient,
        PADDED_BODY_LEN=padded_body_len,
        ALLOW_PADDED=padded,
        BLK=128,
    )
    return s2_t, ok_t.to(torch.bool)


def _hash_to_point_from_flat_torch(
    flat_sigs_t,
    sig_offs_t,
    messages,
    params: FalconParams,
):
    """Device-side `HashToPoint` using nonces already resident in `flat_sigs_t`."""
    import torch

    from ..primitives import sponge

    dev = flat_sigs_t.device
    b = len(messages)
    n = params.n
    if b == 0:
        return (
            torch.empty((0, n), dtype=torch.int32, device=dev),
            torch.empty(0, dtype=torch.int8, device=dev),
        )
    msg_len = len(messages[0])
    joined = b"".join(messages)
    msgs_np = (
        np.frombuffer(joined, dtype=np.uint8).copy() if joined else np.zeros(1, dtype=np.uint8)
    )
    msgs_t = torch.from_numpy(msgs_np).to(dev)

    rate = sponge.SHAKE256.rate
    total = params.nonce_len + msg_len
    padded_len = (total // rate + 1) * rate
    padded_t = torch.empty((b, padded_len), dtype=torch.uint8, device=dev)
    _pad_nonce_msg_kernel[(triton.cdiv(b, 32), triton.cdiv(padded_len, 64))](
        flat_sigs_t,
        sig_offs_t,
        msgs_t,
        padded_t,
        b,
        NONCE_LEN=params.nonce_len,
        MSG_LEN=msg_len,
        PADDED_LEN=padded_len,
        DOMAIN_BYTE=sponge.DOMAIN_SHAKE,
        BLK_B=32,
        BLK_C=64,
    )

    nsamp = n + params.oversample
    outlen = -(-nsamp * 2 // 8) * 8  # whole 64-bit words; extra bytes are ignored
    stream = sponge.sponge_padded(padded_t, rate, outlen)
    nsamp_aligned = outlen // 2
    cols = 1024 if nsamp_aligned <= 1024 else 2048
    blk = 32 if cols == 1024 else 16

    out = torch.empty((b, n), dtype=torch.int32, device=dev)
    ok = torch.empty(b, dtype=torch.int8, device=dev)
    _hash_to_point_kernel[(triton.cdiv(b, blk),)](
        stream, out, ok, b, N=n, NSAMP=nsamp_aligned, COLS=cols, Q=params.q, BLK=blk
    )
    return out, ok


def hash_to_point_batch_torch(nonces, messages, params: FalconParams = FALCON_512, device=None):
    """`HashToPoint` for a batch of equal-length `nonce || message`, all on the device.

    Returns `(c [B, n] int32, ok [B] bool)`. The uniform-length requirement is real: the sponge
    absorbs a fixed number of blocks per row, so messages whose padding lands in different block
    counts cannot share a launch (the ML-DSA signer groups them for the same reason). Callers
    with mixed lengths fall back to `ref.hash_to_point_batch`.
    """
    import torch

    from ..primitives import sponge

    dev = _devices.resolve_device(device)
    n = params.n
    nsamp = n + params.oversample
    outlen = -(-nsamp * 2 // 8) * 8  # whole 64-bit words; the extra bytes are ignored
    lens = {len(no) + len(m) for no, m in zip(nonces, messages, strict=True)}
    if len(lens) != 1:
        raise ValueError(f"device HashToPoint needs one message length, got {sorted(lens)}")
    b = len(nonces)
    msgs = np.empty((b, lens.pop()), dtype=np.uint8)
    for i, (no, m) in enumerate(zip(nonces, messages, strict=True)):
        msgs[i, : len(no)] = np.frombuffer(no, dtype=np.uint8)
        msgs[i, len(no) :] = np.frombuffer(m, dtype=np.uint8)
    padded = sponge.pad_batch(msgs, sponge.SHAKE256.rate, sponge.DOMAIN_SHAKE)
    stream = sponge.sponge_padded(torch.as_tensor(padded, device=dev), sponge.SHAKE256.rate, outlen)
    nsamp_aligned = outlen // 2
    cols = 1024 if nsamp_aligned <= 1024 else 2048
    blk = 32 if cols == 1024 else 16
    out = torch.zeros((b, n), dtype=torch.int32, device=dev)
    ok = torch.zeros(b, dtype=torch.int8, device=dev)
    _hash_to_point_kernel[(triton.cdiv(b, blk),)](
        stream, out, ok, b, N=n, NSAMP=nsamp_aligned, COLS=cols, Q=params.q, BLK=blk
    )
    return out, ok.cpu().numpy().astype(bool)


class Verifier:
    """A public key, uploaded once, that verifies batches of `(message, signature)`."""

    def __init__(
        self,
        pk: bytes,
        params: FalconParams | None = None,
        device=None,
        padded: bool = True,
    ):
        if params is None:
            if len(pk) == FALCON_1024.pk_len and pk and pk[0] == FALCON_1024.pk_header:
                params = FALCON_1024
            else:
                params = FALCON_512
        if len(pk) != params.pk_len:
            raise ValueError(f"public key must be {params.pk_len} bytes, got {len(pk)}")
        if pk[0] != params.pk_header:
            raise ValueError(f"bad public key header {pk[0]:#x}")
        self.params = params
        self.padded = padded
        self.device = _devices.resolve_device(device)
        self.pk_poly = ref.modq_decode(pk[1:], params)
        self._h_hat = None
        self._domain = None

    # ------------------------------------------------------------------ device core

    def _h_ntt(self):
        """`NTT(h)` on the device, computed once: the key is the batch-invariant operand."""
        if self._h_hat is None:
            import torch

            h = torch.tensor(self.pk_poly, dtype=torch.int32, device=self.device)[None, :]
            self._domain = _falcon_domain(self.params)
            self._h_hat = ntt_torch(h, self._domain)[0].contiguous()
            assert self._h_hat.shape == (self.params.n,), self._h_hat.shape
        return self._h_hat

    def verify_core_torch(self, c, s2):
        """Device-only: `c` (hashed point) and `s2` (decoded signature) as `[B, n]` int32.

        Returns `(ok, sqn)` as device tensors. Accepts either NumPy arrays or already-resident
        device tensors without copying.
        """
        import torch

        n, q = self.params.n, self.params.q
        c_t = torch.as_tensor(c, dtype=torch.int32, device=self.device).reshape(-1, n)
        s2_t = torch.as_tensor(s2, dtype=torch.int32, device=self.device).reshape(-1, n)
        batch = c_t.shape[0]
        if batch == 0:
            return (
                torch.empty(0, dtype=torch.int8, device=self.device),
                torch.empty(0, dtype=torch.int64, device=self.device),
            )
        s2_mod = s2_t % q

        domain = _falcon_domain(self.params)
        s2_hat = ntt_torch(s2_mod, domain)
        h_hat = self._h_ntt()
        prod_hat = torch.empty_like(s2_hat)
        _mul_ntt_kernel[(triton_cdiv(batch * n, 4096),)](
            s2_hat, h_hat, prod_hat, batch * n, Q=q, N=n, BLK=4096
        )
        prod = intt_torch(prod_hat, domain)

        sqn = torch.empty(batch, dtype=torch.int64, device=self.device)
        ok = torch.empty(batch, dtype=torch.int8, device=self.device)
        _check_kernel[(batch,)](
            prod, c_t, s2_t, sqn, ok, batch, L2BOUND=self.params.l2bound, Q=q, N=n
        )
        return ok, sqn

    # ------------------------------------------------------------------ public API

    def verify(self, messages, signatures) -> np.ndarray:
        """Verify `len(messages)` signatures end-to-end on the device; returns a bool array."""
        import torch

        if len(messages) != len(signatures):
            raise ValueError(f"{len(messages)} messages but {len(signatures)} signatures")
        b = len(messages)
        if b == 0:
            return np.zeros(0, dtype=bool)

        p = self.params
        skip = 1 + p.nonce_len
        sig_lens = np.fromiter((len(s) for s in signatures), dtype=np.int32, count=b)
        if int(sig_lens.min()) < skip:
            for i, sig in enumerate(signatures):
                if len(sig) < skip:
                    raise ValueError(f"row {i}: signature shorter than header + nonce")

        sig_offs = np.empty(b, dtype=np.int32)
        sig_offs[0] = 0
        if b > 1:
            np.cumsum(sig_lens[:-1], out=sig_offs[1:])
        flat_sigs_np = np.frombuffer(b"".join(signatures), dtype=np.uint8).copy()
        headers = flat_sigs_np[sig_offs]
        if not np.all(headers == p.sig_header):
            bad_idx = int(np.flatnonzero(headers != p.sig_header)[0])
            raise ValueError(f"row {bad_idx}: bad signature header {int(headers[bad_idx]):#x}")

        body_lens = sig_lens - skip
        max_body = int(body_lens.max())

        flat_sigs_t = torch.from_numpy(flat_sigs_np).to(self.device)
        sig_offs_t = torch.from_numpy(sig_offs).to(self.device)
        body_lens_t = torch.from_numpy(body_lens).to(self.device)

        s2_t, decode_ok_t = _comp_decode_from_flat_torch(
            flat_sigs_t, sig_offs_t, body_lens_t, max_body, p, padded=self.padded
        )

        first_mlen = len(messages[0])
        uniform_msgs = all(len(m) == first_mlen for m in messages)
        if uniform_msgs:
            # All messages share one SHAKE256 block count: keep `c_t` on the device end-to-end.
            c_t, hash_ok_t = _hash_to_point_from_flat_torch(flat_sigs_t, sig_offs_t, messages, p)
            core_ok_t, _ = self.verify_core_torch(c_t, s2_t)
            return (decode_ok_t & hash_ok_t & core_ok_t).cpu().numpy().astype(bool)

        # Mixed message lengths: hash on the host, keep GPU-decoded `s2_t` on device.
        c = np.stack(
            [
                ref.hash_to_point(sig[1:skip], msg, p)
                for sig, msg in zip(signatures, messages, strict=True)
            ]
        )
        core_ok_t, _ = self.verify_core_torch(c, s2_t)
        return (decode_ok_t & core_ok_t).cpu().numpy().astype(bool)


_DOMAINS: dict[tuple[int, int, int], object] = {}


def _falcon_domain(params: FalconParams):
    key = (params.q, params.n, params.logn)
    dom = _DOMAINS.get(key)
    if dom is None:
        from ..primitives.ntt import NTTDomain

        dom = NTTDomain(
            f"falcon-{params.n}",
            q=params.q,
            n=params.n,
            zeta=ref.root_of_unity(params.q, params.n),
            layers=params.logn,
        )
        _DOMAINS[key] = dom
    return dom


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
