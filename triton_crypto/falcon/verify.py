"""Falcon-512 verification, batched, with the arithmetic core on the device.

What runs where, and why it is split that way:

| stage | where | why |
|---|---|---|
| public-key decode | host, once per verifier | 14-bit stream, 512 coefficients, negligible |
| signature decode (`comp_decode`) | host, per signature | a variable-length prefix code; see the note below |
| `HashToPoint` (SHAKE256 + rejection) | host | the device sponge + a compaction kernel is the next step |
| `s2_hat = NTT(s2)`, `s2_hat * h_hat`, inverse NTT | **device** | three batched launches |
| `s1 = c - s2h`, normalise, L2 bound | **device** | one launch, one program per signature |

The split is deliberate and measured: the device core is the part with a throughput story
(~1 us/signature), while the host decode is linear in signatures and is what the end-to-end
number actually pays for. `verify_core_torch` is the device-only entry point, so the two can
be reported separately instead of hiding one inside the other.

`comp_decode` is a prefix code (sign + 7 low bits, then a unary continuation), so a batch
cannot be decoded with a vectorised bit gather without a per-coefficient scan; the honest
options are a numpy per-coefficient loop (what this file does) or a device scan kernel. The
loop decodes a signature body in ~200 us, which is why the device core is timed separately.
"""

from __future__ import annotations

import numpy as np
import triton

from .. import devices as _devices
from ..primitives.ntt import intt_torch, ntt_torch
from . import ref
from .kernels import _check_kernel, _hash_to_point_kernel, _mul_ntt_kernel
from .params import FALCON_512, FalconParams

__all__ = ["Verifier", "comp_decode_batch"]


def comp_decode_batch(
    bodies: list[bytes], params: FalconParams = FALCON_512, padded: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Decode one signature body per row into `[B, n]` coefficients, plus a validity mask.

    A malformed signature is *invalid*, not an exception: a verifier's job is to say no.
    The mask is carried instead of returning zeros, because a zero row would sail through the
    norm check as "short" and turn garbage into an accepted signature.
    """
    x, ok, used = ref.comp_decode_batch_np(bodies, params)
    if padded:
        # Falcon-padded pads the body with zeros to a fixed length; everything else must be
        # consumed exactly. Only the rows the strict rule rejected need this second look.
        pad_len = params.sig_padded - 1 - params.nonce_len
        for i in np.nonzero(~ok)[0]:
            body = bodies[i]
            if len(body) == pad_len and not any(body[used[i] :]):
                ok[i] = True
    return x.astype(np.int32), ok


def hash_to_point_batch_torch(nonces, messages, params: FalconParams = FALCON_512, device=None):
    """`HashToPoint` for a batch of equal-length `nonce || message`, all on the device.

    Returns `(c [B, n] int32, ok [B] bool)`. The uniform-length requirement is real: the sponge
    absorbs a fixed number of blocks per row, so messages whose padding lands in different block
    counts cannot share a launch (the ML-DSA signer groups them for the same reason). Callers
    with mixed lengths fall back to `ref.hash_to_point_batch`.
    """
    import torch

    from .. import devices as _devices
    from ..primitives import sponge

    dev = _devices.resolve_device(device)
    n = params.n
    nsamp = n + params.oversample
    outlen = -(-nsamp * 2 // 8) * 8  # whole 64-bit words; the extra bytes are ignored
    lens = {len(no) + len(m) for no, m in zip(nonces, messages, strict=True)}
    if len(lens) != 1:
        raise ValueError(f"device HashToPoint needs one message length, got {sorted(lens)}")
    msgs = np.empty((len(nonces), lens.pop()), dtype=np.uint8)
    for i, (no, m) in enumerate(zip(nonces, messages, strict=True)):
        msgs[i, : len(no)] = np.frombuffer(no, dtype=np.uint8)
        msgs[i, len(no) :] = np.frombuffer(m, dtype=np.uint8)
    padded = sponge.pad_batch(msgs, sponge.SHAKE256.rate, sponge.DOMAIN_SHAKE)
    stream = sponge.sponge_padded(torch.as_tensor(padded, device=dev), sponge.SHAKE256.rate, outlen)
    pairs = stream.reshape(len(nonces), outlen // 2, 2).to(torch.int32)
    samples = (pairs[:, :, 0] << 8) | pairs[:, :, 1]  # big-endian 16-bit samples
    out = torch.zeros((len(nonces), n), dtype=torch.int32, device=dev)
    ok = torch.zeros(len(nonces), dtype=torch.bool, device=dev)
    blk = 32
    _hash_to_point_kernel[(triton.cdiv(len(nonces), blk),)](
        samples, out, ok, len(nonces), N=n, NSAMP=outlen // 2, Q=params.q, BLK=blk
    )
    return out, ok.cpu().numpy()


class Verifier:
    """A public key, uploaded once, that verifies batches of `(message, signature)`."""

    def __init__(
        self, pk: bytes, params: FalconParams = FALCON_512, device=None, padded: bool = True
    ):
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

            q = self.params.q
            h = torch.tensor(self.pk_poly, dtype=torch.int32, device=self.device)[None, :]
            self._domain = _falcon_domain(self.params)
            self._h_hat = ntt_torch(h, self._domain)[0].contiguous()
            assert self._h_hat.shape == (self.params.n,), self._h_hat.shape
            del q
        return self._h_hat

    def verify_core_torch(self, c, s2):
        """Device-only: `c` (hashed point) and `s2` (decoded signature) as `[B, n]` int32.

        Returns `(ok, sqn)` as device tensors. No host decoding, so this is the number to
        quote for the device's throughput.
        """
        import torch

        n, q = self.params.n, self.params.q
        c_t = torch.as_tensor(c, dtype=torch.int32, device=self.device).reshape(-1, n)
        s2_t = torch.as_tensor(s2, dtype=torch.int32, device=self.device).reshape(-1, n)
        batch = c_t.shape[0]
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
        """Verify `len(messages)` signatures; returns a bool array."""
        if len(messages) != len(signatures):
            raise ValueError(f"{len(messages)} messages but {len(signatures)} signatures")
        if not messages:
            return np.zeros(0, dtype=bool)
        p = self.params
        bodies, nonces = [], []
        for i, sig in enumerate(signatures):
            if len(sig) < 1 + p.nonce_len:
                raise ValueError(f"row {i}: signature shorter than header + nonce")
            if sig[0] != p.sig_header:
                raise ValueError(f"row {i}: bad signature header {sig[0]:#x}")
            nonces.append(sig[1 : 1 + p.nonce_len])
            bodies.append(sig[1 + p.nonce_len :])
        s2, decode_ok = comp_decode_batch(bodies, p, padded=self.padded)
        lens = {len(no) + len(m) for no, m in zip(nonces, messages, strict=True)}
        if len(lens) == 1:
            # uniform lengths share one sponge launch (the common case for a benchmark, and for
            # any protocol that fixes the message size); mixed lengths take the host path.
            c_t, hash_ok = hash_to_point_batch_torch(nonces, messages, p, self.device)
            c = c_t.cpu().numpy()
            decode_ok = decode_ok & hash_ok
        else:
            c = np.stack(
                [ref.hash_to_point(n, m, p) for n, m in zip(nonces, messages, strict=True)]
            )
        ok, _ = self.verify_core_torch(c, s2)
        return decode_ok & ok.cpu().numpy().astype(bool)


_DOMAINS: dict[str, object] = {}


def _falcon_domain(params: FalconParams):
    dom = _DOMAINS.get(params.name)
    if dom is None:
        from ..primitives.ntt import NTTDomain

        dom = NTTDomain(
            "falcon",
            q=params.q,
            n=params.n,
            zeta=ref.root_of_unity(params.q, params.n),
            layers=params.logn,
        )
        _DOMAINS[params.name] = dom
    return dom


def triton_cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b
