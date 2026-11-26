"""Batched LMS / HSS verification on the device.

Per LMS level: one message hash (`Q`), p Winternitz chains (one fused launch, see
`kernels.chains`), the LM-OTS public-key hash (`Kc`), and h tree hashes. Everything but the chains
is a SHA-256 over a buffer the host composes, so a whole level costs about `6 + h` launches for the
entire batch.

HSS is a chain of LMS verifications: level i signs the public key of level i+1, and the last level
signs the message. Levels may use different parameter sets (RFC 8554 test case 2 does), so the set
is re-read from each embedded public key.
"""

from __future__ import annotations

import struct

import numpy as np
import torch

from .._inputs import as_bytes_list
from ..devices import resolve_device
from ..primitives import sha2 as _sha2
from . import kernels as _kernels
from .params import LMSSet, get

D_PBLC = 0x8080
D_MESG = 0x8181
D_LEAF = 0x8282
D_INTR = 0x8383


def _coef_rows(q: torch.Tensor, w: int, count: int) -> torch.Tensor:
    """The `count` w-bit digits of each row of `q`, most significant bits first."""
    rows = q.shape[0]
    dev = q.device
    i = torch.arange(count, device=dev)
    byte = ((i * w) // 8).expand(rows, count)
    shift = (8 - w - ((i * w) % 8)).expand(rows, count)
    bits = q.long().gather(1, byte)
    return (bits >> shift) & ((1 << w) - 1)


def _u32_rows(v: torch.Tensor, rows: int) -> torch.Tensor:
    return torch.stack([((v >> (8 * (3 - k))) & 0xFF).to(torch.uint8) for k in range(4)], dim=1)


def _const_rows(rows: int, value: int, dev) -> torch.Tensor:
    out = torch.full((rows, 2), value & 0xFF, dtype=torch.uint8, device=dev)
    out[:, 0] = value >> 8
    return out


class Verifier:
    """Verify LMS / HSS signatures in batches against one HSS public key."""

    def __init__(self, pub: bytes, device=None):
        self.device = resolve_device(device)
        pub = bytes(pub)
        if len(pub) < 4:
            raise ValueError("HSS public key is too short")
        self.levels = struct.unpack(">I", pub[:4])[0]
        if self.levels == 0 or self.levels > 8:
            raise ValueError(f"unsupported HSS level count {self.levels}")
        self.pk = pub[4:]
        self.set = get(*struct.unpack(">II", self.pk[:8]))
        if len(self.pk) != self.set.pk_size:
            raise ValueError(f"the level-0 LMS public key must be {self.set.pk_size} bytes")

    # ---------------- plumbing ----------------
    def _hash(self, data: torch.Tensor, nbytes: int = 32) -> torch.Tensor:
        """SHA-256 of each row of `data` (unpadded), truncated to `nbytes` -> [rows, nbytes]."""
        rows, length = data.shape
        blocks = -(-(length + 9) // 64)
        body = torch.zeros((rows, blocks * 64), dtype=torch.uint8, device=self.device)
        body[:, :length] = data
        body[:, length] = 0x80
        body[:, blocks * 64 - 8 :] = torch.tensor(
            list((8 * length).to_bytes(8, "big")), dtype=torch.uint8, device=self.device
        )
        out = torch.empty((rows, 8), dtype=torch.int32, device=self.device)
        _sha2.sha_absorb(
            _sha2.iv_tensor(256, self.device), body.view(torch.int32), out, rows, 256, blocks
        )
        return out.view(torch.uint8)[:, :nbytes]

    def _chains(self, qint, ident, ys, s: LMSSet, starts) -> torch.Tensor:
        """Run every LM-OTS chain to its end: [rows, p, n] uint8."""
        rows = ys.shape[0]
        dev = self.device
        nw = s.n // 4
        words = ys.reshape(rows * s.p, s.n).contiguous().view(torch.int32)
        ident_words = (
            ident.contiguous().view(torch.int32).repeat_interleave(s.p, dim=0).contiguous()
        )
        qw = qint.repeat_interleave(s.p).to(torch.int32).contiguous()
        cidx = torch.arange(s.p, dtype=torch.int32, device=dev).repeat(rows)
        start_flat = starts.reshape(-1).to(torch.int32).contiguous()
        # rows with fewer remaining steps first, so blocks finish together
        order = torch.argsort(start_flat)
        sorted_words = words[order].contiguous()
        # the per-row address fields must follow the same permutation as the values
        _kernels.chains(
            sorted_words,
            ident_words[order].contiguous(),
            qw[order].contiguous(),
            cidx[order].contiguous(),
            start_flat[order],
            s.w,
            nw,
        )
        out = torch.empty_like(words)
        out[order] = sorted_words
        return out.view(torch.uint8).reshape(rows, s.p, s.n)

    def _header_ok(self, sigs: torch.Tensor, s: LMSSet) -> torch.Tensor:
        """The type words and `q` must match the public key (ACVP's "modify header" cases)."""
        rows = sigs.shape[0]
        q = sigs[:, 0:4].to(torch.int64)
        qint = q[:, 0] << 24 | q[:, 1] << 16 | q[:, 2] << 8 | q[:, 3]
        ots = sigs[:, 4:8].to(torch.int64)
        ots_type = ots[:, 0] << 24 | ots[:, 1] << 16 | ots[:, 2] << 8 | ots[:, 3]
        off = 4 + s.ots_sig_size
        lms = sigs[:, off : off + 4].to(torch.int64)
        lms_type = lms[:, 0] << 24 | lms[:, 1] << 16 | lms[:, 2] << 8 | lms[:, 3]
        ok = torch.ones(rows, dtype=torch.bool, device=self.device)
        ok &= qint < (1 << s.h)
        ok &= ots_type == s.ots_type
        ok &= lms_type == s.lms_type
        return ok

    # ---------------- one LMS level ----------------
    def _lms(
        self, pks: torch.Tensor, msgs: torch.Tensor, sigs: torch.Tensor, s: LMSSet
    ) -> torch.Tensor:
        """Recompute the LMS root for each row."""
        rows = sigs.shape[0]
        dev = self.device
        ident = pks[:, 8:24].contiguous()
        qbytes = sigs[:, 0:4].to(torch.int64)
        qint = qbytes[:, 0] << 24 | qbytes[:, 1] << 16 | qbytes[:, 2] << 8 | qbytes[:, 3]
        ots = sigs[:, 4 : 4 + s.ots_sig_size]
        c = ots[:, 4 : 4 + s.n]
        ys = ots[:, 4 + s.n :].reshape(rows, s.p, s.n)
        path = sigs[:, 4 + s.ots_sig_size + 4 :].reshape(rows, s.h, s.m)

        # Q = H(I || u32(q) || u16(D_MESG) || C || msg)
        qh = self._hash(
            torch.cat([ident, sigs[:, 0:4], _const_rows(rows, D_MESG, dev), c, msgs], dim=1), s.n
        )
        digits = torch.cat([qh, torch.zeros((rows, 2), dtype=torch.uint8, device=dev)], dim=1)
        per = _coef_rows(qh, s.w, (s.n * 8) // s.w)
        cksm = ((1 << s.w) - 1 - per).sum(dim=1) << s.ls
        digits[:, s.n] = (cksm >> 8).to(torch.uint8) & 0xFF
        digits[:, s.n + 1] = (cksm & 0xFF).to(torch.uint8)

        zs = self._chains(qint, ident, ys, s, _coef_rows(digits, s.w, s.p))

        # Kc = H(I || u32(q) || u16(D_PBLC) || z[0] || ... || z[p-1])
        kc = self._hash(
            torch.cat(
                [ident, sigs[:, 0:4], _const_rows(rows, D_PBLC, dev), zs.reshape(rows, s.p * s.n)],
                dim=1,
            ),
            s.n,
        )

        # the tree: leaf hash, then h interior hashes against the sibling in `path`
        node = (1 << s.h) + qint
        root = self._hash(
            torch.cat([ident, _u32_rows(node, rows), _const_rows(rows, D_LEAF, dev), kc], dim=1),
            s.m,
        )
        for k in range(s.h):
            parent = node >> 1
            auth = path[:, k, :]
            odd = ((node & 1) == 1)[:, None]
            left = torch.where(odd, auth, root)
            right = torch.where(odd, root, auth)
            root = self._hash(
                torch.cat(
                    [ident, _u32_rows(parent, rows), _const_rows(rows, D_INTR, dev), left, right],
                    dim=1,
                ),
                s.m,
            )
            node = parent
        return root

    # ---------------- HSS ----------------
    def verify(self, messages, signatures) -> np.ndarray:
        msgs = as_bytes_list(messages, "message")
        sigs = as_bytes_list(signatures, "signature")
        n = len(msgs)
        out = np.zeros(n, dtype=bool)
        if n == 0:
            return out
        if len(sigs) != n:
            raise ValueError(f"got {n} messages but {len(sigs)} signatures")
        # A malformed signature is invalid, not an exception: reject short/absurd level
        # counts here and let the kernel path handle the rest. Previously a single `bytes`
        # message became a list of integer items and raised IndexError.
        valid: list[int] = []
        for i, sig in enumerate(sigs):
            if len(sig) < 4:
                continue
            # RFC 8554 §6: the HSS signature starts with N_spk = N - 1, not with N.
            nspk = struct.unpack(">I", sig[:4])[0]
            if nspk + 1 != self.levels or self._sig_size(sig) != len(sig):
                continue
            valid.append(i)
        if not valid:
            return out
        buckets: dict[tuple[int, int], list[int]] = {}
        for i in valid:
            buckets.setdefault((len(msgs[i]), len(sigs[i])), []).append(i)
        for (length, _sig_len), idxs in buckets.items():
            try:
                out[idxs] = self._verify_group(
                    [msgs[i] for i in idxs], [sigs[i] for i in idxs], length
                )
            except (ValueError, IndexError, RuntimeError, struct.error):
                # invalid embedded type codes / truncated structures: nothing verifies
                out[idxs] = False
        return out

    def _sig_size(self, sig: bytes) -> int | None:
        """Expected signature length, read from the embedded public-key/type words.

        HSS level i's signature contains the level-(i+1) public key, whose size depends on
        the *next* set (levels may mix, e.g. RFC 8554 case 2), so the walk needs the actual
        headers. Returns ``None`` for an unsupported or truncated chain.
        """
        cur = self.pk
        off = 4  # N_spk
        for level in range(self.levels):
            if len(cur) < 8:
                return None
            try:
                s = get(*struct.unpack(">II", cur[:8]))
            except ValueError:
                return None
            off += s.sig_size
            if level + 1 < self.levels:
                if len(sig) < off + 8:
                    return None
                try:
                    ns = get(*struct.unpack(">II", sig[off : off + 8]))
                except ValueError:
                    return None
                off += ns.pk_size
                cur = sig[off - ns.pk_size : off]
        return off

    def _verify_group(self, msgs: list[bytes], sigs: list[bytes], length: int) -> np.ndarray:
        dev = self.device
        rows = len(msgs)
        cur_pub = (
            torch.frombuffer(bytearray(self.pk), dtype=torch.uint8)
            .to(dev)
            .reshape(1, -1)
            .repeat(rows, 1)
        )
        msg_t = (
            torch.frombuffer(bytearray(b"".join(msgs)), dtype=torch.uint8)
            .to(dev)
            .reshape(rows, length)
        )
        all_sigs = (
            torch.frombuffer(bytearray(b"".join(sigs)), dtype=torch.uint8).to(dev).reshape(rows, -1)
        )

        ok = torch.ones(rows, dtype=torch.bool, device=dev)
        sig_off = 4
        for _ in range(self.levels - 1):
            s = get(*struct.unpack(">II", cur_pub[0, :8].cpu().numpy().tobytes()))
            sig_t = all_sigs[:, sig_off : sig_off + s.sig_size]
            # The signed public key follows the LMS signature; its size comes from its own
            # header (the next level may use a different parameter set).
            nxt = sig_off + s.sig_size
            ns = get(*struct.unpack(">II", all_sigs[0, nxt : nxt + 8].cpu().numpy().tobytes()))
            next_pub = all_sigs[:, nxt : nxt + ns.pk_size]
            ok &= self._header_ok(sig_t, s)
            root = self._lms(cur_pub, next_pub, sig_t, s)
            ok &= (root == cur_pub[:, 24 : 24 + s.m]).all(dim=1)
            cur_pub = next_pub
            sig_off += s.sig_size + s.pk_size
            if not bool(ok.any()):
                return ok.cpu().numpy()

        s = get(*struct.unpack(">II", cur_pub[0, :8].cpu().numpy().tobytes()))
        sig_t = all_sigs[:, sig_off:]
        ok &= self._header_ok(sig_t, s)
        root = self._lms(cur_pub, msg_t, sig_t, s)
        ok &= (root == cur_pub[:, 24 : 24 + s.m]).all(dim=1)
        return ok.cpu().numpy()


def verify_batch(pub: bytes, messages, signatures, device=None) -> np.ndarray:
    """One-shot batched verification under one HSS public key."""
    return Verifier(pub, device=device).verify(messages, signatures)
