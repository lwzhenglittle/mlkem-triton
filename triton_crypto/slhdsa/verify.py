"""Batched SLH-DSA (FIPS 205) signature verification on the device.

Scope: the SHAKE parameter sets (6 of 12). The SHA-2 sets need a SHA-256/SHA-512 compression
kernel that this repository does not have yet, so they raise `NotImplementedError` instead of
silently returning wrong answers — the numpy reference covers them.

Shape of the work: one verification is 2k–12k single-block SHAKE256 permutations. Within a
signature they form a dependency chain (layer j+1 needs layer j's root), but across a batch of B
signatures there are `B * 35` independent chains per layer, which is what makes this worth putting
on a device at all. The schedule is therefore "one launch per chain step, over the whole batch":

    per layer:  1 digits step + 15 chain steps + 1 ltree + h' auth-path steps
    per batch:  d layers, then FORS (k trees of height a), then H_msg

Each launch is B*len = 143k permutations wide at B = 4096 for ML-DSA-... for 128f, which is why
the emulation hit ~210 Mperm/s (docs/benchmarks.md).
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from ..devices import resolve_device
from ..primitives import sponge as _sponge
from . import kernels as _kernels
from .params import SLHParams, get
from .ref import FORS_ROOTS, FORS_TREE, OID, PH, TREE, WOTS_HASH, WOTS_PK

__all__ = ["Verifier", "verify_batch"]

RATE = 136  # SHAKE256 rate


def _u32(buf: torch.Tensor, off: int, value) -> None:
    """Write a big-endian uint32 into `buf[:, off:off+4]`; `value` may be a scalar or [rows]."""
    if not torch.is_tensor(value):
        value = torch.full((buf.shape[0],), int(value), dtype=torch.int64, device=buf.device)
    value = value.to(torch.int64)
    for k in range(4):
        buf[:, off + k] = ((value >> (8 * (3 - k))) & 0xFF).to(torch.uint8)


def _u64(buf: torch.Tensor, off: int, value) -> None:
    """Write a big-endian uint64 into `buf[:, off:off+8]`."""
    if not torch.is_tensor(value):
        value = torch.full((buf.shape[0],), int(value), dtype=torch.int64, device=buf.device)
    value = value.to(torch.int64)
    for k in range(8):
        buf[:, off + k] = ((value >> (8 * (7 - k))) & 0xFF).to(torch.uint8)


def _be(value, width: int, rows: int, device) -> torch.Tensor:
    """Big-endian bytes of an int or of an int64 bit pattern ([rows] tensor) -> [rows, width]."""
    if not torch.is_tensor(value):
        value = torch.full((rows,), int(value), dtype=torch.int64, device=device)
    value = value.to(torch.int64)
    cols = [((value >> (8 * (width - 1 - k))) & 0xFF).to(torch.uint8) for k in range(width)]
    return torch.stack(cols, dim=1)


def _bits(b: torch.Tensor, count: int) -> torch.Tensor:
    """[B, k] uint8 -> [B, count] bits, MSB first (the order base_2b uses)."""
    shifts = torch.arange(7, -1, -1, device=b.device)
    return ((b[:, :, None] >> shifts[None, None, :]) & 1).reshape(b.shape[0], -1)[:, :count]


def _bits_to_int(bits: torch.Tensor, width: int) -> torch.Tensor:
    B = bits.shape[0]
    v = bits.to(torch.int64).reshape(B, -1, width)
    out = torch.zeros((B, v.shape[1]), dtype=torch.int64, device=bits.device)
    for j in range(width):
        out = (out << 1) | v[:, :, j]
    return out


def _bytes_to_int(b: torch.Tensor) -> torch.Tensor:
    """[B, k] uint8 big-endian -> [B] int64."""
    out = torch.zeros((b.shape[0],), dtype=torch.int64, device=b.device)
    for j in range(b.shape[1]):
        out = (out << 8) | b[:, j].to(torch.int64)
    return out


class Verifier:
    """Verify SLH-DSA signatures in batches, with one long-lived public key."""

    def __init__(self, ps: str | SLHParams, pk: bytes, device=None):
        self.p = get(ps) if isinstance(ps, str) else ps
        if not self.p.is_shake:
            raise NotImplementedError(
                f"{self.p.name}: the device path implements the SHAKE sets only; the SHA-2 sets "
                "need a SHA-256/SHA-512 kernel (the numpy reference covers them)"
            )
        p = self.p
        if len(pk) != p.pk_size:
            raise ValueError(f"public key must be {p.pk_size} bytes, got {len(pk)}")
        self.device = resolve_device(device)
        self.pk = bytes(pk)
        self.seed_b = self.pk[: p.n]
        self.root_b = self.pk[p.n :]
        self._seed_row = None
        self._root_row = None
        self._seed_w = None

    # ---------------- device plumbing ----------------
    def _dev(self, b: bytes) -> torch.Tensor:
        return torch.frombuffer(bytearray(b), dtype=torch.uint8).to(self.device)

    def _adrs(self, rows, *, layer=0, tree=0, typ=0, kp=None, h=None, index=None):
        buf = torch.zeros((rows, 32), dtype=torch.uint8, device=self._dev_t())
        _u32(buf, 0, layer)
        if torch.is_tensor(tree) and tree.dim() == 2:
            # [batch, 12] of already-encoded tree address bytes (needed when the value exceeds
            # int64, which happens for 256f's full 64-bit tree index); repeat it to `rows`
            t = tree.to(buf.device)
            if t.shape[0] != rows:
                t = t.repeat_interleave(rows // t.shape[0], dim=0)
            buf[:, 4:16] = t
        else:
            _u64(buf, 8, tree)
        _u32(buf, 16, typ)
        if kp is not None:
            _u32(buf, 20, kp)
        if h is not None:
            _u32(buf, 24, h)
        if index is not None:
            _u32(buf, 28, index)
        return buf

    def _dev_t(self):
        return self.device

    def _thash(self, data, kp, chain, index, tree, *, layer=0, typ=0):
        """

        Single-block inputs (n + 32 + dn < 136: every F and H call) go through the fused kernel;
        the ltree hashes (dn = len·n, several blocks) fall back to compose + sponge.
        """
        """`H(PK.seed || ADRS || data)` for every row, in one fused launch.

        `data` is [rows, dn] uint8 with dn = n (F) or 2n (H); `kp`/`chain`/`index`/`tree` are
        [rows] int64 (the ADRS fields), `layer` and `typ` are scalars.
        """
        rows = data.shape[0]
        n = self.p.n
        if n + 32 + data.shape[1] >= RATE:
            return self._thash_multi(data, layer, typ, kp, tree)
        out = torch.empty((rows, n // 8), dtype=torch.int64, device=self._dev_t())
        _kernels.thash(
            self._seed_w,
            data.contiguous().view(torch.int64),
            out,
            kp,
            chain,
            index,
            tree,
            layer,
            typ,
            n,
        )
        return out.view(torch.uint8)

    def _thash_multi(self, data, layer, typ, kp, tree):
        """`H(PK.seed || ADRS || data)` for messages that span several blocks (ltree)."""
        p = self.p
        n = p.n
        rows = data.shape[0]
        dev = self._dev_t()
        adrs = torch.zeros((rows, 32), dtype=torch.uint8, device=dev)
        adrs[:, 0:4] = _be(layer, 4, rows, dev)
        adrs[:, 8:16] = _be(tree, 8, rows, dev)
        adrs[:, 16:20] = _be(typ, 4, rows, dev)
        adrs[:, 20:24] = _be(kp, 4, rows, dev)
        msg = torch.cat([self._seed_row.expand(rows, n), adrs, data], dim=1)
        padded = _sponge.pad_torch(msg, RATE, _sponge.DOMAIN_SHAKE)
        return _sponge.sponge_padded(padded, RATE, n)

    # ---------------- FORS (Alg. 15) ----------------
    def _fors(self, sig, md, idx_leaf, idx_tree):
        p = self.p
        n = p.n
        B = sig.shape[0]
        rows = B * p.k
        dev = self._dev_t()

        vi = _bits_to_int(_bits(md, p.a * p.k).reshape(B, p.k, p.a), p.a)  # [B, k]
        tree_index = ((torch.arange(p.k, device=dev)[None, :] << p.a) + vi).reshape(-1)
        kp_row = idx_leaf.repeat_interleave(p.k)
        tree_row = idx_tree.repeat_interleave(p.k)
        zero_row = torch.zeros(rows, dtype=torch.int64, device=dev)

        fors_view = sig[:, p.n : p.n + p.fors_sig_size].reshape(B, p.k, p.a + 1, n)
        node = self._thash(
            fors_view[:, :, 0, :].reshape(rows, n),
            kp_row,
            zero_row,
            tree_index,
            tree_row,
            layer=0,
            typ=FORS_TREE,
        )
        for j in range(p.a):
            auth = fors_view[:, :, j + 1, :].reshape(rows, n)
            bit = ((vi >> j) & 1).reshape(-1).to(torch.bool)
            left = torch.where(bit[:, None], auth, node)
            right = torch.where(bit[:, None], node, auth)
            node = self._thash(
                torch.cat([left, right], dim=1),
                kp_row,
                torch.full((rows,), j + 1, dtype=torch.int64, device=dev),
                tree_index >> (j + 1),
                tree_row,
                layer=0,
                typ=FORS_TREE,
            )
        zero_b = torch.zeros(B, dtype=torch.int64, device=dev)
        return self._thash(
            node.reshape(B, p.k * n), idx_leaf, zero_b, zero_b, idx_tree, layer=0, typ=FORS_ROOTS
        )

    # ---------------- WOTS+ / XMSS (Alg. 8, 13) ----------------
    def _wots_digits(self, msg):
        """Alg. 5 — the len1 message digits followed by the checksum digits."""
        p = self.p
        B = msg.shape[0]
        nib = torch.empty((B, 2 * msg.shape[1]), dtype=torch.int64, device=msg.device)
        nib[:, 0::2] = msg >> 4
        nib[:, 1::2] = msg & 0x0F
        digits = nib[:, : p.len1]
        csum = (p.w - 1 - digits).sum(dim=1) << ((8 - (p.len2 * p.lg_w) % 8) % 8)
        extra = torch.stack([(csum >> 8) >> 4, ((csum >> 8) & 0xF), (csum & 0xFF) >> 4], dim=1)
        return torch.cat([digits, extra[:, : p.len2]], dim=1)

    def _xmss(self, idx_leaf, sig_layer, msg, layer, idx_tree):
        p = self.p
        n = p.n
        B = msg.shape[0]
        len_ = p.wots_len
        rows = B * len_
        dev = self._dev_t()

        wots = sig_layer[:, : len_ * n].reshape(B, len_, n)
        auth = sig_layer[:, len_ * n :].reshape(B, p.hp, n)

        digits = self._wots_digits(msg)  # [B, len]
        kp_row = idx_leaf.repeat_interleave(len_)
        tree_row = idx_tree.repeat_interleave(len_)
        chain_row = torch.arange(len_, dtype=torch.int64, device=dev).repeat(B)
        pos = digits.reshape(-1).clone()
        remaining = (p.w - 1) - pos
        value = wots.reshape(rows, n)
        for _ in range(p.w - 1):
            hashed = self._thash(
                value, kp_row, chain_row, pos, tree_row, layer=layer, typ=WOTS_HASH
            )
            value = torch.where((remaining > 0)[:, None], hashed, value)
            pos = pos + 1
            remaining = torch.clamp(remaining - 1, min=0)

        zero_b = torch.zeros(B, dtype=torch.int64, device=dev)
        root = self._thash(
            value.reshape(B, len_ * n), idx_leaf, zero_b, zero_b, idx_tree, layer=layer, typ=WOTS_PK
        )
        for k in range(p.hp):
            bit = ((idx_leaf >> k) & 1).to(torch.bool)
            left = torch.where(bit[:, None], auth[:, k, :], root)
            right = torch.where(bit[:, None], root, auth[:, k, :])
            root = self._thash(
                torch.cat([left, right], dim=1),
                zero_b,
                torch.full((B,), k + 1, dtype=torch.int64, device=dev),
                idx_leaf >> (k + 1),
                idx_tree,
                layer=layer,
                typ=TREE,
            )
        return root

    def _ht(self, root, sig_ht, tree_addrs, leaf_idx):
        """Alg. 14 — one XMSS root per layer; `tree_addrs[layer]` is a [B] int64 tensor."""
        p = self.p
        step = p.xmss_sig_size
        for layer in range(p.d):
            root = self._xmss(
                leaf_idx[layer],
                sig_ht[:, layer * step : (layer + 1) * step],
                root,
                layer,
                tree_addrs[layer],
            )
        return root

    # ---------------- Alg. 20 ----------------
    def verify_mprime(self, mprimes, signatures) -> np.ndarray:
        """Verify against already-formatted M' (the "internal" interface); returns bool[B]."""
        p = self.p
        sig = _as_u8(signatures)
        mp = _as_bytes(mprimes)
        B = len(mp)
        out = np.zeros((B,), dtype=bool)
        if B == 0 or sig.shape[0] != B or sig.shape[1] != p.sig_size:
            return out
        self._seed_row = self._dev(self.seed_b)
        self._root_row = self._dev(self.root_b)
        self._seed_w = torch.frombuffer(bytearray(self.seed_b), dtype=torch.int64).to(self._dev_t())
        sig_t = torch.frombuffer(bytearray(np.ascontiguousarray(sig).tobytes()), dtype=torch.uint8)
        sig_t = sig_t.to(self._dev_t()).reshape(B, p.sig_size)

        groups = defaultdict(list)
        for b, m in enumerate(mp):
            groups[len(m)].append(b)

        for length, idxs in groups.items():
            sel = torch.tensor(idxs, dtype=torch.long, device=self._dev_t())
            sub_sig = sig_t.index_select(0, sel)
            g = len(idxs)
            mp_t = (
                torch.frombuffer(bytearray(b"".join(mp[b] for b in idxs)), dtype=torch.uint8)
                .to(self._dev_t())
                .reshape(g, length)
            )

            digest = self._h_msg(sub_sig, mp_t, g)
            md = digest[:, : p.md_bytes]
            # The digest's index fields are unsigned big-endian integers, and for 256f the tree
            # index is a full 64 bits — int64 arithmetic would sign-extend it (and 2**64-1 does
            # not even fit int64), so the split is done on the host and handed over per layer.
            raw = digest.cpu().numpy()
            trees, leaves = [], []
            it = [
                int.from_bytes(raw[b, p.md_bytes : p.md_bytes + p.tree_bytes], "big")
                & ((1 << (p.h - p.hp)) - 1)
                for b in range(g)
            ]
            il = [
                int.from_bytes(raw[b, p.md_bytes + p.tree_bytes :], "big") & ((1 << p.hp) - 1)
                for b in range(g)
            ]
            for _layer in range(p.d):
                trees.append(self._tree_tensor(it))
                leaves.append(torch.tensor(il, dtype=torch.int64, device=self._dev_t()))
                if _layer + 1 < p.d:
                    il = [v & ((1 << p.hp) - 1) for v in it]
                    it = [v >> p.hp for v in it]

            node = self._fors(sub_sig, md, leaves[0], trees[0])
            root = self._ht(node, sub_sig[:, p.n + p.fors_sig_size :], trees, leaves)
            ok = (root == self._root_row.expand_as(root)).all(dim=1).cpu().numpy()
            for j, b in enumerate(idxs):
                out[b] = bool(ok[j])
        return out

    def _tree_tensor(self, values: list[int]) -> torch.Tensor:
        """Tree addresses as int64 *bit patterns* (torch has no uint64; the kernel bitcasts)."""
        arr = np.array([v - (1 << 64) if v >= (1 << 63) else v for v in values], dtype=np.int64)
        return torch.from_numpy(arr).to(self._dev_t())

    def _h_msg(self, sub_sig, mp_t, g):
        """H_msg(R || PK.seed || PK.root || M'), padded to a whole number of blocks."""
        p = self.p
        n = p.n
        head = torch.cat(
            [
                sub_sig[:, :n],
                self._seed_row.expand(g, n),
                self._root_row.expand(g, n),
                mp_t,
            ],
            dim=1,
        )
        padded = _sponge.pad_torch(head, RATE, _sponge.DOMAIN_SHAKE)
        # the sponge squeezes in 8-byte words; m is not a multiple of 8, so take a word-aligned
        # prefix and slice (SHAKE's output is a stream, so this is exactly the same bytes)
        return _sponge.sponge_padded(padded, RATE, -(-p.m // 8) * 8)[:, : p.m]

    def verify(self, messages, signatures, ctx: bytes = b"", prehash=None):
        """External interface: pure (`prehash=None`) or pre-hash (`prehash=<name>`)."""
        if len(ctx) > 255:
            raise ValueError("context must be at most 255 bytes")
        sig = _as_u8(signatures)
        if sig.shape[0] == 0:
            return np.zeros((0,), dtype=bool)
        use = self.p.ph_name if prehash is True else prehash
        mps = []
        for m in _as_bytes(messages):
            if prehash is None:
                mps.append(b"\x00" + bytes([len(ctx)]) + ctx + m)
            else:
                mps.append(b"\x01" + bytes([len(ctx)]) + ctx + OID[use] + PH(use, m))
        return self.verify_mprime(mps, sig)


# ---------------- input plumbing ----------------


def _as_u8(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return np.ascontiguousarray(x, dtype=np.uint8)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.uint8, copy=False).reshape(x.shape[0], -1)
    return np.asarray([np.frombuffer(bytes(s), dtype=np.uint8) for s in x])


def _as_bytes(x) -> list[bytes]:
    if isinstance(x, np.ndarray):
        return [row.tobytes() for row in x]
    return [bytes(m) for m in x]


def verify_batch(ps, pk, messages, signatures, ctx: bytes = b"", prehash=None, device=None):
    return Verifier(ps, pk, device=device).verify(messages, signatures, ctx, prehash)
