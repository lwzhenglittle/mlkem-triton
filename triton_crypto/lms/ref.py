"""LMS / HSS reference implementation (RFC 8554), pure Python on top of `hashlib`.

Verification is the target (signing only exists so the tests have something to verify against a
fresh key); it is written to be obviously the algorithms from the RFC, not to be fast.
"""

from __future__ import annotations

import hashlib
import struct

from .params import LMS_TYPES, OTS_TYPES, LMSSet, M, N, get

D_PBLC = 0x8080
D_MESG = 0x8181
D_LEAF = 0x8282
D_INTR = 0x8383


def u32(v: int) -> bytes:
    return struct.pack(">I", v)


def u16(v: int) -> bytes:
    return struct.pack(">H", v)


def _h(*parts: bytes) -> bytes:
    h = hashlib.sha256()
    for part in parts:
        h.update(part)
    return h.digest()


def coef(s: bytes, i: int, w: int) -> int:
    """The i-th w-bit digit of `s`, most significant bits first (RFC 8554 §3.1.2)."""
    per = (i * w) // 8
    shift = 8 - w - ((i * w) % 8)
    return (s[per] >> shift) & ((1 << w) - 1)


def checksum(q: bytes, w: int, n: int, ls: int) -> int:
    total = 0
    for i in range((n * 8) // w):
        total += (1 << w) - 1 - coef(q, i, w)
    return (total << ls) & 0xFFFF


# ---------------- LM-OTS (Alg. 4b) ----------------
def ots_public_key(ots_type: int, ident: bytes, q: int, msg: bytes, sig: bytes) -> bytes | None:
    """Recompute `Kc` from an LM-OTS signature; `sig` includes the 4-byte type word."""
    if len(sig) < 4:
        return None
    if struct.unpack(">I", sig[:4])[0] != ots_type:
        return None
    w, p, ls = OTS_TYPES[ots_type]
    if len(sig) != 4 + N * (p + 1):
        return None
    c = sig[4 : 4 + N]
    ys = sig[4 + N :]
    qb = u32(q)
    q_digest = _h(ident, qb, u16(D_MESG), c, msg)
    codes = q_digest + u16(checksum(q_digest, w, N, ls))
    zs = []
    for i in range(p):
        a = coef(codes, i, w)
        tmp = ys[i * N : (i + 1) * N]
        for j in range(a, (1 << w) - 1):
            tmp = _h(ident, qb, u16(i), bytes([j]), tmp)
        zs.append(tmp)
    return _h(ident, qb, u16(D_PBLC), *zs)


# ---------------- key generation from a seed (Appendix A) ----------------
def ots_private(ident: bytes, q: int, seed: bytes, p: int) -> list[bytes]:
    """`x_q[i] = H(I || u32(q) || u16(i) || 0xff || SEED)` — Appendix A."""
    return [_h(ident, u32(q), u16(i), b"\xff", seed) for i in range(p)]


def ots_public_from_private(ident: bytes, q: int, xs: list[bytes], w: int) -> bytes:
    qb = u32(q)
    zs = []
    for i, x in enumerate(xs):
        tmp = x
        for j in range((1 << w) - 1):
            tmp = _h(ident, qb, u16(i), bytes([j]), tmp)
        zs.append(tmp)
    return _h(ident, qb, u16(D_PBLC), *zs)


def leaf_from_seed(ident: bytes, q: int, seed: bytes, s: LMSSet) -> bytes:
    return ots_public_from_private(ident, q, ots_private(ident, q, seed, s.p), s.w)


def keygen(seed: bytes, ident: bytes, s: LMSSet) -> tuple[bytes, list[bytes]]:
    """Build the tree from `seed` (Appendix A); returns (public key, leaves)."""
    base = 1 << s.h
    nodes = {}
    leaves = []
    for q in range(base):
        kc = leaf_from_seed(ident, q, seed, s)
        leaves.append(kc)
        nodes[base + q] = _h(ident, u32(base + q), u16(D_LEAF), kc)
    for node in range(base - 1, 0, -1):
        nodes[node] = _h(ident, u32(node), u16(D_INTR), nodes[2 * node], nodes[2 * node + 1])
    return u32(s.lms_type) + u32(s.ots_type) + ident + nodes[1], leaves


def lms_sign(pub: bytes, seed: bytes, q: int, msg: bytes, s: LMSSet, alpha: bytes) -> bytes:
    """Sign one message at leaf `q`. `alpha` is the LM-OTS randomiser C (32 bytes)."""
    ident = pub[8:24]
    base = 1 << s.h
    nodes = {}
    for j in range(base):
        nodes[base + j] = _h(ident, u32(base + j), u16(D_LEAF), leaf_from_seed(ident, j, seed, s))
    for node in range(base - 1, 0, -1):
        nodes[node] = _h(ident, u32(node), u16(D_INTR), nodes[2 * node], nodes[2 * node + 1])
    xs = ots_private(ident, q, seed, s.p)
    qb = u32(q)
    digits = _h(ident, qb, u16(D_MESG), alpha, msg) + b"\x00\x00"
    digits = digits[:-2] + u16(checksum(digits[:-2], s.w, N, s.ls))
    ys = []
    for i in range(s.p):
        a = coef(digits, i, s.w)
        tmp = xs[i]
        for j in range(a):
            tmp = _h(ident, qb, u16(i), bytes([j]), tmp)
        ys.append(tmp)
    # path[i] is the sibling of the node on the way up: ((base+q) >> i) ^ 1
    path = b"".join(nodes[((base + q) >> i) ^ 1] for i in range(s.h))
    return u32(q) + u32(s.ots_type) + alpha + b"".join(ys) + u32(s.lms_type) + path


def hss_keygen(seeds: list[bytes], idents: list[bytes], sets: list[LMSSet]) -> bytes:
    """HSS public key: `u32(L) || pub[0]`."""
    pub0, _ = keygen(seeds[0], idents[0], sets[0])
    return u32(len(sets)) + pub0


def hss_sign(
    seeds: list[bytes],
    idents: list[bytes],
    sets: list[LMSSet],
    qs: list[int],
    msg: bytes,
    alphas: list[bytes],
) -> bytes:
    """Sign with a fixed hierarchy (levels given by `sets`), leaf `qs[i]` at each level."""
    if len(sets) == 1:
        pub0, _ = keygen(seeds[0], idents[0], sets[0])
        return u32(0) + lms_sign(pub0, seeds[0], qs[0], msg, sets[0], alphas[0])
    pubs = [keygen(seeds[i], idents[i], sets[i])[0] for i in range(len(sets))]
    parts = []
    for i in range(len(sets) - 1):
        parts.append(lms_sign(pubs[i], seeds[i], qs[i], pubs[i + 1], sets[i], alphas[i]))
        parts.append(pubs[i + 1])
    parts.append(lms_sign(pubs[-1], seeds[-1], qs[-1], msg, sets[-1], alphas[-1]))
    return u32(len(sets) - 1) + b"".join(parts)


# ---------------- LMS (Alg. 6a) ----------------
def lms_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) < 8:
        return False
    lms_type, ots_type = struct.unpack(">II", pub[:8])
    if lms_type not in _LMS_H or ots_type not in _OTS_T:
        return False
    s = get(lms_type, ots_type)
    if len(pub) != s.pk_size:
        return False
    ident, t1 = pub[8:24], pub[24 : 24 + M]
    if len(sig) < 8:
        return False
    q = struct.unpack(">I", sig[:4])[0]
    otssigtype = struct.unpack(">I", sig[4:8])[0]
    if otssigtype != ots_type:
        return False
    # q(4) | otssigtype(4) | C(n) | y[p*n] | sigtype(4) | path[h*m]: the LM-OTS signature *with*
    # its own type word runs from byte 4 to 8 + n*(p+1), i.e. 4 + ots_sig_size bytes.
    end = 4 + s.ots_sig_size
    if len(sig) < end + 4:
        return False
    ots_sig = sig[4:end]
    sigtype = struct.unpack(">I", sig[end : end + 4])[0]
    if sigtype != lms_type:
        return False
    if q >= (1 << s.h) or len(sig) != s.sig_size:
        return False
    path = sig[end + 4 :]
    kc = ots_public_key(ots_type, ident, q, msg, ots_sig)
    if kc is None:
        return False
    node = (1 << s.h) + q
    tmp = _h(ident, u32(node), u16(D_LEAF), kc)
    for i in range(s.h):
        parent = node // 2
        auth = path[i * M : (i + 1) * M]
        if node & 1:
            tmp = _h(ident, u32(parent), u16(D_INTR), auth, tmp)
        else:
            tmp = _h(ident, u32(parent), u16(D_INTR), tmp, auth)
        node = parent
    return tmp == t1


# ---------------- HSS (RFC 8554 §6.3) ----------------
def hss_verify(pub: bytes, msg: bytes, sig: bytes) -> bool:
    if len(pub) < 4:
        return False
    levels = struct.unpack(">I", pub[:4])[0]
    if levels == 0 or levels > 8:
        return False
    if len(sig) < 4:
        return False
    nspk = struct.unpack(">I", sig[:4])[0]
    if nspk + 1 != levels:
        return False
    off = 4
    cur = pub[4:]
    for _ in range(nspk):
        if len(cur) < 8:
            return False
        lms_type, ots_type = struct.unpack(">II", cur[:8])
        s = get(lms_type, ots_type)
        slen = s.sig_size
        if off + slen + s.pk_size > len(sig):
            return False
        next_pub = sig[off + slen : off + slen + s.pk_size]
        if not lms_verify(cur, next_pub, sig[off : off + slen]):
            return False
        cur = next_pub
        off += slen + s.pk_size
    return lms_verify(cur, msg, sig[off:])


_LMS_H = LMS_TYPES
_OTS_T = {k: v[0] for k, v in OTS_TYPES.items()}
