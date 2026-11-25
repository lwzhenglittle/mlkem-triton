"""FIPS 205 (SLH-DSA) reference implementation — verification side, in plain Python.

This is the oracle the device code is checked against: it follows the algorithms of FIPS 205
by number (Alg. 1–2 toByte/toInt, Alg. 4 base_2b, Alg. 5 WOTS+ checksum, Alg. 8 chain,
Alg. 9/10 wots_pk_from_sig, Alg. 13 xmss_pk_from_sig, Alg. 14 ht_verify, Alg. 15 fors_pk_from_sig,
Alg. 20 slh_verify), and both instantiations of §11 (SHAKE and SHA2) because the ACVP vectors
cover both.

Pure Python on purpose: it is slow (a 128f verification is ~7k SHAKE256 calls) and never on the
hot path, but it is short enough to read against the standard line by line.

Address layout (`ADRS`, 32 bytes, big-endian words), FIPS 205 §11.2:

    bytes  0..4   layer address
    bytes  4..8   tree address, high word (always zero)
    bytes  8..16  tree address, low 64 bits
    bytes 16..20  type
    bytes 20..24  key pair address        (type-dependent)
    bytes 24..28  chain address / tree height
    bytes 28..32  hash address / tree index
"""

from __future__ import annotations

import hashlib

from .params import SLHParams, get

__all__ = ["OID", "PH", "SLH", "VERIFIERS", "Adrs", "base_2b", "to_byte", "to_int"]

# ADRS types (FIPS 205 §11.2)
WOTS_HASH = 0
WOTS_PK = 1
TREE = 2
FORS_TREE = 3
FORS_ROOTS = 4
WOTS_PRF = 5
FORS_PRF = 6

# Pre-hash function identifiers (FIPS 205 Table 3): DER-encoded OID, hash constructor, digest size.
# The OIDs are the standard NIST ones, `2.16.840.1.101.3.4.2.n`, written out as DER.
_OID = bytes.fromhex("06096086480165030402")  # 2.16.840.1.101.3.4.2.<last arc>
OID: dict[str, bytes] = {
    "SHA2-224": _OID + b"\x04",
    "SHA2-256": _OID + b"\x01",
    "SHA2-384": _OID + b"\x02",
    "SHA2-512": _OID + b"\x03",
    "SHA2-512/224": _OID + b"\x05",
    "SHA2-512/256": _OID + b"\x06",
    "SHA3-224": _OID + b"\x07",
    "SHA3-256": _OID + b"\x08",
    "SHA3-384": _OID + b"\x09",
    "SHA3-512": _OID + b"\x0a",
    "SHAKE128": _OID + b"\x0b",
    "SHAKE256": _OID + b"\x0c",
}


_PH_IMPL: dict[str, tuple[str, int]] = {  # name -> (hashlib constructor, XOF output length)
    "SHA2-224": ("sha224", 0),
    "SHA2-256": ("sha256", 0),
    "SHA2-384": ("sha384", 0),
    "SHA2-512": ("sha512", 0),
    "SHA2-512/224": ("sha512_224", 0),
    "SHA2-512/256": ("sha512_256", 0),
    "SHA3-224": ("sha3_224", 0),
    "SHA3-256": ("sha3_256", 0),
    "SHA3-384": ("sha3_384", 0),
    "SHA3-512": ("sha3_512", 0),
    "SHAKE128": ("shake_128", 32),
    "SHAKE256": ("shake_256", 64),
}


def PH(name: str, msg: bytes) -> bytes:
    """The pre-hash functions of FIPS 205 Table 3."""
    fn, out = _PH_IMPL[name]
    h = getattr(hashlib, fn, None)
    h = h(msg) if h is not None else hashlib.new(fn, msg)
    return h.digest() if not out else h.digest(out)


def to_int(x: bytes) -> int:
    """Alg. 1"""
    return int.from_bytes(x, "big") if x else 0


def to_byte(x: int, n: int) -> bytes:
    """Alg. 2"""
    return x.to_bytes(n, "big") if n else b""


def base_2b(x: bytes, b: int, out_len: int) -> list[int]:
    """Alg. 4 — the first `out_len` base-2^b digits of x, big-endian bit order."""
    total = to_int(x)
    bits = len(x) * 8
    return [(total >> (bits - (i + 1) * b)) & ((1 << b) - 1) for i in range(out_len)]


class Adrs:
    """A mutable 32-byte ADRS with the setters of FIPS 205 §11.2."""

    __slots__ = ("b",)

    def __init__(self) -> None:
        self.b = bytearray(32)

    def copy(self) -> Adrs:
        a = Adrs()
        a.b = bytearray(self.b)
        return a

    def _set(self, off: int, v: int, size: int = 4) -> None:
        self.b[off : off + size] = (v & ((1 << (8 * size)) - 1)).to_bytes(size, "big")

    def set_layer(self, v: int) -> None:
        self._set(0, v)

    def set_tree(self, v: int) -> None:
        self._set(8, v >> 32)
        self._set(12, v & 0xFFFFFFFF)

    def set_type(self, t: int) -> None:
        self._set(16, t)

    def set_type_clear(self, t: int) -> None:
        """setTypeAndClear: type = t, final 12 bytes zero."""
        self._set(16, t)
        self.b[20:32] = bytes(12)

    def set_type_clear_keep_kp(self, t: int) -> None:
        """setTypeAndClear, except the key pair field survives (WOTS_PK / FORS_ROOTS)."""
        self._set(16, t)
        self.b[24:32] = bytes(8)

    def set_kp(self, v: int) -> None:
        self._set(20, v)

    def set_chain(self, v: int) -> None:
        self._set(24, v)

    def set_hash(self, v: int) -> None:
        self._set(28, v)

    def set_tree_height(self, v: int) -> None:
        self._set(24, v)

    def set_tree_index(self, v: int) -> None:
        self._set(28, v)

    @property
    def comp(self) -> bytes:
        """The 22-byte compressed address used by the SHA-2 instantiation."""
        return bytes(self.b[3:4] + self.b[8:16] + self.b[19:32])

    def __bytes__(self) -> bytes:
        return bytes(self.b)


class SLH:
    """Verify SLH-DSA signatures for one parameter set."""

    def __init__(self, ps: str | SLHParams):
        self.p = get(ps) if isinstance(ps, str) else ps
        if self.p.n == 16:
            self._xof = hashlib.shake_256
        else:
            self._xof = hashlib.shake_256  # both families use SHAKE for the hash family below

    # ---------------- hash instantiations (FIPS 205 §11) ----------------
    def _h(self, data: bytes, out: int) -> bytes:
        """SHAKE256 or SHA-256/512 according to the parameter set's family."""
        if self.p.is_shake:
            return hashlib.shake_256(data).digest(out)
        fn = hashlib.sha256 if self.p.n == 16 else hashlib.sha512
        return fn(data).digest()[:out]

    def _mgf1(self, seed: bytes, out: int) -> bytes:
        fn = hashlib.sha256 if self.p.n == 16 else hashlib.sha512
        hlen = fn().digest_size
        return b"".join(
            fn(seed + (i // hlen).to_bytes(4, "big")).digest() for i in range(0, out, hlen)
        )[:out]

    def _x(self, seed: bytes, adrs: Adrs, data: bytes, out: int, wide: bool = False) -> bytes:
        """The thash core: `H(PK.seed || [pad] || ADRSc || data)` truncated to `out` bytes.

        FIPS 205's SHA-2 instantiation is deliberately asymmetric: PRF and F always use
        SHA-256 (the seed is padded to a 64-byte block), while H and T_l switch to SHA-512 for
        the 192/256-bit sets (128-byte block). That asymmetry is not cosmetic — using SHA-512
        everywhere agrees with liboqs only for the 128-bit sets.
        """
        if self.p.is_shake:
            return hashlib.shake_256(bytes(seed) + bytes(adrs) + data).digest(out)
        if wide:
            return hashlib.sha512(bytes(seed) + bytes(128 - self.p.n) + adrs.comp + data).digest()[
                :out
            ]
        return hashlib.sha256(bytes(seed) + bytes(64 - self.p.n) + adrs.comp + data).digest()[:out]

    @property
    def _wide(self) -> bool:
        """Whether H/T_l use SHA-512 for this set."""
        return (not self.p.is_shake) and self.p.n > 16

    def h_msg(self, r: bytes, seed: bytes, root: bytes, mprime: bytes) -> bytes:
        p = self.p
        if p.is_shake:
            return hashlib.shake_256(r + seed + root + mprime).digest(p.m)
        inner = self._h(r + seed + root + mprime, 32 if p.n == 16 else 64)
        return self._mgf1(r + seed + inner, p.m)

    def f(self, seed: bytes, adrs: Adrs, m1: bytes) -> bytes:
        return self._x(seed, adrs, m1, self.p.n)

    def h(self, seed: bytes, adrs: Adrs, m1: bytes, m2: bytes) -> bytes:
        return self._x(seed, adrs, m1 + m2, self.p.n, wide=self._wide)

    def t(self, seed: bytes, adrs: Adrs, m: bytes) -> bytes:
        return self._x(seed, adrs, m, self.p.n, wide=self._wide)

    def prf(self, seed: bytes, adrs: Adrs, sk_seed: bytes) -> bytes:
        return self._x(seed, adrs, sk_seed, self.p.n)

    # ---------------- WOTS+ (Alg. 5, 8, 9, 10) ----------------
    def wots_digits(self, msg: bytes) -> list[int]:
        p = self.p
        digits = base_2b(msg, p.lg_w, p.len1)
        csum = sum(p.w - 1 - d for d in digits)
        csum <<= (8 - (p.len2 * p.lg_w) % 8) % 8
        digits += base_2b(to_byte(csum, (p.len2 * p.lg_w + 7) // 8), p.lg_w, p.len2)
        return digits

    def chain(self, x: bytes, i: int, s: int, seed: bytes, adrs: Adrs) -> bytes:
        """Alg. 8 — `s` applications of F starting at chain index `i`."""
        for j in range(i, i + s):
            adrs.set_hash(j)
            x = self.f(seed, adrs, x)
        return x

    def wots_pk_from_sig(self, sig, msg, seed, adrs, kp) -> bytes:
        """Alg. 9/10 — WOTS+ public key recovered from a signature."""
        p = self.p
        digits = self.wots_digits(msg)
        adrs.set_type_clear(WOTS_HASH)
        adrs.set_kp(kp)
        parts = []
        for i in range(p.wots_len):
            adrs.set_chain(i)
            adrs.set_hash(0)
            parts.append(self.chain(sig[i], digits[i], p.w - 1 - digits[i], seed, adrs))
        adrs.set_type_clear_keep_kp(WOTS_PK)
        return self.t(seed, adrs, b"".join(parts))

    # ---------------- XMSS (Alg. 13) ----------------
    def xmss_pk_from_sig(self, idx, sig, msg, seed, adrs) -> bytes:
        """Alg. 13 — one XMSS root from a WOTS+ signature and an authentication path."""
        p = self.p
        adrs.set_type_clear(WOTS_HASH)
        adrs.set_kp(idx)
        root = self.wots_pk_from_sig(sig[: p.wots_len], msg, seed, adrs, idx)
        auth = sig[p.wots_len :]
        adrs.set_type_clear(TREE)
        for k in range(p.hp):
            adrs.set_tree_height(k + 1)
            adrs.set_tree_index(idx >> (k + 1))
            if ((idx >> k) & 1) == 0:
                root = self.h(seed, adrs, root, auth[k])
            else:
                root = self.h(seed, adrs, auth[k], root)
        return root

    # ---------------- hypertree (Alg. 14) ----------------
    def ht_verify(self, root, sig_ht, seed, idx_tree, idx_leaf, adrs) -> bytes:
        p = self.p
        adrs.b = bytearray(32)  # Alg. 14 starts from a cleared address
        adrs.set_tree(idx_tree)
        root = self.xmss_pk_from_sig(idx_leaf, sig_ht[0], root, seed, adrs)
        for j in range(1, p.d):
            idx_leaf = idx_tree & ((1 << p.hp) - 1)
            idx_tree >>= p.hp
            adrs.set_layer(j)
            adrs.set_tree(idx_tree)
            root = self.xmss_pk_from_sig(idx_leaf, sig_ht[j], root, seed, adrs)
        return root

    # ---------------- FORS (Alg. 15) ----------------
    def fors_pk_from_sig(self, sig_fors, md, seed, adrs) -> bytes:
        p = self.p
        indices = base_2b(md, p.a, p.k)
        roots = []
        for i in range(p.k):
            adrs.set_tree_height(0)
            idx = (i << p.a) + indices[i]
            adrs.set_tree_index(idx)
            node = self.f(seed, adrs, sig_fors[i][0])
            for j in range(p.a):
                adrs.set_tree_height(j + 1)
                adrs.set_tree_index(idx >> (j + 1))
                if ((indices[i] >> j) & 1) == 0:
                    node = self.h(seed, adrs, node, sig_fors[i][1 + j])
                else:
                    node = self.h(seed, adrs, sig_fors[i][1 + j], node)
            roots.append(node)
        adrs.set_type_clear_keep_kp(FORS_ROOTS)
        return self.t(seed, adrs, b"".join(roots))

    # ---------------- key generation (Alg. 16, 17) — used by tests only ----------------
    def wots_pkgen(self, seed, sk_seed, adrs, kp) -> bytes:
        """Alg. 6 — WOTS+ public key from the secret seed."""
        p = self.p
        adrs.set_kp(kp)
        parts = []
        for i in range(p.wots_len):
            adrs.set_type(WOTS_PRF)  # must be re-set: the chain below switches it to WOTS_HASH
            adrs.set_chain(i)
            adrs.set_hash(0)
            sk_i = self.prf(seed, adrs, sk_seed)
            adrs.set_type(WOTS_HASH)
            parts.append(self.chain(sk_i, 0, p.w - 1, seed, adrs))
        adrs.set_type_clear_keep_kp(WOTS_PK)
        return self.t(seed, adrs, b"".join(parts))

    def xmss_node(self, sk_seed, seed, i, z, adrs) -> bytes:
        """Alg. 16 — root of the XMSS subtree of height z at leaf i (recursive)."""
        if z == 0:
            return self.wots_pkgen(seed, sk_seed, adrs, i)
        left = self.xmss_node(sk_seed, seed, 2 * i, z - 1, adrs)
        right = self.xmss_node(sk_seed, seed, 2 * i + 1, z - 1, adrs)
        adrs.set_type_clear(TREE)
        adrs.set_tree_height(z)
        adrs.set_tree_index(i)
        return self.h(seed, adrs, left, right)

    def keygen_internal(self, seed: bytes) -> tuple[bytes, bytes]:
        """Alg. 17 — `seed` is 3n bytes (SK.seed ‖ SK.prf ‖ PK.seed)."""
        p = self.p
        if len(seed) != 3 * p.n:
            raise ValueError(f"seed must be {3 * p.n} bytes")
        sk_seed, sk_prf, pk_seed = seed[: p.n], seed[p.n : 2 * p.n], seed[2 * p.n :]
        # The single tree keygen builds is the *top* one: layer d-1, tree address 0.
        adrs = Adrs()
        adrs.set_layer(p.d - 1)
        root = self.xmss_node(sk_seed, pk_seed, 0, p.hp, adrs)
        pk = pk_seed + root
        sk = sk_seed + sk_prf + pk_seed + root
        return sk, pk

    # ---------------- message formatting (FIPS 205 §10.1, Table 3) ----------------
    @staticmethod
    def format_message(msg: bytes, ctx: bytes = b"", prehash: str | None = None) -> bytes:
        """M' for the external interface. `prehash=None` is "pure"; `"internal"` means M' given."""
        if prehash == "internal":
            return msg
        if len(ctx) > 255:
            raise ValueError("context must be at most 255 bytes")
        head = b"\x00" if prehash is None else b"\x01"
        out = head + bytes([len(ctx)]) + ctx
        if prehash is None:
            return out + msg
        # FIPS 205 §10.1: in pre-hash mode M' ends at PH(M) — the message itself is *not*
        # appended (unlike the pure mode above).
        return out + OID[prehash] + PH(prehash, msg)

    # ---------------- Alg. 20 ----------------
    def verify_mprime(self, mprime: bytes, sig: bytes, pk: bytes) -> bool:
        """Verify a signature against an already-formatted M' (internal interface)."""
        p = self.p
        if len(pk) != p.pk_size:
            raise ValueError(f"public key must be {p.pk_size} bytes, got {len(pk)}")
        if len(sig) != p.sig_size:
            return False
        seed, root = pk[: p.n], pk[p.n :]
        r = sig[: p.n]
        digest = self.h_msg(r, seed, root, mprime)

        o = p.md_bytes
        md = digest[:o]
        idx_tree = to_int(digest[o : o + p.tree_bytes]) & ((1 << (p.h - p.hp)) - 1)
        idx_leaf = to_int(digest[o + p.tree_bytes :]) & ((1 << p.hp) - 1)

        body = sig[p.n :]
        n = p.n
        fors = [
            [body[(i * (p.a + 1) + j) * n : (i * (p.a + 1) + j + 1) * n] for j in range(p.a + 1)]
            for i in range(p.k)
        ]
        off = p.fors_sig_size
        ht = []
        step = p.xmss_sig_size
        for j in range(p.d):
            chunk = body[off + j * step : off + (j + 1) * step]
            ht.append([chunk[i * n : (i + 1) * n] for i in range(p.wots_len + p.hp)])

        adrs = Adrs()
        adrs.set_tree(idx_tree)
        adrs.set_type_clear(FORS_TREE)
        adrs.set_kp(idx_leaf)
        node = self.fors_pk_from_sig(fors, md, seed, adrs)
        node = self.ht_verify(node, ht, seed, idx_tree, idx_leaf, adrs)
        return node == root

    def verify(self, msg: bytes, sig: bytes, pk: bytes, ctx: bytes = b"", prehash=None) -> bool:
        """Verify with the external interface: `prehash=None` (pure), or a hash name."""
        return self.verify_mprime(self.format_message(msg, ctx, prehash), sig, pk)


VERIFIERS = {name: SLH(name) for name in get.__globals__["HASHES"]}
