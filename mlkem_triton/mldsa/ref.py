"""ML-DSA (FIPS 204) in numpy — the reference this package tests against.

Complete, deliberate, and slow: key generation (Alg 6), signing with the rejection loop
(Alg 7) and verification (Alg 3), plus the packing (Alg 20-32), sampling (Alg 29-30),
rounding (Alg 35-39) and message formatting (§5.2) they are built from.

Why it exists even though the GPU only verifies:

- it is the oracle for the GPU kernel tests — every intermediate (Â, ĉ, w′, hints, the
  packed w1) can be compared step by step, which is the only practical way to debug a
  data-dependent failure;
- it makes the random-key round-trip tests independent of any third-party package;
- `sign` produces the batch corpora the benchmark needs (one long-lived key, many
  signatures), and is checked byte-for-byte against NIST's ACVP sigGen vectors.

The transform and the sponge are **not** reimplemented here: this module calls
`primitives.ntt` (q = 8380417 instance) and `primitives.sponge`, exactly like the GPU
path does, so a bug in a primitive shows up in both and cannot cancel out.
"""

from __future__ import annotations

import hashlib
import os
from functools import cached_property

import numpy as np

from ..primitives.ntt import DILITHIUM, intt_np, ntt_np
from ..primitives.sponge import SHAKE256
from .params import PARAMS, D, DSAParams, N, Q

__all__ = ["MLDSA", "PREHASH", "Ref", "pack_bits", "unpack_bits"]

D_OFFSET = 1 << D  # 8192


# ---------------------------------------------------------------- bit packing


def pack_bits(vals, bits: int) -> bytes:
    """BitPack: little-endian bit packing of a coefficient vector."""
    vals = np.asarray(vals, dtype=np.uint64).reshape(-1)
    if np.any(vals >= (1 << bits)):
        raise ValueError(f"coefficient out of range for {bits} bits")
    per = ((vals[:, None] >> np.arange(bits, dtype=np.uint64)) & 1).astype(np.uint8)
    return np.packbits(per.reshape(-1), bitorder="little").tobytes()


def unpack_bits(data, bits: int, count: int) -> np.ndarray:
    """BitUnpack: the inverse of `pack_bits` (returns int64)."""
    raw = np.frombuffer(data, dtype=np.uint8)
    per = np.unpackbits(raw, bitorder="little")[: count * bits].reshape(count, bits)
    return (per.astype(np.int64) << np.arange(bits)).sum(axis=1)


# ---------------------------------------------------------------- HashML-DSA pre-hashes

# ACME-style AlgorithmIdentifier OIDs for the pre-hash interface (FIPS 204 §5.4).
_OID = bytes.fromhex("06096086480165030402")
PREHASH = {
    "SHA2-224": (_OID + b"\x04", "sha224", None),
    "SHA2-256": (_OID + b"\x01", "sha256", None),
    "SHA2-384": (_OID + b"\x02", "sha384", None),
    "SHA2-512": (_OID + b"\x03", "sha512", None),
    "SHA2-512/224": (_OID + b"\x05", "sha512_224", None),
    "SHA2-512/256": (_OID + b"\x06", "sha512_256", None),
    "SHA3-224": (_OID + b"\x07", "sha3_224", None),
    "SHA3-256": (_OID + b"\x08", "sha3_256", None),
    "SHA3-384": (_OID + b"\x09", "sha3_384", None),
    "SHA3-512": (_OID + b"\x0a", "sha3_512", None),
    "SHAKE-128": (_OID + b"\x0b", "shake_128", 32),
    "SHAKE-256": (_OID + b"\x0c", "shake_256", 64),
}


def _prehash(name: str, msg: bytes) -> tuple[bytes, bytes]:
    """(OID, Φ) for a HashML-DSA pre-hash algorithm name."""
    try:
        oid, fn, outlen = PREHASH[name.upper()]
    except KeyError:
        raise ValueError(f"unknown pre-hash {name!r}; known: {sorted(PREHASH)}") from None
    h = hashlib.new(fn, msg)  # sha512_224 has no attribute form
    return oid, (h.digest(outlen) if outlen else h.digest())


class MLDSA:
    """One ML-DSA parameter set: every algorithm of FIPS 204 in numpy."""

    def __init__(self, ps: str):
        if ps not in PARAMS:
            raise KeyError(f"unknown parameter set {ps}; known: {sorted(PARAMS)}")
        self.ps = ps
        self.p: DSAParams = PARAMS[ps]
        self.domain = DILITHIUM

    # ---- shorthands ----
    @property
    def k(self) -> int:
        return self.p.k

    @property
    def l(self) -> int:
        return self.p.l

    def __repr__(self) -> str:
        return f"MLDSA({self.ps!r})"

    # ---------------------------------------------------------------- hashing

    @staticmethod
    def H(data: bytes, outlen: int) -> bytes:
        """H = SHAKE256."""
        return SHAKE256.digest_bytes(data, outlen)

    @cached_property
    def _tetas(self) -> tuple[int, int]:
        """Boundaries for RejBoundedPoly: (η=2 -> 15, η=4 -> 9)."""
        return (15, 5) if self.p.eta == 2 else (9, 9)

    # ---------------------------------------------------------------- sampling

    def expand_a(self, rho: bytes) -> np.ndarray:
        """Â = ExpandA(ρ): [k, l, 256] in the NTT domain (FIPS 204 Alg. 32)."""
        A = np.empty((self.k, self.l, N), dtype=np.int64)
        for r in range(self.k):
            for s in range(self.l):
                seed = rho + bytes([s, r])
                A[r, s] = self._rej_ntt_poly(seed)
        return A

    @staticmethod
    def _rej_ntt_poly(seed: bytes) -> np.ndarray:
        """RejNTTPoly: SHAKE128 stream, one 23-bit candidate per 3 bytes (Alg. 30).

        Note this is *not* ML-KEM's SampleNTT: ML-KEM unpacks 3 bytes into **two**
        12-bit candidates, ML-DSA into one 23-bit value (`& 0x7FFFFF`) < q.
        """
        h = hashlib.shake_128(seed)
        nbytes = 168
        while True:
            buf = np.frombuffer(h.digest(nbytes), dtype=np.uint8)
            ntrip = nbytes // 3
            tri = buf[: ntrip * 3].reshape(-1, 3).astype(np.int64)
            cand = tri[:, 0] | (tri[:, 1] << 8) | (tri[:, 2] << 16)
            ok = cand[(cand & 0x7FFFFF) < Q] & 0x7FFFFF
            if ok.size >= N:
                return ok[:N]
            nbytes += 168

    def rej_bounded_poly(self, seed: bytes) -> np.ndarray:
        """RejBoundedPoly: SHAKE256 stream of nibbles -> coefficients in [-η, η]."""
        limit, mod = self._tetas
        h = hashlib.shake_256(seed)
        nbytes = 168
        while True:
            buf = np.frombuffer(h.digest(nbytes), dtype=np.uint8)
            nib = np.stack([buf & 0x0F, buf >> 4], axis=1).reshape(-1)
            vals = self.p.eta - (nib[nib < limit].astype(np.int64) % mod)
            if vals.size >= N:
                return vals[:N]
            nbytes += 168

    def expand_s(self, rhoprime: bytes, nonce: int) -> np.ndarray:
        """s = ExpandS(ρ′, nonce): coefficients in [-η, η]."""
        return self.rej_bounded_poly(rhoprime + int(nonce).to_bytes(2, "little"))

    def expand_mask(self, rhoprime: bytes, kappa: int) -> np.ndarray:
        """y = ExpandMask(ρ′, κ): [l, 256] in [−γ1+1, γ1] (FIPS 204 Alg. 34)."""
        y = np.empty((self.l, N), dtype=np.int64)
        for r in range(self.l):
            seed = rhoprime + int(kappa + r).to_bytes(2, "little")
            raw = SHAKE256.digest_bytes(seed, self.p.polyz_bytes)
            y[r] = self.p.gamma1 - unpack_bits(raw, self.p.z_bits, N)
        return y

    def sample_in_ball(self, ctilde: bytes) -> np.ndarray:
        """c = SampleInBall(c̃) ∈ {−1,0,1}^256 with exactly τ non-zeros (Alg. 29)."""
        p = self.p
        stream = hashlib.shake_256(ctilde)
        buf = stream.digest(136)
        signs = int.from_bytes(buf[:8], "little")
        c = np.zeros(N, dtype=np.int64)
        pos = 8
        for i in range(N - p.tau, N):
            while True:
                if pos >= len(buf):
                    buf = stream.digest(len(buf) + 136)
                j = buf[pos]
                pos += 1
                if j <= i:
                    break
            c[i] = c[j]
            c[j] = 1 - 2 * (signs & 1)
            signs >>= 1
        return c

    # ---------------------------------------------------------------- rounding

    @staticmethod
    def power2round(r):
        """Power2Round: r → (r1, r0) with r = r1·2^d + r0, r0 ∈ (−2^{d−1}, 2^{d−1}]."""
        r = np.asarray(r, dtype=np.int64) % Q
        r0 = r % D_OFFSET
        r0 = np.where(r0 > D_OFFSET // 2, r0 - D_OFFSET, r0)
        return (r - r0) // D_OFFSET, r0

    def decompose(self, r):
        """Decompose_γ2: r → (r1, r0), r = r1·2γ2 + r0 (FIPS 204 Alg. 36)."""
        g2 = self.p.gamma2
        two = 2 * g2
        r = np.asarray(r, dtype=np.int64) % Q
        r0 = r % two
        r0 = np.where(r0 > g2, r0 - two, r0)
        r1 = (r - r0) // two
        special = (r - r0) == Q - 1
        r1 = np.where(special, 0, r1)
        r0 = np.where(special, r0 - 1, r0)
        return r1, r0

    def high_bits(self, r):
        return self.decompose(r)[0]

    def low_bits(self, r):
        return self.decompose(r)[1]

    def use_hint(self, h, r):
        """UseHint_γ2 (FIPS 204 Alg. 37)."""
        m = self.p.m
        r1, r0 = self.decompose(r)
        up = (r1 + 1) % m
        down = (r1 - 1) % m
        hint_up = np.where(r0 > 0, up, down)
        return np.where(np.asarray(h) != 0, hint_up, r1)

    def make_hint(self, z, r):
        """MakeHint_γ2 (FIPS 204 Alg. 39), in the reference's high-bits form."""
        return (self.high_bits(r) != self.high_bits(np.asarray(r) + np.asarray(z))).astype(np.int64)

    # ---------------------------------------------------------------- encodings

    def pack_z(self, z) -> bytes:
        """BitPack of z as γ1 − z (so that unpacking is γ1 − value)."""
        p = self.p
        vals = (p.gamma1 - np.asarray(z, dtype=np.int64)) % (2 * p.gamma1)
        return pack_bits(vals, p.z_bits)

    def unpack_z(self, data) -> np.ndarray:
        p = self.p
        return p.gamma1 - unpack_bits(data, p.z_bits, N)

    def pack_w1(self, w1) -> bytes:
        return pack_bits(np.asarray(w1, dtype=np.int64), self.p.w1_bits)

    def unpack_w1(self, data) -> np.ndarray:
        return unpack_bits(data, self.p.w1_bits, N)

    def pack_eta(self, s) -> bytes:
        p = self.p
        vals = (p.eta - np.asarray(s, dtype=np.int64)) % (2 * p.eta + 1)
        return pack_bits(vals, p.polyeta_bytes * 8 // N)

    def unpack_eta(self, data) -> np.ndarray:
        p = self.p
        bits = p.polyeta_bytes * 8 // N
        return p.eta - unpack_bits(data, bits, N)

    def pack_t1(self, t1) -> bytes:
        return pack_bits(np.asarray(t1, dtype=np.int64), 10)

    def unpack_t1(self, data) -> np.ndarray:
        return unpack_bits(data, 10, N)

    def pack_t0(self, t0) -> bytes:
        # t0 ∈ (−2^{d−1}, 2^{d−1}] -> 2^{d−1} − t0 ∈ [0, 2^d), i.e. **d** bits.
        vals = (D_OFFSET // 2 - np.asarray(t0, dtype=np.int64)) % D_OFFSET
        return pack_bits(vals, D)

    def unpack_t0(self, data) -> np.ndarray:
        return (D_OFFSET // 2) - unpack_bits(data, D, N)

    def pack_h(self, h) -> bytes:
        """HintBitPack (FIPS 204 Alg. 21): τ_max ≤ ω index slots then k counts."""
        p = self.p
        h = np.asarray(h).reshape(self.k, N)
        out = bytearray(p.omega + self.k)
        pos = 0
        for i in range(self.k):
            for j in np.flatnonzero(h[i]):
                out[pos] = int(j)
                pos += 1
            if pos > p.omega:
                raise ValueError("too many hints to encode")
            out[p.omega + i] = pos
        return bytes(out)

    def unpack_h(self, data) -> np.ndarray:
        """HintBitUnpack (FIPS 204 Alg. 20/21); raises on a malformed hint block."""
        p = self.p
        data = np.frombuffer(data, dtype=np.uint8)
        if data.size != p.omega + self.k:
            raise ValueError("wrong hint block length")
        h = np.zeros((self.k, N), dtype=np.int64)
        prev = 0
        for i in range(self.k):
            cnt = int(data[p.omega + i])
            if cnt < prev or cnt > p.omega:
                raise ValueError("hint counts not monotone / too large")
            if cnt > prev:
                idx = data[prev:cnt].astype(np.int64)
                if np.any(idx[1:] <= idx[:-1]):
                    raise ValueError("hint indices not strictly increasing")
                h[i, idx] = 1
            prev = cnt
        if np.any(data[prev : p.omega] != 0):
            raise ValueError("non-zero padding in the hint block")
        return h

    # ---- keys ----

    def pack_pk(self, rho: bytes, t1) -> bytes:
        return rho + b"".join(self.pack_t1(t1[i]) for i in range(self.k))

    def unpack_pk(self, pk: bytes) -> tuple[bytes, np.ndarray]:
        p = self.p
        if len(pk) != p.pk_size:
            raise ValueError(f"public key must be {p.pk_size} bytes, got {len(pk)}")
        rho = pk[:32]
        t1 = np.stack(
            [self.unpack_t1(pk[32 + i * 320 : 32 + (i + 1) * 320]) for i in range(self.k)]
        )
        return rho, t1

    def pack_sk(self, rho, key, tr, s1, s2, t0) -> bytes:
        return (
            rho
            + key
            + tr
            + b"".join(self.pack_eta(s1[i]) for i in range(self.l))
            + b"".join(self.pack_eta(s2[i]) for i in range(self.k))
            + b"".join(self.pack_t0(t0[i]) for i in range(self.k))
        )

    def unpack_sk(self, sk: bytes):
        p, k, l = self.p, self.k, self.l
        if len(sk) != p.sk_size:
            raise ValueError(f"secret key must be {p.sk_size} bytes, got {len(sk)}")
        rho, key, tr = sk[:32], sk[32:64], sk[64:128]
        off = 128
        s1 = np.stack(
            [
                self.unpack_eta(sk[off + i * p.polyeta_bytes : off + (i + 1) * p.polyeta_bytes])
                for i in range(l)
            ]
        )
        off += l * p.polyeta_bytes
        s2 = np.stack(
            [
                self.unpack_eta(sk[off + i * p.polyeta_bytes : off + (i + 1) * p.polyeta_bytes])
                for i in range(k)
            ]
        )
        off += k * p.polyeta_bytes
        t0 = np.stack(
            [
                self.unpack_t0(sk[off + i * p.polyt0_bytes : off + (i + 1) * p.polyt0_bytes])
                for i in range(k)
            ]
        )
        return rho, key, tr, s1, s2, t0

    # ---------------------------------------------------------------- messages

    def format_message(self, msg: bytes, ctx: bytes = b"", prehash: str | None = None) -> bytes:
        """M′ for the external interface (FIPS 204 §5.2), with optional pre-hash."""
        if len(ctx) > 255:
            raise ValueError("ctx must be at most 255 bytes")
        if prehash is None or prehash.lower() == "pure":
            return bytes([0, len(ctx)]) + ctx + msg
        oid, digest = _prehash(prehash, msg)
        return bytes([1, len(ctx)]) + ctx + oid + digest

    def mu(self, pk: bytes, mprime: bytes) -> bytes:
        """μ = H(H(pk, 64) ‖ M′, 64)."""
        return self.H(self.H(pk, 64) + mprime, 64)

    # ---------------------------------------------------------------- algorithms

    def keygen_internal(self, xi: bytes) -> tuple[bytes, bytes]:
        """ML-DSA.KeyGen_internal (FIPS 204 Alg. 6)."""
        k, l = self.k, self.l
        if len(xi) != 32:
            raise ValueError("xi must be 32 bytes")
        # FIPS 204 Alg. 6: (ρ, ρ′, K) ← H(ξ ‖ k ‖ l, 128). The k‖l suffix is the
        # parameter-set domain separation; dropping it makes ρ (and therefore the
        # whole key) wrong while everything else still "works".
        seed = self.H(xi + bytes([k, l]), 128)
        rho, rhoprime, key = seed[:32], seed[32:96], seed[96:]
        A = self.expand_a(rho)
        s1 = np.stack([self.expand_s(rhoprime, i) for i in range(l)])
        s2 = np.stack([self.expand_s(rhoprime, l + i) for i in range(k)])
        s1_hat = ntt_np(s1, self.domain)
        What = (A * s1_hat[None, :, :]).sum(axis=1) % Q
        t = intt_np(What, self.domain) + s2
        t1, t0 = self.power2round(t % Q)
        pk = self.pack_pk(rho, t1)
        tr = self.H(pk, 64)
        return pk, self.pack_sk(rho, key, tr, s1, s2, t0)

    def keygen(self, seed: bytes | None = None) -> tuple[bytes, bytes]:
        return self.keygen_internal(os.urandom(32) if seed is None else seed)

    def sign_internal(
        self, sk: bytes, mprime: bytes, rnd: bytes | None = None, max_attempts: int = 1000
    ) -> bytes:
        """ML-DSA.Sign_internal (FIPS 204 Alg. 7). `rnd=None` ⇒ deterministic."""
        tr = self.unpack_sk(sk)[2]
        return self.sign_mu(sk, self.H(tr + mprime, 64), rnd, max_attempts)

    def sign_mu(
        self, sk: bytes, mu: bytes, rnd: bytes | None = None, max_attempts: int = 1000
    ) -> bytes:
        """Alg. 7 with an externally supplied μ (the external-mu interface)."""
        p, k, l = self.p, self.k, self.l
        rho, key, _tr, s1, s2, t0 = self.unpack_sk(sk)
        rnd = b"\x00" * 32 if rnd is None else rnd
        A = self.expand_a(rho)
        rhoprime = self.H(key + rnd + mu, 64)
        s1_hat = ntt_np(s1, self.domain)
        s2_hat = ntt_np(s2, self.domain)
        t0_hat = ntt_np(t0, self.domain)
        kappa = 0
        for _ in range(max_attempts):
            y = self.expand_mask(rhoprime, kappa)
            y_hat = ntt_np(y, self.domain)
            What = (A * y_hat[None, :, :]).sum(axis=1) % Q
            w = intt_np(What, self.domain)
            w1 = self.high_bits(w)
            ctilde = self.H(mu + b"".join(self.pack_w1(w1[i]) for i in range(k)), p.ctilde_bytes)
            chat = ntt_np(self.sample_in_ball(ctilde), self.domain)
            z = y + intt_np(chat * s1_hat, self.domain)
            z = ((z + Q // 2) % Q) - Q // 2
            if np.max(np.abs(z)) >= p.norm_bound:
                kappa += l
                continue
            w_minus_cs2 = w - intt_np(chat * s2_hat, self.domain)
            r0 = self.low_bits(w_minus_cs2 % Q)
            if np.max(np.abs(r0)) >= p.gamma2 - p.beta:
                kappa += l
                continue
            ct0 = intt_np(chat * t0_hat, self.domain)
            ct0 = ((ct0 + Q // 2) % Q) - Q // 2
            if np.max(np.abs(ct0)) >= p.gamma2:
                kappa += l
                continue
            h = self.make_hint(-ct0, (w_minus_cs2 + ct0) % Q)
            if h.sum() > p.omega:
                kappa += l
                continue
            return b"".join([ctilde] + [self.pack_z(z[i]) for i in range(l)] + [self.pack_h(h)])
        raise RuntimeError("signing did not converge")

    def sign(
        self,
        sk: bytes,
        msg: bytes,
        ctx: bytes = b"",
        prehash: str | None = None,
        rnd: bytes | None = None,
    ) -> bytes:
        """External ML-DSA.Sign: M′ is built from (ctx, msg) first."""
        return self.sign_internal(sk, self.format_message(msg, ctx, prehash), rnd)

    def verify_mu(self, pk: bytes, mu: bytes, sig: bytes) -> bool:
        """ML-DSA.Verify_internal with an externally supplied μ (external-mu interface)."""
        p, k, l = self.p, self.k, self.l
        if len(sig) != p.sig_size:
            return False
        try:
            rho, t1 = self.unpack_pk(pk)
        except ValueError:
            return False
        ctilde = sig[: p.ctilde_bytes]
        z = np.stack(
            [
                self.unpack_z(
                    sig[p.z_offset + i * p.polyz_bytes : p.z_offset + (i + 1) * p.polyz_bytes]
                )
                for i in range(l)
            ]
        )
        if np.max(np.abs(z)) >= p.norm_bound:
            return False
        try:
            h = self.unpack_h(sig[p.h_offset :])
        except ValueError:
            return False

        A = self.expand_a(rho)
        t1_hat = ntt_np((t1 * D_OFFSET) % Q, self.domain)
        chat = ntt_np(self.sample_in_ball(ctilde), self.domain)
        z_hat = ntt_np(z % Q, self.domain)
        w_hat = (A * z_hat[None, :, :]).sum(axis=1) - chat * t1_hat
        w = intt_np(w_hat % Q, self.domain)
        w1 = self.use_hint(h, w)
        c2 = self.H(mu + b"".join(self.pack_w1(w1[i]) for i in range(k)), p.ctilde_bytes)
        return c2 == ctilde

    def verify_internal(self, pk: bytes, mprime: bytes, sig: bytes) -> bool:
        return self.verify_mu(pk, self.mu(pk, mprime), sig)

    def verify(
        self, pk: bytes, msg: bytes, sig: bytes, ctx: bytes = b"", prehash: str | None = None
    ) -> bool:
        """External ML-DSA.Verify (FIPS 204 Alg. 3)."""
        try:
            mprime = self.format_message(msg, ctx, prehash)
        except ValueError:
            return False
        return self.verify_internal(pk, mprime, sig)


Ref = MLDSA
