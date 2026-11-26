"""Falcon-512 verification, numpy reference: the definitions the device path is judged by.

Everything here mirrors `pqclean_falcon-512_clean` line for line, because the details are
where Falcon verification actually lives:

    comp_decode     signatures are *variable length*: per coefficient, one byte carrying the
                    sign and the low 7 bits of the magnitude, then a unary continuation
                    (each 0 bit adds 128, a 1 bit ends it), so 654-666 bytes in practice.
                    "-0" is forbidden and the last byte's unused bits must be zero.
    modq_decode     public keys are a 14-bit MSB-first stream; a coefficient >= q is invalid
                    and the padding bits must be zero.
    hash_to_point   SHAKE256(nonce || m) -> 16-bit big-endian samples -> keep w < 5q (otherwise
                    reject) -> w mod q -> the first n accepted samples, in order. The reference
                    does this with a branch-free compaction over `n + 205` samples; filtering
                    in order gives the same sequence (the probability of running out of samples
                    is below 2^-256).
    is_short        sum(s1[u]^2 + s2[u]^2) <= 34034726, with s1 = c - s2*h normalised into
                    [-q/2, q/2] and s2 the *decoded* signed coefficients.

The public key arrives in NTT+Montgomery form in the reference (`to_ntt_monty`), but that only
matters to the C implementation: a plain NTT, a pointwise product and an inverse NTT compute
the same `s2 * h mod X^n + 1` without Montgomery scalings.
"""

from __future__ import annotations

import hashlib

import numpy as np

from .params import FALCON_512, FalconParams


def comp_decode(buf: bytes, params: FalconParams = FALCON_512) -> tuple[np.ndarray, int]:
    """Decode a compressed signature body into signed coefficients and the bytes consumed."""
    n = params.n
    x = np.zeros(n, dtype=np.int64)
    acc = 0
    acc_len = 0
    pos = 0
    max_in = len(buf)
    for u in range(n):
        if pos >= max_in:
            raise ValueError("truncated signature: ran out of bytes")
        acc = ((acc << 8) | buf[pos]) & 0xFFFFFFFF
        pos += 1
        b = (acc >> acc_len) & 0xFF
        sign = b & 128
        mag = b & 127
        while True:
            if acc_len == 0:
                if pos >= max_in:
                    raise ValueError("truncated signature: ran out of bits")
                acc = ((acc << 8) | buf[pos]) & 0xFFFFFFFF
                pos += 1
                acc_len = 8
            acc_len -= 1
            if (acc >> acc_len) & 1:
                break
            mag += 128
            if mag > params.max_abs_coefficient:
                raise ValueError("signature coefficient out of range (> 2047)")
        if sign and mag == 0:
            raise ValueError("signature contains -0")
        x[u] = -mag if sign else mag
    if acc & ((1 << acc_len) - 1):
        raise ValueError("signature padding bits are not zero")
    return x, pos


def modq_decode(buf: bytes, params: FalconParams = FALCON_512) -> np.ndarray:
    """Decode the 14-bit public-key stream into coefficients in [0, q)."""
    n = params.n
    need = (n * 14 + 7) >> 3
    if need > len(buf):
        raise ValueError(f"public key too short: need {need} bytes, got {len(buf)}")
    out = np.zeros(n, dtype=np.int64)
    acc = 0
    acc_len = 0
    pos = 0
    u = 0
    while u < n:
        acc = ((acc << 8) | buf[pos]) & 0xFFFFFFFF
        pos += 1
        acc_len += 8
        if acc_len >= 14:
            acc_len -= 14
            w = (acc >> acc_len) & 0x3FFF
            if w >= params.q:
                raise ValueError("public key coefficient out of range (>= q)")
            out[u] = w
            u += 1
    if acc & ((1 << acc_len) - 1):
        raise ValueError("public key padding bits are not zero")
    return out


def hash_to_point(nonce: bytes, message: bytes, params: FalconParams = FALCON_512) -> np.ndarray:
    """SHAKE256(nonce || message) -> n coefficients in [0, q), in acceptance order.

    Sample, reject (`w >= 5q`), reduce, keep the first n -- the reference does the same with a
    branch-free compaction over the same 205 extra samples. The filter is vectorised: the
    per-sample Python loop it replaced cost 79 us per signature, which is 60x the SHAKE itself.
    """
    need = (params.n + params.oversample) * 2
    stream = hashlib.shake_256(nonce + message).digest(need)
    w = np.frombuffer(stream, dtype=">u2").astype(np.int64)  # big-endian 16-bit samples
    xs = w[w < 5 * params.q] % params.q
    if xs.size < params.n:
        raise ValueError("SHAKE256 stream exhausted before n samples were accepted")
    return xs[: params.n].astype(np.int64)


def hash_to_point_batch(nonces, messages, params: FalconParams = FALCON_512) -> np.ndarray:
    """`n` coefficients per (nonce, message), stacked into `[B, n]`."""
    if len(nonces) != len(messages):
        raise ValueError(f"{len(nonces)} nonces but {len(messages)} messages")
    if not nonces:
        return np.zeros((0, params.n), dtype=np.int64)
    return np.stack([hash_to_point(no, m, params) for no, m in zip(nonces, messages, strict=True)])


def _ntt(f: np.ndarray, params: FalconParams, inverse: bool = False) -> np.ndarray:
    """Negacyclic NTT mod X^n + 1 over Z_q, via a root of 2n-th order (g = 7 for q = 12289)."""

    dom = _domain(params)
    return dom.intt(f) if inverse else dom.ntt(f)


_DOMAINS: dict[tuple[int, int, int], object] = {}


def root_of_unity(q: int, n: int) -> int:
    """The smallest `zeta` of multiplicative order exactly `2n` mod q.

    Falcon's own generator is 7, but 7 has order 2048 and `7^12` only has order 512; the NTT
    needs order `2n = 1024`, so the primitive root has to be searched for (11 gives 10302,
    the value `eval-triton-crypto-primitives.md` verified by hand). Any element of order 2n
    gives an equivalent transform: the tables are built from it, and pointwise multiplication
    plus the inverse is the negacyclic product either way.
    """
    exp = (q - 1) // (2 * n)
    for g in range(2, 256):
        z = pow(g, exp, q)
        if pow(z, n, q) == q - 1:  # z^n == -1 => order exactly 2n
            return z
    raise ValueError(f"no element of order {2 * n} found mod {q}")


def _domain(params: FalconParams):
    key = (params.q, params.n, params.logn + 1)
    dom = _DOMAINS.get(key)
    if dom is None:
        from triton_crypto.primitives.ntt import NTTDomain

        dom = NTTDomain(
            "falcon",
            q=params.q,
            n=params.n,
            zeta=root_of_unity(params.q, params.n),
            layers=params.logn,
        )
        _DOMAINS[key] = dom
    return dom


def verify_core(
    c: np.ndarray, s2: np.ndarray, pk_poly: np.ndarray, params: FalconParams = FALCON_512
) -> tuple[bool, int]:
    """The part that runs on the device: `s1 = c - s2*h mod q`, then the norm bound.

    Returns `(accept, squared_norm)`; the norm is returned even when it rejects, because that
    number is the only diagnostic a failing verification has.
    """
    q = params.q
    s2_mod = np.asarray(s2, dtype=np.int64) % q
    prod = _ntt(_tt_pointwise(_ntt(s2_mod, params), _ntt(pk_poly, params), q), params, inverse=True)
    # numpy's `%` is floored, so this is already in [0, q) for negative differences too --
    # which is exactly the difference from Triton's truncated `%` that the kernel has to
    # normalise by hand.
    t = (prod - np.asarray(c, dtype=np.int64)) % q
    # normalise into [-q/2, q/2]
    t = np.where(t > q // 2, t - q, t)
    sqn = int(np.sum(t * t) + np.sum(np.asarray(s2, dtype=np.int64) ** 2))
    return sqn <= params.l2bound, sqn


def _tt_pointwise(a: np.ndarray, b: np.ndarray, q: int) -> np.ndarray:
    return a * b % q


def verify(pk: bytes, message: bytes, sig: bytes, params: FalconParams = FALCON_512) -> bool:
    """Verify one Falcon-512 signature. Raises `ValueError` on malformed input."""
    if len(sig) < 1 + params.nonce_len:
        raise ValueError("signature shorter than the header and nonce")
    if sig[0] != params.sig_header:
        raise ValueError(f"bad signature header {sig[0]:#x}, expected {params.sig_header:#x}")
    if len(pk) != params.pk_len:
        raise ValueError(f"public key must be {params.pk_len} bytes, got {len(pk)}")
    if pk[0] != params.pk_header:
        raise ValueError(f"bad public key header {pk[0]:#x}, expected {params.pk_header:#x}")

    nonce = sig[1 : 1 + params.nonce_len]
    body = sig[1 + params.nonce_len :]
    s2, consumed = comp_decode(body, params)
    if consumed != len(body):
        # Falcon-padded pads with zeros to a fixed length; anything else must be exact.
        padded_len = params.sig_padded - 1 - params.nonce_len
        if not (len(body) == padded_len and not any(body[consumed:])):
            raise ValueError("signature has trailing garbage")

    h = modq_decode(pk[1:], params)
    c = hash_to_point(nonce, message, params)
    ok, _ = verify_core(c, s2, h, params)
    return ok


def comp_decode_batch_np(bodies: list[bytes], params: FalconParams = FALCON_512):
    """Vectorised `comp_decode` over a batch: the per-coefficient scan, one numpy op per step.

    The code is a prefix code whose coefficients are not byte-aligned, so a batch cannot be
    decoded with a single bit-gather. But the *scan* can be vectorised over the batch: fix the
    coefficient index `u`, and all rows do step `u` together. 512 steps of ~20 numpy ops each
    decodes 4096 signatures in well under a second, where the per-signature Python loop needs
    ~134 us each.

    Returns `(coefficients [B, n] int64, ok [B] bool, used [B] int64)`. `ok` is the *strict*
    rule (the body must be consumed exactly); the padded variant relaxes that, so the caller
    gets `used` and decides. A malformed row is reported, not raised -- the caller is a verifier.
    """
    B = len(bodies)
    if B == 0:
        return (
            np.zeros((0, params.n), dtype=np.int64),
            np.zeros(0, dtype=bool),
            np.zeros(0, dtype=np.int64),
        )
    lens = np.array([len(b) for b in bodies], dtype=np.int64)
    width = int(lens.max()) + 3  # +3 so the 3-byte windows never read out of bounds
    buf = np.zeros((B, width), dtype=np.uint8)
    for i, b in enumerate(bodies):
        buf[i, : len(b)] = np.frombuffer(b, dtype=np.uint8)
    rows = np.arange(B)

    clip = width - 1

    def window32(byte):
        """32 bits starting at the byte boundary `byte`, zero-padded past the buffer.

        One 32-bit window answers both questions a step asks -- the 8-bit header at `bitpos` and
        the 16-bit unary run at `bitpos + 8` -- so the scan does four gathers per coefficient
        instead of six, which is where two thirds of its time went.
        """
        b0 = buf[rows, np.minimum(byte, clip)].astype(np.int64)
        b1 = buf[rows, np.minimum(byte + 1, clip)].astype(np.int64)
        b2 = buf[rows, np.minimum(byte + 2, clip)].astype(np.int64)
        b3 = buf[rows, np.minimum(byte + 3, clip)].astype(np.int64)
        return (b0 << 24) | (b1 << 16) | (b2 << 8) | b3

    # Leading-zero count of a 16-bit window, as a table: one gather instead of 16 `where`s.
    lead = np.zeros(1 << 16, dtype=np.int64)
    for v in range(1, 1 << 16):
        lead[v] = 16 - int(v).bit_length()
    x = np.zeros((B, params.n), dtype=np.int64)
    pos = np.zeros(B, dtype=np.int64)
    ok = np.ones(B, dtype=bool)
    bits = lens * 8
    max_mag = params.max_abs_coefficient
    for u in range(params.n):
        w = window32(pos >> 3)
        shift = pos & 7
        head = (w >> (24 - shift)) & 0xFF  # 8 bits at `pos`: sign + low 7 magnitude bits
        sign = (head & 0x80) != 0
        cont = (w >> (8 - shift)) & 0xFFFF  # 16 bits at `pos + 8`: the unary run
        k = lead[cont]
        mag = (head & 0x7F) + 128 * k
        bad = cont == 0  # no terminating 1 within 16 bits => |c| would exceed 2047
        bad |= mag > max_mag
        bad |= sign & (mag == 0)  # "-0" is forbidden
        bad |= (pos + 8 + k + 1) > bits  # ran past the end of this signature
        ok &= ~bad
        x[:, u] = np.where(sign, -mag, mag)
        pos = pos + 8 + k + 1

    # Unused bits of the last used byte must be zero, and the body must be fully consumed.
    consumed_bytes = (pos + 7) >> 3
    last_byte = buf[rows, np.minimum(consumed_bytes - 1, width - 1)].astype(np.int64)
    tail_bits = (8 - (pos & 7)) & 7
    ok &= ((last_byte & ((1 << tail_bits) - 1)) == 0) | (tail_bits == 0)
    ok &= consumed_bytes == lens
    return x, ok, consumed_bytes
