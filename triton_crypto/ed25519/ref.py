"""Pure-Python reference for Curve25519 / X25519 (RFC 7748) and Ed25519 (RFC 8032)."""

from __future__ import annotations

import hashlib

__all__ = [
    "CURVE_D",
    "CURVE_L",
    "CURVE_P",
    "ED25519_B",
    "ed25519_public_key_np",
    "ed25519_sign_np",
    "ed25519_verify_np",
    "fe_to_limbs10",
    "point_add",
    "point_decode",
    "point_encode",
    "point_neg",
    "point_to_cached",
    "x25519_base_np",
    "x25519_np",
]

CURVE_P = (1 << 255) - 19
CURVE_L = (1 << 252) + 27742317777372353535851937790883648493
CURVE_D = (-121665 * pow(121666, CURVE_P - 2, CURVE_P)) % CURVE_P
_SQRT_M1 = pow(2, (CURVE_P - 1) // 4, CURVE_P)


def _recover_x(y: int, sign: int) -> int | None:
    if y >= CURVE_P:
        return None
    y2 = (y * y) % CURVE_P
    u = (y2 - 1) % CURVE_P
    v = (CURVE_D * y2 + 1) % CURVE_P
    # x = u * v^3 * (u * v^7)^((p-5)/8)
    v3 = (v * v * v) % CURVE_P
    v7 = (v3 * v3 * v) % CURVE_P
    x = (u * v3 * pow((u * v7) % CURVE_P, (CURVE_P - 5) // 8, CURVE_P)) % CURVE_P
    vx2 = (v * x * x) % CURVE_P
    if vx2 == u:
        pass
    elif vx2 == (-u) % CURVE_P:
        x = (x * _SQRT_M1) % CURVE_P
    else:
        return None
    if x == 0 and sign == 1:
        return None
    if (x & 1) != sign:
        x = CURVE_P - x
    return x


_BY = (4 * pow(5, CURVE_P - 2, CURVE_P)) % CURVE_P
_BX = _recover_x(_BY, 0)
assert _BX is not None
ED25519_B = (_BX, _BY, 1, (_BX * _BY) % CURVE_P)
ED25519_O = (0, 1, 1, 0)


def point_add(p1: tuple[int, int, int, int], p2: tuple[int, int, int, int]):
    """Complete twisted Edwards addition (`add-2008-hwcd-3`) in extended coordinates."""
    x1, y1, z1, t1 = p1
    x2, y2, z2, t2 = p2
    a = ((y1 - x1) * (y2 - x2)) % CURVE_P
    b = ((y1 + x1) * (y2 + x2)) % CURVE_P
    c = (2 * CURVE_D * t1 * t2) % CURVE_P
    d = (2 * z1 * z2) % CURVE_P
    e = (b - a) % CURVE_P
    f = (d - c) % CURVE_P
    g = (d + c) % CURVE_P
    h = (b + a) % CURVE_P
    return (
        (e * f) % CURVE_P,
        (g * h) % CURVE_P,
        (f * g) % CURVE_P,
        (e * h) % CURVE_P,
    )


def point_neg(p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    x, y, z, t = p
    return ((-x) % CURVE_P, y, z, (-t) % CURVE_P)


def point_mul(s: int, p: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    q = ED25519_O
    addend = p
    while s > 0:
        if s & 1:
            q = point_add(q, addend)
        addend = point_add(addend, addend)
        s >>= 1
    return q


def point_encode(p: tuple[int, int, int, int]) -> bytes:
    x, y, z, _ = p
    zi = pow(z, CURVE_P - 2, CURVE_P)
    ax = (x * zi) % CURVE_P
    ay = (y * zi) % CURVE_P
    return (ay | ((ax & 1) << 255)).to_bytes(32, "little")


def point_decode(b: bytes) -> tuple[int, int, int, int] | None:
    if len(b) != 32:
        return None
    val = int.from_bytes(b, "little")
    sign = val >> 255
    y = val & ((1 << 255) - 1)
    x = _recover_x(y, sign)
    if x is None:
        return None
    return (x, y, 1, (x * y) % CURVE_P)


def fe_to_limbs10(v: int) -> tuple[int, ...]:
    """Convert field element `v mod p` to centered 10-limb radix-2^25.5 signed integers."""
    v %= CURVE_P
    h = []
    for i in range(10):
        bits = 26 if i % 2 == 0 else 25
        h.append(v & ((1 << bits) - 1))
        v >>= bits
    h0, h1, h2, h3, h4, h5, h6, h7, h8, h9 = h
    c0 = (h0 + (1 << 25)) >> 26
    h1 += c0
    h0 -= c0 << 26
    c1 = (h1 + (1 << 24)) >> 25
    h2 += c1
    h1 -= c1 << 25
    c2 = (h2 + (1 << 25)) >> 26
    h3 += c2
    h2 -= c2 << 26
    c3 = (h3 + (1 << 24)) >> 25
    h4 += c3
    h3 -= c3 << 25
    c4 = (h4 + (1 << 25)) >> 26
    h5 += c4
    h4 -= c4 << 26
    c5 = (h5 + (1 << 24)) >> 25
    h6 += c5
    h5 -= c5 << 25
    c6 = (h6 + (1 << 25)) >> 26
    h7 += c6
    h6 -= c6 << 26
    c7 = (h7 + (1 << 24)) >> 25
    h8 += c7
    h7 -= c7 << 25
    c8 = (h8 + (1 << 25)) >> 26
    h9 += c8
    h8 -= c8 << 26
    c9 = (h9 + (1 << 24)) >> 25
    h0 += c9 * 19
    h9 -= c9 << 25
    c0 = (h0 + (1 << 25)) >> 26
    h1 += c0
    h0 -= c0 << 26
    return (h0, h1, h2, h3, h4, h5, h6, h7, h8, h9)


def point_to_cached(p: tuple[int, int, int, int]) -> tuple[tuple[int, ...], ...]:
    """Convert extended point `(X, Y, Z, T)` to cached `(Y-X, Y+X, 2Z, 2dT)` in 10-limb form."""
    x, y, z, t = p
    return (
        fe_to_limbs10((y - x) % CURVE_P),
        fe_to_limbs10((y + x) % CURVE_P),
        fe_to_limbs10((2 * z) % CURVE_P),
        fe_to_limbs10((2 * CURVE_D * t) % CURVE_P),
    )


def ed25519_public_key_np(seed: bytes) -> bytes:
    """RFC 8032 §5.1.5 Ed25519 public key derivation from a 32-byte seed."""
    if len(seed) != 32:
        raise ValueError(f"Ed25519 seed must be 32 bytes, got {len(seed)}")
    h = bytearray(hashlib.sha512(seed).digest())
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    a = int.from_bytes(h[:32], "little")
    return point_encode(point_mul(a, ED25519_B))


def ed25519_sign_np(seed: bytes, msg: bytes) -> bytes:
    """RFC 8032 §5.1.6 Ed25519 signing from a 32-byte seed."""
    if len(seed) != 32:
        raise ValueError(f"Ed25519 seed must be 32 bytes, got {len(seed)}")
    h = bytearray(hashlib.sha512(seed).digest())
    h[0] &= 248
    h[31] &= 127
    h[31] |= 64
    a = int.from_bytes(h[:32], "little")
    prefix = bytes(h[32:])
    pk = point_encode(point_mul(a, ED25519_B))

    r = int.from_bytes(hashlib.sha512(prefix + msg).digest(), "little") % CURVE_L
    r_enc = point_encode(point_mul(r, ED25519_B))
    k = int.from_bytes(hashlib.sha512(r_enc + pk + msg).digest(), "little") % CURVE_L
    s = (r + k * a) % CURVE_L
    return r_enc + s.to_bytes(32, "little")


def ed25519_verify_np(pk: bytes, msg: bytes, sig: bytes) -> bool:
    """RFC 8032 §5.1.7 strict Ed25519 verification (`s*B - k*A == R`, canonical A and R, `s < L`)."""
    if len(pk) != 32 or len(sig) != 64:
        return False
    a_pt = point_decode(pk)
    if a_pt is None:
        return False
    r_bytes = sig[:32]
    s = int.from_bytes(sig[32:], "little")
    if s >= CURVE_L:
        return False
    k = int.from_bytes(hashlib.sha512(r_bytes + pk + msg).digest(), "little") % CURVE_L
    lhs = point_add(point_mul(s, ED25519_B), point_mul(k, point_neg(a_pt)))
    return point_encode(lhs) == r_bytes


def x25519_np(scalar: bytes, u_coord: bytes) -> bytes:
    """RFC 7748 §5 X25519 scalar multiplication (32-byte LE scalar, 32-byte LE u-coordinate)."""
    if len(scalar) != 32 or len(u_coord) != 32:
        raise ValueError("X25519 requires 32-byte scalar and 32-byte u-coordinate")
    k_bytes = bytearray(scalar)
    k_bytes[0] &= 248
    k_bytes[31] &= 127
    k_bytes[31] |= 64
    k = int.from_bytes(k_bytes, "little")

    u_bytes = bytearray(u_coord)
    u_bytes[31] &= 127
    x_1 = int.from_bytes(u_bytes, "little") % CURVE_P

    x_2, z_2 = 1, 0
    x_3, z_3 = x_1, 1
    swap = 0
    for t in range(254, -1, -1):
        k_t = (k >> t) & 1
        swap ^= k_t
        if swap:
            x_2, x_3 = x_3, x_2
            z_2, z_3 = z_3, z_2
        swap = k_t
        a = (x_2 + z_2) % CURVE_P
        aa = (a * a) % CURVE_P
        b = (x_2 - z_2) % CURVE_P
        bb = (b * b) % CURVE_P
        e = (aa - bb) % CURVE_P
        c = (x_3 + z_3) % CURVE_P
        d = (x_3 - z_3) % CURVE_P
        da = (d * a) % CURVE_P
        cb = (c * b) % CURVE_P
        x_3 = pow(da + cb, 2, CURVE_P)
        z_3 = (x_1 * pow(da - cb, 2, CURVE_P)) % CURVE_P
        x_2 = (aa * bb) % CURVE_P
        z_2 = (e * (aa + 121665 * e)) % CURVE_P
    if swap:
        x_2, x_3 = x_3, x_2
        z_2, z_3 = z_3, z_2
    res = (x_2 * pow(z_2, CURVE_P - 2, CURVE_P)) % CURVE_P
    return res.to_bytes(32, "little")


def x25519_base_np(scalar: bytes) -> bytes:
    """RFC 7748 X25519 scalar multiplication with the standard base point `u = 9`."""
    return x25519_np(scalar, (9).to_bytes(32, "little"))
