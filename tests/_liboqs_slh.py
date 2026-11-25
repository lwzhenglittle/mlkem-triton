"""Minimal ctypes binding to liboqs' SLH-DSA symbols, used as an independent oracle.

The library is optional: every helper raises `pytest.skip` when it is missing, so the suite still
runs without a liboqs build. Uses the per-algorithm C symbols (`OQS_SIG_slh_dsa_pure_shake_128f_*`)
rather than the `OQS_SIG` struct, so no struct layout is assumed.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = (
    _ROOT.parent / "third_party" / "liboqs" / "build-multi" / "lib" / "liboqs.so",
    _ROOT / "third_party" / "liboqs" / "build-multi" / "lib" / "liboqs.so",
)
_P = ctypes.POINTER(ctypes.c_ubyte)


def _buf(b: bytes) -> ctypes.Array:
    return (ctypes.c_ubyte * max(1, len(b))).from_buffer_copy(b or b"\0")


class LibOQSSLH:
    def __init__(self, lib, base: str, pk_len: int, sk_len: int, sig_len: int):
        self.lib = lib
        self.base = base
        self.pk_len, self.sk_len, self.sig_len = pk_len, sk_len, sig_len

        self._keypair = getattr(lib, f"OQS_SIG_slh_dsa_pure_{base}_keypair")
        self._keypair.restype = ctypes.c_int
        self._keypair.argtypes = [_P, _P]

        self._sign = getattr(lib, f"OQS_SIG_slh_dsa_pure_{base}_sign")
        self._sign.restype = ctypes.c_int
        self._sign.argtypes = [_P, ctypes.POINTER(ctypes.c_size_t), _P, ctypes.c_size_t, _P]

        self._verify = getattr(lib, f"OQS_SIG_slh_dsa_pure_{base}_verify")
        self._verify.restype = ctypes.c_int
        self._verify.argtypes = [_P, ctypes.c_size_t, _P, ctypes.c_size_t, _P]

    @classmethod
    def find(cls, ps: str):
        path = Path(
            os.environ.get(
                "SLHDSA_LIBOQS_SO", _CANDIDATES[0] if _CANDIDATES[0].exists() else _CANDIDATES[1]
            )
        )
        if not path.exists():
            return None
        lib = ctypes.CDLL(str(path))
        base = ps.replace("SLH-DSA-", "").lower().replace("-", "_")
        if not hasattr(lib, f"OQS_SIG_slh_dsa_pure_{base}_verify"):
            return None
        from triton_crypto.slhdsa.params import get

        p = get(ps)
        return cls(lib, base, p.pk_size, p.sk_size, p.sig_size)

    def keypair(self) -> tuple[bytes, bytes]:
        pk = (ctypes.c_ubyte * self.pk_len)()
        sk = (ctypes.c_ubyte * self.sk_len)()
        if self._keypair(pk, sk) != 0:
            raise RuntimeError("liboqs keypair failed")
        return bytes(pk), bytes(sk)

    def sign(self, msg: bytes, sk: bytes) -> bytes:
        sig = (ctypes.c_ubyte * self.sig_len)()
        n = ctypes.c_size_t(self.sig_len)
        if self._sign(sig, ctypes.byref(n), _buf(msg), len(msg), _buf(sk)) != 0:
            raise RuntimeError("liboqs sign failed")
        return bytes(sig)[: n.value]

    def verify(self, msg: bytes, sig: bytes, pk: bytes) -> bool:
        return self._verify(_buf(msg), len(msg), _buf(sig), len(sig), _buf(pk)) == 0


def oracle(ps: str) -> LibOQSSLH:
    lib = LibOQSSLH.find(ps)
    if lib is None:
        pytest.skip("liboqs with SLH-DSA not available")
    return lib
