"""Minimal ctypes binding to a liboqs build that has ML-DSA enabled.

Used as an **independent oracle** (and CPU baseline) for the GPU verifier: liboqs wraps
the `mldsa-native` reference/optimised C implementation, which is a completely separate
codebase from this package. Build it once with

    cd third_party/liboqs
    cmake -B build-mldsa -GNinja -DCMAKE_BUILD_TYPE=Release \
          -DOQS_MINIMAL_BUILD="SIG_ml_dsa_44;SIG_ml_dsa_65;SIG_ml_dsa_87" \
          -DOQS_BUILD_ONLY_LIB=ON -DBUILD_SHARED_LIBS=ON
    ninja -C build-mldsa

The path can be overridden with `MLDSA_LIBOQS_SO`. Everything here degrades to
"unavailable" (None) when the library is missing, so tests can skip.
"""

from __future__ import annotations

import ctypes
import os
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = (
    _ROOT / "third_party" / "liboqs" / "build-bench" / "lib" / "liboqs.so",
    _ROOT / "third_party" / "liboqs" / "build-mldsa" / "lib" / "liboqs.so",
    _ROOT.parent / "third_party" / "liboqs" / "build-bench" / "lib" / "liboqs.so",
    _ROOT.parent / "third_party" / "liboqs" / "build-mldsa" / "lib" / "liboqs.so",
)
DEFAULT_SO = next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0])

SIZES = {
    "ML-DSA-44": (1312, 2560, 2420),
    "ML-DSA-65": (1952, 4032, 3309),
    "ML-DSA-87": (2592, 4896, 4627),
}


def _tag(ps: str) -> str:
    return ps.lower().replace("ml-dsa-", "ml_dsa_")


class LibOQS:
    """The three signatures, bound through ctypes; returns plain bytes."""

    def __init__(self, ps: str, so_path: str | os.PathLike | None = None):
        path = Path(so_path or os.environ.get("MLDSA_LIBOQS_SO", DEFAULT_SO))
        if not path.exists():
            raise FileNotFoundError(path)
        self.ps = ps
        self.pk_size, self.sk_size, self.sig_size = SIZES[ps]
        self.lib = ctypes.CDLL(str(path))
        tag = _tag(ps)
        u8p = ctypes.POINTER(ctypes.c_ubyte)
        size_p = ctypes.POINTER(ctypes.c_size_t)

        self._keypair = getattr(self.lib, f"OQS_SIG_{tag}_keypair")
        self._keypair.argtypes = [u8p, u8p]
        self._keypair.restype = ctypes.c_int

        self._sign = getattr(self.lib, f"OQS_SIG_{tag}_sign")
        self._sign.argtypes = [u8p, size_p, u8p, ctypes.c_size_t, u8p]
        self._sign.restype = ctypes.c_int

        self._verify = getattr(self.lib, f"OQS_SIG_{tag}_verify")
        self._verify.argtypes = [u8p, ctypes.c_size_t, u8p, ctypes.c_size_t, u8p]
        self._verify.restype = ctypes.c_int

    @classmethod
    def available(cls, ps: str = "ML-DSA-65") -> bool:
        path = Path(os.environ.get("MLDSA_LIBOQS_SO", DEFAULT_SO))
        return path.exists()

    def keypair(self) -> tuple[bytes, bytes]:
        pk = (ctypes.c_ubyte * self.pk_size)()
        sk = (ctypes.c_ubyte * self.sk_size)()
        if self._keypair(pk, sk) != 0:
            raise RuntimeError("liboqs keypair failed")
        return bytes(pk), bytes(sk)

    def sign(self, msg: bytes, sk: bytes) -> bytes:
        sig = (ctypes.c_ubyte * self.sig_size)()
        siglen = ctypes.c_size_t(self.sig_size)
        if self._sign(sig, ctypes.byref(siglen), self._as_u8(msg), len(msg), self._as_u8(sk)) != 0:
            raise RuntimeError("liboqs sign failed")
        return bytes(sig[: siglen.value])

    def verify(self, msg: bytes, sig: bytes, pk: bytes) -> bool:
        m = self._as_u8(msg)
        s = self._as_u8(sig)
        p = self._as_u8(pk)
        return self._verify(m, len(msg), s, len(sig), p) == 0

    def verify_many(
        self, msgs: list[bytes], sigs: list[bytes], pk: bytes, threads: int = 1
    ) -> list[bool]:
        """Verify a batch, optionally across `threads` OS threads (GIL is released)."""
        out = [False] * len(msgs)
        if threads <= 1:
            for i, (m, s) in enumerate(zip(msgs, sigs, strict=True)):
                out[i] = self.verify(m, s, pk)
            return out
        chunks = [range(i, len(msgs), threads) for i in range(threads)]

        def work(ids):
            for i in ids:
                out[i] = self.verify(msgs[i], sigs[i], pk)

        ts = [threading.Thread(target=work, args=(c,)) for c in chunks]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        return out

    @staticmethod
    def _as_u8(b: bytes):
        return (ctypes.c_ubyte * len(b)).from_buffer_copy(b)


__all__ = ["DEFAULT_SO", "SIZES", "LibOQS"]
