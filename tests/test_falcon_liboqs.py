"""Falcon against liboqs, when a liboqs build with Falcon is present.

The embedded vector in `tests/test_falcon.py` proves one signature; this proves the batch path
against a different implementation's signatures, and -- more useful -- that both implementations
agree on *rejections*, which is where the two decoders (this one and pqclean's prefix code)
can disagree without either looking broken.

Build the library once:

    cd third_party/liboqs
    cmake -B build-falcon -GNinja -DCMAKE_BUILD_TYPE=Release \\
          -DOQS_MINIMAL_BUILD="SIG_falcon_512;SIG_falcon_padded_512" \\
          -DOQS_BUILD_ONLY_LIB=ON -DBUILD_SHARED_LIBS=ON
    ninja -C build-falcon

`FALCON_LIBOQS_SO` overrides the path; everything degrades to a skip when it is missing.

**The corpus is nondeterministic on purpose, and that is a known limitation.** liboqs signs with a
fresh keypair and a fresh nonce on every run, and this build has neither
`OQS_randombytes_nist_kat_init_256bit` nor a working `OQS_randombytes_switch_algorithm("NIST-KAT")`
(both were tried: the switch returns -1), so the vectors cannot be frozen without adding a data
file. If a run fails here, `probes/falcon-padded-hunt.py` regenerates a few hundred signatures and
reports which ones were rejected and why -- it was written to chase exactly one such failure, and
500 signatures later (320 accepted, 320 tampered rejected by both implementations) that hypothesis
was dead.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

from triton_crypto.falcon import (
    FALCON_512,
    FALCON_1024,
    FALCON_PADDED_512,
    FALCON_PADDED_1024,
    Verifier,
)

TAG_PARAMS = {
    "falcon_512": (FALCON_512, False),
    "falcon_padded_512": (FALCON_PADDED_512, True),
    "falcon_1024": (FALCON_1024, False),
    "falcon_padded_1024": (FALCON_PADDED_1024, True),
}

_ROOT = Path(__file__).resolve().parent.parent
_CANDIDATES = (
    _ROOT / "third_party" / "liboqs" / "build-bench" / "lib" / "liboqs.so",
    _ROOT / "third_party" / "liboqs" / "build-falcon" / "lib" / "liboqs.so",
    _ROOT.parent / "third_party" / "liboqs" / "build-bench" / "lib" / "liboqs.so",
    _ROOT.parent / "third_party" / "liboqs" / "build-falcon" / "lib" / "liboqs.so",
)
DEFAULT_SO = Path(
    os.environ.get("FALCON_LIBOQS_SO", next((p for p in _CANDIDATES if p.exists()), _CANDIDATES[0]))
)

U8P = ctypes.POINTER(ctypes.c_ubyte)
CASES = 64


def _lib(tag: str):
    if not DEFAULT_SO.exists():
        pytest.skip(f"liboqs with Falcon not built at {DEFAULT_SO}")
    lib = ctypes.CDLL(str(DEFAULT_SO))
    keypair = lib[f"OQS_SIG_{tag}_keypair"]
    keypair.argtypes = [U8P, U8P]
    sign = lib[f"OQS_SIG_{tag}_sign"]
    sign.argtypes = [U8P, ctypes.POINTER(ctypes.c_size_t), U8P, ctypes.c_size_t, U8P]
    verify = lib[f"OQS_SIG_{tag}_verify"]
    verify.argtypes = [U8P, ctypes.c_size_t, U8P, ctypes.c_size_t, U8P]
    return keypair, sign, verify


def _corpus(tag: str, n: int = CASES):
    params, _ = TAG_PARAMS[tag]
    keypair, sign, verify = _lib(tag)
    cap = 65536
    pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
    assert keypair(pk, sk) == 0
    pkb = bytes(pk[: params.pk_len])
    msgs, sigs = [], []
    for i in range(n):
        msg = b"liboqs-falcon-%03d" % i
        mbuf = (ctypes.c_ubyte * len(msg)).from_buffer_copy(msg)
        sig = (ctypes.c_ubyte * cap)()
        siglen = ctypes.c_size_t(cap)
        assert sign(sig, ctypes.byref(siglen), mbuf, len(msg), sk) == 0
        msgs.append(msg)
        sigs.append(bytes(sig[: siglen.value]))
    return pkb, msgs, sigs, verify


@pytest.mark.parametrize("tag", list(TAG_PARAMS))
def test_accepts_every_liboqs_signature(tag):
    params, padded = TAG_PARAMS[tag]
    pk, msgs, sigs, _ = _corpus(tag)
    ok = Verifier(pk, params, padded=padded).verify(msgs, sigs)
    assert ok.all(), f"{tag}: {int(ok.sum())}/{len(ok)} accepted"


@pytest.mark.parametrize("tag", list(TAG_PARAMS))
def test_rejects_what_liboqs_rejects(tag):
    params, padded = TAG_PARAMS[tag]
    pk, msgs, sigs, verify = _corpus(tag)
    tampered = []
    for sig in sigs:
        bad = bytearray(sig)
        bad[1 + params.nonce_len + 7] ^= 0x01  # inside the coefficient stream
        tampered.append(bytes(bad))
    ok = Verifier(pk, params, padded=padded).verify(msgs, tampered)
    assert not ok.any(), f"{tag}: {int(ok.sum())} tampered signatures accepted"

    # and liboqs must agree on the same 64 (it is the other implementation of the decoder)
    pkbuf = (ctypes.c_ubyte * len(pk)).from_buffer_copy(pk)
    disagree = 0
    for msg, sig in zip(msgs, tampered, strict=True):
        mbuf = (ctypes.c_ubyte * len(msg)).from_buffer_copy(msg)
        sbuf = (ctypes.c_ubyte * len(sig)).from_buffer_copy(sig)
        disagree += verify(mbuf, len(msg), sbuf, len(sig), pkbuf) == 0
    assert disagree == 0


@pytest.mark.parametrize("tag", ["falcon_512", "falcon_1024"])
def test_mixed_batch_with_one_bad_row(tag):
    """One bad row must not disturb its neighbours -- the batch is one launch, after all."""
    params, _ = TAG_PARAMS[tag]
    pk, msgs, sigs, _ = _corpus(tag, 16)
    bad = bytearray(sigs[7])
    bad[len(bad) // 2] ^= 0x01
    sigs[7] = bytes(bad)
    ok = Verifier(pk, params).verify(msgs, sigs)
    assert ok.tolist() == [True] * 7 + [False] + [True] * 8


@pytest.mark.parametrize(
    ("unpadded_tag", "padded_tag", "params"),
    [
        ("falcon_512", "falcon_padded_512", FALCON_512),
        ("falcon_1024", "falcon_padded_1024", FALCON_1024),
    ],
)
def test_padded_and_unpadded_share_a_verifier(unpadded_tag, padded_tag, params):
    """Both encodings decode to the same coefficients; the verdict must not depend on which."""
    pk_u, msgs_u, sigs_u, _ = _corpus(unpadded_tag, 4)
    pk_p, msgs_p, sigs_p, _ = _corpus(padded_tag, 4)
    assert Verifier(pk_u, params, padded=False).verify(msgs_u, sigs_u).all()
    assert Verifier(pk_p, params, padded=True).verify(msgs_p, sigs_p).all()
    assert all(len(s) <= params.sig_padded for s in sigs_u)
