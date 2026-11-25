"""The numpy ML-DSA reference against the official ACVP vectors (no GPU needed).

Three independent gates, all from usnistgov/ACVP-Server:

    keyGen   75 cases   seed -> (pk, sk), byte-exact
    sigGen  360 cases   (sk, M) -> signature, byte-exact, deterministic *and* hedged
    sigVer  180 cases   (pk, M, σ) -> true/false, including all the invalid ones

The vector files live in `data/acvp-mldsa/`; see `mlkem_triton.mldsa.acvp` for the
download locations. `-k "acvp and keygen or acvp and sigver"` is the fast subset.
"""

from __future__ import annotations

import pytest

from mlkem_triton.mldsa import acvp
from mlkem_triton.mldsa.ref import MLDSA

pytest.importorskip("triton")  # the reference imports the primitives package

PS = ["ML-DSA-44", "ML-DSA-65", "ML-DSA-87"]


def _cases(kind: str):
    try:
        return acvp.load(kind)
    except FileNotFoundError as e:  # pragma: no cover - depends on the checkout
        pytest.skip(str(e))


# ---------------------------------------------------------------- key generation


@pytest.mark.parametrize("ps", PS)
def test_acvp_keygen(ps):
    cases = [c for c in _cases("keyGen") if c["parameterSet"] == ps]
    assert cases
    m = MLDSA(ps)
    for c in cases:
        pk, sk = m.keygen_internal(bytes.fromhex(c["inputs"]["seed"]))
        assert pk.hex().upper() == c["expected"]["pk"].upper(), f"tcId={c['tcId']} pk"
        assert sk.hex().upper() == c["expected"]["sk"].upper(), f"tcId={c['tcId']} sk"


# ---------------------------------------------------------------- signing


@pytest.mark.parametrize("ps", PS)
def test_acvp_siggen_first_of_each_group(ps):
    """One case per (interface, pre-hash, deterministic) combination — the fast subset."""
    cases = [c for c in _cases("sigGen") if c["parameterSet"] == ps]
    seen = set()
    m = MLDSA(ps)
    for c in cases:
        key = (c["interface"], c["preHash"], c["externalMu"], c["deterministic"])
        if key in seen:
            continue
        seen.add(key)
        _check_sign(m, c)


@pytest.mark.slow
@pytest.mark.parametrize("ps", PS)
def test_acvp_siggen_all(ps):
    cases = [c for c in _cases("sigGen") if c["parameterSet"] == ps]
    m = MLDSA(ps)
    for c in cases:
        _check_sign(m, c)


def _check_sign(m: MLDSA, c: dict) -> None:
    sk = bytes.fromhex(c["inputs"]["sk"])
    rnd = bytes.fromhex(c["inputs"]["rnd"]) if "rnd" in c["inputs"] else None
    if c["externalMu"]:
        sig = m.sign_mu(sk, bytes.fromhex(c["inputs"]["mu"]), rnd)
    elif c["interface"] == "internal":
        sig = m.sign_internal(sk, bytes.fromhex(c["inputs"]["message"]), rnd)
    else:
        sig = m.sign(
            sk,
            bytes.fromhex(c["inputs"]["message"]),
            ctx=bytes.fromhex(c["inputs"].get("context", "")),
            prehash=c["inputs"].get("hashAlg"),
            rnd=rnd,
        )
    assert sig.hex().upper() == c["expected"]["signature"].upper(), (
        f"tcId={c['tcId']} tgId={c['tgId']} {c['interface']}/{c['preHash']}"
    )


# ---------------------------------------------------------------- verification


@pytest.mark.parametrize("ps", PS)
def test_acvp_sigver(ps):
    """Includes the invalid signatures: a verifier that always says yes fails here."""
    cases = [c for c in _cases("sigVer") if c["parameterSet"] == ps]
    assert cases
    m = MLDSA(ps)
    n_pass = 0
    for c in cases:
        pk = bytes.fromhex(c["inputs"]["pk"])
        sig = bytes.fromhex(c["inputs"]["signature"])
        if c["externalMu"]:
            got = m.verify_mu(pk, bytes.fromhex(c["inputs"]["mu"]), sig)
        elif c["interface"] == "internal":
            got = m.verify_internal(pk, bytes.fromhex(c["inputs"]["message"]), sig)
        else:
            got = m.verify(
                pk,
                bytes.fromhex(c["inputs"]["message"]),
                sig,
                ctx=bytes.fromhex(c["inputs"].get("context", "")),
                prehash=c["inputs"].get("hashAlg"),
            )
        assert got == c["expected"]["testPassed"], f"tcId={c['tcId']} tgId={c['tgId']}"
        n_pass += got
    assert 0 < n_pass < len(cases)


# ---------------------------------------------------------------- codecs


@pytest.mark.parametrize("ps", PS)
def test_roundtrip_keygen_sign_verify(ps):
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    for msg in (b"", b"x", b"y" * 1000):
        sig = m.sign(sk, msg)
        assert m.verify(pk, msg, sig)
        assert not m.verify(pk, msg + b"!", sig)
        assert not m.verify(pk, msg, sig[:-1])


@pytest.mark.parametrize("ps", PS)
def test_sizes_match_fips204(ps):
    from mlkem_triton.mldsa.params import PARAMS

    spec = {
        "ML-DSA-44": (1312, 2560, 2420),
        "ML-DSA-65": (1952, 4032, 3309),
        "ML-DSA-87": (2592, 4896, 4627),
    }
    pp = PARAMS[ps]
    assert (pp.pk_size, pp.sk_size, pp.sig_size) == spec[ps]
    m = MLDSA(ps)
    pk, sk = m.keygen(bytes(range(32)))
    assert len(pk) == spec[ps][0] and len(sk) == spec[ps][1]
    assert len(m.sign(sk, b"m")) == spec[ps][2]


def test_hint_codec_rejects_malformed_blocks():
    m = MLDSA("ML-DSA-65")
    pk, sk = m.keygen(bytes(range(32)))
    sig = bytearray(m.sign(sk, b"hello"))
    p = m.p
    hb = p.h_offset

    bad = bytearray(sig)
    bad[hb + p.omega] = 0xFF  # count > ω
    with pytest.raises(ValueError):
        m.unpack_h(bytes(bad[hb:]))

    bad = bytearray(sig)
    bad[hb + p.omega + 1] = 0  # counts must be monotone
    with pytest.raises(ValueError):
        m.unpack_h(bytes(bad[hb:]))

    assert not m.verify(pk, b"hello", bytes(bad))
