"""SLH-DSA (FIPS 205) reference: parameter derivation, ACVP keyGen and sigVer.

The reference is pure Python (a 128f verification is ~7k SHAKE256 calls) but it is the definition
of the semantics the device kernels will be checked against, so it is tested first and hardest:

- parameter derivation is checked against the published signature sizes,
- `keygen_internal` is byte-exact against all 12 ACVP keyGen groups (pk *and* sk),
- verification agrees with ACVP for every interface and every pre-hash function,
- and the oracle-free properties (tamper, truncation, wrong pk) are checked directly.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from triton_crypto.slhdsa import PARAMS, SLH
from triton_crypto.slhdsa import acvp as acvp_mod
from triton_crypto.slhdsa.params import get

_DATA = Path(__file__).resolve().parent.parent / "data" / "acvp-slhdsa"

pytestmark = pytest.mark.skipif(
    not (_DATA / "sigVer.json").exists(), reason="ACVP SLH-DSA vectors not present"
)

# Published FIPS 205 sizes; the derivation must reproduce them exactly.
SIZES = {
    "128s": (7856, 32, 64),
    "128f": (17088, 32, 64),
    "192s": (16224, 48, 96),
    "192f": (35664, 48, 96),
    "256s": (29792, 64, 128),
    "256f": (49856, 64, 128),
}


def test_parameter_derivation():
    for name, p in PARAMS.items():
        sig, pk, sk = SIZES[name.rsplit("-", 1)[1]]
        assert (p.sig_size, p.pk_size, p.sk_size) == (sig, pk, sk), name
        assert p.wots_len in (35, 51, 67)
        assert p.m in (30, 34, 39, 42, 47, 49)
        assert p.h == p.d * p.hp
        # the digest split must cover exactly m bytes
        assert p.md_bytes + p.tree_bytes + p.leaf_bytes == p.m


def test_hash_family_ph_default():
    assert PARAMS["SLH-DSA-SHA2-128f"].ph_name == "SHA2-256"
    assert PARAMS["SLH-DSA-SHA2-192f"].ph_name == "SHA2-512"
    assert PARAMS["SLH-DSA-SHAKE-128s"].ph_name == "SHAKE128"
    assert PARAMS["SLH-DSA-SHAKE-256s"].ph_name == "SHAKE256"


def _keygen_cases():
    data = json.loads((_DATA / "keyGen.json").read_text())
    out = []
    for g in data["groups"]:
        for t in g["tests"]:
            out.append(
                pytest.param(
                    g["parameterSet"],
                    base64.b64decode(t["seed"]),
                    base64.b64decode(t["pk"]),
                    base64.b64decode(t["sk"]),
                    id=f"{g['parameterSet']}-tc{t['tcId']}",
                )
            )
    return out


@pytest.mark.parametrize(
    "ps,seed,pk,sk",
    [c for c in _keygen_cases() if "-128f" in c.id or "-192f" in c.id or "-256f" in c.id][:6],
)
def test_acvp_keygen(ps, seed, pk, sk):
    """One keyGen case per parameter set: pk and sk byte-exact."""
    got_sk, got_pk = SLH(ps).keygen_internal(seed)
    assert got_pk == pk
    assert got_sk == sk


@pytest.mark.slow
@pytest.mark.parametrize("ps,seed,pk,sk", _keygen_cases())
def test_acvp_keygen_all(ps, seed, pk, sk):
    got_sk, got_pk = SLH(ps).keygen_internal(seed)
    assert (got_pk, got_sk) == (pk, sk)


@pytest.mark.parametrize("case", acvp_mod.cases(), ids=lambda c: c.label)
def test_acvp_sigver(case):
    v = SLH(case.ps)
    if case.interface == "internal":
        got = v.verify_mprime(case.message, case.signature, case.pk)
    else:
        got = v.verify(case.message, case.signature, case.pk, case.context or b"", case.ph)
    assert got is case.expected


def test_tampered_signature_is_rejected():
    case = next(c for c in acvp_mod.cases() if c.expected)
    sig = bytearray(case.signature)
    sig[len(sig) // 2] ^= 0x01
    assert (
        SLH(case.ps).verify(case.message, bytes(sig), case.pk, case.context or b"", case.ph)
        is False
    )


def test_wrong_public_key_is_rejected():
    cases = [c for c in acvp_mod.cases() if c.expected]
    a = cases[0]
    other = next(c for c in cases if len(c.pk) == len(a.pk) and c.pk != a.pk)
    assert SLH(a.ps).verify(a.message, a.signature, other.pk, a.context or b"", a.ph) is False


def test_truncated_signature_is_rejected():
    case = next(c for c in acvp_mod.cases() if c.expected)
    assert SLH(case.ps).verify(case.message, case.signature[:-1], case.pk) is False


def test_context_changes_the_digest():
    """A valid pure signature must not verify under a different context."""
    case = next(c for c in acvp_mod.cases() if c.expected and c.interface == "external")
    ctx = case.context or b""
    assert SLH(case.ps).verify(case.message, case.signature, case.pk, ctx, case.ph)
    assert not SLH(case.ps).verify(case.message, case.signature, case.pk, ctx + b"x", case.ph)


def test_prehash_interface_requires_the_matching_function():
    """The pre-hash function is part of the semantics: swapping it must fail."""
    case = next(c for c in acvp_mod.cases() if c.prehash and c.expected)
    v = SLH(case.ps)
    assert v.verify(case.message, case.signature, case.pk, case.context or b"", case.ph)
    other = "SHA2-256" if case.ph != "SHA2-256" else "SHA3-384"
    assert v.verify(case.message, case.signature, case.pk, case.context or b"", other) is False


def test_unknown_parameter_set():
    with pytest.raises(KeyError):
        get("SLH-DSA-SHAKE-42f")
