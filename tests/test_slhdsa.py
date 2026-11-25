"""Device SLH-DSA verification: ACVP agreement, batching, and the error paths.

The device path covers the 6 SHAKE parameter sets; the SHA-2 sets are reference-only for now and
must raise rather than return a wrong answer. Everything here is compared against the numpy
reference (`triton_crypto.slhdsa.ref`), which is itself ACVP-checked in `test_slhdsa_ref.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from triton_crypto.slhdsa import acvp
from triton_crypto.slhdsa.ref import SLH
from triton_crypto.slhdsa.verify import Verifier

pytestmark = pytest.mark.skipif(
    not acvp.DEFAULT_DIR.exists(), reason="ACVP SLH-DSA vectors not present"
)


def _shake_cases(*, valid_only=False):
    return [c for c in acvp.cases() if "SHAKE" in c.ps and (c.expected or not valid_only)]


def _same_key(cases, want):
    """A Verifier holds one public key, so a batch must be signatures under the same key."""
    return [c for c in cases if c.pk == cases[0].pk][:want]


def _run(ps, pk, cases):
    v = Verifier(ps, pk)
    out = []
    for c in cases:
        if c.interface == "internal":
            out.append(bool(v.verify_mprime([c.message], [c.signature])[0]))
        else:
            out.append(bool(v.verify([c.message], [c.signature], c.context or b"", c.ph)[0]))
    return out


@pytest.mark.parametrize(
    "ps",
    [
        "SLH-DSA-SHAKE-128f",
        "SLH-DSA-SHAKE-128s",
        "SLH-DSA-SHAKE-192f",
        "SLH-DSA-SHAKE-256f",
        "SLH-DSA-SHAKE-192s",
        "SLH-DSA-SHAKE-256s",
    ],
)
def test_acvp_sample(ps):
    """Two cases per set (one valid, one invalid) through the device path."""
    cases = _same_key([c for c in acvp.cases() if c.ps == ps], 4)
    valid = [c for c in cases if c.expected][:1]
    invalid = [c for c in cases if not c.expected][:1]
    picked = valid + invalid
    assert _run(ps, picked[0].pk, picked) == [c.expected for c in picked]


def test_batched_matches_reference():
    """A batch of 16 signatures (all the same length, so one launch group) vs the reference."""
    cases = [c for c in acvp.cases() if c.ps == "SLH-DSA-SHAKE-128f" and c.interface == "internal"]
    base = _same_key(cases, 4)
    msgs = [c.message for c in base] * 4
    sigs = np.stack([np.frombuffer(c.signature, dtype=np.uint8) for c in base] * 4)
    got = Verifier("SLH-DSA-SHAKE-128f", base[0].pk).verify_mprime(msgs, sigs)
    want = np.array(
        [
            SLH("SLH-DSA-SHAKE-128f").verify_mprime(m, s.tobytes(), base[0].pk)
            for m, s in zip(msgs, sigs, strict=True)
        ]
    )
    assert got.tolist() == want.tolist()


def test_mixed_message_lengths():
    """Different M' lengths are padded at their own length and processed as separate groups."""
    cases = [c for c in acvp.cases() if c.ps == "SLH-DSA-SHAKE-128f" and c.interface == "internal"]
    good = _same_key([c for c in cases if c.expected], 1)[0]
    other = _same_key([c for c in cases if not c.expected], 1)
    bad_msg = good.message + b"\x00"  # a different length *and* an invalid pairing
    msgs = [good.message, bad_msg]
    sigs = np.stack(
        [
            np.frombuffer(good.signature, dtype=np.uint8),
            np.frombuffer(good.signature, dtype=np.uint8),
        ]
    )
    got = Verifier(good.ps, good.pk).verify_mprime(msgs, sigs)
    assert got.tolist() == [True, False], "each length must be padded at its own length"
    assert other is not None


def test_tampered_and_wrong_key_rejected():
    case = next(c for c in _shake_cases(valid_only=True) if c.interface == "internal")
    v = Verifier(case.ps, case.pk)
    assert v.verify_mprime([case.message], np.frombuffer(case.signature, dtype=np.uint8)[None, :])[
        0
    ]
    sig = bytearray(case.signature)
    sig[1000] ^= 0xFF
    assert not v.verify_mprime([case.message], np.frombuffer(bytes(sig), dtype=np.uint8)[None, :])[
        0
    ]
    other = next(
        c for c in _shake_cases(valid_only=True) if len(c.pk) == len(case.pk) and c.pk != case.pk
    )
    assert not Verifier(case.ps, other.pk).verify_mprime(
        [case.message], np.frombuffer(case.signature, dtype=np.uint8)[None, :]
    )[0]


def test_empty_batch():
    case = _shake_cases(valid_only=True)[0]
    v = Verifier(case.ps, case.pk)
    assert v.verify([], np.zeros((0, len(case.signature)), dtype=np.uint8)).shape == (0,)
    assert v.verify_mprime([], np.zeros((0, len(case.signature)), dtype=np.uint8)).shape == (0,)


def test_sha2_sets_are_not_implemented_on_device():
    case = next(c for c in acvp.cases() if "SHA2" in c.ps)
    with pytest.raises(NotImplementedError):
        Verifier(case.ps, case.pk)


def test_context_is_part_of_the_message():
    case = next(
        c
        for c in _shake_cases(valid_only=True)
        if c.interface == "external" and c.prehash is None and not c.context
    )
    v = Verifier(case.ps, case.pk)
    assert v.verify([case.message], [case.signature])[0]
    assert not v.verify([case.message], [case.signature], ctx=b"x")[0]


def test_prehash_function_is_part_of_the_message():
    case = next(c for c in _shake_cases(valid_only=True) if c.prehash and c.expected)
    v = Verifier(case.ps, case.pk)
    assert v.verify([case.message], [case.signature], case.context or b"", case.ph)[0]
    other = "SHAKE128" if case.ph != "SHAKE128" else "SHAKE256"
    assert not v.verify([case.message], [case.signature], case.context or b"", other)[0]
