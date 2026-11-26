"""Encapsulation (FIPS 203 Alg. 17) acceptance: must pass the NIST ACVP vectors byte for byte."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _common import DEV, HAS_GPU

from triton_crypto import acvp, keyprep, refimpl
from triton_crypto.encaps import Encapsulator

pytest.importorskip("triton")

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="requires GPU")

PS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


def _cases(ps):
    return [c for c in acvp.load() if c["function"] == "encapsulation" and c["parameterSet"] == ps]


def _h(v):
    return bytes.fromhex(v)


@pytest.mark.parametrize("ps", PS)
def test_parse_ek_builds_enc_matrix(ps):
    """The W_enc from `parse_ek` must be the same matrix as the formula in `parse_key`.

    Method: build a fake dk from the encapsulation vector's ek (stuff the ek in),
    run it through `parse_key`, and compare.
    """
    cs = _cases(ps)
    ek = _h(cs[0]["inputs"]["ek"])
    ep = keyprep.parse_ek(ps, ek)
    k = ep.k
    # recompute with the ek's own Â/t to confirm W_enc's block layout matches build_W_enc
    t_hat = np.stack([keyprep.byte_decode(ek[384 * j : 384 * (j + 1)], 12) for j in range(k)])
    A_coef = np.stack([[keyprep.intt(ep.A_hat[i, j]) for j in range(k)] for i in range(k)])
    t_coef = np.stack([keyprep.intt(t_hat[j]) for j in range(k)])
    assert np.array_equal(ep.W_enc, keyprep.build_W_enc(A_coef, t_coef, k))
    assert ep.h_ek is not None and len(ep.h_ek) == 32


@pytest.mark.parametrize("ps", PS)
def test_acvp_all_vectors(ps):
    """25 official vectors; ciphertext and shared secret must both be byte-identical."""
    cs = _cases(ps)
    assert len(cs) == 25
    for c in cs:
        enc = Encapsulator(ps, _h(c["inputs"]["ek"]), device=DEV)
        got_c, got_k = enc.encapsulate([_h(c["inputs"]["m"])])
        assert got_c[0].tobytes() == _h(c["expected"]["c"]), f"tcId {c['tcId']} ciphertext"
        assert got_k[0].tobytes() == _h(c["expected"]["k"]), f"tcId {c['tcId']} K"


@pytest.mark.parametrize("ps", PS)
def test_batch_same_message(ps):
    """The same message repeated B times; every output must equal the ACVP expected value."""
    cs = _cases(ps)
    enc = Encapsulator(ps, _h(cs[0]["inputs"]["ek"]), device=DEV)
    m = _h(cs[0]["inputs"]["m"])
    got_c, got_k = enc.encapsulate([m] * 37)
    assert got_c.shape == (37, enc.p.ct_size)
    for i in range(37):
        assert got_c[i].tobytes() == _h(cs[0]["expected"]["c"])
        assert got_k[i].tobytes() == _h(cs[0]["expected"]["k"])


@pytest.mark.parametrize("ps", PS)
def test_roundtrip_with_our_own_decaps(ps):
    """Our encaps → our decaps must return the same K.

    Also decaps independently with kyber-py — only if both directions agree is the
    encapsulation really correct (checking ACVP's c equality alone is already strong,
    but that uses the same batch of vectors; here we switch to random keys).
    """
    pytest.importorskip("kyber_py", reason="requires kyber-py to generate random key pairs")
    from triton_crypto import refimpl
    from triton_crypto.pipeline import Decapsulator

    ek, dk = refimpl.ref_keygen(ps)
    rng = np.random.default_rng(7)
    msgs = [rng.integers(0, 256, 32, dtype=np.uint8).tobytes() for _ in range(16)]

    got_c, got_k = Encapsulator(ps, ek, device=DEV).encapsulate(msgs)
    dec = Decapsulator(ps, dk, device=DEV)
    mine = dec.decaps(got_c)
    for i, _m in enumerate(msgs):
        assert mine[i].tobytes() == got_k[i].tobytes(), f"roundtrip {i} inconsistent"
        assert refimpl.ref_decaps(ps, dk, got_c[i].tobytes()) == got_k[i].tobytes(), (
            f"kyber-py rejects case {i}"
        )


def test_pack_handles_odd_bit_widths():
    """With du/dv = 11/5 one byte spans 3 coefficients; bit packing must be right (regression)."""
    from triton_crypto.kernels.encaps import pack_ciphertext

    for du, dv, k in ((10, 4, 3), (11, 5, 4)):
        ct = 32 * (du * k + dv)
        B = 2
        comp = torch.randint(0, 1 << du, (256 * (k + 1), B), dtype=torch.int16, device=DEV)
        comp[k * 256 :] = torch.randint(0, 1 << dv, (256, B), dtype=torch.int16, device=DEV)
        padded = torch.zeros((B, ((ct + 136) // 136) * 136), dtype=torch.uint8, device=DEV)
        pack_ciphertext(comp, padded, k, du, dv, ct)
        got = padded[:, :ct].cpu().numpy()
        cc = comp.cpu().numpy()
        for b in range(B):
            bits = []
            for j in range(k):
                for c in range(256):
                    bits += [(int(cc[j * 256 + c, b]) >> t) & 1 for t in range(du)]
            for c in range(256):
                bits += [(int(cc[k * 256 + c, b]) >> t) & 1 for t in range(dv)]
            want = np.array(
                [sum(bits[o * 8 + t] << t for t in range(8)) for o in range(ct)], np.uint8
            )
            assert np.array_equal(got[b], want), (du, dv, b)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))


@pytest.mark.parametrize("ps", PS)
def test_empty_batch_and_non_contiguous_messages(ps):
    ek = _h(_cases(ps)[0]["inputs"]["ek"])
    enc = Encapsulator(ps, ek)
    c, k = enc.encapsulate(np.empty((0, 32), np.uint8))
    assert c.shape == (0, enc.p.ct_size) and k.shape == (0, 32)

    msgs = np.arange(4 * 32, dtype=np.uint8).reshape(4, 32)
    base_c, base_k = enc.encapsulate_torch(torch.from_numpy(msgs).to(DEV), 4)
    # a column-strided view has valid values but non-contiguous rows
    wide = torch.zeros((4, 64), dtype=torch.uint8, device=DEV)
    view = wide[:, ::2]
    view.copy_(torch.from_numpy(msgs).to(DEV))
    assert not view.is_contiguous()
    got_c, got_k = enc.encapsulate_torch(view, 4)
    assert torch.equal(base_c, got_c) and torch.equal(base_k, got_k)


def test_mixed_length_message_list_is_rejected():
    ek = _h(_cases("ML-KEM-768")[0]["inputs"]["ek"])
    with pytest.raises(ValueError, match="32 bytes"):
        Encapsulator("ML-KEM-768", ek).encapsulate([b"a" * 31, b"b" * 33])


@pytest.mark.parametrize("ps", PS)
def test_non_canonical_ek_is_rejected(ps):
    ek = bytearray(_h(_cases(ps)[0]["inputs"]["ek"]))
    assert keyprep.parse_ek(ps, bytes(ek), validate=False) is not None
    ek[:384] = b"\xff" * 384  # 12-bit coefficients of 4095 >= q
    assert not refimpl.encaps_key_check(ps, bytes(ek))
    with pytest.raises(ValueError, match="not canonical"):
        Encapsulator(ps, bytes(ek))
    # explicit escape hatch for formally-invalid test vectors
    assert Encapsulator(ps, bytes(ek), validate=False) is not None
