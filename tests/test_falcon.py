"""Falcon-512 verification, host side: decoders, hash-to-point, and one real signature.

The vectors were produced by liboqs 0.16.0 (`pqclean_falcon-512`), so they are an independent
implementation's output rather than this package's. This file checks the decoders and hash against
the pure-Python reference, verifies that a real signature is accepted, and checks that four
different kinds of tampering are not.
"""

from __future__ import annotations

import contextlib

import numpy as np
import pytest

from triton_crypto.falcon import (
    FALCON_512,
    FALCON_1024,
    FALCON_PADDED_512,
    FALCON_PADDED_1024,
    comp_decode,
    hash_to_point,
    modq_decode,
    ref,
    verify_np,
)

PK_HEX = (
    "092bea27c732164e6ff1a0789411fda72e5239ecae4d1b0ecd513d4e4b60699fbb04c4471faec4428323a89462368446"
    "0a0419e6e30b9992a247bd6fd3f6d154b819144a6c002c5efc39c42ccfca3b9846d39ace8fb4f86614a26cf4670932a1"
    "3e58e8850e940bab4786817a1b54e4ab8c277088e384589959e29c37a8cb64902cac670c933b4b35e278b98ba13286f3"
    "8888d3fe1f72550adb4f225c0a561619ecb33dd8a76ad46fdb3f784efa1cc1046d49d2c882a82f4d68d9e7b2dccda466"
    "0e597cd69e78aa40a5fba1382c41820a002ad63ee6ba294e51991bf4ddb9f794450738afc4142dd627fc22c4f14927ae"
    "468e08748d62339c819a84dcae1df272a4265a6049cddee3c780759ee6a21b5603236ed1d3e4102b244652866024e36e"
    "638e03f70a9792692d90401a6b1323954e5657494d5be07019e20c22a17e01b6c009084bcd88e3825d167f29b5b2b36b"
    "d1171418e6890acca33986c225586d45c595322f628c67f9c96297a6dd5a2c6f40f1a8dd205c4eb3f6c89a6ce6187b31"
    "9f17079e2fd4141c231302dfc2cd22dc29e8599ad25bdd85b0364231e35f3916873746a7954b30b992fcc2f78f7ac22a"
    "16c4dbbe0d6cb11dae4b92ea474b26ebb88c56ae000218cd9f88ac42a64629675a37a66a114b0a423324204c48032aa6"
    "b0e24edf28f44b177f5d641da838d4cfeb0058de41374525234024d018144a028e5cacea379e58562e5a67548ea541a8"
    "ce520c02752c69b90705a0ee97e19f05b255430750a238a006ad7da57461f568447d4cb2eaf309695ce18f07820e2e8d"
    "8612facf63217a60e48b71999a3f667e1b0bdb171d727811a51827bb060edace56902e1a78d498cc0ebc407e104cc1cc"
    "59e44807881b4d8525fbbbae20d21208e3273728e21fe849b663c7247c304e18fdc1427e6de094870c3a60728d9b4e29"
    "4a7b612b848a80ce6d460ba54a0fe024e4cfe3722f0e14561ed6f3401d263586b6a968617b58a265e8877b4e71132941"
    "e39f6c95f2ab9d1da862044880cb47d5210440e23411cc85455962d0c8a67ff28a891f15b095d104e99542db8320e4c8"
    "4bc2764894d614bc12294cad163badd4a042be01142347ea56c01186f2dfea49b1ae48a02ab16a7e518642c7934ee566"
    "05f148bb492456b0ede9bd56182e0465958d2c129e96139e5b6b396a95d9ccf5640a7c36e08d417f019feb76e82e2082"
    "a7409b01f12142c8d824e8c2332c9b84e8cff9201e3d40d9a2cb7a7c81b69bcbf4"
)

SIG_HEX = (
    "39d5fbe7bd86a13a9f51c385388cccf7e9d7a36195ca6022d85bba40bc75030915c5153d232d5d7634179153953161d3"
    "1a0373a947484886bf6c95dd8153cdd42d992a995edf14954b23ec16d1504f23ac385e30cdaf5240adf35ffd9f755e45"
    "85b71e76b2ac1ff64c9e2151e832fb161602fa8349eba871be16be454612d666e4069ed5c43085dfcd857fe2b904b0eb"
    "bdb8aef7fe60fd39360f7e606f32a59a1aafa58c8ed5837db06a473a13c5335096b3ee6a174c7f75c16d3c39e4f6b108"
    "29113ad25b9c75905def2612417bed4c8977213527f76ecbeb695edcfe393db63089bb4cdad1ec8d23ebd892b3d7bcdd"
    "3a56b0e6175c01ed54de073743afa56694270a5e742ebe760502c44a130cd5c3817fe876b129fb5916f7cb9c6f6d2acf"
    "84d036c96742f58a22e53ddd951fe9bca34279cea6a461d9862684cf302be2dfdbe921972c890c6d46820b4286ad73ac"
    "2dc7c30de63c1f9ea5cfa7d09bc93c99fabfaa410c560b84fed1855ef1499ff7d1efb1275137c60ca9f854044d796b76"
    "88d3cedf353ad43b9ed14e5a66e909e471e1b65d9d39d14657527686cfa2d418ca1f667fa4bf4afaa73a8c74cca20a6b"
    "cb6b0095313bc6074a44b4a4be079baad0aa92dd4a9b96d06aff14c89fc1d437f202f09dbdf1eea6be70c113ba0c8c52"
    "a46e1a3c94d6b1ec250d4e49a73cc503d3bf7ffc918aca4c44ca8cbcabc5d38450701045b6708495c6f90de7a443bd3a"
    "492b0df31885409b2d0c27c776a83cbdedea334641e3bd076a4fc09b71c96b1b79bfe81844bf73882eba461e8a230839"
    "28c920aaacb95caf9361cd43fc99155a93947bb9db5aa47364e8a642c71e55e790478587e13e68bab685a2f8ff2cc930"
    "56d8782984874748a1ecf975921742eb1bcaadde0c3b1afe2dfb44566bdade8bec"
)

SIG2_HEX = (
    "3918cfc0d1980c8024202659eec40bd93fb0f54cb88d98b6f231bb969e7f49ba2b985687c6e9df7ef7becf3a32a91367"
    "d9c18aa7559420108c8d7ee0933d0834d360f4c7cc75bb7d4be44fa450183b347f7391d27e9ccd6eb2f5848bc49bfd9f"
    "f4ef3a539a0d8bd29b981c2a67f9baf32cb63b19646aa549066e05a5a2ac9c74613c62e1ac5f7626a9eed83c90e643ff"
    "89dd1492344633653850ec2d49ad316b79aa58d028952a7b3368632621ebe3d5e69241f376a04e969db8a6b3c9537be7"
    "14ce85525c65b0b1443383dca076d4473b96a74da6f7bd110e39ea68d1c08e9e689e5aec589223ac1bbb15fbf78ae835"
    "50d66adca3218cea16442abc8e6a9b8c191efac8f32e16b91a2c90a14f2d598d3d3c1d2e895320ab4295fcc31df85685"
    "136b91d50943aecd1914227153513aef2e89a5d376a7b2180f0f669990151e9ce5cc62ce7d7cc4b555f46add128ca6dc"
    "9c318cec46bbd864a63fca6faa6ae59fb98763deaadf1ddf501347615bfad5f43ae514ad4b0dc5d5cd9dd5d14e52070a"
    "bd1faf919ab1d46b72d2124c399ecc2e10aa2abc87499331c7610d79f45cf6be659b47245cacebe377d0d5b2d5340742"
    "9146fa18fcc5055b982aed9674b7555d347187921021c8b172719492fd9b498e5723f5c06f8cc1cc436f687d9f6b0fe9"
    "a66a992b667f58170da38f21af4d314d9ee2f05871ac182a7e7be3b7d3cdbb2da24d184497ed53b0cf5c28e3e2e8acf1"
    "59a7dea6cdd2fac75e2976b243619958b4a3efc14fb4cd6e421e8cb874255f5accc2147eba8fa65b3bf65a7459fbd536"
    "d05100663f28b471032a56f57e35a1ad30b3ca9ee2813133ba4e7f6c97b0bd8609a6c590b693ab6697fe7b2d4a1c78e3"
    "d2043b0aaf622133444095f8d69a71bd9f746953a6edee8e65af6dbdab07"
)

MSG = b"triton-crypto-kernels falcon KAT"
MSG2 = b"second message"
PK = bytes.fromhex(PK_HEX)
SIG = bytes.fromhex(SIG_HEX)
SIG2 = bytes.fromhex(SIG2_HEX)


def test_parameters_match_the_reference_implementation():
    p = FALCON_512
    assert (p.n, p.logn, p.q) == (512, 9, 12289)
    assert p.nonce_len == 40
    assert p.pk_header == 0x09 and p.sig_header == 0x39
    assert len(PK) == p.pk_len == 897
    assert len(SIG) <= p.sig_max == 752
    assert p.sig_padded == 666
    assert p.l2bound == 34034726
    assert p.oversample == 205

    p10 = FALCON_1024
    assert (p10.n, p10.logn, p10.q) == (1024, 10, 12289)
    assert p10.nonce_len == 40
    assert p10.pk_header == 0x0A and p10.sig_header == 0x3A
    assert p10.pk_len == 1793 and p10.sk_len == 2305
    assert p10.sig_max == 1462 and p10.sig_padded == 1280
    assert p10.l2bound == 70265242 and p10.oversample == 287
    assert FALCON_PADDED_1024.sig_padded == 1280 and FALCON_PADDED_512.sig_padded == 666


def test_public_key_decodes_into_range():
    h = modq_decode(PK[1:], FALCON_512)
    assert h.shape == (512,)
    assert h.min() >= 0 and h.max() < FALCON_512.q


def test_hash_to_point_is_deterministic_and_in_range():
    c1 = hash_to_point(SIG[1:41], MSG, FALCON_512)
    c2 = hash_to_point(SIG[1:41], MSG, FALCON_512)
    assert np.array_equal(c1, c2)
    assert c1.shape == (512,)
    assert c1.min() >= 0 and c1.max() < FALCON_512.q


def test_valid_signature_is_accepted():
    assert verify_np(PK, MSG, SIG) is True
    assert verify_np(PK, MSG2, SIG2) is True


def test_wrong_message_is_rejected():
    assert verify_np(PK, MSG2, SIG) is False


def test_wrong_key_is_rejected():
    other = modq_decode(PK[1:], FALCON_512)
    other = (other + 1) % FALCON_512.q
    assert verify_np(PK, MSG, SIG) is True  # sanity
    # the real check: a public key that decodes but is not the signer's
    assert (
        ref.verify_core(
            hash_to_point(SIG[1:41], MSG, FALCON_512),
            comp_decode(SIG[41:], FALCON_512)[0],
            other,
            FALCON_512,
        )[0]
        is False
    )


@pytest.mark.parametrize("flip", [0, 1, 2, 3])
def test_tampering_is_rejected(flip):
    bad = bytearray(SIG)
    if flip == 0:  # header byte
        bad[0] ^= 0x01
    elif flip == 1:  # nonce
        bad[5] ^= 0x01
    elif flip == 2:  # middle of the coefficient stream
        bad[len(bad) // 2] ^= 0x01
    else:  # last byte
        bad[-1] ^= 0x01
    if flip == 0:
        with pytest.raises(ValueError, match="header"):
            verify_np(PK, MSG, bytes(bad))
    else:
        # A tampered body is either rejected on the norm or fails to decode; both are "no".
        with contextlib.suppress(ValueError):
            assert verify_np(PK, MSG, bytes(bad)) is False


def test_malformed_inputs_raise():
    with pytest.raises(ValueError, match="shorter than"):
        verify_np(PK, MSG, SIG[:20])
    with pytest.raises(ValueError, match="public key must be"):
        verify_np(PK[:100], MSG, SIG)
    with pytest.raises(ValueError, match="bad signature header"):
        verify_np(PK, MSG, b"\x00" + SIG[1:])


def test_public_key_rejects_a_coefficient_above_q():
    # 512 coefficients x 14 bits is exactly 896 bytes, so there are no padding bits to test;
    # the decoder's other guard is a coefficient >= q, i.e. the field 0x3FFF.
    h = bytearray(PK[1:])
    h[0] |= 0xC0  # first 14-bit field becomes 0x3FFF-ish
    with pytest.raises(ValueError, match="out of range"):
        modq_decode(bytes(h), FALCON_512)


def test_offline_kat_corpus_accepts_valid_and_rejects_tampered():
    """Deterministic multi-signature KATs exported from liboqs (512 + 1024, unpadded + padded)."""
    import json
    from pathlib import Path

    tag_params = {
        "falcon_512": FALCON_512,
        "falcon_padded_512": FALCON_PADDED_512,
        "falcon_1024": FALCON_1024,
        "falcon_padded_1024": FALCON_PADDED_1024,
    }
    kat_path = Path(__file__).resolve().parent.parent / "data" / "vectors" / "falcon" / "kat.json"
    data = json.loads(kat_path.read_text(encoding="utf-8"))
    seen_tags = set()
    total = 0
    for group in data["groups"]:
        tag = group["tag"]
        seen_tags.add(tag)
        params = tag_params[tag]
        pk = bytes.fromhex(group["pk_hex"])
        assert len(pk) == params.pk_len
        for case in group["cases"]:
            msg = bytes.fromhex(case["msg_hex"])
            sig = bytes.fromhex(case["sig_hex"])
            assert verify_np(pk, msg, sig, params) is True
            bad = bytearray(sig)
            bad[1 + params.nonce_len + 5] ^= 0x01
            with contextlib.suppress(ValueError):
                assert verify_np(pk, msg, bytes(bad), params) is False
            total += 1
    assert seen_tags == set(tag_params)
    assert total >= 64
