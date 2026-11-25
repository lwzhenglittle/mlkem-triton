"""Phase 1 验收：官方 ACVP 向量 + 参考实现自洽性。"""

from __future__ import annotations

import pytest

from mlkem_triton import acvp, refimpl
from mlkem_triton.params import PARAMS

pytest.importorskip("kyber_py", reason="refimpl 需要可选的 kyber-py")

ALL = acvp.load()

PARAM_SETS = ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"]


def _h(b: bytes) -> str:
    return b.hex().upper()


# ---------------- 官方 ACVP 向量 ----------------


def test_acvp_group_summary():
    s = acvp.group_summary(ALL)
    assert len(ALL) > 0
    # 必须有解封装用例（含隐式拒绝）
    assert ("ML-KEM-768", "decapsulation", "VAL") in s
    print(f"\nACVP 用例: {len(ALL)} 条")
    for k, v in sorted(s.items()):
        print(f"  {k[0]:<12} {k[1]:<22} {k[2]:<5} {v:>3}")


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_acvp_encapsulation_aft(ps):
    cases = [c for c in ALL if c["function"] == "encapsulation" and c["parameterSet"] == ps]
    assert cases, f"{ps} 没有封装用例"
    for c in cases:
        ek = bytes.fromhex(c["inputs"]["ek"])
        m = bytes.fromhex(c["inputs"]["m"])
        K, ct = refimpl.ref_encaps(ps, ek, m)
        assert _h(ct) == c["expected"]["c"].upper(), f"tcId={c['tcId']} 密文不符"
        assert _h(K) == c["expected"]["k"].upper(), f"tcId={c['tcId']} 共享密钥不符"


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_acvp_decapsulation_val(ps):
    """含非法密文 / 隐式拒绝用例。"""
    cases = [c for c in ALL if c["function"] == "decapsulation" and c["parameterSet"] == ps]
    assert cases, f"{ps} 没有解封装用例"
    n_reject = 0
    for c in cases:
        dk = bytes.fromhex(c["inputs"]["dk"])
        ct = bytes.fromhex(c["inputs"]["c"])
        got = refimpl.ref_decaps(ps, dk, ct)
        assert _h(got) == c["expected"]["k"].upper(), f"tcId={c['tcId']} 共享密钥不符"
        if got == refimpl.implicit_reject_key(ps, dk, ct):
            n_reject += 1
    print(f"\n{ps}: {len(cases)} 条解封装用例，其中 {n_reject} 条走隐式拒绝")


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_acvp_decapsulation_key_check(ps):
    cases = [c for c in ALL if c["function"] == "decapsulationKeyCheck" and c["parameterSet"] == ps]
    assert cases
    for c in cases:
        got = refimpl.decaps_key_check(ps, bytes.fromhex(c["inputs"]["dk"]))
        assert got == c["expected"]["testPassed"], f"tcId={c['tcId']} got={got}"


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_acvp_encapsulation_key_check(ps):
    cases = [c for c in ALL if c["function"] == "encapsulationKeyCheck" and c["parameterSet"] == ps]
    assert cases
    for c in cases:
        got = refimpl.encaps_key_check(ps, bytes.fromhex(c["inputs"]["ek"]))
        assert got == c["expected"]["testPassed"], f"tcId={c['tcId']} got={got}"


# ---------------- 参考实现自洽性 ----------------


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_keygen_encaps_decaps_roundtrip(ps):
    for _ in range(5):
        ek, dk = refimpl.ref_keygen(ps)
        K, ct = refimpl.ref_encaps(ps, ek)
        assert refimpl.ref_decaps(ps, dk, ct) == K


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_sizes_match_params(ps):
    p = PARAMS[ps]
    ek, dk = refimpl.ref_keygen(ps)
    K, ct = refimpl.ref_encaps(ps, ek)
    assert (len(ek), len(dk), len(ct), len(K)) == (p.ek_size, p.dk_size, p.ct_size, 32)


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_dk_split_layout(ps):
    p = PARAMS[ps]
    ek, dk = refimpl.ref_keygen(ps)
    parts = refimpl.split_dk(ps, dk)
    assert len(parts["dk_pke"]) == 384 * p.k
    assert len(parts["ek"]) == 384 * p.k + 32
    assert len(parts["h_ek"]) == 32
    assert len(parts["z"]) == 32
    assert parts["ek"] == ek


@pytest.mark.parametrize("ps", PARAM_SETS)
def test_implicit_rejection_on_bitflip(ps):
    """伪造密文必须返回 K̄ = J(z‖c)，且与原密钥不同。"""
    ek, dk = refimpl.ref_keygen(ps)
    K, ct = refimpl.ref_encaps(ps, ek)
    for pos in (0, len(ct) // 2, len(ct) - 1):
        bad = bytearray(ct)
        bad[pos] ^= 0x01
        bad = bytes(bad)
        K_bad = refimpl.ref_decaps(ps, dk, bad)
        assert K_bad != K, f"pos={pos} 隐式拒绝失效"
        assert K_bad == refimpl.implicit_reject_key(ps, dk, bad), f"pos={pos} K̄ 不符"


def test_decaps_key_check_detects_tampering():
    ps = "ML-KEM-768"
    _ek, dk = refimpl.ref_keygen(ps)
    assert refimpl.decaps_key_check(ps, dk)
    p = PARAMS[ps]
    for pos in (0, 384 * p.k + 5, 768 * p.k + 32, 768 * p.k + 64, len(dk) - 1):
        bad = bytearray(dk)
        bad[pos] ^= 0x01
        # dk_pke / z 被改动不影响哈希检查（这是 FIPS 203 的已知性质），
        # 只有 ek 或 H(ek) 被改动才会失败。
        affected = pos >= 384 * p.k and pos < 768 * p.k + 64
        if affected:
            assert not refimpl.decaps_key_check(ps, bytes(bad)), f"pos={pos} 应检出"
