"""参考实现封装：统一接口 ref_keygen / ref_encaps / ref_decaps。

底层用 **kyber-py**（第三方 FIPS 203 实现，可选依赖）：
    pip install mlkem-triton[ref]        # 或 pip install kyber-py

它提供确定性内部接口，正好对上 ACVP 向量：
    _keygen_internal(d, z) -> (ek, dk)
    _encaps_internal(ek, m) -> (K, c)
    _decaps_internal(dk, c) -> K

**不是必需的**：正确性主 gate 用 `data/acvp/` 里的 NIST 官方测试向量
（`mlkem_triton.acvp`），不需要任何第三方实现。kyber-py 只在想随机
生成密钥对做往返测试时才用得上。
"""

from __future__ import annotations

import hashlib
import os

from .ntt_ref import byte_decode
from .params import PARAMS, Q

_KEM = None


def _kems() -> dict:
    """惰性导入 kyber-py，缺包时给一条能直接照做的错误信息。"""
    global _KEM
    if _KEM is None:
        try:
            from kyber_py.ml_kem import ML_KEM_512, ML_KEM_768, ML_KEM_1024
        except ModuleNotFoundError as e:  # pragma: no cover
            raise ModuleNotFoundError(
                "这个功能需要 kyber-py（可选依赖）。装它：\n"
                "    pip install mlkem-triton[ref]\n"
                "如果只是想做正确性验证，用 mlkem_triton.acvp + data/acvp/ 下的\n"
                "NIST 官方测试向量，不需要 kyber-py。"
            ) from e
        _KEM = {
            "ML-KEM-512": ML_KEM_512,
            "ML-KEM-768": ML_KEM_768,
            "ML-KEM-1024": ML_KEM_1024,
        }
    return _KEM


def kem(ps: str):
    kems = _kems()
    if ps not in kems:
        raise KeyError(f"未知参数集 {ps}")
    return kems[ps]


# ---------------- 官方接口 ----------------


def ref_keygen(ps: str, d: bytes | None = None, z: bytes | None = None):
    """返回 (ek, dk)。d/z 给定时为确定性。"""
    d = os.urandom(32) if d is None else d
    z = os.urandom(32) if z is None else z
    return kem(ps)._keygen_internal(d, z)


def ref_encaps(ps: str, ek: bytes, m: bytes | None = None):
    """返回 (K, c)；注意 K 在前。"""
    m = os.urandom(32) if m is None else m
    return kem(ps)._encaps_internal(ek, m)


def ref_decaps(ps: str, dk: bytes, c: bytes) -> bytes:
    """返回共享密钥 K（含隐式拒绝）。"""
    return kem(ps)._decaps_internal(dk, c)


# ---------------- FIPS 203 §7 输入校验 ----------------


def decaps_key_check(ps: str, dk: bytes) -> bool:
    """§7.3 解封装密钥检查：长度 + 内嵌 H(ek) 与 ek 一致。"""
    p = PARAMS[ps]
    if len(dk) != p.dk_size:
        return False
    ek = dk[384 * p.k : 768 * p.k + 32]
    h = dk[768 * p.k + 32 : 768 * p.k + 64]
    return hashlib.sha3_256(ek).digest() == h


def encaps_key_check(ps: str, ek: bytes) -> bool:
    """§7.2 封装密钥检查：长度 + 模数检查（每个 12-bit 系数必须 < q）。"""
    p = PARAMS[ps]
    if len(ek) != p.ek_size:
        return False
    # ek 的 t̂ 部分是 k 个多项式，每个 384 字节，需逐个 ByteDecode12
    ok = True
    for i in range(p.k):
        vals = byte_decode(ek[384 * i : 384 * (i + 1)], 12)
        ok = ok and bool((vals < Q).all())
    return ok


# ---------------- 辅助 ----------------


def split_dk(ps: str, dk: bytes):
    """按 FIPS 203 拆 dk：dk_PKE ‖ ek ‖ H(ek) ‖ z。"""
    p = PARAMS[ps]
    k = p.k
    return {
        "dk_pke": dk[: 384 * k],
        "ek": dk[384 * k : 768 * k + 32],
        "h_ek": dk[768 * k + 32 : 768 * k + 64],
        "z": dk[768 * k + 64 : 768 * k + 96],
    }


def implicit_reject_key(ps: str, dk: bytes, c: bytes) -> bytes:
    """K̄ = J(z ‖ c) = SHAKE256(z‖c, 32)。"""
    z = split_dk(ps, dk)["z"]
    return hashlib.shake_256(z + c).digest(32)


__all__ = [
    "decaps_key_check",
    "encaps_key_check",
    "implicit_reject_key",
    "kem",
    "ref_decaps",
    "ref_encaps",
    "ref_keygen",
    "split_dk",
]
