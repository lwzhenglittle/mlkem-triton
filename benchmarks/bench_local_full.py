#!/usr/bin/env python
"""Systematic local GPU vs CPU benchmark suite (full parameter sets, Dev + E2E, breakeven sweep).

Measures the local GPU against `benchmarks/cpu_baseline_barrier.json` in one continuous window:

  Part A  All five schemes across parameter sets and operations (Dev + E2E vs 1T / 16P):
            - ML-KEM-{512,768,1024}: keygen, encaps, decaps (B=4096)
            - ML-DSA-{44,65,87}: verify (B=4096), sign (B=1024)
            - SLH-DSA-{SHAKE,SHA2}-128f: verify (B=4096, matched same-family CPU rival)
            - Falcon-{512,padded-512}: verify core & e2e (B=4096)
            - LMS/HSS RFC 8554 case 0: verify (B=4096; Python reference noted, no C liboqs rival)
  Part B  All 11 scheme-agnostic primitive families (B=65536, size=1024 B):
            both Device-only (tensor resident) and End-to-End (Host NumPy -> Device -> Host)
            against the same barrier 1T / 16P CPU baseline.
  Part C  Batch-size breakeven sweep for four representative workloads:
            ML-KEM-768 decaps, ML-DSA-44 verify, SHA3-256 (1 KiB), ChaCha20 (1 KiB).

Usage:
    python benchmarks/bench_local_full.py --out benchmarks/local-gpu-vs-cpu-20261128.md
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
BENCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BENCH))
sys.path.insert(0, str(ROOT / "tests"))

from bench_16p_barrier import power_state
from bench_primitive import CASES as PRIM_CASES
from bench_primitive import CPU_RIVAL as PRIM_CPU_RIVAL
from bench_primitive import ORDER as PRIM_ORDER
from bench_primitive import bench_copy
from bench_primitive import measure as measure_prim

from triton_crypto import devices
from triton_crypto.falcon import (
    FALCON_512,
    FALCON_1024,
    FALCON_PADDED_512,
    FALCON_PADDED_1024,
)
from triton_crypto.falcon import Verifier as FalconVerifier
from triton_crypto.falcon import ref as falcon_ref
from triton_crypto.falcon.verify import comp_decode_batch
from triton_crypto.lms import ref as lms_ref
from triton_crypto.lms.verify import Verifier as LMSVerifier
from triton_crypto.mldsa.ref import MLDSA
from triton_crypto.mldsa.sign import Signer as MLDSASigner
from triton_crypto.mldsa.verify import Verifier as MLDSAVerifier
from triton_crypto.mlkem import acvp as mlkem_acvp
from triton_crypto.mlkem.encaps import Encapsulator
from triton_crypto.mlkem.keygen import KeyGenerator
from triton_crypto.mlkem.pipeline import Decapsulator
from triton_crypto.primitives import (
    aes,
    blake3,
    chacha20,
    gemm,
    poly1305,
    poseidon2,
    sha2,
    sponge,
)
from triton_crypto.primitives.ntt import LargeNTT
from triton_crypto.slhdsa import acvp as slh_acvp
from triton_crypto.slhdsa.verify import Verifier as SLHVerifier


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=15
        ).stdout.strip()
    except Exception:
        return "unknown"


def _time_fn(fn, dev, reps: int = 9, warmup: int = 3) -> tuple[float, float]:
    """Return (min_s, median_s) including device sync."""
    for _ in range(warmup):
        fn()
    devices.sync(dev)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        devices.sync(dev)
        ts.append(time.perf_counter() - t0)
    ts.sort()
    return ts[0], ts[len(ts) // 2]


def _fmt_ratio(rate: float | None, cpu_rate: float | None) -> str:
    if not rate or not cpu_rate:
        return "n/a"
    r = rate / cpu_rate
    return f"{r:.1f}x" if r >= 10 else f"{r:.2f}x"


def _ratio_val(rate: float | None, cpu_rate: float | None) -> float | None:
    if not rate or not cpu_rate:
        return None
    return rate / cpu_rate


def _oqs_so() -> Path:
    override = os.environ.get("OQS_BENCH_SO")
    if override and Path(override).exists():
        return Path(override)
    for rel in (
        "third_party/liboqs/build-bench/lib/liboqs.so",
        "third_party/liboqs/build-all/lib/liboqs.so",
        "third_party/liboqs/build-falcon/lib/liboqs.so",
    ):
        p = ROOT / rel
        if p.exists():
            return p
    raise FileNotFoundError("no liboqs.so found under third_party/liboqs/")


def _oqs_sig_corpus(so: Path, tag: str, pk_len: int, n: int, msg_len: int = 32):
    lib = ctypes.CDLL(str(so))
    u8p = ctypes.POINTER(ctypes.c_ubyte)
    kp = lib[f"OQS_SIG_{tag}_keypair"]
    kp.argtypes = [u8p, u8p]
    sign = lib[f"OQS_SIG_{tag}_sign"]
    sign.argtypes = [u8p, ctypes.POINTER(ctypes.c_size_t), u8p, ctypes.c_size_t, u8p]
    cap = 65536
    pk, sk = (ctypes.c_ubyte * cap)(), (ctypes.c_ubyte * cap)()
    if kp(pk, sk) != 0:
        raise RuntimeError(f"{tag}: keypair failed")
    msgs, sigs = [], []
    for i in range(n):
        msg = (f"bench-{i:06d}-" + "x" * max(0, msg_len - 13)).encode()[:msg_len]
        mb = (ctypes.c_ubyte * len(msg)).from_buffer_copy(msg)
        sig = (ctypes.c_ubyte * cap)()
        sl = ctypes.c_size_t(cap)
        if sign(sig, ctypes.byref(sl), mb, len(msg), sk) != 0:
            raise RuntimeError(f"{tag}: sign failed")
        msgs.append(msg)
        sigs.append(bytes(sig[: sl.value]))
    return bytes(pk[:pk_len]), msgs, sigs


def _bench_schemes(dev, rivals: dict, oqs_so: Path) -> list[dict]:
    rows: list[dict] = []
    acvp_cases = mlkem_acvp.load()

    def add_row(
        scheme: str,
        op: str,
        batch: int,
        dev_us: float | None,
        e2e_us: float | None,
        rival_key: str | None,
        note: str = "",
    ):
        rv = rivals.get(rival_key) if rival_key else None
        cpu1_us = (1e6 / rv["one_thread"]) if rv else None
        cpu16_us = (1e6 / rv["procs16"]) if rv else None
        dev_rate = (1e6 / dev_us) if dev_us else None
        e2e_rate = (1e6 / e2e_us) if e2e_us else None
        rows.append(
            {
                "scheme": scheme,
                "op": op,
                "batch": batch,
                "dev_us": dev_us,
                "e2e_us": e2e_us,
                "cpu1_us": cpu1_us,
                "cpu16_us": cpu16_us,
                "rival": rival_key,
                "dev_vs_1t": _ratio_val(dev_rate, rv["one_thread"]) if rv else None,
                "dev_vs_16p": _ratio_val(dev_rate, rv["procs16"]) if rv else None,
                "e2e_vs_1t": _ratio_val(e2e_rate, rv["one_thread"]) if rv else None,
                "e2e_vs_16p": _ratio_val(e2e_rate, rv["procs16"]) if rv else None,
                "note": note,
            }
        )
        print(
            f"  {scheme:18s} {op:7s} B={batch:4d}  "
            f"dev={dev_us if dev_us is not None else -1:7.3f}us  "
            f"e2e={e2e_us if e2e_us is not None else -1:7.3f}us  "
            f"16P={cpu16_us if cpu16_us is not None else -1:7.3f}us  "
            f"dev/16P={_fmt_ratio(dev_rate, rv['procs16'] if rv else None):>6s}  "
            f"e2e/16P={_fmt_ratio(e2e_rate, rv['procs16'] if rv else None):>6s}",
            flush=True,
        )

    # 1. ML-KEM (512, 768, 1024) x (keygen, encaps, decaps)
    B_kem = 4096
    for ps in ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"):
        tag = ps.lower().replace("ml-kem-", "mlkem-")
        # keygen
        kg = KeyGenerator(ps, device=dev)
        rng = np.random.default_rng(1)
        d_np = rng.integers(0, 256, (B_kem, 32), dtype=np.uint8)
        z_np = rng.integers(0, 256, (B_kem, 32), dtype=np.uint8)
        d_t = torch.from_numpy(d_np.copy()).to(dev)
        z_t = torch.from_numpy(z_np.copy()).to(dev)
        t_dev, _ = _time_fn(lambda dt=d_t, zt=z_t, kg=kg: kg.generate_torch(dt, zt), dev, reps=9)
        t_e2e, _ = _time_fn(lambda kg=kg: kg.generate(B_kem), dev, reps=5, warmup=2)
        add_row(ps, "keygen", B_kem, t_dev / B_kem * 1e6, t_e2e / B_kem * 1e6, f"{tag}-keygen")

        # encaps
        ec = next(
            c for c in acvp_cases if c["function"] == "encapsulation" and c["parameterSet"] == ps
        )
        ek = bytes.fromhex(ec["inputs"]["ek"])
        m0 = bytes.fromhex(ec["inputs"]["m"])
        enc = Encapsulator(ps, ek, device=dev)
        c_chk, k_chk = enc.encapsulate([m0] * 4)
        assert c_chk[0].tobytes() == bytes.fromhex(ec["expected"]["c"])
        assert k_chk[0].tobytes() == bytes.fromhex(ec["expected"]["k"])
        msgs_kem = [m0] * B_kem
        m_arr = np.frombuffer(b"".join(msgs_kem), np.uint8).reshape(B_kem, 32).copy()
        m_t = torch.from_numpy(m_arr).to(dev)
        t_dev, _ = _time_fn(lambda mt=m_t, enc=enc: enc.encapsulate_torch(mt, B_kem), dev, reps=9)
        t_e2e, _ = _time_fn(lambda ms=msgs_kem, enc=enc: enc.encapsulate(ms), dev, reps=9)
        add_row(ps, "encaps", B_kem, t_dev / B_kem * 1e6, t_e2e / B_kem * 1e6, f"{tag}-encaps")

        # decaps
        dc = next(
            c for c in acvp_cases if c["function"] == "decapsulation" and c["parameterSet"] == ps
        )
        dk = bytes.fromhex(dc["inputs"]["dk"])
        ct1 = bytes.fromhex(dc["inputs"]["c"])
        k1 = bytes.fromhex(dc["expected"]["k"])
        dec = Decapsulator(ps, dk, device=dev)
        got = dec.decaps([ct1] * 4)
        assert all(got[i].tobytes() == k1 for i in range(4))
        cts = [ct1] * B_kem
        ct_arr = np.frombuffer(b"".join(cts), np.uint8).reshape(B_kem, len(ct1)).copy()
        ct_t = torch.from_numpy(ct_arr).to(dev)
        t_dev, _ = _time_fn(lambda ct=ct_t, dec=dec: dec.decaps_torch(ct, B_kem), dev, reps=9)
        t_e2e, _ = _time_fn(lambda cts=cts, dec=dec: dec.decaps(cts), dev, reps=9)
        add_row(ps, "decaps", B_kem, t_dev / B_kem * 1e6, t_e2e / B_kem * 1e6, f"{tag}-decaps")

    # 2. ML-DSA (44, 65, 87) x (verify, sign)
    B_dsa_ver, B_dsa_sign = 4096, 1024
    dsa_pk_sizes = {"ML-DSA-44": 1312, "ML-DSA-65": 1952, "ML-DSA-87": 2592}
    for ps in ("ML-DSA-44", "ML-DSA-65", "ML-DSA-87"):
        tag_oqs = ps.lower().replace("ml-dsa-", "ml_dsa_")
        tag_key = ps.lower().replace("ml-dsa-", "mldsa-")
        # verify
        pk, msgs, sigs = _oqs_sig_corpus(oqs_so, tag_oqs, dsa_pk_sizes[ps], B_dsa_ver, 32)
        ver = MLDSAVerifier(ps, pk, device=dev)
        ref = MLDSA(ps)
        sig_arr = np.stack([np.frombuffer(s, np.uint8) for s in sigs])
        ok = ver.verify(msgs, sig_arr)
        assert int(ok.sum()) == B_dsa_ver
        mu_np = np.stack([np.frombuffer(ref.mu(pk, ref.format_message(m)), np.uint8) for m in msgs])
        mu_d = torch.from_numpy(mu_np.copy()).to(dev)
        sig_d = torch.from_numpy(sig_arr.copy()).to(dev)
        t_dev, _ = _time_fn(lambda md=mu_d, sd=sig_d, v=ver: v.verify_mu(md, sd), dev, reps=9)
        t_e2e, _ = _time_fn(lambda ms=msgs, sa=sig_arr, v=ver: v.verify(ms, sa), dev, reps=7)
        add_row(
            ps,
            "verify",
            B_dsa_ver,
            t_dev / B_dsa_ver * 1e6,
            t_e2e / B_dsa_ver * 1e6,
            f"{tag_key}-verify",
        )

        # sign
        _pk_s, sk_s = ref.keygen(bytes(range(32)))
        signer = MLDSASigner(ps, sk_s, device=dev)
        sign_msgs = [b"message %08d" % i for i in range(B_dsa_sign)]
        mu_s_np = np.stack(
            [np.frombuffer(ref.mu(_pk_s, ref.format_message(m)), np.uint8) for m in sign_msgs]
        )
        mu_s_d = torch.from_numpy(mu_s_np.copy()).to(dev)
        sigs_out = signer.sign(sign_msgs[:4])
        assert sigs_out.shape == (4, ref.p.sig_size)
        t_dev, _ = _time_fn(lambda md=mu_s_d, s=signer: s.sign_mu_rows(md), dev, reps=3, warmup=1)
        t_e2e, _ = _time_fn(lambda ms=sign_msgs, s=signer: s.sign(ms), dev, reps=3, warmup=1)
        add_row(
            ps,
            "sign",
            B_dsa_sign,
            t_dev / B_dsa_sign * 1e6,
            t_e2e / B_dsa_sign * 1e6,
            f"{tag_key}-sign",
        )

    # 3. SLH-DSA (SHAKE-128f and SHA2-128f, matched same-family rivals)
    B_slh = 4096
    for ps, rival_key in (
        ("SLH-DSA-SHAKE-128f", "slhdsa-shake-128f-verify"),
        ("SLH-DSA-SHA2-128f", "slhdsa-sha2-128f-verify"),
    ):
        cases = [c for c in slh_acvp.cases(ps=ps, interface="internal") if c.expected]
        pk = cases[0].pk
        cases = [c for c in cases if c.pk == pk]
        msgs = [cases[i % len(cases)].message for i in range(B_slh)]
        sigs = np.stack(
            [np.frombuffer(cases[i % len(cases)].signature, np.uint8) for i in range(B_slh)]
        )
        v = SLHVerifier(ps, pk, device=dev)
        out = v.verify_mprime(msgs[:8], sigs[:8])
        assert out.all()
        t_e2e, _ = _time_fn(
            lambda ms=msgs, sg=sigs, v=v: v.verify_mprime(ms, sg), dev, reps=3, warmup=1
        )
        # SLH-DSA verify_mprime is dominated by thousands of device hash launches; H2D+D2H is <1%
        add_row(
            ps,
            "verify",
            B_slh,
            t_e2e / B_slh * 1e6,
            t_e2e / B_slh * 1e6,
            rival_key,
            note="mprime e2e≈dev",
        )

    # 4. Falcon-512, Falcon-padded-512, Falcon-1024, Falcon-padded-1024
    B_fal = 4096
    for label, tag_oqs, params, padded, rival_key in (
        ("Falcon-512", "falcon_512", FALCON_512, False, "falcon-512-verify"),
        (
            "Falcon-padded-512",
            "falcon_padded_512",
            FALCON_PADDED_512,
            True,
            "falcon-padded-512-verify",
        ),
        ("Falcon-1024", "falcon_1024", FALCON_1024, False, "falcon-1024-verify"),
        (
            "Falcon-padded-1024",
            "falcon_padded_1024",
            FALCON_PADDED_1024,
            True,
            "falcon-padded-1024-verify",
        ),
    ):
        pk, msgs, sigs = _oqs_sig_corpus(oqs_so, tag_oqs, params.pk_len, B_fal, 32)
        v = FalconVerifier(pk, params, dev, padded=padded)
        s2, _keep = comp_decode_batch(
            [x[1 + params.nonce_len :] for x in sigs], params, padded=padded
        )
        c = falcon_ref.hash_to_point_batch(
            [x[1 : 1 + params.nonce_len] for x in sigs], msgs, params
        )
        c_d = torch.from_numpy(c.astype(np.int32).copy()).to(dev)
        s2_d = torch.from_numpy(s2.astype(np.int32).copy()).to(dev)
        ok, _ = v.verify_core_torch(c_d, s2_d)
        assert int(ok.cpu().numpy().sum()) == B_fal
        ok_e2e = v.verify(msgs, sigs)
        assert int(ok_e2e.sum()) == B_fal
        t_dev, _ = _time_fn(lambda c=c_d, s2=s2_d, v=v: v.verify_core_torch(c, s2), dev, reps=9)
        t_e2e, _ = _time_fn(lambda ms=msgs, sg=sigs, v=v: v.verify(ms, sg), dev, reps=9, warmup=3)
        add_row(label, "verify", B_fal, t_dev / B_fal * 1e6, t_e2e / B_fal * 1e6, rival_key)

    # 5. LMS / HSS (RFC 8554 case 0; no liboqs rival -> Python oracle context)
    B_lms = 4096
    vec = json.loads((ROOT / "data/lms-rfc8554/vectors.json").read_text())[0]
    pub, msg, sig = bytes.fromhex(vec["pub"]), bytes.fromhex(vec["msg"]), bytes.fromhex(vec["sig"])
    v_lms = LMSVerifier(pub, device=dev)
    msgs_lms = [msg] * B_lms
    sigs_lms = np.stack([np.frombuffer(sig, np.uint8)] * B_lms)
    assert v_lms.verify(msgs_lms[:8], sigs_lms[:8]).all()
    t_lms, _ = _time_fn(lambda: v_lms.verify(msgs_lms, sigs_lms), dev, reps=5, warmup=2)
    t0 = time.perf_counter()
    for _ in range(32):
        assert lms_ref.hss_verify(pub, msg, sig)
    py_us = (time.perf_counter() - t0) / 32 * 1e6
    add_row(
        "LMS/HSS (case 0)",
        "verify",
        B_lms,
        t_lms / B_lms * 1e6,
        t_lms / B_lms * 1e6,
        None,
        note=f"no liboqs rival (Python hashlib ref {py_us:.0f} us/op)",
    )
    return rows


def _e2e_prim_fn(name: str, batch: int, size: int, dev):
    """Return a Host NumPy -> Device -> Host E2E callable for `name`, or None if N/A."""
    rng = np.random.default_rng(7)
    if name in ("sha3-256", "sha3-512", "shake128", "shake256"):
        hf = {
            "sha3-256": sponge.SHA3_256,
            "sha3-512": sponge.SHA3_512,
            "shake128": sponge.SHAKE128,
            "shake256": sponge.SHAKE256,
        }[name]
        outlen = hf.outlen or 32
        msgs_np = rng.integers(0, 256, (batch, size), dtype=np.uint8)

        def _fn():
            t = torch.from_numpy(msgs_np).to(dev)
            pad = sponge.pad_torch(t, hf.rate, hf.domain)
            return sponge.sponge_padded(pad, hf.rate, outlen).cpu().numpy()

        return _fn
    if name in ("sha256", "sha512"):
        bits = 256 if name == "sha256" else 512
        block_bytes = 64 if bits == 256 else 128
        dtype = np.int32 if bits == 256 else np.int64
        nblocks = max(1, size // block_bytes)
        msg_np = np.zeros((batch, nblocks * 16), dtype=dtype)
        hin = sha2.iv_tensor(bits, dev)

        def _fn():
            m = torch.from_numpy(msg_np).to(dev)
            out = torch.empty((batch, 8), dtype=m.dtype, device=dev)
            sha2.sha_absorb(hin, m, out, batch, bits, nblocks)
            return out.cpu().numpy()

        return _fn
    if name == "chacha20":
        keys = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        nonces = rng.integers(0, 256, (batch, 12), dtype=np.uint8)

        def _fn():
            return chacha20.chacha20_keystream(keys, nonces, size, device=dev)

        return _fn
    if name == "poseidon2":
        x_np = rng.integers(0, 2013265921, (batch, 16), dtype=np.int32)

        def _fn():
            return poseidon2.poseidon2_permute_torch(x_np, device=dev).cpu().numpy()

        return _fn
    if name == "poly1305":
        keys = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        msgs = rng.integers(0, 256, (batch, size), dtype=np.uint8)

        def _fn():
            return poly1305.poly1305_torch(keys, msgs, device=dev).cpu().numpy()

        return _fn
    if name == "chacha20-poly1305":
        keys = rng.integers(0, 256, (batch, 32), dtype=np.uint8)
        nonces = rng.integers(0, 256, (batch, 12), dtype=np.uint8)
        pt = rng.integers(0, 256, (batch, size), dtype=np.uint8)

        def _fn():
            ct, tag = poly1305.chacha20_poly1305_torch(keys, nonces, pt, device=dev)
            return ct.cpu().numpy(), tag.cpu().numpy()

        return _fn
    if name == "aes128-ctr":
        keys = rng.integers(0, 256, (batch, 16), dtype=np.uint8)
        nonces = rng.integers(0, 256, (batch, 8), dtype=np.uint8)
        rk = torch.as_tensor(
            np.stack([aes.aes128_expand_key_np(bytes(k)) for k in keys]).astype(np.int32),
            device=dev,
        )

        def _fn():
            return aes.aes128_ctr_torch(keys, nonces, size, device=dev, rk=rk).cpu().numpy()

        return _fn
    if name == "blake3":
        msgs = rng.integers(0, 256, (batch, size), dtype=np.uint8)

        def _fn():
            return blake3.blake3_hash_torch(msgs, device=dev).cpu().numpy()

        return _fn
    if name == "gemm-int8":
        a = rng.integers(-128, 128, (1024, 1024), dtype=np.int8)
        b = rng.integers(-128, 128, (1024, 1024), dtype=np.int8)

        def _fn():
            return gemm.gemm_int8_torch(a, b, device=dev).cpu().numpy()

        return _fn
    if name == "ntt-2^20":
        lntt = LargeNTT(1 << 20, 2013265921)
        x = rng.integers(0, 2013265921, 1 << 20, dtype=np.int32)

        def _fn():
            return lntt.forward_torch(x, device=dev).cpu().numpy()

        return _fn
    return None


def _bench_primitives(dev, rivals: dict, batch: int = 65536, size: int = 1024, reps: int = 7):
    copy_t, _ = measure_prim(bench_copy(batch, size, dev), dev, reps)
    copy_bw = 2 * (1 << 26) * 4 / copy_t / 1e9
    rows = []
    for name in PRIM_ORDER:
        bench = PRIM_CASES[name](batch, size, dev)
        t_dev, t_dev_med = (copy_t, copy_t) if name == "copy" else measure_prim(bench, dev, reps)
        e2e_fn = _e2e_prim_fn(name, batch, size, dev)
        t_e2e = None
        if e2e_fn is not None:
            t_e2e, _ = _time_fn(e2e_fn, dev, reps=max(3, reps // 2), warmup=2)

        rival_key = PRIM_CPU_RIVAL.get(name)
        rv = rivals.get(rival_key) if rival_key else None
        if rv is not None:
            is_bytes = rv["unit"] == "B/s"
            dev_rate = (bench.nbytes / t_dev) if is_bytes else (bench.units / t_dev)
            e2e_rate = (
                ((bench.nbytes / t_e2e) if is_bytes else (bench.units / t_e2e))
                if t_e2e is not None
                else None
            )
            dev_1t = dev_rate / rv["one_thread"]
            dev_16p = dev_rate / rv["procs16"]
            e2e_1t = (e2e_rate / rv["one_thread"]) if e2e_rate else None
            e2e_16p = (e2e_rate / rv["procs16"]) if e2e_rate else None
        else:
            dev_1t = dev_16p = e2e_1t = e2e_16p = None

        gbs = bench.nbytes / t_dev / 1e9
        rows.append(
            {
                "name": name,
                "note": bench.note,
                "dev_ms": t_dev * 1e3,
                "dev_med_ms": t_dev_med * 1e3,
                "e2e_ms": None if t_e2e is None else t_e2e * 1e3,
                "gbs": gbs,
                "pct_copy": 100.0 * gbs / copy_bw,
                "dev_vs_1t": dev_1t,
                "dev_vs_16p": dev_16p,
                "e2e_vs_1t": e2e_1t,
                "e2e_vs_16p": e2e_16p,
                "rival": rival_key,
            }
        )
        print(
            f"  {name:18s} dev={t_dev * 1e3:7.3f}ms  "
            f"e2e={t_e2e * 1e3 if t_e2e else -1:7.3f}ms  "
            f"dev/16P={f'{dev_16p:.2f}x' if dev_16p else 'n/a':>7s}  "
            f"e2e/16P={f'{e2e_16p:.2f}x' if e2e_16p else 'n/a':>7s}",
            flush=True,
        )
    return copy_bw, rows


def _bench_breakeven(dev, rivals: dict, oqs_so: Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}

    # 1. ML-KEM-768 decaps
    dc = next(
        c
        for c in mlkem_acvp.load()
        if c["function"] == "decapsulation" and c["parameterSet"] == "ML-KEM-768"
    )
    dec = Decapsulator("ML-KEM-768", bytes.fromhex(dc["inputs"]["dk"]), device=dev)
    ct1 = bytes.fromhex(dc["inputs"]["c"])
    rv_kem = rivals["mlkem-768-decaps"]
    rows_kem = []
    for B in (16, 64, 256, 1024, 4096):
        cts = [ct1] * B
        ct_t = torch.from_numpy(
            np.frombuffer(b"".join(cts), np.uint8).reshape(B, len(ct1)).copy()
        ).to(dev)
        t_dev, _ = _time_fn(lambda ct=ct_t, B=B: dec.decaps_torch(ct, B), dev, reps=9)
        t_e2e, _ = _time_fn(lambda cts=cts: dec.decaps(cts), dev, reps=9)
        rows_kem.append(
            {
                "batch": B,
                "dev_us": t_dev / B * 1e6,
                "e2e_us": t_e2e / B * 1e6,
                "dev_vs_1t": (B / t_dev) / rv_kem["one_thread"],
                "dev_vs_16p": (B / t_dev) / rv_kem["procs16"],
                "e2e_vs_1t": (B / t_e2e) / rv_kem["one_thread"],
                "e2e_vs_16p": (B / t_e2e) / rv_kem["procs16"],
            }
        )
    out["ML-KEM-768 decaps"] = rows_kem

    # 2. ML-DSA-44 verify
    pk, msgs, sigs = _oqs_sig_corpus(oqs_so, "ml_dsa_44", 1312, 4096, 32)
    ver = MLDSAVerifier("ML-DSA-44", pk, device=dev)
    ref = MLDSA("ML-DSA-44")
    sig_all = np.stack([np.frombuffer(s, np.uint8) for s in sigs])
    mu_all = np.stack([np.frombuffer(ref.mu(pk, ref.format_message(m)), np.uint8) for m in msgs])
    rv_dsa = rivals["mldsa-44-verify"]
    rows_dsa = []
    for B in (16, 64, 256, 1024, 4096):
        mu_d = torch.from_numpy(mu_all[:B].copy()).to(dev)
        sig_d = torch.from_numpy(sig_all[:B].copy()).to(dev)
        t_dev, _ = _time_fn(lambda md=mu_d, sd=sig_d: ver.verify_mu(md, sd), dev, reps=9)
        t_e2e, _ = _time_fn(lambda B=B: ver.verify(msgs[:B], sig_all[:B]), dev, reps=7)
        rows_dsa.append(
            {
                "batch": B,
                "dev_us": t_dev / B * 1e6,
                "e2e_us": t_e2e / B * 1e6,
                "dev_vs_1t": (B / t_dev) / rv_dsa["one_thread"],
                "dev_vs_16p": (B / t_dev) / rv_dsa["procs16"],
                "e2e_vs_1t": (B / t_e2e) / rv_dsa["one_thread"],
                "e2e_vs_16p": (B / t_e2e) / rv_dsa["procs16"],
            }
        )
    out["ML-DSA-44 verify"] = rows_dsa

    # 3. Falcon-512 verify
    pk_f, msgs_f, sigs_f = _oqs_sig_corpus(oqs_so, "falcon_512", FALCON_512.pk_len, 4096, 32)
    ver_f = FalconVerifier(pk_f, FALCON_512, dev, padded=False)
    s2_f, _ = comp_decode_batch([x[1 + FALCON_512.nonce_len :] for x in sigs_f], FALCON_512, False)
    c_f = falcon_ref.hash_to_point_batch(
        [x[1 : 1 + FALCON_512.nonce_len] for x in sigs_f], msgs_f, FALCON_512
    )
    rv_f = rivals["falcon-512-verify"]
    rows_f = []
    for B in (16, 64, 256, 1024, 4096):
        c_d = torch.from_numpy(c_f[:B].astype(np.int32).copy()).to(dev)
        s2_d = torch.from_numpy(s2_f[:B].astype(np.int32).copy()).to(dev)
        t_dev, _ = _time_fn(lambda cd=c_d, sd=s2_d: ver_f.verify_core_torch(cd, sd), dev, reps=9)
        t_e2e, _ = _time_fn(lambda B=B: ver_f.verify(msgs_f[:B], sigs_f[:B]), dev, reps=9)
        rows_f.append(
            {
                "batch": B,
                "dev_us": t_dev / B * 1e6,
                "e2e_us": t_e2e / B * 1e6,
                "dev_vs_1t": (B / t_dev) / rv_f["one_thread"],
                "dev_vs_16p": (B / t_dev) / rv_f["procs16"],
                "e2e_vs_1t": (B / t_e2e) / rv_f["one_thread"],
                "e2e_vs_16p": (B / t_e2e) / rv_f["procs16"],
            }
        )
    out["Falcon-512 verify"] = rows_f

    # 4. SHA3-256 (1 KiB) & 5. ChaCha20 (1 KiB)
    for label, prim_name, rival_key in (
        ("SHA3-256 (1 KiB)", "sha3-256", "sha3-256"),
        ("ChaCha20 (1 KiB)", "chacha20", "chacha20"),
    ):
        rv = rivals[rival_key]
        rows_p = []
        for B in (16, 64, 256, 1024, 4096, 16384, 65536):
            bench = PRIM_CASES[prim_name](B, 1024, dev)
            t_dev, _ = measure_prim(bench, dev, reps=7)
            e2e_fn = _e2e_prim_fn(prim_name, B, 1024, dev)
            t_e2e, _ = _time_fn(e2e_fn, dev, reps=5, warmup=2)
            rows_p.append(
                {
                    "batch": B,
                    "dev_us": t_dev / B * 1e6,
                    "e2e_us": t_e2e / B * 1e6,
                    "dev_vs_1t": (bench.nbytes / t_dev) / rv["one_thread"],
                    "dev_vs_16p": (bench.nbytes / t_dev) / rv["procs16"],
                    "e2e_vs_1t": (bench.nbytes / t_e2e) / rv["one_thread"],
                    "e2e_vs_16p": (bench.nbytes / t_e2e) / rv["procs16"],
                }
            )
        out[label] = rows_p

    return out


def _render_md(
    meta: dict, schemes: list[dict], copy_bw: float, prims: list[dict], brk: dict
) -> str:
    def f_us(v):
        if v is None:
            return "n/a"
        return f"{v:.3f}" if v < 10 else f"{v:.2f}" if v < 100 else f"{v:.1f}"

    def f_x(v):
        if v is None:
            return "n/a"
        return f"{v:.1f}×" if v >= 10 else f"{v:.2f}×"

    host_cpu = meta.get("host_cpu") or meta.get("host_arch", "unknown")
    host_nproc = meta.get("host_nproc")
    base_cpu = meta.get("cpu_baseline_cpu") or meta.get("cpu", "unknown")
    base_nproc = meta.get("cpu_baseline_nproc") or meta.get("nproc")
    base_host = meta.get("cpu_baseline_host", "unknown")
    lines = [
        f"# GPU vs CPU Full Benchmark — {meta['timestamp_utc']}",
        "",
        f"- **Host**: `{meta['host']}` (`{host_cpu}`, {host_nproc} logical CPUs)",
        f"- **GPU**: `{meta['gpu']}` (`torch {meta['torch']}`, `triton {meta['triton']}`)",
        f"- **Copy bandwidth (same window)**: **{copy_bw:.1f} GB/s** (read+write)",
        f"- **Git HEAD**: `{meta['git_head']}`{' (dirty)' if meta['git_dirty'] else ''}",
        f"- **CPU baseline**: `{meta['cpu_baseline_file']}` (`{base_cpu}`, {base_nproc} logical CPUs on `{base_host}`, `{meta['cpu_baseline_method']}`, timestamp `{meta['cpu_timestamp']}`)",
        f"- **Power state**: `{json.dumps(meta['power'], sort_keys=True)}`",
        "",
        "## Part A — Schemes × Parameter Sets × Operations (Dev + E2E vs 1T / 16P)",
        "",
        "| Scheme | Op | B | GPU dev (µs/op) | GPU e2e (µs/op) | CPU 1T (µs/op) | CPU 16P (µs/op) | dev vs 1T | dev vs 16P | e2e vs 1T | e2e vs 16P | Note |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in schemes:
        lines.append(
            f"| {r['scheme']} | {r['op']} | {r['batch']} | {f_us(r['dev_us'])} | {f_us(r['e2e_us'])} | "
            f"{f_us(r['cpu1_us'])} | {f_us(r['cpu16_us'])} | {f_x(r['dev_vs_1t'])} | {f_x(r['dev_vs_16p'])} | "
            f"{f_x(r['e2e_vs_1t'])} | {f_x(r['e2e_vs_16p'])} | {r['note']} |"
        )

    lines += [
        "",
        "## Part B — Scheme-Agnostic Primitives (B=65536, size=1024 B; Dev + E2E vs 1T / 16P)",
        "",
        "| Primitive | What | dev ms | e2e ms | GB/s | %copy | dev vs 1T | dev vs 16P | e2e vs 1T | e2e vs 16P |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in prims:
        e2e_s = f"{r['e2e_ms']:.3f}" if r["e2e_ms"] is not None else "n/a"
        lines.append(
            f"| `{r['name']}` | {r['note']} | {r['dev_ms']:.3f} | {e2e_s} | {r['gbs']:.1f} | {r['pct_copy']:.0f}% | "
            f"{f_x(r['dev_vs_1t'])} | {f_x(r['dev_vs_16p'])} | {f_x(r['e2e_vs_1t'])} | {f_x(r['e2e_vs_16p'])} |"
        )

    lines += [
        "",
        "## Part C — Batch Size Breakeven Sweep (GPU vs CPU 1T / 16P)",
        "",
    ]
    for label, rows in brk.items():
        dev_1t = next((r["batch"] for r in rows if r["dev_vs_1t"] >= 1.0), None)
        dev_16p = next((r["batch"] for r in rows if r["dev_vs_16p"] >= 1.0), None)
        e2e_1t = next((r["batch"] for r in rows if r["e2e_vs_1t"] >= 1.0), None)
        e2e_16p = next((r["batch"] for r in rows if r["e2e_vs_16p"] >= 1.0), None)
        lines += [
            f"### {label}",
            "",
            f"- Breakeven (≥1.0×): **dev vs 1T** `B={dev_1t}`, **dev vs 16P** `B={dev_16p}`, **e2e vs 1T** `B={e2e_1t}`, **e2e vs 16P** `B={e2e_16p}`",
            "",
            "| B | GPU dev (µs/op) | GPU e2e (µs/op) | dev vs 1T | dev vs 16P | e2e vs 1T | e2e vs 16P |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for r in rows:
            lines.append(
                f"| {r['batch']} | {f_us(r['dev_us'])} | {f_us(r['e2e_us'])} | "
                f"{f_x(r['dev_vs_1t'])} | {f_x(r['dev_vs_16p'])} | {f_x(r['e2e_vs_1t'])} | {f_x(r['e2e_vs_16p'])} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--baseline", type=Path, default=BENCH / "cpu_baseline_barrier.json")
    ap.add_argument("--out", type=Path, default=BENCH / "local-gpu-vs-cpu-20261128.md")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cpu_data = json.loads(args.baseline.read_text())
    rivals = cpu_data["rivals"]
    cpu_meta = cpu_data["meta"]
    dev = devices.resolve_device(args.device)
    oqs_so = _oqs_so()

    print(f"# GPU: {devices.describe(dev)} | baseline: {args.baseline.name} ({len(rivals)} rivals)")
    print("# Part A: Schemes ...", flush=True)
    schemes = _bench_schemes(dev, rivals, oqs_so)
    print("# Part B: Primitives ...", flush=True)
    copy_bw, prims = _bench_primitives(dev, rivals)
    print("# Part C: Breakeven sweep ...", flush=True)
    brk = _bench_breakeven(dev, rivals, oqs_so)

    def _local_cpu_model() -> str:
        try:
            for ln in Path("/proc/cpuinfo").read_text().splitlines():
                if ln.startswith("model name"):
                    return ln.split(":", 1)[1].strip()
        except OSError:
            pass
        return platform.processor() or platform.machine()

    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "host_arch": platform.machine(),
        "host_cpu": _local_cpu_model(),
        "host_nproc": os.cpu_count(),
        "cpu_baseline_host": cpu_meta.get("host", "unknown"),
        "cpu_baseline_cpu": cpu_meta.get("cpu", "unknown"),
        "cpu_baseline_nproc": cpu_meta.get("nproc"),
        "gpu": devices.describe(dev),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "git_head": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "cpu_baseline_file": f"benchmarks/{args.baseline.name}",
        "cpu_baseline_method": cpu_meta.get("method", "unknown"),
        "cpu_timestamp": cpu_meta.get("timestamp", "unknown"),
        "power": power_state(),
        "copy_bw_gbs": round(copy_bw, 2),
    }

    md = _render_md(meta, schemes, copy_bw, prims, brk)
    args.out.write_text(md)
    json_out = args.out.with_suffix(".json")
    json_out.write_text(
        json.dumps(
            {"meta": meta, "schemes": schemes, "primitives": prims, "breakeven": brk}, indent=2
        )
        + "\n"
    )
    print(f"\nwrote {args.out}\nwrote {json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
