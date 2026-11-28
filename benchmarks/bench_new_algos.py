#!/usr/bin/env python
"""Same-window GPU (Dev + E2E) vs 1T / 16P barrier CPU benchmark for SM3, Ascon, Ed25519/X25519, MAYO, and UOV."""

from __future__ import annotations

import argparse
import ctypes
import json
import multiprocessing as mp
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triton_crypto import devices
from triton_crypto.ed25519 import (
    Verifier as Ed25519Verifier,
)
from triton_crypto.ed25519 import (
    ed25519_public_key_np,
    ed25519_sign_np,
    x25519_base_torch,
)
from triton_crypto.mayo import Verifier as MayoVerifier
from triton_crypto.primitives import ascon, sm3
from triton_crypto.uov import Verifier as UovVerifier

BIN = ROOT / "benchmarks" / "new_algos_cpu"
LIBOQS_MQ = ROOT / "third_party" / "liboqs" / "build-mq" / "lib" / "liboqs.so"


def _worker(args):
    barrier, algo, sz, iters, q = args
    barrier.wait()
    t0 = time.perf_counter()
    p = subprocess.run(
        [str(BIN), algo, str(sz), str(iters)], capture_output=True, text=True, check=True
    )
    t1 = time.perf_counter()
    row = json.loads(p.stdout)
    q.put((t0, t1, row["secs"], row["iters"]))


def bench_cpu_barrier(
    algo: str, sz: int, iters_1t: int, procs: int = 16, reps: int = 2
) -> tuple[float, float]:
    """Return `(rate_1t, rate_16p)` in ops/s (or B/s when multiplied by `sz` by caller)."""
    best_1t = 0.0
    for _ in range(reps):
        p = subprocess.run(
            [str(BIN), algo, str(sz), str(iters_1t)], capture_output=True, text=True, check=True
        )
        row = json.loads(p.stdout)
        best_1t = max(best_1t, row["iters"] / row["secs"])

    iters_p = max(1, iters_1t // 2)
    best_16p = 0.0
    ctx = mp.get_context("spawn")
    for _ in range(reps):
        barrier = ctx.Barrier(procs)
        q = ctx.Queue()
        ps = [
            ctx.Process(target=_worker, args=((barrier, algo, sz, iters_p, q),))
            for _ in range(procs)
        ]
        for proc in ps:
            proc.start()
        for proc in ps:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError(f"CPU worker failed for {algo}")
        res = [q.get() for _ in range(procs)]
        t_start = min(r[0] for r in res)
        t_end = max(r[1] for r in res)
        total_ops = sum(r[3] for r in res)
        best_16p = max(best_16p, total_ops / (t_end - t_start))
    return best_1t, best_16p


def time_gpu(fn, dev, reps: int = 5) -> float:
    fn()
    devices.sync(dev)
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        devices.sync(dev)
        best = min(best, time.perf_counter() - t0)
    return best


def _gen_oqs_sample(alg: str, msg: bytes):
    lib = ctypes.CDLL(str(LIBOQS_MQ))

    class _OQS_SIG(ctypes.Structure):
        _fields_ = [
            ("method_name", ctypes.c_char_p),
            ("alg_version", ctypes.c_char_p),
            ("claimed_nist_level", ctypes.c_uint8),
            ("euf_cma", ctypes.c_bool),
            ("suf_cma", ctypes.c_bool),
            ("sig_with_ctx_support", ctypes.c_bool),
            ("length_public_key", ctypes.c_size_t),
            ("length_secret_key", ctypes.c_size_t),
            ("length_signature", ctypes.c_size_t),
        ]

    lib.OQS_SIG_new.argtypes = [ctypes.c_char_p]
    lib.OQS_SIG_new.restype = ctypes.POINTER(_OQS_SIG)
    lib.OQS_SIG_keypair.argtypes = [ctypes.POINTER(_OQS_SIG), ctypes.c_char_p, ctypes.c_char_p]
    lib.OQS_SIG_sign.argtypes = [
        ctypes.POINTER(_OQS_SIG),
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_char_p,
    ]
    lib.OQS_SIG_free.argtypes = [ctypes.POINTER(_OQS_SIG)]
    obj = lib.OQS_SIG_new(alg.encode())
    s = obj.contents
    pk = ctypes.create_string_buffer(s.length_public_key)
    sk = ctypes.create_string_buffer(s.length_secret_key)
    sig = ctypes.create_string_buffer(s.length_signature)
    siglen = ctypes.c_size_t(0)
    assert lib.OQS_SIG_keypair(obj, pk, sk) == 0
    assert lib.OQS_SIG_sign(obj, sig, ctypes.byref(siglen), msg, len(msg), sk) == 0
    out = (pk.raw, sig.raw[: siglen.value])
    lib.OQS_SIG_free(obj)
    return out


def main() -> int:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    dev = devices.resolve_device(args.device)
    print(f"Device: {devices.describe(dev)} | CPU barrier procs: {args.procs}")
    print()

    rows = []
    rng = np.random.default_rng(42)

    # 1. Byte-oriented primitives at B=65536, 1 KiB messages
    B_prim = 65536
    SZ = 1024
    msgs_np = rng.integers(0, 256, (B_prim, SZ), dtype=np.uint8)
    msgs_dev = torch.from_numpy(msgs_np).to(dev)
    keys_np = rng.integers(0, 256, (B_prim, 16), dtype=np.uint8)
    nonces_np = rng.integers(0, 256, (B_prim, 16), dtype=np.uint8)
    keys_dev = torch.from_numpy(keys_np).to(dev)
    nonces_dev = torch.from_numpy(nonces_np).to(dev)

    # SM3
    sm3_pad = sm3.sm3_pad_torch(msgs_dev)
    t_dev = time_gpu(lambda: sm3.sm3_hash_torch(sm3_pad, device=dev, pre_padded=True), dev)
    t_e2e = time_gpu(lambda: sm3.sm3_hash_torch(msgs_np, device=dev).cpu(), dev)
    c1, c16 = bench_cpu_barrier("sm3", SZ, 150000, args.procs)
    rows.append(
        (
            "sm3 (1 KiB)",
            "GB/s",
            B_prim * SZ / t_dev / 1e9,
            B_prim * SZ / t_e2e / 1e9,
            c1 * SZ / 1e9,
            c16 * SZ / 1e9,
        )
    )

    # Ascon-Hash256
    ascon_nb = (SZ + 1 + 7) // 8
    ascon_pad = torch.zeros((B_prim, ascon_nb * 8), dtype=torch.uint8, device=dev)
    ascon_pad[:, :SZ] = msgs_dev
    ascon_pad[:, SZ] = 0x01
    t_dev = time_gpu(lambda: ascon.ascon_hash256_torch(ascon_pad, device=dev, pre_padded=True), dev)
    t_e2e = time_gpu(lambda: ascon.ascon_hash256_torch(msgs_np, device=dev).cpu(), dev)
    c1, c16 = bench_cpu_barrier("ascon-hash256", SZ, 60000, args.procs)
    rows.append(
        (
            "ascon-hash256 (1 KiB)",
            "GB/s",
            B_prim * SZ / t_dev / 1e9,
            B_prim * SZ / t_e2e / 1e9,
            c1 * SZ / 1e9,
            c16 * SZ / 1e9,
        )
    )

    # Ascon-AEAD128 (SP 800-232) & Ascon-128a (v1.2)
    for name, fn in (
        ("ascon-aead128", ascon.ascon_aead128_torch),
        ("ascon-128a", ascon.ascon128a_torch),
    ):
        t_dev = time_gpu(lambda f=fn: f(keys_dev, nonces_dev, msgs_dev, device=dev), dev)
        t_e2e = time_gpu(
            lambda f=fn: [x.cpu() for x in f(keys_np, nonces_np, msgs_np, device=dev)], dev
        )
        c1, c16 = bench_cpu_barrier(name, SZ, 120000, args.procs)
        rows.append(
            (
                f"{name} (1 KiB)",
                "GB/s",
                B_prim * SZ / t_dev / 1e9,
                B_prim * SZ / t_e2e / 1e9,
                c1 * SZ / 1e9,
                c16 * SZ / 1e9,
            )
        )

    # 2. X25519 & Ed25519 verify at B=4096
    B_ec = 4096
    sc_np = rng.integers(0, 256, (B_ec, 32), dtype=np.uint8)
    sc_dev = torch.from_numpy(sc_np).to(dev)
    t_dev = time_gpu(lambda: x25519_base_torch(sc_dev, device=dev), dev)
    t_e2e = time_gpu(lambda: x25519_base_torch(sc_np, device=dev).cpu(), dev)
    c1, c16 = bench_cpu_barrier("x25519", 32, 15000, args.procs)
    rows.append(
        ("x25519 (B=4096)", "kops/s", B_ec / t_dev / 1e3, B_ec / t_e2e / 1e3, c1 / 1e3, c16 / 1e3)
    )

    seed = rng.integers(0, 256, 32, dtype=np.uint8).tobytes()
    pk_ed = ed25519_public_key_np(seed)
    msg64 = b"x" * 64
    sig_ed = ed25519_sign_np(seed, msg64)
    v_ed = Ed25519Verifier(pk_ed, device=dev)
    msgs_ed_list = [msg64] * B_ec
    sigs_ed_list = [sig_ed] * B_ec
    msgs_ed_dev = (
        torch.from_numpy(np.frombuffer(msg64, dtype=np.uint8).copy())
        .unsqueeze(0)
        .expand(B_ec, 64)
        .contiguous()
        .to(dev)
    )
    sigs_ed_dev = (
        torch.from_numpy(np.frombuffer(sig_ed, dtype=np.uint8).copy())
        .unsqueeze(0)
        .expand(B_ec, 64)
        .contiguous()
        .to(dev)
    )
    t_dev = time_gpu(lambda: v_ed.verify_torch(msgs_ed_dev, sigs_ed_dev), dev)
    t_e2e = time_gpu(lambda: v_ed.verify(msgs_ed_list, sigs_ed_list), dev)
    c1, c16 = bench_cpu_barrier("ed25519-verify", 64, 8000, args.procs)
    rows.append(
        (
            "ed25519-verify (B=4096)",
            "kops/s",
            B_ec / t_dev / 1e3,
            B_ec / t_e2e / 1e3,
            c1 / 1e3,
            c16 / 1e3,
        )
    )

    # 3. MAYO-1 / MAYO-2 / MAYO-3 / MAYO-5 verify at B=4096
    B_mq = 4096
    for alg, iters_c in (("MAYO-1", 4000), ("MAYO-2", 6000), ("MAYO-3", 2000), ("MAYO-5", 1000)):
        pk_m, sig_m = _gen_oqs_sample(alg, msg64)
        vm = MayoVerifier(pk_m, device=dev)
        m_list = [msg64] * B_mq
        s_list = [sig_m] * B_mq
        m_dev = (
            torch.from_numpy(np.frombuffer(msg64, dtype=np.uint8).copy())
            .unsqueeze(0)
            .expand(B_mq, 64)
            .contiguous()
            .to(dev)
        )
        s_dev = (
            torch.from_numpy(np.frombuffer(sig_m, dtype=np.uint8).copy())
            .unsqueeze(0)
            .expand(B_mq, len(sig_m))
            .contiguous()
            .to(dev)
        )
        t_dev = time_gpu(lambda v=vm, md=m_dev, sd=s_dev: v.verify_torch(md, sd), dev)
        t_e2e = time_gpu(lambda v=vm, ml=m_list, sl=s_list: v.verify(ml, sl), dev)
        c1, c16 = bench_cpu_barrier(alg, 64, iters_c, args.procs)
        rows.append(
            (
                f"{alg} verify (B=4096)",
                "kops/s",
                B_mq / t_dev / 1e3,
                B_mq / t_e2e / 1e3,
                c1 / 1e3,
                c16 / 1e3,
            )
        )

    # 4. UOV OV-Is / OV-Ip / OV-III / OV-V verify at B=4096
    for alg, iters_c in (("OV-Is", 4000), ("OV-Ip", 4000), ("OV-III", 1500), ("OV-V", 800)):
        pk_u, sig_u = _gen_oqs_sample(alg, msg64)
        vu = UovVerifier(pk_u, device=dev)
        m_list = [msg64] * B_mq
        s_list = [sig_u] * B_mq
        m_dev = (
            torch.from_numpy(np.frombuffer(msg64, dtype=np.uint8).copy())
            .unsqueeze(0)
            .expand(B_mq, 64)
            .contiguous()
            .to(dev)
        )
        s_dev = (
            torch.from_numpy(np.frombuffer(sig_u, dtype=np.uint8).copy())
            .unsqueeze(0)
            .expand(B_mq, len(sig_u))
            .contiguous()
            .to(dev)
        )
        t_dev = time_gpu(lambda v=vu, md=m_dev, sd=s_dev: v.verify_torch(md, sd), dev)
        t_e2e = time_gpu(lambda v=vu, ml=m_list, sl=s_list: v.verify(ml, sl), dev)
        c1, c16 = bench_cpu_barrier(alg, 64, iters_c, args.procs)
        rows.append(
            (
                f"{alg} verify (B=4096)",
                "kops/s",
                B_mq / t_dev / 1e3,
                B_mq / t_e2e / 1e3,
                c1 / 1e3,
                c16 / 1e3,
            )
        )

    hdr = f"| {'workload':26s} | {'unit':6s} | {'GPU dev':>9s} | {'GPU e2e':>9s} | {'CPU 1T':>9s} | {'CPU 16P':>9s} | {'dev/16P':>8s} | {'e2e/16P':>8s} |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|"
    print(hdr)
    print(sep)
    out_json = []
    for name, unit, gd, ge, c1, c16 in rows:
        print(
            f"| {name:26s} | {unit:6s} | {gd:9.2f} | {ge:9.2f} | {c1:9.2f} | {c16:9.2f} | {gd / c16:7.2f}x | {ge / c16:7.2f}x |"
        )
        out_json.append(
            {
                "name": name,
                "unit": unit,
                "gpu_dev": gd,
                "gpu_e2e": ge,
                "cpu_1t": c1,
                "cpu_16p": c16,
                "dev_vs_16p": gd / c16,
                "e2e_vs_16p": ge / c16,
            }
        )
    if args.json:
        Path(args.json).write_text(json.dumps(out_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
