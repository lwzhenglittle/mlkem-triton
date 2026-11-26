"""The LM-OTS chain, fused: every chain of every signature in one launch, one loop inside.

An LMS signature verification is p Winternitz chains of up to 255 steps, and the step count is
data-dependent (`a = coef(Q || Cksm(Q), i, w)`, so a chain stops after `2^w - 1 - a` steps). Two
things follow that shape this kernel:

* each step is one *single-block* SHA-256 compression — the message is
  `I(16) || u32(q) || u16(i) || u8(j) || value(32)`, 55 bytes, so no prefix state is needed;
* a fixed 255-step loop would be 33x the average work for w=8 (127.5), so rows are sorted by
  remaining steps and the kernel exits per block once its rows are done.

The value is carried as eight 32-bit words; the block words are shifted by 3 bytes because the
value starts at byte 23.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

from ..primitives.sha2 import _k_tensor

KERNELS: dict[int, object] = {}


def gen_source(n_rows_hint: int = 0) -> str:
    L = [
        "import triton",
        "import triton.language as tl",
        "",
        "@triton.jit",
        "def _bs32(x):",
        "    x = x.to(tl.uint32)",
        "    return ((x & 0xFF) << 24) | ((x & 0xFF00) << 8) | ((x >> 8) & 0xFF00) | ((x >> 24) & 0xFF)",
        "",
        "@triton.jit",
        "def _r(x, r):",
        "    x = x.to(tl.uint32)",
        "    return (x >> r) | (x << (32 - r))",
        "",
        "@triton.jit",
        "def lms_chain_kernel(VALUE, IW, QW, CIDX, START, KT, IV, rows, STEPS,",
        "                     BLOCK: tl.constexpr):",
        "    pid = tl.program_id(0)",
        "    row = pid * BLOCK + tl.arange(0, BLOCK)",
        "    m = row < rows",
        "    start = tl.load(START + row, mask=m, other=0).to(tl.int32)",
        "    cidx = tl.load(CIDX + row, mask=m, other=0).to(tl.uint32)",
        "    qw = tl.load(QW + row, mask=m, other=0).to(tl.uint32)",
    ]
    for k in range(4):
        L.append(
            f"    iw{k} = _bs32(tl.load(IW + row * 4 + {k}, mask=m, other=0).to(tl.uint32,"
            f" bitcast=True))"
        )
    for k in range(8):
        L.append(
            f"    v{k} = _bs32(tl.load(VALUE + row * 8 + {k}, mask=m, other=0).to(tl.uint32,"
            f" bitcast=True))"
        )
    L.append("    zero_v = tl.zeros((BLOCK,), dtype=tl.uint32)")
    L.append("    j = start")
    L.append("    while tl.max((j < STEPS).to(tl.int32)) > 0:")
    L.append("        active = (j < STEPS) & m")
    L.append("        jb = tl.where(active, j, 0).to(tl.uint32)")
    L.append("        w0 = iw0")
    L.append("        w1 = iw1")
    L.append("        w2 = iw2")
    L.append("        w3 = iw3")
    L.append("        w4 = qw")
    L.append("        w5 = (cidx << 16) | (jb << 8) | (v0 >> 24)")
    for k in range(1, 8):
        L.append(f"        w{5 + k} = (v{k - 1} << 8) | (v{k} >> 24)")
    L.append("        w13 = (v7 << 8) | tl.full((BLOCK,), 0x80, tl.uint32)")
    L.append("        w14 = tl.zeros((BLOCK,), dtype=tl.uint32)")
    L.append("        w15 = tl.full((BLOCK,), 440, tl.uint32)")
    L.append("        iv0 = tl.load(IV + 0).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a0 = iv0")
    L.append("        iv1 = tl.load(IV + 1).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a1 = iv1")
    L.append("        iv2 = tl.load(IV + 2).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a2 = iv2")
    L.append("        iv3 = tl.load(IV + 3).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a3 = iv3")
    L.append("        iv4 = tl.load(IV + 4).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a4 = iv4")
    L.append("        iv5 = tl.load(IV + 5).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a5 = iv5")
    L.append("        iv6 = tl.load(IV + 6).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a6 = iv6")
    L.append("        iv7 = tl.load(IV + 7).to(tl.uint32, bitcast=True) | zero_v")
    L.append("        a7 = iv7")
    L.append("        for _t in range(0, 64, 1):")
    L.append("            kt = tl.load(KT + _t).to(tl.uint32)")
    L.append("            S1 = _r(a4, 6) ^ _r(a4, 11) ^ _r(a4, 25)")
    L.append("            ch = (a4 & a5) ^ (~a4 & a6)")
    L.append("            t1 = a7 + S1 + ch + kt + w0")
    L.append("            S0 = _r(a0, 2) ^ _r(a0, 13) ^ _r(a0, 22)")
    L.append("            maj = (a0 & a1) ^ (a0 & a2) ^ (a1 & a2)")
    L.append("            t2 = S0 + maj")
    for i in range(7, 0, -1):
        L.append(f"            a{i} = a{i - 1}" if i != 4 else "            a4 = a3 + t1")
    L.append("            a0 = t1 + t2")
    L.append(
        "            nw = w0 + (_r(w1, 7) ^ _r(w1, 18) ^ (w1 >> 3)) + w9"
        " + (_r(w14, 17) ^ _r(w14, 19) ^ (w14 >> 10))"
    )
    for i in range(15):
        L.append(f"            w{i} = w{i + 1}")
    L.append("            w15 = nw")
    for k in range(8):
        L.append(f"        v{k} = tl.where(active, a{k} + iv{k}, v{k})")  # feed-forward
    L.append("        j = j + 1")
    for k in range(8):
        L.append(
            f"    tl.store(VALUE + row * 8 + {k}, _bs32(v{k}).to(tl.int32, bitcast=True), mask=m)"
        )
    return "\n".join(L) + "\n"


def _compile():
    if "k" in KERNELS:
        return KERNELS["k"]
    src = gen_source()
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_lms_chain_{tag}.py"
    if not path.exists():
        path.write_text(src)
    spec = importlib.util.spec_from_file_location(f"_lms_chain_{tag}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    KERNELS["k"] = mod.lms_chain_kernel
    return mod.lms_chain_kernel


def chains(value, ident_words, q_words, cidx, start, w: int, block: int = 128):
    """Advance every (signature, chain) row from `start` to `w-1`, in place.

    `value` is [rows, 8] int32 (the LM-OTS `y[i]` words, updated in place), `ident_words` [rows, 4],
    `q_words`/`cidx`/`start` are [rows].
    """
    import triton

    rows = value.shape[0]
    kern = _compile()
    kern[(triton.cdiv(rows, block),)](
        value,
        ident_words,
        q_words,
        cidx,
        start,
        _k_tensor(256, value.device),
        _iv(value.device),
        rows,
        (1 << w) - 1,
        BLOCK=block,
        num_warps=4,
    )


_IV_CACHE: dict = {}


def _iv(device):
    key = str(device)
    if key not in _IV_CACHE:
        from ..primitives.sha2 import IV256, word_tensor

        _IV_CACHE[key] = word_tensor(IV256, 256, device)
    return _IV_CACHE[key]
