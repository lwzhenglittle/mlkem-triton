"""Batched SHAKE256 / SHA3-512 sponge in Triton (code generation).

Mapping: one thread = one sponge instance, each of the 25 lanes is a `[BLOCK]`
uint64 tensor. Input is pre-padded on the host side (pad10*1 + domain
separation); the GPU side only does a whole-block absorb XOR + 24-round permute
+ single-block squeeze (all our uses have output <= rate).

**The 24 rounds run in a runtime for loop, with round constants read from a
24-element uint64 table** (see "why rolling" below). After rolling, the kernel's
static instruction count drops from 15254 to 790, per-thread scratch spill drops
from 3584 B to 0, throughput at rate=136/out=128 improves by about 1.3x, and
compile time drops from ~25 s to ~3 s.

Why rolling (measured evidence, see docs/design.md):
- On Xe the uint64 of one lane takes 8 GRF (8 bytes per lane under SIMD32), so
  the 25-lane state = 200 GRF, while the hardware only has 256 GRF/thread
  (`-cl-intel-256-GRF-per-thread`; 512-GRF is unsupported on that platform).
- Fully unrolled, the 25 b values of ρπ need another 200 GRF, peaking at
  ~400 GRF, so IGC can only spill the state to scratch: `spill_size: 3584` in
  zeinfo, and 1202 `// spill to offset[..]` / `fill from ..` lines in the ISA.
- After rolling, the live range of the same code shrinks and spill_size = 0.

Usage:
    kern = get_sponge(rate=136, outlen=32)
    kern[(triton.cdiv(n, BLOCK),)](X, OUT, n, nblocks, nblocks*rate//8,
                                   OUTW=32//8, BLOCK=128, num_warps=4)
    X: [n, nblocks*rate/8] int64 (already padded)
    OUT: [n, outlen/8] int64
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from pathlib import Path

import torch

from .keccak import RC, ROT, _rotl_expr

# Round constant table: passed as a runtime kernel argument (the uint64 view is
# carried as int64, bit-for-bit identical).
#
# Cached per device — this used to be a module-level singleton hard-coded to
# "xpu", which blew up on the first call on any non-Intel backend. The table is
# only 24 int64 = 192 B, so one copy per device costs nothing.
_RC_SIGNED = [c - (1 << 64) if c >= (1 << 63) else c for c in RC]
_RC_TENSORS: dict[torch.device, torch.Tensor] = {}


def _rc_tensor(device: torch.device) -> torch.Tensor:
    t = _RC_TENSORS.get(device)
    if t is None:
        t = torch.tensor(_RC_SIGNED, dtype=torch.int64, device=device)
        _RC_TENSORS[device] = t
    return t


def _emit_rounds(L: list[str], rnd: int, ind: str = "        ", rc_expr: str | None = None) -> None:
    rc = rc_expr if rc_expr is not None else str(RC[rnd])
    for x in range(5):
        L.append(f"{ind}c{x} = a{x} ^ a{x + 5} ^ a{x + 10} ^ a{x + 15} ^ a{x + 20}")
    for x in range(5):
        cp, cn = (x + 1) % 5, (x + 4) % 5
        L.append(f"{ind}d{x} = c{cn} ^ {_rotl_expr(f'c{cp}', 1)}")
    for y in range(5):
        for x in range(5):
            i = x + 5 * y
            L.append(f"{ind}a{i} = a{i} ^ d{x}")
    for y in range(5):
        for x in range(5):
            src = x + 5 * y
            dst = y + 5 * ((2 * x + 3 * y) % 5)
            L.append(f"{ind}b{dst} = {_rotl_expr(f'a{src}', ROT[x][y])}")
    for y in range(5):
        for x in range(5):
            L.append(
                f"{ind}a{x + 5 * y} = b{x + 5 * y} ^ "
                f"((~b{(x + 1) % 5 + 5 * y}) & b{(x + 2) % 5 + 5 * y})"
            )
    L.append(f"{ind}a0 = a0 ^ {rc}")


def gen_sponge_source(rate: int, outlen: int) -> str:
    nw = rate // 8
    assert rate % 8 == 0 and outlen % 8 == 0
    assert outlen <= 16 * rate, "too many squeeze blocks"

    L = ["import triton", "import triton.language as tl", "", "@triton.jit"]
    # NB / ROW are runtime arguments: the block count no longer has to be
    # unrolled (24 rounds are already a loop, so the static instruction count
    # is independent of NB). RCP = round constant table (uint64 view).
    L.append("def sponge_kernel(X, OUT, RCP, n, NB, ROW, OUTW: tl.constexpr, BLOCK: tl.constexpr):")
    L.append("    pid = tl.program_id(0)")
    L.append("    offs = pid * BLOCK + tl.arange(0, BLOCK)")
    L.append("    m = offs < n")
    L.append("    base = offs * ROW")
    for i in range(25):
        L.append(f"    a{i} = tl.zeros((BLOCK,), dtype=tl.uint64)")
    L.append("    for _b in range(0, NB, 1):")
    for lane in range(nw):
        L.append(
            f"        a{lane} = a{lane} ^ tl.load(X + base + _b * {nw} + {lane}, "
            f"mask=m, other=0).to(tl.uint64, bitcast=True)"
        )
    # 24-round permute (rolled; it must be inside `for _b` for a correct
    # multi-block sponge)
    L.append("        for _r in range(0, 24, 1):")
    L.append("            rc = tl.load(RCP + _r)")
    _emit_rounds(L, 0, ind="            ", rc_expr="rc")

    # squeeze: may take multiple blocks (e.g. at eta=3 PRF needs 192B > rate 136B)
    nfull = outlen // rate
    rem = outlen - nfull * rate
    blocks = [rate] * nfull + ([rem] if rem else [])
    for bi, blen in enumerate(blocks):
        for lane in range(blen // 8):
            L.append(
                f"    tl.store(OUT + offs * OUTW + {bi * nw + lane}, "
                f"a{lane}.to(tl.int64, bitcast=True), mask=m)"
            )
        if bi < len(blocks) - 1:
            L.append("    for _r in range(0, 24, 1):")
            L.append("        rc = tl.load(RCP + _r)")
            _emit_rounds(L, 0, ind="        ", rc_expr="rc")
    return "\n".join(L) + "\n"


class _SpongeLauncher:
    """Thin wrapper around `sponge_kernel[grid](X, OUT, ...)`: injects the round
    constant table.

    That way the call sites in `ops.py` / `gpu_hash.py` need no change (the
    interface stays as is), while the kernel itself is driven by a runtime table
    across the 24-round loop.

    The round constant table is fetched from `X.device` **at launch time**, not
    bound at `get_sponge()` time — otherwise switching devices within one
    process would be wrong.
    """

    __slots__ = ("_kern",)

    def __init__(self, kern):
        self._kern = kern

    def __getitem__(self, grid):
        kern = self._kern

        def launch(X, OUT, *args, **kwargs):
            return kern[grid](X, OUT, _rc_tensor(X.device), *args, **kwargs)

        return launch

    def __getattr__(self, name):
        return getattr(self._kern, name)


_CACHE: dict[tuple[int, int], object] = {}


def get_sponge(rate: int, outlen: int):
    key = (rate, outlen)
    if key in _CACHE:
        return _CACHE[key]
    src = gen_sponge_source(rate, outlen)
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    path = Path(tempfile.gettempdir()) / f"_mlkem_sponge_{tag}.py"
    if not path.exists():
        path.write_text(src)
    name = f"_mlkem_sponge_{tag}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _CACHE[key] = _SpongeLauncher(mod.sponge_kernel)
    return _CACHE[key]


def pad10star1(msg: bytes, rate: int, domain: int) -> bytes:
    """pad10*1 + domain separation; returns a byte string whose len is a multiple of rate."""
    p = bytearray(msg)
    p.append(domain)
    p.extend(b"\x00" * ((-len(p)) % rate))
    p[-1] |= 0x80
    return bytes(p)


SHAKE_DOMAIN = 0x1F
SHA3_DOMAIN = 0x06


__all__ = [
    "SHA3_DOMAIN",
    "SHAKE_DOMAIN",
    "gen_sponge_source",
    "get_sponge",
    "pad10star1",
]
