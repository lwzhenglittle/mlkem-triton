"""Generic Keccak sponge over a batch: absorb *N* blocks, squeeze *M* blocks.

This is the hash engine shared by ML-KEM (FIPS 203) and ML-DSA (FIPS 204). It knows
nothing about either scheme: an instance is fully described by `(rate, outlen)`, and the
padding is done by the caller (`pad10star1`, `pad_batch`, or the torch helpers below),
because the two standards pad differently (ML-DSA's message formatting is a longer
prefix, and the domain byte differs).

Mapping: one thread = one sponge instance, each of the 25 lanes is a `[BLOCK]`
uint64 tensor. All 24 rounds are unrolled *inside a rolled `for _r in range(0,24)`*
with the round constants read from a 24-entry table passed as a runtime argument.

Why the rounds are rolled (measured, see docs/design.md):
- On Xe the uint64 of one lane takes 8 GRF (8 bytes per lane under SIMD32), so the
  25-lane state = 200 GRF, while the hardware only has 256 GRF/thread
  (`-cl-intel-256-GRF-per-thread`; 512-GRF is unsupported on that platform).
- Fully unrolled, the 25 b values of ρπ need another 200 GRF, peaking at ~400 GRF, so
  IGC can only spill the state to scratch: `spill_size: 3584` in zeinfo, and 1202
  `// spill to offset[..]` / `fill from ..` lines in the ISA.
- After rolling, the live range of the same code shrinks and spill_size = 0: static
  instruction count 15254 → 790, throughput at rate=136/out=128 +~30%, compile time
  ~25 s → ~3 s.

The squeeze may span several blocks (e.g. ExpandA needs 840 B > rate 168 B), the absorb
count is a runtime argument (`NB`) so one compiled kernel serves any message length.

Usage:
    kern = get_sponge(rate=136, outlen=32)
    kern[(triton.cdiv(n, BLOCK),)](X, OUT, n, nblocks, nblocks*rate//8,
                                   OUTW=32//8, BLOCK=128, num_warps=4)
    X: [n, nblocks*rate/8] int64 (already padded)
    OUT: [n, outlen/8] int64

For the non-kernel paths (`hash_np_batch`, `sponge_torch`) use the helpers at the
bottom, which do the padding and the int64 view for you.
"""

from __future__ import annotations

import hashlib
import importlib.util
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .keccak import RC, ROT, _rotl_expr

# ---------------- domain separation bytes (NIST SP 800-185 style) ----------------

DOMAIN_SHA3 = 0x06  # SHA3-224/256/384/512
DOMAIN_SHAKE = 0x1F  # SHAKE128 / SHAKE256
DOMAIN_CSHAKE = 0x04
DOMAIN_RAW = 0x00

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

    That way the call sites need no change (the interface stays as is), while the
    kernel itself is driven by a runtime table across the 24-round loop.

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
    """Return a launcher for `rate`/`outlen`; compiles (and caches) on first use."""
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


# ---------------- padding ----------------

BLOCK = 128
WARPS = 4


def pad10star1(msg: bytes, rate: int, domain: int) -> bytes:
    """pad10*1 + domain separation; returns a byte string whose len is a multiple of rate."""
    p = bytearray(msg)
    p.append(domain)
    p.extend(b"\x00" * ((-len(p)) % rate))
    p[-1] |= 0x80
    return bytes(p)


def padded_len(nbytes: int, rate: int) -> int:
    """Length of `pad10*1(msg)` for a message of `nbytes` bytes."""
    return ((nbytes + 1 + rate - 1) // rate) * rate


def pad_batch(msgs: np.ndarray, rate: int, domain: int) -> np.ndarray:
    """Vectorized pad10*1 + domain separation.

    msgs [n, L] uint8 -> [n, ceil((L+1)/rate)*rate] uint8.
    """
    msgs = np.ascontiguousarray(msgs, dtype=np.uint8)
    n, L = msgs.shape
    total = padded_len(L, rate)
    out = np.zeros((n, total), dtype=np.uint8)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, -1] |= 0x80
    return out


def pad_torch(msgs: torch.Tensor, rate: int, domain: int) -> torch.Tensor:
    """On-device pad10*1 + domain separation; msgs [n,L] uint8 -> [n, total] uint8."""
    n, L = msgs.shape
    total = padded_len(L, rate)
    out = torch.zeros((n, total), dtype=torch.uint8, device=msgs.device)
    out[:, :L] = msgs
    out[:, L] = domain
    out[:, -1] = out[:, -1] | 0x80
    return out


# ---------------- torch-level helpers (absorb a pre-padded batch) ----------------


def sponge_padded(padded: torch.Tensor, rate: int, outlen: int, warps: int = WARPS) -> torch.Tensor:
    """[n, k*rate] uint8 (already padded) -> [n, outlen] uint8, all on device."""
    n, tot = padded.shape
    nblocks = tot // rate
    if nblocks * rate != tot:
        raise ValueError(f"padded length {tot} is not a multiple of rate {rate}")
    x = padded.contiguous().view(torch.int64)
    out = torch.empty((n, outlen // 8), dtype=torch.int64, device=padded.device)
    kern = get_sponge(rate, outlen)
    grid = (max(1, -(-n // BLOCK)),)
    kern[grid](
        x,
        out,
        n,
        nblocks,
        nblocks * rate // 8,
        OUTW=outlen // 8,
        BLOCK=BLOCK,
        num_warps=warps,
    )
    return out.view(torch.uint8).view(n, outlen)


# ---------------- named hash functions ----------------


@dataclass(frozen=True)
class HashFn:
    """A sponge-based hash: `(rate, domain)` plus the fixed output length, if any.

    `ref_name` is the equivalent `hashlib` constructor, used for host-side reference
    digests in tests and for the `HashML-DSA` pre-hash modes.
    """

    name: str
    rate: int
    domain: int
    outlen: int | None  # None for a XOF (caller picks)
    ref_name: str = ""

    def pad(self, msg: bytes) -> bytes:
        return pad10star1(msg, self.rate, self.domain)

    def digest_bytes(self, msg: bytes, outlen: int | None = None) -> bytes:
        """Host-side digest (hashlib); `outlen` is required for XOFs."""
        n = outlen if outlen is not None else self.outlen
        if n is None:
            raise ValueError(f"{self.name} is an XOF: pass outlen")
        fn = getattr(hashlib, self.ref_name or self.name.lower().replace("-", "_"))
        h = fn(msg)
        return h.digest(n) if self.outlen is None else h.digest()

    def hash_torch(self, msgs: torch.Tensor, outlen: int | None = None) -> torch.Tensor:
        """[n, L] uint8 -> [n, outlen] uint8, on device (pads internally)."""
        n = outlen if outlen is not None else self.outlen
        if n is None:
            raise ValueError(f"{self.name} is an XOF: pass outlen")
        return sponge_padded(pad_torch(msgs, self.rate, self.domain), self.rate, n)


SHA3_256 = HashFn("SHA3-256", 136, DOMAIN_SHA3, 32, "sha3_256")
SHA3_512 = HashFn("SHA3-512", 72, DOMAIN_SHA3, 64, "sha3_512")
SHAKE128 = HashFn("SHAKE128", 168, DOMAIN_SHAKE, None, "shake_128")
SHAKE256 = HashFn("SHAKE256", 136, DOMAIN_SHAKE, None, "shake_256")

HASHES = {h.name: h for h in (SHA3_256, SHA3_512, SHAKE128, SHAKE256)}


def hash_np_batch(
    msgs: np.ndarray, hf: HashFn, outlen: int | None = None, device=None
) -> np.ndarray:
    """[n, L] uint8 -> [n, outlen] uint8 using the GPU sponge (round trip via host).

    For ML-KEM's G / J / PRF and for test oracles. The batch stays on the device
    between the pad and the squeeze; only the result comes back.
    """
    from ..devices import resolve_device

    n = outlen if outlen is not None else hf.outlen
    if n is None:
        raise ValueError(f"{hf.name} is an XOF: pass outlen")
    padded = pad_batch(msgs, hf.rate, hf.domain)
    dev = resolve_device(device)
    x = torch.from_numpy(padded.copy()).view(torch.int64).to(dev)
    out = torch.empty((padded.shape[0], n // 8), dtype=torch.int64, device=dev)
    kern = get_sponge(hf.rate, n)
    grid = (max(1, -(-padded.shape[0] // BLOCK)),)
    kern[grid](
        x,
        out,
        padded.shape[0],
        padded.shape[1] // hf.rate,
        padded.shape[1] // 8,
        OUTW=n // 8,
        BLOCK=BLOCK,
        num_warps=WARPS,
    )
    return out.cpu().numpy().view(np.uint8).reshape(padded.shape[0], n)


__all__ = [
    "BLOCK",
    "DOMAIN_CSHAKE",
    "DOMAIN_RAW",
    "DOMAIN_SHA3",
    "DOMAIN_SHAKE",
    "HASHES",
    "SHA3_256",
    "SHA3_512",
    "SHAKE128",
    "SHAKE256",
    "HashFn",
    "gen_sponge_source",
    "get_sponge",
    "hash_np_batch",
    "pad10star1",
    "pad_batch",
    "pad_torch",
    "padded_len",
    "sponge_padded",
]
