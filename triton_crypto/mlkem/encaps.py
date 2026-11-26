"""ML-KEM encapsulation (FIPS 203 Alg. 17), fully device-resident.

As in decaps, the core is **to replace the NTT with Toeplitz matrix multiplication**:

    [u; v] = W_enc @ r + [e1; e2 + Decompress_1(m)]

`W_enc` depends only on the public key `ek` (i.e. `Â = SampleNTT(ρ)` and `t`) and is
fixed for a given `ek`. So in the "same public key, many encapsulations" regime it can
be expanded once and reused — this is exactly our structural advantage over an NTT
implementation. **Pure Triton, no NTT.**

Data flow:

    m(32B) ─┬─▶ pad(m ‖ H(ek)) ─▶ sponge(SHA3-512) ─▶ K_pre ‖ r_seed
            │                        K = K_pre (FIPS 203 has no KDF!)
            │                          ┌───────────────────┴──────────┐
            │                          ▼                              ▼
            │                PRF(r_seed, 0..k-1)          PRF(r_seed, k..2k)
            │                          ▼                              ▼
            │                     CBD_η1 → r                   CBD_η2 → e1, e2
            │                          │                              │
            └──▶ m_bits ───────────────┼────── e2 + 1665·m_bits ─────┘
                                       ▼
                            W_enc @ r + [e1; e2']  ──▶ mod q ──▶ Compress
                                       ▼
                              ByteEncode_du/dv ──▶ c (1088 B)
"""

from __future__ import annotations

import numpy as np
import torch

from .._inputs import as_batch_bytes
from ..devices import resolve_device
from ..primitives.sponge import DOMAIN_SHA3, DOMAIN_SHAKE, SHA3_256
from . import keyprep, ops
from .kernels.encaps import fused_encaps, pack_ciphertext
from .params import PARAMS
from .pipeline import RATE_SHA3_512, RATE_SHAKE, _pad2_gpu, _pad_prf_gpu, sponge_gpu

RATE_SHA3_256 = SHA3_256.rate


class Encapsulator:
    """Load one encapsulation public key onto the device, then batch-encapsulate.

    Args:
        ps: parameter set name.
        ek: encapsulation public key bytes (length ``384k+32``).
        device: ``None`` selects one automatically.
        validate: enforce the FIPS 203 §7.2 encapsulation-key check (length and
            canonical 12-bit coefficients) and reject non-canonical keys. Set to
            ``False`` only for formally-invalid test vectors.
    """

    def __init__(self, ps: str, ek: bytes, device=None, validate: bool = True):
        p = PARAMS[ps]
        self.ps, self.p = ps, p
        self.device = resolve_device(device)
        self.kp = keyprep.parse_ek(ps, ek, validate=validate)
        self.ek = bytes(ek)
        dev = self.device
        self.h_ek = torch.frombuffer(bytearray(self.kp.h_ek), dtype=torch.uint8).to(dev)
        W = torch.from_numpy(np.ascontiguousarray(self.kp.W_enc))
        if W.abs().max().item() > 2048:
            raise ValueError(f"W_enc out of the fp16 integer range: |W|max={W.abs().max().item()}")
        self.W_enc = W.to(torch.float16).to(dev)

    # ---- Main flow ----
    def encapsulate(self, msgs) -> tuple[np.ndarray, np.ndarray]:
        """Batch encapsulation.

        Args:
            msgs: ``[B, 32]`` uint8 array / ``list[bytes]`` / one contiguous ``bytes``
                  block (32-byte messages; the caller is responsible for randomness).

        Returns:
            ``(c, K)``: ciphertext ``[B, ct_size]`` uint8, shared key ``[B, 32]`` uint8.
        """
        arr = _to_msgs2d(msgs)
        B = arr.shape[0]
        if not arr.flags.writeable:  # np.frombuffer over a bytes object
            arr = arr.copy()
        m = torch.from_numpy(arr).to(self.device)
        c, K = self.encapsulate_torch(m, B)
        return c.cpu().numpy(), K.cpu().numpy()

    def encapsulate_torch(self, m: torch.Tensor, B: int):
        """Device-side flow only. `m` is ``[B, 32]`` uint8.

        The input is copied (``.contiguous()``) to the key's device when needed; a
        non-contiguous view used to be consumed with the wrong strides, which produced
        silently incorrect ciphertexts instead of an error.
        """
        p, k, dev = self.p, self.p.k, self.device
        if not torch.is_tensor(m):
            raise TypeError(f"m must be a torch.Tensor, got {type(m).__name__}")
        if m.dtype != torch.uint8:
            raise ValueError(f"m must be uint8, got {m.dtype}")
        if m.shape != (B, 32):
            raise ValueError(f"m must be [B,32] uint8, got {tuple(m.shape)}")
        if B < 0:
            raise ValueError("B must be non-negative")
        m = m.contiguous()
        if m.device != dev:
            m = m.to(dev)
        if B == 0:
            return (
                torch.empty((0, p.ct_size), dtype=torch.uint8, device=dev),
                torch.empty((0, 32), dtype=torch.uint8, device=dev),
            )

        # 1. (K_pre, r_seed) = G(m ‖ H(ek))     — SHA3-512, rate 72
        gr = sponge_gpu(_pad2_gpu(m, self.h_ek, B, RATE_SHA3_512, DOMAIN_SHA3), RATE_SHA3_512, 64)
        k_pre, r_seed = gr[:, :32], gr[:, 32:]

        # 2. PRF: r_seed ‖ [N] → the byte stream fed to CBD
        eta1, eta2 = p.eta1, p.eta2
        y_buf = sponge_gpu(
            _pad_prf_gpu(r_seed, B * k, RATE_SHAKE, DOMAIN_SHAKE, mod=k, off=0),
            RATE_SHAKE,
            64 * eta1,
        )
        e_buf = sponge_gpu(
            _pad_prf_gpu(r_seed, B * (k + 1), RATE_SHAKE, DOMAIN_SHAKE, mod=k + 1, off=k),
            RATE_SHAKE,
            64 * eta2,
        )

        # 3. CBD: r → Y [256k, B] fp16; e1/e2 → E [256(k+1), B] int16
        #    (`cbd_e` also adds 1665·m_bits onto E2, which is exactly Decompress_1(m))
        if eta1 == 2:
            Y = ops.cbd_y(y_buf, B, k)
        else:  # ML-KEM-512's η1=3: generic path
            from .pipeline import cbd_gpu

            Y = (
                cbd_gpu(y_buf, eta1)
                .reshape(B, k, 256)
                .permute(1, 2, 0)
                .reshape(k * 256, B)
                .to(torch.float16)
                .contiguous()
            )
        if eta2 == 2:
            E = ops.cbd_e(e_buf, m, B, k)
        else:
            from .kernels.glue import unpack_mbits
            from .pipeline import cbd_gpu

            e = cbd_gpu(e_buf, eta2).reshape(B, k + 1, 256)
            E1 = e[:, :k].permute(1, 2, 0).reshape(k * 256, B).to(torch.int16)
            E2 = (e[:, k].T + 1665 * unpack_mbits(m, B)).to(torch.int16)
            E = torch.cat([E1, E2], dim=0)

        # 4. Fusion: GEMM + mod q + Compress
        comp = fused_encaps(self.W_enc, Y, E, B, p.du, p.dv)
        if comp is None:  # generic fallback when the shape does not meet the constraints
            acc = torch.mm(self.W_enc.to(torch.float32), Y.to(torch.float32))
            x = (acc.to(torch.int32) + E.to(torch.int32)) % 3329
            x = torch.where(x < 0, x + 3329, x)
            du, dv, pivot = p.du, p.dv, 256 * k
            xu = (((x[:pivot] << du) + 1664) // 3329) & ((1 << du) - 1)
            xv = (((x[pivot:] << dv) + 1664) // 3329) & ((1 << dv) - 1)
            comp = torch.cat([xu, xv], dim=0).to(torch.int16)

        # 5. ByteEncode → c, and prepare the pad10*1 that H(c) needs in the same pass
        #    (this saves a separate pad kernel: the ciphertext itself is 1088 B, plus
        #     the domain-separation byte gives 1089 B; SHA3-256 has rate 136, so it
        #     pads to 9 blocks = 1224 B).
        ct_size = p.ct_size
        total = ((ct_size + 1 + RATE_SHA3_256 - 1) // RATE_SHA3_256) * RATE_SHA3_256
        padded = torch.zeros((B, total), dtype=torch.uint8, device=dev)
        pack_ciphertext(comp, padded, k, p.du, p.dv, ct_size)
        padded[:, ct_size] = DOMAIN_SHA3  # domain separation
        padded[:, total - 1] = 0x80  # last byte of pad10*1
        c = padded[:, :ct_size]  # view, zero-copy

        # 6. FIPS 203 Encaps has **no** KDF step.
        #    Kyber round-3 used K = KDF(K̄ ‖ H(c)); FIPS 203 (Alg 17) just does
        #    `return (K, c)`, and K is the first 32 bytes of G(m‖H(ek)).
        #    (I first wrote the round-3 version: `c` was all correct but `K` was all
        #     wrong, and only ACVP caught it.)
        return c, k_pre


def _to_msgs2d(msgs) -> np.ndarray:
    """Normalize messages to ``[B, 32]`` uint8 (a bytes blob is a batch of 32-byte items)."""
    return as_batch_bytes(msgs, 32, "message", blob_is_batch=True)


def encapsulate(ps: str, ek: bytes, msgs, device=None):
    """One-shot batch encapsulation."""
    return Encapsulator(ps, ek, device=device).encapsulate(msgs)


__all__ = ["Encapsulator", "encapsulate"]
