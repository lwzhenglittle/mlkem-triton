"""Phase 4: end-to-end batch decapsulation pipeline.

Data flow (one batch of B ciphertexts):
  1. unpack c → c1/c2 integers                    (CPU, numpy bit unpacking)
  2. decompress u,v; w = v − W_dec @ u            (GPU, Triton GEMM)
  3. m' = Compress_1(w)                           (CPU)
  4. (K',r') = G(m'‖H(ek)); K̄ = J(z‖c)           (CPU hashlib or GPU batched sponge)
  5. PRF + CBD sample y,e1,e2                     (CPU hashlib or GPU batched sponge)
  6. [u';v'] = W_enc @ y + [e1; e2 + 1665·m']     (GPU, Triton GEMM)
  7. compress and compare; constant-time select K (CPU)

With `use_gpu_hash=True`, steps 4/5 are both replaced by the GPU batched sponge.
"""

from __future__ import annotations

import numpy as np
import torch

from ... import keyprep
from ...devices import resolve_device
from ...kernels import elementwise as ew
from ...kernels.gemm_modq import gemm_modq
from ...params import PARAMS
from .sampling_cpu import G, J, cbd, prf


class ReferenceDecapsulator:
    """Comparison implementation that hashes on the CPU (for benchmark comparisons,
    **not** the recommended entry point)."""

    def __init__(self, ps: str, dk: bytes, device=None, use_gpu_hash: bool = False):
        p = PARAMS[ps]
        device = resolve_device(device)
        self.ps, self.p, self.device = ps, p, device
        self.use_gpu_hash = use_gpu_hash
        self.dk = dk
        self.kp = keyprep.parse_key(ps, dk)
        k = p.k
        self.h_ek = dk[768 * k + 32 : 768 * k + 64]
        self.z = dk[768 * k + 64 : 768 * k + 96]
        self.W_dec = (
            torch.from_numpy(np.ascontiguousarray(self.kp.W_dec)).to(torch.float16).to(device)
        )
        self.W_enc = (
            torch.from_numpy(np.ascontiguousarray(self.kp.W_enc)).to(torch.float16).to(device)
        )

    # ---- sampling: CPU path ----
    def _sample_cpu(self, r_prime: np.ndarray, B: int):
        p, k = self.p, self.p.k
        y = np.empty((256 * k, B), dtype=np.int64)
        e1 = np.empty((256 * k, B), dtype=np.int64)
        e2 = np.empty((256, B), dtype=np.int64)
        for i in range(B):
            r = r_prime[i].tobytes()
            for j in range(k):
                y[256 * j : 256 * (j + 1), i] = cbd(prf(p.eta1, r, j), p.eta1)
                e1[256 * j : 256 * (j + 1), i] = cbd(prf(p.eta2, r, k + j), p.eta2)
            e2[:, i] = cbd(prf(p.eta2, r, 2 * k), p.eta2)
        return y, e1, e2

    # ---- sampling: GPU batched path ----
    def _sample_gpu(self, r_prime: np.ndarray, B: int):
        from .. import gpu_hash

        p, k = self.p, self.p.k
        idx = np.arange(2 * k + 1, dtype=np.uint8)
        inp = np.empty((B, 2 * k + 1, 33), dtype=np.uint8)
        inp[:, :, :32] = r_prime[:, None, :]
        inp[:, :, 32] = idx[None, :]

        # eta1 (y) and eta2 (e1,e2) may have different output lengths, so two batched PRFs
        y_buf = gpu_hash.shake256_batch(
            inp[:, :k, :].reshape(B * k, 33), 64 * p.eta1, device=self.device
        )
        e_buf = gpu_hash.shake256_batch(
            inp[:, k:, :].reshape(B * (k + 1), 33), 64 * p.eta2, device=self.device
        )

        y = gpu_hash.cbd_batch(y_buf, p.eta1).reshape(B, k, 256)
        e = gpu_hash.cbd_batch(e_buf, p.eta2).reshape(B, k + 1, 256)

        Y = np.empty((256 * k, B), dtype=np.int64)
        E1 = np.empty((256 * k, B), dtype=np.int64)
        for j in range(k):
            Y[256 * j : 256 * (j + 1)] = y[:, j, :].T
            E1[256 * j : 256 * (j + 1)] = e[:, j, :].T
        E2 = e[:, k, :].T
        return Y, E1, E2

    # ---- hashing: CPU path ----
    def _hash_cpu(self, m_bytes, cts, B):
        K_prime = np.empty((B, 32), np.uint8)
        r_prime = np.empty((B, 32), np.uint8)
        K_bar = np.empty((B, 32), np.uint8)
        for i in range(B):
            kp, rp = G(m_bytes[i].tobytes() + self.h_ek)
            K_prime[i] = np.frombuffer(kp, np.uint8)
            r_prime[i] = np.frombuffer(rp, np.uint8)
            K_bar[i] = np.frombuffer(J(self.z + cts[i].tobytes()), np.uint8)
        return K_prime, r_prime, K_bar

    # ---- hashing: GPU batched path ----
    def _hash_gpu(self, m_bytes, cts, B):
        from .. import gpu_hash

        h_ek = np.frombuffer(self.h_ek, dtype=np.uint8)
        g_in = np.concatenate([m_bytes, np.broadcast_to(h_ek, (B, 32))], axis=1)
        gr = gpu_hash.sha3_512_batch(g_in, device=self.device)

        z = np.frombuffer(self.z, dtype=np.uint8)
        j_in = np.concatenate([np.broadcast_to(z, (B, 32)), cts], axis=1)
        k_bar = gpu_hash.shake256_batch(j_in, 32, device=self.device)
        return gr[:, :32], gr[:, 32:64], k_bar

    # ---- helpers ----
    def _to_f16(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(a)).to(torch.float16).to(self.device)

    def _to_i16(self, a: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(np.asarray(a, np.int16))).to(self.device)

    # ---- main flow ----
    def decaps(self, cts) -> np.ndarray:
        cts = np.asarray(cts, dtype=np.uint8)
        if cts.ndim == 1:
            cts = cts[None, :]
        B = cts.shape[0]
        p, k = self.p, self.p.k

        # 1. unpack + decompress
        c1, c2 = ew.unpack_ct(cts, k, p.du, p.dv)  # [k,256,B], [256,B]
        u = ew.decompress_np(c1, p.du).reshape(k * 256, B)  # [256k,B]
        v = ew.decompress_np(c2, p.dv)  # [256,B]

        # 2. w = v − W_dec @ u
        u_c = keyprep.center(u)
        w = gemm_modq(self.W_dec, self._to_f16(u_c), self._to_i16(v), mode="dec").cpu().numpy()

        # 3. m'
        m_bits = ew.compress_np(w, 1)  # [256,B] ∈ {0,1}
        m_bytes = ew.pack_bits(m_bits, 1)  # [B,32]

        # 4. G / J
        if self.use_gpu_hash:
            K_prime, r_prime, K_bar = self._hash_gpu(m_bytes, cts, B)
        else:
            K_prime, r_prime, K_bar = self._hash_cpu(m_bytes, cts, B)

        # 5. sampling
        if self.use_gpu_hash:
            y, e1, e2 = self._sample_gpu(r_prime, B)
        else:
            y, e1, e2 = self._sample_cpu(r_prime, B)

        # 6. re-encrypt
        E = np.vstack([e1, e2 + 1665 * m_bits])
        out = gemm_modq(self.W_enc, self._to_f16(y), self._to_i16(E), mode="enc").cpu().numpy()

        # 7. compress + compare
        c1p = ew.compress_np(out[: k * 256], p.du)
        c2p = ew.compress_np(out[k * 256 :], p.dv)
        eq = np.all(c1p == c1.reshape(k * 256, B), axis=0) & np.all(c2p == c2, axis=0)

        # 8. constant-time selection
        return ew.select_bytes(K_prime, K_bar, eq)


def decaps_batch_ref(
    ps: str, dk: bytes, cts, device=None, use_gpu_hash: bool = False
) -> np.ndarray:
    return ReferenceDecapsulator(ps, dk, device=device, use_gpu_hash=use_gpu_hash).decaps(cts)


__all__ = ["ReferenceDecapsulator", "decaps_batch_ref"]
