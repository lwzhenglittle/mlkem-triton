"""LMS / HSS (SP 800-208) signature verification on the GPU, plus the RFC 8554 reference.

    from triton_crypto.lms import Verifier
    v = Verifier(hss_public_key)
    ok = v.verify(messages, signatures)       # -> np.ndarray[bool], batch

`Verifier` accepts any HSS public key up to eight levels, re-reads the parameter set from
each embedded public key (levels may differ), and verifies the whole batch on the device.
`ref` contains the pure-Python implementation used to cross-check it.
"""

from __future__ import annotations

from .params import ALL_SETS, LMS_NAMES, OTS_NAMES, LMSSet, from_name, get
from .verify import Verifier, verify_batch

__all__ = [
    "ALL_SETS",
    "LMS_NAMES",
    "OTS_NAMES",
    "LMSSet",
    "Verifier",
    "from_name",
    "get",
    "verify_batch",
]
