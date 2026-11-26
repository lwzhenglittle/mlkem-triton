"""Backwards-compatible alias for the pre-0.3 package name.

The package was renamed from `mlkem_triton` to `triton_crypto` in 0.3.0, when it grew
from "ML-KEM plus a bit of ML-DSA" into a kernel library with two schemes and a shared
primitive layer. Both names import the *same* module objects, so `isinstance` checks and
`torch`/`triton` caches behave identically; nothing is re-executed under the old name.

This shim exists only so that older code keeps working unchanged. It is deprecated and
will be removed in 1.0 — migrate to `triton_crypto`.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import warnings

import triton_crypto as _new

# Register every submodule under both names up front. Importing it lazily would also work,
# but then `mlkem_triton.foo.bar` and `triton_crypto.foo.bar` could end up as two distinct
# module objects with their own globals — a subtle source of "this class is not that class".
with warnings.catch_warnings():
    # The compatibility shims for the pre-0.8 module paths warn on import; the old package
    # name must keep working without turning `import mlkem_triton` into a warning storm.
    warnings.simplefilter("ignore", DeprecationWarning)
    for _info in pkgutil.walk_packages(_new.__path__, prefix=_new.__name__ + "."):
        _module = importlib.import_module(_info.name)
        sys.modules.setdefault(__name__ + _info.name[len(_new.__name__) :], _module)

# Make `import mlkem_triton` hand back the real package, attributes and all.
sys.modules[__name__] = _new
