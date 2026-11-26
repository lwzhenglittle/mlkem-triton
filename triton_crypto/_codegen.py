"""Compile generated Triton source strings into importable Python modules.

Several kernels are emitted as source text (the sponge, Keccak-f, SHA-2, the padded-concat
kernels, SLH-DSA's thash, LMS chains) and then imported. The old code wrote
``/tmp/_<name>_<hash>.py`` with the default umask and loaded it by path: predictable names
in a world-writable directory are a symlink/TOCTOU target, and a pre-created file could be
executed as this user.

This module gives one implementation:

* the cache lives in a per-user directory created with mode ``0700``;
* the source is written with ``O_CREAT|O_EXCL`` and mode ``0600``, then atomically renamed;
* the module name is derived from the content hash, so caching is safe across processes.

The generated code is still code; nothing here makes untrusted source safe. It only
removes the shared-/tmp failure mode.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import tempfile
from pathlib import Path

__all__ = ["compile_generated"]

_CACHE_DIR: Path | None = None


def _cache_dir() -> Path:
    global _CACHE_DIR
    if _CACHE_DIR is None:
        uid = os.getuid() if hasattr(os, "getuid") else os.getpid()
        d = Path(tempfile.gettempdir()) / f"triton-crypto-codegen-{uid}"
        d.mkdir(mode=0o700, parents=True, exist_ok=True)
        # A directory left behind by another user (or a symlink) must not be reused.
        st = os.lstat(d)
        if not os.path.isdir(d) or os.path.islink(d) or st.st_mode & 0o077:
            raise RuntimeError(f"unsafe codegen cache directory: {d}")
        _CACHE_DIR = d
    return _CACHE_DIR


def compile_generated(src: str, name: str):
    """Write ``src`` once (content-addressed) and return the imported module."""
    tag = hashlib.sha256(src.encode()).hexdigest()[:16]
    mod_name = f"_{name}_{tag}"
    d = _cache_dir()
    path = d / f"{mod_name}.py"
    if not path.exists():
        tmp = d / f".{mod_name}.{os.getpid()}.tmp"
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(src)
            os.replace(tmp, path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load generated module {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
