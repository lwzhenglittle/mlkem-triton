"""Uniform normalization of user-supplied byte inputs.

The scheme entry points accept "natural" Python forms — one ``bytes`` object, a list of
``bytes``, a 2-D ``numpy`` array, a 1-D or 2-D ``torch`` tensor — and the rules for what
counts as one item were previously re-implemented (with slightly different bugs) in every
module. This module is the single implementation:

* one ``bytes``/``bytearray``/``memoryview`` is **one item**, not a sequence of integers;
* a 1-D array/tensor is one item, a 2-D one is a batch;
* an item may be bytes-like or a 1-D array/tensor of ``uint8``;
* every item is length-checked when the caller knows the expected size.

Errors are deliberate: a wrong item length or a batch-size mismatch raises ``ValueError``
rather than being silently reinterpreted. "The signature is invalid" stays a per-item
``False`` at the verification layer; "the caller passed the wrong shape" is a bug in the
caller and must not be mistaken for a failed verification.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = ["as_batch_bytes", "as_bytes_list"]


def _single_bytes(x, name: str, index: int | None = None) -> bytes:
    """One item -> bytes, without reinterpreting integers as byte strings."""
    where = name if index is None else f"{name}[{index}]"
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    if torch.is_tensor(x):
        if x.dim() > 1:
            raise ValueError(f"{where} must be one-dimensional, got shape {tuple(x.shape)}")
        return x.detach().cpu().contiguous().numpy().astype(np.uint8, copy=False).tobytes()
    arr = np.asarray(x)
    if arr.dtype == object:
        raise TypeError(f"{where} must be bytes-like or an array, got {type(x).__name__}")
    if arr.ndim > 1:
        raise ValueError(f"{where} must be one-dimensional, got shape {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.uint8).tobytes()


def as_bytes_list(x, name: str = "items") -> list[bytes]:
    """Normalize a batch of byte strings to ``list[bytes]``.

    Accepts one bytes-like object, a list/tuple of items, or a 1-D/2-D numpy array or
    torch tensor (rows are items). ``str`` and integers are rejected instead of being
    silently coerced.
    """
    if isinstance(x, (bytes, bytearray, memoryview)):
        return [bytes(x)]
    if torch.is_tensor(x):
        if x.dim() == 0:
            raise ValueError(f"{name} must be at least one-dimensional")
        if x.dtype != torch.uint8:
            raise TypeError(f"{name} must be uint8, got {x.dtype}")
        arr = x.detach().cpu().contiguous().numpy()
        if arr.ndim == 1:
            return [arr.tobytes()]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(x.shape)}")
        return [row.tobytes() for row in arr]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            raise ValueError(f"{name} must be at least one-dimensional")
        if x.dtype == object:
            return [_single_bytes(v, name, i) for i, v in enumerate(x)]
        arr = np.ascontiguousarray(x, dtype=np.uint8)
        if arr.ndim == 1:
            return [arr.tobytes()]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 1-D or 2-D, got shape {x.shape}")
        return [row.tobytes() for row in arr]
    if isinstance(x, (str, int)):
        raise TypeError(f"{name} must be bytes-like, not {type(x).__name__}")
    try:
        items = list(x)
    except TypeError as exc:
        raise TypeError(f"{name} must be bytes-like or a sequence, got {type(x).__name__}") from exc
    return [_single_bytes(v, name, i) for i, v in enumerate(items)]


def as_batch_bytes(
    x, item_size: int | None = None, name: str = "items", *, blob_is_batch: bool = False
) -> np.ndarray:
    """Normalize a batch to a ``[B, item_size]`` uint8 array.

    A single item becomes ``B = 1``. ``item_size=None`` accepts variable-length items only
    when the input is a 2-D array (which is necessarily rectangular); otherwise every item
    must have the same length, so the result is well defined. An empty batch returns
    ``(0, item_size)``.

    ``blob_is_batch`` applies to fixed-size schemes (ML-KEM): a single ``bytes`` object or a
    1-D array is then a concatenation of ``item_size``-byte items, provided the length is an
    exact multiple. Lists are always checked per item, so a ``[31, 33]`` pair cannot be
    silently reinterpreted as two 32-byte messages.
    """
    if isinstance(x, torch.Tensor):
        if x.dtype != torch.uint8:
            raise TypeError(f"{name} must be uint8, got {x.dtype}")
        t = x.detach()
        if t.dim() == 1 and blob_is_batch and item_size and t.numel() % item_size == 0:
            t = t.reshape(-1, item_size)
        elif t.dim() == 1:
            t = t.reshape(1, -1)
        if t.dim() != 2:
            raise ValueError(f"{name} must be 1-D or 2-D, got shape {tuple(x.shape)}")
        if item_size is not None and t.shape[1] != item_size:
            kind = name[:-1] if name.endswith("s") else name
            raise ValueError(f"each {kind} must be {item_size} bytes, got {t.shape[1]}")
        return t.contiguous().cpu().numpy()

    if (
        blob_is_batch
        and item_size
        and isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype != object
    ):
        arr = np.ascontiguousarray(x, dtype=np.uint8)
        if arr.size % item_size:
            raise ValueError(f"{name} length {arr.size} is not a multiple of {item_size}")
        if not arr.size:
            return np.empty((0, item_size), dtype=np.uint8)
        return arr.reshape(-1, item_size)

    if blob_is_batch and item_size and isinstance(x, (bytes, bytearray, memoryview)):
        buf = bytes(x)
        if len(buf) % item_size:
            raise ValueError(
                f"input length {len(buf)} is not a multiple of {item_size}-byte {name}"
            )
        if not buf:
            return np.empty((0, item_size), dtype=np.uint8)
        return np.frombuffer(buf, dtype=np.uint8).reshape(-1, item_size)

    rows = as_bytes_list(x, name)
    if not rows:
        return np.empty((0, item_size or 0), dtype=np.uint8)

    sizes = {len(r) for r in rows}
    if item_size is None:
        if len(sizes) != 1:
            detail = ", ".join(str(s) for s in sorted(sizes)[:4])
            raise ValueError(f"{name} have differing lengths ({detail}); expected a rectangle")
        item_size = sizes.pop()
    else:
        for i, r in enumerate(rows):
            if len(r) != item_size:
                raise ValueError(f"{name}[{i}] must be {item_size} bytes, got {len(r)}")
    return np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(len(rows), item_size)
