"""1.0 removed the pre-0.8 import paths. These tests say so, and keep them removed.

They are assertions of *absence*, which is the point: a compatibility shim that creeps back
(or a module the old name gets recreated from by a stray import) is otherwise invisible until
someone publishes a wheel where both layouts coexist. The mapping below is what users get in
the CHANGELOG's migration table; if a row ever comes back, this file goes red.
"""

from __future__ import annotations

import importlib

import pytest

# The pre-0.8 root layout, and the pre-0.3 package name.
REMOVED_MODULES = [
    "mlkem_triton",
    "mlkem_triton.primitives",
    "mlkem_triton.primitives.sponge",
    "mlkem_triton.mldsa",
    "triton_crypto.pipeline",
    "triton_crypto.pipeline_host_hash",
    "triton_crypto.params",
    "triton_crypto.keygen",
    "triton_crypto.encaps",
    "triton_crypto.keyprep",
    "triton_crypto.ops",
    "triton_crypto.acvp",
    "triton_crypto.ntt_ref",
    "triton_crypto.refimpl",
    "triton_crypto.sampling_cpu",
    "triton_crypto.gpu_hash",
    "triton_crypto.kernels",
    "triton_crypto.kernels.ntt",
]

# old path -> new path, with one attribute to import from each (the CHANGELOG table).
MOVED = [
    ("triton_crypto.mlkem.pipeline", "Decapsulator"),
    ("triton_crypto.mlkem.params", "PARAMS"),
    ("triton_crypto.mlkem.keygen", "KeyGenerator"),
    ("triton_crypto.mlkem.encaps", "Encapsulator"),
    ("triton_crypto.mlkem.keyprep", "EncKeyPrep"),
    ("triton_crypto.mlkem.ops", "build_ye"),
    ("triton_crypto.mlkem.ntt_ref", "multiply_ntts"),
    ("triton_crypto.mlkem.acvp", "load"),
    ("triton_crypto.mlkem.gpu_hash", "cbd_batch"),
    ("triton_crypto.mlkem.kernels.ntt", "byte_encode12"),
    ("triton_crypto.mlkem.reference.refimpl", "implicit_reject_key"),
    ("triton_crypto.mlkem.reference.sampling_cpu", "cbd"),
    ("triton_crypto.mlkem.reference.pipeline_host_hash", "ReferenceDecapsulator"),
    ("triton_crypto.slhdsa.thash_sha2", "sha_thash"),
]


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_removed_module_path_is_gone(name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


@pytest.mark.parametrize("module, attr", MOVED)
def test_the_new_home_still_imports(module, attr):
    mod = importlib.import_module(module)
    assert hasattr(mod, attr), f"{module}.{attr} disappeared"


def test_sha_thash_alias_is_gone_from_primitives():
    """It moved to `slhdsa.thash_sha2` in 0.8.0 and the alias was dropped in 1.0."""
    from triton_crypto.primitives import sha2

    assert not hasattr(sha2, "sha_thash")
    assert "sha_thash" not in sha2.__all__


def test_root_package_does_not_advertise_the_old_name():
    import triton_crypto

    assert "mlkem_triton" not in (triton_crypto.__doc__ or "")
    assert triton_crypto.__version__.startswith("1.")


def test_setuptools_finds_only_the_new_tree():
    """`mlkem_triton*` must not be in the package list, or wheels would keep shipping it."""
    from pathlib import Path

    import setuptools

    root = Path(__file__).resolve().parent.parent
    packages = set(setuptools.find_packages(where=str(root)))
    assert packages, "no packages found"
    assert not any(p == "mlkem_triton" or p.startswith("mlkem_triton.") for p in packages)
