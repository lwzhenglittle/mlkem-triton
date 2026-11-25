"""The `mlkem_triton` shim: the old name must hand back the *same* objects, not copies.

A shim that re-imports the modules under a second name would produce two module globals, two
copies of every jit function cache entry and classes that fail `isinstance` across the two names.
These three cases pin down that it does not.
"""

import importlib


def test_old_name_is_the_new_package():
    old = importlib.import_module("mlkem_triton")
    new = importlib.import_module("triton_crypto")
    assert old is new
    assert old.__version__ == new.__version__


def test_submodules_are_shared():
    old = importlib.import_module("mlkem_triton.primitives.sponge")
    new = importlib.import_module("triton_crypto.primitives.sponge")
    assert old is new
    assert old.sponge_padded is new.sponge_padded


def test_scheme_classes_are_shared():
    from mlkem_triton.mldsa import Verifier as OldVerifier

    from triton_crypto.mldsa import Verifier as NewVerifier

    assert OldVerifier is NewVerifier
