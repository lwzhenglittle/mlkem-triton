"""The CPU-only contract, which no GPU machine otherwise exercises.

`docs/support.md` says a machine without an accelerator "can still smoke-test the maths and
reference paths", and `devices.resolve_device` documents that it raises with a message naming the
two numpy modules to use instead. Both are claims about a machine this suite never runs on, so
they are checked by pretending: monkeypatching the availability probes to False and asserting what
follows.
"""

from __future__ import annotations

import importlib.util

import pytest

from triton_crypto import devices


@pytest.fixture
def no_accelerator(monkeypatch):
    import torch

    monkeypatch.setattr(torch.xpu, "is_available", lambda: False, raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("MLKEM_DEVICE", raising=False)
    return devices


def test_no_devices_are_reported(no_accelerator):
    assert no_accelerator.available_devices() == []


def test_resolve_raises_with_the_documented_advice(no_accelerator):
    with pytest.raises(RuntimeError) as exc:
        no_accelerator.resolve_device()
    message = str(exc.value)
    assert "No usable GPU found" in message
    assert "does not support" in message and "CPU" in message
    # the message points a CPU-only reader at these two modules; if either is renamed the advice
    # sends them somewhere that does not exist
    for module in ("triton_crypto.mlkem.ntt_ref", "triton_crypto.mlkem.keyprep"):
        assert importlib.util.find_spec(module) is not None, module


def test_numpy_paths_work_without_a_device(no_accelerator):
    from triton_crypto.falcon import ref
    from triton_crypto.primitives import poly1305

    key = bytes.fromhex("85d6be7857556d337f4452fe42d506a80103808afb0db2fd4abff6af4149f51b")
    assert (
        poly1305.poly1305_np(key, b"Cryptographic Forum Research Group").hex()
        == "a8061dc1305136c6c22b8baf0c0127a9"
    )
    assert len(ref.hash_to_point(bytes(40), b"m")) == 512


def test_an_explicit_cpu_device_is_refused(no_accelerator):
    """`resolve_device("cpu")` must say no rather than hand back a device the kernels cannot use."""
    with pytest.raises(RuntimeError, match="no CPU backend"):
        no_accelerator.resolve_device("cpu")


def test_fused_cfg_disables_pipelining_only_on_xpu():
    """The VA 0x0 fault is an Intel triton-xpu compiler bug; CUDA/ROCm keep `_pick_cfg` stages."""
    import torch

    from triton_crypto.mlkem.kernels.fused_gemm import _fused_cfg, _pick_cfg

    assert _fused_cfg(1024, torch.device("xpu"))["num_stages"] == 1
    assert (
        _fused_cfg(1024, torch.device("cuda"))["num_stages"]
        == _pick_cfg(1024, torch.device("cuda"))["num_stages"]
    )
    assert _fused_cfg(1024, torch.device("cuda"))["num_stages"] == 3


def test_mlkem_custom_ops_use_typing_list_for_torch_2_6_compat():
    """PyTorch 2.4-2.6 custom_op schema inference rejects PEP 585 `list[Tensor]` returns."""
    import ast
    import pathlib

    import torch

    from triton_crypto.mlkem import ops as _ops  # noqa: F401 - registers custom ops

    src = (
        pathlib.Path(__file__).resolve().parent.parent / "triton_crypto" / "mlkem" / "ops.py"
    ).read_text()
    tree = ast.parse(src)
    returns = {
        node.name: ast.unparse(node.returns)
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.returns is not None
    }
    for fn_name in ("pre", "build_ye"):
        assert returns[fn_name] == "typing.List[Tensor]", (fn_name, returns[fn_name])
        schema = str(getattr(torch.ops.triton_crypto, fn_name).default._schema)
        assert "-> Tensor[]" in schema, schema
