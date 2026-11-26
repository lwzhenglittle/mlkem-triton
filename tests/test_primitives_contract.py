"""What counts as "a primitive is done", as a red light instead of a convention.

The 0.8.0 review (`eval-repo-structure-0.8.0.md`, P1-1) asked for this and it has been owed
since. Every module under `primitives/` must:

1. be re-exported from `triton_crypto.primitives` and listed in its `__all__`;
2. ship a numpy reference (`*_np`, `*_np_batch` or `*_ref`) -- the oracle for the device path;
3. ship a device path (`*_torch`, a kernel factory such as `get_kernel`/`get_sponge`, a grid
   launch, or an inline `@triton.jit`), or be listed in `HOST_ONLY` with a reason;
4. have tests: `tests/test_<module>*.py`, or coverage in the shared `test_primitives*.py`;
5. have its kernel contract written down in `docs/kernels.md`.

The point is not the list, it is that adding a module without one of these turns the suite
red. `fields.py` is the only host-only module: it is arithmetic that other modules' kernels
call, so it has no batch entry point of its own.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from triton_crypto import primitives

PRIMITIVES = Path(__file__).resolve().parent.parent / "triton_crypto" / "primitives"
KERNELS_DOC = Path(__file__).resolve().parent.parent / "docs" / "kernels.md"
TESTS = Path(__file__).resolve().parent

HOST_ONLY = {
    "fields": "shared modular arithmetic; kernels that use it live in ntt/ and poseidon2",
}

MODULES = sorted(p.stem for p in PRIMITIVES.glob("*.py") if p.stem not in {"__init__"}) + sorted(
    p.name for p in PRIMITIVES.iterdir() if p.is_dir() and (p / "__init__.py").exists()
)


def _source(name: str) -> str:
    path = PRIMITIVES / name
    if path.is_dir():
        return "\n".join(p.read_text() for p in sorted(path.glob("*.py")))
    return (PRIMITIVES / f"{name}.py").read_text()


GRID_LAUNCH = re.compile(r"\b\w+\s*\[\s*\(?\s*triton\.cdiv|\b\w+\s*\[\s*grid\s*\]")


def _jit_kernels(name: str) -> list[str]:
    """Every function in the module decorated with `@triton.jit`, by source order."""
    found = []
    paths = (
        [PRIMITIVES / name / "kernel.py"]
        if (PRIMITIVES / name).is_dir()
        else [PRIMITIVES / f"{name}.py"]
    )
    for path in paths:
        if not path.exists():
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            for dec in node.decorator_list:
                target = dec.func if isinstance(dec, ast.Call) else dec
                if ast.unparse(target) in ("triton.jit", "tl.jit", "jit"):
                    found.append(node.name)
    return found


def test_modules_are_discoverable():
    assert MODULES, "no primitive modules found -- the glob is wrong"
    assert "poseidon2" in MODULES and "chacha20" in MODULES


@pytest.mark.parametrize("name", MODULES)
def test_exported_and_in_dunder_all(name: str):
    assert hasattr(primitives, name), f"`{name}` is not re-exported from triton_crypto.primitives"
    assert name in primitives.__all__, f"`{name}` is missing from primitives.__all__"


@pytest.mark.parametrize("name", MODULES)
def test_has_a_numpy_reference(name: str):
    module = getattr(primitives, name)
    refs = [
        a
        for a in dir(module)
        if a.endswith(("_np", "_np_batch", "_ref")) and callable(getattr(module, a))
    ]
    assert refs, f"{name}: no `*_np`/`*_ref` reference -- the device path has no oracle"


@pytest.mark.parametrize("name", MODULES)
def test_has_a_device_path_or_says_why_not(name: str):
    module = getattr(primitives, name)
    src = _source(name)
    dev = [a for a in dir(module) if a.endswith("_torch") and callable(getattr(module, a))]
    has_dev = (
        bool(dev)
        or _jit_kernels(name)
        or any(hasattr(module, f) for f in ("get_kernel", "get_sponge", "compile_generated"))
        or bool(GRID_LAUNCH.search(src))
    )
    if name in HOST_ONLY:
        assert HOST_ONLY[name].strip(), "host-only modules must carry a reason"
        return
    assert has_dev, f"{name}: no device entry point found (torch fn, kernel, or grid launch)"


@pytest.mark.parametrize("name", MODULES)
def test_has_tests(name: str):
    hits = sorted(p.name for p in TESTS.glob(f"test_{name}*.py"))
    if hits:
        return
    # Keccak and the sponge share the primitive-level test files; that is still coverage.
    shared = "\n".join(p.read_text() for p in TESTS.glob("test_primitives*.py"))
    assert f"primitives.{name}" in shared or f"primitives import {name}" in shared, (
        f"{name}: no tests/test_{name}*.py and no coverage in test_primitives*.py"
    )


@pytest.mark.parametrize("name", MODULES)
def test_kernel_contract_is_documented(name: str):
    """The doc has to name the module it documents, so a new primitive cannot land with no
    kernel contract just because its kernel happens to be underscore-prefixed."""
    doc = KERNELS_DOC.read_text()
    assert f"primitives/{name}.py" in doc or f"primitives/{name}/" in doc, (
        f"{name}: no kernel contract in docs/kernels.md (expected 'primitives/{name}.py')"
    )


def test_every_kernel_in_the_tree_is_in_the_doc():
    """Not just one line per module: every `@triton.jit` kernel has to be named in
    `docs/kernels.md`, so a kernel added later cannot drift out of the contract.

    This is how `_hash_to_point_kernel` was found -- it had been in `falcon/kernels.py` for
    several commits while the doc described the module it lived in and nothing more.
    """
    doc = KERNELS_DOC.read_text()
    root = TESTS.parent / "triton_crypto"
    defined = set()
    for path in root.rglob("*.py"):
        for m in re.finditer(r"^def (_[a-z0-9_]+_kernels?)\b", path.read_text(), re.M):
            defined.add((m.group(1), path.parent.name))
    assert defined, "no kernels found -- the scan is wrong, not the tree"
    # Same hardening as the export test: ask about the Summary table, which is what claims to list
    # every kernel, not about the document as a whole.
    summary = doc[doc.index("## Summary") : doc.index("## The exported surface")]
    missing = sorted(f"{name} ({mod}/)" for name, mod in defined if name not in summary)
    assert not missing, f"kernels with no row in the Summary table of docs/kernels.md: {missing}"


def test_every_primitive_has_a_bench_case_or_says_why_not():
    """`bench_primitive.py` is the only place the primitive layer is measured; a module that
    is not in it has no number, which is how the structure review described the old state."""
    bench = (TESTS.parent / "benchmarks" / "bench_primitive.py").read_text()
    # The harness may legitimately leave out an arithmetic-only module.
    missing = [m for m in MODULES if m not in bench and m not in HOST_ONLY]
    assert not missing, f"primitives with no bench_primitive case: {missing}"


# The vectors the suite leans on, with the smallest count that still means something. A test that
# loops over a vector file passes vacuously if the file is empty or truncated, and nothing in the
# suite would notice -- the same silent-success shape as a checker whose parser matches nothing.
TEST_VECTORS = {
    "data/vectors/blake3/kat.json": ("vectors", 26),
    "data/vectors/poseidon2/bb16.json": ("vectors", 64),
}


def test_the_vector_files_are_not_empty():
    import json

    root = Path(__file__).resolve().parent.parent
    for rel, (key, floor) in TEST_VECTORS.items():
        data = json.loads((root / rel).read_text())
        entries = data.get(key, data)
        assert len(entries) >= floor, f"{rel}: {len(entries)} entries, expected at least {floor}"


def test_the_acvp_directories_are_populated():
    """Same argument for the ACVP corpora: an empty directory would make several suites vacuous."""
    import json

    root = Path(__file__).resolve().parent.parent / "data"
    # each corpus is a few fat files, so count test *groups* inside them rather than file names
    floors = {"acvp": 20, "acvp-mldsa": 50, "acvp-slhdsa": 20, "acvp-lms": 10}
    for name, floor in floors.items():
        total = 0
        for f in (root / name).rglob("*.json"):
            data = json.loads(f.read_text())
            if isinstance(data, list):  # the LMS files are a bare list of cases
                total += len(data)
            else:
                total += sum(len(data[k]) for k in ("testGroups", "groups") if k in data)
        assert total >= floor, f"data/{name}: {total} groups, expected >= {floor}"


def test_the_bench_labels_do_not_contradict_the_cases():
    """`--list` prints these labels, so a label is the first thing a reader sees.

    One said `gemm-int8 ... 256x256x256` while the case had moved to 1024^3 -- the same
    size-drift that made a headline number quotable at the wrong scale. Cheap to pin: the label
    must mention the size the case's own docstring states.
    """
    import ast
    import re

    src = (Path(__file__).resolve().parent.parent / "benchmarks" / "bench_primitive.py").read_text()
    labels = dict(re.findall(r'^    "([a-z0-9\-\^]+)": "([^"]+)",$', src, re.M))
    docs = {
        node.name: ast.get_docstring(node) or ""
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.FunctionDef) and node.name.startswith("bench_")
    }
    case_for = {"gemm-int8": "bench_gemm_int8", "ntt-2^20": "bench_ntt_large"}
    for name, fn in case_for.items():
        label, doc = labels[name], docs[fn]
        for size in ("256", "512", "1024", "2048"):
            if f"{size}x{size}x{size}" in label or f"{size}^3" in label:
                assert size in doc, f"{name}: label says {size}^3, the case's docstring does not"


def test_every_exported_primitive_is_in_the_doc():
    """The namespace `triton_crypto.primitives` exports is the API a user can call, so every name
    in it should have a contract in `docs/kernels.md`.

    This is the gap the kernel scan above leaves: it matches `_..._kernel` only, so a *public*
    function can be added and documented nowhere -- `gf2_8_matmul_torch` was, for four commits.
    A first attempt at this test demanded an entry for every public `def` in the package, which is
    wrong: the numpy references and helpers are internal, and the doc's contract is the exported
    namespace.
    """
    import ast

    doc = KERNELS_DOC.read_text()
    src = (TESTS.parent / "triton_crypto" / "primitives" / "__init__.py").read_text()
    exported = {
        alias.asname or alias.name
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    assert len(exported) > 50, "the export scan is wrong, not the tree"
    # The first version of this test asked "does the name appear anywhere in the document", and
    # nine exports that were only in the narrative sections passed while the summary table was
    # missing them. Ask about the table instead.
    table_start = doc.index("## The exported surface")
    table = doc[table_start : doc.index("## Primitives", table_start)]
    # `annotations` is a `__future__` feature object that `from __future__ import annotations`
    # leaves bound in the namespace; it is genuinely reachable, so the doc names it explicitly
    # rather than the scan special-casing it.
    missing = sorted(name for name in exported if name not in table)
    assert not missing, (
        f"exported from primitives but absent from the exported-surface table in "
        f"docs/kernels.md: {missing}"
    )

    # ...and the other direction: a name the table promises must actually be importable. The wheel
    # is where this bites -- `gf2_8_matmul_torch` was documented as an export for two commits while
    # `primitives/__init__.py` never imported it, so the documented call raised ImportError from an
    # installed wheel and worked from a source checkout, where the test's own `sys.path` hid it.
    import triton_crypto.primitives as package

    # The first version of this check was circular: it only inspected names that were *both*
    # exported and in the table, so a name in the table but missing from `__init__` -- exactly the
    # `gf2_8_matmul_torch` failure -- was invisible. Read the table itself instead.
    # bare identifiers are top-level promises; a dotted path or a `_kernel` name is a submodule
    # attribute, which is a different claim
    promised = {
        tok
        for tok in re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", table)
        if not tok.startswith("_") and tok not in {"annotations", "__future__"}
    }
    unimplemented = sorted(name for name in promised if not hasattr(package, name))
    assert not unimplemented, (
        f"docs promise these imports but the package lacks them: {unimplemented}"
    )
    in_all = set(getattr(package, "__all__", ()))
    missing_from_all = sorted(name for name in promised if name not in in_all)
    assert not missing_from_all, (
        f"exported in the table but absent from __all__: {missing_from_all}"
    )


def test_the_package_description_names_what_shipped():
    """`pyproject.toml`'s description is what `pip show` and PyPI display.

    It listed the four schemes and five primitives from the previous milestone; Falcon, BLAKE3,
    Poseidon2, AES and the int8 GEMM all shipped since and none were named. Nobody reads a
    description the way they read a README, so it drifts silently -- this pins the names that must
    appear, and fails when a new scheme or primitive is added without updating it.
    """
    import re

    root = Path(__file__).resolve().parent.parent
    text = (root / "pyproject.toml").read_text()
    description = re.search(r'^description = "([^"]+)"', text, re.M).group(1).lower()
    for name in (
        "ml-kem",
        "ml-dsa",
        "slh-dsa",
        "lms",
        "falcon",
        "keccak",
        "sha-2",
        "ntt",
        "chacha20",
        "aes",
        "blake3",
        "poseidon2",
        "gemm",
    ):
        assert name in description, f"pyproject description does not mention {name}"


def test_bench_cpu_refuses_to_shrink_the_baseline():
    """`--only x` without `--merge` replaced `cpu_baseline.json` with a one-rival file.

    Every ratio in every document is computed against that file, so the loss is silent and total;
    it happened in this session and nothing complained. The script now refuses, and this pins it.
    """
    import subprocess

    root = Path(__file__).resolve().parent.parent
    baseline = root / "benchmarks" / "cpu_baseline.json"
    before = baseline.read_text()
    result = subprocess.run(
        [sys.executable, "benchmarks/bench_cpu.py", "--only", "mldsa-44-verify"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2, result.stdout + result.stderr
    assert "--merge" in result.stderr, result.stderr
    assert baseline.read_text() == before, "the baseline changed despite the refusal"
