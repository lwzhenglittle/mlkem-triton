"""The README's code blocks, as executable claims.

Every snippet in `README.md` is illustrative -- it references `dk`, `pk`, `messages` that the
reader supplies -- but the *shape* of every call in it is a promise: these names exist, these
constructors take these arguments in this order, and these constants have these values. A rename
that leaves the README untouched is exactly the kind of drift a documentation test catches and a
test suite does not.

The check is `inspect.signature` rather than execution, so it stays cheap and needs no keys; the
constants in block 4 are checked by value because they are quoted.
"""

from __future__ import annotations

import ast
import inspect
import pathlib

import pytest

from triton_crypto import Decapsulator, Encapsulator, generate_keypairs
from triton_crypto.primitives import BABYBEAR_NTT, DILITHIUM, KYBER


def _params(fn) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_mlkem_block():
    """README block 0: `Decapsulator(scheme, dk)`, `Encapsulator(scheme, ek)`,
    `generate_keypairs(scheme, n)`."""
    # the README writes `Decapsulator("ML-KEM-768", dk)`, so the two positional names matter
    assert _params(Decapsulator)[:2] == ["ps", "dk"]
    assert _params(Encapsulator)[:2] == ["ps", "ek"]
    assert _params(generate_keypairs)[:2] == ["ps", "count"]
    assert hasattr(Decapsulator, "decaps")
    assert hasattr(Encapsulator, "encapsulate")


def test_mldsa_block():
    """README block 1: `Verifier(ps, pk)` with `.verify`, `Signer(ps, sk)` with `.sign(.., ctx,
    prehash, rnd)`."""
    from triton_crypto.mldsa import Signer, Verifier

    assert _params(Verifier)[0] in ("ps", "scheme")
    assert "ctx" in _params(Signer.sign)
    assert "prehash" in _params(Signer.sign)
    assert "rnd" in _params(Signer.sign)


@pytest.mark.parametrize(("module", "cls"), [("slhdsa", "Verifier"), ("lms", "Verifier")])
def test_other_scheme_blocks(module, cls):
    """README blocks 2 and 3: the two verifiers the snippets import."""
    mod = __import__(f"triton_crypto.{module}", fromlist=[cls])
    target = getattr(mod, cls)
    assert hasattr(target, "verify")


def test_primitives_block():
    """README block 4, including the two constant tuples it quotes by value."""
    assert (KYBER.q, KYBER.zeta, KYBER.layers, KYBER.wide) == (3329, 17, 7, False)
    assert (DILITHIUM.q, DILITHIUM.zeta, DILITHIUM.layers, DILITHIUM.wide) == (
        8380417,
        1753,
        8,
        True,
    )
    assert BABYBEAR_NTT.q == 2013265921

    from triton_crypto.primitives import SHA3_256, chacha20_keystream, hash_np_batch

    # the README used to show `(key, nonce, nbytes)`, but the API is batched and takes plurals --
    # one key broadcasts against many nonces, which a singular name would hide
    assert _params(chacha20_keystream)[:3] == ["keys", "nonces", "nbytes"]
    assert "counters" in _params(chacha20_keystream)
    assert _params(hash_np_batch)[:2] == ["msgs", "hf"]
    assert SHA3_256.rate == 136


def test_the_package_parses_as_python_3_10():
    """`pyproject.toml` claims `requires-python = ">=3.10"`; this repository runs 3.14.

    The claim is only checked by the grammar version here -- no 3.10 interpreter is available on
    this machine -- so it catches syntax, not standard-library additions. That is the failure mode
    worth catching: a PEP 695 type parameter or a same-quote f-string nests in 3.12 and produces a
    `SyntaxError` for the very users the metadata promises.
    """
    root = pathlib.Path(__file__).resolve().parent.parent
    bad = []
    paths = list((root / "triton_crypto").rglob("*.py")) + list((root / "benchmarks").glob("*.py"))
    for p in paths:
        try:
            ast.parse(p.read_text(), filename=str(p), feature_version=(3, 10))
        except SyntaxError as exc:
            bad.append(f"{p.relative_to(root)}:{exc.lineno} {exc.msg}")
    assert not bad, "syntax newer than the declared floor: " + "; ".join(bad)


def test_the_documented_test_count_is_the_real_one():
    """A hard-coded count in prose drifts every time a test is added -- measured four times in one
    night. This runs `--collect-only` (which executes nothing) and compares.

    `pytest` inside `pytest` is safe here precisely because collection only imports: no kernel is
    compiled and no device is touched.
    """
    import re
    import subprocess
    import sys

    root = pathlib.Path(__file__).resolve().parent.parent
    out = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--collect-only"],
        cwd=root,
        capture_output=True,
        text=True,
    )
    match = re.search(r"(\d+) tests? collected", out.stdout)
    assert match, f"could not parse the collected count from: {out.stdout[-300:]}"
    actual = int(match.group(1))

    quoted = {}
    for name in ("README.md", "docs/support.md"):
        text = (root / name).read_text()
        quoted[name] = [int(n) for n in re.findall(r"\*\*(\d+) tests(?: green)?\*\*", text)]
    stale = {name: nums for name, nums in quoted.items() if nums and set(nums) != {actual}}
    assert not stale, f"the suite collects {actual} tests, but: {stale}"


def test_the_porting_rule_is_not_the_refuted_one():
    """`docs/patterns.md` §0 used to claim one factor decides everything; the CPU table refutes it.

    SHA-256 has SHA-NI and still wins 1.55x; BLAKE3 has no instruction and loses 0.18x. If either
    half of the old binary rule comes back into the guidance, this fails -- it is the sentence a
    reader will act on.
    """
    text = (pathlib.Path(__file__).resolve().parent.parent / "docs" / "patterns.md").read_text()
    assert "three factors" in text.lower() or "three things" in text.lower(), (
        "the corrected rule should still be there"
    )
    for refuted in (
        "if the CPU has a hardware instruction for it (AES-NI, SHA-NI, PCLMUL), expect 0.06-0.5x and\n"
        "   do not port it for throughput",
    ):
        assert refuted not in text, "the refuted binary rule is back"
    # the two counterexamples must stay named, because they are what makes the rule wrong
    assert "SHA-NI" in text and "1.55" in text
    assert "BLAKE3" in text and "0.18" in text


def test_the_readme_performance_table_matches_the_snapshot_it_names():
    """The README's benchmark table is the first numbers a reader sees, and it had drifted three rows.

    Falcon said 24.5 us e2e while the snapshot said 16.7; the 2^20 NTT said 2.2 ms while it had
    become 0.85-1.2 ms; BLAKE3 carried a ratio from an older window. The table now names its
    snapshot, and this test reads that file and checks the rows that come from it.
    """
    import re

    root = pathlib.Path(__file__).resolve().parent.parent
    readme = (root / "README.md").read_text()

    named = re.search(r"benchmarks/(snapshot-\d{8}T\d{6}Z\.md)", readme)
    assert named, "the performance table must name the snapshot it came from"
    snap = (root / "benchmarks" / named.group(1)).read_text()

    # Falcon: the core figure and both ratios are printed by the falcon section itself
    core = re.search(r"device core ([\d.]+) us/op = ([\d.]+)x 1T, ([\d.]+)x 16P", snap)
    assert core, "the snapshot should carry the falcon core line"
    falcon_row = next(
        line for line in readme.splitlines() if line.startswith("| Falcon-512 verify")
    )
    assert f"{float(core.group(1)):.2f} µs/op" in falcon_row, falcon_row
    assert f"{float(core.group(3)):.1f}×" in falcon_row, falcon_row
    assert f"{float(core.group(2)):.1f}×" in falcon_row, falcon_row

    # BLAKE3: GB/s and the two ratios
    blake = next(line for line in snap.splitlines() if line.startswith("| `blake3`"))
    gbs = re.search(r"\| ([\d.]+)G B/s", blake).group(1)
    ratios = re.findall(r"([\d.]+)x", blake)
    blake_row = next(line for line in readme.splitlines() if "BLAKE3, 1 KiB" in line)
    assert f"{float(gbs):.1f} GB/s" in blake_row, blake_row
    assert f"{float(ratios[-2]):.1f}×" in blake_row, blake_row
    assert f"{float(ratios[-1]):.2f}×" in blake_row, blake_row
