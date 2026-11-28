"""The measurement harness's own guards, which are checkers and therefore need negative cases.

`bench_16p_barrier.py` refuses to record a baseline from a machine whose power state makes the
numbers incomparable. That refusal is the difference between a poisoned baseline and no baseline:
on 2026-11-27 a snapshot taken on battery wrote CPU rates 4.6x low over a good file, and the
1T-drift guard then agreed with itself because both sides were throttled.

These tests run on a machine with no accelerator and no liboqs build: they import the harness,
which reads sysfs and nothing else.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

BENCH = pathlib.Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(BENCH))

bench = pytest.importorskip("bench_16p_barrier")


def test_the_state_guard_catches_both_ways_a_machine_can_be_unrepresentative():
    """The profile and the clock, which are the two things the numbers actually depend on."""
    low_power = {"battery": "Charging", "platform_profile": "low-power", "cur_freq_mhz_max": 4700}
    throttled = {"battery": "Charging", "platform_profile": "performance", "cur_freq_mhz_max": 900}
    for state in (low_power, throttled):
        assert bench.state_warning(state), state


def test_battery_alone_is_not_a_warning_when_the_clock_is_right():
    """Measured on this machine: on battery at 99% with `powerprofilesctl set performance` it
    holds 4.7 GHz and reproduces the stored baseline to within 3%. The clock is the evidence;
    the power source is context for the metadata."""
    on_battery_full_speed = {
        "battery": "Discharging",
        "platform_profile": "performance",
        "cur_freq_mhz_max": 4700,
    }
    assert bench.state_warning(on_battery_full_speed) is None


def test_the_state_guard_stays_quiet_on_a_machine_that_is_representative():
    """The negative control: a guard that always fires is as useless as one that never does."""
    good = {"battery": "Charging", "platform_profile": "performance", "cur_freq_mhz_max": 4700}
    assert bench.state_warning(good) is None
    # ... and one without any sysfs at all must not be treated as broken
    assert bench.state_warning({"cur_freq_mhz_max": 4700}) is None


def test_power_state_reports_what_this_machine_says():
    """Whatever the state is, it must be *read*, not assumed -- the baseline's meta depends on it."""
    state = bench.power_state()
    assert isinstance(state, dict)
    if state:
        assert any(
            k.startswith("cur_freq_mhz") or k in ("battery", "platform_profile") for k in state
        )


def test_bench_rival_bytes_prefers_payload_over_traffic():
    """CPU `B/s` rivals count payload bytes, never sponge padding or read+write traffic."""
    bp = pytest.importorskip("bench_primitive")

    same = bp.Bench(fn=lambda: None, units=1024, unit="B", nbytes=1024)
    assert same.rival_bytes == 1024

    split = bp.Bench(fn=lambda: None, units=16, unit="msg", nbytes=17920, payload_bytes=16384)
    assert split.nbytes == 17920
    assert split.rival_bytes == 16384


@pytest.mark.gpu
def test_primitive_bench_payload_bytes_and_device_staging():
    """Sponge/AEAD report payload bytes for CPU comparisons, and device benches pre-stage tensors."""
    import torch

    from triton_crypto import devices

    bp = pytest.importorskip("bench_primitive")
    dev = devices.resolve_device()
    batch, size = 16, 1024
    expected_payload = batch * size

    for name in ("sha3-256", "sha3-512", "shake128", "shake256"):
        b = bp.CASES[name](batch, size, dev)
        assert b.rival_bytes == expected_payload, name
        assert b.nbytes > expected_payload, name

    aead = bp.CASES["chacha20-poly1305"](batch, size, dev)
    assert aead.rival_bytes == expected_payload
    assert aead.nbytes == 2 * expected_payload
    aead_tensors = [
        c.cell_contents
        for c in (aead.fn.__closure__ or ())
        if isinstance(c.cell_contents, torch.Tensor)
    ]
    assert any(t.device.type == dev.type and t.shape == (batch, size) for t in aead_tensors)

    for name in ("blake3", "poly1305"):
        b = bp.CASES[name](batch, size, dev)
        assert b.rival_bytes == expected_payload
        tensors = [
            c.cell_contents
            for c in (b.fn.__closure__ or ())
            if isinstance(c.cell_contents, torch.Tensor)
        ]
        assert tensors and all(t.device.type == dev.type for t in tensors), name
