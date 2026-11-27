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


def test_the_state_guard_catches_each_way_a_machine_can_be_unrepresentative():
    """Three distinct causes, because they look nothing alike in sysfs."""
    on_battery = {
        "battery": "Discharging",
        "platform_profile": "performance",
        "cur_freq_mhz_max": 4700,
    }
    low_power = {"battery": "Charging", "platform_profile": "low-power", "cur_freq_mhz_max": 4700}
    throttled = {"battery": "Charging", "platform_profile": "performance", "cur_freq_mhz_max": 900}
    for state in (on_battery, low_power, throttled):
        assert bench.state_warning(state), state


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
