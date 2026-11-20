"""Property tests for the per-run summary: coverage, the age-time curve and the command rates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.scenarios import late_joiner
from bench.summarize import age_time_ms, summary_of, watchdog_coverage
from common.commands import MODEL_KEYS
from tests.support.runs import NS_PER_MS, NS_PER_S, T0_NS, command_rows, run_spec, synthetic_run

# One robot's commands of one model, in the order received: the time from the previous
# receipt (or from the start) to each receipt, and the command's age when it arrives.
commands = st.lists(
    st.tuples(
        st.integers(min_value=0, max_value=500 * NS_PER_MS),
        st.integers(min_value=0, max_value=500 * NS_PER_MS),
    ),
    max_size=40,
)


def receipts(start_ns: int, steps: list[tuple[int, int]]) -> tuple[Any, Any]:
    """Return the capture and receive times of the commands, receipts from start_ns."""
    receive_ns = start_ns + np.cumsum([gap_ns for gap_ns, _ in steps], dtype=np.int64)
    capture_ns = receive_ns - np.array([age_ns for _, age_ns in steps], dtype=np.int64)
    return capture_ns, receive_ns


@given(
    steps=commands,
    from_ns=st.integers(min_value=0, max_value=10 * NS_PER_S),
    length_ns=st.integers(min_value=1, max_value=10 * NS_PER_S),
    watchdog_ns=st.integers(min_value=1, max_value=NS_PER_S),
)
def test_watchdog_coverage_is_a_share_of_the_robots_time(
    steps: list[tuple[int, int]], from_ns: int, length_ns: int, watchdog_ns: int
) -> None:
    # The commands start before the window, so some are in effect when it begins and some
    # arrive after it ends.
    capture_ns, receive_ns = receipts(0, steps)

    coverage = watchdog_coverage(capture_ns, receive_ns, watchdog_ns, from_ns, from_ns + length_ns)

    assert 0.0 <= coverage <= 1.0


@given(
    steps=commands.filter(bool),
    from_ns=st.integers(min_value=0, max_value=10 * NS_PER_S),
    tail_ns=st.integers(min_value=1, max_value=500 * NS_PER_MS),
)
def test_the_age_time_bins_add_up_to_the_time_a_command_was_in_effect(
    steps: list[tuple[int, int]], from_ns: int, tail_ns: int
) -> None:
    capture_ns, receive_ns = receipts(from_ns, steps)
    until_ns = int(receive_ns[-1]) + tail_ns

    bins, uncovered_ms = age_time_ms(capture_ns, receive_ns, from_ns, until_ns)

    # A command is in effect from the first receipt to the end of the robot's time.
    assert math.fsum(bins) == pytest.approx((until_ns - int(receive_ns[0])) / NS_PER_MS)
    assert math.fsum(bins) + uncovered_ms == pytest.approx((until_ns - from_ns) / NS_PER_MS)
    assert bins.min(initial=0.0) >= -1e-9


LATE_JOINER = synthetic_run(late_joiner(), run_spec("late_joiner"))
DURATION_NS = round(LATE_JOINER.fleet.scenario.duration_s * NS_PER_S)
# Robot b joins after a has run alone.
JOIN_NS = round(LATE_JOINER.fleet.scenario.robots[1].start_s * NS_PER_S)


def receive_times(start_ns: int) -> st.SearchStrategy[list[int]]:
    """Return receipts in order, from start_ns after t0 to the end of the run."""
    offsets_ns = st.lists(st.integers(min_value=start_ns, max_value=DURATION_NS - 1), max_size=60)
    return offsets_ns.map(lambda offsets: sorted(T0_NS + offset_ns for offset_ns in offsets))


def robot_rates(summary: Mapping[str, Any], model: str, window: str) -> list[float]:
    """Return the command rate of every robot that has time in the window."""
    return [
        models[model][window]["commands_per_s"]
        for models in summary["robots"].values()
        if window in models[model]
    ]


@given(robot_a=receive_times(0), robot_b=receive_times(JOIN_NS))
def test_the_robots_command_rates_add_up_to_the_models(
    robot_a: list[int], robot_b: list[int]
) -> None:
    # Each command arrives at its capture: the rates do not depend on its age.
    commands = {
        robot_id: command_rows("cls", receive_ns, receive_ns)
        for robot_id, receive_ns in (("a", robot_a), ("b", robot_b))
    }
    run = replace(
        LATE_JOINER,
        samples={
            robot_id: replace(robot, commands=commands[robot_id])
            for robot_id, robot in LATE_JOINER.samples.items()
        },
    )

    summary = summary_of(run.run)

    windows = [window.name for window in LATE_JOINER.fleet.scenario.windows]
    assert {
        (model, window): math.fsum(robot_rates(summary, model, window))
        for model in MODEL_KEYS
        for window in windows
    } == pytest.approx(
        {
            (model, window): summary["models"][model][window]["commands_per_s"]
            for model in MODEL_KEYS
            for window in windows
        }
    )
