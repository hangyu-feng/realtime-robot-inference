"""Tests for the comparison candidates and the rule that decides a comparison's outcome."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.candidates import (
    CANDIDATES,
    FRESHNESS_TREND_FLOOR_MS_PER_S,
    MIN_EFFECT_SHARE,
    THROTTLED_FLOOR_MS_PER_S,
    Better,
    Candidate,
    Metric,
    Outcome,
    outcome,
    threshold,
)
from bench.runfiles import ModelSet
from bench.scenarios import steady
from cloud.config import Topology
from robot.settings import RobotSettings

MAIN = "models.cls.all.freshness_ms.p95"
GUARD = "models.cls.all.commands_per_s"
TREND = "models.depth.all.freshness_trend_ms_per_s"
THROTTLED = "cpu.all.throttled_ms_per_s"

# A lower main metric and a higher guard metric are better, and neither has an absolute floor.
CANDIDATE = Candidate(
    name="example",
    hypothesis="The change lowers the main metric.",
    scenario_key="steady",
    topology=Topology.SINGLE,
    model_set=ModelSet.REAL,
    a_overrides={},
    b_overrides={"skip_rule": False},
    main=Metric(MAIN, Better.LOWER),
    guards=(Metric(GUARD, Better.HIGHER),),
)
# The runs of arm A lie 4 apart around a median of zero, so the threshold is their spread.
A_AROUND_ZERO = (-2.0, 2.0)
UNCHANGED = {GUARD: (50.0, 50.0)}

DEPTH_FRESHNESS_P95 = ("models.depth.all.freshness_ms.p95", "lower")
DEPTH_RATE = ("models.depth.all.commands_per_s_per_robot.mean", "higher")
CLS_RATE = ("models.cls.all.commands_per_s_per_robot.mean", "higher")
CLS_FRESHNESS_P99 = ("models.cls.all.freshness_ms.p99", "lower")
DEPTH_CAP_1: Mapping[str, Any] = {"models": {"depth": {"forced_batch_cap": 1}}}


def test_the_candidates_are_listed_in_the_order_they_run() -> None:
    assert [candidate.name for candidate in CANDIDATES] == [
        "fifo_vs_latest",
        "cls_cap_1",
        "depth_cap_2",
        "depth_cap_4",
        "ort_default_threads",
        "depth_unpaced",
    ]


@pytest.mark.parametrize(
    ("name", "scenario_key", "a_overrides", "b_overrides", "main", "guards"),
    [
        (
            "fifo_vs_latest",
            "steady",
            {},
            {"queue_policy": "fifo", "skip_rule": False},
            (TREND, "lower"),
            (("models.depth.all.commands_per_s", "higher"),),
        ),
        (
            "cls_cap_1",
            "massive",
            {},
            {"models": {"cls": {"forced_batch_cap": 1}}},
            ("models.cls.all.freshness_ms.p95", "lower"),
            (CLS_RATE,),
        ),
        (
            "depth_cap_2",
            "massive",
            DEPTH_CAP_1,
            {"models": {"depth": {"forced_batch_cap": 2}}},
            DEPTH_FRESHNESS_P95,
            (DEPTH_RATE,),
        ),
        (
            "depth_cap_4",
            "massive",
            DEPTH_CAP_1,
            {"models": {"depth": {"forced_batch_cap": 4}}},
            DEPTH_FRESHNESS_P95,
            (DEPTH_RATE,),
        ),
        (
            "ort_default_threads",
            "sweep_n5",
            {},
            {"models": {"cls": {"threads": 0}, "depth": {"threads": 0}}},
            (THROTTLED, "lower"),
            (CLS_FRESHNESS_P99, CLS_RATE),
        ),
        (
            "depth_unpaced",
            "massive",
            {},
            {"models": {"depth": {"paced": False}}},
            (THROTTLED, "lower"),
            (CLS_FRESHNESS_P99, ("validity.fire_lag_ms.p99", "lower"), DEPTH_RATE),
        ),
    ],
)
def test_a_candidate_has_its_scenario_its_two_arms_and_its_metrics(
    name: str,
    scenario_key: str,
    a_overrides: Mapping[str, Any],
    b_overrides: Mapping[str, Any],
    main: tuple[str, str],
    guards: tuple[tuple[str, str], ...],
) -> None:
    (candidate,) = (candidate for candidate in CANDIDATES if candidate.name == name)

    assert candidate.scenario_key == scenario_key
    assert candidate.a_overrides == a_overrides
    assert candidate.b_overrides == b_overrides
    assert candidate.main == main
    assert candidate.guards == guards


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda candidate: candidate.name)
def test_every_candidate_runs_on_the_real_models_in_the_single_topology_with_a_hypothesis(
    candidate: Candidate,
) -> None:
    assert candidate.model_set is ModelSet.REAL
    assert candidate.topology is Topology.SINGLE
    assert candidate.hypothesis.endswith(".")


def test_a_trend_at_its_floor_raises_freshness_by_one_frame_period_over_the_steady_run() -> None:
    frame_period_ms = 1000.0 / RobotSettings().frame_rate_hz

    assert FRESHNESS_TREND_FLOOR_MS_PER_S * steady().duration_s == pytest.approx(frame_period_ms)


@pytest.mark.parametrize(
    ("path", "a_values", "b_values", "expected"),
    [
        (MAIN, (90.0, 100.0, 120.0), (100.0, 101.0), 30.0),
        (MAIN, (100.0, 101.0), (90.0, 100.0, 120.0), 30.0),
        (MAIN, (1000.0, 1001.0), (1000.0, 1002.0), MIN_EFFECT_SHARE * 1000.5),
        (MAIN, (-1000.0, -1000.0), (-1000.0, -1000.0), MIN_EFFECT_SHARE * 1000.0),
        (MAIN, (0.0, 0.0), (5.0, 5.0), 0.0),
        (TREND, (0.0, 0.0), (0.0, 0.0), FRESHNESS_TREND_FLOOR_MS_PER_S),
        (THROTTLED, (0.25, 0.5), (0.0, 0.5), THROTTLED_FLOOR_MS_PER_S),
        (THROTTLED, (0.0, 3.0), (0.0, 0.5), 3.0),
    ],
    ids=[
        "the spread of arm A",
        "the spread of arm B",
        "a share of the A median",
        "a share of the size of a negative A median",
        "nothing for equal runs around zero",
        "the floor of a freshness trend",
        "the floor of throttled time",
        "a spread above the floor",
    ],
)
def test_a_threshold_is_the_largest_of_both_spreads_the_share_and_the_floor(
    path: str, a_values: tuple[float, ...], b_values: tuple[float, ...], expected: float
) -> None:
    assert threshold(path, a_values, b_values) == expected


@pytest.mark.parametrize(
    ("b_main", "expected"),
    [
        (-4.5, Outcome.ACCEPTED),
        (-4.0, Outcome.INCONCLUSIVE),
        (-2.5, Outcome.INCONCLUSIVE),
        (-2.0, Outcome.REJECTED_NO_GAIN),
        (0.0, Outcome.REJECTED_NO_GAIN),
        (4.0, Outcome.REJECTED_NO_GAIN),
        (4.5, Outcome.REJECTED_REGRESSION),
    ],
    ids=[
        "better by more than the threshold",
        "better by exactly the threshold",
        "better by more than half the threshold",
        "better by exactly half the threshold",
        "unchanged",
        "worse by exactly the threshold",
        "worse by more than the threshold",
    ],
)
def test_the_main_metric_decides_the_outcome_when_no_guard_is_worse(
    b_main: float, expected: Outcome
) -> None:
    a_values = {MAIN: A_AROUND_ZERO, **UNCHANGED}
    b_values = {MAIN: (b_main, b_main), **UNCHANGED}

    assert outcome(CANDIDATE, a_values, b_values) is expected


@pytest.mark.parametrize(
    ("b_main", "expected"),
    [
        (-0.5, Outcome.ACCEPTED),
        (0.0, Outcome.REJECTED_NO_GAIN),
        (0.5, Outcome.REJECTED_REGRESSION),
    ],
)
def test_any_change_is_an_effect_when_the_runs_of_each_arm_agree_and_the_a_median_is_zero(
    b_main: float, expected: Outcome
) -> None:
    a_values = {MAIN: (0.0, 0.0), **UNCHANGED}
    b_values = {MAIN: (b_main, b_main), **UNCHANGED}

    assert outcome(CANDIDATE, a_values, b_values) is expected


@pytest.mark.parametrize(
    ("b_main", "expected"),
    [(95.5, Outcome.ACCEPTED), (96.0, Outcome.INCONCLUSIVE), (104.5, Outcome.REJECTED_REGRESSION)],
)
def test_a_change_within_the_share_of_the_a_median_is_not_an_effect(
    b_main: float, expected: Outcome
) -> None:
    a_values = {MAIN: (100.0, 100.0), **UNCHANGED}
    b_values = {MAIN: (b_main, b_main), **UNCHANGED}

    assert outcome(CANDIDATE, a_values, b_values) is expected


@pytest.mark.parametrize(
    ("b_guard", "expected"),
    [
        (-4.5, Outcome.REJECTED_GUARD),
        (-4.0, Outcome.ACCEPTED),
        (0.0, Outcome.ACCEPTED),
        (9.0, Outcome.ACCEPTED),
    ],
    ids=[
        "worse by more than the threshold",
        "worse by exactly the threshold",
        "unchanged",
        "better",
    ],
)
def test_a_guard_that_is_worse_by_more_than_its_threshold_rejects_a_gain_in_the_main_metric(
    b_guard: float, expected: Outcome
) -> None:
    a_values = {MAIN: A_AROUND_ZERO, GUARD: A_AROUND_ZERO}
    b_values = {MAIN: (-9.0, -9.0), GUARD: (b_guard, b_guard)}

    assert outcome(CANDIDATE, a_values, b_values) is expected


@pytest.mark.parametrize(
    ("b_trend", "expected"),
    [
        (FRESHNESS_TREND_FLOOR_MS_PER_S, Outcome.REJECTED_NO_GAIN),
        (FRESHNESS_TREND_FLOOR_MS_PER_S * 1.5, Outcome.REJECTED_REGRESSION),
        (-FRESHNESS_TREND_FLOOR_MS_PER_S * 0.75, Outcome.INCONCLUSIVE),
    ],
)
def test_a_freshness_trend_must_change_by_more_than_its_floor_to_be_an_effect(
    b_trend: float, expected: Outcome
) -> None:
    fifo_vs_latest = CANDIDATES[0]
    (guard,) = fifo_vs_latest.guards
    a_values = {TREND: (0.0, 0.0), guard.path: (30.0, 30.0)}
    b_values = {TREND: (b_trend, b_trend), guard.path: (30.0, 30.0)}

    assert outcome(fifo_vs_latest, a_values, b_values) is expected


def test_one_b_run_far_from_the_other_is_no_effect_though_it_moves_the_median() -> None:
    a_values = {MAIN: (100.0, 100.0), GUARD: (100.0, 100.0)}
    b_values = {MAIN: (100.0, 20.0), GUARD: (100.0, 20.0)}

    assert outcome(CANDIDATE, a_values, b_values) is Outcome.REJECTED_NO_GAIN


_VALUES = st.floats(min_value=-1e9, max_value=1e9)


@given(level=_VALUES, far_main=_VALUES, far_guard=_VALUES)
def test_one_b_run_far_from_the_other_never_accepts_or_rejects_for_a_change(
    level: float, far_main: float, far_guard: float
) -> None:
    a_values = {MAIN: (level, level), GUARD: (level, level)}
    b_values = {MAIN: (level, far_main), GUARD: (level, far_guard)}

    assert outcome(CANDIDATE, a_values, b_values) in (
        Outcome.INCONCLUSIVE,
        Outcome.REJECTED_NO_GAIN,
    )
