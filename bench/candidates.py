"""The comparisons of DESIGN.md section 12: each candidate, its metrics, and the outcome rule.

A candidate is written down here before it runs: its hypothesis, the one change of its arm
B, a main metric and guard metrics, each with the direction that counts as better. The
outcome compares the two arms' medians, metric by metric, against a threshold that the
arms' own spread sets. Nothing here reads a file or starts a run.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, NamedTuple

from bench.runfiles import ModelSet
from bench.scenarios import steady
from cloud.config import Topology
from robot.settings import RobotSettings


class Better(StrEnum):
    LOWER = "lower"
    HIGHER = "higher"


class Metric(NamedTuple):
    """A number of summary.json, by its dotted path, and the direction that counts as better."""

    path: str
    better: Better


@dataclass(frozen=True)
class Candidate:
    """One comparison: a scenario run with the overrides of arm A and with those of arm B."""

    name: str
    hypothesis: str
    scenario_key: str
    topology: Topology
    model_set: ModelSet
    # Overrides of the server config, laid over those of the model set.
    a_overrides: Mapping[str, Any]
    b_overrides: Mapping[str, Any]
    main: Metric
    guards: tuple[Metric, ...]


class Outcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED_GUARD = "rejected_guard"
    REJECTED_REGRESSION = "rejected_regression"
    INCONCLUSIVE = "inconclusive"
    REJECTED_NO_GAIN = "rejected_no_gain"


# A change below this share of the A median is not called an effect. A design choice: the
# run-to-run variation of the container spike informed its size.
MIN_EFFECT_SHARE = 0.04
# ASSUMPTION: A16. Every demo sends at the robots' default frame rate.
# One frame period over the length of the steady run: the trend at which freshness rises by
# one period within the run, which is the scale of the steady run's own check that freshness
# has no upward trend. That check takes the period from the frame rate the robots send at.
FRESHNESS_TREND_FLOOR_MS_PER_S = 1000.0 / RobotSettings().frame_rate_hz / steady().duration_s
# One millisecond of freeze per second: one hundredth of a quota period in each second, below
# any effect DESIGN.md 6.4 discusses.
THROTTLED_FLOOR_MS_PER_S = 1.0
# The absolute floor of a metric, by the last segment of its path; every other metric has none.
_ABSOLUTE_FLOORS: Mapping[str, float] = {
    "freshness_trend_ms_per_s": FRESHNESS_TREND_FLOOR_MS_PER_S,
    "throttled_ms_per_s": THROTTLED_FLOOR_MS_PER_S,
}

_DEPTH_FRESHNESS_P95 = Metric("models.depth.all.freshness_ms.p95", Better.LOWER)
_DEPTH_RATE_PER_ROBOT = Metric("models.depth.all.commands_per_s_per_robot.mean", Better.HIGHER)
_CLS_RATE_PER_ROBOT = Metric("models.cls.all.commands_per_s_per_robot.mean", Better.HIGHER)
_CLS_FRESHNESS_P99 = Metric("models.cls.all.freshness_ms.p99", Better.LOWER)
_THROTTLED = Metric("cpu.all.throttled_ms_per_s", Better.LOWER)
# With a forced cap the depth worker serves the patched graph, so an arm A that forces the
# default cap runs the same graph as arm B (DESIGN.md 4.6). The classifier's file takes any
# batch size, so its comparison keeps the defaults in arm A.
_DEPTH_DEFAULT_CAP_FORCED: Mapping[str, Any] = {"models": {"depth": {"forced_batch_cap": 1}}}

# In the order they run.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate(
        name="fifo_vs_latest",
        hypothesis=("Under FIFO, freshness grows for the whole run; under latest-wins it is flat."),
        scenario_key="steady",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides={},
        # The configuration refuses the skip rule on a queue.
        b_overrides={"queue_policy": "fifo", "skip_rule": False},
        main=Metric("models.depth.all.freshness_trend_ms_per_s", Better.LOWER),
        guards=(Metric("models.depth.all.commands_per_s", Better.HIGHER),),
    ),
    Candidate(
        name="cls_cap_1",
        hypothesis=(
            "A cap of 1 lowers the classifier's freshness tail at 50 robots without lowering "
            "its throughput (H3)."
        ),
        scenario_key="massive",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides={},
        b_overrides={"models": {"cls": {"forced_batch_cap": 1}}},
        main=Metric("models.cls.all.freshness_ms.p95", Better.LOWER),
        guards=(_CLS_RATE_PER_ROBOT,),
    ),
    Candidate(
        name="depth_cap_2",
        hypothesis=(
            "A larger depth cap spreads the dispatch cost, and lowers depth's freshness tail "
            "at 50 robots only if that cost is large (H3)."
        ),
        scenario_key="massive",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides=_DEPTH_DEFAULT_CAP_FORCED,
        b_overrides={"models": {"depth": {"forced_batch_cap": 2}}},
        main=_DEPTH_FRESHNESS_P95,
        guards=(_DEPTH_RATE_PER_ROBOT,),
    ),
    Candidate(
        name="depth_cap_4",
        hypothesis="As `depth_cap_2`, with a larger cap.",
        scenario_key="massive",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides=_DEPTH_DEFAULT_CAP_FORCED,
        b_overrides={"models": {"depth": {"forced_batch_cap": 4}}},
        main=_DEPTH_FRESHNESS_P95,
        guards=(_DEPTH_RATE_PER_ROBOT,),
    ),
    Candidate(
        name="ort_default_threads",
        hypothesis="Default thread counts oversubscribe the quota and add throttling.",
        scenario_key="sweep_n5",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides={},
        # A thread count of 0 leaves the count to ONNX Runtime.
        b_overrides={"models": {"cls": {"threads": 0}, "depth": {"threads": 0}}},
        main=_THROTTLED,
        guards=(_CLS_FRESHNESS_P99, _CLS_RATE_PER_ROBOT),
    ),
    Candidate(
        name="depth_unpaced",
        hypothesis=(
            "Without pacing, two saturated inference threads plus the frame I/O exceed the "
            "quota at 50 robots, and the kernel freezes the whole container for part of each "
            "period (H5)."
        ),
        scenario_key="massive",
        topology=Topology.SINGLE,
        model_set=ModelSet.REAL,
        a_overrides={},
        b_overrides={"models": {"depth": {"paced": False}}},
        main=_THROTTLED,
        # The depth rate puts the price of pacing in the ledger.
        guards=(
            _CLS_FRESHNESS_P99,
            Metric("validity.fire_lag_ms.p99", Better.LOWER),
            _DEPTH_RATE_PER_ROBOT,
        ),
    ),
)


def threshold(path: str, a_values: Sequence[float], b_values: Sequence[float]) -> float:
    """Return the change in a metric's median that counts as an effect, from its runs per arm.

    It is the largest of the spread of each arm, a share of the size of the A median, and
    the metric's absolute floor. Both spreads count, so that one slow run in arm B is not
    called an effect.
    """
    return max(
        max(a_values) - min(a_values),
        max(b_values) - min(b_values),
        MIN_EFFECT_SHARE * abs(statistics.median(a_values)),
        _ABSOLUTE_FLOORS.get(path.rpartition(".")[2], 0.0),
    )


def _gain(metric: Metric, a_values: Sequence[float], b_values: Sequence[float]) -> float:
    """Return how far the B median lies from the A median in the metric's better direction."""
    change = statistics.median(b_values) - statistics.median(a_values)
    return change if metric.better is Better.HIGHER else -change


def outcome(
    candidate: Candidate,
    a_values: Mapping[str, Sequence[float]],
    b_values: Mapping[str, Sequence[float]],
) -> Outcome:
    """Return a candidate's outcome from each metric's run values per arm, keyed by its path.

    Each arm needs at least one run value of the main metric and of every guard.
    """
    for guard in candidate.guards:
        a_runs, b_runs = a_values[guard.path], b_values[guard.path]
        if -_gain(guard, a_runs, b_runs) > threshold(guard.path, a_runs, b_runs):
            return Outcome.REJECTED_GUARD
    a_runs, b_runs = a_values[candidate.main.path], b_values[candidate.main.path]
    gain = _gain(candidate.main, a_runs, b_runs)
    effect = threshold(candidate.main.path, a_runs, b_runs)
    if gain > effect:
        return Outcome.ACCEPTED
    if -gain > effect:
        return Outcome.REJECTED_REGRESSION
    if gain > effect / 2:
        return Outcome.INCONCLUSIVE
    return Outcome.REJECTED_NO_GAIN
