"""Tests for the capacity model: its input reader, the command gap bound and the predictions."""

import copy
import functools
import json
import operator
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bench.capacity_model import (
    QUOTA_CORES,
    CapacityInput,
    CapacityInputError,
    command_gap_bound_ns,
    longest_cycle_ns,
    main,
    parse_input,
    predict,
)
from bench.scenarios import disconnect, late_joiner, massive, steady, sweep
from bench.simulation import Overheads, PhaseMode, ServiceTable, SimulationResult, simulate
from bench.stats import summarize
from cloud.config import ServerConfig
from common.commands import MODEL_KEYS

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
CONFIG = ServerConfig()
MODELS = CONFIG.models
# The period every simulated robot declares, in whole nanoseconds.
FRAME_PERIOD_MS = round(NS_PER_S / CONFIG.frame_rate_hz) / NS_PER_MS
# Four different handling times, so that the recorded inputs show each under its own key.
OVERHEADS = Overheads(
    transport_up_ns=NS_PER_MS,
    dispatch_request_ns=NS_PER_MS // 2,
    dispatch_result_ns=NS_PER_MS // 4,
    deliver_ns=2 * NS_PER_MS,
)
# The measured cost of the frame I/O is not the configured 0.15 ms per frame, so that the
# predictions show which of the two each value takes.
IO_CPU_MS_PER_FRAME = 0.4
# The wall time per frame of every batch. The classifier's cap is then 8 and depth's 2
# (DESIGN.md 6.2).
FRAME_MS = {"cls": 5.0, "depth": 40.0}
# A batch's CPU time over its wall time: not 1, so that the predictions show which of the two
# each value takes.
CPU_PER_WALL = 1.5
# Depth's service time alone is above its 150 ms threshold, so every depth command is stale.
STALE_DEPTH_FRAME_MS = {"cls": 5.0, "depth": 200.0}
# Depth gets a pacing share at 10 robots, and none at 1 or 2.
SHORT_CURVE = (1, 2, 10)
# Not the default seed 0, so that the predictions show which seed they were made with. Under
# this seed robot c's depth rate before its stop also differs from the survivors' mean, so that
# the disconnect facts show which robots they cover.
SEED = 2

FACT_KEYS = {
    "predict.steady.cls.freshness_ms.p50",
    "predict.steady.cls.freshness_ms.p95",
    "predict.steady.cls.freshness_ms.p99",
    "predict.steady.cls.freshness_ms.max",
    "predict.steady.depth.freshness_ms.p50",
    "predict.steady.depth.freshness_ms.p95",
    "predict.steady.depth.freshness_ms.p99",
    "predict.steady.depth.freshness_ms.max",
    "predict.steady.cls.commands_per_s",
    "predict.steady.depth.commands_per_s",
    "predict.steady.cls.command_age_peak_ms.p95",
    "predict.steady.depth.command_age_peak_ms.p95",
    "predict.late_joiner.after.a.cls.commands_per_s",
    "predict.late_joiner.after.a.depth.commands_per_s",
    "predict.sweep.n1.cls.commands_per_s_per_robot.mean",
    "predict.sweep.n3.cls.commands_per_s_per_robot.mean",
    "predict.sweep.n5.cls.commands_per_s_per_robot.mean",
    "predict.sweep.n10.cls.commands_per_s_per_robot.mean",
    "predict.sweep.n1.depth.commands_per_s_per_robot.mean",
    "predict.sweep.n3.depth.commands_per_s_per_robot.mean",
    "predict.sweep.n5.depth.commands_per_s_per_robot.mean",
    "predict.sweep.n10.depth.commands_per_s_per_robot.mean",
    "predict.disconnect.before.survivors.depth.commands_per_s_per_robot",
    "predict.disconnect.after.survivors.depth.commands_per_s_per_robot",
    "predict.massive.cls.commands_per_s_per_robot.mean",
    "predict.massive.depth.commands_per_s_per_robot.mean",
}
CURVE_FIELDS = {
    "commands_per_s_per_robot",
    "freshness_ms_p50",
    "freshness_ms_p95",
    "command_gap_ms_max",
    "command_gap_bound_ms",
    "batch_size_mean",
    "utilization",
}
# Each fact that is one robot's commands per second, or a mean of them, with its window's length.
PER_ROBOT_RATE_FACTS = {
    **{f"predict.steady.{model}.commands_per_s": 30.0 for model in MODEL_KEYS},
    **{f"predict.late_joiner.after.a.{model}.commands_per_s": 20.0 for model in MODEL_KEYS},
    **{
        f"predict.sweep.n{count}.{model}.commands_per_s_per_robot.mean": 30.0
        for count in (1, 3, 5, 10)
        for model in MODEL_KEYS
    },
    "predict.disconnect.before.survivors.depth.commands_per_s_per_robot": 20.0,
    "predict.disconnect.after.survivors.depth.commands_per_s_per_robot": 20.0,
    **{f"predict.massive.{model}.commands_per_s_per_robot.mean": 30.0 for model in MODEL_KEYS},
}
# Each per-robot mean over every robot of a 30 s run, with its model and the run's robot count.
FLEET_RATE_FACTS = [
    (f"predict.{run}.{model}.commands_per_s_per_robot.mean", model, robot_count)
    for run, robot_count in [
        ("sweep.n1", 1),
        ("sweep.n3", 3),
        ("sweep.n5", 5),
        ("sweep.n10", 10),
        ("massive", 50),
    ]
    for model in MODEL_KEYS
]
# The cycles that each bound of SHORT_CURVE counts: ceil(N / cap) to reach a robot once its
# frame is pending, and one more between the two writes that end a gap. A classifier batch of
# up to 8 frames is back 0.75 ms plus 5 ms per frame after its dispatch, and a depth batch of
# up to 2 frames after up to 80.75 ms. At 10 robots the smallest pacing share the run sets,
# 0.855 cores once the tenth robot has started, holds depth after a batch of 2 for its 120 ms
# of CPU time over the share, which is longer.
GAP_CYCLES_MS = {
    "cls": [2 * 40.75, 2 * 40.75, 3 * 40.75],
    "depth": [2 * 80.75, 2 * 80.75, 6 * 120 / 0.855],
}
WALL_MS = ("models", "cls", "threads", "1", "2", "wall_ms")

# The input file: sizes 1 to 8 at the configured single thread, an extra size the interpolation
# may use, and another thread count that is not read.
VALID_DOCUMENT: dict[str, Any] = {
    "models": {
        "cls": {
            "threads": {
                "1": {
                    "1": {"wall_ms": 5.0, "cpu_ms": 4.5},
                    "2": {"wall_ms": 9.0, "cpu_ms": 8.5},
                    "3": {"wall_ms": 13.0, "cpu_ms": 12.5},
                    "4": {"wall_ms": 17.25, "cpu_ms": 16},
                    "8": {"wall_ms": 33.0, "cpu_ms": 31.0},
                },
                "2": {"1": {"wall_ms": "not read"}},
            }
        },
        "depth": {
            "threads": {
                "1": {
                    "1": {"wall_ms": 60.0, "cpu_ms": 58.0},
                    "2": {"wall_ms": 118.0, "cpu_ms": 115.0},
                    "4": {"wall_ms": 230.0, "cpu_ms": 226.0},
                    "8": {"wall_ms": 455.0, "cpu_ms": 450.0},
                }
            }
        },
    },
    "overheads_ms": {
        "transport_up": 1.0,
        "dispatch_request": 0.5,
        "dispatch_result": 0.25,
        "deliver": 2,
    },
    "io_cpu_ms_per_frame": 0.2,
    "label": "two single-thread workers",
}
VALID_INPUT = CapacityInput(
    tables={
        "cls": ServiceTable(
            wall_ns={1: 5_000_000, 2: 9_000_000, 3: 13_000_000, 4: 17_250_000, 8: 33_000_000},
            cpu_ns={1: 4_500_000, 2: 8_500_000, 3: 12_500_000, 4: 16_000_000, 8: 31_000_000},
        ),
        "depth": ServiceTable(
            wall_ns={1: 60_000_000, 2: 118_000_000, 4: 230_000_000, 8: 455_000_000},
            cpu_ns={1: 58_000_000, 2: 115_000_000, 4: 226_000_000, 8: 450_000_000},
        ),
    },
    overheads=Overheads(
        transport_up_ns=1_000_000,
        dispatch_request_ns=500_000,
        dispatch_result_ns=250_000,
        deliver_ns=2_000_000,
    ),
    io_cpu_ms_per_frame=0.2,
    label="two single-thread workers",
)


def tables(frame_ms: Mapping[str, float]) -> dict[str, ServiceTable]:
    """Return tables in which a batch takes frame_ms of wall time per frame of each model.

    Its CPU time is CPU_PER_WALL times its wall time.
    """
    return {
        model: ServiceTable(
            wall_ns={size: round(size * time_ms * NS_PER_MS) for size in (1, 2, 4, 8)},
            cpu_ns={
                size: round(size * time_ms * CPU_PER_WALL * NS_PER_MS) for size in (1, 2, 4, 8)
            },
        )
        for model, time_ms in frame_ms.items()
    }


TABLES = tables(FRAME_MS)


def capacity_hz(model: str, batch_cap: int) -> float:
    """Return the most frames per second a worker serves, with no pacing, from FRAME_MS."""
    dispatch_ms = (OVERHEADS.dispatch_request_ns + OVERHEADS.dispatch_result_ns) / NS_PER_MS
    return max(
        size * 1_000 / (size * FRAME_MS[model] + dispatch_ms) for size in range(1, batch_cap + 1)
    )


def fleet(robot_count: int) -> list[str]:
    """Return the robot ids of a sweep or massive run."""
    return [f"r{number:02d}" for number in range(1, robot_count + 1)]


def in_ms(samples_ns: Iterable[int]) -> list[float]:
    return [ns / NS_PER_MS for ns in samples_ns]


def mean_rate(result: SimulationResult, window: str, model: str, robot_ids: Iterable[str]) -> float:
    """Return the mean commands per second that the robots received from the model in the window."""
    return statistics.fmean(
        result.windows[window][model][robot_id].commands_per_s for robot_id in robot_ids
    )


def sweep_statistics(result: SimulationResult, model: str) -> dict[str, float]:
    """Return the curve values of a model that are statistics of one sweep run."""
    robots = result.windows["all"][model].values()
    freshness_ms = summarize(in_ms(ns for robot in robots for ns in robot.freshness_ns))
    batch_size = result.models[model].batch_size
    return {
        "commands_per_s_per_robot": statistics.fmean(robot.commands_per_s for robot in robots),
        "freshness_ms_p50": freshness_ms["p50"],
        "freshness_ms_p95": freshness_ms["p95"],
        "command_gap_ms_max": max(robot.command_gap_max_ns for robot in robots) / NS_PER_MS,
        "batch_size_mean": sum(size * count for size, count in batch_size.items())
        / sum(batch_size.values()),
    }


def document_without(path: tuple[str, ...]) -> bytes:
    document = copy.deepcopy(VALID_DOCUMENT)
    del functools.reduce(operator.getitem, path[:-1], document)[path[-1]]
    return json.dumps(document).encode()


def document_with(path: tuple[str, ...], value: object) -> bytes:
    document = copy.deepcopy(VALID_DOCUMENT)
    functools.reduce(operator.getitem, path[:-1], document)[path[-1]] = value
    return json.dumps(document).encode()


@pytest.fixture(scope="module")
def predictions() -> dict[str, Any]:
    return predict(
        CONFIG, TABLES, OVERHEADS, IO_CPU_MS_PER_FRAME, QUOTA_CORES, SEED, SHORT_CURVE, "linear"
    )


@pytest.fixture(scope="module")
def predictions_without_skip_rule() -> dict[str, Any]:
    config = replace(CONFIG, skip_rule=False)
    return predict(
        config, TABLES, OVERHEADS, IO_CPU_MS_PER_FRAME, QUOTA_CORES, SEED, SHORT_CURVE, "linear"
    )


@pytest.fixture(scope="module")
def runs() -> dict[str, SimulationResult]:
    """Return the runs behind the predictions fixture, simulated again from the same inputs."""
    scenarios = {
        "steady": steady(),
        "late_joiner": late_joiner(),
        "disconnect": disconnect(),
        "massive": massive(),
        # The sweeps of the facts, and those of the curves.
        **{f"sweep.n{count}": sweep(count) for count in {1, 3, 5, 10, *SHORT_CURVE}},
    }
    return {
        name: simulate(
            CONFIG,
            scenario,
            TABLES,
            OVERHEADS,
            IO_CPU_MS_PER_FRAME,
            QUOTA_CORES,
            PhaseMode.RANDOM,
            SEED,
        )
        for name, scenario in scenarios.items()
    }


@pytest.fixture(scope="module")
def expected_facts(runs: dict[str, SimulationResult]) -> dict[str, float]:
    """Return each fact from the runs, over the window and robots and with the statistic it names.

    A per-robot mean covers the robots present for the whole window, and the survivors are the
    robots that run to the end.
    """
    facts: dict[str, float] = {}
    for model in MODEL_KEYS:
        steady_robot = runs["steady"].windows["all"][model]["a"]
        freshness_ms = summarize(in_ms(steady_robot.freshness_ns))
        for name in ("p50", "p95", "p99", "max"):
            facts[f"predict.steady.{model}.freshness_ms.{name}"] = freshness_ms[name]
        facts[f"predict.steady.{model}.commands_per_s"] = steady_robot.commands_per_s
        facts[f"predict.steady.{model}.command_age_peak_ms.p95"] = summarize(
            in_ms(steady_robot.command_age_peak_ns)
        )["p95"]
        facts[f"predict.late_joiner.after.a.{model}.commands_per_s"] = (
            runs["late_joiner"].windows["after"][model]["a"].commands_per_s
        )
        for count in (1, 3, 5, 10):
            facts[f"predict.sweep.n{count}.{model}.commands_per_s_per_robot.mean"] = mean_rate(
                runs[f"sweep.n{count}"], "all", model, fleet(count)
            )
        facts[f"predict.massive.{model}.commands_per_s_per_robot.mean"] = mean_rate(
            runs["massive"], "all", model, fleet(50)
        )
    for window in ("before", "after"):
        facts[f"predict.disconnect.{window}.survivors.depth.commands_per_s_per_robot"] = mean_rate(
            runs["disconnect"], window, "depth", ("a", "b")
        )
    return facts


@pytest.mark.parametrize(
    ("robot_count", "batch_cap", "skip_max_unserved_ns", "bound_ns"),
    [
        (1, 1, None, 100 + 2 * 1_000),
        (8, 8, None, 100 + 2 * 1_000),
        (9, 8, None, 100 + 3 * 1_000),
        (50, 8, None, 100 + 8 * 1_000),
        (50, 8, 5_000, 100 + 8 * 1_000 + 5_000),
        (3, 1, 0, 100 + 4 * 1_000),
    ],
)
def test_the_gap_bound_is_a_period_and_a_cycle_more_than_the_cycles_to_reach_every_robot(
    robot_count: int, batch_cap: int, skip_max_unserved_ns: int | None, bound_ns: int
) -> None:
    assert (
        command_gap_bound_ns(robot_count, batch_cap, 100, 1_000, skip_max_unserved_ns) == bound_ns
    )


@pytest.mark.parametrize(
    ("batch_cap", "shares_cores", "cycle_ns"),
    [
        pytest.param(4, [], 2 + 40 + 3, id="unpaced"),
        pytest.param(3, [], 2 + 30 + 3, id="interpolated-cap"),
        pytest.param(4, [2.0], 2 + 40 + 3, id="hold-shorter-than-the-batch"),
        # The hold counts the CPU time: the wall time over the share would end within the batch.
        pytest.param(4, [1.0], 60, id="hold-longer-than-the-batch"),
        pytest.param(4, [2.0, 0.5], 120, id="the-smallest-share-counts"),
        # The float 0.3 lies just under three tenths, so 60 ns of CPU time need just over 200.
        pytest.param(4, [0.3], 201, id="hold-rounded-up"),
    ],
)
def test_the_longest_cycle_is_the_batch_and_its_dispatch_or_the_pacing_hold_if_longer(
    batch_cap: int, shares_cores: list[float], cycle_ns: int
) -> None:
    # A batch's CPU time is 1.5 times its wall time.
    table = ServiceTable(wall_ns={1: 10, 2: 20, 4: 40, 8: 80}, cpu_ns={1: 15, 2: 30, 4: 60, 8: 120})
    overheads = Overheads(
        transport_up_ns=0, dispatch_request_ns=2, dispatch_result_ns=3, deliver_ns=0
    )

    assert longest_cycle_ns(table, overheads, batch_cap, shares_cores) == cycle_ns


def test_the_predictions_have_exactly_the_fact_keys_and_one_curve_value_per_robot_count(
    predictions: dict[str, Any],
) -> None:
    curves = predictions["curves"]
    counts = len(SHORT_CURVE)

    assert set(predictions) == {"label", "inputs", "facts", "curves"}
    assert set(predictions["facts"]) == FACT_KEYS
    assert len(FACT_KEYS) == 26
    assert set(curves) == {"robots", "cpu_demand_cores", *MODEL_KEYS}
    assert (curves["robots"], len(curves["cpu_demand_cores"])) == (list(SHORT_CURVE), counts)
    assert {model: set(curves[model]) for model in MODEL_KEYS} == dict.fromkeys(
        MODEL_KEYS, CURVE_FIELDS
    )
    assert {
        (model, field): len(values)
        for model in MODEL_KEYS
        for field, values in curves[model].items()
    } == {(model, field): counts for model in MODEL_KEYS for field in CURVE_FIELDS}


def test_the_inputs_record_the_settings_and_the_plan_each_model_ran_with(
    predictions: dict[str, Any],
) -> None:
    inputs = predictions["inputs"]

    assert (inputs["seed"], inputs["quota_cores"], inputs["robot_counts"]) == (
        SEED,
        2.0,
        list(SHORT_CURVE),
    )
    assert (inputs["io_cpu_ms_per_frame"], inputs["configured_io_cpu_ms_per_frame"]) == (
        IO_CPU_MS_PER_FRAME,
        CONFIG.io_cpu_ms_per_frame,
    )
    assert inputs["overheads_ms"] == {
        "transport_up": 1.0,
        "dispatch_request": 0.5,
        "dispatch_result": 0.25,
        "deliver": 2.0,
    }
    assert inputs["tables"]["depth"] == {
        "threads": 1,
        "wall_ms": {"1": 40.0, "2": 80.0, "4": 160.0, "8": 320.0},
        "cpu_ms": {"1": 60.0, "2": 120.0, "4": 240.0, "8": 480.0},
    }
    # Depth's budget is its 150 ms threshold less one frame period and the 10 ms allowance.
    assert inputs["plans"]["depth"] == {
        "batch_cap": 2,
        "service_ms": 80.0,
        "return_allowance_ms": 10.0,
        "skip_rule": True,
        "skip_max_unserved_ms": 1000.0,
        "budget_ms": pytest.approx(150 - 1000 / 30 - 10, abs=1e-6),
        "margin_ms": pytest.approx(150 - 1000 / 30 - 10 - 80, abs=1e-6),
        "feasible": True,
    }


@pytest.mark.parametrize("key", sorted(FACT_KEYS))
def test_each_fact_is_its_statistic_over_the_window_and_the_robots_it_names(
    predictions: dict[str, Any], expected_facts: dict[str, float], key: str
) -> None:
    assert predictions["facts"][key] == pytest.approx(expected_facts[key], rel=1e-12)


def test_the_survivors_before_the_stop_leave_out_the_robot_that_stops(
    predictions: dict[str, Any], runs: dict[str, SimulationResult]
) -> None:
    key = "predict.disconnect.before.survivors.depth.commands_per_s_per_robot"

    # Robot c also ran for the whole window, at another rate.
    with_c_per_s = mean_rate(runs["disconnect"], "before", "depth", ("a", "b", "c"))

    assert predictions["facts"][key] != pytest.approx(with_c_per_s)


@pytest.mark.parametrize(("key", "window_s"), PER_ROBOT_RATE_FACTS.items())
def test_a_robot_never_gets_more_than_the_frame_rate_plus_one_command_over_the_window(
    predictions: dict[str, Any], key: str, window_s: float
) -> None:
    # A window need not be a whole number of frame periods, so it can hold one frame more than
    # its length times the rate.
    assert predictions["facts"][key] <= CONFIG.frame_rate_hz + 1 / window_s


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_curve_never_gives_a_robot_more_than_the_frame_rate_plus_one_command_over_the_run(
    predictions: dict[str, Any], model: str
) -> None:
    rates_per_s = predictions["curves"][model]["commands_per_s_per_robot"]

    assert max(rates_per_s) <= CONFIG.frame_rate_hz + 1 / 30.0


@pytest.mark.parametrize(("key", "model", "robot_count"), FLEET_RATE_FACTS)
def test_the_robots_together_never_get_more_than_the_capacity_plus_one_batch_over_the_run(
    predictions: dict[str, Any], key: str, model: str, robot_count: int
) -> None:
    batch_cap = predictions["inputs"]["plans"][model]["batch_cap"]

    total_per_s = robot_count * predictions["facts"][key]

    assert total_per_s <= capacity_hz(model, batch_cap) + batch_cap / 30.0


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_curve_never_gives_the_robots_together_more_than_the_capacity_plus_one_batch(
    predictions: dict[str, Any], model: str
) -> None:
    batch_cap = predictions["inputs"]["plans"][model]["batch_cap"]
    rates_per_s = predictions["curves"][model]["commands_per_s_per_robot"]

    totals_per_s = [
        robot_count * rate_per_s
        for robot_count, rate_per_s in zip(SHORT_CURVE, rates_per_s, strict=True)
    ]

    assert max(totals_per_s) <= capacity_hz(model, batch_cap) + batch_cap / 30.0


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_curves_longest_gap_is_within_its_bound_once_the_first_frame_and_command_travel(
    predictions: dict[str, Any], model: str
) -> None:
    curve = predictions["curves"][model]
    # A robot's longest gap includes the wait from its start to its first command, which also
    # takes the uplink of its first frame and the delivery of that command.
    travel_ms = (OVERHEADS.transport_up_ns + OVERHEADS.deliver_ns) / NS_PER_MS

    over_ms = [
        gap_ms - bound_ms - travel_ms
        for gap_ms, bound_ms in zip(
            curve["command_gap_ms_max"], curve["command_gap_bound_ms"], strict=True
        )
    ]

    assert max(over_ms) <= 0


@pytest.mark.parametrize(
    "field",
    [
        "commands_per_s_per_robot",
        "freshness_ms_p50",
        "freshness_ms_p95",
        "command_gap_ms_max",
        "batch_size_mean",
    ],
)
@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_curve_value_is_its_statistic_over_the_sweep_run_at_that_robot_count(
    predictions: dict[str, Any], runs: dict[str, SimulationResult], model: str, field: str
) -> None:
    expected = [sweep_statistics(runs[f"sweep.n{count}"], model)[field] for count in SHORT_CURVE]

    assert predictions["curves"][model][field] == pytest.approx(expected, rel=1e-12)


def test_the_demand_curve_is_the_predicted_demand_of_the_sweep_run_at_each_robot_count(
    predictions: dict[str, Any], runs: dict[str, SimulationResult]
) -> None:
    assert predictions["curves"]["cpu_demand_cores"] == [
        runs[f"sweep.n{count}"].cpu_demand_cores for count in SHORT_CURVE
    ]


def test_the_measured_io_cost_moves_the_demand_curve_and_no_fact_or_model_curve(
    predictions: dict[str, Any],
) -> None:
    # Depth is paced at 10 and at 50 robots, so a share computed from the measured cost would
    # move its rates away from those of a measured cost equal to the configured one.
    equal_costs: dict[str, Any] = predict(
        CONFIG,
        TABLES,
        OVERHEADS,
        CONFIG.io_cpu_ms_per_frame,
        QUOTA_CORES,
        SEED,
        SHORT_CURVE,
        "linear",
    )

    assert predictions["facts"] == equal_costs["facts"]
    assert {model: predictions["curves"][model] for model in MODEL_KEYS} == {
        model: equal_costs["curves"][model] for model in MODEL_KEYS
    }
    assert all(
        higher > lower
        for higher, lower in zip(
            predictions["curves"]["cpu_demand_cores"],
            equal_costs["curves"]["cpu_demand_cores"],
            strict=True,
        )
    )


@pytest.mark.parametrize(
    ("model", "utilization"),
    [
        # 30 frames per second per robot at 7.5 ms of CPU time each, over the classifier's one
        # thread.
        ("cls", [0.225, 0.45, 2.25]),
        # At 60 ms of CPU time each, over depth's one thread at 1 and 2 robots. At 10 robots,
        # over its pacing share: what the quota leaves after the 0.1 core of headroom, the
        # classifier's full thread and the frame I/O of 10 robots at the configured 0.15 ms per
        # frame.
        ("depth", [1.8, 3.6, 18 / (1.9 - 1 - 0.045)]),
    ],
)
def test_utilization_is_the_offered_cpu_time_over_the_cores_the_worker_may_use(
    predictions: dict[str, Any], model: str, utilization: list[float]
) -> None:
    assert predictions["curves"][model]["utilization"] == pytest.approx(utilization)


def test_a_worker_that_leaves_its_thread_count_to_the_runtime_may_use_the_whole_quota() -> None:
    models = {**MODELS, "cls": replace(MODELS["cls"], threads=0)}

    output: dict[str, Any] = predict(
        replace(CONFIG, models=models),
        TABLES,
        OVERHEADS,
        IO_CPU_MS_PER_FRAME,
        QUOTA_CORES,
        SEED,
        (1,),
        "",
    )

    # 30 frames per second at 7.5 ms of CPU time each, over the 2 cores of the quota.
    assert output["curves"]["cls"]["utilization"] == pytest.approx([0.1125])


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_with_the_skip_rule_the_gap_bound_adds_its_interval_to_a_period_and_the_cycles(
    predictions: dict[str, Any], model: str
) -> None:
    assert predictions["curves"][model]["command_gap_bound_ms"] == pytest.approx(
        [FRAME_PERIOD_MS + cycles_ms + 1_000 for cycles_ms in GAP_CYCLES_MS[model]]
    )


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_without_the_skip_rule_the_gap_bound_is_a_period_and_the_cycles(
    predictions_without_skip_rule: dict[str, Any], model: str
) -> None:
    curve = predictions_without_skip_rule["curves"][model]

    assert curve["command_gap_bound_ms"] == pytest.approx(
        [FRAME_PERIOD_MS + cycles_ms for cycles_ms in GAP_CYCLES_MS[model]]
    )


def test_a_model_whose_every_command_is_stale_has_a_null_peak_age_and_every_other_fact() -> None:
    stale_tables = tables(STALE_DEPTH_FRAME_MS)
    null_key = "predict.steady.depth.command_age_peak_ms.p95"

    output: dict[str, Any] = predict(
        CONFIG, stale_tables, OVERHEADS, IO_CPU_MS_PER_FRAME, QUOTA_CORES, SEED, (1,), ""
    )

    facts = output["facts"]
    assert {key: value is None for key, value in facts.items()} == {
        key: key == null_key for key in FACT_KEYS
    }
    assert facts["predict.steady.depth.freshness_ms.p50"] > MODELS["depth"].stale_ms


def test_runs_in_which_no_frame_arrives_give_gaps_as_long_as_the_run_and_no_statistic() -> None:
    # An uplink longer than every run.
    silent = replace(OVERHEADS, transport_up_ns=60 * NS_PER_S)
    counts = len(SHORT_CURVE)
    empty_curve = {
        "commands_per_s_per_robot": [0.0] * counts,
        "freshness_ms_p50": [None] * counts,
        "freshness_ms_p95": [None] * counts,
        # A robot that receives nothing waits for its command for the whole 30 s run.
        "command_gap_ms_max": [30_000.0] * counts,
        "batch_size_mean": [None] * counts,
        "utilization": [None] * counts,
    }

    output: dict[str, Any] = predict(
        CONFIG, TABLES, silent, IO_CPU_MS_PER_FRAME, QUOTA_CORES, SEED, SHORT_CURVE, ""
    )

    curves = output["curves"]
    assert {
        model: {field: curves[model][field] for field in empty_curve} for model in MODEL_KEYS
    } == dict.fromkeys(MODEL_KEYS, empty_curve)
    assert output["facts"] == {
        **dict.fromkeys(FACT_KEYS),
        **dict.fromkeys(PER_ROBOT_RATE_FACTS, 0.0),
    }


def test_a_valid_input_is_read_in_nanoseconds_with_every_size_of_the_configured_threads() -> None:
    assert parse_input(json.dumps(VALID_DOCUMENT).encode(), MODELS) == VALID_INPUT


@pytest.mark.parametrize("zero", [0, 0.0], ids=["int", "float"])
def test_a_time_of_zero_is_read_as_zero(zero: float) -> None:
    measured = parse_input(document_with(("overheads_ms", "deliver"), zero), MODELS)

    assert measured.overheads == replace(VALID_INPUT.overheads, deliver_ns=0)


@pytest.mark.parametrize(
    "path",
    [
        ("models",),
        ("models", "depth"),
        ("models", "cls", "threads"),
        ("models", "cls", "threads", "1"),
        ("models", "depth", "threads", "1", "8"),
        ("models", "cls", "threads", "1", "3", "wall_ms"),
        ("models", "depth", "threads", "1", "1", "cpu_ms"),
        ("overheads_ms",),
        ("overheads_ms", "transport_up"),
        ("overheads_ms", "dispatch_request"),
        ("overheads_ms", "dispatch_result"),
        ("overheads_ms", "deliver"),
        ("io_cpu_ms_per_frame",),
        ("label",),
    ],
    ids=".".join,
)
def test_a_missing_key_is_refused_with_its_path(path: tuple[str, ...]) -> None:
    with pytest.raises(CapacityInputError, match=re.escape(f"{'.'.join(path)}: missing")):
        parse_input(document_without(path), MODELS)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        pytest.param(("models",), [], "models: must be an object", id="models-a-list"),
        pytest.param(
            ("models", "cls", "threads", "1"),
            [5.0],
            "models.cls.threads.1: must be an object",
            id="sizes-a-list",
        ),
        pytest.param(
            ("models", "cls", "threads", "1", "2"),
            9.0,
            "models.cls.threads.1.2: must be an object",
            id="size-a-number",
        ),
        pytest.param(
            ("overheads_ms",), "fast", "overheads_ms: must be an object", id="overheads-a-string"
        ),
        pytest.param(WALL_MS, "9", "wall_ms: must be a number", id="string"),
        pytest.param(WALL_MS, True, "wall_ms: must be a number", id="bool"),
        pytest.param(WALL_MS, None, "wall_ms: must be a number", id="null"),
        pytest.param(WALL_MS, -0.5, "wall_ms: must be finite, 0 or more", id="negative"),
        pytest.param(WALL_MS, float("nan"), "wall_ms: must be finite, 0 or more", id="nan"),
        pytest.param(WALL_MS, float("inf"), "wall_ms: must be finite, 0 or more", id="infinity"),
        pytest.param(WALL_MS, 1e300, "wall_ms: must be finite, 0 or more", id="beyond-the-clock"),
        pytest.param(WALL_MS, 10**400, "wall_ms: must be finite, 0 or more", id="huge-int"),
        pytest.param(
            ("models", "depth", "threads", "1", "4", "cpu_ms"),
            -1,
            "models.depth.threads.1.4.cpu_ms: must be finite, 0 or more",
            id="negative-cpu",
        ),
        pytest.param(
            ("overheads_ms", "deliver"),
            -2,
            "overheads_ms.deliver: must be finite, 0 or more",
            id="negative-overhead",
        ),
        pytest.param(
            ("io_cpu_ms_per_frame",),
            "0.15",
            "io_cpu_ms_per_frame: must be a number",
            id="io-a-string",
        ),
        pytest.param(("label",), 7, "label: must be a string", id="label-a-number"),
    ],
)
def test_a_bad_value_is_refused_with_its_path(
    path: tuple[str, ...], value: object, message: str
) -> None:
    with pytest.raises(CapacityInputError, match=re.escape(message)):
        parse_input(document_with(path, value), MODELS)


@pytest.mark.parametrize(
    "key",
    ["0", "-1", "01", "+1", " 1", "1.5", "x", "", "1" * 5_000],
    ids=["zero", "negative", "leading-zero", "plus", "space", "fraction", "word", "empty", "long"],
)
def test_a_size_key_that_is_not_a_batch_size_is_refused_with_its_path(key: str) -> None:
    data = document_with(("models", "cls", "threads", "1", key), {"wall_ms": 1.0, "cpu_ms": 1.0})

    with pytest.raises(
        CapacityInputError, match=re.escape(f"models.cls.threads.1.{key}: must name a batch size")
    ):
        parse_input(data, MODELS)


@pytest.mark.parametrize(
    "data",
    [b"{", b"\x80{}", b"[" * 100_000, b'{"label": 1' + b"0" * 5_000 + b"}"],
    ids=["syntax", "not-utf-8", "nested-too-deep", "integer-too-long"],
)
def test_a_file_that_is_not_json_is_refused(data: bytes) -> None:
    with pytest.raises(CapacityInputError, match="not a JSON document"):
        parse_input(data, MODELS)


def test_an_input_that_is_not_an_object_is_refused() -> None:
    with pytest.raises(CapacityInputError, match="the input: must be an object"):
        parse_input(b"[]", MODELS)


@pytest.fixture(scope="module")
def written(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """Return what main() writes for VALID_DOCUMENT under seed 3, as json reads it back."""
    directory = tmp_path_factory.mktemp("main")
    t_b = directory / "t_b.json"
    t_b.write_text(json.dumps(VALID_DOCUMENT))
    out = directory / "predictions.json"
    main(["--t-b", str(t_b), "--out", str(out), "--seed", "3"])
    predictions: dict[str, Any] = json.loads(out.read_text())
    return predictions


def test_main_writes_predictions_that_json_reads_back(written: dict[str, Any]) -> None:
    assert (written["label"], written["inputs"]["seed"]) == ("two single-thread workers", 3)
    assert written["inputs"]["robot_counts"] == list(range(1, 51))
    assert set(written["facts"]) == FACT_KEYS


def test_main_records_the_files_io_cost_beside_the_configured_one(written: dict[str, Any]) -> None:
    inputs = written["inputs"]

    assert (inputs["io_cpu_ms_per_frame"], inputs["configured_io_cpu_ms_per_frame"]) == (0.2, 0.15)


def test_main_computes_the_pacing_share_from_the_configured_io_cost(
    written: dict[str, Any],
) -> None:
    # At 10 robots the classifier fills its thread, and depth serves every frame alone, at 58 ms
    # of CPU time: 17.4 cores offered. Depth's share is what the quota leaves after the 0.1 core
    # of headroom, the classifier's thread and the frame I/O of 10 robots at the configured
    # 0.15 ms per frame. The file's 0.2 ms would leave 0.84 cores.
    assert written["curves"]["depth"]["utilization"][9] == pytest.approx(17.4 / 0.855)


def test_main_seeds_the_robots_phases_with_0_by_default(tmp_path: Path) -> None:
    t_b = tmp_path / "t_b.json"
    # An uplink longer than every run: no frame arrives, so the runs at 50 robot counts are quick.
    t_b.write_bytes(document_with(("overheads_ms", "transport_up"), 60_000.0))
    out = tmp_path / "predictions.json"

    main(["--t-b", str(t_b), "--out", str(out)])

    assert json.loads(out.read_text())["inputs"]["seed"] == 0


def test_main_reports_a_bad_input_as_a_usage_error_that_names_the_key(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    t_b = tmp_path / "t_b.json"
    t_b.write_bytes(document_without(("overheads_ms", "deliver")))
    out = tmp_path / "predictions.json"

    with pytest.raises(SystemExit) as raised:
        main(["--t-b", str(t_b), "--out", str(out)])

    assert raised.value.code == 2
    assert "overheads_ms.deliver: missing" in capsys.readouterr().err
    assert not out.exists()


@pytest.mark.parametrize(
    ("t_b_name", "out_name", "flag"),
    [
        ("missing.json", "predictions.json", "argument --t-b"),
        ("t_b.json", "missing/predictions.json", "argument --out"),
        ("t_b.json", ".", "argument --out"),
    ],
    ids=["t-b-missing", "out-directory-missing", "out-a-directory"],
)
def test_main_refuses_a_path_it_cannot_use_before_it_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], t_b_name: str, out_name: str, flag: str
) -> None:
    (tmp_path / "t_b.json").write_text(json.dumps(VALID_DOCUMENT))

    with pytest.raises(SystemExit) as raised:
        main(["--t-b", str(tmp_path / t_b_name), "--out", str(tmp_path / out_name)])

    assert raised.value.code == 2
    assert flag in capsys.readouterr().err
