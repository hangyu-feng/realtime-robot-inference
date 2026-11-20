"""The tables of DESIGN.md 11.5, written from summary/facts.json and the summaries it names.

A measured cell shows a fact's value with the lowest and highest value of its counted runs:
a rendered placeholder shows only the value, so these tables are where the spread between
runs is visible. Run ids come from facts.json alone and each summary is opened by its id,
so a run directory that is not counted never reaches a table. The same inputs give the
same bytes.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.candidates import CANDIDATES, Metric
from bench.facts import BREAKDOWN_STAGES, REGISTRY
from bench.render import RenderError, format_value, read_facts
from bench.runfiles import FACTS_FILE, SUMMARY_FILE, Fact, Facts
from bench.runplan import SWEEP_ROBOT_COUNTS
from cloud.config import Topology
from common.commands import MODEL_KEYS

TABLES_DIRECTORY = "tables"
# The directory of the results root that holds the run directories.
RAW_DIRECTORY = "raw"
NOT_RUN = "not run"

_PREDICT_PREFIX = "predict."
# The serving scenario keys, in the order of the evidence plan (DESIGN.md 11.2).
_SERVING_SCENARIOS = (
    "steady",
    "late_joiner",
    *(f"sweep_n{count}" for count in SWEEP_ROBOT_COUNTS),
    "disconnect",
    "worker_kill",
    "massive",
    "soak",
)
_BREAKDOWN_STATISTICS = ("mean", "p50", "p95")
_DISCONNECT_WINDOWS = ("before", "after")
_CELL_NOTE = (
    "A measured cell shows the fact's value, with the lowest and highest value of its counted "
    "runs in parentheses."
)


class TablesError(Exception):
    """facts.json lacks a fact a table shows, or a counted run's summary is missing or malformed."""


@dataclass(frozen=True)
class _Measured:
    """A cell that shows one fact."""

    key: str


@dataclass(frozen=True)
class _RunCount:
    """A cell that shows how many counted runs stand behind the measured cells of its row."""


@dataclass(frozen=True)
class _Gap:
    """A cell that shows the fact key minus its prediction, the fact predict.<key>."""

    key: str


@dataclass(frozen=True)
class _Spread:
    """A cell that shows a fact's highest minus its lowest run value."""

    key: str


@dataclass(frozen=True)
class _ToExplain:
    """A cell that marks a gap between prediction and measurement larger than the spread."""

    key: str


@dataclass(frozen=True)
class _Phrase:
    """A cell of text with facts in it, each shown as a measured cell shows its fact."""

    parts: tuple[str | _Measured, ...]


type _Cell = str | _Measured | _RunCount | _Gap | _Spread | _ToExplain | _Phrase
type _Row = tuple[_Cell, ...]


@dataclass(frozen=True)
class _Layout:
    """A table as it is planned: its labels, and the fact each cell shows."""

    heading: str
    header: tuple[str, ...]
    rows: tuple[_Row, ...]


@dataclass(frozen=True)
class _Section:
    """A table as it is written, with the lines under it."""

    heading: str
    header: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    footer: tuple[str, ...]


def _cell_keys(cell: _Cell) -> tuple[str, ...]:
    match cell:
        case _Measured(key) | _Spread(key):
            return (key,)
        case _Gap(key) | _ToExplain(key):
            return (_PREDICT_PREFIX + key, key)
        case _Phrase(parts):
            return tuple(part.key for part in parts if isinstance(part, _Measured))
        case _:
            return ()


def _row_keys(rows: Iterable[_Row]) -> Iterator[str]:
    return (key for row in rows for cell in row for key in _cell_keys(cell))


def _is_prediction(key: str) -> bool:
    return key.startswith(_PREDICT_PREFIX)


def _run_ids(facts: Facts, keys: Iterable[str]) -> set[str]:
    return {run_id for key in keys for run_id in facts.facts[key].run_ids}


def _measured_run_ids(facts: Facts, keys: Iterable[str]) -> set[str]:
    """Return the counted runs behind the facts among keys; a prediction's runs are apart."""
    return _run_ids(facts, (key for key in keys if not _is_prediction(key)))


def _fact_text(fact: Fact) -> str:
    value = format_value(fact.value, fact.unit)
    if fact.min is None and fact.max is None:
        return value
    if fact.min is None or fact.max is None:
        raise TablesError(f"a fact has only one of its lowest and highest run values: {fact}")
    return f"{value} ({format_value(fact.min, fact.unit)} to {format_value(fact.max, fact.unit)})"


def _number(facts: Facts, key: str) -> float | None:
    value = facts.facts[key].value
    if isinstance(value, bool | str):
        raise TablesError(f"{FACTS_FILE}: {key} is not a number, so it has no gap")
    return value


def _gap(facts: Facts, key: str) -> float | None:
    predicted = _number(facts, _PREDICT_PREFIX + key)
    measured = _number(facts, key)
    return None if predicted is None or measured is None else measured - predicted


def _spread(fact: Fact) -> float | None:
    return None if fact.min is None or fact.max is None else fact.max - fact.min


def _cell_text(cell: _Cell, row: _Row, facts: Facts) -> str:
    match cell:
        case str():
            return cell
        case _Measured(key):
            return _fact_text(facts.facts[key])
        case _RunCount():
            return str(len(_measured_run_ids(facts, _row_keys((row,)))))
        case _Gap(key):
            return format_value(_gap(facts, key), facts.facts[key].unit)
        case _Spread(key):
            return format_value(_spread(facts.facts[key]), facts.facts[key].unit)
        case _ToExplain(key):
            gap = _gap(facts, key)
            spread = _spread(facts.facts[key])
            return "yes" if gap is not None and spread is not None and abs(gap) > spread else ""
        case _Phrase(parts):
            return "".join(
                part if isinstance(part, str) else _fact_text(facts.facts[part.key])
                for part in parts
            )


def _rows_text(rows: Iterable[_Row], facts: Facts) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(_cell_text(cell, row, facts) for cell in row) for row in rows)


def _not_run_row(width: int) -> tuple[str, ...]:
    return (NOT_RUN, *("",) * (width - 1))


def _fill(layout: _Layout, facts: Facts | None) -> _Section:
    """Write a planned table, as one row "not run" when no counted run stands behind it."""
    keys = tuple(_row_keys(layout.rows))
    run_ids = set() if facts is None else _measured_run_ids(facts, keys)
    footer = [f"Counted runs: {len(run_ids)}."]
    predictions = [key for key in keys if _is_prediction(key)]
    prediction_runs = set() if facts is None else _run_ids(facts, predictions)
    if predictions:
        footer.append(f"Counted runs behind the predictions: {len(prediction_runs)}.")
    # A table of predictions alone rests on the runs behind them.
    standing = prediction_runs if len(predictions) == len(keys) else run_ids
    if facts is None or not standing:
        return _Section(
            layout.heading, layout.header, (_not_run_row(len(layout.header)),), tuple(footer)
        )
    return _Section(layout.heading, layout.header, _rows_text(layout.rows, facts), tuple(footer))


_MODEL_HEADER = ("Quantity", *(f"`{model}`" for model in MODEL_KEYS))


def _per_model(heading: str, prefix: str, metrics: Sequence[tuple[str, str]]) -> _Layout:
    """Plan quantities against the two models; a row's key is prefix.<model>.<metric>."""
    rows: tuple[_Row, ...] = tuple(
        (label, *(_Measured(f"{prefix}.{model}.{metric}") for model in MODEL_KEYS))
        for label, metric in metrics
    )
    return _Layout(heading, _MODEL_HEADER, rows)


def _quantities(heading: str, keys: Sequence[tuple[str, str]]) -> _Layout:
    """Plan single quantities, each a label and the whole key of its fact."""
    return _Layout(
        heading, ("Quantity", "Value"), tuple((label, _Measured(key)) for label, key in keys)
    )


def _steady() -> tuple[_Layout, ...]:
    # The quantities of DESIGN.md 9.1, each with whether the capacity model predicts it.
    metrics = (
        ("Freshness p50 (ms)", "freshness_ms.p50", True),
        ("Freshness p95 (ms)", "freshness_ms.p95", True),
        ("Freshness p99 (ms)", "freshness_ms.p99", True),
        ("Freshness max (ms)", "freshness_ms.max", True),
        ("Commands per second", "commands_per_s", True),
        ("Freshness trend (ms per s)", "freshness_trend_ms_per_s", False),
        ("Peak command age p95 (ms)", "command_age_peak_ms.p95", True),
        ("Commands received", "commands_received", False),
        ("Stale commands", "stale_count", False),
        ("Stale ratio", "stale_ratio", False),
        ("Out-of-order commands", "out_of_order_count", False),
    )
    header = (
        "Quantity",
        *(f"`{model}` {kind}" for model in MODEL_KEYS for kind in ("predicted", "measured")),
    )
    rows = tuple(
        (
            label,
            *(
                cell
                for model in MODEL_KEYS
                for cell in (
                    _Measured(f"{_PREDICT_PREFIX}steady.{model}.{metric}") if predicted else "",
                    _Measured(f"steady.{model}.{metric}"),
                )
            ),
        )
        for label, metric, predicted in metrics
    )
    return (_Layout("One robot", header, rows),)


def _late_joiner() -> tuple[_Layout, ...]:
    header = (
        "Robot",
        "Window",
        *(
            f"`{model}` {quantity}"
            for model in MODEL_KEYS
            for quantity in ("commands/s", "p95 freshness (ms)")
        ),
    )
    measured: tuple[_Row, ...] = tuple(
        (
            label,
            window,
            *(
                _Measured(f"late_joiner.{window}.{robot}.{model}.{metric}")
                for model in MODEL_KEYS
                for metric in ("commands_per_s", "freshness_ms.p95")
            ),
        )
        for label, robot, window in (("A", "a", "before"), ("A", "a", "after"), ("B", "b", "after"))
    )
    predicted: _Row = (
        "A, predicted",
        "after",
        *(
            cell
            for model in MODEL_KEYS
            for cell in (
                _Measured(f"{_PREDICT_PREFIX}late_joiner.after.a.{model}.commands_per_s"),
                "",
            )
        ),
    )
    join = _Layout(
        "After the join",
        _MODEL_HEADER,
        (
            (
                "B's first command after its join (ms)",
                *(_Measured(f"late_joiner.b.{model}.first_command_ms") for model in MODEL_KEYS),
            ),
            (
                "A and B settle at equal rates, counted in 1 s bins (s)",
                *(_Measured(f"late_joiner.{model}.settle_s") for model in MODEL_KEYS),
            ),
        ),
    )
    return (_Layout("Per robot and window", header, (*measured, predicted)), join)


def _sweep_model(model: str) -> _Layout:
    metrics = (
        "commands_per_s_per_robot.min",
        "commands_per_s_per_robot.mean",
        "commands_per_s_per_robot.max",
        "freshness_ms.p50",
        "freshness_ms.p95",
        "batch_size.mean",
        "batch_size.share_1",
        "utilization",
    )
    header = (
        "Robots",
        "Runs",
        "Commands/s per robot, predicted",
        "Commands/s per robot, min",
        "Commands/s per robot, mean",
        "Commands/s per robot, max",
        "Freshness p50 (ms)",
        "Freshness p95 (ms)",
        "Mean batch size",
        "Share of batches of 1",
        "Utilization",
    )
    rows: tuple[_Row, ...] = tuple(
        (
            str(count),
            _RunCount(),
            _Measured(f"{_PREDICT_PREFIX}sweep.n{count}.{model}.commands_per_s_per_robot.mean"),
            *(_Measured(f"sweep.n{count}.{model}.{metric}") for metric in metrics),
        )
        for count in SWEEP_ROBOT_COUNTS
    )
    return _Layout(f"`{model}`", header, rows)


def _sweep() -> tuple[_Layout, ...]:
    cpu = _Layout(
        "CPU",
        ("Robots", "Runs", "CPU utilization", "Throttled (ms per s)"),
        tuple(
            (
                str(count),
                _RunCount(),
                _Measured(f"sweep.n{count}.cpu.utilization"),
                _Measured(f"sweep.n{count}.cpu.throttled_ms_per_s"),
            )
            for count in SWEEP_ROBOT_COUNTS
        ),
    )
    inflection = _Layout(
        "Inflection point, from the sweep and the 50-robot run",
        (
            "Model",
            "Capacity at the first saturated point (frames/s)",
            "Inflection N* (robots)",
            "Utilization crosses 1",
        ),
        tuple(
            (
                f"`{model}`",
                *(
                    _Measured(f"sweep.{model}.{metric}")
                    for metric in ("capacity_fps", "inflection_n", "inflection_between")
                ),
            )
            for model in MODEL_KEYS
        ),
    )
    return (*(_sweep_model(model) for model in MODEL_KEYS), cpu, inflection)


def _breakdown_mean_row(label: str, metric: str) -> _Row:
    """Plan a row that has a mean per model and no tail."""
    return (
        label,
        *(
            cell
            for model in MODEL_KEYS
            for cell in (_Measured(f"sweep.n5.{model}.{metric}"), "", "")
        ),
    )


def _breakdown() -> tuple[_Layout, ...]:
    header = (
        "Stage",
        *(f"`{model}` {statistic}" for model in MODEL_KEYS for statistic in _BREAKDOWN_STATISTICS),
    )
    stages: tuple[_Row, ...] = tuple(
        (
            f"`{stage}`",
            *(
                _Measured(f"sweep.n5.{model}.stage.{stage}_ms.{statistic}")
                for model in MODEL_KEYS
                for statistic in _BREAKDOWN_STATISTICS
            ),
        )
        for stage in BREAKDOWN_STAGES
    )
    end_to_end: _Row = (
        "End to end",
        *(
            _Measured(f"sweep.n5.{model}.{metric}")
            for model in MODEL_KEYS
            for metric in ("e2e_ms.mean", "freshness_ms.p50", "freshness_ms.p95")
        ),
    )
    rows: tuple[_Row, ...] = (
        *stages,
        _breakdown_mean_row("Sum of stage means", "stage_sum_ms.mean"),
        end_to_end,
        _breakdown_mean_row("Residual", "residual_ms.mean"),
    )
    return (_Layout("Stages, all values in ms", header, rows),)


def _disconnect() -> tuple[_Layout, ...]:
    metrics = (
        ("Survivors' `depth` commands/s per robot", "", "depth.commands_per_s_per_robot"),
        ("Survivors' `cls` commands/s per robot", "", "cls.commands_per_s_per_robot"),
        ("Survivors' `depth` p95 freshness (ms)", "", "depth.freshness_ms.p95"),
        ("Survivors' `cls` p95 freshness (ms)", "", "cls.freshness_ms.p95"),
        (
            "Predicted survivors' `depth` commands/s per robot",
            _PREDICT_PREFIX,
            "depth.commands_per_s_per_robot",
        ),
    )
    survivors = _Layout(
        "Survivors",
        ("Quantity", "Before the kill", "After the kill"),
        tuple(
            (
                label,
                *(
                    _Measured(f"{prefix}disconnect.{window}.survivors.{metric}")
                    for window in _DISCONNECT_WINDOWS
                ),
            )
            for label, prefix, metric in metrics
        ),
    )
    times = _quantities(
        "After the kill",
        (
            (
                "Time from the kill until the server closes the connection (ms)",
                "disconnect.detect_ms",
            ),
            *(
                (
                    f"Time until the survivors' `{model}` rate settles, counted in 1 s bins (s)",
                    f"disconnect.{model}.recovery_s",
                )
                for model in MODEL_KEYS
            ),
        ),
    )
    return (survivors, times)


_MASSIVE_PER_MODEL = (
    ("Commands/s per robot, min", "commands_per_s_per_robot.min"),
    ("Commands/s per robot, mean", "commands_per_s_per_robot.mean"),
    ("Commands/s per robot, max", "commands_per_s_per_robot.max"),
    ("Commands received", "commands_received"),
    ("Lowest over mean rate", "fairness.min_over_mean"),
    ("Jain index", "fairness.jain"),
    ("Longest command gap (ms)", "command_gap_ms.max"),
    ("Command gap bound (ms)", "command_gap_bound_ms"),
    ("Freshness p50 (ms)", "freshness_ms.p50"),
    ("Freshness p95 (ms)", "freshness_ms.p95"),
    ("Freshness p99 (ms)", "freshness_ms.p99"),
    ("Freshness max (ms)", "freshness_ms.max"),
    ("Stale commands", "stale_count"),
    ("Stale ratio", "stale_ratio"),
    ("Out-of-order commands", "out_of_order_count"),
    ("Watchdog coverage, min", "watchdog_coverage.min"),
    ("Watchdog coverage, mean", "watchdog_coverage.mean"),
)
_MASSIVE_SYSTEM = (
    ("Connections accepted", "massive.connections.accepted"),
    ("Connections refused", "massive.connections.rejected"),
    ("CPU utilization", "massive.cpu.utilization"),
    ("Throttled (ms per s)", "massive.cpu.throttled_ms_per_s"),
    ("Gateway CPU (cores)", "massive.gateway.cpu_cores"),
    ("Gateway CPU per frame (ms)", "massive.gateway.cpu_ms_per_frame"),
    ("Gateway event-loop lag p99 (ms)", "massive.gateway.loop_lag_ms.p99"),
    ("Bytes received per second", "massive.bytes_in_per_s"),
    ("Stage `transport_up` p50 (ms)", "massive.stage.transport_up_ms.p50"),
    ("Stage `transport_up` p99 (ms)", "massive.stage.transport_up_ms.p99"),
    ("Stage `dispatch_rt` p50 (ms)", "massive.stage.dispatch_rt_ms.p50"),
    ("Simulator CPU (cores)", "massive.fleet.cpu_cores"),
    ("Simulator fire lag p99 (ms)", "massive.fleet.fire_lag_ms.p99"),
    (
        "Memory, anonymous + socket, after the connections settle (MB)",
        "massive.memory.anon_sock_mb_start",
    ),
    ("Memory, anonymous + socket, at the end (MB)", "massive.memory.anon_sock_mb_end"),
    ("Socket memory at the end (MB)", "massive.memory.sock_mb"),
)
_SOAK_SYSTEM = (
    ("Memory slope (MB per minute)", "soak.memory.slope_mb_per_min"),
    ("Change in open file descriptors", "soak.fds.delta"),
    ("Change in event-loop tasks", "soak.tasks.delta"),
)
_SOAK_PER_MODEL = (
    ("Commands/s per robot, min", "commands_per_s_per_robot.min"),
    ("Commands/s per robot, mean", "commands_per_s_per_robot.mean"),
    ("Commands/s per robot, max", "commands_per_s_per_robot.max"),
    ("Lowest over mean rate", "fairness.min_over_mean"),
    ("Jain index", "fairness.jain"),
)


def _massive() -> tuple[_Layout, ...]:
    # Each table has runs of its own: the predictions rest on the model benchmark, the useful
    # robots on every scale point, the soak on its own scenario.
    return (
        _per_model("Per model", "massive", _MASSIVE_PER_MODEL),
        _per_model(
            "Predicted",
            f"{_PREDICT_PREFIX}massive",
            (("Commands/s per robot, mean", "commands_per_s_per_robot.mean"),),
        ),
        _quantities("Connections, CPU and memory", _MASSIVE_SYSTEM),
        _per_model(
            "Within the watchdog limits, from the sweep and the 50-robot run",
            "massive",
            (("Robots served within the watchdog limit, of the counts measured", "useful_robots"),),
        ),
        _quantities("Soak, 5 minutes", _SOAK_SYSTEM),
        _per_model("Soak, per model", "soak", _SOAK_PER_MODEL),
    )


def _memory_budget() -> tuple[_Layout, ...]:
    # The components and predicted bounds of DESIGN.md 8.3.
    components = (
        (
            "Gateway process",
            "base + N x (one frame in the slot + connection buffers)",
            "gateway.rss_mb",
        ),
        ("Worker `cls`", "model + arena for the largest batch", "worker_cls.rss_mb"),
        ("Worker `depth`", "model + arena for the largest batch", "worker_depth.rss_mb"),
        ("Robot fleet", "base + frame pool + N x robot state + samples", "fleet.rss_mb"),
        ("Kernel socket memory", "N x (receive buffer + send buffer), both ends", "memory.sock_mb"),
        ("Container total (anonymous + socket)", "sum of the above", "memory.anon_sock_mb_end"),
    )
    return (
        _Layout(
            "At 50 robots",
            ("Component", "Predicted bound", "Measured (MB)"),
            tuple((label, bound, _Measured(f"massive.{key}")) for label, bound, key in components),
        ),
    )


def _env() -> tuple[_Layout, ...]:
    return (
        _quantities(
            "Test machine",
            (
                ("CPU model", "env.cpu_model"),
                ("Kernel", "env.kernel"),
                ("Container engine", "env.engine"),
                ("Repeats: the fewest counted runs behind a scenario fact", "env.repeats"),
            ),
        ),
    )


def _predicted_vs_measured(facts: Facts | None) -> _Layout:
    measured_keys = sorted(
        key.removeprefix(_PREDICT_PREFIX)
        for key in ({} if facts is None else facts.facts)
        if key.startswith(_PREDICT_PREFIX)
    )
    return _Layout(
        "Every prediction",
        ("Key", "Predicted", "Measured", "Gap", "Spread", "Runs", "To explain"),
        tuple(
            (
                f"`{key}`",
                _Measured(_PREDICT_PREFIX + key),
                _Measured(key),
                _Gap(key),
                _Spread(key),
                _RunCount(),
                _ToExplain(key),
            )
            for key in measured_keys
        ),
    )


_LEDGER_HEADER = (
    "Candidate",
    "Scenario",
    "Arm A",
    "Arm B",
    "Hypothesis",
    "Main metric",
    "Guard metrics",
    "Main change (B minus A)",
    "Baseline spread",
    "Runs",
    "Outcome",
)


def _overrides(overrides: Mapping[str, Any]) -> str:
    if not overrides:
        return "defaults"
    return f"`{json.dumps(overrides, sort_keys=True, separators=(', ', ': '))}`"


def _metric(metric: Metric) -> str:
    return f"`{metric.path}`, {metric.better}"


def _ledger_rows() -> tuple[_Row, ...]:
    return tuple(
        (
            f"`{candidate.name}`",
            f"`{candidate.scenario_key}`, `{candidate.topology}`",
            _overrides(candidate.a_overrides),
            _overrides(candidate.b_overrides),
            candidate.hypothesis,
            _metric(candidate.main),
            "; ".join(_metric(guard) for guard in candidate.guards),
            _Measured(f"ledger.{candidate.name}.main_delta"),
            _Measured(f"ledger.{candidate.name}.baseline_spread"),
            _RunCount(),
            _Measured(f"ledger.{candidate.name}.outcome"),
        )
        for candidate in CANDIDATES
    )


def _ledger(facts: Facts | None) -> _Section:
    """Write every candidate, so a candidate that did not run is listed as not run."""
    rows = _ledger_rows()
    if facts is None:
        # Without facts.json no comparison has been aggregated: every outcome is not run.
        text = tuple(
            (*(cell if isinstance(cell, str) else "" for cell in row[:-1]), NOT_RUN) for row in rows
        )
        run_count = 0
    else:
        text = _rows_text(rows, facts)
        run_count = len(_measured_run_ids(facts, _row_keys(rows)))
    return _Section("Candidates", _LEDGER_HEADER, text, (f"Counted runs: {run_count}.",))


@dataclass(frozen=True)
class _FaultCase:
    """A row of the fault table: a case, the scenario key that plays it, checks and recovery."""

    case: str
    scenario: str
    checks: _Cell
    recovery: _Cell

    @property
    def row(self) -> _Row:
        return (self.case, f"`{self.scenario}`", self.checks, self.recovery)


# The fault runs of DESIGN.md 11.5.
_FAULT_CASES = (
    _FaultCase(
        "A robot's process killed with SIGKILL",
        "disconnect",
        "`closed_abnormal`, `killed_robot_released`",
        _Phrase(
            (
                "session closed after ",
                _Measured("disconnect.detect_ms"),
                " ms; survivors' `depth` rate risen after ",
                _Measured("disconnect.depth.recovery_s"),
                " s",
            )
        ),
    ),
    _FaultCase(
        "The `depth` worker killed with SIGKILL",
        "worker_kill",
        _Phrase(
            (
                "`worker_restarted`; `cls` commands/s per robot before / after: ",
                _Measured("worker_kill.before.cls.commands_per_s_per_robot.mean"),
                " / ",
                _Measured("worker_kill.after.cls.commands_per_s_per_robot.mean"),
            )
        ),
        _Phrase(("`depth` commands again after ", _Measured("worker_kill.depth.outage_ms"), " ms")),
    ),
)
_FAULTS_HEADER = ("Case", "Scenario", "Checks", "Recovery")


def _faults(facts: Facts | None) -> _Section:
    """Write each fault case, as not run where no counted run of its scenario stands behind it.

    The footer gives the counted runs of each scenario, and the checks that failed in them.
    """
    rows: list[tuple[str, ...]] = []
    counts: list[str] = []
    failed: dict[str, tuple[str, ...]] = {}
    for case in _FAULT_CASES:
        keys = tuple(_row_keys((case.row,)))
        run_ids = set() if facts is None else _measured_run_ids(facts, keys)
        counts.append(f"{len(run_ids)} of `{case.scenario}`")
        if facts is None or not run_ids:
            rows.append((case.case, f"`{case.scenario}`", NOT_RUN, ""))
            continue
        rows.extend(_rows_text((case.row,), facts))
        for key in keys:
            failed.update(facts.facts[key].failed_checks)
    failures = "; ".join(
        f"{run_id}: {', '.join(f'`{check}`' for check in checks)}"
        for run_id, checks in sorted(failed.items())
    )
    footer = (f"Counted runs: {', '.join(counts)}.", f"Failed checks: {failures or 'none'}.")
    return _Section("Cases", _FAULTS_HEADER, tuple(rows), footer)


def _summary_value(summary: Mapping[str, Any], path: Sequence[str], run_id: str) -> object:
    value: object = summary
    for depth, key in enumerate(path):
        if not isinstance(value, dict) or key not in value:
            raise TablesError(f"the summary of {run_id} has no {'.'.join(path[: depth + 1])}")
        value = value[key]
    return value


def _summary_number(summary: Mapping[str, Any], path: Sequence[str], run_id: str) -> float | None:
    value = _summary_value(summary, path, run_id)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TablesError(f"the summary of {run_id}: {'.'.join(path)} is not a number")
    return value


type _Read = Callable[[Mapping[str, Any], str], float | None]


def _fleet_cores(window: str) -> _Read:
    """Return the reader of the cores of the fleet processes over a window of the run.

    The fleet processes run the robots; the driver is not one.
    """

    def read(summary: Mapping[str, Any], run_id: str) -> float | None:
        path = ("cpu", window, "cores")
        cores = _summary_value(summary, path, run_id)
        if not isinstance(cores, dict):
            raise TablesError(f"the summary of {run_id}: {'.'.join(path)} is not an object")
        names = sorted(name for name in cores if name == "fleet" or name.startswith("fleet_"))
        if not names:
            raise TablesError(f"the summary of {run_id}: {'.'.join(path)} holds no fleet process")
        values = [_summary_number(cores, (name,), run_id) for name in names]
        present = [value for value in values if value is not None]
        # A process without a value makes the sum unknown, not smaller.
        return sum(present) if len(present) == len(values) else None

    return read


@dataclass(frozen=True)
class _Quantity:
    """A row of the topology tables: one number read from each counted run's summary."""

    label: str
    unit: str
    read: _Read


def _at(*path: str) -> _Read:
    def read(summary: Mapping[str, Any], run_id: str) -> float | None:
        return _summary_number(summary, path, run_id)

    return read


def _topology_quantities(scenario: str) -> tuple[_Quantity, ...]:
    """Return the quantities of DESIGN.md 9.6 for a scenario key, over each run's window "all".

    The simulator's CPU of disconnect is read before the kill instead: the killed robot's
    own fleet process has exited by the last scrape of "all", so the fleet's sum is unknown
    there in every run.
    """
    simulator = (
        _Quantity("Simulator CPU before the kill (cores)", "cores", _fleet_cores("before"))
        if scenario == "disconnect"
        else _Quantity("Simulator CPU (cores)", "cores", _fleet_cores("all"))
    )
    return (
        *(
            _Quantity(
                f"`{model}` commands/s per robot",
                "commands/s",
                _at("models", model, "all", "commands_per_s_per_robot", "mean"),
            )
            for model in MODEL_KEYS
        ),
        *(
            _Quantity(
                f"`{model}` freshness p95 (ms)",
                "ms",
                _at("models", model, "all", "freshness_ms", "p95"),
            )
            for model in MODEL_KEYS
        ),
        simulator,
        _Quantity("Simulator fire lag p99 (ms)", "ms", _at("validity", "fire_lag_ms", "p99")),
        _Quantity("Throttled (ms per s)", "ms/s", _at("cpu", "all", "throttled_ms_per_s")),
    )


def _combined_text(values: Sequence[float | None], unit: str) -> str:
    """Show the median of the runs' values with the lowest and highest, as a fact is shown."""
    present = [value for value in values if value is not None]
    if not present:
        return format_value(None, unit)
    median = format_value(statistics.median(present), unit)
    return f"{median} ({format_value(min(present), unit)} to {format_value(max(present), unit)})"


def _topology_section(
    scenario: str,
    runs: Mapping[tuple[str, Topology], tuple[str, ...]],
    summaries: Mapping[str, Mapping[str, Any]],
) -> _Section:
    header = ("Quantity", *(f"`{topology}`" for topology in Topology))
    counts = [f"{len(runs[scenario, topology])} in `{topology}`" for topology in Topology]
    footer = (f"Counted runs: {', '.join(counts)}.",)
    if not any(runs[scenario, topology] for topology in Topology):
        return _Section(f"`{scenario}`", header, (_not_run_row(len(header)),), footer)
    rows = tuple(
        (
            quantity.label,
            *(
                _combined_text(
                    [
                        quantity.read(summaries[run_id], run_id)
                        for run_id in runs[scenario, topology]
                    ],
                    quantity.unit,
                )
                for topology in Topology
            ),
        )
        for quantity in _topology_quantities(scenario)
    )
    return _Section(f"`{scenario}`", header, rows, footer)


def topology_runs(facts: Facts | None) -> dict[tuple[str, Topology], tuple[str, ...]]:
    """Return the counted runs of each serving scenario key and topology, from facts.json.

    They are the run ids of the facts the registry places in that scenario key and
    topology. A key computed across scale points belongs to no scenario key, and its run
    ids to several.
    """
    owned = {
        key: (entry.scenario_key, entry.topology)
        for key, entry in REGISTRY.items()
        if entry.scenario_key in _SERVING_SCENARIOS and entry.topology is not None
    }
    runs: dict[tuple[str, Topology], set[str]] = {
        (scenario, topology): set() for scenario in _SERVING_SCENARIOS for topology in Topology
    }
    if facts is None:
        return {owner: () for owner in runs}
    missing = sorted(key for key in owned if key not in facts.facts)
    if missing:
        raise TablesError(f"{FACTS_FILE} has no fact {', '.join(missing)}")
    for key, owner in owned.items():
        runs[owner].update(facts.facts[key].run_ids)
    return {owner: tuple(sorted(run_ids)) for owner, run_ids in runs.items()}


def _file_text(title: str, intro: str, sections: Iterable[_Section]) -> str:
    lines = [f"# {title}", "", intro, ""]
    for section in sections:
        lines += [f"## {section.heading}", "", _markdown_row(section.header)]
        lines.append(_markdown_row(tuple("---" for _ in section.header)))
        lines += [_markdown_row(row) for row in section.rows]
        lines += ["", *section.footer, ""]
    return "\n".join(lines)


def _markdown_row(cells: tuple[str, ...]) -> str:
    # A bar inside a cell would end the cell.
    return "| " + " | ".join(cell.replace("|", "\\|") for cell in cells) + " |"


# Each table file of facts: its name, title, the sentence under the title, and its tables.
_FACT_FILES: tuple[tuple[str, str, str, Callable[[], tuple[_Layout, ...]]], ...] = (
    ("steady.md", "Freshness under steady load", "DESIGN.md 9.1, `single` topology.", _steady),
    ("late_joiner.md", "Late joiner", "DESIGN.md 9.2, `single` topology.", _late_joiner),
    ("sweep.md", "Scaling sweep", "DESIGN.md 9.3 and 6.5, `single` topology.", _sweep),
    ("disconnect.md", "Disconnect", "DESIGN.md 9.4, `single` topology.", _disconnect),
    ("massive.md", "Massive connections", "DESIGN.md 9.5, `single` topology.", _massive),
    (
        "breakdown_n5.md",
        "Latency breakdown at 5 robots",
        "DESIGN.md 7, `single` topology. Each model's stage rows, sum, end-to-end mean and "
        "residual come from one run.",
        _breakdown,
    ),
    ("memory_budget.md", "Memory budget", "DESIGN.md 8.3, `single` topology.", _memory_budget),
    ("env.md", "Test machines", "Where the counted runs were measured.", _env),
)
_PREDICTED_INTRO = (
    "Every prediction of the capacity model beside its measurement. The gap is the measured "
    "value minus the prediction and the spread is the highest minus the lowest run value; a "
    "gap larger than the spread is marked, and DESIGN.md explains it in its text."
)
_TOPOLOGY_INTRO = (
    "DESIGN.md 9.6 for every serving scenario, over the whole run unless the row names another "
    "span. A cell shows the median over the counted runs, with the lowest and highest run value "
    "in parentheses."
)
_FAULTS_INTRO = (
    "DESIGN.md 11.2, `single` topology: a robot killed in `disconnect` and the `depth` worker "
    "killed in `worker_kill`, each with the checks of its runs and the time the service takes "
    "to recover."
)


def render_tables(
    facts: Facts | None, summaries: Mapping[str, Mapping[str, Any]]
) -> dict[str, str]:
    """Return the text of every table file by its name.

    facts is None when the results root holds no facts.json, and every table is then
    written as not run. summaries holds the summary of every run topology_runs names.
    """
    layouts = {name: plan() for name, _, _, plan in _FACT_FILES}
    comparison = _predicted_vs_measured(facts)
    if facts is not None:
        planned = (*(layout for each in layouts.values() for layout in each), comparison)
        keys = {*_row_keys(row for layout in planned for row in layout.rows)}
        keys.update(_row_keys(_ledger_rows()))
        keys.update(_row_keys(case.row for case in _FAULT_CASES))
        missing = sorted(key for key in keys if key not in facts.facts)
        if missing:
            raise TablesError(f"{FACTS_FILE} has no fact {', '.join(missing)}")
    texts = {
        name: _file_text(
            title, f"{intro} {_CELL_NOTE}", (_fill(layout, facts) for layout in layouts[name])
        )
        for name, title, intro, _ in _FACT_FILES
    }
    texts["predicted_vs_measured.md"] = _file_text(
        "Predicted against measured",
        f"{_PREDICTED_INTRO} {_CELL_NOTE}",
        (_fill(comparison, facts),),
    )
    runs = topology_runs(facts)
    texts["topology.md"] = _file_text(
        "Topology comparison",
        _TOPOLOGY_INTRO,
        (_topology_section(scenario, runs, summaries) for scenario in _SERVING_SCENARIOS),
    )
    texts["faults.md"] = _file_text(
        "Fault runs",
        f"{_FAULTS_INTRO} {_CELL_NOTE}",
        (_faults(facts),),
    )
    texts["ledger.md"] = _file_text(
        "Tried and rejected",
        f"DESIGN.md 12: every candidate, with its outcome from the ledger. {_CELL_NOTE}",
        (_ledger(facts),),
    )
    return dict(sorted(texts.items()))


def _read_summary(path: Path, run_id: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TablesError(f"{path} is missing, though {FACTS_FILE} counts {run_id}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise TablesError(f"{path} is not JSON: {error}") from error
    if not isinstance(document, dict):
        raise TablesError(f"{path} must hold an object")
    return document


def write_tables(results: Path, out: Path) -> None:
    """Write every table under out/tables/, from the results root."""
    facts = read_facts(results)
    summaries = {
        run_id: _read_summary(results / RAW_DIRECTORY / scenario / run_id / SUMMARY_FILE, run_id)
        for (scenario, _), run_ids in topology_runs(facts).items()
        for run_id in run_ids
    }
    texts = render_tables(facts, summaries)
    directory = out / TABLES_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    for name, text in texts.items():
        (directory / name).write_bytes(text.encode("utf-8"))


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.tables", description="Write the tables of DESIGN.md 11.5."
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results"), help="results root (default: %(default)s)"
    )
    parser.add_argument(
        "--out", type=Path, help="the directory to write tables/ in (default: the results root)"
    )
    arguments = parser.parse_args(argv)
    try:
        write_tables(arguments.results, arguments.out or arguments.results)
    except (TablesError, RenderError) as error:
        print(f"bench.tables: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
