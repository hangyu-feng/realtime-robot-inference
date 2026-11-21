"""The plots of DESIGN.md 11.5, drawn from the facts, the ledger, the predictions and run summaries.

    python -m bench.plots --results DIR

writes one PNG per plot under DIR/plots/. A plot takes its runs only from summary/facts.json and
ledger.jsonl and opens each run's summary.json by its id: it never lists raw/ and never reads
samples or scrapes. Whatever a plot draws that is not a fact comes from one run, the counted run
of the single topology with the lowest repeat index (per arm for a comparison). An item with no
counted run is still written, as a figure that says "not run". The same inputs give the same
bytes.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import re
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.style
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import NullLocator

from bench.capacity_model import QUOTA_CORES
from bench.runfiles import ARMS, FACTS_FILE, SUMMARY_FILE, Fact, Facts, RunFileError
from bench.scenarios import Scenario, disconnect, late_joiner, massive
from cloud.config import Topology
from common.commands import MODEL_KEYS
from common.protocol import STAGE_NAMES
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS

type Array = npt.NDArray[np.float64]

# Names under the results root.
RAW_DIRECTORY = "raw"
COMPARE_DIRECTORY = "compare"
SUMMARY_DIRECTORY = "summary"
PLOTS_DIRECTORY = "plots"
PREDICTIONS_FILE = "predictions.json"
LEDGER_FILE = "ledger.jsonl"

# Every figure has one size and resolution, so a figure's bytes depend only on what it draws.
FIGURE_SIZE_IN = (11.0, 4.5)
FIGURE_DPI = 100
# The trailing span of the rolling p95 in the steady plot: a robot at 30 fps receives about 30
# classifier commands in it, enough for a 95th percentile that moves with the run.
ROLLING_P95_WINDOW_S = 1.0
# Bins of the per-robot rate histogram of the 50-robot run.
FAIRNESS_BINS = 20

# The fact whose run ids name the counted runs of a scenario key in the single topology. Each is
# a count or a rate that every counted run has, so its run ids are every counted run of the key.
RUNS_FACT_KEYS: Mapping[str, str] = {
    "steady": "steady.cls.commands_received",
    "late_joiner": "late_joiner.before.a.cls.commands_per_s",
    "sweep_n1": "sweep.n1.cls.commands_per_s_per_robot.mean",
    "sweep_n3": "sweep.n3.cls.commands_per_s_per_robot.mean",
    "sweep_n5": "sweep.n5.cls.commands_per_s_per_robot.mean",
    "sweep_n10": "sweep.n10.cls.commands_per_s_per_robot.mean",
    "disconnect": "disconnect.before.survivors.cls.commands_per_s_per_robot",
    "massive": "massive.cls.commands_received",
    "soak": "soak.cls.commands_per_s_per_robot.mean",
    "model_bench": "model_bench.dispatch_rt_ms",
}
# The serving scenario keys, in the order of the evidence plan.
SERVING_KEYS = tuple(key for key in RUNS_FACT_KEYS if key != "model_bench")
# The sweep's scale points: scenario key, robot count, and the prefix of its fact keys.
SWEEP_POINTS = (
    ("sweep_n1", 1, "sweep.n1"),
    ("sweep_n3", 3, "sweep.n3"),
    ("sweep_n5", 5, "sweep.n5"),
    ("sweep_n10", 10, "sweep.n10"),
)
MASSIVE_POINT = ("massive", len(massive().robots), "massive")
# The ten stages of DESIGN.md 7, in the order a command passes them.
BREAKDOWN_STAGES = ("fire_lag", *STAGE_NAMES, "deliver")
# The model benchmark's thread counts and batch sizes (DESIGN.md 11.4, Model bench).
BENCH_THREADS = (1, 2)
BENCH_BATCH_SIZES = (1, 2, 4, 8)
# The comparison whose arms fifo_vs_latest.png draws, and what each arm runs.
FIFO_CANDIDATE = "fifo_vs_latest"
FIFO_ARM_LABELS: Mapping[str, str] = {"a": "latest wins", "b": "FIFO control"}
# The curves of predictions.json that the sweep plots draw, per model.
PREDICTED_FIELDS = (
    "commands_per_s_per_robot",
    "freshness_ms_p50",
    "freshness_ms_p95",
    "batch_size_mean",
)

_RUN_ID = re.compile(r"(?P<topology>single|split)-[0-9a-f]{12}-[0-9a-f]{8}-r(?P<repeat>[0-9]+)")


class PlotInputError(Exception):
    """A file the plots read is missing, is not JSON, or lacks a field or a fact they draw."""


@dataclass(frozen=True)
class RunSummary:
    """The summary.json of one counted run, read by dotted paths."""

    run_id: str
    document: Mapping[str, Any]

    def at(self, path: str) -> object:
        value: object = self.document
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                raise PlotInputError(f"{SUMMARY_FILE} of run {self.run_id}: {path} is missing")
            value = value[key]
        return value

    def number(self, path: str) -> float:
        value = self.at(path)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise PlotInputError(f"{SUMMARY_FILE} of run {self.run_id}: {path} must be a number")
        return float(value)

    def numbers(self, path: str) -> Array:
        values = self.at(path)
        if not isinstance(values, list) or not all(
            isinstance(value, int | float) and not isinstance(value, bool) for value in values
        ):
            raise PlotInputError(
                f"{SUMMARY_FILE} of run {self.run_id}: {path} must be a list of numbers"
            )
        return np.array(values, dtype=np.float64)

    def keys(self, path: str) -> list[str]:
        value = self.at(path)
        if not isinstance(value, dict):
            raise PlotInputError(f"{SUMMARY_FILE} of run {self.run_id}: {path} must be an object")
        return sorted(value)

    def pair(self, path: str, x_name: str, y_name: str) -> tuple[Array, Array]:
        """Return the two lists of a series object, ordered by the first."""
        x = self.numbers(f"{path}.{x_name}")
        y = self.numbers(f"{path}.{y_name}")
        if x.size != y.size:
            raise PlotInputError(
                f"{SUMMARY_FILE} of run {self.run_id}: {path} has lists of different lengths"
            )
        order = np.argsort(x, kind="stable")
        return x[order], y[order]


@dataclass(frozen=True)
class Curves:
    """The curves of predictions.json over robot counts, null values as NaN."""

    robots: Array
    models: Mapping[str, Mapping[str, Array]]
    cpu_demand_cores: Array


@dataclass(frozen=True)
class PlotInputs:
    facts: Mapping[str, Fact]
    # None when no prediction was written: no model benchmark was counted, or the null model set.
    curves: Curves | None
    # Per serving scenario key with a counted single run, the summary of the one plots draw.
    one_run: Mapping[str, RunSummary]
    # Per arm of the FIFO control, the summary of its run with the lowest repeat; empty unless
    # both arms have counted runs.
    fifo_arms: Mapping[str, RunSummary]


@dataclass(frozen=True)
class Plot:
    png: bytes
    drawn: bool  # false for a figure that says "not run"


def stale_ratio_curve(freshness_counts: Array) -> Array:
    """Return the share of commands at or above each threshold of i ms, from counts in 1 ms bins.

    At the bins' resolution a command of bin i counts as older than i ms. The curve is empty
    when there is no command.
    """
    total = freshness_counts.sum()
    if total == 0:
        return np.zeros(0)
    return np.cumsum(freshness_counts[::-1])[::-1] / total


def coverage_curve(age_time_ms: Array, uncovered_ms: float) -> Array:
    """Return, for each watchdog limit of i ms, the share of the robots' time that is covered.

    age_time_ms holds the time the age of the command in effect spends in each 1 ms bin, and
    uncovered_ms the time before the first applied command. A limit of i ms covers the time
    spent in the bins below i. The curve is empty when the robots ran for no time.
    """
    total_ms = age_time_ms.sum() + uncovered_ms
    if total_ms == 0:
        return np.zeros(0)
    return np.concatenate(([0.0], np.cumsum(age_time_ms))) / total_ms


def rolling_p95(t_s: Array, values: Array, window_s: float) -> Array:
    """Return at each sample the p95 of the samples in the span of window_s that ends at it.

    t_s is in ascending order. The span is open at its start and closed at its end.
    """
    starts = np.searchsorted(t_s, t_s - window_s, side="right")
    return np.array(
        [np.percentile(values[start : end + 1], 95) for end, start in enumerate(starts)],
        dtype=np.float64,
    )


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        raise PlotInputError(f"cannot read {path}: {error}") from error


def _parse_json(text: str, label: str) -> object:
    try:
        return json.loads(text)
    except (ValueError, RecursionError) as error:
        raise PlotInputError(f"{label} is not JSON: {error}") from error


def _first_single_run(run_ids: Iterable[str], source: str) -> str | None:
    """Return the run of the single topology with the lowest repeat index; None for none."""
    repeats: dict[str, int] = {}
    for run_id in run_ids:
        match = _RUN_ID.fullmatch(run_id)
        if match is None:
            raise PlotInputError(f"{source}: {run_id!r} is not a run id")
        if match["topology"] == Topology.SINGLE:
            repeats[run_id] = int(match["repeat"])
    return min(repeats, key=lambda run_id: (repeats[run_id], run_id), default=None)


def _read_summary(run_directory: Path, run_id: str) -> RunSummary:
    document = _parse_json(
        _read_text(run_directory / SUMMARY_FILE), f"{SUMMARY_FILE} of run {run_id}"
    )
    if not isinstance(document, dict):
        raise PlotInputError(f"{SUMMARY_FILE} of run {run_id} must be an object")
    return RunSummary(run_id, document)


def _fact(facts: Mapping[str, Fact], key: str) -> Fact:
    if key not in facts:
        raise PlotInputError(f"{FACTS_FILE}: {key} is missing")
    return facts[key]


def _value(fact: Fact, key: str) -> float:
    """Return a fact's value, NaN for null, which a plot leaves out."""
    if fact.value is None:
        return math.nan
    if isinstance(fact.value, bool | str):
        raise PlotInputError(f"{FACTS_FILE}: {key} must be a number or null")
    return float(fact.value)


def _read_curves(results: Path) -> Curves | None:
    path = results / SUMMARY_DIRECTORY / PREDICTIONS_FILE
    if not path.exists():
        return None
    document = _parse_json(_read_text(path), PREDICTIONS_FILE)
    curves = document.get("curves") if isinstance(document, dict) else None
    if not isinstance(curves, dict):
        raise PlotInputError(f"{PREDICTIONS_FILE}: curves must be an object")
    robots = _curve(curves, "robots", "curves.robots", None)
    models: dict[str, dict[str, Array]] = {}
    for model in MODEL_KEYS:
        fields = curves.get(model)
        if not isinstance(fields, dict):
            raise PlotInputError(f"{PREDICTIONS_FILE}: curves.{model} must be an object")
        models[model] = {
            field: _curve(fields, field, f"curves.{model}.{field}", robots.size)
            for field in PREDICTED_FIELDS
        }
    cpu_demand_cores = _curve(curves, "cpu_demand_cores", "curves.cpu_demand_cores", robots.size)
    return Curves(robots, models, cpu_demand_cores)


def _curve(document: Mapping[str, Any], key: str, path: str, length: int | None) -> Array:
    """Return one list of a curve, null as NaN; length is that of the robot counts."""
    values = document.get(key)
    if (
        not isinstance(values, list)
        or not all(
            value is None or (isinstance(value, int | float) and not isinstance(value, bool))
            for value in values
        )
        or (length is not None and len(values) != length)
    ):
        raise PlotInputError(
            f"{PREDICTIONS_FILE}: {path} must be a list of numbers, one per robot count"
        )
    return np.array([math.nan if value is None else value for value in values], dtype=np.float64)


def _read_ledger_run_ids(results: Path) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return per candidate of the ledger its run ids per arm."""
    run_ids: dict[str, dict[str, tuple[str, ...]]] = {}
    for number, line in enumerate(_read_text(results / LEDGER_FILE).splitlines(), start=1):
        label = f"{LEDGER_FILE} line {number}"
        document = _parse_json(line, label)
        if not isinstance(document, dict) or not isinstance(document.get("name"), str):
            raise PlotInputError(f"{label}: name must be a string")
        arms = document.get("run_ids")
        if not (
            isinstance(arms, dict)
            and arms.keys() == set(ARMS)
            and all(
                isinstance(ids, list) and all(isinstance(run_id, str) for run_id in ids)
                for ids in arms.values()
            )
        ):
            raise PlotInputError(f"{label}: run_ids must hold a list of run ids per arm")
        run_ids[document["name"]] = {arm: tuple(arms[arm]) for arm in ARMS}
    return run_ids


def _read_fifo_arms(results: Path, arms: Mapping[str, tuple[str, ...]]) -> dict[str, RunSummary]:
    first = {arm: _first_single_run(arms[arm], f"{LEDGER_FILE}: {FIFO_CANDIDATE}") for arm in ARMS}
    directory = results / RAW_DIRECTORY / COMPARE_DIRECTORY / FIFO_CANDIDATE
    return {
        arm: _read_summary(directory / arm / run_id, run_id)
        for arm, run_id in first.items()
        if run_id is not None
    }


def load_inputs(results: Path) -> PlotInputs | None:
    """Read what the plots draw from the results root; None when it holds no facts.json.

    Aggregation writes no facts.json for a results root without runs, and then nothing is
    counted.
    """
    facts_path = results / SUMMARY_DIRECTORY / FACTS_FILE
    if not facts_path.exists():
        return None
    try:
        facts = Facts.from_json(_read_text(facts_path)).facts
    except RunFileError as error:
        raise PlotInputError(str(error)) from error
    one_run: dict[str, RunSummary] = {}
    for key in SERVING_KEYS:
        fact_key = RUNS_FACT_KEYS[key]
        run_id = _first_single_run(_fact(facts, fact_key).run_ids, f"{FACTS_FILE}: {fact_key}")
        if run_id is not None:
            one_run[key] = _read_summary(results / RAW_DIRECTORY / key / run_id, run_id)
    ledger = _read_ledger_run_ids(results)
    fifo_arms = _read_fifo_arms(results, ledger[FIFO_CANDIDATE]) if FIFO_CANDIDATE in ledger else {}
    return PlotInputs(
        facts=facts,
        curves=_read_curves(results),
        one_run=one_run,
        fifo_arms=fifo_arms if fifo_arms.keys() == set(ARMS) else {},
    )


def _figure(title: str, columns: int) -> tuple[Figure, list[Axes]]:
    figure = Figure(figsize=FIGURE_SIZE_IN, dpi=FIGURE_DPI, layout="constrained")
    grid = figure.subplots(1, columns, squeeze=False)
    figure.suptitle(title)
    return figure, [grid[0, column] for column in range(columns)]


def _not_run(title: str) -> Figure:
    figure = Figure(figsize=FIGURE_SIZE_IN, dpi=FIGURE_DPI)
    figure.text(0.5, 0.5, f"{title}\nnot run", ha="center", va="center", fontsize="x-large")
    return figure


def _png(figure: Figure) -> bytes:
    buffer = io.BytesIO()
    # Without the software entry the file holds only the drawing, so equal inputs give equal
    # bytes. Agg writes no date into a PNG.
    figure.savefig(buffer, format="png", dpi=FIGURE_DPI, metadata={"Software": None})
    return buffer.getvalue()


def _label(axes: Axes, title: str, x_label: str, y_label: str) -> None:
    axes.set_title(title)
    axes.set_xlabel(x_label)
    axes.set_ylabel(y_label)


def _count_ticks(axes: Axes, counts: Sequence[int]) -> None:
    axes.set_xticks(counts, labels=[str(count) for count in counts])
    axes.xaxis.set_minor_locator(NullLocator())


def _fact_points(facts: Mapping[str, Fact], keys: Sequence[str]) -> tuple[Array, Array]:
    """Return the facts' values and, as error bars, their distances to the lowest and the
    highest run value; NaN where a fact has none."""
    values: list[float] = []
    below: list[float] = []
    above: list[float] = []
    for key in keys:
        fact = _fact(facts, key)
        value = _value(fact, key)
        values.append(value)
        below.append(math.nan if fact.min is None else value - fact.min)
        above.append(math.nan if fact.max is None else fact.max - value)
    return np.array(values, dtype=np.float64), np.array([below, above], dtype=np.float64)


def _predicted(
    axes: Axes, inputs: PlotInputs, model: str, field: str, largest_count: int, color: str
) -> None:
    """Draw a predicted curve up to a robot count; nothing when no prediction was written."""
    if inputs.curves is None:
        return
    keep = inputs.curves.robots <= largest_count
    axes.plot(
        inputs.curves.robots[keep],
        inputs.curves.models[model][field][keep],
        color=color,
        linestyle=":",
        label="predicted",
    )


def _freshness_series(axes: Axes, run: RunSummary, model: str, label: str, color: str) -> None:
    """Draw the freshness of every command against time, and its rolling p95."""
    t_s, freshness_ms = run.pair(f"series.freshness.{model}", "t_s", "freshness_ms")
    axes.plot(t_s, freshness_ms, linestyle="none", marker=".", markersize=2, color=color, alpha=0.4)
    axes.plot(
        t_s,
        rolling_p95(t_s, freshness_ms, ROLLING_P95_WINDOW_S),
        color=color,
        label=f"{label}: each command, and p95 over the trailing second",
    )


def _stale_threshold(axes: Axes, model: str) -> None:
    # The robots of the evidence runs declare the default thresholds.
    axes.axhline(DEFAULT_STALE_MS[model], color="k", linestyle="--", label="stale threshold")


def _steady_freshness(inputs: PlotInputs, title: str) -> Figure | None:
    run = inputs.one_run.get("steady")
    if run is None:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        _freshness_series(axes, run, model, "freshness", "C0")
        _stale_threshold(axes, model)
        _label(axes, model, "time from t0 (s)", "freshness (ms)")
        axes.legend(fontsize="small")
    return figure


def _fifo_vs_latest(inputs: PlotInputs, title: str) -> Figure | None:
    if not inputs.fifo_arms:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        for index, arm in enumerate(ARMS):
            run = inputs.fifo_arms[arm]
            _freshness_series(axes, run, model, FIFO_ARM_LABELS[arm], f"C{index}")
        _stale_threshold(axes, model)
        _label(axes, model, "time from t0 (s)", "freshness (ms)")
        axes.legend(fontsize="small")
    return figure


def _throughput_over_time(
    inputs: PlotInputs, key: str, scenario: Scenario, title: str
) -> Figure | None:
    """Draw each robot's commands per second in 1 s bins, with the planned joins and kills."""
    run = inputs.one_run.get(key)
    if run is None:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        # A killed robot writes no samples, so the series holds the robots that ran on.
        for robot_id in run.keys("series.commands_per_bin"):
            counts = run.numbers(f"series.commands_per_bin.{robot_id}.{model}")
            axes.stairs(counts, np.arange(counts.size + 1), label=f"robot {robot_id}")
        for plan in scenario.robots:
            if plan.start_s > 0:
                axes.axvline(
                    plan.start_s, color="k", linestyle="--", label=f"{plan.robot_id} joins"
                )
            if plan.stop_s is not None:
                axes.axvline(plan.stop_s, color="k", linestyle=":", label=f"{plan.robot_id} killed")
        _label(axes, model, "time from t0 (s)", "commands per second")
        axes.legend(fontsize="small")
    return figure


def _robot_values(
    inputs: PlotInputs, points: Sequence[tuple[str, int, str]], model: str, metric: str
) -> tuple[list[int], list[float]]:
    """Return, at each robot count, each robot's value of a metric in the one run of that count."""
    counts: list[int] = []
    values: list[float] = []
    for key, count, _ in points:
        run = inputs.one_run.get(key)
        if run is not None:
            for robot_id in run.keys("robots"):
                counts.append(count)
                values.append(run.number(f"robots.{robot_id}.{model}.all.{metric}"))
    return counts, values


def _scale_points(
    axes: Axes, inputs: PlotInputs, model: str, fact_metric: str, robot_metric: str, color: str
) -> None:
    """Draw a metric against robot count: the fact at each count with its range over the runs,
    beside each robot's value in one run, and the 50-robot run as a point of its own."""
    for points, marker, robot_marker, name in (
        (SWEEP_POINTS, "o", ".", "sweep"),
        ((MASSIVE_POINT,), "D", "x", "massive"),
    ):
        values, errors = _fact_points(
            inputs.facts, [f"{prefix}.{model}.{fact_metric}" for _, _, prefix in points]
        )
        axes.errorbar(
            [count for _, count, _ in points],
            values,
            yerr=errors,
            color=color,
            marker=marker,
            capsize=3,
            label=f"{fact_metric}, {name}",
        )
        robot_counts, robot_values = _robot_values(inputs, points, model, robot_metric)
        axes.plot(
            robot_counts,
            robot_values,
            linestyle="none",
            marker=robot_marker,
            color=color,
            alpha=0.4,
            label=f"each robot of one run, {name}",
        )


def _sweep_runs(inputs: PlotInputs) -> list[tuple[int, RunSummary]]:
    """Return the robot count and the one run of each sweep point that has a counted run."""
    return [(count, inputs.one_run[key]) for key, count, _ in SWEEP_POINTS if key in inputs.one_run]


_SWEEP_COUNTS = [count for _, count, _ in SWEEP_POINTS]
_SCALE_COUNTS = [*_SWEEP_COUNTS, MASSIVE_POINT[1]]


def _sweep_throughput(inputs: PlotInputs, title: str) -> Figure | None:
    if not _sweep_runs(inputs):
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        _scale_points(axes, inputs, model, "commands_per_s_per_robot.mean", "commands_per_s", "C0")
        _predicted(axes, inputs, model, "commands_per_s_per_robot", MASSIVE_POINT[1], "C0")
        axes.set_xscale("log")
        _count_ticks(axes, _SCALE_COUNTS)
        _label(axes, model, "robots", "commands per second per robot")
        axes.legend(fontsize="small")
    return figure


def _sweep_freshness(inputs: PlotInputs, title: str) -> Figure | None:
    if not _sweep_runs(inputs):
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        for statistic, color in (("p50", "C0"), ("p95", "C1")):
            metric = f"freshness_ms.{statistic}"
            _scale_points(axes, inputs, model, metric, metric, color)
            _predicted(axes, inputs, model, f"freshness_ms_{statistic}", MASSIVE_POINT[1], color)
        axes.set_xscale("log")
        _count_ticks(axes, _SCALE_COUNTS)
        _label(axes, model, "robots", "freshness (ms)")
        axes.legend(fontsize="small")
    return figure


def _batch_shares(run: RunSummary, model: str) -> dict[int, float]:
    """Return the share of the batches of each size in window all; empty for no batch."""
    path = f"models.{model}.all.batch_size.distribution"
    counts: dict[int, float] = {}
    for size in run.keys(path):
        if not size.isdecimal():
            raise PlotInputError(f"{SUMMARY_FILE} of run {run.run_id}: {path} has key {size!r}")
        counts[int(size)] = run.number(f"{path}.{size}")
    total = sum(counts.values())
    return {size: count / total for size, count in counts.items()} if total else {}


def _sweep_batch_size(inputs: PlotInputs, title: str) -> Figure | None:
    present = _sweep_runs(inputs)
    if not present:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        shares = [_batch_shares(run, model) for _, run in present]
        bottoms = np.zeros(len(present))
        for index, size in enumerate(
            sorted({size for run_shares in shares for size in run_shares})
        ):
            heights = np.array([run_shares.get(size, 0.0) for run_shares in shares])
            axes.bar(
                [count for count, _ in present],
                heights,
                bottom=bottoms,
                color=f"C{index}",
                label=f"share of batches of {size}, one run",
            )
            bottoms += heights
        mean_axes = axes.twinx()
        values, errors = _fact_points(
            inputs.facts, [f"{prefix}.{model}.batch_size.mean" for _, _, prefix in SWEEP_POINTS]
        )
        mean_axes.errorbar(
            _SWEEP_COUNTS, values, yerr=errors, color="k", marker="o", capsize=3, label="mean size"
        )
        _predicted(mean_axes, inputs, model, "batch_size_mean", _SWEEP_COUNTS[-1], "k")
        mean_axes.set_ylabel("mean batch size")
        _count_ticks(axes, _SWEEP_COUNTS)
        _label(axes, model, "robots", "share of batches")
        handles, labels = axes.get_legend_handles_labels()
        mean_handles, mean_labels = mean_axes.get_legend_handles_labels()
        axes.legend(handles + mean_handles, labels + mean_labels, fontsize="small")
    return figure


def _process_shares(axes: Axes, present: Sequence[tuple[int, RunSummary]]) -> None:
    """Draw each process's cores in the one run of each count, as a share of the CPU limit."""
    processes = sorted({process for _, run in present for process in run.keys("cpu.all.cores")})
    for index, process in enumerate(processes):
        cores = [
            (count, run.number(f"cpu.all.cores.{process}"))
            for count, run in present
            if process in run.keys("cpu.all.cores")
        ]
        # The limit of a counted run is QUOTA_CORES: the limits gate compares them exactly.
        axes.plot(
            [count for count, _ in cores],
            [process_cores / QUOTA_CORES for _, process_cores in cores],
            color=f"C{index}",
            marker=".",
            label=f"{process}, one run",
        )


def _sweep_cpu(inputs: PlotInputs, title: str) -> Figure | None:
    present = _sweep_runs(inputs)
    if not present:
        return None
    figure, (share_axes, throttled_axes) = _figure(title, 2)
    for axes, metric, label in (
        (share_axes, "utilization", "total"),
        (throttled_axes, "throttled_ms_per_s", "throttled"),
    ):
        values, errors = _fact_points(
            inputs.facts, [f"{prefix}.cpu.{metric}" for _, _, prefix in SWEEP_POINTS]
        )
        axes.errorbar(
            _SWEEP_COUNTS, values, yerr=errors, color="k", marker="o", capsize=3, label=label
        )
    _process_shares(share_axes, present)
    if inputs.curves is not None:
        keep = inputs.curves.robots <= _SWEEP_COUNTS[-1]
        share_axes.plot(
            inputs.curves.robots[keep],
            inputs.curves.cpu_demand_cores[keep] / QUOTA_CORES,
            color="k",
            linestyle=":",
            label="predicted demand",
        )
    for axes, axes_title, y_label in (
        (share_axes, "utilization", "share of the CPU limit"),
        (throttled_axes, "throttling", "throttled time (ms per s)"),
    ):
        _count_ticks(axes, _SWEEP_COUNTS)
        _label(axes, axes_title, "robots", y_label)
        axes.legend(fontsize="small")
    return figure


def _massive_fairness(inputs: PlotInputs, title: str) -> Figure | None:
    run = inputs.one_run.get("massive")
    if run is None:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        rates = np.array(
            [
                run.number(f"robots.{robot_id}.{model}.all.commands_per_s")
                for robot_id in run.keys("robots")
            ]
        )
        axes.hist(rates, bins=FAIRNESS_BINS, label="robots of one run")
        axes.axvline(float(rates.mean()), color="k", linestyle="--", label="mean")
        _label(axes, model, "commands per second", "robots")
        axes.legend(fontsize="small")
    return figure


def _memory(inputs: PlotInputs, key: str, title: str) -> Figure | None:
    run = inputs.one_run.get(key)
    if run is None:
        return None
    figure, (axes,) = _figure(title, 1)
    t_s, memory_mb = run.pair("series.memory", "t_s", "anon_sock_mb")
    axes.plot(t_s, memory_mb, marker=".", label="anonymous plus socket memory, one run")
    _label(axes, "", "time from t0 (s)", "MB")
    axes.legend(fontsize="small")
    return figure


def _breakdown(inputs: PlotInputs, title: str) -> Figure | None:
    if "sweep_n5" not in inputs.one_run:
        return None
    figure, (axes,) = _figure(title, 1)
    positions = np.arange(len(MODEL_KEYS), dtype=np.float64)

    def means(metric: str) -> Array:
        keys = [f"sweep.n5.{model}.{metric}" for model in MODEL_KEYS]
        return np.array([_value(_fact(inputs.facts, key), key) for key in keys])

    colors = matplotlib.colormaps["tab20"]
    tops = np.zeros(len(MODEL_KEYS))
    for index, stage in enumerate(BREAKDOWN_STAGES):
        heights = means(f"stage.{stage}_ms.mean")
        axes.bar(positions, heights, bottom=tops, color=colors(index), label=stage)
        tops += heights
    axes.bar(
        positions, means("residual_ms.mean"), bottom=tops, color="0.5", hatch="//", label="residual"
    )
    axes.plot(
        positions,
        means("e2e_ms.mean"),
        linestyle="none",
        marker="_",
        markersize=40,
        color="k",
        label="end-to-end mean",
    )
    axes.set_xticks(positions, labels=list(MODEL_KEYS))
    _label(axes, "", "model", "mean time (ms)")
    axes.legend(fontsize="small", loc="upper left", bbox_to_anchor=(1.0, 1.0))
    return figure


def _stale_ratio_vs_threshold(inputs: PlotInputs, title: str) -> Figure | None:
    present = [(key, inputs.one_run[key]) for key in SERVING_KEYS if key in inputs.one_run]
    if not present:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        for index, (key, run) in enumerate(present):
            curve = stale_ratio_curve(run.numbers(f"models.{model}.curves.freshness_counts"))
            axes.plot(np.arange(curve.size), curve, color=f"C{index}", label=key)
        axes.axvline(DEFAULT_STALE_MS[model], color="k", linestyle="--", label="stale threshold")
        _label(axes, model, "threshold (ms)", "share of commands older at receipt")
        axes.legend(fontsize="small")
    return figure


def _coverage_vs_watchdog(inputs: PlotInputs, title: str) -> Figure | None:
    present = [
        (count, inputs.one_run[key])
        for key, count, _ in (*SWEEP_POINTS, MASSIVE_POINT)
        if key in inputs.one_run
    ]
    if not present:
        return None
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        for index, (count, run) in enumerate(present):
            curve = coverage_curve(
                run.numbers(f"models.{model}.curves.age_time_ms"),
                run.number(f"models.{model}.curves.uncovered_ms"),
            )
            axes.plot(np.arange(curve.size), curve, color=f"C{index}", label=f"robots: {count}")
        axes.axvline(DEFAULT_WATCHDOG_MS[model], color="k", linestyle="--", label="watchdog limit")
        _label(axes, model, "watchdog limit (ms)", "share of time covered")
        axes.legend(fontsize="small")
    return figure


def _model_tb(inputs: PlotInputs, title: str) -> Figure | None:
    if not _fact(inputs.facts, RUNS_FACT_KEYS["model_bench"]).run_ids:
        return None
    sizes = np.array(BENCH_BATCH_SIZES, dtype=np.float64)
    figure, axes_list = _figure(title, len(MODEL_KEYS))
    for axes, model in zip(axes_list, MODEL_KEYS, strict=True):
        for index, threads in enumerate(BENCH_THREADS):
            for metric, name, style in (("wall_ms", "wall", "-"), ("cpu_ms", "CPU", "--")):
                keys = [
                    f"model_bench.{model}.t{threads}.b{size}.{metric}" for size in BENCH_BATCH_SIZES
                ]
                values, errors = _fact_points(inputs.facts, keys)
                # The facts are the times of a whole batch.
                axes.errorbar(
                    sizes,
                    values / sizes,
                    yerr=errors / sizes,
                    color=f"C{index}",
                    linestyle=style,
                    marker="o",
                    capsize=3,
                    label=f"{name}, threads: {threads}",
                )
        axes.set_xscale("log", base=2)
        _count_ticks(axes, BENCH_BATCH_SIZES)
        _label(axes, model, "batch size", "time per frame (ms)")
        axes.legend(fontsize="small")
    return figure


type _Draw = Callable[[PlotInputs, str], Figure | None]

# Every plot of DESIGN.md 11.5: its file, its title, and how it is drawn.
_PLOTS: tuple[tuple[str, str, _Draw], ...] = (
    ("steady_freshness.png", "Steady load: freshness of every command", _steady_freshness),
    (
        "late_joiner_throughput.png",
        "Late joiner: commands per second per robot",
        lambda inputs, title: _throughput_over_time(inputs, "late_joiner", late_joiner(), title),
    ),
    ("sweep_throughput.png", "Scaling sweep: commands per second per robot", _sweep_throughput),
    ("sweep_freshness.png", "Scaling sweep: freshness", _sweep_freshness),
    ("sweep_batch_size.png", "Scaling sweep: batch sizes", _sweep_batch_size),
    ("sweep_cpu.png", "Scaling sweep: CPU", _sweep_cpu),
    (
        "disconnect_throughput.png",
        "Disconnect: commands per second per robot",
        lambda inputs, title: _throughput_over_time(inputs, "disconnect", disconnect(), title),
    ),
    ("massive_fairness.png", "Massive: per-robot command rates", _massive_fairness),
    (
        "massive_memory.png",
        "Massive: anonymous plus socket memory",
        lambda inputs, title: _memory(inputs, "massive", title),
    ),
    (
        "soak_memory.png",
        "Soak: anonymous plus socket memory",
        lambda inputs, title: _memory(inputs, "soak", title),
    ),
    ("breakdown_n5.png", "Latency breakdown in the sweep of 5 robots", _breakdown),
    ("stale_ratio_vs_threshold.png", "Stale ratio against threshold", _stale_ratio_vs_threshold),
    ("coverage_vs_watchdog.png", "Watchdog coverage against limit", _coverage_vs_watchdog),
    ("model_tb.png", "Model benchmark: time per frame against batch size", _model_tb),
    (
        "fifo_vs_latest.png",
        "Freshness under latest wins and under the FIFO control",
        _fifo_vs_latest,
    ),
)


def render_plots(inputs: PlotInputs | None) -> dict[str, Plot]:
    """Draw every plot by its file name; with no inputs, every plot says "not run"."""
    plots: dict[str, Plot] = {}
    # The default style, whatever a matplotlibrc on the machine sets.
    with matplotlib.style.context("default"):
        for name, title, draw in _PLOTS:
            figure = None if inputs is None else draw(inputs, title)
            png = _png(_not_run(title) if figure is None else figure)
            plots[name] = Plot(png, drawn=figure is not None)
    return plots


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.plots",
        description="Draw the plots of the design document from a results root.",
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("results"),
        metavar="DIR",
        help="the results root (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    try:
        plots = render_plots(load_inputs(arguments.results))
    except PlotInputError as error:
        print(f"bench.plots: {error}", file=sys.stderr)
        return 1
    directory = arguments.results / PLOTS_DIRECTORY
    try:
        directory.mkdir(parents=True, exist_ok=True)
        for name, plot in plots.items():
            (directory / name).write_bytes(plot.png)
    except OSError as error:
        print(f"bench.plots: cannot write {directory}: {error}", file=sys.stderr)
        return 1
    for name, plot in plots.items():
        print(f"{PLOTS_DIRECTORY}/{name}" + ("" if plot.drawn else ": not run"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
