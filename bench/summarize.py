"""The summary of one finished run: every number of the run that aggregation reads.

summarize_run computes a serving run's summary from its files and its samples, and
summarize_model_bench the model benchmark's from its reports. write_summary writes summary.json
and then records its hash in DONE; it never changes the hashes of the run's other files. Each
quantity is defined in the code that computes it.

    python -m bench.summarize --results DIR

writes the summary of every finished run under DIR/raw whose summary is missing, was written by
another version of this code, or does not match the hash in DONE. It exits 1 when a summary
could not be computed, after trying every run.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
import math
import os
import re
import sys
import zipfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.capacity_model import command_gap_bound_ns, longest_cycle_ns
from bench.live_check import conservation_violations
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    EVENTS_FILE,
    GATE_FILE,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    SUMMARY_FILE,
    Done,
    Event,
    EventName,
    FleetConfig,
    Gate,
    RunConfig,
    RunFileError,
    RunSpec,
    Scrape,
)
from bench.scenarios import RobotPlan, Scenario, Window, WorkerKill
from bench.simulation import Overheads, ServiceTable
from bench.stats import summarize
from cloud.counters import WorkerState
from common.commands import MODEL_KEYS
from common.histogram import Histogram, difference
from common.protocol import STAGE_NAMES
from robot.control import Verdict
from robot.samples import COMMAND_DTYPE, TICK_DTYPE, VERDICTS, TickOutcome, load_samples

# A run counts as limited by the simulator when the p99 of its fire lag is above this share of
# the frame period: freshness differences under a tenth of a period are below what any table
# compares.
SIMULATOR_FIRE_LAG_SHARE = 0.1
# The memory baseline is taken this long after t0. Every robot of the 50-robot runs connects at
# t0, so a baseline taken there would count the connections and the runtime's arenas as growth.
MEMORY_SLOPE_SKIP_S = 5.0
# The width of an age bin of the two curves of DESIGN.md 5.3: finer than any threshold or
# watchdog limit that a curve is read at.
CURVE_BIN_MS = 1
# The width of a bin of the per-robot command series: the resolution at which the settle and
# recovery times are read.
SERIES_BIN_S = 1

# The scenario key of the model benchmark. It is not taken from the run plan: a change to the
# plan must not change this module's digest and with it make every summary stale.
MODEL_BENCH_KEY = "model_bench"
REPORTS_DIRECTORY = "reports"
RAW_DIRECTORY = "raw"
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
ALL_WINDOW = "all"

_SUMMARIZER = "bench/summarize.py"
_ATTEMPT = re.compile(r"\.attempt[0-9]+$")
_SAMPLES_PATTERN = "robot_*.npz"

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
_NS_PER_US = 1_000
_US_PER_MS = 1_000
_US_PER_S = 1_000_000
_MS_PER_S = 1_000
_S_PER_MIN = 60
_BYTES_PER_MB = 2**20
# Later than any reading of the monotonic clock: the replacement time of a command that no
# later command replaces.
_NEVER_NS = np.iinfo(np.int64).max

_APPLIED = VERDICTS.index(Verdict.APPLIED)
_STALE = VERDICTS.index(Verdict.STALE)
_OUT_OF_ORDER = VERDICTS.index(Verdict.OUT_OF_ORDER)
# The stages between a batch's dispatch and its result at the gateway.
_AFTER_DISPATCH = [
    STAGE_NAMES.index(name)
    for name in ("dispatch_rt", "decode", "preprocess", "batch_peers", "inference", "postprocess")
]
# The stages of a frame that both of its commands carry.
_PER_FRAME_STAGES = ("fire_lag_ms", "transport_up_ms")
_STAGE_STATISTICS = ("mean", "p50", "p95", "p99")
_GATEWAY_PROCESSES = ("gateway", *(f"worker_{model}" for model in MODEL_KEYS))

type Rows = npt.NDArray[np.void]
type IntArray = npt.NDArray[np.int64]
type FloatArray = npt.NDArray[np.float64]


class SummaryError(Exception):
    """A run's files do not hold what its summary is computed from."""


def in_window(times_ns: IntArray, start_ns: int, end_ns: int) -> npt.NDArray[np.bool_]:
    """Return which times belong to the window [start_ns, end_ns)."""
    # ASSUMPTION: A14. The workers warm up before any robot connects, so no window holds
    # warm-up, and a statistic covers every sample of its window: nothing is trimmed at the
    # start. A command belongs to the window that holds its receive time, a tick to the one
    # that holds its capture time.
    return (times_ns >= start_ns) & (times_ns < end_ns)


def least_squares_slope(x: FloatArray, y: FloatArray) -> float | None:
    """Return the slope of the least-squares line of y against x.

    None for fewer than two points, and where x does not vary.
    """
    if x.size < 2:
        return None
    centred = x - x.mean()
    spread = float(centred @ centred)
    if spread == 0:
        return None
    return float(centred @ (y - y.mean())) / spread


@dataclass(frozen=True)
class CommandGaps:
    """The intervals without a command of one robot and model over its time in a window."""

    # From the later of the window's start and the robot's start to its first command.
    first_ns: int
    # Between consecutive commands.
    closed_ns: IntArray
    # From its last command to the earlier of the window's end and the robot's stop.
    last_ns: int

    @property
    def longest_ns(self) -> int:
        return max(self.first_ns, self.last_ns, int(self.closed_ns.max(initial=0)))


def command_gaps(receive_ns: IntArray, from_ns: int, until_ns: int) -> CommandGaps:
    """Return the gaps between the commands received, in order, within [from_ns, until_ns)."""
    if receive_ns.size == 0:
        # A robot that received nothing went its whole time in the window without a command,
        # so both open intervals are that time.
        return CommandGaps(until_ns - from_ns, np.empty(0, np.int64), until_ns - from_ns)
    return CommandGaps(
        int(receive_ns[0]) - from_ns, np.diff(receive_ns), until_ns - int(receive_ns[-1])
    )


def peak_ages_ns(capture_ns: IntArray, receive_ns: IntArray, until_ns: int) -> IntArray:
    """Return the age each applied command reached before it was replaced.

    The commands are one robot's applied commands of one model, in the order received. A
    command is replaced when the next one is received; the last one is not, and its peak is
    its age at until_ns, the end of the robot's time in the window.
    """
    if capture_ns.size == 0:
        return np.empty(0, np.int64)
    peaks: IntArray = np.append(receive_ns[1:], until_ns) - capture_ns
    return peaks


def watchdog_coverage(
    capture_ns: IntArray, receive_ns: IntArray, watchdog_ns: int, from_ns: int, until_ns: int
) -> float:
    """Return the share of [from_ns, until_ns) in which the robot held an unexpired command.

    The commands are every applied command of one robot and model in the run, in the order
    received. Command i covers from its receipt until the next applied command replaces it
    or its age reaches the watchdog limit, whichever is first. The time before the first
    applied command is not covered.
    """
    replaced_ns = np.append(receive_ns[1:], _NEVER_NS)
    ends_ns = np.clip(np.minimum(replaced_ns, capture_ns + watchdog_ns), from_ns, until_ns)
    starts_ns = np.clip(receive_ns, from_ns, until_ns)
    return float(np.maximum(ends_ns - starts_ns, 0).sum()) / (until_ns - from_ns)


def age_time_ms(
    capture_ns: IntArray, receive_ns: IntArray, from_ns: int, until_ns: int
) -> tuple[FloatArray, float]:
    """Return the time one robot spent at each command age, and its time without a command.

    The commands are the robot's applied commands of one model received in [from_ns,
    until_ns), in order. Command i is in effect from its receipt to the receipt of the next
    one, or to until_ns, and its age rises linearly over that time, so bin j of the first
    result holds the milliseconds during which the age was in [j, j + 1) x CURVE_BIN_MS. The
    second result is the time before the first command, in ms. The two add up to the length
    of [from_ns, until_ns).
    """
    if receive_ns.size == 0:
        return np.zeros(0), (until_ns - from_ns) / _NS_PER_MS
    replaced_ns = np.append(receive_ns[1:], until_ns)
    youngest = (receive_ns - capture_ns) / _NS_PER_MS / CURVE_BIN_MS
    oldest = (replaced_ns - capture_ns) / _NS_PER_MS / CURVE_BIN_MS
    # A command replaced at the instant it arrived is never in effect.
    held = oldest > youngest
    youngest, oldest = youngest[held], oldest[held]
    low = np.floor(youngest).astype(np.int64)
    high = np.ceil(oldest).astype(np.int64)
    # Every bin from low to high - 1 gets one whole bin width, and then the first and the last
    # give back the part of them that the age did not cross.
    steps = np.zeros(int(high.max(initial=0)) + 1)
    np.add.at(steps, low, 1.0)
    np.add.at(steps, high, -1.0)
    bins = np.cumsum(steps)[:-1]
    np.subtract.at(bins, low, youngest - low)
    np.subtract.at(bins, high - 1, high - oldest)
    return bins * CURVE_BIN_MS, (int(receive_ns[0]) - from_ns) / _NS_PER_MS


def fairness(rates_per_s: Sequence[float]) -> dict[str, float | None]:
    """Return the Jain index and the lowest over the mean of the robots' command rates.

    Both are None for fewer than two robots, and where no robot received a command.
    """
    # ASSUMPTION: A12. A robot is starved when its command rate falls away from the others',
    # which the ratio of the lowest rate to the mean shows; the Jain index,
    # (sum x)^2 / (n * sum x^2), is reported beside it. The longest command gap is the other
    # half of the definition.
    total = math.fsum(rates_per_s)
    if len(rates_per_s) < 2 or total == 0:
        return {"jain": None, "min_over_mean": None}
    squares = math.fsum(rate * rate for rate in rates_per_s)
    return {
        "jain": total * total / (len(rates_per_s) * squares),
        "min_over_mean": min(rates_per_s) * len(rates_per_s) / total,
    }


def _dispatch_ns(commands: Rows) -> IntArray:
    """Return when the batch of each command was dispatched.

    A batch's dispatch time is its result time minus the stages between the two: the
    dispatch round trip and the worker's stages, which each of its commands carries.
    """
    after_dispatch_us = commands["stage_us"][:, _AFTER_DISPATCH].astype(np.int64).sum(axis=1)
    dispatch: IntArray = commands["result_ns"] - _NS_PER_US * after_dispatch_us
    return dispatch


def dispatch_times_ns(commands: Rows) -> IntArray:
    """Return when each batch behind the commands of one model was dispatched, ascending."""
    # Every command of a batch carries the batch's result time, so one row per value is one
    # row per batch.
    _, first_rows = np.unique(commands["result_ns"], return_index=True)
    return np.sort(_dispatch_ns(commands[first_rows]))


def measured_cycle_ns(
    dispatch_ns: IntArray, window_end_ns: int, robot_count: int, batch_cap: int
) -> tuple[int, int]:
    """Return the cycle that a run's own gap bound is computed with, and its longest interval.

    The intervals are those between consecutive dispatches up to the window's end, with the
    interval from the last dispatch to the window's end. A robot waits at most
    K = ceil(robot_count / batch_cap) + 1 consecutive cycles, so the cycle is the mean of the
    K consecutive intervals with the largest sum, rounded up to a whole nanosecond: one slow
    batch counts once, not K times. With fewer than K intervals it is their sum over K.
    """
    times_ns = np.append(dispatch_ns[dispatch_ns <= window_end_ns], window_end_ns)
    intervals_ns = np.diff(times_ns)
    cycles = math.ceil(robot_count / batch_cap) + 1
    if intervals_ns.size < cycles:
        longest_sum_ns = int(intervals_ns.sum())
    else:
        sums_ns = np.cumsum(np.insert(intervals_ns, 0, 0))
        longest_sum_ns = int((sums_ns[cycles:] - sums_ns[:-cycles]).max())
    return -(-longest_sum_ns // cycles), int(intervals_ns.max(initial=0))


def module_files(root: Path, start: str) -> dict[str, bytes]:
    """Return the file at start and every module under root that it imports, transitively.

    The keys are paths relative to root, with "/" between their parts. The imports are read
    from the import statements of each file, wherever they stand in it. Importing a module
    runs the __init__.py of every package above it, so those files count too.
    """
    files: dict[str, bytes] = {}
    pending = [start]
    while pending:
        path = pending.pop()
        if path in files:
            continue
        files[path] = (root / path).read_bytes()
        package = path.split("/")[:-1]
        for name in _imported_names(ast.parse(files[path]), package):
            parts = name.split(".")
            for depth in range(1, len(parts) + 1):
                module = "/".join(parts[:depth])
                pending.extend(
                    candidate
                    for candidate in (f"{module}.py", f"{module}/__init__.py")
                    if (root / candidate).is_file()
                )
    return files


def _imported_names(tree: ast.AST, package: Sequence[str]) -> Iterator[str]:
    """Yield the dotted name of everything a module imports; package is the module's own."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom):
            # One dot is the module's own package, and each further dot one package up.
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            yield module
            # A name imported from a package may be one of its modules.
            for alias in node.names:
                yield f"{module}.{alias.name}"


@cache
def summarizer_digest() -> str:
    """Return the SHA-256 over this module and every module of the repository it imports.

    A summary records it, so a summary written by other code can be told from a current one.
    """
    digest = hashlib.sha256()
    for path, content in sorted(module_files(REPOSITORY_ROOT, _SUMMARIZER).items()):
        # Each part goes in behind its length, so two different sets of files never give the
        # same stream of bytes.
        for part in (path.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


@dataclass(frozen=True)
class RobotSamples:
    """The samples file of one robot."""

    commands: Rows
    ticks: Rows
    # Rows that found their array full.
    dropped_ticks: int
    dropped_commands: int


@dataclass(frozen=True)
class Run:
    """The files of one finished serving run."""

    spec: RunSpec
    run_id: str
    fleet: FleetConfig
    gated: bool
    events: tuple[Event, ...]
    scrapes: tuple[Scrape, ...]
    # The samples of every robot that ran to the end, by robot id. A robot that the scenario
    # kills writes no samples: it is left out of every statistic computed from samples, and
    # its server-side counts come from the scrapes.
    samples: Mapping[str, RobotSamples]


@dataclass(frozen=True)
class _Reading:
    """A scrape that succeeded, with its time on the gateway's clock."""

    ts_ns: int
    # The /metrics object, whose shape is the contract of DESIGN.md 10.
    metrics: Any
    scrape: Scrape


@dataclass(frozen=True)
class _Timeline:
    """A run with its time zero and its successful scrapes in time order."""

    run: Run
    t0_ns: int
    readings: tuple[_Reading, ...]
    # The scrape one second before t0, and the one after the fleet has left; None for one
    # that failed.
    before_t0: _Reading | None
    final: _Reading | None

    @property
    def scenario(self) -> Scenario:
        return self.run.fleet.scenario

    def at_ns(self, t_s: float) -> int:
        return self.t0_ns + round(t_s * _NS_PER_S)

    def stop_s(self, plan: RobotPlan) -> float:
        return self.scenario.duration_s if plan.stop_s is None else plan.stop_s


@dataclass(frozen=True)
class _Span:
    """A window on the shared clock, with the two scrapes its counter differences are taken at.

    The scrapes are the successful ones with the smallest time at or after the window's start
    and the largest at or before its end, and before the end where a robot or a worker is
    killed there. They are None when the window does not hold two.
    """

    window: Window
    start_ns: int
    end_ns: int
    readings: tuple[_Reading, _Reading] | None

    @property
    def length_s(self) -> float:
        return self.window.end_s - self.window.start_s

    @property
    def span_s(self) -> float | None:
        """Return the time between the two scrapes, which every counter rate is taken over."""
        if self.readings is None:
            return None
        earlier, later = self.readings
        return (later.ts_ns - earlier.ts_ns) / _NS_PER_S


@dataclass(frozen=True)
class _RobotWindow:
    """One robot's commands of one model in one window, and what is computed per robot."""

    robot_id: str
    # Whether the robot runs for the whole window.
    present: bool
    from_ns: int
    until_ns: int
    rows: Rows
    commands_per_s: float
    applied_per_s: float
    gaps: CommandGaps
    peak_ages_ns: IntArray
    coverage: float


def _timeline(run: Run) -> _Timeline:
    if not run.events or run.events[0].t0_mono_ns is None:
        raise SummaryError(f"the first line of {EVENTS_FILE} is not the t0 event")
    readings = sorted(
        (
            _Reading(scrape.metrics["ts_mono_ns"], scrape.metrics, scrape)
            for scrape in run.scrapes
            if scrape.metrics is not None
        ),
        key=lambda reading: reading.ts_ns,
    )
    return _Timeline(
        run,
        run.events[0].t0_mono_ns,
        tuple(readings),
        before_t0=next((reading for reading in readings if reading.scrape.k == -1), None),
        final=next((reading for reading in readings if reading.scrape.k is None), None),
    )


def _span(timeline: _Timeline, window: Window) -> _Span:
    start_ns, end_ns = timeline.at_ns(window.start_s), timeline.at_ns(window.end_s)
    # A scrape taken at the instant of a kill, of a robot or of a worker, may already show
    # its effect.
    fault = timeline.scenario.fault
    ends_at_kill = any(plan.stop_s == window.end_s for plan in timeline.scenario.robots) or (
        fault is not None and fault.at_s == window.end_s
    )
    inside = [
        reading
        for reading in timeline.readings
        if start_ns <= reading.ts_ns <= end_ns and not (ends_at_kill and reading.ts_ns == end_ns)
    ]
    return _Span(window, start_ns, end_ns, (inside[0], inside[-1]) if len(inside) > 1 else None)


def _is_present(timeline: _Timeline, plan: RobotPlan, window: Window) -> bool:
    return plan.start_s <= window.start_s and timeline.stop_s(plan) >= window.end_s


def _model_rows(commands: Rows, model: str) -> Rows:
    rows: Rows = commands[commands["model"] == MODEL_KEYS.index(model)]
    return rows


def _robot_window(
    timeline: _Timeline, plan: RobotPlan, model: str, span: _Span
) -> _RobotWindow | None:
    """Return the robot's statistics in the window; None for no samples or no time in it."""
    samples = timeline.run.samples.get(plan.robot_id)
    from_ns = timeline.at_ns(max(span.window.start_s, plan.start_s))
    until_ns = timeline.at_ns(min(span.window.end_s, timeline.stop_s(plan)))
    if samples is None or until_ns <= from_ns:
        return None
    commands = _model_rows(samples.commands, model)
    rows = commands[in_window(commands["receive_ns"], span.start_ns, span.end_ns)]
    applied = rows[rows["verdict"] == _APPLIED]
    # Coverage follows the command in effect, which at the window's start may be one from
    # before it, so it takes the applied commands of the whole run.
    every_applied = commands[commands["verdict"] == _APPLIED]
    # W of DESIGN.md 6.3: the limit the robot declared.
    watchdog_ns = round(timeline.run.fleet.watchdog_ms[model] * _NS_PER_MS)
    return _RobotWindow(
        robot_id=plan.robot_id,
        present=_is_present(timeline, plan, span.window),
        from_ns=from_ns,
        until_ns=until_ns,
        rows=rows,
        # ASSUMPTION: A8. Throughput is the commands a robot received per second, of each
        # model, whatever the robot then did with them: the count of its command samples in
        # the window over the window's length. The applied part is reported beside it.
        commands_per_s=rows.size / span.length_s,
        applied_per_s=applied.size / span.length_s,
        gaps=command_gaps(rows["receive_ns"], from_ns, until_ns),
        peak_ages_ns=peak_ages_ns(applied["capture_ns"], applied["receive_ns"], until_ns),
        coverage=watchdog_coverage(
            every_applied["capture_ns"], every_applied["receive_ns"], watchdog_ns, from_ns, until_ns
        ),
    )


def _pooled(parts: Sequence[Rows]) -> Rows:
    pooled: Rows = np.concatenate([np.empty(0, COMMAND_DTYPE), *parts])
    return pooled


def _statistics(values: FloatArray, names: Sequence[str]) -> dict[str, float | None]:
    """Return the named statistics of bench.stats.summarize; None for each without a sample."""
    if values.size == 0:
        return dict.fromkeys(names)
    summary = summarize(values.tolist())
    return {name: summary[name] for name in names}


def _min_mean_max(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "mean": None, "max": None}
    return {"min": min(values), "mean": math.fsum(values) / len(values), "max": max(values)}


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    """Return the quotient; None where a part is not known or the divisor is 0."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def _minus(later: float | None, earlier: float | None) -> float | None:
    """Return the difference; None where a part is not known."""
    if later is None or earlier is None:
        return None
    return later - earlier


def _all_known(values: Sequence[float | None]) -> list[float] | None:
    """Return the values; None where one of them is not known."""
    known = [value for value in values if value is not None]
    return known if len(known) == len(values) else None


def _value(metrics: Any, *path: str) -> Any:
    """Return the value at a key path of a /metrics object; None where a step is null or absent."""
    value = metrics
    for key in path:
        if not isinstance(value, Mapping) or value.get(key) is None:
            return None
        value = value[key]
    return value


def _rise(readings: tuple[_Reading, _Reading], *path: str) -> float | None:
    """Return how far a counter moved between two scrapes; None where either lacks it."""
    earlier, later = (_value(reading.metrics, *path) for reading in readings)
    return _minus(later, earlier)


def _processes(reading: _Reading) -> dict[str, Any]:
    """Return the counters of every process of a scrape, the fleet's and the driver's included."""
    return {
        **reading.metrics["processes"],
        **{name: asdict(stats) for name, stats in reading.scrape.fleet_processes.items()},
    }


def _process_cpu_ms(earlier: Any, later: Any) -> float | None:
    """Return the CPU time a process used between two readings of its counters.

    None when it was not one live process at both: a worker's counters restart with a new
    process, and a process that has exited is absent.
    """
    if earlier is None or later is None or earlier["pid"] != later["pid"]:
        return None
    used: float = (
        later["cpu_user_ms"]
        + later["cpu_system_ms"]
        - earlier["cpu_user_ms"]
        - earlier["cpu_system_ms"]
    )
    return used


def _throughput_block(
    stats: Sequence[_RobotWindow], present: Sequence[_RobotWindow]
) -> dict[str, object]:
    return {
        "commands_received": sum(stat.rows.size for stat in stats),
        "commands_per_s": math.fsum(stat.commands_per_s for stat in stats),
        "commands_per_s_per_robot": _min_mean_max([stat.commands_per_s for stat in present]),
        "applied_per_s_per_robot": _min_mean_max([stat.applied_per_s for stat in present]),
    }


def _freshness_ms(rows: Rows) -> FloatArray:
    """Return the freshness of every command: receive time minus capture time (DESIGN.md 10.4)."""
    freshness: FloatArray = (rows["receive_ns"] - rows["capture_ns"]) / _NS_PER_MS
    return freshness


def _freshness_block(rows: Rows, t0_ns: int) -> dict[str, object]:
    """Return the freshness and verdict statistics of the commands, of any verdict."""
    freshness_ms = _freshness_ms(rows)
    stale = int((rows["verdict"] == _STALE).sum())
    return {
        "freshness_ms": _statistics(freshness_ms, ("p50", "p95", "p99", "max", "mean")),
        # The slope of freshness against receive time: flat when the system keeps up.
        "freshness_trend_ms_per_s": least_squares_slope(
            (rows["receive_ns"] - t0_ns) / _NS_PER_S, freshness_ms
        ),
        "stale_count": stale,
        "stale_ratio": _ratio(stale, rows.size),
        "out_of_order_count": int((rows["verdict"] == _OUT_OF_ORDER).sum()),
    }


def _gap_block(stats: Sequence[_RobotWindow], present: Sequence[_RobotWindow]) -> dict[str, object]:
    closed_ms = np.concatenate([np.empty(0), *(stat.gaps.closed_ns for stat in stats)]) / _NS_PER_MS
    peaks_ms = np.concatenate([np.empty(0), *(stat.peak_ages_ns for stat in stats)]) / _NS_PER_MS
    longest_ns = [stat.gaps.longest_ns for stat in stats]
    first_ns = [stat.gaps.first_ns for stat in present]
    coverage = [stat.coverage for stat in present]
    return {
        "command_gap_ms": {
            **_statistics(closed_ms, ("p50", "p95")),
            # The longest gap also counts the open intervals at both ends of a robot's time.
            "max": max(longest_ns) / _NS_PER_MS if longest_ns else None,
        },
        "first_gap_ms": {"max": max(first_ns) / _NS_PER_MS if first_ns else None},
        "command_age_peak_ms": _statistics(peaks_ms, ("p50", "p95", "max")),
        "watchdog_coverage": {
            "min": min(coverage, default=None),
            "mean": math.fsum(coverage) / len(coverage) if coverage else None,
        },
    }


def _stage_values_ms(rows: Rows) -> dict[str, FloatArray]:
    """Return the ten stage durations of every command in ms, in the order of DESIGN.md 7."""
    server_ms = rows["stage_us"] / _US_PER_MS
    return {
        "fire_lag_ms": (rows["send_ns"] - rows["capture_ns"]) / _NS_PER_MS,
        **{f"{name}_ms": server_ms[:, index] for index, name in enumerate(STAGE_NAMES)},
        "deliver_ms": (rows["receive_ns"] - rows["result_ns"]) / _NS_PER_MS,
    }


def _timed(rows: Rows) -> Rows:
    """Return the commands that have a send time."""
    # A command whose tick was dropped from a full sample array has send time 0, so neither
    # its fire lag nor a sum that needs it is known. validity.samples_dropped counts the ticks.
    timed: Rows = rows[rows["send_ns"] != 0]
    return timed


def _stage_block(rows: Rows) -> dict[str, object]:
    """Return the stage statistics of the commands, and how their means add up."""
    timed = _timed(rows)
    values = _stage_values_ms(timed)
    block: dict[str, object] = {
        "stage": {name: _statistics(stage, _STAGE_STATISTICS) for name, stage in values.items()}
    }
    e2e_ms = stage_sum_ms = residual_ms = None
    if timed.size > 0:
        # Means add and percentiles do not, so the check of the breakdown is on means.
        e2e_ms = float(_freshness_ms(timed).mean())
        stage_sum_ms = math.fsum(float(stage.mean()) for stage in values.values())
        residual_ms = e2e_ms - stage_sum_ms
    block["e2e_ms"] = {"mean": e2e_ms}
    block["stage_sum_ms"] = {"mean": stage_sum_ms}
    block["residual_ms"] = {"mean": residual_ms}
    return block


def _stage_all(stats: Sequence[_RobotWindow], robot_ids: Sequence[str]) -> dict[str, object]:
    """Return each stage over the commands of both models, pooled.

    A frame is offered to both models, so its fire lag and its uplink reach the robot in two
    commands: those two stages are taken once per frame, which a robot, a session and a frame
    number identify.
    """
    parts = [_timed(stat.rows) for stat in stats]
    rows = _pooled(parts)
    frames = np.stack(
        [
            np.repeat(
                np.array([robot_ids.index(stat.robot_id) for stat in stats], dtype=np.int64),
                [part.size for part in parts],
            ),
            rows["session"].astype(np.int64),
            rows["frame_number"].astype(np.int64),
        ],
        axis=1,
    )
    _, once = np.unique(frames, axis=0, return_index=True)
    return {
        name: _statistics(stage[once] if name in _PER_FRAME_STAGES else stage, _STAGE_STATISTICS)
        for name, stage in _stage_values_ms(rows).items()
    }


def _batch_block(readings: tuple[_Reading, _Reading] | None, model: str) -> dict[str, object]:
    """Return the batch statistics of a window, from the differences of its two scrapes.

    The scrapes are a second apart, so that is the resolution of these statistics in time. A
    batch time read from the histogram is the upper edge of its bucket, about 9% wide.
    """
    if readings is None:
        return {
            "batch_size": {"distribution": {}, "mean": None, "share_1": None},
            "batch_ms": {"p50": None, "p95": None},
        }
    earlier, later = (reading.metrics["models"][model] for reading in readings)
    counts = {
        size: count - earlier["batch_size"].get(size, 0)
        for size, count in later["batch_size"].items()
    }
    batches = sum(counts.values())
    batch_ms = difference(
        Histogram.from_dict(later["batch_ms"]), Histogram.from_dict(earlier["batch_ms"])
    )
    quantiles = {"p50": 0.5, "p95": 0.95}
    return {
        "batch_size": {
            "distribution": {size: count for size, count in counts.items() if count},
            "mean": _ratio(sum(int(size) * count for size, count in counts.items()), batches),
            "share_1": _ratio(counts.get("1", 0), batches),
        },
        "batch_ms": {
            name: batch_ms.quantile_ms(q) if batch_ms.count else None
            for name, q in quantiles.items()
        },
    }


def _share_cores(later: Any, model: str) -> tuple[float | None, str | None]:
    """Return the cores the model's worker may use at a scrape, or the reason it is not known."""
    worker = later["workers"][model]
    if worker["pacing_share"] is not None:
        return worker["pacing_share"], None
    threads, limit_cores = worker["threads"], later["cgroup"]["cpu_limit_cores"]
    if threads > 0:
        return (threads if limit_cores is None else min(threads, limit_cores)), None
    # With threads 0 the runtime sizes its pool from the host's cores, so the limit bounds it.
    if limit_cores is None:
        return None, "the runtime chose its thread count and the container has no CPU limit"
    return limit_cores, None


def _capacity_block(
    readings: tuple[_Reading, _Reading] | None, model: str, robots: int, frame_rate_hz: float
) -> dict[str, object]:
    """Return the measured capacity and utilization of a model in a window (DESIGN.md 10.5).

    c is the worker's CPU time per frame served; s is its pacing share at the later scrape
    when it is paced, else the cores it can use; capacity is 1000 * s / c frames per second;
    utilization is the demand of the robots present over it.
    """
    block: dict[str, object] = dict.fromkeys(
        (
            "cpu_ms_per_frame",
            "pacing_share",
            "share_cores",
            "capacity_fps",
            "utilization",
            "capacity_note",
        )
    )
    if readings is None:
        return {**block, "capacity_note": "the window does not hold two successful scrapes"}
    earlier, later = (reading.metrics for reading in readings)
    worker = f"worker_{model}"
    cpu_ms = _process_cpu_ms(earlier["processes"][worker], later["processes"][worker])
    frames = later["models"][model]["frames"] - earlier["models"][model]["frames"]
    share_cores, note = _share_cores(later, model)
    block["pacing_share"] = later["workers"][model]["pacing_share"]
    block["share_cores"] = share_cores
    if cpu_ms is None:
        note = "the worker was not one live process over the window"
    elif frames == 0:
        note = "the worker served no frame in the window"
    elif cpu_ms == 0:
        note = "the worker used no CPU time in the window: a runtime that only waits"
    if note is not None or cpu_ms is None or share_cores is None:
        return {**block, "capacity_note": note}
    capacity_fps = _MS_PER_S * share_cores * frames / cpu_ms
    return {
        **block,
        "cpu_ms_per_frame": cpu_ms / frames,
        "capacity_fps": capacity_fps,
        "utilization": robots * frame_rate_hz / capacity_fps,
    }


def _model_window(
    timeline: _Timeline, model: str, span: _Span, stats: Sequence[_RobotWindow]
) -> dict[str, object]:
    """Return models.<m>.<window>: the statistics of one model over one window."""
    present = [stat for stat in stats if stat.present]
    rows = _pooled([stat.rows for stat in stats])
    # Every robot planned for the whole window, also one that is killed later and leaves no
    # samples: the demand on the model.
    robots = sum(_is_present(timeline, plan, span.window) for plan in timeline.scenario.robots)
    return {
        "robots": robots,
        **_throughput_block(stats, present),
        **_freshness_block(rows, timeline.t0_ns),
        **_gap_block(stats, present),
        "fairness": fairness([stat.commands_per_s for stat in present]),
        **_batch_block(span.readings, model),
        **_capacity_block(span.readings, model, robots, timeline.run.fleet.frame_rate_hz),
        **_stage_block(rows),
    }


def _curves(stats: Sequence[_RobotWindow]) -> dict[str, object]:
    """Return models.<m>.curves: what the two curves of DESIGN.md 5.3 are drawn from.

    freshness_counts holds the commands by freshness, and age_time_ms the time by the age
    of the command in effect, both in bins of CURVE_BIN_MS from 0 and summed over the robots
    present for the whole run; uncovered_ms is their time before a first applied command.
    """
    present = [stat for stat in stats if stat.present]
    freshness_bins = np.floor(
        _freshness_ms(_pooled([stat.rows for stat in present])) / CURVE_BIN_MS
    ).astype(np.int64)
    age_time = np.zeros(0)
    uncovered_ms = 0.0
    for stat in present:
        applied = stat.rows[stat.rows["verdict"] == _APPLIED]
        bins, before_first_ms = age_time_ms(
            applied["capture_ns"], applied["receive_ns"], stat.from_ns, stat.until_ns
        )
        longer, shorter = (age_time, bins) if age_time.size >= bins.size else (bins, age_time)
        age_time = longer.copy()
        age_time[: shorter.size] += shorter
        uncovered_ms += before_first_ms
    return {
        "bin_ms": CURVE_BIN_MS,
        "freshness_counts": np.bincount(freshness_bins).tolist(),
        "age_time_ms": age_time.tolist(),
        "uncovered_ms": uncovered_ms,
    }


def _robot_block(stat: _RobotWindow) -> dict[str, object]:
    """Return robots.<id>.<m>.<window>."""
    return {
        "commands_per_s": stat.commands_per_s,
        "freshness_ms": _statistics(_freshness_ms(stat.rows), ("p50", "p95", "p99", "max")),
        "stale_count": int((stat.rows["verdict"] == _STALE).sum()),
        "command_gap_ms": {"max": stat.gaps.longest_ns / _NS_PER_MS},
        "first_gap_ms": stat.gaps.first_ns / _NS_PER_MS,
        "watchdog_coverage": stat.coverage,
    }


def _survivors_block(stats: Sequence[_RobotWindow]) -> dict[str, object]:
    """Return survivors.<m>.<window>, over the robots that are not killed: those with samples."""
    rates = [stat.commands_per_s for stat in stats]
    freshness_ms = _freshness_ms(_pooled([stat.rows for stat in stats]))
    return {
        "commands_per_s_per_robot": math.fsum(rates) / len(rates) if rates else None,
        "freshness_ms": _statistics(freshness_ms, ("p95",)),
    }


def _cpu_block(span: _Span) -> dict[str, object]:
    """Return cpu.<window>, from the differences of the window's two scrapes."""
    if span.readings is None or span.span_s is None:
        return {
            "utilization": None,
            "cores": {},
            "throttled_ms_per_s": None,
            "throttled_share": None,
            "robots_container_cores": None,
        }
    earlier, later = span.readings
    before, after = _processes(earlier), _processes(later)
    span_ms = span.span_s * _MS_PER_S
    limit_cores = _value(later.metrics, "cgroup", "cpu_limit_cores")
    usage_us = _rise(span.readings, "cgroup", "cpu_usage_us")
    throttled_us = _rise(span.readings, "cgroup", "throttled_us")
    robots_usage_us = _minus(
        *(
            None if cgroup is None else cgroup.cpu_usage_us
            for cgroup in (later.scrape.fleet_cgroup, earlier.scrape.fleet_cgroup)
        )
    )
    return {
        # ASSUMPTION: A15. CPU utilization is the CPU time the container's cgroup used, as a
        # share of what its CPU limit allows over the same time, with the cores each process
        # used beside it.
        "utilization": _ratio(
            usage_us, None if limit_cores is None else span.span_s * limit_cores * _US_PER_S
        ),
        "cores": {
            name: _ratio(_process_cpu_ms(before.get(name), after.get(name)), span_ms)
            for name in sorted({*before, *after})
        },
        "throttled_ms_per_s": _ratio(
            None if throttled_us is None else throttled_us / _US_PER_MS, span.span_s
        ),
        "throttled_share": _ratio(
            _rise(span.readings, "cgroup", "nr_throttled"),
            _rise(span.readings, "cgroup", "nr_periods"),
        ),
        # The robots' own container, which only the split topology has.
        "robots_container_cores": _ratio(robots_usage_us, span.span_s * _US_PER_S),
    }


def _gateway_block(span: _Span) -> dict[str, object]:
    """Return gateway.<window>, from the differences of the window's two scrapes."""
    if span.readings is None or span.span_s is None:
        return {
            "cpu_cores": None,
            "cpu_ms_per_frame": None,
            "loop_lag_ms": {"p99": None},
            "bytes_in_per_s": None,
        }
    earlier, later = (reading.metrics for reading in span.readings)
    cpu_ms = _process_cpu_ms(earlier["processes"]["gateway"], later["processes"]["gateway"])
    # A robot that connected after the earlier scrape has no entry there: it had sent nothing.
    frames_in = sum(
        robot["frames_in"] - (_value(earlier, "robots", robot_id, "frames_in") or 0)
        for robot_id, robot in later["robots"].items()
    )
    lag_ms = difference(
        Histogram.from_dict(later["loop"]["lag_ms"]), Histogram.from_dict(earlier["loop"]["lag_ms"])
    )
    return {
        "cpu_cores": _ratio(cpu_ms, span.span_s * _MS_PER_S),
        "cpu_ms_per_frame": _ratio(cpu_ms, frames_in),
        "loop_lag_ms": {"p99": lag_ms.quantile_ms(0.99) if lag_ms.count else None},
        "bytes_in_per_s": _ratio(_rise(span.readings, "connections", "bytes_in"), span.span_s),
    }


def _anon_sock_mb(metrics: Any) -> float | None:
    """Return the anonymous plus socket memory of the container; None where the kernel hides one."""
    anon_bytes, sock_bytes = (
        metrics["cgroup"][name] for name in ("memory_anon_bytes", "memory_sock_bytes")
    )
    if anon_bytes is None or sock_bytes is None:
        return None
    megabytes: float = (anon_bytes + sock_bytes) / _BYTES_PER_MB
    return megabytes


def _planned_connections(timeline: _Timeline, ts_ns: int) -> int:
    return sum(
        timeline.at_ns(plan.start_s) <= ts_ns < timeline.at_ns(timeline.stop_s(plan))
        for plan in timeline.scenario.robots
    )


def _growth(
    timeline: _Timeline, settled: Sequence[_Reading]
) -> tuple[dict[str, object], list[str]]:
    """Return the memory growth between the first and the last settled scrape, and the notes."""
    growth: dict[str, object] = dict.fromkeys(
        ("anon_sock_mb_start", "anon_sock_mb_end", "slope_mb_per_min", "fds_delta", "tasks_delta")
    )
    memory_mb = _all_known([_anon_sock_mb(reading.metrics) for reading in settled])
    if len(settled) < 2 or memory_mb is None:
        return growth, []
    first, last = settled[0], settled[-1]
    minutes = np.array([reading.ts_ns for reading in settled]) / _NS_PER_S / _S_PER_MIN
    fds_last, fds_first = (
        _all_known(
            [_value(reading.metrics, "processes", name, "fds") for name in _GATEWAY_PROCESSES]
        )
        for reading in (last, first)
    )
    growth.update(
        anon_sock_mb_start=memory_mb[0],
        anon_sock_mb_end=memory_mb[-1],
        # ASSUMPTION: A13. A leak is judged by the slope of anonymous plus socket memory once
        # the connections have settled: page cache from writing files is not a leak, and the
        # 5-minute soak gives a slow one time to show.
        slope_mb_per_min=least_squares_slope(minutes, np.array(memory_mb, dtype=np.float64)),
        fds_delta=_minus(*(None if fds is None else sum(fds) for fds in (fds_last, fds_first))),
        tasks_delta=last.metrics["loop"]["tasks"] - first.metrics["loop"]["tasks"],
    )
    differing = [
        f"{reading.metrics['connections']['current']} connections at "
        f"{(reading.ts_ns - timeline.t0_ns) / _NS_PER_S:.1f} s, "
        f"{_planned_connections(timeline, reading.ts_ns)} planned"
        for reading in (first, last)
        if reading.metrics["connections"]["current"]
        != _planned_connections(timeline, reading.ts_ns)
    ]
    notes = [f"memory scrapes with other connections than planned: {'; '.join(differing)}"]
    return growth, notes if differing else []


def _memory_block(timeline: _Timeline, span: _Span) -> tuple[dict[str, object], list[str]]:
    """Return memory, from the scrapes of the whole run taken before any robot stops.

    The growth is measured from the first scrape at or after t0 + MEMORY_SLOPE_SKIP_S. The
    peak is over every scrape from t0, and the sizes per process are those of the last one.
    """
    stops_ns = [
        timeline.at_ns(plan.stop_s) for plan in timeline.scenario.robots if plan.stop_s is not None
    ]
    steady = [
        reading
        for reading in timeline.readings
        if span.start_ns <= reading.ts_ns < min([span.end_ns, *stops_ns])
    ]
    settled_ns = timeline.at_ns(MEMORY_SLOPE_SKIP_S)
    growth, notes = _growth(timeline, [r for r in steady if r.ts_ns >= settled_ns])
    memory_mb = _all_known([_anon_sock_mb(reading.metrics) for reading in steady])
    block: dict[str, object] = {
        **growth,
        "anon_sock_mb_max": max(memory_mb) if memory_mb else None,
        "memory_peak_bytes": None,
        "sock_mb": None,
        "rss_mb": {},
    }
    if steady:
        last = steady[-1]
        block["memory_peak_bytes"] = last.metrics["cgroup"]["memory_peak_bytes"]
        block["sock_mb"] = _ratio(last.metrics["cgroup"]["memory_sock_bytes"], _BYTES_PER_MB)
        block["rss_mb"] = {
            name: process["rss_bytes"] / _BYTES_PER_MB
            for name, process in sorted(_processes(last).items())
            if process is not None
        }
    return block, notes


def _connections_block(timeline: _Timeline) -> dict[str, object] | None:
    """Return connections: the counters of the final scrape minus those before t0.

    The scrape before t0 is the baseline, so a connection of an earlier run of the same
    gateway would not count. None when either scrape failed.
    """
    if timeline.before_t0 is None or timeline.final is None:
        return None
    earlier, later = (
        reading.metrics["connections"] for reading in (timeline.before_t0, timeline.final)
    )
    closed, rejected = (
        {reason: count - earlier[kind][reason] for reason, count in later[kind].items()}
        for kind in ("closed", "rejected")
    )
    return {
        "accepted": later["accepted"] - earlier["accepted"],
        "closed": closed,
        "rejected": sum(rejected.values()),
        "rejected_by_reason": rejected,
    }


def _series(timeline: _Timeline, span: _Span) -> dict[str, object]:
    """Return series: what the plots of one run draw over time."""
    scenario, samples = timeline.scenario, timeline.run.samples
    bins = math.ceil(scenario.duration_s / SERIES_BIN_S)
    commands_per_bin: dict[str, dict[str, list[int]]] = {}
    freshness: dict[str, dict[str, list[float]]] = {}
    for model in timeline.run.fleet.models:
        pooled: list[Rows] = []
        for robot_id, robot in samples.items():
            rows = _model_rows(robot.commands, model)
            rows = rows[in_window(rows["receive_ns"], span.start_ns, span.end_ns)]
            second = (rows["receive_ns"] - timeline.t0_ns) // (SERIES_BIN_S * _NS_PER_S)
            commands_per_bin.setdefault(robot_id, {})[model] = np.bincount(
                second, minlength=bins
            ).tolist()
            pooled.append(rows)
        # Only the one-robot run draws every command: its points are few enough to keep.
        if scenario.name == "steady":
            rows = _pooled(pooled)
            freshness[model] = {
                "t_s": ((rows["receive_ns"] - timeline.t0_ns) / _NS_PER_S).tolist(),
                "freshness_ms": _freshness_ms(rows).tolist(),
            }
    inside = [
        reading for reading in timeline.readings if span.start_ns <= reading.ts_ns < span.end_ns
    ]
    return {
        "bin_s": SERIES_BIN_S,
        "commands_per_bin": commands_per_bin,
        "freshness": freshness,
        "memory": {
            "t_s": [(reading.ts_ns - timeline.t0_ns) / _NS_PER_S for reading in inside],
            "anon_sock_mb": [_anon_sock_mb(reading.metrics) for reading in inside],
        },
    }


def _validity(
    timeline: _Timeline, span: _Span, throttled_ms_per_s: object, notes: Sequence[str]
) -> dict[str, object]:
    """Return validity: whether the run's numbers describe the service and not the harness."""
    run = timeline.run
    ticks = np.concatenate(
        [np.empty(0, TICK_DTYPE), *(robot.ticks for robot in run.samples.values())]
    )
    ticks = ticks[in_window(ticks["capture_ns"], span.start_ns, span.end_ns)]
    sent = ticks[ticks["outcome"] == TickOutcome.SENT]
    fire_lag_p99_ms = _statistics((sent["send_ns"] - sent["capture_ns"]) / _NS_PER_MS, ("p99",))[
        "p99"
    ]
    outcomes = {
        outcome.name.lower(): int((ticks["outcome"] == outcome).sum()) for outcome in TickOutcome
    }
    limit_ms = SIMULATOR_FIRE_LAG_SHARE * _MS_PER_S / run.fleet.frame_rate_hz
    first, final = (timeline.readings[0] if timeline.readings else None), timeline.final
    return {
        "fire_lag_ms": {"p99": fire_lag_p99_ms},
        "ticks": outcomes,
        # A simulator that sends late, or passes a tick, shapes the freshness it reports.
        "simulator_limited": (
            fire_lag_p99_ms is None or fire_lag_p99_ms > limit_ms or outcomes["missed_late"] > 0
        ),
        "throttled_ms_per_s": throttled_ms_per_s,
        # The identities of DESIGN.md 10.3, in the final scrape; None when it failed.
        "conservation_ok": None if final is None else not conservation_violations(final.metrics),
        # ASSUMPTION: A20. The driver and the gateway read one monotonic clock, so the
        # gateway's reading of a scrape lies between the driver's two readings around it.
        "shared_clock_ok": first is not None
        and first.scrape.driver_before_ns <= first.ts_ns <= first.scrape.driver_after_ns,
        "failed_scrapes": sum(not scrape.ok for scrape in run.scrapes),
        "samples_dropped": {
            "ticks": sum(robot.dropped_ticks for robot in run.samples.values()),
            "commands": sum(robot.dropped_commands for robot in run.samples.values()),
        },
        "final_scrape_settled": next(
            (event.settled for event in run.events if event.name is EventName.FINAL_POLL), None
        ),
        "notes": list(notes),
    }


def _fault_event(timeline: _Timeline, fault: WorkerKill) -> Event:
    """Return the run's fault event; SummaryError unless it has one, of the killed model."""
    events = [event for event in timeline.run.events if event.name is EventName.FAULT]
    if len(events) != 1:
        raise SummaryError(
            f"a run that kills the {fault.model} worker needs one {EventName.FAULT} event in "
            f"{EVENTS_FILE}, and has {len(events)}"
        )
    (event,) = events
    if event.model != fault.model:
        raise SummaryError(
            f"the {EventName.FAULT} event names the {event.model} worker, and the scenario kills "
            f"the {fault.model} worker"
        )
    return event


def _restarts(timeline: _Timeline) -> dict[str, int] | None:
    """Return each worker's restarts from the scrape before t0 to the final one.

    None when either scrape failed.
    """
    if timeline.before_t0 is None or timeline.final is None:
        return None
    earlier, later = (
        reading.metrics["workers"] for reading in (timeline.before_t0, timeline.final)
    )
    return {
        model: worker["restarts"] - earlier[model]["restarts"] for model, worker in later.items()
    }


def _outage_ms(timeline: _Timeline, span: _Span, model: str, kill_ns: int) -> float | None:
    """Return the time from the kill to the first command of the model served after it.

    The commands are every robot's of the model in the window. Only a batch dispatched after
    the kill shows the worker serving again: one dispatched before it came from the killed
    process, also where its command arrived after the kill. None when no command came from a
    batch dispatched after the kill.
    """
    commands = _pooled(
        [_model_rows(robot.commands, model) for robot in timeline.run.samples.values()]
    )
    commands = commands[in_window(commands["receive_ns"], span.start_ns, span.end_ns)]
    served = commands[_dispatch_ns(commands) > kill_ns]
    if served.size == 0:
        return None
    return (int(served["receive_ns"].min()) - kill_ns) / _NS_PER_MS


def _fault_block(timeline: _Timeline, span: _Span) -> dict[str, object] | None:
    """Return fault: the worker the run killed, and how the service came back; None for none.

    The kill is at t0 plus the fault event's time, read in the gateway's container right
    after the kill returned. late_ms is how far that is past the planned time, and restarts
    each worker's restarts over the run.
    """
    fault = timeline.scenario.fault
    if fault is None:
        return None
    event = _fault_event(timeline, fault)
    return {
        "model": event.model,
        "pid": event.pid,
        "planned_s": fault.at_s,
        "kill_s": event.t_s,
        "late_ms": (event.t_s - fault.at_s) * _MS_PER_S,
        "outage_ms": _outage_ms(timeline, span, fault.model, timeline.at_ns(event.t_s)),
        "restarts": _restarts(timeline),
    }


def _check(name: str, passed: bool | None, observed: object) -> dict[str, object]:
    return {"name": name, "passed": passed, "observed": observed}


def _worker_restarted(timeline: _Timeline, fault: Mapping[str, Any]) -> dict[str, object]:
    """Return worker_restarted: the killed worker alone restarted, once, and serves again.

    The final scrape shows it ready under another pid than the killed one, and a command of
    its model came from a batch dispatched after the kill.
    """
    model, restarts = fault["model"], fault["restarts"]
    worker = _value(None if timeline.final is None else timeline.final.metrics, "workers", model)
    state, pid = (None, None) if worker is None else (worker["state"], worker["pid"])
    # The killed worker is among the names, so counts that lack it never match.
    once = restarts is not None and restarts == {
        name: 1 if name == model else 0 for name in (*restarts, model)
    }
    return _check(
        "worker_restarted",
        once
        and state == WorkerState.READY
        and pid not in (None, fault["pid"])
        and fault["outage_ms"] is not None,
        {
            "restarts": restarts,
            "state": state,
            "pid": pid,
            "killed_pid": fault["pid"],
            "outage_ms": fault["outage_ms"],
        },
    )


def _common_checks(
    timeline: _Timeline, validity: Mapping[str, Any], fault: Mapping[str, Any] | None
) -> list[dict[str, object]]:
    """Return the checks of every serving scenario.

    A run that kills a worker checks that the worker restarted, where every other run checks
    that no worker did.
    """
    final = None if timeline.final is None else timeline.final.metrics
    restarts = None if final is None else {m: w["restarts"] for m, w in final["workers"].items()}
    internal_errors = _value(final, "connections", "closed", "internal_error")
    return [
        _check(
            "conservation",
            validity["conservation_ok"] is True,
            None if final is None else conservation_violations(final),
        ),
        _check("no_worker_restart", restarts is not None and not any(restarts.values()), restarts)
        if fault is None
        else _worker_restarted(timeline, fault),
        _check("no_internal_error", internal_errors == 0, internal_errors),
        _check(
            "simulator_valid",
            not validity["simulator_limited"],
            {
                "fire_lag_ms_p99": validity["fire_lag_ms"]["p99"],
                "missed_late": validity["ticks"]["missed_late"],
            },
        ),
    ]


def _trend_checks(timeline: _Timeline, models: Mapping[str, Any]) -> list[dict[str, object]]:
    """Return no_upward_trend.<m>: freshness drifts by less than one frame period over the run."""
    fleet = timeline.run.fleet
    period_ms = _MS_PER_S / fleet.frame_rate_hz
    checks = []
    for model in fleet.models:
        trend = models[model][ALL_WINDOW]["freshness_trend_ms_per_s"]
        passed = trend is not None and abs(trend) * timeline.scenario.duration_s < period_ms
        checks.append(_check(f"no_upward_trend.{model}", passed, trend))
    return checks


def _join_checks(timeline: _Timeline) -> list[dict[str, object]]:
    """Return b_served_after_join: every robot that joins late got commands of every model."""
    run = timeline.run
    received = {
        plan.robot_id: {
            model: int(
                (
                    _model_rows(run.samples[plan.robot_id].commands, model)["receive_ns"]
                    >= timeline.at_ns(plan.start_s)
                ).sum()
            )
            for model in run.fleet.models
        }
        for plan in timeline.scenario.robots
        if plan.start_s > 0 and plan.robot_id in run.samples
    }
    passed = bool(received) and all(all(counts.values()) for counts in received.values())
    return [_check("b_served_after_join", passed, received)]


def _killed_released(timeline: _Timeline, robot_id: str) -> dict[str, object]:
    """Return killed_robot_released: the final scrape shows the robot gone and its slot empty."""
    robot = None if timeline.final is None else timeline.final.metrics["robots"].get(robot_id)
    if robot is None:
        return _check("killed_robot_released", False, None)
    pending = {model: counters["frames_pending"] for model, counters in robot["models"].items()}
    return _check(
        "killed_robot_released",
        not robot["connected"] and not any(pending.values()),
        {"connected": robot["connected"], "frames_pending": pending},
    )


def _closed_abnormal(timeline: _Timeline, robot_id: str) -> dict[str, object]:
    """Return closed_abnormal: the kill ended one session as abnormal, and nothing else closed.

    The counters are compared between the last scrape that shows the robot connected and
    the first after it that does not. detect_ms is the time from the kill to the gateway's
    close of the session: the session's start, from a scrape taken while it was open, plus
    its length once closed, minus the time of the kill event.
    """
    name = "closed_abnormal"
    connected = [
        _value(reading.metrics, "robots", robot_id, "connected") for reading in timeline.readings
    ]
    if True not in connected:
        return _check(name, False, None)
    last_open = len(connected) - 1 - connected[::-1].index(True)
    if False not in connected[last_open:]:
        return _check(name, False, None)
    before = timeline.readings[last_open]
    after = timeline.readings[connected.index(False, last_open)]
    rises = {
        reason: count - before.metrics["connections"]["closed"][reason]
        for reason, count in after.metrics["connections"]["closed"].items()
    }
    kill_s = next(
        (
            event.t_s
            for event in timeline.run.events
            if event.name is EventName.ROBOT_KILL and event.robot_id == robot_id
        ),
        None,
    )
    session_start_ns = before.ts_ns - before.metrics["robots"][robot_id]["connected_s"] * _NS_PER_S
    closed_ns = session_start_ns + after.metrics["robots"][robot_id]["connected_s"] * _NS_PER_S
    return _check(
        name,
        rises["abnormal"] == 1 and sum(rises.values()) == 1,
        {
            "closed": {reason: rise for reason, rise in rises.items() if rise},
            "detect_ms": None
            if kill_s is None
            else (closed_ns - timeline.at_ns(kill_s)) / _NS_PER_MS,
        },
    )


def _kill_checks(timeline: _Timeline) -> list[dict[str, object]]:
    # The disconnect scenario kills one robot, and both checks are about that robot.
    (killed,) = (plan.robot_id for plan in timeline.scenario.robots if plan.stop_s is not None)
    return [_killed_released(timeline, killed), _closed_abnormal(timeline, killed)]


def _all_accepted(timeline: _Timeline, connections: Mapping[str, Any] | None) -> dict[str, object]:
    """Return all_accepted: every robot's connection was accepted and none was refused."""
    if connections is None:
        return _check("all_accepted", False, None)
    observed = {"accepted": connections["accepted"], "rejected": connections["rejected"]}
    robots = len(timeline.scenario.robots)
    return _check("all_accepted", observed == {"accepted": robots, "rejected": 0}, observed)


def _frames_skipped_rose(timeline: _Timeline, model: str) -> bool:
    """Whether the skip rule passed over a frame of the model between t0 and the final scrape."""
    if timeline.before_t0 is None or timeline.final is None:
        return False
    skipped = [
        sum(
            robot["models"][model]["frames_skipped"]
            for robot in reading.metrics["robots"].values()
            if model in robot["models"]
        )
        for reading in (timeline.before_t0, timeline.final)
    ]
    return bool(skipped[1] > skipped[0])


def _calibrated_cycle_ns(
    timeline: _Timeline, span: _Span, model: str, batch_cap: int, dispatch_rt_p99_ms: float | None
) -> int | None:
    """Return the longest cycle that the worker's warm-up calibration predicts; None without one.

    Each dispatch leg is taken as the window's whole dispatch round trip at p99, and the
    shares are every pacing share that a scrape inside the window shows.
    """
    worker = span.readings[1].metrics["workers"][model] if span.readings else None
    if worker is None or worker["t_b_ms"] is None or dispatch_rt_p99_ms is None:
        return None
    table = ServiceTable(
        *(
            {int(size): round(ms * _NS_PER_MS) for size, ms in worker[key].items()}
            for key in ("t_b_ms", "t_b_cpu_ms")
        )
    )
    leg_ns = round(dispatch_rt_p99_ms * _NS_PER_MS)
    shares = [
        reading.metrics["workers"][model]["pacing_share"]
        for reading in timeline.readings
        if span.start_ns <= reading.ts_ns <= span.end_ns
        and reading.metrics["workers"][model]["pacing_share"] is not None
    ]
    return longest_cycle_ns(table, Overheads(0, leg_ns, leg_ns, 0), batch_cap, shares)


def _gaps_within_bound(
    timeline: _Timeline,
    span: _Span,
    model: str,
    stats: Sequence[_RobotWindow],
    stage: Mapping[str, Any],
) -> dict[str, object]:
    """Return gaps_within_bound.<m>: no robot waited longer than the scheduler's bound.

    The bound is bench.capacity_model.command_gap_bound_ns with the cycle measured from the
    run's own dispatches: the pacing hold follows each batch's measured CPU time, so the
    calibrated cycle, a warm-up median, is observed beside it for comparison only. A robot's
    closed gaps and its closing open interval are held to the bound, and its wait for a first
    command to the bound plus the p99 of the uplink and of the delivery, which that wait
    also holds.
    """
    name = f"gaps_within_bound.{model}"
    run = timeline.run
    batch_cap = _value(
        span.readings[1].metrics if span.readings else None, "workers", model, "batch_cap"
    )
    if batch_cap is None:
        return _check(name, False, None)
    commands = _pooled([_model_rows(robot.commands, model) for robot in run.samples.values()])
    cycle_ns, longest_interval_ns = measured_cycle_ns(
        dispatch_times_ns(commands), span.end_ns, len(timeline.scenario.robots), batch_cap
    )
    config = run.spec.server_config
    bound_ns = command_gap_bound_ns(
        len(timeline.scenario.robots),
        batch_cap,
        round(_NS_PER_S / run.fleet.frame_rate_hz),
        cycle_ns,
        # The function's docstring speaks only of the skip rule being on or off. The interval
        # is passed only when the rule is on and passed over a frame in this run: a run in
        # which no frame was skipped has the schedule of the rule off.
        round(config["skip_max_unserved_ms"] * _NS_PER_MS)
        if config["skip_rule"] and _frames_skipped_rose(timeline, model)
        else None,
    )
    allowance_ms = sum(stage[key]["p99"] or 0.0 for key in ("transport_up_ms", "deliver_ms"))
    later_gaps_ns = [
        max(stat.gaps.last_ns, int(stat.gaps.closed_ns.max(initial=0))) for stat in stats
    ]
    first_gaps_ns = [stat.gaps.first_ns for stat in stats]
    calibrated_ns = _calibrated_cycle_ns(
        timeline, span, model, batch_cap, stage["dispatch_rt_ms"]["p99"]
    )
    return _check(
        name,
        bool(stats)
        and max(later_gaps_ns) <= bound_ns
        and max(first_gaps_ns) <= bound_ns + allowance_ms * _NS_PER_MS,
        {
            "largest_gap_ms": max(later_gaps_ns) / _NS_PER_MS if stats else None,
            "largest_first_gap_ms": max(first_gaps_ns) / _NS_PER_MS if stats else None,
            "bound_ms": bound_ns / _NS_PER_MS,
            "cycle_ms": cycle_ns / _NS_PER_MS,
            "longest_interval_ms": longest_interval_ns / _NS_PER_MS,
            "calibrated_cycle_ms": None if calibrated_ns is None else calibrated_ns / _NS_PER_MS,
        },
    )


def _checks(
    timeline: _Timeline,
    span: _Span,
    stats: Mapping[str, Sequence[_RobotWindow]],
    summary: Mapping[str, Any],
) -> list[dict[str, object]]:
    """Return checks: each with a criterion that is a design constant or a derived bound.

    None is a target: the design document states no numeric target for a demo. A check
    whose passed is None has no criterion at all and only records what was observed.
    """
    name = timeline.scenario.name
    checks = _common_checks(timeline, summary["validity"], summary["fault"])
    if name == "steady":
        checks += _trend_checks(timeline, summary["models"])
    if name == "late_joiner":
        checks += _join_checks(timeline)
    if name == "disconnect":
        checks += _kill_checks(timeline)
    if name in ("massive", "soak"):
        checks.append(_all_accepted(timeline, summary["connections"]))
        checks += [
            _gaps_within_bound(
                timeline, span, model, stats[model], summary["models"][model][ALL_WINDOW]["stage"]
            )
            for model in timeline.run.fleet.models
        ]
    if name == "soak":
        growth = ("slope_mb_per_min", "fds_delta", "tasks_delta")
        checks.append(
            _check("memory_growth", None, {key: summary["memory"][key] for key in growth})
        )
    return checks


def summary_of(run: Run) -> dict[str, Any]:
    """Compute the summary of a serving run from its files, without its summarizer_digest."""
    timeline = _timeline(run)
    scenario, models = timeline.scenario, run.fleet.models
    spans = {window.name: _span(timeline, window) for window in scenario.windows}
    whole = spans[ALL_WINDOW]
    stats = {
        model: {
            name: [
                stat
                for plan in scenario.robots
                if (stat := _robot_window(timeline, plan, model, span)) is not None
            ]
            for name, span in spans.items()
        }
        for model in models
    }
    cpu = {name: _cpu_block(span) for name, span in spans.items()}
    memory, notes = _memory_block(timeline, whole)
    killed = any(plan.stop_s is not None for plan in scenario.robots)
    robot_ids = [plan.robot_id for plan in scenario.robots]
    summary: dict[str, Any] = {
        "run": {
            "scenario": run.spec.scenario_key,
            "run_id": run.run_id,
            "repeat": run.spec.repeat,
            "topology": run.spec.topology.value,
            "duration_s": scenario.duration_s,
            "gated": run.gated,
        },
        "validity": _validity(timeline, whole, cpu[ALL_WINDOW]["throttled_ms_per_s"], notes),
        "windows": {
            name: {
                "start_s": span.window.start_s,
                "end_s": span.window.end_s,
                "span_s": span.span_s,
            }
            for name, span in spans.items()
        },
        "models": {
            model: {
                **{
                    name: _model_window(timeline, model, span, stats[model][name])
                    for name, span in spans.items()
                },
                "curves": _curves(stats[model][ALL_WINDOW]),
            }
            for model in models
        },
        "stage_all": _stage_all(
            [stat for model in models for stat in stats[model][ALL_WINDOW]], robot_ids
        ),
        "gateway": {name: _gateway_block(span) for name, span in spans.items()},
        "robots": {
            robot_id: {
                model: {
                    name: _robot_block(stat)
                    for name in spans
                    for stat in stats[model][name]
                    if stat.robot_id == robot_id
                }
                for model in models
            }
            for robot_id in run.samples
        },
        "survivors": {
            model: {name: _survivors_block(stats[model][name]) for name in spans}
            for model in models
            if killed
        },
        "fault": _fault_block(timeline, whole),
        "cpu": cpu,
        "memory": memory,
        "connections": _connections_block(timeline),
        "series": _series(timeline, whole),
    }
    summary["checks"] = _checks(
        timeline, whole, {model: stats[model][ALL_WINDOW] for model in models}, summary
    )
    return summary


def _dropped(meta: Mapping[str, object], key: str, path: Path) -> int:
    count = meta.get(key)
    if isinstance(count, bool) or not isinstance(count, int):
        raise SummaryError(f"{path}: the meta has no count {key}")
    return count


def read_run(run_dir: Path) -> Run:
    """Read the files of a finished serving run."""
    config = RunConfig.from_json((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    if config.spec is None or config.run_id is None or config.fleet is None:
        raise SummaryError(f"{run_dir} is not a serving run of a plan")
    with gzip.open(run_dir / SCRAPES_FILE, "rt", encoding="utf-8") as file:
        scrapes = tuple(Scrape.from_json(line) for line in file)
    samples: dict[str, RobotSamples] = {}
    for plan in config.fleet.scenario.robots:
        if plan.stop_s is None:
            path = run_dir / SAMPLES_DIRECTORY / f"robot_{plan.robot_id}.npz"
            # DONE does not cover the samples, so nothing has checked these files yet.
            try:
                commands, ticks, meta = load_samples(path)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                raise SummaryError(f"{path} is not a samples file: {error!r}") from error
            samples[plan.robot_id] = RobotSamples(
                commands,
                ticks,
                _dropped(meta, "dropped_ticks", path),
                _dropped(meta, "dropped_commands", path),
            )
    return Run(
        spec=config.spec,
        run_id=config.run_id,
        fleet=config.fleet,
        gated=Gate.from_json((run_dir / GATE_FILE).read_text(encoding="utf-8")).gated,
        events=tuple(
            Event.from_json(line)
            for line in (run_dir / EVENTS_FILE).read_text(encoding="utf-8").splitlines()
        ),
        scrapes=scrapes,
        samples=samples,
    )


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Return the summary of the finished serving run in run_dir."""
    return {"summarizer_digest": summarizer_digest(), **summary_of(read_run(run_dir))}


def _mean_cost(entry: Any) -> dict[str, float]:
    """Return the mean wall and CPU time of one entry of a model benchmark report."""
    return {"wall_ms": entry["wall_ms"]["mean"], "cpu_ms": entry["mean_cpu_ms"]}


def _batch_table(report: Any) -> tuple[str, str, str, dict[str, dict[str, float]]]:
    """Return the model, the thread count, the graph and the table of a batch report."""
    arguments = report["args"]
    table = {str(entry["batch_size"]): _mean_cost(entry) for entry in report["results"]}
    return arguments["model"], f"t{arguments['threads']}", arguments["graph"], table


def _stage_costs(report: Any) -> dict[str, dict[str, float]]:
    return {stage: _mean_cost(entry) for stage, entry in report["results"].items()}


def _dispatch_costs(report: Any) -> dict[str, dict[str, float]]:
    return {
        str(entry["batch_size"]): {
            "wall_ms_mean": entry["wall_ms"]["mean"],
            "wall_ms_p50": entry["wall_ms"]["p50"],
            "gateway_cpu_us": entry["gateway_cpu_us_per_round_trip"],
            "worker_cpu_us": entry["worker_cpu_us_per_round_trip"],
        }
        for entry in report["results"]
    }


def _websocket_costs(report: Any) -> dict[str, float]:
    results = report["results"]
    return {
        "round_trip_ms_mean": results["sender"]["round_trip_ms"]["mean"],
        "receiver_cpu_us_per_message": results["cpu_us_per_message"]["receiver"],
        "sender_cpu_us_per_message": results["cpu_us_per_message"]["sender"],
    }


def _read_report[T](path: Path, read: Callable[[Any], T]) -> T:
    """Read what a summary takes from one report; SummaryError names a file that lacks it."""
    try:
        return read(json.loads(path.read_bytes()))
    except (ValueError, KeyError, TypeError, AttributeError) as error:
        raise SummaryError(f"{path} is not a report of its kind: {error!r}") from error


def summarize_model_bench(run_dir: Path) -> dict[str, Any]:
    """Return the summary of the model benchmark run in run_dir, from its reports alone.

    tables holds, per model, thread count ("t1"), graph ("published" or "batched") and batch
    size, the mean wall and CPU time of one batch; stages the mean wall and CPU time of one
    call of each stage outside inference; dispatch, per batch size, the round trip of one
    dispatch and its CPU time on each side; websocket the round trip and the CPU time per
    message of the 50-robot run without a model.
    """
    by_kind: dict[str, list[Path]] = {}
    for path in sorted((run_dir / REPORTS_DIRECTORY).glob("*.json")):
        by_kind.setdefault(_read_report(path, lambda report: str(report["kind"])), []).append(path)
    tables: dict[str, dict[str, dict[str, dict[str, dict[str, float]]]]] = {}
    for path in by_kind.pop("model_bench.batch", []):
        model, threads, graph, table = _read_report(path, _batch_table)
        graphs = tables.setdefault(model, {}).setdefault(threads, {})
        if graph in graphs:
            raise SummaryError(f"{path} repeats the table of another batch report")
        graphs[graph] = table
    summary: dict[str, Any] = {"summarizer_digest": summarizer_digest(), "tables": tables}
    single: Mapping[str, tuple[str, Callable[[Any], object]]] = {
        "stages": ("model_bench.stages", _stage_costs),
        "dispatch": ("transport_bench.dispatch", _dispatch_costs),
        "websocket": ("transport_bench.websocket", _websocket_costs),
    }
    for key, (kind, read) in single.items():
        paths = by_kind.pop(kind, [])
        if len(paths) != 1:
            raise SummaryError(f"{run_dir} needs one {kind} report, and has {len(paths)}")
        summary[key] = _read_report(paths[0], read)
    if by_kind:
        raise SummaryError(f"{run_dir} has reports of other kinds: {', '.join(sorted(by_kind))}")
    return summary


def run_files(run_dir: Path) -> dict[str, bytes]:
    """Return the files that DONE's hashes are checked against: every file but the samples.

    The keys are paths relative to run_dir, with "/" between their parts.
    """
    files: dict[str, bytes] = {}
    for path in sorted(run_dir.rglob("*")):
        relative = path.relative_to(run_dir).as_posix()
        if path.is_file() and not relative.startswith(f"{SAMPLES_DIRECTORY}/"):
            files[relative] = path.read_bytes()
    return files


def _is_model_bench(run_dir: Path) -> bool:
    config = RunConfig.from_json((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    return config.spec is not None and config.spec.scenario_key == MODEL_BENCH_KEY


def _replace_file(path: Path, content: bytes) -> None:
    """Put content at path in one step, so a reader never sees a part of it."""
    # The temporary file stands beside the run directory, not in it: DONE covers every file
    # of the directory, so a temporary file left there by an interrupted write would make a
    # finished run invalid.
    temporary = path.parent.with_name(f"{path.parent.name}.{path.name}.tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def write_summary(run_dir: Path) -> None:
    """Write summary.json of the finished run in run_dir, then its hash into DONE.

    Only summary_sha256 of DONE changes. A write interrupted between the two files leaves
    a summary that does not match DONE, which is then not current and is written again.
    """
    done = Done.from_json((run_dir / DONE_FILE).read_text(encoding="utf-8"))
    summary = summarize_model_bench(run_dir) if _is_model_bench(run_dir) else summarize_run(run_dir)
    # One line: the curves and the series are long lists of numbers.
    content = (
        json.dumps(summary, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    _replace_file(run_dir / SUMMARY_FILE, content)
    done = replace(done, summary_sha256=hashlib.sha256(content).hexdigest())
    _replace_file(run_dir / DONE_FILE, done.to_json().encode())


def summary_is_current(run_dir: Path, done: Done) -> bool:
    """Whether the run's summary exists, matches DONE's hash and was written by this code."""
    path = run_dir / SUMMARY_FILE
    if not path.is_file():
        return False
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != done.summary_sha256:
        return False
    recorded: object = json.loads(content)["summarizer_digest"]
    return recorded == summarizer_digest()


def refresh_summary(run_dir: Path) -> str:
    """Write the run's summary where it is finished and not current; return what was done."""
    if _ATTEMPT.search(run_dir.name):
        return "left: an attempt"
    try:
        done = Done.from_json((run_dir / DONE_FILE).read_text(encoding="utf-8"))
    except OSError, RunFileError:
        return "left: no valid DONE"
    if not done.valid_for(run_files(run_dir)):
        return "left: no valid DONE"
    if summary_is_current(run_dir, done):
        return "current"
    # A model benchmark's summary comes from its reports, which are kept with the results;
    # a serving run's needs the samples, which are not.
    if not _is_model_bench(run_dir) and not any(
        (run_dir / SAMPLES_DIRECTORY).glob(_SAMPLES_PATTERN)
    ):
        return "left: its summary is not current and it has no samples to write it from"
    write_summary(run_dir)
    return "summary written"


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.summarize",
        description="Write the summary of every finished run whose summary is not current.",
    )
    parser.add_argument(
        "--results", type=Path, required=True, metavar="DIR", help="the results root"
    )
    results: Path = parser.parse_args(argv).results
    failed: list[str] = []
    # A run directory is one that holds config.json, which every run writes first.
    for config_path in sorted((results / RAW_DIRECTORY).rglob(CONFIG_FILE)):
        run_dir = config_path.parent
        name = run_dir.relative_to(results).as_posix()
        try:
            print(f"{name}: {refresh_summary(run_dir)}")
        except Exception as error:
            # The boundary of one run: a summary that cannot be computed is reported, the
            # run keeps its DONE, and the other runs are still tried.
            print(f"{name}: failed")
            failed.append(f"{run_dir.name}: {error!r}")
    for failure in failed:
        print(f"bench.summarize: no summary for {failure}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
