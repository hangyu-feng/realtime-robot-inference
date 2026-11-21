"""The fact keys of the design document: where each value comes from and how runs combine.

A fact is one measured value of DESIGN.md or the README (DESIGN.md 11.4). The registry holds
one entry per key: the scenario key and topology whose counted runs it reads, where its value
lies in a run's summary.json or which rule computes it, its unit, how the values of several
runs combine, and whether it is a whole number. The rules that need more than one path of
one summary are here as functions of plain values: the inflection point, the useful robot
count, the settle and recovery times, and the runs a model's breakdown and its gap bound are
taken from. The placeholder that names a key in a template is defined beside the keys.
Nothing here reads a file.
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from bench.candidates import CANDIDATES
from bench.runplan import MODEL_BENCH_KEY, SWEEP_ROBOT_COUNTS
from bench.scenarios import massive, sweep, worker_kill
from cloud.config import Topology
from common.commands import MODEL_KEYS
from common.protocol import STAGE_NAMES

# A measured value in a template: a fact key in double braces. Where the placeholder is the
# whole of a code span, the two backticks belong to it, so the rendered value is plain text.
# A backtick on one side only is the edge of a longer code span and stays.
KEY_PATTERN = r"[a-z][a-z0-9_.]*"
PLACEHOLDER = re.compile(rf"(?P<tick>`)?\{{\{{(?P<key>{KEY_PATTERN})\}}\}}(?(tick)`)")

# Where utilization and the peak command age are followed over robot counts: the sweep's
# runs and the massive run, as (scenario key, robot count) in order of robot count.
SCALE_POINTS: tuple[tuple[str, int], ...] = (
    *((f"sweep_n{count}", len(sweep(count).robots)) for count in SWEEP_ROBOT_COUNTS),
    ("massive", len(massive().robots)),
)
# The thread counts and the batch sizes of the model benchmark's tables (DESIGN.md 11.4).
BENCH_THREADS = (1, 2)
BENCH_BATCH_SIZES = (1, 2, 4, 8)
# The stages of DESIGN.md 7, in the order of its table: the robot's, the server's, the robot's.
BREAKDOWN_STAGES = ("fire_lag", *STAGE_NAMES, "deliver")

# Two robots have settled at equal shares from the first 1 s bin after which their counts
# stay this close. A count is whole commands, so a difference of one command is counting;
# above that, a tenth of the bin's mean count separates a robot that is still catching up
# from two robots served in turn.
SETTLE_SHARE = 0.1
# The survivors have recovered in the first 1 s bin whose mean count reaches this share of
# the rate the capacity model predicts once the dead robot's share is free, N / (N - 1)
# times their rate before the kill. The missing tenth allows for whole commands in one bin.
RECOVERY_SHARE = 0.9

_SPLIT_PREFIX = "split."
_PREDICT_PREFIX = "predict."
_RATIO = "ratio"
_COMMANDS = "commands"
_COMMAND_RATE = "commands/s"
_MS = "ms"
_MB = "MB"
# The unit of each metric of a summary, by its name without the statistic that may end it.
_UNITS: Mapping[str, str] = MappingProxyType(
    {
        "freshness_ms": _MS,
        "freshness_trend_ms_per_s": "ms/s",
        "command_gap_ms": _MS,
        "command_age_peak_ms": _MS,
        "first_gap_ms": _MS,
        "fire_lag_ms": _MS,
        "commands_per_s": _COMMAND_RATE,
        "commands_per_s_per_robot": _COMMAND_RATE,
        "commands_received": _COMMANDS,
        "stale_count": _COMMANDS,
        "out_of_order_count": _COMMANDS,
        "stale_ratio": _RATIO,
        "watchdog_coverage": _RATIO,
        "fairness.jain": _RATIO,
        "fairness.min_over_mean": _RATIO,
        "batch_size": "frames",
        "batch_size.share_1": _RATIO,
        "utilization": _RATIO,
        "throttled_ms_per_s": "ms/s",
    }
)
_STATISTICS = frozenset(("min", "mean", "max", "p50", "p95", "p99"))
_INTEGER_METRICS = frozenset(("commands_received", "stale_count", "out_of_order_count"))
_BINS_UNIT = "s (1 s bins)"


class Source(StrEnum):
    """Where a key's value comes from."""

    SUMMARY = "summary"  # each counted run's summary.json, at the entry's path
    BREAKDOWN = "breakdown"  # the summary of the one run a model's breakdown is taken from
    GAP_BOUND = "gap_bound"  # the summary of the run with the model's longest gap
    SETTLE = "settle"  # per run, from the two robots' commands in 1 s bins
    RECOVERY = "recovery"  # per run, from the survivors' commands in 1 s bins
    # Per run, at the entry's path, only where the run's check of the abnormal close passed.
    DETECT = "detect"
    INFLECTION = "inflection"  # across the scale points; the path names the result
    USEFUL_ROBOTS = "useful_robots"  # across the scale points
    BATCH_PLAN = "batch_plan"  # from the model benchmark's table; the path names the result
    PREDICTION = "prediction"  # predictions.json, under the entry's key
    LEDGER = "ledger"  # the candidate's line of the ledger; the path names the field
    ENVIRONMENT = "environment"  # the counted runs' env.json; the path names the field


class Combine(StrEnum):
    """How the values of a key's counted runs become its value."""

    MEDIAN = "median"
    LARGEST = "largest"
    SMALLEST = "smallest"


@dataclass(frozen=True)
class FactKey:
    """One key of facts.json."""

    key: str
    source: Source
    # None for a key that reads several scenario keys or none.
    scenario_key: str | None
    # None for a key that belongs to no topology: a prediction, the ledger, the environment.
    topology: Topology | None
    # The model of a key whose rule needs it; None otherwise.
    model: str | None
    path: tuple[str, ...]
    unit: str
    combine: Combine
    integer: bool


def placeholder_keys(text: str) -> list[str]:
    """Return the keys of a template's placeholders, each once, in order of appearance."""
    return list(dict.fromkeys(match["key"] for match in PLACEHOLDER.finditer(text)))


def metric_unit(path: str) -> str:
    """Return the unit of the value at a dotted path of summary.json.

    KeyError for a path that ends in no metric of the unit table.
    """
    parts = path.split(".")
    for start in range(len(parts)):
        name = _metric_name(".".join(parts[start:]))
        if name in _UNITS:
            return _UNITS[name]
    raise KeyError(f"no unit is declared for {path}")


def _metric_name(metric: str) -> str:
    """Return a metric without its statistic, unless the unit table has it with one."""
    name, _, last = metric.rpartition(".")
    return name if last in _STATISTICS and metric not in _UNITS else metric


def combine_rule(key: str) -> Combine:
    """Return how a key's run values combine.

    A statistic that is itself an extreme takes the extreme over the runs, so a reported
    extreme is a value some run observed and the extreme of every counted run.
    """
    last = key.rpartition(".")[2]
    if last == "max":
        return Combine.LARGEST
    if last == "min" or key.endswith("fairness.min_over_mean"):
        return Combine.SMALLEST
    return Combine.MEDIAN


def stored(value: float, integer: bool) -> int | float:
    """Return a number as its key stores it.

    A whole-number key keeps an integral value as an int, so the median of 50 and 50 is 50
    and renders as a count; any other value is a float.
    """
    return int(value) if integer and float(value).is_integer() else float(value)


def combined(values: Sequence[float], entry: FactKey) -> int | float:
    """Combine a key's run values by its rule. ValueError for no values."""
    if not values:
        raise ValueError(f"{entry.key}: no run values to combine")
    if entry.combine is Combine.LARGEST:
        return stored(max(values), entry.integer)
    if entry.combine is Combine.SMALLEST:
        return stored(min(values), entry.integer)
    return stored(statistics.median(values), entry.integer)


def breakdown_run(e2e_mean_ms: Mapping[str, float]) -> str:
    """Return the run a model's breakdown is taken from, given each run's end-to-end mean.

    It is the run whose mean is the median: the lower middle for an even count, and the
    smaller run id where two runs share that mean. Medians taken stage by stage would come
    from different runs, and the stage means would not add up to their sum.
    """
    if not e2e_mean_ms:
        raise ValueError("no run to take the breakdown from")
    ordered = sorted(e2e_mean_ms.values())
    middle_ms = ordered[(len(ordered) - 1) // 2]
    return min(run_id for run_id, mean_ms in e2e_mean_ms.items() if mean_ms == middle_ms)


def gap_bound_run(gap_max_ms: Mapping[str, float]) -> str:
    """Return the run a model's gap bound is taken from, given each run's longest gap.

    It is the run with the longest gap, the smaller run id where two runs share it. Shown
    beside the longest gap, the bound is then the one that gap was checked against; a bound
    combined over the runs could be another run's, tighter than this run's, and the pair
    would read as a violation that no run had.
    """
    if not gap_max_ms:
        raise ValueError("no run to take the gap bound from")
    longest_ms = max(gap_max_ms.values())
    return min(run_id for run_id, gap_ms in gap_max_ms.items() if gap_ms == longest_ms)


@dataclass(frozen=True)
class ScalePoint:
    """One robot count of the scale points, with the medians over its counted runs."""

    robot_count: int
    utilization: float | None
    capacity_fps: float | None


@dataclass(frozen=True)
class Inflection:
    """Where a model's utilization crosses 1 over the scale points (DESIGN.md 6.5)."""

    capacity_fps: float
    inflection_n: float
    inflection_between: str
    # The robot count of the lowest saturated point. No point above it changes the result.
    robot_count: int


def inflection(points: Sequence[ScalePoint], frame_rate_hz: float) -> Inflection | None:
    """Return the inflection of a model from its scale points, in order of robot count.

    None when a point has no utilization or capacity (a point without a counted run, or a
    runtime with no CPU cost), or when no point is saturated. The capacity is the one at the
    lowest saturated point; nothing is interpolated between points or extrapolated past them.
    """
    measured = [
        (point.robot_count, point.utilization, point.capacity_fps)
        for point in points
        if point.utilization is not None and point.capacity_fps is not None
    ]
    if len(measured) != len(points):
        return None
    for index, (robot_count, utilization, capacity_fps) in enumerate(measured):
        if utilization >= 1:
            between = (
                "below 1 robot"
                if index == 0
                else f"between {measured[index - 1][0]} and {robot_count} robots"
            )
            return Inflection(capacity_fps, capacity_fps / frame_rate_hz, between, robot_count)
    return None


def useful_robots(
    peak_age_p95_ms: Sequence[tuple[int, float | None]], watchdog_ms: float
) -> int | None:
    """Return the largest robot count measured within the watchdog limit.

    peak_age_p95_ms holds, in order of robot count, each scale point's count and the median
    over its runs of the p95 peak command age. The result is the largest count at which that
    median, and the median at every lower count, is at or below the limit; 0 when one robot
    is already above it; None when a point has no median. It is a count that was measured,
    not a capacity.
    """
    measured = [
        (robot_count, peak_ms) for robot_count, peak_ms in peak_age_p95_ms if peak_ms is not None
    ]
    if len(measured) != len(peak_age_p95_ms):
        return None
    within = 0
    for robot_count, peak_ms in measured:
        if peak_ms > watchdog_ms:
            break
        within = robot_count
    return within


def settle_s(first_counts: Sequence[int], second_counts: Sequence[int]) -> int | None:
    """Return the seconds after a join from which two robots' command counts stay equal.

    Each list holds one robot's commands in the complete 1 s bins from the join to the end
    of the run. The result is the end of the first bin from which every bin has the two
    counts within the larger of one command and SETTLE_SHARE of their mean; None when the
    last bin does not. Its smallest value is 1, the resolution of the bins.
    """
    settled_from: int | None = None
    pairs = list(zip(first_counts, second_counts, strict=True))
    for index in reversed(range(len(pairs))):
        first, second = pairs[index]
        if abs(first - second) > max(1.0, SETTLE_SHARE * (first + second) / 2):
            break
        settled_from = index
    return None if settled_from is None else settled_from + 1


def recovery_s(
    survivor_mean_counts: Sequence[float], before_per_s: float, robot_count: int
) -> int | None:
    """Return the seconds after a kill at which the survivors' rate has risen.

    survivor_mean_counts holds the survivors' mean command count in each complete 1 s bin
    from the kill to the end of the run, before_per_s their rate per robot before the kill,
    and robot_count the robots before it. The result is the end of the first bin at or above
    RECOVERY_SHARE of robot_count / (robot_count - 1) times the rate before; None when no
    bin is.
    """
    recovered = RECOVERY_SHARE * robot_count / (robot_count - 1) * before_per_s
    for index, count in enumerate(survivor_mean_counts):
        if count >= recovered:
            return index + 1
    return None


def _entry(
    key: str,
    source: Source,
    scenario_key: str | None,
    topology: Topology | None,
    path: tuple[str, ...],
    unit: str,
    *,
    model: str | None = None,
    integer: bool = False,
) -> FactKey:
    return FactKey(
        key, source, scenario_key, topology, model, path, unit, combine_rule(key), integer
    )


def _statistics(metric: str, *names: str) -> tuple[str, ...]:
    return tuple(f"{metric}.{name}" for name in names)


def _metrics(
    key_prefix: str,
    scenario_key: str,
    topology: Topology,
    summary_path: tuple[str, ...],
    metrics: Sequence[str],
) -> Iterator[FactKey]:
    """Yield one key per metric: the key and the path both end in the metric's own parts."""
    for metric in metrics:
        yield _entry(
            f"{key_prefix}.{metric}",
            Source.SUMMARY,
            scenario_key,
            topology,
            (*summary_path, *metric.split(".")),
            _UNITS[_metric_name(metric)],
            integer=metric in _INTEGER_METRICS,
        )


_RATE_PER_ROBOT = _statistics("commands_per_s_per_robot", "min", "mean", "max")
_FAIRNESS = _statistics("fairness", "jain", "min_over_mean")
_STEADY_METRICS = (
    *_statistics("freshness_ms", "p50", "p95", "p99", "max"),
    "commands_per_s",
    "commands_received",
    "stale_count",
    "stale_ratio",
    "out_of_order_count",
    "freshness_trend_ms_per_s",
    "command_age_peak_ms.p95",
)
_SWEEP_METRICS = (
    *_RATE_PER_ROBOT,
    *_statistics("freshness_ms", "p50", "p95"),
    *_statistics("batch_size", "mean", "share_1"),
    "utilization",
)
_MASSIVE_METRICS = (
    *_RATE_PER_ROBOT,
    "commands_received",
    *_FAIRNESS,
    "command_gap_ms.max",
    *_statistics("freshness_ms", "p50", "p95", "p99", "max"),
    "stale_count",
    "stale_ratio",
    "out_of_order_count",
    *_statistics("watchdog_coverage", "min", "mean"),
)
_CPU_METRICS = ("utilization", "throttled_ms_per_s")
_WINDOWS = ("before", "after")


def _steady(prefix: str, topology: Topology) -> Iterator[FactKey]:
    for model in MODEL_KEYS:
        yield from _metrics(
            f"{prefix}steady.{model}", "steady", topology, ("models", model, "all"), _STEADY_METRICS
        )


def _late_joiner(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "late_joiner"
    for model in MODEL_KEYS:
        # Robot b joins at the end of the window before, so it has keys after the join only.
        for window, robot_id in (("before", "a"), ("after", "a"), ("after", "b")):
            yield from _metrics(
                f"{prefix}late_joiner.{window}.{robot_id}.{model}",
                scenario_key,
                topology,
                ("robots", robot_id, model, window),
                ("commands_per_s", "freshness_ms.p95"),
            )
        yield _entry(
            f"{prefix}late_joiner.{model}.settle_s",
            Source.SETTLE,
            scenario_key,
            topology,
            (),
            _BINS_UNIT,
            model=model,
            integer=True,
        )
        # The time from b's planned join to its first command, which the 1 s bins cannot show.
        yield _entry(
            f"{prefix}late_joiner.b.{model}.first_command_ms",
            Source.SUMMARY,
            scenario_key,
            topology,
            ("robots", "b", model, "after", "first_gap_ms"),
            _MS,
        )


def _sweep(prefix: str, topology: Topology) -> Iterator[FactKey]:
    for count in SWEEP_ROBOT_COUNTS:
        scenario_key = f"sweep_n{count}"
        for model in MODEL_KEYS:
            yield from _metrics(
                f"{prefix}sweep.n{count}.{model}",
                scenario_key,
                topology,
                ("models", model, "all"),
                _SWEEP_METRICS,
            )
        yield from _metrics(
            f"{prefix}sweep.n{count}.cpu", scenario_key, topology, ("cpu", "all"), _CPU_METRICS
        )
    for model in MODEL_KEYS:
        for result, unit in (
            ("capacity_fps", "frames/s"),
            ("inflection_n", "robots"),
            ("inflection_between", ""),
        ):
            yield _entry(
                f"{prefix}sweep.{model}.{result}",
                Source.INFLECTION,
                None,
                topology,
                (result,),
                unit,
                model=model,
            )


def _breakdown(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "sweep_n5"
    for model in MODEL_KEYS:
        window = ("models", model, "all")
        key_prefix = f"{prefix}sweep.n5.{model}"
        for stage in BREAKDOWN_STAGES:
            for statistic in ("mean", "p50", "p95"):
                yield _entry(
                    f"{key_prefix}.stage.{stage}_ms.{statistic}",
                    Source.BREAKDOWN,
                    scenario_key,
                    topology,
                    (*window, "stage", f"{stage}_ms", statistic),
                    _MS,
                    model=model,
                )
        for total in ("stage_sum_ms", "e2e_ms", "residual_ms"):
            yield _entry(
                f"{key_prefix}.{total}.mean",
                Source.BREAKDOWN,
                scenario_key,
                topology,
                (*window, total, "mean"),
                _MS,
                model=model,
            )


def _disconnect(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "disconnect"
    for model in MODEL_KEYS:
        for window in _WINDOWS:
            yield from _metrics(
                f"{prefix}disconnect.{window}.survivors.{model}",
                scenario_key,
                topology,
                ("survivors", model, window),
                ("commands_per_s_per_robot", "freshness_ms.p95"),
            )
        yield _entry(
            f"{prefix}disconnect.{model}.recovery_s",
            Source.RECOVERY,
            scenario_key,
            topology,
            (),
            _BINS_UNIT,
            model=model,
            integer=True,
        )
    # The summarizer reads the time from the kill to the close off the scrapes, beside its
    # check of the close: aggregation reads no scrape.
    yield _entry(
        f"{prefix}disconnect.detect_ms",
        Source.DETECT,
        scenario_key,
        topology,
        ("checks", "closed_abnormal", "observed", "detect_ms"),
        _MS,
    )


def _worker_kill(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "worker_kill"
    fault = worker_kill().fault
    if fault is None:
        raise ValueError(f"the {scenario_key} scenario kills no worker")
    # From the kill to the first command of the killed worker's model served after it.
    yield _entry(
        f"{prefix}{scenario_key}.{fault.model}.outage_ms",
        Source.SUMMARY,
        scenario_key,
        topology,
        ("fault", "outage_ms"),
        _MS,
    )
    for model in MODEL_KEYS:
        for window in _WINDOWS:
            yield from _metrics(
                f"{prefix}{scenario_key}.{window}.{model}",
                scenario_key,
                topology,
                ("models", model, window),
                ("commands_per_s_per_robot.mean",),
            )


def _massive(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "massive"

    def summary(key: str, path: tuple[str, ...], unit: str, integer: bool = False) -> FactKey:
        return _entry(
            f"{prefix}massive.{key}",
            Source.SUMMARY,
            scenario_key,
            topology,
            path,
            unit,
            integer=integer,
        )

    for model in MODEL_KEYS:
        yield from _metrics(
            f"{prefix}massive.{model}",
            scenario_key,
            topology,
            ("models", model, "all"),
            _MASSIVE_METRICS,
        )
        # The bound of a run's own check of its gaps (DESIGN.md 6.3), from the run with the
        # longest gap.
        yield _entry(
            f"{prefix}massive.{model}.command_gap_bound_ms",
            Source.GAP_BOUND,
            scenario_key,
            topology,
            ("checks", f"gaps_within_bound.{model}", "observed", "bound_ms"),
            _MS,
            model=model,
        )
        yield _entry(
            f"{prefix}massive.{model}.useful_robots",
            Source.USEFUL_ROBOTS,
            None,
            topology,
            (),
            "robots",
            model=model,
            integer=True,
        )
    yield summary("stage.transport_up_ms.p50", ("stage_all", "transport_up_ms", "p50"), _MS)
    yield summary("stage.transport_up_ms.p99", ("stage_all", "transport_up_ms", "p99"), _MS)
    yield summary("stage.dispatch_rt_ms.p50", ("stage_all", "dispatch_rt_ms", "p50"), _MS)
    yield from _metrics(
        f"{prefix}massive.cpu", scenario_key, topology, ("cpu", "all"), _CPU_METRICS
    )
    yield summary("bytes_in_per_s", ("gateway", "all", "bytes_in_per_s"), "bytes/s")
    yield summary("connections.accepted", ("connections", "accepted"), "connections", True)
    yield summary("connections.rejected", ("connections", "rejected"), "connections", True)
    yield summary("gateway.cpu_cores", ("gateway", "all", "cpu_cores"), "cores")
    yield summary("gateway.cpu_ms_per_frame", ("gateway", "all", "cpu_ms_per_frame"), _MS)
    yield summary("gateway.loop_lag_ms.p99", ("gateway", "all", "loop_lag_ms", "p99"), _MS)
    yield summary("fleet.cpu_cores", ("cpu", "all", "cores", "fleet"), "cores")
    yield summary("fleet.fire_lag_ms.p99", ("validity", "fire_lag_ms", "p99"), _MS)
    for process in ("gateway", "fleet", "worker_cls", "worker_depth"):
        yield summary(f"{process}.rss_mb", ("memory", "rss_mb", process), _MB)
    for name in ("anon_sock_mb_start", "anon_sock_mb_end", "sock_mb"):
        yield summary(f"memory.{name}", ("memory", name), _MB)


def _soak(prefix: str, topology: Topology) -> Iterator[FactKey]:
    scenario_key = "soak"

    def memory(key: str, name: str, unit: str, integer: bool = False) -> FactKey:
        return _entry(
            f"{prefix}soak.{key}",
            Source.SUMMARY,
            scenario_key,
            topology,
            ("memory", name),
            unit,
            integer=integer,
        )

    yield memory("memory.slope_mb_per_min", "slope_mb_per_min", "MB/min")
    yield memory("fds.delta", "fds_delta", "file descriptors", True)
    yield memory("tasks.delta", "tasks_delta", "tasks", True)
    for model in MODEL_KEYS:
        # Over five minutes a robot receives ten times the commands of the massive run, so
        # these figures are not dominated by counting.
        yield from _metrics(
            f"{prefix}soak.{model}",
            scenario_key,
            topology,
            ("models", model, "all"),
            (*_FAIRNESS, *_RATE_PER_ROBOT),
        )


def bench_graph(model: str, threads: int, batch_size: int) -> str:
    """Return the graph whose benchmark gives a model's batch time at a thread count and size.

    The published depth graph takes one frame, and serves a cap of 1 at one thread; every
    other depth entry comes from the patched copy, which serves the larger caps
    (DESIGN.md 4.6). The classifier's published file takes any batch size.
    """
    if model == "depth" and (threads, batch_size) != (1, 1):
        return "batched"
    return "published"


def _model_bench() -> Iterator[FactKey]:
    def summary(key: str, path: tuple[str, ...]) -> FactKey:
        return _entry(
            f"{MODEL_BENCH_KEY}.{key}", Source.SUMMARY, MODEL_BENCH_KEY, Topology.SINGLE, path, _MS
        )

    for model in MODEL_KEYS:
        for threads in BENCH_THREADS:
            for size in BENCH_BATCH_SIZES:
                table = ("tables", model, f"t{threads}", bench_graph(model, threads, size))
                for field in ("wall_ms", "cpu_ms"):
                    yield summary(f"{model}.t{threads}.b{size}.{field}", (*table, str(size), field))
        yield _entry(
            f"{MODEL_BENCH_KEY}.{model}.batch_cap",
            Source.BATCH_PLAN,
            MODEL_BENCH_KEY,
            Topology.SINGLE,
            ("batch_cap",),
            "frames",
            model=model,
            integer=True,
        )
        yield summary(f"{model}.preprocess_ms", ("stages", f"preprocess_{model}_full", "wall_ms"))
        yield summary(f"{model}.postprocess_ms", ("stages", f"postprocess_{model}", "wall_ms"))
    yield _entry(
        f"{MODEL_BENCH_KEY}.depth.margin_ms",
        Source.BATCH_PLAN,
        MODEL_BENCH_KEY,
        Topology.SINGLE,
        ("margin_ms",),
        _MS,
        model="depth",
    )
    yield summary("decode.full_ms", ("stages", "decode_full", "wall_ms"))
    yield summary("decode.half_ms", ("stages", "decode_half", "wall_ms"))
    yield summary("dispatch_rt_ms", ("dispatch", "1", "wall_ms_p50"))


def _predictions() -> Iterator[FactKey]:
    """Yield exactly the keys that bench.capacity_model writes under facts."""

    def prediction(key: str) -> FactKey:
        return _entry(
            f"{_PREDICT_PREFIX}{key}", Source.PREDICTION, None, None, (), metric_unit(key)
        )

    for model in MODEL_KEYS:
        for statistic in ("p50", "p95", "p99", "max"):
            yield prediction(f"steady.{model}.freshness_ms.{statistic}")
        yield prediction(f"steady.{model}.commands_per_s")
        yield prediction(f"steady.{model}.command_age_peak_ms.p95")
        yield prediction(f"late_joiner.after.a.{model}.commands_per_s")
        for count in SWEEP_ROBOT_COUNTS:
            yield prediction(f"sweep.n{count}.{model}.commands_per_s_per_robot.mean")
        yield prediction(f"massive.{model}.commands_per_s_per_robot.mean")
    for window in _WINDOWS:
        yield prediction(f"disconnect.{window}.survivors.depth.commands_per_s_per_robot")


def _ledger() -> Iterator[FactKey]:
    for candidate in CANDIDATES:
        main_path, _ = candidate.main
        for field, unit in (
            ("outcome", ""),
            ("main_delta", metric_unit(main_path)),
            ("baseline_spread", metric_unit(main_path)),
        ):
            yield _entry(
                f"ledger.{candidate.name}.{field}",
                Source.LEDGER,
                None,
                None,
                (candidate.name, field),
                unit,
            )


def _environment() -> Iterator[FactKey]:
    for field in ("cpu_model", "kernel", "engine"):
        yield _entry(f"env.{field}", Source.ENVIRONMENT, None, None, (field,), "")
    yield _entry("env.repeats", Source.ENVIRONMENT, None, None, ("repeats",), "runs", integer=True)


def _registry() -> dict[str, FactKey]:
    entries: list[FactKey] = []
    # Without a prefix a key means the single topology; split. selects the split one.
    for topology, prefix in ((Topology.SINGLE, ""), (Topology.SPLIT, _SPLIT_PREFIX)):
        for family in (
            _steady,
            _late_joiner,
            _sweep,
            _breakdown,
            _disconnect,
            _worker_kill,
            _massive,
            _soak,
        ):
            entries.extend(family(prefix, topology))
    entries.extend(_model_bench())
    entries.extend(_predictions())
    entries.extend(_ledger())
    entries.extend(_environment())
    registry = {entry.key: entry for entry in entries}
    if len(registry) != len(entries):
        raise ValueError("two entries of the fact registry share a key")
    return registry


REGISTRY: Mapping[str, FactKey] = MappingProxyType(_registry())
