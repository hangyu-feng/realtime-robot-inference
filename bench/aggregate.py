"""Aggregate the counted runs of a results root into the facts and the files beside them.

python -m bench.aggregate build --results DIR --models {real,fake,null} [--out DIR]
    [--digest current|recorded]
python -m bench.aggregate status --results DIR [--templates DIR]

build decides which run directories under raw/ count (DESIGN.md 11.4) and writes, under
--out (default the results root): summary/t_b.json and summary/predictions.json (the
capacity model's input and its predictions, from the model benchmark alone),
summary/facts.json, ledger.jsonl, summary/excluded.json and summary/failed_checks.json. It
reads summaries and never writes one, and never reads samples or scrapes. It exits 0 when
it has written them or there is no run directory, 1 when an input cannot be read or the
summary of a counted run is not current, and 2 when run directories exist and none counts.
With a non-zero exit it has written nothing.

status prints the excluded runs, the failed checks and the templated keys that are null. It
exits 1 when a check failed, or when the facts are of the real model set and a templated key
is null.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from bench.candidates import CANDIDATES, Candidate, outcome, threshold
from bench.capacity_model import QUOTA_CORES, CapacityInputError, parse_input, predict
from bench.facts import (
    BENCH_BATCH_SIZES,
    BENCH_THREADS,
    REGISTRY,
    SCALE_POINTS,
    FactKey,
    ScalePoint,
    Source,
    breakdown_run,
    combined,
    gap_bound_run,
    inflection,
    placeholder_keys,
    recovery_s,
    settle_s,
    stored,
    useful_robots,
)
from bench.render import RenderError, read_templates
from bench.runfiles import (
    ARMS,
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    FACTS_FILE,
    GATE_FILE,
    SUMMARY_FILE,
    Done,
    Fact,
    Facts,
    Gate,
    ModelSet,
    RunConfig,
    RunEnvironment,
    RunFileError,
    RunSpec,
)
from bench.runplan import (
    MODEL_BENCH_KEY,
    code_digest,
    code_paths,
    compare_plan,
    evidence_plan,
    overrides_for,
    run_id,
)
from bench.scenarios import Scenario, massive
from bench.summarize import RAW_DIRECTORY, run_files, summarizer_digest
from cloud.config import OVERRIDES_VARIABLE, ServerConfig, Topology, load_config
from cloud.scheduler import plan_model
from common.commands import MODEL_KEYS
from common.thresholds import DEFAULT_WATCHDOG_MS

COMPARE_DIRECTORY = "compare"
SUMMARY_DIRECTORY = "summary"
T_B_FILE = "t_b.json"
PREDICTIONS_FILE = "predictions.json"
EXCLUDED_FILE = "excluded.json"
FAILED_CHECKS_FILE = "failed_checks.json"
LEDGER_FILE = "ledger.jsonl"
# The outcome of a candidate that lacks a counted run in an arm.
NOT_RUN = "not run"

_REPOSITORY = Path(__file__).resolve().parent.parent
# Listing the files of the working tree is a local command; this only keeps a hung one from
# hanging the build.
_GIT_TIMEOUT_S = 60.0
_ATTEMPT = re.compile(r"\.attempt[0-9]+$")
_NS_PER_MS = 1_000_000
_US_PER_MS = 1_000
# The predictions are made as the capacity model's own command makes them: one seed for
# the robots' phases, and a curve from one robot to the robots of the massive demo.
_PREDICTION_SEED = 0
_PREDICTION_ROBOT_COUNTS = range(1, len(massive().robots) + 1)
_BENCH_GROUP = f"{MODEL_BENCH_KEY}:{Topology.SINGLE}"
_COMMAND_BINS = ("series", "commands_per_bin")
_CLOSED_ABNORMAL_PASSED = ("checks", "closed_abnormal", "passed")


class AggregateError(Exception):
    """A file the aggregation needs is missing or malformed, or holds what it cannot use."""


class DoneState(StrEnum):
    ABSENT = "absent"
    INVALID = "invalid"  # malformed, or its hashes are not those of the run's files
    VALID = "valid"


class Reason(StrEnum):
    """Why a run directory is not counted, in the order the reasons are listed."""

    HAND_RUN = "hand run"
    NO_DONE = "no DONE"
    DONE_MISMATCH = "DONE does not match the files"
    GATE_FAILED = "gate failed"
    OTHER_CODE = "another code digest"
    NOT_IN_PLAN = "spec not in the plan"
    ATTEMPT = "failed attempt"
    NOT_RECORDED = "not recorded in facts.json"


@dataclass(frozen=True)
class Exclusion:
    reason: Reason
    detail: str | None = None

    def __str__(self) -> str:
        return self.reason.value if self.detail is None else f"{self.reason}: {self.detail}"


@dataclass(frozen=True)
class RunDirectory:
    """One directory under raw/, with what decides whether it counts."""

    # Relative to the results root, with "/" between its parts.
    path: str
    # None when the file is absent or malformed, which only an unfinished run may be.
    config: RunConfig | None
    gate: Gate | None
    done: DoneState
    summary_sha256: str | None

    @property
    def name(self) -> str:
        return self.path.rpartition("/")[2]

    @property
    def attempt(self) -> bool:
        return _ATTEMPT.search(self.path) is not None


@dataclass(frozen=True)
class Current:
    """Count the runs of the working tree's code; each one's summary must be current."""

    code_digest: str
    summarizer_digest: str


@dataclass(frozen=True)
class Recorded:
    """Count the runs that facts.json recorded, with their summaries as they are."""

    code_digest: str
    run_ids: frozenset[str]


@dataclass(frozen=True)
class CountedRun:
    """A counted run, with what the facts read of it."""

    run_id: str
    path: str
    spec: RunSpec
    # None for the model benchmark, which runs no robot.
    scenario: Scenario | None
    summary: Mapping[str, Any]
    environment: RunEnvironment
    failed_checks: tuple[str, ...]


@dataclass(frozen=True)
class _Inputs:
    """Everything the facts are built from."""

    # The counted runs by group: KEY:TOPOLOGY, or compare/CANDIDATE/ARM.
    runs: Mapping[str, tuple[CountedRun, ...]]
    # The run directories of each group's spec and code, and those of them not counted.
    attempts: Mapping[str, int]
    failed_attempts: Mapping[str, int]
    # The groups of the evidence plan, in its order.
    evidence_groups: tuple[str, ...]
    # The facts object of predictions.json; None when no predictions were made.
    predictions: Mapping[str, Any] | None
    ledger: Mapping[str, Mapping[str, Any]]
    config: ServerConfig


@dataclass(frozen=True)
class _Behind:
    """The runs behind one fact, and what was tried for them."""

    runs: tuple[CountedRun, ...]
    # The smallest number of counted runs in any group the fact reads.
    count: int
    attempts: int
    failed_attempts: int


def lookup(document: object, path: Sequence[str]) -> object:
    """Return the value at a path of a JSON document; None where the path leaves it.

    A step into a list selects the item whose name is the step, which is how the checks of
    a summary are addressed. A path leaves the document at a null (a value that was not
    measured) or at a key that is absent (a process that was not running).
    """
    value = document
    for step in path:
        if isinstance(value, list):
            value = next(
                (item for item in value if isinstance(item, dict) and item.get("name") == step),
                None,
            )
        elif isinstance(value, dict):
            value = value.get(step)
        else:
            return None
    return value


def _number(document: object, path: Sequence[str], where: str) -> float | None:
    """Return the number at a path, None for null; AggregateError for any other value."""
    value = lookup(document, path)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AggregateError(f"{where}: {'.'.join(path)} must be a number or null")
    return value


def _required(document: object, path: Sequence[str], where: str) -> float:
    value = _number(document, path, where)
    if value is None:
        raise AggregateError(f"{where}: {'.'.join(path)} is missing")
    return value


def _read_bytes(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except OSError as error:
        raise AggregateError(f"cannot read {path}: {error}") from error


def _read_record[T](read: Callable[[str], T], path: Path) -> T:
    try:
        return read(_read_bytes(path).decode("utf-8"))
    except (UnicodeDecodeError, RunFileError) as error:
        raise AggregateError(f"{path}: {error}") from error


def _directories(parent: Path) -> list[Path]:
    try:
        return sorted(child for child in parent.iterdir() if child.is_dir())
    except OSError as error:
        raise AggregateError(f"cannot list {parent}: {error}") from error


def run_directories(results: Path) -> list[Path]:
    """Return every run directory of a results root: evidence runs, comparison runs, attempts."""
    raw = results / RAW_DIRECTORY
    found: list[Path] = []
    for directory in _directories(raw) if raw.is_dir() else []:
        if directory.name == COMPARE_DIRECTORY:
            for candidate in _directories(directory):
                for arm in _directories(candidate):
                    found.extend(_directories(arm))
        else:
            found.extend(_directories(directory))
    return found


def _record_of[T](
    read: Callable[[str], T], files: Mapping[str, bytes], name: str, where: str, finished: bool
) -> T | None:
    """Return a record of a run's files; None where an unfinished run lacks it or has it broken.

    An interrupted run may stop before or inside any file, and is not counted whatever its
    files hold. A finished run holds every file whole, so there the same is an error.
    """
    content = files.get(name)
    if content is None:
        if finished:
            raise AggregateError(f"{where}: a finished run without {name}")
        return None
    try:
        return read(content.decode("utf-8"))
    except (UnicodeDecodeError, RunFileError) as error:
        if finished:
            raise AggregateError(f"{where}: {error}") from error
        return None


def scan_run(results: Path, directory: Path) -> RunDirectory:
    """Read what decides whether a run directory counts."""
    path = directory.relative_to(results).as_posix()
    try:
        files = run_files(directory)
    except OSError as error:
        raise AggregateError(f"cannot read {directory}: {error}") from error
    done, summary_sha256 = DoneState.ABSENT, None
    if DONE_FILE in files:
        try:
            record = Done.from_json(files[DONE_FILE].decode("utf-8"))
        except UnicodeDecodeError, RunFileError:
            done = DoneState.INVALID
        else:
            done = DoneState.VALID if record.valid_for(files) else DoneState.INVALID
            summary_sha256 = record.summary_sha256
    finished = done is DoneState.VALID
    return RunDirectory(
        path,
        _record_of(RunConfig.from_json, files, CONFIG_FILE, path, finished),
        _record_of(Gate.from_json, files, GATE_FILE, path, finished),
        done,
        summary_sha256,
    )


def run_path(spec: RunSpec) -> str:
    """Return where the run of a spec lies under the results root (DESIGN.md 11.1)."""
    if spec.candidate is None:
        return f"{RAW_DIRECTORY}/{spec.scenario_key}/{run_id(spec)}"
    return f"{RAW_DIRECTORY}/{COMPARE_DIRECTORY}/{spec.candidate}/{spec.arm}/{run_id(spec)}"


def group_of(spec: RunSpec) -> str:
    """Return the group of a spec's runs: the runs that are repeats of each other."""
    if spec.candidate is None:
        return f"{spec.scenario_key}:{spec.topology}"
    return f"{COMPARE_DIRECTORY}/{spec.candidate}/{spec.arm}"


def planned_specs(model_set: ModelSet, digest: str) -> list[RunSpec]:
    """Return the specs whose runs count, each once, as its first repeat.

    The evidence plan's specs feed the facts, and the comparisons' specs the ledger alone.
    """
    comparisons = [
        spec
        for candidate in CANDIDATES
        for spec in compare_plan(candidate, digest)
        if spec.repeat == 1
    ]
    return [*evidence_plan(1, model_set, digest), *comparisons]


def _failed_gates(gate: Gate) -> list[str]:
    passed = {
        "tests_passed": gate.tests_passed.passed,
        "image_built": gate.image_built.built,
        "limits_ok": gate.limits_ok.ok,
        "fresh": gate.fresh.ok,
        "config_ok": gate.config_ok is None or gate.config_ok.ok,
    }
    return [name for name, ok in passed.items() if not ok]


def _planned_as(spec: RunSpec, digest: str) -> RunSpec:
    """Return a spec as the plan would hold it: any repeat of it, run with any code."""
    return replace(spec, repeat=1, code_digest=digest)


def exclusions(run: RunDirectory, planned: Sequence[RunSpec], digest: str) -> tuple[Exclusion, ...]:
    """Return every reason a run directory is not counted for this code; none when it counts."""
    reasons: list[Exclusion] = []
    spec = None if run.config is None else run.config.spec
    if run.config is not None and run.config.hand_run:
        reasons.append(Exclusion(Reason.HAND_RUN))
    if run.done is DoneState.ABSENT:
        reasons.append(Exclusion(Reason.NO_DONE))
    elif run.done is DoneState.INVALID:
        reasons.append(Exclusion(Reason.DONE_MISMATCH))
    if run.gate is not None and not run.gate.gated:
        reasons.append(Exclusion(Reason.GATE_FAILED, ", ".join(_failed_gates(run.gate))))
    if spec is not None:
        if spec.code_digest != digest:
            reasons.append(Exclusion(Reason.OTHER_CODE, spec.code_digest))
        if _planned_as(spec, digest) not in planned:
            reasons.append(Exclusion(Reason.NOT_IN_PLAN))
        elif not run.attempt and run.path != run_path(spec):
            # A copy of a run under another name would count twice, and the tables open a
            # run by the path of its spec.
            reasons.append(Exclusion(Reason.NOT_IN_PLAN, "not in the directory of its spec"))
    if run.attempt:
        reasons.append(Exclusion(Reason.ATTEMPT))
    return tuple(reasons)


def classify(
    runs: Sequence[RunDirectory], planned: Sequence[RunSpec], counting: Current | Recorded
) -> dict[str, tuple[Exclusion, ...]]:
    """Return, for each run directory by its path, why it is not counted; none when it is."""
    if isinstance(counting, Current):
        return {run.path: exclusions(run, planned, counting.code_digest) for run in runs}
    recorded = {
        run.name: run.path
        for run in runs
        if run.name in counting.run_ids and run.done is DoneState.VALID
    }
    unusable = sorted(counting.run_ids - set(recorded))
    if unusable:
        raise AggregateError(
            f"runs that {FACTS_FILE} recorded are missing or have no valid {DONE_FILE}: "
            + ", ".join(unusable)
        )
    return {
        run.path: ()
        if run.path in recorded.values()
        else (*exclusions(run, planned, counting.code_digest), Exclusion(Reason.NOT_RECORDED))
        for run in runs
    }


def _current_summary(
    directory: Path, run: RunDirectory, digest: str | None
) -> dict[str, Any] | None:
    """Return a finished run's summary; None when it is not current.

    A summary is current when it exists, has the hash DONE recorded for it and, where a
    digest is given, was written by the summarizer of that digest.
    """
    path = directory / SUMMARY_FILE
    if not path.is_file():
        return None
    content = _read_bytes(path)
    if hashlib.sha256(content).hexdigest() != run.summary_sha256:
        return None
    try:
        summary = json.loads(content)
    except ValueError as error:
        raise AggregateError(f"{path} is not JSON: {error}") from error
    if not isinstance(summary, dict):
        raise AggregateError(f"{path} must be an object")
    if digest is not None and summary.get("summarizer_digest") != digest:
        return None
    return summary


def _failed_checks(summary: Mapping[str, Any], where: str) -> tuple[str, ...]:
    checks = summary.get("checks")
    if not isinstance(checks, list) or not all(isinstance(check, dict) for check in checks):
        raise AggregateError(f"{where}: checks must be a list of objects")
    # A check whose passed is null has no criterion: it only records what was observed.
    return tuple(str(check.get("name")) for check in checks if check.get("passed") is False)


def _counted_run(directory: Path, run: RunDirectory, summary: dict[str, Any]) -> CountedRun:
    """Read the rest of a counted run: its environment and its failed checks."""
    if run.config is None or run.config.spec is None:
        raise AggregateError(f"{run.path}: a counted run without a run spec")
    fleet = run.config.fleet
    return CountedRun(
        run_id=run.name,
        path=run.path,
        spec=run.config.spec,
        scenario=None if fleet is None else fleet.scenario,
        summary=summary,
        environment=_read_record(RunEnvironment.from_json, directory / ENVIRONMENT_FILE),
        # The model benchmark runs no robot and checks nothing.
        failed_checks=() if fleet is None else _failed_checks(summary, run.path),
    )


def _median(runs: Sequence[CountedRun], path: Sequence[str]) -> float | None:
    """Return the median over the runs of the number at a path; None when no run has one."""
    values = [value for run in runs if (value := _number(run.summary, path, run.path)) is not None]
    return statistics.median(values) if values else None


def _bench_median(runs: Sequence[CountedRun], path: Sequence[str]) -> float:
    """Return the median over the model benchmark runs of a number each of them must hold."""
    return statistics.median(_required(run.summary, path, run.path) for run in runs)


def _bench_fact(runs: Sequence[CountedRun], key: str) -> float:
    return _bench_median(runs, REGISTRY[f"{MODEL_BENCH_KEY}.{key}"].path)


def _measured_tables(runs: Sequence[CountedRun]) -> dict[str, object]:
    """Return the batch times of the model files, per model, thread count and batch size."""
    return {
        model: {
            "threads": {
                str(threads): {
                    str(size): {
                        "wall_ms": _bench_fact(runs, f"{model}.t{threads}.b{size}.wall_ms"),
                        # A batch also costs the CPU time of its dispatch, on both sides.
                        "cpu_ms": _bench_fact(runs, f"{model}.t{threads}.b{size}.cpu_ms")
                        + sum(
                            _bench_median(runs, ("dispatch", str(size), side))
                            for side in ("gateway_cpu_us", "worker_cpu_us")
                        )
                        / _US_PER_MS,
                    }
                    for size in BENCH_BATCH_SIZES
                }
                for threads in BENCH_THREADS
            }
        }
        for model in MODEL_KEYS
    }


def _fake_tables(config: ServerConfig) -> dict[str, object]:
    """Return the service times of the fake runtime, which waits and uses no CPU time.

    Every size of the default config is given, at one thread, so the table holds whatever
    sizes the fake set's config calibrates.
    """
    default = ServerConfig()
    return {
        model: {
            "threads": {
                "1": {
                    str(size): {"wall_ms": config.models[model].fake_frame_ms * size, "cpu_ms": 0.0}
                    for size in default.models[model].batch_sizes
                }
            }
        }
        for model in MODEL_KEYS
    }


def t_b_document(
    model_set: ModelSet, config: ServerConfig, runs: Sequence[CountedRun]
) -> dict[str, object]:
    """Return the capacity model's input, from the counted model benchmark runs.

    Every measured input of a prediction comes from here, and so from no demo run.
    """
    run_ids = ", ".join(sorted(run.run_id for run in runs))
    round_trip_ms = _bench_median(runs, ("websocket", "round_trip_ms_mean"))
    # Each leg is set to the whole measured round trip that contains it: an upper bound.
    dispatch_ms = max(
        _bench_median(runs, ("dispatch", str(size), "wall_ms_mean")) for size in BENCH_BATCH_SIZES
    )
    fake = model_set is ModelSet.FAKE
    return {
        "label": (
            f"fake service times; overheads and I/O cost from model benchmark runs {run_ids}"
            if fake
            else f"model benchmark runs {run_ids}"
        ),
        "models": _fake_tables(config) if fake else _measured_tables(runs),
        "overheads_ms": {
            "transport_up": round_trip_ms,
            "dispatch_request": dispatch_ms,
            "dispatch_result": dispatch_ms,
            "deliver": round_trip_ms,
        },
        "io_cpu_ms_per_frame": sum(
            _bench_median(runs, ("websocket", f"{side}_cpu_us_per_message"))
            for side in ("receiver", "sender")
        )
        / _US_PER_MS,
    }


def predictions_document(config: ServerConfig, t_b_text: str) -> dict[str, object]:
    """Return the capacity model's predictions for the input file's text.

    The config is the one the evidence runs use, unchanged, so every scheduler setting and
    the pacing share are theirs. The file's I/O cost is the measured one, which enters only
    the predicted CPU demand.
    """
    try:
        measured = parse_input(t_b_text.encode(), config.models)
    except CapacityInputError as error:
        raise AggregateError(f"{T_B_FILE}: {error}") from error
    return predict(
        config,
        measured.tables,
        measured.overheads,
        measured.io_cpu_ms_per_frame,
        QUOTA_CORES,
        _PREDICTION_SEED,
        _PREDICTION_ROBOT_COUNTS,
        measured.label,
    )


def _metric_values(runs: Sequence[CountedRun], path: str) -> list[float]:
    return [_required(run.summary, path.split("."), run.path) for run in runs]


def ledger_line(
    candidate: Candidate, arms: Mapping[str, Sequence[CountedRun]]
) -> dict[str, object]:
    """Return a candidate's line of the ledger, from its counted runs per arm."""
    metrics = (candidate.main, *candidate.guards)
    line: dict[str, object] = {
        "name": candidate.name,
        "hypothesis": candidate.hypothesis,
        "scenario_key": candidate.scenario_key,
        "topology": candidate.topology.value,
        "model_set": candidate.model_set.value,
        "a_overrides": dict(candidate.a_overrides),
        "b_overrides": dict(candidate.b_overrides),
        "main": candidate.main._asdict(),
        "guards": [guard._asdict() for guard in candidate.guards],
        "run_ids": {arm: sorted(run.run_id for run in arms[arm]) for arm in ARMS},
        "metrics": {},
        "outcome": NOT_RUN,
        "main_delta": None,
        "baseline_spread": None,
    }
    if not all(arms[arm] for arm in ARMS):
        return line
    a_values, b_values = (
        {metric.path: _metric_values(arms[arm], metric.path) for metric in metrics} for arm in ARMS
    )
    measured = {
        metric.path: {
            "better": metric.better.value,
            "a_median": statistics.median(a_values[metric.path]),
            "b_median": statistics.median(b_values[metric.path]),
            "change": statistics.median(b_values[metric.path])
            - statistics.median(a_values[metric.path]),
            # The spread of arm A is the baseline spread.
            "a_spread": max(a_values[metric.path]) - min(a_values[metric.path]),
            "b_spread": max(b_values[metric.path]) - min(b_values[metric.path]),
            "threshold": threshold(metric.path, a_values[metric.path], b_values[metric.path]),
        }
        for metric in metrics
    }
    return {
        **line,
        "metrics": measured,
        "outcome": outcome(candidate, a_values, b_values).value,
        "main_delta": measured[candidate.main.path]["change"],
        "baseline_spread": measured[candidate.main.path]["a_spread"],
    }


def _behind(inputs: _Inputs, groups: Sequence[str]) -> _Behind:
    counted = [inputs.runs.get(group, ()) for group in groups]
    return _Behind(
        runs=tuple(run for runs in counted for run in runs),
        count=min(len(runs) for runs in counted),
        attempts=sum(inputs.attempts.get(group, 0) for group in groups),
        failed_attempts=sum(inputs.failed_attempts.get(group, 0) for group in groups),
    )


def _fact(
    entry: FactKey,
    value: int | float | str | None,
    behind: _Behind,
    run_values: Sequence[float] = (),
) -> Fact:
    return Fact(
        value=value,
        unit=entry.unit,
        runs=behind.count,
        min=stored(min(run_values), entry.integer) if run_values else None,
        max=stored(max(run_values), entry.integer) if run_values else None,
        run_ids=tuple(sorted(run.run_id for run in behind.runs)),
        dirty_runs=sum(run.environment.dirty for run in behind.runs),
        failed_checks={run.run_id: run.failed_checks for run in behind.runs if run.failed_checks},
        attempts=behind.attempts,
        failed_attempts=behind.failed_attempts,
    )


def _groups(entry: FactKey, inputs: _Inputs) -> list[str]:
    """Return the groups whose runs a key reads."""
    if entry.source is Source.LEDGER:
        return [f"{COMPARE_DIRECTORY}/{entry.path[0]}/{arm}" for arm in ARMS]
    if entry.source is Source.ENVIRONMENT:
        return list(inputs.evidence_groups)
    if entry.source is Source.PREDICTION:
        return [_BENCH_GROUP]
    return [f"{entry.scenario_key}:{entry.topology}"]


def _scale_groups(entry: FactKey, through_robot_count: int | None) -> list[str]:
    """Return the groups of the scale points up to a robot count; every point's for None.

    A key computed across the scale points reads every point, and is null when one has no
    median, but its value can be decided below the last point. Its fact then rests on the
    runs of the points that decided it, and carries their failed checks alone.
    """
    return [
        f"{scenario_key}:{entry.topology}"
        for scenario_key, robot_count in SCALE_POINTS
        if through_robot_count is None or robot_count <= through_robot_count
    ]


def _scenario(run: CountedRun) -> Scenario:
    if run.scenario is None:
        raise AggregateError(f"{run.path}: a run without robots has no command series")
    return run.scenario


def _bins(run: CountedRun, robot_id: str, model: str, from_s: float) -> list[int] | None:
    """Return a robot's commands of a model in the complete 1 s bins from a time to the end.

    None where the summary has no series of the robot and model: the robot left no samples,
    or did not ask for the model.
    """
    path = (*_COMMAND_BINS, robot_id, model)
    counts = lookup(run.summary, path)
    if counts is None:
        return None
    whole_s = math.floor(_scenario(run).duration_s)
    if (
        not isinstance(counts, list)
        or len(counts) < whole_s
        or not all(isinstance(count, int) and not isinstance(count, bool) for count in counts)
    ):
        raise AggregateError(
            f"{run.path}: {'.'.join(path)} must hold a whole number of commands for each "
            "second of the run"
        )
    # The scenarios join and kill at whole seconds, so the summary's bins from t0 are also
    # the bins from the join or the kill.
    return counts[math.ceil(from_s) : whole_s]


def _settle_s(run: CountedRun, model: str) -> int | None:
    robots = _scenario(run).robots
    first = [robot for robot in robots if robot.start_s == 0]
    joiners = [robot for robot in robots if robot.start_s > 0]
    if len(first) != 1 or len(joiners) != 1:
        raise AggregateError(
            f"{run.path}: a settle time needs one robot from the start and one that joins"
        )
    join_s = joiners[0].start_s
    first_bins = _bins(run, first[0].robot_id, model, join_s)
    joiner_bins = _bins(run, joiners[0].robot_id, model, join_s)
    if first_bins is None or joiner_bins is None:
        return None
    return settle_s(first_bins, joiner_bins)


def _recovery_s(run: CountedRun, model: str) -> int | None:
    robots = _scenario(run).robots
    kills = [robot.stop_s for robot in robots if robot.stop_s is not None]
    if len(kills) != 1 or len(robots) < 2:
        raise AggregateError(
            f"{run.path}: a recovery time needs one robot that is killed and one that survives"
        )
    before_per_s = _number(
        run.summary, ("survivors", model, "before", "commands_per_s_per_robot"), run.path
    )
    series = [
        _bins(run, robot.robot_id, model, kills[0]) for robot in robots if robot.stop_s is None
    ]
    survivors = [bins for bins in series if bins is not None]
    if before_per_s is None or len(survivors) != len(series):
        return None
    return recovery_s(
        [statistics.fmean(counts) for counts in zip(*survivors, strict=True)],
        before_per_s,
        len(robots),
    )


def _model_of(entry: FactKey) -> str:
    if entry.model is None:
        raise ValueError(f"{entry.key}: its rule needs a model, and the registry gives none")
    return entry.model


_RUN_RULES: Mapping[Source, Callable[[CountedRun, str], int | None]] = {
    Source.SETTLE: _settle_s,
    Source.RECOVERY: _recovery_s,
}


def _run_value(entry: FactKey, run: CountedRun) -> float | None:
    # The detection time is that of the session the check saw close abnormally; where the
    # check did not see that, the time is of no such close.
    if entry.source is Source.DETECT and lookup(run.summary, _CLOSED_ABNORMAL_PASSED) is not True:
        return None
    if entry.source in _RUN_RULES:
        return _RUN_RULES[entry.source](run, _model_of(entry))
    return _number(run.summary, entry.path, run.path)


def _per_run_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    """Build a fact that has one value per run, combined by the key's rule."""
    behind = _behind(inputs, _groups(entry, inputs))
    values = [value for run in behind.runs if (value := _run_value(entry, run)) is not None]
    return _fact(entry, combined(values, entry) if values else None, behind, values)


def _one_run_fact(
    entry: FactKey,
    inputs: _Inputs,
    choose: Callable[[Mapping[str, float]], str],
    by_path: Sequence[str],
) -> Fact:
    """Build a fact whose value is one run's, chosen from each run's number at by_path."""
    behind = _behind(inputs, _groups(entry, inputs))
    by_run = {
        run.run_id: number
        for run in behind.runs
        if (number := _number(run.summary, by_path, run.path)) is not None
    }
    values = [value for run in behind.runs if (value := _run_value(entry, run)) is not None]
    if not by_run:
        return _fact(entry, None, behind, values)
    chosen = choose(by_run)
    (run,) = (run for run in behind.runs if run.run_id == chosen)
    value = _number(run.summary, entry.path, run.path)
    return _fact(entry, None if value is None else stored(value, entry.integer), behind, values)


def _breakdown_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    """Build a fact of the breakdown: its value comes from one run per model (DESIGN.md 7)."""
    return _one_run_fact(entry, inputs, breakdown_run, (*entry.path[:3], "e2e_ms", "mean"))


def _gap_bound_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    """Build a gap bound: that of the run whose longest gap is the value of the gap fact."""
    gap_path = ("models", _model_of(entry), "all", "command_gap_ms", "max")
    return _one_run_fact(entry, inputs, gap_bound_run, gap_path)


def _point_median(
    inputs: _Inputs, entry: FactKey, scenario_key: str, metric: Sequence[str]
) -> float | None:
    runs = inputs.runs.get(f"{scenario_key}:{entry.topology}", ())
    return _median(runs, ("models", _model_of(entry), "all", *metric))


def _inflection_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    points = [
        ScalePoint(
            robot_count,
            _point_median(inputs, entry, scenario_key, ("utilization",)),
            _point_median(inputs, entry, scenario_key, ("capacity_fps",)),
        )
        for scenario_key, robot_count in SCALE_POINTS
    ]
    found = inflection(points, inputs.config.frame_rate_hz)
    if found is None:
        return _fact(entry, None, _behind(inputs, _scale_groups(entry, None)))
    groups = _scale_groups(entry, found.robot_count)
    return _fact(entry, asdict(found)[entry.path[0]], _behind(inputs, groups))


def _useful_robots_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    peaks = [
        (robot_count, _point_median(inputs, entry, scenario_key, ("command_age_peak_ms", "p95")))
        for scenario_key, robot_count in SCALE_POINTS
    ]
    value = useful_robots(peaks, DEFAULT_WATCHDOG_MS[_model_of(entry)])
    # The point after the largest count within the limit is the first one above it, and no
    # later point changes the value; with no such point, or no value, every point decides.
    above = None if value is None else next((count for count, _ in peaks if count > value), None)
    return _fact(entry, value, _behind(inputs, _scale_groups(entry, above)))


def _batch_plan_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    """Build a fact of the batch cap, as the gateway plans it from the measured batch times."""
    behind = _behind(inputs, _groups(entry, inputs))
    if not behind.runs:
        return _fact(entry, None, behind)
    model = _model_of(entry)
    # The default config and the one-thread table: what the gateway calibrates and plans
    # with at warm-up, whatever model set the demos of this root ran on.
    default = ServerConfig()
    plan = plan_model(
        model,
        default,
        {
            size: round(_bench_fact(behind.runs, f"{model}.t1.b{size}.wall_ms") * _NS_PER_MS)
            for size in default.models[model].batch_sizes
        },
    )
    values = {"batch_cap": plan.scheduler.batch_cap, "margin_ms": plan.margin_ns / _NS_PER_MS}
    return _fact(entry, stored(values[entry.path[0]], entry.integer), behind)


def _prediction_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    behind = _behind(inputs, _groups(entry, inputs))
    if inputs.predictions is None:
        return _fact(entry, None, behind)
    value = _number(inputs.predictions, (entry.key,), PREDICTIONS_FILE)
    return _fact(entry, None if value is None else float(value), behind)


def _ledger_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    candidate, *fields = entry.path
    # A candidate that was not run has no metrics, so a path into them leaves its line.
    value = lookup(inputs.ledger[candidate], fields)
    if isinstance(value, bool) or not isinstance(value, int | float | str | None):
        where = ".".join(entry.path)
        raise AggregateError(f"{LEDGER_FILE}: {where} must be a number, a string or null")
    return _fact(entry, value, _behind(inputs, _groups(entry, inputs)))


def _environment_fact(entry: FactKey, inputs: _Inputs) -> Fact:
    behind = _behind(inputs, _groups(entry, inputs))
    (field,) = entry.path
    if field == "repeats":
        return _fact(entry, behind.count, behind)
    if not behind.runs:
        return _fact(entry, None, behind)
    # A string has no median: the fact is the value of the run that started last.
    latest = max(behind.runs, key=lambda run: (run.environment.inside.start_utc, run.run_id))
    environment = latest.environment
    values = {
        "cpu_model": environment.inside.cpu_model,
        "kernel": environment.inside.kernel,
        "engine": f"{environment.engine} {environment.engine_version}",
    }
    return _fact(entry, values[field], behind)


_BUILDERS: Mapping[Source, Callable[[FactKey, _Inputs], Fact]] = {
    Source.SUMMARY: _per_run_fact,
    Source.SETTLE: _per_run_fact,
    Source.RECOVERY: _per_run_fact,
    Source.DETECT: _per_run_fact,
    Source.BREAKDOWN: _breakdown_fact,
    Source.GAP_BOUND: _gap_bound_fact,
    Source.INFLECTION: _inflection_fact,
    Source.USEFUL_ROBOTS: _useful_robots_fact,
    Source.BATCH_PLAN: _batch_plan_fact,
    Source.PREDICTION: _prediction_fact,
    Source.LEDGER: _ledger_fact,
    Source.ENVIRONMENT: _environment_fact,
}


def _json_text(document: object) -> str:
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _tally(
    runs: Sequence[RunDirectory],
    reference: Sequence[RunSpec],
    digest: str,
    excluded: Mapping[str, tuple[Exclusion, ...]],
) -> tuple[Counter[str], Counter[str]]:
    """Count, per group, the directories of its spec and code, and those of them not counted."""
    attempts: Counter[str] = Counter()
    failed: Counter[str] = Counter()
    for run in runs:
        spec = None if run.config is None else run.config.spec
        if spec is None or spec.code_digest != digest or _planned_as(spec, digest) not in reference:
            continue
        attempts[group_of(spec)] += 1
        failed[group_of(spec)] += bool(excluded[run.path])
    return attempts, failed


def _model_costs(
    model_set: ModelSet, config: ServerConfig, bench_runs: Sequence[CountedRun]
) -> tuple[str | None, str | None, Mapping[str, Any] | None]:
    """Return the text of t_b.json, that of predictions.json, and the predicted facts.

    None for all three where no prediction is made: without a counted model benchmark run,
    and on the null model set, whose models have no cost to predict from.
    """
    if not bench_runs or model_set is ModelSet.NULL:
        return None, None, None
    t_b_text = _json_text(t_b_document(model_set, config, bench_runs))
    document = predictions_document(config, t_b_text)
    predicted = document["facts"]
    if not isinstance(predicted, dict):
        raise AggregateError(f"{PREDICTIONS_FILE}: facts must be an object")
    # Written as the capacity model's own command writes it.
    return t_b_text, json.dumps(document, indent=2, allow_nan=False) + "\n", predicted


def _listings(
    counted: Sequence[CountedRun], excluded: Mapping[str, tuple[Exclusion, ...]]
) -> tuple[str, str]:
    """Return the text of excluded.json and that of failed_checks.json."""
    excluded_runs = [
        {"path": path, "reasons": [str(reason) for reason in reasons]}
        for path, reasons in sorted(excluded.items())
        if reasons
    ]
    failed_runs = [
        {"run_id": run.run_id, "path": run.path, "checks": list(run.failed_checks)}
        for run in sorted(counted, key=lambda run: run.path)
        if run.failed_checks
    ]
    return _json_text({"runs": excluded_runs}), _json_text({"runs": failed_runs})


def _outputs(
    model_set: ModelSet,
    digest: str,
    evidence_groups: tuple[str, ...],
    counted: Sequence[CountedRun],
    tally: tuple[Counter[str], Counter[str]],
    excluded: Mapping[str, tuple[Exclusion, ...]],
) -> dict[str, str | None]:
    """Return the text of every output by its path under --out, in the order they are written.

    None stands for a file that this build does not produce, so that one left by an
    earlier build does not stay beside facts it does not belong to.
    """
    runs: dict[str, tuple[CountedRun, ...]] = {}
    for run in sorted(counted, key=lambda run: run.run_id):
        runs[group_of(run.spec)] = (*runs.get(group_of(run.spec), ()), run)
    config = load_config({OVERRIDES_VARIABLE: json.dumps(overrides_for(model_set, {}))})
    t_b_text, predictions_text, predicted = _model_costs(
        model_set, config, runs.get(_BENCH_GROUP, ())
    )
    lines = [
        ledger_line(
            candidate,
            {arm: runs.get(f"{COMPARE_DIRECTORY}/{candidate.name}/{arm}", ()) for arm in ARMS},
        )
        for candidate in sorted(CANDIDATES, key=lambda candidate: candidate.name)
    ]
    inputs = _Inputs(
        runs=runs,
        attempts=tally[0],
        failed_attempts=tally[1],
        evidence_groups=evidence_groups,
        predictions=predicted,
        ledger={str(line["name"]): line for line in lines},
        config=config,
    )
    facts = Facts(
        model_set=model_set,
        code_digest=digest,
        counted_run_ids=tuple(sorted(run.run_id for run in counted)),
        facts={key: _BUILDERS[entry.source](entry, inputs) for key, entry in REGISTRY.items()},
    )
    excluded_text, failed_checks_text = _listings(counted, excluded)
    return {
        f"{SUMMARY_DIRECTORY}/{T_B_FILE}": t_b_text,
        f"{SUMMARY_DIRECTORY}/{PREDICTIONS_FILE}": predictions_text,
        f"{SUMMARY_DIRECTORY}/{FACTS_FILE}": facts.to_json(),
        LEDGER_FILE: "".join(
            json.dumps(line, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for line in lines
        ),
        f"{SUMMARY_DIRECTORY}/{EXCLUDED_FILE}": excluded_text,
        f"{SUMMARY_DIRECTORY}/{FAILED_CHECKS_FILE}": failed_checks_text,
    }


def _read_facts(results: Path) -> Facts | None:
    """Return the facts a results root holds; None when it holds none."""
    path = results / SUMMARY_DIRECTORY / FACTS_FILE
    return _read_record(Facts.from_json, path) if path.is_file() else None


def _refuse(
    results: Path,
    runs: Sequence[RunDirectory],
    excluded: Mapping[str, tuple[Exclusion, ...]],
    digest: str,
) -> int:
    """Say why no run counts, and what can be done; return the exit status of the refusal."""
    print(f"no run under {results / RAW_DIRECTORY} counts for the code digest {digest}:")
    per_reason = Counter(reason.reason for reasons in excluded.values() for reason in reasons)
    for reason in Reason:
        if per_reason[reason]:
            print(f"  {per_reason[reason]} excluded: {reason}")
    others = sorted(
        {
            run.config.spec.code_digest
            for run in runs
            if run.config is not None and run.config.spec is not None
        }
        - {digest}
    )
    if others:
        print(f"  code digests of the runs found: {', '.join(others)}")
    recorded = _read_facts(results)
    if recorded is not None and recorded.code_digest != digest:
        print(
            f"{FACTS_FILE} records the code digest {recorded.code_digest}, and the code is "
            f"now {digest}. Two ways on:\n"
            "  make evidence                a new set of runs, for the code as it is\n"
            "  make report DIGEST=recorded  render the recorded evidence again"
        )
    print("nothing written")
    return 2


def _no_runs(results: Path) -> int:
    print(f"no run directory under {results / RAW_DIRECTORY}: nothing written")
    return 0


def _load_counted(
    results: Path, runs: Sequence[RunDirectory], counting: Current | Recorded
) -> list[CountedRun]:
    """Read the runs that count; AggregateError names those whose summary is not current."""
    digest = counting.summarizer_digest if isinstance(counting, Current) else None
    counted: list[CountedRun] = []
    stale: list[str] = []
    for run in runs:
        summary = _current_summary(results / run.path, run, digest)
        if summary is None:
            stale.append(run.path)
        else:
            counted.append(_counted_run(results / run.path, run, summary))
    if stale:
        raise AggregateError(
            "the summary of a counted run is missing, does not match its DONE or was written "
            f"by another summarizer: {', '.join(stale)}. Write them with "
            f"python -m bench.summarize --results {results}"
        )
    return counted


def build(results: Path, out: Path, model_set: ModelSet, counting: Current | Recorded) -> int:
    """Aggregate a results root into the files under out; return the exit status."""
    directories = run_directories(results)
    if not directories:
        return _no_runs(results)
    runs = [scan_run(results, directory) for directory in directories]
    digest = counting.code_digest
    planned = planned_specs(model_set, digest)
    excluded = classify(runs, planned, counting)
    counting_runs = [run for run in runs if not excluded[run.path]]
    if not counting_runs:
        return _refuse(results, runs, excluded, digest)
    counted = _load_counted(results, counting_runs, counting)
    # What was tried is counted against the specs of the plan, and against those of the
    # counted runs themselves: the runs that facts.json recorded keep their attempts when
    # the plan has changed since.
    reference = [*planned]
    for run in counted:
        if _planned_as(run.spec, digest) not in reference:
            reference.append(_planned_as(run.spec, digest))
    outputs = _outputs(
        model_set,
        digest,
        tuple(group_of(spec) for spec in planned if spec.candidate is None),
        counted,
        _tally(runs, reference, digest, excluded),
        excluded,
    )
    for name, text in outputs.items():
        path = out / name
        if text is None:
            path.unlink(missing_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            # Bytes, so the newlines are the same on every platform.
            path.write_bytes(text.encode("utf-8"))
            print(f"wrote {path}")
    print(f"{len(counted)} runs counted, {len(runs) - len(counted)} excluded")
    return 0


def tree_code_digest(repository: Path) -> str:
    """Return the code digest of a working tree, from the files git lists in it."""
    try:
        listing = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=True,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise AggregateError(f"cannot list the files of {repository} with git: {error}") from error
    # git also lists a tracked file that the working tree has lost; the digest is of the
    # files that are there, as a run would have them.
    paths = [
        path for path in code_paths(listing.stdout.splitlines()) if (repository / path).is_file()
    ]
    return code_digest({path: _read_bytes(repository / path) for path in paths})


def _listed(document: object, path: Path) -> list[dict[str, Any]]:
    """Return the runs of an excluded.json or a failed_checks.json."""
    runs = lookup(document, ("runs",))
    if not isinstance(runs, list) or not all(isinstance(run, dict) for run in runs):
        raise AggregateError(f"{path}: runs must be a list of objects")
    return runs


def _read_listed(path: Path) -> list[dict[str, Any]]:
    try:
        return _listed(json.loads(_read_bytes(path)), path)
    except ValueError as error:
        raise AggregateError(f"{path} is not JSON: {error}") from error


def status(results: Path, templates: Path) -> int:
    """Print what a reader of the report must know of its facts; return the exit status."""
    facts = _read_facts(results)
    if facts is None:
        print(f"no {FACTS_FILE} under {results / SUMMARY_DIRECTORY}")
        return 0
    excluded = _read_listed(results / SUMMARY_DIRECTORY / EXCLUDED_FILE)
    failed = _read_listed(results / SUMMARY_DIRECTORY / FAILED_CHECKS_FILE)
    templated = {
        key for text in read_templates(templates).values() for key in placeholder_keys(text)
    }
    null_keys = sorted(
        key for key in templated if key in facts.facts and facts.facts[key].value is None
    )
    print(f"excluded runs: {len(excluded)}")
    for run in excluded:
        print(f"  {run.get('path')}: {'; '.join(map(str, run.get('reasons', [])))}")
    print(f"runs with a failed check: {len(failed)}")
    for run in failed:
        print(f"  {run.get('path')}: {', '.join(map(str, run.get('checks', [])))}")
    print(f"templated keys that are null: {len(null_keys)}")
    for key in null_keys:
        print(f"  {key}")
    dirty = sum(fact.dirty_runs > 0 for fact in facts.facts.values())
    if dirty:
        print(f"warning: {dirty} facts rest on a run of a dirty tree")
    # A null on the fake or the null model set is expected: those sets have no model costs.
    return 1 if failed or (facts.model_set is ModelSet.REAL and null_keys) else 0


def _build_command(arguments: argparse.Namespace) -> int:
    results: Path = arguments.results
    # Before any digest is computed: a root without runs needs neither git nor the facts.
    if not run_directories(results):
        return _no_runs(results)
    counting: Current | Recorded
    if arguments.digest == "recorded":
        recorded = _read_facts(results)
        if recorded is None:
            print(
                f"no {FACTS_FILE} under {results / SUMMARY_DIRECTORY} records a counted run: "
                "nothing written"
            )
            return 2
        counting = Recorded(recorded.code_digest, frozenset(recorded.counted_run_ids))
    else:
        counting = Current(tree_code_digest(_REPOSITORY), summarizer_digest())
    return build(results, arguments.out or results, arguments.models, counting)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.aggregate", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build", help="write the facts and the files beside them")
    build_parser.add_argument("--results", type=Path, required=True, help="the results root")
    build_parser.add_argument(
        "--models",
        type=ModelSet,
        choices=list(ModelSet),
        required=True,
        help="the model set of the evidence runs",
    )
    build_parser.add_argument("--out", type=Path, help="where to write (default: the results root)")
    build_parser.add_argument(
        "--digest",
        choices=("current", "recorded"),
        default="current",
        help="count the runs of the working tree's code, or the runs facts.json recorded",
    )
    status_parser = commands.add_parser("status", help="list what the report rests on")
    status_parser.add_argument("--results", type=Path, required=True, help="the results root")
    status_parser.add_argument("--templates", type=Path, default=Path("docs"), help="the templates")
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "status":
            return status(arguments.results, arguments.templates)
        return _build_command(arguments)
    except (AggregateError, RenderError) as error:
        print(f"aggregate: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
