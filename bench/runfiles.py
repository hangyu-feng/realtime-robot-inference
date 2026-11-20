"""The files of a run directory and of the results root, as typed records.

Each record is a dataclass: to_json returns the text of its file or of its line, and
from_json reads that text back and refuses anything malformed with a RunFileError that
names the field. Nothing here opens a file. Writers and readers pass text and bytes, and
agree on a format by using its record.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection, Mapping
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from types import NoneType, UnionType
from typing import Any, NoReturn, get_args, get_type_hints

from bench.scenarios import RobotPlan, Scenario, Window, WorkerKill
from cloud.config import Topology
from common.cgroup import CgroupStats, Limits
from common.errors import ConfigError
from common.procstat import ProcessStats
from robot.frames import FrameSetStats
from robot.settings import LinkSettings

# Names inside a run directory (DESIGN.md 11.3). DONE's hashes are keyed by these names,
# with "/" between a directory and its file.
CONFIG_FILE = "config.json"
ENVIRONMENT_FILE = "env.json"
INSIDE_ENVIRONMENT_FILE = "env_inside.json"
GATE_FILE = "gate.json"
EVENTS_FILE = "events.jsonl"
SCRAPES_FILE = "scrapes.jsonl.gz"
SUMMARY_FILE = "summary.json"
DONE_FILE = "DONE"
PLANS_DIRECTORY = "plans"
SAMPLES_DIRECTORY = "samples"
# Under summary/ in the results root.
FACTS_FILE = "facts.json"

# How a run's robots are offset from each other, as robot.fleet's --phase takes it.
PHASE_MODES = ("random", "aligned")
# The two arms of a comparison, which are also the names of their directories.
ARMS = ("a", "b")


class RunFileError(Exception):
    """A run file or one of its lines is not JSON, or one of its fields is missing or malformed."""


class ModelSet(StrEnum):
    REAL = "real"  # the published models
    FAKE = "fake"  # a fixed service time per frame and no model file
    NULL = "null"  # a result at once: isolates transport and scheduling


type _Read[T] = Callable[[object, str], T]


def _finite_float(text: str) -> float:
    number = float(text)
    # A literal beyond the float range parses as infinity, which JSON cannot write back.
    if not math.isfinite(number):
        raise ValueError(f"{text} is beyond the range of a float")
    return number


def _refuse_constant(text: str) -> NoReturn:
    raise ValueError(f"{text} is not a JSON value")


def _parse(text: str, label: str) -> object:
    try:
        return json.loads(text, parse_float=_finite_float, parse_constant=_refuse_constant)
    except (ValueError, RecursionError) as error:
        # A JSONDecodeError is a ValueError, as is an integer of too many digits; deeply
        # nested brackets raise RecursionError.
        raise RunFileError(f"{label} is not JSON: {error}") from error


def _file_text(document: Mapping[str, object], label: str) -> str:
    """Return the text of a whole file. Keys are sorted, so equal records give equal bytes."""
    try:
        return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    except ValueError as error:
        raise RunFileError(f"{label} cannot be written: {error}") from error


def _line_text(document: Mapping[str, object], label: str) -> str:
    """Return one line of a JSON-lines file, without its newline."""
    try:
        return json.dumps(document, separators=(",", ":"), allow_nan=False)
    except ValueError as error:
        raise RunFileError(f"{label} cannot be written: {error}") from error


def _child(path: str, key: str) -> str:
    """Name a field for an error: "config.json: spec.repeat"."""
    return f"{path}.{key}" if ": " in path else f"{path}: {key}"


class _Fields:
    """The fields of one JSON object, each read with the path that names it in an error."""

    def __init__(
        self, value: object, path: str, required: Collection[str], optional: Collection[str] = ()
    ) -> None:
        if not isinstance(value, dict):
            raise RunFileError(f"{path} must be an object")
        for key in required:
            if key not in value:
                raise RunFileError(f"{_child(path, key)} is missing")
        for key in value:
            if key not in required and key not in optional:
                raise RunFileError(f"{_child(path, key)} is not a field of it")
        self._document: dict[str, Any] = value
        self._path = path

    def __contains__(self, key: str) -> bool:
        return key in self._document

    def read[T](self, key: str, read: _Read[T]) -> T:
        return read(self._document[key], _child(self._path, key))


def _string(value: object, path: str) -> str:
    if not isinstance(value, str):
        raise RunFileError(f"{path} must be a string")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise RunFileError(f"{path} must be true or false")
    return value


def _integer(value: object, path: str) -> int:
    # A bool is an int to isinstance, and true is not a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise RunFileError(f"{path} must be an integer")
    return value


def _number_as_written(value: object, path: str) -> int | float:
    """Return a JSON number with its kind kept: a fact that is an int stays one."""
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RunFileError(f"{path} must be a number")
    return value


def _number(value: object, path: str) -> float:
    """Return a JSON number as a float."""
    try:
        return float(_number_as_written(value, path))
    except OverflowError as error:
        raise RunFileError(f"{path} must be within the range of a float") from error


def _json_object(value: object, path: str) -> dict[str, Any]:
    """Return an object whose content belongs to its writer: a config echo, a metrics snapshot."""
    if not isinstance(value, dict):
        raise RunFileError(f"{path} must be an object")
    return value


def _optional[T](read: _Read[T]) -> _Read[T | None]:
    def read_optional(value: object, path: str) -> T | None:
        return None if value is None else read(value, path)

    return read_optional


def _sequence[T](read: _Read[T]) -> _Read[tuple[T, ...]]:
    def read_sequence(value: object, path: str) -> tuple[T, ...]:
        if not isinstance(value, list):
            raise RunFileError(f"{path} must be a list")
        return tuple(read(item, f"{path}[{index}]") for index, item in enumerate(value))

    return read_sequence


def _mapping[T](read: _Read[T]) -> _Read[dict[str, T]]:
    def read_mapping(value: object, path: str) -> dict[str, T]:
        if not isinstance(value, dict):
            raise RunFileError(f"{path} must be an object")
        return {key: read(item, _child(path, key)) for key, item in value.items()}

    return read_mapping


def _member[E: StrEnum](members: type[E]) -> _Read[E]:
    def read_member(value: object, path: str) -> E:
        if not isinstance(value, str) or value not in members:
            raise RunFileError(f"{path} must be one of {', '.join(members)}")
        return members(value)

    return read_member


def _one_of(choices: tuple[str, ...]) -> _Read[str]:
    def read_choice(value: object, path: str) -> str:
        if not isinstance(value, str) or value not in choices:
            raise RunFileError(f"{path} must be one of {', '.join(choices)}")
        return value

    return read_choice


_SCALARS: Mapping[object, _Read[Any]] = {
    bool: _boolean,
    int: _integer,
    float: _number,
    str: _string,
}


def _record[T](record_type: Callable[..., T]) -> _Read[T]:
    """Return the reader of a dataclass whose fields are numbers, strings and booleans.

    The reader follows the declared types, so a record that another module publishes (the
    cgroup limits, a process's counters) is read here as that module defines it.
    """
    readers: dict[str, _Read[Any]] = {}
    for name, declared in get_type_hints(record_type).items():
        if isinstance(declared, UnionType):
            # The only union among these fields is a scalar or None.
            (scalar,) = (member for member in get_args(declared) if member is not NoneType)
            readers[name] = _optional(_SCALARS[scalar])
        else:
            readers[name] = _SCALARS[declared]

    def read_record(value: object, path: str) -> T:
        document = _Fields(value, path, readers)
        return record_type(**{name: document.read(name, read) for name, read in readers.items()})

    return read_record


_read_limits = _record(Limits)
_read_cgroup_stats = _record(CgroupStats)
_read_process_stats = _record(ProcessStats)
_read_frame_set_stats = _record(FrameSetStats)
_read_robot_plan = _record(RobotPlan)
_read_window = _record(Window)
_read_worker_kill = _record(WorkerKill)
_read_link_settings = _record(LinkSettings)


def _read_link(value: object, path: str) -> LinkSettings:
    try:
        return _read_link_settings(value, path)
    except ConfigError as error:
        # The settings refuse a delay or a jitter below zero themselves.
        raise RunFileError(f"{path}: {error}") from error


@dataclass(frozen=True)
class RunSpec:
    """What identifies a run: runs whose specs differ only in the repeat index are repeats.

    The record is defined with the files because config.json stores it. The run plan builds
    the specs and derives a run's id from this record's JSON.
    """

    scenario_key: str
    topology: Topology
    # One of PHASE_MODES; None for the model benchmark, which runs no robot.
    phase_mode: str | None
    model_set: ModelSet
    # The gateway's config echo (DESIGN.md 10.8) with an empty version: the commit is not
    # part of what a run is.
    server_config: Mapping[str, Any]
    uplink: LinkSettings
    downlink: LinkSettings
    # The SHA-256 of a compose file laid over the topology's own; None without one.
    extra_compose: str | None
    # A comparison's candidate and its arm, one of ARMS; both None for an evidence run.
    candidate: str | None
    arm: str | None
    code_digest: str
    repeat: int

    def to_json(self) -> str:
        return _file_text(_spec_document(self), "a run spec")

    @classmethod
    def from_json(cls, text: str) -> RunSpec:
        return _read_spec(_parse(text, "a run spec"), "a run spec")


def _spec_document(spec: RunSpec) -> dict[str, object]:
    return {
        "scenario_key": spec.scenario_key,
        "topology": spec.topology.value,
        "phase_mode": spec.phase_mode,
        "model_set": spec.model_set.value,
        "server_config": dict(spec.server_config),
        "uplink": asdict(spec.uplink),
        "downlink": asdict(spec.downlink),
        "extra_compose": spec.extra_compose,
        "candidate": spec.candidate,
        "arm": spec.arm,
        "code_digest": spec.code_digest,
        "repeat": spec.repeat,
    }


def _read_spec(value: object, path: str) -> RunSpec:
    document = _Fields(value, path, [field.name for field in fields(RunSpec)])
    spec = RunSpec(
        scenario_key=document.read("scenario_key", _string),
        topology=document.read("topology", _member(Topology)),
        phase_mode=document.read("phase_mode", _optional(_one_of(PHASE_MODES))),
        model_set=document.read("model_set", _member(ModelSet)),
        server_config=document.read("server_config", _json_object),
        uplink=document.read("uplink", _read_link),
        downlink=document.read("downlink", _read_link),
        extra_compose=document.read("extra_compose", _optional(_string)),
        candidate=document.read("candidate", _optional(_string)),
        arm=document.read("arm", _optional(_one_of(ARMS))),
        code_digest=document.read("code_digest", _string),
        repeat=document.read("repeat", _integer),
    )
    if (spec.arm is None) != (spec.candidate is None):
        raise RunFileError(f"{_child(path, 'arm')} must be null exactly when candidate is")
    return spec


class EventName(StrEnum):
    T0 = "t0"  # the first line: time zero on the shared monotonic clock
    SERVER_READY = "server_ready"  # by the gateway's own clock, so before t0
    ROBOT_START = "robot_start"
    ROBOT_JOIN = "robot_join"  # a start after t0
    ROBOT_KILL = "robot_kill"
    ROBOT_STOP = "robot_stop"  # a robot that ran to the end
    SCENARIO_END = "scenario_end"
    FINAL_POLL = "final_poll"  # the wait, after the fleet, for the gateway's last counts
    # A worker killed with SIGKILL, timed in the gateway's container right after the kill.
    FAULT = "fault"


_ROBOT_EVENT_FIELDS = ("robot_id",)
_EVENT_FIELDS: Mapping[EventName, tuple[str, ...]] = {
    EventName.T0: ("t0_mono_ns",),
    EventName.SERVER_READY: (),
    EventName.ROBOT_START: _ROBOT_EVENT_FIELDS,
    EventName.ROBOT_JOIN: _ROBOT_EVENT_FIELDS,
    EventName.ROBOT_KILL: _ROBOT_EVENT_FIELDS,
    EventName.ROBOT_STOP: _ROBOT_EVENT_FIELDS,
    EventName.SCENARIO_END: (),
    EventName.FINAL_POLL: ("settled", "waited_s"),
    EventName.FAULT: ("model", "pid"),
}
_EVENT_READERS: Mapping[str, _Read[Any]] = {
    "robot_id": _string,
    "t0_mono_ns": _integer,
    "settled": _boolean,
    "waited_s": _number,
    "model": _string,
    "pid": _integer,
}
_EVENT_LINE = f"a line of {EVENTS_FILE}"


@dataclass(frozen=True)
class Event:
    """One line of events.jsonl, or one planned event of config.json.

    t_s is in seconds from t0. A line holds, beside its name and time, only the fields of
    its kind, and the other fields here are None.
    """

    name: EventName
    t_s: float
    robot_id: str | None = None  # the four robot events
    t0_mono_ns: int | None = None  # t0
    settled: bool | None = None  # final_poll: whether the poll saw every count it waited for
    waited_s: float | None = None  # final_poll: how long the poll took
    model: str | None = None  # fault: the model whose worker was killed
    pid: int | None = None  # fault: the process that was killed

    def to_json(self) -> str:
        return _line_text(_event_document(self), _EVENT_LINE)

    @classmethod
    def from_json(cls, line: str) -> Event:
        return _read_event(_parse(line, _EVENT_LINE), _EVENT_LINE)


def _event_document(event: Event) -> dict[str, object]:
    return {
        "event": event.name.value,
        "t_s": event.t_s,
        **{key: getattr(event, key) for key in _EVENT_FIELDS[event.name]},
    }


def _read_event(value: object, path: str) -> Event:
    name = _Fields(value, path, ("event",), ("t_s", *_EVENT_READERS)).read(
        "event", _member(EventName)
    )
    document = _Fields(value, path, ("event", "t_s", *_EVENT_FIELDS[name]))
    return Event(
        name,
        document.read("t_s", _number),
        **{key: document.read(key, _EVENT_READERS[key]) for key in _EVENT_FIELDS[name]},
    )


@dataclass(frozen=True)
class FleetConfig:
    """The robots of a serving run: their timeline, what each declares, and what is planned."""

    scenario: Scenario
    frame_rate_hz: float
    models: tuple[str, ...]
    stale_ms: Mapping[str, float]
    watchdog_ms: Mapping[str, float]
    # The events the timeline fixes in advance, each at its planned time from t0.
    planned_events: tuple[Event, ...]


def _fleet_config_document(fleet: FleetConfig) -> dict[str, object]:
    return {
        "scenario": asdict(fleet.scenario),
        "frame_rate_hz": fleet.frame_rate_hz,
        "models": list(fleet.models),
        "stale_ms": dict(fleet.stale_ms),
        "watchdog_ms": dict(fleet.watchdog_ms),
        "planned_events": [_event_document(event) for event in fleet.planned_events],
    }


def _read_scenario(value: object, path: str) -> Scenario:
    document = _Fields(value, path, [field.name for field in fields(Scenario)])
    return Scenario(
        name=document.read("name", _string),
        duration_s=document.read("duration_s", _number),
        robots=document.read("robots", _sequence(_read_robot_plan)),
        windows=document.read("windows", _sequence(_read_window)),
        fault=document.read("fault", _optional(_read_worker_kill)),
    )


def _read_fleet_config(value: object, path: str) -> FleetConfig:
    document = _Fields(value, path, [field.name for field in fields(FleetConfig)])
    return FleetConfig(
        scenario=document.read("scenario", _read_scenario),
        frame_rate_hz=document.read("frame_rate_hz", _number),
        models=document.read("models", _sequence(_string)),
        stale_ms=document.read("stale_ms", _mapping(_number)),
        watchdog_ms=document.read("watchdog_ms", _mapping(_number)),
        planned_events=document.read("planned_events", _sequence(_read_event)),
    )


@dataclass(frozen=True)
class RunConfig:
    """config.json: what a run is, written before it starts.

    A run that the scenario driver started by hand has no spec and no run id, and is never
    counted. The model benchmark has no fleet.
    """

    spec: RunSpec | None
    run_id: str | None
    seed: int
    # The compose file whose hash the spec holds, as the runner was given it.
    extra_compose_path: str | None
    fleet: FleetConfig | None

    @property
    def hand_run(self) -> bool:
        return self.spec is None

    def to_json(self) -> str:
        document = {
            "hand_run": self.hand_run,
            "spec": None if self.spec is None else _spec_document(self.spec),
            "run_id": self.run_id,
            "seed": self.seed,
            "extra_compose_path": self.extra_compose_path,
            "fleet": None if self.fleet is None else _fleet_config_document(self.fleet),
        }
        return _file_text(document, CONFIG_FILE)

    @classmethod
    def from_json(cls, text: str) -> RunConfig:
        document = _Fields(
            _parse(text, CONFIG_FILE),
            CONFIG_FILE,
            ("hand_run", "spec", "run_id", "seed", "extra_compose_path", "fleet"),
        )
        config = cls(
            spec=document.read("spec", _optional(_read_spec)),
            run_id=document.read("run_id", _optional(_string)),
            seed=document.read("seed", _integer),
            extra_compose_path=document.read("extra_compose_path", _optional(_string)),
            fleet=document.read("fleet", _optional(_read_fleet_config)),
        )
        if document.read("hand_run", _boolean) != config.hand_run:
            raise RunFileError(f"{CONFIG_FILE}: hand_run must be true exactly when spec is null")
        if (config.run_id is None) != config.hand_run:
            raise RunFileError(f"{CONFIG_FILE}: run_id must be null exactly when spec is")
        return config


_PLAN_FILE = f"a plan of {PLANS_DIRECTORY}/"


@dataclass(frozen=True)
class FleetPlan:
    """A file of plans/: the robots of one fleet process, as robot.fleet's --plan reads them.

    t0_mono_ns is time zero on the shared monotonic clock. The fleet's own format lets it
    be null, but every plan of a run carries it: the run's processes share one time zero.
    """

    t0_mono_ns: int
    duration_s: float
    robots: tuple[RobotPlan, ...]

    def to_json(self) -> str:
        return _file_text(asdict(self), _PLAN_FILE)

    @classmethod
    def from_json(cls, text: str) -> FleetPlan:
        document = _Fields(
            _parse(text, _PLAN_FILE), _PLAN_FILE, ("t0_mono_ns", "duration_s", "robots")
        )
        return cls(
            t0_mono_ns=document.read("t0_mono_ns", _integer),
            duration_s=document.read("duration_s", _number),
            robots=document.read("robots", _sequence(_read_robot_plan)),
        )


_SCRAPE_LINE = f"a line of {SCRAPES_FILE}"
_SCRAPE_KEYS = ("k", "driver_before_ns", "driver_after_ns", "ok", "fleet")


@dataclass(frozen=True)
class Scrape:
    """One line of scrapes.jsonl.gz: one reading of /metrics and of the fleet's processes.

    A scrape that succeeded holds the metrics object as the gateway sent it. One that
    failed, or took longer than its limit, holds the error and no metrics.
    """

    # The planned second from t0; None for the final scrape, taken after the fleet has left.
    k: int | None
    # The driver's monotonic clock just before the request and just after the response.
    driver_before_ns: int
    driver_after_ns: int
    metrics: Mapping[str, Any] | None
    error: str | None
    # The fleet processes that were alive, and the scenario driver, each by its name.
    fleet_processes: Mapping[str, ProcessStats]
    # The counters of the robots' container; None in the single topology, which has none.
    fleet_cgroup: CgroupStats | None

    @property
    def ok(self) -> bool:
        return self.metrics is not None

    def to_json(self) -> str:
        document: dict[str, object] = {
            "k": self.k,
            "driver_before_ns": self.driver_before_ns,
            "driver_after_ns": self.driver_after_ns,
            "ok": self.ok,
        }
        if self.metrics is None:
            document["error"] = self.error
        else:
            document["metrics"] = dict(self.metrics)
        fleet: dict[str, object] = {
            "processes": {name: asdict(stats) for name, stats in self.fleet_processes.items()}
        }
        # The key is written only where there is a robots container.
        if self.fleet_cgroup is not None:
            fleet["cgroup"] = asdict(self.fleet_cgroup)
        document["fleet"] = fleet
        return _line_text(document, _SCRAPE_LINE)

    @classmethod
    def from_json(cls, line: str) -> Scrape:
        value = _parse(line, _SCRAPE_LINE)
        ok = _Fields(value, _SCRAPE_LINE, _SCRAPE_KEYS, ("metrics", "error")).read("ok", _boolean)
        document = _Fields(value, _SCRAPE_LINE, (*_SCRAPE_KEYS, "metrics" if ok else "error"))
        processes, cgroup = document.read("fleet", _read_scrape_fleet)
        return cls(
            k=document.read("k", _optional(_integer)),
            driver_before_ns=document.read("driver_before_ns", _integer),
            driver_after_ns=document.read("driver_after_ns", _integer),
            metrics=document.read("metrics", _json_object) if ok else None,
            error=None if ok else document.read("error", _string),
            fleet_processes=processes,
            fleet_cgroup=cgroup,
        )


def _read_scrape_fleet(
    value: object, path: str
) -> tuple[dict[str, ProcessStats], CgroupStats | None]:
    document = _Fields(value, path, ("processes",), ("cgroup",))
    return (
        document.read("processes", _mapping(_read_process_stats)),
        document.read("cgroup", _read_cgroup_stats) if "cgroup" in document else None,
    )


@dataclass(frozen=True)
class InsideEnvironment:
    """env_inside.json: what the scenario driver read inside its container at the start.

    The fields of bench.environment.collect_environment, and the sizes of the frame set.
    """

    cpu_model: str | None
    cpu_count_visible: int | None
    machine: str
    kernel: str
    cgroup: Limits
    python: str
    packages: Mapping[str, str | None]
    model_sha256: Mapping[str, str]
    start_utc: str
    frames: FrameSetStats

    def to_json(self) -> str:
        return _file_text(_inside_document(self), INSIDE_ENVIRONMENT_FILE)

    @classmethod
    def from_json(cls, text: str) -> InsideEnvironment:
        document = _Fields(
            _parse(text, INSIDE_ENVIRONMENT_FILE), INSIDE_ENVIRONMENT_FILE, _INSIDE_KEYS
        )
        return _read_inside(document)


_INSIDE_KEYS = tuple(field.name for field in fields(InsideEnvironment))


def _inside_document(inside: InsideEnvironment) -> dict[str, object]:
    return {
        "cpu_model": inside.cpu_model,
        "cpu_count_visible": inside.cpu_count_visible,
        "machine": inside.machine,
        "kernel": inside.kernel,
        "cgroup": asdict(inside.cgroup),
        "python": inside.python,
        "packages": dict(inside.packages),
        "model_sha256": dict(inside.model_sha256),
        "start_utc": inside.start_utc,
        "frames": asdict(inside.frames),
    }


def _read_inside(document: _Fields) -> InsideEnvironment:
    return InsideEnvironment(
        cpu_model=document.read("cpu_model", _optional(_string)),
        cpu_count_visible=document.read("cpu_count_visible", _optional(_integer)),
        machine=document.read("machine", _string),
        kernel=document.read("kernel", _string),
        cgroup=document.read("cgroup", _read_limits),
        python=document.read("python", _string),
        packages=document.read("packages", _mapping(_optional(_string))),
        model_sha256=document.read("model_sha256", _mapping(_string)),
        start_utc=document.read("start_utc", _string),
        frames=document.read("frames", _read_frame_set_stats),
    )


@dataclass(frozen=True)
class RunEnvironment:
    """env.json: where a run was measured (DESIGN.md 11.3).

    The file holds the inside fields beside its own. Their cgroup is that of the container
    the gate checks, the gateway's; in the split topology robots_cgroup holds the limits
    of the robots' container.
    """

    inside: InsideEnvironment
    host_memory_total_bytes: int
    engine: str
    engine_version: str
    commit: str
    dirty: bool
    code_digest: str
    image_digest: str
    robots_cgroup: Limits | None

    def to_json(self) -> str:
        document = {
            **_inside_document(self.inside),
            "host_memory_total_bytes": self.host_memory_total_bytes,
            "engine": self.engine,
            "engine_version": self.engine_version,
            "commit": self.commit,
            "dirty": self.dirty,
            "code_digest": self.code_digest,
            "image_digest": self.image_digest,
            "robots_cgroup": None if self.robots_cgroup is None else asdict(self.robots_cgroup),
        }
        return _file_text(document, ENVIRONMENT_FILE)

    @classmethod
    def from_json(cls, text: str) -> RunEnvironment:
        own_keys = [field.name for field in fields(cls) if field.name != "inside"]
        document = _Fields(
            _parse(text, ENVIRONMENT_FILE), ENVIRONMENT_FILE, (*_INSIDE_KEYS, *own_keys)
        )
        return cls(
            inside=_read_inside(document),
            host_memory_total_bytes=document.read("host_memory_total_bytes", _integer),
            engine=document.read("engine", _string),
            engine_version=document.read("engine_version", _string),
            commit=document.read("commit", _string),
            dirty=document.read("dirty", _boolean),
            code_digest=document.read("code_digest", _string),
            image_digest=document.read("image_digest", _string),
            robots_cgroup=document.read("robots_cgroup", _optional(_read_limits)),
        )


_SUITE_GATE = "a tests gate"


@dataclass(frozen=True)
class SuiteGate:
    """The tests gate: the result of the test suite inside one image, for one code digest.

    It is the tests_passed object of gate.json, and the content of the file under gates/
    that gate_file names, beside the output that output_sha256 is the hash of.
    """

    passed: bool
    image_digest: str
    code_digest: str
    commit: str
    dirty: bool
    passed_count: int
    failed_count: int
    skipped_count: int
    deselected_count: int
    gate_file: str
    output_sha256: str

    def to_json(self) -> str:
        return _file_text(asdict(self), _SUITE_GATE)

    @classmethod
    def from_json(cls, text: str) -> SuiteGate:
        return _read_suite_gate(_parse(text, _SUITE_GATE), _SUITE_GATE)


_read_suite_gate = _record(SuiteGate)


@dataclass(frozen=True)
class ImageGate:
    """Whether the image was built in this invocation and every container of the run has it."""

    built: bool
    digest: str
    containers: Mapping[str, str]  # each container's name and the digest of its image


@dataclass(frozen=True)
class GatedLimits:
    """The limits the gate compares; None for one that was not read or is not reported."""

    cpu_limit_cores: float | None
    memory_limit_bytes: int | None
    memory_swap_limit_bytes: int | None


_read_gated_limits = _record(GatedLimits)


@dataclass(frozen=True)
class LimitsGate:
    ok: bool
    expected: GatedLimits
    read: GatedLimits


@dataclass(frozen=True)
class FreshGate:
    """Whether every container of the run was new and no connection came before t0."""

    ok: bool
    new_containers: bool
    # None where no gateway ran, or where the scrape before t0 failed.
    accepted_before_t0: int | None


_read_fresh_gate = _record(FreshGate)


@dataclass(frozen=True)
class ConfigGate:
    """Whether the gateway's config echo is the run spec's server config."""

    ok: bool
    differences: tuple[str, ...]  # the dotted paths whose values differ


@dataclass(frozen=True)
class Gate:
    """gate.json: the gates of one run. A run counts only when it is gated."""

    tests_passed: SuiteGate
    image_built: ImageGate
    limits_ok: LimitsGate
    fresh: FreshGate
    # None for the model benchmark, which starts no gateway.
    config_ok: ConfigGate | None

    @property
    def gated(self) -> bool:
        """Whether every gate that applies to the run passed."""
        return (
            self.tests_passed.passed
            and self.image_built.built
            and self.limits_ok.ok
            and self.fresh.ok
            and (self.config_ok is None or self.config_ok.ok)
        )

    def to_json(self) -> str:
        document = {
            "tests_passed": asdict(self.tests_passed),
            "image_built": {
                "built": self.image_built.built,
                "digest": self.image_built.digest,
                "containers": dict(self.image_built.containers),
            },
            "limits_ok": asdict(self.limits_ok),
            "fresh": asdict(self.fresh),
            "config_ok": None if self.config_ok is None else asdict(self.config_ok),
            "gated": self.gated,
        }
        return _file_text(document, GATE_FILE)

    @classmethod
    def from_json(cls, text: str) -> Gate:
        document = _Fields(
            _parse(text, GATE_FILE),
            GATE_FILE,
            ("tests_passed", "image_built", "limits_ok", "fresh", "config_ok", "gated"),
        )
        gate = cls(
            tests_passed=document.read("tests_passed", _read_suite_gate),
            image_built=document.read("image_built", _read_image_gate),
            limits_ok=document.read("limits_ok", _read_limits_gate),
            fresh=document.read("fresh", _read_fresh_gate),
            config_ok=document.read("config_ok", _optional(_read_config_gate)),
        )
        if document.read("gated", _boolean) != gate.gated:
            raise RunFileError(
                f"{GATE_FILE}: gated must be true exactly when every gate that is not null passed"
            )
        return gate


def _read_image_gate(value: object, path: str) -> ImageGate:
    document = _Fields(value, path, ("built", "digest", "containers"))
    return ImageGate(
        built=document.read("built", _boolean),
        digest=document.read("digest", _string),
        containers=document.read("containers", _mapping(_string)),
    )


def _read_limits_gate(value: object, path: str) -> LimitsGate:
    document = _Fields(value, path, ("ok", "expected", "read"))
    return LimitsGate(
        ok=document.read("ok", _boolean),
        expected=document.read("expected", _read_gated_limits),
        read=document.read("read", _read_gated_limits),
    )


def _read_config_gate(value: object, path: str) -> ConfigGate:
    document = _Fields(value, path, ("ok", "differences"))
    return ConfigGate(
        ok=document.read("ok", _boolean),
        differences=document.read("differences", _sequence(_string)),
    )


def done_files(files: Mapping[str, bytes]) -> dict[str, str]:
    """Return the SHA-256 of every file that DONE covers, from a run's files by their paths.

    A path is relative to the run directory, with "/" between its parts. The samples, the
    summary and DONE itself are left out: the samples are not kept with the results, and
    the summary can be written again after the run. Every other file counts, so a file
    that a writer leaves behind in a finished run makes its DONE invalid.
    """
    return {
        path: hashlib.sha256(content).hexdigest()
        for path, content in sorted(files.items())
        if path not in (SUMMARY_FILE, DONE_FILE) and not path.startswith(f"{SAMPLES_DIRECTORY}/")
    }


@dataclass(frozen=True)
class Done:
    """DONE: the mark of a finished run, with the hashes that make it checkable."""

    files: Mapping[str, str]
    # The hash of summary.json; None until a summary is written, and the only part of the
    # file that is ever written again.
    summary_sha256: str | None

    def valid_for(self, files: Mapping[str, bytes]) -> bool:
        """Whether the run's files, by their paths, are exactly the ones this DONE recorded."""
        return dict(self.files) == done_files(files)

    def to_json(self) -> str:
        document = {"files": dict(self.files), "summary_sha256": self.summary_sha256}
        return _file_text(document, DONE_FILE)

    @classmethod
    def from_json(cls, text: str) -> Done:
        document = _Fields(_parse(text, DONE_FILE), DONE_FILE, ("files", "summary_sha256"))
        return cls(
            files=document.read("files", _mapping(_string)),
            summary_sha256=document.read("summary_sha256", _optional(_string)),
        )


@dataclass(frozen=True)
class Fact:
    """One number or string of the design document, combined from its counted runs."""

    # None for a key with no counted run, or one computed across runs that has no answer.
    value: int | float | str | bool | None
    unit: str
    runs: int
    # The lowest and the highest run value; None where the fact has no value per run.
    min: int | float | None
    max: int | float | None
    run_ids: tuple[str, ...]
    dirty_runs: int
    failed_checks: Mapping[str, tuple[str, ...]]  # a run id and the checks that failed in it
    attempts: int
    failed_attempts: int


def _fact_document(fact: Fact) -> dict[str, object]:
    return {
        "value": fact.value,
        "unit": fact.unit,
        "runs": fact.runs,
        "min": fact.min,
        "max": fact.max,
        "run_ids": list(fact.run_ids),
        "dirty_runs": fact.dirty_runs,
        "failed_checks": {run_id: list(checks) for run_id, checks in fact.failed_checks.items()},
        "attempts": fact.attempts,
        "failed_attempts": fact.failed_attempts,
    }


def _fact_value(value: object, path: str) -> int | float | str | bool | None:
    if value is not None and not isinstance(value, bool | int | float | str):
        raise RunFileError(f"{path} must be a number, a string, true, false or null")
    return value


def _read_fact(value: object, path: str) -> Fact:
    document = _Fields(value, path, [field.name for field in fields(Fact)])
    return Fact(
        value=document.read("value", _fact_value),
        unit=document.read("unit", _string),
        runs=document.read("runs", _integer),
        min=document.read("min", _optional(_number_as_written)),
        max=document.read("max", _optional(_number_as_written)),
        run_ids=document.read("run_ids", _sequence(_string)),
        dirty_runs=document.read("dirty_runs", _integer),
        failed_checks=document.read("failed_checks", _mapping(_sequence(_string))),
        attempts=document.read("attempts", _integer),
        failed_attempts=document.read("failed_attempts", _integer),
    )


@dataclass(frozen=True)
class Facts:
    """facts.json: every fact, with the model set and the code its runs were counted for."""

    model_set: ModelSet
    code_digest: str
    # The evidence runs behind the facts and the comparison runs behind the ledger.
    counted_run_ids: tuple[str, ...]
    facts: Mapping[str, Fact]

    def to_json(self) -> str:
        document = {
            "model_set": self.model_set.value,
            "code_digest": self.code_digest,
            "counted_run_ids": list(self.counted_run_ids),
            "facts": {key: _fact_document(fact) for key, fact in self.facts.items()},
        }
        return _file_text(document, FACTS_FILE)

    @classmethod
    def from_json(cls, text: str) -> Facts:
        document = _Fields(
            _parse(text, FACTS_FILE),
            FACTS_FILE,
            ("model_set", "code_digest", "counted_run_ids", "facts"),
        )
        return cls(
            model_set=document.read("model_set", _member(ModelSet)),
            code_digest=document.read("code_digest", _string),
            counted_run_ids=document.read("counted_run_ids", _sequence(_string)),
            facts=document.read("facts", _mapping(_read_fact)),
        )
