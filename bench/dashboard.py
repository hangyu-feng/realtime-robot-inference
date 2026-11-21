"""The dashboard: a run second by second in a browser, replayed from a counted run or live.

    python -m bench.dashboard [--replay KEY]

It serves http://127.0.0.1:8090/ on the host. It is an observation tool outside the system:
nothing in the container starts it. Live, it polls the gateway's /metrics at each whole
second of the monotonic clock. With --replay it plays the first counted single run of a
serving scenario from results/, releasing one successful scrape a second from the moment it
starts, and touches no service. Both sources feed one series of points, each taken from two
consecutive snapshots; GET /data returns the series as JSON and GET / the page that draws it.

It exits 0 on Ctrl-C, 1 when it cannot start (a key with no counted run, a run whose files
cannot be read, a port in use, a missing page file) and 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
import threading
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from bench.render import RenderError, read_facts
from bench.run import (
    DEFAULT_METRICS_URL,
    SCRAPE_TIMEOUT_S,
    MetricsUnavailable,
    fetch_metrics,
    next_scrape_k,
)
from bench.runfiles import EVENTS_FILE, SCRAPES_FILE, Event, EventName, RunFileError, Scrape
from bench.tables import RAW_DIRECTORY, TablesError, topology_runs
from cloud.config import Topology
from common.clock import Clock, MonotonicClock
from common.commands import MODEL_KEYS
from common.histogram import Histogram, difference

HOST = "127.0.0.1"
PORT = 8090
# Two minutes at one point a second: every demo but the soak fits whole.
HISTORY_POINTS = 120
LOOP_LAG_QUANTILE = 0.99
FRESHNESS_QUANTILE = 0.95
PAGE_FILE = Path(__file__).with_name("dashboard.html")
STATUS_OK = "ok"
STATUS_FINISHED = "replay finished"
STATUS_NO_SNAPSHOT = "no snapshot yet"
# The processes of /metrics whose CPU time the page draws (docs/metrics.md 10.7).
PROCESS_NAMES = ("gateway", *(f"worker_{model}" for model in MODEL_KEYS))
# A source waits at most one scrape's timeout before it sees that it must stop.
SOURCE_JOIN_TIMEOUT_S = 2 * SCRAPE_TIMEOUT_S

_NS_PER_S = 1_000_000_000
_US_PER_S = 1_000_000
_MS_PER_S = 1_000
# Counters stay within a signed 64-bit integer, as the gateway keeps them.
_MAX_COUNT = 2**63 - 1
_RESULTS = Path(__file__).resolve().parent.parent / "results"

# A source: it feeds the series until the event is set, or until it has nothing left.
type _Player = Callable[[threading.Event], None]


class ViewError(Exception):
    """A snapshot cannot follow the previous one: a field is missing or a counter went down."""


class DashboardError(Exception):
    """The dashboard cannot start: no counted run to replay, an unreadable run, port or page."""


@dataclass(frozen=True)
class ProcessReading:
    pid: int
    cpu_ms: int  # user plus system time


@dataclass(frozen=True)
class ModelReading:
    frames: int
    batches: int
    stale_ms: float  # the configured threshold, config.models.<m>.stale_ms
    worker_state: str
    worker_restarts: int


@dataclass(frozen=True)
class RobotModelReading:
    commands_sent: int
    frames_overwritten: int


@dataclass(frozen=True)
class ReportReading:
    """One model's part of a robot's last telemetry report."""

    freshness: Histogram
    commands_stale: int


@dataclass(frozen=True)
class RobotReading:
    session: int
    connected: bool
    # The models this id has subscribed to in any session.
    models: Mapping[str, RobotModelReading]
    # None before the session's first report.
    telemetry: Mapping[str, ReportReading] | None


@dataclass(frozen=True)
class Snapshot:
    """A /metrics object as the view reads it (docs/metrics.md)."""

    ts_mono_ns: int
    connections: int
    memory_bytes: int | None
    cpu_limit_cores: float | None
    cpu_usage_us: int | None
    loop_lag: Histogram
    # By the names of PROCESS_NAMES; None while a worker has no live process.
    processes: Mapping[str, ProcessReading | None]
    models: Mapping[str, ModelReading]
    robots: Mapping[str, RobotReading]


@dataclass(frozen=True)
class ModelPoint:
    served_per_s: float
    overwritten_per_s: float
    mean_batch_size: float | None
    stale_ms: float
    worker_state: str
    worker_restarts: int


@dataclass(frozen=True)
class RobotModelPoint:
    commands_per_s: float
    freshness_p95_ms: float | None
    commands_stale: int | None


@dataclass(frozen=True)
class RobotPoint:
    connected: bool
    models: Mapping[str, RobotModelPoint]


@dataclass(frozen=True)
class Point:
    """What the gateway did between two consecutive snapshots, at the later one's time."""

    t_s: float
    connections: int
    memory_bytes: int | None
    loop_lag_p99_ms: float | None
    cpu_limit_cores: float | None
    # container, then each of PROCESS_NAMES, in cores.
    cpu_cores: Mapping[str, float | None]
    models: Mapping[str, ModelPoint]
    robots: Mapping[str, RobotPoint]
    events: tuple[str, ...]


def _whole(value: object) -> int:
    # A bool is an int to isinstance, and true is not a count.
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= _MAX_COUNT:
        raise ValueError(f"{value!r} is not a whole number in [0, 2**63 - 1]")
    return value


def _number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{value!r} is not a number")
    # json.loads reads NaN, Infinity and 1e999, which strict JSON cannot carry back out.
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{value!r} is not finite")
    return number


def _text(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{value!r} is not a string")
    return value


def _flag(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{value!r} is not true or false")
    return value


def _process(entry: Any) -> ProcessReading | None:
    if entry is None:
        return None
    cpu_ms = _whole(entry["cpu_user_ms"]) + _whole(entry["cpu_system_ms"])
    return ProcessReading(_whole(entry["pid"]), cpu_ms)


def _model(metrics: Mapping[str, Any], model: str) -> ModelReading:
    counters, worker = metrics["models"][model], metrics["workers"][model]
    return ModelReading(
        frames=_whole(counters["frames"]),
        batches=_whole(counters["batches"]),
        stale_ms=_number(metrics["config"]["models"][model]["stale_ms"]),
        worker_state=_text(worker["state"]),
        worker_restarts=_whole(worker["restarts"]),
    )


def _robot(entry: Any) -> RobotReading:
    # Absent until the session's first report (docs/metrics.md 10.4).
    telemetry = entry.get("telemetry")
    return RobotReading(
        session=_whole(entry["session"]),
        connected=_flag(entry["connected"]),
        models={
            _text(model): RobotModelReading(
                _whole(counters["commands_sent"]), _whole(counters["frames_overwritten"])
            )
            for model, counters in entry["models"].items()
        },
        telemetry=None
        if telemetry is None
        else {
            _text(model): ReportReading(
                Histogram.from_dict(fields["freshness_ms"]), _whole(fields["commands_stale"])
            )
            for model, fields in telemetry["models"].items()
        },
    )


def _snapshot(metrics: Mapping[str, Any]) -> Snapshot:
    cgroup = metrics["cgroup"]
    memory_bytes, cpu_limit_cores = cgroup["memory_current_bytes"], cgroup["cpu_limit_cores"]
    cpu_usage_us = cgroup["cpu_usage_us"]
    return Snapshot(
        ts_mono_ns=_whole(metrics["ts_mono_ns"]),
        connections=_whole(metrics["connections"]["current"]),
        memory_bytes=None if memory_bytes is None else _whole(memory_bytes),
        cpu_limit_cores=None if cpu_limit_cores is None else _number(cpu_limit_cores),
        cpu_usage_us=None if cpu_usage_us is None else _whole(cpu_usage_us),
        loop_lag=Histogram.from_dict(metrics["loop"]["lag_ms"]),
        processes={name: _process(metrics["processes"][name]) for name in PROCESS_NAMES},
        models={model: _model(metrics, model) for model in MODEL_KEYS},
        robots={_text(robot_id): _robot(entry) for robot_id, entry in metrics["robots"].items()},
    )


def read_snapshot(metrics: Mapping[str, Any]) -> Snapshot:
    """Return a /metrics object as the view reads it; ViewError when it lacks a field it reads."""
    try:
        return _snapshot(metrics)
    except (KeyError, TypeError, AttributeError, ValueError, OverflowError) as error:
        # A missing key, a list or a null where an object belongs, and a value of the wrong
        # type; float() of an int beyond the float range overflows.
        raise ViewError(
            f"the snapshot is not /metrics as docs/metrics.md describes it: {error!r}"
        ) from error


def _rise(later: int, earlier: int, name: str) -> int:
    if later < earlier:
        raise ViewError(f"{name} went down from {earlier} to {later}")
    return later - earlier


def _window_quantile(later: Histogram, earlier: Histogram, q: float, name: str) -> float | None:
    """Return the quantile of what later recorded after earlier; None when it recorded nothing."""
    try:
        window = difference(later, earlier)
    except ValueError as error:
        raise ViewError(f"{name} went down: {error}") from error
    return window.quantile_ms(q) if window.count else None


def _cpu_cores(earlier: Snapshot, later: Snapshot, elapsed_s: float) -> dict[str, float | None]:
    container = (
        None
        if earlier.cpu_usage_us is None or later.cpu_usage_us is None
        else _rise(later.cpu_usage_us, earlier.cpu_usage_us, "cgroup.cpu_usage_us")
        / _US_PER_S
        / elapsed_s
    )
    cores: dict[str, float | None] = {"container": container}
    for name in PROCESS_NAMES:
        before, after = earlier.processes[name], later.processes[name]
        # A new process's counters start again from zero.
        if before is None or after is None or before.pid != after.pid:
            cores[name] = None
            continue
        used_ms = _rise(after.cpu_ms, before.cpu_ms, f"the CPU time of processes.{name}")
        cores[name] = used_ms / _MS_PER_S / elapsed_s
    return cores


def _overwritten(earlier: Snapshot, later: Snapshot, model: str) -> int:
    """Return the frames of the model overwritten in the slots of every robot between the two."""
    total = 0
    for robot_id, robot in later.robots.items():
        counters = robot.models.get(model)
        if counters is None:
            continue
        before = earlier.robots.get(robot_id)
        # A robot absent before, or new to the model, had none overwritten.
        before_counters = None if before is None else before.models.get(model)
        total += _rise(
            counters.frames_overwritten,
            0 if before_counters is None else before_counters.frames_overwritten,
            f"robots.{robot_id}.models.{model}.frames_overwritten",
        )
    return total


def _model_point(earlier: Snapshot, later: Snapshot, model: str, elapsed_s: float) -> ModelPoint:
    before, after = earlier.models[model], later.models[model]
    frames = _rise(after.frames, before.frames, f"models.{model}.frames")
    batches = _rise(after.batches, before.batches, f"models.{model}.batches")
    return ModelPoint(
        served_per_s=frames / elapsed_s,
        overwritten_per_s=_overwritten(earlier, later, model) / elapsed_s,
        mean_batch_size=frames / batches if batches else None,
        stale_ms=after.stale_ms,
        worker_state=after.worker_state,
        worker_restarts=after.worker_restarts,
    )


def _robot_model_point(
    robot_id: str, earlier: RobotReading | None, later: RobotReading, model: str, elapsed_s: float
) -> RobotModelPoint:
    before = None if earlier is None else earlier.models.get(model)
    sent = _rise(
        later.models[model].commands_sent,
        0 if before is None else before.commands_sent,
        f"robots.{robot_id}.models.{model}.commands_sent",
    )
    report = None if later.telemetry is None else later.telemetry.get(model)
    if report is None:
        return RobotModelPoint(sent / elapsed_s, None, None)
    # Telemetry restarts with each session (docs/metrics.md 10.4), so the window of a new
    # session starts from an empty histogram.
    previous_report = None
    if earlier is not None and earlier.session == later.session and earlier.telemetry is not None:
        previous_report = earlier.telemetry.get(model)
    previous = Histogram() if previous_report is None else previous_report.freshness
    freshness_p95_ms = _window_quantile(
        report.freshness,
        previous,
        FRESHNESS_QUANTILE,
        f"robots.{robot_id}.telemetry.models.{model}.freshness_ms",
    )
    return RobotModelPoint(sent / elapsed_s, freshness_p95_ms, report.commands_stale)


def _events(earlier: Snapshot, later: Snapshot) -> tuple[str, ...]:
    """Return the robots that disconnected and connected, and the workers that restarted."""

    def connected_in(snapshot: Snapshot, robot_id: str, session: int) -> bool:
        robot = snapshot.robots.get(robot_id)
        return robot is not None and robot.connected and robot.session == session

    disconnected = [
        f"robot {robot_id} disconnected"
        for robot_id, robot in sorted(earlier.robots.items())
        if robot.connected and not connected_in(later, robot_id, robot.session)
    ]
    connected = [
        f"robot {robot_id} connected"
        for robot_id, robot in sorted(later.robots.items())
        if robot.connected and not connected_in(earlier, robot_id, robot.session)
    ]
    restarted = [
        f"worker {model} restarted"
        for model in MODEL_KEYS
        if _rise(
            later.models[model].worker_restarts,
            earlier.models[model].worker_restarts,
            f"workers.{model}.restarts",
        )
    ]
    return (*disconnected, *connected, *restarted)


def point(earlier: Snapshot, later: Snapshot, origin_ns: int) -> Point:
    """Return the point of two consecutive snapshots, at the later one's time from origin_ns.

    ViewError when the gateway's clock did not advance between them or a counter went down:
    later is then not the next snapshot of the same gateway.
    """
    elapsed_ns = later.ts_mono_ns - earlier.ts_mono_ns
    if elapsed_ns <= 0:
        raise ViewError(f"the gateway's clock moved by {elapsed_ns} ns between two snapshots")
    elapsed_s = elapsed_ns / _NS_PER_S
    return Point(
        t_s=(later.ts_mono_ns - origin_ns) / _NS_PER_S,
        connections=later.connections,
        memory_bytes=later.memory_bytes,
        loop_lag_p99_ms=_window_quantile(
            later.loop_lag, earlier.loop_lag, LOOP_LAG_QUANTILE, "loop.lag_ms"
        ),
        cpu_limit_cores=later.cpu_limit_cores,
        cpu_cores=_cpu_cores(earlier, later, elapsed_s),
        models={model: _model_point(earlier, later, model, elapsed_s) for model in MODEL_KEYS},
        robots={
            robot_id: RobotPoint(
                robot.connected,
                {
                    model: _robot_model_point(
                        robot_id, earlier.robots.get(robot_id), robot, model, elapsed_s
                    )
                    for model in robot.models
                },
            )
            for robot_id, robot in later.robots.items()
        },
        events=_events(earlier, later),
    )


class Series:
    """The latest points of a run and why the last snapshot is missing, if it is.

    It takes one input at a time, a snapshot or a failure, and is the only place that decides
    what a failure does: the reason becomes the status and the previous snapshot is
    forgotten, so the next snapshot gives no point and the one after it does. A gateway
    restart then shows as a gap and never as a negative rate.
    """

    def __init__(self, origin_ns: int | None) -> None:
        # None: the first snapshot's clock reading becomes the origin.
        self._origin_ns = origin_ns
        self._previous: Snapshot | None = None
        # Bounded: when it is full, the oldest point is dropped.
        self._points: deque[Point] = deque(maxlen=HISTORY_POINTS)
        self.status = STATUS_NO_SNAPSHOT

    @property
    def points(self) -> tuple[Point, ...]:
        return tuple(self._points)

    def add_snapshot(self, metrics: Mapping[str, Any]) -> None:
        try:
            snapshot = read_snapshot(metrics)
            if self._origin_ns is None:
                self._origin_ns = snapshot.ts_mono_ns
            if self._previous is not None:
                self._points.append(point(self._previous, snapshot, self._origin_ns))
        except ViewError as error:
            self.add_failure(str(error))
            return
        self._previous = snapshot
        self.status = STATUS_OK

    def add_failure(self, reason: str) -> None:
        self._previous = None
        self.status = reason

    def finish(self) -> None:
        self.status = STATUS_FINISHED


def data_document(source: str, status: str, points: Sequence[Point]) -> bytes:
    """Return the body of GET /data: the source, the status and the points, as strict JSON."""
    document = {"source": source, "status": status, "points": [asdict(item) for item in points]}
    return json.dumps(document, allow_nan=False, separators=(",", ":")).encode("utf-8")


def replay_run_id(runs: Mapping[tuple[str, Topology], Sequence[str]], key: str) -> str:
    """Return the first counted single run of a scenario key, from what topology_runs returns.

    DashboardError for a key with none, which names the keys that have one.
    """
    run_ids = runs.get((key, Topology.SINGLE), ())
    if run_ids:
        return run_ids[0]
    keys = [
        scenario
        for (scenario, topology), counted in runs.items()
        if topology is Topology.SINGLE and counted
    ]
    raise DashboardError(
        f"{key!r} has no counted single run to replay; the keys that have one: "
        f"{', '.join(keys) or 'none'}"
    )


@dataclass(frozen=True)
class StoredRun:
    """What a replay plays: the run's t0 and its successful scrapes' /metrics, in order."""

    t0_mono_ns: int
    snapshots: tuple[Mapping[str, Any], ...]


def read_run(run_dir: Path) -> StoredRun:
    """Return a run directory's t0 and successful scrapes; DashboardError if they are unreadable."""
    try:
        with (run_dir / EVENTS_FILE).open(encoding="utf-8") as file:
            first = Event.from_json(file.readline())
        with gzip.open(run_dir / SCRAPES_FILE, "rt", encoding="utf-8") as file:
            scrapes = [Scrape.from_json(line) for line in file]
    except (OSError, EOFError, ValueError, RunFileError) as error:
        # A file that is not gzip is an OSError, one cut short an EOFError, and bytes that are
        # not UTF-8 a ValueError.
        raise DashboardError(f"{run_dir} cannot be read: {error}") from error
    if first.name is not EventName.T0 or first.t0_mono_ns is None:
        raise DashboardError(f"the first line of {run_dir / EVENTS_FILE} is not the t0 event")
    return StoredRun(
        first.t0_mono_ns, tuple(scrape.metrics for scrape in scrapes if scrape.metrics is not None)
    )


@dataclass
class _Shared:
    """The series and its source, written by the source thread and read by the handlers."""

    source: str
    series: Series
    lock: threading.Lock = field(default_factory=threading.Lock)

    def document(self) -> bytes:
        with self.lock:
            status, points = self.series.status, self.series.points
        return data_document(self.source, status, points)


def _wait_for_second(k: int, t0_ns: int, clock: Clock, stop: threading.Event) -> bool:
    """Wait until second k from t0_ns; return False when told to stop first."""
    due_ns = t0_ns + k * _NS_PER_S
    # Re-checked against the clock: a wait may end early.
    while (remaining_ns := due_ns - clock.now_ns()) > 0:
        if stop.wait(remaining_ns / _NS_PER_S):
            return False
    return True


def _play_live(shared: _Shared, clock: Clock, stop: threading.Event) -> None:
    """Poll /metrics at each whole second of the monotonic clock until told to stop."""
    k = next_scrape_k(-1, 0, clock.now_ns())
    while _wait_for_second(k, 0, clock, stop):
        try:
            metrics = fetch_metrics(DEFAULT_METRICS_URL)
        except MetricsUnavailable as error:
            with shared.lock:
                shared.series.add_failure(f"gateway not reachable: {error}")
        else:
            with shared.lock:
                shared.series.add_snapshot(metrics)
        k = next_scrape_k(k, 0, clock.now_ns())


def _play_replay(shared: _Shared, run: StoredRun, clock: Clock, stop: threading.Event) -> None:
    """Release one snapshot a second from now, then mark the replay finished."""
    start_ns, k = clock.now_ns(), 0
    for metrics in run.snapshots:
        if not _wait_for_second(k, start_ns, clock, stop):
            return
        with shared.lock:
            shared.series.add_snapshot(metrics)
        k = next_scrape_k(k, start_ns, clock.now_ns())
    with shared.lock:
        shared.series.finish()


def _feed(play: _Player, shared: _Shared, stop: threading.Event) -> None:
    try:
        play(stop)
    except Exception as error:
        # The boundary of the source thread: the page shows why it stopped and the server
        # stays up.
        with shared.lock:
            shared.series.add_failure(f"the source stopped: {error!r}")
        print(f"bench.dashboard: the source stopped: {error!r}", file=sys.stderr, flush=True)


def _handler(shared: _Shared, page: bytes) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                self._send(HTTPStatus.OK, "text/html; charset=utf-8", page)
            elif self.path == "/data":
                self._send(HTTPStatus.OK, "application/json", shared.document())
            else:
                self._send(HTTPStatus.NOT_FOUND, "text/plain; charset=utf-8", b"not found\n")

        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            try:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            except ConnectionError:
                # A page closed mid-reply; the next request is another page's.
                return

        def log_message(self, format: str, *args: Any) -> None:
            # No line per request: the page asks once a second.
            return

    return Handler


def _replay_source(key: str) -> tuple[str, StoredRun]:
    """Return the name and the content of a key's first counted single run under results/."""
    try:
        runs = topology_runs(read_facts(_RESULTS))
    except (RenderError, TablesError) as error:
        raise DashboardError(f"the counted runs cannot be read: {error}") from error
    run_id = replay_run_id(runs, key)
    return f"replay of {key} run {run_id}", read_run(_RESULTS / RAW_DIRECTORY / key / run_id)


def _start(key: str | None) -> tuple[_Shared, _Player, ThreadingHTTPServer]:
    try:
        page = PAGE_FILE.read_bytes()
    except OSError as error:
        raise DashboardError(f"the page {PAGE_FILE} cannot be read: {error}") from error
    clock = MonotonicClock()
    play: _Player
    if key is None:
        shared = _Shared(f"live from {DEFAULT_METRICS_URL}", Series(None))
        play = partial(_play_live, shared, clock)
    else:
        source, run = _replay_source(key)
        shared = _Shared(source, Series(run.t0_mono_ns))
        play = partial(_play_replay, shared, run, clock)
    try:
        server = ThreadingHTTPServer((HOST, PORT), _handler(shared, page))
    except OSError as error:
        raise DashboardError(f"cannot serve on {HOST}:{PORT}: {error.strerror}") from error
    return shared, play, server


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.dashboard",
        description=f"Serve the dashboard at http://{HOST}:{PORT}/, live or replaying a run.",
    )
    parser.add_argument(
        "--replay", metavar="KEY", help="replay the first counted single run of a scenario key"
    )
    arguments = parser.parse_args(argv)
    try:
        shared, play, server = _start(arguments.replay)
    except DashboardError as error:
        print(f"bench.dashboard: {error}", file=sys.stderr)
        return 1
    print(f"bench.dashboard: serving http://{HOST}:{PORT}/, {shared.source}", flush=True)
    stop = threading.Event()
    source = threading.Thread(target=_feed, args=(play, shared, stop), name="source", daemon=True)
    source.start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        source.join(timeout=SOURCE_JOIN_TIMEOUT_S)
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
