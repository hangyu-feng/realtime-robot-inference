"""The scenario driver: one serving run, played in the robots' container, and its scraper.

    python -m bench.run --run-dir DIR [--scenario KEY] [--url URL] [--metrics-url URL]
        [--robots-cgroup]

The evidence runner writes a run's config.json, then starts the driver in the container that
holds the robots, with the host user's ids. The driver describes its container in
env_inside.json, sets time zero, writes each fleet process's plan, starts the processes,
kills a robot that the scenario kills at its planned time, and writes events.jsonl. A thread
beside it scrapes /metrics and the fleet's processes once a second into scrapes.jsonl.gz.
After the fleet has left, it waits for the gateway's last counts and takes the final scrape.

Run by hand against a running service, with --scenario and a directory that has no
config.json, it writes config.json itself, as a hand run that nothing counts, and ends by
printing each robot's commands and freshness per model.

It exits 0 when every fleet process exited 0 and every killed one died of SIGKILL, 1 when
one did not or the run could not be played, and 2 for bad arguments or a directory that
already holds a run.
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import IO, Any

import numpy as np
import numpy.typing as npt

from bench.environment import collect_environment
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    EVENTS_FILE,
    INSIDE_ENVIRONMENT_FILE,
    PLANS_DIRECTORY,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    Event,
    EventName,
    FleetConfig,
    InsideEnvironment,
    RunConfig,
    RunFileError,
    RunSpec,
    Scrape,
)
from bench.runplan import FleetProcess, fleet_plans, planned_events, scenario_for
from bench.scenarios import Scenario
from bench.stats import summarize
from common.cgroup import CgroupError, CgroupStats, read_stats
from common.clock import Clock, MonotonicClock
from common.commands import MODEL_KEYS
from common.procstat import ProcessGoneError, ProcessStats, read_process_stats
from robot.control import Verdict
from robot.frames import DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from robot.samples import VERDICTS, load_samples
from robot.settings import LinkSettings, RobotSettings

type Rows = npt.NDArray[np.void]

# Every fleet process builds its frame set and allocates its sample arrays before time zero.
FLEET_LEAD_S = 4.0
# A scrape that takes longer is recorded as failed: its reading of the gateway's clock could
# lie anywhere in the wait, and the next scrape is due a second after this one was.
SCRAPE_TIMEOUT_S = 1.0
# The first scrape is one second before time zero: the gateway before any robot connects.
FIRST_SCRAPE_K = -1
# After the fleet, /metrics is polled this often, for at most this long, until the gateway
# shows every robot gone and its last counts in: the poll of a peer the driver does not
# control is bounded.
FINAL_POLL_INTERVAL_S = 0.1
FINAL_POLL_LIMIT_S = 10.0
# Every fleet process ends at the end of the run; one that has not ended this long after it
# is hung.
FLEET_EXIT_MARGIN_S = 60.0
# The scraper's last scrape is due at the end of the run, before the fleet has ended, so
# joining it waits at most for one scrape.
SCRAPER_JOIN_TIMEOUT_S = 10.0
# A killed or stopped process is reaped at once; this only bounds a wait the kernel ends.
REAP_TIMEOUT_S = 5.0
# The gateway in this container, at its default port.
DEFAULT_METRICS_URL = "http://127.0.0.1:8080/metrics"
# The directory of the model files, as the image sets it. Where it is, the models are hashed.
MODEL_DIR_VARIABLE = "MODEL_DIR"
# A hand run has no spec: its robots take the fleet's defaults, its seed among them.
HAND_RUN_SEED = 0
HAND_RUN_PHASE = "random"
# The name robot.fleet gives each robot's samples file.
SAMPLES_FILE = "robot_{robot_id}.npz"

_NS_PER_S = 1_000_000_000
_REPOSITORY = Path(__file__).resolve().parent.parent


class DriverError(Exception):
    """The run cannot be played: its config, a file it writes, or a process failed."""


class RunDirectoryError(DriverError):
    """The run directory cannot take this run: it holds one, or has no config to play."""


class MetricsUnavailable(Exception):
    """/metrics did not answer with a JSON object."""


def fleet_config(scenario: Scenario) -> FleetConfig:
    """Return the robots of a run: the scenario's timeline and what every robot declares.

    Each robot runs with the default settings of robot.settings.
    """
    settings = RobotSettings()
    return FleetConfig(
        scenario=scenario,
        frame_rate_hz=settings.frame_rate_hz,
        models=settings.models,
        stale_ms=dict(settings.stale_ms),
        watchdog_ms=dict(settings.watchdog_ms),
        planned_events=planned_events(scenario),
    )


def hand_run_config(scenario: Scenario) -> RunConfig:
    """Return the config of a run started by hand: no spec, so no run id and no count."""
    return RunConfig(
        spec=None,
        run_id=None,
        seed=HAND_RUN_SEED,
        extra_compose_path=None,
        fleet=fleet_config(scenario),
    )


def plan_path(run_dir: Path, process: FleetProcess) -> Path:
    return run_dir / PLANS_DIRECTORY / f"{process.name}.json"


def _pairs(values: Mapping[str, float]) -> str:
    return ",".join(f"{model}={value!r}" for model, value in values.items())


def fleet_command(
    process: FleetProcess,
    run_dir: Path,
    url: str,
    fleet: FleetConfig,
    spec: RunSpec | None,
    python: str,
) -> list[str]:
    """Return the command line of one fleet process of the run.

    The phases and the link model are the spec's; a hand run takes the fleet's defaults.
    """
    if spec is None:
        phase, uplink, downlink = HAND_RUN_PHASE, LinkSettings(), LinkSettings()
    elif spec.phase_mode is None:
        raise DriverError(f"the spec of {spec.scenario_key} runs no robots")
    else:
        phase, uplink, downlink = spec.phase_mode, spec.uplink, spec.downlink
    return [
        python,
        "-m",
        "robot.fleet",
        "--url",
        url,
        "--plan",
        str(plan_path(run_dir, process)),
        "--samples-dir",
        str(run_dir / SAMPLES_DIRECTORY),
        "--seed",
        str(process.seed),
        "--phase",
        phase,
        "--frame-rate-hz",
        repr(fleet.frame_rate_hz),
        "--models",
        ",".join(fleet.models),
        "--stale-ms",
        _pairs(fleet.stale_ms),
        "--watchdog-ms",
        _pairs(fleet.watchdog_ms),
        "--uplink-delay-ms",
        repr(uplink.delay_ms),
        "--uplink-jitter-ms",
        repr(uplink.jitter_ms),
        "--downlink-delay-ms",
        repr(downlink.delay_ms),
        "--downlink-jitter-ms",
        repr(downlink.jitter_ms),
    ]


def kill_targets(processes: Sequence[FleetProcess], scenario: Scenario) -> dict[str, str]:
    """Return, for each robot that the scenario kills, the fleet process that runs it alone."""
    killed = {robot.robot_id for robot in scenario.robots if robot.stop_s is not None}
    return {
        robot.robot_id: process.name
        for process in processes
        for robot in process.plan.robots
        if robot.robot_id in killed
    }


def last_scrape_k(duration_s: float) -> int:
    """Return the planned second of the last scrape: the last whole second of the run."""
    return math.floor(duration_s)


def next_scrape_k(k: int, t0_ns: int, now_ns: int) -> int:
    """Return the planned second of the scrape after scrape k, at now_ns after it was taken.

    It is the next whole second from time zero that has not passed, so a scrape that ran
    late is taken once and the ones it overran are not.
    """
    whole_seconds_passed = -((t0_ns - now_ns) // _NS_PER_S)
    return max(k + 1, whole_seconds_passed)


def scrape_record(
    k: int | None,
    before_ns: int,
    after_ns: int,
    metrics: Mapping[str, Any] | None,
    error: str | None,
    processes: Mapping[str, ProcessStats],
    cgroup: CgroupStats | None,
) -> Scrape:
    """Return a scrape line; a reply that took longer than SCRAPE_TIMEOUT_S counts as failed."""
    elapsed_ns = after_ns - before_ns
    if metrics is not None and elapsed_ns > SCRAPE_TIMEOUT_S * _NS_PER_S:
        metrics = None
        error = f"the reply took {elapsed_ns / _NS_PER_S:.3f} s, longer than {SCRAPE_TIMEOUT_S:g} s"
    return Scrape(k, before_ns, after_ns, metrics, error, dict(processes), cgroup)


def server_ready_event(metrics: Mapping[str, Any], t0_ns: int) -> Event | None:
    """Return when the gateway became ready, by its own clock; None before it is ready."""
    uptime_s = metrics["uptime_s"]
    if uptime_s is None:
        return None
    ready_ns = metrics["ts_mono_ns"] - uptime_s * _NS_PER_S
    return Event(EventName.SERVER_READY, (ready_ns - t0_ns) / _NS_PER_S)


def final_counts_arrived(
    metrics: Mapping[str, Any],
    robot_ids: Collection[str],
    commands: Mapping[str, Rows],
    models: Sequence[str],
) -> bool:
    """Whether the gateway shows every planned robot gone, with the last counts of each survivor.

    A robot is gone when its record shows connected false. A robot with samples has its last
    counts in when the commands_received of each model in its telemetry equals its samples of
    that model in the session that the record names.
    """
    robots = metrics["robots"]
    if any(robot_id not in robots or robots[robot_id]["connected"] for robot_id in robot_ids):
        return False
    for robot_id, rows in commands.items():
        record = robots.get(robot_id)
        if record is None or "telemetry" not in record:
            return False
        session_rows = rows[rows["session"] == record["session"]]
        for model in models:
            count = int(np.count_nonzero(session_rows["model"] == MODEL_KEYS.index(model)))
            if record["telemetry"]["models"][model]["commands_received"] != count:
                return False
    return True


def exit_status(returncodes: Mapping[str, int | None], killed: Collection[str]) -> int:
    """Return the driver's exit status from the return codes of its fleet processes.

    0 when every process exited 0, but the killed ones, which must have died of SIGKILL.
    A process that did not end has None.
    """
    expected = {name: -signal.SIGKILL if name in killed else 0 for name in returncodes}
    return 0 if all(returncodes[name] == expected[name] for name in returncodes) else 1


def _report_line(robot_id: str, model: str, rows: Rows) -> str:
    selected = rows[rows["model"] == MODEL_KEYS.index(model)]
    line = f"robot {robot_id} {model}: {selected.size} commands"
    if selected.size == 0:
        return line
    # Every command, whatever its verdict (DESIGN.md 10.4).
    freshness_ms = (
        selected["receive_ns"].astype(np.int64) - selected["capture_ns"].astype(np.int64)
    ) / 1e6
    freshness = summarize(freshness_ms.tolist())
    stale = int(np.count_nonzero(selected["verdict"] == VERDICTS.index(Verdict.STALE)))
    return (
        f"{line}, freshness ms p50 {freshness['p50']:.1f} p95 {freshness['p95']:.1f} "
        f"p99 {freshness['p99']:.1f} max {freshness['max']:.1f}, {stale} stale"
    )


def report_lines(commands: Mapping[str, Rows], models: Sequence[str]) -> list[str]:
    """Return a hand run's report: per robot and model, its commands, freshness and stale count."""
    return [
        _report_line(robot_id, model, commands[robot_id])
        for robot_id in sorted(commands)
        for model in models
    ]


def prepare(run_dir: Path, scenario: Scenario | None) -> tuple[RunConfig, FleetConfig]:
    """Return the run's config and its robots, writing a hand run's config where there is none.

    RunDirectoryError for a directory that holds a run already, and for one that has no
    config.json when no scenario is given, or has one when a scenario is.
    """
    held = [name for name in (DONE_FILE, EVENTS_FILE, SCRAPES_FILE) if (run_dir / name).exists()]
    if held:
        raise RunDirectoryError(f"{run_dir} already holds a run: it has {', '.join(held)}")
    config_path = run_dir / CONFIG_FILE
    if config_path.exists():
        if scenario is not None:
            raise RunDirectoryError(
                f"{config_path} exists: --scenario is for a directory without one"
            )
        try:
            config = RunConfig.from_json(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, RunFileError) as error:
            raise DriverError(f"cannot read {config_path}: {error}") from error
    elif scenario is None:
        raise RunDirectoryError(f"{run_dir} has no {CONFIG_FILE}: give --scenario for a hand run")
    else:
        config = hand_run_config(scenario)
        run_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config.to_json(), encoding="utf-8")
    if config.fleet is None:
        raise RunDirectoryError(f"{config_path} is the config of a run without robots")
    for name in (PLANS_DIRECTORY, SAMPLES_DIRECTORY):
        (run_dir / name).mkdir(exist_ok=True)
    return config, config.fleet


def inside_environment() -> InsideEnvironment:
    """Describe the container the driver runs in, and the frame set every robot sends."""
    directory = os.environ.get(MODEL_DIR_VARIABLE)
    # The image holds the models; a container without them records no hashes.
    model_dir = Path(directory) if directory and Path(directory).is_dir() else None
    frames = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).stats()
    document = {**collect_environment(model_dir), "frames": asdict(frames)}
    return InsideEnvironment.from_json(json.dumps(document))


def fetch_metrics(url: str) -> dict[str, Any]:
    """Return the /metrics object at url."""
    try:
        with urllib.request.urlopen(url, timeout=SCRAPE_TIMEOUT_S) as reply:
            body = reply.read()
    except (OSError, http.client.HTTPException) as error:
        # URLError and a timeout are OSErrors; a reply cut short is an HTTPException.
        raise MetricsUnavailable(f"GET {url} failed: {error!r}") from error
    try:
        snapshot = json.loads(body)
    except ValueError as error:
        raise MetricsUnavailable(f"{url} is not JSON: {error}") from error
    if not isinstance(snapshot, dict):
        raise MetricsUnavailable(f"{url} is not a JSON object")
    return snapshot


def _t_s(at_ns: int, t0_ns: int) -> float:
    return (at_ns - t0_ns) / _NS_PER_S


class _EventLog:
    """events.jsonl, written a line at a time by the driver and by its scraper."""

    def __init__(self, file: IO[str]) -> None:
        self._file = file
        self._lock = threading.Lock()

    def write(self, event: Event) -> None:
        with self._lock:
            self._file.write(event.to_json() + "\n")
            self._file.flush()


def _process_stats(pids: Mapping[str, int]) -> dict[str, ProcessStats]:
    """Return the counters of the processes that are alive; an exited process is absent."""
    stats: dict[str, ProcessStats] = {}
    for name, pid in pids.items():
        try:
            stats[name] = read_process_stats(pid)
        except ProcessGoneError:
            continue
    return stats


class _Scrapes:
    """scrapes.jsonl.gz, and the taking of one scrape."""

    def __init__(
        self,
        file: gzip.GzipFile,
        metrics_url: str,
        pids: Mapping[str, int],
        robots_cgroup: bool,
    ) -> None:
        self._file = file
        self._metrics_url = metrics_url
        self._pids = pids
        self._robots_cgroup = robots_cgroup

    def take(self, k: int | None, clock: Clock) -> Scrape:
        before_ns = clock.now_ns()
        metrics: dict[str, Any] | None
        try:
            metrics, error = fetch_metrics(self._metrics_url), None
        except MetricsUnavailable as failure:
            metrics, error = None, str(failure)
        after_ns = clock.now_ns()
        scrape = scrape_record(
            k,
            before_ns,
            after_ns,
            metrics,
            error,
            _process_stats(self._pids),
            read_stats() if self._robots_cgroup else None,
        )
        self._file.write(scrape.to_json().encode("utf-8") + b"\n")
        return scrape


class _Scraper(threading.Thread):
    """The thread that takes a scrape at every whole second of time zero to the run's end."""

    def __init__(
        self, scrapes: _Scrapes, events: _EventLog, t0_ns: int, last_k: int, clock: Clock
    ) -> None:
        super().__init__(name="scraper", daemon=True)
        self._scrapes = scrapes
        self._events = events
        self._t0_ns = t0_ns
        self._last_k = last_k
        self._clock = clock
        self.stop = threading.Event()
        self.failure: Exception | None = None

    def run(self) -> None:
        try:
            self._scrape_to_the_end()
        except Exception as error:
            # The boundary of the scraper: the driver fails the run with what stopped it.
            self.failure = error

    def _scrape_to_the_end(self) -> None:
        k, ready_written = FIRST_SCRAPE_K, False
        while k <= self._last_k:
            due_ns = self._t0_ns + k * _NS_PER_S
            # Re-checked against the clock; the driver sets stop when the run ends early.
            while (remaining_ns := due_ns - self._clock.now_ns()) > 0:
                if self.stop.wait(remaining_ns / _NS_PER_S):
                    return
            scrape = self._scrapes.take(k, self._clock)
            if not ready_written and scrape.metrics is not None:
                ready = server_ready_event(scrape.metrics, self._t0_ns)
                if ready is not None:
                    self._events.write(ready)
                    ready_written = True
            k = next_scrape_k(k, self._t0_ns, self._clock.now_ns())


def _wait_for_fleet(
    children: Mapping[str, subprocess.Popen[bytes]], deadline_ns: int, clock: Clock
) -> dict[str, int | None]:
    """Return each fleet process's return code, or None for one still running at the deadline."""
    returncodes: dict[str, int | None] = {}
    for name, child in children.items():
        try:
            returncodes[name] = child.wait(
                timeout=max(0.0, (deadline_ns - clock.now_ns()) / _NS_PER_S)
            )
        except subprocess.TimeoutExpired:
            returncodes[name] = None
    return returncodes


def _survivor_commands(run_dir: Path, scenario: Scenario) -> dict[str, Rows]:
    """Return the command samples of every robot that ran to the end and wrote them."""
    commands: dict[str, Rows] = {}
    for robot in scenario.robots:
        path = run_dir / SAMPLES_DIRECTORY / SAMPLES_FILE.format(robot_id=robot.robot_id)
        if robot.stop_s is None and path.is_file():
            try:
                commands[robot.robot_id] = load_samples(path)[0]
            except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
                raise DriverError(f"{path} is not a samples file: {error!r}") from error
    return commands


def _final_poll(
    metrics_url: str, fleet: FleetConfig, commands: Mapping[str, Rows], clock: Clock
) -> tuple[bool, float]:
    """Poll /metrics until the gateway has every robot's last counts; return whether and how long.

    A poll that fails counts as one that has not seen the counts yet.
    """
    robot_ids = [robot.robot_id for robot in fleet.scenario.robots]
    start_ns = clock.now_ns()
    deadline_ns = start_ns + round(FINAL_POLL_LIMIT_S * _NS_PER_S)
    while True:
        try:
            settled = final_counts_arrived(
                fetch_metrics(metrics_url), robot_ids, commands, fleet.models
            )
        except MetricsUnavailable:
            settled = False
        now_ns = clock.now_ns()
        if settled or now_ns >= deadline_ns:
            return settled, _t_s(now_ns, start_ns)
        time.sleep(FINAL_POLL_INTERVAL_S)


def _play(arguments: argparse.Namespace) -> int:
    run_dir: Path = arguments.run_dir
    config, fleet = prepare(run_dir, arguments.scenario)
    clock = MonotonicClock()
    (run_dir / INSIDE_ENVIRONMENT_FILE).write_text(inside_environment().to_json(), encoding="utf-8")
    # ASSUMPTION: A20. Time zero is a reading of the host's monotonic clock, which the fleet
    # processes and the gateway read too, in this container or the other.
    t0_ns = clock.now_ns() + round(FLEET_LEAD_S * _NS_PER_S)
    processes = fleet_plans(fleet.scenario, t0_ns, config.seed)
    for process in processes:
        plan_path(run_dir, process).write_text(process.plan.to_json(), encoding="utf-8")
    with (
        (run_dir / EVENTS_FILE).open("x", encoding="utf-8") as events_file,
        (run_dir / SCRAPES_FILE).open("xb") as scrapes_file,
        gzip.GzipFile(filename="", mode="wb", fileobj=scrapes_file, mtime=0) as scrapes_gzip,
    ):
        events = _EventLog(events_file)
        events.write(Event(EventName.T0, 0.0, t0_mono_ns=t0_ns))
        return _run_fleet(arguments, config, fleet, processes, events, scrapes_gzip, t0_ns, clock)


def _follow_timeline(
    processes: Sequence[FleetProcess],
    children: Mapping[str, subprocess.Popen[bytes]],
    scenario: Scenario,
    events: _EventLog,
    t0_ns: int,
    clock: Clock,
) -> None:
    """Write each planned event at its time, and kill a robot the scenario kills there."""
    targets = kill_targets(processes, scenario)
    for event in planned_events(scenario):
        due_ns = t0_ns + round(event.t_s * _NS_PER_S)
        # Re-checked against the clock: a sleep may end early.
        while (remaining_ns := due_ns - clock.now_ns()) > 0:
            time.sleep(remaining_ns / _NS_PER_S)
        if event.name is EventName.ROBOT_KILL and event.robot_id is not None:
            # ASSUMPTION: A11. A robot's stop time is a SIGKILL of its own process: it closes
            # nothing and writes no samples.
            children[targets[event.robot_id]].send_signal(signal.SIGKILL)
        # Read right after the kill has returned.
        events.write(replace(event, t_s=_t_s(clock.now_ns(), t0_ns)))


def _after_the_fleet(
    arguments: argparse.Namespace,
    fleet: FleetConfig,
    scrapes: _Scrapes,
    events: _EventLog,
    t0_ns: int,
    clock: Clock,
) -> dict[str, Rows]:
    """Wait for the gateway's last counts, take the final scrape; return the survivors' commands."""
    try:
        commands = _survivor_commands(arguments.run_dir, fleet.scenario)
        settled, waited_s = _final_poll(arguments.metrics_url, fleet, commands, clock)
        scrapes.take(None, clock)
    except CgroupError as error:
        raise DriverError(f"cannot read the robots' cgroup: {error}") from error
    events.write(
        Event(EventName.FINAL_POLL, _t_s(clock.now_ns(), t0_ns), settled=settled, waited_s=waited_s)
    )
    return commands


def _run_fleet(
    arguments: argparse.Namespace,
    config: RunConfig,
    fleet: FleetConfig,
    processes: Sequence[FleetProcess],
    events: _EventLog,
    scrapes_file: gzip.GzipFile,
    t0_ns: int,
    clock: Clock,
) -> int:
    """Start the fleet and the scraper, follow the timeline, then take the final scrape."""
    children: dict[str, subprocess.Popen[bytes]] = {}
    scraper: _Scraper | None = None
    try:
        for process in processes:
            command = fleet_command(
                process, arguments.run_dir, arguments.url, fleet, config.spec, sys.executable
            )
            children[process.name] = subprocess.Popen(
                command, stdin=subprocess.DEVNULL, cwd=_REPOSITORY
            )
        pids = {**{name: child.pid for name, child in children.items()}, "driver": os.getpid()}
        scrapes = _Scrapes(scrapes_file, arguments.metrics_url, pids, arguments.robots_cgroup)
        scraper = _Scraper(scrapes, events, t0_ns, last_scrape_k(fleet.scenario.duration_s), clock)
        scraper.start()
        _follow_timeline(processes, children, fleet.scenario, events, t0_ns, clock)
        deadline_ns = t0_ns + round((fleet.scenario.duration_s + FLEET_EXIT_MARGIN_S) * _NS_PER_S)
        returncodes = _wait_for_fleet(children, deadline_ns, clock)
        scraper.join(timeout=SCRAPER_JOIN_TIMEOUT_S)
        if scraper.is_alive():
            raise DriverError(f"the scraper did not end within {SCRAPER_JOIN_TIMEOUT_S:g} s")
        if scraper.failure is not None:
            raise DriverError(f"a scrape failed: {scraper.failure!r}") from scraper.failure
        commands = _after_the_fleet(arguments, fleet, scrapes, events, t0_ns, clock)
        if config.hand_run:
            for line in report_lines(commands, fleet.models):
                print(line)
        return exit_status(returncodes, set(kill_targets(processes, fleet.scenario).values()))
    finally:
        if scraper is not None:
            scraper.stop.set()
        for child in children.values():
            if child.poll() is None:
                child.kill()
                child.wait(timeout=REAP_TIMEOUT_S)


def _scenario(key: str) -> Scenario:
    try:
        return scenario_for(key)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.run",
        description="Play one serving run against the gateway, and scrape it.",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="the run's directory")
    parser.add_argument(
        "--scenario",
        type=_scenario,
        help="a scenario key, for a hand run in a directory without config.json",
    )
    parser.add_argument(
        "--url", default=RobotSettings().url, help="the gateway's WebSocket address"
    )
    parser.add_argument(
        "--metrics-url", default=DEFAULT_METRICS_URL, help="the gateway's /metrics address"
    )
    parser.add_argument(
        "--robots-cgroup",
        action="store_true",
        help="record this container's cgroup in every scrape: the robots' own container",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        return _play(arguments)
    except RunDirectoryError as error:
        print(f"bench.run: {error}", file=sys.stderr)
        return 2
    except (DriverError, OSError) as error:
        # An OSError here is a file of the run directory that could not be written.
        print(f"bench.run: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
