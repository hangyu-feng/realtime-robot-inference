"""The evidence runner: every planned run in fresh containers, gated, finished and summarized.

    python -m bench.evidence run [--results DIR] [--repeats R] [--models real|fake|null]
        [--only KEY:TOPOLOGY[,KEY:TOPOLOGY...]] [--extra-compose FILE]
    python -m bench.evidence compare CANDIDATE|all [--results DIR]

It runs on the host, outside every container, with the docker CLI and git. It computes the
code digest of the working tree and skips every planned run that is finished. Before the
first run left, it builds the image and runs the test suite in it, once per image and code
digest (the tests gate). Then it plays each run in fresh containers: it starts them, has the
scenario driver play the run inside, removes them, and writes the run's environment, its
gates and DONE, then has its summary written. In a run whose scenario kills a worker, the
fault helper (bench.fault) kills it from the gateway's container while the driver plays,
and the runner adds the event the helper prints to the run's events. A directory of the
same run without a valid DONE is kept as an attempt, and the run is played again.

It prints one line per run. It exits 0 when every run finished with its summary; 1 when the
tests gate failed, or a run failed or its summary could not be written; 2 for bad
arguments; and 129, 130 or 143 after SIGHUP, SIGINT or SIGTERM, which remove the current
run's containers and leave it without DONE.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from types import FrameType
from typing import Any

from bench import docker_ops
from bench.aggregate import AggregateError, run_path, tree_code_digest
from bench.candidates import CANDIDATES, Candidate
from bench.docker_ops import CONTAINER_PREFIX, Compose, DockerError
from bench.gates import (
    GateError,
    gate_record,
    model_bench_gate_record,
    read_report_cgroups,
    read_scrapes,
    stored_suite_gate,
    suite_gate_file,
    suite_gate_record,
)
from bench.run import fleet_config
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    EVENTS_FILE,
    GATE_FILE,
    INSIDE_ENVIRONMENT_FILE,
    PLANS_DIRECTORY,
    SAMPLES_DIRECTORY,
    Done,
    Event,
    EventName,
    Gate,
    InsideEnvironment,
    RunConfig,
    RunEnvironment,
    RunFileError,
    Scrape,
    SuiteGate,
    done_files,
)
from bench.runplan import (
    MODEL_BENCH_KEY,
    ModelSet,
    RunSpec,
    code_paths,
    compare_plan,
    evidence_plan,
    only,
    overrides_for,
    run_id,
    scenario_for,
)
from bench.scenarios import Scenario, massive
from bench.summarize import REPORTS_DIRECTORY, run_files, write_summary
from cloud.config import Topology
from common.cgroup import Limits
from robot.settings import RobotSettings

REPOSITORY = Path(__file__).resolve().parent.parent
COMPOSE_FILES: Mapping[Topology, Path] = {
    Topology.SINGLE: REPOSITORY / "docker-compose.yml",
    Topology.SPLIT: REPOSITORY / "docker-compose.split.yml",
}
# Compose names a project after the directory of its files, in lower case with only letters,
# digits, "-" and "_". The runner names it the same way, so a service started by hand from the
# repository belongs to the project and is removed before a run.
PROJECT = re.sub(r"[^a-z0-9_-]", "", REPOSITORY.name.lower())
# Where both compose files mount the results root.
CONTAINER_RESULTS = PurePosixPath("/app/results")
GATEWAY_SERVICE = "system"
ROBOTS_SERVICE = "robots"
ROBOTS_PROFILE = "robots"
READY_PREFIX = "ready:"
ENGINE = "docker"
# Three repeats of every run.
DEFAULT_REPEATS = 3
# The tests gate: the suite in the image, as its own user, without the tests that need the
# host's docker CLI; the skip reasons are printed, and no cache is written.
SUITE_COMMAND = ("pytest", "-q", "-p", "no:cacheprovider", "-rs", "-m", "not docker")

# The first build downloads and verifies both models.
BUILD_TIMEOUT_S = 3600.0
# Far above the suite's duration in the image under its CPU limit.
SUITE_TIMEOUT_S = 3600.0
# A query of the engine or of git, or the start or the removal of containers, which waits
# out the stop grace period of the compose files.
DOCKER_TIMEOUT_S = 120.0
GIT_TIMEOUT_S = 60.0
# The ready line follows the warm-up of every model; this covers the gateway's own start.
READY_MARGIN_S = 30.0
# The driver's run is the scenario plus the fleet's lead before time zero, the final poll
# and the environment it records.
DRIVER_MARGIN_S = 120.0
# The fault helper exits at its kill, long before the driver ends; this bounds the wait for it
# after the driver.
FAULT_EXIT_TIMEOUT_S = 30.0
# One command of the model benchmark: a batch command times each batch size for some
# seconds after its warm-up iterations.
BENCH_COMMAND_TIMEOUT_S = 1800.0
# The lines of output a failure quotes.
TAIL_LINES = 30

# The batch sizes the classifier serves and the patched depth graph takes; the published
# depth graph takes one frame per run (DESIGN.md 4.6).
BENCH_BATCH_SIZES = "1,2,4,8"
PUBLISHED_DEPTH_BATCH_SIZES = "1"
BENCH_THREADS = (1, 2)
BENCH_LOOP = "uvloop"
# The WebSocket run carries the massive demo's robots at the robots' frame rate, for long
# enough that the CPU per message is read over many seconds.
WEBSOCKET_ROBOTS = len(massive().robots)
WEBSOCKET_FPS = round(RobotSettings().frame_rate_hz)
WEBSOCKET_DURATION_S = 20


class EvidenceError(Exception):
    """The invocation cannot go on: git, the build or the tests gate failed."""


class RunFailure(Exception):
    """A run could not be played to its end; its directory stays without DONE."""


class Interrupted(BaseException):
    """A signal stopped the runner.

    Not an Exception, so that the boundary of a run's summary, which catches every Exception,
    lets it through.
    """

    def __init__(self, signal_number: int) -> None:
        super().__init__(f"stopped by signal {signal_number}")
        self.signal_number = signal_number


class Outcome(StrEnum):
    SKIPPED = "finished before, skipped"
    FINISHED = "finished"
    SUMMARY_FAILED = "finished, its summary failed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunResult:
    """What became of one planned run."""

    path: str  # the run's directory under the results root
    outcome: Outcome
    detail: str

    def line(self) -> str:
        return f"{self.path}: {self.outcome}" + (f"; {self.detail}" if self.detail else "")


@dataclass(frozen=True)
class RepositoryState:
    """The working tree the runs are of."""

    commit: str
    dirty: bool  # whether any code-digest path differs from the commit
    code_digest: str


@dataclass(frozen=True)
class Invocation:
    """What one invocation established before its first run, and each of its runs shares."""

    results: Path  # absolute
    image_digest: str
    # The tests gate of the image and the code digest, which may have passed for an earlier
    # commit of the same code.
    tests_passed: SuiteGate
    commit: str
    dirty: bool
    engine_version: str
    host_memory_total_bytes: int
    # The host user's ids, uid:gid: every process that writes under the results root runs
    # with them, so the files belong to whoever owns the results root.
    user: str


def _git(*arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=REPOSITORY,
            capture_output=True,
            text=True,
            check=True,
            timeout=GIT_TIMEOUT_S,
        )
    except subprocess.CalledProcessError as error:
        raise EvidenceError(f"git {arguments[0]} failed: {error.stderr.strip()}") from error
    except (OSError, subprocess.SubprocessError) as error:
        raise EvidenceError(f"git {arguments[0]} failed: {error}") from error
    return completed.stdout


def repository_state() -> RepositoryState:
    """Read the commit, whether the code paths are dirty, and the code digest of the tree."""
    listed = _git("ls-files", "-co", "--exclude-standard").splitlines()
    status = _git("status", "--porcelain", "--untracked-files=all", "--", *code_paths(listed))
    return RepositoryState(
        commit=_git("rev-parse", "HEAD").strip(),
        dirty=bool(status.strip()),
        code_digest=tree_code_digest(REPOSITORY),
    )


def prepare_invocation(
    results: Path, image_digest: str, tests_passed: SuiteGate, state: RepositoryState
) -> Invocation:
    """Return the invocation's shared state, reading the engine's version and host memory."""
    return Invocation(
        results=results,
        image_digest=image_digest,
        tests_passed=tests_passed,
        commit=state.commit,
        dirty=state.dirty,
        engine_version=docker_ops.engine_version(DOCKER_TIMEOUT_S),
        host_memory_total_bytes=docker_ops.host_memory_total_bytes(DOCKER_TIMEOUT_S),
        user=f"{os.getuid()}:{os.getgid()}",
    )


def candidates_named(name: str) -> list[Candidate]:
    """Return the candidate of that name, or every candidate in order for "all"."""
    if name == "all":
        return list(CANDIDATES)
    named = [candidate for candidate in CANDIDATES if candidate.name == name]
    if not named:
        names = ", ".join(candidate.name for candidate in CANDIDATES)
        raise ValueError(f"{name} is not a candidate; they are {names}, or all")
    return named


def run_overrides(spec: RunSpec) -> dict[str, Any]:
    """Return a run's RRI_OVERRIDES object: its model set's settings, under its arm's."""
    if spec.candidate is None or spec.arm is None:
        return overrides_for(spec.model_set, {})
    (candidate,) = candidates_named(spec.candidate)
    arms = {"a": candidate.a_overrides, "b": candidate.b_overrides}
    return overrides_for(spec.model_set, arms[spec.arm])


def run_compose(
    topology: Topology,
    extra_compose_path: Path | None,
    overrides: Mapping[str, Any],
    results: Path,
    commit: str,
) -> Compose:
    """Return the compose project of a run: its topology's file, then any file laid over it."""
    extra = () if extra_compose_path is None else (extra_compose_path,)
    return Compose(
        project=PROJECT,
        files=(COMPOSE_FILES[topology], *extra),
        profiles=(ROBOTS_PROFILE,) if topology is Topology.SPLIT else (),
        variables={
            "RRI_OVERRIDES": json.dumps(overrides),
            "RRI_RESULTS": str(results),
            "GIT_COMMIT": commit,
        },
    )


def ready_timeout_s(server_config: Mapping[str, Any]) -> float:
    """Return how long a gateway may take to print its ready line: every model's warm-up, then a
    margin."""
    warmup_s: float = server_config["timeouts_ms"]["warmup"] / 1000
    return len(server_config["models"]) * warmup_s + READY_MARGIN_S


def driver_command(spec: RunSpec) -> list[str]:
    """Return the scenario driver's command line in the robots' container."""
    command = ["python", "-m", "bench.run", "--run-dir", str(CONTAINER_RESULTS / run_path(spec))]
    if spec.topology is Topology.SPLIT:
        # The robots reach the gateway in the other container by its service's name.
        port = spec.server_config["listen"]["port"]
        command += [
            "--url",
            f"ws://{GATEWAY_SERVICE}:{port}/ws",
            "--metrics-url",
            f"http://{GATEWAY_SERVICE}:{port}/metrics",
            "--robots-cgroup",
        ]
    return command


def robots_container(spec: RunSpec) -> str:
    return f"{CONTAINER_PREFIX}robots-{run_id(spec)}"


def model_bench_commands(seed: int, reports: PurePosixPath) -> list[tuple[str, list[str]]]:
    """Return the model benchmark's commands in the order they run, each with its report's name.

    Each graph is timed at one and at two threads: the classifier's file at every batch
    size, the published depth graph at one frame and its patched copy at every size. Then
    the stages outside inference, the dispatch round trip, and the WebSocket run without a
    model.
    """
    graphs = {
        "cls": (("published", BENCH_BATCH_SIZES),),
        "depth": (("published", PUBLISHED_DEPTH_BATCH_SIZES), ("batched", BENCH_BATCH_SIZES)),
    }
    commands: list[tuple[str, list[str]]] = []
    for model, model_graphs in graphs.items():
        for threads, (graph, sizes) in itertools.product(BENCH_THREADS, model_graphs):
            name = f"{model}_t{threads}_{graph}"
            commands.append(
                (
                    name,
                    [
                        *("python", "-m", "bench.model_bench", "batch", "--model", model),
                        *("--graph", graph, "--threads", str(threads), "--batch-sizes", sizes),
                    ],
                )
            )
    commands.append(("stages", ["python", "-m", "bench.model_bench", "stages"]))
    commands.append(
        ("dispatch", ["python", "-m", "bench.transport_bench", "dispatch", "--loop", BENCH_LOOP])
    )
    commands.append(
        (
            "websocket",
            [
                *("python", "-m", "bench.transport_bench", "websocket"),
                *("--robots", str(WEBSOCKET_ROBOTS), "--fps", str(WEBSOCKET_FPS)),
                *("--duration-s", str(WEBSOCKET_DURATION_S), "--loop", BENCH_LOOP),
                *("--seed", str(seed)),
            ],
        )
    )
    return [
        (name, [*command, "--out", str(reports / f"{name}.json")]) for name, command in commands
    ]


def next_free(path_of: Callable[[int], Path]) -> Path:
    """Return the first path of path_of(1), path_of(2), ... that does not exist."""
    k = 1
    while path_of(k).exists():
        k += 1
    return path_of(k)


def is_finished(run_dir: Path) -> bool:
    """Whether a run directory holds a finished run: a DONE whose hashes match its files."""
    done_path = run_dir / DONE_FILE
    if not done_path.is_file():
        return False
    try:
        done = Done.from_json(done_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError, RunFileError:
        return False
    return done.valid_for(run_files(run_dir))


def keep_as_attempt(run_dir: Path) -> Path | None:
    """Rename an unfinished run directory to its next free attempt; return the new path."""
    if not run_dir.exists():
        return None
    attempt = next_free(lambda k: run_dir.with_name(f"{run_dir.name}.attempt{k}"))
    run_dir.rename(attempt)
    return attempt


def failed_gates(gate: Gate) -> list[str]:
    passed = {
        "tests_passed": gate.tests_passed.passed,
        "image_built": gate.image_built.built,
        "limits_ok": gate.limits_ok.ok,
        "fresh": gate.fresh.ok,
        "config_ok": gate.config_ok is None or gate.config_ok.ok,
    }
    return [name for name, ok in passed.items() if not ok]


def _with_gateway_limits(inside: InsideEnvironment, scrapes: Sequence[Scrape]) -> InsideEnvironment:
    """Return the inside fields with the gateway's limits, from the first scrape that succeeded."""
    first = next((scrape.metrics for scrape in scrapes if scrape.metrics is not None), None)
    if first is None:
        raise RunFailure(
            "no scrape succeeded, so the limits of the gateway's container are unknown"
        )
    document = json.loads(inside.to_json())
    document["cgroup"] = {field.name: first["cgroup"].get(field.name) for field in fields(Limits)}
    return InsideEnvironment.from_json(json.dumps(document))


def run_environment(
    spec: RunSpec, inside: InsideEnvironment, scrapes: Sequence[Scrape], invocation: Invocation
) -> RunEnvironment:
    """Return env.json of a serving run.

    The inside fields are the driver's. In the split topology the driver ran in the robots'
    container, so its limits are the robots', and the gateway's come from the scrapes.
    """
    robots_cgroup = None
    if spec.topology is Topology.SPLIT:
        robots_cgroup = inside.cgroup
        inside = _with_gateway_limits(inside, scrapes)
    return RunEnvironment(
        inside=inside,
        host_memory_total_bytes=invocation.host_memory_total_bytes,
        engine=ENGINE,
        engine_version=invocation.engine_version,
        commit=invocation.commit,
        dirty=invocation.dirty,
        code_digest=spec.code_digest,
        image_digest=invocation.image_digest,
        robots_cgroup=robots_cgroup,
    )


def _tail(output: bytes | str) -> str:
    text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else output
    return "\n".join(text.strip().splitlines()[-TAIL_LINES:])


def _make_run_directory(run_dir: Path, subdirectories: Sequence[str]) -> None:
    # Before any docker command that mounts it: the engine creates a missing bind source as
    # root.
    run_dir.mkdir(parents=True)
    for name in subdirectories:
        (run_dir / name).mkdir()


def _write_config(
    run_dir: Path, spec: RunSpec, extra_compose_path: Path | None, scenario: Scenario | None
) -> None:
    config = RunConfig(
        spec=spec,
        run_id=run_id(spec),
        seed=spec.repeat,
        extra_compose_path=None if extra_compose_path is None else str(extra_compose_path),
        fleet=None if scenario is None else fleet_config(scenario),
    )
    (run_dir / CONFIG_FILE).write_text(config.to_json(), encoding="utf-8")


def _finish_removal[T](removal: Callable[[], T]) -> T:
    """Run a removal of containers to its end, even when the runner's first signal lands in it.

    The signal cuts the docker command it lands in short. Its handler ignores every later
    signal, so the removal runs once more, to its end, before the signal stops the runner.
    """
    try:
        return removal()
    except Interrupted:
        removal()
        raise


def _clear_project(compose: Compose) -> list[str]:
    """Remove every container of the project; return the ids that were there."""

    def clear() -> list[str]:
        there = docker_ops.project_containers(compose.project, DOCKER_TIMEOUT_S)
        docker_ops.down(compose, DOCKER_TIMEOUT_S)
        docker_ops.remove(there, DOCKER_TIMEOUT_S)
        return there

    return _finish_removal(clear)


def _start_serving(compose: Compose, spec: RunSpec) -> dict[str, str]:
    """Start the run's containers; return each one's id by its name in the gate."""
    docker_ops.up(compose, GATEWAY_SERVICE, DOCKER_TIMEOUT_S)
    containers = {
        GATEWAY_SERVICE: docker_ops.service_container(compose, GATEWAY_SERVICE, DOCKER_TIMEOUT_S)
    }
    if spec.topology is Topology.SPLIT:
        containers[ROBOTS_SERVICE] = docker_ops.run_detached(
            compose, ROBOTS_SERVICE, robots_container(spec), ["sleep", "infinity"], DOCKER_TIMEOUT_S
        )
    return containers


def _gateway_log(container: str) -> str:
    try:
        return _tail(docker_ops.log_tail(container, TAIL_LINES, DOCKER_TIMEOUT_S))
    except DockerError as error:
        return f"(no log: {error})"


def fault_event(exit_status: int, stdout: bytes) -> Event:
    """Return the fault event that the fault helper printed.

    RunFailure unless the helper exited 0 having printed exactly one line, a fault event.
    """
    if exit_status != 0:
        raise RunFailure(f"the fault helper exited with {exit_status}")
    lines = stdout.decode("utf-8", errors="replace").splitlines()
    if len(lines) != 1:
        raise RunFailure(
            f"the fault helper printed {len(lines)} lines, not one {EventName.FAULT} event"
        )
    try:
        event = Event.from_json(lines[0])
    except RunFileError as error:
        raise RunFailure(f"the fault helper printed no event: {error}") from error
    if event.name is not EventName.FAULT:
        raise RunFailure(
            f"the fault helper printed a {event.name} event, not a {EventName.FAULT} event"
        )
    return event


def _drive(
    compose: Compose, spec: RunSpec, scenario: Scenario, invocation: Invocation, gateway: str
) -> None:
    """Have the scenario driver play the run in the robots' container; RunFailure when it fails."""
    timeout_s = scenario.duration_s + DRIVER_MARGIN_S
    if spec.topology is Topology.SPLIT:
        completed = docker_ops.exec_in_container(
            robots_container(spec), invocation.user, driver_command(spec), timeout_s
        )
    else:
        completed = docker_ops.exec_in_service(
            compose, GATEWAY_SERVICE, invocation.user, driver_command(spec), timeout_s
        )
    if completed.exit_status != 0:
        raise RunFailure(
            f"the driver exited with {completed.exit_status}:\n{_tail(completed.output)}\n"
            f"gateway log:\n{_gateway_log(gateway)}"
        )


def _drive_with_fault(
    compose: Compose, spec: RunSpec, scenario: Scenario, invocation: Invocation, gateway: str
) -> None:
    """Have the driver play a run that kills a worker, beside the fault helper that kills it.

    The fault event the helper prints follows the driver's last line of events.jsonl, so the
    driver writes only the events its timeline plans. RunFailure when the driver fails, or
    the helper has not printed one fault event and exited 0 soon after the driver's end.
    """
    command = ["python", "-m", "bench.fault", "--run-dir", str(CONTAINER_RESULTS / run_path(spec))]
    # In the gateway's container in either topology, and as the image's user: a signal needs
    # the worker's pid namespace and its owner, and the host user's uid may not be the image's.
    helper = docker_ops.start_in_service(compose, GATEWAY_SERVICE, command)
    with helper:
        try:
            _drive(compose, spec, scenario, invocation, gateway)
            try:
                stdout, stderr = helper.communicate(timeout=FAULT_EXIT_TIMEOUT_S)
            except subprocess.TimeoutExpired as error:
                raise RunFailure(
                    f"the fault helper did not exit within {FAULT_EXIT_TIMEOUT_S:g} s of the "
                    "driver's end"
                ) from error
        finally:
            # Ends the docker client of a helper still running; removing the containers ends
            # the process inside. A helper that has exited is not signalled.
            helper.kill()
            helper.wait(timeout=docker_ops.REAP_TIMEOUT_S)
    try:
        event = fault_event(helper.returncode, stdout)
    except RunFailure as error:
        raise RunFailure(f"{error}; what it wrote:\n{_tail(stdout + stderr)}") from error
    with (invocation.results / run_path(spec) / EVENTS_FILE).open("a", encoding="utf-8") as file:
        file.write(event.to_json() + "\n")


def _play_serving(
    compose: Compose,
    spec: RunSpec,
    scenario: Scenario,
    containers: Mapping[str, str],
    invocation: Invocation,
) -> dict[str, str]:
    """Wait for the gateway and have the driver play the run; return each container's image.

    RunFailure when the driver or the fault helper fails, or a container did not run to the
    end of the run.
    """
    gateway = containers[GATEWAY_SERVICE]
    try:
        docker_ops.wait_for_line(gateway, READY_PREFIX, ready_timeout_s(spec.server_config))
    except DockerError as error:
        raise RunFailure(f"{error}\ngateway log:\n{_gateway_log(gateway)}") from error
    if scenario.fault is None:
        _drive(compose, spec, scenario, invocation, gateway)
    else:
        _drive_with_fault(compose, spec, scenario, invocation, gateway)
    for name, id_ in containers.items():
        # In the split topology the driver outlives the gateway, and the fleet reconnects
        # and exits 0 all the same: a container that stopped or restarted during the run
        # exited early.
        state = docker_ops.container_state(id_, DOCKER_TIMEOUT_S)
        if state.status != "running" or state.restart_count != 0:
            raise RunFailure(
                f"the {name} container did not run to the end: {state.status}, exit code "
                f"{state.exit_code}, {state.restart_count} restarts, out of memory "
                f"{str(state.oom_killed).lower()}\ngateway log:\n{_gateway_log(gateway)}"
            )
    return {name: docker_ops.image_of(id_, DOCKER_TIMEOUT_S) for name, id_ in containers.items()}


def _serving_run(
    spec: RunSpec,
    scenario: Scenario,
    extra_compose_path: Path | None,
    invocation: Invocation,
    run_dir: Path,
) -> Gate:
    """Play a serving run and write its environment and its gates; return the gates."""
    _make_run_directory(run_dir, (PLANS_DIRECTORY, SAMPLES_DIRECTORY))
    _write_config(run_dir, spec, extra_compose_path, scenario)
    compose = run_compose(
        spec.topology,
        extra_compose_path,
        run_overrides(spec),
        invocation.results,
        invocation.commit,
    )
    earlier = _clear_project(compose)
    try:
        containers = _start_serving(compose, spec)
        images = _play_serving(compose, spec, scenario, containers, invocation)
    finally:
        _clear_project(compose)
    scrapes = read_scrapes(run_dir)
    inside = InsideEnvironment.from_json(
        (run_dir / INSIDE_ENVIRONMENT_FILE).read_text(encoding="utf-8")
    )
    environment = run_environment(spec, inside, scrapes, invocation)
    (run_dir / ENVIRONMENT_FILE).write_text(environment.to_json(), encoding="utf-8")
    gate = gate_record(
        invocation.tests_passed,
        invocation.image_digest,
        images,
        list(containers.values()),
        earlier,
        spec.server_config,
        scrapes,
    )
    (run_dir / GATE_FILE).write_text(gate.to_json(), encoding="utf-8")
    return gate


def model_bench_environment(
    spec: RunSpec, report_path: Path, invocation: Invocation
) -> RunEnvironment:
    """Return env.json of the model benchmark, from the environment block of its first report."""
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        inside = InsideEnvironment.from_json(
            json.dumps({**report["environment"], "frames": report["frames"]})
        )
    except (ValueError, KeyError, TypeError) as error:
        raise RunFailure(f"{report_path} has no environment to record: {error!r}") from error
    return RunEnvironment(
        inside=inside,
        host_memory_total_bytes=invocation.host_memory_total_bytes,
        engine=ENGINE,
        engine_version=invocation.engine_version,
        commit=invocation.commit,
        dirty=invocation.dirty,
        code_digest=spec.code_digest,
        image_digest=invocation.image_digest,
        robots_cgroup=None,
    )


def _model_bench_run(
    spec: RunSpec, extra_compose_path: Path | None, invocation: Invocation, run_dir: Path
) -> Gate:
    """Run each command of the model benchmark in its own container; return the gates."""
    _make_run_directory(run_dir, (REPORTS_DIRECTORY,))
    _write_config(run_dir, spec, extra_compose_path, None)
    compose = run_compose(
        Topology.SINGLE,
        extra_compose_path,
        run_overrides(spec),
        invocation.results,
        invocation.commit,
    )
    earlier = _clear_project(compose)
    commands = model_bench_commands(
        spec.repeat, CONTAINER_RESULTS / run_path(spec) / REPORTS_DIRECTORY
    )
    images: dict[str, str] = {}
    container_ids: list[str] = []
    for k, (name, command) in enumerate(commands, start=1):
        container = f"{CONTAINER_PREFIX}bench-{run_id(spec)}-{k}"
        try:
            completed = docker_ops.run_named(
                compose,
                GATEWAY_SERVICE,
                container,
                command,
                invocation.user,
                BENCH_COMMAND_TIMEOUT_S,
            )
            if completed.exit_status != 0:
                raise RunFailure(
                    f"{name} exited with {completed.exit_status}:\n{_tail(completed.output)}"
                )
            container_ids.append(docker_ops.container_id(container, DOCKER_TIMEOUT_S))
            images[name] = docker_ops.image_of(container, DOCKER_TIMEOUT_S)
        finally:
            _finish_removal(functools.partial(docker_ops.remove, [container], DOCKER_TIMEOUT_S))
    environment = model_bench_environment(
        spec, run_dir / REPORTS_DIRECTORY / f"{commands[0][0]}.json", invocation
    )
    (run_dir / ENVIRONMENT_FILE).write_text(environment.to_json(), encoding="utf-8")
    gate = model_bench_gate_record(
        invocation.tests_passed,
        invocation.image_digest,
        images,
        container_ids,
        earlier,
        read_report_cgroups(run_dir),
    )
    (run_dir / GATE_FILE).write_text(gate.to_json(), encoding="utf-8")
    return gate


def run_one(
    spec: RunSpec,
    scenario: Scenario | None,
    extra_compose_path: Path | None,
    invocation: Invocation,
) -> RunResult:
    """Play one planned run unless it is finished; scenario is None for the model benchmark.

    A directory of the run without a valid DONE is kept as an attempt first. A run that
    fails stays without DONE. A run whose gate fails is finished all the same, not gated.
    """
    path = run_path(spec)
    run_dir = invocation.results / path
    if is_finished(run_dir):
        return RunResult(path, Outcome.SKIPPED, "")
    kept = keep_as_attempt(run_dir)
    notes = [] if kept is None else [f"the directory before it is kept as {kept.name}"]
    try:
        if scenario is None:
            gate = _model_bench_run(spec, extra_compose_path, invocation, run_dir)
        else:
            gate = _serving_run(spec, scenario, extra_compose_path, invocation, run_dir)
    except (RunFailure, DockerError, GateError, RunFileError, OSError) as error:
        # An OSError is a file of the run that could not be read or written.
        return RunResult(path, Outcome.FAILED, "; ".join([*notes, str(error)]))
    # The run is finished from here on: the hashes of its raw files, and no summary yet.
    done = Done(files=done_files(run_files(run_dir)), summary_sha256=None)
    (run_dir / DONE_FILE).write_text(done.to_json(), encoding="utf-8")
    notes.append("gated" if gate.gated else f"not gated: {', '.join(failed_gates(gate))}")
    try:
        write_summary(run_dir)
    except Exception as error:
        # The boundary of one run's summary: the run stays finished, and python -m
        # bench.summarize writes the summary once the summarizer is fixed.
        return RunResult(path, Outcome.SUMMARY_FAILED, "; ".join([*notes, repr(error)]))
    return RunResult(path, Outcome.FINISHED, "; ".join(notes))


def suite_gate(
    compose: Compose, results: Path, image_digest: str, state: RepositoryState
) -> SuiteGate:
    """Return the tests gate of the image and the code, running the suite unless it passed before.

    A stored gate that failed, or that cannot be trusted, is run again, and its output is
    kept beside the new one as .fail<k>.txt. EvidenceError when the gate fails.
    """
    stored = stored_suite_gate(results, image_digest, state.code_digest)
    if stored is not None and stored.passed:
        return stored
    gate_path = results / suite_gate_file(image_digest, state.code_digest)
    output_path = gate_path.with_suffix(".txt")
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.rename(
            next_free(lambda k: gate_path.with_name(f"{gate_path.stem}.fail{k}.txt"))
        )
    container = f"{CONTAINER_PREFIX}tests-{gate_path.stem}"
    print(f"the tests gate: running the suite in the image, into {output_path}", flush=True)
    # An invocation that was killed before its cleanup leaves a container of this name, and
    # the engine refuses a second one.
    docker_ops.remove([container], DOCKER_TIMEOUT_S)
    try:
        completed = docker_ops.run_named(
            compose, GATEWAY_SERVICE, container, SUITE_COMMAND, None, SUITE_TIMEOUT_S
        )
    finally:
        _finish_removal(functools.partial(docker_ops.remove, [container], DOCKER_TIMEOUT_S))
    output_path.write_bytes(completed.output)
    try:
        record = suite_gate_record(
            completed.output,
            completed.exit_status,
            image_digest,
            state.code_digest,
            state.commit,
            state.dirty,
        )
    except GateError as error:
        raise EvidenceError(
            f"the tests gate failed: {error}; the output is {output_path}:\n"
            f"{_tail(completed.output)}"
        ) from error
    gate_path.write_text(record.to_json(), encoding="utf-8")
    if not record.passed:
        raise EvidenceError(
            f"the tests gate failed; the output is {output_path}:\n{_tail(completed.output)}"
        )
    return record


class SignalWatch:
    """SIGHUP, SIGINT and SIGTERM: the first stops the runner where it is, the rest are ignored.

    Ignoring them lets the current run's containers be removed after the first.
    """

    HANDLED = (signal.SIGHUP, signal.SIGINT, signal.SIGTERM)

    def __init__(self) -> None:
        self.received: int | None = None
        for number in self.HANDLED:
            signal.signal(number, self._stop)

    def _stop(self, signal_number: int, frame: FrameType | None) -> None:
        for number in self.HANDLED:
            signal.signal(number, signal.SIG_IGN)
        self.received = signal_number
        raise Interrupted(signal_number)


def _play_plan(
    specs: Sequence[RunSpec],
    results: Path,
    extra_compose_path: Path | None,
    state: RepositoryState,
    watch: SignalWatch,
) -> int:
    """Play every run of the plan that is not finished; return the exit status."""
    results.mkdir(parents=True, exist_ok=True)
    pending = [spec for spec in specs if not is_finished(results / run_path(spec))]
    for spec in specs:
        if spec not in pending:
            print(RunResult(run_path(spec), Outcome.SKIPPED, "").line(), flush=True)
    if not pending:
        return 0
    compose = run_compose(Topology.SINGLE, None, {}, results, state.commit)
    print("building the image", flush=True)
    image_digest = docker_ops.build(compose, BUILD_TIMEOUT_S)
    print(f"image {image_digest}, code {state.code_digest[:8]}, dirty {state.dirty}", flush=True)
    tests_passed = suite_gate(compose, results, image_digest, state)
    invocation = prepare_invocation(results, image_digest, tests_passed, state)
    print(f"tests gate passed: {invocation.tests_passed.gate_file}", flush=True)
    failed: list[RunResult] = []
    for spec in pending:
        scenario = None if spec.scenario_key == MODEL_BENCH_KEY else scenario_for(spec.scenario_key)
        result = run_one(spec, scenario, extra_compose_path, invocation)
        print(result.line(), flush=True)
        if result.outcome in (Outcome.FAILED, Outcome.SUMMARY_FAILED):
            failed.append(result)
        if watch.received is not None:
            return 128 + watch.received
    for result in failed:
        print(f"bench.evidence: {result.path}: {result.outcome}", file=sys.stderr)
    return 1 if failed else 0


def _entries(text: str) -> list[str]:
    return [entry for entry in text.split(",") if entry]


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be 1 or more, got {value}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.evidence",
        description="Play the planned runs in fresh containers, gated, and summarize them.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="the runs of the evidence plan")
    run.add_argument("--repeats", type=_positive, default=DEFAULT_REPEATS)
    run.add_argument("--models", type=ModelSet, choices=list(ModelSet), default=ModelSet.REAL)
    run.add_argument(
        "--only", type=_entries, metavar="KEY:TOPOLOGY,...", help="these runs of the plan alone"
    )
    run.add_argument(
        "--extra-compose", type=Path, metavar="FILE", help="a compose file laid over the topology's"
    )
    compare = commands.add_parser("compare", help="a candidate's runs, A, B, B, A")
    compare.add_argument("candidate", help="a candidate of bench/candidates.py, or all")
    for command in (run, compare):
        command.add_argument(
            "--results", type=Path, default=Path("results"), help="the results root"
        )
    return parser


def planned_runs(
    arguments: argparse.Namespace, code_digest: str
) -> tuple[list[RunSpec], Path | None]:
    """Return the runs a command line asks for, in their order, and any extra compose file.

    ValueError for an entry of --only or a candidate that the plan does not have.
    """
    if arguments.command == "compare":
        specs = [
            spec
            for candidate in candidates_named(arguments.candidate)
            for spec in compare_plan(candidate, code_digest)
        ]
        return specs, None
    specs = evidence_plan(arguments.repeats, arguments.models, code_digest)
    if arguments.only is not None:
        specs = only(specs, arguments.only)
    extra: Path | None = arguments.extra_compose
    if extra is None:
        return specs, None
    extra = extra.resolve()
    digest = hashlib.sha256(extra.read_bytes()).hexdigest()
    return [replace(spec, extra_compose=digest) for spec in specs], extra


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    results: Path = arguments.results.resolve()
    try:
        state = repository_state()
        specs, extra_compose_path = planned_runs(arguments, state.code_digest)
    except (ValueError, OSError) as error:
        print(f"bench.evidence: {error}", file=sys.stderr)
        return 2
    except (EvidenceError, AggregateError) as error:
        print(f"bench.evidence: {error}", file=sys.stderr)
        return 1
    watch = SignalWatch()
    try:
        return _play_plan(specs, results, extra_compose_path, state, watch)
    except (Interrupted, EvidenceError, DockerError, GateError, OSError) as error:
        # A cleanup that fails after a signal raises its own error; the signal still decides.
        print(f"bench.evidence: {error}", file=sys.stderr)
        return 1 if watch.received is None else 128 + watch.received


if __name__ == "__main__":
    sys.exit(main())
