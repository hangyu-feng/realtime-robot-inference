"""Tests for the fault injector: when it kills the worker, what it reports, and when it refuses."""

from __future__ import annotations

import math
import signal
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from bench.fault import POLL_INTERVAL_S, T0_WAIT_S, FaultError, inject, main
from bench.live_check import MetricsUnavailable
from bench.runfiles import CONFIG_FILE, EVENTS_FILE, Event, EventName, FleetConfig, RunConfig
from bench.runplan import planned_events, run_id
from bench.scenarios import Scenario, disconnect, worker_kill
from common.clock import FakeClock
from common.commands import MODEL_KEYS
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from tests.support.runs import run_spec

NS_PER_S = 1_000_000_000
# The runner starts the injector just before the driver, which sets t0 a few seconds ahead.
START_NS = 100 * NS_PER_S
T0_NS = 106 * NS_PER_S
T0_LINE = Event(EventName.T0, 0.0, t0_mono_ns=T0_NS).to_json() + "\n"
# worker_kill kills the depth worker 20 s after t0.
KILL_NS = T0_NS + 20 * NS_PER_S
DEPTH_PID = 4242
READY: Mapping[str, object] = {
    "workers": {
        "cls": {"state": "ready", "pid": 4241},
        "depth": {"state": "ready", "pid": DEPTH_PID},
    }
}
# What the kill itself takes, so that the time read after it differs from the time before.
KILL_TAKES_NS = 250_000


@dataclass
class Container:
    """The clock, /metrics and the processes of the gateway's container, with real behaviour.

    A sleep advances the clock by its share of what it was asked for, at least 1 ns, and then
    appends to events.jsonl each write that is due by the new time. /metrics answers with the
    given snapshot, or fails where it is None. A kill takes KILL_TAKES_NS, or raises the given
    error.
    """

    run_dir: Path
    writes: list[tuple[int, str]] = field(default_factory=list)
    metrics: Mapping[str, object] | None = field(default_factory=lambda: READY)
    sleep_share: float = 1.0
    kill_error: OSError | None = None
    clock: FakeClock = field(default_factory=lambda: FakeClock(START_NS))
    fetched_ns: list[int] = field(default_factory=list)
    killed: list[tuple[int, int, int]] = field(default_factory=list)

    def write_due(self) -> None:
        while self.writes and self.writes[0][0] <= self.clock.now_ns():
            _, text = self.writes.pop(0)
            with (self.run_dir / EVENTS_FILE).open("a", encoding="utf-8") as file:
                file.write(text)

    def sleep(self, seconds: float) -> None:
        self.clock.advance_ns(max(1, math.ceil(round(seconds * NS_PER_S) * self.sleep_share)))
        self.write_due()

    def fetch(self) -> Mapping[str, object]:
        self.fetched_ns.append(self.clock.now_ns())
        if self.metrics is None:
            raise MetricsUnavailable("GET http://127.0.0.1:8080/metrics failed: refused")
        return self.metrics

    def kill(self, pid: int, signal_number: int) -> None:
        if self.kill_error is not None:
            raise self.kill_error
        self.killed.append((pid, signal_number, self.clock.now_ns()))
        self.clock.advance_ns(KILL_TAKES_NS)

    def run_injector(self) -> Event:
        self.write_due()
        return inject(self.run_dir, self.clock, self.sleep, self.fetch, self.kill)


def write_config(run_dir: Path, scenario: Scenario) -> None:
    spec = run_spec(scenario.name)
    fleet = FleetConfig(
        scenario=scenario,
        frame_rate_hz=30.0,
        models=MODEL_KEYS,
        stale_ms=DEFAULT_STALE_MS,
        watchdog_ms=DEFAULT_WATCHDOG_MS,
        planned_events=planned_events(scenario),
    )
    config = RunConfig(spec=spec, run_id=run_id(spec), seed=1, extra_compose_path=None, fleet=fleet)
    (run_dir / CONFIG_FILE).write_text(config.to_json(), encoding="utf-8")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """A worker_kill run directory, before the driver has written anything."""
    write_config(tmp_path, worker_kill())
    return tmp_path


@pytest.mark.parametrize("sleep_share", [1.0, 0.5], ids=["sleeps on time", "sleeps that end early"])
def test_the_worker_is_killed_with_sigkill_at_t0_plus_the_planned_time(
    run_dir: Path, sleep_share: float
) -> None:
    container = Container(run_dir, writes=[(START_NS, T0_LINE)], sleep_share=sleep_share)

    event = container.run_injector()

    assert container.fetched_ns == [KILL_NS]
    assert container.killed == [(DEPTH_PID, signal.SIGKILL, KILL_NS)]
    # The time is read once the kill has returned.
    assert event == Event(EventName.FAULT, 20.00025, model="depth", pid=DEPTH_PID)


def test_a_t0_line_that_appears_after_polls_is_waited_for_until_it_is_complete(
    run_dir: Path,
) -> None:
    # No events.jsonl at the start; then the driver's first line, written in two parts.
    writes = [(START_NS + 250_000_000, T0_LINE[:12]), (START_NS + 450_000_000, T0_LINE[12:])]
    container = Container(run_dir, writes=writes)

    event = container.run_injector()

    assert container.killed == [(DEPTH_PID, signal.SIGKILL, KILL_NS)]
    assert event.t_s == pytest.approx(20.00025)


@pytest.mark.parametrize(
    "writes",
    [[], [(START_NS, T0_LINE[:-1])]],
    ids=["no events file", "a first line without its end"],
)
def test_no_t0_line_within_the_bound_kills_nothing(
    run_dir: Path, writes: list[tuple[int, str]]
) -> None:
    container = Container(run_dir, writes=writes)

    with pytest.raises(FaultError, match=f"no t0 line in {EVENTS_FILE} within 60 s"):
        container.run_injector()

    assert (container.fetched_ns, container.killed) == ([], [])
    waited_ns = container.clock.now_ns() - START_NS
    assert T0_WAIT_S * NS_PER_S <= waited_ns < (T0_WAIT_S + POLL_INTERVAL_S) * NS_PER_S


def test_a_first_line_that_is_not_t0_kills_nothing(run_dir: Path) -> None:
    ready_line = Event(EventName.SERVER_READY, -5.0).to_json() + "\n"
    container = Container(run_dir, writes=[(START_NS, ready_line + T0_LINE)])

    with pytest.raises(FaultError, match="is a server_ready event, not t0"):
        container.run_injector()

    assert container.killed == []


@pytest.mark.parametrize(
    "metrics",
    [
        {"workers": {"depth": {"state": "restarting", "pid": DEPTH_PID}}},
        {"workers": {"depth": {"state": "ready", "pid": None}}},
        {"workers": {"depth": {"state": "ready", "pid": True}}},
        {"workers": {"cls": {"state": "ready", "pid": 4241}}},
        {},
        None,
    ],
    ids=["restarting", "no pid", "a pid that is a bool", "no depth worker", "no workers", "down"],
)
def test_a_worker_that_is_not_ready_with_a_pid_is_not_killed(
    run_dir: Path, metrics: Mapping[str, object] | None
) -> None:
    container = Container(run_dir, writes=[(START_NS, T0_LINE)], metrics=metrics)

    with pytest.raises(FaultError):
        container.run_injector()

    assert container.fetched_ns == [KILL_NS]
    assert container.killed == []


def test_a_run_whose_scenario_kills_no_worker_is_refused_without_waiting(tmp_path: Path) -> None:
    write_config(tmp_path, disconnect())
    container = Container(tmp_path, writes=[(START_NS, T0_LINE)])

    with pytest.raises(FaultError, match="kills no worker"):
        container.run_injector()

    assert (container.clock.now_ns(), container.fetched_ns, container.killed) == (START_NS, [], [])


def test_a_run_directory_without_a_config_is_refused(tmp_path: Path) -> None:
    with pytest.raises(FaultError, match=f"cannot read the run's {CONFIG_FILE}"):
        Container(tmp_path).run_injector()


@pytest.mark.parametrize(
    "error",
    [ProcessLookupError(3, "No such process"), PermissionError(1, "Operation not permitted")],
    ids=["gone", "not ours"],
)
def test_a_kill_that_is_refused_is_an_error(run_dir: Path, error: OSError) -> None:
    container = Container(run_dir, writes=[(START_NS, T0_LINE)], kill_error=error)

    with pytest.raises(FaultError, match=f"cannot kill the depth worker, pid {DEPTH_PID}"):
        container.run_injector()


@pytest.mark.parametrize(
    "argv",
    [[], ["--run-dir"], ["--run-dir", "run", "--model", "depth"]],
    ids=["none", "no value", "unknown"],
)
def test_main_exits_2_on_bad_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(argv)

    assert exit_info.value.code == 2


def test_main_exits_1_with_one_line_on_stderr_when_it_kills_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_config(tmp_path, disconnect())

    status = main(["--run-dir", str(tmp_path)])

    out, err = capsys.readouterr()
    assert (status, out) == (1, "")
    assert err.startswith("bench.fault: the scenario of")
    assert err.count("\n") == 1
