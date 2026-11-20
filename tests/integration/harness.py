"""Runs the gateway and robot fleets as real processes for the integration tests.

The service under test runs on real time in other processes, so a test waits for process
events: a line on the gateway's stdout, a process exit. wait_for_metrics is the one bounded
poll of the test tree, for a state of the service or for a planned instant of a test, which is
a predicate on the snapshot's monotonic time. No test sleeps.

Fleets and the gateway write their output to files, never to a pipe that nobody reads, except
the gateway's stdout, which a thread reads to the end.

open_raw_session and raw_frame play a robot message by message, for the peers that a robot
process never is: one that breaks a rule, floods, or comes and goes a thousand times.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import select
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.live_check import fetch_metrics
from cloud.config import OVERRIDES_VARIABLE, TOPOLOGY_VARIABLE
from common.commands import MODEL_KEYS
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    FrameHeader,
    Hello,
    decode_welcome,
    encode_frame,
    encode_hello,
)
from common.transport_ws import ClientOptions, WsConnection, connect_ws
from robot.samples import load_samples
from robot.settings import RobotSettings

REPO_ROOT = Path(__file__).resolve().parents[2]
NS_PER_S = 1_000_000_000

# Fake models whose service times are far from every boundary the scheduler draws: depth at
# 40 ms per frame cannot keep up with one robot at 30 fps, and cls at 2 ms per frame keeps up
# with ten. One calibration run keeps the warm-up short.
FAKE_SET: Mapping[str, object] = {
    "calibration_runs": 1,
    "models": {
        "cls": {"runtime": "fake", "fake_frame_ms": 2.0},
        "depth": {"runtime": "fake", "fake_frame_ms": 40.0, "batch_sizes": [1]},
    },
}
# The bounds only keep a broken service from hanging the suite; a healthy one takes seconds.
LISTENING_TIMEOUT_S = 30.0
READY_TIMEOUT_S = 60.0
# The gateway's shutdown is bounded at 7 s with the default settings (the compose files give it
# 10 s); this adds the interpreter's exit.
EXIT_TIMEOUT_S = 15.0
# A fleet process builds its frame set before its time zero, so that several processes started
# one after another share one time zero.
FLEET_LEAD_NS = 2 * NS_PER_S
# A fleet ends at its plan's stop time; this is the bound past it for its last report, its close
# and its samples file.
FLEET_EXIT_MARGIN_S = 30.0
POLL_INTERVAL_S = 0.1
# cls is far from saturation and serves every frame; the rest of a robot's frames allows for
# those in flight or skipped at the end of its run.
SERVED_SHARE = 0.9
# A raw client is a default robot on the wire: its hello, its socket settings and its limits.
ROBOT_SETTINGS = RobotSettings()
ROBOT_CLIENT_OPTIONS = ClientOptions(
    max_message_bytes=ROBOT_SETTINGS.max_message_bytes,
    rcvbuf_bytes=ROBOT_SETTINGS.rcvbuf_bytes,
    sndbuf_bytes=ROBOT_SETTINGS.sndbuf_bytes,
    notsent_lowat_bytes=ROBOT_SETTINGS.notsent_lowat_bytes,
    open_timeout_s=ROBOT_SETTINGS.connect_timeout_ms / 1000,
    close_timeout_s=ROBOT_SETTINGS.close_timeout_ms / 1000,
)
# The JPEG of a raw frame: it passes the gateway's start-marker check and the fake runtime's.
RAW_JPEG = b"\xff\xd8\xff" + bytes(1000)
_LISTENING_PREFIX = "listening on "
_READY_PREFIX = "ready:"

Snapshot = dict[str, Any]
Commands = npt.NDArray[np.void]


@dataclass(frozen=True)
class Service:
    """A gateway process that has bound its port."""

    port: int
    http_url: str
    ws_url: str
    process: subprocess.Popen[str]
    _stdout_lines: queue.Queue[str | None]

    def wait_ready(self, timeout_s: float) -> None:
        """Wait for the ready line, which the gateway prints when it starts accepting."""
        _wait_for_line(self._stdout_lines, _READY_PREFIX, timeout_s)


def _wait_for_line(lines: queue.Queue[str | None], prefix: str, timeout_s: float) -> str:
    """Return the next stdout line that starts with prefix; fail at the deadline or at EOF."""
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            line = lines.get(timeout=max(0.0, deadline - time.monotonic()))
        except queue.Empty:
            raise AssertionError(
                f"no {prefix!r} line from the gateway within {timeout_s} s"
            ) from None
        if line is None:
            raise AssertionError(f"the gateway closed its stdout before a {prefix!r} line")
        if line.startswith(prefix):
            return line


def _merged(base: Mapping[str, object], overrides: Mapping[str, object]) -> dict[str, object]:
    """Merge overrides over base, object by object, as RRI_OVERRIDES merges over the defaults."""
    merged = dict(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = _merged(current, value)
        else:
            merged[key] = value
    return merged


def _read_lines(stream: Iterator[str], lines: queue.Queue[str | None], copy: Path) -> None:
    """Copy each line to the queue and to a file until EOF, then put None."""
    with copy.open("w") as file:
        for line in stream:
            file.write(line)
            file.flush()
            lines.put(line)
    lines.put(None)


@contextlib.contextmanager
def running_service(
    overrides: Mapping[str, object],
    log_dir: Path,
    topology: str = "single",
    wait_ready: bool = True,
    expect_exit: int | None = 0,
    base: Mapping[str, object] = FAKE_SET,
) -> Iterator[Service]:
    """Run `python -m cloud.main` with overrides merged over base, on a free local port.

    On exit, SIGTERM unless the process has already exited, then the exit status is checked
    against expect_exit (None for a test that kills the gateway) and stderr for a traceback.
    """
    settings = _merged(_merged(base, overrides), {"host": "127.0.0.1", "port": 0})
    environment = {
        **os.environ,
        TOPOLOGY_VARIABLE: topology,
        OVERRIDES_VARIABLE: json.dumps(settings),
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    stderr_path = log_dir / "gateway.err"
    lines: queue.Queue[str | None] = queue.Queue()
    with stderr_path.open("w") as stderr:
        process = subprocess.Popen(
            [sys.executable, "-m", "cloud.main"],
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=stderr,
            text=True,
        )
    assert process.stdout is not None
    reader = threading.Thread(
        target=_read_lines, args=(process.stdout, lines, log_dir / "gateway.out"), daemon=True
    )
    reader.start()
    try:
        line = _wait_for_line(lines, _LISTENING_PREFIX, LISTENING_TIMEOUT_S)
        port = int(line.strip().rsplit(":", 1)[1])
        service = Service(
            port, f"http://127.0.0.1:{port}", f"ws://127.0.0.1:{port}/ws", process, lines
        )
        if wait_ready:
            service.wait_ready(READY_TIMEOUT_S)
        yield service
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
        try:
            process.wait(EXIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise AssertionError(f"the gateway did not exit within {EXIT_TIMEOUT_S} s") from None
        finally:
            # Workers inherit the pipe and exit after the gateway, so EOF can come a little later.
            reader.join(EXIT_TIMEOUT_S)
            process.stdout.close()
    errors = stderr_path.read_text()
    assert "Traceback" not in errors, errors
    if expect_exit is not None:
        assert process.returncode == expect_exit, errors


@contextlib.contextmanager
def killed_on_exit(process: subprocess.Popen[Any]) -> Iterator[subprocess.Popen[Any]]:
    """Yield process, and kill it on exit if it is still running, so no test leaves one behind."""
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def time_zero_ns() -> int:
    """Return a fleet time zero FLEET_LEAD_NS ahead on the monotonic clock all processes share.

    Take it once the gateway is ready, just before starting the fleets: the lead is the fleets'
    time to build their frame sets, and a gateway still starting would spend it.
    """
    # ASSUMPTION: A20. Every process on this host reads one monotonic clock, so the plan's time
    # zero, the robots' timestamps and the gateway's ts_mono_ns are comparable.
    return time.monotonic_ns() + FLEET_LEAD_NS


def plan(
    t0_ns: int, duration_s: float, robots: Sequence[tuple[str, float, float | None]]
) -> dict[str, object]:
    """Return a fleet plan: each robot as (robot_id, start_s, stop_s or None for the end)."""
    return {
        "t0_mono_ns": t0_ns,
        "duration_s": duration_s,
        "robots": [
            {"robot_id": robot_id, "start_s": start_s, "stop_s": stop_s}
            for robot_id, start_s, stop_s in robots
        ],
    }


def start_fleet(
    service: Service, fleet_plan: Mapping[str, object], samples_dir: Path, *args: str
) -> subprocess.Popen[bytes]:
    """Start one fleet process on the plan; its samples and its log go to samples_dir.

    Each fleet process has a samples directory of its own, which holds plan.json, fleet.log
    (stdout and stderr) and robot_<id>.npz.
    """
    samples_dir.mkdir(parents=True)
    plan_path = samples_dir / "plan.json"
    plan_path.write_text(json.dumps(fleet_plan))
    with (samples_dir / "fleet.log").open("wb") as log:
        return subprocess.Popen(
            [
                sys.executable,
                "-m",
                "robot.fleet",
                *("--url", service.ws_url),
                *("--plan", str(plan_path)),
                *("--samples-dir", str(samples_dir)),
                *args,
            ],
            cwd=REPO_ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def fleet_timeout_s(fleet_plan: Mapping[str, object]) -> float:
    """Return how long to wait, from now, for a fleet started on this plan to exit."""
    t0_ns = fleet_plan["t0_mono_ns"]
    duration_s = fleet_plan["duration_s"]
    assert isinstance(t0_ns, int)
    assert isinstance(duration_s, int | float)
    return (t0_ns - time.monotonic_ns()) / 1e9 + duration_s + FLEET_EXIT_MARGIN_S


def run_fleets(
    service: Service, fleets: Sequence[tuple[Mapping[str, object], Path, tuple[str, ...]]]
) -> None:
    """Run fleet processes side by side, each (plan, samples_dir, args), until each exits 0."""
    with contextlib.ExitStack() as stack:
        processes = [
            stack.enter_context(
                killed_on_exit(start_fleet(service, fleet_plan, samples_dir, *args))
            )
            for fleet_plan, samples_dir, args in fleets
        ]
        statuses = [
            process.wait(fleet_timeout_s(fleet_plan))
            for process, (fleet_plan, _, _) in zip(processes, fleets, strict=True)
        ]
    assert statuses == [0] * len(fleets)


def fleet_log(samples_dir: Path) -> str:
    return (samples_dir / "fleet.log").read_text()


def robot_commands(samples_dir: Path, robot_id: str) -> Commands:
    commands, _ticks, _meta = load_samples(samples_dir / f"robot_{robot_id}.npz")
    return commands


def rate_per_s(commands: Commands, model: str, start_ns: int, end_ns: int) -> float:
    """Return the rate of the commands of a model that the robot received in [start_ns, end_ns)."""
    received_ns = commands["receive_ns"][commands["model"] == MODEL_KEYS.index(model)]
    count = np.count_nonzero((received_ns >= start_ns) & (received_ns < end_ns))
    return int(count) / ((end_ns - start_ns) / NS_PER_S)


def silent_seconds(
    commands: Commands, model: str, t0_ns: int, start_s: int, end_s: int
) -> list[int]:
    """Return each whole second from start_s to end_s after t0 with no command of the model."""
    return [
        second
        for second in range(start_s, end_s)
        if rate_per_s(commands, model, t0_ns + second * NS_PER_S, t0_ns + (second + 1) * NS_PER_S)
        == 0
    ]


def first_receive_ns(commands: Commands, model: str) -> int:
    """Return when the robot received its first command of a model."""
    return int(commands["receive_ns"][commands["model"] == MODEL_KEYS.index(model)].min())


def relative_spread(values: Sequence[float]) -> float:
    """Return (max - min) / max: how far apart a set of counts or rates is."""
    return (max(values) - min(values)) / max(values)


def scrape(service: Service) -> Snapshot:
    return fetch_metrics(service.http_url)


def wait_for_metrics(
    service: Service, predicate: Callable[[Snapshot], bool], timeout_s: float
) -> Snapshot:
    """Scrape every POLL_INTERVAL_S until predicate holds, and return that snapshot.

    The one bounded poll of the test tree: the service runs on real time in other processes, and
    a planned instant of a test is a predicate on the snapshot's ts_mono_ns.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        snapshot = scrape(service)
        if predicate(snapshot):
            return snapshot
        if time.monotonic() >= deadline:
            raise AssertionError(f"/metrics did not reach the awaited state within {timeout_s} s")
        time.sleep(POLL_INTERVAL_S)


def gains(before: Snapshot, after: Snapshot, group: str) -> dict[str, int]:
    """Return how much each reason of connections.<group> rose from before to after."""
    return {
        reason: count - before["connections"][group][reason]
        for reason, count in after["connections"][group].items()
    }


def at_or_after(instant_ns: int) -> Callable[[Snapshot], bool]:
    """Return a predicate that holds once the gateway's clock has reached instant_ns."""
    return lambda snapshot: snapshot["ts_mono_ns"] >= instant_ns


def wait_for_exit(pid: int, timeout_s: float) -> bool:
    """Wait for a process that is not this one's child to exit; whether it did by the deadline."""
    try:
        descriptor = os.pidfd_open(pid)
    except ProcessLookupError:
        return True
    try:
        poller = select.poll()
        poller.register(descriptor, select.POLLIN)
        # A pidfd becomes readable when its process exits.
        return bool(poller.poll(round(timeout_s * 1000)))
    finally:
        os.close(descriptor)


async def open_raw_session(ws_url: str, robot_id: str) -> tuple[WsConnection, int]:
    """Dial the gateway as a default robot, say its hello, and return the session it is given.

    The caller drives the connection message by message, which a robot process cannot do.
    """
    connection = await connect_ws(ws_url, ROBOT_CLIENT_OPTIONS)
    with contextlib.ExitStack() as stack:
        stack.callback(connection.abort)
        hello = Hello(
            PROTOCOL_VERSION,
            robot_id,
            ROBOT_SETTINGS.token,
            ROBOT_SETTINGS.models,
            ROBOT_SETTINGS.stale_ms,
            ROBOT_SETTINGS.watchdog_ms,
            ROBOT_SETTINGS.frame_rate_hz,
            FRAME_WIDTH_PX,
            FRAME_HEIGHT_PX,
        )
        connection.send_nowait(encode_hello(hello))
        async with asyncio.timeout(ROBOT_SETTINGS.welcome_timeout_ms / 1000):
            welcome = await connection.recv()
        assert isinstance(welcome, str), welcome
        session = decode_welcome(welcome).session
        # The caller owns the connection from here on.
        stack.pop_all()
    return connection, session


def raw_frame(session: int, frame_number: int) -> bytes:
    """Return a frame of the session captured and sent now."""
    # ASSUMPTION: A20. The gateway reads the same monotonic clock, so these times are its own.
    now_ns = time.monotonic_ns()
    return encode_frame(FrameHeader(session, frame_number, now_ns, now_ns), RAW_JPEG)
