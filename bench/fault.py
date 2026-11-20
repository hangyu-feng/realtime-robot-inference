"""Kills one model's worker process at the time a fault run plans, from the gateway's container.

    python -m bench.fault --run-dir DIR

reads the worker to kill and its time from the scenario in DIR/config.json, waits for the
scenario driver's t0 line in DIR/events.jsonl, and at t0 plus the planned time kills with
SIGKILL the process that /metrics names for that worker. It prints the fault event, which the
runner adds to events.jsonl once the driver has finished, and writes nothing under DIR: the
directory belongs to the host user. It exits 0 once the worker is killed, 1 when it could not
kill it, and 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path

from bench.live_check import MetricsUnavailable, fetch_metrics
from bench.runfiles import CONFIG_FILE, EVENTS_FILE, Event, EventName, RunConfig, RunFileError
from bench.scenarios import WorkerKill
from cloud.config import load_config
from cloud.counters import WorkerState
from common.clock import Clock, MonotonicClock
from common.errors import ConfigError

# The driver writes its t0 line a few seconds after it starts, and the runner starts this
# process just before the driver. A poll every 100 ms sees the line well before t0; the wait
# is bounded for a driver that never writes it.
POLL_INTERVAL_S = 0.1
T0_WAIT_S = 60.0

_NS_PER_S = 1_000_000_000

type Fetch = Callable[[], Mapping[str, object]]
type Kill = Callable[[int, int], None]


class FaultError(Exception):
    """The worker was not killed: no kill planned, no t0 in time, or no live worker to kill."""


def _planned_kill(run_dir: Path) -> WorkerKill:
    """Return the worker kill of the run's scenario."""
    try:
        config = RunConfig.from_json((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, RunFileError) as error:
        raise FaultError(f"cannot read the run's {CONFIG_FILE}: {error}") from error
    fault = None if config.fleet is None else config.fleet.scenario.fault
    if fault is None:
        raise FaultError(f"the scenario of {run_dir} kills no worker")
    return fault


def _t0_ns(events_path: Path, clock: Clock, sleep: Callable[[float], None]) -> int:
    """Return t0 from the first line of events.jsonl, polling until the driver has written it."""
    deadline_ns = clock.now_ns() + round(T0_WAIT_S * _NS_PER_S)
    while True:
        try:
            with events_path.open(encoding="utf-8") as file:
                line = file.readline()
        except FileNotFoundError:
            line = ""
        except (OSError, UnicodeDecodeError) as error:
            raise FaultError(f"cannot read {EVENTS_FILE}: {error}") from error
        # A line without its newline is still being written.
        if line.endswith("\n"):
            break
        if clock.now_ns() >= deadline_ns:
            raise FaultError(f"no t0 line in {EVENTS_FILE} within {T0_WAIT_S:g} s")
        sleep(POLL_INTERVAL_S)
    try:
        event = Event.from_json(line)
    except RunFileError as error:
        raise FaultError(str(error)) from error
    # Only a t0 line has the field. The driver writes t0 first, so a first line of any other
    # kind will never become one.
    if event.t0_mono_ns is None:
        raise FaultError(f"the first line of {EVENTS_FILE} is a {event.name} event, not t0")
    return event.t0_mono_ns


def _ready_pid(fetch: Fetch, model: str) -> int:
    """Return the pid of the model's worker, which /metrics must show ready."""
    try:
        metrics = fetch()
    except MetricsUnavailable as error:
        raise FaultError(str(error)) from error
    workers = metrics.get("workers")
    worker = workers.get(model) if isinstance(workers, Mapping) else None
    if not isinstance(worker, Mapping):
        raise FaultError(f"/metrics shows no {model} worker")
    state, pid = worker.get("state"), worker.get("pid")
    # A bool is an int to isinstance, and no pid.
    if state != WorkerState.READY or isinstance(pid, bool) or not isinstance(pid, int):
        raise FaultError(f"the {model} worker is not ready with a pid: state {state}, pid {pid}")
    return pid


def inject(
    run_dir: Path, clock: Clock, sleep: Callable[[float], None], fetch: Fetch, kill: Kill
) -> Event:
    """Kill the worker that the run's scenario kills, at its planned time; return the fault event.

    The event's time is the clock read at once after the kill returned, from t0. FaultError
    when the scenario kills no worker, the driver's t0 line has not come within T0_WAIT_S,
    /metrics shows the worker not ready or without a pid, or the kill is refused.
    """
    fault = _planned_kill(run_dir)
    t0_ns = _t0_ns(run_dir / EVENTS_FILE, clock, sleep)
    kill_at_ns = t0_ns + round(fault.at_s * _NS_PER_S)
    # A sleep may end before its time, so the clock decides when it is over.
    while (now_ns := clock.now_ns()) < kill_at_ns:
        sleep((kill_at_ns - now_ns) / _NS_PER_S)
    pid = _ready_pid(fetch, fault.model)
    try:
        kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError) as error:
        raise FaultError(f"cannot kill the {fault.model} worker, pid {pid}: {error}") from error
    killed_ns = clock.now_ns()
    # ASSUMPTION: A20. This container's monotonic clock is the one the driver read t0 on, in
    # either topology, so the kill is timed from t0 here.
    return Event(EventName.FAULT, (killed_ns - t0_ns) / _NS_PER_S, model=fault.model, pid=pid)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.fault",
        description="Kill the worker that a fault run's scenario kills, at its planned time.",
    )
    parser.add_argument("--run-dir", type=Path, required=True, help="the run's directory")
    arguments = parser.parse_args(argv)
    try:
        # The gateway of this container, at the port its own environment gives it.
        port = load_config(os.environ).port
        event = inject(
            arguments.run_dir,
            MonotonicClock(),
            time.sleep,
            partial(fetch_metrics, f"http://127.0.0.1:{port}"),
            os.kill,
        )
    except (ConfigError, FaultError) as error:
        print(f"bench.fault: {error}", file=sys.stderr)
        return 1
    print(event.to_json())
    return 0


if __name__ == "__main__":
    sys.exit(main())
