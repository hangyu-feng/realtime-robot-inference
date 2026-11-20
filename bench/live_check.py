"""Checks a running service against the robots' samples, and reads its CPU while it is idle.

agreement compares what one robot reported in its telemetry with the per-command samples it
wrote, conservation_violations checks the identities of DESIGN.md 10.3 in a /metrics snapshot,
and boundary_violations compares the robot's counters with the gateway's. Each returns the
problems it found, empty when everything agrees.

    python -m bench.live_check agreement --url http://HOST:PORT --samples-dir DIR
    python -m bench.live_check idle --url http://HOST:PORT --window-s S

Both commands state numbers and make no judgment of them beyond the checks named here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.stats import summarize
from common.commands import MODEL_KEYS
from common.histogram import BUCKETS_PER_OCTAVE, Histogram
from robot.samples import load_samples

# The quantiles the done criterion compares.
QUANTILES = (0.5, 0.95, 0.99)
# A histogram quantile is the upper edge of the bucket that holds the sample of its rank, so it
# lies in [sample, sample * BUCKET_RATIO): DESIGN.md 10.1.
BUCKET_RATIO = 2 ** (1 / BUCKETS_PER_OCTAVE)
# The bucket edges are computed with a power and the samples with a division, so a value on an
# edge may differ from it in the last bits.
RELATIVE_TOLERANCE = 1e-9
# The eight server stages travel as whole microseconds, floored from cumulative offsets, so on a
# shared clock their sum is below the robot's own interval by less than one microsecond.
RESIDUAL_LIMIT_NS = 1000
# A scrape of a local service answers in milliseconds; the bound only keeps a hung service from
# hanging the command.
FETCH_TIMEOUT_S = 10.0
SAMPLES_PATTERN = "robot_*.npz"
_SAMPLES_PREFIX = "robot_"
_PROCESSES = ("gateway", "worker_cls", "worker_depth")


class MetricsUnavailable(Exception):
    """The service did not answer /metrics with a JSON object."""


def fetch_metrics(url: str) -> dict[str, Any]:
    """Return the /metrics snapshot of the service at url (http://HOST:PORT)."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/metrics", timeout=FETCH_TIMEOUT_S) as reply:
            body = reply.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise MetricsUnavailable(f"GET {url}/metrics failed: {error}") from error
    try:
        snapshot = json.loads(body)
    except ValueError as error:
        raise MetricsUnavailable(f"/metrics is not JSON: {error}") from error
    if not isinstance(snapshot, dict):
        raise MetricsUnavailable("/metrics is not a JSON object")
    return snapshot


def _session_rows(commands: npt.NDArray[np.void], session: int, model: str) -> npt.NDArray[np.void]:
    selected = (commands["session"] == session) & (commands["model"] == MODEL_KEYS.index(model))
    rows: npt.NDArray[np.void] = commands[selected]
    return rows


def session_freshness_ms(
    commands: npt.NDArray[np.void], session: int, model: str
) -> npt.NDArray[np.float64]:
    """Return the freshness of one session's commands of one model, as the robot records it."""
    rows = _session_rows(commands, session, model)
    # The quotient the robot's histogram records (receive minus capture, in ns, over 1e6): it is
    # correctly rounded, so the maxima compare exactly.
    freshness_ns = rows["receive_ns"].astype(np.int64) - rows["capture_ns"].astype(np.int64)
    return freshness_ns / 1e6


def agreement(
    metrics: Mapping[str, object], robot_id: str, commands: npt.NDArray[np.void]
) -> list[str]:
    """Compare a robot's telemetry in /metrics with the samples of its current session."""
    # JSON from /metrics, whose shape is the contract of DESIGN.md 10.
    snapshot: Any = metrics
    robot = snapshot["robots"].get(robot_id)
    if robot is None:
        return [f"{robot_id}: not in /metrics"]
    telemetry = robot.get("telemetry")
    if telemetry is None:
        return [f"{robot_id}: no telemetry report"]
    session = robot["session"]
    problems: list[str] = []
    for model in robot["subscribed"]:
        label = f"{robot_id} {model}"
        freshness_ms = np.sort(session_freshness_ms(commands, session, model))
        reported = telemetry["models"][model]
        received = reported["commands_received"]
        if received != freshness_ms.size:
            problems.append(f"{label}: commands_received {received}, {freshness_ms.size} samples")
        problems.extend(
            _histogram_problems(label, Histogram.from_dict(reported["freshness_ms"]), freshness_ms)
        )
        if snapshot["config"]["shared_clock"]:
            problems.extend(_residual_problems(label, commands, session, model))
    return problems


def _histogram_problems(
    label: str, histogram: Histogram, freshness_ms: npt.NDArray[np.float64]
) -> list[str]:
    """Check that a histogram holds exactly the given sorted samples, within its buckets."""
    count = freshness_ms.size
    if histogram.count != count:
        return [f"{label}: histogram count {histogram.count}, {count} samples"]
    if count == 0:
        return []
    problems: list[str] = []
    samples_max_ms = float(freshness_ms[-1])
    if histogram.max_ms != samples_max_ms:
        problems.append(f"{label}: histogram max {histogram.max_ms} ms, samples {samples_max_ms}")
    for q in QUANTILES:
        sample_ms = float(freshness_ms[min(count, max(1, math.ceil(q * count))) - 1])
        quantile_ms = histogram.quantile_ms(q)
        low_ms = sample_ms * (1 - RELATIVE_TOLERANCE)
        high_ms = sample_ms * BUCKET_RATIO * (1 + RELATIVE_TOLERANCE)
        if not low_ms <= quantile_ms <= high_ms:
            problems.append(
                f"{label}: histogram q{q} {quantile_ms} ms is outside the bucket of the "
                f"sample of its rank, {sample_ms} ms"
            )
    return problems


def _residual_problems(
    label: str, commands: npt.NDArray[np.void], session: int, model: str
) -> list[str]:
    """Check that the server stages add up to result minus send, within one microsecond."""
    rows = _session_rows(commands, session, model)
    stages_ns = 1000 * rows["stage_us"].astype(np.int64).sum(axis=1)
    residual_ns = rows["result_ns"].astype(np.int64) - rows["send_ns"].astype(np.int64) - stages_ns
    outside = residual_ns[(residual_ns < 0) | (residual_ns >= RESIDUAL_LIMIT_NS)]
    if outside.size == 0:
        return []
    return [
        f"{label}: {outside.size} samples with result - send - stages outside "
        f"[0, {RESIDUAL_LIMIT_NS}) ns, first {int(outside[0])} ns"
    ]


def conservation_violations(metrics: Mapping[str, object]) -> list[str]:
    """Check the three identities of DESIGN.md 10.3 for every robot and model."""
    snapshot: Any = metrics
    problems: list[str] = []
    for robot_id, robot in snapshot["robots"].items():
        if robot["frames_in"] != robot["frames_rejected"] + robot["frames_accepted"]:
            problems.append(f"{robot_id}: frames_in != frames_rejected + frames_accepted")
        for model, counters in robot["models"].items():
            received = (
                counters["frames_served"]
                + counters["frames_overwritten"]
                + counters["frames_skipped"]
                + counters["frames_dropped_disconnect"]
                + counters["frames_pending"]
            )
            if counters["frames_received"] != received:
                problems.append(f"{robot_id} {model}: frames_received is not conserved")
            served = (
                counters["commands_sent"]
                + counters["commands_queued"]
                + counters["commands_dropped_downlink"]
                + counters["results_dropped_disconnect"]
                + counters["frames_failed"]
                + counters["in_flight"]
            )
            if counters["frames_served"] != served:
                problems.append(f"{robot_id} {model}: frames_served is not conserved")
    return problems


def boundary_violations(metrics: Mapping[str, object], robot_id: str) -> list[str]:
    """Compare a robot's counters with the gateway's across the connection between them.

    Valid for a robot with one session (telemetry reconnects 0) and no link delay, whose last
    report has arrived: frames it sent are frames the gateway took in, and commands the gateway
    sent are commands it received, but for those still in flight when it stopped. The gateway
    keeps a robot id's counters across sessions, and its session numbers are not counted per id,
    so the id must also be new to the gateway: run the check against a gateway started for it.
    """
    snapshot: Any = metrics
    robot = snapshot["robots"].get(robot_id)
    if robot is None:
        return [f"{robot_id}: not in /metrics"]
    telemetry = robot.get("telemetry")
    if telemetry is None:
        return [f"{robot_id}: no telemetry report"]
    if telemetry["reconnects"] != 0:
        return [f"{robot_id}: reconnects {telemetry['reconnects']}, the check needs one session"]
    problems: list[str] = []
    if telemetry["frames_sent"] != robot["frames_in"]:
        problems.append(
            f"{robot_id}: frames_sent {telemetry['frames_sent']}, frames_in {robot['frames_in']}"
        )
    # Commands popped for the robot but not yet read when it stopped are at most what its
    # downlink queue holds: a tolerance, not an identity.
    tolerance = snapshot["config"]["downlink_queue_commands"]
    for model in robot["subscribed"]:
        sent = robot["models"][model]["commands_sent"]
        received = telemetry["models"][model]["commands_received"]
        if not 0 <= sent - received <= tolerance:
            problems.append(
                f"{robot_id} {model}: commands_sent {sent}, commands_received {received}, "
                f"outside [0, {tolerance}] apart"
            )
    return problems


def _has_link_delay(meta: Mapping[str, object]) -> bool:
    """Whether a robot's samples were taken with a delay or jitter in either direction."""
    settings: Any = meta["settings"]
    return any(
        settings[direction][field] != 0
        for direction in ("uplink", "downlink")
        for field in ("delay_ms", "jitter_ms")
    )


def _model_report(
    metrics: Mapping[str, object], robot_id: str, commands: npt.NDArray[np.void]
) -> dict[str, object]:
    """Return, per model, the exact summary of the samples beside the histogram's."""
    snapshot: Any = metrics
    robot = snapshot["robots"].get(robot_id)
    if robot is None or "telemetry" not in robot:
        return {}
    report: dict[str, object] = {}
    for model in robot["subscribed"]:
        freshness_ms = session_freshness_ms(commands, robot["session"], model)
        histogram = Histogram.from_dict(robot["telemetry"]["models"][model]["freshness_ms"])
        report[model] = {
            "samples_ms": summarize(freshness_ms.tolist()) if freshness_ms.size else None,
            "histogram_ms": {
                "count": histogram.count,
                **{f"p{round(q * 100)}": histogram.quantile_ms(q) for q in QUANTILES},
                "max": histogram.max_ms,
            }
            if histogram.count
            else None,
        }
    return report


def _agreement_command(arguments: argparse.Namespace) -> int:
    paths = sorted(arguments.samples_dir.glob(SAMPLES_PATTERN))
    # A directory without samples would otherwise pass every check vacuously.
    if not paths:
        print(f"live_check: no {SAMPLES_PATTERN} in {arguments.samples_dir}", file=sys.stderr)
        return 2
    metrics = fetch_metrics(arguments.url)
    robots: dict[str, object] = {}
    failed = False
    for path in paths:
        robot_id = path.stem.removeprefix(_SAMPLES_PREFIX)
        commands, _ticks, meta = load_samples(path)
        problems = agreement(metrics, robot_id, commands)
        if not _has_link_delay(meta):
            problems += boundary_violations(metrics, robot_id)
        failed = failed or bool(problems)
        robots[robot_id] = {
            "models": _model_report(metrics, robot_id, commands),
            "problems": problems,
        }
    conservation = conservation_violations(metrics)
    print(json.dumps({"robots": robots, "conservation": conservation}, indent=2))
    return 1 if failed or conservation else 0


def _cpu_ms(process: Mapping[str, int]) -> int:
    return process["cpu_user_ms"] + process["cpu_system_ms"]


def _idle_command(arguments: argparse.Namespace) -> int:
    first = fetch_metrics(arguments.url)
    # The command's own wait: the window it reports on.
    time.sleep(arguments.window_s)
    second = fetch_metrics(arguments.url)
    processes: dict[str, int | None] = {}
    for name in _PROCESSES:
        before = first["processes"][name]
        after = second["processes"][name]
        # A process that is not live, or was replaced, at either scrape has no window.
        same = before is not None and after is not None and before["pid"] == after["pid"]
        processes[name] = _cpu_ms(after) - _cpu_ms(before) if same else None
    connected = [first["connections"]["current"], second["connections"]["current"]]
    report = {
        "window_s": (second["ts_mono_ns"] - first["ts_mono_ns"]) / 1e9,
        "cpu_ms": processes,
        "cgroup_cpu_us": second["cgroup"]["cpu_usage_us"] - first["cgroup"]["cpu_usage_us"],
        "robots_connected": connected,
    }
    print(json.dumps(report, indent=2))
    # A robot connected at either scrape means the window was not idle.
    return 1 if any(connected) else 0


def _positive_seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {text}")
    return value


_COMMANDS: Mapping[str, Callable[[argparse.Namespace], int]] = {
    "agreement": _agreement_command,
    "idle": _idle_command,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.live_check", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("agreement", help="check /metrics against the robots' samples")
    check.add_argument("--url", required=True, help="the service, http://HOST:PORT")
    check.add_argument("--samples-dir", type=Path, required=True, help="the fleet's samples")
    idle = commands.add_parser("idle", help="CPU of the service over a window with no robot")
    idle.add_argument("--url", required=True, help="the service, http://HOST:PORT")
    idle.add_argument("--window-s", type=_positive_seconds, required=True, help="window length")
    arguments = parser.parse_args(argv)
    try:
        return _COMMANDS[arguments.command](arguments)
    except MetricsUnavailable as error:
        print(f"live_check: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
