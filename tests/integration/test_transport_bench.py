"""Smoke tests of the transport benchmark: both subcommands run end to end and report consistently.

Each test runs the tool as a separate process, as it is run by hand, so its
children start the way they do in a measurement. --saturate is not run here:
its load needs the model files, which exist only in the image.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from bench.transport_bench import LOOPS

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the transport benchmark needs Linux: uvloop, TCP_NOTSENT_LOWAT, /proc and cgroup files",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# A run takes seconds. The bound only keeps a hung run from hanging the suite.
RUN_TIMEOUT_S = 120

REPORT_KEYS = {"kind", "args", "environment", "frames", "results"}
DISPATCH_ENTRY_KEYS = {
    "batch_size",
    "round_trips",
    "wall_ms",
    "gateway_cpu_us_per_round_trip",
    "worker_cpu_us_per_round_trip",
}
WEBSOCKET_RESULT_KEYS = {
    "window_s",
    "processes",
    "cpu_us_per_message",
    "cgroup",
    "sender",
    "receiver",
    "saturate",
}
PROCESS_KEYS = {"pid", "cpu_user_s", "cpu_system_s", "cpu_s", "cpu_cores"}
CGROUP_KEYS = {"cpu_usage_us", "nr_periods", "nr_throttled", "throttled_us"}
SENDER_KEYS = {
    "ticks",
    "frames_sent",
    "ticks_skipped",
    "replies_received",
    "fire_lag_ms",
    "round_trip_ms",
    "socket",
    "stalled_path",
}
STALLED_PATH_KEYS = {"frames_taken", "receiver_queued_bytes", "robot_unsent_bytes"}
RECEIVER_KEYS = {"messages", "loop_lag_ms", "socket"}


def run_bench(out: Path, *arguments: str) -> dict[str, Any]:
    """Run the benchmark, check that it ended cleanly, and return its report."""
    completed = subprocess.run(
        [sys.executable, "-m", "bench.transport_bench", *arguments, "--out", str(out)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )
    assert (completed.returncode, completed.stderr) == (0, "")
    assert completed.stdout.endswith(f"wrote {out}\n")
    report: dict[str, Any] = json.loads(out.read_text())
    return report


@pytest.mark.parametrize("loop", LOOPS)
def test_dispatch_reports_each_batch_size_in_order(tmp_path: Path, loop: str) -> None:
    report = run_bench(
        tmp_path / "dispatch.json",
        *("dispatch", "--batch-sizes", "1,3", "--round-trips", "5", "--loop", loop),
    )

    results = report["results"]
    assert set(report) == REPORT_KEYS
    assert report["kind"] == "transport_bench.dispatch"
    assert [set(entry) for entry in results] == [DISPATCH_ENTRY_KEYS] * 2
    assert [entry["batch_size"] for entry in results] == [1, 3]
    assert [entry["wall_ms"]["count"] for entry in results] == [5, 5]
    assert all(entry["gateway_cpu_us_per_round_trip"] > 0 for entry in results)
    assert all(entry["worker_cpu_us_per_round_trip"] > 0 for entry in results)


@pytest.mark.parametrize("loop", LOOPS)
def test_websocket_counts_every_frame_once_on_each_side(tmp_path: Path, loop: str) -> None:
    report = run_bench(
        tmp_path / "websocket.json",
        *("websocket", "--robots", "2", "--fps", "30", "--duration-s", "1"),
        *("--loop", loop, "--seed", "1"),
    )

    results = report["results"]
    sender = results["sender"]
    assert set(report) == REPORT_KEYS
    assert report["kind"] == "transport_bench.websocket"
    assert set(results) == WEBSOCKET_RESULT_KEYS
    assert {name: set(entry) for name, entry in results["processes"].items()} == {
        "receiver": PROCESS_KEYS,
        "sender": PROCESS_KEYS,
    }
    assert set(results["cpu_us_per_message"]) == {"receiver", "sender"}
    assert set(results["cgroup"]) == CGROUP_KEYS
    assert (set(sender), set(results["receiver"]), results["saturate"]) == (
        SENDER_KEYS,
        RECEIVER_KEYS,
        [],
    )
    assert set(sender["stalled_path"]) == STALLED_PATH_KEYS
    # A path whose receiver reads nothing still holds what its window took.
    assert sender["stalled_path"]["receiver_queued_bytes"] > 0
    assert sender["frames_sent"] > 0
    assert sender["frames_sent"] + sender["ticks_skipped"] == sender["ticks"]
    assert (
        sender["replies_received"],
        results["receiver"]["messages"],
        sender["fire_lag_ms"]["count"],
        sender["round_trip_ms"]["count"],
    ) == (sender["frames_sent"],) * 4
