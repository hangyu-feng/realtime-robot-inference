"""The running service against its contracts: the /metrics fields, thread counts, the C mask."""

import importlib
import importlib.machinery
import os
import signal
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.integration.harness import (
    REPO_ROOT,
    Service,
    Snapshot,
    fleet_timeout_s,
    killed_on_exit,
    plan,
    run_fleets,
    running_service,
    scrape,
    start_fleet,
    time_zero_ns,
    wait_for_metrics,
)
from tests.support.contract import contract_fields, snapshot_fields

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# DESIGN.md 4.6: the image's OpenBLAS and OpenMP settings and the explicit OpenCV and ONNX
# Runtime settings keep each process at a few threads, whatever the host's CPU count.
MAX_THREADS = 4
MODEL_DIR_VARIABLE = "MODEL_DIR"
# A report arrives once per second; this bounds the wait for the first one.
TELEMETRY_TIMEOUT_S = 15.0


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Service]:
    with running_service({}, tmp_path_factory.mktemp("gateway")) as running:
        yield running


def test_live_metrics_have_exactly_the_contract_fields(service: Service, tmp_path: Path) -> None:
    t0_ns = time_zero_ns()
    gone_plan = plan(t0_ns, 2.0, [("t16gone", 0.0, None)])
    connected_plan = plan(t0_ns, 60.0, [("t16here", 0.0, None)])

    with killed_on_exit(start_fleet(service, connected_plan, tmp_path / "here")) as connected:
        run_fleets(service, [(gone_plan, tmp_path / "gone", ())])

        def both_robots_reported(snapshot: Snapshot) -> bool:
            robots = snapshot["robots"]
            return "telemetry" in robots.get("t16here", {}) and not robots["t16gone"]["connected"]

        snapshot = wait_for_metrics(service, both_robots_reported, TELEMETRY_TIMEOUT_S)
        # SIGTERM is the fleet's planned stop: a last report and a close with 1000.
        connected.send_signal(signal.SIGTERM)
        connected_status = connected.wait(fleet_timeout_s(connected_plan))

    design = (REPO_ROOT / "DESIGN.md").read_text()
    assert connected_status == 0
    assert snapshot["robots"]["t16here"]["connected"]
    assert snapshot_fields(snapshot) == contract_fields(design)


@pytest.mark.skipif(
    os.environ.get(MODEL_DIR_VARIABLE) is None,
    reason=f"{MODEL_DIR_VARIABLE} is not set: the thread settings are the image's environment",
)
def test_each_process_runs_a_few_threads_whatever_the_host(service: Service) -> None:
    threads = {name: process["threads"] for name, process in scrape(service)["processes"].items()}

    assert set(threads) == {"gateway", "worker_cls", "worker_depth"}
    assert max(threads.values()) <= MAX_THREADS, threads


def test_the_websocket_mask_runs_in_the_c_extension() -> None:
    # Robots mask every frame they send; the pure-Python mask would be charged to the fleet.
    speedups = importlib.import_module("websockets.speedups")

    assert speedups.__file__ is not None
    assert speedups.__file__.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES))
