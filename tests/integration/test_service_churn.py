"""The gateway's tasks, descriptors and connections after robots come and go.

A baseline is the event loop's task count and the gateway's descriptor count that two
consecutive scrapes agree on while no robot is connected: a scrape's own connection, and the
closing of the one before it, can move either count by one.
"""

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from bench.live_check import conservation_violations
from common.transport_ws import WsConnection
from tests.integration.harness import (
    Service,
    Snapshot,
    gains,
    open_raw_session,
    plan,
    raw_frame,
    run_fleets,
    running_service,
    time_zero_ns,
    wait_for_metrics,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# Each only keeps a broken service from hanging the suite. A healthy gateway settles in well
# under a second, and the thousand cycles take about ten.
BASELINE_TIMEOUT_S = 30.0
CHURN_TIMEOUT_S = 120.0
CHURN_CYCLES = 500


def resources(snapshot: Snapshot) -> tuple[int, int]:
    """Return the event loop's task count and the gateway's descriptor count."""
    return snapshot["loop"]["tasks"], snapshot["processes"]["gateway"]["fds"]


def settled_baseline(service: Service) -> Snapshot:
    """Return a snapshot with no robot connected whose resources the scrape before it shares."""
    previous: list[tuple[int, int]] = []

    def settled(snapshot: Snapshot) -> bool:
        idle = snapshot["connections"]["current"] == 0
        agrees = idle and previous == [resources(snapshot)]
        previous[:] = [resources(snapshot)] if idle else []
        return agrees

    return wait_for_metrics(service, settled, BASELINE_TIMEOUT_S)


def back_at(baseline: Snapshot) -> Callable[[Snapshot], bool]:
    """Return a predicate that holds once no robot is connected and the resources are back."""
    return lambda snapshot: (
        snapshot["connections"]["current"] == 0 and resources(snapshot) == resources(baseline)
    )


async def session_with_a_frame(service: Service, robot_id: str) -> WsConnection:
    """Open a session and send it one frame."""
    connection, session = await open_raw_session(service.ws_url, robot_id)
    connection.send_nowait(raw_frame(session, 1))
    return connection


def test_tasks_descriptors_and_connections_return_to_their_baseline_after_robots_churn(
    tmp_path: Path,
) -> None:
    fleet_plan = plan(
        time_zero_ns(),
        3.0,
        [
            ("t15a", 0.0, None),
            ("t15b", 0.5, 2.0),
            ("t15c", 1.0, None),
            ("t15d", 1.5, 2.5),
            ("t15e", 2.0, None),
        ],
    )

    with running_service({}, tmp_path / "gateway") as service:
        baseline = settled_baseline(service)
        run_fleets(service, [(fleet_plan, tmp_path / "fleet", ())])
        settled = wait_for_metrics(service, back_at(baseline), BASELINE_TIMEOUT_S)

    assert gains(baseline, settled, "closed")["internal_error"] == 0
    assert conservation_violations(settled) == []


async def test_a_thousand_connect_and_disconnect_cycles_leave_descriptors_and_tasks_at_baseline(
    tmp_path: Path,
) -> None:
    with running_service({}, tmp_path / "gateway") as service:
        before = await asyncio.to_thread(settled_baseline, service)
        async with asyncio.timeout(CHURN_TIMEOUT_S):
            # Half the robots leave with a close, half drop the connection as a killed one does.
            for cycle in range(CHURN_CYCLES):
                connection = await session_with_a_frame(service, f"churn{cycle:04d}")
                await connection.close(1000, "")
            for cycle in range(CHURN_CYCLES, 2 * CHURN_CYCLES):
                connection = await session_with_a_frame(service, f"churn{cycle:04d}")
                connection.abort()
            connection, _ = await open_raw_session(service.ws_url, "churn-last")
            await connection.close(1000, "")
        after = await asyncio.to_thread(
            wait_for_metrics, service, back_at(before), BASELINE_TIMEOUT_S
        )

    closed = gains(before, after, "closed")
    rejected = gains(before, after, "rejected")
    restarts = [
        after["workers"][model]["restarts"] - worker["restarts"]
        for model, worker in before["workers"].items()
    ]
    assert (
        after["connections"]["accepted"] - before["connections"]["accepted"] == 2 * CHURN_CYCLES + 1
    )
    assert (closed["client_close"], closed["abnormal"]) == (CHURN_CYCLES + 1, CHURN_CYCLES)
    assert closed["internal_error"] == 0
    assert set(rejected.values()) == {0}
    # The robot table stays bounded under 1001 distinct ids.
    assert len(after["robots"]) <= after["config"]["max_robot_entries"]
    assert restarts == [0] * len(restarts)
    assert conservation_violations(after) == []
