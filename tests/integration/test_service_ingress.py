"""The service with real processes when a robot stops reading, breaks a frame rule or floods.

Tests that share a service assert on robot ids they created and on differences of counters.
"""

import asyncio
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from bench.live_check import conservation_violations
from common.transport import TransportClosed
from common.transport_ws import WsConnection
from tests.integration.harness import (
    EXIT_TIMEOUT_S,
    REPO_ROOT,
    SERVED_SHARE,
    Service,
    fleet_timeout_s,
    gains,
    killed_on_exit,
    open_raw_session,
    plan,
    raw_frame,
    robot_commands,
    running_service,
    scrape,
    silent_seconds,
    start_fleet,
    time_zero_ns,
    wait_for_metrics,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# Each session's bucket holds four messages, so a test can overflow it with five at once.
INGRESS_MAX_MESSAGES_PER_S = 4.0
MAX_MESSAGE_BYTES = 65_536
# Bounds every wait on the gateway of a raw session; a healthy one answers in milliseconds.
GATEWAY_TIMEOUT_S = 10.0
# Bounds the wait for the slow-consumer close: the client's buffers fill in about 5 s.
SLOW_CONSUMER_CLOSE_TIMEOUT_S = 20.0


@pytest.fixture(scope="module")
def limited_service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Service]:
    overrides = {
        "ingress_max_messages_per_s": INGRESS_MAX_MESSAGES_PER_S,
        "max_message_bytes": MAX_MESSAGE_BYTES,
    }
    with running_service(overrides, tmp_path_factory.mktemp("gateway")) as running:
        yield running


async def peer_close_code(connection: WsConnection) -> int | None:
    """Read until the connection closes, and return the code of the gateway's close frame.

    Commands of frames accepted before the close may arrive first.
    """
    async with asyncio.timeout(GATEWAY_TIMEOUT_S):
        while True:
            try:
                await connection.recv()
            except TransportClosed as closed:
                return closed.peer_code


def test_a_robot_that_stops_reading_is_closed_as_a_slow_consumer_and_the_others_keep_their_rate(
    tmp_path: Path,
) -> None:
    duration_s = 10

    with running_service({"slow_consumer_timeout_ms": 500}, tmp_path / "gateway") as service:
        before = scrape(service)
        t0_ns = time_zero_ns()
        fleet_plan = plan(t0_ns, duration_s, [("t10normal", 0.0, None)])
        with (
            killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet,
            (tmp_path / "client.log").open("wb") as client_log,
        ):
            client = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "tests.integration.non_reading_client",
                    *("--url", service.ws_url),
                    *("--robot-id", "t10deaf"),
                    *("--duration-s", str(duration_s)),
                ],
                cwd=REPO_ROOT,
                stdout=client_log,
                stderr=subprocess.STDOUT,
            )
            with killed_on_exit(client):
                wait_for_metrics(
                    service,
                    lambda snapshot: gains(before, snapshot, "closed")["slow_consumer"] > 0,
                    SLOW_CONSUMER_CLOSE_TIMEOUT_S,
                )
                client_status = client.wait(EXIT_TIMEOUT_S)
                fleet_status = fleet.wait(fleet_timeout_s(fleet_plan))
        after = scrape(service)

    normal = after["robots"]["t10normal"]
    commands = robot_commands(tmp_path / "fleet", "t10normal")
    served_share = (
        normal["telemetry"]["models"]["cls"]["commands_received"]
        / normal["telemetry"]["frames_sent"]
    )
    assert client_status == 0, (tmp_path / "client.log").read_text()
    assert fleet_status == 0
    assert gains(before, after, "closed")["slow_consumer"] == 1
    assert gains(before, after, "closed")["internal_error"] == 0
    assert after["robots"]["t10deaf"]["connected"] is False
    assert silent_seconds(commands, "cls", t0_ns, 1, duration_s) == []
    assert served_share >= SERVED_SHARE
    assert conservation_violations(after) == []


async def test_a_frame_with_a_bad_header_is_dropped_and_the_next_frame_is_accepted(
    limited_service: Service,
) -> None:
    before = await asyncio.to_thread(scrape, limited_service)
    connection, session = await open_raw_session(limited_service.ws_url, "t11a")
    try:
        # Another session's number.
        connection.send_nowait(raw_frame(session + 1, 1))
        connection.send_nowait(raw_frame(session, 2))
        after = await asyncio.to_thread(
            wait_for_metrics,
            limited_service,
            lambda snapshot: snapshot["robots"]["t11a"]["frames_in"] == 2,
            GATEWAY_TIMEOUT_S,
        )
    finally:
        await connection.close(1000, "")

    robot = after["robots"]["t11a"]
    assert robot["frames_rejected"] == 1
    assert robot["frames_accepted"] == 1
    assert robot["connected"] is True
    assert gains(before, after, "frames_rejected")["malformed"] == 1


async def test_a_message_over_the_size_limit_closes_the_connection_with_1009(
    limited_service: Service,
) -> None:
    before = await asyncio.to_thread(scrape, limited_service)
    connection, _ = await open_raw_session(limited_service.ws_url, "t11b")
    try:
        connection.send_nowait(bytes(MAX_MESSAGE_BYTES + 1))
        close_code = await peer_close_code(connection)
    finally:
        connection.abort()
    after = await asyncio.to_thread(
        wait_for_metrics,
        limited_service,
        lambda snapshot: not snapshot["robots"]["t11b"]["connected"],
        GATEWAY_TIMEOUT_S,
    )

    assert close_code == 1009
    assert gains(before, after, "frames_rejected")["oversize"] == 1
    assert gains(before, after, "closed")["protocol_error"] == 1
    assert after["robots"]["t11b"]["frames_rejected"] == 1


async def test_a_flood_over_the_message_rate_closes_the_connection_with_1008(
    limited_service: Service,
) -> None:
    before = await asyncio.to_thread(scrape, limited_service)
    connection, session = await open_raw_session(limited_service.ws_url, "t11c")
    try:
        # One more than the bucket that a session starts with; the hello is not counted.
        for frame_number in range(1, round(INGRESS_MAX_MESSAGES_PER_S) + 2):
            connection.send_nowait(raw_frame(session, frame_number))
        close_code = await peer_close_code(connection)
    finally:
        connection.abort()
    after = await asyncio.to_thread(
        wait_for_metrics,
        limited_service,
        lambda snapshot: not snapshot["robots"]["t11c"]["connected"],
        GATEWAY_TIMEOUT_S,
    )

    assert close_code == 1008
    assert gains(before, after, "frames_rejected")["rate_limited"] == 1
    assert gains(before, after, "closed")["ingress_limit"] == 1
    assert after["robots"]["t11c"]["frames_accepted"] == round(INGRESS_MAX_MESSAGES_PER_S)
