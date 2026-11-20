"""The service end to end with real processes, sockets and time: commands, fairness, conservation.

Tests that share a service assert on robot ids they created and on differences of counters.
"""

import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from bench.live_check import agreement, boundary_violations, conservation_violations
from common.commands import MODEL_KEYS
from common.protocol import STAGE_NAMES
from tests.integration.harness import (
    NS_PER_S,
    Service,
    Snapshot,
    fleet_timeout_s,
    killed_on_exit,
    plan,
    rate_per_s,
    relative_spread,
    robot_commands,
    run_fleets,
    running_service,
    scrape,
    start_fleet,
    time_zero_ns,
    wait_for_metrics,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# Counts and rates that a fair share makes equal may differ by this fraction of the largest.
FAIRNESS_TOLERANCE = 0.10
UPLINK_DELAY_MS = 20
DOWNLINK_DELAY_MS = 30


@pytest.fixture(scope="module")
def service(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Service]:
    with running_service({}, tmp_path_factory.mktemp("gateway")) as running:
        yield running


# DESIGN.md 6.1: the scheduler's fairness holds with the skip rule off, which then never passes
# over a robot.
@pytest.fixture(scope="module")
def service_without_skip_rule(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Service]:
    with running_service({"skip_rule": False}, tmp_path_factory.mktemp("gateway")) as running:
        yield running


def test_one_robot_end_to_end(service: Service, tmp_path: Path) -> None:
    t0_ns = time_zero_ns()
    plain_plan = plan(t0_ns, 3.0, [("t1plain", 0.0, None)])
    delayed_plan = plan(t0_ns, 3.0, [("t1delayed", 0.0, None)])
    delays = (
        *("--uplink-delay-ms", str(UPLINK_DELAY_MS)),
        *("--downlink-delay-ms", str(DOWNLINK_DELAY_MS)),
    )

    run_fleets(
        service,
        [(plain_plan, tmp_path / "plain", ()), (delayed_plan, tmp_path / "delayed", delays)],
    )

    snapshot = scrape(service)
    commands = robot_commands(tmp_path / "plain", "t1plain")
    delayed = robot_commands(tmp_path / "delayed", "t1delayed")
    stages_ns = 1000 * commands["stage_us"].astype(np.int64).sum(axis=1)
    fire_lag_ns = commands["send_ns"].astype(np.int64) - commands["capture_ns"]
    deliver_ns = commands["receive_ns"].astype(np.int64) - commands["result_ns"]
    freshness_ns = commands["receive_ns"].astype(np.int64) - commands["capture_ns"]
    residual_ns = freshness_ns - (fire_lag_ns + stages_ns + deliver_ns)
    telemetry = snapshot["robots"]["t1plain"]["telemetry"]["models"]
    assert all(np.count_nonzero(commands["model"] == index) > 0 for index in range(len(MODEL_KEYS)))
    assert residual_ns.min() >= 0
    assert residual_ns.max() < 1000
    assert [
        telemetry[model]["commands_received"]
        == telemetry[model]["commands_applied"]
        + telemetry[model]["commands_stale"]
        + telemetry[model]["commands_out_of_order"]
        for model in MODEL_KEYS
    ] == [True] * len(MODEL_KEYS)
    assert conservation_violations(snapshot) == []
    assert agreement(snapshot, "t1plain", commands) == []
    assert boundary_violations(snapshot, "t1plain") == []
    transport_up_us = delayed["stage_us"][:, STAGE_NAMES.index("transport_up")]
    delayed_deliver_ns = delayed["receive_ns"].astype(np.int64) - delayed["result_ns"]
    assert delayed.size > 0
    assert transport_up_us.min() >= 1000 * UPLINK_DELAY_MS
    assert delayed_deliver_ns.min() >= DOWNLINK_DELAY_MS * 1_000_000


def test_five_robots_share_the_saturated_model_evenly(
    service_without_skip_rule: Service, tmp_path: Path
) -> None:
    robot_ids = [f"t2r{index}" for index in range(1, 6)]
    fleet_plan = plan(time_zero_ns(), 4.0, [(robot_id, 0.0, None) for robot_id in robot_ids])

    run_fleets(service_without_skip_rule, [(fleet_plan, tmp_path / "fleet", ())])

    robots = scrape(service_without_skip_rule)["robots"]
    depth_sent = [robots[robot_id]["models"]["depth"]["commands_sent"] for robot_id in robot_ids]
    cls_sent = [robots[robot_id]["models"]["cls"]["commands_sent"] for robot_id in robot_ids]
    frames_sent = [robots[robot_id]["telemetry"]["frames_sent"] for robot_id in robot_ids]
    assert relative_spread(depth_sent) <= FAIRNESS_TOLERANCE
    assert relative_spread(cls_sent + frames_sent) <= FAIRNESS_TOLERANCE


def test_a_late_joiner_reaches_the_incumbents_share(
    service_without_skip_rule: Service, tmp_path: Path
) -> None:
    t0_ns = time_zero_ns()
    fleet_plan = plan(t0_ns, 5.0, [("t3a", 0.0, None), ("t3b", 2.0, None)])

    run_fleets(service_without_skip_rule, [(fleet_plan, tmp_path / "fleet", ())])

    last_window = (t0_ns + 5 * NS_PER_S // 2, t0_ns + 5 * NS_PER_S)
    rates = [
        rate_per_s(robot_commands(tmp_path / "fleet", robot_id), "depth", *last_window)
        for robot_id in ("t3a", "t3b")
    ]
    assert min(rates) > 0
    assert relative_spread(rates) <= FAIRNESS_TOLERANCE


def test_every_scrape_keeps_the_identities_while_robots_join_and_leave(
    service: Service, tmp_path: Path
) -> None:
    fleet_plan = plan(
        time_zero_ns(),
        4.0,
        [
            ("t6a", 0.0, None),
            ("t6b", 0.5, 2.5),
            ("t6c", 1.0, 3.5),
            ("t6d", 2.0, None),
            ("t6e", 2.5, 3.0),
        ],
    )
    violations: list[str] = []

    with killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet:

        def check(snapshot: Snapshot) -> bool:
            violations.extend(conservation_violations(snapshot))
            return fleet.poll() is not None

        wait_for_metrics(service, check, fleet_timeout_s(fleet_plan))
        exit_status = fleet.wait()

    violations.extend(conservation_violations(scrape(service)))
    assert exit_status == 0
    assert violations == []
