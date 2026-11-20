"""The service with real processes when a robot or a worker dies, freezes or hangs, a robot id
is used twice, and the gateway is stopped or killed."""

import contextlib
import os
import signal
import sys
import time
from pathlib import Path

import pytest

from bench.live_check import conservation_violations
from common.commands import MODEL_KEYS
from tests.integration.harness import (
    EXIT_TIMEOUT_S,
    NS_PER_S,
    SERVED_SHARE,
    at_or_after,
    first_receive_ns,
    fleet_log,
    fleet_timeout_s,
    gains,
    killed_on_exit,
    plan,
    rate_per_s,
    relative_spread,
    robot_commands,
    run_fleets,
    running_service,
    scrape,
    silent_seconds,
    start_fleet,
    time_zero_ns,
    wait_for_exit,
    wait_for_metrics,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# Three robots share the saturated depth model; two share it after one is killed: 1.5 times
# the rate each, of which this much is required.
MIN_RATE_GAIN = 1.3
RATE_TOLERANCE = 0.10
# A window starts this long after the event that opens it, so it holds no transition.
SETTLE_NS = NS_PER_S // 2
# Waiting for a planned instant of a plan whose fleet has not started yet.
INSTANT_TIMEOUT_S = 30.0
# Only keeps a broken service from hanging the suite: a fake worker restarts in about a second.
RESTART_TIMEOUT_S = 30.0


def test_a_killed_robot_frees_its_share_and_its_state(tmp_path: Path) -> None:
    duration_s = 5.0

    with (
        running_service({}, tmp_path / "gateway") as service,
        contextlib.ExitStack() as stack,
    ):
        t0_ns = time_zero_ns()
        kill_at_ns = t0_ns + 2 * NS_PER_S
        plans = {robot_id: plan(t0_ns, duration_s, [(robot_id, 0.0, None)]) for robot_id in "abc"}
        before = scrape(service)
        fleets = {
            robot_id: stack.enter_context(
                killed_on_exit(start_fleet(service, robot_plan, tmp_path / robot_id))
            )
            for robot_id, robot_plan in plans.items()
        }
        wait_for_metrics(service, at_or_after(kill_at_ns), INSTANT_TIMEOUT_S)
        # ASSUMPTION: A11. "Kill one" is SIGKILL of that robot's process, with no clean close.
        fleets["c"].kill()
        killed_ns = time.monotonic_ns()
        gone = wait_for_metrics(
            service, lambda snapshot: not snapshot["robots"]["c"]["connected"], EXIT_TIMEOUT_S
        )
        statuses = {
            robot_id: fleets[robot_id].wait(fleet_timeout_s(plans[robot_id]))
            for robot_id in ("a", "b", "c")
        }
        after = scrape(service)

    survivors = [robot_commands(tmp_path / robot_id, robot_id) for robot_id in ("a", "b")]
    end_ns = t0_ns + round(duration_s * NS_PER_S)
    # The windows are planned from time zero: a survivor first served after its first window
    # opened would show too low a rate before the kill, and so too high a gain.
    first_served_ns = [first_receive_ns(commands, "depth") - t0_ns for commands in survivors]
    rate_before = sum(
        rate_per_s(commands, "depth", t0_ns + SETTLE_NS, kill_at_ns) for commands in survivors
    )
    rate_after = sum(
        rate_per_s(commands, "depth", killed_ns + SETTLE_NS, end_ns) for commands in survivors
    )
    depth = gone["robots"]["c"]["models"]["depth"]
    assert max(first_served_ns) <= SETTLE_NS, first_served_ns
    assert statuses == {"a": 0, "b": 0, "c": -signal.SIGKILL}
    assert (
        after["connections"]["closed"]["abnormal"] - before["connections"]["closed"]["abnormal"]
        == 1
    )
    # /metrics shows a disconnected record's frames_pending as 0 whatever the mailbox holds, so
    # the release is checked where it shows. Depth is saturated, so c always holds a depth
    # frame, pending or at the worker: a pending one is counted as dropped at the disconnect,
    # and one at the worker stays in flight until its result is dropped. The identities then
    # hold, and only the survivors can be eligible.
    assert (
        depth["frames_dropped_disconnect"]
        + depth["in_flight"]
        + depth["results_dropped_disconnect"]
        >= 1
    )
    assert conservation_violations(gone) == []
    assert gone["models"]["depth"]["eligible"] <= 2
    assert gone["connections"]["current"] == 2
    assert rate_after >= MIN_RATE_GAIN * rate_before > 0
    assert [after["workers"][model]["restarts"] for model in MODEL_KEYS] == [
        before["workers"][model]["restarts"] for model in MODEL_KEYS
    ]


def test_a_frozen_robot_is_closed_and_reconnects(tmp_path: Path) -> None:
    duration_s = 7.0

    with running_service({"idle_timeout_ms": 1000}, tmp_path / "gateway") as service:
        t0_ns = time_zero_ns()
        stop_at_ns = t0_ns + 3 * NS_PER_S // 2
        continue_at_ns = stop_at_ns + 2 * NS_PER_S
        frozen_plan = plan(t0_ns, duration_s, [("frozen", 0.0, None)])
        steady_plan = plan(t0_ns, duration_s, [("steady", 0.0, None)])
        with (
            killed_on_exit(start_fleet(service, frozen_plan, tmp_path / "frozen")) as frozen,
            killed_on_exit(start_fleet(service, steady_plan, tmp_path / "steady")) as steady,
        ):
            before = wait_for_metrics(service, at_or_after(stop_at_ns), INSTANT_TIMEOUT_S)
            frozen.send_signal(signal.SIGSTOP)
            wait_for_metrics(service, at_or_after(continue_at_ns), INSTANT_TIMEOUT_S)
            frozen.send_signal(signal.SIGCONT)
            statuses = [
                frozen.wait(fleet_timeout_s(frozen_plan)),
                steady.wait(fleet_timeout_s(steady_plan)),
            ]
            after = scrape(service)

    frozen_commands = robot_commands(tmp_path / "frozen", "frozen")
    steady_commands = robot_commands(tmp_path / "steady", "steady")
    end_ns = t0_ns + round(duration_s * NS_PER_S)
    # Both robots must have been served before the first window opened: the steady one for its
    # rate, the frozen one to be connected when it is stopped.
    first_served_ns = [
        first_receive_ns(commands, "cls") - t0_ns for commands in (frozen_commands, steady_commands)
    ]
    rates_per_s = [
        rate_per_s(steady_commands, "cls", t0_ns + SETTLE_NS, stop_at_ns),
        rate_per_s(steady_commands, "cls", continue_at_ns + SETTLE_NS, end_ns),
    ]
    frozen_after = after["robots"]["frozen"]
    closed_before = before["connections"]["closed"]["idle_timeout"]
    assert max(first_served_ns) <= SETTLE_NS, first_served_ns
    assert statuses == [0, 0]
    assert after["connections"]["closed"]["idle_timeout"] - closed_before == 1
    assert frozen_after["session"] > before["robots"]["frozen"]["session"]
    assert frozen_after["telemetry"]["reconnects"] == 1
    assert min(rates_per_s) > 0
    assert relative_spread(rates_per_s) <= RATE_TOLERANCE


def test_sigterm_while_streaming_stops_the_gateway_and_its_workers(tmp_path: Path) -> None:
    with running_service({}, tmp_path / "gateway") as service:
        t0_ns = time_zero_ns()
        fleet_plan = plan(t0_ns, 6.0, [("t12a", 0.0, None), ("t12b", 0.0, None)])
        with killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet:
            streaming = wait_for_metrics(
                service, at_or_after(t0_ns + 2 * NS_PER_S), INSTANT_TIMEOUT_S
            )
            worker_pids = [streaming["workers"][model]["pid"] for model in MODEL_KEYS]
            service.process.send_signal(signal.SIGTERM)
            # The compose files give the gateway 10 s to stop.
            gateway_status = service.process.wait(10)
            workers_exited = [wait_for_exit(pid, EXIT_TIMEOUT_S) for pid in worker_pids]
            fleet_status = fleet.wait(fleet_timeout_s(fleet_plan))

    assert gateway_status == 0
    assert workers_exited == [True, True]
    assert fleet_status == 0
    # The robots log the close code of each session that ends: 1001, the gateway going away.
    assert "1001" in fleet_log(tmp_path / "fleet")


def test_a_second_process_with_the_same_robot_id_replaces_the_first_which_stops_for_good(
    tmp_path: Path,
) -> None:
    duration_s = 6.0

    with running_service({}, tmp_path / "gateway") as service:
        before = scrape(service)
        t0_ns = time_zero_ns()
        first_plan = plan(t0_ns, duration_s, [("t7", 0.0, None)])
        second_plan = plan(t0_ns, duration_s, [("t7", 2.0, None)])
        with (
            killed_on_exit(start_fleet(service, first_plan, tmp_path / "first")) as first,
            killed_on_exit(start_fleet(service, second_plan, tmp_path / "second")) as second,
        ):
            first_session = wait_for_metrics(
                service, at_or_after(t0_ns + 3 * NS_PER_S // 2), INSTANT_TIMEOUT_S
            )["robots"]["t7"]["session"]
            statuses = [
                first.wait(fleet_timeout_s(first_plan)),
                second.wait(fleet_timeout_s(second_plan)),
            ]
        after = scrape(service)

    commands = robot_commands(tmp_path / "second", "t7")
    session = after["robots"]["t7"]["session"]
    # The first robot is told its session was replaced, a close that a reconnect would only
    # repeat, so it stops and its process fails.
    assert statuses == [1, 0]
    assert "4001" in fleet_log(tmp_path / "first")
    assert gains(before, after, "closed")["replaced"] == 1
    assert gains(before, after, "closed")["internal_error"] == 0
    assert session > first_session
    assert set(commands["model"].tolist()) == set(range(len(MODEL_KEYS)))
    assert set(commands["session"].tolist()) == {session}
    assert conservation_violations(after) == []


def test_a_killed_depth_worker_is_restarted_and_cls_keeps_its_rate(tmp_path: Path) -> None:
    duration_s = 5.0
    robot_ids = ["t8a", "t8b"]

    with running_service({}, tmp_path / "gateway") as service:
        before = scrape(service)
        restarts = before["workers"]["depth"]["restarts"]
        t0_ns = time_zero_ns()
        fleet_plan = plan(t0_ns, duration_s, [(robot_id, 0.0, None) for robot_id in robot_ids])
        with killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet:
            streaming = wait_for_metrics(
                service, at_or_after(t0_ns + 3 * NS_PER_S // 2), INSTANT_TIMEOUT_S
            )
            pid = streaming["workers"]["depth"]["pid"]
            os.kill(pid, signal.SIGKILL)
            killed_ns = time.monotonic_ns()
            worker_exited = wait_for_exit(pid, EXIT_TIMEOUT_S)
            restarted = wait_for_metrics(
                service,
                lambda snapshot: (
                    snapshot["workers"]["depth"]["restarts"] == restarts + 1
                    and snapshot["workers"]["depth"]["state"] == "ready"
                ),
                RESTART_TIMEOUT_S,
            )
            fleet_status = fleet.wait(fleet_timeout_s(fleet_plan))
        after = scrape(service)

    end_ns = t0_ns + round(duration_s * NS_PER_S)
    rates_per_s = {}
    for robot_id in robot_ids:
        commands = robot_commands(tmp_path / "fleet", robot_id)
        rates_per_s[robot_id] = [
            rate_per_s(commands, "cls", t0_ns + SETTLE_NS, killed_ns),
            rate_per_s(commands, "cls", killed_ns + SETTLE_NS, end_ns),
        ]
    assert worker_exited
    assert fleet_status == 0
    assert restarted["workers"]["depth"]["pid"] != pid
    assert after["workers"]["depth"]["restarts"] - restarts == 1
    assert after["workers"]["cls"]["restarts"] == before["workers"]["cls"]["restarts"]
    assert all(min(rates) > 0 for rates in rates_per_s.values()), rates_per_s
    assert all(relative_spread(rates) <= RATE_TOLERANCE for rates in rates_per_s.values()), (
        rates_per_s
    )
    assert conservation_violations(after) == []
    assert gains(before, after, "closed")["internal_error"] == 0


def test_a_depth_batch_over_the_inference_timeout_restarts_the_worker_and_cls_keeps_serving(
    tmp_path: Path,
) -> None:
    duration_s = 6
    robot_ids = ["t9a", "t9b"]
    # Every depth batch takes 300 ms, past the 200 ms timeout at the await of its result, so
    # depth never commands.
    overrides = {"inference_timeout_ms": 200, "models": {"depth": {"fake_frame_ms": 300.0}}}

    with running_service(overrides, tmp_path / "gateway") as service:
        t0_ns = time_zero_ns()
        fleet_plan = plan(t0_ns, duration_s, [(robot_id, 0.0, None) for robot_id in robot_ids])
        run_fleets(service, [(fleet_plan, tmp_path / "fleet", ())])
        after = scrape(service)

    workers = after["workers"]
    robots = after["robots"]
    depth_sent = [robots[robot_id]["models"]["depth"]["commands_sent"] for robot_id in robot_ids]
    silent = {
        robot_id: silent_seconds(
            robot_commands(tmp_path / "fleet", robot_id), "cls", t0_ns, 1, duration_s
        )
        for robot_id in robot_ids
    }
    served_share = {
        robot_id: robots[robot_id]["telemetry"]["models"]["cls"]["commands_received"]
        / robots[robot_id]["telemetry"]["frames_sent"]
        for robot_id in robot_ids
    }
    assert workers["depth"]["timeouts"] >= 1
    assert workers["depth"]["restarts"] >= 1
    assert [workers["cls"]["timeouts"], workers["cls"]["restarts"]] == [0, 0]
    assert depth_sent == [0] * len(robot_ids)
    assert silent == {robot_id: [] for robot_id in robot_ids}
    assert min(served_share.values()) >= SERVED_SHARE, served_share
    assert conservation_violations(after) == []


def test_the_workers_exit_when_the_gateway_is_killed(tmp_path: Path) -> None:
    with running_service({}, tmp_path / "gateway", expect_exit=None) as service:
        t0_ns = time_zero_ns()
        fleet_plan = plan(t0_ns, 5.0, [("t14", 0.0, None)])
        with killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet:
            # A robot streams, so the depth worker is mid-batch when its link closes.
            streaming = wait_for_metrics(service, at_or_after(t0_ns + NS_PER_S), INSTANT_TIMEOUT_S)
            worker_pids = [streaming["workers"][model]["pid"] for model in MODEL_KEYS]
            service.process.kill()
            gateway_status = service.process.wait(EXIT_TIMEOUT_S)
            workers_exited = [wait_for_exit(pid, EXIT_TIMEOUT_S) for pid in worker_pids]
            # The robot retries until its plan ends.
            fleet_status = fleet.wait(fleet_timeout_s(fleet_plan))

    assert gateway_status == -signal.SIGKILL
    assert workers_exited == [True] * len(MODEL_KEYS)
    assert fleet_status == 0
