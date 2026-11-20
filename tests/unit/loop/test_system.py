"""Tests of the whole system in one process: the gateway, fake workers and real robots.

Every timing is virtual and every service time is fixed, so the scheduling outcomes of the
demos of DESIGN.md section 9 can be asserted exactly.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.capacity_model import command_gap_bound_ns
from common.commands import MODEL_KEYS
from common.protocol import STAGE_NAMES
from robot.samples import load_samples
from robot.settings import LinkSettings
from tests.support.system import InProcessSystem, identity_violations

MS = 1_000_000
SECOND = 1_000_000_000
# The robots' default rate of 30 frames per second.
PERIOD_NS = round(SECOND / 30)
# Depth takes longer per frame than a frame period, so it is saturated from the first robot;
# its batch cap is then 2, the largest size whose batch time fits the freshness budget
# (150 ms - one period - 10 ms of allowance). The skip rule is off, so every robot with a
# frame pending is served in turn.
DEPTH_MS = 40
DEPTH_CAP = 2
SATURATED_MS = {"cls": 2, "depth": DEPTH_MS}
SKIP_RULE_OFF = {"skip_rule": False}
# Half a millisecond past a whole one: every event of the system falls on a whole millisecond
# (see the first test), so a snapshot taken here never shares an instant with one.
BETWEEN_EVENTS_NS = MS // 2


async def advance_to(system: InProcessSystem, instant_ns: int) -> None:
    await system.advance((instant_ns - system.clock.now_ns()) / 1e9)


def depth_sent(snapshot: Mapping[str, Any], robot_ids: Sequence[str]) -> list[int]:
    return [
        snapshot["robots"][robot_id]["models"]["depth"]["commands_sent"] for robot_id in robot_ids
    ]


async def test_one_robot_gets_commands_whose_stages_add_up_and_sit_where_they_belong(
    tmp_path: Path,
) -> None:
    # Every delay and service time is whole milliseconds, and sleep_until sleeps whole
    # milliseconds, so every event of the virtual loop falls on a whole millisecond and each
    # stage below is exact. A stage stamped at the wrong place moves time between neighbouring
    # stages without changing their sum; only these equalities see that.
    uplink_ms, downlink_ms = 10, 5
    # The worker's send, the link's stamp of the result and the dispatcher's wake-up are three
    # instants apart, as on a real host, so a result stamped at either neighbour of the link's
    # stamp moves time into or out of dispatch_rt.
    result_leg_ms, wake_up_ms = 1, 3
    frame_ms = {"cls": 2, "depth": 20}
    async with InProcessSystem(
        frame_ms, result_leg_ms=result_leg_ms, dispatcher_wake_up_ms=wake_up_ms
    ) as system:
        start_ns = system.clock.now_ns()
        system.add_robot(
            "r01",
            start_ns,
            start_ns + 3 * SECOND,
            uplink=LinkSettings(delay_ms=uplink_ms),
            downlink=LinkSettings(delay_ms=downlink_ms),
        )
        await system.advance(4)
        path = tmp_path / "r01.npz"
        system.samples("r01").save(path, {})
    commands, _, _ = load_samples(path)
    stage_us = commands["stage_us"].astype(np.int64)
    fire_lag_ns = commands["send_ns"] - commands["capture_ns"]
    deliver_ns = commands["receive_ns"] - commands["result_ns"]
    freshness_ns = commands["receive_ns"] - commands["capture_ns"]
    # The stages travel in whole microseconds, rounded down from cumulative offsets.
    unaccounted_ns = freshness_ns - (fire_lag_ns + 1000 * stage_us.sum(axis=1) + deliver_ns)
    service_us = 1000 * np.array([frame_ms[model] for model in MODEL_KEYS])[commands["model"]]
    assert set(commands["model"].tolist()) == {MODEL_KEYS.index("cls"), MODEL_KEYS.index("depth")}
    assert ((unaccounted_ns >= 0) & (unaccounted_ns < 1000)).all()
    assert (stage_us[:, STAGE_NAMES.index("transport_up")] == 1000 * uplink_ms).all()
    # The fake worker has a batch at the instant it is written, so the round trip is the
    # result's leg alone; the dispatcher's wake-up belongs to deliver.
    assert (stage_us[:, STAGE_NAMES.index("dispatch_rt")] == 1000 * result_leg_ms).all()
    assert (deliver_ns == (wake_up_ms + downlink_ms) * MS).all()
    assert (stage_us[:, STAGE_NAMES.index("inference")] == service_us).all()
    assert (stage_us[:, STAGE_NAMES.index("batch_peers")] == 0).all()


@pytest.mark.parametrize(
    "offset_ns",
    [
        # Started 6 ms apart, so their frames arrive at different instants.
        pytest.param(6 * MS, id="staggered"),
        # Started together, so every robot's frame arrives at one instant.
        pytest.param(0, id="aligned"),
    ],
)
async def test_five_robots_on_a_saturated_depth_get_depth_commands_in_turn(
    offset_ns: int,
) -> None:
    robot_ids = [f"r{index:02}" for index in range(1, 6)]
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        for index, robot_id in enumerate(robot_ids):
            system.add_robot(robot_id, index * offset_ns, 12 * SECOND)
        spreads = []
        for _ in range(10):
            await system.advance(1)
            sent = depth_sent(await system.metrics(), robot_ids)
            spreads.append(max(sent) - min(sent))
        snapshot = await system.metrics()
    overwritten = [
        snapshot["robots"][robot_id]["models"]["depth"]["frames_overwritten"]
        for robot_id in robot_ids
    ]
    assert all(spread <= 1 for spread in spreads), spreads
    # Saturated: every robot's slot was overwritten before depth could serve it.
    assert all(count > 0 for count in overwritten), overwritten


async def test_a_late_joiner_gets_the_incumbents_depth_rate_within_one_rotation() -> None:
    join_ns = 2 * SECOND
    # Within one rotation of the join: the joiner's first frame, a period later and rounded up
    # to the timer's millisecond; the incumbent's batch then at the worker; and the joiner's
    # own batch at the cap, which takes the incumbent's pending frame too.
    served_by_ns = join_ns + PERIOD_NS + MS + DEPTH_MS * MS + DEPTH_CAP * DEPTH_MS * MS
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        system.add_robot("a", 0, 10 * SECOND)
        system.add_robot("b", join_ns, 10 * SECOND)
        await advance_to(system, served_by_ns + BETWEEN_EVENTS_NS)
        first = depth_sent(await system.metrics(), ["a", "b"])
        gains = []
        for _ in range(5):
            await system.advance(1)
            now = depth_sent(await system.metrics(), ["a", "b"])
            gains.append((now[0] - first[0], now[1] - first[1]))
    assert first[1] >= 1
    # Each batch from the joiner's first on holds both robots' frames.
    assert all(gain_a == gain_b for gain_a, gain_b in gains), gains
    assert gains[0][0] > 0


async def depth_gains(
    system: InProcessSystem, robot_ids: Sequence[str], start_ns: int, window_ns: int
) -> list[int]:
    """Return each robot's depth commands sent over the window from start_ns."""
    await advance_to(system, start_ns)
    before = depth_sent(await system.metrics(), robot_ids)
    await advance_to(system, start_ns + window_ns)
    after = depth_sent(await system.metrics(), robot_ids)
    return [late - early for early, late in zip(before, after, strict=True)]


async def test_an_aborted_robot_is_released_and_its_depth_share_goes_to_the_survivors() -> None:
    robot_count = 3
    survivors = ["r1", "r2"]
    # Thirty batches at the cap: ten rotations of three robots before the abort, thirty of two
    # after it, so each window holds whole rotations.
    window_ns = 30 * DEPTH_CAP * DEPTH_MS * MS
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        for index, robot_id in enumerate([*survivors, "r3"]):
            system.add_robot(robot_id, index * 11 * MS, 10 * SECOND)
        before = await depth_gains(system, survivors, SECOND + BETWEEN_EVENTS_NS, window_ns)
        await advance_to(system, 3500 * MS + BETWEEN_EVENTS_NS)
        held = await system.metrics()
        system.abort("r3")
        # Long enough for the gateway to see the loss, and short of the next event.
        await advance_to(system, 3500 * MS + BETWEEN_EVENTS_NS + MS // 4)
        gone = await system.metrics()
        after = await depth_gains(system, survivors, 4500 * MS + BETWEEN_EVENTS_NS, window_ns)
        snapshot = await system.metrics()
    # The robot held a depth frame when it was killed. /metrics shows a disconnected record's
    # frames_pending as 0 whatever the mailbox holds, so the release is seen in the frames
    # counted as dropped and in the robots eligible for depth.
    pending = [held["robots"]["r3"]["models"][model]["frames_pending"] for model in MODEL_KEYS]
    dropped = [
        gone["robots"]["r3"]["models"][model]["frames_dropped_disconnect"]
        - held["robots"]["r3"]["models"][model]["frames_dropped_disconnect"]
        for model in MODEL_KEYS
    ]
    eligible = [
        held["models"][model]["eligible"] - gone["models"][model]["eligible"]
        for model in MODEL_KEYS
    ]
    assert pending[MODEL_KEYS.index("depth")] == 1
    assert dropped == eligible == pending
    assert snapshot["connections"]["closed"]["abnormal"] == 1
    assert snapshot["connections"]["current"] == len(survivors)
    assert snapshot["robots"]["r3"]["connected"] is False
    assert [late * (robot_count - 1) for late in after] == [early * robot_count for early in before]
    assert min(before) > 0


def service_gaps(snapshot: Mapping[str, Any], robot_ids: Sequence[str]) -> dict[str, float]:
    return {
        f"{robot_id}.{model}": counters["max_service_gap_ms"]
        for robot_id in robot_ids
        for model, counters in snapshot["robots"][robot_id]["models"].items()
    }


async def test_a_robot_back_after_an_abort_gets_a_later_session_and_disturbs_no_one() -> None:
    others = ["r1", "r2"]
    phase_ns = 22 * MS
    # The robot's process is killed and started again. The new one keeps the phase of the
    # first schedule, so its frames meet the others' at the same instants as before.
    return_ns = phase_ns + 105 * PERIOD_NS
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        system.add_robot("r1", 0, 8 * SECOND)
        system.add_robot("r2", 11 * MS, 8 * SECOND)
        system.add_robot("r3", phase_ns, 8 * SECOND)
        await advance_to(system, 2500 * MS + BETWEEN_EVENTS_NS)
        earlier = await system.metrics()
        await advance_to(system, 3 * SECOND)
        system.abort("r3")
        system.add_robot("r3", return_ns, 8 * SECOND)
        await advance_to(system, 7 * SECOND + BETWEEN_EVENTS_NS)
        later = await system.metrics()
    assert later["robots"]["r3"]["connected"] is True
    assert later["robots"]["r3"]["session"] > earlier["robots"]["r3"]["session"]
    assert service_gaps(later, others) == service_gaps(earlier, others)


async def test_every_identity_holds_at_every_second_while_robots_join_and_leave() -> None:
    # (robot, start s, stop s): joins and leaves at different instants, overlapping.
    plan = [("r1", 0.0, 9.0), ("r2", 1.0, 5.0), ("r3", 2.5, 10.0), ("r4", 4.2, 6.5)]
    async with InProcessSystem(SATURATED_MS) as system:
        for robot_id, start_s, stop_s in plan:
            system.add_robot(robot_id, round(start_s * SECOND), round(stop_s * SECOND))
        violations = []
        seen: set[str] = set()
        for _ in range(11):
            await system.advance(1)
            snapshot = await system.metrics()
            violations.extend(identity_violations(snapshot))
            seen.update(snapshot["robots"])
        closed = snapshot["connections"]["closed"]
    assert violations == []
    assert seen == {robot_id for robot_id, _, _ in plan}
    assert closed["client_close"] == len(plan)


def model_counters(snapshot: Mapping[str, Any], robot_id: str, model: str) -> Any:
    return snapshot["robots"][robot_id]["models"][model]


async def test_robots_that_all_leave_at_one_instant_leave_the_gateway_idle() -> None:
    robot_ids = [f"r{index}" for index in range(1, 6)]
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        for index, robot_id in enumerate(robot_ids):
            system.add_robot(robot_id, index * 6 * MS, 60 * SECOND)
        await advance_to(system, 2 * SECOND + BETWEEN_EVENTS_NS)
        for robot_id in robot_ids:
            system.leave(robot_id)
        await advance_to(system, 3 * SECOND + BETWEEN_EVENTS_NS)
        idle = await system.metrics()
        await advance_to(system, 4 * SECOND + BETWEEN_EVENTS_NS)
        later = await system.metrics()
    held = [
        model_counters(idle, robot_id, model)[name]
        for robot_id in robot_ids
        for model in MODEL_KEYS
        for name in ("in_flight", "commands_queued")
    ]
    workers = [
        (idle["workers"][model]["state"], idle["workers"][model]["restarts"])
        for model in MODEL_KEYS
    ]
    assert idle["connections"]["current"] == 0
    assert idle["connections"]["closed"]["client_close"] == len(robot_ids)
    assert idle["connections"]["closed"]["internal_error"] == 0
    assert [idle["models"][model]["eligible"] for model in MODEL_KEYS] == [0] * len(MODEL_KEYS)
    assert held == [0] * len(held)
    assert identity_violations(idle) == []
    assert workers == [("ready", 0)] * len(MODEL_KEYS)
    # With no robot connected, nothing samples the loop's lag and nothing is dispatched.
    assert later["loop"]["lag_ms"]["count"] == idle["loop"]["lag_ms"]["count"]
    assert [later["models"][model]["batches"] for model in MODEL_KEYS] == [
        idle["models"][model]["batches"] for model in MODEL_KEYS
    ]


async def test_fifty_robots_at_one_instant_are_all_accepted_and_served_within_the_bound() -> None:
    robot_ids = [f"r{index:02}" for index in range(1, 51)]
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        for robot_id in robot_ids:
            system.add_robot(robot_id, 0, 10 * SECOND)
        await advance_to(system, 3 * SECOND + BETWEEN_EVENTS_NS)
        snapshot = await system.metrics()
    # Robots started at one instant connect at one instant, and a cycle is at most the cap
    # times the fake service time: the fake links answer exactly one service time per frame
    # after the dispatch and resume the dispatcher at once.
    bounds_ms = {}
    for model in MODEL_KEYS:
        cap = snapshot["workers"][model]["batch_cap"]
        cycle_ns = cap * SATURATED_MS[model] * MS
        bounds_ms[model] = (
            command_gap_bound_ns(len(robot_ids), cap, PERIOD_NS, cycle_ns, None) / 1e6
        )
    connections = snapshot["connections"]
    unserved = [
        f"{robot_id}.{model}"
        for robot_id in robot_ids
        for model in MODEL_KEYS
        if model_counters(snapshot, robot_id, model)["commands_sent"] < 1
    ]
    over_bound = {
        f"{robot_id}.{model}": model_counters(snapshot, robot_id, model)["max_service_gap_ms"]
        for robot_id in robot_ids
        for model in MODEL_KEYS
        if model_counters(snapshot, robot_id, model)["max_service_gap_ms"] > bounds_ms[model]
    }
    assert connections["accepted"] == len(robot_ids)
    assert connections["current"] == len(robot_ids)
    assert set(connections["rejected"].values()) == {0}
    assert connections["closed"]["internal_error"] == 0
    assert unserved == []
    assert over_bound == {}
    assert identity_violations(snapshot) == []


async def test_a_robot_that_reconnects_at_once_with_restarted_frame_numbers_is_served(
    tmp_path: Path,
) -> None:
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        system.add_robot("r1", 0, 6 * SECOND)
        system.add_robot("r2", 11 * MS, 6 * SECOND)
        system.add_robot("r3", 22 * MS, 6 * SECOND)
        await advance_to(system, 2 * SECOND + BETWEEN_EVENTS_NS)
        before = await system.metrics()
        system.abort("r3")
        # A new process with the same id, whose capture schedule and frame numbers start
        # again from 0, connects at the instant the old one dies.
        system.add_robot("r3", system.clock.now_ns(), 6 * SECOND)
        await advance_to(system, 4 * SECOND + BETWEEN_EVENTS_NS)
        after = await system.metrics()
        path = tmp_path / "r3.npz"
        system.samples("r3").save(path, {})
    commands, _, _ = load_samples(path)
    session = after["robots"]["r3"]["session"]
    closed_before = before["connections"]["closed"]
    closed_after = after["connections"]["closed"]
    # The gateway may see the new hello before the old connection's end, as on a real host,
    # so the old session ends either abnormal or replaced, and exactly one of them.
    ended = sum(closed_after[reason] - closed_before[reason] for reason in ("abnormal", "replaced"))
    assert after["robots"]["r3"]["connected"] is True
    assert session > before["robots"]["r3"]["session"]
    assert after["robots"]["r3"]["frames_rejected"] == before["robots"]["r3"]["frames_rejected"]
    assert set(commands["model"].tolist()) == set(range(len(MODEL_KEYS)))
    assert set(commands["session"].tolist()) == {session}
    assert ended == 1
    assert closed_after["internal_error"] == 0
    assert after["connections"]["current"] == 3
    assert identity_violations(after) == []


async def test_a_robot_sending_five_frames_per_second_is_served_every_frame() -> None:
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        system.add_robot("r1", 0, 10 * SECOND)
        system.add_robot("r2", 11 * MS, 10 * SECOND)
        # Depth serves the two fast robots at its cap of two 40 ms frames, so a frame of the
        # slow robot waits at most one batch, far within its 200 ms period.
        system.add_robot("slow", 22 * MS, 10 * SECOND, frame_rate_hz=5.0)
        await advance_to(system, 6 * SECOND + BETWEEN_EVENTS_NS)
        snapshot = await system.metrics()
    slow_lost = [
        model_counters(snapshot, "slow", model)[name]
        for model in MODEL_KEYS
        for name in ("frames_overwritten", "frames_skipped")
    ]
    slow_depth = model_counters(snapshot, "slow", "depth")
    fast_sent = [
        model_counters(snapshot, robot_id, "depth")["commands_sent"] for robot_id in ("r1", "r2")
    ]
    fast_overwritten = [
        model_counters(snapshot, robot_id, "depth")["frames_overwritten"]
        for robot_id in ("r1", "r2")
    ]
    assert slow_lost == [0] * len(slow_lost)
    assert slow_depth["commands_sent"] >= slow_depth["frames_received"] - 1
    assert max(fast_sent) - min(fast_sent) <= 1
    assert min(fast_overwritten) > 0
    assert identity_violations(snapshot) == []


async def test_a_depth_worker_killed_mid_batch_restarts_while_cls_serves_every_frame() -> None:
    robot_ids = ["r1", "r2"]
    async with InProcessSystem(SATURATED_MS, SKIP_RULE_OFF) as system:
        system.add_robot("r1", 0, 10 * SECOND)
        system.add_robot("r2", 11 * MS, 10 * SECOND)
        await advance_to(system, 2 * SECOND + BETWEEN_EVENTS_NS)
        held = await system.metrics()
        system.crash_worker("depth")
        await advance_to(system, 2 * SECOND + BETWEEN_EVENTS_NS + MS)
        crashed = await system.metrics()
        await advance_to(system, 3 * SECOND + BETWEEN_EVENTS_NS)
        back = await system.metrics()
        await advance_to(system, 4 * SECOND + BETWEEN_EVENTS_NS)
        later = await system.metrics()

    def total(snapshot: Mapping[str, Any], model: str, name: str) -> int:
        return sum(model_counters(snapshot, robot_id, model)[name] for robot_id in robot_ids)

    def gain(model: str, name: str, early: Mapping[str, Any]) -> list[int]:
        return [
            model_counters(later, robot_id, model)[name]
            - model_counters(early, robot_id, model)[name]
            for robot_id in robot_ids
        ]

    in_flight = total(held, "depth", "in_flight")
    cls_kept = [
        model_counters(snapshot, robot_id, "cls")[name]
        for robot_id in robot_ids
        for name in ("frames_failed", "frames_overwritten")
        for snapshot in (held, later)
    ]
    cls_sent = gain("cls", "commands_sent", held)
    cls_received = gain("cls", "frames_received", held)
    assert in_flight >= 1
    assert total(crashed, "depth", "frames_failed") - total(held, "depth", "frames_failed") == (
        in_flight
    )
    assert crashed["workers"]["depth"]["state"] == "restarting"
    assert crashed["workers"]["cls"]["state"] == "ready"
    assert later["workers"]["depth"]["state"] == "ready"
    assert later["workers"]["depth"]["restarts"] == 1
    assert later["workers"]["cls"]["restarts"] == 0
    assert min(gain("depth", "commands_sent", back)) > 0
    assert cls_kept[0::2] == cls_kept[1::2]
    assert all(
        sent >= received - 1 for sent, received in zip(cls_sent, cls_received, strict=True)
    ), (cls_sent, cls_received)
    assert later["connections"]["current"] == len(robot_ids)
    assert later["connections"]["closed"]["internal_error"] == 0
    assert identity_violations(later) == []
