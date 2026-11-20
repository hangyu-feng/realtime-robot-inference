"""Tests for the discrete-event simulation that drives the scheduling core on a simulated clock."""

import functools
import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.capacity_model import command_gap_bound_ns, longest_cycle_ns
from bench.scenarios import (
    RobotPlan,
    Scenario,
    Window,
    disconnect,
    late_joiner,
    massive,
    steady,
    sweep,
)
from bench.simulation import (
    FinalCounts,
    Overheads,
    PacingShare,
    PhaseMode,
    ServiceTable,
    SimulationResult,
    simulate,
)
from cloud.config import QueuePolicy, ServerConfig
from cloud.registry import ModelCounters
from cloud.scheduler import pacing_release_ns
from common.commands import MODEL_KEYS

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
SEED = 0
QUOTA_CORES = 2.0
CONFIG = ServerConfig()
SKIP_OFF = replace(CONFIG, skip_rule=False)
# The uplink and the delivery differ, so that a run shows which of the two a time includes.
OVERHEADS = Overheads(
    transport_up_ns=NS_PER_MS,
    dispatch_request_ns=NS_PER_MS // 2,
    dispatch_result_ns=NS_PER_MS // 2,
    deliver_ns=2 * NS_PER_MS,
)
# The measured cost of the frame I/O is not the configured 0.15 ms per frame, so that a run
# shows which of the two a value takes.
IO_CPU_MS_PER_FRAME = 0.4


def linear_table(wall_ms: float, cpu_ms: float) -> ServiceTable:
    """Return a table in which a batch takes wall_ms of wall and cpu_ms of CPU time per frame."""
    return ServiceTable(
        wall_ns={size: round(size * wall_ms * NS_PER_MS) for size in (1, 2, 4, 8)},
        cpu_ns={size: round(size * cpu_ms * NS_PER_MS) for size in (1, 2, 4, 8)},
    )


# Under the default configuration the classifier's cap is 8 (40 ms within its 456.7 ms budget)
# and depth's is 2 (80 ms within its 106.7 ms budget). With the overheads, the classifier serves
# at most 8 frames in 41 ms and depth 2 frames in 81 ms, so depth is saturated by one robot. A
# batch's CPU time is 1.5 times its wall time, so that a run shows which of the two it takes.
TABLES = {"cls": linear_table(5.0, 7.5), "depth": linear_table(40.0, 60.0)}
# A service time above the frame period for both models.
SLOW = {"cls": linear_table(50.0, 75.0), "depth": linear_table(50.0, 75.0)}
# A classifier whose CPU time, 15 ms per frame, leaves depth a pacing share from the second
# robot on; its wall time is longer.
HEAVY_CLS = {"cls": linear_table(20.0, 15.0), "depth": linear_table(40.0, 60.0)}
# A classifier whose CPU time, 30 ms per frame, leaves depth a pacing share with one robot.
PACED_ALONE = {"cls": linear_table(20.0, 30.0), "depth": linear_table(40.0, 60.0)}
# Every frame served alone, well within a frame period.
FAST = {"cls": linear_table(1.0, 1.5), "depth": linear_table(1.0, 1.5)}
# Depth's service time alone is above its 150 ms threshold.
STALE = {"cls": linear_table(5.0, 7.5), "depth": linear_table(200.0, 300.0)}

# One robot for 30 s, with windows that meet at 45 ms.
FAST_SCENARIO = Scenario(
    "fast",
    30.0,
    (RobotPlan("a", 0.0, None),),
    (Window("all", 0.0, 30.0), Window("before", 0.0, 0.045), Window("after", 0.045, 30.0)),
)
# One robot that stops at 100 ms of a 1 s run.
STOP_SCENARIO = Scenario("stop", 1.0, (RobotPlan("a", 0.0, 0.1),), (Window("all", 0.0, 1.0),))


@dataclass(frozen=True)
class Case:
    """The inputs of one simulated run, besides the seed."""

    config: ServerConfig
    scenario: Scenario
    tables: Mapping[str, ServiceTable] = field(default_factory=lambda: TABLES)
    quota_cores: float | None = QUOTA_CORES
    phase_mode: PhaseMode = PhaseMode.RANDOM
    overheads: Overheads = OVERHEADS
    io_cpu_ms_per_frame: float = IO_CPU_MS_PER_FRAME


DEMOS = {
    "steady": Case(CONFIG, steady()),
    "late_joiner": Case(CONFIG, late_joiner()),
    "sweep_n1": Case(CONFIG, sweep(1)),
    "sweep_n3": Case(CONFIG, sweep(3)),
    "sweep_n5": Case(CONFIG, sweep(5)),
    "sweep_n10": Case(CONFIG, sweep(10)),
    "disconnect": Case(CONFIG, disconnect()),
    "massive": Case(CONFIG, massive()),
}
CASES = {
    **DEMOS,
    "fifo": Case(
        replace(SKIP_OFF, queue_policy=QueuePolicy.FIFO), steady(), SLOW, quota_cores=None
    ),
    "latest": Case(SKIP_OFF, steady(), SLOW, quota_cores=None),
    "saturated": Case(SKIP_OFF, sweep(10)),
    "late_joiner_skip_off": Case(SKIP_OFF, late_joiner()),
    # Without pacing, depth's capacity does not change when a robot leaves.
    "disconnect_unpaced": Case(SKIP_OFF, disconnect(), quota_cores=None),
    "aligned": Case(CONFIG, sweep(5), phase_mode=PhaseMode.ALIGNED),
    "paced": Case(CONFIG, disconnect(), HEAVY_CLS),
    "paced_alone": Case(CONFIG, steady(), PACED_ALONE, phase_mode=PhaseMode.ALIGNED),
    # At 25 frames per second the frame period is exactly 40 ms, so 30 s hold 750 ticks.
    "fast": Case(
        replace(CONFIG, frame_rate_hz=25.0),
        FAST_SCENARIO,
        FAST,
        quota_cores=None,
        phase_mode=PhaseMode.ALIGNED,
    ),
    "stop": Case(
        replace(SKIP_OFF, frame_rate_hz=25.0),
        STOP_SCENARIO,
        SLOW,
        quota_cores=None,
        phase_mode=PhaseMode.ALIGNED,
    ),
    "stale": Case(CONFIG, steady(), STALE),
    "long_uplink": Case(
        CONFIG,
        steady(),
        phase_mode=PhaseMode.ALIGNED,
        overheads=replace(OVERHEADS, transport_up_ns=25 * NS_PER_MS),
    ),
    # No frame arrives within the run.
    "silent": Case(CONFIG, steady(), overheads=replace(OVERHEADS, transport_up_ns=60 * NS_PER_S)),
}


@st.composite
def short_runs(draw: st.DrawFn) -> Case:
    """Draw a 3 s run of up to six robots that join and leave at any time, under any settings."""
    robots = []
    for number in range(1, draw(st.integers(min_value=1, max_value=6)) + 1):
        start_ms = draw(st.integers(min_value=0, max_value=2_000))
        stop_s = draw(
            st.none()
            | st.integers(min_value=start_ms + 1, max_value=3_000).map(lambda ms: ms / 1_000)
        )
        robots.append(RobotPlan(f"r{number:02d}", start_ms / 1_000, stop_s))
    policy = draw(st.sampled_from(QueuePolicy))
    # The configuration refuses the skip rule on a queue.
    skip_rule = policy is QueuePolicy.LATEST and draw(st.booleans())
    # Up to a depth service time above its 150 ms threshold, which guard (a) of the skip rule
    # serves. The CPU time is drawn apart from the wall time: it sets the pacing hold.
    frame_ms = st.floats(min_value=0.5, max_value=250.0)
    return Case(
        replace(CONFIG, queue_policy=policy, skip_rule=skip_rule),
        Scenario("short", 3.0, tuple(robots), (Window("all", 0.0, 3.0),)),
        {model: linear_table(draw(frame_ms), draw(frame_ms)) for model in MODEL_KEYS},
        draw(st.sampled_from([None, QUOTA_CORES])),
        draw(st.sampled_from(PhaseMode)),
    )


def simulate_case(case: Case, seed: int) -> SimulationResult:
    return simulate(
        case.config,
        case.scenario,
        case.tables,
        case.overheads,
        case.io_cpu_ms_per_frame,
        case.quota_cores,
        case.phase_mode,
        seed,
    )


@functools.cache
def run(name: str) -> SimulationResult:
    """Return the result of a case of CASES, simulated once for every test that reads it."""
    return simulate_case(CASES[name], SEED)


def imbalances(result: SimulationResult) -> list[tuple[int, int]]:
    """Return how far each robot and model is off the conservation identities of DESIGN.md 10.3."""
    off: list[tuple[int, int]] = []
    for models in result.final.values():
        for final in models.values():
            counters = final.counters
            frames = counters.frames_received - (
                counters.frames_served
                + counters.frames_overwritten
                + counters.frames_skipped
                + counters.frames_dropped_disconnect
                + final.pending_count
            )
            commands = counters.frames_served - (
                counters.commands_sent
                + counters.commands_queued
                + counters.commands_dropped_downlink
                + counters.results_dropped_disconnect
                + counters.frames_failed
                + counters.in_flight
            )
            off.append((frames, commands))
    return off


def gaps_over_the_bound(
    case: Case, result: SimulationResult
) -> dict[tuple[str, str, str], tuple[int, int]]:
    """Return every command gap of a run above its bound, as (gap, bound) by model and robot.

    The gaps are the interval from a robot's start to its first command, which may also take
    the uplink and the delivery, and the intervals after it: between two commands, and from the
    last to the robot's stop or the end of the run. A robot that receives nothing waits from its
    start to its stop for its first command.
    """
    end_ns = round(case.scenario.duration_s * NS_PER_S)
    over: dict[tuple[str, str, str], tuple[int, int]] = {}
    for model in MODEL_KEYS:
        scheduler = result.models[model].plan.scheduler
        shares_cores = [
            share.share_cores
            for share in result.models[model].shares
            if share.share_cores is not None
        ]
        bound_ns = command_gap_bound_ns(
            len(case.scenario.robots),
            scheduler.batch_cap,
            round(NS_PER_S / case.config.frame_rate_hz),
            longest_cycle_ns(case.tables[model], case.overheads, scheduler.batch_cap, shares_cores),
            scheduler.skip_max_unserved_ns if scheduler.skip_rule else None,
        )
        first_bound_ns = bound_ns + case.overheads.transport_up_ns + case.overheads.deliver_ns
        for plan in case.scenario.robots:
            stop_ns = end_ns if plan.stop_s is None else round(plan.stop_s * NS_PER_S)
            received_ns = result.windows["all"][model][plan.robot_id].received_ns
            edges_ns = (round(plan.start_s * NS_PER_S), *received_ns, stop_ns)
            first_ns = edges_ns[1] - edges_ns[0]
            later_ns = max(
                (later - earlier for earlier, later in itertools.pairwise(edges_ns[1:])), default=0
            )
            if first_ns > first_bound_ns:
                over[model, plan.robot_id, "first"] = (first_ns, first_bound_ns)
            if later_ns > bound_ns:
                over[model, plan.robot_id, "later"] = (later_ns, bound_ns)
    return over


def first_captures_ns(result: SimulationResult) -> list[int]:
    """Return when each robot captured the frame of the first classifier command it received."""
    return [
        robot.received_ns[0] - robot.freshness_ns[0]
        for robot in result.windows["all"]["cls"].values()
    ]


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_under_fifo_the_age_of_the_served_frames_grows_for_the_whole_run(model: str) -> None:
    freshness_ns = run("fifo").windows["all"][model]["a"].freshness_ns

    assert len(freshness_ns) > 500
    assert all(later > earlier for earlier, later in itertools.pairwise(freshness_ns))


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_under_latest_the_age_of_the_served_frames_stays_within_a_period_a_service_and_overheads(
    model: str,
) -> None:
    # With one robot every batch holds one frame, so the service time is that of a batch of one.
    service_ns, _ = SLOW[model].at(1)
    overheads_ns = (
        OVERHEADS.transport_up_ns
        + OVERHEADS.dispatch_request_ns
        + OVERHEADS.dispatch_result_ns
        + OVERHEADS.deliver_ns
    )
    frame_period_ns = round(NS_PER_S / CONFIG.frame_rate_hz)

    freshness_ns = run("latest").windows["all"][model]["a"].freshness_ns

    assert len(freshness_ns) > 500
    assert max(freshness_ns) <= frame_period_ns + service_ns + overheads_ns


@pytest.mark.parametrize("name", list(CASES))
def test_both_conservation_identities_hold_at_the_end_of_every_run(name: str) -> None:
    off = imbalances(run(name))

    assert len(off) == len(CASES[name].scenario.robots) * len(MODEL_KEYS)
    assert off == [(0, 0)] * len(off)


@pytest.mark.parametrize("name", list(DEMOS))
def test_every_command_is_written_at_once_and_received_unless_it_is_still_travelling(
    name: str,
) -> None:
    result = run(name)
    finals = [
        (robot_id, model, final.counters)
        for robot_id, models in result.final.items()
        for model, final in models.items()
    ]

    # A command written less than one delivery before its robot's stop or the end of the run
    # is sent but not received, and the batches of a model complete more than one delivery
    # apart, so at most one command of a robot and model is still travelling.
    unreceived = {
        counters.commands_sent - len(result.windows["all"][model][robot_id].received_ns)
        for robot_id, model, counters in finals
    }

    assert unreceived <= {0, 1}
    assert {
        (counters.commands_queued, counters.commands_dropped_downlink) for *_, counters in finals
    } == {(0, 0)}
    assert min(counters.max_service_gap_ns for *_, counters in finals) > 0


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_robots_of_a_saturated_model_are_served_within_one_frame_of_each_other(model: str) -> None:
    counters = [models[model].counters for models in run("saturated").final.values()]
    served = [robot.frames_served for robot in counters]

    # Every robot lost frames to overwriting: the model could not serve them all.
    assert min(robot.frames_overwritten for robot in counters) > 0
    assert max(served) - min(served) <= 1


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_robot_that_joins_late_reaches_the_incumbents_rate_within_one_batch_over_the_window(
    model: str,
) -> None:
    result = run("late_joiner_skip_off")
    after = result.windows["after"][model]
    batch_cap = result.models[model].plan.scheduler.batch_cap

    difference_per_s = abs(after["b"].commands_per_s - after["a"].commands_per_s)

    # The window after the join runs from 10 s to 30 s.
    assert difference_per_s <= batch_cap / 20.0


@pytest.mark.parametrize("robot_id", ["a", "b"])
def test_without_pacing_a_survivor_gets_n_over_n_minus_one_times_its_depth_rate_after_a_stop(
    robot_id: str,
) -> None:
    windows = run("disconnect_unpaced").windows
    before_per_s = windows["before"]["depth"][robot_id].commands_per_s
    after_per_s = windows["after"]["depth"][robot_id].commands_per_s

    # Saturated before and after: below the frame rate in both windows.
    assert max(before_per_s, after_per_s) < CONFIG.frame_rate_hz
    assert after_per_s / before_per_s == pytest.approx(3 / 2, rel=0.05)


@pytest.mark.parametrize("name", list(CASES))
def test_every_command_gap_is_within_the_bound_the_scheduler_guarantees(name: str) -> None:
    assert gaps_over_the_bound(CASES[name], run(name)) == {}


@given(case=short_runs(), seed=st.integers(min_value=0, max_value=2**32 - 1))
def test_in_any_run_the_identities_hold_and_every_gap_is_within_the_bound(
    case: Case, seed: int
) -> None:
    result = simulate_case(case, seed)

    off = imbalances(result)
    assert off == [(0, 0)] * len(off)
    assert gaps_over_the_bound(case, result) == {}


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_with_spare_capacity_each_command_reaches_the_robot_one_round_trip_after_its_capture(
    model: str,
) -> None:
    # The robot captures frame k at 40k ms. The frame reaches the gateway 1 ms later, both idle
    # workers take it at once and are back 2 ms after that, and the command reaches the robot
    # 2 ms later still. Each command is applied over the one before, 45 ms after that one's
    # capture, and the longest wait for a command is one frame period.
    robot = run("fast").windows["all"][model]["a"]

    assert robot.received_ns == tuple((40 * frame + 5) * NS_PER_MS for frame in range(750))
    assert robot.freshness_ns == (5 * NS_PER_MS,) * 750
    assert robot.command_age_peak_ns == (45 * NS_PER_MS,) * 749
    assert (robot.commands_per_s, robot.stale, robot.out_of_order) == (25.0, 0, 0)
    assert robot.command_gap_max_ns == 40 * NS_PER_MS


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_command_received_at_the_end_of_a_window_counts_in_the_next_one(model: str) -> None:
    # The window before ends at 45 ms, as the second command arrives.
    windows = run("fast").windows
    before = windows["before"][model]["a"]
    after = windows["after"][model]["a"]

    assert before.received_ns == (5 * NS_PER_MS,)
    assert after.received_ns[:2] == (45 * NS_PER_MS, 85 * NS_PER_MS)
    # Each rate is over the window's whole length.
    assert (before.commands_per_s, after.commands_per_s) == pytest.approx((1 / 0.045, 749 / 29.955))


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_with_spare_capacity_every_frame_is_served_and_its_command_written_as_its_batch_completes(
    model: str,
) -> None:
    # The last frame is dispatched as it arrives, at 29961 ms, and its command written at
    # 29963 ms. Writes come 40 ms apart, and the first 3 ms after the connect.
    assert run("fast").final["a"][model] == FinalCounts(
        ModelCounters(
            frames_received=750,
            frames_served=750,
            commands_sent=750,
            last_served_seq=750,
            last_served_ns=29_961 * NS_PER_MS,
            last_command_ns=29_963 * NS_PER_MS,
            max_service_gap_ns=40 * NS_PER_MS,
        ),
        pending_count=0,
    )


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_model_run_books_the_batches_wall_time_as_busy_and_the_waits_for_frames_as_idle(
    model: str,
) -> None:
    # Each worker waits 1 ms for the first frame, then 38 ms after each 2 ms round trip for the
    # next. A batch takes 1 ms of wall time and 1.5 ms of CPU time.
    model_run = run("fast").models[model]

    assert model_run.batch_size == {1: 750}
    assert (model_run.busy_ns, model_run.paced_ns, model_run.idle_ns) == (
        750 * NS_PER_MS,
        0,
        (1 + 749 * 38) * NS_PER_MS,
    )
    assert model_run.cpu_ns == 1_125 * NS_PER_MS
    assert model_run.shares == (PacingShare(0, None),)


def test_the_predicted_demand_is_the_workers_cpu_time_and_the_frame_io_over_the_run() -> None:
    # One robot sends 750 frames, each served alone by each model at 1.5 ms of CPU time, and
    # each costs the measured 0.4 ms of frame I/O: 2.55 s of CPU time over 30 s.
    assert run("fast").cpu_demand_cores == pytest.approx(2.55 / 30)


def test_the_measured_io_cost_moves_the_predicted_demand_and_nothing_else() -> None:
    # Depth is paced from the second robot on, so a share computed from the measured cost would
    # move its batches.
    with_cost = run("paced")

    without_cost = simulate_case(replace(CASES["paced"], io_cpu_ms_per_frame=0.0), SEED)

    assert replace(without_cost, cpu_demand_cores=with_cost.cpu_demand_cores) == with_cost
    assert without_cost.cpu_demand_cores < with_cost.cpu_demand_cores


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_stop_drops_the_frame_still_pending_and_the_result_still_at_the_worker(
    model: str,
) -> None:
    # Frames come every 40 ms and a batch is back 51 ms after its dispatch. Frame 0 arrives at
    # 1 ms and its command is written at 52 ms, when frame 1, which arrived at 41 ms, is
    # dispatched. Frame 2 arrives at 81 ms and waits. The robot stops at 100 ms, before it
    # captures frame 3, and the result of frame 1 comes back at 103 ms.
    result = run("stop")

    assert result.final["a"][model] == FinalCounts(
        ModelCounters(
            frames_received=3,
            frames_served=2,
            frames_dropped_disconnect=1,
            commands_sent=1,
            results_dropped_disconnect=1,
            last_served_seq=2,
            last_served_ns=52 * NS_PER_MS,
            last_command_ns=52 * NS_PER_MS,
            max_service_gap_ns=52 * NS_PER_MS,
        ),
        pending_count=0,
    )
    assert result.windows["all"][model]["a"].received_ns == (54 * NS_PER_MS,)


def test_a_stopped_robot_receives_nothing_from_its_stop_on() -> None:
    result = run("disconnect")

    assert max(max(result.windows["all"][model]["c"].received_ns) for model in MODEL_KEYS) < (
        20 * NS_PER_S
    )
    assert {model: "c" in result.windows["after"][model] for model in MODEL_KEYS} == {
        "cls": False,
        "depth": False,
    }


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_stopped_robot_holds_no_frame_and_is_served_no_more_after_its_stop(model: str) -> None:
    final = run("disconnect").final["c"][model]
    served_ns = final.counters.last_served_ns

    assert final.pending_count == 0
    assert served_ns is not None
    assert served_ns < 20 * NS_PER_S


@pytest.mark.parametrize(
    ("name", "window", "robot_id", "from_s", "to_s"),
    [
        pytest.param("late_joiner", "all", "b", 10.0, 30.0, id="joins-within-the-window"),
        pytest.param("late_joiner", "after", "b", 10.0, 30.0, id="joins-at-the-start"),
        pytest.param("disconnect", "all", "c", 0.0, 20.0, id="stops-within-the-window"),
        pytest.param("disconnect", "before", "c", 0.0, 20.0, id="stops-at-the-end"),
        pytest.param("disconnect", "after", "a", 20.0, 40.0, id="runs-through"),
        pytest.param("stop", "all", "a", 0.0, 0.1, id="one-command"),
        pytest.param("silent", "all", "a", 0.0, 30.0, id="no-command"),
    ],
)
@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_windows_longest_gap_runs_from_the_robots_start_through_its_commands_to_its_stop(
    name: str, window: str, robot_id: str, from_s: float, to_s: float, model: str
) -> None:
    robot = run(name).windows[window][model][robot_id]

    edges_ns = (round(from_s * NS_PER_S), *robot.received_ns, round(to_s * NS_PER_S))

    assert robot.command_gap_max_ns == max(
        later - earlier for earlier, later in itertools.pairwise(edges_ns)
    )


@pytest.mark.parametrize(
    ("name", "window", "robot_id", "window_s"),
    [
        pytest.param("late_joiner", "all", "b", 30.0, id="joins-within-the-window"),
        pytest.param("disconnect", "all", "c", 40.0, id="stops-within-the-window"),
        pytest.param("stop", "all", "a", 1.0, id="one-command"),
    ],
)
@pytest.mark.parametrize("model", MODEL_KEYS)
def test_a_robots_rate_is_its_commands_over_the_whole_window_even_if_it_ran_for_part_of_it(
    name: str, window: str, robot_id: str, window_s: float, model: str
) -> None:
    robot = run(name).windows[window][model][robot_id]

    assert robot.commands_per_s == pytest.approx(len(robot.received_ns) / window_s)


def test_a_command_older_than_its_threshold_is_stale_and_replaces_no_command() -> None:
    # Depth's service time alone is above its 150 ms threshold, so every depth command arrives
    # stale, and none is applied over another.
    robot = run("stale").windows["all"]["depth"]["a"]

    assert len(robot.received_ns) > 0
    assert (robot.stale, robot.out_of_order, robot.command_age_peak_ns) == (
        len(robot.received_ns),
        0,
        (),
    )


def test_the_skip_rule_counts_a_frames_uplink_twice_in_its_age() -> None:
    # A depth frame is doomed once its dwell, two 25 ms links, the 80 ms service time at the cap
    # of 2 and the 10 ms allowance exceed its 150 ms threshold: after 10 ms of waiting. Frames
    # come every 33.3 ms and a batch of one is back after 41 ms, so of every three frames depth
    # takes the first as it arrives and the second after 7.7 ms, and skips the third after
    # 15.3 ms. The run ends with the second of the last three in flight and the third pending.
    final = run("long_uplink").final["a"]["depth"]

    assert (final.counters.frames_served, final.counters.frames_skipped) == (600, 299)
    assert (final.counters.in_flight, final.pending_count) == (1, 1)


def test_the_paced_model_gets_the_share_the_gateway_derives_after_every_start_and_stop() -> None:
    # Each robot offers 30 frames per second, at 15 ms of classifier CPU time and 0.15 ms of
    # frame I/O each: the configured cost, as in the gateway, and not the measured one. With one
    # robot depth may use its whole thread. With two, it gets what the classifier and the frame
    # I/O leave of the 1.9 cores that the quota keeps after the headroom; from three on, the
    # classifier's thread is full.
    result = run("paced")
    shares = result.models["depth"].shares

    assert [share.at_ns for share in shares] == [0, 0, 0, 20 * NS_PER_S]
    assert [share.share_cores for share in shares] == [
        None,
        pytest.approx(1.9 - 2 * 0.45 - 2 * 0.0045),
        pytest.approx(1.9 - 1 - 3 * 0.0045),
        pytest.approx(1.9 - 2 * 0.45 - 2 * 0.0045),
    ]
    assert {share.share_cores for share in result.models["cls"].shares} == {None}
    assert result.models["depth"].paced_ns > 0


def test_a_paced_worker_waits_for_its_batch_cpu_time_over_the_share_when_that_outlasts_it() -> None:
    # One robot: the classifier's 30 ms of CPU time per frame and the frame I/O leave depth
    # 0.9955 of the 1.9 cores. A depth batch holds one frame, takes 60 ms of CPU time and is
    # back 41 ms after its dispatch, so the hold, 60 ms over the share, is the longer.
    result = run("paced_alone")
    depth = result.models["depth"]
    (share,) = depth.shares

    assert share.share_cores is not None
    assert share.share_cores == pytest.approx(1.9 - 0.9 - 0.0045)
    hold_ns = pacing_release_ns(0, 60 * NS_PER_MS, share.share_cores)
    received_ns = result.windows["all"]["depth"]["a"].received_ns
    assert {later - earlier for earlier, later in itertools.pairwise(received_ns)} == {hold_ns}
    # The worker waits 1 ms for the first frame, and is held after each batch until the next.
    dispatches = sum(depth.batch_size.values())
    assert (depth.idle_ns, depth.paced_ns) == (
        NS_PER_MS,
        (dispatches - 1) * (hold_ns - 41 * NS_PER_MS),
    )


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_without_a_quota_no_model_is_paced(model: str) -> None:
    run_of_model = run("disconnect_unpaced").models[model]

    assert run_of_model.shares == (
        PacingShare(0, None),
        PacingShare(0, None),
        PacingShare(0, None),
        PacingShare(20 * NS_PER_S, None),
    )
    assert run_of_model.paced_ns == 0


def test_aligned_phases_give_the_classifier_larger_batches_than_random_phases() -> None:
    def mean_batch_size(result: SimulationResult) -> float:
        batch_size = result.models["cls"].batch_size
        return sum(size * count for size, count in batch_size.items()) / sum(batch_size.values())

    assert mean_batch_size(run("aligned")) > mean_batch_size(run("sweep_n5"))


def test_random_phases_lie_within_one_frame_period_and_follow_the_seed() -> None:
    # The five robots start at 0 s, and the classifier serves the first frame of each: a batch
    # of all five is back within 26 ms, less than a frame period. So the capture time of each
    # robot's first command is its phase.
    period_ns = round(NS_PER_S / CONFIG.frame_rate_hz)
    phases_ns = first_captures_ns(run("sweep_n5"))
    other_phases_ns = first_captures_ns(simulate_case(CASES["sweep_n5"], SEED + 1))

    assert [phase_ns // period_ns for phase_ns in phases_ns + other_phases_ns] == [0] * 10
    assert phases_ns != other_phases_ns


def test_aligned_phases_capture_every_robots_first_frame_at_its_start() -> None:
    assert first_captures_ns(run("aligned")) == [0] * 5


def test_a_run_is_deterministic_for_a_given_seed() -> None:
    def simulated() -> SimulationResult:
        return simulate(
            CONFIG,
            sweep(3),
            TABLES,
            OVERHEADS,
            IO_CPU_MS_PER_FRAME,
            QUOTA_CORES,
            PhaseMode.RANDOM,
            7,
        )

    assert simulated() == simulated()


def test_simulate_refuses_a_table_without_a_configured_batch_size() -> None:
    times_ns = {1: NS_PER_MS, 2: 2 * NS_PER_MS, 4: 4 * NS_PER_MS}
    tables = {"cls": ServiceTable(times_ns, times_ns), "depth": TABLES["depth"]}

    with pytest.raises(ValueError, match="service table of cls"):
        simulate(
            CONFIG, steady(), tables, OVERHEADS, IO_CPU_MS_PER_FRAME, None, PhaseMode.RANDOM, SEED
        )


@pytest.mark.parametrize(
    ("batch_size", "times_ns"),
    [(1, (10, 9)), (3, (35, 33)), (4, (50, 46)), (6, (70, 66)), (7, (80, 76)), (8, (90, 86))],
)
def test_a_service_table_interpolates_linearly_between_the_nearest_sizes_it_has(
    batch_size: int, times_ns: tuple[int, int]
) -> None:
    table = ServiceTable(wall_ns={1: 10, 2: 20, 4: 50, 8: 90}, cpu_ns={1: 9, 2: 20, 4: 46, 8: 86})

    assert table.at(batch_size) == times_ns


@pytest.mark.parametrize(("batch_size", "wall_ns"), [(2, 3), (3, 7)])
def test_an_interpolated_time_is_rounded_to_whole_nanoseconds(
    batch_size: int, wall_ns: int
) -> None:
    table = ServiceTable(wall_ns={1: 0, 4: 10}, cpu_ns={1: 0, 4: 10})

    assert table.at(batch_size)[0] == wall_ns
