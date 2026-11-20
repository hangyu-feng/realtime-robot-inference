"""Tests for the scheduling core: the derivations of its settings, and the per-model scheduler."""

import math
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest

from cloud.config import QueuePolicy, ServerConfig
from cloud.mailbox import Frame, Mailbox
from cloud.registry import ModelCounters, RobotRecord, RobotRegistry
from cloud.scheduler import (
    Batch,
    Dispatch,
    Idle,
    ModelPlan,
    ModelScheduler,
    SchedulerConfig,
    WaitUntil,
    batch_cap,
    other_demand_cores,
    pacing_release_ns,
    pacing_share,
    plan_model,
)

NS_PER_MS = 1_000_000

# At 25 frames per second the frame period is exactly 40 ms, so every budget below is exact.
PLAN_CONFIG = ServerConfig(frame_rate_hz=25.0)
# Calibrated wall times per batch size, in ms.
CLS_T_B_MS = {1: 100, 2: 200, 4: 400, 8: 800}

# The scheduler tests below count time in plain nanoseconds on the gateway's clock.
MODEL = "cls"
STALE_NS = 100
SERVICE_NS = 10
ALLOWANCE_NS = 5
SKIP_MAX_UNSERVED_NS = 1_000
FRAME_PERIOD_NS = 40
RETENTION_NS = 1_000
MAX_ENTRIES = 8
BASE = SchedulerConfig(
    model=MODEL,
    batch_cap=2,
    service_ns=SERVICE_NS,
    return_allowance_ns=ALLOWANCE_NS,
    skip_rule=False,
    skip_max_unserved_ns=SKIP_MAX_UNSERVED_NS,
)
# The scheduler adds the service time and the allowance to a frame's age, so a frame is doomed
# when its fire lag, twice its uplink and its dwell add up to DOOMING_NS or more. DOOMING_NS is
# even, so an uplink of half of it dooms a frame too.
DOOMING_NS = STALE_NS - SERVICE_NS - ALLOWANCE_NS + 1
# What next() returns, and the robot's frames_served and frames_skipped after it.
SERVED = (Dispatch, 1, 0)
SKIPPED = (Idle, 0, 1)


def ns(table_ms: dict[int, int]) -> dict[int, int]:
    return {size: time_ms * NS_PER_MS for size, time_ms in table_ms.items()}


# The changes are typed Any: mypy checks the value type of a ** dict passed to
# dataclasses.replace against every field, and the callers change fields of different types.
def with_model(config: ServerConfig, model: str, **changes: Any) -> ServerConfig:
    """Return config with the settings of one model changed."""
    return replace(
        config, models={**config.models, model: replace(config.models[model], **changes)}
    )


class World:
    """A registry, a mailbox and one scheduler, driven the way the gateway drives them."""

    def __init__(self, config: SchedulerConfig = BASE) -> None:
        self.registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
        self.mailbox = Mailbox(QueuePolicy.LATEST, fifo_capacity_frames=1)
        self.scheduler = ModelScheduler(config, self.mailbox, 0)

    def connect(self, robot_id: str, now_ns: int, stale_ns: int = STALE_NS) -> RobotRecord:
        record = self.registry.connect(
            robot_id, (MODEL,), {MODEL: stale_ns}, {MODEL: 2 * stale_ns}, FRAME_PERIOD_NS, now_ns
        )
        self.mailbox.open(record)
        return record

    def disconnect(self, robot_id: str, now_ns: int) -> None:
        self.mailbox.close(robot_id)
        self.registry.disconnect(robot_id, now_ns)

    def put(
        self,
        robot_id: str,
        frame_number: int,
        received_ns: int,
        fire_lag_ns: int = 0,
        transport_up_ns: int = 0,
    ) -> None:
        capture_ns = received_ns - transport_up_ns - fire_lag_ns
        self.mailbox.put(
            robot_id,
            Frame(frame_number, capture_ns, received_ns, fire_lag_ns, transport_up_ns, b"\xff\xd8"),
        )

    def dispatch(self, now_ns: int) -> Batch:
        decision = self.scheduler.next(now_ns)
        assert isinstance(decision, Dispatch)
        return decision.batch

    def serve(self, robot_ids: str, frame_number: int, now_ns: int) -> list[str]:
        """Put a frame for each robot, dispatch and complete one batch; return whom it served."""
        for robot_id in robot_ids:
            self.put(robot_id, frame_number, now_ns)
        batch = self.dispatch(now_ns)
        self.scheduler.complete(batch, (), 1, 1, now_ns + 1)
        return served_ids(batch)

    def counters(self, robot_id: str) -> ModelCounters:
        return self.registry.get(robot_id).models[MODEL]


def served_ids(batch: Batch) -> list[str]:
    return [item.record.robot_id for item in batch.items]


def complete_without_failures(scheduler: ModelScheduler, batch: Batch) -> object:
    return scheduler.complete(batch, (), 5, 5, 3 * RETENTION_NS)


def fail_batch(scheduler: ModelScheduler, batch: Batch) -> object:
    scheduler.fail(batch, 3 * RETENTION_NS)
    return None


def test_plan_model_derives_every_setting_from_the_configuration_and_the_batch_times() -> None:
    plan = plan_model("cls", PLAN_CONFIG, ns(CLS_T_B_MS))

    # The budget is the 500 ms threshold minus the 40 ms frame period and the 10 ms allowance.
    assert plan == ModelPlan(
        scheduler=SchedulerConfig(
            model="cls",
            batch_cap=4,
            service_ns=400 * NS_PER_MS,
            return_allowance_ns=10 * NS_PER_MS,
            skip_rule=True,
            skip_max_unserved_ns=1_000 * NS_PER_MS,
        ),
        budget_ns=450 * NS_PER_MS,
        margin_ns=50 * NS_PER_MS,
        feasible=True,
    )


def test_plan_model_copies_the_skip_rule_settings_and_the_allowance() -> None:
    config = replace(
        PLAN_CONFIG, skip_rule=False, skip_max_unserved_ms=250.0, overhead_allowance_ms=0.0
    )

    plan = plan_model("cls", config, ns(CLS_T_B_MS))

    assert (plan.scheduler.skip_rule, plan.scheduler.skip_max_unserved_ns) == (
        False,
        250 * NS_PER_MS,
    )
    assert (plan.scheduler.return_allowance_ns, plan.budget_ns) == (0, 460 * NS_PER_MS)


@pytest.mark.parametrize(
    ("model", "forced_batch_cap", "t_b_ms", "cap", "margin_ms", "feasible"),
    [
        pytest.param("cls", None, {1: 100, 2: 200, 4: 450, 8: 800}, 4, 0, True, id="at-budget"),
        pytest.param("cls", 8, CLS_T_B_MS, 8, -350, False, id="forced-above-the-rule"),
        pytest.param("cls", 1, CLS_T_B_MS, 1, 350, True, id="forced-below-the-rule"),
        # The depth budget is 150 - 40 - 10 = 100 ms.
        pytest.param("depth", None, {1: 120, 2: 230, 4: 450, 8: 900}, 1, -20, True, id="no-fit"),
        pytest.param(
            "depth", None, {1: 150, 2: 290, 4: 580, 8: 1160}, 1, -50, True, id="at-threshold"
        ),
        pytest.param(
            "depth", None, {1: 151, 2: 290, 4: 580, 8: 1160}, 1, -51, False, id="over-threshold"
        ),
    ],
)
def test_plan_model_picks_the_cap_and_judges_the_service_time_against_the_threshold(
    model: str,
    forced_batch_cap: int | None,
    t_b_ms: dict[int, int],
    cap: int,
    margin_ms: int,
    feasible: bool,
) -> None:
    config = with_model(PLAN_CONFIG, model, forced_batch_cap=forced_batch_cap)

    plan = plan_model(model, config, ns(t_b_ms))

    assert (plan.scheduler.batch_cap, plan.scheduler.service_ns) == (cap, ns(t_b_ms)[cap])
    assert (plan.margin_ns, plan.feasible) == (margin_ms * NS_PER_MS, feasible)


@pytest.mark.parametrize(
    "t_b_ms",
    [{1: 100, 2: 200, 4: 400}, {1: 100, 2: 200, 3: 300, 4: 400, 8: 800}, {}],
    ids=["a-size-missing", "an-uncalibrated-size", "empty"],
)
def test_plan_model_refuses_a_table_whose_sizes_are_not_the_calibrated_ones(
    t_b_ms: dict[int, int],
) -> None:
    with pytest.raises(ValueError, match="not the calibrated sizes"):
        plan_model("cls", PLAN_CONFIG, ns(t_b_ms))


@pytest.mark.parametrize(
    ("t_b_ns", "budget_ns", "cap"),
    [
        ({1: 10, 2: 20, 4: 40}, 1_000, 4),
        ({1: 10, 2: 20, 4: 40}, 40, 4),
        ({1: 10, 2: 20, 4: 40}, 39, 2),
        ({1: 10, 2: 20, 4: 40}, 10, 1),
        ({1: 10, 2: 20, 4: 40}, 9, 1),
        ({1: 10, 2: 20, 4: 40}, -5, 1),
        ({1: 30, 2: 20, 4: 50}, 25, 2),
        ({4: 40, 1: 10, 2: 20}, 5, 1),
    ],
    ids=[
        "every-size-fits",
        "largest-at-the-budget",
        "largest-just-over-the-budget",
        "only-the-smallest-fits",
        "none-fits",
        "negative-budget",
        "times-not-ascending",
        "none-fits-sizes-not-ascending",
    ],
)
def test_batch_cap_is_the_largest_size_within_the_budget_or_the_smallest(
    t_b_ns: dict[int, int], budget_ns: int, cap: int
) -> None:
    assert batch_cap(t_b_ns, budget_ns) == cap


@pytest.mark.parametrize(
    ("start_ns", "cpu_ns", "share_cores", "release_ns"),
    [
        (1_000, 300, 0.5, 1_600),
        (0, 1_000, 2.0, 500),
        (5, 0, 0.25, 5),
        # A third of a nanosecond rounds up to a whole one.
        (0, 1, 3.0, 1),
        # The float 0.3 lies just under three tenths, so 3 ns of CPU time need just over 10 ns.
        (0, 3, 0.3, 11),
    ],
)
def test_pacing_release_is_the_cpu_time_over_the_share_rounded_up_after_the_start(
    start_ns: int, cpu_ns: int, share_cores: float, release_ns: int
) -> None:
    assert pacing_release_ns(start_ns, cpu_ns, share_cores) == release_ns


@pytest.mark.parametrize(
    ("headroom_cores", "threads", "other_cores", "share_cores"),
    [
        pytest.param(0.25, 1, 1.25, 0.5, id="remainder-below-the-threads"),
        pytest.param(0.25, 1, 0.75, None, id="remainder-at-the-threads"),
        pytest.param(0.25, 1, 0.5, None, id="remainder-above-the-threads"),
        pytest.param(0.25, 2, 0.25, 1.5, id="remainder-below-two-threads"),
        pytest.param(0.25, 0, 0.0, 1.75, id="runtime-threads-count-as-the-quota"),
        pytest.param(0.0, 0, 0.0, None, id="runtime-threads-with-the-whole-quota"),
        pytest.param(0.25, 1, 1.625, 0.25, id="remainder-below-the-floor"),
        pytest.param(0.25, 1, 3.0, 0.25, id="nothing-left"),
    ],
)
def test_pacing_share_gives_the_paced_worker_what_the_quota_leaves_it(
    headroom_cores: float, threads: int, other_cores: float, share_cores: float | None
) -> None:
    # Every value is a sum of powers of two, so the float arithmetic is exact.
    assert pacing_share(2.0, headroom_cores, 0.25, threads, other_cores) == share_cores


# Robots declare 25 fps, so each offers 25 frames per second: 0.0125 cores of frame I/O at
# 0.5 ms per frame, 0.25 cores of cls at 10 ms per frame and 0.5 cores of depth at 20 ms.
DEMAND_CONFIG = ServerConfig(io_cpu_ms_per_frame=0.5)
DEMAND_CPU_NS_PER_FRAME = {"cls": 10 * NS_PER_MS, "depth": 20 * NS_PER_MS}
BOTH = ("cls", "depth")
CLS_PACED = with_model(with_model(DEMAND_CONFIG, "depth", paced=False), "cls", paced=True)


@pytest.mark.parametrize(
    ("config", "connected", "disconnected", "expected_cores"),
    [
        pytest.param(DEMAND_CONFIG, [], [], 0.0, id="no-robots"),
        pytest.param(DEMAND_CONFIG, [BOTH], [], 0.2625, id="below-the-thread-limit"),
        pytest.param(DEMAND_CONFIG, [BOTH] * 4, [], 1.05, id="at-the-thread-limit"),
        pytest.param(DEMAND_CONFIG, [BOTH] * 6, [], 1.075, id="over-the-thread-limit"),
        pytest.param(
            with_model(DEMAND_CONFIG, "cls", threads=0),
            [BOTH] * 10,
            [],
            2.125,
            id="runtime-threads-limited-by-the-quota",
        ),
        pytest.param(DEMAND_CONFIG, [("depth",)], [], 0.0125, id="subscribed-to-the-paced-only"),
        pytest.param(DEMAND_CONFIG, [BOTH], [BOTH], 0.2625, id="a-disconnected-record"),
        pytest.param(CLS_PACED, [BOTH], [], 0.5125, id="the-paced-model-left-out"),
        pytest.param(
            with_model(DEMAND_CONFIG, "depth", paced=False), [BOTH], [], 0.7625, id="none-paced"
        ),
    ],
)
def test_other_demand_counts_frame_io_and_each_unpaced_model_up_to_its_cores(
    config: ServerConfig,
    connected: list[tuple[str, ...]],
    disconnected: list[tuple[str, ...]],
    expected_cores: float,
) -> None:
    registry = RobotRegistry(RETENTION_NS, max_entries=16)
    records = [
        registry.connect(
            f"r{index:02d}",
            subscribed,
            dict.fromkeys(subscribed, 500),
            dict.fromkeys(subscribed, 1_000),
            40 * NS_PER_MS,
            0,
        )
        for index, subscribed in enumerate([*connected, *disconnected])
    ]
    for record in records[len(connected) :]:
        registry.disconnect(record.robot_id, 1)

    predicted_cores = other_demand_cores(config, registry.records(), DEMAND_CPU_NS_PER_FRAME, 2.0)

    assert predicted_cores == pytest.approx(expected_cores)


def test_the_first_call_is_idle_when_no_frame_is_pending() -> None:
    world = World()
    world.connect("a", 0)

    assert world.scheduler.next(10) == Idle()


def test_robots_never_served_go_first_by_connect_time_then_robot_id() -> None:
    world = World(replace(BASE, batch_cap=3))
    world.connect("c", 1)
    world.connect("b", 2)
    world.connect("a", 2)
    for robot_id in "abc":
        world.put(robot_id, 1, 10)

    assert served_ids(world.dispatch(10)) == ["c", "a", "b"]


def test_a_dispatch_records_the_session_frame_and_service_order_of_each_robot() -> None:
    world = World()
    records = [world.connect("a", 0), world.connect("b", 0)]
    for robot_id in "ab":
        world.put(robot_id, 1, 5)

    batch = world.dispatch(10)

    assert [(item.session, item.frame.frame_number) for item in batch.items] == [(1, 1), (2, 1)]
    assert [
        (record.models[MODEL].last_served_seq, record.models[MODEL].last_served_ns)
        for record in records
    ] == [(1, 10), (2, 10)]


def test_robots_are_served_least_recently_served_first_and_a_newcomer_goes_first_once() -> None:
    world = World(replace(BASE, batch_cap=1))
    world.connect("a", 0)
    world.connect("b", 0)

    served = [world.serve("ab", 1, 100), world.serve("ab", 2, 200)]
    world.connect("d", 250)
    served += [
        world.serve("abd", frame_number, 100 * frame_number) for frame_number in (3, 4, 5, 6)
    ]

    assert served == [["a"], ["b"], ["d"], ["a"], ["b"], ["d"]]


def test_robots_served_in_one_batch_go_next_time_in_the_order_they_were_served() -> None:
    world = World()
    for connected_ns, robot_id in enumerate("bac"):
        world.connect(robot_id, connected_ns)

    served = [world.serve("bac", 1, 100), world.serve("bac", 2, 200)]

    # c, never served, goes first; then b, served ahead of a in the first batch, although a
    # comes first by robot id and both were served at the same time.
    assert served == [["b", "a"], ["c", "b"]]


def test_a_batch_holds_at_most_the_cap_and_robots_left_out_keep_their_frames() -> None:
    world = World(replace(BASE, skip_rule=True))
    for robot_id in "abc":
        world.connect(robot_id, 0)
    world.put("a", 1, 500)
    world.put("b", 1, 500)
    # Doomed, but the walk stops at the cap before it reaches this frame.
    world.put("c", 1, 500 - DOOMING_NS)

    batch = world.dispatch(500)

    assert served_ids(batch) == ["a", "b"]
    assert (world.mailbox.pending_count("c", MODEL), world.counters("c").frames_skipped) == (1, 0)


def test_a_robot_with_no_new_frame_is_not_served_again() -> None:
    world = World()
    world.connect("a", 0)
    world.connect("b", 0)
    world.serve("ab", 1, 10)

    second = world.serve("b", 2, 30)

    assert (second, world.scheduler.next(50)) == (["b"], Idle())


@pytest.mark.parametrize(
    ("skip_rule", "fire_lag_ns", "transport_up_ns", "dwell_ns", "unserved_ns", "outcome"),
    [
        pytest.param(True, 0, 0, DOOMING_NS, 100, SKIPPED, id="doomed"),
        pytest.param(True, 0, 0, DOOMING_NS - 1, 100, SERVED, id="arrives-at-the-threshold"),
        pytest.param(True, DOOMING_NS - 1, 0, 1, 100, SKIPPED, id="fixed-part-at-the-threshold"),
        pytest.param(True, DOOMING_NS, 0, 0, 100, SERVED, id="fire-lag-alone-dooms-it"),
        pytest.param(True, 0, DOOMING_NS // 2, 0, 100, SERVED, id="uplink-alone-dooms-it"),
        # The downlink is estimated as equal to the uplink, so the uplink counts twice.
        pytest.param(
            True, 0, 20, DOOMING_NS - 1 - 2 * 20, 100, SERVED, id="uplink-twice-at-the-threshold"
        ),
        pytest.param(True, 0, 20, DOOMING_NS - 2 * 20, 100, SKIPPED, id="uplink-twice-doomed"),
        # The dwell starts at the frame's arrival, so the fire lag is not counted a second time.
        pytest.param(
            True, 40, 0, DOOMING_NS - 1 - 40, 100, SERVED, id="fire-lag-and-dwell-at-the-threshold"
        ),
        pytest.param(
            True,
            0,
            0,
            DOOMING_NS,
            SKIP_MAX_UNSERVED_NS - 1,
            SKIPPED,
            id="unserved-since-connect-just-under-the-limit",
        ),
        pytest.param(
            True,
            0,
            0,
            DOOMING_NS,
            SKIP_MAX_UNSERVED_NS,
            SERVED,
            id="unserved-since-connect-for-the-limit",
        ),
        pytest.param(False, 0, 0, DOOMING_NS, 100, SERVED, id="skip-rule-off"),
    ],
)
def test_the_skip_rule_passes_over_a_doomed_frame_unless_a_guard_holds(
    skip_rule: bool,
    fire_lag_ns: int,
    transport_up_ns: int,
    dwell_ns: int,
    unserved_ns: int,
    outcome: tuple[type, int, int],
) -> None:
    now_ns = 5_000
    world = World(replace(BASE, skip_rule=skip_rule))
    world.connect("a", now_ns - unserved_ns)
    world.put("a", 1, now_ns - dwell_ns, fire_lag_ns, transport_up_ns)

    decision = world.scheduler.next(now_ns)

    counters = world.counters("a")
    assert (type(decision), counters.frames_served, counters.frames_skipped) == outcome
    assert world.mailbox.pending_count("a", MODEL) == 0


@pytest.mark.parametrize("capture_ns", [-(10**12), 0, 10**12], ids=["behind", "zero", "ahead"])
@pytest.mark.parametrize(
    ("dwell_ns", "outcome"),
    [(DOOMING_NS - 1, SERVED), (DOOMING_NS, SKIPPED)],
    ids=["arrives-at-the-threshold", "doomed"],
)
def test_the_skip_rule_measures_the_dwell_on_the_gateway_clock_whatever_the_robot_clock_reads(
    capture_ns: int, dwell_ns: int, outcome: tuple[type, int, int]
) -> None:
    # Without a shared clock the capture time is on the robot's own clock, which the gateway
    # never subtracts from its own (DESIGN.md 5.4).
    now_ns = 5_000
    world = World(replace(BASE, skip_rule=True))
    world.connect("a", now_ns - 100)
    world.mailbox.put("a", Frame(1, capture_ns, now_ns - dwell_ns, 0, 0, b"\xff\xd8"))

    decision = world.scheduler.next(now_ns)

    counters = world.counters("a")
    assert (type(decision), counters.frames_served, counters.frames_skipped) == outcome


@pytest.mark.parametrize(
    ("since_served_ns", "outcome"),
    [(SKIP_MAX_UNSERVED_NS - 1, (Idle, 1, 1)), (SKIP_MAX_UNSERVED_NS, (Dispatch, 2, 0))],
    ids=["just-under-the-limit", "for-the-limit"],
)
def test_the_skip_rule_counts_the_unserved_time_from_the_last_service(
    since_served_ns: int, outcome: tuple[type, int, int]
) -> None:
    world = World(replace(BASE, skip_rule=True))
    world.connect("a", 0)
    world.serve("a", 1, 10)
    now_ns = 10 + since_served_ns
    world.put("a", 2, now_ns - DOOMING_NS)

    decision = world.scheduler.next(now_ns)

    counters = world.counters("a")
    assert (type(decision), counters.frames_served, counters.frames_skipped) == outcome


@pytest.mark.parametrize(
    ("since_served_ns", "outcome"),
    [(SKIP_MAX_UNSERVED_NS - 1, (Idle, 1, 1)), (SKIP_MAX_UNSERVED_NS, (Dispatch, 2, 0))],
    ids=["just-under-the-limit", "for-the-limit"],
)
def test_the_skip_rule_counts_the_unserved_time_from_a_service_in_an_earlier_session(
    since_served_ns: int, outcome: tuple[type, int, int]
) -> None:
    # A reconnect keeps the fairness memory, so it does not restart the unserved time.
    world = World(replace(BASE, skip_rule=True))
    world.connect("a", 0)
    world.serve("a", 1, 10)
    world.disconnect("a", 20)
    world.connect("a", 30)
    now_ns = 10 + since_served_ns
    world.put("a", 1, now_ns - DOOMING_NS)

    decision = world.scheduler.next(now_ns)

    counters = world.counters("a")
    assert (type(decision), counters.frames_served, counters.frames_skipped) == outcome


def test_a_skipped_robot_takes_no_place_in_the_batch_and_the_walk_moves_on() -> None:
    world = World(replace(BASE, batch_cap=1, skip_rule=True))
    world.connect("a", 0)
    world.connect("b", 0)
    world.put("a", 1, 500 - DOOMING_NS)
    world.put("b", 1, 500)

    batch = world.dispatch(500)

    assert served_ids(batch) == ["b"]
    assert (world.counters("a").frames_skipped, world.counters("a").last_served_seq) == (1, None)


def test_the_skip_rule_judges_a_frame_against_the_threshold_its_robot_declared() -> None:
    world = World(replace(BASE, skip_rule=True))
    world.connect("strict", 0, stale_ns=STALE_NS)
    world.connect("lenient", 0, stale_ns=2 * STALE_NS)
    for robot_id in ("strict", "lenient"):
        world.put(robot_id, 1, 500 - DOOMING_NS)

    batch = world.dispatch(500)

    assert served_ids(batch) == ["lenient"]
    assert world.counters("strict").frames_skipped == 1


def test_pacing_holds_the_worker_until_the_cpu_time_over_the_share_has_passed() -> None:
    # At half a core, a batch dispatched at 100 that used 40 ns of CPU releases the worker at 180.
    world = World()
    world.scheduler.set_pacing_share(0.5)
    world.connect("a", 0)
    world.put("a", 1, 100)
    first = world.dispatch(100)
    world.scheduler.complete(first, (), worker_ns=30, cpu_ns=40, now_ns=150)
    world.put("a", 2, 160)

    holds = [world.scheduler.next(160), world.scheduler.next(179)]
    world.put("a", 3, 179)
    second = world.dispatch(180)

    assert holds == [WaitUntil(180), WaitUntil(180)]
    assert second.items[0].frame.frame_number == 3


def test_a_pacing_hold_leaves_a_doomed_frame_pending_with_the_skip_rule_on() -> None:
    # At half a core, a batch dispatched at 1000 that used 200 ns of CPU releases the worker at
    # 1400. The frame pending at 1300 is doomed, and the robot was served too recently for
    # guard (b), so only a dispatch would skip it.
    world = World(replace(BASE, skip_rule=True))
    world.scheduler.set_pacing_share(0.5)
    world.connect("a", 0)
    world.put("a", 1, 1_000)
    first = world.dispatch(1_000)
    world.scheduler.complete(first, (), worker_ns=30, cpu_ns=200, now_ns=1_050)
    world.put("a", 2, 1_300 - DOOMING_NS)

    held = world.scheduler.next(1_300)

    assert held == WaitUntil(1_400)
    assert (world.counters("a").frames_skipped, world.mailbox.pending_count("a", MODEL)) == (0, 1)


@pytest.mark.parametrize(
    ("complete_ns", "dispatch_ns", "paced_ns", "idle_ns"),
    [(150, 180, 30, 100), (150, 200, 30, 120), (190, 200, 0, 110)],
    ids=[
        "dispatched-at-the-release",
        "dispatched-after-the-release",
        "completed-after-the-release",
    ],
)
def test_the_wait_before_a_dispatch_is_paced_up_to_the_release_and_idle_after_it(
    complete_ns: int, dispatch_ns: int, paced_ns: int, idle_ns: int
) -> None:
    # The worker is free from 0 and idle until the first dispatch at 100. At half a core the
    # 40 ns of CPU time of that batch release the worker at 180.
    world = World()
    world.scheduler.set_pacing_share(0.5)
    world.connect("a", 0)
    world.put("a", 1, 100)
    first = world.dispatch(100)
    world.scheduler.complete(first, (), worker_ns=30, cpu_ns=40, now_ns=complete_ns)
    world.put("a", 2, dispatch_ns)

    world.dispatch(dispatch_ns)

    assert (world.scheduler.stats.paced_ns, world.scheduler.stats.idle_ns) == (paced_ns, idle_ns)


def test_a_share_applies_from_the_next_complete_and_a_release_already_set_stands() -> None:
    # The quarter core set while the first batch is at the worker paces it: its 40 ns of CPU
    # release the worker 160 ns after its dispatch at 100. The share removed after that
    # complete() leaves that release in place and paces nothing after it.
    world = World()
    world.scheduler.set_pacing_share(0.5)
    world.connect("a", 0)
    world.put("a", 1, 100)
    first = world.dispatch(100)
    world.scheduler.set_pacing_share(0.25)
    world.scheduler.complete(first, (), worker_ns=30, cpu_ns=40, now_ns=150)
    world.scheduler.set_pacing_share(None)
    world.put("a", 2, 200)

    held = world.scheduler.next(200)
    second = world.dispatch(260)
    world.scheduler.complete(second, (), worker_ns=30, cpu_ns=40, now_ns=300)
    world.put("a", 3, 300)

    assert held == WaitUntil(260)
    assert isinstance(world.scheduler.next(300), Dispatch)


@pytest.mark.parametrize("share_cores", [0.0, -0.5, math.nan, math.inf])
def test_a_pacing_share_that_is_not_a_finite_number_above_zero_is_refused(
    share_cores: float,
) -> None:
    with pytest.raises(ValueError, match="pacing share"):
        World().scheduler.set_pacing_share(share_cores)


def test_next_raises_while_a_batch_is_in_flight() -> None:
    world = World()
    world.connect("a", 0)
    world.put("a", 1, 5)
    batch = world.dispatch(10)
    world.put("a", 2, 15)

    with pytest.raises(RuntimeError, match="is in flight"):
        world.scheduler.next(20)

    assert world.scheduler.in_flight is batch


def test_dispatches_number_their_batches_and_book_sizes_frames_and_worker_time() -> None:
    world = World()
    for robot_id in "abc":
        world.connect(robot_id, 0)
        world.put(robot_id, 1, 5)

    first = world.dispatch(10)
    world.scheduler.complete(first, (), worker_ns=2 * NS_PER_MS, cpu_ns=1, now_ns=20)
    second = world.dispatch(20)
    world.scheduler.complete(second, (), worker_ns=3 * NS_PER_MS, cpu_ns=1, now_ns=30)

    stats = world.scheduler.stats
    assert [(batch.batch_id, batch.dispatched_ns) for batch in (first, second)] == [
        (1, 10),
        (2, 20),
    ]
    assert (stats.batches, stats.frames, stats.batch_size) == (2, 3, {2: 1, 1: 1})
    assert (stats.busy_ns, stats.batch_ms.count, stats.batch_ms.sum_ms) == (5 * NS_PER_MS, 2, 5.0)
    assert world.scheduler.in_flight is None


def test_complete_delivers_and_counts_failed_disconnected_and_replaced_items_once_each() -> None:
    world = World(replace(BASE, batch_cap=5))
    for robot_id in "abcde":
        world.connect(robot_id, 0)
        world.put(robot_id, 1, 5)
    batch = world.dispatch(10)
    # While the batch is at the worker, c leaves and d connects again; b's frame then fails.
    world.disconnect("c", 12)
    world.disconnect("d", 12)
    world.connect("d", 13)

    deliverable = world.scheduler.complete(batch, [1], worker_ns=5, cpu_ns=5, now_ns=20)

    assert (served_ids(batch), deliverable) == (["a", "b", "c", "d", "e"], (0, 4))
    assert {
        robot_id: (
            world.counters(robot_id).in_flight,
            world.counters(robot_id).frames_failed,
            world.counters(robot_id).results_dropped_disconnect,
        )
        for robot_id in "abcde"
    } == {"a": (0, 0, 0), "b": (0, 1, 0), "c": (0, 0, 1), "d": (0, 0, 1), "e": (0, 0, 0)}


def test_an_item_both_failed_and_disconnected_counts_once_as_failed() -> None:
    world = World()
    world.connect("a", 0)
    world.put("a", 1, 5)
    batch = world.dispatch(10)
    world.disconnect("a", 12)

    deliverable = world.scheduler.complete(batch, [0], worker_ns=5, cpu_ns=5, now_ns=20)

    counters = world.counters("a")
    assert (deliverable, counters.in_flight) == ((), 0)
    assert (counters.frames_failed, counters.results_dropped_disconnect) == (1, 0)


@pytest.mark.parametrize(
    ("after_the_drop", "counters_in_the_registry"),
    [
        pytest.param(
            lambda world: None,
            {"b": ModelCounters(last_command_ns=21 + RETENTION_NS)},
            id="id-unknown",
        ),
        pytest.param(
            lambda world: world.connect("a", 22 + RETENTION_NS),
            {
                "b": ModelCounters(last_command_ns=21 + RETENTION_NS),
                "a": ModelCounters(last_command_ns=22 + RETENTION_NS),
            },
            id="id-back-as-new-record",
        ),
    ],
)
@pytest.mark.parametrize(
    ("finish", "result", "outcome"),
    [(complete_without_failures, (), (0, 1, 0)), (fail_batch, None, (0, 0, 1))],
    ids=["complete", "fail"],
)
def test_a_result_for_a_record_the_registry_dropped_counts_on_that_record_alone(
    finish: Callable[[ModelScheduler, Batch], object],
    result: object,
    outcome: tuple[int, int, int],
    after_the_drop: Callable[[World], object],
    counters_in_the_registry: dict[str, ModelCounters],
) -> None:
    world = World()
    old = world.connect("a", 0)
    world.put("a", 1, 5)
    batch = world.dispatch(10)
    world.disconnect("a", 20)
    # The next connect drops the record of a, which has been gone longer than the retention.
    world.connect("b", 21 + RETENTION_NS)
    after_the_drop(world)

    returned = finish(world.scheduler, batch)

    counters = old.models[MODEL]
    assert returned == result
    assert (
        counters.in_flight,
        counters.results_dropped_disconnect,
        counters.frames_failed,
    ) == outcome
    assert {
        record.robot_id: record.models[MODEL] for record in world.registry.records()
    } == counters_in_the_registry


def test_fail_counts_every_item_failed_and_sets_no_pacing_hold() -> None:
    world = World()
    world.scheduler.set_pacing_share(0.25)
    for robot_id in "ab":
        world.connect(robot_id, 0)
        world.put(robot_id, 1, 5)
    batch = world.dispatch(10)

    world.scheduler.fail(batch, 20)
    world.put("a", 2, 30)

    assert isinstance(world.scheduler.next(30), Dispatch)
    assert [
        (world.counters(robot_id).frames_failed, world.counters(robot_id).frames_served)
        for robot_id in "ab"
    ] == [(1, 2), (1, 1)]


def test_the_time_from_a_failure_to_the_next_call_is_not_booked() -> None:
    world = World()
    world.connect("a", 0)
    world.put("a", 1, 10)
    batch = world.dispatch(10)
    world.scheduler.fail(batch, 20)

    restarted = world.scheduler.next(500)
    world.put("a", 2, 520)
    world.dispatch(520)

    assert restarted == Idle()
    assert (world.scheduler.stats.idle_ns, world.scheduler.stats.paced_ns) == (10 + 20, 0)


@pytest.mark.parametrize(
    "finish", [complete_without_failures, fail_batch], ids=["complete", "fail"]
)
def test_complete_and_fail_refuse_a_batch_that_is_not_in_flight(
    finish: Callable[[ModelScheduler, Batch], object],
) -> None:
    world = World()
    world.connect("a", 0)
    world.put("a", 1, 5)
    batch = world.dispatch(10)
    world.scheduler.fail(batch, 20)

    with pytest.raises(ValueError, match="not the batch in flight"):
        finish(world.scheduler, batch)


@pytest.mark.parametrize(
    "finish", [complete_without_failures, fail_batch], ids=["complete", "fail"]
)
def test_complete_and_fail_refuse_an_earlier_batch_and_leave_the_one_in_flight_alone(
    finish: Callable[[ModelScheduler, Batch], object],
) -> None:
    world = World()
    world.connect("a", 0)
    world.put("a", 1, 5)
    old = world.dispatch(10)
    world.scheduler.fail(old, 20)
    world.put("a", 2, 30)
    new = world.dispatch(30)

    with pytest.raises(ValueError, match="not the batch in flight"):
        finish(world.scheduler, old)

    assert world.scheduler.in_flight is new
    assert (world.counters("a").in_flight, world.counters("a").frames_failed) == (1, 1)


@pytest.mark.parametrize("failed", [[-1], [2], [0, 2]])
def test_complete_refuses_a_failed_index_outside_the_batch_and_changes_nothing(
    failed: list[int],
) -> None:
    world = World()
    for robot_id in "ab":
        world.connect(robot_id, 0)
        world.put(robot_id, 1, 5)
    batch = world.dispatch(10)

    with pytest.raises(ValueError, match="does not have"):
        world.scheduler.complete(batch, failed, worker_ns=5, cpu_ns=5, now_ns=20)

    assert world.scheduler.in_flight is batch
    assert [world.counters(robot_id).in_flight for robot_id in "ab"] == [1, 1]
