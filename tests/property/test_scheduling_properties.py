"""Properties of the per-model scheduler: fairness, the longest gap, batches and pacing.

The fairness, faster-sender, reconnect and longest-gap properties run robots that send one frame
per period, with next() called exactly when the gateway calls it: those properties need robots
that keep sending, and more of them pending than a batch takes. The batch, hold and pacing
properties run the state machine of scheduling_machine.py, which interleaves every step a caller
takes and also calls next() at times the gateway would not.
"""

import itertools
import math
from collections.abc import Iterable, Sequence
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.stateful import invariant, precondition, rule, run_state_machine_as_test

from cloud.registry import RobotRecord
from cloud.scheduler import Batch, Decision, Dispatch, Idle, WaitUntil
from tests.property.scheduling_machine import (
    MACHINE_SETTINGS,
    SHARES_CORES,
    Scenario,
    SchedulingMachine,
    World,
    churning_scenarios,
    core_setups,
    dispatches,
    faster_sender_scenarios,
    periodic_scenarios,
    simulate,
)


def spread_breaks(
    stretches: Iterable[tuple[Sequence[RobotRecord], Batch]],
) -> list[tuple[str, str]]:
    """Return the robot ids of each two robots whose service counts drift more than one apart.

    stretches gives, for each dispatch in order, the robots it considers and its batch. For each
    two robots, the difference of their service counts is followed over every run of
    consecutive dispatches that consider both. Their counts over any stretch of such a run differ
    by the change of the difference across it, so they differ by at most one over every stretch
    exactly when the difference stays within a range of one. Robots are told apart by record.
    """
    breaks = []
    # For each two records, by identity: their difference, and its lowest and highest values.
    runs: dict[tuple[int, int], tuple[int, int, int]] = {}
    for considered, batch in stretches:
        served = {id(item.record) for item in batch.items}
        continued = {}
        for first, second in itertools.combinations(sorted(considered, key=id), 2):
            key = (id(first), id(second))
            difference, lowest, highest = runs.get(key, (0, 0, 0))
            difference += (key[0] in served) - (key[1] in served)
            continued[key] = (difference, min(lowest, difference), max(highest, difference))
            if max(highest, difference) - min(lowest, difference) > 1:
                breaks.append((first.robot_id, second.robot_id))
        runs = continued
    return breaks


def churn_end_ns(scenario: Scenario) -> int:
    """Return when a run of churning_scenarios() ends: after every robot has come back.

    Robots leave and come back within 36 of the shortest frame periods.
    """
    return 40 * min(sender.period_ns for sender in scenario.senders)


@given(scenario=churning_scenarios())
def test_robots_pending_throughout_a_stretch_are_served_within_one_of_each_other(
    scenario: Scenario,
) -> None:
    world = simulate(scenario, churn_end_ns(scenario))

    breaks = spread_breaks(
        (decided.pending, batch) for decided, batch in dispatches(world, scenario.config.model)
    )

    assert breaks == []


def stretches_with_everyone_pending(
    world: World, model: str
) -> list[tuple[Sequence[RobotRecord], Batch]]:
    """Return each dispatch with the robots it considers: all when all have a frame pending.

    A dispatch at which some robot of the run has no frame pending considers none, which ends
    every run.
    """
    return [
        (decided.pending if len(decided.pending) == len(world.records) else (), batch)
        for decided, batch in dispatches(world, model)
    ]


@given(scenario=faster_sender_scenarios(), dispatch_count=st.integers(10, 30))
def test_a_robot_sending_two_or_four_times_as_fast_is_served_as_often_as_the_others(
    scenario: Scenario, dispatch_count: int
) -> None:
    # No batch takes more than two of the others' frame periods, so the run holds at least
    # dispatch_count dispatches.
    period_ns = max(sender.period_ns for sender in scenario.senders)
    world = simulate(scenario, dispatch_count * 2 * period_ns)
    faster = min(scenario.senders, key=lambda sender: sender.period_ns).robot_id

    breaks = spread_breaks(stretches_with_everyone_pending(world, scenario.config.model))

    assert [pair for pair in breaks if faster in pair] == []


def reconnect_overtakes(world: World, model: str) -> list[tuple[int, str, list[str]]]:
    """Return each dispatch that served a robot back from a reconnect ahead of a longer waiter.

    A robot is back from a reconnect at its first service in a later session than its last one,
    its record kept. A robot that has waited longer is a pending one left out of the batch that
    the model never served or last served before that robot. Services are ordered by dispatch
    and, within a batch, by position, kept per record across its sessions.
    """
    overtakes = []
    # For each record, by identity: the dispatch index and position of its last service, and
    # the session it was in.
    last_service: dict[int, tuple[int, int, int]] = {}
    for index, (decided, batch) in enumerate(dispatches(world, model)):
        served = {id(item.record) for item in batch.items}
        left_out = [record for record in decided.pending if id(record) not in served]
        for item in batch.items:
            previous = last_service.get(id(item.record))
            if previous is not None and previous[2] < item.session:
                waited_longer = [
                    record.robot_id
                    for record in left_out
                    if last_service.get(id(record), (-1, -1, 0))[:2] < previous[:2]
                ]
                overtakes.append((decided.now_ns, item.record.robot_id, waited_longer))
        for position, item in enumerate(batch.items):
            last_service[id(item.record)] = (index, position, item.session)
    return [overtake for overtake in overtakes if overtake[2]]


@given(scenario=churning_scenarios())
def test_a_robot_that_reconnects_is_not_served_ahead_of_robots_that_have_waited_longer(
    scenario: Scenario,
) -> None:
    world = simulate(scenario, churn_end_ns(scenario))

    assert reconnect_overtakes(world, scenario.config.model) == []


def gap_bound_ns(scenario: Scenario) -> int:
    """Return tau + (ceil(N / batch_cap) + 1) * C, the longest-gap bound of DESIGN.md 6.3.

    tau is the frame period, and N counts every robot of the scenario, all connected at 0. C is
    the longest time from a dispatch until the worker can take its next batch: the batch's wall
    time, or with a pacing share the batch's CPU time over the share, in exact arithmetic, when
    that is longer.
    """
    period_ns = max(sender.period_ns for sender in scenario.senders)
    cycle_ns = max(
        max(
            worker_ns,
            0
            if scenario.share_cores is None
            else math.ceil(Fraction(cpu_ns) / Fraction(scenario.share_cores)),
        )
        for worker_ns, cpu_ns in scenario.batch_times_ns
    )
    rounds = math.ceil(len(scenario.senders) / scenario.config.batch_cap) + 1
    return period_ns + rounds * cycle_ns


def longest_waits_ns(world: World, end_ns: int) -> dict[str, int]:
    """Return, per robot, the longest time it went without a dispatch that included it.

    Every robot stays connected for the whole run. The times run from its connect to its first
    dispatch, between consecutive dispatches, and from its last dispatch to end_ns, the end of
    the run: a robot left without a dispatch for longer than a bound before the end would show
    it there.
    """
    times_ns = {
        robot_id: [connection.record.connected_ns]
        for robot_id, connection in world.connections.items()
    }
    for model in world.schedulers:
        for decided, batch in dispatches(world, model):
            for item in batch.items:
                times_ns[item.record.robot_id].append(decided.now_ns)
    return {
        robot_id: max(
            later_ns - earlier_ns
            for earlier_ns, later_ns in itertools.pairwise([*robot_times_ns, end_ns])
        )
        for robot_id, robot_times_ns in times_ns.items()
    }


@given(scenario=periodic_scenarios(skip_rule=False), rounds=st.integers(2, 3))
def test_with_the_skip_rule_off_no_robot_waits_longer_than_the_gap_bound_for_a_dispatch(
    scenario: Scenario, rounds: int
) -> None:
    bound_ns = gap_bound_ns(scenario)
    end_ns = rounds * bound_ns

    waits_ns = longest_waits_ns(simulate(scenario, end_ns), end_ns)

    assert {robot_id: wait_ns for robot_id, wait_ns in waits_ns.items() if wait_ns > bound_ns} == {}


@given(scenario=periodic_scenarios(skip_rule=True), rounds=st.integers(2, 3))
def test_with_the_skip_rule_on_no_robot_waits_longer_than_its_interval_plus_the_gap_bound(
    scenario: Scenario, rounds: int
) -> None:
    # Once a robot has gone skip_max_unserved_ns unserved, guard (b) keeps its frames from
    # being skipped, and the bound of the skip rule off runs from there.
    bound_ns = scenario.config.skip_max_unserved_ns + gap_bound_ns(scenario)
    end_ns = rounds * bound_ns

    waits_ns = longest_waits_ns(simulate(scenario, end_ns), end_ns)

    assert {robot_id: wait_ns for robot_id, wait_ns in waits_ns.items() if wait_ns > bound_ns} == {}


def oversized_or_repeating(world: World) -> list[tuple[str, int, list[str]]]:
    """Return each batch larger than its model's cap or holding one robot twice."""
    caps = {setup.config.model: setup.config.batch_cap for setup in world.setup.models}
    batches = [
        (model, batch.batch_id, [item.record.robot_id for item in batch.items])
        for model in world.schedulers
        for _, batch in dispatches(world, model)
    ]
    return [
        (model, batch_id, robot_ids)
        for model, batch_id, robot_ids in batches
        if len(robot_ids) > caps[model] or len(set(robot_ids)) < len(robot_ids)
    ]


class OneBatchMachine(SchedulingMachine):
    """Checks that a worker holds one batch at a time, of at most the cap, each robot once."""

    SETUPS = core_setups()

    @precondition(lambda self: bool(self.world.in_flight))
    @rule(data=st.data())
    def busy_worker_asks(self, data: st.DataObject) -> None:
        model = data.draw(st.sampled_from(sorted(self.world.in_flight)), label="model")
        scheduler = self.world.schedulers[model]

        with pytest.raises(RuntimeError):
            scheduler.next(self.now_ns)

        assert scheduler.in_flight is self.world.in_flight[model]

    @invariant()
    def each_scheduler_holds_the_batch_its_worker_holds(self) -> None:
        mismatched = [
            model
            for model, scheduler in self.world.schedulers.items()
            if scheduler.in_flight is not self.world.in_flight.get(model)
        ]
        assert mismatched == []

    @invariant()
    def every_batch_holds_at_most_the_cap_and_each_robot_once(self) -> None:
        assert oversized_or_repeating(self.world) == []


def test_a_worker_holds_one_batch_at_a_time_of_at_most_the_cap_with_each_robot_once() -> None:
    run_state_machine_as_test(OneBatchMachine, settings=MACHINE_SETTINGS)


def hold_breaks(world: World) -> list[tuple[str, int, Decision]]:
    """Return each call of next() by a paced worker that did not keep to the hold in force.

    complete() holds the worker from its batch's dispatch until the batch's CPU time over the
    share has passed, rounded up to a whole nanosecond in exact arithmetic. fail() is given no
    CPU time, so a failed batch sets no hold, and nothing holds the worker before its first
    batch. A call before the hold ends must return WaitUntil at its end, and every other call
    something other than a WaitUntil. A worker asks only once its last batch is back, so the
    hold in force at a call is the one its last batch set.
    """
    shares_cores = {
        setup.config.model: Fraction(setup.share_cores)
        for setup in world.setup.models
        if setup.share_cores is not None
    }
    breaks = []
    for model, share_cores in shares_cores.items():
        # By batch id, the end of the hold the batch set when it came back. Batch 0 stands for
        # the start, where the worker is free from time 0.
        releases_ns = {0: 0} | {
            finish.batch.batch_id: finish.batch.dispatched_ns
            + math.ceil(finish.cpu_ns / share_cores)
            for finish in world.finishes
            if finish.model == model
        }
        last_batch_id = 0
        for decided in (decided for decided in world.decisions if decided.model == model):
            release_ns = releases_ns[last_batch_id]
            match decided.decision:
                case WaitUntil(deadline_ns):
                    kept = decided.now_ns < release_ns and deadline_ns == release_ns
                case Dispatch(batch):
                    kept = decided.now_ns >= release_ns
                    last_batch_id = batch.batch_id
                case Idle():
                    kept = decided.now_ns >= release_ns
            if not kept:
                breaks.append((model, decided.now_ns, decided.decision))
    return breaks


class HoldMachine(SchedulingMachine):
    """Checks with both workers paced that every call of next() keeps to the hold in force."""

    SETUPS = core_setups(shares_cores=SHARES_CORES)

    @precondition(
        lambda self: any(
            deadline_ns > self.now_ns for deadline_ns in self.hold_deadlines_ns().values()
        )
    )
    @rule(data=st.data())
    def held_worker_asks_just_before_its_deadline(self, data: st.DataObject) -> None:
        """Let a held worker ask one nanosecond before its deadline, where the hold must stand."""
        deadlines_ns = {
            model: deadline_ns
            for model, deadline_ns in self.hold_deadlines_ns().items()
            if deadline_ns > self.now_ns
        }
        model = data.draw(st.sampled_from(sorted(deadlines_ns)), label="model")
        self.now_ns = deadlines_ns[model] - 1
        self.world.decide(model, self.now_ns)

    @invariant()
    def every_call_keeps_to_the_hold_in_force(self) -> None:
        assert hold_breaks(self.world) == []


def test_a_pacing_hold_ends_at_its_deadline_which_lies_after_the_call() -> None:
    run_state_machine_as_test(HoldMachine, settings=MACHINE_SETTINGS)


def pacing_excesses(world: World) -> list[tuple[str, int, int]]:
    """Return each two dispatches of a paced model between which its batches overran the share.

    Between dispatches i < j the CPU time of batches i to j - 1 may be at most
    share * (dispatched_j - dispatched_i). The share is the exact value of its float, as the
    fraction numerator / denominator, so the comparison runs in integers. fail() is given no CPU
    time, so a failed batch counts as none.
    """
    cpu_ns = {(finish.model, finish.batch.batch_id): finish.cpu_ns for finish in world.finishes}
    shares_cores = {
        setup.config.model: setup.share_cores.as_integer_ratio()
        for setup in world.setup.models
        if setup.share_cores is not None
    }
    excesses = []
    for model, (numerator, denominator) in shares_cores.items():
        model_dispatches = dispatches(world, model)
        times_ns = [decided.now_ns for decided, _ in model_dispatches]
        # Every batch but the last has finished, because the next one was dispatched after it.
        used_ns = list(
            itertools.accumulate(
                (cpu_ns[(model, batch.batch_id)] for _, batch in model_dispatches[:-1]), initial=0
            )
        )
        excesses += [
            (model, times_ns[first], times_ns[last])
            for first, last in itertools.combinations(range(len(times_ns)), 2)
            if (used_ns[last] - used_ns[first]) * denominator
            > numerator * (times_ns[last] - times_ns[first])
        ]
    return excesses


class PacingMachine(SchedulingMachine):
    """Checks with both workers paced that each keeps to its share."""

    SETUPS = core_setups(shares_cores=SHARES_CORES)

    @invariant()
    def every_paced_worker_keeps_to_its_share(self) -> None:
        assert pacing_excesses(self.world) == []


def test_a_paced_worker_uses_at_most_its_share_between_any_two_dispatches() -> None:
    run_state_machine_as_test(PacingMachine, settings=MACHINE_SETTINGS)
