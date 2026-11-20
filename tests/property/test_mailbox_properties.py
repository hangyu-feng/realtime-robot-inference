"""Properties of a frame's way through the scheduling core: conservation, newest first, order.

Each test runs the state machine of scheduling_machine.py, which interleaves every step a caller
takes, and checks its property after every step.
"""

from collections import Counter

from hypothesis import strategies as st
from hypothesis.stateful import invariant, precondition, rule, run_state_machine_as_test

from cloud.config import QueuePolicy
from tests.property.scheduling_machine import (
    DECLARATIONS,
    MACHINE_SETTINGS,
    ROBOT_IDS,
    SchedulingMachine,
    World,
    core_setups,
    dispatches,
)


def identity_gaps(world: World) -> dict[tuple[str, int, str], tuple[int, int]]:
    """Return each record and model whose identities fail, with the excess of each left side.

    The first identity: frames_received = frames_served + frames_overwritten + frames_skipped
    + frames_dropped_disconnect + frames_pending, where frames_pending is Mailbox.pending_count
    of the record's id and model, which is 0 for an id that is not open. The second:
    frames_served = commands_sent + commands_queued + commands_dropped_downlink
    + results_dropped_disconnect + frames_failed + in_flight. The gauge does not describe a
    record the registry dropped whose id is open again with a new record: it counts the new
    record's frames, and the dropped one holds none.
    """
    gaps: dict[tuple[str, int, str], tuple[int, int]] = {}
    for record in world.records.values():
        connection = world.connections.get(record.robot_id)
        superseded = connection is not None and connection.record is not record
        for model, counters in record.models.items():
            frames_pending = (
                0 if superseded else world.mailbox.pending_count(record.robot_id, model)
            )
            gaps[(record.robot_id, record.session, model)] = (
                counters.frames_received
                - counters.frames_served
                - counters.frames_overwritten
                - counters.frames_skipped
                - counters.frames_dropped_disconnect
                - frames_pending,
                counters.frames_served
                - counters.commands_sent
                - counters.commands_queued
                - counters.commands_dropped_downlink
                - counters.results_dropped_disconnect
                - counters.frames_failed
                - counters.in_flight,
            )
    return {key: gap for key, gap in gaps.items() if gap != (0, 0)}


def gauge_errors(world: World) -> dict[tuple[str, int], tuple[object, object]]:
    """Return each record whose in_flight or commands_queued gauges miscount, with both counts.

    in_flight counts the record's items in the batches at the workers, and commands_queued,
    summed over its models, the commands waiting in the downlink queue of its open session. The
    identities hold only if each gauge counts what it names.
    """
    at_workers = Counter(
        (id(item.record), model) for model, batch in world.in_flight.items() for item in batch.items
    )
    errors: dict[tuple[str, int], tuple[object, object]] = {}
    for record in world.records.values():
        connection = world.connections.get(record.robot_id)
        gauges = (
            {model: counters.in_flight for model, counters in record.models.items()},
            sum(counters.commands_queued for counters in record.models.values()),
        )
        counted = (
            {model: at_workers[(id(record), model)] for model in record.models},
            # The id of a record the registry dropped can be open again with a new record,
            # whose queue holds none of this record's commands.
            len(connection.downlink)
            if connection is not None and connection.record is record
            else 0,
        )
        errors[(record.robot_id, record.session)] = (gauges, counted)
    return {key: error for key, error in errors.items() if error[0] != error[1]}


def misdelivered(world: World) -> list[tuple[str, int, tuple[int, ...], tuple[int, ...]]]:
    """Return each batch whose results were not delivered to exactly the items due a command.

    An item is due one when its frame did not fail and its session was still open when the batch
    left its worker. An item of a session that had ended, its record dropped or not, gets none.
    """
    deliveries = [
        (
            finish.model,
            finish.batch.batch_id,
            finish.delivered,
            tuple(
                index
                for index, item in enumerate(finish.batch.items)
                if index not in finish.failed and item.session in finish.open_sessions
            ),
        )
        for finish in world.finishes
    ]
    return [delivery for delivery in deliveries if delivery[2] != delivery[3]]


def strangers_touched(world: World) -> list[tuple[str, int, list[str]]]:
    """Return each batch whose complete() or fail() changed a record that none of its items holds.

    That covers a new record of the id of an item whose record the registry dropped.
    """
    touched = [
        (
            finish.model,
            finish.batch.batch_id,
            [
                record.robot_id
                for record in finish.touched
                if all(record is not item.record for item in finish.batch.items)
            ],
        )
        for finish in world.finishes
    ]
    return [entry for entry in touched if entry[2]]


class ConservationMachine(SchedulingMachine):
    """Checks frame conservation under both queue policies, with records dropped mid-batch.

    Five robot ids share a table of three or four entries with a short retention time, and a
    robot whose frame is at a worker can leave for longer than the retention time, so the
    registry drops records while their frames are at a worker. The skip rule is on wherever the
    configuration allows it, under QueuePolicy.LATEST, so that skipped frames are counted too:
    a run in which it skips nothing serves frames as one with the rule off does.
    """

    SETUPS = core_setups(skip_rules=st.just(True))

    @precondition(
        lambda self: any(
            item.record.connected for batch in self.world.in_flight.values() for item in batch.items
        )
    )
    @rule(data=st.data())
    def robot_at_a_worker_leaves_until_its_record_expires(self, data: st.DataObject) -> None:
        """Close a robot whose frame is at a worker, and say a hello past the retention time.

        The hello comes from the same id, which gets a new record, or from another id, after
        which the robot's id is unknown. Either way connect() drops the robot's record while
        its frame is still at the worker.
        """
        robot_ids = {
            item.record.robot_id
            for batch in self.world.in_flight.values()
            for item in batch.items
            if item.record.connected
        }
        robot_id = data.draw(st.sampled_from(sorted(robot_ids)), label="robot_id")
        self.world.close(robot_id, self.now_ns)
        self.now_ns += self.world.setup.retention_ns + 1
        hello_ids = [
            robot_id,
            *(other for other in ROBOT_IDS if other not in (robot_id, *self.world.connections)),
        ]
        hello_id = data.draw(st.sampled_from(hello_ids), label="hello_id")
        self.world.hello(hello_id, data.draw(DECLARATIONS, label="stale_ns"), self.now_ns)

    @invariant()
    def both_identities_hold_for_every_record(self) -> None:
        assert identity_gaps(self.world) == {}
        assert gauge_errors(self.world) == {}

    @invariant()
    def results_reach_open_sessions_alone_and_touch_only_their_records(self) -> None:
        assert misdelivered(self.world) == []
        assert strangers_touched(self.world) == []


def test_every_frame_is_counted_once_after_every_step_even_when_its_record_is_dropped() -> None:
    run_state_machine_as_test(ConservationMachine, settings=MACHINE_SETTINGS)


def stale_serves(world: World) -> list[tuple[str, str, int, int]]:
    """Return each frame served that was not the newest its robot had put in its session."""
    return [
        (
            model,
            item.record.robot_id,
            item.frame.frame_number,
            decided.newest[item.session].frame_number,
        )
        for model in world.schedulers
        for decided, batch in dispatches(world, model)
        for item in batch.items
        if item.frame is not decided.newest[item.session]
    ]


class LatestMachine(SchedulingMachine):
    """Checks under QueuePolicy.LATEST that a model serves a robot's newest frame."""

    SETUPS = core_setups(policies=st.just(QueuePolicy.LATEST))

    @invariant()
    def every_frame_served_was_the_newest(self) -> None:
        assert stale_serves(self.world) == []


def test_under_latest_a_model_serves_the_newest_frame_its_robot_has_put() -> None:
    run_state_machine_as_test(LatestMachine, settings=MACHINE_SETTINGS)


def order_breaks(world: World) -> list[tuple[str, int, int, int]]:
    """Return each frame a model served whose number was not above the last it served that session.

    Session numbers are never used twice, so a model and a session name one robot's session.
    """
    breaks = []
    for model in world.schedulers:
        last_served: dict[int, int] = {}
        for _, batch in dispatches(world, model):
            for item in batch.items:
                previous = last_served.get(item.session)
                if previous is not None and item.frame.frame_number <= previous:
                    breaks.append((model, item.session, previous, item.frame.frame_number))
                last_served[item.session] = item.frame.frame_number
    return breaks


def served_twice(world: World) -> list[tuple[str, int, int]]:
    """Return each frame that one model served more than once.

    A session and a frame number name one frame, because numbers rise within a session.
    """
    served = Counter(
        (model, item.session, item.frame.frame_number)
        for model in world.schedulers
        for _, batch in dispatches(world, model)
        for item in batch.items
    )
    return [frame for frame, count in served.items() if count > 1]


class OrderMachine(SchedulingMachine):
    """Checks under both queue policies the order in which a model serves a robot's frames."""

    SETUPS = core_setups()

    @invariant()
    def frames_are_served_in_order_and_once(self) -> None:
        assert order_breaks(self.world) == []
        assert served_twice(self.world) == []


def test_served_frame_numbers_rise_per_model_and_session_and_no_frame_is_served_twice() -> None:
    run_state_machine_as_test(OrderMachine, settings=MACHINE_SETTINGS)
