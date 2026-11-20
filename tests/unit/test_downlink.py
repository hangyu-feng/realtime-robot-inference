"""Tests for the downlink queue: its bound, the service gap and the second conservation identity."""

from collections.abc import Callable, Collection

import pytest

from cloud.config import QueuePolicy
from cloud.downlink import DownlinkQueue, QueuedCommand
from cloud.mailbox import Frame, Mailbox
from cloud.registry import ModelCounters, RobotRegistry
from cloud.scheduler import Batch, Dispatch, ModelScheduler, SchedulerConfig

# Times are in nanoseconds on the gateway's clock.
MODEL = "cls"
FRAME_PERIOD_NS = 40
RETENTION_NS = 10_000
MAX_ENTRIES = 8


class Gateway:
    """The registry, the mailbox, one scheduler and a downlink queue per connection.

    Each method is one synchronous step of the gateway, with the open and close sequences of
    the registry's docstring, and a completion that pushes the command of every item that
    complete() returns.
    """

    def __init__(self, downlink_queue_commands: int) -> None:
        self.registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
        self.mailbox = Mailbox(QueuePolicy.LATEST, fifo_capacity_frames=1)
        config = SchedulerConfig(
            model=MODEL,
            batch_cap=2,
            service_ns=10,
            return_allowance_ns=5,
            skip_rule=False,
            skip_max_unserved_ns=1_000,
        )
        self.scheduler = ModelScheduler(config, self.mailbox, 0)
        self.downlink_queue_commands = downlink_queue_commands
        self.downlinks: dict[str, DownlinkQueue] = {}
        self.batches: list[Batch] = []

    def open(self, robot_id: str, now_ns: int) -> None:
        record = self.registry.connect(
            robot_id, (MODEL,), {MODEL: 100}, {MODEL: 200}, FRAME_PERIOD_NS, now_ns
        )
        self.mailbox.open(record)
        self.downlinks[robot_id] = DownlinkQueue(self.downlink_queue_commands)

    def close(self, robot_id: str, now_ns: int) -> None:
        self.mailbox.close(robot_id)
        self.downlinks[robot_id].drop_all()
        self.registry.disconnect(robot_id, now_ns)

    def put(self, robot_id: str, frame_number: int, now_ns: int) -> None:
        self.mailbox.put(robot_id, Frame(frame_number, now_ns - 7, now_ns, 2, 5, b"\xff\xd8"))

    def dispatch(self, now_ns: int) -> None:
        decision = self.scheduler.next(now_ns)
        assert isinstance(decision, Dispatch)
        self.batches.append(decision.batch)

    def complete(self, failed: Collection[int], now_ns: int) -> None:
        batch = self.batches[-1]
        for index in self.scheduler.complete(batch, failed, 1, 1, now_ns):
            record = batch.items[index].record
            self.downlinks[record.robot_id].push(QueuedCommand(b"command", record.models[MODEL]))

    def fail(self, now_ns: int) -> None:
        self.scheduler.fail(self.batches[-1], now_ns)

    def pop(self, robot_id: str, now_ns: int) -> None:
        self.downlinks[robot_id].pop(now_ns)


def second_identity_gaps(registry: RobotRegistry) -> dict[str, int]:
    """Return every robot whose second identity fails, with served minus the rest."""
    gaps = {
        record.robot_id: counters.frames_served
        - counters.commands_sent
        - counters.commands_queued
        - counters.commands_dropped_downlink
        - counters.results_dropped_disconnect
        - counters.frames_failed
        - counters.in_flight
        for record in registry.records()
        for counters in record.models.values()
    }
    return {robot_id: gap for robot_id, gap in gaps.items() if gap != 0}


def test_a_full_queue_drops_its_oldest_command_as_dropped_downlink() -> None:
    commands = [
        QueuedCommand(bytes([index]), ModelCounters(last_command_ns=0)) for index in range(3)
    ]
    queue = DownlinkQueue(capacity=2)

    for command in commands:
        queue.push(command)

    assert len(queue) == 2
    assert [
        (command.counters.commands_queued, command.counters.commands_dropped_downlink)
        for command in commands
    ] == [(0, 1), (1, 0), (1, 0)]
    assert [queue.pop(10), queue.pop(20)] == commands[1:]


def test_pop_returns_none_when_the_queue_is_empty() -> None:
    assert DownlinkQueue(capacity=1).pop(0) is None


def test_pop_counts_a_sent_command_and_the_longest_gap_since_the_session_start() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = registry.connect("a", (MODEL,), {MODEL: 100}, {MODEL: 200}, FRAME_PERIOD_NS, 100)
    counters = record.models[MODEL]
    queue = DownlinkQueue(capacity=4)

    gaps_ns = []
    for pop_ns in (250, 300, 600):
        queue.push(QueuedCommand(b"command", counters))
        queue.pop(pop_ns)
        gaps_ns.append(counters.max_service_gap_ns)

    assert gaps_ns == [150, 150, 300]
    assert (counters.commands_sent, counters.commands_queued, counters.last_command_ns) == (
        3,
        0,
        600,
    )


def test_pop_refuses_a_command_of_a_model_with_no_session_start_and_keeps_it() -> None:
    counters = ModelCounters()
    queue = DownlinkQueue(capacity=2)
    queue.push(QueuedCommand(b"command", counters))

    with pytest.raises(ValueError, match="no session start"):
        queue.pop(5)

    assert (len(queue), counters.commands_queued, counters.commands_sent) == (1, 1, 0)


def test_drop_all_counts_every_queued_command_as_a_result_dropped_at_disconnect() -> None:
    cls_counters = ModelCounters(last_command_ns=0)
    depth_counters = ModelCounters(last_command_ns=0)
    queue = DownlinkQueue(capacity=4)
    for counters in (cls_counters, depth_counters, cls_counters):
        queue.push(QueuedCommand(b"command", counters))

    queue.drop_all()

    assert len(queue) == 0
    assert [
        (
            counters.commands_queued,
            counters.results_dropped_disconnect,
            counters.commands_dropped_downlink,
        )
        for counters in (cls_counters, depth_counters)
    ] == [(0, 2, 0), (0, 1, 0)]


def test_the_second_identity_holds_after_complete_and_push_and_after_every_other_call() -> None:
    gateway = Gateway(downlink_queue_commands=1)
    steps: list[Callable[[], None]] = [
        lambda: gateway.open("a", 0),
        lambda: gateway.open("b", 0),
        lambda: gateway.put("a", 1, 5),
        lambda: gateway.put("b", 1, 5),
        lambda: gateway.dispatch(10),
        lambda: gateway.pop("a", 12),
        # a is delivered and its command waits; b's frame failed.
        lambda: gateway.complete([1], 20),
        lambda: gateway.put("a", 2, 25),
        lambda: gateway.put("b", 2, 25),
        lambda: gateway.dispatch(30),
        # a's new command pushes its unsent one out of the full queue.
        lambda: gateway.complete([], 40),
        lambda: gateway.pop("b", 45),
        lambda: gateway.put("a", 3, 50),
        lambda: gateway.dispatch(55),
        lambda: gateway.close("b", 56),
        lambda: gateway.fail(60),
        lambda: gateway.put("a", 4, 65),
        lambda: gateway.dispatch(70),
        # a's queued command and its batch at the worker both outlive its session.
        lambda: gateway.close("a", 75),
        lambda: gateway.complete([], 80),
        lambda: gateway.open("a", 90),
        lambda: gateway.put("a", 1, 95),
        lambda: gateway.dispatch(100),
        lambda: gateway.complete([], 110),
        lambda: gateway.pop("a", 120),
    ]

    gaps_after_each_step = []
    for step in steps:
        step()
        gaps_after_each_step.append(second_identity_gaps(gateway.registry))

    assert gaps_after_each_step == [{}] * len(steps)
    assert [
        (
            counters.frames_served,
            counters.commands_sent,
            counters.commands_queued,
            counters.commands_dropped_downlink,
            counters.results_dropped_disconnect,
            counters.frames_failed,
            counters.in_flight,
        )
        for counters in (gateway.registry.get(robot_id).models[MODEL] for robot_id in "ab")
    ] == [(5, 1, 0, 1, 2, 1, 0), (2, 1, 0, 0, 0, 1, 0)]
