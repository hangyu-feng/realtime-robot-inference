"""Tests for the mailbox: the two policies, take and skip, close, and the first identity."""

from collections.abc import Callable

import pytest

from cloud.config import QueuePolicy
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRecord, RobotRegistry

# Times are in nanoseconds.
FRAME_PERIOD_NS = 40
RETENTION_NS = 10_000
MAX_ENTRIES = 8
FIFO_CAPACITY_FRAMES = 2
BOTH = ("cls", "depth")

Operation = Callable[[Mailbox, str, str], object]
OPERATIONS: list[Operation] = [Mailbox.head, Mailbox.take, Mailbox.skip]
OPERATION_IDS = ["head", "take", "skip"]


def new_mailbox(policy: QueuePolicy) -> tuple[RobotRegistry, Mailbox]:
    return RobotRegistry(RETENTION_NS, MAX_ENTRIES), Mailbox(policy, FIFO_CAPACITY_FRAMES)


def open_robot(
    registry: RobotRegistry, mailbox: Mailbox, robot_id: str, subscribed: tuple[str, ...] = BOTH
) -> RobotRecord:
    """Run the open sequence of a session for robot_id."""
    record = registry.connect(
        robot_id,
        subscribed,
        dict.fromkeys(subscribed, 500),
        dict.fromkeys(subscribed, 1_000),
        FRAME_PERIOD_NS,
        0,
    )
    mailbox.open(record)
    return record


def frame(frame_number: int) -> Frame:
    capture_ns = frame_number * FRAME_PERIOD_NS
    return Frame(
        frame_number=frame_number,
        capture_ns=capture_ns,
        received_ns=capture_ns + 5,
        fire_lag_ns=1,
        transport_up_ns=2,
        jpeg=b"\xff\xd8\xff\xd9",
    )


def first_identity_gaps(registry: RobotRegistry, mailbox: Mailbox) -> dict[tuple[str, str], int]:
    """Return every record and model whose first identity fails, with received minus the rest."""
    gaps = {
        (record.robot_id, model): counters.frames_received
        - counters.frames_served
        - counters.frames_overwritten
        - counters.frames_skipped
        - counters.frames_dropped_disconnect
        - mailbox.pending_count(record.robot_id, model)
        for record in registry.records()
        for model, counters in record.models.items()
    }
    return {key: gap for key, gap in gaps.items() if gap != 0}


def test_latest_keeps_only_the_newest_frame_and_counts_each_replaced_one_per_model() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    record = open_robot(registry, mailbox, "a")

    for frame_number in (1, 2, 3):
        mailbox.put("a", frame(frame_number))

    assert [mailbox.head("a", model).frame_number for model in BOTH] == [3, 3]
    assert [mailbox.pending_count("a", model) for model in BOTH] == [1, 1]
    assert [
        (counters.frames_received, counters.frames_overwritten)
        for counters in record.models.values()
    ] == [(3, 2), (3, 2)]


def test_the_models_share_one_frame_object() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a")
    put = frame(1)

    mailbox.put("a", put)

    assert mailbox.head("a", "cls") is put
    assert mailbox.head("a", "depth") is put


def test_fifo_drops_the_new_frame_for_a_model_whose_queue_is_full() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.FIFO)
    record = open_robot(registry, mailbox, "a")
    for frame_number in (1, 2, 3):
        mailbox.put("a", frame(frame_number))
    mailbox.take("a", "cls")

    mailbox.put("a", frame(4))

    taken = {
        model: [mailbox.take("a", model).frame_number for _ in range(FIFO_CAPACITY_FRAMES)]
        for model in BOTH
    }
    assert taken == {"cls": [2, 4], "depth": [1, 2]}
    assert {model: counters.frames_overwritten for model, counters in record.models.items()} == {
        "cls": 1,
        "depth": 2,
    }


@pytest.mark.parametrize(
    ("policy", "frame_number"),
    [(QueuePolicy.LATEST, 2), (QueuePolicy.FIFO, 1)],
    ids=["latest-gives-the-newest", "fifo-gives-the-oldest"],
)
def test_take_removes_and_returns_the_head_and_counts_it_served_and_in_flight(
    policy: QueuePolicy, frame_number: int
) -> None:
    registry, mailbox = new_mailbox(policy)
    record = open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(1))
    mailbox.put("a", frame(2))
    head = mailbox.head("a", "cls")

    taken = mailbox.take("a", "cls")

    assert taken is head
    assert taken.frame_number == frame_number
    counters = record.models["cls"]
    assert (counters.frames_served, counters.in_flight, counters.frames_skipped) == (1, 1, 0)


@pytest.mark.parametrize(
    ("policy", "pending_after"),
    [(QueuePolicy.LATEST, 0), (QueuePolicy.FIFO, 1)],
    ids=["latest", "fifo"],
)
def test_skip_removes_the_head_and_counts_it_skipped(
    policy: QueuePolicy, pending_after: int
) -> None:
    registry, mailbox = new_mailbox(policy)
    record = open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(1))
    mailbox.put("a", frame(2))

    mailbox.skip("a", "cls")

    counters = record.models["cls"]
    assert (counters.frames_skipped, counters.frames_served, counters.in_flight) == (1, 0, 0)
    assert mailbox.pending_count("a", "cls") == pending_after


def test_skip_under_fifo_removes_the_oldest_frame() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.FIFO)
    open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(1))
    mailbox.put("a", frame(2))

    mailbox.skip("a", "cls")

    assert mailbox.head("a", "cls").frame_number == 2


def test_take_leaves_the_frame_pending_for_the_other_model() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(1))

    taken = mailbox.take("a", "cls")

    assert mailbox.head("a", "depth") is taken
    assert [mailbox.pending_count("a", model) for model in BOTH] == [0, 1]


@pytest.mark.parametrize("operation", OPERATIONS, ids=OPERATION_IDS)
@pytest.mark.parametrize(
    ("robot_id", "model"),
    [("a", "cls"), ("a", "depth"), ("b", "cls")],
    ids=["nothing-pending", "unsubscribed-model", "id-not-open"],
)
def test_head_take_and_skip_raise_when_no_frame_is_pending(
    operation: Operation, robot_id: str, model: str
) -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a", ("cls",))

    with pytest.raises(ValueError, match="is pending"):
        operation(mailbox, robot_id, model)


@pytest.mark.parametrize(
    ("policy", "dropped"),
    [(QueuePolicy.LATEST, {"cls": 0, "depth": 1}), (QueuePolicy.FIFO, {"cls": 1, "depth": 2})],
    ids=["latest", "fifo"],
)
def test_close_counts_each_pending_frame_once_per_model(
    policy: QueuePolicy, dropped: dict[str, int]
) -> None:
    registry, mailbox = new_mailbox(policy)
    record = open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(1))
    mailbox.put("a", frame(2))
    mailbox.take("a", "cls")

    mailbox.close("a")

    assert {
        model: counters.frames_dropped_disconnect for model, counters in record.models.items()
    } == dropped
    assert [mailbox.pending_count("a", model) for model in BOTH] == [0, 0]
    assert mailbox.pending_robots("depth") == []


def test_pending_count_is_zero_for_an_unsubscribed_model_and_an_id_that_is_not_open() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a", ("cls",))
    mailbox.put("a", frame(1))
    counts = [mailbox.pending_count("a", "depth"), mailbox.pending_count("b", "cls")]

    mailbox.close("a")

    assert [*counts, mailbox.pending_count("a", "cls")] == [0, 0, 0]


def test_pending_robots_lists_the_open_robots_with_a_frame_for_the_model() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a")
    open_robot(registry, mailbox, "b", ("cls",))
    open_robot(registry, mailbox, "c", ("depth",))
    open_robot(registry, mailbox, "d")
    for robot_id in "abc":
        mailbox.put(robot_id, frame(1))

    mailbox.take("a", "cls")

    assert [record.robot_id for record in mailbox.pending_robots("cls")] == ["b"]
    assert sorted(record.robot_id for record in mailbox.pending_robots("depth")) == ["a", "c"]


def test_opening_an_open_id_raises() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    record = open_robot(registry, mailbox, "a")

    with pytest.raises(ValueError, match="'a' is already open"):
        mailbox.open(record)


def test_closing_an_id_that_is_not_open_raises() -> None:
    _, mailbox = new_mailbox(QueuePolicy.LATEST)

    with pytest.raises(KeyError):
        mailbox.close("a")


def test_a_closed_id_opens_again_for_its_next_session_with_nothing_pending() -> None:
    registry, mailbox = new_mailbox(QueuePolicy.LATEST)
    open_robot(registry, mailbox, "a")
    mailbox.put("a", frame(7))
    mailbox.close("a")
    registry.disconnect("a", 10)

    open_robot(registry, mailbox, "a", ("depth",))
    mailbox.put("a", frame(1))

    assert [mailbox.pending_count("a", model) for model in BOTH] == [0, 1]
    assert mailbox.head("a", "depth").frame_number == 1


@pytest.mark.parametrize("policy", [QueuePolicy.LATEST, QueuePolicy.FIFO])
def test_the_first_identity_holds_after_every_step(policy: QueuePolicy) -> None:
    registry, mailbox = new_mailbox(policy)
    steps: list[Callable[[], object]] = [
        lambda: open_robot(registry, mailbox, "a"),
        lambda: open_robot(registry, mailbox, "b", ("cls",)),
        lambda: mailbox.put("a", frame(1)),
        lambda: mailbox.put("b", frame(1)),
        lambda: mailbox.put("a", frame(2)),
        lambda: mailbox.put("a", frame(3)),
        lambda: mailbox.take("a", "cls"),
        lambda: mailbox.skip("a", "depth"),
        lambda: mailbox.put("a", frame(4)),
        lambda: mailbox.take("b", "cls"),
        lambda: mailbox.put("b", frame(2)),
        lambda: mailbox.close("a"),
        lambda: registry.disconnect("a", 10),
        lambda: open_robot(registry, mailbox, "a", ("depth",)),
        lambda: mailbox.put("a", frame(1)),
        lambda: mailbox.close("b"),
    ]

    gaps_after_each_step = []
    for step in steps:
        step()
        gaps_after_each_step.append(first_identity_gaps(registry, mailbox))

    assert gaps_after_each_step == [{}] * len(steps)
