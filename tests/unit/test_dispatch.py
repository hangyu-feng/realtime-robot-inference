"""Tests for one model's result accounting: the result check, the stages and complete()."""

import copy
from collections.abc import Callable, Sequence
from dataclasses import replace

import pytest

from cloud.config import ServerConfig
from cloud.counters import CloseReason, ConnectionTotals
from cloud.dispatch import ModelDispatch, check_result, stage_durations
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRecord, RobotRegistry
from cloud.scheduler import Batch, BatchItem, Dispatch, plan_model
from cloud.sessions import Opened, Session, SessionTable
from common.commands import ClsCommand, decode_command, encode_command
from common.histogram import Histogram
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    STAGE_NAMES,
    CommandHeader,
    FrameHeader,
    Hello,
    StageDurations,
    decode_command_message,
    encode_frame,
    encode_hello,
)
from common.worker_protocol import (
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
CONFIG = ServerConfig()
JPEG = b"\xff\xd8\xff\xe0jpeg"
# The command header of common/protocol.py, which the body follows.
COMMAND_HEADER_BYTES = 64
# At these batch times every size fits the cls budget, so the cap is 8.
CLS_READY = WorkerReady(
    model="cls",
    pid=100,
    model_file=None,
    model_sha256=None,
    threads=1,
    t_b_ns={1: NS_PER_MS, 2: 2 * NS_PER_MS, 4: 4 * NS_PER_MS, 8: 8 * NS_PER_MS},
    t_b_cpu_ns={1: NS_PER_MS, 2: 2 * NS_PER_MS, 4: 4 * NS_PER_MS, 8: 8 * NS_PER_MS},
)

# The worker's stages of every frame in a result built by answer(): the worker has the batch
# 300.3 us after its dispatch and sends the result 5 ms later.
TO_WORKER_NS = 300_300
WORKER_NS = 5 * NS_PER_MS
DECODE_NS = 1_000_400
PREPROCESS_NS = 500_300
POSTPROCESS_NS = 250_200
INFERENCE_NS = 2_000_100
OWN_NS = DECODE_NS + PREPROCESS_NS + POSTPROCESS_NS + INFERENCE_NS
# A frame of the robots of World is sent 500.5 us before it arrives, 2 ms after its capture.
TRANSPORT_UP_NS = 500_500
FIRE_LAG_NS = 2 * NS_PER_MS


def body(index: int) -> bytes:
    """Return the command body a worker computes for the frame at index of its batch."""
    return encode_command(ClsCommand(class_id=index, confidence=0.5))


def answer(batch: Batch, statuses: Sequence[FrameStatus]) -> BatchResult:
    """Return a worker's result for the batch, one frame per status, with the stages above."""
    received_ns = batch.dispatched_ns + TO_WORKER_NS
    return BatchResult(
        batch_id=batch.batch_id,
        received_ns=received_ns,
        sent_ns=received_ns + WORKER_NS,
        inference_ns=INFERENCE_NS,
        cpu_ns=WORKER_NS,
        frames=tuple(
            FrameResult(
                status,
                body(index) if status is FrameStatus.OK else b"",
                DECODE_NS,
                PREPROCESS_NS,
                POSTPROCESS_NS,
            )
            for index, status in enumerate(statuses)
        ),
    )


def record(robot_id: str, session: int) -> RobotRecord:
    return RobotRecord(
        robot_id=robot_id,
        session=session,
        connected_ns=0,
        disconnected_ns=None,
        subscribed=("cls",),
        stale_ns={"cls": 500 * NS_PER_MS},
        watchdog_ns={"cls": NS_PER_S},
        frame_period_ns=33 * NS_PER_MS,
        models={},
    )


FRAME = Frame(1, 0, 10 * NS_PER_MS, FIRE_LAG_NS, TRANSPORT_UP_NS, JPEG)
BATCH = Batch(
    batch_id=7,
    items=(BatchItem(record("a", 1), 1, FRAME), BatchItem(record("b", 2), 2, FRAME)),
    dispatched_ns=20 * NS_PER_MS,
)
GOOD = answer(BATCH, [FrameStatus.OK, FrameStatus.OK])
GOOD_RESULT_NS = GOOD.sent_ns + 200_200


class World:
    """A session table and the cls dispatch over a real registry and mailbox."""

    def __init__(self) -> None:
        self.registry = RobotRegistry(round(CONFIG.robot_retention_s * NS_PER_S), 16)
        self.mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)
        self.table = SessionTable(
            CONFIG, self.registry, self.mailbox, ConnectionTotals(), lambda: None
        )
        self.dispatch = ModelDispatch("cls", CONFIG, CLS_READY, self.mailbox, 0)

    def open(self, robot_id: str, now_ns: int = 0) -> Session:
        hello = Hello(
            protocol_version=PROTOCOL_VERSION,
            robot_id=robot_id,
            token=None,
            models=("cls",),
            stale_ms={"cls": 500.0},
            watchdog_ms={"cls": 1_000.0},
            frame_rate_hz=30.0,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
        outcome = self.table.hello(encode_hello(hello), lambda: None, now_ns)
        assert isinstance(outcome, Opened)
        return outcome.session

    def send(self, session: Session, frame_number: int, received_ns: int) -> None:
        send_ns = received_ns - TRANSPORT_UP_NS
        header = FrameHeader(session.number, frame_number, send_ns - FIRE_LAG_NS, send_ns)
        assert self.table.ingest(session, encode_frame(header, JPEG), received_ns, None)

    def dispatch_batch(self, now_ns: int) -> Batch:
        decision = self.dispatch.scheduler.next(now_ns)
        assert isinstance(decision, Dispatch)
        return decision.batch


def drain(session: Session, now_ns: int) -> list[bytes]:
    """Pop every command queued for the session and return their messages, oldest first."""
    messages: list[bytes] = []
    while (command := session.downlink.pop(now_ns)) is not None:
        messages.append(command.data)
    return messages


def identity_breaks(registry: RobotRegistry) -> list[str]:
    """Return the robots whose cls counters break the second conservation identity."""
    return [
        record.robot_id
        for record in registry.records()
        for counters in [record.models["cls"]]
        if counters.frames_served
        != counters.commands_sent
        + counters.commands_queued
        + counters.commands_dropped_downlink
        + counters.results_dropped_disconnect
        + counters.frames_failed
        + counters.in_flight
    ]


def test_check_result_accepts_stamps_that_meet_at_every_boundary() -> None:
    # The worker has the batch at its dispatch, spends exactly the frames' own stages on it,
    # and its result arrives the moment it is sent.
    result = replace(GOOD, received_ns=BATCH.dispatched_ns, sent_ns=BATCH.dispatched_ns + OWN_NS)

    check_result(BATCH, result, result.sent_ns)


@pytest.mark.parametrize(
    ("result", "result_ns"),
    [
        pytest.param(replace(GOOD, batch_id=8), GOOD_RESULT_NS, id="another-batch"),
        pytest.param(replace(GOOD, frames=GOOD.frames[:1]), GOOD_RESULT_NS, id="a-frame-missing"),
        pytest.param(replace(GOOD, frames=GOOD.frames * 2), GOOD_RESULT_NS, id="frames-too-many"),
        pytest.param(
            replace(GOOD, received_ns=BATCH.dispatched_ns - 1),
            GOOD_RESULT_NS,
            id="received-before-the-dispatch",
        ),
        pytest.param(
            replace(GOOD, sent_ns=GOOD.received_ns - 1),
            GOOD_RESULT_NS,
            id="sent-before-it-was-received",
        ),
        pytest.param(GOOD, GOOD.sent_ns - 1, id="at-the-gateway-before-it-was-sent"),
        pytest.param(
            replace(GOOD, inference_ns=INFERENCE_NS + WORKER_NS - OWN_NS + 1),
            GOOD_RESULT_NS,
            id="own-stages-over-the-worker-time",
        ),
        pytest.param(
            replace(
                GOOD,
                frames=(
                    GOOD.frames[0],
                    replace(GOOD.frames[1], postprocess_ns=POSTPROCESS_NS + WORKER_NS),
                ),
            ),
            GOOD_RESULT_NS,
            id="one-frame-over-the-worker-time",
        ),
    ],
)
def test_check_result_refuses_a_result_that_does_not_fit_its_batch(
    result: BatchResult, result_ns: int
) -> None:
    with pytest.raises(WorkerProtocolError):
        check_result(BATCH, result, result_ns)


def stages_of(
    transport_up_ns: int,
    slot_wait_ns: int,
    to_worker_ns: int,
    worker_ns: int,
    from_worker_ns: int,
    frame_stages_ns: tuple[int, int, int],
    inference_ns: int,
) -> StageDurations:
    """Return the stages of a frame received at 1 s whose intervals are the ones given.

    frame_stages_ns are the frame's decode, preprocess and postprocess.
    """
    received_ns = NS_PER_S
    dispatched_ns = received_ns + slot_wait_ns
    worker_received_ns = dispatched_ns + to_worker_ns
    worker_sent_ns = worker_received_ns + worker_ns
    decode_ns, preprocess_ns, postprocess_ns = frame_stages_ns
    frame_result = FrameResult(FrameStatus.OK, body(0), decode_ns, preprocess_ns, postprocess_ns)
    result = BatchResult(1, worker_received_ns, worker_sent_ns, inference_ns, 0, (frame_result,))
    frame = Frame(1, 0, received_ns, FIRE_LAG_NS, transport_up_ns, JPEG)
    return stage_durations(
        frame, dispatched_ns, result, frame_result, worker_sent_ns + from_worker_ns
    )


@pytest.mark.parametrize(
    ("intervals_ns", "stages_us"),
    [
        pytest.param(
            (2_000, 7_000, 1_000, 9_000, 2_000, (3_000, 1_000, 1_000), 2_000),
            (2, 7, 3, 3, 1, 2, 2, 1),
            id="whole-microseconds",
        ),
        # Each stage rounded down on its own would add up to 15 999 us, two short of the
        # interval of 16 001 us.
        pytest.param(
            (
                500_500,
                10_000_000,
                300_300,
                5_000_000,
                200_200,
                (1_000_400, 500_300, 250_200),
                2_000_100,
            ),
            (500, 10_000, 501, 1_000, 500, 1_249, 2_000, 251),
            id="fractions-carried-to-the-stage-that-completes-a-microsecond",
        ),
        pytest.param(
            (600, 600, 0, 0, 0, (0, 0, 0), 0),
            (0, 1, 0, 0, 0, 0, 0, 0),
            id="stages-under-a-microsecond-each",
        ),
        pytest.param(
            (0, 0, 0, 4_000, 0, (1_000, 1_000, 1_000), 1_000),
            (0, 0, 0, 1, 1, 0, 1, 1),
            id="no-time-on-other-frames",
        ),
    ],
)
def test_the_stages_are_cumulative_offsets_rounded_down_to_the_microsecond(
    intervals_ns: tuple[int, int, int, int, int, tuple[int, int, int], int],
    stages_us: tuple[int, ...],
) -> None:
    assert stages_of(*intervals_ns) == StageDurations(*stages_us)


def test_a_dispatch_plans_its_model_from_the_calibration_and_starts_with_empty_counts() -> None:
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)

    dispatch = ModelDispatch("cls", CONFIG, CLS_READY, mailbox, 5)

    assert (dispatch.model, dispatch.calibration) == ("cls", CLS_READY)
    assert dispatch.plan == plan_model("cls", CONFIG, CLS_READY.t_b_ns)
    assert dispatch.stages == {name: Histogram() for name in STAGE_NAMES}
    assert dispatch.errors == {"decode": 0, "inference": 0}
    assert dispatch.scheduler.in_flight is None


def test_the_scheduler_dispatches_up_to_the_planned_cap_from_the_given_time() -> None:
    world = World()
    sessions = [world.open(f"r{index}") for index in range(9)]
    for session in sessions:
        world.send(session, 1, 10 * NS_PER_MS)

    batch = world.dispatch_batch(25 * NS_PER_MS)

    assert (len(batch.items), world.dispatch.plan.scheduler.batch_cap) == (8, 8)
    # The worker counts as free from the time the dispatch was built at, 0.
    assert world.dispatch.scheduler.stats.idle_ns == 25 * NS_PER_MS


def test_a_calibration_of_other_sizes_than_the_configured_ones_is_refused() -> None:
    ready = replace(CLS_READY, t_b_ns={1: NS_PER_MS, 2: NS_PER_MS}, t_b_cpu_ns={1: 1, 2: 1})
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)

    with pytest.raises(ValueError, match="not the calibrated sizes"):
        ModelDispatch("cls", CONFIG, ready, mailbox, 0)


def command_header(batch: Batch, index: int, result: BatchResult, result_ns: int) -> CommandHeader:
    """Return the header of the command for the item at index of the batch."""
    item = batch.items[index]
    return CommandHeader(
        model="cls",
        session=item.session,
        frame_number=item.frame.frame_number,
        capture_ns=item.frame.capture_ns,
        batch_size=len(batch.items),
        result_ns=result_ns,
        stages=stage_durations(
            item.frame, batch.dispatched_ns, result, result.frames[index], result_ns
        ),
    )


def test_complete_queues_a_command_for_each_deliverable_frame_and_counts_the_others() -> None:
    world = World()
    sessions = {robot_id: world.open(robot_id) for robot_id in "abcdef"}
    for index, robot_id in enumerate("abcdef"):
        world.send(sessions[robot_id], 1, (10 + index) * NS_PER_MS)
    batch = world.dispatch_batch(20 * NS_PER_MS)
    # While the batch is at the worker, d leaves and e connects again.
    world.table.end(sessions["d"], CloseReason.ABNORMAL, 21 * NS_PER_MS)
    replacement = world.open("e", 22 * NS_PER_MS)
    statuses = [
        FrameStatus.OK,
        FrameStatus.DECODE_ERROR,
        FrameStatus.INFERENCE_ERROR,
        FrameStatus.OK,
        FrameStatus.OK,
        FrameStatus.OK,
    ]
    result = answer(batch, statuses)
    result_ns = result.sent_ns + 200_200

    woken = world.dispatch.complete(batch, result, result_ns, world.table.sessions)

    assert [item.record.robot_id for item in batch.items] == list("abcdef")
    assert woken == [sessions["a"], sessions["f"]]
    assert identity_breaks(world.registry) == []
    assert world.dispatch.errors == {"decode": 1, "inference": 1}
    assert {name: histogram.count for name, histogram in world.dispatch.stages.items()} == (
        dict.fromkeys(STAGE_NAMES, 2)
    )
    assert {
        robot_id: (
            counters.commands_queued,
            counters.frames_failed,
            counters.results_dropped_disconnect,
        )
        for robot_id in "abcdef"
        for counters in [world.registry.get(robot_id).models["cls"]]
    } == {
        "a": (1, 0, 0),
        "b": (0, 1, 0),
        "c": (0, 1, 0),
        "d": (0, 0, 1),
        "e": (0, 0, 1),
        "f": (1, 0, 0),
    }
    assert {
        robot_id: [decode_command_message(message) for message in drain(session, result_ns)]
        for robot_id, session in [*sessions.items(), ("e new", replacement)]
    } == {
        "a": [(command_header(batch, 0, result, result_ns), decode_command("cls", body(0)))],
        "b": [],
        "c": [],
        "d": [],
        "e": [],
        "e new": [],
        "f": [(command_header(batch, 5, result, result_ns), decode_command("cls", body(5)))],
    }


def test_the_command_carries_the_worker_body_unchanged_after_its_header() -> None:
    world = World()
    session = world.open("a")
    world.send(session, 1, 10 * NS_PER_MS)
    batch = world.dispatch_batch(20 * NS_PER_MS)
    result = answer(batch, [FrameStatus.OK])

    world.dispatch.complete(batch, result, result.sent_ns, world.table.sessions)

    (message,) = drain(session, result.sent_ns)
    assert message[COMMAND_HEADER_BYTES:] == body(0)


def test_the_stage_histograms_count_exactly_the_commands_queued() -> None:
    world = World()
    sessions = [world.open(robot_id) for robot_id in "abc"]
    for index, session in enumerate(sessions):
        world.send(session, 1, (10 + index) * NS_PER_MS)
    batch = world.dispatch_batch(20 * NS_PER_MS)
    world.table.end(sessions[2], CloseReason.CLIENT_CLOSE, 21 * NS_PER_MS)
    result = answer(batch, [FrameStatus.OK, FrameStatus.DECODE_ERROR, FrameStatus.OK])

    world.dispatch.complete(batch, result, result.sent_ns, world.table.sessions)

    stages = world.dispatch.stages
    # Only a got a command: its frame waited 10 ms for the dispatch, 10 000 whole microseconds.
    assert {name: histogram.count for name, histogram in stages.items()} == dict.fromkeys(
        STAGE_NAMES, 1
    )
    assert (stages["slot_wait"].min_ms, stages["slot_wait"].max_ms) == (10.0, 10.0)
    assert stages["transport_up"].sum_ms == 0.5


@pytest.mark.parametrize(
    ("share_cores", "waits_ns"),
    [
        pytest.param(None, (0, 20 * NS_PER_MS), id="unpaced"),
        # At a quarter of a core, the batch's 4 ms of CPU time holds the worker until 16 ms
        # after its dispatch, 26 ms: paced from the result at 18 ms until then, idle after.
        pytest.param(0.25, (8 * NS_PER_MS, 12 * NS_PER_MS), id="paced"),
    ],
)
def test_the_wait_after_a_batch_counts_from_the_gateways_reading_of_its_result(
    share_cores: float | None, waits_ns: tuple[int, int]
) -> None:
    world = World()
    world.dispatch.scheduler.set_pacing_share(share_cores)
    session = world.open("a")
    world.send(session, 1, 10 * NS_PER_MS)
    batch = world.dispatch_batch(10 * NS_PER_MS)
    # The worker sends its result at 15 ms and the gateway has it at 18 ms. The worker is
    # ready for the next batch only from then: the 3 ms back are part of dispatch_rt, neither
    # paced nor idle.
    result = replace(
        answer(batch, [FrameStatus.OK]),
        received_ns=11 * NS_PER_MS,
        sent_ns=15 * NS_PER_MS,
        cpu_ns=4 * NS_PER_MS,
    )
    world.dispatch.complete(batch, result, 18 * NS_PER_MS, world.table.sessions)
    stats = world.dispatch.scheduler.stats
    paced_before_ns, idle_before_ns = stats.paced_ns, stats.idle_ns
    world.send(session, 2, 38 * NS_PER_MS)

    world.dispatch_batch(38 * NS_PER_MS)

    assert (stats.paced_ns - paced_before_ns, stats.idle_ns - idle_before_ns) == waits_ns


@pytest.mark.parametrize(
    "spoil",
    [
        pytest.param(
            lambda result: replace(result, batch_id=result.batch_id + 1), id="another-batch"
        ),
        pytest.param(lambda result: replace(result, frames=()), id="no-frames"),
        pytest.param(
            lambda result: replace(result, received_ns=result.received_ns - TO_WORKER_NS - 1),
            id="received-before-the-dispatch",
        ),
    ],
)
def test_a_refused_result_raises_before_anything_is_booked(
    spoil: Callable[[BatchResult], BatchResult],
) -> None:
    world = World()
    session = world.open("a")
    world.send(session, 1, 10 * NS_PER_MS)
    batch = world.dispatch_batch(20 * NS_PER_MS)
    result = spoil(answer(batch, [FrameStatus.DECODE_ERROR]))
    counters = copy.deepcopy(session.record.models["cls"])

    with pytest.raises(WorkerProtocolError):
        world.dispatch.complete(batch, result, result.sent_ns + 1, world.table.sessions)

    stats = world.dispatch.scheduler.stats
    assert world.dispatch.scheduler.in_flight is batch
    assert session.record.models["cls"] == counters
    assert world.dispatch.errors == {"decode": 0, "inference": 0}
    assert (stats.busy_ns, stats.batch_ms.count, len(session.downlink)) == (0, 0, 0)
    assert {histogram.count for histogram in world.dispatch.stages.values()} == {0}
    # The batch is still the scheduler's to fail, which counts its frame once.
    world.dispatch.scheduler.fail(batch, result.sent_ns + 2)
    assert (session.record.models["cls"].frames_failed, identity_breaks(world.registry)) == (1, [])
