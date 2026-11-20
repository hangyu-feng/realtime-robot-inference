"""Tests for the session table: hellos, admission and replacement, ingress, and session ends."""

import copy
from collections.abc import Callable
from dataclasses import replace

import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, precondition, rule, run_state_machine_as_test

from cloud.config import ServerConfig
from cloud.counters import (
    FRAME_REJECT_REASONS,
    REJECT_REASONS,
    CloseReason,
    ConnectionTotals,
)
from cloud.downlink import QueuedCommand
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRegistry
from cloud.scheduler import Dispatch, ModelScheduler, SchedulerConfig
from cloud.sessions import CLOSE_CODES, Opened, Refused, Session, SessionTable
from common.commands import MODEL_KEYS
from common.histogram import Histogram
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    CloseCode,
    FrameHeader,
    Hello,
    ModelTelemetry,
    Telemetry,
    Welcome,
    decode_welcome,
    encode_frame,
    encode_hello,
    encode_telemetry,
    telemetry_to_dict,
)

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
CONFIG = ServerConfig()
# One message per second, so that a second message at the same instant is over the limit.
ONE_MESSAGE_PER_S = replace(CONFIG, ingress_max_messages_per_s=1.0)
TOKEN_CONFIG = replace(CONFIG, shared_token="secret")
ONE_CONNECTION = replace(CONFIG, max_connections=1)
# A frame may be at most 100 ms old, so that the age rules apply to small times.
MAX_AGE_NS = 100 * NS_PER_MS
AGE_CONFIG = replace(CONFIG, max_frame_age_ms=100.0)
JPEG = b"\xff\xd8\xff\xe0jpeg"

MODEL_TELEMETRY = ModelTelemetry(
    commands_received=3,
    commands_applied=2,
    commands_stale=1,
    commands_out_of_order=0,
    freshness_ms=Histogram(),
    command_gap_ms=Histogram(),
    command_age_peak_ms=Histogram(),
    deliver_ms=Histogram(),
    watchdog_trips=0,
    watchdog_expired_ms=40,
    watchdog_state="ok",
)
TELEMETRY = Telemetry(
    ts_mono_ns=123_456_789,
    frames_captured=30,
    frames_sent=29,
    frames_skipped_send=1,
    fire_lag_ms=Histogram(),
    reconnects=2,
    models={"cls": MODEL_TELEMETRY, "depth": MODEL_TELEMETRY},
)


def hello_text(
    robot_id: str = "a",
    *,
    token: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
    models: tuple[str, ...] = MODEL_KEYS,
    stale_ms: float = 150.0,
    watchdog_ms: float = 300.0,
    frame_rate_hz: float = 30.0,
) -> str:
    return encode_hello(
        Hello(
            protocol_version=protocol_version,
            robot_id=robot_id,
            token=token,
            models=models,
            stale_ms=dict.fromkeys(models, stale_ms),
            watchdog_ms=dict.fromkeys(models, watchdog_ms),
            frame_rate_hz=frame_rate_hz,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
    )


def frame_at(
    session_number: int, frame_number: int, capture_ns: int, send_ns: int, jpeg: bytes = JPEG
) -> bytes:
    return encode_frame(FrameHeader(session_number, frame_number, capture_ns, send_ns), jpeg)


def good_frame(session: Session, frame_number: int, received_ns: int) -> bytes:
    """Return a frame of the session captured 3 ms and sent 1 ms before received_ns."""
    return frame_at(
        session.number, frame_number, received_ns - 3 * NS_PER_MS, received_ns - NS_PER_MS
    )


def closed(**counts: int) -> dict[str, int]:
    """Return connections.closed with these counts and every other reason at 0."""
    return {reason.value: 0 for reason in CloseReason} | counts


def rejected(**counts: int) -> dict[str, int]:
    """Return connections.rejected with these counts and every other reason at 0."""
    return dict.fromkeys(REJECT_REASONS, 0) | counts


def frames_rejected(**counts: int) -> dict[str, int]:
    """Return connections.frames_rejected with these counts and every other reason at 0."""
    return dict.fromkeys(FRAME_REJECT_REASONS, 0) | counts


class Wake:
    """A session's wake that counts its calls."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> None:
        self.calls += 1


class World:
    """A session table over a real registry, mailbox and totals, checked after every step.

    Commands are queued as a dispatcher queues them, through one real scheduler per model.
    After every step: for every record, frames_in = frames_rejected + frames_accepted and both
    conservation identities of each model hold; the per-robot counters add up to the totals;
    accepted = connected_count + sum(closed); and the table holds exactly the sessions of the
    connected records, each with its record's session number.
    """

    def __init__(self, config: ServerConfig = CONFIG) -> None:
        self.registry = RobotRegistry(
            round(config.robot_retention_s * NS_PER_S), config.max_robot_entries
        )
        self.mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        self.totals = ConnectionTotals()
        # The connected count that each call of the membership callback saw.
        self.memberships: list[int] = []
        self.table = SessionTable(
            config,
            self.registry,
            self.mailbox,
            self.totals,
            lambda: self.memberships.append(self.registry.connected_count()),
        )
        self.wakes: dict[Session, Wake] = {}
        self.schedulers = {
            model: ModelScheduler(
                SchedulerConfig(
                    model,
                    batch_cap=8,
                    service_ns=1,
                    return_allowance_ns=0,
                    skip_rule=False,
                    skip_max_unserved_ns=1,
                ),
                self.mailbox,
                0,
            )
            for model in MODEL_KEYS
        }

    def hello(
        self, message: str | bytes, now_ns: int = 0, wake: Callable[[], None] = lambda: None
    ) -> Opened | Refused:
        outcome = self.table.hello(message, wake, now_ns)
        self.check()
        return outcome

    def open(
        self,
        robot_id: str = "a",
        now_ns: int = 0,
        token: str | None = None,
        models: tuple[str, ...] = MODEL_KEYS,
    ) -> Session:
        wake = Wake()
        outcome = self.hello(hello_text(robot_id, token=token, models=models), now_ns, wake)
        assert isinstance(outcome, Opened)
        self.wakes[outcome.session] = wake
        return outcome.session

    def ingest(
        self, session: Session, message: str | bytes, received_ns: int, rtt_ns: int | None = None
    ) -> bool:
        accepted = self.table.ingest(session, message, received_ns, rtt_ns)
        self.check()
        return accepted

    def lost(
        self, session: Session, peer_code: int | None, local_code: int | None, now_ns: int
    ) -> None:
        self.table.lost(session, peer_code, local_code, now_ns)
        self.check()

    def end(self, session: Session, reason: CloseReason, now_ns: int) -> None:
        self.table.end(session, reason, now_ns)
        self.check()

    def queue_command(self, model: str, now_ns: int) -> None:
        """Serve every frame pending for the model in one batch and queue a command for each."""
        scheduler = self.schedulers[model]
        decision = scheduler.next(now_ns)
        assert isinstance(decision, Dispatch)
        batch = decision.batch
        for index in scheduler.complete(batch, (), 1, 1, now_ns):
            record = batch.items[index].record
            command = QueuedCommand(b"command", record.models[model])
            self.table.sessions[record.robot_id].downlink.push(command)
        self.check()

    def check(self) -> None:
        records = list(self.registry.records())
        assert [
            record.robot_id
            for record in records
            if record.frames_in != record.frames_rejected + record.frames_accepted
        ] == []
        assert [
            (record.robot_id, model)
            for record in records
            for model, counters in record.models.items()
            if counters.frames_received
            != counters.frames_served
            + counters.frames_overwritten
            + counters.frames_skipped
            + counters.frames_dropped_disconnect
            + self.mailbox.pending_count(record.robot_id, model)
            or counters.frames_served
            != counters.commands_sent
            + counters.commands_queued
            + counters.commands_dropped_downlink
            + counters.results_dropped_disconnect
            + counters.frames_failed
            + counters.in_flight
        ] == []
        assert sum(record.bytes_in for record in records) == self.totals.bytes_in
        assert sum(record.frames_rejected for record in records) == sum(
            self.totals.frames_rejected.values()
        )
        assert self.totals.accepted == self.registry.connected_count() + sum(
            self.totals.closed.values()
        )
        assert {robot_id: session.number for robot_id, session in self.table.sessions.items()} == {
            record.robot_id: record.session for record in records if record.connected
        }


def test_each_end_closes_with_its_code_or_aborts_the_connection() -> None:
    assert dict(CLOSE_CODES) == {
        CloseReason.CLIENT_CLOSE: None,
        CloseReason.ABNORMAL: None,
        CloseReason.IDLE_TIMEOUT: CloseCode.IDLE_TIMEOUT,
        CloseReason.SLOW_CONSUMER: None,
        CloseReason.INGRESS_LIMIT: CloseCode.POLICY,
        CloseReason.PROTOCOL_ERROR: CloseCode.PROTOCOL_ERROR,
        CloseReason.REPLACED: CloseCode.SESSION_REPLACED,
        CloseReason.SHUTDOWN: CloseCode.SERVER_SHUTDOWN,
        CloseReason.INTERNAL_ERROR: None,
    }


def test_a_good_hello_opens_a_session_and_answers_with_its_welcome() -> None:
    world = World()
    message = hello_text("a")
    wake = Wake()

    outcome = world.hello(message, 5 * NS_PER_MS, wake)

    assert isinstance(outcome, Opened)
    session = outcome.session
    record = world.registry.get("a")
    assert decode_welcome(outcome.welcome) == Welcome(PROTOCOL_VERSION, 1, True)
    assert session.record is record
    assert (session.robot_id, session.number, session.last_frame_number, session.ended) == (
        "a",
        1,
        None,
        False,
    )
    assert (record.connected, record.connected_ns, record.subscribed) == (
        True,
        5 * NS_PER_MS,
        MODEL_KEYS,
    )
    assert dict(world.table.sessions) == {"a": session}
    assert (world.totals.accepted, record.bytes_in, world.totals.bytes_in) == (
        1,
        len(message),
        len(message),
    )
    assert (world.memberships, wake.calls) == ([1], 0)
    with pytest.raises(ValueError, match="already open"):
        world.mailbox.open(record)


@pytest.mark.parametrize(
    ("models", "stale_ms", "watchdog_ms", "frame_rate_hz", "thresholds_ns", "frame_period_ns"),
    [
        pytest.param(
            ("cls", "depth"), 150.0, 300.0, 30.0, (150 * NS_PER_MS, 300 * NS_PER_MS), 33_333_333
        ),
        pytest.param(("depth",), 120.25, 600.5, 60.0, (120_250_000, 600_500_000), 16_666_667),
        pytest.param(("cls",), 0.0004, 0.0006, 1.0, (400, 600), NS_PER_S),
    ],
)
def test_a_hello_declares_its_thresholds_and_frame_period_in_nanoseconds(
    models: tuple[str, ...],
    stale_ms: float,
    watchdog_ms: float,
    frame_rate_hz: float,
    thresholds_ns: tuple[int, int],
    frame_period_ns: int,
) -> None:
    world = World()

    world.hello(
        hello_text(
            "a",
            models=models,
            stale_ms=stale_ms,
            watchdog_ms=watchdog_ms,
            frame_rate_hz=frame_rate_hz,
        )
    )

    record = world.registry.get("a")
    stale_ns, watchdog_ns = thresholds_ns
    assert (record.subscribed, record.stale_ns, record.watchdog_ns) == (
        models,
        dict.fromkeys(models, stale_ns),
        dict.fromkeys(models, watchdog_ns),
    )
    assert record.frame_period_ns == frame_period_ns


@pytest.mark.parametrize("shared_clock", [True, False])
def test_the_welcome_says_whether_robot_and_gateway_share_a_clock(shared_clock: bool) -> None:
    world = World(replace(CONFIG, shared_clock=shared_clock))

    outcome = world.hello(hello_text("a"))

    assert isinstance(outcome, Opened)
    assert decode_welcome(outcome.welcome).shared_clock is shared_clock


@pytest.mark.parametrize(
    ("config", "message", "label", "code"),
    [
        pytest.param(
            CONFIG, frame_at(1, 1, 0, 0), "bad_hello", CloseCode.PROTOCOL_ERROR, id="binary"
        ),
        pytest.param(CONFIG, "not json", "bad_hello", CloseCode.PROTOCOL_ERROR, id="not-json"),
        pytest.param(
            CONFIG,
            encode_telemetry(TELEMETRY),
            "bad_hello",
            CloseCode.PROTOCOL_ERROR,
            id="telemetry-first",
        ),
        pytest.param(
            CONFIG, hello_text("a", protocol_version=2), "version", CloseCode.POLICY, id="version"
        ),
        pytest.param(
            CONFIG,
            hello_text("a", models=("cls", "pose")),
            "unknown_model",
            CloseCode.POLICY,
            id="unknown-model",
        ),
        pytest.param(TOKEN_CONFIG, hello_text("a"), "auth", CloseCode.POLICY, id="no-token"),
        pytest.param(
            TOKEN_CONFIG, hello_text("a", token="guess"), "auth", CloseCode.POLICY, id="bad-token"
        ),
        pytest.param(
            ONE_CONNECTION, hello_text("a"), "capacity", CloseCode.OVER_CAPACITY, id="capacity"
        ),
    ],
)
def test_a_refused_hello_is_counted_under_its_label_and_registers_nothing(
    config: ServerConfig, message: str | bytes, label: str, code: CloseCode
) -> None:
    world = World(config)
    other = world.open("b", token=config.shared_token)

    outcome = world.hello(message, 10 * NS_PER_MS)

    assert outcome == Refused(label, code)
    assert world.totals.rejected == rejected(**{label: 1})
    assert dict(world.table.sessions) == {"b": other}
    assert [record.robot_id for record in world.registry.records()] == ["b"]
    assert (world.totals.accepted, world.totals.bytes_in, world.memberships) == (
        1,
        len(hello_text("b", token=config.shared_token)),
        [1],
    )


def test_no_hello_within_the_handshake_timeout_is_refused_as_an_idle_timeout() -> None:
    world = World()

    refused = world.table.refuse_handshake_timeout()

    assert refused == Refused("handshake_timeout", CloseCode.IDLE_TIMEOUT)
    assert world.totals.rejected == rejected(handshake_timeout=1)
    world.check()


def test_the_token_is_checked_before_the_connection_cap() -> None:
    world = World(replace(CONFIG, shared_token="secret", max_connections=1))
    world.open("b", token="secret")

    outcome = world.hello(hello_text("a", token="guess"))

    assert outcome == Refused("auth", CloseCode.POLICY)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        pytest.param(TOKEN_CONFIG, hello_text("a", token="guess"), id="bad-token"),
        pytest.param(CONFIG, hello_text("a", protocol_version=2), id="version"),
        pytest.param(CONFIG, hello_text("a", models=("pose",)), id="unknown-model"),
    ],
)
def test_a_refused_second_hello_leaves_the_first_session_open(
    config: ServerConfig, message: str
) -> None:
    world = World(config)
    first = world.open("a", token=config.shared_token)
    world.ingest(first, good_frame(first, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)

    outcome = world.hello(message, 20 * NS_PER_MS)

    assert isinstance(outcome, Refused)
    assert (first.ended, world.wakes[first].calls, world.table.sessions["a"] is first) == (
        False,
        0,
        True,
    )
    assert world.registry.get("a").session == first.number
    assert world.mailbox.pending_count("a", "cls") == 1
    assert world.totals.closed == closed()


def test_a_hello_that_replaces_a_session_is_not_refused_for_capacity() -> None:
    world = World(ONE_CONNECTION)
    first = world.open("a")

    second = world.open("a", 10 * NS_PER_MS)

    assert (first.end_reason, second.number, world.totals.rejected) == (
        CloseReason.REPLACED,
        first.number + 1,
        rejected(),
    )


def test_a_second_hello_with_the_same_id_replaces_the_first_session() -> None:
    world = World()
    old = world.open("a")
    world.ingest(old, good_frame(old, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)
    # cls serves the frame and queues its command; depth leaves it pending.
    world.queue_command("cls", 20 * NS_PER_MS)

    new = world.open("a", 30 * NS_PER_MS)

    counters = world.registry.get("a").models
    assert (old.end_reason, world.wakes[old].calls, world.wakes[new].calls) == (
        CloseReason.REPLACED,
        1,
        0,
    )
    assert (new.number, world.table.sessions["a"] is new) == (old.number + 1, True)
    assert world.totals.closed == closed(replaced=1)
    assert counters["depth"].frames_dropped_disconnect == 1
    assert counters["cls"].results_dropped_disconnect == 1
    assert (len(new.downlink), world.mailbox.pending_count("a", "depth")) == (0, 0)
    assert world.memberships == [1, 0, 1]


@pytest.mark.parametrize(
    "late",
    [
        pytest.param(
            lambda world, old: world.ingest(
                old, good_frame(old, 2, 40 * NS_PER_MS), 40 * NS_PER_MS
            ),
            id="frame",
        ),
        pytest.param(
            lambda world, old: world.ingest(old, encode_telemetry(TELEMETRY), 40 * NS_PER_MS),
            id="telemetry",
        ),
        pytest.param(
            lambda world, old: world.lost(old, None, CloseCode.MESSAGE_TOO_BIG, 40 * NS_PER_MS),
            id="oversize",
        ),
        pytest.param(lambda world, old: world.lost(old, 1000, None, 40 * NS_PER_MS), id="close"),
        pytest.param(
            lambda world, old: world.end(old, CloseReason.IDLE_TIMEOUT, 40 * NS_PER_MS), id="end"
        ),
    ],
)
def test_what_a_replaced_session_returns_later_counts_nothing_and_leaves_the_new_one_open(
    late: Callable[[World, Session], object],
) -> None:
    world = World()
    old = world.open("a")
    new = world.open("a", 30 * NS_PER_MS)
    record = copy.deepcopy(world.registry.get("a"))
    totals = copy.deepcopy(world.totals)

    late(world, old)

    assert (world.registry.get("a"), world.totals) == (record, totals)
    assert (world.table.sessions["a"] is new, new.ended, world.wakes[new].calls) == (
        True,
        False,
        0,
    )
    assert old.end_reason is CloseReason.REPLACED


@pytest.mark.parametrize(
    ("shared_clock", "rtt_ns", "transport_up_ns"),
    [
        pytest.param(True, None, NS_PER_MS, id="shared-clock"),
        pytest.param(True, 9 * NS_PER_MS, NS_PER_MS, id="shared-clock-ignores-the-round-trip"),
        pytest.param(False, 2_000_001, 1_000_000, id="half-the-round-trip"),
        pytest.param(False, None, 0, id="no-round-trip-yet"),
    ],
)
def test_an_accepted_frame_goes_into_the_mailbox_with_its_times_and_a_view_of_its_jpeg(
    shared_clock: bool, rtt_ns: int | None, transport_up_ns: int
) -> None:
    world = World(replace(CONFIG, shared_clock=shared_clock))
    session = world.open("a")
    received_ns = 50 * NS_PER_MS
    message = frame_at(session.number, 7, received_ns - 3 * NS_PER_MS, received_ns - NS_PER_MS)

    accepted = world.ingest(session, message, received_ns, rtt_ns)

    head = world.mailbox.head("a", "depth")
    record = session.record
    assert accepted is True
    assert head == Frame(
        7, received_ns - 3 * NS_PER_MS, received_ns, 2 * NS_PER_MS, transport_up_ns, JPEG
    )
    assert isinstance(head.jpeg, memoryview)
    assert head.jpeg.obj is message
    assert world.mailbox.head("a", "cls") is head
    assert (record.frames_in, record.frames_accepted, session.last_frame_number) == (1, 1, 7)
    assert record.bytes_in == len(hello_text("a")) + len(message)


def test_a_frame_over_the_rate_limit_is_rejected_and_ends_the_session() -> None:
    world = World(ONE_MESSAGE_PER_S)
    session = world.open("a")
    first = good_frame(session, 1, 10 * NS_PER_MS)
    second = good_frame(session, 2, 10 * NS_PER_MS)

    accepted = [
        world.ingest(session, first, 10 * NS_PER_MS),
        world.ingest(session, second, 10 * NS_PER_MS),
    ]

    record = session.record
    assert accepted == [True, False]
    assert (record.frames_in, record.frames_rejected, record.frames_accepted) == (2, 1, 1)
    assert world.totals.frames_rejected == frames_rejected(rate_limited=1)
    assert (session.end_reason, world.totals.closed) == (
        CloseReason.INGRESS_LIMIT,
        closed(ingress_limit=1),
    )
    assert record.bytes_in == len(hello_text("a")) + len(first) + len(second)
    assert world.wakes[session].calls == 1


# The bad message arrives at BAD_RECEIVED_NS, after frame 5 was accepted, and frame 6 follows it.
BAD_RECEIVED_NS = 1_000 * NS_PER_MS
BAD_CAPTURE_NS = BAD_RECEIVED_NS - 3 * NS_PER_MS
BAD_SEND_NS = BAD_RECEIVED_NS - NS_PER_MS


@pytest.mark.parametrize(
    "bad_frame",
    [
        pytest.param(lambda number: b"\x01\x00\x00", id="shorter-than-its-header"),
        pytest.param(
            lambda number: b"\x02" + frame_at(number, 6, BAD_CAPTURE_NS, BAD_SEND_NS)[1:],
            id="not-a-frame",
        ),
        pytest.param(
            lambda number: (
                frame_at(number, 6, BAD_CAPTURE_NS, BAD_SEND_NS)[:1]
                + b"\x01"
                + frame_at(number, 6, BAD_CAPTURE_NS, BAD_SEND_NS)[2:]
            ),
            id="reserved-field-set",
        ),
        pytest.param(
            lambda number: frame_at(number, 6, BAD_CAPTURE_NS, BAD_SEND_NS, b"GIF89a"),
            id="no-jpeg-marker",
        ),
        pytest.param(
            lambda number: frame_at(number + 1, 6, BAD_CAPTURE_NS, BAD_SEND_NS),
            id="another-session",
        ),
        pytest.param(
            lambda number: frame_at(number, 5, BAD_CAPTURE_NS, BAD_SEND_NS),
            id="frame-number-repeated",
        ),
        pytest.param(
            lambda number: frame_at(number, 4, BAD_CAPTURE_NS, BAD_SEND_NS),
            id="frame-number-lower",
        ),
        pytest.param(
            lambda number: frame_at(number, 6, BAD_SEND_NS, BAD_CAPTURE_NS),
            id="sent-before-capture",
        ),
        pytest.param(
            lambda number: frame_at(number, 6, BAD_SEND_NS - MAX_AGE_NS - 1, BAD_SEND_NS),
            id="fire-lag-over-the-age-limit",
        ),
        pytest.param(
            lambda number: frame_at(
                number, 6, BAD_RECEIVED_NS - MAX_AGE_NS - 1, BAD_RECEIVED_NS - MAX_AGE_NS // 2
            ),
            id="captured-before-the-age-limit",
        ),
        pytest.param(
            lambda number: frame_at(number, 6, BAD_CAPTURE_NS, BAD_RECEIVED_NS + 1),
            id="sent-after-receipt",
        ),
    ],
)
def test_a_malformed_frame_is_dropped_and_the_next_good_frame_is_accepted(
    bad_frame: Callable[[int], bytes],
) -> None:
    world = World(AGE_CONFIG)
    session = world.open("a")
    world.ingest(session, good_frame(session, 5, 500 * NS_PER_MS), 500 * NS_PER_MS)

    accepted = [
        world.ingest(session, bad_frame(session.number), BAD_RECEIVED_NS),
        world.ingest(session, good_frame(session, 6, 2_000 * NS_PER_MS), 2_000 * NS_PER_MS),
    ]

    record = session.record
    assert accepted == [False, True]
    assert (record.frames_in, record.frames_rejected, record.frames_accepted) == (3, 1, 2)
    assert world.totals.frames_rejected == frames_rejected(malformed=1)
    assert (session.ended, session.last_frame_number) == (False, 6)
    assert world.mailbox.head("a", "cls").frame_number == 6
    assert world.totals.closed == closed()


@pytest.mark.parametrize(
    ("shared_clock", "accepted"), [(True, False), (False, True)], ids=["shared", "unshared"]
)
@pytest.mark.parametrize(
    ("capture_ns", "send_ns"),
    [
        (BAD_CAPTURE_NS, BAD_RECEIVED_NS + 1),
        (BAD_RECEIVED_NS - MAX_AGE_NS - 1, BAD_RECEIVED_NS - MAX_AGE_NS // 2),
    ],
    ids=["sent-after-receipt", "captured-before-the-age-limit"],
)
def test_frame_times_are_compared_with_the_gateway_clock_only_when_it_is_shared(
    shared_clock: bool, accepted: bool, capture_ns: int, send_ns: int
) -> None:
    world = World(replace(AGE_CONFIG, shared_clock=shared_clock))
    session = world.open("a")

    result = world.ingest(
        session, frame_at(session.number, 1, capture_ns, send_ns), BAD_RECEIVED_NS
    )

    assert (result, session.record.frames_accepted, session.ended) == (
        accepted,
        int(accepted),
        False,
    )


@pytest.mark.parametrize(
    ("fire_lag_ns", "accepted"),
    [(MAX_AGE_NS, True), (MAX_AGE_NS + 1, False)],
    ids=["at-the-limit", "over-the-limit"],
)
def test_the_age_limit_of_a_frame_is_the_configured_one(fire_lag_ns: int, accepted: bool) -> None:
    world = World(replace(AGE_CONFIG, shared_clock=False))
    session = world.open("a")

    result = world.ingest(session, frame_at(session.number, 1, 0, fire_lag_ns), BAD_RECEIVED_NS)

    assert result is accepted


def test_telemetry_is_stored_as_published_and_the_round_trip_is_read_with_each_report() -> None:
    world = World()
    session = world.open("a")
    later_telemetry = replace(TELEMETRY, ts_mono_ns=TELEMETRY.ts_mono_ns + NS_PER_S)
    first = encode_telemetry(TELEMETRY)
    later = encode_telemetry(later_telemetry)

    stored = [world.ingest(session, first, 10 * NS_PER_MS, 2 * NS_PER_MS)]
    after_first = (session.record.telemetry, session.record.rtt_ns)
    stored.append(world.ingest(session, later, 1_010 * NS_PER_MS, None))

    record = session.record
    assert stored == [False, False]
    assert after_first == (telemetry_to_dict(TELEMETRY), 2 * NS_PER_MS)
    assert (record.telemetry, record.rtt_ns) == (telemetry_to_dict(later_telemetry), None)
    assert (record.frames_in, session.ended) == (0, False)
    assert record.bytes_in == len(hello_text("a")) + len(first) + len(later)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param("not json", id="not-json"),
        pytest.param(hello_text("a"), id="a-second-hello"),
        pytest.param(
            encode_telemetry(
                replace(TELEMETRY, models={"cls": replace(MODEL_TELEMETRY, commands_received=4)})
            ),
            id="commands-that-do-not-add-up",
        ),
    ],
)
def test_malformed_text_ends_the_session_as_a_protocol_error(text: str) -> None:
    world = World()
    session = world.open("a")

    stored = world.ingest(session, text, 10 * NS_PER_MS, NS_PER_MS)

    record = session.record
    assert (stored, session.end_reason, world.totals.closed) == (
        False,
        CloseReason.PROTOCOL_ERROR,
        closed(protocol_error=1),
    )
    assert (record.telemetry, record.rtt_ns, record.frames_in) == (None, None, 0)
    assert record.bytes_in == len(hello_text("a")) + len(text)


def test_a_frame_over_the_rate_limit_is_refused_before_it_is_decoded() -> None:
    world = World(ONE_MESSAGE_PER_S)
    session = world.open("a")
    world.ingest(session, good_frame(session, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)

    world.ingest(session, b"\x01\x00\x00", 10 * NS_PER_MS)

    assert world.totals.frames_rejected == frames_rejected(rate_limited=1)
    assert session.end_reason is CloseReason.INGRESS_LIMIT


def test_text_over_the_rate_limit_ends_the_session_before_it_is_parsed() -> None:
    world = World(ONE_MESSAGE_PER_S)
    session = world.open("a")
    world.ingest(session, good_frame(session, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)

    world.ingest(session, "not json", 10 * NS_PER_MS)

    assert (session.end_reason, world.totals.closed) == (
        CloseReason.INGRESS_LIMIT,
        closed(ingress_limit=1),
    )
    assert world.totals.frames_rejected == frames_rejected()


@pytest.mark.parametrize(
    "message",
    [
        pytest.param(frame_at(1, 1, 7 * NS_PER_MS, 9 * NS_PER_MS), id="frame"),
        pytest.param(encode_telemetry(TELEMETRY), id="telemetry"),
        pytest.param("not json", id="malformed-text"),
    ],
)
def test_a_message_for_a_session_that_has_ended_counts_nothing(message: str | bytes) -> None:
    world = World()
    session = world.open("a")
    world.end(session, CloseReason.SHUTDOWN, 5 * NS_PER_MS)
    record = copy.deepcopy(session.record)
    totals = copy.deepcopy(world.totals)

    accepted = world.ingest(session, message, 10 * NS_PER_MS, NS_PER_MS)

    assert (accepted, session.record, world.totals) == (False, record, totals)
    assert (session.end_reason, world.wakes[session].calls) == (CloseReason.SHUTDOWN, 1)


@pytest.mark.parametrize(
    ("peer_code", "local_code", "reason"),
    [
        pytest.param(1000, None, CloseReason.CLIENT_CLOSE, id="peer-closed"),
        pytest.param(1001, None, CloseReason.CLIENT_CLOSE, id="peer-going-away"),
        pytest.param(None, 1002, CloseReason.PROTOCOL_ERROR, id="protocol-error"),
        pytest.param(None, 1007, CloseReason.PROTOCOL_ERROR, id="text-not-utf8"),
        pytest.param(None, 1011, CloseReason.ABNORMAL, id="keepalive-timeout"),
        pytest.param(None, None, CloseReason.ABNORMAL, id="no-close-frame"),
    ],
)
def test_a_lost_connection_ends_with_the_reason_its_close_codes_give(
    peer_code: int | None, local_code: int | None, reason: CloseReason
) -> None:
    world = World()
    session = world.open("a")

    world.lost(session, peer_code, local_code, 10 * NS_PER_MS)

    assert (session.end_reason, world.totals.closed) == (reason, closed(**{reason.value: 1}))
    assert (session.record.frames_in, world.totals.frames_rejected) == (0, frames_rejected())
    assert world.wakes[session].calls == 1


def test_a_message_too_big_counts_as_an_oversize_frame_and_ends_as_a_protocol_error() -> None:
    world = World()
    session = world.open("a")
    world.ingest(session, good_frame(session, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)

    world.lost(session, None, CloseCode.MESSAGE_TOO_BIG, 20 * NS_PER_MS)

    record = session.record
    assert (record.frames_in, record.frames_rejected, record.frames_accepted) == (2, 1, 1)
    assert world.totals.frames_rejected == frames_rejected(oversize=1)
    assert (session.end_reason, world.totals.closed) == (
        CloseReason.PROTOCOL_ERROR,
        closed(protocol_error=1),
    )


def test_a_message_too_big_after_the_end_counts_nothing() -> None:
    world = World()
    session = world.open("a")
    world.end(session, CloseReason.SLOW_CONSUMER, 10 * NS_PER_MS)

    world.lost(session, None, CloseCode.MESSAGE_TOO_BIG, 20 * NS_PER_MS)

    assert (session.record.frames_in, session.record.frames_rejected) == (0, 0)
    assert world.totals.frames_rejected == frames_rejected()
    assert world.totals.closed == closed(slow_consumer=1)


def test_end_runs_the_close_sequence_then_counts_and_wakes_the_handler_last() -> None:
    world = World()
    # What the handler would find when woken: whether the table still holds the session,
    # whether its record is connected, the count of its reason and the membership calls.
    seen: list[tuple[bool, bool, int, list[int]]] = []
    outcome = world.hello(
        hello_text("a"),
        0,
        lambda: seen.append(
            (
                "a" in world.table.sessions,
                world.registry.get("a").connected,
                world.totals.closed["idle_timeout"],
                list(world.memberships),
            )
        ),
    )
    assert isinstance(outcome, Opened)
    session = outcome.session
    world.ingest(session, good_frame(session, 1, 10 * NS_PER_MS), 10 * NS_PER_MS)
    world.queue_command("cls", 20 * NS_PER_MS)

    world.end(session, CloseReason.IDLE_TIMEOUT, 30 * NS_PER_MS)

    counters = world.registry.get("a").models
    assert seen == [(False, False, 1, [1, 0])]
    assert counters["depth"].frames_dropped_disconnect == 1
    assert counters["cls"].results_dropped_disconnect == 1
    assert (world.registry.get("a").disconnected_ns, len(session.downlink)) == (30 * NS_PER_MS, 0)


def test_a_session_ends_once_and_the_first_reason_wins() -> None:
    world = World()
    session = world.open("a")

    world.end(session, CloseReason.IDLE_TIMEOUT, 10 * NS_PER_MS)
    world.end(session, CloseReason.SHUTDOWN, 20 * NS_PER_MS)
    world.lost(session, None, None, 30 * NS_PER_MS)

    assert session.end_reason is CloseReason.IDLE_TIMEOUT
    assert world.totals.closed == closed(idle_timeout=1)
    assert (world.wakes[session].calls, world.memberships) == (1, [1, 0])
    assert world.registry.get("a").disconnected_ns == 10 * NS_PER_MS


# Three robot ids under a cap of three, and a limit of two messages per second against waits
# of at most a quarter second: a run meets replacements, rate limits and every kind of end
# without a refused hello.
MACHINE_CONFIG = replace(
    CONFIG, max_connections=3, ingress_max_messages_per_s=2.0, max_frame_age_ms=100.0
)
MACHINE_WAIT_NS = NS_PER_S // 4
END_REASONS = st.sampled_from(CloseReason)
CLOSE_CODE_PAIRS = st.sampled_from(
    [(1000, None), (None, 1002), (None, 1007), (None, 1009), (None, 1011), (None, None)]
)


class SessionMachine(RuleBasedStateMachine):
    """Hellos, messages and ends of sessions, interleaved at random and checked after each step.

    Every session ever opened stays addressable, so messages and ends also reach sessions that
    a replacement or an end has already closed.
    """

    def __init__(self) -> None:
        super().__init__()
        self.world = World(MACHINE_CONFIG)
        self.opened: list[Session] = []
        self.now_ns = NS_PER_S

    @rule(robot_id=st.sampled_from("abc"))
    def hello(self, robot_id: str) -> None:
        self.opened.append(self.world.open(robot_id, self.now_ns))

    @precondition(lambda self: self.opened)
    @rule(index=st.integers(min_value=0), step=st.integers(min_value=0, max_value=2))
    def frame(self, index: int, step: int) -> None:
        """Send a frame numbered step above the session's last, so step 0 repeats a number."""
        session = self.opened[index % len(self.opened)]
        frame_number = (session.last_frame_number or 0) + step
        self.world.ingest(session, good_frame(session, frame_number, self.now_ns), self.now_ns)

    @precondition(lambda self: self.opened)
    @rule(index=st.integers(min_value=0), text=st.sampled_from(["telemetry", "garbage"]))
    def text(self, index: int, text: str) -> None:
        message = {"telemetry": encode_telemetry(TELEMETRY), "garbage": "not json"}[text]
        self.world.ingest(self.opened[index % len(self.opened)], message, self.now_ns, NS_PER_MS)

    @precondition(lambda self: self.opened)
    @rule(index=st.integers(min_value=0), codes=CLOSE_CODE_PAIRS)
    def lost(self, index: int, codes: tuple[int | None, int | None]) -> None:
        self.world.lost(self.opened[index % len(self.opened)], *codes, self.now_ns)

    @precondition(lambda self: self.opened)
    @rule(index=st.integers(min_value=0), reason=END_REASONS)
    def end(self, index: int, reason: CloseReason) -> None:
        self.world.end(self.opened[index % len(self.opened)], reason, self.now_ns)

    @precondition(lambda self: self.world.mailbox.pending_robots("cls"))
    @rule()
    def serve_cls(self) -> None:
        self.world.queue_command("cls", self.now_ns)

    @precondition(lambda self: self.world.mailbox.pending_robots("depth"))
    @rule()
    def serve_depth(self) -> None:
        self.world.queue_command("depth", self.now_ns)

    @rule(elapsed_ns=st.integers(min_value=0, max_value=MACHINE_WAIT_NS))
    def wait(self, elapsed_ns: int) -> None:
        self.now_ns += elapsed_ns


def test_every_interleaving_of_steps_keeps_the_counts_and_the_table_consistent() -> None:
    run_state_machine_as_test(
        SessionMachine, settings=settings(max_examples=100, stateful_step_count=40)
    )
