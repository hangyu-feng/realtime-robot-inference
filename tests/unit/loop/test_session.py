"""Tests for the coroutine of one robot connection, on virtual time over in-memory connections."""

import asyncio
import functools
import logging
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, replace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cloud.config import ServerConfig
from cloud.counters import REJECT_REASONS, CloseReason, ConnectionTotals
from cloud.downlink import QueuedCommand
from cloud.mailbox import Mailbox
from cloud.registry import RobotRegistry
from cloud.session import SessionContext, run_connection
from cloud.sessions import Session, SessionTable
from common.commands import MODEL_KEYS
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    CloseCode,
    FrameHeader,
    Hello,
    Welcome,
    decode_welcome,
    encode_frame,
    encode_hello,
)
from common.transport import Connection, TransportClosed
from tests.support.memory_transport import MemoryConnection, PipeLimits, memory_pair
from tests.support.virtual_time import LoopClock, run_virtual

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
CONFIG = ServerConfig()
TOKEN_CONFIG = replace(CONFIG, shared_token="secret")
ONE_CONNECTION = replace(CONFIG, max_connections=1)
# One message per second, so that a second frame at the same instant is over the limit.
ONE_MESSAGE_PER_S = replace(CONFIG, ingress_max_messages_per_s=1.0)
UNSHARED_CLOCK = replace(CONFIG, shared_clock=False)
HANDSHAKE_NS = round(CONFIG.handshake_timeout_ms * NS_PER_MS)
IDLE_MS = round(CONFIG.idle_timeout_ms)
IDLE_S = CONFIG.idle_timeout_ms / 1000
IDLE_NS = round(CONFIG.idle_timeout_ms * NS_PER_MS)
SLOW_CONSUMER_NS = round(CONFIG.slow_consumer_timeout_ms * NS_PER_MS)
JPEG = b"\xff\xd8\xff\xe0jpeg"
COMMAND_BYTES = 64
# The robot's end. Its inbox takes two commands; after that the gateway's bytes stay unsent.
ROBOT_END = PipeLimits(
    capacity_bytes=2 * COMMAND_BYTES,
    write_limit_bytes=65_536,
    max_message_bytes=4_096,
    close_timeout_ns=NS_PER_S,
)
# The gateway's end, with its write limit of 0.
GATEWAY_END = PipeLimits(
    capacity_bytes=65_536,
    write_limit_bytes=0,
    max_message_bytes=1_000,
    close_timeout_ns=round(CONFIG.close_timeout_ms * NS_PER_MS),
    receive_queue_frames=CONFIG.receive_queue_frames,
)
# What the gateway holds for a robot that has stopped reading: its downlink queue, the command
# the kernel refused and the one command the drain then waits on.
HELD_BOUND_BYTES = (CONFIG.downlink_queue_commands + 2) * COMMAND_BYTES
BROKEN_ROBOT = "broken"


def hello_text(
    robot_id: str = "a",
    *,
    token: str | None = None,
    protocol_version: int = PROTOCOL_VERSION,
    models: tuple[str, ...] = MODEL_KEYS,
) -> str:
    return encode_hello(
        Hello(
            protocol_version=protocol_version,
            robot_id=robot_id,
            token=token,
            models=models,
            stale_ms=dict.fromkeys(models, 150.0),
            watchdog_ms=dict.fromkeys(models, 300.0),
            frame_rate_hz=30.0,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
    )


def frame(session_number: int, frame_number: int, now_ns: int) -> bytes:
    """Return a frame captured and sent at now_ns."""
    return encode_frame(FrameHeader(session_number, frame_number, now_ns, now_ns), JPEG)


def command(number: int) -> bytes:
    """Return a command message of COMMAND_BYTES bytes that carries its number."""
    return number.to_bytes(COMMAND_BYTES)


def deliver(session: Session, data: bytes) -> None:
    """Queue a command for the session and wake it, as a dispatcher does after a batch."""
    session.downlink.push(QueuedCommand(data, session.record.models["cls"]))
    session.wake()


def closed(**counts: int) -> dict[str, int]:
    """Return connections.closed with these counts and every other reason at 0."""
    return {reason.value: 0 for reason in CloseReason} | counts


def rejected(**counts: int) -> dict[str, int]:
    """Return connections.rejected with these counts and every other reason at 0."""
    return dict.fromkeys(REJECT_REASONS, 0) | counts


def codes(closed_error: TransportClosed) -> tuple[int | None, int | None]:
    return closed_error.peer_code, closed_error.local_code


def error_lines(caplog: pytest.LogCaptureFixture) -> list[tuple[str, bool]]:
    """Return each logged error as its message and whether it carries a traceback."""
    return [
        (record.getMessage(), record.exc_info is not None)
        for record in caplog.records
        if record.levelno >= logging.ERROR
    ]


async def read_to_end(connection: MemoryConnection) -> tuple[list[str | bytes], TransportClosed]:
    """Read until recv() raises; return the messages read and what it raised."""
    messages: list[str | bytes] = []
    while True:
        try:
            messages.append(await connection.recv())
        except TransportClosed as closed_error:
            return messages, closed_error


async def settle() -> None:
    """Let every task run until all of them wait: only then does virtual time move, by 1 ms."""
    await asyncio.sleep(0.001)


async def close_normally(client: MemoryConnection) -> None:
    await client.close(CloseCode.NORMAL, "")


async def drop(client: MemoryConnection) -> None:
    client.abort()


# How a peer leaves, and the reason its session then ends with.
LEAVES = [
    pytest.param(close_normally, CloseReason.CLIENT_CLOSE, id="close"),
    pytest.param(drop, CloseReason.ABNORMAL, id="abort"),
]


def as_it_is(end: MemoryConnection) -> Connection:
    return end


class BrokenSendNowait:
    """A connection whose synchronous write fails as a fault in the gateway's own code would."""

    def __init__(self, inner: MemoryConnection) -> None:
        self._inner = inner

    async def recv(self) -> str | bytes:
        return await self._inner.recv()

    async def send(self, message: str | bytes) -> None:
        await self._inner.send(message)

    def send_nowait(self, message: str | bytes) -> None:
        raise RuntimeError("the synchronous write is broken")

    def unsent_bytes(self) -> int:
        return self._inner.unsent_bytes()

    def rtt_ns(self) -> int | None:
        return self._inner.rtt_ns()

    async def close(self, code: int, reason: str) -> None:
        await self._inner.close(code, reason)

    def abort(self) -> None:
        self._inner.abort()


class TableWithBrokenIngest(SessionTable):
    """A session table whose ingress step fails for one robot, as a fault in the gateway would."""

    def ingest(
        self, session: Session, message: str | bytes, received_ns: int, rtt_ns: int | None
    ) -> bool:
        if session.robot_id == BROKEN_ROBOT:
            raise RuntimeError("the ingress step is broken")
        return super().ingest(session, message, received_ns, rtt_ns)


class TableThatRecordsIngest(SessionTable):
    """A session table that records the session number of every call of its ingress step."""

    def __init__(
        self,
        config: ServerConfig,
        registry: RobotRegistry,
        mailbox: Mailbox,
        totals: ConnectionTotals,
        on_membership_change: Callable[[], None],
    ) -> None:
        super().__init__(config, registry, mailbox, totals, on_membership_change)
        self.ingested: list[int] = []

    def ingest(
        self, session: Session, message: str | bytes, received_ns: int, rtt_ns: int | None
    ) -> bool:
        self.ingested.append(session.number)
        return super().ingest(session, message, received_ns, rtt_ns)


@dataclass(frozen=True)
class Peer:
    """One connection: the robot's end, the gateway's end and the gateway's handler task."""

    client: MemoryConnection
    server: MemoryConnection
    handler: asyncio.Task[None]


@dataclass(frozen=True)
class Robot:
    """A connection with an open session, and the welcome that opened it."""

    client: MemoryConnection
    server: MemoryConnection
    handler: asyncio.Task[None]
    session: Session
    welcome: str


class World:
    """The connection side of a gateway over a real session table, registry, mailbox and totals."""

    def __init__(
        self, config: ServerConfig = CONFIG, table_type: type[SessionTable] = SessionTable
    ) -> None:
        self.clock = LoopClock()
        self.registry = RobotRegistry(
            round(config.robot_retention_s * NS_PER_S), config.max_robot_entries
        )
        self.mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        self.totals = ConnectionTotals()
        self.table = table_type(config, self.registry, self.mailbox, self.totals, lambda: None)
        # The model of every poke, in the order the pokes came.
        self.poked: list[str] = []
        self.accepting = True
        self.context = SessionContext(
            config=config,
            clock=self.clock,
            table=self.table,
            totals=self.totals,
            pokes={model: functools.partial(self.poked.append, model) for model in MODEL_KEYS},
            accepting=lambda: self.accepting,
        )

    def connect(
        self,
        handlers: asyncio.TaskGroup,
        gateway_end: Callable[[MemoryConnection], Connection] = as_it_is,
        robot_end: PipeLimits = ROBOT_END,
    ) -> Peer:
        """Start a handler on the gateway's end of a new connection."""
        client, server = memory_pair(robot_end, GATEWAY_END)
        handler = handlers.create_task(run_connection(gateway_end(server), self.context))
        return Peer(client, server, handler)

    async def open(
        self,
        handlers: asyncio.TaskGroup,
        robot_id: str = "a",
        *,
        models: tuple[str, ...] = MODEL_KEYS,
        gateway_end: Callable[[MemoryConnection], Connection] = as_it_is,
    ) -> Robot:
        """Connect a robot, say hello and read its welcome."""
        peer = self.connect(handlers, gateway_end)
        peer.client.send_nowait(hello_text(robot_id, models=models))
        welcome = await peer.client.recv()
        assert isinstance(welcome, str)
        return Robot(peer.client, peer.server, peer.handler, self.table.sessions[robot_id], welcome)

    def check(self) -> None:
        """Every accepted session is open or has ended once: accepted = current + sum(closed)."""
        assert self.totals.accepted == self.registry.connected_count() + sum(
            self.totals.closed.values()
        )


@pytest.fixture
def make_world() -> Iterator[Callable[..., World]]:
    """Build the worlds of a test, and check each one's connection identity after the test."""
    worlds: list[World] = []

    def make(config: ServerConfig = CONFIG, table_type: type[SessionTable] = SessionTable) -> World:
        world = World(config, table_type)
        worlds.append(world)
        return world

    yield make
    for world in worlds:
        world.check()


async def test_a_good_hello_is_answered_with_the_welcome_of_session_one(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        connected = world.registry.get("a").connected
        await robot.client.close(CloseCode.NORMAL, "")

    assert decode_welcome(robot.welcome) == Welcome(PROTOCOL_VERSION, 1, CONFIG.shared_clock)
    assert (connected, world.totals.accepted) == (True, 1)


@pytest.mark.parametrize(
    ("config", "message", "label", "code"),
    [
        pytest.param(CONFIG, b"\x01binary", "bad_hello", CloseCode.PROTOCOL_ERROR, id="binary"),
        pytest.param(CONFIG, "{}", "bad_hello", CloseCode.PROTOCOL_ERROR, id="not_a_hello"),
        pytest.param(
            CONFIG,
            hello_text(protocol_version=PROTOCOL_VERSION + 1),
            "version",
            CloseCode.POLICY,
            id="version",
        ),
        pytest.param(
            CONFIG, hello_text(models=("pose",)), "unknown_model", CloseCode.POLICY, id="model"
        ),
        pytest.param(TOKEN_CONFIG, hello_text(token="wrong"), "auth", CloseCode.POLICY, id="auth"),
    ],
)
async def test_a_refused_hello_reaches_the_peer_as_its_close_code_and_counts_once(
    make_world: Callable[..., World],
    config: ServerConfig,
    message: str | bytes,
    label: str,
    code: CloseCode,
) -> None:
    world = make_world(config)

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers)
        peer.client.send_nowait(message)
        messages, at_client = await read_to_end(peer.client)

    assert (messages, codes(at_client)) == ([], (code, None))
    assert (world.totals.rejected, world.totals.accepted) == (rejected(**{label: 1}), 0)


async def test_a_hello_over_the_connection_cap_is_closed_with_1013_and_the_open_session_stays(
    make_world: Callable[..., World],
) -> None:
    world = make_world(ONE_CONNECTION)

    async with asyncio.TaskGroup() as handlers:
        first = await world.open(handlers, "a")
        second = world.connect(handlers)
        second.client.send_nowait(hello_text("b"))
        messages, at_second = await read_to_end(second.client)
        first_ended = first.session.ended
        await first.client.close(CloseCode.NORMAL, "")

    assert (messages, codes(at_second)) == ([], (CloseCode.OVER_CAPACITY, None))
    assert (world.totals.rejected, first_ended) == (rejected(capacity=1), False)


async def test_no_hello_within_the_handshake_timeout_is_closed_with_4003_at_the_timeout(
    make_world: Callable[..., World],
) -> None:
    world = make_world()
    start_ns = world.clock.now_ns()

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers)
        messages, at_client = await read_to_end(peer.client)
        waited_ns = world.clock.now_ns() - start_ns

    assert (messages, codes(at_client)) == ([], (CloseCode.IDLE_TIMEOUT, None))
    assert (waited_ns, world.totals.rejected) == (HANDSHAKE_NS, rejected(handshake_timeout=1))


@pytest.mark.parametrize("leave", [close_normally, drop])
async def test_a_peer_that_leaves_before_its_hello_counts_nothing(
    make_world: Callable[..., World], leave: Callable[[MemoryConnection], Awaitable[None]]
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers)
        await settle()
        await leave(peer.client)

    assert world.totals == ConnectionTotals()


async def test_a_hello_after_shutdown_has_begun_is_closed_with_1001_and_counts_nothing(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers)
        await settle()
        world.accepting = False
        peer.client.send_nowait(hello_text())
        messages, at_client = await read_to_end(peer.client)

    assert (messages, codes(at_client)) == ([], (CloseCode.SERVER_SHUTDOWN, None))
    assert world.totals == ConnectionTotals()


@pytest.mark.parametrize("models", [("depth",), MODEL_KEYS], ids=["depth", "both"])
async def test_a_frame_is_stamped_when_it_arrives_and_pokes_the_models_its_robot_subscribes_to(
    make_world: Callable[..., World], models: tuple[str, ...]
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers, models=models)
        await asyncio.sleep(0.25)
        arrival_ns = world.clock.now_ns()
        robot.client.send_nowait(frame(robot.session.number, 1, arrival_ns))
        await settle()
        received_ns = [world.mailbox.head("a", model).received_ns for model in models]
        robot.client.send_nowait(frame(robot.session.number, 2, world.clock.now_ns()))
        await settle()
        await robot.client.close(CloseCode.NORMAL, "")

    assert received_ns == [arrival_ns] * len(models)
    # Each subscribed model is poked once for each frame accepted.
    assert world.poked == [*models, *models]


async def test_the_ingress_step_gets_the_last_ping_round_trip_of_the_connection(
    make_world: Callable[..., World],
) -> None:
    world = make_world(UNSHARED_CLOCK)

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        robot.server.set_rtt_ns(4 * NS_PER_MS)
        robot.client.send_nowait(frame(robot.session.number, 1, world.clock.now_ns()))
        await settle()
        transport_up_ns = world.mailbox.head("a", "cls").transport_up_ns
        await robot.client.close(CloseCode.NORMAL, "")

    assert transport_up_ns == 2 * NS_PER_MS


async def test_a_malformed_frame_is_dropped_and_the_next_good_frame_is_accepted(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        now_ns = world.clock.now_ns()
        robot.client.send_nowait(frame(robot.session.number + 1, 1, now_ns))
        robot.client.send_nowait(frame(robot.session.number, 2, now_ns))
        await settle()
        record = robot.session.record
        counted = (record.frames_in, record.frames_rejected, record.frames_accepted)
        ended = robot.session.ended
        await robot.client.close(CloseCode.NORMAL, "")

    assert (counted, ended) == ((2, 1, 1), False)
    assert world.totals.frames_rejected["malformed"] == 1


async def test_a_frame_over_the_ingress_limit_closes_the_connection_with_1008(
    make_world: Callable[..., World],
) -> None:
    world = make_world(ONE_MESSAGE_PER_S)

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        now_ns = world.clock.now_ns()
        robot.client.send_nowait(frame(robot.session.number, 1, now_ns))
        robot.client.send_nowait(frame(robot.session.number, 2, now_ns))
        messages, at_client = await read_to_end(robot.client)

    assert (messages, codes(at_client)) == ([], (CloseCode.POLICY, None))
    assert world.totals.frames_rejected["rate_limited"] == 1
    assert world.totals.closed == closed(ingress_limit=1)


async def test_an_oversized_message_closes_with_1009_and_counts_oversize_and_protocol_error(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        robot.client.send_nowait(bytes(GATEWAY_END.max_message_bytes + 1))
        messages, at_client = await read_to_end(robot.client)

    assert (messages, codes(at_client)) == ([], (CloseCode.MESSAGE_TOO_BIG, None))
    assert world.totals.frames_rejected["oversize"] == 1
    assert world.totals.closed == closed(protocol_error=1)


@pytest.mark.parametrize(
    ("message_times", "closed_at"),
    [
        pytest.param((), 1.25, id="no_message"),
        pytest.param((0.24,), 1.25, id="a_message_in_the_first_quarter"),
        pytest.param((0.25,), 1.5, id="a_message_at_the_first_quarter"),
        pytest.param((0.25, 0.49), 1.5, id="a_message_within_a_quarter_of_the_last_move"),
        pytest.param((0.25, 0.5), 1.75, id="a_message_a_quarter_after_the_last_move"),
    ],
)
async def test_the_idle_deadline_moves_only_with_a_message_a_quarter_idle_timeout_after_it_was_set(
    make_world: Callable[..., World], message_times: tuple[float, ...], closed_at: float
) -> None:
    """Times are in idle timeouts, counted from the start of the session."""
    world = make_world()
    loop = asyncio.get_running_loop()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        start_s = loop.time()
        start_ns = world.clock.now_ns()
        for frame_number, message_time in enumerate(message_times, start=1):
            await asyncio.sleep(start_s + message_time * IDLE_S - loop.time())
            robot.client.send_nowait(
                frame(robot.session.number, frame_number, world.clock.now_ns())
            )
        messages, at_client = await read_to_end(robot.client)
        closed_ns = world.clock.now_ns() - start_ns

    assert (messages, codes(at_client)) == ([], (CloseCode.IDLE_TIMEOUT, None))
    assert (closed_ns, world.totals.closed) == (round(closed_at * IDLE_NS), closed(idle_timeout=1))


async def silence_until_closed(gaps_ms: list[int]) -> tuple[int, int | None]:
    """Send a frame after each gap, then nothing more.

    Return how long after the last message the gateway closed the connection, and its code.
    """
    world = World()
    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        for frame_number, gap_ms in enumerate(gaps_ms, start=1):
            await asyncio.sleep(gap_ms / 1000)
            robot.client.send_nowait(
                frame(robot.session.number, frame_number, world.clock.now_ns())
            )
        last_ns = world.clock.now_ns()
        _, at_client = await read_to_end(robot.client)
        silent_ns = world.clock.now_ns() - last_ns
    world.check()
    return silent_ns, at_client.peer_code


@settings(max_examples=60)
@given(gaps_ms=st.lists(st.integers(1, IDLE_MS - 1), max_size=10))
def test_a_silent_session_is_closed_as_idle_between_one_and_one_and_a_quarter_idle_timeouts(
    gaps_ms: list[int],
) -> None:
    silent_ns, code = run_virtual(silence_until_closed(gaps_ms))

    assert code == CloseCode.IDLE_TIMEOUT
    assert IDLE_NS <= silent_ns <= 5 * IDLE_NS // 4


async def test_a_session_that_keeps_sending_is_never_closed_as_idle(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        # Two hours of frames, each sent just before the idle timeout would be reached.
        for frame_number in range(1, 2_668):
            await asyncio.sleep(0.9 * IDLE_S)
            robot.client.send_nowait(
                frame(robot.session.number, frame_number, world.clock.now_ns())
            )
        await settle()
        ended = robot.session.ended
        accepted = robot.session.record.frames_accepted
        await robot.client.close(CloseCode.NORMAL, "")

    assert (ended, accepted) == (False, 2_667)


async def test_commands_are_written_in_order_on_the_callers_task_while_the_peer_takes_them(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        counters = robot.session.record.models["cls"]
        deliver(robot.session, command(0))
        deliver(robot.session, command(1))
        # Nothing has been awaited since the deliveries, so only wake() can have written.
        written = (counters.commands_sent, len(robot.session.downlink), robot.server.unsent_bytes())
        received = [await robot.client.recv(), await robot.client.recv()]
        await robot.client.close(CloseCode.NORMAL, "")

    assert (written, received) == ((2, 0, 0), [command(0), command(1)])


async def test_once_the_handler_has_drained_the_queue_commands_are_written_at_once_again(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        counters = robot.session.record.models["cls"]
        # The kernel refuses the third command, and the fourth waits for the handler's drain.
        for number in range(4):
            deliver(robot.session, command(number))
        drained = [await robot.client.recv() for _ in range(4)]
        await settle()
        deliver(robot.session, command(4))
        # Nothing has been awaited since this delivery, so only wake() can have written it.
        written = (counters.commands_sent, len(robot.session.downlink))
        last = await robot.client.recv()
        await robot.client.close(CloseCode.NORMAL, "")

    assert (written, [*drained, last]) == ((5, 0), [command(number) for number in range(5)])


async def test_a_peer_that_never_takes_its_welcome_is_aborted_at_the_slow_consumer_timeout(
    make_world: Callable[..., World],
) -> None:
    world = make_world()
    start_ns = world.clock.now_ns()

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers, robot_end=replace(ROBOT_END, capacity_bytes=0))
        peer.client.send_nowait(hello_text())
        messages, at_client = await read_to_end(peer.client)
        ended_ns = world.clock.now_ns() - start_ns

    assert (messages, codes(at_client), ended_ns) == ([], (None, None), SLOW_CONSUMER_NS)
    assert (world.totals.accepted, world.totals.closed) == (1, closed(slow_consumer=1))


async def test_a_peer_that_stops_reading_is_aborted_at_the_slow_consumer_timeout(
    make_world: Callable[..., World],
) -> None:
    world = make_world()
    held_bytes: list[int] = []

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        counters = robot.session.record.models["cls"]
        start_ns = world.clock.now_ns()
        for number in range(16):
            deliver(robot.session, command(number))
            held_bytes.append(
                len(robot.session.downlink) * COMMAND_BYTES + robot.server.unsent_bytes()
            )
            await asyncio.sleep(0.01)
        await asyncio.wait({robot.handler})
        ended_ns = world.clock.now_ns() - start_ns
        messages, at_client = await read_to_end(robot.client)

    # The robot's end took two commands and the kernel refused the third, so the fourth, 30 ms
    # in, was the first to wait for the handler's drain, whose timer started then.
    assert ended_ns == 30 * NS_PER_MS + SLOW_CONSUMER_NS
    assert (messages, codes(at_client)) == ([command(0), command(1)], (None, None))
    assert world.totals.closed == closed(slow_consumer=1)
    assert (
        counters.commands_sent,
        counters.commands_dropped_downlink,
        counters.results_dropped_disconnect,
        counters.commands_queued,
    ) == (4, 4, 8, 0)
    assert max(held_bytes) <= HELD_BOUND_BYTES


async def test_bytes_out_counts_the_welcome_and_every_command_written_at_once_or_by_the_drain(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        # The robot's end takes two commands, the kernel refuses the third, and the other three
        # wait for the handler's drain.
        for number in range(6):
            deliver(robot.session, command(number))
        left_to_drain = len(robot.session.downlink)
        received = [await robot.client.recv() for _ in range(6)]
        await robot.client.close(CloseCode.NORMAL, "")

    expected_bytes = len(robot.welcome) + 6 * COMMAND_BYTES
    assert (left_to_drain, received) == (3, [command(number) for number in range(6)])
    assert (robot.session.record.bytes_out, world.totals.bytes_out) == (
        expected_bytes,
        expected_bytes,
    )


async def test_a_synchronous_write_that_raises_ends_its_session_alone(
    make_world: Callable[..., World], caplog: pytest.LogCaptureFixture
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        broken = await world.open(handlers, "a", gateway_end=BrokenSendNowait)
        healthy = await world.open(handlers, "b")
        # One batch's commands for two sessions, woken one after the other as a dispatcher does.
        deliver(broken.session, command(0))
        deliver(healthy.session, command(1))
        ended_at_once = broken.session.end_reason
        received = await healthy.client.recv()
        _, at_broken = await read_to_end(broken.client)
        await healthy.client.close(CloseCode.NORMAL, "")

    assert (ended_at_once, codes(at_broken)) == (CloseReason.INTERNAL_ERROR, (None, None))
    assert received == command(1)
    assert error_lines(caplog) == [("robot a: writing a command failed", True)]
    assert world.totals.closed == closed(internal_error=1, client_close=1)


async def test_after_a_closed_connection_refused_a_write_no_further_command_is_handed_to_it(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        counters = robot.session.record.models["cls"]
        # The close has reached the gateway's end, and its reader has not run since.
        await robot.client.close(CloseCode.NORMAL, "")
        deliver(robot.session, command(0))
        deliver(robot.session, command(1))

    assert (counters.commands_sent, counters.results_dropped_disconnect) == (1, 1)
    assert robot.session.end_reason is CloseReason.CLIENT_CLOSE


async def test_a_replaced_session_is_closed_with_4001_and_its_queued_frame_is_not_put(
    make_world: Callable[..., World],
) -> None:
    world = make_world(table_type=TableThatRecordsIngest)
    table = world.table
    assert isinstance(table, TableThatRecordsIngest)

    async with asyncio.TaskGroup() as handlers:
        old = await world.open(handlers, "a")
        old.client.send_nowait(frame(old.session.number, 1, world.clock.now_ns()))
        new = world.connect(handlers)
        await settle()
        # The new hello is handled first, so the old connection's second frame is still queued
        # when its session is replaced.
        new.client.send_nowait(hello_text("a"))
        old.client.send_nowait(frame(old.session.number, 2, world.clock.now_ns()))
        messages, at_old = await read_to_end(old.client)
        new_session_number = world.table.sessions["a"].number
        pending = world.mailbox.pending_count("a", "cls")
        await new.client.close(CloseCode.NORMAL, "")

    assert (messages, codes(at_old)) == ([], (CloseCode.SESSION_REPLACED, None))
    assert (table.ingested, new_session_number) == ([old.session.number], old.session.number + 1)
    assert (world.registry.get("a").frames_in, pending) == (1, 0)
    assert world.totals.closed == closed(replaced=1, client_close=1)


async def test_a_session_ended_for_shutdown_is_closed_with_1001(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        await asyncio.sleep(0.25)
        end_ns = world.clock.now_ns()
        world.table.end(robot.session, CloseReason.SHUTDOWN, end_ns)
        messages, at_client = await read_to_end(robot.client)
        await asyncio.wait({robot.handler})
        returned_ns = world.clock.now_ns()

    assert (messages, codes(at_client)) == ([], (CloseCode.SERVER_SHUTDOWN, None))
    assert (returned_ns, world.totals.closed) == (end_ns, closed(shutdown=1))


async def test_an_end_while_bytes_are_unsent_aborts_at_once_and_releases_the_blocked_writer(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        for number in range(4):
            deliver(robot.session, command(number))
        # The handler takes the fourth command and waits in its send behind the third.
        await settle()
        unsent_before = robot.server.unsent_bytes()
        end_ns = world.clock.now_ns()
        world.table.end(robot.session, CloseReason.SHUTDOWN, end_ns)
        unsent_after = robot.server.unsent_bytes()
        await asyncio.wait({robot.handler})
        released_ns = world.clock.now_ns()
        messages, at_client = await read_to_end(robot.client)

    assert (unsent_before, unsent_after, released_ns) == (2 * COMMAND_BYTES, 0, end_ns)
    assert (messages, codes(at_client)) == ([command(0), command(1)], (None, None))
    assert world.totals.closed == closed(shutdown=1)


@pytest.mark.parametrize(("leave", "reason"), LEAVES)
async def test_a_peer_that_leaves_an_open_session_ends_it_with_the_reason_of_its_leaving(
    make_world: Callable[..., World],
    leave: Callable[[MemoryConnection], Awaitable[None]],
    reason: CloseReason,
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        await settle()
        await leave(robot.client)

    assert world.totals.closed == closed(**{reason.value: 1})


@pytest.mark.parametrize(("leave", "reason"), LEAVES)
async def test_a_leaving_that_the_writer_sees_first_ends_the_session_with_the_same_reason(
    make_world: Callable[..., World],
    leave: Callable[[MemoryConnection], Awaitable[None]],
    reason: CloseReason,
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        peer = world.connect(handlers)
        # The peer has gone before the handler first runs, so the welcome meets the closed
        # connection before the reader has started.
        peer.client.send_nowait(hello_text())
        await leave(peer.client)

    assert (world.totals.accepted, world.totals.closed) == (1, closed(**{reason.value: 1}))


async def test_a_reader_that_raises_ends_its_session_as_an_internal_error_and_no_other(
    make_world: Callable[..., World], caplog: pytest.LogCaptureFixture
) -> None:
    world = make_world(table_type=TableWithBrokenIngest)

    async with asyncio.TaskGroup() as handlers:
        broken = await world.open(handlers, BROKEN_ROBOT)
        healthy = await world.open(handlers, "b")
        now_ns = world.clock.now_ns()
        broken.client.send_nowait(frame(broken.session.number, 1, now_ns))
        _, at_broken = await read_to_end(broken.client)
        healthy.client.send_nowait(frame(healthy.session.number, 1, now_ns))
        await settle()
        accepted = healthy.session.record.frames_accepted
        await healthy.client.close(CloseCode.NORMAL, "")

    assert (codes(at_broken), accepted) == ((None, None), 1)
    assert error_lines(caplog) == [("robot broken: the connection's handler failed", True)]
    assert world.totals.closed == closed(internal_error=1, client_close=1)


async def test_cancelling_the_handler_ends_its_session_as_an_internal_error_and_propagates(
    make_world: Callable[..., World],
) -> None:
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        await settle()
        robot.handler.cancel()
        messages, at_client = await read_to_end(robot.client)

    assert (messages, codes(at_client)) == ([], (None, None))
    assert (robot.handler.cancelled(), world.totals.closed) == (True, closed(internal_error=1))


async def test_a_session_logs_its_opening_and_its_end_and_nothing_for_a_frame(
    make_world: Callable[..., World], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="cloud.session")
    world = make_world()

    async with asyncio.TaskGroup() as handlers:
        robot = await world.open(handlers)
        robot.client.send_nowait(frame(robot.session.number, 1, world.clock.now_ns()))
        await settle()
        await robot.client.close(CloseCode.NORMAL, "")

    assert [record.getMessage() for record in caplog.records] == [
        "robot a: session 1 opened",
        "robot a: session 1 ended: client_close",
    ]
