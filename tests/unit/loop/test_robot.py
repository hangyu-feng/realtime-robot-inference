"""Tests for one robot's life, on the virtual-time loop.

The test plays the gateway over in-memory pairs: the robot's connector hands out the robot's
end of each new pair, and the test reads and writes the other end.
"""

import asyncio
import random
from collections import deque
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from cloud.config import ServerConfig
from common.aio import sleep_until
from common.backoff import Backoff
from common.commands import ClsCommand, DepthCommand, encode_command
from common.protocol import (
    PROTOCOL_VERSION,
    CommandHeader,
    FrameHeader,
    Hello,
    StageDurations,
    Telemetry,
    Welcome,
    decode_frame,
    decode_hello,
    decode_telemetry,
    encode_command_message,
    encode_welcome,
)
from common.transport import Connection, TransportClosed, TransportConnectError
from robot.robot import Robot
from robot.samples import SampleLog, load_samples
from robot.schedule import CaptureSchedule
from robot.settings import LinkSettings, RobotSettings
from tests.support.memory_transport import MemoryConnection, PipeLimits, memory_pair
from tests.support.virtual_time import LoopClock

MS = 1_000_000
SECOND = 1_000_000_000
# 20 frames per second: a period of whole milliseconds, so every tick is an instant that
# sleep_until, which sleeps whole milliseconds, reaches exactly.
PERIOD_NS = 50 * MS
SCHEDULE = CaptureSchedule(start_ns=0, period_ns=PERIOD_NS)
JPEGS = (b"\xff\xd8\xff" + b"a" * 61, b"\xff\xd8\xff" + b"b" * 125)
SEED = 11
SETTINGS = RobotSettings(
    models=("cls",),
    stale_ms={"cls": 500.0},
    watchdog_ms={"cls": 1000.0},
    frame_rate_hz=20.0,
    token="secret",
    welcome_timeout_ms=400.0,
    close_timeout_ms=100.0,
    stall_timeout_ms=300.0,
    reconnect_min_ms=100.0,
    reconnect_max_ms=800.0,
)
ROBOT_END = PipeLimits(
    capacity_bytes=1_000_000,
    write_limit_bytes=0,
    max_message_bytes=4096,
    close_timeout_ns=100 * MS,
)
# The test plays the gateway, so its end stops reading where the gateway's library does.
RECEIVE_QUEUE_FRAMES = ServerConfig().receive_queue_frames
GATEWAY_END = PipeLimits(
    capacity_bytes=1_000_000,
    write_limit_bytes=0,
    max_message_bytes=1_000_000,
    close_timeout_ns=SECOND,
    receive_queue_frames=RECEIVE_QUEUE_FRAMES,
)
# A gateway whose buffers hold one message: a second one stays unsent at the robot.
NARROW_GATEWAY_END = PipeLimits(
    capacity_bytes=1,
    write_limit_bytes=0,
    max_message_bytes=1_000_000,
    close_timeout_ns=SECOND,
    receive_queue_frames=RECEIVE_QUEUE_FRAMES,
)


class Gateway:
    """The test's side of every connection the robot opens."""

    def __init__(
        self, limits: PipeLimits = GATEWAY_END, refusals: Sequence[int | None] = ()
    ) -> None:
        self._limits = limits
        # The HTTP status of the answer to each of the first connects; None refuses the
        # connection itself.
        self._refusals = deque(refusals)
        self._opened: asyncio.Queue[MemoryConnection] = asyncio.Queue()
        # The virtual time of every connect the robot tried.
        self.connect_ns: list[int] = []

    async def connect(self) -> Connection:
        self.connect_ns.append(LoopClock().now_ns())
        if self._refusals:
            raise TransportConnectError("refused", self._refusals.popleft())
        robot_end, gateway_end = memory_pair(ROBOT_END, self._limits)
        self._opened.put_nowait(gateway_end)
        return robot_end

    async def next_connection(self) -> MemoryConnection:
        """Return the gateway's end of the robot's next connection."""
        return await self._opened.get()

    async def accept(self, session: int) -> tuple[MemoryConnection, Hello]:
        """Take the robot's next connection, read its hello and welcome it."""
        connection = await self.next_connection()
        hello = decode_hello(str(await connection.recv()))
        connection.send_nowait(encode_welcome(Welcome(PROTOCOL_VERSION, session, True)))
        return connection, hello


def make_robot(
    gateway: Gateway,
    settings: RobotSettings = SETTINGS,
    start_ns: int = 0,
    stop_ns: int = 3600 * SECOND,
    samples: SampleLog | None = None,
    schedule: CaptureSchedule = SCHEDULE,
    seed: int = SEED,
) -> Robot:
    return Robot(
        "r01",
        settings,
        schedule,
        JPEGS,
        start_ns,
        stop_ns,
        gateway.connect,
        LoopClock(),
        random.Random(seed),
        samples,
    )


def pauses_ns(count: int) -> list[int]:
    """Return the robot's first reconnect pauses, as its timers run them.

    A pause is timed in whole milliseconds, rounded up.
    """
    backoff = Backoff(100 * MS, 800 * MS, random.Random(SEED))
    return [-(-backoff.next_ns() // MS) * MS for _ in range(count)]


def cls_command(session: int, frame_number: int, capture_ns: int) -> bytes:
    header = CommandHeader(
        model="cls",
        session=session,
        frame_number=frame_number,
        capture_ns=capture_ns,
        batch_size=1,
        result_ns=capture_ns,
        stages=StageDurations(0, 0, 0, 0, 0, 0, 0, 0),
    )
    return encode_command_message(header, encode_command(ClsCommand(class_id=1, confidence=0.5)))


# A well-formed command of the model that the robots of these tests do not subscribe to.
DEPTH_COMMAND = encode_command_message(
    CommandHeader("depth", 1, 1, 0, 1, 0, StageDurations(0, 0, 0, 0, 0, 0, 0, 0)),
    encode_command(DepthCommand(0.0, 0.0, 0.0, 0.0, 1.0)),
)


async def next_frame(connection: MemoryConnection) -> tuple[int, FrameHeader, bytes]:
    """Read a frame; return the virtual time of its arrival, its header and its JPEG."""
    message = await connection.recv()
    assert isinstance(message, bytes)
    header, jpeg = decode_frame(message)
    return LoopClock().now_ns(), header, bytes(jpeg)


async def next_report(connection: MemoryConnection) -> Telemetry:
    """Read up to the next telemetry report and return it."""
    message = await connection.recv()
    while isinstance(message, bytes):
        message = await connection.recv()
    return decode_telemetry(message)


async def read_to_end(connection: MemoryConnection) -> tuple[list[str | bytes], TransportClosed]:
    """Read until recv() raises; return the messages read and what it raised."""
    messages: list[str | bytes] = []
    while True:
        try:
            messages.append(await connection.recv())
        except TransportClosed as closed:
            return messages, closed


def saved(samples: SampleLog, tmp_path: Path) -> tuple[npt.NDArray[np.void], npt.NDArray[np.void]]:
    """Return the commands and the ticks of a sample log, as its file holds them."""
    samples.save(tmp_path / "robot_r01.npz", {})
    commands, ticks, _ = load_samples(tmp_path / "robot_r01.npz")
    return commands, ticks


async def test_the_hello_declares_the_robot_s_id_token_models_thresholds_and_frames() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        _, hello = await gateway.accept(session=1)
        stop.set()

    assert hello == Hello(
        protocol_version=PROTOCOL_VERSION,
        robot_id="r01",
        token="secret",
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        frame_width_px=640,
        frame_height_px=480,
    )


async def test_frames_leave_at_their_scheduled_ticks_with_the_tick_time_as_send_time() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=3)
        frames = [await next_frame(connection) for _ in range(3)]
        stop.set()

    assert frames == [
        (50 * MS, FrameHeader(3, 1, 50 * MS, 50 * MS), JPEGS[1]),
        (100 * MS, FrameHeader(3, 2, 100 * MS, 100 * MS), JPEGS[0]),
        (150 * MS, FrameHeader(3, 3, 150 * MS, 150 * MS), JPEGS[1]),
    ]


async def test_a_tick_taken_after_its_due_time_sends_the_clock_reading_as_its_send_time() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    settings = RobotSettings(
        models=("cls",), stale_ms={"cls": 500.0}, watchdog_ms={"cls": 1000.0}, frame_rate_hz=30.0
    )
    # A 30 fps period is not a whole number of milliseconds, and sleep_until sleeps whole
    # milliseconds, rounded up, so each tick is taken up to a millisecond after it is due.
    robot = make_robot(gateway, settings=settings, schedule=CaptureSchedule(0, 33_333_333))

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        frames = [await next_frame(connection) for _ in range(3)]
        stop.set()

    # The fire lag, send_ns - capture_ns, shows the lateness.
    assert [
        (arrival_ns, header.capture_ns, header.send_ns) for arrival_ns, header, _ in frames
    ] == [
        (34 * MS, 33_333_333, 34 * MS),
        (67 * MS, 66_666_666, 67 * MS),
        (100 * MS, 99_999_999, 100 * MS),
    ]


async def test_a_robot_waits_for_its_start_time_before_it_connects() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway, start_ns=2 * SECOND + 500_000)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        arrival_ns, header, _ = await next_frame(connection)
        stop.set()

    # The start is not a whole millisecond, and the wait is rounded up to one.
    assert gateway.connect_ns == [2 * SECOND + 1 * MS]
    assert (arrival_ns, header.frame_number) == (2 * SECOND + 50 * MS, 41)


async def test_the_report_follows_the_frame_of_the_first_tick_a_second_into_the_session() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        messages = [await connection.recv() for _ in range(21)]
        arrival_ns = LoopClock().now_ns()
        stop.set()

    assert [type(message) for message in messages] == [bytes] * 20 + [str]
    report = decode_telemetry(str(messages[20]))
    assert (arrival_ns, report.ts_mono_ns) == (1 * SECOND, 1 * SECOND)
    assert (report.frames_captured, report.frames_sent, report.reconnects) == (20, 20, 0)


async def test_a_report_waits_for_a_tick_that_leaves_nothing_unsent() -> None:
    gateway = Gateway(limits=NARROW_GATEWAY_END)
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        # Read through the tick at 900 ms, then leave the frames of 950 ms and 1 s unread:
        # the second stays unsent at the robot, over the tick at which the report is due.
        for _ in range(18):
            await connection.recv()
        await sleep_until(LoopClock(), 1060 * MS)
        frames = [await next_frame(connection) for _ in range(3)]
        report = await next_report(connection)
        stop.set()

    assert [header.frame_number for _, header, _ in frames] == [19, 20, 22]
    assert (report.ts_mono_ns, report.frames_sent, report.frames_skipped_send) == (
        1100 * MS,
        21,
        1,
    )


async def test_a_gateway_that_stops_reading_makes_ticks_skip_then_the_robot_aborts_and_reconnects(
    tmp_path: Path,
) -> None:
    gateway = Gateway(limits=NARROW_GATEWAY_END)
    stop = asyncio.Event()
    samples = SampleLog(tick_capacity=64, command_capacity=64)
    robot = make_robot(gateway, samples=samples)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        # The frame of 50 ms fills the gateway's buffers and is never read, so the frame of
        # 100 ms stays unsent, and every tick from 150 ms on is skipped.
        first, _ = await gateway.accept(session=1)
        await gateway.accept(session=2)
        stop.set()
    messages, closed = await read_to_end(first)

    # Skipped from 150 ms, so the stall timeout of 300 ms has passed at the tick of 450 ms.
    assert gateway.connect_ns == [0, 450 * MS + pauses_ns(1)[0]]
    _, ticks = saved(samples, tmp_path)
    assert ticks["frame_number"].tolist() == list(range(1, 10))
    assert ticks["outcome"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1, 1]
    assert len(messages) == 1
    assert (closed.peer_code, closed.local_code) == (None, None)
    assert robot.fatal_code is None


async def test_after_a_stall_and_a_reconnect_the_frames_stay_on_the_robot_s_one_schedule(
    tmp_path: Path,
) -> None:
    gateway = Gateway(limits=NARROW_GATEWAY_END)
    stop = asyncio.Event()
    samples = SampleLog(tick_capacity=64, command_capacity=64)
    robot = make_robot(gateway, samples=samples)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        await gateway.accept(session=1)
        second, _ = await gateway.accept(session=2)
        frames = [await next_frame(second) for _ in range(3)]
        _, answered, _ = frames[0]
        second.send_nowait(cls_command(2, answered.frame_number, answered.capture_ns))
        stop.set()

    # The first session skips from 150 ms on and is aborted at the tick of 450 ms; the second
    # is welcomed one pause later. The outage moves no tick, so the frame numbers go on.
    first = SCHEDULE.latest_due(450 * MS + pauses_ns(1)[0]) + 1
    numbers = [first, first + 1, first + 2]
    assert [
        (arrival_ns, header.frame_number, header.capture_ns) for arrival_ns, header, _ in frames
    ] == [(SCHEDULE.capture_ns(number), number, SCHEDULE.capture_ns(number)) for number in numbers]
    # The frame numbers of the samples rise across the sessions, so a command finds the tick of
    # its frame when they are saved.
    commands, ticks = saved(samples, tmp_path)
    assert ticks["frame_number"].tolist() == [*range(1, 10), *numbers]
    assert commands[["frame_number", "send_ns"]].tolist() == [(first, SCHEDULE.capture_ns(first))]


async def test_a_reconnect_after_an_abort_counts_once_and_starts_the_counters_again() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        first, _ = await gateway.accept(session=1)
        await sleep_until(LoopClock(), 2 * SECOND + 10 * MS)
        first.abort()
        second, _ = await gateway.accept(session=2)
        report = await next_report(second)
        stop.set()

    # The robot sees the loss at its tick of 2.05 s.
    welcome_ns = 2050 * MS + pauses_ns(1)[0]
    assert gateway.connect_ns == [0, welcome_ns]
    assert (robot.reconnects, report.reconnects) == (1, 1)
    session_frames = SCHEDULE.latest_due(report.ts_mono_ns) - SCHEDULE.latest_due(welcome_ns)
    assert (report.frames_captured, report.frames_sent) == (session_frames, session_frames)
    assert session_frames in (20, 21)


async def test_every_welcome_after_the_first_counts_one_reconnect() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        first, _ = await gateway.accept(session=1)
        await first.close(1013, "over capacity")
        second, _ = await gateway.accept(session=2)
        await second.close(1001, "shutdown")
        third, _ = await gateway.accept(session=3)
        report = await next_report(third)
        stop.set()

    assert (robot.reconnects, report.reconnects) == (2, 2)
    assert robot.fatal_code is None


async def test_refused_connects_are_retried_with_backoff_and_are_not_reconnects() -> None:
    gateway = Gateway(refusals=[503, None, 503])
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        report = await next_report(connection)
        stop.set()

    first, second, third = pauses_ns(3)
    assert gateway.connect_ns == [0, first, first + second, first + second + third]
    assert first <= 100 * MS < second <= 200 * MS < third <= 400 * MS
    assert (robot.reconnects, report.reconnects) == (0, 0)


async def test_a_welcome_starts_the_backoff_again_from_its_minimum() -> None:
    gateway = Gateway(refusals=[503, 503])
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        await connection.close(1013, "over capacity")
        await gateway.accept(session=2)
        stop.set()

    welcome_ns = gateway.connect_ns[2]
    # The robot sees the close at its first tick after the welcome.
    seen_ns = SCHEDULE.capture_ns(SCHEDULE.latest_due(welcome_ns) + 1)
    backoff = Backoff(100 * MS, 800 * MS, random.Random(SEED))
    backoff.next_ns()
    backoff.next_ns()
    backoff.reset()
    assert gateway.connect_ns[3] - seen_ns == -(-backoff.next_ns() // MS) * MS
    assert gateway.connect_ns[3] - seen_ns <= 100 * MS


@pytest.mark.parametrize("code", [1002, 1007, 1008, 1009, 4001])
async def test_a_close_that_a_retry_would_repeat_stops_the_robot_for_good(code: int) -> None:
    gateway = Gateway()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        await connection.close(code, "refused")

    assert robot.fatal_code == code
    assert len(gateway.connect_ns) == 1


@pytest.mark.parametrize("code", [1000, 1001, 1011, 1013, 4002, 4003])
async def test_any_other_close_is_followed_by_a_reconnect(code: int) -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        await connection.close(code, "closed")
        await gateway.accept(session=2)
        stop.set()

    assert robot.fatal_code is None
    assert len(gateway.connect_ns) == 2


@pytest.mark.parametrize(("code", "fatal_code", "connects"), [(1008, 1008, 1), (1013, None, 2)])
async def test_a_hello_answered_with_a_close_is_final_or_retried_by_its_code(
    code: int, fatal_code: int | None, connects: int
) -> None:
    gateway = Gateway()
    # Long enough for one reconnect pause, and over before a second hello times out.
    robot = make_robot(gateway, stop_ns=300 * MS)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection = await gateway.next_connection()
        await connection.recv()
        await connection.close(code, "refused")

    assert robot.fatal_code == fatal_code
    assert (len(gateway.connect_ns), robot.reconnects) == (connects, 0)


async def test_a_hello_that_gets_no_welcome_in_time_is_dropped_and_retried() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        first = await gateway.next_connection()
        await gateway.accept(session=1)
        stop.set()
    _, closed = await read_to_end(first)

    assert gateway.connect_ns == [0, 400 * MS + pauses_ns(1)[0]]
    assert (closed.peer_code, closed.local_code) == (None, None)
    assert robot.reconnects == 0


@pytest.mark.parametrize(
    "welcome",
    [
        b"\x02welcome",
        "welcome",
        '{"type":"welcome","protocol_version":1,"session":0,"shared_clock":true}',
        encode_welcome(Welcome(PROTOCOL_VERSION + 1, 1, True)),
    ],
)
async def test_a_first_message_that_is_not_a_welcome_is_closed_with_1002_for_good(
    welcome: str | bytes,
) -> None:
    gateway = Gateway()
    robot = make_robot(gateway)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection = await gateway.next_connection()
        connection.send_nowait(welcome)
        _, closed = await read_to_end(connection)

    assert closed.peer_code == 1002
    assert robot.fatal_code == 1002
    assert len(gateway.connect_ns) == 1


@pytest.mark.parametrize(
    "message", ["telemetry?", b"\x02short", cls_command(1, 1, 0)[:-1], DEPTH_COMMAND]
)
@pytest.mark.parametrize("downlink", [LinkSettings(), LinkSettings(delay_ms=30.0)])
async def test_a_message_that_is_not_a_command_of_the_session_is_closed_with_1002_for_good(
    message: str | bytes, downlink: LinkSettings
) -> None:
    gateway = Gateway()
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        downlink=downlink,
    )
    robot = make_robot(gateway, settings=settings)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        connection.send_nowait(message)
        _, closed = await read_to_end(connection)

    assert closed.peer_code == 1002
    assert robot.fatal_code == 1002
    assert len(gateway.connect_ns) == 1


async def test_with_a_link_model_messages_arrive_in_order_one_delay_after_they_were_sent(
    tmp_path: Path,
) -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    clock = LoopClock()
    samples = SampleLog(tick_capacity=64, command_capacity=64)
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        uplink=LinkSettings(delay_ms=20.0),
        downlink=LinkSettings(delay_ms=30.0),
    )
    robot = make_robot(gateway, settings=settings, samples=samples)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        frames = [await next_frame(connection) for _ in range(3)]
        connection.send_nowait(cls_command(1, 2, 100 * MS))
        await sleep_until(clock, 185 * MS)
        connection.send_nowait(cls_command(1, 3, 150 * MS))
        await sleep_until(clock, 300 * MS)
        stop.set()

    # The send time is stamped before the uplink line, and the receive time after the
    # downlink line.
    assert [(arrival_ns, header.send_ns) for arrival_ns, header, _ in frames] == [
        (70 * MS, 50 * MS),
        (120 * MS, 100 * MS),
        (170 * MS, 150 * MS),
    ]
    commands, _ = saved(samples, tmp_path)
    assert commands[["frame_number", "send_ns", "receive_ns", "verdict"]].tolist() == [
        (2, 100 * MS, 200 * MS, 0),
        (3, 150 * MS, 215 * MS, 0),
    ]


async def test_with_an_uplink_delay_the_report_follows_its_frame_through_the_delay_line() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        uplink=LinkSettings(delay_ms=20.0),
    )
    robot = make_robot(gateway, settings=settings)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        messages = [await connection.recv() for _ in range(21)]
        arrival_ns = LoopClock().now_ns()
        stop.set()

    # The report of the tick at 1 s is as late as that tick's frame and never overtakes it, as
    # TCP would not let it.
    assert [type(message) for message in messages] == [bytes] * 20 + [str]
    report = decode_telemetry(str(messages[20]))
    assert (arrival_ns, report.ts_mono_ns) == (1020 * MS, 1 * SECOND)


async def uplink_lateness_ns(seed: int, commands_per_frame: int) -> list[int]:
    """Run a robot with a jittered link both ways; return each frame's arrival minus its send.

    The test's gateway answers each of the robot's first twelve frames with that many commands.
    """
    gateway = Gateway()
    stop = asyncio.Event()
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        uplink=LinkSettings(delay_ms=10.0, jitter_ms=30.0),
        downlink=LinkSettings(delay_ms=10.0, jitter_ms=30.0),
    )
    robot = make_robot(gateway, settings=settings, start_ns=LoopClock().now_ns(), seed=seed)
    lateness_ns: list[int] = []
    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        for _ in range(12):
            arrival_ns, header, _ = await next_frame(connection)
            lateness_ns.append(arrival_ns - header.send_ns)
            for _ in range(commands_per_frame):
                connection.send_nowait(cls_command(1, header.frame_number, header.capture_ns))
        stop.set()
    return lateness_ns


async def test_the_link_jitter_repeats_for_one_seed_and_changes_with_the_seed() -> None:
    first = await uplink_lateness_ns(seed=5, commands_per_frame=1)
    again = await uplink_lateness_ns(seed=5, commands_per_frame=1)
    other = await uplink_lateness_ns(seed=6, commands_per_frame=1)

    assert first == again
    assert first != other


async def test_the_uplink_jitter_does_not_depend_on_the_downlink_traffic() -> None:
    quiet = await uplink_lateness_ns(seed=5, commands_per_frame=0)
    busy = await uplink_lateness_ns(seed=5, commands_per_frame=1)

    # Each direction draws from a generator of its own.
    assert quiet == busy
    assert len(set(quiet)) > 1


async def stopped_session(
    tmp_path: Path, stop_ns: int, stop_event_after_s: float
) -> tuple[Robot, Gateway, list[str | bytes], TransportClosed, npt.NDArray[np.void]]:
    """Run one session to a planned stop at the tick of 2 s, with a command sent at that tick.

    Returns the robot, the gateway, every message the gateway got after the hello, how the
    connection closed, and the robot's command samples.
    """
    gateway = Gateway()
    stop = asyncio.Event()
    clock = LoopClock()
    samples = SampleLog(tick_capacity=64, command_capacity=64)
    robot = make_robot(gateway, stop_ns=stop_ns, samples=samples)
    stop_timer = asyncio.get_running_loop().call_later(stop_event_after_s, stop.set)
    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        await sleep_until(clock, 500 * MS)
        connection.send_nowait(cls_command(1, 5, 250 * MS))
        await sleep_until(clock, 700 * MS)
        connection.send_nowait(cls_command(1, 9, 450 * MS))
        # The instant of the stop tick: this command is in the robot's connection when the
        # tick loop stops the session, and is never taken out of it.
        await sleep_until(clock, 2 * SECOND)
        connection.send_nowait(cls_command(1, 38, 1900 * MS))
        messages, closed = await read_to_end(connection)
    stop_timer.cancel()
    commands, _ = saved(samples, tmp_path)
    return robot, gateway, messages, closed, commands


# The stop time of the plan, and the stop Event that a signal sets: either way the session
# stops at the tick of 2 s.
PLANNED_STOPS = pytest.mark.parametrize(
    ("stop_ns", "stop_event_after_s"), [(2 * SECOND, 3600.0), (3600 * SECOND, 1.99)]
)


@PLANNED_STOPS
async def test_a_planned_stop_sends_the_last_report_then_closes_with_1000(
    tmp_path: Path, stop_ns: int, stop_event_after_s: float
) -> None:
    robot, gateway, messages, closed, _ = await stopped_session(
        tmp_path, stop_ns, stop_event_after_s
    )

    report = decode_telemetry(str(messages[-1]))
    # Reports at the ticks of 1 s and 2 s, and the frames of the 39 ticks before the stop.
    assert [type(message) for message in messages].count(str) == 2
    assert (report.ts_mono_ns, report.frames_sent) == (2 * SECOND, 39)
    assert (closed.peer_code, closed.local_code) == (1000, None)
    assert robot.fatal_code is None
    assert len(gateway.connect_ns) == 1


@PLANNED_STOPS
async def test_a_command_that_arrives_at_the_stop_is_in_neither_the_samples_nor_the_last_report(
    tmp_path: Path, stop_ns: int, stop_event_after_s: float
) -> None:
    _, _, messages, _, commands = await stopped_session(tmp_path, stop_ns, stop_event_after_s)

    cls = decode_telemetry(str(messages[-1])).models["cls"]
    freshness_ms = (commands["receive_ns"] - commands["capture_ns"]) / 1e6
    assert commands["frame_number"].tolist() == [5, 9]
    assert cls.commands_received == 2
    assert (cls.freshness_ms.count, cls.freshness_ms.max_ms) == (2, freshness_ms.max())
    assert freshness_ms.tolist() == [250.0, 250.0]


async def test_the_last_report_goes_past_the_uplink_delay_line_whose_frames_are_dropped() -> None:
    gateway = Gateway()
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        uplink=LinkSettings(delay_ms=70.0),
    )
    robot = make_robot(gateway, settings=settings, stop_ns=2 * SECOND)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        messages, closed = await read_to_end(connection)

    # At the stop tick of 2 s the frame of 1.95 s is still in the line, until 2.02 s: it is
    # dropped like a packet in flight, and the report reaches the gateway before the close.
    assert [type(message) for message in messages].count(bytes) == 38
    assert isinstance(messages[-1], str)
    report = decode_telemetry(messages[-1])
    assert (report.ts_mono_ns, report.frames_sent, closed.peer_code) == (2 * SECOND, 39, 1000)


async def test_the_last_report_is_written_behind_a_frame_that_is_still_unsent() -> None:
    gateway = Gateway(limits=NARROW_GATEWAY_END)
    robot = make_robot(gateway, stop_ns=2 * SECOND)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        # The frames of 50 ms to 1.85 s and the report of 1 s.
        for _ in range(38):
            await connection.recv()
        # Unread from here: the frame of 1.9 s fills the gateway's buffers, so the frame of
        # 1.95 s is still unsent at the robot at the stop tick of 2 s.
        await sleep_until(LoopClock(), 2 * SECOND + 1 * MS)
        messages, closed = await read_to_end(connection)

    assert [type(message) for message in messages] == [bytes, bytes, str]
    report = decode_telemetry(str(messages[-1]))
    assert (report.ts_mono_ns, report.frames_sent, closed.peer_code) == (2 * SECOND, 39, 1000)


async def test_a_replacement_seen_only_after_a_refused_send_still_stops_the_robot() -> None:
    gateway = Gateway()
    # At 20 commands per second a downlink delay of 500 ms gives a line of 12 messages.
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        close_timeout_ms=100.0,
        downlink=LinkSettings(delay_ms=500.0),
    )
    robot = make_robot(gateway, settings=settings)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        await sleep_until(LoopClock(), 110 * MS)
        # Twelve commands fill the line, so the receive loop waits for room with the
        # thirteenth in hand and does not reach the close behind it. The tick of 150 ms then
        # finds its send refused.
        for frame_number in range(1, 14):
            connection.send_nowait(cls_command(1, frame_number, 0))
        await connection.close(4001, "replaced")
    ended_ns = LoopClock().now_ns()

    assert robot.fatal_code == 4001
    assert len(gateway.connect_ns) == 1
    # The line frees room at 610 ms, long after the wait of one close timeout ended in an
    # abort at 250 ms.
    assert ended_ns == 610 * MS


async def test_a_protocol_error_in_the_downlink_line_ends_the_wait_after_a_refused_send() -> None:
    gateway = Gateway()
    # At 20 commands per second a downlink delay of 2 s gives a line of 42 messages.
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        close_timeout_ms=100.0,
        downlink=LinkSettings(delay_ms=2000.0),
    )
    robot = make_robot(gateway, settings=settings)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(asyncio.Event()))
        connection, _ = await gateway.accept(session=1)
        await sleep_until(LoopClock(), 110 * MS)
        # The line takes the first 42 messages and the receive loop waits for room with the
        # 43rd in hand, so the tick of 150 ms finds its send refused and the wait after it
        # ends in the robot's own abort at 250 ms. The second message is a command of a model
        # the robot did not subscribe to: it stops the pump when the line releases it.
        connection.send_nowait(cls_command(1, 1, 0))
        connection.send_nowait(DEPTH_COMMAND)
        for frame_number in range(3, 101):
            connection.send_nowait(cls_command(1, frame_number, 0))
        connection.abort()
    ended_ns = LoopClock().now_ns()

    assert robot.fatal_code == 1002
    assert len(gateway.connect_ns) == 1
    # The pump stops when the line releases the first 42, at 2.11 s, and the robot stops
    # then: nothing frees room for the receive loop after that, and the planned stop is an
    # hour away.
    assert ended_ns == 2110 * MS


async def test_an_uplink_send_that_finds_the_connection_closed_ends_the_session() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    settings = RobotSettings(
        models=("cls",),
        stale_ms={"cls": 500.0},
        watchdog_ms={"cls": 1000.0},
        frame_rate_hz=20.0,
        close_timeout_ms=100.0,
        reconnect_min_ms=100.0,
        reconnect_max_ms=800.0,
        uplink=LinkSettings(delay_ms=20.0),
        downlink=LinkSettings(delay_ms=500.0),
    )
    robot = make_robot(gateway, settings=settings)

    async with asyncio.TaskGroup() as group:
        group.create_task(robot.run(stop))
        connection, _ = await gateway.accept(session=1)
        await sleep_until(LoopClock(), 110 * MS)
        # The downlink line is held full as in the test above, so only the uplink pump, which
        # sends the frame of 100 ms at 120 ms, meets the closed connection.
        for frame_number in range(1, 14):
            connection.send_nowait(cls_command(1, frame_number, 0))
        await connection.close(1001, "shutdown")
        await gateway.accept(session=2)
        stop.set()

    assert robot.fatal_code is None
    assert gateway.connect_ns == [0, 610 * MS + pauses_ns(1)[0]]
    assert robot.reconnects == 1


async def test_the_stop_event_ends_the_wait_for_the_start_time() -> None:
    gateway = Gateway()
    stop = asyncio.Event()
    robot = make_robot(gateway, start_ns=10 * SECOND)
    asyncio.get_running_loop().call_later(1.0, stop.set)

    await robot.run(stop)

    assert LoopClock().now_ns() == 1 * SECOND
    assert gateway.connect_ns == []


async def test_the_stop_event_ends_a_reconnect_pause() -> None:
    gateway = Gateway(refusals=[503, 503, 503])
    stop = asyncio.Event()
    settings = RobotSettings(reconnect_min_ms=60_000.0, reconnect_max_ms=60_000.0)
    robot = make_robot(gateway, settings=settings)
    asyncio.get_running_loop().call_later(1.0, stop.set)

    await robot.run(stop)

    assert LoopClock().now_ns() == 1 * SECOND
    assert gateway.connect_ns == [0]


async def test_a_reconnect_pause_ends_at_the_stop_time() -> None:
    gateway = Gateway(refusals=[503, 503, 503])
    settings = RobotSettings(reconnect_min_ms=60_000.0, reconnect_max_ms=60_000.0)
    robot = make_robot(gateway, settings=settings, stop_ns=2 * SECOND)

    await robot.run(asyncio.Event())

    assert LoopClock().now_ns() == 2 * SECOND
    assert gateway.connect_ns == [0]
    assert robot.fatal_code is None


async def test_a_robot_whose_stop_time_has_passed_never_connects() -> None:
    gateway = Gateway()
    robot = make_robot(gateway, start_ns=0, stop_ns=0)

    await robot.run(asyncio.Event())

    assert gateway.connect_ns == []
