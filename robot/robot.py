"""One simulated robot's life: connect, hello, tick loop, receive loop, link pumps, reconnect.

This is the I/O layer of a robot. It moves messages between a Connection and the session
logic of robot/session.py, reads the clock once per event, and owns every task and wait of
the robot.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field

from common.aio import sleep_until
from common.backoff import Backoff
from common.clock import Clock
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    CloseCode,
    Fault,
    Hello,
    ProtocolError,
    Welcome,
    decode_welcome,
    encode_hello,
)
from common.transport import Connection, TransportClosed, TransportConnectError
from robot.link import DelayLine, line_capacity
from robot.samples import SampleLog
from robot.schedule import CaptureSchedule
from robot.session import RobotSession
from robot.settings import LinkSettings, RobotSettings

_LOG = logging.getLogger(__name__)

# Opens a connection to the gateway, bounded by its own connect timeout. Raises
# TransportConnectError.
Connector = Callable[[], Awaitable[Connection]]

# After these closes the robot stops for good: a retry would repeat the refusal, or start a
# replacement loop between two processes with one id (DESIGN.md 4.3). 1007 is the library's
# own close for a text message that is not UTF-8.
_FATAL_CODES = frozenset(
    (
        CloseCode.PROTOCOL_ERROR,
        1007,
        CloseCode.POLICY,
        CloseCode.MESSAGE_TOO_BIG,
        CloseCode.SESSION_REPLACED,
    )
)


@dataclass(frozen=True, slots=True)
class _SessionEnd:
    """How one connection ended, as far as the robot's next step depends on it."""

    # For the log.
    reason: str
    # The close code that ended it: the gateway's, the library's, or this robot's own 1002.
    # None when it ended without a close frame, and for a planned stop.
    code: int | None
    planned: bool = False


@dataclass(slots=True)
class _Link:
    """What the tasks of one session share."""

    connection: Connection
    session: RobotSession
    # The delay line of each direction that has a delay.
    uplink: DelayLine | None
    downlink: DelayLine | None
    # Set by every push onto the line; its pump waits on it while the line is empty.
    uplink_pushed: asyncio.Event = field(default_factory=asyncio.Event)
    downlink_pushed: asyncio.Event = field(default_factory=asyncio.Event)
    # Set by the downlink pump whenever it has taken messages off the line.
    downlink_room: asyncio.Event = field(default_factory=asyncio.Event)
    # What the receive loop's recv() raised when the connection closed.
    closed: TransportClosed | None = None
    # A message the gateway may not send, met by the receive loop or the downlink pump.
    protocol_error: ProtocolError | None = None
    # The uplink pump's send found the connection closed.
    uplink_refused: bool = False


def _close_end(closed: TransportClosed) -> _SessionEnd:
    code = closed.peer_code if closed.peer_code is not None else closed.local_code
    if code is None:
        return _SessionEnd("the connection ended without a close frame", None)
    return _SessionEnd(f"closed with {code}", code)


def _lost_end(link: _Link) -> _SessionEnd:
    """Return the end of a session that a protocol error or the connection's close ended."""
    if link.protocol_error is not None:
        return _SessionEnd(f"protocol error: {link.protocol_error}", CloseCode.PROTOCOL_ERROR)
    if link.closed is None:
        return _SessionEnd("the connection ended without a close frame", None)
    return _close_end(link.closed)


def _delay_line(
    settings: LinkSettings, messages_per_s: float, rng: random.Random
) -> DelayLine | None:
    """Return an empty delay line for a direction that has a delay, else None."""
    if not settings.enabled:
        return None
    return DelayLine(settings, line_capacity(settings, messages_per_s), rng)


def _log_session_end(session: RobotSession, reason: str) -> None:
    """Log why a session ended, and what the robot did with each model's commands in it."""
    _LOG.info("robot %s: session %d ended: %s", session.robot_id, session.number, reason)
    for model, control in session.controls.items():
        # ASSUMPTION: A10. The stale count is reported here, in the robot's log, once per
        # session and model, and at 1 Hz through telemetry. Nothing is logged per command.
        _LOG.info(
            "robot %s: session %d %s: received %d, applied %d, stale %d, out of order %d",
            session.robot_id,
            session.number,
            model,
            control.received,
            control.applied,
            control.stale,
            control.out_of_order,
        )


class Robot:
    """One simulated robot, from its start time to its stop time.

    reconnects counts the sessions after the first in this robot process (DESIGN.md 10.4).
    fatal_code is the close code that stopped the robot for good, and None while it runs
    and after it has completed its plan.
    """

    def __init__(
        self,
        robot_id: str,
        settings: RobotSettings,
        schedule: CaptureSchedule,
        jpegs: Sequence[bytes],
        start_ns: int,
        stop_ns: int,
        connect: Connector,
        clock: Clock,
        rng: random.Random,
        samples: SampleLog | None,
    ) -> None:
        self._robot_id = robot_id
        self._settings = settings
        self._schedule = schedule
        self._jpegs = jpegs
        self._start_ns = start_ns
        self._stop_ns = stop_ns
        self._connect = connect
        self._clock = clock
        self._samples = samples
        self._backoff = Backoff(
            round(settings.reconnect_min_ms * 1_000_000),
            round(settings.reconnect_max_ms * 1_000_000),
            rng,
        )
        # The jitter of each direction has its own generator, seeded from a copy of rng, so
        # one seed fixes the reconnect pauses and both jitter sequences, and the pauses are
        # the ones that seed gives whether or not a link model is on.
        seeds = random.Random()
        seeds.setstate(rng.getstate())
        self._uplink_rng = random.Random(seeds.getrandbits(64))
        self._downlink_rng = random.Random(seeds.getrandbits(64))
        self._welcomed = False
        self.reconnects = 0
        self.fatal_code: int | None = None

    async def run(self, stop: asyncio.Event) -> None:
        """Run sessions until the stop time, the stop Event, or a close that is final."""
        await self._wait_until(self._start_ns, stop)
        while self._clock.now_ns() < self._stop_ns and not stop.is_set():
            try:
                # Bounded by the connector, which runs name resolution, the TCP connect and
                # the upgrade under the connect timeout.
                connection = await self._connect()
            except TransportConnectError as error:
                # A refused connection or an HTTP 503, as while the service warms up.
                _LOG.info("robot %s: no connection: %s", self._robot_id, error)
            else:
                end = await self._serve(connection, stop)
                if end.planned:
                    return
                if end.code in _FATAL_CODES:
                    self.fatal_code = end.code
                    return
            pause_ns = self._backoff.next_ns()
            await self._wait_until(min(self._clock.now_ns() + pause_ns, self._stop_ns), stop)

    async def _wait_until(self, deadline_ns: int, stop: asyncio.Event) -> None:
        """Return once the clock reads deadline_ns or later, or stop is set."""
        # The timer rule of common.aio.sleep_until, with the stop Event as a second way out:
        # whole milliseconds, rounded up, and the clock read again, so the wait never ends
        # before its deadline and never spins.
        while not stop.is_set() and (remaining_ns := deadline_ns - self._clock.now_ns()) > 0:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(-(-remaining_ns // 1_000_000) / 1000):
                    # Completed by whoever stops the fleet: a signal sets the Event.
                    await stop.wait()

    async def _serve(self, connection: Connection, stop: asyncio.Event) -> _SessionEnd:
        """Run one connection from its hello to its end, and leave it closed."""
        try:
            try:
                welcome, welcome_ns = await self._handshake(connection)
            except TransportClosed as closed:
                end = _close_end(closed)
            except TimeoutError:
                end = _SessionEnd("no welcome within the timeout", None)
            except ProtocolError as error:
                end = _SessionEnd(f"protocol error: {error}", CloseCode.PROTOCOL_ERROR)
                await connection.close(CloseCode.PROTOCOL_ERROR, "the welcome is malformed")
            else:
                return await self._stream(connection, welcome, welcome_ns, stop)
            _LOG.info("robot %s: no session: %s", self._robot_id, end.reason)
            return end
        finally:
            # A no-op once the connection has closed, so this drops only a connection that
            # no closing handshake ended: a lost one, a stalled one, or one whose robot task
            # is being cancelled.
            connection.abort()

    async def _handshake(self, connection: Connection) -> tuple[Welcome, int]:
        """Say hello and return the welcome with the clock's reading at its arrival."""
        settings = self._settings
        hello = Hello(
            PROTOCOL_VERSION,
            self._robot_id,
            settings.token,
            settings.models,
            settings.stale_ms,
            settings.watchdog_ms,
            settings.frame_rate_hz,
            FRAME_WIDTH_PX,
            FRAME_HEIGHT_PX,
        )
        connection.send_nowait(encode_hello(hello))
        async with asyncio.timeout(settings.welcome_timeout_ms / 1000):
            message = await connection.recv()
        welcome_ns = self._clock.now_ns()
        if isinstance(message, bytes):
            raise ProtocolError(Fault.MALFORMED, "a binary message before the welcome")
        return decode_welcome(message), welcome_ns

    async def _stream(
        self, connection: Connection, welcome: Welcome, welcome_ns: int, stop: asyncio.Event
    ) -> _SessionEnd:
        """Run the session of an accepted hello: frames out and commands in, until it ends."""
        settings = self._settings
        if self._welcomed:
            self.reconnects += 1
        self._welcomed = True
        self._backoff.reset()
        session = RobotSession(
            self._robot_id,
            settings,
            welcome,
            self._schedule,
            self._jpegs,
            welcome_ns,
            self._samples,
        )
        _LOG.info("robot %s: session %d open", self._robot_id, welcome.session)
        # A line holds what its delay covers at its direction's message rate: the frames
        # and the 1 Hz report on the way up, a command per frame and model on the way down.
        link = _Link(
            connection,
            session,
            uplink=_delay_line(settings.uplink, settings.frame_rate_hz + 1, self._uplink_rng),
            downlink=_delay_line(
                settings.downlink,
                settings.frame_rate_hz * len(settings.models),
                self._downlink_rng,
            ),
        )
        async with asyncio.TaskGroup() as group:
            receiver = group.create_task(self._receive_loop(link))
            children = [receiver]
            if link.uplink is not None:
                children.append(group.create_task(self._pump_uplink(link, link.uplink)))
            if link.downlink is not None:
                children.append(group.create_task(self._pump_downlink(link, link.downlink)))
            try:
                end = await self._tick_loop(link, receiver, children, stop)
            finally:
                for child in children:
                    child.cancel()
        if end.planned:
            await connection.close(CloseCode.NORMAL, "the robot has stopped")
        elif link.protocol_error is not None:
            await connection.close(CloseCode.PROTOCOL_ERROR, "the gateway broke the protocol")
        _log_session_end(session, end.reason)
        return end

    async def _tick_loop(
        self,
        link: _Link,
        receiver: asyncio.Task[None],
        children: Sequence[asyncio.Task[None]],
        stop: asyncio.Event,
    ) -> _SessionEnd:
        """Send the session's frames and reports, one tick at a time, until the session ends."""
        session = link.session
        while True:
            await sleep_until(self._clock, self._schedule.capture_ns(session.next_frame))
            # The one clock reading of the tick. It is the send time of the tick's frame.
            now_ns = self._clock.now_ns()
            if now_ns >= self._stop_ns or stop.is_set():
                # The planned stop is one step with no await. A cancelled task's next step
                # is the CancelledError thrown at its await, also when the future it awaited
                # has just completed, and recv() leaves a message it had not returned in the
                # library's queue. So a command that arrives from here on reaches neither
                # the samples nor the last report, which keeps the two equal: /metrics keeps
                # the last report, and the run's checks compare it with the samples exactly.
                for child in children:
                    child.cancel()
                # Every command recorded so far was received at or before this tick's
                # reading, because no other task ran since it was taken. The report goes
                # straight to the connection, behind whatever is unsent, and past the uplink
                # delay line, whose frames are dropped with the session like packets in
                # flight. A connection that has closed loses the report; the robot stops all
                # the same.
                with contextlib.suppress(TransportClosed):
                    link.connection.send_nowait(session.telemetry(now_ns, self.reconnects))
                return _SessionEnd("planned stop", None, planned=True)
            if receiver.done() or link.protocol_error is not None:
                return _lost_end(link)
            if link.uplink_refused:
                return await self._end_after_refused_send(link, receiver)
            try:
                stalled = self._send_tick(link, now_ns)
            except TransportClosed:
                return await self._end_after_refused_send(link, receiver)
            if stalled:
                # The robot's half-open rule: the frame stream is the heartbeat in this
                # direction too, and a gateway that takes no frame is not serving.
                link.connection.abort()
                return _SessionEnd("no frame left the robot within the stall timeout", None)

    def _send_tick(self, link: _Link, now_ns: int) -> bool:
        """Send what the tick at now_ns has due; return True when the path has stalled.

        Raises TransportClosed when a send finds the connection closed.
        """
        connection, session, uplink = link.connection, link.session, link.uplink
        # How "the previous frame is still unsent" is detected: the socket's send buffer and
        # TCP_NOTSENT_LOWAT make the kernel refuse new bytes while unsent ones are above the
        # mark, and the event loop writes inside the call and keeps only what the kernel
        # refused. So unsent bytes at this tick mean that the previous frame has not fully
        # left this process. The tick is then skipped, and so is one that finds the uplink
        # delay line full.
        message = session.tick(
            now_ns, connection.unsent_bytes() > 0 or (uplink is not None and uplink.full())
        )
        if message is not None:
            self._send_up(link, message, now_ns)
        elif session.stalled(now_ns):
            return True
        # The report follows a frame, and only when nothing is unsent: after a skipped tick
        # the path is busy, so the report waits for the next frame that leaves.
        if (
            session.telemetry_due(now_ns)
            and connection.unsent_bytes() == 0
            and not (uplink is not None and uplink.full())
        ):
            self._send_up(link, session.telemetry(now_ns, self.reconnects), now_ns)
        return False

    def _send_up(self, link: _Link, message: str | bytes, now_ns: int) -> None:
        """Send a message through the uplink delay line, or straight to the connection."""
        if link.uplink is None:
            link.connection.send_nowait(message)
        else:
            # The frame's send time was stamped before this line, so the injected delay
            # lands in the uplink stage.
            link.uplink.push(message, now_ns)
            link.uplink_pushed.set()

    async def _end_after_refused_send(
        self, link: _Link, receiver: asyncio.Task[None]
    ) -> _SessionEnd:
        """Wait for the receive loop's close codes after a send found the connection closed."""
        # The library refuses every send as soon as the gateway's close frame has arrived,
        # but recv() raises only once the gateway has closed the TCP connection too, about
        # one round trip later, so the receive loop may not have the close code yet. Without
        # this wait, a robot replaced with 4001 or refused with 1008 whose tick fell in that
        # window would see no code and reconnect: the replacement loop between two processes
        # with one id that DESIGN.md 1.4 rules out.
        try:
            async with asyncio.timeout(self._settings.close_timeout_ms / 1000):
                # asyncio.wait cancels nothing it waits on, so the timeout leaves the
                # receive loop running.
                await asyncio.wait({receiver})
        except TimeoutError:
            # After an abort, recv() returns the messages already queued and then raises
            # with the close frame the library has already parsed.
            link.connection.abort()
            # Completed by the abort. With a downlink delay line the receive loop first
            # passes the queued messages into it, which their release times bound while the
            # downlink pump runs; once the pump has stopped on a protocol error, the receive
            # loop returns at its next message.
            await asyncio.wait({receiver})
        return _lost_end(link)

    async def _receive_loop(self, link: _Link) -> None:
        """Take the gateway's messages until the connection closes; record how it closed."""
        connection, session, downlink = link.connection, link.session, link.downlink
        # The bound of an await on a peer that may send nothing for a long time: a starved
        # model sends no command, and the gateway's idle timeout and this robot's stall rule
        # end a dead connection. The session never outlives its planned stop, which the tick
        # loop handles at the first tick at or after the stop time, so on expiry this loop
        # returns with no codes and that tick still stops the session as planned.
        loop = asyncio.get_running_loop()
        deadline_s = (
            loop.time()
            + (self._stop_ns - self._clock.now_ns()) / 1e9
            + self._settings.close_timeout_ms / 1000
        )
        try:
            async with asyncio.timeout_at(deadline_s):
                while True:
                    message = await connection.recv()
                    receive_ns = self._clock.now_ns()
                    if downlink is None:
                        session.receive(message, receive_ns)
                        continue
                    while downlink.full() and link.protocol_error is None:
                        link.downlink_room.clear()
                        # Completed by the downlink pump, which frees room at each message's
                        # release time. Until then nothing is read, which pushes back on the
                        # gateway like any slow reader.
                        await link.downlink_room.wait()
                    if link.protocol_error is not None:
                        # The downlink pump has stopped on a message the gateway may not
                        # send, so no room will come and the session ends with 1002.
                        return
                    downlink.push(message, receive_ns)
                    link.downlink_pushed.set()
        except TimeoutError:
            return
        except ProtocolError as error:
            link.protocol_error = error
        except TransportClosed as closed:
            link.closed = closed

    async def _pump_uplink(self, link: _Link, uplink: DelayLine) -> None:
        """Send each message of the uplink delay line at its release time."""
        while True:
            await self._head_release(uplink, link.uplink_pushed)
            try:
                for message in uplink.pop_due(self._clock.now_ns()):
                    link.connection.send_nowait(message)
            except TransportClosed:
                # Recorded and returned, never raised: a child's exception would leave the
                # session's TaskGroup as an ExceptionGroup and reach the fleet's, which
                # would cancel every robot of the process. The tick loop ends the session.
                link.uplink_refused = True
                return

    async def _pump_downlink(self, link: _Link, downlink: DelayLine) -> None:
        """Hand each message of the downlink delay line to the session at its release time."""
        while True:
            await self._head_release(downlink, link.downlink_pushed)
            # The receive time is stamped after the line, so the injected delay lands in the
            # deliver stage and in freshness.
            now_ns = self._clock.now_ns()
            messages = downlink.pop_due(now_ns)
            # Set before the messages are handed over, so a receive loop that waits for room
            # also wakes when one of them stops this pump, and then returns.
            link.downlink_room.set()
            try:
                for message in messages:
                    link.session.receive(message, now_ns)
            except ProtocolError as error:
                # Recorded, as in the receive loop: the tick loop closes the session with
                # 1002 and the robot stops for good.
                link.protocol_error = error
                return

    async def _head_release(self, line: DelayLine, pushed: asyncio.Event) -> None:
        """Return once the message at the head of the line is due."""
        while (release_ns := line.next_release_ns()) is None:
            pushed.clear()
            # Completed by the next push onto this line. No timer runs while it is empty.
            await pushed.wait()
        await sleep_until(self._clock, release_ns)
