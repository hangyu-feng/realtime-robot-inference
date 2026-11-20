"""The coroutine of one robot connection: its hello, its reader, its writer and its close.

run_connection is the isolation boundary of one connection (DESIGN.md 1.4). It moves messages
between the connection and the session table and decides nothing about them: every rule for a
hello, a frame, a report or the end of a session is in cloud/sessions.py. The handler task is
the session's writer and the only task that closes its connection; the reader is its one child.
Every close() here is bounded by the connection's own close timeout (common/transport.py).

Known inexactness: DownlinkQueue.pop counts commands_sent when a command is handed to the
connection, so a write that fails because the connection has just closed is still counted, at
most once per session end.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from cloud.config import ServerConfig
from cloud.counters import CloseReason, ConnectionTotals
from cloud.sessions import CLOSE_CODES, Refused, Session, SessionTable
from common.clock import Clock
from common.protocol import CloseCode
from common.transport import Connection, TransportClosed

_LOGGER = logging.getLogger(__name__)
_MS_PER_S = 1000
# The reader's idle deadline is set this fraction of the idle timeout past the timeout itself,
# and moved only once that fraction of the idle timeout has passed since it was set.
_IDLE_SLACK = 0.25


@dataclass(frozen=True)
class SessionContext:
    """What the connections of one gateway share."""

    config: ServerConfig
    clock: Clock
    table: SessionTable
    # The counters the table counts in; bytes_out is counted here, where messages are written.
    totals: ConnectionTotals
    # Per model, wakes that model's dispatcher when a frame was accepted for it.
    pokes: Mapping[str, Callable[[], None]]
    # False once shutdown has begun.
    accepting: Callable[[], bool]


async def run_connection(connection: Connection, context: SessionContext) -> None:
    """Serve one connection from its hello to its end.

    A fault in this connection's own code ends here: it is logged once, and a session that is
    still open ends as internal_error, so every accepted session ends with exactly one reason.
    """
    writer = _Writer(connection, context)
    session: Session | None = None
    try:
        message = await _receive_hello(connection, context)
        if message is None:
            return
        outcome = context.table.hello(message, writer.wake, context.clock.now_ns())
        if isinstance(outcome, Refused):
            await connection.close(outcome.code, outcome.reason)
            return
        session = writer.session = outcome.session
        await _run_session(connection, context, session, writer, outcome.welcome)
    except Exception:
        _LOGGER.exception(
            "robot %s: the connection's handler failed",
            "<no session>" if session is None else session.robot_id,
        )
    finally:
        if session is not None and not session.ended:
            # Every intended path ends the session before this point, so this counts a fault
            # or a cancellation of the handler.
            context.table.end(session, CloseReason.INTERNAL_ERROR, context.clock.now_ns())
        # A no-op on a closed connection. It is how a session that ended without a close code is
        # closed, and after it the transport's own close, which follows the handler, has nothing
        # to wait for.
        connection.abort()


async def _receive_hello(connection: Connection, context: SessionContext) -> str | bytes | None:
    """Return the connection's first message, or None once a connection with none is closed."""
    try:
        async with asyncio.timeout(context.config.handshake_timeout_ms / _MS_PER_S):
            message = await connection.recv()
    except TimeoutError:
        refusal = context.table.refuse_handshake_timeout()
        await connection.close(refusal.code, refusal.reason)
        return None
    except TransportClosed:
        # The peer left before its hello, which is neither accepted nor rejected.
        return None
    if not context.accepting():
        # A hello that arrives after shutdown has begun is neither accepted nor rejected
        # (DESIGN.md 10.7); the code tells the robot to connect again later.
        await connection.close(CloseCode.SERVER_SHUTDOWN, CloseReason.SHUTDOWN.value)
        return None
    return message


async def _run_session(
    connection: Connection,
    context: SessionContext,
    session: Session,
    writer: _Writer,
    welcome: str,
) -> None:
    """Run an open session until it has ended, then close its connection with the reason's code."""
    _LOGGER.info("robot %s: session %d opened", session.robot_id, session.number)
    async with asyncio.TaskGroup() as tasks:
        reader = tasks.create_task(_read(connection, context, session))
        try:
            await writer.run(welcome)
        finally:
            reader.cancel()
    reason = session.end_reason
    if reason is None:
        raise RuntimeError("the writer returned while its session was open")
    _LOGGER.info("robot %s: session %d ended: %s", session.robot_id, session.number, reason.value)
    code = CLOSE_CODES[reason]
    # A reason without a code ends in the abort that follows in run_connection.
    if code is not None:
        await connection.close(code, reason.value)


async def _read(connection: Connection, context: SessionContext, session: Session) -> None:
    """Pass the session's messages to the table until the session has ended.

    It awaits nothing but recv(), so it always drains the transport's receive queue
    (DESIGN.md 5.1), and nothing on this path logs.
    """
    table = context.table
    clock = context.clock
    loop = asyncio.get_running_loop()
    idle_s = context.config.idle_timeout_ms / _MS_PER_S
    # One timer for the whole loop instead of one per frame. The deadline is set 1.25 idle
    # timeouts ahead, and only the first message that arrives a quarter of an idle timeout or
    # more after that sets it again. So it fires between 1 and 1.25 idle timeouts after the last
    # message, and the timer moves at most once per quarter of an idle timeout. Both times are
    # the loop's, which is the clock of the timer itself.
    start_s = loop.time()
    reschedule_from_s = start_s + _IDLE_SLACK * idle_s
    try:
        async with asyncio.timeout_at(start_s + (1 + _IDLE_SLACK) * idle_s) as idle_deadline:
            while True:
                message = await connection.recv()
                received_ns = clock.now_ns()
                # A session that was replaced or ended elsewhere may still get a message its
                # connection had queued; it must not reach the table's state.
                if session.ended:
                    return
                now_s = loop.time()
                if now_s >= reschedule_from_s:
                    idle_deadline.reschedule(now_s + (1 + _IDLE_SLACK) * idle_s)
                    reschedule_from_s = now_s + _IDLE_SLACK * idle_s
                accepted = table.ingest(session, message, received_ns, connection.rtt_ns())
                if session.ended:
                    return
                if accepted:
                    for model in session.record.subscribed:
                        context.pokes[model]()
    except TimeoutError:
        table.end(session, CloseReason.IDLE_TIMEOUT, clock.now_ns())
    except TransportClosed as closed:
        table.lost(session, closed.peer_code, closed.local_code, clock.now_ns())


class _Writer:
    """The writing side of one session: the handler's send loop, and the wake the core calls.

    A command reaches the connection on one of two paths. wake() writes it at once, on its
    caller's task, when the connection holds no unsent bytes. The gateway's connections have a
    write limit of 0, so "no unsent bytes" means the kernel has taken everything written so
    far: such a write goes into an empty transport and waits behind nothing. When the kernel
    refuses bytes, the commands that follow stay in the session's bounded downlink queue, and
    the handler task drains it with sends that wait, under the slow-consumer timeout. A
    connection waits for its transport to empty only inside a send, after that send's own
    write, so the first send of a drain writes its command behind what the kernel refused: the
    transport holds at most the remainder of one command and one whole command, and every
    other command waits in the queue. Commands stay in order because one path at a time hands
    them to the connection: each path pops a command and hands it over in the same step, with
    no await in between, and while the handler waits in a send the connection holds unsent
    bytes, so wake() writes nothing then.
    """

    # Set by run_connection in the step that opens the session, before anything can call wake().
    session: Session

    def __init__(self, connection: Connection, context: SessionContext) -> None:
        self._connection = connection
        self._context = context
        self._slow_consumer_s = context.config.slow_consumer_timeout_ms / _MS_PER_S
        self._woken = asyncio.Event()
        # While true, wake() only signals the handler: from the start until the welcome has
        # been sent, and from a write that left commands behind until the handler has drained
        # them.
        self._draining = True
        # A synchronous write found the connection closed. Nothing more is written, so at most
        # one command is counted as sent without being sent, and the reader ends the session
        # with the reason its recv() gives: a refused write carries no close code.
        self._write_refused = False

    def wake(self) -> None:
        """Tell the session's handler that the session has ended or that commands were queued.

        Synchronous, and called from other tasks: by the table when the session ends, and by a
        model's dispatcher for the sessions of a completed batch.
        """
        if self.session.ended:
            # A close frame would wait behind bytes the robot is not reading, so a connection
            # that holds unsent bytes is dropped at once, which also releases a handler that
            # waits in a send.
            if self._connection.unsent_bytes() > 0:
                self._connection.abort()
            self._woken.set()
        elif self._draining:
            self._woken.set()
        elif not self._write_refused:
            self._write_now()

    async def run(self, welcome: str) -> None:
        """Send the welcome, then drain the queue whenever wake() hands over; return at the end."""
        session = self.session
        connection = self._connection
        table = self._context.table
        clock = self._context.clock
        try:
            async with asyncio.timeout(self._slow_consumer_s):
                self._count_bytes_out(len(welcome))
                await connection.send(welcome)
            self._draining = False
            while True:
                # No timeout by design: completed by wake(), which code in this process calls
                # when the session ends and when commands are left for this task to drain.
                await self._woken.wait()
                self._woken.clear()
                if session.ended:
                    return
                # One timer per drain, and none per command.
                async with asyncio.timeout(self._slow_consumer_s):
                    while (command := session.downlink.pop(clock.now_ns())) is not None:
                        self._count_bytes_out(len(command.data))
                        await connection.send(command.data)
                self._draining = False
        except TimeoutError:
            table.end(session, CloseReason.SLOW_CONSUMER, clock.now_ns())
        except TransportClosed as closed:
            table.lost(session, closed.peer_code, closed.local_code, clock.now_ns())

    def _write_now(self) -> None:
        """Write queued commands on the caller's task for as long as the kernel takes them whole."""
        session = self.session
        connection = self._connection
        now_ns = self._context.clock.now_ns()
        try:
            while (
                connection.unsent_bytes() == 0
                and (command := session.downlink.pop(now_ns)) is not None
            ):
                self._count_bytes_out(len(command.data))
                connection.send_nowait(command.data)
        except TransportClosed:
            self._write_refused = True
            return
        except Exception:
            # This write runs on a dispatcher's task, outside run_connection, so it is its own
            # isolation boundary of this one connection (DESIGN.md 1.4): the dispatcher goes on
            # with its next session.
            _LOGGER.exception("robot %s: writing a command failed", session.robot_id)
            self._context.table.end(session, CloseReason.INTERNAL_ERROR, now_ns)
            connection.abort()
            return
        if len(session.downlink) > 0:
            # A write that would wait is the handler's: it drains what is left.
            self._draining = True
            self._woken.set()

    def _count_bytes_out(self, message_bytes: int) -> None:
        """Count a message handed to the connection; a text message counts its characters."""
        self.session.record.bytes_out += message_bytes
        self._context.totals.bytes_out += message_bytes
