"""An in-memory Connection pair and Listener, the transport of the async unit tests.

The pair is an executable model, at the level of whole messages, of the WebSocket library
behaviour that the gateway and the robots rely on. Each end has an inbox, standing for the
kernel's receive buffer and the library's queue together, and an outbox, standing for the
transport's write buffer. A sent message moves into the peer's inbox while that inbox holds
less than the peer's capacity_bytes, and otherwise waits in the sender's outbox, where it counts
as unsent.

A close frame travels the same way, behind the messages sent before it. The library at the
other end reads it from the socket whether or not the application is in recv(), but only while
it reads at all: with more than its receive queue's messages unread it stops reading, and starts
again once the application has taken all but a quarter of them. When it reads a close it answers
with the same code, behind the messages its own end has not sent yet. The closing handshake is
over once each end has read the other's close, and a close() whose handshake is not over within
its end's close timeout aborts the pair.

Two library details are left out: the library checks a message's size when it reads the frame,
where the pair checks it when the message enters the inbox, and an end that aborts loses what
its library had not yet read from the socket, which the pair keeps.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections import deque
from dataclasses import dataclass
from http import HTTPStatus
from types import TracebackType
from typing import Self

from common.protocol import CloseCode
from common.transport import Handler, HttpHandler, TransportClosed, TransportConnectError


@dataclass(frozen=True)
class PipeLimits:
    """The buffers, the receive queue and the close timeout of one end."""

    # What this end's inbox takes before the peer's bytes stay unsent: the kernel buffers.
    capacity_bytes: int
    # send() returns once this end's unsent bytes are at or under it; the gateway's end uses 0.
    write_limit_bytes: int
    # A longer message sent to this end ends the pair with 1009, as the library's max_size does.
    max_message_bytes: int
    # How long this end's close waits for the closing handshake to end before it aborts the
    # pair: the adapter's close timeout.
    close_timeout_ns: int
    # The library's receive queue high-water mark, in messages: while more are unread, the
    # library reads nothing from the socket, a close frame included, until the application has
    # taken all but a quarter of them. The default is the library's own, which the robots keep;
    # the gateway sets receive_queue_frames from its configuration.
    receive_queue_frames: int = 16


class _Pair:
    """What the two ends of one connection share."""

    def __init__(self) -> None:
        # The end that dropped the connection: by abort(), at its close timeout, or as the
        # receiver of a message over its limit. Nothing moves between the ends once it is set.
        self.aborted_by: MemoryConnection | None = None
        self.changed = asyncio.Event()

    @property
    def aborted(self) -> bool:
        return self.aborted_by is not None

    def notify(self) -> None:
        """Wake every task that waits on either end, to check its condition again."""
        self.changed.set()
        self.changed = asyncio.Event()


def _size_bytes(message: str | bytes) -> int:
    return len(message.encode()) if isinstance(message, str) else len(message)


class MemoryConnection:
    """One end of an in-memory connection.

    recv() takes from the inbox and refills it from the peer's outbox. send() returns once
    unsent_bytes() is at or under this end's write limit; send_nowait() never waits. A message
    longer than the receiver's max_message_bytes ends the pair as the library's 1009 failure
    does: the receiver sends a close with 1009, unless it has sent or read a close already, and
    drops the connection, so nothing more reaches either end.

    close(code, reason): this end stops sending, and its close follows its outbox to the peer.
    Once the peer has read it, the peer's send_nowait() raises TransportClosed(None, None), as
    the adapter refuses a send in any state but OPEN, and its send() raises
    TransportClosed(code, None); this end's send_nowait() and send() raise as soon as it has
    called close(), the latter with TransportClosed(None, code). recv() on either end returns
    what its inbox holds and raises only once the connection is lost: the handshake has ended
    or the pair is aborted. Every TransportClosed with codes carries the first close its end
    sent or read, as the library builds the exception from the close frames it has sent and
    read: TransportClosed(None, code) on an end that sent its close first, also when that close
    timed out, TransportClosed(code, None) on an end that read the peer's close first, and no
    codes on an end that did neither. A close() on an end that has already sent or read a
    close sends nothing and waits the same way, so it returns at once when the handshake has
    ended.

    abort() ends both ends at once, and is a no-op once the connection is lost: both outboxes
    are discarded, with any close still waiting in them, and every send raises TransportClosed,
    send() with the codes recv() would report. Each recv() still returns what its inbox holds,
    and a close that had entered an inbox is read after the messages ahead of it. A send()
    blocked at the write limit waits, as the library's drain does, until its bytes have gone or
    the connection is lost; reading the peer's close does not end it. It raises
    TransportClosed when the peer drops the connection, by abort() or at the end of the peer's
    close timeout. When this end aborts, it returns without raising, as the library's does (its
    drain waiter is completed, not failed, when the transport is dropped locally), and this
    end's next send raises.

    A recv() while another recv() of this end waits raises RuntimeError, as the library raises
    its ConcurrencyError, a RuntimeError.
    """

    # Set by memory_pair, which has to make both ends before it can link them.
    _peer: MemoryConnection

    def __init__(self, limits: PipeLimits, pair: _Pair) -> None:
        self._limits = limits
        self._pair = pair
        # Takes messages while it holds less than capacity_bytes; while it is full, the peer's
        # messages wait in the peer's outbox.
        self._inbox: deque[str | bytes] = deque()
        self._inbox_bytes = 0
        # False while the library reads nothing from the socket. The last messages_in_socket of
        # the inbox came in meanwhile, so they join the library's queue only once it reads
        # again.
        self._reading = True
        self._messages_in_socket = 0
        # The peer's close, behind every message of the inbox, which the library has not read.
        self._close_in_socket: int | None = None
        # Unbounded, as the library's write buffer is: the code under test bounds what it leaves
        # here, through the write limit and its own queues.
        self._outbox: deque[str | bytes] = deque()
        self._outbox_bytes = 0
        # This end's close or its answer to the peer's, waiting behind the outbox.
        self._outbox_close: int | None = None
        # The first close this end sent or read, with the end that sent it: the source of this
        # end's codes. This end sends nothing more once it is set.
        self._first_close: tuple[MemoryConnection, int] | None = None
        self._read_close = False
        self._rtt_ns: int | None = None
        self._receiving = False

    async def recv(self) -> str | bytes:
        if self._receiving:
            raise RuntimeError("another recv() is already waiting on this connection")
        self._receiving = True
        try:
            while not self._inbox:
                if self._lost():
                    raise TransportClosed(*self._codes())
                # Completed by a message entering this inbox, the end of the closing handshake,
                # or an abort.
                await self._pair.changed.wait()
        finally:
            # A cancelled recv() frees the connection for the next one.
            self._receiving = False
        message = self._inbox.popleft()
        self._inbox_bytes -= _size_bytes(message)
        queued_messages = len(self._inbox) - self._messages_in_socket
        if not self._reading and queued_messages <= self._limits.receive_queue_frames // 4:
            # The library reads again, and takes what waited in the socket at once: the
            # messages, then a close behind them.
            self._reading = True
            self._messages_in_socket = 0
            self._take_close()
            self._reading = len(self._inbox) <= self._limits.receive_queue_frames
        self._peer._flush()
        return message

    async def send(self, message: str | bytes) -> None:
        if not self._open():
            raise TransportClosed(*self._codes())
        self.send_nowait(message)
        while self._outbox_bytes > self._limits.write_limit_bytes:
            # Completed by the peer reading from its inbox or by an abort. The handshake cannot
            # end meanwhile, because this end's close or answer waits behind these bytes.
            await self._pair.changed.wait()
            if self._pair.aborted_by is self:
                # The library completes its drain waiter, and does not fail it, when its own
                # side drops the transport.
                return
            if self._pair.aborted:
                raise TransportClosed(*self._codes())

    def send_nowait(self, message: str | bytes) -> None:
        if not self._open():
            # The adapter refuses a send in any state but OPEN, before any code is known.
            raise TransportClosed(None, None)
        self._outbox.append(message)
        self._outbox_bytes += _size_bytes(message)
        self._flush()

    def unsent_bytes(self) -> int:
        return self._outbox_bytes

    def rtt_ns(self) -> int | None:
        return self._rtt_ns

    def set_rtt_ns(self, value: int | None) -> None:
        """Set what rtt_ns() returns: the round trip of the library's latest ping."""
        self._rtt_ns = value

    async def close(self, code: int, reason: str) -> None:
        if self._open():
            self._first_close = (self, code)
            self._outbox_close = code
            self._flush()
        try:
            with contextlib.suppress(TimeoutError):
                async with asyncio.timeout(self._limits.close_timeout_ns / 1e9):
                    while not self._lost():
                        # Completed by the end of the closing handshake or by an abort.
                        await self._pair.changed.wait()
        finally:
            self.abort()

    def abort(self) -> None:
        if self._lost():
            return
        self._pair.aborted_by = self
        for end in (self, self._peer):
            end._outbox.clear()
            end._outbox_bytes = 0
            end._outbox_close = None
        self._pair.notify()

    def _open(self) -> bool:
        return self._first_close is None and not self._pair.aborted

    def _lost(self) -> bool:
        """Tell whether the connection is lost: aborted, or each end has read a close."""
        return self._pair.aborted or (self._read_close and self._peer._read_close)

    def _codes(self) -> tuple[int | None, int | None]:
        """Return (peer_code, local_code) from the first close this end sent or read."""
        if self._first_close is None:
            return None, None
        sender, code = self._first_close
        return (None, code) if sender is self else (code, None)

    def _take_close(self) -> None:
        """Read the peer's close behind the inbox, once it has come and the library reads."""
        if self._close_in_socket is None or not self._reading:
            return
        code = self._close_in_socket
        self._close_in_socket = None
        self._read_close = True
        if self._first_close is None:
            self._first_close = (self._peer, code)
            if not self._pair.aborted:
                self._outbox_close = code
                self._flush()

    def _flush(self) -> None:
        """Move unsent messages, then a close, into the peer's inbox while it has room."""
        peer = self._peer
        while self._outbox and peer._inbox_bytes < peer._limits.capacity_bytes:
            message = self._outbox.popleft()
            size_bytes = _size_bytes(message)
            self._outbox_bytes -= size_bytes
            if size_bytes > peer._limits.max_message_bytes:
                # The peer's library fails the connection as soon as it reads the frame header:
                # it closes with 1009, unless it has sent or read a close already, whose code
                # then stands, and drops the connection, so nothing else reaches this end. The
                # peer drops it, so a send() of this end blocked at the write limit raises.
                closes = peer._first_close is None
                if closes:
                    peer._first_close = (peer, CloseCode.MESSAGE_TOO_BIG)
                peer.abort()
                if closes:
                    self._close_in_socket = CloseCode.MESSAGE_TOO_BIG
                    self._take_close()
                return
            peer._inbox.append(message)
            peer._inbox_bytes += size_bytes
            if peer._reading:
                peer._reading = len(peer._inbox) <= peer._limits.receive_queue_frames
            else:
                peer._messages_in_socket += 1
        if (
            self._outbox_close is not None
            and not self._outbox
            and peer._inbox_bytes < peer._limits.capacity_bytes
        ):
            peer._close_in_socket = self._outbox_close
            self._outbox_close = None
            peer._take_close()
        self._pair.notify()


def memory_pair(
    client: PipeLimits, server: PipeLimits
) -> tuple[MemoryConnection, MemoryConnection]:
    """Return the two ends of a new connection: the client's, then the server's."""
    pair = _Pair()
    client_end = MemoryConnection(client, pair)
    server_end = MemoryConnection(server, pair)
    client_end._peer = server_end
    server_end._peer = client_end
    return client_end, server_end


class MemoryListener:
    """Serves in-memory connections as the WebSocket server serves sockets.

    An async context manager that owns a TaskGroup of the handler tasks, as the library owns
    its handler tasks. Its port is 0.
    """

    def __init__(
        self, handler: Handler, http: HttpHandler, client: PipeLimits, server: PipeLimits
    ) -> None:
        self._handler = handler
        self._http = http
        self._client = client
        self._server = server
        self._tasks = asyncio.TaskGroup()
        # The handlers still running, which wait_closed() waits for.
        self._handlers: set[asyncio.Task[None]] = set()
        self._stopped = asyncio.Event()

    async def __aenter__(self) -> Self:
        await self._tasks.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self._tasks.__aexit__(exc_type, exc, traceback)

    @property
    def port(self) -> int:
        return 0

    @property
    def stopped(self) -> bool:
        """True once stop_accepting() has been called."""
        return self._stopped.is_set()

    async def connect(self, path: str) -> MemoryConnection:
        """Open a connection to path and return the client's end.

        The HTTP handler answers first, and its reply refuses the upgrade with the reply's
        status, as the real server answers before the upgrade; otherwise the handler starts on
        the server's end.
        """
        if self.stopped:
            raise TransportConnectError(
                "the listener has stopped accepting", HTTPStatus.SERVICE_UNAVAILABLE
            )
        reply = await self._http(path)
        if reply is not None:
            raise TransportConnectError(
                f"{path} was answered with {reply.status} instead of an upgrade", reply.status
            )
        if self.stopped:
            # The library answers 503 to an upgrade still in progress when it stops accepting.
            raise TransportConnectError(
                "the listener stopped accepting during the upgrade", HTTPStatus.SERVICE_UNAVAILABLE
            )
        client_end, server_end = memory_pair(self._client, self._server)
        handler = self._tasks.create_task(self._serve(server_end))
        self._handlers.add(handler)
        handler.add_done_callback(self._handlers.discard)
        return client_end

    def stop_accepting(self) -> None:
        self._stopped.set()

    async def wait_closed(self) -> None:
        # Completed by stop_accepting(): like the library's, this returns only after it.
        await self._stopped.wait()
        if self._handlers:
            # Completed by the handler tasks, which end their connections themselves.
            await asyncio.wait(self._handlers)

    async def _serve(self, connection: MemoryConnection) -> None:
        # The TaskGroup takes a coroutine, while a Handler may return any awaitable.
        await self._handler(connection)
