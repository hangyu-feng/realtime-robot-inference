"""The WebSocket adapter: a Connection and a Listener over the websockets library.

The only module that imports websockets. The server side is the gateway's listener with its
plain HTTP routes, and the client side is a robot's dial. Both set their socket options before
the connection exists, so the kernel's buffers are bounded from the first byte. The options
are Linux's, where the services run.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
from dataclasses import dataclass
from http import HTTPStatus

from websockets.asyncio.client import connect
from websockets.asyncio.connection import Connection as WebSocket
from websockets.asyncio.server import Server, ServerConnection, broadcast, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed, InvalidHandshake, InvalidStatus
from websockets.http11 import Request, Response
from websockets.protocol import State
from websockets.uri import parse_uri

from common.transport import Handler, HttpHandler, TransportClosed, TransportConnectError

# Linux's number for the option, from linux/tcp.h. Some CPython builds, such as the standalone
# builds that uv installs, do not export socket.TCP_NOTSENT_LOWAT.
TCP_NOTSENT_LOWAT = 25


@dataclass(frozen=True)
class ServerOptions:
    """What the listener needs: its address, its limits, its socket buffers and its timeouts."""

    host: str
    port: int
    max_message_bytes: int
    receive_queue_frames: int
    rcvbuf_bytes: int
    sndbuf_bytes: int
    ping_interval_s: float
    ping_timeout_s: float
    open_timeout_s: float
    close_timeout_s: float


@dataclass(frozen=True)
class ClientOptions:
    """What a dial needs: its message limit, its socket settings and its timeouts."""

    max_message_bytes: int
    rcvbuf_bytes: int
    sndbuf_bytes: int
    notsent_lowat_bytes: int
    open_timeout_s: float
    close_timeout_s: float


@dataclass(frozen=True)
class SocketBuffers:
    """The buffer sizes of a socket as the kernel granted them; Linux reports twice the request."""

    rcvbuf_bytes: int
    sndbuf_bytes: int


def _transport_closed(error: ConnectionClosed) -> TransportClosed:
    """Translate the library's report of a closed connection, by whose close frame came first."""
    # The exception's own code and reason are deprecated and warn, so only its frames are read.
    if error.rcvd is not None and (error.sent is None or error.rcvd_then_sent):
        return TransportClosed(error.rcvd.code, None)
    if error.sent is not None:
        return TransportClosed(None, error.sent.code)
    return TransportClosed(None, None)


class WsConnection:
    """A Connection over one websockets connection, on either end.

    recv and send wait for as long as the peer makes them; the caller sets the bound.
    """

    def __init__(self, websocket: WebSocket, close_timeout_s: float) -> None:
        self._websocket = websocket
        self._close_timeout_s = close_timeout_s

    async def recv(self) -> str | bytes:
        try:
            return await self._websocket.recv()
        except ConnectionClosed as error:
            raise _transport_closed(error) from error

    async def send(self, message: str | bytes) -> None:
        # A transport that has been dropped on this side discards a write, and the library
        # still counts the connection open until the loop reports the loss, so its send would
        # return as if the message had gone.
        if self._websocket.state is State.OPEN and self._websocket.transport.is_closing():
            raise TransportClosed(None, None)
        try:
            await self._websocket.send(message)
        except ConnectionClosed as error:
            raise _transport_closed(error) from error

    def send_nowait(self, message: str | bytes) -> None:
        # broadcast is the library's public synchronous send: it writes and never awaits a
        # drain. On its own it would skip a connection that is not open without a word, and a
        # dropped transport discards the write as it does in send, so both are refused here.
        if self._websocket.state is not State.OPEN or self._websocket.transport.is_closing():
            raise TransportClosed(None, None)
        try:
            broadcast((self._websocket,), message, raise_exceptions=True)
        except ExceptionGroup as error:
            raise TransportClosed(None, None) from error

    def unsent_bytes(self) -> int:
        # What a closing transport still holds is no longer this side's to send: an abort has
        # discarded it, a close sends it on its own, and send() refuses from then on.
        transport = self._websocket.transport
        return 0 if transport.is_closing() else transport.get_write_buffer_size()

    def rtt_ns(self) -> int | None:
        # The library measures a round trip with loop.time(), which uvloop serves in whole
        # milliseconds. On one host a round trip is far below a millisecond, so a pong usually
        # measures 0.0 and this is usually None, also after pongs have arrived.
        latency_s = self._websocket.latency
        return None if latency_s == 0.0 else round(latency_s * 1e9)

    async def close(self, code: int, reason: str) -> None:
        try:
            async with asyncio.timeout(self._close_timeout_s):
                await self._websocket.close(code, reason)
        except TimeoutError:
            # The expected outcome on a peer that does not read: the library writes the close
            # frame behind the unsent bytes and waits for them to drain before its own close
            # timeout starts.
            pass
        finally:
            # A no-op on a transport that has closed. It drops a connection whose handshake
            # did not end in time, also when the caller is cancelled.
            self._websocket.transport.abort()

    def abort(self) -> None:
        self._websocket.transport.abort()


class WsListener:
    """The Listener over a websockets server, with the socket buffers its connections inherit."""

    def __init__(self, server: Server, port: int, buffers: SocketBuffers) -> None:
        self._server = server
        self.port = port
        self.buffers = buffers

    def stop_accepting(self) -> None:
        # The only call to the server's close, which acts once. It leaves the open connections
        # alone, because this program closes each one itself, with that connection's own code.
        self._server.close(close_connections=False)

    async def wait_closed(self) -> None:
        # Returns only after stop_accepting; the caller sets the bound.
        await self._server.wait_closed()


async def serve_ws(handler: Handler, http: HttpHandler, options: ServerOptions) -> WsListener:
    """Listen on the address of the options, answer plain HTTP through http, and run handler.

    Raises OSError when the address cannot be bound.
    """

    async def route(connection: ServerConnection, request: Request) -> Response | None:
        # The library runs this inside its open timeout, which bounds the handler too.
        reply = await http(request.path.partition("?")[0])
        if reply is None:
            return None
        # The headers are built whole: assigning an item to Headers appends to it.
        headers = Headers(
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(reply.body))),
                ("Connection", "close"),
            ]
        )
        return Response(reply.status, HTTPStatus(reply.status).phrase, headers, reply.body)

    async def wrapper(websocket: ServerConnection) -> None:
        await handler(WsConnection(websocket, options.close_timeout_s))

    with contextlib.ExitStack() as stack:
        # ASSUMPTION: A21. Plain ws on IPv4: TLS ends at a load balancer in front of the gateway.
        sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Set before bind: every accepted socket inherits them, and the window scale is
        # negotiated for the small receive buffer.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, options.rcvbuf_bytes)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, options.sndbuf_bytes)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.bind((options.host, options.port))
        buffers = SocketBuffers(
            rcvbuf_bytes=sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
            sndbuf_bytes=sock.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
        )
        port: int = sock.getsockname()[1]
        server = await serve(
            wrapper,
            sock=sock,
            # JPEG does not compress further (DESIGN.md 4.3).
            compression=None,
            max_size=options.max_message_bytes,
            max_queue=options.receive_queue_frames,
            # The transport pauses as soon as any byte stays in user space. A paused transport
            # then means unsent bytes, which the slow-consumer rule acts on, and a send whose
            # bytes the kernel took returns without suspending.
            write_limit=0,
            ping_interval=options.ping_interval_s,
            ping_timeout=options.ping_timeout_s,
            open_timeout=options.open_timeout_s,
            close_timeout=options.close_timeout_s,
            process_request=route,
            server_header=None,
        )
        # The server owns the socket from here on.
        stack.pop_all()
    return WsListener(server, port, buffers)


async def connect_ws(url: str, options: ClientOptions) -> WsConnection:
    """Dial the URL and return the open connection.

    Raises TransportConnectError when the name does not resolve, the connection or the upgrade
    is refused, or all of it takes longer than the open timeout.
    """
    loop = asyncio.get_running_loop()
    target = parse_uri(url)
    with contextlib.ExitStack() as stack:
        try:
            sock = stack.enter_context(socket.socket(socket.AF_INET, socket.SOCK_STREAM))
            # Set before connecting, so the kernel stops taking new frame bytes once that
            # much is unsent (DESIGN.md 4.3) and never holds more than the buffers allow.
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, options.sndbuf_bytes)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, options.rcvbuf_bytes)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.IPPROTO_TCP, TCP_NOTSENT_LOWAT, options.notsent_lowat_bytes)
            sock.setblocking(False)
            # One bound for the name lookup, the TCP connection and the upgrade: each is a
            # wait on a peer, the lookup too when the name is served by Docker's DNS.
            async with asyncio.timeout(options.open_timeout_s):
                addresses = await loop.getaddrinfo(
                    target.host, target.port, family=socket.AF_INET, type=socket.SOCK_STREAM
                )
                await loop.sock_connect(sock, addresses[0][4])
                # No pings from this side: a robot's liveness rule is its send stall, and the
                # library still answers the gateway's pings.
                websocket = await connect(
                    url,
                    sock=sock,
                    proxy=None,
                    compression=None,
                    max_size=options.max_message_bytes,
                    ping_interval=None,
                    open_timeout=None,
                    close_timeout=options.close_timeout_s,
                    user_agent_header=None,
                )
        except (OSError, InvalidHandshake, ValueError) as error:
            # A redirect arrives as a ValueError, because the library cannot follow one on a
            # socket it was given; the refused upgrade is then its cause.
            refusal = error.__cause__ if isinstance(error, ValueError) else error
            http_status = (
                refusal.response.status_code if isinstance(refusal, InvalidStatus) else None
            )
            raise TransportConnectError(f"cannot connect to {url}: {error}", http_status) from error
        # The connection owns the socket from here on.
        stack.pop_all()
    return WsConnection(websocket, options.close_timeout_s)
