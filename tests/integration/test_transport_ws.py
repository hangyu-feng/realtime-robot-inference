"""The WebSocket adapter on real loopback sockets and uvloop, seen from both ends.

Each test checks what an end is told about messages, closes, aborts, limits and refusals. Every
wait is bounded by one asyncio.timeout around the body of its test.
"""

from __future__ import annotations

import asyncio
import contextlib
import socket
import sys
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace

import pytest

from common.transport import Connection, HttpReply, TransportClosed, TransportConnectError
from common.transport_ws import (
    ClientOptions,
    ServerOptions,
    SocketBuffers,
    WsConnection,
    WsListener,
    connect_ws,
    serve_ws,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the adapter sets Linux socket options, and the services' loop is uvloop",
)

LOOPBACK = "127.0.0.1"
# Bounds every wait of a test. A healthy run ends long before it.
TEST_TIMEOUT_S = 20.0
# A wait that is still unfinished after this long is taken to be blocked.
BLOCKED_S = 0.2
# The close timeout of the end that closes on a peer that does not read.
SHORT_CLOSE_TIMEOUT_S = 0.1
MAX_MESSAGE_BYTES = 64 * 1024
RCVBUF_BYTES = 40_960
SNDBUF_BYTES = 4096

# The pings are far apart, so none is sent during a test.
SERVER_OPTIONS = ServerOptions(
    host=LOOPBACK,
    port=0,
    max_message_bytes=MAX_MESSAGE_BYTES,
    receive_queue_frames=4,
    rcvbuf_bytes=RCVBUF_BYTES,
    sndbuf_bytes=SNDBUF_BYTES,
    ping_interval_s=60.0,
    ping_timeout_s=60.0,
    open_timeout_s=5.0,
    close_timeout_s=5.0,
)
CLIENT_OPTIONS = ClientOptions(
    max_message_bytes=MAX_MESSAGE_BYTES,
    rcvbuf_bytes=SNDBUF_BYTES,
    sndbuf_bytes=RCVBUF_BYTES,
    notsent_lowat_bytes=16 * 1024,
    open_timeout_s=5.0,
    close_timeout_s=5.0,
)

FLOOD_MESSAGE = bytes(1024)
# A megabyte: far more than the path to a peer that reads nothing can take, which is its
# kernel buffers of a few kilobytes and the few frames its library queues before it stops
# reading the socket. So most of a flood stays unsent for as long as the peer reads nothing.
FLOOD_MESSAGES = 1024

NO_ROUTES: Mapping[str, HttpReply] = {}


@dataclass(frozen=True)
class Service:
    """A listener under test, and what its handler has been given."""

    listener: WsListener
    url: str
    # The server end of every connection, in the order of the upgrades. One entry per
    # connection that a test opens, so the test bounds it.
    accepted: asyncio.Queue[Connection]
    # Ends every handler.
    release: asyncio.Event


@contextlib.asynccontextmanager
async def serving(
    options: ServerOptions = SERVER_OPTIONS, routes: Mapping[str, HttpReply] = NO_ROUTES
) -> AsyncIterator[Service]:
    """Run a listener whose handler keeps each connection until released; close it on exit."""
    accepted: asyncio.Queue[Connection] = asyncio.Queue()
    server_ends: list[Connection] = []
    release = asyncio.Event()

    async def handler(connection: Connection) -> None:
        server_ends.append(connection)
        accepted.put_nowait(connection)
        # Set by the test, or below when the test has ended.
        await release.wait()

    async def http(path: str) -> HttpReply | None:
        return routes.get(path)

    listener = await serve_ws(handler, http, options)
    try:
        yield Service(listener, f"ws://{LOOPBACK}:{listener.port}/ws", accepted, release)
    finally:
        release.set()
        # A failed test may leave a connection whose closing handshake could not end.
        for server_end in server_ends:
            server_end.abort()
        listener.stop_accepting()
        await listener.wait_closed()


@contextlib.asynccontextmanager
async def connected(
    service: Service, options: ClientOptions = CLIENT_OPTIONS
) -> AsyncIterator[tuple[WsConnection, Connection]]:
    """Open one connection to the service; yield its client end and its server end."""
    client = await connect_ws(service.url, options)
    try:
        yield client, await service.accepted.get()
    finally:
        client.abort()


@contextlib.asynccontextmanager
async def answering(reply: bytes) -> AsyncIterator[str]:
    """Run a server that answers every request with the same bytes; yield its ws URL."""

    async def answer(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        await reader.readuntil(b"\r\n\r\n")
        writer.write(reply)
        writer.close()
        await writer.wait_closed()

    server = await asyncio.start_server(answer, LOOPBACK, 0)
    async with server:
        yield f"ws://{LOOPBACK}:{server.sockets[0].getsockname()[1]}/ws"


async def read_until_closed(connection: Connection) -> TransportClosed:
    """Read the connection to its end and return the error that ends it."""
    with pytest.raises(TransportClosed) as raised:
        while True:
            await connection.recv()
    return raised.value


def flood(connection: Connection) -> None:
    for _ in range(FLOOD_MESSAGES):
        connection.send_nowait(FLOOD_MESSAGE)


async def send_until_blocked(connection: Connection) -> list[int]:
    """Send until a send stays unfinished; return the unsent bytes after each send that ended."""
    unsent_bytes_after_each_send: list[int] = []
    while True:
        try:
            async with asyncio.timeout(BLOCKED_S):
                await connection.send(FLOOD_MESSAGE)
        except TimeoutError:
            return unsent_bytes_after_each_send
        unsent_bytes_after_each_send.append(connection.unsent_bytes())


async def http_get(port: int, target: str) -> bytes:
    """Send a plain GET and return everything the server answers before it closes."""
    reader, writer = await asyncio.open_connection(LOOPBACK, port)
    try:
        writer.write(f"GET {target} HTTP/1.1\r\nHost: {LOOPBACK}\r\n\r\n".encode())
        return await reader.read()
    finally:
        writer.close()
        await writer.wait_closed()


async def test_text_and_binary_messages_arrive_unchanged_in_both_directions() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        await client.send("hello")
        client.send_nowait(b"\x00frame")
        await server_end.send("welcome")
        server_end.send_nowait(b"\x01command")
        at_server = [await server_end.recv(), await server_end.recv()]
        at_client = [await client.recv(), await client.recv()]

    assert at_server == ["hello", b"\x00frame"]
    assert at_client == ["welcome", b"\x01command"]


async def test_a_message_of_exactly_the_limit_arrives() -> None:
    message = bytes(range(256)) * (MAX_MESSAGE_BYTES // 256)
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        await client.send(message)
        received = await server_end.recv()

    assert received == message


async def test_an_oversized_message_closes_the_connection_with_1009_from_the_receiver() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        client.send_nowait(bytes(MAX_MESSAGE_BYTES + 1))
        at_receiver = await read_until_closed(server_end)
        at_sender = await read_until_closed(client)

    assert (at_receiver.peer_code, at_receiver.local_code) == (None, 1009)
    assert (at_sender.peer_code, at_sender.local_code) == (1009, None)


@pytest.mark.parametrize("closer", [0, 1], ids=["the client closes", "the server closes"])
async def test_a_close_gives_its_code_as_local_to_the_closer_and_as_peer_to_the_other(
    closer: int,
) -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as ends,
    ):
        await ends[closer].close(4001, "replaced")
        at_closer = await read_until_closed(ends[closer])
        at_other = await read_until_closed(ends[1 - closer])

    assert (at_closer.peer_code, at_closer.local_code) == (None, 4001)
    assert (at_other.peer_code, at_other.local_code) == (4001, None)


@pytest.mark.parametrize("aborter", [0, 1], ids=["the client aborts", "the server aborts"])
async def test_an_abort_gives_neither_end_a_code(aborter: int) -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as ends,
    ):
        ends[aborter].abort()
        at_aborter = await read_until_closed(ends[aborter])
        at_other = await read_until_closed(ends[1 - aborter])

    assert (at_aborter.peer_code, at_aborter.local_code) == (None, None)
    assert (at_other.peer_code, at_other.local_code) == (None, None)


async def test_a_second_close_returns_and_the_first_code_stands() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, _),
    ):
        await client.close(1000, "done")
        await client.close(4001, "again")
        at_client = await read_until_closed(client)

    assert (at_client.peer_code, at_client.local_code) == (None, 1000)


async def test_a_close_after_an_abort_returns_and_sends_no_code() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        server_end.abort()
        await server_end.close(1000, "after the abort")
        at_client = await read_until_closed(client)

    assert (at_client.peer_code, at_client.local_code) == (None, None)


async def test_a_send_to_a_client_that_never_reads_blocks_and_a_bounded_close_aborts() -> None:
    options = replace(SERVER_OPTIONS, close_timeout_s=SHORT_CLOSE_TIMEOUT_S)
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving(options) as service,
        connected(service) as (client, server_end),
    ):
        unsent_bytes_after_each_send = await send_until_blocked(server_end)
        flood(server_end)
        unsent_bytes = server_end.unsent_bytes()
        server_end.send_nowait(FLOOD_MESSAGE)
        unsent_after_one_more_bytes = server_end.unsent_bytes()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(BLOCKED_S):
                await server_end.send(FLOOD_MESSAGE)
        # Far above the close timeout and far below the bound of the test: the close must
        # return by its own timeout.
        async with asyncio.timeout(TEST_TIMEOUT_S / 4):
            await server_end.close(1000, "done")
        at_client = await read_until_closed(client)

    # With a write limit of 0, a send returns only when the kernel has taken every byte.
    assert set(unsent_bytes_after_each_send) == {0}
    assert unsent_bytes > 0
    assert unsent_after_one_more_bytes >= unsent_bytes + len(FLOOD_MESSAGE)
    # The close frame waited behind the unsent bytes and was dropped with them.
    assert (at_client.peer_code, at_client.local_code) == (None, None)


async def test_a_blocked_send_returns_when_its_own_side_aborts_and_the_next_send_raises() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        flood(server_end)
        # The group's exit awaits the send, so it passes only if the send returns without raising.
        async with asyncio.TaskGroup() as group:
            blocked = group.create_task(server_end.send(FLOOD_MESSAGE))
            _, still_blocked = await asyncio.wait({blocked}, timeout=BLOCKED_S)
            server_end.abort()
        with pytest.raises(TransportClosed) as raised:
            await server_end.send(FLOOD_MESSAGE)

    assert still_blocked == {blocked}
    assert (raised.value.peer_code, raised.value.local_code) == (None, None)


async def test_send_nowait_after_a_close_raises() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, _),
    ):
        await client.close(1000, "done")
        with pytest.raises(TransportClosed) as raised:
            client.send_nowait(b"late")

    assert (raised.value.peer_code, raised.value.local_code) == (None, None)


async def test_send_nowait_raises_while_a_close_waits_behind_unsent_bytes() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        flood(server_end)
        # The abort ends the close, which the group's exit awaits.
        async with asyncio.TaskGroup() as group:
            closing = group.create_task(server_end.close(1000, "done"))
            _, still_closing = await asyncio.wait({closing}, timeout=BLOCKED_S)
            with pytest.raises(TransportClosed) as raised:
                server_end.send_nowait(FLOOD_MESSAGE)
            server_end.abort()

    assert still_closing == {closing}
    assert (raised.value.peer_code, raised.value.local_code) == (None, None)


async def test_send_nowait_right_after_an_abort_raises() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        server_end.abort()
        with pytest.raises(TransportClosed) as raised:
            server_end.send_nowait(b"late")

    assert (raised.value.peer_code, raised.value.local_code) == (None, None)


async def test_unsent_bytes_are_none_right_after_an_abort() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        flood(server_end)
        before = server_end.unsent_bytes()
        server_end.abort()
        # In the step of the abort, before the loop has reported the loss.
        after = server_end.unsent_bytes()

    assert before > 0
    assert after == 0


async def test_send_right_after_an_abort_raises() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        server_end.abort()
        with pytest.raises(TransportClosed) as raised:
            await server_end.send(b"late")

    assert (raised.value.peer_code, raised.value.local_code) == (None, None)


async def test_after_the_servers_close_the_client_cannot_send_and_still_reads_the_code() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        await server_end.send("last")
        await server_end.close(4003, "idle")
        with pytest.raises(TransportClosed) as refused:
            client.send_nowait(b"late")
        last = await client.recv()
        at_client = await read_until_closed(client)

    assert (refused.value.peer_code, refused.value.local_code) == (None, None)
    assert last == "last"
    assert (at_client.peer_code, at_client.local_code) == (4003, None)


async def test_the_round_trip_is_unknown_before_the_first_pong() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        round_trips = (client.rtt_ns(), server_end.rtt_ns())

    assert round_trips == (None, None)


async def test_the_listener_reports_the_buffers_the_kernel_granted() -> None:
    async with asyncio.timeout(TEST_TIMEOUT_S), serving() as service:
        buffers = service.listener.buffers

    # Linux doubles a requested size (socket(7)).
    assert buffers == SocketBuffers(rcvbuf_bytes=2 * RCVBUF_BYTES, sndbuf_bytes=2 * SNDBUF_BYTES)


@pytest.mark.parametrize(
    ("status", "status_line", "body"),
    [
        (200, b"HTTP/1.1 200 OK", b'{"status":"ok"}'),
        (503, b"HTTP/1.1 503 Service Unavailable", b'{"status":"starting"}'),
        (200, b"HTTP/1.1 200 OK", b'{"pad":"' + b"x" * 300_000 + b'"}'),
    ],
    ids=["a reply", "a refusal", "a body larger than the socket buffers"],
)
async def test_a_plain_get_receives_the_handlers_status_and_json_body(
    status: int, status_line: bytes, body: bytes
) -> None:
    routes = {"/metrics": HttpReply(status, body)}
    async with asyncio.timeout(TEST_TIMEOUT_S), serving(routes=routes) as service:
        answer = await http_get(service.listener.port, "/metrics?verbose=1")

    head, _, received_body = answer.partition(b"\r\n\r\n")
    assert head.split(b"\r\n") == [
        status_line,
        b"Content-Type: application/json",
        b"Content-Length: %d" % len(body),
        b"Connection: close",
    ]
    assert received_body == body


async def test_a_reply_on_the_upgrade_path_refuses_the_connection_with_its_status() -> None:
    routes = {"/ws": HttpReply(503, b'{"status":"starting"}')}
    async with asyncio.timeout(TEST_TIMEOUT_S), serving(routes=routes) as service:
        with pytest.raises(TransportConnectError) as raised:
            await connect_ws(service.url, CLIENT_OPTIONS)

    assert raised.value.http_status == 503


@pytest.mark.parametrize(
    ("reply", "http_status"),
    [
        (b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n", 404),
        (
            b"HTTP/1.1 301 Moved Permanently\r\nLocation: ws://127.0.0.1:1/ws\r\n"
            b"Content-Length: 0\r\n\r\n",
            301,
        ),
        (b"not an HTTP response\r\n\r\n", None),
    ],
    ids=["another status", "a redirect", "not HTTP"],
)
async def test_an_answer_other_than_the_upgrade_is_a_connect_error_with_its_status(
    reply: bytes, http_status: int | None
) -> None:
    async with asyncio.timeout(TEST_TIMEOUT_S), answering(reply) as url:
        with pytest.raises(TransportConnectError) as raised:
            await connect_ws(url, CLIENT_OPTIONS)

    assert raised.value.http_status == http_status


async def test_a_refused_connection_is_a_connect_error_without_a_status() -> None:
    # A socket that is bound and does not listen refuses every connection to its port.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as closed_door:
        closed_door.bind((LOOPBACK, 0))
        url = f"ws://{LOOPBACK}:{closed_door.getsockname()[1]}/ws"
        async with asyncio.timeout(TEST_TIMEOUT_S):
            with pytest.raises(TransportConnectError) as raised:
                await connect_ws(url, CLIENT_OPTIONS)

    assert raised.value.http_status is None
    assert isinstance(raised.value.__cause__, ConnectionRefusedError)


async def test_an_upgrade_that_is_never_answered_is_a_connect_error_at_the_open_timeout() -> None:
    options = replace(CLIENT_OPTIONS, open_timeout_s=BLOCKED_S)
    # The kernel completes connections to a listening socket that nothing accepts from.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as silent:
        silent.bind((LOOPBACK, 0))
        silent.listen(1)
        url = f"ws://{LOOPBACK}:{silent.getsockname()[1]}/ws"
        async with asyncio.timeout(TEST_TIMEOUT_S):
            with pytest.raises(TransportConnectError) as raised:
                await connect_ws(url, options)

    assert raised.value.http_status is None
    assert isinstance(raised.value.__cause__, TimeoutError)


async def test_stop_accepting_refuses_new_connections_and_keeps_the_open_one() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (client, server_end),
    ):
        service.listener.stop_accepting()
        with pytest.raises(TransportConnectError):
            await connect_ws(service.url, CLIENT_OPTIONS)
        await client.send(b"still open")
        received = await server_end.recv()

    assert received == b"still open"


async def test_wait_closed_returns_once_the_open_connection_and_its_handler_have_gone() -> None:
    async with (
        asyncio.timeout(TEST_TIMEOUT_S),
        serving() as service,
        connected(service) as (_, server_end),
    ):
        service.listener.stop_accepting()
        with pytest.raises(TimeoutError):
            async with asyncio.timeout(BLOCKED_S):
                await service.listener.wait_closed()
        await server_end.close(1001, "shutdown")
        service.release.set()
        await service.listener.wait_closed()
