"""Tests for the in-memory connection pair and listener that the async unit tests run on."""

import asyncio

import pytest

from common.transport import (
    Connection,
    HttpReply,
    Listener,
    TransportClosed,
    TransportConnectError,
)
from tests.support.memory_transport import MemoryConnection, MemoryListener, PipeLimits, memory_pair
from tests.support.virtual_time import LoopClock

# The two ends differ in every limit, so a test fails when the pair applies a limit of the wrong
# end. The robot's end.
CLIENT = PipeLimits(
    capacity_bytes=100,
    write_limit_bytes=50,
    max_message_bytes=200,
    close_timeout_ns=1_000_000_000,
    receive_queue_frames=4,
)
# The gateway's end, with its write limit of 0.
SERVER = PipeLimits(
    capacity_bytes=60,
    write_limit_bytes=0,
    max_message_bytes=1_000,
    close_timeout_ns=2_000_000_000,
    receive_queue_frames=2,
)
# Fills the inbox of either end on its own.
FILLER = b"f" * 100


def codes(closed: TransportClosed) -> tuple[int | None, int | None]:
    return closed.peer_code, closed.local_code


async def read_to_end(connection: MemoryConnection) -> tuple[list[str | bytes], TransportClosed]:
    """Read until recv() raises; return the messages read and what it raised."""
    messages: list[str | bytes] = []
    while True:
        try:
            messages.append(await connection.recv())
        except TransportClosed as closed:
            return messages, closed


async def still_running_after(task: asyncio.Task[object], seconds: float) -> bool:
    """Wait up to the given virtual time for the task; tell whether it is still running."""
    _, pending = await asyncio.wait({task}, timeout=seconds)
    return task in pending


async def send_error(connection: MemoryConnection, message: bytes) -> TransportClosed:
    """Send the message and return the TransportClosed that the send raises."""
    with pytest.raises(TransportClosed) as raised:
        await connection.send(message)
    return raised.value


async def upgrade(path: str) -> HttpReply | None:
    return None


async def echo_once(connection: Connection) -> None:
    """Send the first message back, then close."""
    await connection.send(await connection.recv())
    await connection.close(1000, "done")


async def test_messages_reach_the_peer_whole_and_in_order() -> None:
    client, server = memory_pair(CLIENT, SERVER)

    client.send_nowait("hello")
    client.send_nowait(b"\xff\xd8\xff frame")

    assert [await server.recv(), await server.recv()] == ["hello", b"\xff\xd8\xff frame"]


async def test_send_blocks_at_the_write_limit_until_the_peer_reads() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(server.send(b"command"))
        blocked = await still_running_after(sending, 3_600)
        first = await client.recv()

    assert (blocked, first, await client.recv()) == (True, FILLER, b"command")


@pytest.mark.parametrize(
    ("message_bytes", "blocks"),
    [(CLIENT.write_limit_bytes, False), (CLIENT.write_limit_bytes + 1, True)],
)
async def test_send_waits_only_while_unsent_bytes_are_above_a_write_limit_above_zero(
    message_bytes: int, blocks: bool
) -> None:
    client, server = memory_pair(CLIENT, SERVER)
    client.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(client.send(b"t" * message_bytes))
        blocked = await still_running_after(sending, 3_600)
        await server.recv()

    assert blocked is blocks


def test_each_inbox_fills_at_its_own_capacity() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    # Fills the server's inbox and leaves room in the client's.
    between = b"b" * 70

    client.send_nowait(between)
    client.send_nowait(b"frame")
    server.send_nowait(between)
    server.send_nowait(b"command")

    assert (client.unsent_bytes(), server.unsent_bytes()) == (len(b"frame"), 0)


async def test_unsent_bytes_count_what_the_peer_has_not_taken_and_fall_as_it_reads() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)
    server.send_nowait(FILLER)
    server.send_nowait(b"c" * 30)

    unsent_bytes = [server.unsent_bytes()]
    await client.recv()
    unsent_bytes.append(server.unsent_bytes())
    await client.recv()
    unsent_bytes.append(server.unsent_bytes())

    assert unsent_bytes == [130, 30, 0]


def test_send_nowait_leaves_what_the_peer_cannot_take_unsent() -> None:
    _, server = memory_pair(CLIENT, SERVER)

    server.send_nowait(FILLER)
    server.send_nowait(b"command")

    assert server.unsent_bytes() == len(b"command")


def test_a_text_message_counts_its_utf8_bytes() -> None:
    _, server = memory_pair(CLIENT, SERVER)

    server.send_nowait(FILLER)
    # Five characters, six bytes: the e with an acute accent takes two.
    server.send_nowait("h\N{LATIN SMALL LETTER E WITH ACUTE}llo")

    assert server.unsent_bytes() == 6


async def test_a_close_follows_the_messages_sent_before_it_and_both_ends_see_its_code() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    client.send_nowait(FILLER)
    client.send_nowait(b"last frame")

    async with asyncio.TaskGroup() as tasks:
        closing = tasks.create_task(client.close(1000, "done"))
        # Half the client's close timeout: the close is still waiting behind the last frame.
        waited_behind_data = await still_running_after(closing, CLIENT.close_timeout_ns / 2e9)
        server_messages, at_server = await read_to_end(server)
    client_messages, at_client = await read_to_end(client)

    assert waited_behind_data
    assert (server_messages, codes(at_server)) == ([FILLER, b"last frame"], (1000, None))
    assert (client_messages, codes(at_client)) == ([], (None, 1000))


async def test_once_the_peers_close_has_arrived_sends_fail_but_earlier_messages_are_read() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(b"command")
    await server.close(4001, "replaced")

    with pytest.raises(TransportClosed) as refused:
        client.send_nowait(b"frame")
    messages, closed = await read_to_end(client)

    assert (codes(refused.value), messages, codes(closed)) == (
        (None, None),
        [b"command"],
        (4001, None),
    )


async def test_once_the_peers_close_has_arrived_send_raises_its_code_and_close_returns() -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    await server.close(4003, "idle")

    refused = await send_error(client, b"frame")
    start_ns = clock.now_ns()
    await client.close(1000, "bye")

    assert (codes(refused), clock.now_ns()) == ((4003, None), start_ns)


async def test_once_its_close_has_arrived_an_end_refuses_sends_and_reports_its_own_code() -> None:
    _, server = memory_pair(CLIENT, SERVER)
    await server.close(1001, "shutdown")

    with pytest.raises(TransportClosed) as refused_now:
        server.send_nowait(b"command")
    refused = await send_error(server, b"command")
    messages, closed = await read_to_end(server)

    assert [codes(refused_now.value), codes(refused), codes(closed)] == [
        (None, None),
        (None, 1001),
        (None, 1001),
    ]
    assert messages == []


async def test_a_close_that_cannot_enter_a_full_inbox_aborts_at_its_timeout_keeping_its_code() -> (
    None
):
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)
    start_ns = clock.now_ns()

    await server.close(1001, "shutdown")
    waited_ns = clock.now_ns() - start_ns
    client_messages, at_client = await read_to_end(client)
    _, at_server = await read_to_end(server)

    assert waited_ns == SERVER.close_timeout_ns
    # The library counts its close as sent once it has written it, so the closing end reports
    # its own code although the close never arrived.
    assert (client_messages, codes(at_client), codes(at_server)) == (
        [FILLER],
        (None, None),
        (None, 1001),
    )


@pytest.mark.parametrize(
    ("unread_messages", "waited_ns"),
    [
        (CLIENT.receive_queue_frames, 0),
        (CLIENT.receive_queue_frames + 1, SERVER.close_timeout_ns),
    ],
)
async def test_a_close_is_read_only_while_the_peer_has_at_most_its_receive_queue_unread(
    unread_messages: int, waited_ns: int
) -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    commands = [b"c"] * unread_messages
    for command in commands:
        server.send_nowait(command)
    start_ns = clock.now_ns()

    await server.close(4001, "replaced")
    closed_after_ns = clock.now_ns() - start_ns
    client_messages, at_client = await read_to_end(client)
    _, at_server = await read_to_end(server)

    # Past its receive queue the client's library stops reading, so the close waits unread
    # behind the commands; the client reads it after them, also once the pair is aborted.
    assert closed_after_ns == waited_ns
    assert (client_messages, codes(at_client), codes(at_server)) == (
        commands,
        (4001, None),
        (None, 4001),
    )


@pytest.mark.parametrize(
    ("unread_messages", "waited_ns"),
    [
        (SERVER.receive_queue_frames, 0),
        (SERVER.receive_queue_frames + 1, SERVER.close_timeout_ns),
    ],
)
async def test_a_close_ends_only_while_the_closing_end_has_at_most_its_receive_queue_unread(
    unread_messages: int, waited_ns: int
) -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    frames = [b"f"] * unread_messages
    for frame in frames:
        client.send_nowait(frame)
    start_ns = clock.now_ns()

    await server.close(4001, "replaced")
    closed_after_ns = clock.now_ns() - start_ns
    client_messages, at_client = await read_to_end(client)
    server_messages, at_server = await read_to_end(server)

    # The client reads the close and answers it, and the answer waits unread behind the frames
    # once the server's library has stopped reading.
    assert closed_after_ns == waited_ns
    assert (client_messages, codes(at_client)) == ([], (4001, None))
    assert (server_messages, codes(at_server)) == (frames, (None, 4001))


async def test_an_end_that_stopped_reading_reads_again_at_a_quarter_of_its_receive_queue() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    # One above the client's receive queue: its library stops reading.
    commands = [b"c"] * (CLIENT.receive_queue_frames + 1)
    # They arrive while the library does not read, so they count only once it reads again.
    late = [b"d", b"d"]
    for command in commands + late:
        server.send_nowait(command)
    taken_before_resume = len(commands) - CLIENT.receive_queue_frames // 4 - 1
    taken: list[str | bytes] = []

    async with asyncio.TaskGroup() as tasks:
        closing = tasks.create_task(server.close(4001, "replaced"))
        for _ in range(taken_before_resume):
            taken.append(await client.recv())
        # A quarter of the server's close timeout each.
        waiting_before_resume = await still_running_after(closing, SERVER.close_timeout_ns / 4e9)
        taken.append(await client.recv())
        waiting_after_resume = await still_running_after(closing, SERVER.close_timeout_ns / 4e9)
    rest, at_client = await read_to_end(client)

    assert (waiting_before_resume, waiting_after_resume) == (True, False)
    assert (taken + rest, codes(at_client)) == (commands + late, (4001, None))


async def test_a_close_returns_once_its_answer_has_come_behind_the_peers_unsent_messages() -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)
    server.send_nowait(b"command")
    start_ns = clock.now_ns()

    async with asyncio.TaskGroup() as tasks:
        closing = tasks.create_task(client.close(1000, "done"))
        # Half the client's close timeout: the server has read the close, and its answer waits
        # behind the command that the client's full inbox cannot take.
        waiting = await still_running_after(closing, CLIENT.close_timeout_ns / 2e9)
        messages, at_client = await read_to_end(client)
    closed_after_ns = clock.now_ns() - start_ns
    _, at_server = await read_to_end(server)

    assert (waiting, closed_after_ns) == (True, CLIENT.close_timeout_ns // 2)
    assert (messages, codes(at_client), codes(at_server)) == (
        [FILLER, b"command"],
        (None, 1000),
        (1000, None),
    )


async def test_a_close_whose_caller_is_cancelled_aborts_the_pair() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(SERVER.close_timeout_ns / 2e9):
            await server.close(1001, "shutdown")
    messages, closed = await read_to_end(client)

    assert (messages, codes(closed)) == ([FILLER], (None, None))


async def test_a_second_close_returns_at_once_and_changes_nothing() -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    await server.close(1001, "shutdown")
    start_ns = clock.now_ns()

    await server.close(1000, "again")
    _, closed = await read_to_end(client)

    assert (clock.now_ns(), codes(closed)) == (start_ns, (1001, None))


async def test_a_second_close_leaves_the_code_of_a_close_still_waiting_to_be_delivered() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(server.close(1001, "shutdown"))
        tasks.create_task(server.close(1000, "again"))
        # Half the server's close timeout: both closes are waiting behind the full inbox.
        await asyncio.sleep(SERVER.close_timeout_ns / 2e9)
        messages, closed = await read_to_end(client)

    assert (messages, codes(closed)) == ([FILLER], (1001, None))


async def test_a_close_after_an_abort_returns_at_once() -> None:
    clock = LoopClock()
    client, server = memory_pair(CLIENT, SERVER)
    client.abort()
    start_ns = clock.now_ns()

    await server.close(1000, "bye")

    assert clock.now_ns() == start_ns


async def test_abort_discards_both_outboxes_and_leaves_both_inboxes_to_read() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    client.send_nowait(FILLER)
    client.send_nowait(b"frame")
    server.send_nowait(FILLER)
    server.send_nowait(b"command")

    client.abort()
    unsent_bytes = (client.unsent_bytes(), server.unsent_bytes())
    client_messages, at_client = await read_to_end(client)
    server_messages, at_server = await read_to_end(server)

    assert unsent_bytes == (0, 0)
    assert (client_messages, codes(at_client)) == ([FILLER], (None, None))
    assert (server_messages, codes(at_server)) == ([FILLER], (None, None))


async def test_after_an_abort_every_send_raises_without_codes() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.abort()

    with pytest.raises(TransportClosed) as refused_now:
        client.send_nowait(b"frame")
    refused = await send_error(client, b"frame")

    assert (codes(refused_now.value), codes(refused)) == ((None, None), (None, None))


async def test_after_an_abort_a_close_already_in_the_inbox_still_gives_its_code() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    client.send_nowait(FILLER)
    client.send_nowait(b"frame")
    server.send_nowait(b"command")

    async with asyncio.TaskGroup() as tasks:
        closing = tasks.create_task(server.close(4001, "replaced"))
        # Half the server's close timeout: the client has read the close, and its answer waits
        # behind the frame that the server's full inbox cannot take.
        waiting = await still_running_after(closing, SERVER.close_timeout_ns / 2e9)
        client.abort()
    client_messages, at_client = await read_to_end(client)
    server_messages, at_server = await read_to_end(server)

    assert waiting
    assert (client_messages, codes(at_client)) == ([b"command"], (4001, None))
    assert (server_messages, codes(at_server)) == ([FILLER], (None, 4001))


async def test_a_send_blocked_at_the_write_limit_raises_when_the_peer_aborts() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(send_error(server, b"command"))
        blocked = await still_running_after(sending, 3_600)
        client.abort()

    assert (blocked, codes(sending.result())) == (True, (None, None))


async def test_a_send_blocked_at_the_write_limit_returns_when_its_own_end_aborts() -> None:
    _, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(server.send(b"command"))
        blocked = await still_running_after(sending, 3_600)
        server.abort()
    refused = await send_error(server, b"next")

    assert (blocked, sending.exception(), codes(refused)) == (True, None, (None, None))


async def test_a_send_blocked_behind_a_message_over_the_receivers_limit_raises_1009() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)
    server.send_nowait(b"x" * (CLIENT.max_message_bytes + 1))

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(send_error(server, b"command"))
        blocked = await still_running_after(sending, 3_600)
        await client.recv()

    assert (blocked, codes(sending.result())) == (True, (1009, None))


async def test_a_send_blocked_at_the_write_limit_raises_when_the_peers_close_times_out() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)

    async with asyncio.TaskGroup() as tasks:
        sending = tasks.create_task(send_error(server, b"command"))
        blocked = await still_running_after(sending, 3_600)
        tasks.create_task(client.close(1000, "done"))
        # Half the client's close timeout: the server has read the close, which leaves its
        # drain waiting, and its answer waits behind the command the client does not take.
        blocked_through_the_close = await still_running_after(
            sending, CLIENT.close_timeout_ns / 2e9
        )

    assert (blocked, blocked_through_the_close, codes(sending.result())) == (
        True,
        True,
        (1000, None),
    )


async def test_a_message_over_the_limit_behind_a_close_the_receiver_sent_changes_no_code() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(FILLER)
    server.send_nowait(b"x" * (CLIENT.max_message_bytes + 1))

    async with asyncio.TaskGroup() as tasks:
        closing = tasks.create_task(client.close(1000, "done"))
        # Half the client's close timeout: the server has read the close, and its answer waits
        # behind the message that the client's full inbox cannot take.
        waiting = await still_running_after(closing, CLIENT.close_timeout_ns / 2e9)
        client_messages, at_client = await read_to_end(client)
    _, at_server = await read_to_end(server)

    # The client's library has sent its close, so it drops the connection without another.
    assert (waiting, client_messages) == (True, [FILLER])
    assert (codes(at_client), codes(at_server)) == ((None, 1000), (1000, None))


async def test_a_message_of_exactly_the_receivers_limit_arrives() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    message = b"x" * SERVER.max_message_bytes

    client.send_nowait(message)

    assert await server.recv() == message


async def test_a_message_over_the_receivers_limit_ends_the_pair_with_1009() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    server.send_nowait(b"command")
    client.send_nowait(b"frame")

    client.send_nowait(b"x" * (SERVER.max_message_bytes + 1))
    server_messages, at_receiver = await read_to_end(server)
    client_messages, at_sender = await read_to_end(client)

    assert (server_messages, codes(at_receiver)) == ([b"frame"], (None, 1009))
    assert (client_messages, codes(at_sender)) == ([b"command"], (1009, None))


async def test_a_message_under_the_senders_own_limit_and_over_the_receivers_ends_the_pair() -> None:
    client, server = memory_pair(CLIENT, SERVER)

    server.send_nowait(b"x" * (CLIENT.max_message_bytes + 1))
    _, at_receiver = await read_to_end(client)
    _, at_sender = await read_to_end(server)

    assert (codes(at_receiver), codes(at_sender)) == ((None, 1009), (1009, None))


async def test_nothing_sent_behind_a_message_over_the_limit_reaches_the_receiver() -> None:
    client, server = memory_pair(CLIENT, SERVER)
    # Two messages fit the client's inbox; the next two wait in the server's outbox.
    server.send_nowait(b"c" * 60)
    server.send_nowait(b"c" * 60)
    server.send_nowait(b"x" * (CLIENT.max_message_bytes + 1))
    server.send_nowait(b"after")

    messages, _ = await read_to_end(client)

    assert (messages, server.unsent_bytes()) == ([b"c" * 60, b"c" * 60], 0)


async def test_a_recv_cancelled_as_its_message_arrives_leaves_the_message_queued() -> None:
    client, server = memory_pair(CLIENT, SERVER)

    async with asyncio.TaskGroup() as tasks:
        receiving = tasks.create_task(client.recv())
        await still_running_after(receiving, 1)
        server.send_nowait(b"command")
        receiving.cancel()

    assert (receiving.cancelled(), await client.recv()) == (True, b"command")


async def test_a_second_recv_while_one_waits_raises_and_the_first_still_gets_its_message() -> None:
    client, server = memory_pair(CLIENT, SERVER)

    async with asyncio.TaskGroup() as tasks:
        first = tasks.create_task(client.recv())
        await still_running_after(first, 1)
        with pytest.raises(RuntimeError, match="already waiting"):
            await client.recv()
        server.send_nowait(b"command")

    assert first.result() == b"command"


def test_rtt_is_none_until_a_round_trip_is_set() -> None:
    client, _ = memory_pair(CLIENT, SERVER)

    before = client.rtt_ns()
    client.set_rtt_ns(250_000)

    assert (before, client.rtt_ns()) == (None, 250_000)


def test_a_transport_closed_with_both_codes_is_refused() -> None:
    with pytest.raises(ValueError, match="peer_code=1000 and local_code=1001"):
        TransportClosed(1000, 1001)


def test_the_memory_listener_is_a_listener_on_port_zero() -> None:
    listener: Listener = MemoryListener(echo_once, upgrade, CLIENT, SERVER)

    assert listener.port == 0


async def test_a_reply_from_the_http_handler_refuses_the_upgrade_with_its_status() -> None:
    paths: list[str] = []

    async def warming_up(path: str) -> HttpReply | None:
        paths.append(path)
        return HttpReply(503, b'{"status": "warming"}')

    async with MemoryListener(echo_once, warming_up, CLIENT, SERVER) as listener:
        with pytest.raises(TransportConnectError) as refused:
            await listener.connect("/ws")

    assert (paths, refused.value.http_status) == (["/ws"], 503)


async def test_the_listener_runs_the_handler_on_the_server_end_of_each_connection() -> None:
    async with MemoryListener(echo_once, upgrade, CLIENT, SERVER) as listener:
        first = await listener.connect("/ws")
        second = await listener.connect("/ws")
        first.send_nowait(b"one")
        second.send_nowait(b"two")
        first_messages, first_closed = await read_to_end(first)
        second_messages, second_closed = await read_to_end(second)

    assert (first_messages, codes(first_closed)) == ([b"one"], (1000, None))
    assert (second_messages, codes(second_closed)) == ([b"two"], (1000, None))


async def test_the_listener_gives_the_handler_the_servers_end_and_the_caller_the_clients() -> None:
    received: list[str | bytes] = []
    # Over the client's limit and under the server's: only the server's end takes it.
    frame = b"f" * (CLIENT.max_message_bytes + 1)

    async def receive_once(connection: Connection) -> None:
        received.append(await connection.recv())
        await connection.close(1000, "done")

    async with MemoryListener(receive_once, upgrade, CLIENT, SERVER) as listener:
        client = await listener.connect("/ws")
        client.send_nowait(frame)
        _, closed = await read_to_end(client)

    assert (received, codes(closed)) == ([frame], (1000, None))


async def test_after_stop_accepting_a_connect_is_refused_with_503() -> None:
    async with MemoryListener(echo_once, upgrade, CLIENT, SERVER) as listener:
        stopped_before = listener.stopped
        listener.stop_accepting()
        listener.stop_accepting()
        with pytest.raises(TransportConnectError) as refused:
            await listener.connect("/ws")

    assert (stopped_before, listener.stopped, refused.value.http_status) == (False, True, 503)


async def test_an_upgrade_in_progress_when_the_listener_stops_is_refused_with_503() -> None:
    answered = asyncio.Event()

    async def slow_upgrade(path: str) -> HttpReply | None:
        # Completed by the timer below, which fires after the listener has stopped.
        await answered.wait()
        return None

    async with MemoryListener(echo_once, slow_upgrade, CLIENT, SERVER) as listener:
        loop = asyncio.get_running_loop()
        loop.call_later(1.0, listener.stop_accepting)
        loop.call_later(2.0, answered.set)
        with pytest.raises(TransportConnectError) as refused:
            await listener.connect("/ws")

    assert refused.value.http_status == 503


async def test_wait_closed_returns_only_once_stop_accepting_has_been_called() -> None:
    async with (
        MemoryListener(echo_once, upgrade, CLIENT, SERVER) as listener,
        asyncio.TaskGroup() as tasks,
    ):
        waiting = tasks.create_task(listener.wait_closed())
        waited_before_stop = await still_running_after(waiting, 3_600)
        listener.stop_accepting()

    assert (waited_before_stop, waiting.done()) == (True, True)


async def test_wait_closed_waits_for_every_handler_still_running() -> None:
    async with MemoryListener(echo_once, upgrade, CLIENT, SERVER) as listener:
        client = await listener.connect("/ws")
        listener.stop_accepting()
        async with asyncio.TaskGroup() as tasks:
            waiting = tasks.create_task(listener.wait_closed())
            waited_for_handler = await still_running_after(waiting, 3_600)
            client.send_nowait(b"ping")
        messages, closed = await read_to_end(client)

    assert (waited_for_handler, messages, codes(closed)) == (True, [b"ping"], (1000, None))
