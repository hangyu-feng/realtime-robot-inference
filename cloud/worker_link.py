"""One model's worker process and the gateway's end of its link.

start_worker spawns the process with one end of a Unix socket pair and returns a WorkerLink
over the other end. The link splits the worker's byte stream into the messages of
common/worker_protocol.py and stamps each one when its last byte arrives, writes batch
requests, and stops, kills and reaps the process. cloud/supervisor.py drives it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import socket
import sys
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol

from cloud.config import ServerConfig
from common.clock import Clock
from common.worker_protocol import (
    LENGTH_PREFIX,
    BatchResult,
    WorkerProtocolError,
    WorkerReady,
    decode_worker_message,
    encode_batch_request,
    payload_length,
)

logger = logging.getLogger(__name__)

_MS_PER_S = 1_000


class WorkerFailed(Exception):
    """A worker can no longer serve; kind says how it failed.

    "died": its link reached EOF or failed. "timeout": no result within the inference timeout,
    or no WorkerReady within the warm-up timeout. "protocol": a message that does not decode, a
    message of the wrong kind, or a result that does not answer its batch.
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class WorkerStartError(Exception):
    """A worker could not be spawned, or its first start failed. Its text is a one-line reason."""


class PayloadSplitter:
    """Turns the chunks of a worker's byte stream into whole payloads. Pure: it does no I/O.

    Every message is a length prefix and then that many payload bytes, and any chunking of one
    stream gives the same payloads. After a WorkerProtocolError the stream has no known message
    boundary, so the caller feeds nothing more.
    """

    def __init__(self) -> None:
        self._prefix = bytearray()
        # The payload being filled, None between payloads, and the length its prefix gave.
        self._payload: bytearray | None = None
        self._payload_bytes = 0

    def feed(self, data: bytes) -> list[memoryview]:
        """Take the next chunk of the stream and return the payloads it completes, in order.

        WorkerProtocolError for a length that no payload may have, raised before anything is
        allocated for that payload.
        """
        payloads: list[memoryview] = []
        chunk = memoryview(data)
        while chunk:
            if self._payload is None:
                missing_bytes = LENGTH_PREFIX.size - len(self._prefix)
                self._prefix += chunk[:missing_bytes]
                chunk = chunk[missing_bytes:]
                if len(self._prefix) == LENGTH_PREFIX.size:
                    prefix = bytes(self._prefix)
                    self._prefix.clear()
                    self._payload_bytes = payload_length(prefix)
                    # Each payload is a new buffer that grows with the bytes that arrive, so a
                    # prefix alone commits no memory, and payload_length has bounded what the
                    # buffer can grow to at MAX_PAYLOAD_BYTES.
                    self._payload = bytearray()
            else:
                missing_bytes = self._payload_bytes - len(self._payload)
                self._payload += chunk[:missing_bytes]
                chunk = chunk[missing_bytes:]
                if len(self._payload) == self._payload_bytes:
                    payloads.append(memoryview(self._payload))
                    self._payload = None
        return payloads


class WorkerLink(Protocol):
    """One worker process and the gateway's end of its link."""

    @property
    def pid(self) -> int: ...

    @property
    def lost(self) -> bool:
        """True once the socket has reached EOF or an error, or this side has closed it."""

    @property
    def failure(self) -> WorkerFailed | None:
        """The failure that the link has recorded, which ready() and result() raise; or None.

        A protocol break records "protocol" when it closes the link, and the loss of the
        socket records "died". The link can be lost a moment before the loss is recorded.
        """

    async def ready(self) -> WorkerReady:
        """Return the worker's first message; WorkerFailed when it is anything else."""

    def write_batch(self, batch_id: int, jpegs: Sequence[bytes | memoryview]) -> None:
        """Write a batch request. It never waits."""

    async def result(self) -> tuple[BatchResult, int]:
        """Return the next result and the gateway's clock reading when its last byte arrived.

        WorkerFailed when the link is lost or the worker sends anything else.
        """

    def kill(self) -> None:
        """Send SIGKILL, unless the process has been reaped."""

    async def stop(self) -> int | None:
        """Close the link, wait for the process to exit, kill it if it does not, and reap it.

        Returns the exit status, or None when the process could not be reaped. It may be
        called again, and it raises no exception of its own: cancelled while it waits, it kills
        the process and lets the CancelledError propagate.
        """


# The model, the batch cap fixed at the first start (None for the first start itself), and
# on_lost, which the link calls once when its socket is lost.
WorkerLauncher = Callable[[str, int | None, Callable[[], None]], Awaitable[WorkerLink]]


class _Receiver(asyncio.Protocol):
    """The read side of a link: splits the stream, and holds one stamped payload for take()."""

    # Set by connection_made, which the loop calls before any other callback.
    _transport: asyncio.BaseTransport

    def __init__(self, clock: Clock, on_lost: Callable[[], None]) -> None:
        self._clock = clock
        self._on_lost = on_lost
        self._splitter = PayloadSplitter()
        # A one-slot buffer: a payload with the clock reading of its last byte, until take()
        # returns it. The worker sends one message per request, so a second payload while one
        # waits here breaks the protocol, and nothing queues behind the slot.
        self._arrived: tuple[memoryview, int] | None = None
        # The kind and the message of the WorkerFailed that every take() raises from now on.
        self._failure: tuple[str, str] | None = None
        self._waiter: asyncio.Future[None] | None = None

    @property
    def failure(self) -> WorkerFailed | None:
        return None if self._failure is None else WorkerFailed(*self._failure)

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        self._transport = transport

    def data_received(self, data: bytes) -> None:
        # Read before anything else: for a result this is result_ns, taken when its last byte
        # arrived and before the dispatcher's task resumes, so that task's wake-up is not in
        # the dispatch round trip (DESIGN.md 4.5).
        received_ns = self._clock.now_ns()
        try:
            payloads = self._splitter.feed(data)
        except WorkerProtocolError as error:
            self._break_protocol(str(error))
            return
        if not payloads:
            return
        if len(payloads) > 1 or self._arrived is not None:
            self._break_protocol("the worker sent a message before its previous one was taken")
            return
        self._arrived = (payloads[0], received_ns)
        self._wake()

    def connection_lost(self, exc: Exception | None) -> None:
        if self._failure is None:
            self._failure = (
                "died",
                "the worker's link closed" if exc is None else f"the worker's link failed: {exc}",
            )
        self._wake()
        self._on_lost()

    async def take(self) -> tuple[memoryview, int]:
        """Return the next payload and the clock reading taken when its last byte arrived.

        A payload that arrived before the link was lost is still returned. WorkerFailed once
        the link has failed and no payload waits.
        """
        while self._arrived is None:
            if (failure := self.failure) is not None:
                raise failure
            self._waiter = asyncio.get_running_loop().create_future()
            try:
                # Completed by the worker's next message or by the loss of the link. The
                # caller bounds the wait, with the warm-up or the inference timeout.
                await self._waiter
            finally:
                self._waiter = None
        arrived, self._arrived = self._arrived, None
        return arrived

    def _break_protocol(self, message: str) -> None:
        # The stream has no known message boundary any more, so nothing read from it counts.
        self._arrived = None
        self._failure = ("protocol", message)
        self._wake()
        self._transport.close()

    def _wake(self) -> None:
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)


def _kill(process: asyncio.subprocess.Process) -> None:
    # ProcessLookupError: the process has exited and been reaped, so there is nothing to kill.
    with contextlib.suppress(ProcessLookupError):
        process.kill()


class _ProcessLink:
    """The WorkerLink of a spawned worker process."""

    def __init__(
        self,
        model: str,
        process: asyncio.subprocess.Process,
        transport: asyncio.Transport,
        receiver: _Receiver,
        stop_timeout_s: float,
    ) -> None:
        self._model = model
        self._process = process
        self._transport = transport
        self._receiver = receiver
        self._stop_timeout_s = stop_timeout_s

    @property
    def pid(self) -> int:
        return self._process.pid

    @property
    def lost(self) -> bool:
        # The transport is closing from the moment the loop reads EOF or meets an error, which
        # is before it calls connection_lost: a request written in between would be dropped.
        return self._transport.is_closing()

    @property
    def failure(self) -> WorkerFailed | None:
        return self._receiver.failure

    async def ready(self) -> WorkerReady:
        message, _ = await self._message()
        if not isinstance(message, WorkerReady):
            raise WorkerFailed("protocol", "the worker's first message is not a WorkerReady")
        return message

    def write_batch(self, batch_id: int, jpegs: Sequence[bytes | memoryview]) -> None:
        # One writev with the frames' own buffers. No drain is needed: at most one batch is in
        # flight, and the next request is written only after the result of the one before, so
        # the transport never holds more than one request.
        self._transport.writelines(encode_batch_request(batch_id, jpegs))

    async def result(self) -> tuple[BatchResult, int]:
        message, received_ns = await self._message()
        if not isinstance(message, BatchResult):
            raise WorkerFailed("protocol", "the worker sent a WorkerReady where a result was due")
        return message, received_ns

    def kill(self) -> None:
        _kill(self._process)

    async def stop(self) -> int | None:
        self._transport.close()
        try:
            try:
                async with asyncio.timeout(self._stop_timeout_s):
                    return await self._process.wait()
            except TimeoutError:
                # A worker in the middle of a batch, or stuck in native code, does not see the
                # link close.
                _kill(self._process)
            try:
                async with asyncio.timeout(self._stop_timeout_s):
                    return await self._process.wait()
            except TimeoutError:
                logger.error(
                    "worker %s (pid %d) could not be reaped after SIGKILL",
                    self._model,
                    self._process.pid,
                )
                return None
        finally:
            # A cancellation can land in either wait: the shutdown's last stage may cancel a
            # supervisor that is already stopping its worker. Swallowing it would break the
            # shutdown stages, and letting it pass before the SIGKILL would leave a hung worker
            # running after the gateway, outside a container. So a process that has not been
            # reaped is killed here, the cancellation propagates, and the event loop's child
            # watcher reaps the process.
            if self._process.returncode is None:
                _kill(self._process)

    async def _message(self) -> tuple[WorkerReady | BatchResult, int]:
        payload, received_ns = await self._receiver.take()
        try:
            return decode_worker_message(payload, self._model), received_ns
        except WorkerProtocolError as error:
            raise WorkerFailed("protocol", str(error)) from error


async def start_worker(
    model: str,
    batch_cap: int | None,
    on_lost: Callable[[], None],
    *,
    config: ServerConfig,
    clock: Clock,
) -> WorkerLink:
    """Spawn the model's worker process and return the link to it.

    batch_cap is None for the first start, and the cap fixed at the first start for a restart.
    The worker inherits this process's environment, and so loads the same configuration.
    WorkerStartError when the process cannot be spawned.
    """
    # ASSUMPTION: A17. Workers are separate processes by choice; the brief does not require it.
    # A process per model isolates the models from each other and the compute from the
    # gateway's I/O, and a link per process makes one worker's death one socket's EOF.
    cap_arguments = () if batch_cap is None else ("--batch-cap", str(batch_cap))
    parent_end, child_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with contextlib.ExitStack() as undo:
        undo.callback(parent_end.close)
        # This process's copy of the worker's end is closed as soon as the worker has its own:
        # while a copy stays open here, the gateway never sees EOF when the worker dies.
        with child_end:
            try:
                # stdout and stderr are inherited, not piped, so no buffer hides between the
                # worker and the container's log. A session, and so a process group, of its own
                # keeps a Ctrl-C in a terminal, which signals the foreground group, from
                # reaching the worker before it ignores SIGINT: the gateway's staged shutdown
                # ends its workers. uvloop takes start_new_session, not process_group.
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    *("-m", "workers.worker", "--model", model),
                    *("--fd", str(child_end.fileno()), *cap_arguments),
                    stdin=asyncio.subprocess.DEVNULL,
                    pass_fds=(child_end.fileno(),),
                    start_new_session=True,
                )
            except OSError as error:
                raise WorkerStartError(f"its process could not be spawned: {error}") from error
        undo.callback(_kill, process)
        receiver = _Receiver(clock, on_lost)
        # create_unix_connection is the call that takes a protocol factory, and the link is a
        # Protocol so that a result is stamped in data_received.
        transport, _ = await asyncio.get_running_loop().create_unix_connection(
            lambda: receiver, sock=parent_end
        )
        undo.pop_all()
    return _ProcessLink(
        model, process, transport, receiver, config.worker_stop_timeout_ms / _MS_PER_S
    )
