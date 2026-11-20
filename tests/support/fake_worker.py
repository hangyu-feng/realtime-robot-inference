"""A fake worker link and its launcher, for the async unit tests on the virtual-time loop.

A FakeWorkerLink behaves as the link of a worker that runs the fake runtime: it answers after
the service time of each batch, with the statuses and command bodies that FakeRunner gives, and
it can be made to crash, hang, break the protocol, answer for another batch, or refuse to exit.
"""

from __future__ import annotations

import asyncio
import signal
from collections.abc import Callable, Mapping, Sequence

from cloud.config import ServerConfig
from cloud.worker_link import WorkerFailed, WorkerStartError
from common.clock import Clock
from common.commands import encode_command
from common.worker_protocol import BatchResult, FrameResult, FrameStatus, WorkerReady
from workers.models.fake import FakeRunner

# What a link whose exit is held waits for the exit, before its kill and once more after it:
# the default worker_stop_timeout_ms.
STOP_TIMEOUT_NS = round(ServerConfig().worker_stop_timeout_ms * 1_000_000)

_FIRST_PID = 1000


class FakeWorkerLink:
    """A WorkerLink whose worker is simulated on the event loop's clock.

    requests lists every batch written, as (batch_id, the JPEGs as bytes). cpu_ns is the CPU
    time every result reports: 0 unless a test sets it, as the fake runtime uses almost none.
    batch_cap is the cap the link was launched with, None for a first start. killed records a
    kill() and stopped a stop().
    """

    def __init__(
        self,
        model: str,
        pid: int,
        batch_cap: int | None,
        clock: Clock,
        frame_ns: int,
        batch_sizes: Sequence[int],
        ready_delay_ns: int,
        on_lost: Callable[[], None],
    ) -> None:
        self.model = model
        self.batch_cap = batch_cap
        self.requests: list[tuple[int, list[bytes]]] = []
        self.cpu_ns = 0
        self.killed = False
        self.stopped = False
        self._pid = pid
        self._clock = clock
        self._frame_ns = frame_ns
        self._on_lost = on_lost
        # The fake runtime of workers/worker.py, without its wait: the loop's timer stands for
        # the service time.
        self._runner = FakeRunner(model, frame_ns, lambda wait_s: None)
        self._lost = False
        self._broken = False
        self._hung = False
        self._garbled = False
        self._exit_held = False
        self._exit_status: int | None = None
        # The answer on its way and the one that has arrived, as on a real link: at most one
        # of each, because the supervisor keeps one batch in flight.
        self._timer: asyncio.TimerHandle | None = None
        self._arrived: WorkerReady | BatchResult | None = None
        self._waiter: asyncio.Future[None] | None = None
        ready = WorkerReady(
            model=model,
            pid=pid,
            model_file=None,
            model_sha256=None,
            threads=1,
            t_b_ns={size: size * frame_ns for size in batch_sizes},
            t_b_cpu_ns=dict.fromkeys(batch_sizes, 0),
        )
        self._timer = asyncio.get_running_loop().call_later(
            ready_delay_ns / 1e9, self._arrive, ready
        )

    @property
    def pid(self) -> int:
        return self._pid

    @property
    def lost(self) -> bool:
        return self._lost

    @property
    def failure(self) -> WorkerFailed | None:
        if not self._lost:
            return None
        if self._broken:
            return WorkerFailed("protocol", "the fake worker broke the protocol")
        return WorkerFailed("died", "the fake worker's link is lost")

    async def ready(self) -> WorkerReady:
        message = await self._take()
        if not isinstance(message, WorkerReady):
            raise WorkerFailed("protocol", "the fake worker's first message is not a WorkerReady")
        return message

    def write_batch(self, batch_id: int, jpegs: Sequence[bytes | memoryview]) -> None:
        frames = [bytes(jpeg) for jpeg in jpegs]
        self.requests.append((batch_id, frames))
        if self._lost or self._hung:
            return
        received_ns = self._clock.now_ns()
        self._timer = asyncio.get_running_loop().call_later(
            len(frames) * self._frame_ns / 1e9, self._finish, batch_id, frames, received_ns
        )

    async def result(self) -> tuple[BatchResult, int]:
        message = await self._take()
        if not isinstance(message, BatchResult):
            raise WorkerFailed("protocol", "the fake worker sent a WorkerReady instead of a result")
        return message, message.sent_ns

    def kill(self) -> None:
        self.killed = True
        if not self._exit_held:
            self._lose(-signal.SIGKILL)

    async def stop(self) -> int | None:
        self.stopped = True
        if not self._exit_held:
            # The worker exits with status 0 when its link closes, unless it had ended before.
            self._lose(0)
            return self._exit_status
        self._lose(None)
        try:
            await asyncio.sleep(STOP_TIMEOUT_NS / 1e9)
            self.kill()
            await asyncio.sleep(STOP_TIMEOUT_NS / 1e9)
            return None
        finally:
            # As the real link does: a cancellation in either wait kills the process too.
            self.kill()

    def crash(self) -> None:
        """End the worker process: the link is lost, and a waiting result raises "died"."""
        self._lose(1)

    def break_protocol(self) -> None:
        """Send bytes that break the protocol and live on, as a worker stuck in a loop would.

        The link is lost, as a real link closes its end on a protocol break, and every later
        ready() or result() raises "protocol". The process exits only when it is killed or
        sees its link close.
        """
        self._broken = True
        self._arrived = None
        self._lose(None)

    def hang(self) -> None:
        """Send no further answer, as a worker stuck in native code."""
        self._hung = True
        self._cancel_timer()

    def garble(self) -> None:
        """Make the next result carry another batch's id."""
        self._garbled = True

    def hold_exit(self) -> None:
        """Make stop() wait like a process that never exits, not even when killed.

        stop() then takes both stop bounds, kills between them and when it is cancelled, and
        returns None.
        """
        self._exit_held = True

    def _lose(self, exit_status: int | None) -> None:
        """Lose the link once, and record the exit status, None while the process lives.

        The first exit sets the status.
        """
        if self._exit_status is None:
            self._exit_status = exit_status
        if self._lost:
            return
        self._lost = True
        self._cancel_timer()
        self._wake()
        self._on_lost()

    def _cancel_timer(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _finish(self, batch_id: int, frames: list[bytes], received_ns: int) -> None:
        outcome = self._runner.run_batch(frames)
        result = BatchResult(
            batch_id=batch_id + 1 if self._garbled else batch_id,
            received_ns=received_ns,
            sent_ns=self._clock.now_ns(),
            inference_ns=outcome.inference_ns,
            cpu_ns=self.cpu_ns,
            frames=tuple(
                FrameResult(FrameStatus.OK, encode_command(frame.command), 0, 0, 0)
                if frame.command is not None
                else FrameResult(FrameStatus.DECODE_ERROR, b"", 0, 0, 0)
                for frame in outcome.frames
            ),
        )
        self._garbled = False
        self._arrive(result)

    def _arrive(self, message: WorkerReady | BatchResult) -> None:
        self._timer = None
        self._arrived = message
        self._wake()

    def _wake(self) -> None:
        if self._waiter is not None and not self._waiter.done():
            self._waiter.set_result(None)

    async def _take(self) -> WorkerReady | BatchResult:
        while self._arrived is None:
            if (failure := self.failure) is not None:
                raise failure
            self._waiter = asyncio.get_running_loop().create_future()
            try:
                # Completed by the answer's timer or by the loss of the link; the caller
                # bounds the wait, as for a real link.
                await self._waiter
            finally:
                self._waiter = None
        message, self._arrived = self._arrived, None
        return message


class FakeLauncher:
    """A WorkerLauncher that builds FakeWorkerLinks.

    links lists every link launched, in order. fail_starts[model] = n makes the next n starts
    of that model raise WorkerStartError.
    """

    def __init__(
        self,
        clock: Clock,
        frame_ns: Mapping[str, int],
        batch_sizes: Mapping[str, Sequence[int]],
        ready_delay_ns: int = 0,
    ) -> None:
        self.links: list[FakeWorkerLink] = []
        self.fail_starts: dict[str, int] = {}
        self._clock = clock
        self._frame_ns = frame_ns
        self._batch_sizes = batch_sizes
        self._ready_delay_ns = ready_delay_ns

    async def __call__(
        self, model: str, batch_cap: int | None, on_lost: Callable[[], None]
    ) -> FakeWorkerLink:
        if self.fail_starts.get(model, 0) > 0:
            self.fail_starts[model] -= 1
            raise WorkerStartError("the fake launcher was told to fail this start")
        link = FakeWorkerLink(
            model,
            _FIRST_PID + len(self.links),
            batch_cap,
            self._clock,
            self._frame_ns[model],
            # A restarted worker calibrates only the sizes up to the cap it is given.
            [size for size in self._batch_sizes[model] if batch_cap is None or size <= batch_cap],
            self._ready_delay_ns,
            on_lost,
        )
        self.links.append(link)
        return link
