"""One model's worker lifecycle and its dispatch loop.

A ModelSupervisor starts the model's worker, waits for its warm-up, serves batches with it, and
starts it again after a failure, with a capped backoff, for as long as the gateway runs. The
scheduling decisions and the accounting of results are the pure logic of cloud/dispatch.py;
this module awaits, bounds every wait on the worker, and writes to the link.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from collections.abc import Mapping

from cloud.config import ServerConfig
from cloud.counters import WorkerState, WorkerStatus
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Mailbox
from cloud.scheduler import Batch, Dispatch, Idle, WaitUntil
from cloud.sessions import Session
from cloud.worker_link import WorkerFailed, WorkerLauncher, WorkerLink, WorkerStartError
from common.aio import sleep_until
from common.backoff import Backoff
from common.clock import Clock
from common.worker_protocol import WorkerProtocolError

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000
_MS_PER_S = 1_000


class ModelSupervisor:
    """Keeps one model's worker alive and dispatches the model's batches to it.

    dispatch is None until the first WorkerReady of the run and is then the same object for the
    life of the gateway, so the fairness memory, the batch ids, the counters and the batch cap
    survive a restart. ready is set at that first WorkerReady.
    """

    def __init__(
        self,
        model: str,
        config: ServerConfig,
        clock: Clock,
        launch: WorkerLauncher,
        mailbox: Mailbox,
        sessions: Mapping[str, Session],
        rng: random.Random,
    ) -> None:
        self.model = model
        self.status = WorkerStatus()
        self.dispatch: ModelDispatch | None = None
        self.ready = asyncio.Event()
        self._config = config
        self._clock = clock
        self._launch = launch
        self._mailbox = mailbox
        self._sessions = sessions
        self._backoff = Backoff(
            round(config.worker_restart_min_ms * _NS_PER_MS),
            round(config.worker_restart_max_ms * _NS_PER_MS),
            rng,
        )
        self._stop = asyncio.Event()
        # Ends the dispatch loop's Idle wait.
        self._wake = asyncio.Event()
        self._idle = False

    def poke(self) -> None:
        """Tell the dispatch loop that a frame for this model was accepted."""
        # A frame changes the decision only while the loop waits in Idle: during a batch the
        # loop calls next() anyway once the result is in, and no frame can change a pacing
        # hold. So every other frame costs this one attribute test and no wake-up.
        if self._idle:
            self._wake.set()

    def request_stop(self) -> None:
        """Dispatch no new batch, let the batch in flight settle, and end run().

        It also ends a restart pause and an Idle wait.
        """
        self._stop.set()
        self._wake.set()

    async def run(self) -> None:
        """Keep the model's worker alive and serving until a stop is requested.

        WorkerStartError when the first start fails, so that start-up fails fast. Later starts
        are retried for as long as the gateway runs.
        """
        while not self._stop.is_set():
            reached_ready = await self._run_worker()
            if not self._stop.is_set():
                self.status.state = WorkerState.RESTARTING if reached_ready else WorkerState.FAILED
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(self._backoff.next_ns() / 1e9):
                        # Completed by request_stop(). The timeout is the normal end of the
                        # pause.
                        await self._stop.wait()

    async def _run_worker(self) -> bool:
        """Start one worker and serve with it until it fails or a stop is requested.

        Returns whether this start reached READY. WorkerStartError when the first start does
        not.
        """
        # The dispatch exists from the first WorkerReady of the run on, and a start that does
        # not reach it ends the run, so only the first start finds none.
        first_start = self.dispatch is None
        if first_start:
            self.status.state = WorkerState.WARMING
        else:
            self.status.state = WorkerState.RESTARTING
            self.status.restarts += 1
        batch_cap = None if self.dispatch is None else self.dispatch.plan.scheduler.batch_cap
        try:
            # The link's loss sets the wake Event whatever the loop is doing: the top of the
            # dispatch loop tests link.lost, and the Event only has to end an Idle wait.
            link = await self._launch(self.model, batch_cap, self._wake.set)
        except WorkerStartError as error:
            if first_start:
                raise
            logger.error("worker %s could not be started again: %s", self.model, error)
            return False
        # The process is live while it warms up, so its pid shows from here on.
        self.status.pid = link.pid
        failure: WorkerFailed | None = None
        reached_ready = False
        try:
            dispatch = await self._warm_up(link)
            reached_ready = True
            self.status.state = WorkerState.READY
            self._backoff.reset()
            await self._serve(link, dispatch)
        except WorkerFailed as error:
            failure = error
            if error.kind != "died":
                # The state of a worker that timed out or broke the protocol is unknown, and
                # one stuck in native code runs no Python handler: only SIGKILL ends it for
                # certain.
                link.kill()
        finally:
            try:
                exit_status = await link.stop()
            finally:
                # The pid stays visible while the process may still be live.
                self.status.pid = None
        if failure is not None:
            if self.dispatch is None:
                raise WorkerStartError(f"{failure} (exit status {exit_status})") from failure
            logger.error(
                "worker %s failed (%s): %s; exit status %s",
                self.model,
                failure.kind,
                failure,
                exit_status,
            )
        return reached_ready

    async def _warm_up(self, link: WorkerLink) -> ModelDispatch:
        """Wait for the worker's WorkerReady and return the model's dispatch.

        The first WorkerReady of the run builds the dispatch. A later one is logged and not
        adopted: the batch cap and the calibration are fixed for the run (DESIGN.md 1.3 rule 9).
        """
        try:
            async with asyncio.timeout(self._config.warmup_timeout_ms / _MS_PER_S):
                ready = await link.ready()
        except TimeoutError as error:
            raise WorkerFailed(
                "timeout", f"no WorkerReady within {self._config.warmup_timeout_ms:g} ms"
            ) from error
        t_b_ms = {size: batch_ns / 1e6 for size, batch_ns in ready.t_b_ns.items()}
        if self.dispatch is not None:
            logger.info(
                "worker %s is ready again; its calibration is not adopted: T(b) in ms %s",
                self.model,
                t_b_ms,
            )
            return self.dispatch
        try:
            dispatch = ModelDispatch(
                self.model, self._config, ready, self._mailbox, self._clock.now_ns()
            )
        except ValueError as error:
            # The worker calibrated other sizes than the configured ones.
            raise WorkerFailed("protocol", str(error)) from error
        plan = dispatch.plan
        logger.info(
            "worker %s is ready: file %s, threads %d, T(b) in ms %s, batch cap %d, "
            "margin %.3f ms, stale threshold %s",
            self.model,
            ready.model_file,
            ready.threads,
            t_b_ms,
            plan.scheduler.batch_cap,
            plan.margin_ns / 1e6,
            "feasible" if plan.feasible else "not feasible",
        )
        self.dispatch = dispatch
        self.ready.set()
        return dispatch

    async def _serve(self, link: WorkerLink, dispatch: ModelDispatch) -> None:
        """Dispatch batches to the worker until a stop is requested; WorkerFailed when it fails.

        At most one batch is in flight by construction: the loop awaits a batch's result before
        it calls next() again.
        """
        scheduler = dispatch.scheduler
        # The sessions that got a command from the last result. They are woken, and so write
        # their commands, only after the next request is on its way: between two batches the
        # worker waits for the scheduler's work and one wake-up of this task, and for nothing
        # else (DESIGN.md 4.5).
        touched: list[Session] = []
        try:
            while not self._stop.is_set():
                if link.lost:
                    # The worker died, or broke the protocol, while it was idle. Tested before
                    # next(), so no frame is taken for a worker that cannot serve it. The kind
                    # the link recorded decides the kill: a worker that broke the protocol may
                    # still be running.
                    raise link.failure or WorkerFailed(
                        "died", "the worker's link was lost while it was idle"
                    )
                # The frames are chosen now, when the worker is free, so they are the newest.
                decision = scheduler.next(self._clock.now_ns())
                if isinstance(decision, Dispatch):
                    link.write_batch(
                        decision.batch.batch_id, [item.frame.jpeg for item in decision.batch.items]
                    )
                for session in touched:
                    session.wake()
                touched = []
                match decision:
                    case Dispatch(batch):
                        touched = await self._settle(link, dispatch, batch)
                    case WaitUntil(deadline_ns):
                        # No frame can change a pacing hold (cloud/scheduler.py), so frames do
                        # not wake this wait.
                        await sleep_until(self._clock, deadline_ns)
                    case Idle():
                        # Cleared before the flag is set, with no await since next(): a frame
                        # accepted from here on finds the flag and sets the Event, so none is
                        # missed.
                        self._wake.clear()
                        self._idle = True
                        # Completed by poke() (a frame for this model), by the link's loss or
                        # by request_stop(). Unbounded by design: with no frame, a live worker
                        # and no stop there is nothing to do.
                        await self._wake.wait()
                        self._idle = False
        finally:
            # The one place a batch fails, whatever ended the loop: the worker's death, a
            # timeout, a refused result or a cancellation.
            if scheduler.in_flight is not None:
                scheduler.fail(scheduler.in_flight, self._clock.now_ns())
            for session in touched:
                session.wake()

    async def _settle(
        self, link: WorkerLink, dispatch: ModelDispatch, batch: Batch
    ) -> list[Session]:
        """Await the batch's result and book it; return the sessions that got a command."""
        try:
            async with asyncio.timeout(self._config.inference_timeout_ms / _MS_PER_S):
                # The link stamps the result in its socket callback, so the wake-up of this
                # task is not in the dispatch round trip.
                result, result_ns = await link.result()
        except TimeoutError as error:
            self.status.timeouts += 1
            raise WorkerFailed(
                "timeout",
                f"no result for batch {batch.batch_id} within "
                f"{self._config.inference_timeout_ms:g} ms",
            ) from error
        try:
            return dispatch.complete(batch, result, result_ns, self._sessions)
        except WorkerProtocolError as error:
            raise WorkerFailed("protocol", str(error)) from error
