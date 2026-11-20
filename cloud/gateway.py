"""The gateway's composition: its shared state, the accept gate, the HTTP routes and its life.

A Gateway builds the robot table, the mailbox, the session table and one supervisor per model,
and joins them: it serves each connection, answers the plain HTTP paths, warms the models up
one at a time, and shuts down in stages. It decides nothing about a frame or a batch: those
rules are in cloud/sessions.py and cloud/dispatch.py.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
from collections.abc import Callable, Mapping

from cloud.config import ServerConfig, config_echo
from cloud.counters import CloseReason, ConnectionTotals
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Mailbox
from cloud.metrics import (
    MAX_CONCURRENT_RENDERS,
    MetricsSources,
    SystemStatsReader,
    health_reply,
    probe_loop_lag,
    render_metrics,
)
from cloud.pacing import Pacing
from cloud.registry import RobotRegistry
from cloud.scheduler import ModelPlan
from cloud.session import SessionContext, run_connection
from cloud.sessions import SessionTable
from cloud.supervisor import ModelSupervisor
from cloud.worker_link import WorkerLauncher, WorkerStartError
from common.cgroup import Limits
from common.clock import Clock
from common.commands import MODEL_KEYS
from common.histogram import Histogram
from common.protocol import PROTOCOL_VERSION
from common.transport import Connection, HttpReply, Listener

logger = logging.getLogger(__name__)

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
_MS_PER_S = 1_000
_UNAVAILABLE = HttpReply(503, b'{"status":"unavailable"}')
_BUSY = HttpReply(503, b'{"status":"busy"}')
_STATS_UNAVAILABLE = HttpReply(503, b'{"status":"stats unavailable"}')
_NOT_FOUND = HttpReply(404, b'{"status":"not found"}')


async def _supervise(supervisor: ModelSupervisor) -> None:
    """Run a model's supervisor; a failed first start is raised with the model's name in it."""
    try:
        await supervisor.run()
    except WorkerStartError as error:
        # The supervisor's reason does not say which model it is for, and the one line the
        # process prints before it exits has to.
        raise WorkerStartError(f"{supervisor.model} failed to start: {error}") from error


class Gateway:
    """One gateway: the state its connections and its workers share, and the tasks around it."""

    def __init__(
        self,
        config: ServerConfig,
        clock: Clock,
        wall_clock_ms: Callable[[], int],
        limits: Limits,
        version: str,
        launch: WorkerLauncher,
        read_system: SystemStatsReader,
        rng: random.Random,
    ) -> None:
        self._config = config
        self._clock = clock
        self._limits = limits
        self._registry = RobotRegistry(
            round(config.robot_retention_s * _NS_PER_S), config.max_robot_entries
        )
        mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        totals = ConnectionTotals()
        self._table = SessionTable(
            config, self._registry, mailbox, totals, self._membership_changed
        )
        # Each supervisor draws its restart pauses from a generator of its own, so the pauses
        # of one model do not depend on how often the other has failed.
        self._supervisors = {
            model: ModelSupervisor(
                model,
                config,
                clock,
                launch,
                mailbox,
                self._table.sessions,
                random.Random(rng.getrandbits(64)),
            )
            for model in MODEL_KEYS
        }
        loop_lag_ms = Histogram()
        # Set exactly while a robot is connected: the loop-lag probe runs only then.
        self._probe_active = asyncio.Event()
        # Every connection whose handler runs, with or without a session. The upgrade gate
        # counts them, and the last shutdown stage aborts them.
        self._connections: set[Connection] = set()
        # Set exactly while no connection is live. It starts set because the set starts empty.
        self._drained = asyncio.Event()
        self._drained.set()
        # None until both models are warm.
        self._pacing: Pacing | None = None
        self._accepting = False
        self._accepting_since_ns: int | None = None
        self._renders = 0
        self._context = SessionContext(
            config=config,
            clock=clock,
            table=self._table,
            totals=totals,
            pokes={model: supervisor.poke for model, supervisor in self._supervisors.items()},
            accepting=lambda: self._accepting,
        )
        self._sources = MetricsSources(
            config_echo=config_echo(config, version=version, protocol_version=PROTOCOL_VERSION),
            limits=limits,
            clock=clock,
            wall_clock_ms=wall_clock_ms,
            gateway_pid=os.getpid(),
            registry=self._registry,
            mailbox=mailbox,
            totals=totals,
            supervisors=self._supervisors,
            pacing_shares=self._pacing_shares,
            accepting_since_ns=lambda: self._accepting_since_ns,
            loop_lag_ms=loop_lag_ms,
            read_system=read_system,
        )

    async def handle(self, connection: Connection) -> None:
        """Serve one connection, from its upgrade to its end."""
        self._connections.add(connection)
        self._drained.clear()
        try:
            await run_connection(connection, self._context)
        finally:
            self._connections.discard(connection)
            # Only when the set is empty again: a connection that closed earlier must not let
            # the shutdown pass its drain stage while other sessions are still closing.
            if not self._connections:
                self._drained.set()

    async def http(self, path: str) -> HttpReply | None:
        """Answer a plain HTTP request for path; None lets it continue as a WebSocket upgrade."""
        if path == "/healthz":
            states = {
                model: supervisor.status.state for model, supervisor in self._supervisors.items()
            }
            return health_reply(states, self._accepting)
        if path == "/metrics":
            if self._renders >= MAX_CONCURRENT_RENDERS:
                return _BUSY
            self._renders += 1
            try:
                return HttpReply(200, await render_metrics(self._sources))
            except TimeoutError:
                # The read of the kernel's statistics did not return within its bound.
                return _STATS_UNAVAILABLE
            finally:
                self._renders -= 1
        if path == "/ws":
            if not self._accepting:
                # During warm-up, and from the first shutdown stage on.
                return _UNAVAILABLE
            # Sessions plus connections still in their handshake, so twice the session cap: a
            # reconnect storm of every allowed robot can complete while every old session is
            # still open. The bound is approximate by the upgrades in flight between this
            # check and the start of their handlers.
            if len(self._connections) >= 2 * self._config.max_connections:
                return _BUSY
            return None
        return _NOT_FOUND

    async def run(
        self,
        listener: Listener,
        stop: asyncio.Event,
        announce: Callable[[Mapping[str, ModelPlan]], None],
    ) -> None:
        """Warm the models up, accept robots until stop is set, then shut down in stages.

        announce is called once, with each model's plan, when the gateway starts accepting. A
        first start that fails raises WorkerStartError, in the ExceptionGroup of the tasks
        this runs. After stop is set this returns within shutdown_grace_ms plus twice
        worker_stop_timeout_ms, and the caller then waits for the listener within
        close_timeout_ms, so the whole shutdown is bounded by
        shutdown_grace_ms + 2 * worker_stop_timeout_ms + close_timeout_ms.
        """
        async with asyncio.TaskGroup() as tasks:
            probe = tasks.create_task(
                probe_loop_lag(
                    self._clock,
                    self._sources.loop_lag_ms,
                    round(self._config.loop_probe_ms * _NS_PER_MS),
                    self._probe_active,
                )
            )
            running, dispatches = await self._warm_up(tasks, stop)
            accepted = not stop.is_set()
            if accepted:
                self._start_accepting(dispatches)
                announce({model: dispatch.plan for model, dispatch in dispatches.items()})
                # No timeout by design: completed by the caller's signal handler, and the wait
                # is the lifetime of the service.
                await stop.wait()

            # Stage 1 runs on every path, also for a stop during warm-up, when the gate was
            # never open: the listener's wait_closed(), which the caller awaits next, returns
            # only once the listener has been told to stop, and a listener left open would
            # take connections until the process exits.
            self._accepting = False
            listener.stop_accepting()
            logger.info("shutdown, stage 1: stopped accepting connections")
            if accepted:
                await self._drain(running)
            logger.info(
                "shutdown, stage 3: aborting %d connections and cancelling what still runs",
                len(self._connections),
            )
            # This also covers the connections that never said hello.
            for connection in self._connections:
                connection.abort()
            # A supervisor that is cancelled stops and reaps its worker, within its bounds, and
            # one that is already stopping its worker kills it at once.
            for task in running:
                task.cancel()
            probe.cancel()

    async def _warm_up(
        self, tasks: asyncio.TaskGroup, stop: asyncio.Event
    ) -> tuple[list[asyncio.Task[None]], dict[str, ModelDispatch]]:
        """Start the supervisors one by one, each after the first calibration of the one before.

        Returns the supervisors' tasks and the dispatch of each model that became ready: of
        every model unless stop was set first.
        """
        running: list[asyncio.Task[None]] = []
        dispatches: dict[str, ModelDispatch] = {}
        # One model at a time: two workers calibrating at once would compete for the CPU
        # quota, and a throttled calibration gives a wrong T(b) and so a wrong batch cap
        # (DESIGN.md 6.2).
        for model, supervisor in self._supervisors.items():
            if stop.is_set():
                break
            running.append(tasks.create_task(_supervise(supervisor)))
            waiters = [
                asyncio.create_task(supervisor.ready.wait()),
                asyncio.create_task(stop.wait()),
            ]
            try:
                # No timeout by design: completed by the supervisor's first WorkerReady, which
                # the supervisor itself awaits under the warm-up timeout, or by the stop Event.
                await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
            finally:
                # asyncio.wait cancels nothing it waits on, so without this the waiter that
                # did not complete would outlive the wait, also when a failed first start
                # cancels this task. A waiter that is done ignores the cancel.
                for waiter in waiters:
                    waiter.cancel()
            # None when the stop came first.
            if supervisor.dispatch is not None:
                dispatches[model] = supervisor.dispatch
        return running, dispatches

    def _start_accepting(self, dispatches: Mapping[str, ModelDispatch]) -> None:
        """Open the accept gate, once every model has its dispatch."""
        self._pacing = Pacing(self._config, self._limits.cpu_limit_cores, dispatches)
        # No robot is connected yet, so this is the paced worker's share with none.
        self._pacing.update(self._registry.records())
        # ASSUMPTION: A14. Warm-up finishes before robots connect: the gate opens only here,
        # so no window of a run includes a calibration.
        self._accepting = True
        self._accepting_since_ns = self._clock.now_ns()
        logger.info("accepting connections")

    async def _drain(self, running: list[asyncio.Task[None]]) -> None:
        """Run shutdown stage 2: end every session and let the batches in flight settle."""
        for supervisor in self._supervisors.values():
            supervisor.request_stop()
        now_ns = self._clock.now_ns()
        # Each end releases the session's state at once, and its handler then closes the
        # connection with 1001. A result still at a worker is counted by the core as dropped
        # for a robot that has gone.
        sessions = list(self._table.sessions.values())
        for session in sessions:
            self._table.end(session, CloseReason.SHUTDOWN, now_ns)
        logger.info(
            "shutdown, stage 2: ended %d sessions; draining for up to %g ms",
            len(sessions),
            self._config.shutdown_grace_ms,
        )
        try:
            async with asyncio.timeout(self._config.shutdown_grace_ms / _MS_PER_S):
                # Completed by handle(), when the last connection's handler has returned.
                await self._drained.wait()
                # Completed by the supervisors: each returns once its batch in flight has
                # settled and its worker has exited.
                await asyncio.wait(running)
        except TimeoutError:
            logger.warning("shutdown: not drained within the grace period")

    def _membership_changed(self) -> None:
        """Follow a robot's join or leave: the paced worker's share and the loop-lag probe."""
        if self._pacing is None:
            # The accept gate opens only after both models are warm, so no session can
            # change before then.
            raise RuntimeError("a session changed before the models were ready")
        self._pacing.update(self._registry.records())
        if self._registry.connected_count() > 0:
            self._probe_active.set()
        else:
            self._probe_active.clear()

    def _pacing_shares(self) -> Mapping[str, float | None]:
        """Return the share each model's worker is paced to; nothing is paced during warm-up."""
        return {} if self._pacing is None else self._pacing.shares()
