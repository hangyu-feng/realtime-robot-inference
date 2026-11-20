"""The whole system in one process: the gateway, fake workers and robots on one virtual-time loop.

InProcessSystem runs the real gateway with fake worker links behind a MemoryListener, and real
robots whose connector opens in-memory connections to it. Everything reads the loop's clock,
so a test drives minutes of traffic in moments, and the times it asserts are exact.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from http import HTTPStatus
from types import MappingProxyType, TracebackType
from typing import Any, Self

from cloud.config import OVERRIDES_VARIABLE, load_config
from cloud.gateway import Gateway
from cloud.metrics import SystemStats
from cloud.scheduler import ModelPlan
from cloud.worker_link import WorkerFailed, WorkerLink
from common.aio import sleep_until
from common.cgroup import CgroupStats, Limits
from common.clock import Clock
from common.commands import MODEL_KEYS
from common.transport import Connection
from common.worker_protocol import BatchResult, WorkerReady
from robot.robot import Robot
from robot.samples import SampleLog
from robot.schedule import CaptureSchedule
from robot.settings import LinkSettings, RobotSettings
from tests.support.fake_worker import FakeLauncher, FakeWorkerLink
from tests.support.memory_transport import MemoryListener, PipeLimits
from tests.support.virtual_time import LoopClock

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000

# The container of the demos: a CPU quota of 2 cores and 4 GiB of memory.
LIMITS = Limits(
    version=2,
    cpu_limit_cores=2.0,
    cpu_period_us=100_000,
    memory_limit_bytes=4 * 2**30,
    memory_swap_limit_bytes=0,
    cpuset=None,
)
# Every frame passes the gateway's start-marker check and the fake runtime's, and the sizes
# differ, so a frame's own bytes can be told apart at the worker.
JPEGS: tuple[bytes, ...] = tuple(b"\xff\xd8\xff" + bytes(size - 3) for size in (600, 900, 1200))
# A direction of the link model that holds nothing back.
NO_DELAY = LinkSettings()

_NO_OVERRIDES: Mapping[str, object] = MappingProxyType({})
# Every robot runs with the default settings but for its link model, so one pair of limits
# serves every connection.
_ROBOT_DEFAULTS = RobotSettings()
# The terms of the second and third identities of DESIGN.md 10.3.
_RECEIVED_TERMS = (
    "frames_served",
    "frames_overwritten",
    "frames_skipped",
    "frames_dropped_disconnect",
    "frames_pending",
)
_SERVED_TERMS = (
    "commands_sent",
    "commands_queued",
    "commands_dropped_downlink",
    "results_dropped_disconnect",
    "frames_failed",
    "in_flight",
)


async def _read_no_stats(pids: Mapping[str, int | None]) -> SystemStats:
    """Report no cgroup usage and no live process: the virtual loop reads no kernel file."""
    cgroup = CgroupStats(
        cpu_usage_us=None,
        cpu_user_us=None,
        cpu_system_us=None,
        nr_periods=None,
        nr_throttled=None,
        throttled_us=None,
        memory_current_bytes=None,
        memory_peak_bytes=None,
        memory_anon_bytes=None,
        memory_file_bytes=None,
        memory_sock_bytes=None,
        oom_kills=None,
    )
    return SystemStats(cgroup, dict.fromkeys(pids))


class _DelayedResultLink:
    """A fake worker link that hands each result over as a real link and its dispatcher do.

    A real link stamps a result in its socket callback once the result has crossed the socket,
    and the dispatcher's task resumes after that. The fake link alone stamps a result at the
    instant the worker sends it and resumes the dispatcher at that instant too. This link moves
    the stamp result_leg_ns after the worker's send, and the dispatcher's resumption
    wake_up_ns after the stamp.
    """

    def __init__(
        self, link: FakeWorkerLink, clock: Clock, result_leg_ns: int, wake_up_ns: int
    ) -> None:
        self._link = link
        self._clock = clock
        self._result_leg_ns = result_leg_ns
        self._wake_up_ns = wake_up_ns

    @property
    def pid(self) -> int:
        return self._link.pid

    @property
    def lost(self) -> bool:
        return self._link.lost

    @property
    def failure(self) -> WorkerFailed | None:
        return self._link.failure

    async def ready(self) -> WorkerReady:
        return await self._link.ready()

    def write_batch(self, batch_id: int, jpegs: Sequence[bytes | memoryview]) -> None:
        self._link.write_batch(batch_id, jpegs)

    async def result(self) -> tuple[BatchResult, int]:
        result, _ = await self._link.result()
        result_ns = result.sent_ns + self._result_leg_ns
        # The fake link resumes this task at the worker's send. With both delays 0 that is the
        # deadline, so sleep_until returns without yielding and the dispatcher resumes exactly
        # as it does with the fake link alone.
        await sleep_until(self._clock, result_ns + self._wake_up_ns)
        return result, result_ns

    def kill(self) -> None:
        self._link.kill()

    async def stop(self) -> int | None:
        return await self._link.stop()


@dataclass(frozen=True)
class _RobotHandle:
    """One robot of the system: its task, its stop Event, its samples and its connections."""

    task: asyncio.Task[None]
    stop: asyncio.Event
    samples: SampleLog
    # The robot's end of every connection it has opened, oldest first.
    connections: list[Connection]


class InProcessSystem:
    """The gateway with fake workers, and the robots a test adds, as an async context manager.

    frame_ms gives each model's fake service time per frame, in whole milliseconds. overrides
    sets any other setting as RRI_OVERRIDES does; its models entry is replaced by the fake
    runtimes of frame_ms. result_leg_ms is the time a result takes from the worker's send to
    the gateway's link, which stamps it then, and dispatcher_wake_up_ms the time the model's
    dispatcher takes to resume after that stamp, both in whole milliseconds. At 0, the
    default, the worker's send, the link's stamp and the dispatcher's resumption are one
    instant. Entering starts the gateway and returns once it accepts robots. Leaving stops
    every robot as its stop signal does, then shuts the gateway down and joins every task.
    """

    def __init__(
        self,
        frame_ms: Mapping[str, int],
        overrides: Mapping[str, object] = _NO_OVERRIDES,
        *,
        result_leg_ms: int = 0,
        dispatcher_wake_up_ms: int = 0,
    ) -> None:
        models = {
            model: {"runtime": "fake", "fake_frame_ms": frame_ms[model]} for model in MODEL_KEYS
        }
        # The path a scenario's settings take into the real service, so the test runs with the
        # configuration they give there.
        self.config = load_config({OVERRIDES_VARIABLE: json.dumps({**overrides, "models": models})})
        self.clock = LoopClock()
        self._launcher = FakeLauncher(
            self.clock,
            {model: frame_ms[model] * _NS_PER_MS for model in MODEL_KEYS},
            {model: settings.batch_sizes for model, settings in self.config.models.items()},
        )

        async def launch(
            model: str, batch_cap: int | None, on_lost: Callable[[], None]
        ) -> WorkerLink:
            return _DelayedResultLink(
                await self._launcher(model, batch_cap, on_lost),
                self.clock,
                result_leg_ms * _NS_PER_MS,
                dispatcher_wake_up_ms * _NS_PER_MS,
            )

        self._gateway = Gateway(
            self.config,
            self.clock,
            lambda: 0,
            LIMITS,
            "test",
            launch,
            _read_no_stats,
            random.Random(0),
        )
        self._listener = MemoryListener(
            self._gateway.handle,
            self._gateway.http,
            PipeLimits(
                capacity_bytes=_ROBOT_DEFAULTS.rcvbuf_bytes,
                # A robot only calls send_nowait(), which no write limit holds back.
                write_limit_bytes=0,
                max_message_bytes=_ROBOT_DEFAULTS.max_message_bytes,
                close_timeout_ns=round(_ROBOT_DEFAULTS.close_timeout_ms * _NS_PER_MS),
            ),
            PipeLimits(
                capacity_bytes=self.config.socket_rcvbuf_bytes,
                # The gateway runs its transports with a write limit of 0.
                write_limit_bytes=0,
                max_message_bytes=self.config.max_message_bytes,
                close_timeout_ns=round(self.config.close_timeout_ms * _NS_PER_MS),
                receive_queue_frames=self.config.receive_queue_frames,
            ),
        )
        self._tasks = asyncio.TaskGroup()
        self._stack = contextlib.AsyncExitStack()
        self._stop = asyncio.Event()
        self._ready = asyncio.Event()
        self._handles: list[_RobotHandle] = []
        # The latest robot added under each id.
        self._robots: dict[str, _RobotHandle] = {}

    async def __aenter__(self) -> Self:
        async with contextlib.AsyncExitStack() as stack:
            await stack.enter_async_context(self._listener)
            await stack.enter_async_context(self._tasks)
            self._tasks.create_task(self._gateway.run(self._listener, self._stop, self._announce))
            # Completed by the gateway's announcement that it accepts robots. A first worker
            # start that fails ends run(), and the TaskGroup then cancels this wait.
            await self._ready.wait()
            self._stack = stack.pop_all()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        for handle in self._handles:
            handle.stop.set()
        running = [handle.task for handle in self._handles if not handle.task.done()]
        if exc is None and running:
            # Completed by each robot's planned stop at its next tick, so the robots close
            # their sessions before the gateway shuts down, as at the end of a run.
            await asyncio.wait(running)
        self._stop.set()
        return await self._stack.__aexit__(exc_type, exc, traceback)

    def add_robot(
        self,
        robot_id: str,
        start_ns: int,
        stop_ns: int,
        *,
        uplink: LinkSettings = NO_DELAY,
        downlink: LinkSettings = NO_DELAY,
        frame_rate_hz: float = _ROBOT_DEFAULTS.frame_rate_hz,
    ) -> Robot:
        """Start a robot that streams from start_ns, where its capture schedule starts, to stop_ns.

        A robot added under the id of a connected one replaces that session, as a second
        process with the id would.
        """
        settings = RobotSettings(uplink=uplink, downlink=downlink, frame_rate_hz=frame_rate_hz)
        period_ns = round(_NS_PER_S / settings.frame_rate_hz)
        # Every tick of the robot's time with a second of slack, and a command per tick and
        # model, as robot.fleet sizes its logs.
        tick_capacity = math.ceil((stop_ns - start_ns) / period_ns) + math.ceil(
            settings.frame_rate_hz
        )
        samples = SampleLog(tick_capacity, tick_capacity * len(settings.models))
        connections: list[Connection] = []

        async def connect() -> Connection:
            connection = await self._listener.connect("/ws")
            connections.append(connection)
            return connection

        robot = Robot(
            robot_id,
            settings,
            CaptureSchedule(start_ns, period_ns),
            JPEGS,
            start_ns,
            stop_ns,
            connect,
            self.clock,
            random.Random(robot_id),
            samples,
        )
        stop = asyncio.Event()
        handle = _RobotHandle(self._tasks.create_task(robot.run(stop)), stop, samples, connections)
        self._handles.append(handle)
        self._robots[robot_id] = handle
        return robot

    def samples(self, robot_id: str) -> SampleLog:
        """Return the sample log of the latest robot added under robot_id."""
        return self._robots[robot_id].samples

    def leave(self, robot_id: str) -> None:
        """Stop the latest robot of robot_id as its stop signal does: a last report, then 1000."""
        self._robots[robot_id].stop.set()

    def abort(self, robot_id: str) -> None:
        """Kill the latest robot of robot_id as SIGKILL kills its process: no close is sent.

        Its connection is aborted and its task cancelled, so it never connects again.
        """
        handle = self._robots[robot_id]
        for connection in handle.connections:
            # A no-op for a connection that has already closed.
            connection.abort()
        handle.task.cancel()

    def crash_worker(self, model: str) -> None:
        """End the process behind the model's latest worker link; its supervisor restarts it."""
        links = [link for link in self._launcher.links if link.model == model]
        links[-1].crash()

    async def advance(self, seconds: float) -> None:
        """Let the system run for this much virtual time."""
        await asyncio.sleep(seconds)

    async def metrics(self) -> dict[str, Any]:
        """Return the gateway's /metrics snapshot, parsed."""
        reply = await self._gateway.http("/metrics")
        if reply is None or reply.status != HTTPStatus.OK:
            raise RuntimeError(f"/metrics did not answer with a snapshot: {reply}")
        snapshot: dict[str, Any] = json.loads(reply.body)
        return snapshot

    def _announce(self, plans: Mapping[str, ModelPlan]) -> None:
        self._ready.set()


def identity_violations(snapshot: Mapping[str, Any]) -> list[str]:
    """Return each identity of DESIGN.md 10.3 that a /metrics snapshot breaks, by its path.

    It also checks the connection count, accepted = current + sum(closed), which holds because
    every accepted session ends exactly once.
    """
    violations: list[str] = []
    for robot_id, robot in snapshot["robots"].items():
        if robot["frames_in"] != robot["frames_rejected"] + robot["frames_accepted"]:
            violations.append(f"robots.{robot_id}: frames_in")
        for model, counters in robot["models"].items():
            path = f"robots.{robot_id}.models.{model}"
            if counters["frames_received"] != sum(counters[name] for name in _RECEIVED_TERMS):
                violations.append(f"{path}: frames_received")
            if counters["frames_served"] != sum(counters[name] for name in _SERVED_TERMS):
                violations.append(f"{path}: frames_served")
    connections = snapshot["connections"]
    if connections["accepted"] != connections["current"] + sum(connections["closed"].values()):
        violations.append("connections: accepted")
    return violations
