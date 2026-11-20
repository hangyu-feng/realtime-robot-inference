"""Tests for the gateway's composition: warm-up, the accept gate, the HTTP routes and shutdown.

They run on the virtual-time loop, with fake workers that answer after known times, an
in-memory listener and a reader of system statistics that reads nothing. What the gateway holds
is read as a client reads it: through /healthz and /metrics.
"""

import asyncio
import contextlib
import json
import random
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace
from typing import Any

import pytest

from cloud.config import ServerConfig
from cloud.counters import REJECT_REASONS, CloseReason
from cloud.gateway import Gateway
from cloud.metrics import MAX_CONCURRENT_RENDERS, STATS_READ_TIMEOUT_S, SystemStats
from cloud.scheduler import ModelPlan
from cloud.worker_link import WorkerStartError
from common.cgroup import CgroupStats, Limits
from common.commands import MODEL_KEYS
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    CloseCode,
    FrameHeader,
    Hello,
    decode_welcome,
    encode_frame,
    encode_hello,
)
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from common.transport import HttpReply, TransportClosed, TransportConnectError
from tests.support.fake_worker import FakeLauncher
from tests.support.memory_transport import MemoryConnection, MemoryListener, PipeLimits
from tests.support.virtual_time import LoopClock

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
CONFIG = ServerConfig()
# The fake workers' service time per frame. With the default thresholds the cls cap is then the
# largest calibrated size, and the depth cap is 2: a depth batch of 4 would take 160 ms.
FRAME_MS = {"cls": 2, "depth": 40}
BATCH_CAPS = {"cls": 8, "depth": 2}
# How long each fake worker takes to send its WorkerReady, so that warm-up has a middle.
READY_DELAY_MS = 100
CLOSE_NS = round(CONFIG.close_timeout_ms * NS_PER_MS)
GRACE_NS = round(CONFIG.shutdown_grace_ms * NS_PER_MS)
LIMITS = Limits(
    version=2,
    cpu_limit_cores=2.0,
    cpu_period_us=100_000,
    memory_limit_bytes=4 * 2**30,
    memory_swap_limit_bytes=0,
    cpuset="0-1",
)
CGROUP = CgroupStats(
    cpu_usage_us=0,
    cpu_user_us=0,
    cpu_system_us=0,
    nr_periods=0,
    nr_throttled=0,
    throttled_us=0,
    memory_current_bytes=0,
    memory_peak_bytes=0,
    memory_anon_bytes=0,
    memory_file_bytes=0,
    memory_sock_bytes=0,
    oom_kills=0,
)
ROBOT_END = PipeLimits(
    capacity_bytes=65_536,
    write_limit_bytes=65_536,
    max_message_bytes=4_096,
    close_timeout_ns=NS_PER_S,
)
# A robot that holds one message and never makes room: the welcome fills its inbox, so the
# gateway's close cannot reach it.
SILENT_ROBOT_END = replace(ROBOT_END, capacity_bytes=1)
# The gateway's end, with its write limit of 0.
GATEWAY_END = PipeLimits(
    capacity_bytes=65_536,
    write_limit_bytes=0,
    max_message_bytes=CONFIG.max_message_bytes,
    close_timeout_ns=CLOSE_NS,
    receive_queue_frames=CONFIG.receive_queue_frames,
)
JPEG = b"\xff\xd8\xff" + bytes(61)
WARMING = {"cls": "warming", "depth": "warming"}


def hello_text(robot_id: str, models: tuple[str, ...] = MODEL_KEYS) -> str:
    return encode_hello(
        Hello(
            protocol_version=PROTOCOL_VERSION,
            robot_id=robot_id,
            token=None,
            models=models,
            stale_ms={model: DEFAULT_STALE_MS[model] for model in models},
            watchdog_ms={model: DEFAULT_WATCHDOG_MS[model] for model in models},
            frame_rate_hz=30.0,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
    )


def decoded(reply: HttpReply | None) -> tuple[int, Any]:
    """Return an HTTP reply as its status and its JSON body."""
    assert reply is not None
    return reply.status, json.loads(reply.body)


def closed(**counts: int) -> dict[str, int]:
    """Return connections.closed with these counts and every other reason at 0."""
    return {reason.value: 0 for reason in CloseReason} | counts


def codes(closed_error: TransportClosed) -> tuple[int | None, int | None]:
    return closed_error.peer_code, closed_error.local_code


async def read_to_end(connection: MemoryConnection) -> tuple[list[str | bytes], TransportClosed]:
    """Read until recv() raises; return the messages read and what it raised."""
    messages: list[str | bytes] = []
    while True:
        try:
            messages.append(await connection.recv())
        except TransportClosed as closed_error:
            return messages, closed_error


async def settle() -> None:
    """Let every task run until all of them wait: only then does virtual time move, by 1 ms."""
    await asyncio.sleep(0.001)


def other_tasks() -> set[asyncio.Task[Any]]:
    """Return every task of the loop but the one that asks."""
    return asyncio.all_tasks() - {asyncio.current_task()}


class World:
    """A gateway with fake workers, and what a test needs to drive it.

    announced lists the plans of every announce call, and ready is set by the first. While
    stats_gate is clear, a read of the system statistics does not return.
    """

    def __init__(self, config: ServerConfig = CONFIG) -> None:
        self.clock = LoopClock()
        self.launcher = FakeLauncher(
            self.clock,
            {model: frame_ms * NS_PER_MS for model, frame_ms in FRAME_MS.items()},
            {model: settings.batch_sizes for model, settings in config.models.items()},
            READY_DELAY_MS * NS_PER_MS,
        )
        self.stop = asyncio.Event()
        self.ready = asyncio.Event()
        self.announced: list[Mapping[str, ModelPlan]] = []
        self.stats_gate = asyncio.Event()
        self.stats_gate.set()
        self.gateway = Gateway(
            config,
            self.clock,
            lambda: 0,
            LIMITS,
            "test",
            self.launcher,
            self.read_system,
            random.Random(0),
        )

    def announce(self, plans: Mapping[str, ModelPlan]) -> None:
        self.announced.append(plans)
        self.ready.set()

    async def read_system(self, pids: Mapping[str, int | None]) -> SystemStats:
        # Completed by the test that cleared the gate; the scrape's own timeout bounds it.
        await self.stats_gate.wait()
        return SystemStats(CGROUP, dict.fromkeys(pids))

    async def get(self, path: str) -> tuple[int, Any]:
        return decoded(await self.gateway.http(path))

    async def metrics(self) -> Any:
        status, snapshot = await self.get("/metrics")
        assert status == 200
        return snapshot

    def links(self) -> list[tuple[str, bool]]:
        """Return every worker launched so far: its model and whether it has been stopped."""
        return [(link.model, link.stopped) for link in self.launcher.links]


@dataclass(frozen=True)
class Service:
    """A gateway that runs behind a listener: the listener and the task of Gateway.run."""

    listener: MemoryListener
    run: asyncio.Task[None]


@contextlib.asynccontextmanager
async def serving(world: World, robot_end: PipeLimits = ROBOT_END) -> AsyncIterator[Service]:
    """Run the world's gateway behind an in-memory listener; at the end, stop it and join it."""
    async with (
        MemoryListener(
            world.gateway.handle, world.gateway.http, robot_end, GATEWAY_END
        ) as listener,
        asyncio.TaskGroup() as tasks,
    ):
        run = tasks.create_task(world.gateway.run(listener, world.stop, world.announce))
        yield Service(listener, run)
        world.stop.set()


async def open_robot(
    listener: MemoryListener, robot_id: str, models: tuple[str, ...] = MODEL_KEYS
) -> tuple[MemoryConnection, int]:
    """Connect a robot, say hello and read its welcome; return its end and its session number."""
    client = await listener.connect("/ws")
    client.send_nowait(hello_text(robot_id, models))
    welcome = await client.recv()
    assert isinstance(welcome, str)
    return client, decode_welcome(welcome).session


async def test_during_warm_up_health_is_503_with_the_states_and_an_upgrade_is_refused() -> None:
    world = World()

    async with serving(world) as service:
        await settle()
        health = await world.get("/healthz")
        upgrade = await world.get("/ws")
        with pytest.raises(TransportConnectError) as refused:
            await service.listener.connect("/ws")

    assert health == (503, {"status": "unavailable", "accepting": False, "workers": WARMING})
    assert upgrade == (503, {"status": "unavailable"})
    assert refused.value.http_status == 503


async def test_during_warm_up_metrics_render_with_no_uptime_and_no_calibration() -> None:
    world = World()

    async with serving(world):
        await settle()
        snapshot = await world.metrics()

    workers = {model: entry["state"] for model, entry in snapshot["workers"].items()}
    caps = {model: entry["batch_cap"] for model, entry in snapshot["workers"].items()}
    assert (snapshot["uptime_s"], workers, caps) == (None, WARMING, {"cls": None, "depth": None})
    assert snapshot["config"]["version"] == "test"


async def test_the_depth_worker_is_launched_only_after_the_cls_worker_is_ready() -> None:
    world = World()

    async with serving(world):
        await asyncio.sleep((READY_DELAY_MS - 1) / 1000)
        before_cls_ready = world.links()
        await asyncio.sleep(0.002)
        after_cls_ready = world.links()

    assert before_cls_ready == [("cls", False)]
    assert after_cls_ready == [("cls", False), ("depth", False)]


async def test_once_both_models_are_ready_the_gateway_is_healthy_and_accepts_a_robot() -> None:
    world = World()

    async with serving(world) as service:
        await world.ready.wait()
        ready_ns = world.clock.now_ns()
        health = await world.get("/healthz")
        client, session = await open_robot(service.listener, "a")
        snapshot = await world.metrics()
        await client.close(CloseCode.NORMAL, "")

    assert ready_ns == 2 * READY_DELAY_MS * NS_PER_MS
    assert health == (200, {"status": "ok"})
    assert (session, snapshot["connections"]["current"], snapshot["uptime_s"]) == (1, 1, 0.0)


async def test_announce_is_called_once_with_the_plan_of_each_model() -> None:
    world = World()

    async with serving(world):
        await world.ready.wait()
        await asyncio.sleep(1.0)
        snapshot = await world.metrics()

    announced = [
        {model: plan.scheduler.batch_cap for model, plan in plans.items()}
        for plans in world.announced
    ]
    served = {model: entry["batch_cap"] for model, entry in snapshot["workers"].items()}
    assert (announced, served) == ([BATCH_CAPS], BATCH_CAPS)


async def test_an_upgrade_is_refused_as_busy_once_live_connections_reach_twice_the_cap() -> None:
    world = World(replace(CONFIG, max_connections=1))

    async with serving(world) as service:
        await world.ready.wait()
        # Neither connection says hello: one in its handshake counts as one with a session.
        first = await service.listener.connect("/ws")
        await settle()
        below_the_bound = await world.gateway.http("/ws")
        second = await service.listener.connect("/ws")
        await settle()
        at_the_bound = await world.get("/ws")
        with pytest.raises(TransportConnectError) as refused:
            await service.listener.connect("/ws")
        first.abort()
        await settle()
        after_one_left = await world.gateway.http("/ws")
        second.abort()

    assert (below_the_bound, after_one_left) == (None, None)
    assert (at_the_bound, refused.value.http_status) == ((503, {"status": "busy"}), 503)


async def test_a_scrape_is_refused_as_busy_while_the_allowed_renders_are_running() -> None:
    world = World()

    async with serving(world):
        world.stats_gate.clear()
        async with asyncio.TaskGroup() as scrapes:
            held = [
                scrapes.create_task(world.gateway.http("/metrics"))
                for _ in range(MAX_CONCURRENT_RENDERS)
            ]
            await settle()
            one_more = await world.get("/metrics")
            world.stats_gate.set()
        after = await world.get("/metrics")

    assert one_more == (503, {"status": "busy"})
    assert [decoded(scrape.result())[0] for scrape in held] == [200] * MAX_CONCURRENT_RENDERS
    assert after[0] == 200


async def test_a_scrape_whose_statistics_do_not_arrive_is_answered_503_at_the_read_timeout() -> (
    None
):
    world = World()

    async with serving(world):
        world.stats_gate.clear()
        start_ns = world.clock.now_ns()
        unavailable = await world.get("/metrics")
        waited_ns = world.clock.now_ns() - start_ns
        world.stats_gate.set()
        after = await world.get("/metrics")

    assert unavailable == (503, {"status": "stats unavailable"})
    assert (waited_ns, after[0]) == (round(STATS_READ_TIMEOUT_S * NS_PER_S), 200)


@pytest.mark.parametrize("path", ["/", "/metrics/", "/ws/robot", "/health"])
async def test_any_other_path_is_answered_404(path: str) -> None:
    world = World()

    assert await world.get(path) == (404, {"status": "not found"})


@pytest.mark.parametrize(
    ("model", "launched"),
    [
        pytest.param("cls", [], id="cls"),
        pytest.param("depth", [("cls", True)], id="depth"),
    ],
)
async def test_a_failed_first_start_raises_out_of_run_and_leaves_no_task(
    model: str, launched: list[tuple[str, bool]]
) -> None:
    world = World()
    world.launcher.fail_starts[model] = 1

    async with MemoryListener(
        world.gateway.handle, world.gateway.http, ROBOT_END, GATEWAY_END
    ) as listener:
        with pytest.RaisesGroup(
            pytest.RaisesExc(WorkerStartError, match=f"^{model} failed to start: ")
        ):
            await world.gateway.run(listener, world.stop, world.announce)
        left = other_tasks()

    assert (left, world.links(), world.announced) == (set(), launched, [])


@pytest.mark.parametrize(
    ("stop_ms", "launched"),
    [
        pytest.param(READY_DELAY_MS // 2, [("cls", True)], id="while_cls_warms_up"),
        pytest.param(
            READY_DELAY_MS + READY_DELAY_MS // 2,
            [("cls", True), ("depth", True)],
            id="while_depth_warms_up",
        ),
    ],
)
async def test_a_stop_during_warm_up_ends_run_at_once_with_the_workers_and_the_listener_stopped(
    stop_ms: int, launched: list[tuple[str, bool]]
) -> None:
    world = World()

    async with serving(world) as service:
        await asyncio.sleep(stop_ms / 1000)
        world.stop.set()
        await service.run
        ended_ns = world.clock.now_ns()
        stopped = service.listener.stopped
        await service.listener.wait_closed()
        left = other_tasks()
        health = await world.get("/healthz")

    assert (ended_ns, stopped, left) == (stop_ms * NS_PER_MS, True, set())
    assert (world.links(), world.announced, health[0]) == (launched, [], 503)


async def test_a_stop_that_is_set_before_run_starts_launches_no_worker() -> None:
    world = World()
    world.stop.set()

    async with serving(world) as service:
        await service.run
        stopped = service.listener.stopped

    assert (world.links(), world.announced, stopped) == ([], [], True)


async def test_shutdown_closes_every_session_with_1001_and_counts_it_as_shutdown() -> None:
    world = World()

    async with serving(world) as service:
        await world.ready.wait()
        robots = [await open_robot(service.listener, robot_id) for robot_id in ("a", "b")]
        await settle()
        world.stop.set()
        ends = [await read_to_end(client) for client, _ in robots]
        await service.run
        await service.listener.wait_closed()
        left = other_tasks()
        upgrade = await world.get("/ws")
        snapshot = await world.metrics()

    assert [(messages, codes(at_client)) for messages, at_client in ends] == [
        ([], (CloseCode.SERVER_SHUTDOWN, None))
    ] * 2
    connections = snapshot["connections"]
    assert (connections["accepted"], connections["current"]) == (2, 0)
    assert connections["closed"] == closed(shutdown=2)
    assert (upgrade, left) == ((503, {"status": "unavailable"}), set())
    assert world.links() == [("cls", True), ("depth", True)]


async def test_shutdown_lets_the_batch_in_flight_finish_and_counts_its_result_as_dropped() -> None:
    world = World()

    async with serving(world) as service:
        await world.ready.wait()
        client, session = await open_robot(service.listener, "a", models=("depth",))
        sent_ns = world.clock.now_ns()
        client.send_nowait(encode_frame(FrameHeader(session, 1, sent_ns, sent_ns), JPEG))
        await asyncio.sleep(FRAME_MS["depth"] / 4 / 1000)
        world.stop.set()
        await service.run
        ended_ns = world.clock.now_ns()
        snapshot = await world.metrics()

    depth = snapshot["robots"]["a"]["models"]["depth"]
    assert ended_ns - sent_ns == FRAME_MS["depth"] * NS_PER_MS
    assert (depth["frames_received"], depth["in_flight"], depth["frames_failed"]) == (1, 0, 0)
    assert (depth["results_dropped_disconnect"], depth["commands_sent"]) == (1, 0)


async def stop_with_a_silent_peer(
    world: World, service: Service
) -> tuple[int, int, TransportClosed]:
    """Open a session for a peer that never reads, then stop the gateway.

    Return how long run() took after the stop, how many messages the peer's end then held, and
    how its connection ended.
    """
    client = await service.listener.connect("/ws")
    client.send_nowait(hello_text("silent"))
    await settle()
    stop_ns = world.clock.now_ns()
    world.stop.set()
    await service.run
    took_ns = world.clock.now_ns() - stop_ns
    messages, at_client = await read_to_end(client)
    return took_ns, len(messages), at_client


async def test_a_peer_that_never_answers_its_close_is_aborted_at_the_close_timeout() -> None:
    world = World()

    async with serving(world, SILENT_ROBOT_END) as service:
        await world.ready.wait()
        took_ns, held, at_client = await stop_with_a_silent_peer(world, service)
        snapshot = await world.metrics()

    # The welcome is all the peer's end holds: the close frame never entered it.
    assert (took_ns, held, codes(at_client)) == (CLOSE_NS, 1, (None, None))
    assert snapshot["connections"]["closed"] == closed(shutdown=1)


async def test_a_connection_that_closed_before_the_stop_does_not_end_the_drain_early() -> None:
    world = World()

    async with serving(world, SILENT_ROBOT_END) as service:
        await world.ready.wait()
        earlier, _ = await open_robot(service.listener, "earlier")
        await earlier.close(CloseCode.NORMAL, "")
        await settle()
        took_ns, _, at_client = await stop_with_a_silent_peer(world, service)
        snapshot = await world.metrics()

    assert (took_ns, codes(at_client)) == (CLOSE_NS, (None, None))
    assert snapshot["connections"]["closed"] == closed(client_close=1, shutdown=1)


async def test_the_drain_waits_for_the_last_connection_after_another_has_closed() -> None:
    world = World()

    async with serving(world, SILENT_ROBOT_END) as service:
        await world.ready.wait()
        # This robot has read its welcome, so the close of the shutdown reaches it at once,
        # while the silent peer's connection is still closing.
        reader, _ = await open_robot(service.listener, "reader")
        took_ns, _, at_client = await stop_with_a_silent_peer(world, service)
        _, at_reader = await read_to_end(reader)
        snapshot = await world.metrics()

    assert (took_ns, codes(at_client)) == (CLOSE_NS, (None, None))
    assert codes(at_reader) == (CloseCode.SERVER_SHUTDOWN, None)
    assert snapshot["connections"]["closed"] == closed(shutdown=2)


async def test_a_connection_that_never_said_hello_is_aborted_at_the_grace_deadline() -> None:
    world = World()

    async with serving(world) as service:
        await world.ready.wait()
        client = await service.listener.connect("/ws")
        await settle()
        stop_ns = world.clock.now_ns()
        world.stop.set()
        await service.run
        took_ns = world.clock.now_ns() - stop_ns
        messages, at_client = await read_to_end(client)
        await service.listener.wait_closed()
        left = other_tasks()
        snapshot = await world.metrics()

    assert (took_ns, messages, codes(at_client)) == (GRACE_NS, [], (None, None))
    connections = snapshot["connections"]
    assert (connections["accepted"], connections["rejected"]) == (
        0,
        dict.fromkeys(REJECT_REASONS, 0),
    )
    assert left == set()


async def test_the_paced_share_is_recomputed_when_a_robot_joins_and_when_it_leaves() -> None:
    # One robot's 30 frames per second at 40 ms each are 1.2 cores of I/O, which leave the
    # paced worker 0.7 of the 2 cores after the headroom of 0.1; with no robot it is not paced.
    world = World(replace(CONFIG, io_cpu_ms_per_frame=40.0))
    shares: list[float | None] = []

    async with serving(world) as service:
        await world.ready.wait()
        shares.append((await world.metrics())["workers"]["depth"]["pacing_share"])
        client, _ = await open_robot(service.listener, "a")
        shares.append((await world.metrics())["workers"]["depth"]["pacing_share"])
        await client.close(CloseCode.NORMAL, "")
        await settle()
        shares.append((await world.metrics())["workers"]["depth"]["pacing_share"])

    assert shares == [None, pytest.approx(0.7), None]


async def test_the_loop_probe_samples_only_while_a_robot_is_connected() -> None:
    world = World()
    period_s = CONFIG.loop_probe_ms / 1000
    samples: list[int] = []

    async with serving(world) as service:
        await world.ready.wait()
        await asyncio.sleep(1.0)
        samples.append((await world.metrics())["loop"]["lag_ms"]["count"])
        client, _ = await open_robot(service.listener, "a")
        # Half a period past the tenth target, so that no sample falls on this instant.
        await asyncio.sleep(10.5 * period_s)
        samples.append((await world.metrics())["loop"]["lag_ms"]["count"])
        await client.close(CloseCode.NORMAL, "")
        await asyncio.sleep(1.0)
        samples.append((await world.metrics())["loop"]["lag_ms"]["count"])

    assert samples == [0, 10, 10]
