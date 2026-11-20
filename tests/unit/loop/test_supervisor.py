"""Tests for a model's supervisor: start, warm-up, the dispatch loop, failures and restarts.

They run on the virtual-time loop, with a real scheduler, mailbox, registry and session table,
and fake workers that answer after a known service time.
"""

import asyncio
import contextlib
import logging
import random
from collections.abc import AsyncIterator, Awaitable, Callable
from functools import partial

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cloud.config import ServerConfig
from cloud.counters import CloseReason, ConnectionTotals, WorkerState
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Mailbox
from cloud.registry import ModelCounters, RobotRegistry
from cloud.scheduler import Decision, ModelScheduler
from cloud.sessions import Opened, Session, SessionTable
from cloud.supervisor import ModelSupervisor
from cloud.worker_link import WorkerStartError
from common.aio import sleep_until
from common.commands import ClsCommand, Command
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    FrameHeader,
    Hello,
    decode_command_message,
    encode_frame,
    encode_hello,
)
from tests.support.fake_worker import STOP_TIMEOUT_NS, FakeLauncher, FakeWorkerLink
from tests.support.virtual_time import LoopClock, run_virtual

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
# The skip rule is off so that a frame that waited out a restart is still served: these tests
# are about the loop around the scheduler, not about the rule.
CONFIG = ServerConfig(skip_rule=False)
# The fake worker's service time per frame. Every batch size then fits the cls budget, so the
# cap is the largest size.
FRAME_MS = 10
BATCH_CAP = 8
INFERENCE_TIMEOUT_MS = round(CONFIG.inference_timeout_ms)
WARMUP_TIMEOUT_MS = round(CONFIG.warmup_timeout_ms)
# The first restart pause is jittered between half the minimum and the minimum, and the second
# between the minimum and twice the minimum.
RESTART_MIN_MS = round(CONFIG.worker_restart_min_ms)
STOP_TIMEOUT_MS = STOP_TIMEOUT_NS // NS_PER_MS
JPEG_START = b"\xff\xd8\xff"
# Passes the gateway's two-byte prefilter and fails in the worker as a frame that does not
# decode.
BROKEN_JPEG_START = b"\xff\xd8\x00"


@pytest.fixture
def next_calls(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """The time in ms of every ModelScheduler.next() call of the test, in call order."""
    calls: list[int] = []

    class RecordingScheduler(ModelScheduler):
        def next(self, now_ns: int) -> Decision:
            calls.append(now_ns // NS_PER_MS)
            return super().next(now_ns)

    monkeypatch.setattr("cloud.dispatch.ModelScheduler", RecordingScheduler)
    return calls


class World:
    """The cls supervisor over a real session table, mailbox and registry, with fake workers."""

    def __init__(
        self,
        ready_delay_ms: int = 0,
        calibrated_sizes: tuple[int, ...] = CONFIG.models["cls"].batch_sizes,
    ) -> None:
        self.clock = LoopClock()
        self.registry = RobotRegistry(
            round(CONFIG.robot_retention_s * NS_PER_S), CONFIG.max_robot_entries
        )
        self.mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)
        self.table = SessionTable(
            CONFIG, self.registry, self.mailbox, ConnectionTotals(), lambda: None
        )
        self.launcher = FakeLauncher(
            self.clock,
            {"cls": FRAME_MS * NS_PER_MS},
            {"cls": calibrated_sizes},
            ready_delay_ms * NS_PER_MS,
        )
        self.supervisor = ModelSupervisor(
            "cls",
            CONFIG,
            self.clock,
            self.launcher,
            self.mailbox,
            self.table.sessions,
            random.Random(0),
        )
        # Every wake of a session's handler: the robot id, the requests written to workers by
        # then, and the commands queued for the session by then.
        self.wakes: list[tuple[str, int, int]] = []
        self._opened: dict[str, Session] = {}
        self._frames_sent = 0

    @property
    def dispatch(self) -> ModelDispatch:
        dispatch = self.supervisor.dispatch
        assert dispatch is not None
        return dispatch

    @property
    def status(self) -> tuple[WorkerState, int | None, int, int]:
        """Return the worker's state, pid, restarts and timeouts."""
        status = self.supervisor.status
        return status.state, status.pid, status.restarts, status.timeouts

    def link(self, index: int) -> FakeWorkerLink:
        return self.launcher.links[index]

    def requests(self) -> list[tuple[int, list[bytes]]]:
        """Return every request written so far, over every worker launched."""
        return [request for link in self.launcher.links for request in link.requests]

    def open(self, robot_id: str) -> Session:
        hello = Hello(
            protocol_version=PROTOCOL_VERSION,
            robot_id=robot_id,
            token=None,
            models=("cls",),
            stale_ms={"cls": 500.0},
            watchdog_ms={"cls": 1_000.0},
            frame_rate_hz=30.0,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )

        def wake() -> None:
            self.wakes.append(
                (robot_id, len(self.requests()), len(self._opened[robot_id].downlink))
            )

        outcome = self.table.hello(encode_hello(hello), wake, self.clock.now_ns())
        assert isinstance(outcome, Opened)
        self._opened[robot_id] = outcome.session
        return outcome.session

    def send(self, session: Session, jpeg_start: bytes = JPEG_START) -> bytes:
        """Deliver the session's next frame as its reader does, and return the frame's JPEG.

        Every JPEG has its own length, which names the frame in a request and, as the class id
        of the fake runtime, in its command.
        """
        self._frames_sent += 1
        jpeg = jpeg_start + bytes(self._frames_sent)
        now_ns = self.clock.now_ns()
        header = FrameHeader(session.number, (session.last_frame_number or 0) + 1, now_ns, now_ns)
        assert self.table.ingest(session, encode_frame(header, jpeg), now_ns, None)
        self.supervisor.poke()
        return jpeg

    def commands(self, session: Session) -> list[tuple[int, Command]]:
        """Pop every command queued for the session: its frame number and the command."""
        popped: list[tuple[int, Command]] = []
        while (queued := session.downlink.pop(self.clock.now_ns())) is not None:
            header, command = decode_command_message(queued.data)
            popped.append((header.frame_number, command))
        return popped

    def counters(self, robot_id: str) -> ModelCounters:
        return self.registry.get(robot_id).models["cls"]

    def identity_breaks(self) -> list[str]:
        """Return the robots whose cls counters break a conservation identity of DESIGN.md 10.3."""
        frames = [
            f"{record.robot_id}: frames"
            for record in self.registry.records()
            for counters in [record.models["cls"]]
            if counters.frames_received
            != counters.frames_served
            + counters.frames_overwritten
            + counters.frames_skipped
            + counters.frames_dropped_disconnect
            + self.mailbox.pending_count(record.robot_id, "cls")
        ]
        commands = [
            f"{record.robot_id}: commands"
            for record in self.registry.records()
            for counters in [record.models["cls"]]
            if counters.frames_served
            != counters.commands_sent
            + counters.commands_queued
            + counters.commands_dropped_downlink
            + counters.results_dropped_disconnect
            + counters.frames_failed
            + counters.in_flight
        ]
        return frames + commands

    async def until_ms(self, time_ms: int) -> None:
        """Let virtual time pass until time_ms, and everything due before it happen."""
        await sleep_until(self.clock, time_ms * NS_PER_MS)


async def started(world: World) -> asyncio.Task[None]:
    """Start the supervisor and return its task once the first WorkerReady has arrived."""
    task = asyncio.create_task(world.supervisor.run())
    # Completed by the supervisor at its first WorkerReady.
    await world.supervisor.ready.wait()
    return task


@contextlib.asynccontextmanager
async def running(world: World) -> AsyncIterator[None]:
    """Run the supervisor for the block, from its first WorkerReady until a stop."""
    async with asyncio.TaskGroup() as group:
        group.create_task(world.supervisor.run())
        # Completed by the supervisor at its first WorkerReady.
        await world.supervisor.ready.wait()
        try:
            yield
        finally:
            world.supervisor.request_stop()


async def test_the_worker_is_idle_until_a_frame_arrives_and_then_gets_exactly_one_request(
    next_calls: list[int],
) -> None:
    world = World()
    async with running(world):
        session = world.open("a")
        await world.until_ms(1_000)
        requests_while_idle = world.requests()
        jpeg = world.send(session)
        await world.until_ms(1_000 + FRAME_MS + 5)

    assert requests_while_idle == []
    assert world.requests() == [(1, [jpeg])]
    # One call when the worker became ready, one for the frame, one after its result. Nothing
    # polls in between.
    assert next_calls == [0, 1_000, 1_000 + FRAME_MS]
    assert world.identity_breaks() == []


async def test_frames_during_a_batch_cause_no_request_and_the_next_batch_takes_the_newest(
    next_calls: list[int],
) -> None:
    world = World()
    async with running(world):
        first, second = world.open("a"), world.open("b")
        dispatched = world.send(first)
        await world.until_ms(3)
        world.send(first)
        newest_of_first = world.send(first)
        only_of_second = world.send(second)
        await world.until_ms(FRAME_MS - 1)
        requests_during_the_batch = world.requests()
        calls_during_the_batch = list(next_calls)
        await world.until_ms(FRAME_MS + 1)

    assert requests_during_the_batch == [(1, [dispatched])]
    assert calls_during_the_batch == [0, 0]
    # The robot never served goes first, and each robot's frame is its newest.
    assert world.requests()[1] == (2, [only_of_second, newest_of_first])
    assert world.counters("a").frames_overwritten == 1
    assert world.identity_breaks() == []


async def test_a_pacing_hold_dispatches_at_its_release_with_the_frames_newest_then(
    next_calls: list[int],
) -> None:
    world = World()
    async with running(world):
        # Each batch reports 30 ms of CPU time, which a share of half a core holds for 60 ms
        # from its dispatch.
        world.link(0).cpu_ns = 30 * NS_PER_MS
        world.dispatch.scheduler.set_pacing_share(0.5)
        session = world.open("a")
        first = world.send(session)
        await world.until_ms(20)
        world.send(session)
        await world.until_ms(40)
        newest_at_the_release = world.send(session)
        await world.until_ms(59)
        requests_during_the_hold = world.requests()
        await world.until_ms(125)

    assert requests_during_the_hold == [(1, [first])]
    assert world.requests() == [(1, [first]), (2, [newest_at_the_release])]
    # After the result at 10 ms the loop waits for the release at 60 ms, and the frames at 20
    # and 40 ms cause no call. The second batch is held from 70 ms to 120 ms in the same way.
    assert next_calls == [0, 0, 10, 60, 70, 120]
    assert world.dispatch.scheduler.stats.paced_ns == 50 * NS_PER_MS
    assert world.identity_breaks() == []


async def test_a_command_is_queued_and_its_session_woken_after_the_next_request_is_written() -> (
    None
):
    world = World()
    async with running(world):
        session = world.open("a")
        first = world.send(session)
        await world.until_ms(3)
        world.send(session)
        await world.until_ms(FRAME_MS + 1)
        wakes = list(world.wakes)
        commands = world.commands(session)

    # At the wake the second request is written and the first frame's command is queued.
    assert wakes == [("a", 2, 1)]
    assert commands == [(1, ClsCommand(class_id=len(first), confidence=1.0))]
    assert world.identity_breaks() == []


async def test_a_session_is_woken_for_its_command_when_no_batch_follows() -> None:
    world = World()
    async with running(world):
        session = world.open("a")
        world.send(session)
        await world.until_ms(FRAME_MS + 1)

    assert world.wakes == [("a", 1, 1)]
    assert world.counters("a").commands_queued == 1


async def test_a_frame_that_does_not_decode_fails_alone() -> None:
    world = World()
    async with running(world):
        good, bad = world.open("a"), world.open("b")
        world.send(good)
        world.send(bad, BROKEN_JPEG_START)
        await world.until_ms(2 * FRAME_MS + 1)

    assert (world.counters("a").commands_queued, world.counters("b").frames_failed) == (1, 1)
    assert world.dispatch.errors == {"decode": 1, "inference": 0}
    assert world.status == (WorkerState.READY, None, 0, 0)
    assert world.identity_breaks() == []


async def test_a_launch_that_fails_on_the_first_start_raises() -> None:
    world = World()
    world.launcher.fail_starts["cls"] = 1

    with pytest.raises(WorkerStartError):
        await world.supervisor.run()

    assert (world.supervisor.ready.is_set(), world.supervisor.dispatch) == (False, None)
    assert world.launcher.links == []


@pytest.mark.parametrize(
    ("spoil", "killed", "reason"),
    [
        pytest.param(FakeWorkerLink.crash, False, "link is lost", id="died-in-warm-up"),
        pytest.param(FakeWorkerLink.hang, True, "no WorkerReady within", id="warm-up-timeout"),
    ],
)
async def test_a_first_warm_up_that_fails_raises_and_stops_the_worker(
    spoil: Callable[[FakeWorkerLink], None], killed: bool, reason: str
) -> None:
    world = World(ready_delay_ms=1_000)
    task = asyncio.create_task(world.supervisor.run())
    await world.until_ms(1)
    spoil(world.link(0))

    with pytest.raises(WorkerStartError, match=reason):
        await task

    assert (world.link(0).killed, world.link(0).stopped) == (killed, True)
    # A warm-up timeout is not an inference timeout.
    assert world.status == (WorkerState.WARMING, None, 0, 0)
    assert world.supervisor.ready.is_set() is False


async def test_a_first_calibration_of_other_sizes_than_the_configured_ones_raises() -> None:
    world = World(calibrated_sizes=(1, 2))

    with pytest.raises(WorkerStartError, match="not the calibrated sizes"):
        await world.supervisor.run()

    assert (world.link(0).killed, world.link(0).stopped) == (True, True)
    assert (world.supervisor.ready.is_set(), world.supervisor.dispatch) == (False, None)


async def test_the_pid_shows_while_the_worker_warms_up_and_is_cleared_after_the_stop() -> None:
    world = World(ready_delay_ms=1_000)
    task = asyncio.create_task(world.supervisor.run())
    await world.until_ms(500)
    while_warming = world.status
    await world.until_ms(1_001)
    while_ready = world.status
    world.supervisor.request_stop()
    await task

    pid = world.link(0).pid
    assert while_warming == (WorkerState.WARMING, pid, 0, 0)
    assert while_ready == (WorkerState.READY, pid, 0, 0)
    assert world.supervisor.status.pid is None
    assert world.link(0).stopped is True


async def test_a_crash_during_a_batch_fails_its_frames_and_restarts_with_the_same_scheduler() -> (
    None
):
    world = World()
    async with running(world):
        dispatch = world.dispatch
        first, second = world.open("a"), world.open("b")
        world.send(first)
        world.send(second)
        await world.until_ms(5)
        world.link(0).crash()
        await world.until_ms(6)
        after_the_crash = world.status
        failed = (world.counters("a").frames_failed, world.counters("b").frames_failed)
        breaks_after_the_crash = world.identity_breaks()
        await world.until_ms(5 + RESTART_MIN_MS // 2 - 1)
        links_during_the_pause = len(world.launcher.links)
        await world.until_ms(5 + RESTART_MIN_MS + 1)
        after_the_restart = world.status
        jpeg = world.send(first)
        await world.until_ms(5 + RESTART_MIN_MS + FRAME_MS + 2)

    assert failed == (1, 1)
    assert breaks_after_the_crash == []
    assert after_the_crash == (WorkerState.RESTARTING, None, 0, 0)
    assert links_during_the_pause == 1
    assert after_the_restart == (WorkerState.READY, world.link(1).pid, 1, 0)
    # The restart is launched with the cap fixed at the first start, and the first
    # calibration, the scheduler and its batch ids live on.
    assert (world.link(0).batch_cap, world.link(1).batch_cap) == (None, BATCH_CAP)
    assert world.supervisor.dispatch is dispatch
    assert dispatch.calibration.pid == world.link(0).pid
    assert world.link(1).requests == [(2, [jpeg])]
    assert (world.link(0).killed, world.link(0).stopped) == (False, True)
    assert world.identity_breaks() == []


async def test_a_crash_while_idle_is_noticed_without_a_frame_taken(next_calls: list[int]) -> None:
    world = World()
    async with running(world):
        session = world.open("a")
        await world.until_ms(5)
        # The frame arrives in the same step as the crash, before the loop runs again.
        world.link(0).crash()
        jpeg = world.send(session)
        await world.until_ms(6)
        after_the_crash = world.status
        calls_after_the_crash = list(next_calls)
        counters = world.counters("a")
        taken = (counters.frames_served, counters.frames_failed)
        await world.until_ms(5 + RESTART_MIN_MS + FRAME_MS + 2)

    assert after_the_crash == (WorkerState.RESTARTING, None, 0, 0)
    assert calls_after_the_crash == [0]
    assert taken == (0, 0)
    assert world.link(0).requests == []
    # The frame waited in the mailbox and went to the restarted worker.
    assert world.link(1).requests == [(1, [jpeg])]
    assert world.identity_breaks() == []


async def test_a_protocol_break_while_idle_kills_the_worker_at_once_and_restarts_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.ERROR, logger="cloud.supervisor")
    world = World()
    async with running(world):
        await world.until_ms(5)
        world.link(0).break_protocol()
        await world.until_ms(6)
        after_the_break = world.status
        killed_before_the_stop_bound = world.link(0).killed
        await world.until_ms(5 + RESTART_MIN_MS + 1)

    assert after_the_break == (WorkerState.RESTARTING, None, 0, 0)
    assert killed_before_the_stop_bound is True
    assert world.status == (WorkerState.READY, None, 1, 0)
    assert [record.getMessage().split(":")[0] for record in caplog.records] == [
        "worker cls failed (protocol)"
    ]


async def test_a_worker_that_hangs_times_out_is_killed_and_restarted() -> None:
    world = World()
    async with running(world):
        session = world.open("a")
        world.send(session)
        await world.until_ms(5)
        world.link(0).hang()
        await world.until_ms(INFERENCE_TIMEOUT_MS - 1)
        before_the_timeout = world.status
        await world.until_ms(INFERENCE_TIMEOUT_MS + 1)
        after_the_timeout = world.status
        failed = world.counters("a").frames_failed
        await world.until_ms(INFERENCE_TIMEOUT_MS + RESTART_MIN_MS + 1)

    assert before_the_timeout == (WorkerState.READY, world.link(0).pid, 0, 0)
    assert after_the_timeout == (WorkerState.RESTARTING, None, 0, 1)
    assert failed == 1
    assert (world.link(0).killed, world.link(0).stopped) == (True, True)
    assert world.status == (WorkerState.READY, None, 1, 1)
    assert world.identity_breaks() == []


async def test_a_warm_up_timeout_on_a_restart_counts_no_timeout_and_is_retried() -> None:
    world = World(ready_delay_ms=1_000)
    async with running(world):
        await world.until_ms(1_500)
        world.link(0).crash()
        # The restart is launched within the first pause and would be ready a second later.
        await world.until_ms(1_500 + RESTART_MIN_MS + 1)
        world.link(1).hang()
        launched_ms = 1_500 + RESTART_MIN_MS
        await world.until_ms(launched_ms + WARMUP_TIMEOUT_MS + 1)
        after_the_timeout = world.status
        await world.until_ms(launched_ms + WARMUP_TIMEOUT_MS + 2 * RESTART_MIN_MS + 1_000 + 1)
        after_the_retry = world.status

    assert after_the_timeout == (WorkerState.FAILED, None, 1, 0)
    assert (world.link(1).killed, world.link(1).stopped) == (True, True)
    assert after_the_retry == (WorkerState.READY, world.link(2).pid, 2, 0)
    assert [link.batch_cap for link in world.launcher.links] == [None, BATCH_CAP, BATCH_CAP]


async def test_a_result_for_another_batch_fails_the_batch_and_restarts_the_worker() -> None:
    world = World()
    async with running(world):
        session = world.open("a")
        world.link(0).garble()
        world.send(session)
        await world.until_ms(FRAME_MS + 1)
        after_the_result = world.status
        counters = world.counters("a")
        booked = (counters.frames_failed, counters.commands_queued)
        await world.until_ms(FRAME_MS + RESTART_MIN_MS + 1)

    assert after_the_result == (WorkerState.RESTARTING, None, 0, 0)
    assert booked == (1, 0)
    assert (world.link(0).killed, world.link(0).stopped) == (True, True)
    assert world.status == (WorkerState.READY, None, 1, 0)
    assert world.wakes == []
    assert world.identity_breaks() == []


async def test_a_restart_that_fails_shows_failed_and_is_retried_after_the_backoff() -> None:
    world = World()
    async with running(world):
        world.launcher.fail_starts["cls"] = 1
        await world.until_ms(5)
        world.link(0).crash()
        await world.until_ms(6)
        after_the_crash = world.status
        await world.until_ms(5 + RESTART_MIN_MS + 1)
        after_the_failed_restart = world.status
        links_after_the_failed_restart = len(world.launcher.links)
        # The second pause is at most twice the minimum.
        await world.until_ms(5 + 3 * RESTART_MIN_MS + 1)
        after_the_retry = world.status

    assert after_the_crash == (WorkerState.RESTARTING, None, 0, 0)
    assert after_the_failed_restart == (WorkerState.FAILED, None, 1, 0)
    assert links_after_the_failed_restart == 1
    assert after_the_retry == (WorkerState.READY, world.link(1).pid, 2, 0)


async def test_request_stop_lets_the_batch_in_flight_finish_and_stops_the_worker() -> None:
    world = World()
    task = await started(world)
    session = world.open("a")
    first = world.send(session)
    await world.until_ms(3)
    world.send(session)
    world.supervisor.request_stop()
    await task

    # run() returns when the batch settles, with its command queued and its session woken, and
    # dispatches nothing more.
    assert world.clock.now_ns() == FRAME_MS * NS_PER_MS
    assert world.requests() == [(1, [first])]
    assert world.wakes == [("a", 1, 1)]
    assert world.mailbox.pending_count("a", "cls") == 1
    assert world.dispatch.scheduler.in_flight is None
    assert (world.link(0).killed, world.link(0).stopped) == (False, True)
    assert world.supervisor.status.pid is None
    assert world.identity_breaks() == []


async def test_request_stop_ends_a_restart_pause_at_once() -> None:
    world = World()
    task = await started(world)
    await world.until_ms(5)
    world.link(0).crash()
    await world.until_ms(6)
    world.supervisor.request_stop()
    await task

    assert world.clock.now_ns() == 6 * NS_PER_MS
    assert world.status == (WorkerState.RESTARTING, None, 0, 0)
    assert len(world.launcher.links) == 1


async def test_a_cancellation_fails_the_batch_in_flight_stops_the_worker_and_propagates() -> None:
    world = World()
    task = await started(world)
    session = world.open("a")
    world.send(session)
    await world.until_ms(5)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert world.counters("a").frames_failed == 1
    assert world.dispatch.scheduler.in_flight is None
    assert world.link(0).stopped is True
    assert world.supervisor.status.pid is None
    assert world.identity_breaks() == []


async def test_a_cancellation_inside_the_stop_kills_the_worker_and_propagates() -> None:
    world = World()
    task = await started(world)
    world.link(0).hold_exit()
    await world.until_ms(5)
    world.supervisor.request_stop()
    await world.until_ms(6)
    inside_the_stop = (world.link(0).stopped, world.link(0).killed, world.supervisor.status.pid)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    # The pid shows until the stop has ended, and the kill does not wait for the stop bound.
    assert inside_the_stop == (True, False, world.link(0).pid)
    assert world.link(0).killed is True
    assert world.supervisor.status.pid is None
    assert world.clock.now_ns() == 6 * NS_PER_MS


async def test_a_worker_that_does_not_exit_is_killed_after_the_stop_bound() -> None:
    world = World()
    task = await started(world)
    world.link(0).hold_exit()
    world.supervisor.request_stop()
    await world.until_ms(STOP_TIMEOUT_MS - 1)
    killed_within_the_bound = world.link(0).killed
    await task

    assert (killed_within_the_bound, world.link(0).killed) == (False, True)
    assert world.clock.now_ns() == 2 * STOP_TIMEOUT_NS
    assert world.supervisor.status.pid is None


async def test_both_conservation_identities_hold_after_every_step() -> None:
    world = World()
    async with running(world):
        first, second, third = world.open("a"), world.open("b"), world.open("c")
        for session in (first, second, third):
            world.send(session)
        assert world.identity_breaks() == []
        await world.until_ms(3)
        # While the batch is at the worker: a overwrites its slot, b leaves, c is replaced.
        world.send(first)
        world.send(first)
        world.table.end(second, CloseReason.ABNORMAL, world.clock.now_ns())
        replacement = world.open("c")
        assert world.identity_breaks() == []
        # The batch of three frames takes three service times.
        await world.until_ms(3 * FRAME_MS + 1)
        assert world.identity_breaks() == []
        # The second batch, the newest frame of a, gets a result for another batch.
        world.link(0).garble()
        world.send(replacement)
        world.send(first, BROKEN_JPEG_START)
        assert world.identity_breaks() == []
        await world.until_ms(4 * FRAME_MS + 1)
        assert world.identity_breaks() == []
        # The restarted worker serves the two frames that waited, one of which does not decode.
        restarted_ms = 4 * FRAME_MS + RESTART_MIN_MS
        await world.until_ms(restarted_ms + 2 * FRAME_MS + 1)
        assert world.identity_breaks() == []
        world.send(replacement)
        await world.until_ms(restarted_ms + 2 * FRAME_MS + 4)
        world.link(1).crash()
        assert world.identity_breaks() == []
        await world.until_ms(restarted_ms + 2 * FRAME_MS + 5)
        assert world.identity_breaks() == []
        restarted_again_ms = restarted_ms + 2 * FRAME_MS + 4 + RESTART_MIN_MS
        await world.until_ms(restarted_again_ms + 1)
        world.send(replacement)
        assert world.identity_breaks() == []
        await world.until_ms(restarted_again_ms + FRAME_MS + 2)
        assert world.identity_breaks() == []

    assert {
        robot_id: (
            counters.frames_received,
            counters.frames_served,
            counters.commands_queued,
            counters.results_dropped_disconnect,
            counters.frames_failed,
        )
        for robot_id in "abc"
        for counters in [world.counters(robot_id)]
    } == {"a": (4, 3, 1, 0, 2), "b": (1, 1, 0, 1, 0), "c": (4, 4, 2, 1, 1)}
    assert world.status == (WorkerState.READY, None, 2, 0)


async def send_frame(world: World, robot_id: str, jpeg_start: bytes) -> None:
    world.send(world.table.sessions[robot_id], jpeg_start)


async def replace_session(world: World, robot_id: str) -> None:
    world.open(robot_id)


async def advance(world: World, time_ms: int) -> None:
    await world.until_ms(world.clock.now_ns() // NS_PER_MS + time_ms)


async def spoil_newest_worker(world: World, spoil: Callable[[FakeWorkerLink], None]) -> None:
    spoil(world.link(-1))


# Each step acts on the world as one event of a run does. The times span a batch, a restart
# pause, the inference timeout and the warm-up timeout.
STEPS: list[Callable[[World], Awaitable[None]]] = [
    *(partial(send_frame, robot_id=robot_id, jpeg_start=JPEG_START) for robot_id in "abc"),
    partial(send_frame, robot_id="a", jpeg_start=BROKEN_JPEG_START),
    *(partial(replace_session, robot_id=robot_id) for robot_id in "ab"),
    *(
        partial(advance, time_ms=time_ms)
        for time_ms in (1, FRAME_MS, RESTART_MIN_MS, INFERENCE_TIMEOUT_MS, WARMUP_TIMEOUT_MS)
    ),
    partial(spoil_newest_worker, spoil=FakeWorkerLink.crash),
    partial(spoil_newest_worker, spoil=FakeWorkerLink.hang),
    partial(spoil_newest_worker, spoil=FakeWorkerLink.garble),
]


@given(st.lists(st.sampled_from(STEPS), max_size=40))
def test_the_identities_hold_after_every_step_of_any_script_and_a_stop_always_ends_the_run(
    script: list[Callable[[World], Awaitable[None]]],
) -> None:
    async def play() -> list[list[str]]:
        world = World()
        breaks: list[list[str]] = []
        async with running(world):
            for robot_id in "abc":
                world.open(robot_id)
            for step in script:
                await step(world)
                breaks.append(world.identity_breaks())
        return [*breaks, world.identity_breaks()]

    assert run_virtual(play()) == [[]] * (len(script) + 1)
