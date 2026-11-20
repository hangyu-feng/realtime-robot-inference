"""Tests for the /metrics field builders, the render, the stats reader and /healthz."""

import asyncio
import itertools
import json
import os
import random
import sys
import threading
from collections.abc import Callable, Mapping
from dataclasses import replace

import pytest

from cloud.config import ModelConfig, ServerConfig, config_echo
from cloud.counters import CloseReason, ConnectionTotals, WorkerState, WorkerStatus
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Frame, Mailbox
from cloud.metrics import (
    STATS_READ_TIMEOUT_S,
    MetricsSources,
    SystemStats,
    cgroup_entry,
    connections_entry,
    health_reply,
    loop_entry,
    model_entry,
    processes_entry,
    read_system_stats,
    render_metrics,
    robot_entry,
    stages_entry,
    worker_entry,
)
from cloud.registry import ModelCounters, RobotRecord, RobotRegistry
from cloud.scheduler import Batch, Dispatch
from cloud.sessions import Opened, Session, SessionTable
from cloud.supervisor import ModelSupervisor
from cloud.worker_link import WorkerLink
from common.cgroup import CgroupStats, Limits
from common.clock import FakeClock
from common.commands import MODEL_KEYS, ClsCommand, encode_command
from common.histogram import Histogram
from common.procstat import ProcessStats
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    PROTOCOL_VERSION,
    STAGE_NAMES,
    FrameHeader,
    Hello,
    ModelTelemetry,
    Telemetry,
    encode_frame,
    encode_hello,
    encode_telemetry,
    telemetry_to_dict,
)
from common.worker_protocol import BatchResult, FrameResult, FrameStatus, WorkerReady
from tests.support.virtual_time import run_virtual

NS_PER_MS = 1_000_000
NS_PER_S = 1_000_000_000
# The thresholds are stated here because the expected margins below are computed from them.
CONFIG = ServerConfig(
    models={
        "cls": ModelConfig(stale_ms=150.0, watchdog_ms=300.0),
        "depth": ModelConfig(stale_ms=500.0, watchdog_ms=1_000.0, paced=True),
    }
)
# Room for a robot table as large as the configuration allows.
MANY_ROBOTS = replace(CONFIG, max_connections=CONFIG.max_robot_entries)
JPEG = b"\xff\xd8\xff\xe0jpeg"
WALL_MS = 1_700_000_000_000
GATEWAY_PID = 10
# Every cls size fits its budget, so its cap is 8. No depth size fits, so its cap is 1, its
# margin is negative, and the batch time alone is above its threshold.
READY = {
    "cls": WorkerReady(
        model="cls",
        pid=11,
        model_file="cls.onnx",
        model_sha256="ab" * 32,
        threads=1,
        t_b_ns={1: 2 * NS_PER_MS, 2: 3 * NS_PER_MS, 4: 5 * NS_PER_MS, 8: 9 * NS_PER_MS},
        t_b_cpu_ns={1: 2_500_000, 2: 3_500_000, 4: 5_500_000, 8: 9_500_000},
    ),
    "depth": WorkerReady(
        model="depth",
        pid=12,
        model_file=None,
        model_sha256=None,
        threads=1,
        t_b_ns={
            1: 600 * NS_PER_MS,
            2: 1_200 * NS_PER_MS,
            4: 2_400 * NS_PER_MS,
            8: 4_800 * NS_PER_MS,
        },
        t_b_cpu_ns={
            1: 590 * NS_PER_MS,
            2: 1_180 * NS_PER_MS,
            4: 2_360 * NS_PER_MS,
            8: 4_720 * NS_PER_MS,
        },
    ),
}
LIMITS = Limits(
    version=2,
    cpu_limit_cores=2.0,
    cpu_period_us=100_000,
    memory_limit_bytes=4 * 2**30,
    memory_swap_limit_bytes=None,
    cpuset="0-3",
)
CGROUP = CgroupStats(
    cpu_usage_us=900,
    cpu_user_us=600,
    cpu_system_us=300,
    nr_periods=50,
    nr_throttled=2,
    throttled_us=1_500,
    memory_current_bytes=300 * 2**20,
    memory_peak_bytes=310 * 2**20,
    memory_anon_bytes=200 * 2**20,
    memory_file_bytes=90 * 2**20,
    memory_sock_bytes=None,
    oom_kills=0,
)
MODEL_TELEMETRY = ModelTelemetry(
    commands_received=3,
    commands_applied=2,
    commands_stale=1,
    commands_out_of_order=0,
    freshness_ms=Histogram(),
    command_gap_ms=Histogram(),
    command_age_peak_ms=Histogram(),
    deliver_ms=Histogram(),
    watchdog_trips=0,
    watchdog_expired_ms=40,
    watchdog_state="ok",
)
TELEMETRY = Telemetry(
    ts_mono_ns=123_456_789,
    frames_captured=30,
    frames_sent=29,
    frames_skipped_send=1,
    fire_lag_ms=Histogram(),
    reconnects=0,
    models={"cls": MODEL_TELEMETRY, "depth": MODEL_TELEMETRY},
)
EMPTY_HISTOGRAM = Histogram().to_dict()
# The settings of either model in the config echo that workers.<m> repeats.
WORKER_SETTINGS: dict[str, object] = {"threads": 1, "batch_sizes": [1, 2, 4, 8]}


def process_stats(pid: int) -> ProcessStats:
    return ProcessStats(
        pid=pid, cpu_user_ms=40, cpu_system_ms=20, rss_bytes=50 * 2**20, threads=3, fds=9, nice=0
    )


def live_process(pid: int) -> dict[str, object]:
    """Return the processes.<p> entry of the process that process_stats(pid) describes."""
    return {
        "pid": pid,
        "cpu_user_ms": 40,
        "cpu_system_ms": 20,
        "rss_bytes": 50 * 2**20,
        "threads": 3,
        "fds": 9,
        "nice": 0,
    }


async def no_worker(model: str, batch_cap: int | None, on_lost: Callable[[], None]) -> WorkerLink:
    raise AssertionError("these tests set a supervisor's state by hand and never run it")


class World:
    """The state a snapshot reads, over a real registry, mailbox, session table and totals.

    The supervisors are real and never run: a test sets what a snapshot reads from them, the
    status and the dispatch, by hand. Frames and results go through the real core.
    """

    def __init__(self, config: ServerConfig = CONFIG) -> None:
        self.config = config
        self.echo = config_echo(config, version="test", protocol_version=PROTOCOL_VERSION)
        self.clock = FakeClock()
        self.registry = RobotRegistry(
            round(config.robot_retention_s * NS_PER_S), config.max_robot_entries
        )
        self.mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        self.totals = ConnectionTotals()
        self.table = SessionTable(config, self.registry, self.mailbox, self.totals, lambda: None)
        self.supervisors = {
            model: ModelSupervisor(
                model,
                config,
                self.clock,
                no_worker,
                self.mailbox,
                self.table.sessions,
                random.Random(0),
            )
            for model in MODEL_KEYS
        }
        self.shares: dict[str, float | None] = dict.fromkeys(MODEL_KEYS)
        self.accepting_since_ns: int | None = None
        self.lag_ms = Histogram()
        self.pids_read: list[dict[str, int | None]] = []
        self.sources = MetricsSources(
            config_echo=self.echo,
            limits=LIMITS,
            clock=self.clock,
            wall_clock_ms=lambda: WALL_MS,
            gateway_pid=GATEWAY_PID,
            registry=self.registry,
            mailbox=self.mailbox,
            totals=self.totals,
            supervisors=self.supervisors,
            pacing_shares=lambda: self.shares,
            accepting_since_ns=lambda: self.accepting_since_ns,
            loop_lag_ms=self.lag_ms,
            read_system=self.read_system,
        )

    async def read_system(self, pids: Mapping[str, int | None]) -> SystemStats:
        self.pids_read.append(dict(pids))
        return SystemStats(
            CGROUP,
            {name: None if pid is None else process_stats(pid) for name, pid in pids.items()},
        )

    def ready(self, model: str) -> ModelDispatch:
        """Put the model's supervisor in the state it has after its first calibration."""
        supervisor = self.supervisors[model]
        dispatch = ModelDispatch(
            model, self.config, READY[model], self.mailbox, self.clock.now_ns()
        )
        supervisor.dispatch = dispatch
        supervisor.status.state = WorkerState.READY
        supervisor.status.pid = READY[model].pid
        return dispatch

    def open(self, robot_id: str, models: tuple[str, ...] = MODEL_KEYS) -> Session:
        hello = Hello(
            protocol_version=PROTOCOL_VERSION,
            robot_id=robot_id,
            token=None,
            models=models,
            stale_ms=dict.fromkeys(models, 150.0),
            watchdog_ms=dict.fromkeys(models, 300.0),
            frame_rate_hz=30.0,
            frame_width_px=FRAME_WIDTH_PX,
            frame_height_px=FRAME_HEIGHT_PX,
        )
        outcome = self.table.hello(encode_hello(hello), lambda: None, self.clock.now_ns())
        assert isinstance(outcome, Opened)
        return outcome.session

    def send(self, session: Session, frame_number: int) -> None:
        """Deliver a frame captured 3 ms and sent 1 ms before now."""
        now_ns = self.clock.now_ns()
        header = FrameHeader(
            session.number, frame_number, now_ns - 3 * NS_PER_MS, now_ns - NS_PER_MS
        )
        assert self.table.ingest(session, encode_frame(header, JPEG), now_ns, None)

    def dispatch(self, model: str) -> Batch:
        dispatch = self.supervisors[model].dispatch
        assert dispatch is not None
        decision = dispatch.scheduler.next(self.clock.now_ns())
        assert isinstance(decision, Dispatch)
        return decision.batch

    def complete(self, model: str, batch: Batch) -> None:
        """Book a result for the batch, which the worker had for 3 of the 5 ms since dispatch."""
        dispatch = self.supervisors[model].dispatch
        assert dispatch is not None
        self.clock.advance_ns(5 * NS_PER_MS)
        received_ns = batch.dispatched_ns + NS_PER_MS
        result = BatchResult(
            batch_id=batch.batch_id,
            received_ns=received_ns,
            sent_ns=received_ns + 3 * NS_PER_MS,
            inference_ns=NS_PER_MS,
            cpu_ns=3 * NS_PER_MS,
            frames=tuple(
                FrameResult(
                    FrameStatus.OK,
                    encode_command(ClsCommand(class_id=1, confidence=0.5)),
                    200_000,
                    200_000,
                    200_000,
                )
                for _ in batch.items
            ),
        )
        dispatch.complete(batch, result, self.clock.now_ns(), self.table.sessions)

    def snapshot(self) -> dict[str, object]:
        document: dict[str, object] = json.loads(run_virtual(render_metrics(self.sources)))
        return document


def warming_world() -> World:
    """Return a gateway whose workers have not sent their first calibration."""
    world = World()
    world.clock.advance_ns(NS_PER_S)
    world.supervisors["cls"].status.pid = READY["cls"].pid
    return world


def serving_world() -> World:
    """Return a gateway with a batch in flight for each model, commands queued and three robots.

    Robots a and b are connected. Each got a cls command, of which a's was written and b's is
    queued; each has a second frame at the cls worker; a's second frame is at the depth worker
    and b's is pending for depth. Robot a has reported telemetry. Robot c has disconnected.
    """
    world = World()
    world.clock.advance_ns(NS_PER_S)
    world.ready("cls")
    world.ready("depth")
    world.accepting_since_ns = world.clock.now_ns()
    world.shares["depth"] = 0.75
    world.clock.advance_ns(NS_PER_S)
    a = world.open("a")
    b = world.open("b")
    world.clock.advance_ns(10 * NS_PER_MS)
    world.send(a, 1)
    world.send(b, 1)
    world.complete("cls", world.dispatch("cls"))
    assert a.downlink.pop(world.clock.now_ns()) is not None
    world.clock.advance_ns(30 * NS_PER_MS)
    world.send(a, 2)
    world.send(b, 2)
    world.dispatch("cls")
    world.dispatch("depth")
    world.table.ingest(a, encode_telemetry(TELEMETRY), world.clock.now_ns(), 2 * NS_PER_MS)
    world.table.end(world.open("c"), CloseReason.CLIENT_CLOSE, world.clock.now_ns())
    world.clock.advance_ns(20 * NS_PER_MS)
    return world


def depth_down_world() -> World:
    """Return the serving gateway after its depth worker has crashed, then timed out and been
    killed: two restarts, one of them after a timeout.
    """
    world = serving_world()
    status = world.supervisors["depth"].status
    status.state = WorkerState.RESTARTING
    status.pid = None
    status.restarts = 2
    status.timeouts = 1
    return world


STALE_NS = {"cls": 150 * NS_PER_MS, "depth": 500 * NS_PER_MS}


def robot_record(
    *,
    disconnected_ns: int | None = None,
    subscribed: tuple[str, ...] = MODEL_KEYS,
    models: dict[str, ModelCounters] | None = None,
) -> RobotRecord:
    """Return a record of robot a, in session 3 since 2 s."""
    return RobotRecord(
        robot_id="a",
        session=3,
        connected_ns=2 * NS_PER_S,
        disconnected_ns=disconnected_ns,
        subscribed=subscribed,
        stale_ns={model: STALE_NS[model] for model in subscribed},
        watchdog_ns={model: 2 * STALE_NS[model] for model in subscribed},
        frame_period_ns=33_333_333,
        models={} if models is None else models,
        bytes_in=880_000,
        bytes_out=540,
        frames_in=22,
        frames_rejected=2,
        frames_accepted=20,
    )


def mailbox_holding_a_frame_of(record: RobotRecord) -> Mailbox:
    """Return a mailbox in which the record is open with one frame pending for each model.

    The record's counters are replaced afterwards by the ones a test states.
    """
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)
    for model in record.subscribed:
        record.models[model] = ModelCounters()
    mailbox.open(record)
    mailbox.put(record.robot_id, Frame(1, 0, 0, 0, 0, JPEG))
    return mailbox


def test_a_connected_robot_entry_has_every_server_side_field() -> None:
    record = robot_record()
    mailbox = mailbox_holding_a_frame_of(record)
    record.rtt_ns = 3 * NS_PER_MS
    record.models["cls"] = ModelCounters(
        frames_received=20,
        frames_served=12,
        frames_overwritten=5,
        frames_skipped=2,
        frames_dropped_disconnect=0,
        in_flight=1,
        commands_queued=2,
        commands_sent=6,
        commands_dropped_downlink=1,
        results_dropped_disconnect=0,
        frames_failed=2,
        last_served_seq=9,
        last_served_ns=4_900 * NS_PER_MS,
        last_command_ns=4_600 * NS_PER_MS,
        max_service_gap_ns=250 * NS_PER_MS,
    )
    # Depth has never served this robot: the open interval runs from the session's start.
    record.models["depth"] = ModelCounters(
        frames_received=20, frames_overwritten=19, last_command_ns=2 * NS_PER_S
    )

    entry = robot_entry(record, mailbox, 5 * NS_PER_S)

    assert entry == {
        "session": 3,
        "connected": True,
        "connected_s": 3.0,
        "subscribed": ["cls", "depth"],
        "stale_ms": {"cls": 150.0, "depth": 500.0},
        "watchdog_ms": {"cls": 300.0, "depth": 1_000.0},
        "rtt_ms": 3.0,
        "bytes_in": 880_000,
        "bytes_out": 540,
        "frames_in": 22,
        "frames_rejected": 2,
        "frames_accepted": 20,
        "models": {
            "cls": {
                "frames_received": 20,
                "frames_served": 12,
                "frames_overwritten": 5,
                "frames_skipped": 2,
                "frames_dropped_disconnect": 0,
                "frames_pending": 1,
                "in_flight": 1,
                "commands_queued": 2,
                "commands_sent": 6,
                "commands_dropped_downlink": 1,
                "results_dropped_disconnect": 0,
                "frames_failed": 2,
                "last_served_age_ms": 100.0,
                "max_service_gap_ms": 400.0,
            },
            "depth": {
                "frames_received": 20,
                "frames_served": 0,
                "frames_overwritten": 19,
                "frames_skipped": 0,
                "frames_dropped_disconnect": 0,
                "frames_pending": 1,
                "in_flight": 0,
                "commands_queued": 0,
                "commands_sent": 0,
                "commands_dropped_downlink": 0,
                "results_dropped_disconnect": 0,
                "frames_failed": 0,
                "last_served_age_ms": None,
                "max_service_gap_ms": 3_000.0,
            },
        },
    }


def test_a_disconnected_robot_entry_stops_its_session_at_the_disconnect() -> None:
    record = robot_record(
        disconnected_ns=4 * NS_PER_S,
        subscribed=("cls",),
        models={
            "cls": ModelCounters(
                frames_received=7,
                frames_served=5,
                frames_overwritten=1,
                frames_dropped_disconnect=1,
                commands_sent=4,
                results_dropped_disconnect=1,
                last_served_seq=5,
                last_served_ns=3_500 * NS_PER_MS,
                last_command_ns=3_400 * NS_PER_MS,
                max_service_gap_ns=200 * NS_PER_MS,
            )
        },
    )
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)

    entry = robot_entry(record, mailbox, 9 * NS_PER_S)

    assert entry == {
        "session": 3,
        "connected": False,
        "connected_s": 2.0,
        "subscribed": ["cls"],
        "stale_ms": {"cls": 150.0},
        "watchdog_ms": {"cls": 300.0},
        "rtt_ms": None,
        "bytes_in": 880_000,
        "bytes_out": 540,
        "frames_in": 22,
        "frames_rejected": 2,
        "frames_accepted": 20,
        "models": {
            "cls": {
                "frames_received": 7,
                "frames_served": 5,
                "frames_overwritten": 1,
                "frames_skipped": 0,
                "frames_dropped_disconnect": 1,
                "frames_pending": 0,
                "in_flight": 0,
                "commands_queued": 0,
                "commands_sent": 4,
                "commands_dropped_downlink": 0,
                "results_dropped_disconnect": 1,
                "frames_failed": 0,
                "last_served_age_ms": 5_500.0,
                "max_service_gap_ms": 600.0,
            }
        },
    }


def test_a_robot_back_with_fewer_models_lists_its_session_s_models_and_every_model_s_counters() -> (
    None
):
    # Counters are kept per robot id: depth's are from an earlier session, which subscribed to
    # both models.
    record = robot_record(
        subscribed=("cls",),
        models={
            "cls": ModelCounters(
                frames_received=3, frames_overwritten=3, last_command_ns=2 * NS_PER_S
            ),
            "depth": ModelCounters(
                frames_received=4,
                frames_served=4,
                commands_sent=4,
                last_served_seq=2,
                last_served_ns=NS_PER_S,
                max_service_gap_ns=90 * NS_PER_MS,
            ),
        },
    )
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)

    entry = robot_entry(record, mailbox, 5 * NS_PER_S)

    assert entry == {
        "session": 3,
        "connected": True,
        "connected_s": 3.0,
        "subscribed": ["cls"],
        "stale_ms": {"cls": 150.0},
        "watchdog_ms": {"cls": 300.0},
        "rtt_ms": None,
        "bytes_in": 880_000,
        "bytes_out": 540,
        "frames_in": 22,
        "frames_rejected": 2,
        "frames_accepted": 20,
        "models": {
            "cls": {
                "frames_received": 3,
                "frames_served": 0,
                "frames_overwritten": 3,
                "frames_skipped": 0,
                "frames_dropped_disconnect": 0,
                "frames_pending": 0,
                "in_flight": 0,
                "commands_queued": 0,
                "commands_sent": 0,
                "commands_dropped_downlink": 0,
                "results_dropped_disconnect": 0,
                "frames_failed": 0,
                "last_served_age_ms": None,
                "max_service_gap_ms": 3_000.0,
            },
            "depth": {
                "frames_received": 4,
                "frames_served": 4,
                "frames_overwritten": 0,
                "frames_skipped": 0,
                "frames_dropped_disconnect": 0,
                "frames_pending": 0,
                "in_flight": 0,
                "commands_queued": 0,
                "commands_sent": 4,
                "commands_dropped_downlink": 0,
                "results_dropped_disconnect": 0,
                "frames_failed": 0,
                "last_served_age_ms": 4_000.0,
                "max_service_gap_ms": 90.0,
            },
        },
    }


@pytest.mark.parametrize(
    ("disconnected_ns", "last_command_ns", "closed_gap_ns", "gap_ms"),
    [
        pytest.param(None, 8_900 * NS_PER_MS, 400 * NS_PER_MS, 400.0, id="closed-gap-is-longest"),
        pytest.param(None, 8_300 * NS_PER_MS, 100 * NS_PER_MS, 700.0, id="open-until-the-scrape"),
        pytest.param(None, 2 * NS_PER_S, 0, 7_000.0, id="no-command-yet-in-the-session"),
        pytest.param(
            3_500 * NS_PER_MS,
            3 * NS_PER_S,
            100 * NS_PER_MS,
            500.0,
            id="open-until-the-disconnect",
        ),
        pytest.param(
            3_500 * NS_PER_MS,
            3_400 * NS_PER_MS,
            250 * NS_PER_MS,
            250.0,
            id="closed-gap-is-longest-after-the-disconnect",
        ),
        pytest.param(None, None, 250 * NS_PER_MS, 250.0, id="not-subscribed-in-this-session"),
    ],
)
def test_the_longest_service_gap_covers_the_open_interval_to_the_scrape_or_the_disconnect(
    disconnected_ns: int | None, last_command_ns: int | None, closed_gap_ns: int, gap_ms: float
) -> None:
    counters = ModelCounters(last_command_ns=last_command_ns, max_service_gap_ns=closed_gap_ns)
    record = robot_record(disconnected_ns=disconnected_ns, models={"cls": counters})
    mailbox = Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames)

    entry = robot_entry(record, mailbox, 9 * NS_PER_S)

    assert entry["models"] == {
        "cls": {
            "frames_received": 0,
            "frames_served": 0,
            "frames_overwritten": 0,
            "frames_skipped": 0,
            "frames_dropped_disconnect": 0,
            "frames_pending": 0,
            "in_flight": 0,
            "commands_queued": 0,
            "commands_sent": 0,
            "commands_dropped_downlink": 0,
            "results_dropped_disconnect": 0,
            "frames_failed": 0,
            "last_served_age_ms": None,
            "max_service_gap_ms": gap_ms,
        }
    }


def test_a_sessions_last_interval_still_counts_after_the_robot_reconnects() -> None:
    world = World()
    world.ready("cls")
    first = world.open("a", ("cls",))
    world.send(first, 1)
    world.complete("cls", world.dispatch("cls"))
    assert first.downlink.pop(world.clock.now_ns()) is not None
    # No command for 2 s, and then the robot leaves: the disconnect ends that interval.
    world.clock.advance_ns(2 * NS_PER_S)
    world.table.end(first, CloseReason.CLIENT_CLOSE, world.clock.now_ns())
    world.clock.advance_ns(10 * NS_PER_MS)
    away = world.snapshot()["robots"]
    world.open("a", ("cls",))
    world.clock.advance_ns(50 * NS_PER_MS)
    back = world.snapshot()["robots"]

    assert isinstance(away, dict)
    assert isinstance(back, dict)
    assert [robots["a"]["models"]["cls"]["max_service_gap_ms"] for robots in (away, back)] == [
        2_000.0,
        2_000.0,
    ]


def test_a_disconnected_record_shows_no_pending_frame_of_a_newer_record_with_its_id() -> None:
    # The registry dropped the old record, and its id connected again as a new one.
    old = robot_record(disconnected_ns=4 * NS_PER_S, models={"cls": ModelCounters()})
    mailbox = mailbox_holding_a_frame_of(robot_record())

    entry = robot_entry(old, mailbox, 9 * NS_PER_S)

    assert mailbox.pending_count("a", "cls") == 1
    assert entry["models"] == {
        "cls": {
            "frames_received": 0,
            "frames_served": 0,
            "frames_overwritten": 0,
            "frames_skipped": 0,
            "frames_dropped_disconnect": 0,
            "frames_pending": 0,
            "in_flight": 0,
            "commands_queued": 0,
            "commands_sent": 0,
            "commands_dropped_downlink": 0,
            "results_dropped_disconnect": 0,
            "frames_failed": 0,
            "last_served_age_ms": None,
            "max_service_gap_ms": 0.0,
        }
    }


def test_a_robot_entry_has_no_telemetry_until_a_report_exists() -> None:
    record = robot_record()

    entry = robot_entry(record, Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames), 0)

    assert "telemetry" not in entry


def test_a_robot_entry_embeds_the_stored_report_itself() -> None:
    record = robot_record()
    record.telemetry = telemetry_to_dict(TELEMETRY)

    entry = robot_entry(record, Mailbox(CONFIG.queue_policy, CONFIG.fifo_capacity_frames), 0)

    assert entry["telemetry"] is record.telemetry


def test_a_model_entry_is_all_zeros_before_the_first_calibration() -> None:
    world = World()

    assert model_entry("cls", None, world.mailbox) == {
        "batches": 0,
        "frames": 0,
        "batch_size": {},
        "batch_ms": EMPTY_HISTOGRAM,
        "busy_ms": 0,
        "paced_ms": 0,
        "idle_ms": 0,
        "eligible": 0,
    }


def test_a_model_entry_reports_the_scheduler_counters_in_whole_milliseconds() -> None:
    world = World()
    dispatch = world.ready("cls")
    stats = dispatch.scheduler.stats
    stats.batches = 3
    stats.frames = 5
    stats.batch_size = {1: 1, 2: 2}
    stats.batch_ms.record(4.5)
    stats.busy_ns = 13_999_999
    stats.paced_ns = 2 * NS_PER_MS
    stats.idle_ns = 999_999
    world.send(world.open("a"), 1)
    world.open("b")

    assert model_entry("cls", dispatch, world.mailbox) == {
        "batches": 3,
        "frames": 5,
        "batch_size": {"1": 1, "2": 2},
        "batch_ms": stats.batch_ms.to_dict(),
        "busy_ms": 13,
        "paced_ms": 2,
        "idle_ms": 0,
        "eligible": 1,
    }


def test_eligible_counts_only_the_robots_with_a_frame_pending_for_that_model() -> None:
    world = World()
    world.send(world.open("a", ("cls",)), 1)
    world.send(world.open("b"), 1)

    eligible = {model: model_entry(model, None, world.mailbox)["eligible"] for model in MODEL_KEYS}

    assert eligible == {"cls": 2, "depth": 1}


def test_every_stage_histogram_is_empty_before_the_first_calibration() -> None:
    assert stages_entry(None) == {
        "transport_up_ms": EMPTY_HISTOGRAM,
        "slot_wait_ms": EMPTY_HISTOGRAM,
        "dispatch_rt_ms": EMPTY_HISTOGRAM,
        "decode_ms": EMPTY_HISTOGRAM,
        "preprocess_ms": EMPTY_HISTOGRAM,
        "batch_peers_ms": EMPTY_HISTOGRAM,
        "inference_ms": EMPTY_HISTOGRAM,
        "postprocess_ms": EMPTY_HISTOGRAM,
    }


@pytest.mark.parametrize("stage", STAGE_NAMES)
def test_a_stage_histogram_is_published_under_its_name_in_milliseconds(stage: str) -> None:
    dispatch = World().ready("cls")
    dispatch.stages[stage].record(1.5)

    entry = stages_entry(dispatch)

    assert entry == {f"{name}_ms": EMPTY_HISTOGRAM for name in STAGE_NAMES} | {
        f"{stage}_ms": dispatch.stages[stage].to_dict()
    }
    assert entry[f"{stage}_ms"] != EMPTY_HISTOGRAM


def test_connections_show_the_current_count_and_every_total() -> None:
    world = World()
    a = world.open("a")
    world.open("b")
    world.table.end(a, CloseReason.CLIENT_CLOSE, 0)
    world.table.hello("not a hello", lambda: None, 0)
    world.table.ingest(world.table.sessions["b"], b"not a frame", 0, None)
    world.totals.bytes_out = 77

    entry = connections_entry(world.registry, world.totals)

    assert entry == {
        "current": 1,
        "accepted": 2,
        "closed": {
            "client_close": 1,
            "abnormal": 0,
            "idle_timeout": 0,
            "slow_consumer": 0,
            "ingress_limit": 0,
            "protocol_error": 0,
            "replaced": 0,
            "shutdown": 0,
            "internal_error": 0,
        },
        "rejected": {
            "capacity": 0,
            "version": 0,
            "auth": 0,
            "bad_hello": 1,
            "unknown_model": 0,
            "handshake_timeout": 0,
        },
        "frames_rejected": {"malformed": 1, "oversize": 0, "rate_limited": 0},
        "bytes_in": world.totals.bytes_in,
        "bytes_out": 77,
    }
    assert world.totals.bytes_in > 0


def test_a_worker_entry_before_the_first_calibration_nulls_exactly_what_it_sets() -> None:
    status = WorkerStatus(pid=11)

    entry = worker_entry(WORKER_SETTINGS, status, None, None)

    assert entry == {
        "state": "warming",
        "pid": 11,
        "restarts": 0,
        "model_file": None,
        "model_sha256": None,
        "threads": 1,
        "batch_sizes": [1, 2, 4, 8],
        "t_b_ms": None,
        "t_b_cpu_ms": None,
        "batch_cap": None,
        "pacing_share": None,
        "margin_ms": None,
        "feasible": None,
        "timeouts": 0,
        "errors": {"decode": 0, "inference": 0},
    }


def test_a_ready_worker_entry_shows_its_calibration_and_the_plan_made_from_it() -> None:
    world = World()
    dispatch = world.ready("cls")
    dispatch.errors["decode"] = 2
    status = world.supervisors["cls"].status

    entry = worker_entry(WORKER_SETTINGS, status, dispatch, None)

    # The budget is 150 ms less one frame period at 30 fps and the 10 ms allowance:
    # 106.666667 ms, which the 9 ms batch of 8 fits with 97.666667 ms to spare.
    assert entry == {
        "state": "ready",
        "pid": 11,
        "restarts": 0,
        "model_file": "cls.onnx",
        "model_sha256": "ab" * 32,
        "threads": 1,
        "batch_sizes": [1, 2, 4, 8],
        "t_b_ms": {"1": 2.0, "2": 3.0, "4": 5.0, "8": 9.0},
        "t_b_cpu_ms": {"1": 2.5, "2": 3.5, "4": 5.5, "8": 9.5},
        "batch_cap": 8,
        "pacing_share": None,
        "margin_ms": 97.666667,
        "feasible": True,
        "timeouts": 0,
        "errors": {"decode": 2, "inference": 0},
    }


def test_a_worker_that_is_down_keeps_its_calibration_and_shows_no_pid() -> None:
    world = depth_down_world()
    supervisor = world.supervisors["depth"]

    entry = worker_entry(WORKER_SETTINGS, supervisor.status, supervisor.dispatch, 0.75)

    # The budget is 500 ms less one frame period and the allowance: 456.666667 ms, which no
    # size fits, so the cap is 1 and its 600 ms batch leaves a negative margin.
    assert entry == {
        "state": "restarting",
        "pid": None,
        "restarts": 2,
        "model_file": None,
        "model_sha256": None,
        "threads": 1,
        "batch_sizes": [1, 2, 4, 8],
        "t_b_ms": {"1": 600.0, "2": 1_200.0, "4": 2_400.0, "8": 4_800.0},
        "t_b_cpu_ms": {"1": 590.0, "2": 1_180.0, "4": 2_360.0, "8": 4_720.0},
        "batch_cap": 1,
        "pacing_share": 0.75,
        "margin_ms": -143.333333,
        "feasible": False,
        "timeouts": 1,
        "errors": {"decode": 0, "inference": 0},
    }


def test_the_error_kinds_shown_before_the_first_calibration_are_those_counted_after_it() -> None:
    before = worker_entry(WORKER_SETTINGS, WorkerStatus(), None, None)

    assert before["errors"] == World().ready("cls").errors


def test_the_cgroup_entry_joins_the_limits_and_the_usage_and_keeps_nulls() -> None:
    assert cgroup_entry(LIMITS, CGROUP) == {
        "version": 2,
        "cpu_limit_cores": 2.0,
        "cpu_period_us": 100_000,
        "memory_limit_bytes": 4 * 2**30,
        "memory_swap_limit_bytes": None,
        "cpuset": "0-3",
        "cpu_usage_us": 900,
        "cpu_user_us": 600,
        "cpu_system_us": 300,
        "nr_periods": 50,
        "nr_throttled": 2,
        "throttled_us": 1_500,
        "memory_current_bytes": 300 * 2**20,
        "memory_peak_bytes": 310 * 2**20,
        "memory_anon_bytes": 200 * 2**20,
        "memory_file_bytes": 90 * 2**20,
        "memory_sock_bytes": None,
        "oom_kills": 0,
    }


def test_a_process_that_is_not_live_is_null() -> None:
    entry = processes_entry({"gateway": process_stats(10), "worker_cls": None})

    assert entry == {
        "gateway": {
            "pid": 10,
            "cpu_user_ms": 40,
            "cpu_system_ms": 20,
            "rss_bytes": 50 * 2**20,
            "threads": 3,
            "fds": 9,
            "nice": 0,
        },
        "worker_cls": None,
    }


def test_the_loop_entry_counts_the_live_tasks_of_the_running_loop() -> None:
    lag_ms = Histogram()
    lag_ms.record(0.25)

    async def entry_beside_two_tasks() -> dict[str, object]:
        async with asyncio.TaskGroup() as tasks:
            tasks.create_task(asyncio.sleep(1))
            tasks.create_task(asyncio.sleep(1))
            return loop_entry(lag_ms)

    assert run_virtual(entry_beside_two_tasks()) == {"lag_ms": lag_ms.to_dict(), "tasks": 3}


@pytest.mark.parametrize(
    ("build", "uptime_s", "workers"),
    [
        pytest.param(
            warming_world,
            None,
            {"worker_cls": live_process(11), "worker_depth": None},
            id="warming",
        ),
        pytest.param(
            serving_world,
            1.065,
            {"worker_cls": live_process(11), "worker_depth": live_process(12)},
            id="serving",
        ),
        pytest.param(
            depth_down_world,
            1.065,
            {"worker_cls": live_process(11), "worker_depth": None},
            id="depth-down",
        ),
    ],
)
def test_the_render_is_one_json_object_whose_every_field_is_what_its_builder_gives(
    build: Callable[[], World], uptime_s: float | None, workers: dict[str, object]
) -> None:
    world = build()
    now_ns = world.clock.now_ns()

    snapshot = world.snapshot()

    assert snapshot == {
        "schema": 1,
        "ts_mono_ns": now_ns,
        "ts_wall_ms": WALL_MS,
        "uptime_s": uptime_s,
        "config": world.echo,
        "robots": {
            record.robot_id: robot_entry(record, world.mailbox, now_ns)
            for record in world.registry.records()
        },
        "models": {
            model: model_entry(model, supervisor.dispatch, world.mailbox)
            for model, supervisor in world.supervisors.items()
        },
        "stages": {
            model: stages_entry(supervisor.dispatch)
            for model, supervisor in world.supervisors.items()
        },
        "connections": connections_entry(world.registry, world.totals),
        "workers": {
            model: worker_entry(
                WORKER_SETTINGS, supervisor.status, supervisor.dispatch, world.shares[model]
            )
            for model, supervisor in world.supervisors.items()
        },
        "cgroup": cgroup_entry(LIMITS, CGROUP),
        "processes": {"gateway": live_process(GATEWAY_PID)} | workers,
        # The scrape runs as the only task of its loop.
        "loop": {"lag_ms": EMPTY_HISTOGRAM, "tasks": 1},
    }


def test_each_worker_entry_repeats_the_threads_and_batch_sizes_of_its_own_model() -> None:
    models = {
        "cls": CONFIG.models["cls"],
        "depth": replace(CONFIG.models["depth"], threads=2, batch_sizes=(1, 2)),
    }
    workers = World(replace(CONFIG, models=models)).snapshot()["workers"]

    assert isinstance(workers, dict)
    assert {
        model: (entry["threads"], entry["batch_sizes"]) for model, entry in workers.items()
    } == {"cls": (1, [1, 2, 4, 8]), "depth": (2, [1, 2])}


def test_uptime_is_null_until_the_gateway_accepts_and_counts_from_then() -> None:
    world = warming_world()

    before = world.snapshot()["uptime_s"]
    world.accepting_since_ns = world.clock.now_ns()
    world.clock.advance_ns(2_500 * NS_PER_MS)
    after = world.snapshot()["uptime_s"]

    assert (before, after) == (None, 2.5)


def test_the_stats_reader_is_given_the_pid_of_every_process_and_none_for_one_not_live() -> None:
    world = depth_down_world()

    world.snapshot()

    assert world.pids_read == [{"gateway": GATEWAY_PID, "worker_cls": 11, "worker_depth": None}]


class Bystander:
    """Another task of the loop, which counts its turns and acts once, in its second turn.

    Under render_beside the second turn is the one the render's first yield gives it. It
    stops after MAX_TURNS: on the virtual-time loop a task that only yields keeps the clock
    still, so a render that waited for a timer would otherwise never end, and with the bound
    it ends and shows too many turns.
    """

    MAX_TURNS = 64

    def __init__(self, in_second_turn: Callable[[], object] = lambda: None) -> None:
        self.turns = 0
        self._in_second_turn = in_second_turn

    async def run(self) -> None:
        while self.turns < self.MAX_TURNS:
            self.turns += 1
            if self.turns == 2:
                self._in_second_turn()
            await asyncio.sleep(0)


async def render_beside(world: World, bystander: Bystander) -> tuple[bytes, int]:
    """Render while the bystander runs; return the body and the turns it took meanwhile."""
    task = asyncio.create_task(bystander.run())
    # The bystander takes its first turn now, so that each later turn is a yield of the render.
    await asyncio.sleep(0)
    turns_before = bystander.turns
    body = await render_metrics(world.sources)
    turns = bystander.turns - turns_before
    task.cancel()
    # Completed by the cancellation, or by the bystander's last turn before it.
    await asyncio.wait({task})
    return body, turns


@pytest.mark.parametrize(
    ("robots", "yields"), [(0, 0), (1, 0), (16, 0), (17, 1), (32, 1), (33, 2), (256, 15)]
)
def test_the_render_yields_to_the_loop_between_chunks_of_sixteen_robots(
    robots: int, yields: int
) -> None:
    world = World(MANY_ROBOTS)
    for index in range(robots):
        world.open(f"robot-{index}")

    body, turns = run_virtual(render_beside(world, Bystander()))

    assert turns == yields
    assert json.loads(body)["robots"].keys() == {f"robot-{index}" for index in range(robots)}


def test_a_chunked_render_equals_the_entries_built_in_one_step() -> None:
    world = World()
    world.clock.advance_ns(NS_PER_S)
    for index in range(40):
        world.send(world.open(f"robot-{index}"), 1)
    world.clock.advance_ns(NS_PER_S)

    snapshot = world.snapshot()

    assert snapshot["robots"] == {
        record.robot_id: robot_entry(record, world.mailbox, world.clock.now_ns())
        for record in world.registry.records()
    }


def test_a_connect_that_drops_a_record_during_a_yield_does_not_break_the_render() -> None:
    # The table is full and its last record is disconnected, so the next new id replaces it.
    world = World(replace(CONFIG, max_connections=40, max_robot_entries=40))
    for index in range(39):
        world.open(f"robot-{index}")
    world.table.end(world.open("gone"), CloseReason.CLIENT_CLOSE, 0)
    before = [record.robot_id for record in world.registry.records()]

    body, _ = run_virtual(render_beside(world, Bystander(lambda: world.open("late"))))

    after = [record.robot_id for record in world.registry.records()]
    assert (before[-1], after[-1]) == ("gone", "late")
    assert list(json.loads(body)["robots"]) == before


def test_each_chunk_of_robots_is_measured_from_a_clock_reading_of_its_own() -> None:
    world = World()
    for index in range(17):
        world.open(f"robot-{index}")
    world.clock.advance_ns(NS_PER_S)
    bystander = Bystander(lambda: world.clock.advance_ns(5 * NS_PER_MS))

    body, _ = run_virtual(render_beside(world, bystander))

    # Whatever the other tasks did during a yield happened before the next chunk's reading, so
    # no age in that chunk can be negative.
    snapshot = json.loads(body)
    connected_s = {robot_id: entry["connected_s"] for robot_id, entry in snapshot["robots"].items()}
    assert snapshot["ts_mono_ns"] == NS_PER_S
    assert connected_s == {f"robot-{index}": 1.0 for index in range(16)} | {"robot-16": 1.005}


def scrape_while_b_connects_and_is_served() -> tuple[bytes, int]:
    """Scrape while robot b connects and has its frame dispatched for cls, during the stats read.

    Robot a connected before the scrape. The stats read yields to the loop once, as the real one
    does while its thread reads the files, and the bystander acts in that turn: b connects 50 ms
    into the read and its frame is dispatched at once, and the read returns 20 ms later. Return
    the body and the clock reading when the read returned.
    """
    world = World()
    world.clock.advance_ns(NS_PER_S)
    world.ready("cls")
    world.open("a")

    async def read_in_a_thread(pids: Mapping[str, int | None]) -> SystemStats:
        await asyncio.sleep(0)
        return await world.read_system(pids)

    def b_connects_and_is_served() -> None:
        world.clock.advance_ns(50 * NS_PER_MS)
        world.send(world.open("b"), 1)
        world.dispatch("cls")
        world.clock.advance_ns(20 * NS_PER_MS)

    world.sources = replace(world.sources, read_system=read_in_a_thread)
    body, _ = run_virtual(render_beside(world, Bystander(b_connects_and_is_served)))
    return body, world.clock.now_ns()


def test_a_robot_that_connects_during_the_stats_read_is_listed_and_counted() -> None:
    body, _ = scrape_while_b_connects_and_is_served()

    snapshot = json.loads(body)
    assert (list(snapshot["robots"]), snapshot["connections"]["current"]) == (["a", "b"], 2)


def test_the_gauges_of_a_scrape_are_measured_from_a_clock_reading_after_its_stats_read() -> None:
    body, read_returned_ns = scrape_while_b_connects_and_is_served()

    # From a reading taken before the read, b's session and the time since its dispatch would
    # both come out negative.
    snapshot = json.loads(body)
    b = snapshot["robots"]["b"]
    assert (snapshot["ts_mono_ns"], b["connected_s"], b["models"]["cls"]["last_served_age_ms"]) == (
        read_returned_ns,
        0.02,
        20.0,
    )


def test_the_three_identities_hold_with_a_batch_in_flight_and_a_command_queued() -> None:
    robots = serving_world().snapshot()["robots"]

    assert isinstance(robots, dict)
    counters = [
        (robot, entry["models"][model]) for robot, entry in robots.items() for model in MODEL_KEYS
    ]
    assert sum(model["in_flight"] for _, model in counters) == 3
    assert sum(model["commands_queued"] for _, model in counters) == 1
    assert sum(model["frames_pending"] for _, model in counters) == 1
    assert [
        robot
        for robot, entry in robots.items()
        if entry["frames_in"] != entry["frames_rejected"] + entry["frames_accepted"]
    ] == []
    assert [
        robot
        for robot, model in counters
        if model["frames_received"]
        != model["frames_served"]
        + model["frames_overwritten"]
        + model["frames_skipped"]
        + model["frames_dropped_disconnect"]
        + model["frames_pending"]
        or model["frames_served"]
        != model["commands_sent"]
        + model["commands_queued"]
        + model["commands_dropped_downlink"]
        + model["results_dropped_disconnect"]
        + model["frames_failed"]
        + model["in_flight"]
    ] == []


def test_a_stats_read_that_does_not_return_fails_the_scrape_at_the_read_timeout() -> None:
    async def never_returns(pids: Mapping[str, int | None]) -> SystemStats:
        # Nothing completes this future: only the render's timeout ends the wait.
        answer: asyncio.Future[SystemStats] = asyncio.get_running_loop().create_future()
        return await answer

    sources = replace(World().sources, read_system=never_returns)

    async def scrape() -> float:
        with pytest.raises(TimeoutError):
            await render_metrics(sources)
        return asyncio.get_running_loop().time()

    assert run_virtual(scrape()) == STATS_READ_TIMEOUT_S


def test_the_system_stats_are_read_off_the_event_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    # The kernel builds these files when they are read, so a read on the loop would hold every
    # connection for as long as it takes.
    reader_threads: list[int] = []

    def read_stats() -> CgroupStats:
        reader_threads.append(threading.get_ident())
        return CGROUP

    def read_process_stats(pid: int) -> ProcessStats:
        reader_threads.append(threading.get_ident())
        return process_stats(pid)

    monkeypatch.setattr("cloud.metrics.read_stats", read_stats)
    monkeypatch.setattr("cloud.metrics.read_process_stats", read_process_stats)

    async def read_beside_the_loop() -> tuple[SystemStats, int]:
        stats = await read_system_stats(
            {"gateway": GATEWAY_PID, "worker_cls": 11, "worker_depth": None}
        )
        return stats, threading.get_ident()

    # A real loop, because the virtual-time loop reports a wait for a thread as a deadlock.
    stats, loop_thread = asyncio.run(read_beside_the_loop())

    assert stats == SystemStats(
        CGROUP,
        {
            "gateway": process_stats(GATEWAY_PID),
            "worker_cls": process_stats(11),
            "worker_depth": None,
        },
    )
    assert (len(reader_threads), loop_thread in reader_threads) == (3, False)


@pytest.mark.skipif(sys.platform != "linux", reason="reads the kernel's cgroup and /proc files")
def test_the_system_stats_cover_live_processes_and_null_the_others() -> None:
    # Above the kernel's largest pid (2**22), so no process has it.
    no_such_pid = 2**22 + 1

    stats = asyncio.run(
        read_system_stats({"gateway": os.getpid(), "worker_cls": None, "worker_depth": no_such_pid})
    )

    gateway = stats.processes["gateway"]
    assert gateway is not None
    assert gateway.pid == os.getpid()
    assert (stats.processes["worker_cls"], stats.processes["worker_depth"]) == (None, None)
    assert isinstance(stats.cgroup, CgroupStats)


def test_health_is_ok_while_the_gateway_accepts_and_every_worker_is_ready() -> None:
    reply = health_reply(dict.fromkeys(MODEL_KEYS, WorkerState.READY), accepting=True)

    assert (reply.status, json.loads(reply.body)) == (200, {"status": "ok"})


@pytest.mark.parametrize(
    ("cls", "depth", "accepting"),
    [
        combination
        for combination in itertools.product(WorkerState, WorkerState, [True, False])
        if combination != (WorkerState.READY, WorkerState.READY, True)
    ],
)
def test_health_is_unavailable_with_the_states_in_every_other_case(
    cls: WorkerState, depth: WorkerState, accepting: bool
) -> None:
    reply = health_reply({"cls": cls, "depth": depth}, accepting)

    assert (reply.status, json.loads(reply.body)) == (
        503,
        {
            "status": "unavailable",
            "accepting": accepting,
            "workers": {"cls": cls.value, "depth": depth.value},
        },
    )
