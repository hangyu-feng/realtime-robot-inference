"""The /metrics snapshot, /healthz, the event-loop lag probe and the reader of system statistics.

The field builders shape the objects of DESIGN.md section 10 from the state the rest of the
gateway keeps: each reads counters and histograms that are already aggregated and changes
nothing. render_metrics joins them into one JSON object without stalling the event loop, and
probe_loop_lag measures how late that loop runs its timers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

from cloud.counters import ConnectionTotals, WorkerState, WorkerStatus
from cloud.dispatch import ModelDispatch
from cloud.mailbox import Mailbox
from cloud.registry import RobotRecord, RobotRegistry
from cloud.scheduler import ModelStats
from cloud.supervisor import ModelSupervisor
from common.aio import sleep_until
from common.cgroup import CgroupStats, Limits, read_stats
from common.clock import Clock
from common.histogram import Histogram
from common.procstat import ProcessGoneError, ProcessStats, read_process_stats
from common.protocol import STAGE_NAMES
from common.transport import HttpReply

# The scraper reads /metrics once per second, so more renders at once means that something
# else polls, and each render holds a whole snapshot in memory.
MAX_CONCURRENT_RENDERS = 4
# Far above any healthy read of kernel pseudo-files. A slower read fails the scrape rather
# than holding it.
STATS_READ_TIMEOUT_S = 1.0

_NS_PER_MS = 1_000_000
# Robot entries serialized between two yields to the event loop.
_ROBOTS_PER_CHUNK = 16
# The keys of workers.<m>.errors, which cloud.dispatch.ModelDispatch counts from the first
# calibration on. They are listed here because the object shows every kind, at 0, before it.
_ERROR_KINDS = ("decode", "inference")
# Strict JSON, which has no NaN or infinity, in its compact form: the event loop serializes
# every scrape.
_JSON_ENCODER = json.JSONEncoder(allow_nan=False, separators=(",", ":"))


@dataclass(frozen=True)
class SystemStats:
    """What one scrape reads from the kernel: the cgroup's usage and each process's counters.

    processes maps a process name of /metrics to its counters, None for one that is not live.
    """

    cgroup: CgroupStats
    processes: Mapping[str, ProcessStats | None]


# Reads the statistics of the processes given by name, each with its pid or None when it has
# no live process.
SystemStatsReader = Callable[[Mapping[str, int | None]], Awaitable[SystemStats]]


def _read_stats(pids: Mapping[str, int | None]) -> SystemStats:
    processes: dict[str, ProcessStats | None] = {}
    for name, pid in pids.items():
        if pid is None:
            processes[name] = None
            continue
        try:
            processes[name] = read_process_stats(pid)
        except ProcessGoneError:
            # A worker that has just died, before its supervisor has cleared the pid.
            processes[name] = None
    return SystemStats(read_stats(), processes)


async def read_system_stats(pids: Mapping[str, int | None]) -> SystemStats:
    """Read the cgroup's usage and the counters of every live process in a worker thread.

    The kernel builds these files when they are read, so no file is read on the event loop.
    The caller bounds the wait; a read that outlives it finishes in its thread and is dropped.
    """
    return await asyncio.to_thread(_read_stats, pids)


@dataclass(frozen=True)
class MetricsSources:
    """Everything a /metrics snapshot reads.

    config_echo is the object cloud.config.config_echo builds. pacing_shares returns the share
    each model's worker is paced to now; a model without an entry is not paced.
    accepting_since_ns returns when the gateway began accepting connections, None before.
    """

    config_echo: Mapping[str, object]
    limits: Limits
    clock: Clock
    wall_clock_ms: Callable[[], int]
    gateway_pid: int
    registry: RobotRegistry
    mailbox: Mailbox
    totals: ConnectionTotals
    supervisors: Mapping[str, ModelSupervisor]
    pacing_shares: Callable[[], Mapping[str, float | None]]
    accepting_since_ns: Callable[[], int | None]
    loop_lag_ms: Histogram
    read_system: SystemStatsReader


def robot_entry(record: RobotRecord, mailbox: Mailbox, now_ns: int) -> dict[str, object]:
    """Build robots.<id> (DESIGN.md 10.3) for a scrape at now_ns.

    The whole entry is read in one synchronous step, and the core keeps the three conservation
    identities after every call, so they hold in every entry, gauges included.
    """
    # The session ends at the disconnect; an open one runs to the scrape.
    end_ns = now_ns if record.disconnected_ns is None else record.disconnected_ns
    models: dict[str, object] = {}
    for model, counters in record.models.items():
        # The longest gap also covers the interval that is still open, so a robot that is
        # starved now, or never got a command in its session, shows it. last_command_ns is
        # None for a model the current session does not subscribe to.
        open_gap_ns = 0 if counters.last_command_ns is None else end_ns - counters.last_command_ns
        models[model] = {
            "frames_received": counters.frames_received,
            "frames_served": counters.frames_served,
            "frames_overwritten": counters.frames_overwritten,
            "frames_skipped": counters.frames_skipped,
            "frames_dropped_disconnect": counters.frames_dropped_disconnect,
            # The mailbox is keyed by robot id. A disconnected record holds no frame, and its
            # id may belong to a newer record by now, whose frames are not this one's.
            "frames_pending": (
                mailbox.pending_count(record.robot_id, model) if record.connected else 0
            ),
            "in_flight": counters.in_flight,
            "commands_queued": counters.commands_queued,
            "commands_sent": counters.commands_sent,
            "commands_dropped_downlink": counters.commands_dropped_downlink,
            "results_dropped_disconnect": counters.results_dropped_disconnect,
            "frames_failed": counters.frames_failed,
            "last_served_age_ms": (
                None
                if counters.last_served_ns is None
                else (now_ns - counters.last_served_ns) / 1e6
            ),
            "max_service_gap_ms": max(counters.max_service_gap_ns, open_gap_ns) / 1e6,
        }
    entry: dict[str, object] = {
        "session": record.session,
        "connected": record.connected,
        "connected_s": (end_ns - record.connected_ns) / 1e9,
        "subscribed": list(record.subscribed),
        "stale_ms": {model: stale_ns / 1e6 for model, stale_ns in record.stale_ns.items()},
        "watchdog_ms": {
            model: watchdog_ns / 1e6 for model, watchdog_ns in record.watchdog_ns.items()
        },
        "rtt_ms": None if record.rtt_ns is None else record.rtt_ns / 1e6,
        "bytes_in": record.bytes_in,
        "bytes_out": record.bytes_out,
        "frames_in": record.frames_in,
        "frames_rejected": record.frames_rejected,
        "frames_accepted": record.frames_accepted,
        "models": models,
    }
    if record.telemetry is not None:
        # Stored as the object it is published as, so no histogram is serialized per scrape.
        entry["telemetry"] = record.telemetry
    return entry


def model_entry(model: str, dispatch: ModelDispatch | None, mailbox: Mailbox) -> dict[str, object]:
    """Build models.<m> (DESIGN.md 10.5); dispatch is None before the first calibration."""
    stats = ModelStats() if dispatch is None else dispatch.scheduler.stats
    return {
        "batches": stats.batches,
        "frames": stats.frames,
        "batch_size": {str(size): count for size, count in stats.batch_size.items()},
        "batch_ms": stats.batch_ms.to_dict(),
        # Whole milliseconds, rounded down, so each stays a counter that only grows.
        "busy_ms": stats.busy_ns // _NS_PER_MS,
        "paced_ms": stats.paced_ns // _NS_PER_MS,
        "idle_ms": stats.idle_ns // _NS_PER_MS,
        "eligible": len(mailbox.pending_robots(model)),
    }


def stages_entry(dispatch: ModelDispatch | None) -> dict[str, object]:
    """Build stages.<m> (DESIGN.md 10.6); every histogram is empty before the first calibration."""
    stages = dict.fromkeys(STAGE_NAMES, Histogram()) if dispatch is None else dispatch.stages
    return {f"{name}_ms": stages[name].to_dict() for name in STAGE_NAMES}


def connections_entry(registry: RobotRegistry, totals: ConnectionTotals) -> dict[str, object]:
    """Build connections (DESIGN.md 10.7)."""
    return {
        "current": registry.connected_count(),
        "accepted": totals.accepted,
        "closed": dict(totals.closed),
        "rejected": dict(totals.rejected),
        "frames_rejected": dict(totals.frames_rejected),
        "bytes_in": totals.bytes_in,
        "bytes_out": totals.bytes_out,
    }


def worker_entry(
    settings: Mapping[str, object],
    status: WorkerStatus,
    dispatch: ModelDispatch | None,
    pacing_share_cores: float | None,
) -> dict[str, object]:
    """Build workers.<m> (DESIGN.md 10.7).

    settings is the model's entry of the config echo. Its threads and batch sizes are what the
    worker reports once it is ready, so they are published from the start. dispatch is None
    before the first calibration of the run, and exactly the fields that calibration sets are
    null until then.
    """
    entry: dict[str, object] = {
        "state": status.state.value,
        "pid": status.pid,
        "restarts": status.restarts,
        "model_file": None,
        "model_sha256": None,
        "threads": settings["threads"],
        "batch_sizes": settings["batch_sizes"],
        "t_b_ms": None,
        "t_b_cpu_ms": None,
        "batch_cap": None,
        "pacing_share": pacing_share_cores,
        "margin_ms": None,
        "feasible": None,
        "timeouts": status.timeouts,
        "errors": dict.fromkeys(_ERROR_KINDS, 0),
    }
    if dispatch is not None:
        calibration = dispatch.calibration
        entry.update(
            model_file=calibration.model_file,
            model_sha256=calibration.model_sha256,
            t_b_ms={str(size): batch_ns / 1e6 for size, batch_ns in calibration.t_b_ns.items()},
            t_b_cpu_ms={str(size): cpu_ns / 1e6 for size, cpu_ns in calibration.t_b_cpu_ns.items()},
            batch_cap=dispatch.plan.scheduler.batch_cap,
            margin_ms=dispatch.plan.margin_ns / 1e6,
            feasible=dispatch.plan.feasible,
            errors=dict(dispatch.errors),
        )
    return entry


def cgroup_entry(limits: Limits, stats: CgroupStats) -> dict[str, object]:
    """Build cgroup (DESIGN.md 10.7): the limits and the usage, field by field, nulls kept."""
    return {**asdict(limits), **asdict(stats)}


def processes_entry(processes: Mapping[str, ProcessStats | None]) -> dict[str, object]:
    """Build processes (DESIGN.md 10.7): null for a process that is not live."""
    return {name: None if stats is None else asdict(stats) for name, stats in processes.items()}


def loop_entry(lag_ms: Histogram) -> dict[str, object]:
    """Build loop (DESIGN.md 10.7); it counts the tasks of the running event loop."""
    return {"lag_ms": lag_ms.to_dict(), "tasks": len(asyncio.all_tasks())}


async def render_metrics(sources: MetricsSources) -> bytes:
    """Return the /metrics object of DESIGN.md 10.2 as JSON, built so the loop is not stalled.

    The kernel files are read first, off the loop and under STATS_READ_TIMEOUT_S; TimeoutError
    propagates, and the caller answers that the statistics are unavailable. Everything but the
    robots is then built and serialized in one synchronous step. The robot entries follow in
    chunks, with a yield to the loop between chunks so that waiting frames are read. Each
    entry is built in one synchronous step, so the identities of DESIGN.md 10.3 hold for every
    robot in every scrape, while the totals of the first step may differ from the sum over
    the robots by what happened during the yields.
    """
    pids: dict[str, int | None] = {"gateway": sources.gateway_pid}
    for model, supervisor in sources.supervisors.items():
        pids[f"worker_{model}"] = supervisor.status.pid
    async with asyncio.timeout(STATS_READ_TIMEOUT_S):
        system = await sources.read_system(pids)
    # Other tasks ran during the read, so the clock and the records are read after it: a robot
    # that connected or was served meanwhile is under robots as well as in connections.current,
    # and none of its gauges comes out negative.
    now_ns = sources.clock.now_ns()
    accepting_since_ns = sources.accepting_since_ns()
    shares = sources.pacing_shares()
    # cloud.config.config_echo builds models as one object of settings per model.
    settings = cast(Mapping[str, Mapping[str, object]], sources.config_echo["models"])
    head = _JSON_ENCODER.encode(
        {
            "schema": 1,
            "ts_mono_ns": now_ns,
            # Recorded for the reader of a scrape, and never subtracted.
            "ts_wall_ms": sources.wall_clock_ms(),
            "uptime_s": (
                None if accepting_since_ns is None else (now_ns - accepting_since_ns) / 1e9
            ),
            "config": sources.config_echo,
            "models": {
                model: model_entry(model, supervisor.dispatch, sources.mailbox)
                for model, supervisor in sources.supervisors.items()
            },
            "stages": {
                model: stages_entry(supervisor.dispatch)
                for model, supervisor in sources.supervisors.items()
            },
            "connections": connections_entry(sources.registry, sources.totals),
            "workers": {
                model: worker_entry(
                    settings[model], supervisor.status, supervisor.dispatch, shares.get(model)
                )
                for model, supervisor in sources.supervisors.items()
            },
            "cgroup": cgroup_entry(sources.limits, system.cgroup),
            "processes": processes_entry(system.processes),
            "loop": loop_entry(sources.loop_lag_ms),
        }
    )
    # Taken once: a robot that connects, or a record that is dropped, during a yield below
    # cannot break the iteration.
    records = list(sources.registry.records())
    robots: list[str] = []
    for start in range(0, len(records), _ROBOTS_PER_CHUNK):
        if start > 0:
            await asyncio.sleep(0)
            # Other tasks ran during the yield, and a gauge measured from an earlier reading
            # could come out negative for a robot they served.
            now_ns = sources.clock.now_ns()
        robots.extend(
            f"{_JSON_ENCODER.encode(record.robot_id)}:"
            f"{_JSON_ENCODER.encode(robot_entry(record, sources.mailbox, now_ns))}"
            for record in records[start : start + _ROBOTS_PER_CHUNK]
        )
    # The head is one object, and the robots become its last member.
    return "".join((head[:-1], ',"robots":{', ",".join(robots), "}}")).encode()


def health_reply(states: Mapping[str, WorkerState], accepting: bool) -> HttpReply:
    """Answer /healthz: 200 while the gateway accepts and every worker is ready, else 503.

    It is 503 during warm-up, during shutdown, and while a worker restarts, when connected
    robots keep their sessions and new robots are still accepted.
    """
    if accepting and all(state is WorkerState.READY for state in states.values()):
        return HttpReply(200, _JSON_ENCODER.encode({"status": "ok"}).encode())
    body = {
        "status": "unavailable",
        "accepting": accepting,
        "workers": {model: state.value for model, state in states.items()},
    }
    return HttpReply(503, _JSON_ENCODER.encode(body).encode())


async def probe_loop_lag(
    clock: Clock, lag_ms: Histogram, period_ns: int, active: asyncio.Event
) -> None:
    """Record for ever how late the event loop runs a timer of period_ns, while active is set.

    The targets are fixed steps of the period, so a late wake-up does not shift the next one.
    A stall that passes several targets gives one sample. A lag below about 1 ms is the
    resolution of the loop's timers, not a stall. It runs until it is cancelled.
    """
    while True:
        # No timeout by design: the gateway sets active when the first robot connects. While
        # none is connected the probe is parked here, so an idle gateway runs no timer and
        # idle samples do not dilute the histogram.
        await active.wait()
        # A target kept across the park would record the whole parked time as one sample.
        target_ns = clock.now_ns() + period_ns
        while True:
            # The configuration keeps the period at 1 ms or more, and the probe relies on
            # that: with a period below the time between two clock readings, sleep_until
            # would find every target already passed and return without ever yielding.
            await sleep_until(clock, target_ns)
            if not active.is_set():
                # The last robot left during the wait, so this wake-up is not sampled.
                break
            now_ns = clock.now_ns()
            lag_ms.record((now_ns - target_ns) / 1e6)
            target_ns += period_ns
            if target_ns <= now_ns:
                # A stall passed the next target too. Each target it passed would add a
                # sample of the same stall, so the targets start again from now.
                target_ns = now_ns + period_ns
