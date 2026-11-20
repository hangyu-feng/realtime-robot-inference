"""Builders of synthetic run directories, for the tests of everything that reads a results root.

synthetic_run invents a whole serving run whose files agree with each other: robots that
receive commands at fixed rates, and scrapes whose counters follow them. worker_kill_run
invents one in which the scenario's worker is killed and comes back. A test changes
what it is about with dataclasses.replace, edit_metrics, command_rows and tick_rows, and
then either summarizes it in memory (SyntheticRun.run) or writes it under a results root.
The writers make the kinds of directory a results root can hold: a finished run, a run
whose gate failed or whose code is other code (through the spec and the gated flag), an
attempt, a hand run and a model benchmark run. write_evidence makes a whole results root
of such runs, aggregated.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from bench.aggregate import Current, build
from bench.candidates import CANDIDATES
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    EVENTS_FILE,
    GATE_FILE,
    INSIDE_ENVIRONMENT_FILE,
    PLANS_DIRECTORY,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    SUMMARY_FILE,
    ConfigGate,
    Done,
    Event,
    EventName,
    FleetConfig,
    FreshGate,
    Gate,
    GatedLimits,
    ImageGate,
    InsideEnvironment,
    LimitsGate,
    ModelSet,
    RunConfig,
    RunEnvironment,
    RunSpec,
    Scrape,
    SuiteGate,
    done_files,
)
from bench.runplan import (
    MODEL_BENCH_KEY,
    compare_plan,
    evidence_plan,
    fleet_plans,
    overrides_for,
    planned_events,
    resolved_config,
    run_id,
    scenario_for,
)
from bench.scenarios import RobotPlan, Scenario, sweep
from bench.summarize import (
    RAW_DIRECTORY,
    REPORTS_DIRECTORY,
    RobotSamples,
    Run,
    run_files,
    summarizer_digest,
    write_summary,
)
from cloud.config import Topology
from common.cgroup import CgroupStats, Limits
from common.commands import MODEL_KEYS
from common.histogram import Histogram
from common.procstat import ProcessStats
from common.protocol import STAGE_NAMES
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from robot.control import Verdict
from robot.frames import FrameSetStats
from robot.samples import COMMAND_DTYPE, TICK_DTYPE, VERDICTS, TickOutcome
from robot.settings import LinkSettings

type Rows = npt.NDArray[np.void]
type Times = Sequence[int] | npt.NDArray[np.int64]

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
FRAME_RATE_HZ = 30.0
FRAME_PERIOD_NS = round(NS_PER_S / FRAME_RATE_HZ)
# Time zero of a synthetic run on the shared monotonic clock.
T0_NS = 1_000 * NS_PER_S
CODE_DIGEST = "c0de" * 16
OTHER_CODE_DIGEST = "0dd5" * 16
IMAGE_DIGEST = "sha256:" + "1a" * 32
# How a synthetic robot is served unless a test says otherwise: each model answers every
# so many of its frames, after a fixed time from capture to receipt.
EVERY_FRAMES: Mapping[str, int] = {"cls": 1, "depth": 3}
FRESHNESS_MS: Mapping[str, float] = {"cls": 20.0, "depth": 80.0}
# The worker's time on one frame, which is also its CPU time, and the batch cap it reports.
SERVICE_MS: Mapping[str, float] = {"cls": 5.0, "depth": 20.0}
BATCH_CAP: Mapping[str, int] = {"cls": 8, "depth": 1}
# A scrape's reading of the gateway's clock comes this long after its planned second, between
# the driver's two readings, which are DRIVER_MARGIN_NS before and after it.
SCRAPE_LAG_NS = 2 * NS_PER_MS
DRIVER_MARGIN_NS = NS_PER_MS
# The final scrape, after the fleet has left.
FINAL_SCRAPE_AFTER_NS = 1_500 * NS_PER_MS
# How long after a kill the gateway closes the dead robot's session.
KILL_DETECT_NS = 50 * NS_PER_MS
# A worker kill comes this long after its planned time; the killed worker's model sends no
# command for this long after it; the new process takes this pid.
KILL_LATE_NS = 400_000
WORKER_OUTAGE_NS = 1_500 * NS_PER_MS
RESTARTED_PID = 13
FIRE_LAG_NS = 200_000
DELIVER_NS = 500_000
FRAME_BYTES = 40_000
GATEWAY_CPU_MS_PER_FRAME = 0.1
LIMITS = Limits(
    version=2,
    cpu_limit_cores=2.0,
    cpu_period_us=100_000,
    memory_limit_bytes=4 * 2**30,
    memory_swap_limit_bytes=0,
    cpuset="0-1",
)
_CLOSE_REASONS = (
    "client_close",
    "abnormal",
    "idle_timeout",
    "slow_consumer",
    "ingress_limit",
    "protocol_error",
    "replaced",
    "shutdown",
    "internal_error",
)
_REJECT_REASONS = ("capacity", "version", "auth", "bad_hello", "unknown_model", "handshake_timeout")
_FRAME_REJECT_REASONS = ("malformed", "oversize", "rate_limited")
_PIDS = {"gateway": 1, "worker_cls": 11, "worker_depth": 12, "fleet": 21, "driver": 20}
# The server stages of a synthetic command that do not depend on its model, in microseconds.
# The slot wait takes what the command's time from send to result leaves.
_FIXED_STAGE_US = {
    "transport_up": 500,
    "dispatch_rt": 200,
    "decode": 300,
    "preprocess": 200,
    "batch_peers": 0,
    "postprocess": 100,
}


def run_spec(
    scenario_key: str = "steady",
    *,
    topology: Topology = Topology.SINGLE,
    model_set: ModelSet = ModelSet.FAKE,
    overrides: Mapping[str, Any] | None = None,
    code_digest: str = CODE_DIGEST,
    repeat: int = 1,
    candidate: str | None = None,
    arm: str | None = None,
    extra_compose: str | None = None,
) -> RunSpec:
    """Return a run spec whose server config is the model set's under the given overrides.

    Without overrides, extra compose file or candidate it is the evidence plan's spec for the
    key, the topology, the model set, the code digest and the repeat, so a run of it counts
    (the plan runs the model benchmark on the real set only); with a candidate and its arm's
    overrides it is that arm's spec in the comparison plan.
    """
    return RunSpec(
        scenario_key=scenario_key,
        topology=topology,
        phase_mode=None if scenario_key == MODEL_BENCH_KEY else "random",
        model_set=model_set,
        server_config=resolved_config(topology, overrides_for(model_set, overrides or {})),
        uplink=LinkSettings(),
        downlink=LinkSettings(),
        extra_compose=extra_compose,
        candidate=candidate,
        arm=arm,
        code_digest=code_digest,
        repeat=repeat,
    )


def run_directory(results: Path, spec: RunSpec) -> Path:
    """Return where a run of the spec lies under a results root (DESIGN.md 11.1)."""
    if spec.candidate is not None and spec.arm is not None:
        return results / RAW_DIRECTORY / "compare" / spec.candidate / spec.arm / run_id(spec)
    return results / RAW_DIRECTORY / spec.scenario_key / run_id(spec)


def command_rows(
    model: str,
    capture_ns: Times,
    receive_ns: Times,
    *,
    verdicts: Sequence[Verdict] | None = None,
    frame_numbers: Times | None = None,
    session: int = 1,
    batch_size: int = 1,
    fire_lag_ns: int = 0,
    deliver_ns: int = 0,
    stage_us: Mapping[str, int] | None = None,
) -> Rows:
    """Return command samples of one model, one per capture and receive time, in that order.

    Every command is applied unless verdicts says otherwise. A command is sent fire_lag_ns
    after its capture and its result leaves the gateway deliver_ns before its receipt. The
    given stages are the same for every command, and the slot wait takes the rest of the
    time from send to result, so the stages add up as the gateway's do.
    """
    rows = np.zeros(len(capture_ns), dtype=COMMAND_DTYPE)
    rows["session"] = session
    rows["model"] = MODEL_KEYS.index(model)
    rows["verdict"] = [
        VERDICTS.index(verdict) for verdict in verdicts or [Verdict.APPLIED] * len(capture_ns)
    ]
    rows["batch_size"] = batch_size
    rows["frame_number"] = range(1, len(capture_ns) + 1) if frame_numbers is None else frame_numbers
    rows["capture_ns"] = capture_ns
    rows["send_ns"] = rows["capture_ns"] + fire_lag_ns
    rows["receive_ns"] = receive_ns
    rows["result_ns"] = rows["receive_ns"] - deliver_ns
    stages = dict.fromkeys(STAGE_NAMES, 0) | dict(stage_us or {})
    for index, name in enumerate(STAGE_NAMES):
        rows["stage_us"][:, index] = stages[name]
    others_us = sum(value for name, value in stages.items() if name != "slot_wait")
    rows["stage_us"][:, STAGE_NAMES.index("slot_wait")] = (
        rows["result_ns"] - rows["send_ns"]
    ) // 1000 - others_us
    return rows


def merged(*parts: Rows) -> Rows:
    """Return one robot's commands of several models in the order they were received."""
    rows = np.concatenate(parts)
    ordered: Rows = rows[np.argsort(rows["receive_ns"], kind="stable")]
    return ordered


def tick_rows(
    capture_ns: Times,
    *,
    fire_lag_ns: int = 0,
    outcomes: Sequence[TickOutcome] | None = None,
) -> Rows:
    """Return tick samples, one per capture time; every tick is sent unless outcomes says so."""
    rows = np.zeros(len(capture_ns), dtype=TICK_DTYPE)
    rows["frame_number"] = range(1, len(capture_ns) + 1)
    rows["capture_ns"] = capture_ns
    rows["outcome"] = outcomes or [TickOutcome.SENT] * len(capture_ns)
    rows["send_ns"] = np.where(
        rows["outcome"] == TickOutcome.SENT, rows["capture_ns"] + fire_lag_ns, 0
    )
    return rows


def samples(commands: Rows, ticks: Rows) -> RobotSamples:
    """Return one robot's samples, with no row dropped."""
    return RobotSamples(commands, ticks, dropped_ticks=0, dropped_commands=0)


def histogram(value_ms: float, count: int) -> dict[str, object]:
    """Return the /metrics form of a histogram that recorded one value count times."""
    if count == 0:
        return Histogram().to_dict()
    once = Histogram()
    once.record(value_ms)
    # Written out, not recorded count times: a long run's histograms hold many thousands.
    buckets = once.to_dict()["buckets"]
    assert isinstance(buckets, dict)
    (bucket,) = buckets
    return {
        "count": count,
        "sum": value_ms * count,
        "min": value_ms,
        "max": value_ms,
        "buckets": {bucket: count},
    }


def process(
    pid: int, *, cpu_ms: int = 0, rss_bytes: int = 50 * 2**20, fds: int = 10
) -> ProcessStats:
    """Return a process's counters, with a quarter of its CPU time spent in the kernel.

    A process's CPU time is its user time plus its system time. Both parts grow here, so a
    reader that takes only one of them gets every CPU quantity wrong.
    """
    system_ms = cpu_ms // 4
    return ProcessStats(
        pid=pid,
        cpu_user_ms=cpu_ms - system_ms,
        cpu_system_ms=system_ms,
        rss_bytes=rss_bytes,
        threads=1,
        fds=fds,
        nice=0,
    )


def robot_entry(
    *,
    connected: bool = True,
    connected_s: float = 0.0,
    frames_in: int = 0,
    commands_sent: Mapping[str, int] | None = None,
    frames_skipped: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Return robots.<id> of /metrics (DESIGN.md 10.3), with the conservation identities kept.

    Every frame that came in was accepted; a model served the frames whose commands were
    sent, passed over frames_skipped, and the rest were overwritten in the slot.
    """
    sent = dict.fromkeys(MODEL_KEYS, 0) | dict(commands_sent or {})
    skipped = dict.fromkeys(MODEL_KEYS, 0) | dict(frames_skipped or {})
    return {
        "session": 1,
        "connected": connected,
        "connected_s": connected_s,
        "subscribed": list(MODEL_KEYS),
        "stale_ms": dict(DEFAULT_STALE_MS),
        "watchdog_ms": dict(DEFAULT_WATCHDOG_MS),
        "rtt_ms": None,
        "bytes_in": frames_in * FRAME_BYTES,
        "bytes_out": 0,
        "frames_in": frames_in,
        "frames_rejected": 0,
        "frames_accepted": frames_in,
        "models": {
            model: {
                "frames_received": frames_in,
                "frames_served": sent[model],
                "frames_overwritten": frames_in - sent[model] - skipped[model],
                "frames_skipped": skipped[model],
                "frames_dropped_disconnect": 0,
                "frames_pending": 0,
                "in_flight": 0,
                "commands_queued": 0,
                "commands_sent": sent[model],
                "commands_dropped_downlink": 0,
                "results_dropped_disconnect": 0,
                "frames_failed": 0,
                "last_served_age_ms": None,
                "max_service_gap_ms": 0.0,
            }
            for model in MODEL_KEYS
        },
    }


def metrics(
    ts_mono_ns: int, changes: Mapping[str, object] | None = None, *, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return a /metrics object (DESIGN.md 10) of a gateway that has served nothing yet.

    changes maps dotted paths to the values that replace the idle ones: "connections.accepted",
    "robots.a", "workers.depth.pacing_share".
    """
    batch_sizes = [1, 2, 4, 8]
    snapshot: dict[str, Any] = {
        "schema": 1,
        "ts_mono_ns": ts_mono_ns,
        "ts_wall_ms": 1_700_000_000_000 + ts_mono_ns // NS_PER_MS,
        "uptime_s": (ts_mono_ns - T0_NS) / NS_PER_S + 10.0,
        "config": {**config, "version": "0123abc"},
        "robots": {},
        "models": {
            model: {
                "batches": 0,
                "frames": 0,
                "batch_size": {},
                "batch_ms": Histogram().to_dict(),
                "busy_ms": 0,
                "paced_ms": 0,
                "idle_ms": 0,
                "eligible": 0,
            }
            for model in MODEL_KEYS
        },
        "stages": {
            model: {f"{name}_ms": Histogram().to_dict() for name in STAGE_NAMES}
            for model in MODEL_KEYS
        },
        "connections": {
            "current": 0,
            "accepted": 0,
            "closed": dict.fromkeys(_CLOSE_REASONS, 0),
            "rejected": dict.fromkeys(_REJECT_REASONS, 0),
            "frames_rejected": dict.fromkeys(_FRAME_REJECT_REASONS, 0),
            "bytes_in": 0,
            "bytes_out": 0,
        },
        "workers": {
            model: {
                "state": "ready",
                "pid": _PIDS[f"worker_{model}"],
                "restarts": 0,
                "model_file": None,
                "model_sha256": None,
                "threads": 1,
                "batch_sizes": batch_sizes,
                "t_b_ms": {str(size): SERVICE_MS[model] * size for size in batch_sizes},
                "t_b_cpu_ms": {str(size): SERVICE_MS[model] * size for size in batch_sizes},
                "batch_cap": BATCH_CAP[model],
                "pacing_share": None,
                "margin_ms": 0.0,
                "feasible": True,
                "timeouts": 0,
                "errors": {"decode": 0, "inference": 0},
            }
            for model in MODEL_KEYS
        },
        "cgroup": {
            **asdict(LIMITS),
            **asdict(
                CgroupStats(
                    cpu_usage_us=0,
                    cpu_user_us=0,
                    cpu_system_us=0,
                    nr_periods=0,
                    nr_throttled=0,
                    throttled_us=0,
                    memory_current_bytes=400 * 2**20,
                    memory_peak_bytes=450 * 2**20,
                    memory_anon_bytes=300 * 2**20,
                    memory_file_bytes=50 * 2**20,
                    memory_sock_bytes=2**20,
                    oom_kills=0,
                )
            ),
        },
        "processes": {
            name: asdict(process(_PIDS[name])) for name in ("gateway", "worker_cls", "worker_depth")
        },
        "loop": {"lag_ms": Histogram().to_dict(), "tasks": 5},
    }
    for path, value in (changes or {}).items():
        *parents, last = path.split(".")
        target = snapshot
        for key in parents:
            target = target[key]
        target[last] = value
    return snapshot


def scrape(
    k: int | None,
    snapshot: Mapping[str, Any],
    *,
    fleet: Mapping[str, ProcessStats] | None = None,
    fleet_cgroup: CgroupStats | None = None,
) -> Scrape:
    """Return a successful scrape line: the driver read its clock just around the gateway's."""
    ts_mono_ns: int = snapshot["ts_mono_ns"]
    return Scrape(
        k=k,
        driver_before_ns=ts_mono_ns - DRIVER_MARGIN_NS,
        driver_after_ns=ts_mono_ns + DRIVER_MARGIN_NS,
        metrics=snapshot,
        error=None,
        fleet_processes=fleet or {},
        fleet_cgroup=fleet_cgroup,
    )


def failed_scrape(k: int, at_ns: int) -> Scrape:
    return Scrape(
        k=k,
        driver_before_ns=at_ns,
        driver_after_ns=at_ns + NS_PER_S,
        metrics=None,
        error="timed out",
        fleet_processes={},
        fleet_cgroup=None,
    )


def gate(spec: RunSpec, *, gated: bool = True) -> Gate:
    """Return a run's gate: every gate passed, or, for gated false, the limits gate failed."""
    expected = GatedLimits(
        cpu_limit_cores=2.0, memory_limit_bytes=4 * 2**30, memory_swap_limit_bytes=0
    )
    serving = spec.scenario_key != MODEL_BENCH_KEY
    return Gate(
        tests_passed=SuiteGate(
            passed=True,
            image_digest=IMAGE_DIGEST,
            code_digest=spec.code_digest,
            commit="0123abc",
            dirty=False,
            passed_count=100,
            failed_count=0,
            skipped_count=0,
            deselected_count=1,
            gate_file=f"gates/{IMAGE_DIGEST[7:19]}-{spec.code_digest[:8]}.json",
            output_sha256="0" * 64,
        ),
        image_built=ImageGate(built=True, digest=IMAGE_DIGEST, containers={"system": IMAGE_DIGEST}),
        limits_ok=LimitsGate(
            ok=gated,
            expected=expected,
            read=expected if gated else replace(expected, cpu_limit_cores=1.0),
        ),
        fresh=FreshGate(ok=True, new_containers=True, accepted_before_t0=0 if serving else None),
        config_ok=ConfigGate(ok=True, differences=()) if serving else None,
    )


@dataclass(frozen=True)
class SyntheticRun:
    """A serving run before it is written: what its files will hold."""

    config: RunConfig
    gate: Gate
    events: tuple[Event, ...]
    scrapes: tuple[Scrape, ...]
    # The samples of every robot that runs to the end, by robot id.
    samples: Mapping[str, RobotSamples]

    @property
    def spec(self) -> RunSpec:
        if self.config.spec is None:
            raise ValueError("a hand run has no spec")
        return self.config.spec

    @property
    def fleet(self) -> FleetConfig:
        if self.config.fleet is None:
            raise ValueError("a serving run has a fleet")
        return self.config.fleet

    @property
    def run(self) -> Run:
        """Return the run as bench.summarize reads it from its files."""
        return Run(
            spec=self.spec,
            run_id=run_id(self.spec),
            fleet=self.fleet,
            gated=self.gate.gated,
            events=self.events,
            scrapes=self.scrapes,
            samples=self.samples,
        )


@dataclass(frozen=True)
class _Robot:
    """One synthetic robot: when it runs, and everything it sent and received."""

    plan: RobotPlan
    start_ns: int
    # When it stops sending: its planned stop, or the end of the run.
    stop_ns: int
    # When the gateway closes its session: later than the stop for a robot that is killed.
    close_ns: int
    ticks: Rows
    commands: Mapping[str, Rows]


def _robot(
    plan: RobotPlan,
    scenario: Scenario,
    t0_ns: int,
    every_frames: Mapping[str, int],
    freshness_ms: Mapping[str, float],
) -> _Robot:
    start_ns = t0_ns + round(plan.start_s * NS_PER_S)
    stopped = plan.stop_s is not None
    stop_s = scenario.duration_s if plan.stop_s is None else plan.stop_s
    stop_ns = t0_ns + round(stop_s * NS_PER_S)
    captures = np.arange(start_ns, stop_ns, FRAME_PERIOD_NS)
    commands = {}
    for model in MODEL_KEYS:
        served = captures[:: every_frames[model]]
        receive = served + round(freshness_ms[model] * NS_PER_MS)
        kept = receive < stop_ns
        commands[model] = command_rows(
            model,
            served[kept],
            receive[kept],
            frame_numbers=(np.arange(captures.size)[:: every_frames[model]] + 1)[kept],
            fire_lag_ns=FIRE_LAG_NS,
            deliver_ns=DELIVER_NS,
            stage_us={**_FIXED_STAGE_US, "inference": round(SERVICE_MS[model] * 1000)},
        )
    return _Robot(
        plan=plan,
        start_ns=start_ns,
        stop_ns=stop_ns,
        close_ns=stop_ns + KILL_DETECT_NS if stopped else stop_ns,
        ticks=tick_rows(captures, fire_lag_ns=FIRE_LAG_NS),
        commands=commands,
    )


def _count(times_ns: npt.NDArray[np.int64], ts_ns: int) -> int:
    """Return how many of the ascending times are at or before ts_ns."""
    return int(np.searchsorted(times_ns, ts_ns, side="right"))


def _snapshot(
    ts_ns: int, robots: Sequence[_Robot], t0_ns: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the /metrics object the gateway of a synthetic run shows at ts_ns."""
    started = [robot for robot in robots if robot.start_ns <= ts_ns]
    connected = [robot for robot in started if ts_ns < robot.close_ns]
    closed = [robot for robot in started if ts_ns >= robot.close_ns]
    frames_in = {robot.plan.robot_id: _count(robot.ticks["send_ns"], ts_ns) for robot in started}
    sent = {
        robot.plan.robot_id: {
            model: _count(rows["receive_ns"], ts_ns) for model, rows in robot.commands.items()
        }
        for robot in started
    }
    frames = {model: sum(counts[model] for counts in sent.values()) for model in MODEL_KEYS}
    cpu_ms = {
        "gateway": round(sum(frames_in.values()) * GATEWAY_CPU_MS_PER_FRAME),
        **{f"worker_{model}": round(frames[model] * SERVICE_MS[model]) for model in MODEL_KEYS},
    }
    changes: dict[str, object] = {
        "connections.current": len(connected),
        "connections.accepted": len(started),
        "connections.closed.client_close": sum(robot.plan.stop_s is None for robot in closed),
        "connections.closed.abnormal": sum(robot.plan.stop_s is not None for robot in closed),
        "connections.bytes_in": sum(frames_in.values()) * FRAME_BYTES,
        "cgroup.cpu_usage_us": sum(cpu_ms.values()) * 1000,
        "cgroup.nr_periods": (ts_ns - t0_ns + 10 * NS_PER_S) // (100 * NS_PER_MS),
        "loop.tasks": 5 + 2 * len(connected),
        "loop.lag_ms": histogram(0.5, 50 * max(0, (ts_ns - t0_ns) // NS_PER_S)),
    }
    for name, used_ms in cpu_ms.items():
        fds = 20 + len(connected) if name == "gateway" else 10
        changes[f"processes.{name}"] = asdict(process(_PIDS[name], cpu_ms=used_ms, fds=fds))
    for model in MODEL_KEYS:
        changes[f"models.{model}.batches"] = frames[model]
        changes[f"models.{model}.frames"] = frames[model]
        changes[f"models.{model}.batch_size"] = {"1": frames[model]} if frames[model] else {}
        changes[f"models.{model}.batch_ms"] = histogram(SERVICE_MS[model], frames[model])
    snapshot = metrics(ts_ns, changes, config=config)
    for robot in started:
        robot_id = robot.plan.robot_id
        snapshot["robots"][robot_id] = robot_entry(
            connected=robot in connected,
            connected_s=(min(ts_ns, robot.close_ns) - robot.start_ns) / NS_PER_S,
            frames_in=frames_in[robot_id],
            commands_sent=sent[robot_id],
        )
    return snapshot


def _fleet_processes(ts_ns: int, robots: Sequence[_Robot], t0_ns: int) -> dict[str, ProcessStats]:
    """Return the fleet's processes that are alive at ts_ns, and the driver."""
    elapsed_ms = max(0, ts_ns - t0_ns) // NS_PER_MS
    processes = {"driver": process(_PIDS["driver"], cpu_ms=elapsed_ms // 100)}
    if any(robot.plan.stop_s is None for robot in robots):
        processes["fleet"] = process(_PIDS["fleet"], cpu_ms=elapsed_ms // 10)
    for index, robot in enumerate(robots):
        if robot.plan.stop_s is not None and ts_ns < robot.stop_ns:
            processes[f"fleet_{robot.plan.robot_id}"] = process(30 + index, cpu_ms=elapsed_ms // 50)
    return processes


def _events(scenario: Scenario, t0_ns: int) -> tuple[Event, ...]:
    """Return the events of a run in which everything happened at its planned time."""
    return (
        Event(EventName.T0, 0.0, t0_mono_ns=t0_ns),
        Event(EventName.SERVER_READY, -5.0),
        *planned_events(scenario),
        Event(EventName.FINAL_POLL, scenario.duration_s + 0.2, settled=True, waited_s=0.2),
    )


def synthetic_run(
    scenario: Scenario,
    spec: RunSpec,
    *,
    t0_ns: int = T0_NS,
    every_frames: Mapping[str, int] = EVERY_FRAMES,
    freshness_ms: Mapping[str, float] = FRESHNESS_MS,
    gated: bool = True,
) -> SyntheticRun:
    """Return a serving run of the scenario in which nothing went wrong.

    Every robot sends a frame each frame period from its start. Model m answers every
    every_frames[m]-th frame of each robot, freshness_ms[m] after its capture, each in a
    batch of one, so a robot's rate and freshness per model are known in advance. The
    scrapes come SCRAPE_LAG_NS after each whole second from -1 to the duration, and the final
    one after the fleet has left; their counters follow the robots. A robot with a stop time
    is killed there: it leaves no samples, and the gateway closes its session as abnormal
    KILL_DETECT_NS later.
    """
    robots = [_robot(plan, scenario, t0_ns, every_frames, freshness_ms) for plan in scenario.robots]
    return _played(scenario, spec, robots, t0_ns, gated)


def _played(
    scenario: Scenario, spec: RunSpec, robots: Sequence[_Robot], t0_ns: int, gated: bool
) -> SyntheticRun:
    """Return the run of the robots: the scrapes that follow them, its events and its samples."""
    end_ns = t0_ns + round(scenario.duration_s * NS_PER_S)
    times: list[tuple[int | None, int]] = [
        (k, t0_ns + k * NS_PER_S + SCRAPE_LAG_NS) for k in range(-1, round(scenario.duration_s) + 1)
    ]
    times.append((None, end_ns + FINAL_SCRAPE_AFTER_NS))
    split = spec.topology is Topology.SPLIT
    scrapes = tuple(
        scrape(
            k,
            _snapshot(ts_ns, robots, t0_ns, spec.server_config),
            fleet=_fleet_processes(ts_ns, robots, t0_ns),
            fleet_cgroup=_fleet_cgroup(ts_ns - t0_ns) if split else None,
        )
        for k, ts_ns in times
    )
    fleet = FleetConfig(
        scenario=scenario,
        frame_rate_hz=FRAME_RATE_HZ,
        models=MODEL_KEYS,
        stale_ms=DEFAULT_STALE_MS,
        watchdog_ms=DEFAULT_WATCHDOG_MS,
        planned_events=planned_events(scenario),
    )
    return SyntheticRun(
        config=RunConfig(
            spec=spec, run_id=run_id(spec), seed=spec.repeat, extra_compose_path=None, fleet=fleet
        ),
        gate=gate(spec, gated=gated),
        events=_events(scenario, t0_ns),
        scrapes=scrapes,
        samples={
            robot.plan.robot_id: samples(merged(*robot.commands.values()), robot.ticks)
            for robot in robots
            if robot.plan.stop_s is None
        },
    )


def worker_kill_run(
    scenario: Scenario, spec: RunSpec, *, outage_ns: int = WORKER_OUTAGE_NS
) -> SyntheticRun:
    """Return a serving run of a scenario that kills a worker, from which the service recovered.

    The run is synthetic_run's, but for the worker of the scenario's fault, which is killed
    KILL_LATE_NS after its planned time. Its model's commands that would have arrived in the
    outage_ns after the kill never come, and those frames are overwritten in their slots.
    From the first scrape after the kill the worker has restarted once, under RESTARTED_PID,
    and it is ready from the first scrape after the outage. The fault event is the last line
    of events.jsonl, where the runner appends it once the driver has finished.
    """
    if scenario.fault is None:
        raise ValueError(f"the {scenario.name} scenario kills no worker")
    model = scenario.fault.model
    kill_ns = T0_NS + round(scenario.fault.at_s * NS_PER_S) + KILL_LATE_NS
    robots = []
    for plan in scenario.robots:
        robot = _robot(plan, scenario, T0_NS, EVERY_FRAMES, FRESHNESS_MS)
        rows = robot.commands[model]
        lost = (rows["receive_ns"] >= kill_ns) & (rows["receive_ns"] < kill_ns + outage_ns)
        robots.append(replace(robot, commands={**robot.commands, model: rows[~lost]}))

    def restart(k: int | None, metrics: dict[str, Any]) -> None:
        ts_ns = metrics["ts_mono_ns"]
        if ts_ns < kill_ns:
            return
        worker = metrics["workers"][model]
        worker["restarts"] += 1
        worker["pid"] = RESTARTED_PID
        worker["state"] = "ready" if ts_ns >= kill_ns + outage_ns else "restarting"
        metrics["processes"][f"worker_{model}"]["pid"] = RESTARTED_PID

    run = edit_metrics(_played(scenario, spec, robots, T0_NS, gated=True), restart)
    killed = Event(
        EventName.FAULT, (kill_ns - T0_NS) / NS_PER_S, model=model, pid=_PIDS[f"worker_{model}"]
    )
    return replace(run, events=(*run.events, killed))


def _fleet_cgroup(elapsed_ns: int) -> CgroupStats:
    """Return the counters of the robots' own container, which uses a tenth of a core."""
    return CgroupStats(
        cpu_usage_us=max(0, elapsed_ns) // 10_000,
        cpu_user_us=None,
        cpu_system_us=None,
        nr_periods=None,
        nr_throttled=None,
        throttled_us=None,
        memory_current_bytes=200 * 2**20,
        memory_peak_bytes=None,
        memory_anon_bytes=150 * 2**20,
        memory_file_bytes=None,
        memory_sock_bytes=2**20,
        oom_kills=0,
    )


def edit_metrics(
    run: SyntheticRun, edit: Callable[[int | None, dict[str, Any]], None]
) -> SyntheticRun:
    """Return the run with every successful scrape's /metrics object changed by edit.

    edit gets a scrape's planned second, None for the final one, and a copy of its metrics
    to change in place.
    """
    scrapes = []
    for line in run.scrapes:
        if line.metrics is None:
            scrapes.append(line)
            continue
        changed = copy.deepcopy(dict(line.metrics))
        edit(line.k, changed)
        scrapes.append(replace(line, metrics=changed))
    return replace(run, scrapes=tuple(scrapes))


def _inside_environment() -> InsideEnvironment:
    return InsideEnvironment(
        cpu_model="Synthetic CPU",
        cpu_count_visible=2,
        machine="x86_64",
        kernel="6.6.0-synthetic",
        cgroup=LIMITS,
        python="3.14.0",
        packages={"numpy": "2.0.0", "onnxruntime": None},
        model_sha256={},
        start_utc="2026-01-01T00:00:00+00:00",
        frames=FrameSetStats(
            count=16, jpeg_quality=80, min_bytes=30_000, mean_bytes=40_000.0, max_bytes=50_000
        ),
    )


def _environment(spec: RunSpec) -> RunEnvironment:
    return RunEnvironment(
        inside=_inside_environment(),
        host_memory_total_bytes=32 * 2**30,
        engine="docker",
        engine_version="27.0.0",
        commit="0123abc",
        dirty=False,
        code_digest=spec.code_digest,
        image_digest=IMAGE_DIGEST,
        robots_cgroup=replace(LIMITS, cpu_limit_cores=None)
        if spec.topology is Topology.SPLIT
        else None,
    )


def _write(path: Path, text: str) -> None:
    # Bytes, so the file is the same on every platform and its hash with it.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode())


def _write_fleet_files(run_dir: Path, run: SyntheticRun) -> None:
    """Write what the scenario driver and the fleet write: plans, events, scrapes, samples."""
    scenario = run.fleet.scenario
    t0_ns = run.events[0].t0_mono_ns
    if t0_ns is None:
        raise ValueError("the first event of a run is its t0")
    for fleet_process in fleet_plans(scenario, t0_ns, run.config.seed):
        _write(
            run_dir / PLANS_DIRECTORY / f"{fleet_process.name}.json", fleet_process.plan.to_json()
        )
    _write(run_dir / INSIDE_ENVIRONMENT_FILE, _inside_environment().to_json())
    _write(run_dir / EVENTS_FILE, "".join(f"{event.to_json()}\n" for event in run.events))
    lines = "".join(f"{line.to_json()}\n" for line in run.scrapes)
    (run_dir / SCRAPES_FILE).write_bytes(gzip.compress(lines.encode(), mtime=0))
    (run_dir / SAMPLES_DIRECTORY).mkdir()
    for robot_id, robot in run.samples.items():
        meta = {
            "robot_id": robot_id,
            "dropped_ticks": robot.dropped_ticks,
            "dropped_commands": robot.dropped_commands,
        }
        np.savez(
            run_dir / SAMPLES_DIRECTORY / f"robot_{robot_id}.npz",
            commands=robot.commands,
            ticks=robot.ticks,
            meta=np.array(json.dumps(meta)),
        )


def _write_done(run_dir: Path) -> None:
    done = Done(files=done_files(run_files(run_dir)), summary_sha256=None)
    _write(run_dir / DONE_FILE, done.to_json())


def store_summary(run_dir: Path, summary: Mapping[str, object]) -> None:
    """Put a summary a test made up in a finished run, as bench.summarize stores its own.

    It carries the current summarizer digest unless the test gives it another.
    """
    document = {"summarizer_digest": summarizer_digest(), **summary}
    content = (json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (run_dir / SUMMARY_FILE).write_bytes(content)
    done = Done.from_json((run_dir / DONE_FILE).read_text(encoding="utf-8"))
    done = replace(done, summary_sha256=hashlib.sha256(content).hexdigest())
    _write(run_dir / DONE_FILE, done.to_json())


def write_run(run_dir: Path, run: SyntheticRun) -> None:
    """Write a finished serving run without a summary: its files, and a DONE that is valid."""
    _write(run_dir / CONFIG_FILE, run.config.to_json())
    _write_fleet_files(run_dir, run)
    _write(run_dir / ENVIRONMENT_FILE, _environment(run.spec).to_json())
    _write(run_dir / GATE_FILE, run.gate.to_json())
    _write_done(run_dir)


def write_finished_run(
    results: Path, run: SyntheticRun, summary: Mapping[str, object] | None = None
) -> Path:
    """Write a finished run under a results root with a current summary; return its directory.

    The summary is computed from the run's files, or is the given one: for a test of a
    reader that needs chosen values in it.
    """
    run_dir = run_directory(results, run.spec)
    write_run(run_dir, run)
    if summary is None:
        write_summary(run_dir)
    else:
        store_summary(run_dir, summary)
    return run_dir


def write_attempt(
    results: Path,
    run: SyntheticRun,
    summary: Mapping[str, object] | None = None,
    *,
    attempt: int = 1,
) -> Path:
    """Write a run that was renamed to an attempt and still holds its summary.json."""
    run_dir = write_finished_run(results, run, summary)
    return run_dir.rename(run_dir.with_name(f"{run_dir.name}.attempt{attempt}"))


def write_hand_run(results: Path, run: SyntheticRun, name: str = "manual") -> Path:
    """Write what the scenario driver leaves when it is started by hand; return its directory.

    The driver wrote config.json itself, with no run spec. Nothing wrote a gate, a DONE or a
    summary.
    """
    run_dir = results / RAW_DIRECTORY / run.fleet.scenario.name / name
    config = RunConfig(spec=None, run_id=None, seed=0, extra_compose_path=None, fleet=run.fleet)
    _write(run_dir / CONFIG_FILE, config.to_json())
    _write_fleet_files(run_dir, run)
    return run_dir


def _batch_report(
    model: str, threads: int, graph: str, sizes: Sequence[int], frame_ms: float
) -> dict[str, Any]:
    """Return a model_bench batch report whose batch of b frames takes b times frame_ms."""
    return {
        "kind": "model_bench.batch",
        "args": {"command": "batch", "model": model, "graph": graph, "threads": threads},
        "environment": {"cgroup": asdict(LIMITS)},
        "frames": {},
        "results": [
            {
                "batch_size": size,
                "iterations": 30,
                "wall_ms": _summary_ms(frame_ms * size / threads),
                "mean_cpu_ms": frame_ms * size,
            }
            for size in sizes
        ],
    }


def _summary_ms(mean_ms: float) -> dict[str, float]:
    """Return a bench.stats summary of times around a mean."""
    return {
        "count": 30,
        "mean": mean_ms,
        "min": mean_ms * 0.9,
        "p50": mean_ms * 0.98,
        "p95": mean_ms * 1.2,
        "p99": mean_ms * 1.3,
        "max": mean_ms * 1.5,
    }


def model_bench_reports(scale: float = 1.0) -> dict[str, dict[str, Any]]:
    """Return the reports of a model benchmark run by file name, every time multiplied by scale.

    The classifier is timed on its published graph, and depth on the published graph at
    batch size 1 and on the batched copy at every size, each at one and at two threads.
    """
    frame_ms = {"cls": 4.0 * scale, "depth": 60.0 * scale}
    reports: dict[str, dict[str, Any]] = {}
    for threads in (1, 2):
        reports[f"batch_cls_t{threads}_published"] = _batch_report(
            "cls", threads, "published", (1, 2, 4, 8), frame_ms["cls"]
        )
        reports[f"batch_depth_t{threads}_published"] = _batch_report(
            "depth", threads, "published", (1,), frame_ms["depth"]
        )
        reports[f"batch_depth_t{threads}_batched"] = _batch_report(
            "depth", threads, "batched", (1, 2, 4, 8), frame_ms["depth"] * 1.05
        )
    stage_ms = {
        "decode_full": 2.0,
        "decode_half": 1.0,
        **{f"preprocess_{model}_{size}": 0.5 for model in MODEL_KEYS for size in ("full", "half")},
        **{f"postprocess_{model}": 0.05 for model in MODEL_KEYS},
    }
    reports["stages"] = {
        "kind": "model_bench.stages",
        "args": {"command": "stages"},
        "environment": {"cgroup": asdict(LIMITS)},
        "frames": {},
        "results": {
            stage: {"wall_ms": _summary_ms(cost * scale), "mean_cpu_ms": cost * scale}
            for stage, cost in stage_ms.items()
        },
    }
    reports["dispatch"] = {
        "kind": "transport_bench.dispatch",
        "args": {"command": "dispatch", "loop": "uvloop"},
        "environment": {"cgroup": asdict(LIMITS)},
        "frames": {},
        "results": [
            {
                "batch_size": size,
                "round_trips": 2000,
                "wall_ms": _summary_ms(0.1 * size * scale),
                "gateway_cpu_us_per_round_trip": 40.0 * size * scale,
                "worker_cpu_us_per_round_trip": 30.0 * size * scale,
            }
            for size in (1, 2, 4, 8)
        ],
    }
    reports["websocket"] = {
        "kind": "transport_bench.websocket",
        "args": {"command": "websocket", "robots": 50, "fps": 30},
        "environment": {"cgroup": asdict(LIMITS)},
        "frames": {},
        "results": {
            "window_s": 20.0,
            "cpu_us_per_message": {"receiver": 60.0 * scale, "sender": 50.0 * scale},
            "sender": {"round_trip_ms": _summary_ms(0.4 * scale)},
            "receiver": {},
        },
    }
    return reports


def write_model_bench_run(
    results: Path,
    spec: RunSpec,
    reports: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    gated: bool = True,
) -> Path:
    """Write a finished model benchmark run with a current summary; return its directory."""
    run_dir = run_directory(results, spec)
    config = RunConfig(
        spec=spec, run_id=run_id(spec), seed=spec.repeat, extra_compose_path=None, fleet=None
    )
    _write(run_dir / CONFIG_FILE, config.to_json())
    _write(run_dir / ENVIRONMENT_FILE, _environment(spec).to_json())
    _write(run_dir / GATE_FILE, gate(spec, gated=gated).to_json())
    for name, report in (model_bench_reports() if reports is None else reports).items():
        _write(run_dir / REPORTS_DIRECTORY / f"{name}.json", json.dumps(report, indent=2) + "\n")
    _write_done(run_dir)
    write_summary(run_dir)
    return run_dir


def write_evidence(results: Path) -> None:
    """Write a results root as make report leaves it, of synthetic runs that nothing went wrong in.

    It holds a finished run of every entry of the fake set's evidence plan, one repeat, and
    the four runs of the fifo_vs_latest comparison, which a plot draws. Every summary is
    the summarizer's own, and the facts, the predictions and the ledger are those
    aggregation builds from them, so a reader of the root meets the names they write. The
    soak runs three robots for 30 s, which its summary and facts allow, and a scenario that
    kills a worker is worker_kill_run's.
    """
    (fifo,) = (candidate for candidate in CANDIDATES if candidate.name == "fifo_vs_latest")
    for spec in (*evidence_plan(1, ModelSet.FAKE, CODE_DIGEST), *compare_plan(fifo, CODE_DIGEST)):
        if spec.scenario_key == MODEL_BENCH_KEY:
            write_model_bench_run(results, spec)
            continue
        scenario = scenario_for(spec.scenario_key)
        if scenario.name == "soak":
            scenario = replace(sweep(3), name="soak")
        run = (
            synthetic_run(scenario, spec)
            if scenario.fault is None
            else worker_kill_run(scenario, spec)
        )
        write_finished_run(results, run)
    counting = Current(CODE_DIGEST, summarizer_digest())
    if build(results, results, ModelSet.FAKE, counting) != 0:
        raise ValueError(f"aggregation counted no run of {results}")
