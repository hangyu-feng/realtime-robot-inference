"""The robot fleet's command line: the plan, the frame set, the robots in one loop, the samples.

    python -m robot.fleet [--url URL] (--robots N --duration-s S | --plan PATH)
        [--frame-rate-hz 30] [--phase random|aligned] [--seed 0]
        [--models cls,depth] [--stale-ms cls=500,depth=150] [--watchdog-ms cls=1000,depth=300]
        [--uplink-delay-ms D] [--uplink-jitter-ms J]
        [--downlink-delay-ms D] [--downlink-jitter-ms J]
        [--samples-dir DIR] [--token T]

A plan file is the JSON object

    {"t0_mono_ns": int | null, "duration_s": float,
     "robots": [{"robot_id": "a", "start_s": 0.0, "stop_s": null}]}

whose times are seconds from the fleet's time zero; a stop_s of null runs to the end. The
process exits 0 when every robot completed its plan, 1 when one stopped on a close that is
final, and 2 for bad arguments.
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import math
import os
import random
import signal
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from pathlib import Path

from common import aio
from common.clock import Clock, MonotonicClock
from common.commands import MODEL_KEYS
from common.errors import ConfigError
from common.protocol import ROBOT_ID_PATTERN, STAGE_NAMES
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from common.transport_ws import ClientOptions, connect_ws
from robot.frames import DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from robot.robot import Robot
from robot.samples import VERDICTS, SampleLog
from robot.schedule import CaptureSchedule
from robot.settings import LinkSettings, RobotSettings

_LOG = logging.getLogger(__name__)

_MAX_ROBOTS = 99
# Time zero and the duration are each at most this, about 73 years, so every instant of a run
# fits the i64 of nanoseconds that a frame header carries.
_MAX_INSTANT_NS = 2**61
_MAX_RUN_S = _MAX_INSTANT_NS / 1e9
_PLAN_KEYS = frozenset(("t0_mono_ns", "duration_s", "robots"))
_ROBOT_KEYS = frozenset(("robot_id", "start_s", "stop_s"))


class Phase(StrEnum):
    """How the robots' capture schedules are offset from each other."""

    RANDOM = "random"  # each robot's first tick falls at a seeded random point of a period
    ALIGNED = "aligned"  # every robot ticks at the same instants: the worst case for bursts


@dataclass(frozen=True)
class RobotPlan:
    """When one robot runs, in seconds from time zero; a stop_s of None runs to the end."""

    robot_id: str
    start_s: float
    stop_s: float | None


@dataclass(frozen=True)
class FleetPlan:
    """The robots of one fleet process and their common timeline.

    t0_mono_ns is time zero on the monotonic clock; None takes the clock's reading once the
    frame set is built.
    """

    t0_mono_ns: int | None
    duration_s: float
    robots: tuple[RobotPlan, ...]


@dataclass(frozen=True)
class FleetArguments:
    """A validated command line."""

    settings: RobotSettings
    plan: FleetPlan
    phase: Phase
    seed: int
    samples_dir: Path | None


@dataclass(frozen=True)
class RobotRun:
    """One robot of the plan, placed on the monotonic clock."""

    robot_id: str
    start_ns: int
    stop_ns: int
    # Its first tick is its start plus its phase.
    schedule: CaptureSchedule
    # The robot's reconnect pauses, and the jitter of its link model, come from it.
    rng: random.Random


def _seconds(value: object, name: str) -> float:
    """Return a time of a plan, in seconds from time zero; ConfigError unless it is one."""
    # A comparison with NaN is false, the bounds refuse infinity, and an integer of any size
    # compares exactly, so the conversion below cannot overflow.
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not 0 <= value <= _MAX_RUN_S
    ):
        raise ConfigError(f"{name} must be a number from 0 to {_MAX_RUN_S:g}")
    return float(value)


def _duration(value: object, name: str) -> float:
    duration_s = _seconds(value, name)
    if duration_s == 0:
        raise ConfigError(f"{name} must be above 0")
    return duration_s


def read_plan(path: Path) -> FleetPlan:
    """Read and validate a plan file; ConfigError for anything wrong with it."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ConfigError(f"cannot read the plan {path}: {error}") from error
    except (ValueError, RecursionError) as error:
        # A JSONDecodeError and a UnicodeDecodeError are ValueErrors; deeply nested brackets
        # raise RecursionError.
        raise ConfigError(f"the plan {path} is not JSON: {error}") from error
    if not isinstance(document, dict) or document.keys() != _PLAN_KEYS:
        raise ConfigError(
            f"a plan must be an object with exactly the keys {', '.join(sorted(_PLAN_KEYS))}"
        )
    t0_mono_ns = document["t0_mono_ns"]
    if t0_mono_ns is not None and (
        type(t0_mono_ns) is not int or not 0 <= t0_mono_ns <= _MAX_INSTANT_NS
    ):
        raise ConfigError(f"t0_mono_ns must be null or an integer from 0 to {_MAX_INSTANT_NS}")
    duration_s = _duration(document["duration_s"], "duration_s")
    entries = document["robots"]
    if not isinstance(entries, list) or not entries:
        raise ConfigError("robots must be a list of at least one robot")
    robots = tuple(_read_robot(entry, duration_s) for entry in entries)
    if len({robot.robot_id for robot in robots}) != len(robots):
        raise ConfigError("robots must have distinct ids")
    return FleetPlan(t0_mono_ns, duration_s, robots)


def _read_robot(entry: object, duration_s: float) -> RobotPlan:
    if not isinstance(entry, dict) or entry.keys() != _ROBOT_KEYS:
        raise ConfigError(
            f"a robot must be an object with exactly the keys {', '.join(sorted(_ROBOT_KEYS))}"
        )
    robot_id = entry["robot_id"]
    # The rule of a hello's robot id, which is also part of the samples' file name.
    if not isinstance(robot_id, str) or ROBOT_ID_PATTERN.fullmatch(robot_id) is None:
        raise ConfigError("robot_id must be 1 to 64 ASCII letters, digits, '_', '-' or '.'")
    start_s = _seconds(entry["start_s"], f"{robot_id}: start_s")
    if start_s >= duration_s:
        raise ConfigError(f"{robot_id}: start_s must be below duration_s")
    if entry["stop_s"] is None:
        return RobotPlan(robot_id, start_s, None)
    stop_s = _seconds(entry["stop_s"], f"{robot_id}: stop_s")
    if not start_s < stop_s <= duration_s:
        raise ConfigError(f"{robot_id}: stop_s must be above start_s and at most duration_s")
    return RobotPlan(robot_id, start_s, stop_s)


def phases_ns(phase: Phase, seed: int, robot_count: int, period_ns: int) -> list[int]:
    """Return each robot's offset of its first tick from its start time, in plan order."""
    if phase is Phase.ALIGNED:
        return [0] * robot_count
    rng = random.Random(seed)
    return [rng.randrange(period_ns) for _ in range(robot_count)]


def robot_runs(arguments: FleetArguments, t0_ns: int) -> list[RobotRun]:
    """Return each robot of the plan placed on the monotonic clock from t0_ns, in plan order."""
    settings, plan = arguments.settings, arguments.plan
    period_ns = round(1e9 / settings.frame_rate_hz)
    phases = phases_ns(arguments.phase, arguments.seed, len(plan.robots), period_ns)
    runs: list[RobotRun] = []
    for robot, phase_ns in zip(plan.robots, phases, strict=True):
        start_ns = t0_ns + round(robot.start_s * 1e9)
        stop_s = plan.duration_s if robot.stop_s is None else robot.stop_s
        runs.append(
            RobotRun(
                robot.robot_id,
                start_ns,
                t0_ns + round(stop_s * 1e9),
                CaptureSchedule(start_ns + phase_ns, period_ns),
                # A run is repeatable for a seed, and each robot draws a stream of its own.
                random.Random(f"{arguments.seed}:{robot.robot_id}:backoff"),
            )
        )
    return runs


def sample_capacities(arguments: FleetArguments) -> tuple[int, int]:
    """Return the ticks, then the commands, that each robot's sample log holds."""
    settings = arguments.settings
    # Every tick of the run, with a second of slack for a late stop, and a command per tick
    # and model.
    tick_capacity = math.ceil(arguments.plan.duration_s * settings.frame_rate_hz) + math.ceil(
        settings.frame_rate_hz
    )
    return tick_capacity, tick_capacity * len(settings.models)


def _robot_count(text: str) -> int:
    count = int(text)
    if not 1 <= count <= _MAX_ROBOTS:
        raise argparse.ArgumentTypeError(f"must be from 1 to {_MAX_ROBOTS}")
    return count


def _model_list(text: str) -> tuple[str, ...]:
    return tuple(text.split(","))


def _thresholds(text: str) -> dict[str, float]:
    """Read MODEL=MS pairs separated by commas."""
    thresholds: dict[str, float] = {}
    for pair in text.split(","):
        model, separator, value = pair.partition("=")
        if not separator or model in thresholds:
            raise argparse.ArgumentTypeError("must be MODEL=MS pairs of distinct models")
        thresholds[model] = float(value)
    return thresholds


def _build_parser() -> argparse.ArgumentParser:
    defaults = RobotSettings()
    parser = argparse.ArgumentParser(
        prog="python -m robot.fleet",
        description="Run simulated robots against the gateway, and write their samples.",
    )
    parser.add_argument("--url", default=defaults.url, help="the gateway's WebSocket address")
    parser.add_argument("--robots", type=_robot_count, help="robots r01 to rNN for the duration")
    parser.add_argument("--duration-s", type=float, help="length of the run with --robots")
    parser.add_argument("--plan", type=Path, help="a plan file, instead of --robots")
    parser.add_argument("--frame-rate-hz", type=float, default=defaults.frame_rate_hz)
    parser.add_argument("--phase", type=Phase, choices=list(Phase), default=Phase.RANDOM)
    parser.add_argument("--seed", type=int, default=0, help="seeds phases, jitter and backoff")
    parser.add_argument("--models", type=_model_list, default=defaults.models, metavar="M,M")
    parser.add_argument(
        "--stale-ms",
        type=_thresholds,
        default={},
        metavar="MODEL=MS,...",
        help="stale thresholds, for the models named; the others keep their defaults",
    )
    parser.add_argument("--watchdog-ms", type=_thresholds, default={}, metavar="MODEL=MS,...")
    for direction in ("uplink", "downlink"):
        parser.add_argument(f"--{direction}-delay-ms", type=float, default=0.0)
        parser.add_argument(f"--{direction}-jitter-ms", type=float, default=0.0)
    parser.add_argument("--samples-dir", type=Path, help="write robot_<id>.npz here at the end")
    parser.add_argument("--token", help="the shared token, when the gateway asks for one")
    return parser


def parse_arguments(argv: Sequence[str] | None) -> FleetArguments:
    """Read and validate the command line; exits with status 2 on anything wrong."""
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if (arguments.plan is None) == (arguments.robots is None):
        parser.error("give either --robots with --duration-s, or --plan")
    if (arguments.robots is None) != (arguments.duration_s is None):
        parser.error("--robots and --duration-s go together, and --plan carries its own duration")
    models: tuple[str, ...] = arguments.models
    try:
        if arguments.plan is not None:
            plan = read_plan(arguments.plan)
        else:
            # The ids carry two digits, so they sort in robot order.
            plan = FleetPlan(
                None,
                _duration(arguments.duration_s, "--duration-s"),
                tuple(
                    RobotPlan(f"r{number:02d}", 0.0, None)
                    for number in range(1, arguments.robots + 1)
                ),
            )
        settings = RobotSettings(
            url=arguments.url,
            models=models,
            # A subscribed model that a flag does not name keeps its default threshold.
            stale_ms={
                **{model: ms for model, ms in DEFAULT_STALE_MS.items() if model in models},
                **arguments.stale_ms,
            },
            watchdog_ms={
                **{model: ms for model, ms in DEFAULT_WATCHDOG_MS.items() if model in models},
                **arguments.watchdog_ms,
            },
            frame_rate_hz=arguments.frame_rate_hz,
            token=arguments.token,
            uplink=LinkSettings(arguments.uplink_delay_ms, arguments.uplink_jitter_ms),
            downlink=LinkSettings(arguments.downlink_delay_ms, arguments.downlink_jitter_ms),
        )
    except ConfigError as error:
        parser.error(str(error))
    return FleetArguments(settings, plan, arguments.phase, arguments.seed, arguments.samples_dir)


def save_samples(
    samples_dir: Path,
    arguments: FleetArguments,
    runs: Sequence[RobotRun],
    logs: Mapping[str, SampleLog],
) -> None:
    """Write each robot's log to robot_<id>.npz, with what a reader needs to know of its run."""
    settings = arguments.settings
    # Every setting but the token.
    settings_meta = {
        **{
            field.name: getattr(settings, field.name)
            for field in fields(settings)
            if field.name != "token"
        },
        "stale_ms": dict(settings.stale_ms),
        "watchdog_ms": dict(settings.watchdog_ms),
        "uplink": asdict(settings.uplink),
        "downlink": asdict(settings.downlink),
    }
    for run in runs:
        log = logs[run.robot_id]
        log.save(
            samples_dir / f"robot_{run.robot_id}.npz",
            {
                "robot_id": run.robot_id,
                "seed": arguments.seed,
                "phase": arguments.phase.value,
                "settings": settings_meta,
                "schedule_start_ns": run.schedule.start_ns,
                "period_ns": run.schedule.period_ns,
                "model_keys": MODEL_KEYS,
                "stage_names": STAGE_NAMES,
                "verdict_names": [verdict.value for verdict in VERDICTS],
                "dropped_ticks": log.dropped_ticks,
                "dropped_commands": log.dropped_commands,
            },
        )


def exit_status(fatal_codes: Iterable[int | None]) -> int:
    """Return the process's exit status from the close code that stopped each robot for good.

    0 when every robot completed its plan, 1 when one stopped on a close that is final.
    """
    return 1 if any(code is not None for code in fatal_codes) else 0


async def _run_robots(robots: Sequence[Robot]) -> None:
    """Run every robot to the end of its plan, or to SIGTERM or SIGINT."""
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    # On either signal the robots send their last report and close with 1000.
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(signal_number, stop.set)
    async with asyncio.TaskGroup() as group:
        for robot in robots:
            group.create_task(robot.run(stop))


def _build_robots(
    settings: RobotSettings,
    runs: Sequence[RobotRun],
    logs: Mapping[str, SampleLog],
    jpegs: Sequence[bytes],
    clock: Clock,
) -> list[Robot]:
    """Return a robot for each run, with its sample log if it has one, in plan order."""
    connect = functools.partial(
        connect_ws,
        settings.url,
        ClientOptions(
            max_message_bytes=settings.max_message_bytes,
            rcvbuf_bytes=settings.rcvbuf_bytes,
            sndbuf_bytes=settings.sndbuf_bytes,
            notsent_lowat_bytes=settings.notsent_lowat_bytes,
            open_timeout_s=settings.connect_timeout_ms / 1000,
            close_timeout_s=settings.close_timeout_ms / 1000,
        ),
    )
    return [
        Robot(
            run.robot_id,
            settings,
            run.schedule,
            jpegs,
            run.start_ns,
            run.stop_ns,
            connect,
            clock,
            run.rng,
            logs.get(run.robot_id),
        )
        for run in runs
    ]


def _run_fleet(arguments: FleetArguments, clock: Clock) -> int:
    """Build the frame set and the robots, run them, write the samples; return the exit status."""
    settings, plan, samples_dir = arguments.settings, arguments.plan, arguments.samples_dir
    # ASSUMPTION: A5. The frame set is encoded once, here, before time zero, and every robot
    # cycles through it.
    frames = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    # Allocated before time zero too, because committing the arrays takes time. Without
    # --samples-dir no robot keeps a log.
    tick_capacity, command_capacity = sample_capacities(arguments)
    logs: dict[str, SampleLog] = (
        {}
        if samples_dir is None
        else {robot.robot_id: SampleLog(tick_capacity, command_capacity) for robot in plan.robots}
    )
    # ASSUMPTION: A20. A time zero given by the plan is a reading of the monotonic clock that
    # every process on this host shares, so several fleet processes and their runner can
    # agree on it.
    t0_ns = clock.now_ns() if plan.t0_mono_ns is None else plan.t0_mono_ns
    runs = robot_runs(arguments, t0_ns)
    robots = _build_robots(settings, runs, logs, frames.jpegs, clock)
    stats = frames.stats()
    _LOG.info(
        "fleet: %d robots for %g s to %s; %d frames of %d to %d bytes",
        len(robots),
        plan.duration_s,
        settings.url,
        stats.count,
        stats.min_bytes,
        stats.max_bytes,
    )
    aio.run(_run_robots(robots))
    # Written here, after the loop has ended: nothing on the frame path writes a file.
    # ASSUMPTION: A11. A robot that a scenario kills runs as its own process with a one-robot
    # plan and dies by SIGKILL, with no clean close, so it never gets here and leaves no file.
    if samples_dir is not None:
        save_samples(samples_dir, arguments, runs, logs)
    return exit_status(robot.fatal_code for robot in robots)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = parse_arguments(argv)
    samples_dir = arguments.samples_dir
    if samples_dir is not None:
        # Checked before the frame set is built and any robot starts: np.savez creates no
        # directory, and a run would otherwise lose its samples at its very end.
        try:
            samples_dir.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            print(f"robot.fleet: cannot create --samples-dir: {error}", file=sys.stderr)
            return 2
        if not os.access(samples_dir, os.W_OK | os.X_OK):
            print(f"robot.fleet: cannot write to --samples-dir {samples_dir}", file=sys.stderr)
            return 2
    logging.basicConfig(
        level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s"
    )
    # The library logs every connection it opens at INFO.
    logging.getLogger("websockets").setLevel(logging.WARNING)
    return _run_fleet(arguments, MonotonicClock())


if __name__ == "__main__":
    sys.exit(main())
