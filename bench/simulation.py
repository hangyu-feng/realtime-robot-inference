"""A discrete-event simulation of the gateway's scheduling core under a simulated clock.

It drives the real registry, mailbox, downlink queues and schedulers of cloud/, with the
scheduler settings and pacing share that cloud.scheduler derives, and each robot's own command
control from robot/, through the open and close sequences that cloud.registry.RobotRegistry
states. Batch times come from a calibration table, and every other handling time from fixed
overheads. Time is integer nanoseconds of one clock that the robots and the gateway share.
Events wait in a heap ordered by (time_ns, sequence number), so a run is deterministic for a
given seed; nothing sleeps and no real clock is read.

It does not model the kernel freezing the container when the demand passes the CPU quota. It
reports the predicted total CPU demand instead, so that the caller can flag that case.
"""

from __future__ import annotations

import heapq
import itertools
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from fractions import Fraction
from functools import partial

from bench.scenarios import Scenario, Window
from cloud.config import ServerConfig
from cloud.downlink import DownlinkQueue, QueuedCommand
from cloud.mailbox import Frame, Mailbox
from cloud.registry import ModelCounters, RobotRecord, RobotRegistry
from cloud.scheduler import (
    Batch,
    Dispatch,
    Idle,
    ModelPlan,
    ModelScheduler,
    WaitUntil,
    other_demand_cores,
    pacing_share,
    plan_model,
)
from robot.control import CommandOutcome, ModelControl, Verdict
from robot.schedule import CaptureSchedule

_NS_PER_S = 1_000_000_000
_NS_PER_MS = 1_000_000
# Nothing decodes a simulated frame: the start and end markers of a JPEG stand in for one.
_PLACEHOLDER_JPEG = b"\xff\xd8\xff\xd9"


@dataclass(frozen=True)
class ServiceTable:
    """The wall and CPU time of one whole batch per batch size, of one model at its threads."""

    wall_ns: Mapping[int, int]
    cpu_ns: Mapping[int, int]

    def at(self, batch_size: int) -> tuple[int, int]:
        """Return the wall and CPU time of one batch of this size.

        A size the table lacks, between two sizes it has, is interpolated linearly between the
        nearest of them.
        """
        return _interpolate(self.wall_ns, batch_size), _interpolate(self.cpu_ns, batch_size)


def _interpolate(times_ns: Mapping[int, int], batch_size: int) -> int:
    lower = max(size for size in times_ns if size <= batch_size)
    upper = min(size for size in times_ns if size >= batch_size)
    if lower == upper:
        return times_ns[lower]
    fraction = Fraction(batch_size - lower, upper - lower)
    return times_ns[lower] + round((times_ns[upper] - times_ns[lower]) * fraction)


@dataclass(frozen=True)
class Overheads:
    """The handling times outside the worker, the same for every frame and command."""

    transport_up_ns: int  # robot send to gateway receive
    dispatch_request_ns: int  # dispatch to the whole batch at the worker
    dispatch_result_ns: int  # results sent to results at the gateway
    deliver_ns: int  # results at the gateway to the command at the robot


class PhaseMode(StrEnum):
    """Where each robot's capture ticks fall within the frame period (DESIGN.md 11.2)."""

    RANDOM = "random"  # uniform within one period, drawn from the seed: the default
    ALIGNED = "aligned"  # every robot captures at the same instants


@dataclass(frozen=True)
class RobotWindow:
    """What one robot received from one model within one window of the run.

    received_ns holds the receive time of every command, whatever its verdict, so its length is
    the number of commands received, and freshness_ns their freshness in the same order.
    commands_per_s is that number over the window's length. command_age_peak_ns holds the peak
    age of each applied command that a later one replaced (CommandOutcome.replaced_age_ns).
    command_gap_max_ns counts the open intervals too: from the later of the window's start and
    the robot's start to its first command, and from its last command to the earlier of the
    window's end and the robot's stop, so a robot that receives nothing has a gap as long as its
    time in the window.
    """

    received_ns: tuple[int, ...]
    commands_per_s: float
    freshness_ns: tuple[int, ...]
    command_age_peak_ns: tuple[int, ...]
    stale: int
    out_of_order: int
    command_gap_max_ns: int


@dataclass(frozen=True)
class PacingShare:
    """The pacing share set at at_ns, after a robot started or stopped; None when not paced."""

    at_ns: int
    share_cores: float | None


@dataclass(frozen=True)
class ModelRun:
    """One model's scheduler over the run.

    batch_size, busy_ns, paced_ns and idle_ns come from its ModelStats. cpu_ns is the CPU time of
    every batch dispatched, from the table, and shares holds the share set after each start and
    stop of a robot, in order.
    """

    plan: ModelPlan
    batch_size: Mapping[int, int]
    busy_ns: int
    paced_ns: int
    idle_ns: int
    cpu_ns: int
    shares: tuple[PacingShare, ...]


@dataclass(frozen=True)
class FinalCounts:
    """A robot's counters for one model at the end of the run, and its frames still pending."""

    counters: ModelCounters
    pending_count: int


@dataclass(frozen=True)
class SimulationResult:
    """What one run predicts.

    windows is keyed by window name, model and robot id, and holds every robot that ran within
    the window. final is keyed by robot id and model. cpu_demand_cores is the predicted total
    CPU demand: the workers' CPU time, and the measured cost of the frame I/O for every frame
    sent, over the run's duration.
    """

    windows: Mapping[str, Mapping[str, Mapping[str, RobotWindow]]]
    models: Mapping[str, ModelRun]
    final: Mapping[str, Mapping[str, FinalCounts]]
    cpu_demand_cores: float


@dataclass(frozen=True)
class _Robot:
    """One robot of the scenario on the simulated clock."""

    robot_id: str
    start_ns: int
    # Its stop, or the end of the run for a robot that runs to the end. No event at the end of
    # the run is processed, so such a robot is never closed and keeps its pending frames.
    stop_ns: int
    # Ticks from its start plus its phase. It sends each frame at its tick: no fire lag.
    ticks: CaptureSchedule


@dataclass(frozen=True)
class _Session:
    """A running robot: its record, its connection's downlink queue and its control per model."""

    record: RobotRecord
    downlink: DownlinkQueue
    controls: dict[str, ModelControl]


@dataclass(frozen=True)
class _Received:
    """A command the robot received, and what its control made of it."""

    received_ns: int
    outcome: CommandOutcome


def simulate(
    config: ServerConfig,
    scenario: Scenario,
    tables: Mapping[str, ServiceTable],
    overheads: Overheads,
    io_cpu_ms_per_frame: float,
    quota_cores: float | None,
    phase_mode: PhaseMode,
    seed: int,
) -> SimulationResult:
    """Run the scenario and return what it predicts.

    tables holds every model's service table, with a time for each size of its batch_sizes;
    other sizes serve the interpolation. io_cpu_ms_per_frame is the measured CPU time of the
    frame I/O per frame, and enters only the predicted CPU demand: the pacing share comes from
    config.io_cpu_ms_per_frame, as in the gateway. Without a CPU quota (quota_cores None)
    nothing is paced.
    """
    for model, model_config in config.models.items():
        table = tables.get(model)
        if table is None or not all(
            size in table.wall_ns and size in table.cpu_ns for size in model_config.batch_sizes
        ):
            raise ValueError(
                f"the service table of {model} needs a time for each of its batch sizes "
                f"{list(model_config.batch_sizes)}"
            )
    end_ns = round(scenario.duration_s * _NS_PER_S)
    period_ns = round(_NS_PER_S / config.frame_rate_hz)
    phases = random.Random(seed)
    robots: list[_Robot] = []
    for plan in scenario.robots:
        start_ns = round(plan.start_s * _NS_PER_S)
        stop_ns = end_ns if plan.stop_s is None else min(end_ns, round(plan.stop_s * _NS_PER_S))
        phase_ns = phases.randrange(period_ns) if phase_mode is PhaseMode.RANDOM else 0
        ticks = CaptureSchedule(start_ns + phase_ns, period_ns)
        robots.append(_Robot(plan.robot_id, start_ns, stop_ns, ticks))
    simulation = _Simulation(
        config, tables, overheads, io_cpu_ms_per_frame, quota_cores, end_ns, robots
    )
    simulation.run()
    return simulation.result(scenario.windows)


class _Simulation:
    """The state of one run, with a method per kind of event."""

    def __init__(
        self,
        config: ServerConfig,
        tables: Mapping[str, ServiceTable],
        overheads: Overheads,
        io_cpu_ms_per_frame: float,
        quota_cores: float | None,
        end_ns: int,
        robots: Sequence[_Robot],
    ) -> None:
        self._config = config
        self._overheads = overheads
        self._io_cpu_ms_per_frame = io_cpu_ms_per_frame
        self._quota_cores = quota_cores
        self._end_ns = end_ns
        self._robots = robots
        self._stale_ns = {
            model: round(model_config.stale_ms * _NS_PER_MS)
            for model, model_config in config.models.items()
        }
        self._watchdog_ns = {
            model: round(model_config.watchdog_ms * _NS_PER_MS)
            for model, model_config in config.models.items()
        }
        self._cpu_ns_per_frame = {model: tables[model].cpu_ns[1] for model in config.models}
        self._registry = RobotRegistry(
            round(config.robot_retention_s * _NS_PER_S), config.max_robot_entries
        )
        self._mailbox = Mailbox(config.queue_policy, config.fifo_capacity_frames)
        # The settings the gateway derives from a worker's calibration, which holds the
        # configured batch sizes only. The overheads time the events and set no scheduler field.
        self._plans = {
            model: plan_model(
                model,
                config,
                {size: tables[model].wall_ns[size] for size in model_config.batch_sizes},
            )
            for model, model_config in config.models.items()
        }
        # ASSUMPTION: A14. The workers are warm before the first robot connects, so each is free
        # from the start of the run.
        self._schedulers = {
            model: ModelScheduler(plan.scheduler, self._mailbox, 0)
            for model, plan in self._plans.items()
        }
        # The wall and CPU time of every batch size a scheduler can dispatch.
        self._service = {
            model: {size: tables[model].at(size) for size in range(1, plan.scheduler.batch_cap + 1)}
            for model, plan in self._plans.items()
        }
        # Whether each scheduler's last decision was Idle, so that a frame arrival calls next().
        self._idle = dict.fromkeys(config.models, False)
        self._shares: dict[str, list[PacingShare]] = {model: [] for model in config.models}
        self._events: list[tuple[int, int, Callable[[int], None]]] = []
        self._sequence = itertools.count()
        self._sessions: dict[str, _Session] = {}
        # Kept after a robot stops, and after the registry drops its record.
        self._records: dict[str, RobotRecord] = {}
        self._received: dict[str, dict[str, list[_Received]]] = {}

    def run(self) -> None:
        # Starts and stops go on the heap first, so at any instant they come before every other
        # event: a robot is running at its start, and stopped at its stop.
        for robot in self._robots:
            self._at(robot.start_ns, partial(self._start, robot))
            self._at(robot.stop_ns, partial(self._stop, robot))
        for model in self._schedulers:
            self._next(model, 0)
        while self._events and self._events[0][0] < self._end_ns:
            time_ns, _, event = heapq.heappop(self._events)
            event(time_ns)

    def result(self, windows: Sequence[Window]) -> SimulationResult:
        models = {model: self._model_run(model) for model in self._config.models}
        # A robot sends one frame at each of its ticks before its stop.
        frames_sent = sum(robot.ticks.latest_due(robot.stop_ns - 1) + 1 for robot in self._robots)
        io_cpu_ns = frames_sent * self._io_cpu_ms_per_frame * _NS_PER_MS
        cpu_ns = sum(run.cpu_ns for run in models.values()) + io_cpu_ns
        return SimulationResult(
            windows={
                window.name: {model: self._window(window, model) for model in self._config.models}
                for window in windows
            },
            models=models,
            final={
                robot_id: {
                    model: FinalCounts(counters, self._mailbox.pending_count(robot_id, model))
                    for model, counters in record.models.items()
                }
                for robot_id, record in self._records.items()
            },
            cpu_demand_cores=cpu_ns / self._end_ns,
        )

    def _at(self, time_ns: int, event: Callable[[int], None]) -> None:
        heapq.heappush(self._events, (time_ns, next(self._sequence), event))

    def _start(self, robot: _Robot, now_ns: int) -> None:
        """Open the robot's session, and put its first frame on the uplink."""
        # ASSUMPTION: A2. Every frame is offered to both models: every robot subscribes to every
        # model, and declares the configured thresholds.
        subscribed = tuple(self._config.models)
        record = self._registry.connect(
            robot.robot_id,
            subscribed,
            self._stale_ns,
            self._watchdog_ns,
            robot.ticks.period_ns,
            now_ns,
        )
        self._mailbox.open(record)
        controls = {
            model: ModelControl(
                record.session, self._stale_ns[model], self._watchdog_ns[model], now_ns
            )
            for model in subscribed
        }
        downlink = DownlinkQueue(self._config.downlink_queue_commands)
        self._sessions[robot.robot_id] = _Session(record, downlink, controls)
        self._records[robot.robot_id] = record
        self._received[robot.robot_id] = {model: [] for model in subscribed}
        self._set_pacing_shares(now_ns)
        self._send(robot, 0)

    def _stop(self, robot: _Robot, now_ns: int) -> None:
        """Close the robot's session."""
        session = self._sessions.pop(robot.robot_id)
        self._mailbox.close(robot.robot_id)
        session.downlink.drop_all()
        self._registry.disconnect(robot.robot_id, now_ns)
        self._set_pacing_shares(now_ns)

    def _set_pacing_shares(self, now_ns: int) -> None:
        """Set each model's pacing share for the robots connected now, as the gateway does."""
        for model, model_config in self._config.models.items():
            share_cores: float | None = None
            if model_config.paced and self._quota_cores is not None:
                share_cores = pacing_share(
                    self._quota_cores,
                    self._config.pacing_headroom_cores,
                    self._config.pacing_min_share_cores,
                    model_config.threads,
                    other_demand_cores(
                        self._config,
                        self._registry.records(),
                        self._cpu_ns_per_frame,
                        self._quota_cores,
                    ),
                )
            self._schedulers[model].set_pacing_share(share_cores)
            self._shares[model].append(PacingShare(now_ns, share_cores))

    def _send(self, robot: _Robot, frame_number: int) -> None:
        """Put the frame on the uplink if the robot sends it: it does at each tick while it runs."""
        capture_ns = robot.ticks.capture_ns(frame_number)
        if capture_ns < robot.stop_ns:
            self._at(
                capture_ns + self._overheads.transport_up_ns,
                partial(self._arrive, robot, frame_number, capture_ns),
            )

    def _arrive(self, robot: _Robot, frame_number: int, capture_ns: int, now_ns: int) -> None:
        """Put an arriving frame in the mailbox, wake every idle scheduler, and send the next."""
        if robot.robot_id not in self._sessions:
            # The robot stopped while its frame travelled, and it sends no more.
            return
        # ASSUMPTION: A20. The robots and the gateway read one clock, so the uplink is exact.
        uplink_ns = self._overheads.transport_up_ns
        frame = Frame(frame_number, capture_ns, now_ns, 0, uplink_ns, _PLACEHOLDER_JPEG)
        self._mailbox.put(robot.robot_id, frame)
        for model, idle in self._idle.items():
            if idle:
                self._next(model, now_ns)
        self._send(robot, frame_number + 1)

    def _next(self, model: str, now_ns: int) -> None:
        """Ask the model's scheduler what its free worker does now, and schedule what follows."""
        decision = self._schedulers[model].next(now_ns)
        self._idle[model] = isinstance(decision, Idle)
        if isinstance(decision, Dispatch):
            wall_ns, _ = self._service[model][len(decision.batch.items)]
            overheads = self._overheads
            done_ns = (
                now_ns + overheads.dispatch_request_ns + wall_ns + overheads.dispatch_result_ns
            )
            self._at(done_ns, partial(self._complete, model, decision.batch))
        elif isinstance(decision, WaitUntil):
            self._at(decision.deadline_ns, partial(self._next, model))

    def _complete(self, model: str, batch: Batch, now_ns: int) -> None:
        """Book the finished batch, write each deliverable command at once, and free the worker."""
        wall_ns, cpu_ns = self._service[model][len(batch.items)]
        items = [
            batch.items[index]
            for index in self._schedulers[model].complete(batch, (), wall_ns, cpu_ns, now_ns)
        ]
        # complete() delivers only to a connected record in the item's session, which here is a
        # running robot.
        sessions = [self._sessions[item.record.robot_id] for item in items]
        # Every command is pushed before any queue is popped, as cloud.registry.RobotRegistry
        # tells every caller: the second identity holds again once the last one is pushed.
        for item, session in zip(items, sessions, strict=True):
            session.downlink.push(QueuedCommand(b"", item.record.models[model]))
        # The simulated connection is never slow, so a command is written as soon as it is
        # queued. It writes no bytes, and the real queue still counts the command and its pop
        # measures the service gap.
        for item, session in zip(items, sessions, strict=True):
            session.downlink.pop(now_ns)
            self._at(
                now_ns + self._overheads.deliver_ns,
                partial(self._deliver, item.record.robot_id, model, item.session, item.frame),
            )
        self._next(model, now_ns)

    def _deliver(self, robot_id: str, model: str, session: int, frame: Frame, now_ns: int) -> None:
        """Hand a command to its robot's control, unless the robot stopped while it travelled."""
        running = self._sessions.get(robot_id)
        if running is None:
            return
        control = running.controls[model]
        outcome = control.on_command(session, frame.frame_number, frame.capture_ns, now_ns)
        self._received[robot_id][model].append(_Received(now_ns, outcome))

    def _model_run(self, model: str) -> ModelRun:
        stats = self._schedulers[model].stats
        service = self._service[model]
        return ModelRun(
            plan=self._plans[model],
            batch_size=dict(stats.batch_size),
            busy_ns=stats.busy_ns,
            paced_ns=stats.paced_ns,
            idle_ns=stats.idle_ns,
            # Booked at dispatch, as the scheduler books its batch and frame counts.
            cpu_ns=sum(count * service[size][1] for size, count in stats.batch_size.items()),
            shares=tuple(self._shares[model]),
        )

    def _window(self, window: Window, model: str) -> dict[str, RobotWindow]:
        """Return what each robot that ran within the window received from the model."""
        start_ns = round(window.start_s * _NS_PER_S)
        end_ns = round(window.end_s * _NS_PER_S)
        robots: dict[str, RobotWindow] = {}
        for robot in self._robots:
            from_ns = max(start_ns, robot.start_ns)
            to_ns = min(end_ns, robot.stop_ns)
            if from_ns < to_ns:
                commands = [
                    command
                    for command in self._received[robot.robot_id][model]
                    if start_ns <= command.received_ns < end_ns
                ]
                robots[robot.robot_id] = _robot_window(commands, from_ns, to_ns, end_ns - start_ns)
        return robots


def _robot_window(
    commands: Sequence[_Received], from_ns: int, to_ns: int, window_ns: int
) -> RobotWindow:
    """Summarize the commands a robot received while it ran from from_ns to to_ns in a window."""
    received_ns = tuple(command.received_ns for command in commands)
    outcomes = [command.outcome for command in commands]
    return RobotWindow(
        received_ns=received_ns,
        # ASSUMPTION: A8. Throughput is the commands a robot receives from a model per second,
        # whatever their verdict; the stale ones are counted apart.
        commands_per_s=len(commands) * _NS_PER_S / window_ns,
        freshness_ns=tuple(outcome.freshness_ns for outcome in outcomes),
        command_age_peak_ns=tuple(
            outcome.replaced_age_ns for outcome in outcomes if outcome.replaced_age_ns is not None
        ),
        stale=sum(outcome.verdict is Verdict.STALE for outcome in outcomes),
        out_of_order=sum(outcome.verdict is Verdict.OUT_OF_ORDER for outcome in outcomes),
        command_gap_max_ns=max(
            later - earlier for earlier, later in itertools.pairwise((from_ns, *received_ns, to_ns))
        ),
    )
