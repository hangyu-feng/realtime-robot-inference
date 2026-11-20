"""Drivers of the scheduling core for its property tests.

World stands in for the gateway: it holds the registry, the mailbox, one scheduler per model and
the open connections, takes every step as the docstring of cloud.registry.RobotRegistry sets out,
and records what each step observed. SchedulingMachine interleaves those steps at random.
simulate() runs robots that each send one frame per period, and may leave and come back, against
one model's scheduler, which it calls exactly when the gateway does.

Times are small integers of nanoseconds, so that coincidences are common: a frame arriving at a
dispatch, a hold ending at a call, an age at a threshold.
"""

import heapq
import itertools
from collections.abc import Mapping
from dataclasses import astuple, dataclass
from typing import ClassVar

from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, precondition, rule

from cloud.config import QueuePolicy
from cloud.downlink import DownlinkQueue, QueuedCommand
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRecord, RobotRegistry
from cloud.scheduler import (
    Batch,
    Decision,
    Dispatch,
    Idle,
    ModelScheduler,
    SchedulerConfig,
    WaitUntil,
)
from common.commands import MODEL_KEYS

MACHINE_SETTINGS = settings(max_examples=100, stateful_step_count=50)

# More ids than a table of the machine has entries, so that the registry drops records.
ROBOT_IDS = ("a", "b", "c", "d", "e")
COMMAND = b"command"
JPEG = b"\xff\xd8"
# Only other_demand_cores reads a record's frame period, and no test here calls it.
FRAME_PERIOD_NS = 30

SUBSCRIPTIONS: tuple[tuple[str, ...], ...] = (*((model,) for model in MODEL_KEYS), MODEL_KEYS)
# What a hello declares: the models it subscribes to, each with its stale threshold.
DECLARATIONS = st.sampled_from(SUBSCRIPTIONS).flatmap(
    lambda subscribed: st.fixed_dictionaries({model: st.integers(10, 150) for model in subscribed})
)
# The fire lag and the uplink of a frame. With a model's service time and allowance they make
# up the part of a frame's age that the skip rule cannot shorten, mostly below the thresholds of
# DECLARATIONS, so that a wait decides whether a frame is doomed.
FRAME_AGES_NS = st.tuples(st.integers(0, 40), st.integers(0, 20))
# 0.3 and most drawn floats are not binary fractions with few digits, so the CPU time over the
# share is rarely a whole number of nanoseconds.
SHARES_CORES = st.sampled_from((0.3, 0.5, 1.0, 2.0)) | st.floats(min_value=0.25, max_value=4.0)
OPTIONAL_SHARES_CORES = st.none() | SHARES_CORES
QUEUE_POLICIES = st.sampled_from(QueuePolicy)
SKIP_RULES = st.booleans()


@dataclass(frozen=True)
class ModelSetup:
    """One model's scheduler settings and the pacing share of its worker, None when unpaced."""

    config: SchedulerConfig
    share_cores: float | None


@dataclass(frozen=True)
class CoreSetup:
    """The settings of a world."""

    policy: QueuePolicy
    fifo_capacity_frames: int
    models: tuple[ModelSetup, ...]
    retention_ns: int
    max_entries: int
    downlink_queue_commands: int


@dataclass
class Connection:
    """An open session as the gateway holds it, and the newest frame put in it."""

    record: RobotRecord
    downlink: DownlinkQueue
    newest: Frame | None = None


@dataclass(frozen=True)
class Decided:
    """A call of ModelScheduler.next() that returned a decision, and the state it decided on."""

    model: str
    now_ns: int
    # The robots with a frame pending for the model just before the call.
    pending: tuple[RobotRecord, ...]
    # The newest frame put in each open session by then, by session number.
    newest: Mapping[int, Frame]
    decision: Decision


@dataclass(frozen=True)
class Finished:
    """A batch taken off its worker by complete() or fail(), and what the call changed."""

    model: str
    batch: Batch
    # The items that produced no command: those named to complete(), or every item for fail().
    failed: frozenset[int]
    # The CPU time complete() was given. fail() is given none, so 0.
    cpu_ns: int
    delivered: tuple[int, ...]
    open_sessions: frozenset[int]
    # The records whose counters the call, with the pushes that follow it, changed.
    touched: tuple[RobotRecord, ...]


class World:
    """The scheduling core as the gateway drives it, with a record of what each step observed.

    Each method is one synchronous step of a caller. Only open sessions put frames and close,
    so the check a connection makes against the registry before either always passes here.
    """

    def __init__(self, setup: CoreSetup) -> None:
        self.setup = setup
        self.registry = RobotRegistry(setup.retention_ns, setup.max_entries)
        self.mailbox = Mailbox(setup.policy, setup.fifo_capacity_frames)
        self.schedulers: dict[str, ModelScheduler] = {}
        for model_setup in setup.models:
            scheduler = ModelScheduler(model_setup.config, self.mailbox, 0)
            scheduler.set_pacing_share(model_setup.share_cores)
            self.schedulers[model_setup.config.model] = scheduler
        self.connections: dict[str, Connection] = {}
        # Every record connect() returned, by identity, also those the registry dropped since:
        # a batch item can still hold one, and its counters must still add up.
        self.records: dict[int, RobotRecord] = {}
        # The batch each busy worker holds, as the gateway knows it from the dispatch.
        self.in_flight: dict[str, Batch] = {}
        self.decisions: list[Decided] = []
        self.finishes: list[Finished] = []

    def hello(self, robot_id: str, stale_ns: Mapping[str, int], now_ns: int) -> None:
        """Open a session subscribed to the models of stale_ns, first closing the id's open one."""
        # ASSUMPTION: A22. A second connection with the same robot id replaces the first session.
        if robot_id in self.connections:
            self.close(robot_id, now_ns)
        watchdog_ns = {model: 2 * threshold_ns for model, threshold_ns in stale_ns.items()}
        record = self.registry.connect(
            robot_id, tuple(stale_ns), stale_ns, watchdog_ns, FRAME_PERIOD_NS, now_ns
        )
        self.mailbox.open(record)
        downlink = DownlinkQueue(self.setup.downlink_queue_commands)
        self.connections[robot_id] = Connection(record, downlink)
        self.records.setdefault(id(record), record)

    def close(self, robot_id: str, now_ns: int) -> None:
        self.mailbox.close(robot_id)
        self.connections.pop(robot_id).downlink.drop_all()
        self.registry.disconnect(robot_id, now_ns)

    def put(self, robot_id: str, frame: Frame) -> None:
        """Put a frame of the robot's open session, numbered above the session's newest."""
        self.mailbox.put(robot_id, frame)
        self.connections[robot_id].newest = frame

    def decide(self, model: str, now_ns: int) -> Decision:
        """Ask what the model's free worker does at now_ns, and hand it the batch of a dispatch."""
        pending = tuple(self.mailbox.pending_robots(model))
        newest = {
            connection.record.session: connection.newest
            for connection in self.connections.values()
            if connection.newest is not None
        }
        decision = self.schedulers[model].next(now_ns)
        self.decisions.append(Decided(model, now_ns, pending, newest, decision))
        if isinstance(decision, Dispatch):
            self.in_flight[model] = decision.batch
        return decision

    def complete(
        self, model: str, failed: frozenset[int], worker_ns: int, cpu_ns: int, now_ns: int
    ) -> None:
        """Book the results of the model's batch and push the command of each item delivered."""
        batch = self.in_flight.pop(model)
        before = self._counters()
        delivered = self.schedulers[model].complete(batch, failed, worker_ns, cpu_ns, now_ns)
        for index in delivered:
            record = batch.items[index].record
            command = QueuedCommand(COMMAND, record.models[model])
            self.connections[record.robot_id].downlink.push(command)
        self._finish(model, batch, failed, cpu_ns, delivered, before)

    def fail(self, model: str, now_ns: int) -> None:
        """Drop the model's batch: its worker died, timed out or broke the protocol."""
        batch = self.in_flight.pop(model)
        before = self._counters()
        self.schedulers[model].fail(batch, now_ns)
        self._finish(model, batch, frozenset(range(len(batch.items))), 0, (), before)

    def pop(self, robot_id: str, now_ns: int) -> None:
        """Write the next command queued for the robot's open session."""
        self.connections[robot_id].downlink.pop(now_ns)

    def _counters(self) -> dict[int, tuple[tuple[object, ...], ...]]:
        """Return every record's counters by identity, as values that later calls cannot change."""
        return {
            key: tuple(astuple(counters) for counters in record.models.values())
            for key, record in self.records.items()
        }

    def _finish(
        self,
        model: str,
        batch: Batch,
        failed: frozenset[int],
        cpu_ns: int,
        delivered: tuple[int, ...],
        before: dict[int, tuple[tuple[object, ...], ...]],
    ) -> None:
        after = self._counters()
        self.finishes.append(
            Finished(
                model=model,
                batch=batch,
                failed=failed,
                cpu_ns=cpu_ns,
                delivered=delivered,
                open_sessions=frozenset(
                    connection.record.session for connection in self.connections.values()
                ),
                touched=tuple(self.records[key] for key in after if after[key] != before[key]),
            )
        )


def dispatches(world: World, model: str) -> list[tuple[Decided, Batch]]:
    """Return the model's calls of next() that dispatched a batch, with the batch, in call order."""
    return [
        (decided, decided.decision.batch)
        for decided in world.decisions
        if decided.model == model and isinstance(decided.decision, Dispatch)
    ]


def arriving_frame(frame_number: int, now_ns: int, ages_ns: tuple[int, int]) -> Frame:
    """Return a frame that arrives at now_ns after its fire lag and uplink, on a shared clock."""
    fire_lag_ns, transport_up_ns = ages_ns
    capture_ns = now_ns - transport_up_ns - fire_lag_ns
    return Frame(frame_number, capture_ns, now_ns, fire_lag_ns, transport_up_ns, JPEG)


@st.composite
def core_setups(
    draw: st.DrawFn,
    *,
    policies: st.SearchStrategy[QueuePolicy] = QUEUE_POLICIES,
    skip_rules: st.SearchStrategy[bool] = SKIP_RULES,
    shares_cores: st.SearchStrategy[float | None] = OPTIONAL_SHARES_CORES,
) -> CoreSetup:
    """Draw the settings of a world whose table has fewer entries than there are ROBOT_IDS.

    A cap of one or two, with three or four entries, keeps more robots pending than a batch
    takes. The skip rule's interval is mostly longer than a robot waits for a batch here, so
    that guard (b) does not save every doomed frame and the rule does skip.
    """
    policy = draw(policies)
    # The configuration refuses the skip rule on a FIFO queue.
    skip_rule = policy is QueuePolicy.LATEST and draw(skip_rules)
    models = tuple(
        ModelSetup(
            SchedulerConfig(
                model=model,
                batch_cap=draw(st.integers(1, 2)),
                service_ns=draw(st.integers(0, 40)),
                return_allowance_ns=draw(st.integers(0, 20)),
                skip_rule=skip_rule,
                skip_max_unserved_ns=draw(st.integers(100, 1000)),
            ),
            draw(shares_cores),
        )
        for model in MODEL_KEYS
    )
    return CoreSetup(
        policy=policy,
        fifo_capacity_frames=draw(st.integers(1, 3)),
        models=models,
        retention_ns=draw(st.integers(0, 200)),
        max_entries=draw(st.integers(3, len(ROBOT_IDS) - 1)),
        downlink_queue_commands=draw(st.integers(1, 3)),
    )


class SchedulingMachine(RuleBasedStateMachine):
    """Every step a caller of the scheduling core takes, interleaved at random.

    A subclass sets SETUPS, the settings its runs draw, and adds the checks of one property.
    Time is now_ns, which moves only forward. Workers ask for batches as the gateway's do: a
    worker asks as soon as it has returned its results, and at a frame arrival every free worker
    that no pacing hold keeps waiting asks, which also brings a worker back after fail(). On top
    of that, worker_asks lets a free worker ask at any other time, a burst of frames lets it ask
    late, and a held worker asks exactly at its deadline.

    Hypothesis leaves a random part of the rules out of each run, so no single rule may be what
    keeps frames and batches moving: the first robots connect at the start, frames arrive with
    the passing of time, and a worker that returns results asks again by itself.
    """

    SETUPS: ClassVar[st.SearchStrategy[CoreSetup]]
    world: World
    now_ns: int

    @initialize(data=st.data())
    def start(self, data: st.DataObject) -> None:
        setup = data.draw(self.SETUPS, label="setup")
        self.world = World(setup)
        self.now_ns = 0
        first_ids = data.draw(
            st.lists(
                st.sampled_from(ROBOT_IDS), min_size=2, max_size=setup.max_entries, unique=True
            ),
            label="first_robot_ids",
        )
        for robot_id in first_ids:
            self.world.hello(robot_id, data.draw(DECLARATIONS, label="stale_ns"), self.now_ns)

    @precondition(lambda self: len(self.world.connections) < self.world.setup.max_entries)
    @rule(data=st.data())
    def connect(self, data: st.DataObject) -> None:
        """Say hello for a robot with no open session, while the table has room for it."""
        closed_ids = [robot_id for robot_id in ROBOT_IDS if robot_id not in self.world.connections]
        robot_id = data.draw(st.sampled_from(closed_ids), label="robot_id")
        self.world.hello(robot_id, data.draw(DECLARATIONS, label="stale_ns"), self.now_ns)

    @precondition(lambda self: bool(self.world.connections))
    @rule(data=st.data())
    def reconnect(self, data: st.DataObject) -> None:
        """Say hello again for a robot with an open session, which replaces its session."""
        robot_id = data.draw(st.sampled_from(sorted(self.world.connections)), label="robot_id")
        self.world.hello(robot_id, data.draw(DECLARATIONS, label="stale_ns"), self.now_ns)

    @precondition(lambda self: bool(self.world.connections))
    @rule(data=st.data())
    def disconnect(self, data: st.DataObject) -> None:
        robot_id = data.draw(st.sampled_from(sorted(self.world.connections)), label="robot_id")
        self.world.close(robot_id, self.now_ns)

    @precondition(lambda self: bool(self.world.connections))
    @rule(data=st.data(), elapsed_ns=st.integers(0, 40))
    def frame_arrives(self, data: st.DataObject, elapsed_ns: int) -> None:
        """Advance time and let a frame of one robot arrive."""
        self.now_ns += elapsed_ns
        robot_id = data.draw(st.sampled_from(sorted(self.world.connections)), label="robot_id")
        self._arrive(robot_id, data)
        self._free_workers_ask()

    @precondition(lambda self: bool(self.world.connections))
    @rule(data=st.data(), elapsed_ns=st.integers(0, 40))
    def every_robot_sends(self, data: st.DataObject, elapsed_ns: int) -> None:
        """Advance time and let a frame of every robot with an open session arrive."""
        self.now_ns += elapsed_ns
        self._every_robot_sends(data)

    @precondition(lambda self: bool(self.world.connections))
    @rule(data=st.data())
    def burst_of_frames(self, data: st.DataObject) -> None:
        """Let frames of every robot arrive one after another, and only then let free workers ask.

        An idle worker asks later than the gateway would, so the frames chosen have waited and
        the skip rule may find them doomed.
        """
        for robot_id in sorted(self.world.connections):
            self._arrive(robot_id, data)
            self.now_ns += data.draw(st.integers(0, 40), label="elapsed_ns")
        self._free_workers_ask()

    @precondition(lambda self: len(self.world.in_flight) < len(self.world.schedulers))
    @rule(data=st.data())
    def worker_asks(self, data: st.DataObject) -> None:
        """Let a free worker ask for a batch at a time the gateway would not ask."""
        free_models = [model for model in MODEL_KEYS if model not in self.world.in_flight]
        model = data.draw(st.sampled_from(free_models), label="model")
        self.world.decide(model, self.now_ns)

    @precondition(lambda self: bool(self.hold_deadlines_ns()))
    @rule(data=st.data())
    def pacing_releases(self, data: st.DataObject) -> None:
        """Advance time to the end of a pacing hold and let the held worker ask again."""
        deadlines_ns = self.hold_deadlines_ns()
        model = data.draw(st.sampled_from(sorted(deadlines_ns)), label="model")
        self.now_ns = deadlines_ns[model]
        self.world.decide(model, self.now_ns)

    @precondition(lambda self: bool(self.hold_deadlines_ns()) and bool(self.world.connections))
    @rule(data=st.data())
    def robots_send_until_a_hold_ends(self, data: st.DataObject) -> None:
        """Let frames arrive while a worker is held, and let the worker ask when the hold ends."""
        deadlines_ns = self.hold_deadlines_ns()
        model = data.draw(st.sampled_from(sorted(deadlines_ns)), label="model")
        self._every_robot_sends(data)
        self.now_ns = deadlines_ns[model]
        self.world.decide(model, self.now_ns)

    @precondition(lambda self: bool(self.world.in_flight))
    @rule(data=st.data(), elapsed_ns=st.integers(0, 40))
    def worker_returns_results(self, data: st.DataObject, elapsed_ns: int) -> None:
        """Advance time and let a busy worker return its results, some failed, and ask again."""
        self.now_ns += elapsed_ns
        model = data.draw(st.sampled_from(sorted(self.world.in_flight)), label="model")
        batch = self.world.in_flight[model]
        failed = data.draw(st.frozensets(st.sampled_from(range(len(batch.items)))), label="failed")
        worker_ns = data.draw(st.integers(0, self.now_ns - batch.dispatched_ns), label="worker_ns")
        # A worker with several threads uses more CPU time than wall time.
        cpu_ns = data.draw(st.integers(0, 3 * worker_ns + 30), label="cpu_ns")
        self.world.complete(model, failed, worker_ns, cpu_ns, self.now_ns)
        self.world.decide(model, self.now_ns)

    @precondition(lambda self: bool(self.world.in_flight))
    @rule(data=st.data())
    def worker_fails(self, data: st.DataObject) -> None:
        """Let a busy worker die; it asks again once it is back, at the next frame at the latest."""
        model = data.draw(st.sampled_from(sorted(self.world.in_flight)), label="model")
        self.world.fail(model, self.now_ns)

    @precondition(
        lambda self: any(connection.downlink for connection in self.world.connections.values())
    )
    @rule(data=st.data())
    def command_written(self, data: st.DataObject) -> None:
        waiting_ids = sorted(
            robot_id
            for robot_id, connection in self.world.connections.items()
            if connection.downlink
        )
        robot_id = data.draw(st.sampled_from(waiting_ids), label="robot_id")
        self.world.pop(robot_id, self.now_ns)

    @rule(elapsed_ns=st.integers(0, 40) | st.integers(0, 400))
    def time_passes(self, elapsed_ns: int) -> None:
        self.now_ns += elapsed_ns

    def hold_deadlines_ns(self) -> dict[str, int]:
        """Return the deadline of each worker whose last decision is a hold not over by now."""
        last = {decided.model: decided.decision for decided in self.world.decisions}
        return {
            model: decision.deadline_ns
            for model, decision in last.items()
            if isinstance(decision, WaitUntil) and decision.deadline_ns >= self.now_ns
        }

    def _every_robot_sends(self, data: st.DataObject) -> None:
        for robot_id in sorted(self.world.connections):
            self._arrive(robot_id, data)
        self._free_workers_ask()

    def _arrive(self, robot_id: str, data: st.DataObject) -> None:
        """Put a frame of the robot that arrives now, numbered above its session's newest."""
        newest = self.world.connections[robot_id].newest
        # Frame numbers rise within a session and restart with the next one.
        frame_number = data.draw(st.integers(1, 3), label="number_step") + (
            0 if newest is None else newest.frame_number
        )
        ages_ns = data.draw(FRAME_AGES_NS, label="ages_ns")
        self.world.put(robot_id, arriving_frame(frame_number, self.now_ns, ages_ns))

    def _free_workers_ask(self) -> None:
        """Let every free worker that no pacing hold keeps waiting ask for a batch."""
        deadlines_ns = self.hold_deadlines_ns()
        for model in MODEL_KEYS:
            if model not in self.world.in_flight and model not in deadlines_ns:
                self.world.decide(model, self.now_ns)


@dataclass(frozen=True)
class Sender:
    """A robot connected at time 0 whose frame k arrives at phase_ns + k * period_ns.

    offline_ns holds the times its session closes and opens again, in order; it sends nothing
    while its session is closed.
    """

    robot_id: str
    period_ns: int
    phase_ns: int
    stale_ns: int
    # The fire lag and the uplink of its successive frames, repeated.
    ages_ns: tuple[tuple[int, int], ...]
    offline_ns: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Scenario:
    """Robots sending to one model's scheduler, and its worker's batches."""

    config: SchedulerConfig
    share_cores: float | None
    senders: tuple[Sender, ...]
    # The wall time and the CPU time of the worker's successive batches, repeated.
    batch_times_ns: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _Arrival:
    sender: int
    frame_number: int


@dataclass(frozen=True)
class _Close:
    sender: int


@dataclass(frozen=True)
class _Open:
    sender: int


@dataclass(frozen=True)
class _Completion:
    worker_ns: int
    cpu_ns: int


@dataclass(frozen=True)
class _Release:
    """The end of a pacing hold."""


_Event = _Arrival | _Close | _Open | _Completion | _Release


class _Simulation:
    """The state of a run of simulate(): the world, the events to come and the worker's state."""

    def __init__(self, scenario: Scenario, end_ns: int) -> None:
        self.scenario = scenario
        self.model = scenario.config.model
        self.world = World(
            CoreSetup(
                policy=QueuePolicy.LATEST,
                fifo_capacity_frames=1,
                models=(ModelSetup(scenario.config, scenario.share_cores),),
                # The table keeps every record for the whole run, so a robot that leaves comes
                # back to its own record.
                retention_ns=end_ns,
                max_entries=len(scenario.senders),
                downlink_queue_commands=1,
            )
        )
        self.events: list[tuple[int, int, _Event]] = []
        self.order = itertools.count()
        self.batch_times_ns = itertools.cycle(scenario.batch_times_ns)
        # Whether the worker's last decision was Idle, so that the next frame makes it ask.
        self.idle = True

    def schedule(self, time_ns: int, event: _Event) -> None:
        # The order runs the events of one time in the order they were scheduled.
        heapq.heappush(self.events, (time_ns, next(self.order), event))

    def handle(self, now_ns: int, event: _Event) -> None:
        match event:
            case _Arrival(index, frame_number):
                sender = self.scenario.senders[index]
                if sender.robot_id in self.world.connections:
                    ages_ns = sender.ages_ns[frame_number % len(sender.ages_ns)]
                    self.world.put(sender.robot_id, arriving_frame(frame_number, now_ns, ages_ns))
                    if self.idle:
                        self.ask(now_ns)
                self.schedule(now_ns + sender.period_ns, _Arrival(index, frame_number + 1))
            case _Close(index):
                self.world.close(self.scenario.senders[index].robot_id, now_ns)
            case _Open(index):
                sender = self.scenario.senders[index]
                self.world.hello(sender.robot_id, {self.model: sender.stale_ns}, now_ns)
            case _Completion(worker_ns, cpu_ns):
                self.world.complete(self.model, frozenset(), worker_ns, cpu_ns, now_ns)
                for robot_id in self.world.connections:
                    self.world.pop(robot_id, now_ns)
                self.ask(now_ns)
            case _Release():
                self.ask(now_ns)

    def ask(self, now_ns: int) -> None:
        """Let the free worker ask for a batch, and schedule what its decision leads to."""
        decision = self.world.decide(self.model, now_ns)
        self.idle = isinstance(decision, Idle)
        match decision:
            case Dispatch():
                worker_ns, cpu_ns = next(self.batch_times_ns)
                self.schedule(now_ns + worker_ns, _Completion(worker_ns, cpu_ns))
            case WaitUntil(deadline_ns):
                # A deadline that is not after now would hold the run at this time forever.
                assert deadline_ns > now_ns, f"{decision} returned at {now_ns}"
                self.schedule(deadline_ns, _Release())


def simulate(scenario: Scenario, end_ns: int) -> World:
    """Run the scenario until end_ns, calling the scheduler exactly when the gateway calls it.

    Every robot connects at 0, and leaves and comes back with a new session as its offline_ns
    says. next() is called at every completion, at every pacing release and at every frame
    arrival while the last decision is Idle, and every command is written as soon as it is
    queued.
    """
    simulation = _Simulation(scenario, end_ns)
    for index, sender in enumerate(scenario.senders):
        simulation.world.hello(sender.robot_id, {simulation.model: sender.stale_ns}, 0)
        simulation.schedule(sender.phase_ns, _Arrival(index, 0))
        for close_ns, open_ns in sender.offline_ns:
            simulation.schedule(close_ns, _Close(index))
            simulation.schedule(open_ns, _Open(index))
    while simulation.events and simulation.events[0][0] <= end_ns:
        now_ns, _, event = heapq.heappop(simulation.events)
        simulation.handle(now_ns, event)
    return simulation.world


def batch_times_ns(
    shortest_ns: int, longest_ns: int
) -> st.SearchStrategy[tuple[tuple[int, int], ...]]:
    """Draw the wall and CPU time of up to four batches, the CPU time up to twice the wall time."""
    return st.lists(
        st.integers(shortest_ns, longest_ns).flatmap(
            lambda wall_ns: st.tuples(st.just(wall_ns), st.integers(0, 2 * wall_ns))
        ),
        min_size=1,
        max_size=4,
    ).map(tuple)


@st.composite
def periodic_scenarios(draw: st.DrawFn, *, skip_rule: bool) -> Scenario:
    """Draw one to five robots that all send one frame per period, at one period, and stay.

    Each robot's stale threshold lies within a period of the fixed part of its first frame's
    age (fire lag, twice the uplink, service time and allowance), so that its frames, whose ages
    vary, are fresh, doomed only after a wait, or doomed however short the wait.
    """
    period_ns = draw(st.integers(20, 60))
    config = SchedulerConfig(
        model="cls",
        batch_cap=draw(st.integers(1, 4)),
        service_ns=draw(st.integers(0, period_ns)),
        return_allowance_ns=draw(st.integers(0, period_ns // 2)),
        skip_rule=skip_rule,
        skip_max_unserved_ns=draw(st.integers(1, 4 * period_ns)),
    )
    ages_ns = st.lists(
        st.tuples(st.integers(0, 3 * period_ns), st.integers(0, period_ns)), min_size=1, max_size=4
    )
    senders = []
    for index in range(draw(st.integers(1, 5))):
        sender_ages_ns = tuple(draw(ages_ns))
        fire_lag_ns, transport_up_ns = sender_ages_ns[0]
        fixed_ns = (
            fire_lag_ns + 2 * transport_up_ns + config.service_ns + config.return_allowance_ns
        )
        senders.append(
            Sender(
                robot_id=f"r{index}",
                period_ns=period_ns,
                phase_ns=draw(st.integers(0, period_ns - 1)),
                # A declared threshold is above 0.
                stale_ns=max(1, fixed_ns + draw(st.integers(-period_ns, period_ns))),
                ages_ns=sender_ages_ns,
                offline_ns=(),
            )
        )
    return Scenario(
        config, draw(OPTIONAL_SHARES_CORES), tuple(senders), draw(batch_times_ns(1, 2 * period_ns))
    )


@st.composite
def faster_sender_scenarios(draw: st.DrawFn) -> Scenario:
    """Draw two to five robots, one of which sends 2 or 4 times as many frames as the others.

    Every batch takes longer than the others' frame period, so from the second dispatch on,
    every robot has a frame pending at every dispatch.
    """
    period_ns = 4 * draw(st.integers(5, 15))
    robot_count = draw(st.integers(2, 5))
    periods_ns = [period_ns] * robot_count
    periods_ns[draw(st.integers(0, robot_count - 1))] //= draw(st.sampled_from((2, 4)))
    senders = tuple(
        Sender(
            robot_id=f"r{index}",
            period_ns=sender_period_ns,
            phase_ns=draw(st.integers(0, sender_period_ns - 1)),
            stale_ns=1,
            ages_ns=((0, 0),),
            offline_ns=(),
        )
        for index, sender_period_ns in enumerate(periods_ns)
    )
    config = SchedulerConfig(
        model="cls",
        batch_cap=draw(st.integers(1, robot_count)),
        service_ns=0,
        return_allowance_ns=0,
        skip_rule=False,
        skip_max_unserved_ns=1,
    )
    return Scenario(config, None, senders, draw(batch_times_ns(period_ns + 1, 2 * period_ns)))


@st.composite
def churning_scenarios(draw: st.DrawFn) -> Scenario:
    """Draw two to five robots that leave and come back, each sending at a base rate or below.

    One robot sends one frame per base period and each other one per one, two or three base
    periods, so the base period is the shortest frame period of the run. The batch cap is below
    the robot count and a batch takes one to three base periods, so robots are left out of
    batches, and a slower robot can be pending at some dispatches and not at others. A robot
    leaves up to twice, each time after one to fifteen base periods, and comes back at once,
    with a new session, or up to three base periods later, within 36 base periods.
    """
    period_ns = draw(st.integers(10, 30))
    robot_count = draw(st.integers(2, 5))
    multiples = [draw(st.sampled_from((1, 2, 3))) for _ in range(robot_count)]
    multiples[draw(st.integers(0, robot_count - 1))] = 1
    senders = []
    for index, multiple in enumerate(multiples):
        sender_period_ns = period_ns * multiple
        offline_ns = []
        open_ns = 0
        for _ in range(draw(st.integers(0, 2))):
            close_ns = open_ns + period_ns * draw(st.integers(1, 15))
            open_ns = close_ns + draw(st.just(0) | st.integers(0, 3 * period_ns))
            offline_ns.append((close_ns, open_ns))
        senders.append(
            Sender(
                robot_id=f"r{index}",
                period_ns=sender_period_ns,
                phase_ns=draw(st.integers(0, sender_period_ns - 1)),
                stale_ns=1,
                ages_ns=((0, 0),),
                offline_ns=tuple(offline_ns),
            )
        )
    config = SchedulerConfig(
        model="cls",
        batch_cap=draw(st.integers(1, robot_count - 1)),
        service_ns=0,
        return_allowance_ns=0,
        skip_rule=False,
        skip_max_unserved_ns=1,
    )
    return Scenario(
        config,
        draw(OPTIONAL_SHARES_CORES),
        tuple(senders),
        draw(batch_times_ns(period_ns, 3 * period_ns)),
    )
