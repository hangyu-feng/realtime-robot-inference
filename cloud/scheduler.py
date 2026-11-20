"""One scheduler per model, and the derivations of its settings.

Derivations: pure functions turn the configuration and a worker's calibration into the
scheduler's settings (plan_model, batch_cap) and into the pacing share (pacing_share,
other_demand_cores, pacing_release_ns). The gateway calls them at startup and when a robot
joins or leaves, and the simulated-clock capacity model calls the same ones. Nothing else
derives these values, so the predictions are made with the settings the service runs with.

Scheduling: a ModelScheduler decides, whenever its worker is free, which robots' frames go into
the next batch (DESIGN.md 6.1): frames are taken from the mailbox at dispatch time, robots are
served least recently served first, and the skip rule (5.5), the batch cap (6.2) and the
pacing hold (6.4) apply.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction

from cloud.config import ServerConfig
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRecord
from common.histogram import Histogram

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
_MS_PER_S = 1_000


@dataclass(frozen=True, slots=True)
class SchedulerConfig:
    """The settings of one model's scheduler, fixed for the run.

    service_ns is the calibrated wall time of one batch at the cap: the skip rule's estimate
    of the service time.
    """

    model: str
    batch_cap: int
    service_ns: int
    return_allowance_ns: int
    skip_rule: bool
    skip_max_unserved_ns: int


@dataclass(frozen=True, slots=True)
class ModelPlan:
    """A model's scheduler settings, with the freshness budget and margin they leave.

    margin_ns is workers.<m>.margin_ms and feasible is workers.<m>.feasible of /metrics.
    """

    scheduler: SchedulerConfig
    budget_ns: int
    margin_ns: int
    feasible: bool


def batch_cap(t_b_ns: Mapping[int, int], budget_ns: int) -> int:
    """Return the largest size whose batch time is at most budget_ns, or the smallest size.

    The smallest size is the cap when no size fits (DESIGN.md 6.2): the model is then served
    one frame at a time with a negative margin.
    """
    return max(
        (size for size, batch_ns in t_b_ns.items() if batch_ns <= budget_ns),
        default=min(t_b_ns),
    )


def plan_model(model: str, config: ServerConfig, t_b_ns: Mapping[int, int]) -> ModelPlan:
    """Derive a model's scheduler settings from the configuration and its calibrated batch times.

    t_b_ns is the calibrated wall time of one batch of each size. ValueError unless its sizes
    are exactly the model's batch_sizes: only calibrated sizes can become the cap, so a caller
    with a table of other sizes cuts it first.
    """
    model_config = config.models[model]
    if set(t_b_ns) != set(model_config.batch_sizes):
        raise ValueError(
            f"{model}: t_b_ns has the sizes {sorted(t_b_ns)}, "
            f"not the calibrated sizes {list(model_config.batch_sizes)}"
        )
    stale_ns = round(model_config.stale_ms * _NS_PER_MS)
    # One allowance serves the budget and the skip rule because both test the same inequality,
    # wait + service time + handling time within the threshold: the budget with a wait of one
    # frame period, the skip rule with the frame's measured age. A second tunable would let
    # them disagree. It excludes link delay, which the skip rule takes from the frame's uplink.
    return_allowance_ns = round(config.overhead_allowance_ms * _NS_PER_MS)
    # The model's configured threshold, not a declared one: the cap is fixed before any robot
    # declares its own. The budget may be negative.
    budget_ns = stale_ns - round(_NS_PER_S / config.frame_rate_hz) - return_allowance_ns
    cap = (
        batch_cap(t_b_ns, budget_ns)
        if model_config.forced_batch_cap is None
        else model_config.forced_batch_cap
    )
    service_ns = t_b_ns[cap]
    return ModelPlan(
        scheduler=SchedulerConfig(
            model=model,
            batch_cap=cap,
            service_ns=service_ns,
            return_allowance_ns=return_allowance_ns,
            skip_rule=config.skip_rule,
            skip_max_unserved_ns=round(config.skip_max_unserved_ms * _NS_PER_MS),
        ),
        budget_ns=budget_ns,
        margin_ns=budget_ns - service_ns,
        # Against the whole threshold, not the budget: a negative margin means that some frames
        # arrive stale, while a service time above the threshold means that every frame does,
        # however short its wait (DESIGN.md 5.5).
        feasible=service_ns <= stale_ns,
    )


def pacing_release_ns(start_ns: int, cpu_ns: int, share_cores: float) -> int:
    """Return when a worker paced to share_cores may start again after a batch begun at start_ns.

    From start_ns to the returned time, the batch's CPU time over the wall time is at or under
    the share. This is the one definition of the rule.
    """
    # Whole nanoseconds from exact arithmetic: a float quotient could round down and release
    # the worker early, with its duty over the share.
    return start_ns + math.ceil(cpu_ns / Fraction(share_cores))


def _usable_cores(threads: int, quota_cores: float) -> float:
    """Return the cores a worker with this many intra-op threads can use under the quota."""
    # Threads 0 leaves the pool size to ONNX Runtime, which sizes it from the host's cores, so
    # the quota bounds what the worker can use.
    return threads if threads > 0 else quota_cores


def pacing_share(
    quota_cores: float,
    headroom_cores: float,
    min_share_cores: float,
    threads: int,
    other_demand_cores: float,
) -> float | None:
    """Return the CPU share the paced worker gets, in cores, or None when it is not paced.

    The share is what the quota leaves after the headroom and the predicted demand of every
    other consumer, and never below min_share_cores. It is None when that remainder is at least
    what the worker can use: its threads, or the quota when threads is 0.

    The share is feed-forward, not a feedback loop (DESIGN.md 1.3 rule 9, 6.4):
    other_demand_cores is predicted from the frame rates the connected robots declared, the
    calibrated CPU costs of the workers and the configured I/O cost, and the gateway recomputes
    the share when a robot joins or leaves. A caller whose container has no CPU quota does not
    call this and paces nothing: under a cpuset limit there is no freeze to prevent.
    """
    # ASSUMPTION: A19. The CPU limit is a quota, and the kernel freezes the whole container
    # once its demand passes the quota; the paced worker gives way so that it does not.
    remainder_cores = quota_cores - headroom_cores - other_demand_cores
    if remainder_cores >= _usable_cores(threads, quota_cores):
        return None
    return max(remainder_cores, min_share_cores)


def other_demand_cores(
    config: ServerConfig,
    records: Iterable[RobotRecord],
    cpu_ns_per_frame: Mapping[str, int],
    quota_cores: float,
) -> float:
    """Predict the CPU, in cores, of everything but the paced worker while the robots keep sending.

    The inputs: the connected records, each sending at its declared frame rate
    (1e9 / frame_period_ns) to the models it subscribed to; config.io_cpu_ms_per_frame, the
    configured CPU time per frame of the frame I/O (the gateway and, in the single topology,
    the simulator); cpu_ns_per_frame, per model, the calibrated CPU time of one batch of size 1
    (t_b_cpu_ns[1] of the worker's calibration); and the threads of each model that is not
    paced, or quota_cores for one whose threads is 0. The demand is the frame I/O of every
    connected robot, plus each unpaced model's offered frames, capped at the cores it can use.
    Every offered frame is costed as a batch of one: that is the cost when frames come one at
    a time, and an upper bound when the per-frame cost does not fall with batch size
    (DESIGN.md 6.3, H3). The paced model is left out.
    """
    connected = [record for record in records if record.connected]
    sent_hz = sum(_NS_PER_S / record.frame_period_ns for record in connected)
    demand_cores = sent_hz * config.io_cpu_ms_per_frame / _MS_PER_S
    for model, model_config in config.models.items():
        if model_config.paced:
            continue
        offered_hz = sum(
            _NS_PER_S / record.frame_period_ns for record in connected if model in record.subscribed
        )
        demand_cores += min(
            _usable_cores(model_config.threads, quota_cores),
            offered_hz * cpu_ns_per_frame[model] / _NS_PER_S,
        )
    return demand_cores


@dataclass(frozen=True, slots=True)
class BatchItem:
    """One robot's frame in a batch, with the session the robot was in when next() took it.

    The item holds the record itself, so complete() and fail() never look a robot up: the
    registry may drop a disconnected record while its frame is at the worker, and the item
    still counts on the right counters.
    """

    record: RobotRecord
    session: int
    frame: Frame


@dataclass(frozen=True, slots=True)
class Batch:
    """A batch dispatched to the worker. batch_id counts up from 1 in each scheduler."""

    batch_id: int
    items: tuple[BatchItem, ...]
    dispatched_ns: int


@dataclass(frozen=True, slots=True)
class Dispatch:
    """Send this batch to the worker."""

    batch: Batch


@dataclass(frozen=True, slots=True)
class WaitUntil:
    """Pacing holds the worker: call next() again at deadline_ns."""

    deadline_ns: int


@dataclass(frozen=True, slots=True)
class Idle:
    """No frame is eligible: call next() again when a frame arrives."""


Decision = Dispatch | WaitUntil | Idle


@dataclass(slots=True)
class ModelStats:
    """The scheduling counters of /metrics models.<m> (DESIGN.md 10.5).

    batch_size counts the batches of each size; a size never dispatched is absent. batch_ms
    holds the worker's wall time per batch.
    """

    batches: int = 0
    frames: int = 0
    batch_size: dict[int, int] = field(default_factory=dict)
    batch_ms: Histogram = field(default_factory=Histogram)
    busy_ns: int = 0
    paced_ns: int = 0
    idle_ns: int = 0


class ModelScheduler:
    """The scheduler of one model's worker, which takes one batch at a time.

    There is one scheduler per model for the life of the gateway, and a worker restart is
    fail() followed by next() on the same scheduler: the last_served_seq values in the records
    come from this scheduler's counter, and a new counter would put every robot it serves ahead
    of all the others. A restarted worker serves the cap this scheduler was built with: the
    gateway starts it with that cap, and it loads the model file for that cap
    (common/worker_protocol.py, WorkerReady).

    The FIFO policy needs no change here: Mailbox.head() is then the oldest frame, and
    cloud/config.py refuses FIFO with the skip rule on.
    """

    def __init__(self, config: SchedulerConfig, mailbox: Mailbox, now_ns: int) -> None:
        """Start with the worker free from now_ns."""
        self._config = config
        self._mailbox = mailbox
        self.stats = ModelStats()
        self._in_flight: Batch | None = None
        self._share_cores: float | None = None
        # The pacing release set by the last complete(); None when nothing holds the worker.
        self._release_ns: int | None = None
        # When the worker became free; None while a batch is in flight and after fail(),
        # until the first next() after it.
        self._free_since_ns: int | None = now_ns
        self._last_batch_id = 0
        self._last_served_seq = 0

    @property
    def in_flight(self) -> Batch | None:
        return self._in_flight

    def set_pacing_share(self, share_cores: float | None) -> None:
        """Pace the worker to share_cores from now on, or stop pacing it with None.

        The share applies from the next complete(); a release already set stands. ValueError
        for a share that is not a finite number above 0.
        """
        if share_cores is not None and not (math.isfinite(share_cores) and share_cores > 0):
            raise ValueError(f"a pacing share must be finite and above 0 cores, got {share_cores}")
        self._share_cores = share_cores

    def next(self, now_ns: int) -> Decision:
        """Decide what the free worker does at now_ns: dispatch a batch, wait for pacing, or idle.

        A caller calls it when the worker becomes free (once after construction, after
        complete(), and after fail() once the restarted worker is ready), when a WaitUntil
        deadline is reached, and when a frame arrives while the last decision was Idle. A
        WaitUntil comes only from pacing, and no frame arrival can change it, so a caller need
        not call next() for a frame that arrives during one: doing so returns the same decision.
        RuntimeError while a batch is in flight, because a worker takes one batch at a time.
        """
        if self._in_flight is not None:
            raise RuntimeError(
                f"{self._config.model}: batch {self._in_flight.batch_id} is in flight"
            )
        free_since_ns = now_ns if self._free_since_ns is None else self._free_since_ns
        self._free_since_ns = free_since_ns
        if self._release_ns is not None and now_ns < self._release_ns:
            # The mailbox is not read during the hold, so the frames chosen once it ends are
            # the newest at dispatch time (DESIGN.md 6.1 step 2).
            return WaitUntil(self._release_ns)
        model = self._config.model
        robots = self._mailbox.pending_robots(model)
        # Robots this model never served go first, by connect time and then robot id, so a
        # robot id goes first once; then the others, least recently served first. The sequence
        # number counts every robot this scheduler serves, so the order is total even inside
        # one batch.
        robots.sort(
            key=lambda record: (
                (0, record.connected_ns, record.robot_id)
                if record.models[model].last_served_seq is None
                else (1, record.models[model].last_served_seq, record.robot_id)
            )
        )
        items: list[BatchItem] = []
        for record in robots:
            if len(items) == self._config.batch_cap:
                # A robot the walk does not reach keeps its frame pending.
                break
            if self._config.skip_rule and self._skips(record, now_ns):
                self._mailbox.skip(record.robot_id, model)
                continue
            self._last_served_seq += 1
            counters = record.models[model]
            counters.last_served_seq = self._last_served_seq
            counters.last_served_ns = now_ns
            items.append(
                BatchItem(record, record.session, self._mailbox.take(record.robot_id, model))
            )
        if not items:
            # Nothing polls: the caller calls next() again when a frame arrives.
            return Idle()
        self._last_batch_id += 1
        batch = Batch(self._last_batch_id, tuple(items), now_ns)
        # The wait since the worker became free is paced up to the release and idle after it.
        # Both move only at a dispatch, so these /metrics counters lag while a worker waits.
        paced_ns = 0 if self._release_ns is None else max(0, self._release_ns - free_since_ns)
        self.stats.paced_ns += paced_ns
        self.stats.idle_ns += now_ns - free_since_ns - paced_ns
        self.stats.batches += 1
        self.stats.frames += len(items)
        self.stats.batch_size[len(items)] = self.stats.batch_size.get(len(items), 0) + 1
        self._in_flight = batch
        self._free_since_ns = None
        self._release_ns = None
        return Dispatch(batch)

    def complete(
        self, batch: Batch, failed: Collection[int], worker_ns: int, cpu_ns: int, now_ns: int
    ) -> tuple[int, ...]:
        """Book the results of the batch in flight and return the indices of deliverable items.

        failed holds the indices of the items whose frame produced no command. worker_ns is the
        worker's wall time for the batch (BatchResult.sent_ns - received_ns), cpu_ns its CPU
        time, and now_ns the gateway's reading when it had the results (the result_ns of these
        commands); the worker is free from now_ns. The indices come in batch order, and the
        caller pushes their commands at once (the second identity, cloud/downlink.py).
        ValueError unless batch is the batch in flight and every index in failed is an index
        of its items.
        """
        if batch is not self._in_flight:
            raise ValueError(f"batch {batch.batch_id} is not the batch in flight")
        if not all(0 <= index < len(batch.items) for index in failed):
            raise ValueError(f"failed names an item that batch {batch.batch_id} does not have")
        deliverable: list[int] = []
        for index, item in enumerate(batch.items):
            counters = item.record.models[self._config.model]
            counters.in_flight -= 1
            # One outcome per item, tested in this order. A failed frame counts as failed
            # whether its robot is still connected or not, as in fail(). A robot that left, or
            # whose id connected again, has a record that is not connected or is in a later
            # session; the result belongs to a session that has ended.
            if index in failed:
                counters.frames_failed += 1
            elif not item.record.connected or item.record.session != item.session:
                counters.results_dropped_disconnect += 1
            else:
                deliverable.append(index)
        self.stats.busy_ns += worker_ns
        self.stats.batch_ms.record(worker_ns / _NS_PER_MS)
        # With a share, the worker is held until the batch's CPU time over the share has passed
        # since its dispatch. So for any two dispatches i < j under one share, the CPU time of
        # batches i to j - 1 is at most share * (dispatched_j - dispatched_i). Over an arbitrary
        # window the worker can exceed the share by the CPU of the batches that straddle the
        # window's ends: the share is met per dispatch-to-dispatch span, not per quota period.
        # A failed batch reports no CPU time and sets no hold.
        self._release_ns = (
            None
            if self._share_cores is None
            else pacing_release_ns(batch.dispatched_ns, cpu_ns, self._share_cores)
        )
        self._in_flight = None
        self._free_since_ns = now_ns
        return tuple(deliverable)

    def fail(self, batch: Batch, now_ns: int) -> None:
        """Count every item of the batch in flight as failed; ValueError unless it is in flight.

        The worker died, timed out or broke the protocol. No pacing hold follows, and the worker
        counts as free again only from the first next() after this call, so nothing is booked
        at now_ns: restart and warm-up time are neither paced nor idle.
        """
        if batch is not self._in_flight:
            raise ValueError(f"batch {batch.batch_id} is not the batch in flight")
        for item in batch.items:
            counters = item.record.models[self._config.model]
            counters.in_flight -= 1
            counters.frames_failed += 1
        self._in_flight = None

    def _skips(self, record: RobotRecord, now_ns: int) -> bool:
        """Return whether the skip rule passes over the robot's head frame at now_ns.

        The frame is doomed when it would reach the robot older than the threshold the robot
        declared. Its age there is its fire lag, its uplink, its dwell here, the service time,
        the handling time and the downlink, estimated as equal to its uplink. service_ns is the
        calibrated time of a batch at the cap, so the age of a frame in a smaller batch is
        over-estimated, which errs toward skipping: a frame that a smaller batch would deliver
        in time can be judged doomed, and guard (b) bounds the cost. It errs toward serving
        only through guard (a).

        Two guards keep the rule from leaving a robot unserved for long; a doomed frame is
        served when either holds:
        (a) the fixed part of its age, everything but its dwell here, alone exceeds the
            threshold. A perfectly fresh frame from this robot could not make it either,
            because the hardware, the link or the robot's own lateness rules it out, so
            skipping would only starve the robot (DESIGN.md 5.5: skip only when a fresher
            frame could still make it).
        (b) the robot has gone unserved by this model for skip_max_unserved_ns or longer,
            counted from its last service, or from its connect if this model never served it.
        """
        config = self._config
        frame = self._mailbox.head(record.robot_id, config.model)
        threshold_ns = record.stale_ns[config.model]
        fixed_ns = (
            frame.fire_lag_ns
            + 2 * frame.transport_up_ns
            + config.service_ns
            + config.return_allowance_ns
        )
        last_served_ns = record.models[config.model].last_served_ns
        unserved_since_ns = record.connected_ns if last_served_ns is None else last_served_ns
        return (
            fixed_ns + now_ns - frame.received_ns > threshold_ns
            and fixed_ns <= threshold_ns
            and now_ns - unserved_since_ns < config.skip_max_unserved_ns
        )
