"""Properties of the robot-side rules: command verdicts, the watchdog and the capture schedule."""

import bisect
from dataclasses import dataclass
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from robot.control import CommandOutcome, ModelControl, Verdict, WatchdogState
from robot.schedule import CaptureSchedule

SESSION = 7

# Three in four commands belong to the session; the others test the session rule.
SESSIONS = st.sampled_from((SESSION, SESSION, SESSION, SESSION + 1))


@dataclass(frozen=True)
class Arrival:
    """A command as the robot receives it, and a watchdog query in the wait before it."""

    query_ns: int
    receive_ns: int
    session: int
    frame_number: int
    capture_ns: int


@dataclass(frozen=True)
class Timeline:
    """A session of one model, and the grid step of its brute-force integration."""

    stale_ns: int
    watchdog_ns: int
    start_ns: int
    arrivals: tuple[Arrival, ...]
    end_ns: int
    grid_step_ns: int


@dataclass(frozen=True)
class Sample:
    """The watchdog as one query saw it."""

    now_ns: int
    applied_count: int
    state: WatchdogState
    trips: int
    expired_ns: int


@dataclass(frozen=True)
class Run:
    """What a ModelControl did with a timeline."""

    control: ModelControl
    outcomes: tuple[CommandOutcome, ...]
    samples: tuple[Sample, ...]


@st.composite
def timelines(draw: st.DrawFn) -> Timeline:
    # The rules are integer arithmetic, so the scale of the durations does not matter, and a
    # few tens of nanoseconds make coincidences common: commands received at the same time,
    # ages exactly at the stale threshold, replacements exactly at expiry.
    stale_ns = draw(st.integers(min_value=0, max_value=60))
    watchdog_ns = stale_ns + draw(st.integers(min_value=0, max_value=60))
    start_ns = draw(st.integers(min_value=-(2**62), max_value=2**62))
    now_ns = start_ns
    frame_number = 0
    arrivals = []
    for _ in range(draw(st.integers(min_value=0, max_value=30))):
        wait_ns = draw(st.integers(min_value=0, max_value=60))
        query_ns = now_ns + draw(st.integers(min_value=0, max_value=wait_ns))
        now_ns += wait_ns
        # Mostly rising frame numbers, with repeats and drops for the order rule.
        frame_number = max(0, frame_number + draw(st.integers(min_value=-2, max_value=4)))
        # Ages around the stale threshold, two in three of them at or below it. An age below
        # zero, a capture time after the receipt, can only come from a faulty gateway, and the
        # rules still hold for it.
        age_ns = stale_ns + draw(st.integers(min_value=-40, max_value=20))
        arrivals.append(Arrival(query_ns, now_ns, draw(SESSIONS), frame_number, now_ns - age_ns))
    end_ns = now_ns + draw(st.integers(min_value=0, max_value=150))
    grid_step_ns = draw(st.integers(min_value=1, max_value=20))
    return Timeline(stale_ns, watchdog_ns, start_ns, tuple(arrivals), end_ns, grid_step_ns)


def query(control: ModelControl, now_ns: int) -> Sample:
    return Sample(
        now_ns=now_ns,
        applied_count=control.applied,
        state=control.watchdog_state(now_ns),
        trips=control.watchdog_trips(now_ns),
        expired_ns=control.watchdog_expired_ns(now_ns),
    )


def run(timeline: Timeline) -> Run:
    """Feed the timeline to a new ModelControl, querying the watchdog around every command."""
    control = ModelControl(SESSION, timeline.stale_ns, timeline.watchdog_ns, timeline.start_ns)
    outcomes = []
    samples = [query(control, timeline.start_ns)]
    for arrival in timeline.arrivals:
        samples.append(query(control, arrival.query_ns))
        samples.append(query(control, arrival.receive_ns))
        outcomes.append(
            control.on_command(
                arrival.session, arrival.frame_number, arrival.capture_ns, arrival.receive_ns
            )
        )
        samples.append(query(control, arrival.receive_ns))
    samples.append(query(control, timeline.end_ns))
    return Run(control, tuple(outcomes), tuple(samples))


def applied_arrivals(timeline: Timeline, result: Run) -> list[Arrival]:
    return [
        arrival
        for arrival, outcome in zip(timeline.arrivals, result.outcomes, strict=True)
        if outcome.verdict is Verdict.APPLIED
    ]


def expected_state(sample: Sample, applied: list[Arrival], watchdog_ns: int) -> WatchdogState:
    """Derive the state from the age of the command in effect when the sample was taken."""
    if sample.applied_count == 0:
        return WatchdogState.NONE
    age_ns = sample.now_ns - applied[sample.applied_count - 1].capture_ns
    return WatchdogState.EXPIRED if age_ns > watchdog_ns else WatchdogState.OK


def covered_ns_on_grid(timeline: Timeline, applied: list[Arrival]) -> int:
    """Sum the grid cells at whose start a command in effect has an age of at most watchdog_ns."""
    # The grid runs from start_ns to end_ns, and the last cell is cut at end_ns.
    receive_times_ns = [arrival.receive_ns for arrival in applied]
    covered_ns = 0
    for point_ns in range(timeline.start_ns, timeline.end_ns, timeline.grid_step_ns):
        in_effect = bisect.bisect_right(receive_times_ns, point_ns) - 1
        if in_effect >= 0 and point_ns - applied[in_effect].capture_ns <= timeline.watchdog_ns:
            covered_ns += min(point_ns + timeline.grid_step_ns, timeline.end_ns) - point_ns
    return covered_ns


@given(timeline=timelines())
def test_every_received_command_gets_exactly_one_verdict(timeline: Timeline) -> None:
    result = run(timeline)
    control = result.control
    verdicts = [outcome.verdict for outcome in result.outcomes]

    assert control.received == len(timeline.arrivals)
    assert control.received == control.applied + control.stale + control.out_of_order
    assert (control.applied, control.stale, control.out_of_order) == (
        verdicts.count(Verdict.APPLIED),
        verdicts.count(Verdict.STALE),
        verdicts.count(Verdict.OUT_OF_ORDER),
    )


@given(timeline=timelines())
def test_a_command_is_stale_exactly_when_its_age_at_receipt_is_above_the_threshold(
    timeline: Timeline,
) -> None:
    outcomes = run(timeline).outcomes

    assert [outcome.freshness_ns for outcome in outcomes] == [
        arrival.receive_ns - arrival.capture_ns for arrival in timeline.arrivals
    ]
    assert [outcome.verdict is Verdict.STALE for outcome in outcomes] == [
        outcome.freshness_ns > timeline.stale_ns for outcome in outcomes
    ]


@given(timeline=timelines())
def test_applied_frame_numbers_strictly_increase_within_the_session(timeline: Timeline) -> None:
    applied = applied_arrivals(timeline, run(timeline))

    assert all(arrival.session == SESSION for arrival in applied)
    assert all(earlier.frame_number < later.frame_number for earlier, later in pairwise(applied))


@given(timeline=timelines())
def test_watchdog_expired_time_never_decreases_and_never_exceeds_the_session_time(
    timeline: Timeline,
) -> None:
    samples = run(timeline).samples

    assert all(earlier.expired_ns <= later.expired_ns for earlier, later in pairwise(samples))
    assert all(0 <= sample.expired_ns <= sample.now_ns - timeline.start_ns for sample in samples)


@given(timeline=timelines())
def test_watchdog_trips_never_decrease(timeline: Timeline) -> None:
    samples = run(timeline).samples

    assert all(earlier.trips <= later.trips for earlier, later in pairwise(samples))


@given(timeline=timelines())
def test_watchdog_state_follows_the_age_of_the_command_in_effect(timeline: Timeline) -> None:
    result = run(timeline)
    applied = applied_arrivals(timeline, result)

    assert [sample.state for sample in result.samples] == [
        expected_state(sample, applied, timeline.watchdog_ns) for sample in result.samples
    ]


@given(timeline=timelines())
def test_coverage_matches_a_brute_force_integration_over_a_grid(timeline: Timeline) -> None:
    result = run(timeline)
    applied = applied_arrivals(timeline, result)
    # Coverage times the session time, which keeps the comparison in exact integers.
    session_ns = timeline.end_ns - timeline.start_ns
    covered_ns = session_ns - result.control.watchdog_expired_ns(timeline.end_ns)
    # Each applied command covers one interval, from its application to the next one or to
    # its expiry, and the grid misjudges the length of each by at most one step.
    tolerance_ns = timeline.grid_step_ns * len(applied)

    assert abs(covered_ns - covered_ns_on_grid(timeline, applied)) <= tolerance_ns


@given(
    start_ns=st.integers(min_value=-(2**62), max_value=2**62),
    period_ns=st.integers(min_value=1, max_value=10**10),
    frame_number=st.integers(min_value=0, max_value=2**40),
)
def test_latest_due_at_a_capture_time_is_that_frame_number(
    start_ns: int, period_ns: int, frame_number: int
) -> None:
    schedule = CaptureSchedule(start_ns, period_ns)

    assert schedule.latest_due(schedule.capture_ns(frame_number)) == frame_number


@given(
    start_ns=st.integers(min_value=-(2**62), max_value=2**62),
    period_ns=st.integers(min_value=1, max_value=10**10),
    frame_number=st.integers(min_value=0, max_value=2**40),
)
def test_latest_due_just_before_a_capture_time_is_the_frame_before(
    start_ns: int, period_ns: int, frame_number: int
) -> None:
    schedule = CaptureSchedule(start_ns, period_ns)

    assert schedule.latest_due(schedule.capture_ns(frame_number) - 1) == frame_number - 1
