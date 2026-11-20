"""Tests for the saturating load of the model benchmark: the pacing arithmetic, and the loop."""

import itertools
from collections.abc import Sequence
from fractions import Fraction

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.model_bench import LoadResults, StopRequest, paced_start_ns, run_load
from bench.stats import summarize
from common.clock import FakeClock

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

START_NS = st.integers(min_value=0, max_value=2**63 - 1)
CPU_NS = st.integers(min_value=0, max_value=10**12)
SHARE_CORES = st.floats(min_value=1e-6, max_value=1e3)
# Per batch: its CPU time, and how long after its release the next batch starts.
# A batch that runs for longer than its share asks ends after its release, so the
# next one starts late.
BATCHES = st.lists(
    st.tuples(CPU_NS, st.integers(min_value=0, max_value=10**12)), min_size=1, max_size=20
)

# Per batch of a load: its wall time and its CPU time. Either can be the larger:
# a preempted batch waits for a core, and a session of several threads uses CPU
# time faster than wall time passes. Every batch takes some wall time, so the
# wall time of a load is never zero.
LOAD_BATCHES = st.lists(
    st.tuples(st.integers(min_value=1, max_value=10**9), st.integers(min_value=0, max_value=10**9)),
    min_size=1,
    max_size=20,
)
# With the CPU times above, this keeps every pacing wait short enough that a
# sleep given in seconds converts back to the whole nanoseconds the load asked for.
LOAD_SHARE_CORES = st.floats(min_value=0.01, max_value=64.0)
LOAD_START_NS = 7 * NS_PER_S
LOAD_BATCH_SIZE = 3
# Longer than any load of LOAD_BATCHES runs, so that only a stop request ends one.
ENDLESS_DURATION_S = 1e6

HALF_CORE = 0.5
# At half a core, a batch that takes BATCH_NS of wall and CPU time releases the
# next one CYCLE_NS after it starts.
BATCH_NS = 3 * NS_PER_MS
CYCLE_NS = 2 * BATCH_NS


class FakeProcess:
    """A process whose clocks move only while it runs a batch or sleeps.

    Its batches take the wall and CPU times it is given, and it requests a stop
    during the last of them.
    """

    def __init__(self, batches: Sequence[tuple[int, int]]) -> None:
        self.clock = FakeClock(LOAD_START_NS)
        self.cpu_clock = FakeClock()
        self.stop = StopRequest()
        self.starts_ns: list[int] = []
        self.slept_ns = 0
        self._batches = batches

    def run_batch(self, jpegs: Sequence[bytes]) -> None:
        wall_ns, cpu_ns = self._batches[len(self.starts_ns)]
        self.starts_ns.append(self.clock.now_ns())
        self.clock.advance_ns(wall_ns)
        self.cpu_clock.advance_ns(cpu_ns)
        self.stop.requested = len(self.starts_ns) == len(self._batches)

    def sleep(self, duration_s: float) -> None:
        # The load asks for whole nanoseconds, in the seconds that time.sleep takes.
        duration_ns = round(duration_s * NS_PER_S)
        self.clock.advance_ns(duration_ns)
        self.slept_ns += duration_ns

    def run_load(
        self, share_cores: float | None, duration_s: float = ENDLESS_DURATION_S
    ) -> LoadResults:
        """Run the load of the model benchmark on this process."""
        return run_load(
            self.run_batch,
            itertools.repeat((b"jpeg",) * LOAD_BATCH_SIZE),
            duration_s=duration_s,
            share_cores=share_cores,
            stop=self.stop,
            clock=self.clock,
            process_time_ns=self.cpu_clock.now_ns,
            sleep=self.sleep,
        )


def cycle_starts(start_ns: int, batches: list[tuple[int, int]], share_cores: float) -> list[int]:
    """Return the start of every batch, and the end of the cycle of the last one."""
    starts = [start_ns]
    for cpu_ns, delay_ns in batches:
        starts.append(paced_start_ns(starts[-1], cpu_ns, share_cores) + delay_ns)
    return starts


def earliest_starts(batches: Sequence[tuple[int, int]], share_cores: float) -> list[int]:
    """Return the earliest start of every batch of a load, and the end of the last cycle.

    A batch can start once the batch before it has ended and the share has
    released it.
    """
    starts = [LOAD_START_NS]
    for wall_ns, cpu_ns in batches:
        starts.append(max(starts[-1] + wall_ns, paced_start_ns(starts[-1], cpu_ns, share_cores)))
    return starts


@pytest.mark.parametrize(
    ("start_ns", "cpu_ns", "share_cores", "expected_ns"),
    [
        (1_000, 300, 0.5, 1_600),
        (0, 1_000, 2.0, 500),
        (5, 0, 0.25, 5),
        # A third of a nanosecond rounds up to a whole one.
        (0, 1, 3.0, 1),
        # The float 0.3 lies just under three tenths, so 3 ns of CPU time at that
        # share need just over 10 ns.
        (0, 3, 0.3, 11),
    ],
)
def test_next_batch_starts_the_cpu_time_over_the_share_after_the_last(
    start_ns: int, cpu_ns: int, share_cores: float, expected_ns: int
) -> None:
    assert paced_start_ns(start_ns, cpu_ns, share_cores) == expected_ns


@given(start_ns=START_NS, cpu_ns=CPU_NS, share_cores=SHARE_CORES)
def test_paced_start_is_the_earliest_that_keeps_the_cycle_within_the_share(
    start_ns: int, cpu_ns: int, share_cores: float
) -> None:
    release_ns = paced_start_ns(start_ns, cpu_ns, share_cores)

    share = Fraction(share_cores)
    assert cpu_ns <= share * (release_ns - start_ns)
    assert cpu_ns > share * (release_ns - 1 - start_ns)


@given(start_ns=START_NS, batches=BATCHES, share_cores=SHARE_CORES)
def test_duty_over_any_whole_number_of_cycles_is_at_most_the_share(
    start_ns: int, batches: list[tuple[int, int]], share_cores: float
) -> None:
    starts = cycle_starts(start_ns, batches, share_cores)

    cpu_ns = [batch_cpu_ns for batch_cpu_ns, _ in batches]
    for first, last in itertools.combinations(range(len(starts)), 2):
        assert sum(cpu_ns[first:last]) <= Fraction(share_cores) * (starts[last] - starts[first])


@given(batches=LOAD_BATCHES, share_cores=LOAD_SHARE_CORES)
def test_paced_load_starts_each_batch_as_soon_as_the_share_allows(
    batches: list[tuple[int, int]], share_cores: float
) -> None:
    process = FakeProcess(batches)

    process.run_load(share_cores)

    assert process.starts_ns == earliest_starts(batches, share_cores)[:-1]


@given(batches=LOAD_BATCHES, share_cores=LOAD_SHARE_CORES)
def test_stop_ends_the_load_after_the_whole_cycle_of_the_running_batch(
    batches: list[tuple[int, int]], share_cores: float
) -> None:
    process = FakeProcess(batches)

    process.run_load(share_cores)

    assert process.clock.now_ns() == earliest_starts(batches, share_cores)[-1]


@given(batches=LOAD_BATCHES)
def test_unpaced_load_runs_its_batches_back_to_back(batches: list[tuple[int, int]]) -> None:
    process = FakeProcess(batches)

    process.run_load(None)

    walls_ns = [wall_ns for wall_ns, _ in batches]
    assert process.starts_ns == list(itertools.accumulate(walls_ns[:-1], initial=LOAD_START_NS))
    assert process.slept_ns == 0


@pytest.mark.parametrize(("duration_s", "cycles"), [(0.0, 1), (0.010, 2)])
def test_load_ends_with_the_first_cycle_that_ends_past_its_duration(
    duration_s: float, cycles: int
) -> None:
    # One batch more than the cycles, so that the stop request never comes.
    process = FakeProcess([(BATCH_NS, BATCH_NS)] * (cycles + 1))

    results = process.run_load(HALF_CORE, duration_s)

    assert results.batches == cycles
    assert process.clock.now_ns() - LOAD_START_NS == cycles * CYCLE_NS


@given(batches=LOAD_BATCHES, share_cores=st.none() | LOAD_SHARE_CORES)
def test_load_reports_its_batches_their_times_and_the_time_it_slept(
    batches: list[tuple[int, int]], share_cores: float | None
) -> None:
    process = FakeProcess(batches)

    results = process.run_load(share_cores)

    wall_s = (process.clock.now_ns() - LOAD_START_NS) / NS_PER_S
    cpu_s = sum(cpu_ns for _, cpu_ns in batches) / NS_PER_S
    assert results == LoadResults(
        batches=len(batches),
        frames=len(batches) * LOAD_BATCH_SIZE,
        wall_s=wall_s,
        cpu_s=cpu_s,
        duty=cpu_s / wall_s,
        wall_ms=summarize([wall_ns / NS_PER_MS for wall_ns, _ in batches]),
        paced_ms=process.slept_ns / NS_PER_MS,
    )
