"""Tests for the injectable clocks."""

import time

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.clock import Clock, FakeClock, MonotonicClock

# The annotation makes mypy check that both classes satisfy the protocol.
CLOCKS: tuple[Clock, ...] = (MonotonicClock(), FakeClock(start_ns=7))


@pytest.mark.parametrize("clock", CLOCKS, ids=["monotonic", "fake"])
def test_clock_never_goes_back(clock: Clock) -> None:
    first_ns = clock.now_ns()

    assert clock.now_ns() >= first_ns


def test_monotonic_clock_reads_the_shared_monotonic_timeline() -> None:
    before_ns = time.monotonic_ns()
    now_ns = MonotonicClock().now_ns()
    after_ns = time.monotonic_ns()

    assert before_ns <= now_ns <= after_ns


def test_fake_clock_starts_at_zero_by_default() -> None:
    assert FakeClock().now_ns() == 0


def test_fake_clock_starts_at_the_given_time() -> None:
    assert FakeClock(start_ns=1_500).now_ns() == 1_500


def test_fake_clock_stands_still_between_advances() -> None:
    clock = FakeClock(start_ns=40)

    assert (clock.now_ns(), clock.now_ns()) == (40, 40)


@given(
    start_ns=st.integers(min_value=0, max_value=2**63),
    deltas_ns=st.lists(st.integers(min_value=0, max_value=2**40)),
)
def test_fake_clock_reads_the_start_plus_every_advance(start_ns: int, deltas_ns: list[int]) -> None:
    clock = FakeClock(start_ns)

    for delta_ns in deltas_ns:
        clock.advance_ns(delta_ns)

    assert clock.now_ns() == start_ns + sum(deltas_ns)


def test_fake_clock_rejects_a_negative_advance() -> None:
    clock = FakeClock(start_ns=100)

    with pytest.raises(ValueError, match="delta_ns=-1"):
        clock.advance_ns(-1)

    assert clock.now_ns() == 100
