"""Tests for the exponential backoff with jitter."""

import math
import random

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.backoff import Backoff

# The largest value random() returns: no float in [0, 1) is greater.
HIGHEST_DRAW = math.nextafter(1.0, 0.0)


class FixedDraw(random.Random):
    """A generator whose every draw from [0, 1) is the same value."""

    def __init__(self, draw: float) -> None:
        super().__init__(0)
        self._draw = draw

    def random(self) -> float:
        return self._draw


def delays_ns(backoff: Backoff, count: int) -> list[int]:
    return [backoff.next_ns() for _ in range(count)]


@pytest.mark.parametrize(
    ("attempt", "ceiling_ns"),
    [(0, 100), (1, 200), (2, 400), (3, 800), (4, 1_000), (5, 1_000), (40, 1_000)],
)
def test_each_attempt_draws_between_half_its_ceiling_and_its_ceiling(
    attempt: int, ceiling_ns: int
) -> None:
    lowest = Backoff(100, 1_000, FixedDraw(0.0))
    highest = Backoff(100, 1_000, FixedDraw(HIGHEST_DRAW))

    lowest_ns = delays_ns(lowest, attempt + 1)[attempt]
    highest_ns = delays_ns(highest, attempt + 1)[attempt]

    assert (lowest_ns, highest_ns) == (ceiling_ns // 2, ceiling_ns)


def test_the_ceiling_doubles_from_the_minimum() -> None:
    backoff = Backoff(100, 10_000, FixedDraw(HIGHEST_DRAW))

    assert delays_ns(backoff, 5) == [100, 200, 400, 800, 1_600]


def test_the_ceiling_stops_at_the_maximum() -> None:
    backoff = Backoff(100, 300, FixedDraw(HIGHEST_DRAW))

    assert delays_ns(backoff, 6) == [100, 200, 300, 300, 300, 300]


def test_reset_starts_again_from_the_minimum() -> None:
    backoff = Backoff(100, 10_000, FixedDraw(HIGHEST_DRAW))
    delays_ns(backoff, 4)

    backoff.reset()

    assert delays_ns(backoff, 2) == [100, 200]


def test_the_same_seed_gives_the_same_delays() -> None:
    first = Backoff(500_000_000, 10_000_000_000, random.Random(11))
    second = Backoff(500_000_000, 10_000_000_000, random.Random(11))

    assert delays_ns(first, 12) == delays_ns(second, 12)


@pytest.mark.parametrize(("min_ns", "max_ns"), [(0, 10), (-1, 10), (11, 10)])
def test_bounds_must_satisfy_zero_below_minimum_up_to_maximum(min_ns: int, max_ns: int) -> None:
    with pytest.raises(ValueError, match=f"min_ns={min_ns}, max_ns={max_ns}"):
        Backoff(min_ns, max_ns, random.Random(0))


# Bounds up to 2**53, below which every integer is a float exactly, so a delay never leaves its
# ceiling through the product with the factor.
@given(
    bounds_ns=st.tuples(st.integers(1, 2**53), st.integers(1, 2**53)).map(sorted),
    rng=st.randoms(),
    count=st.integers(1, 64),
)
def test_every_delay_is_between_half_the_minimum_and_the_maximum(
    bounds_ns: list[int], rng: random.Random, count: int
) -> None:
    min_ns, max_ns = bounds_ns

    drawn_ns = delays_ns(Backoff(min_ns, max_ns, rng), count)

    # In whole nanoseconds: half an odd minimum rounds half to even.
    assert all(round(min_ns / 2) <= delay_ns <= max_ns for delay_ns in drawn_ns)
