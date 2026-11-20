"""Tests for the link model's delay line: release times, order, capacity and the seed."""

import random

import pytest

from robot.link import DelayLine, line_capacity
from robot.settings import LinkSettings

MS = 1_000_000


def line(delay_ms: float, jitter_ms: float, capacity: int = 100, seed: int = 0) -> DelayLine:
    return DelayLine(LinkSettings(delay_ms, jitter_ms), capacity, random.Random(seed))


@pytest.mark.parametrize(
    ("delay_ms", "jitter_ms", "messages_per_s", "capacity"),
    [
        (0.0, 0.0, 30.0, 2),
        (100.0, 0.0, 30.0, 5),
        (100.0, 50.0, 30.0, 7),
        (20.0, 0.0, 31.0, 3),
        (1000.0, 0.0, 60.0, 62),
        (0.0, 1.0, 1.0, 3),
    ],
)
def test_a_line_holds_what_its_longest_delay_covers_at_the_message_rate_plus_two(
    delay_ms: float, jitter_ms: float, messages_per_s: float, capacity: int
) -> None:
    assert line_capacity(LinkSettings(delay_ms, jitter_ms), messages_per_s) == capacity


@pytest.mark.parametrize("message", [b"\x01frame", "telemetry"])
def test_a_message_is_released_one_delay_after_its_push(message: str | bytes) -> None:
    delayed = line(delay_ms=20.0, jitter_ms=0.0)

    release_ns = delayed.push(message, 5 * MS)

    assert release_ns == 25 * MS
    assert delayed.next_release_ns() == 25 * MS
    assert delayed.pop_due(25 * MS - 1) == []
    assert delayed.pop_due(25 * MS) == [message]


def test_an_empty_line_has_no_next_release() -> None:
    delayed = line(delay_ms=20.0, jitter_ms=0.0)

    assert delayed.next_release_ns() is None
    assert delayed.pop_due(10**18) == []
    assert len(delayed) == 0


def test_every_message_due_is_taken_at_once_and_in_the_order_pushed() -> None:
    delayed = line(delay_ms=10.0, jitter_ms=0.0)
    delayed.push(b"a", 0)
    delayed.push(b"b", 1 * MS)
    delayed.push(b"c", 5 * MS)

    assert delayed.pop_due(11 * MS) == [b"a", b"b"]
    assert delayed.next_release_ns() == 15 * MS
    assert len(delayed) == 1


def test_a_jitter_adds_between_nothing_and_itself_to_the_delay() -> None:
    delayed = line(delay_ms=10.0, jitter_ms=4.0, capacity=1000)

    # Pushed 100 ms apart, far more than the jitter, so no release is held back for order.
    lateness_ns = [delayed.push(b"m", k * 100 * MS) - k * 100 * MS for k in range(1000)]

    assert min(lateness_ns) >= 10 * MS
    assert max(lateness_ns) <= 14 * MS
    assert len(set(lateness_ns)) > 900


def test_a_message_with_a_short_jitter_is_not_released_before_the_one_ahead_of_it() -> None:
    delayed = line(delay_ms=0.0, jitter_ms=50.0, capacity=1000)

    # Pushed at one instant, so only the order rule separates them.
    release_ns = [delayed.push(bytes([k % 256]), 0) for k in range(1000)]

    assert release_ns == sorted(release_ns)
    assert delayed.pop_due(50 * MS) == [bytes([k % 256]) for k in range(1000)]


def test_a_release_is_not_held_back_by_a_message_that_has_already_left() -> None:
    delayed = line(delay_ms=10.0, jitter_ms=0.0)
    delayed.push(b"a", 0)
    delayed.pop_due(10 * MS)

    assert delayed.push(b"b", 50 * MS) == 60 * MS


def test_a_full_line_says_so_and_refuses_a_push() -> None:
    delayed = line(delay_ms=10.0, jitter_ms=0.0, capacity=2)
    delayed.push(b"a", 0)
    assert not delayed.full()
    delayed.push(b"b", 0)

    assert delayed.full()
    with pytest.raises(ValueError, match="already holds its 2 messages"):
        delayed.push(b"c", 0)
    assert len(delayed) == 2


def test_a_line_has_room_again_once_a_message_is_released() -> None:
    delayed = line(delay_ms=10.0, jitter_ms=0.0, capacity=1)
    delayed.push(b"a", 0)

    delayed.pop_due(10 * MS)

    assert not delayed.full()
    assert delayed.push(b"b", 10 * MS) == 20 * MS


def test_the_same_seed_gives_the_same_releases_and_another_seed_gives_others() -> None:
    def releases(seed: int) -> list[int]:
        delayed = line(delay_ms=5.0, jitter_ms=20.0, seed=seed)
        return [delayed.push(b"m", k * 100 * MS) for k in range(50)]

    assert releases(7) == releases(7)
    assert releases(7) != releases(8)
