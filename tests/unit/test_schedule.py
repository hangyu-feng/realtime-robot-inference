"""Tests for the open-loop capture schedule."""

import pytest

from robot.schedule import CaptureSchedule

START_NS = 1_000
PERIOD_NS = 100
SCHEDULE = CaptureSchedule(START_NS, PERIOD_NS)


@pytest.mark.parametrize(("frame_number", "capture_ns"), [(0, START_NS), (1, 1_100), (25, 3_500)])
def test_the_capture_time_of_frame_k_is_k_periods_after_the_start(
    frame_number: int, capture_ns: int
) -> None:
    assert SCHEDULE.capture_ns(frame_number) == capture_ns


@pytest.mark.parametrize(
    ("now_ns", "frame_number"),
    [
        (START_NS - 10 * PERIOD_NS, -1),
        (START_NS - 1, -1),
        (START_NS, 0),
        (START_NS + PERIOD_NS // 2, 0),
        (START_NS + PERIOD_NS - 1, 0),
        (START_NS + PERIOD_NS, 1),
        (START_NS + 25 * PERIOD_NS + PERIOD_NS // 2, 25),
    ],
    ids=[
        "long-before-the-start",
        "just-before-the-start",
        "at-the-start",
        "between-the-first-two-ticks",
        "just-before-the-second-tick",
        "at-the-second-tick",
        "between-later-ticks",
    ],
)
def test_latest_due_is_the_last_frame_whose_tick_is_at_or_before_now(
    now_ns: int, frame_number: int
) -> None:
    assert SCHEDULE.latest_due(now_ns) == frame_number
