"""Property tests for the link model's delay line."""

import random

from hypothesis import given
from hypothesis import strategies as st

from robot.link import DelayLine
from robot.settings import LinkSettings

delays_ms = st.floats(min_value=0.0, max_value=1000.0, allow_nan=False)
# Each step is the time since the previous push, and how far before the push the line is
# drained: at the push itself, or so long before it that nothing is due.
steps = st.lists(
    st.tuples(st.integers(min_value=0, max_value=2_000_000_000), st.sampled_from((0, -(10**13)))),
    min_size=1,
    max_size=60,
)


@given(delay_ms=delays_ms, jitter_ms=delays_ms, seed=st.integers(), steps=steps)
def test_releases_never_decrease_and_never_precede_the_push_time_plus_the_delay(
    delay_ms: float, jitter_ms: float, seed: int, steps: list[tuple[int, int]]
) -> None:
    delayed = DelayLine(LinkSettings(delay_ms, jitter_ms), len(steps), random.Random(seed))
    now_ns = 0
    pushes: list[tuple[int, int]] = []
    for gap_ns, drain_offset_ns in steps:
        now_ns += gap_ns
        delayed.pop_due(now_ns + drain_offset_ns)
        pushes.append((now_ns, delayed.push(b"m", now_ns)))

    releases = [release_ns for _, release_ns in pushes]
    assert releases == sorted(releases)
    assert all(
        release_ns >= push_ns + round(delay_ms * 1_000_000) for push_ns, release_ns in pushes
    )


@given(delay_ms=delays_ms, jitter_ms=delays_ms, seed=st.integers(), steps=steps)
def test_messages_leave_in_the_order_pushed_and_none_before_its_release(
    delay_ms: float, jitter_ms: float, seed: int, steps: list[tuple[int, int]]
) -> None:
    delayed = DelayLine(LinkSettings(delay_ms, jitter_ms), len(steps), random.Random(seed))
    now_ns = 0
    release_of: dict[str | bytes, int] = {}
    left: list[tuple[str | bytes, int]] = []
    for number, (gap_ns, drain_offset_ns) in enumerate(steps):
        now_ns += gap_ns
        drain_ns = now_ns + drain_offset_ns
        left.extend((message, drain_ns) for message in delayed.pop_due(drain_ns))
        message = number.to_bytes(2, "little")
        release_of[message] = delayed.push(message, now_ns)
    end_ns = max(release_of.values())
    left.extend((message, end_ns) for message in delayed.pop_due(end_ns))

    assert [message for message, _ in left] == list(release_of)
    assert all(left_ns >= release_of[message] for message, left_ns in left)
    assert len(delayed) == 0
