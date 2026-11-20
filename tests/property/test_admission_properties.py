"""Properties of the ingress limiter, checked against bounds computed in exact arithmetic.

For messages offered at times t_1 <= t_2 <= ..., the bound of the run of messages i to j is each
bucket's capacity plus its refill over the run: rate * (burst_s + t_j - t_i), for the number of
messages and for their bytes.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import accumulate

from hypothesis import given
from hypothesis import strategies as st

from cloud.admission import IngressLimiter

NS_PER_S = 10**9
MAX_MESSAGES = 40


@dataclass(frozen=True)
class Settings:
    """The float settings of a limiter, and the bounds they set in exact arithmetic."""

    max_messages_per_s: float
    max_bytes_per_s: float
    burst_s: float

    def message_bound(self, interval_ns: int) -> Fraction:
        """Return the most messages that a run spanning interval_ns may hold."""
        return Fraction(self.max_messages_per_s) * (
            Fraction(self.burst_s) + Fraction(interval_ns, NS_PER_S)
        )

    def byte_bound(self, interval_ns: int) -> Fraction:
        """Return the most bytes that a run spanning interval_ns may weigh."""
        return Fraction(self.max_bytes_per_s) * (
            Fraction(self.burst_s) + Fraction(interval_ns, NS_PER_S)
        )


@dataclass(frozen=True)
class Message:
    message_bytes: int
    now_ns: int


@dataclass(frozen=True)
class Traffic:
    """The settings and the construction time of a limiter, and the messages offered to it."""

    settings: Settings
    start_ns: int
    messages: tuple[Message, ...]


def whole_nanosecond_rates(max_twos: int, max_fives: int) -> st.SearchStrategy[float]:
    """Draw rates of 2**a * 5**b per second, with a from -2 and b from 0.

    Such a rate refills one token in a whole number of nanoseconds, because 10**9 is
    2**9 * 5**9. With a burst that is a whole number of nanoseconds too, a run can then meet a
    bound exactly, where any rounding error would refuse a message.
    """
    return st.builds(
        lambda twos, fives: 2.0**twos * 5**fives,
        st.integers(min_value=-2, max_value=max_twos),
        st.integers(min_value=0, max_value=max_fives),
    )


# The configuration's defaults, the demo's frame rate and a short burst, rates as above, and
# arbitrary settings. Message rates stay low enough that a full message bucket often holds fewer
# messages than a draw sends. As in the configuration, a full message bucket holds at least one
# message.
SETTINGS = st.builds(
    Settings,
    max_messages_per_s=st.sampled_from((128.0, 30.0))
    | whole_nanosecond_rates(max_twos=6, max_fives=3)
    | st.floats(min_value=0.5, max_value=500.0),
    max_bytes_per_s=st.just(8.0 * 2**20)
    | whole_nanosecond_rates(max_twos=9, max_fives=9)
    | st.floats(min_value=1.0, max_value=1e8),
    burst_s=st.sampled_from((1.0, 0.25)) | st.floats(min_value=0.01, max_value=10.0),
).filter(lambda settings: settings.message_bound(0) >= 1)
START_TIMES_NS = st.integers(min_value=0, max_value=2**62)


def message_refill_ns(settings: Settings) -> int:
    return math.ceil(NS_PER_S / Fraction(settings.max_messages_per_s))


def draw_list[T](draw: st.DrawFn, elements: st.SearchStrategy[T]) -> list[T]:
    """Draw a list whose length is drawn evenly from 0 to MAX_MESSAGES.

    A list strategy keeps most lists short, and a bucket runs out only after many messages.
    """
    count = draw(st.integers(min_value=0, max_value=MAX_MESSAGES))
    return draw(st.lists(elements, min_size=count, max_size=count))


@st.composite
def any_traffic(draw: st.DrawFn) -> Traffic:
    """Draw messages at one of three paces, sized around the byte capacity."""
    settings = draw(SETTINGS)
    byte_capacity = settings.byte_bound(0)
    # Sizes that share the byte capacity among a full burst of messages, so that either bucket
    # may run out first, and sizes up to twice the byte capacity, which only a bucket that
    # ignored its capacity could admit.
    sizes_bytes = st.integers(
        min_value=0, max_value=math.ceil(byte_capacity / settings.message_bound(0))
    ) | st.integers(min_value=0, max_value=math.ceil(2 * byte_capacity))
    # A sender far above the message rate, one around it, and one that bursts after idle gaps
    # of one to two times burst_s, the time a bucket takes to fill from empty, after which a
    # bucket that ignored its capacity would hold more than it.
    fill_ns = math.ceil(Fraction(settings.burst_s) * NS_PER_S)
    gaps_ns = draw(
        st.sampled_from(
            (
                st.integers(min_value=0, max_value=message_refill_ns(settings) // 4),
                st.integers(min_value=0, max_value=2 * message_refill_ns(settings)),
                st.just(0) | st.integers(min_value=fill_ns, max_value=2 * fill_ns),
            )
        )
    )
    start_ns = draw(START_TIMES_NS)
    gaps_and_sizes = draw_list(draw, st.tuples(gaps_ns, sizes_bytes))
    elapsed_ns = accumulate(gap_ns for gap_ns, _ in gaps_and_sizes)
    messages = tuple(
        Message(message_bytes, start_ns + offset_ns)
        for (_, message_bytes), offset_ns in zip(gaps_and_sizes, elapsed_ns, strict=True)
    )
    return Traffic(settings, start_ns, messages)


def earliest_within_bounds_ns(
    settings: Settings, sent: Sequence[Message], message_bytes: int, not_before_ns: int
) -> int:
    """Return the earliest nanosecond from not_before_ns at which a message keeps both bounds.

    The message follows the messages sent, and every run of messages that ends with it must be
    within both bounds.
    """
    seconds_per_message = 1 / Fraction(settings.max_messages_per_s)
    seconds_per_byte = 1 / Fraction(settings.max_bytes_per_s)
    earliest_ns = not_before_ns
    run_bytes = message_bytes
    for run_messages, earlier in enumerate(reversed(sent), start=2):
        run_bytes += earlier.message_bytes
        # t - t_i >= max(messages / rate, bytes / rate) - burst_s, in seconds.
        needed_s = max(run_messages * seconds_per_message, run_bytes * seconds_per_byte)
        earliest_ns = max(
            earliest_ns,
            earlier.now_ns + math.ceil((needed_s - Fraction(settings.burst_s)) * NS_PER_S),
        )
    return earliest_ns


@st.composite
def traffic_within_bounds(draw: st.DrawFn) -> Traffic:
    """Draw messages, each sent when every run up to it just meets both bounds, or a delay later.

    A run of one message is within the bounds because a full message bucket holds one message
    and no message weighs more than a full byte bucket.
    """
    settings = draw(SETTINGS)
    sizes_and_delays = draw_list(
        draw,
        st.tuples(
            st.integers(min_value=0, max_value=math.floor(settings.byte_bound(0))),
            st.just(0) | st.integers(min_value=0, max_value=message_refill_ns(settings)),
        ),
    )
    start_ns = draw(START_TIMES_NS)
    messages: list[Message] = []
    now_ns = start_ns
    for message_bytes, delay_ns in sizes_and_delays:
        now_ns = earliest_within_bounds_ns(settings, messages, message_bytes, now_ns) + delay_ns
        messages.append(Message(message_bytes, now_ns))
    return Traffic(settings, start_ns, tuple(messages))


def new_limiter(traffic: Traffic) -> IngressLimiter:
    settings = traffic.settings
    return IngressLimiter(
        settings.max_messages_per_s, settings.max_bytes_per_s, settings.burst_s, traffic.start_ns
    )


def admissions(traffic: Traffic) -> list[bool]:
    limiter = new_limiter(traffic)
    return [limiter.admit(message.message_bytes, message.now_ns) for message in traffic.messages]


def offer_after_history(traffic: Traffic, index: int, times_ns: Sequence[int]) -> list[bool]:
    """Replay the messages before index on a new limiter, then offer message index at times_ns."""
    limiter = new_limiter(traffic)
    for message in traffic.messages[:index]:
        limiter.admit(message.message_bytes, message.now_ns)
    message_bytes = traffic.messages[index].message_bytes
    return [limiter.admit(message_bytes, now_ns) for now_ns in times_ns]


def bound_violations(traffic: Traffic, admitted: Sequence[bool]) -> list[tuple[int, int]]:
    """Return every run, as its first and last index, whose admitted messages break a bound."""
    counts = list(accumulate((int(flag) for flag in admitted), initial=0))
    weights_bytes = list(
        accumulate(
            (
                message.message_bytes * flag
                for message, flag in zip(traffic.messages, admitted, strict=True)
            ),
            initial=0,
        )
    )
    times_ns = [message.now_ns for message in traffic.messages]
    settings = traffic.settings
    return [
        (first, last)
        for last in range(len(times_ns))
        for first in range(last + 1)
        if counts[last + 1] - counts[first]
        > settings.message_bound(times_ns[last] - times_ns[first])
        or weights_bytes[last + 1] - weights_bytes[first]
        > settings.byte_bound(times_ns[last] - times_ns[first])
    ]


@given(traffic=any_traffic())
def test_the_admitted_messages_of_every_run_stay_within_both_bounds(traffic: Traffic) -> None:
    assert bound_violations(traffic, admissions(traffic)) == []


@given(traffic=traffic_within_bounds())
def test_traffic_whose_every_run_is_within_both_bounds_is_admitted_whole(
    traffic: Traffic,
) -> None:
    assert all(admissions(traffic))


@given(traffic=any_traffic())
def test_a_message_refused_at_a_time_is_refused_again_at_that_time(traffic: Traffic) -> None:
    repeated = [
        offer_after_history(traffic, index, [message.now_ns, message.now_ns])
        for index, message in enumerate(traffic.messages)
    ]

    assert all(first or not second for first, second in repeated)


@given(
    traffic=any_traffic(),
    delay_ns=st.integers(min_value=0, max_value=10) | st.integers(min_value=0, max_value=NS_PER_S),
)
def test_a_message_admitted_at_a_time_is_admitted_at_any_later_time_after_the_same_history(
    traffic: Traffic, delay_ns: int
) -> None:
    at_time = [
        offer_after_history(traffic, index, [message.now_ns])[0]
        for index, message in enumerate(traffic.messages)
    ]
    later = [
        offer_after_history(traffic, index, [message.now_ns + delay_ns])[0]
        for index, message in enumerate(traffic.messages)
    ]

    assert all(
        not admitted or admitted_later
        for admitted, admitted_later in zip(at_time, later, strict=True)
    )
