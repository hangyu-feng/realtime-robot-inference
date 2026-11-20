"""Tests for admission: the token and the cap, the ingress limit and the frame header check."""

from collections.abc import Sequence
from dataclasses import dataclass

import pytest

from cloud.admission import IngressLimiter, RejectReason, admission_refusal, frame_header_valid

SHARED_TOKEN = "fleet-token"
MAX_CONNECTIONS = 4

# A full bucket holds 2 messages and 500 bytes. One message refills in 250 ms, and in that time
# 250 bytes refill.
MESSAGES_PER_S = 4.0
BYTES_PER_S = 1_000.0
BURST_S = 0.5
MESSAGE_REFILL_NS = 250_000_000
START_NS = 5_000

# A frame header that passes every rule unless a test changes it: a fire lag of 400 ns and an
# uplink of 500 ns, both far below the maximum age.
SESSION = 7
CAPTURE_NS = 1_000_000
SEND_NS = CAPTURE_NS + 400
RECEIVED_NS = SEND_NS + 500
MAX_AGE_NS = 5_000


@dataclass(frozen=True)
class Header:
    """The arguments of one frame_header_valid call that the tests vary."""

    session: int = SESSION
    frame_number: int = 10
    send_ns: int = SEND_NS
    received_ns: int = RECEIVED_NS
    last_frame_number: int | None = 9
    shared_clock: bool = True


def valid(header: Header) -> bool:
    return frame_header_valid(
        header.session,
        header.frame_number,
        CAPTURE_NS,
        header.send_ns,
        header.received_ns,
        connection_session=SESSION,
        last_frame_number=header.last_frame_number,
        shared_clock=header.shared_clock,
        max_age_ns=MAX_AGE_NS,
    )


def new_limiter() -> IngressLimiter:
    return IngressLimiter(MESSAGES_PER_S, BYTES_PER_S, BURST_S, START_NS)


def admit_all(limiter: IngressLimiter, sizes_bytes: Sequence[int], now_ns: int) -> list[bool]:
    return [limiter.admit(message_bytes, now_ns) for message_bytes in sizes_bytes]


def test_reject_reasons_are_the_labels_of_connections_rejected() -> None:
    assert [reason.value for reason in RejectReason] == ["auth", "capacity"]


@pytest.mark.parametrize("token", [None, "any-token"], ids=["missing", "given"])
def test_without_a_shared_token_no_token_is_checked(token: str | None) -> None:
    refusal = admission_refusal(
        token, shared_token=None, connected=0, max_connections=MAX_CONNECTIONS
    )

    assert refusal is None


def test_the_shared_token_is_admitted() -> None:
    refusal = admission_refusal(
        SHARED_TOKEN, shared_token=SHARED_TOKEN, connected=0, max_connections=MAX_CONNECTIONS
    )

    assert refusal is None


@pytest.mark.parametrize(
    "token",
    [None, "fleet-tokeN", "other-token", "fleet-token-2", "fleet"],
    ids=["missing", "last-character-off", "same-length", "longer", "prefix"],
)
def test_a_missing_or_wrong_token_is_refused_for_auth(token: str | None) -> None:
    refusal = admission_refusal(
        token, shared_token=SHARED_TOKEN, connected=0, max_connections=MAX_CONNECTIONS
    )

    assert refusal is RejectReason.AUTH


def test_the_token_is_checked_before_the_cap() -> None:
    refusal = admission_refusal(
        "other-token",
        shared_token=SHARED_TOKEN,
        connected=MAX_CONNECTIONS,
        max_connections=MAX_CONNECTIONS,
    )

    assert refusal is RejectReason.AUTH


@pytest.mark.parametrize("shared_token", [None, SHARED_TOKEN], ids=["token-off", "token-on"])
@pytest.mark.parametrize(
    ("connected", "refusal"),
    [
        (0, None),
        (MAX_CONNECTIONS - 1, None),
        (MAX_CONNECTIONS, RejectReason.CAPACITY),
        (MAX_CONNECTIONS + 1, RejectReason.CAPACITY),
    ],
    ids=["none-open", "one-below-the-cap", "at-the-cap", "above-the-cap"],
)
def test_the_cap_refuses_a_hello_once_max_connections_sessions_are_open(
    shared_token: str | None, connected: int, refusal: RejectReason | None
) -> None:
    result = admission_refusal(
        SHARED_TOKEN,
        shared_token=shared_token,
        connected=connected,
        max_connections=MAX_CONNECTIONS,
    )

    assert result is refusal


@pytest.mark.parametrize(
    ("message_bytes", "admitted"),
    [(500, True), (501, False)],
    ids=["the-whole-byte-capacity", "one-byte-more"],
)
def test_a_new_limiter_starts_with_full_buckets(message_bytes: int, admitted: bool) -> None:
    assert new_limiter().admit(message_bytes, START_NS) is admitted


def test_a_burst_up_to_the_capacity_is_admitted_at_once_and_the_next_message_refused() -> None:
    assert admit_all(new_limiter(), [200, 300, 0], START_NS) == [True, True, False]


def test_a_refused_message_takes_nothing_from_either_bucket() -> None:
    # The first message leaves one message and 200 bytes. Had the refused second one taken the
    # last message or any bytes, the third could not pass.
    assert admit_all(new_limiter(), [300, 201, 200], START_NS) == [True, False, True]


@pytest.mark.parametrize(
    ("elapsed_ns", "message_bytes", "admitted"),
    [
        (MESSAGE_REFILL_NS - 1, 0, False),
        (MESSAGE_REFILL_NS, 0, True),
        (MESSAGE_REFILL_NS, 250, True),
        (MESSAGE_REFILL_NS, 251, False),
    ],
    ids=["one-ns-before-a-message", "a-message", "its-bytes", "one-byte-more"],
)
def test_both_buckets_refill_at_their_rates(
    elapsed_ns: int, message_bytes: int, admitted: bool
) -> None:
    limiter = new_limiter()
    admit_all(limiter, [250, 250], START_NS)

    assert limiter.admit(message_bytes, START_NS + elapsed_ns) is admitted


@pytest.mark.parametrize(
    ("burst_sizes_bytes", "message_bytes"),
    [([250, 250], 250), ([0, 0], 0), ([500], 250)],
    ids=["refused-by-both-buckets", "refused-by-the-message-bucket", "refused-by-the-byte-bucket"],
)
def test_a_refused_message_keeps_the_refill_since_the_previous_call(
    burst_sizes_bytes: list[int], message_bytes: int
) -> None:
    # Half a refill after the burst, each short bucket holds half of what the message needs and
    # the message is refused. One refill after the burst it passes only if the refusal kept
    # that half.
    limiter = new_limiter()
    admit_all(limiter, burst_sizes_bytes, START_NS)

    admitted = [
        limiter.admit(message_bytes, START_NS + MESSAGE_REFILL_NS // 2),
        limiter.admit(message_bytes, START_NS + MESSAGE_REFILL_NS),
    ]

    assert admitted == [False, True]


def test_a_message_that_meets_the_bound_exactly_after_uneven_gaps_is_admitted() -> None:
    # A full bucket holds 6.25 * 0.25 = 1.5625 messages, and three messages within 230 ms meet
    # the bound 6.25 * (0.25 + 0.23) = 3 exactly. The uneven gaps refill amounts that are not
    # binary fractions, so a float level can end a rounding error short of the third message.
    limiter = IngressLimiter(6.25, BYTES_PER_S, 0.25, 0)

    admitted = [limiter.admit(0, now_ns) for now_ns in (0, 71_333_170, 230_000_000)]

    assert admitted == [True, True, True]


@pytest.mark.parametrize(
    ("sizes_bytes", "admitted"),
    [([250, 250, 0], [True, True, False]), ([500, 1], [True, False])],
    ids=["messages", "bytes"],
)
def test_a_long_idle_refills_no_bucket_above_its_capacity(
    sizes_bytes: list[int], admitted: list[bool]
) -> None:
    limiter = new_limiter()
    admit_all(limiter, [250, 250], START_NS)

    assert admit_all(limiter, sizes_bytes, START_NS + 3_600 * 10**9) == admitted


@pytest.mark.parametrize(
    ("sizes_bytes", "admitted"),
    [([0, 0, 0], [True, True, False]), ([500, 1, 0], [True, False, True])],
    ids=["message-bucket-empty-byte-bucket-full", "byte-bucket-empty-a-message-left"],
)
def test_each_bucket_refuses_on_its_own(sizes_bytes: list[int], admitted: list[bool]) -> None:
    assert admit_all(new_limiter(), sizes_bytes, START_NS) == admitted


def test_a_call_before_the_construction_time_raises() -> None:
    with pytest.raises(ValueError, match="now_ns=4999 is before"):
        new_limiter().admit(0, START_NS - 1)


def test_a_call_before_the_previous_call_raises_and_changes_nothing() -> None:
    # The step back is a whole message refill, so a limiter that kept the bad reading as its
    # previous call would refill that interval a second time and admit both messages after it.
    limiter = new_limiter()
    limiter.admit(0, START_NS + MESSAGE_REFILL_NS)

    with pytest.raises(ValueError, match="now_ns=5000 is before"):
        limiter.admit(0, START_NS)

    assert admit_all(limiter, [0, 0], START_NS + MESSAGE_REFILL_NS) == [True, False]


@pytest.mark.parametrize("shared_clock", [True, False])
def test_a_frame_header_that_meets_every_rule_is_valid(shared_clock: bool) -> None:
    assert valid(Header(shared_clock=shared_clock))


@pytest.mark.parametrize("session", [SESSION - 1, SESSION + 1], ids=["lower", "higher"])
def test_a_frame_of_another_session_is_refused(session: int) -> None:
    assert not valid(Header(session=session))


@pytest.mark.parametrize(
    ("last_frame_number", "expected"),
    [(None, True), (9, True), (10, False), (11, False)],
    ids=["first-frame", "last-one-below", "last-one-equal", "last-one-above"],
)
def test_a_frame_number_must_be_above_the_last_accepted_one(
    last_frame_number: int | None, expected: bool
) -> None:
    assert valid(Header(frame_number=10, last_frame_number=last_frame_number)) is expected


@pytest.mark.parametrize("shared_clock", [True, False])
@pytest.mark.parametrize(
    ("fire_lag_ns", "expected"),
    [(-1, False), (0, True), (MAX_AGE_NS, True), (MAX_AGE_NS + 1, False)],
    ids=["send-before-capture", "send-at-capture", "at-max-age", "above-max-age"],
)
def test_the_send_time_must_be_from_zero_to_max_age_after_the_capture_time(
    shared_clock: bool, fire_lag_ns: int, expected: bool
) -> None:
    send_ns = CAPTURE_NS + fire_lag_ns
    header = Header(send_ns=send_ns, received_ns=send_ns, shared_clock=shared_clock)

    assert valid(header) is expected


@pytest.mark.parametrize(
    ("received_ns", "expected"),
    [
        (SEND_NS - 1, False),
        (SEND_NS, True),
        (CAPTURE_NS + MAX_AGE_NS, True),
        (CAPTURE_NS + MAX_AGE_NS + 1, False),
    ],
    ids=["before-the-send", "at-the-send", "age-at-max", "age-above-max"],
)
def test_on_a_shared_clock_a_frame_arrives_after_its_send_and_within_max_age_of_capture(
    received_ns: int, expected: bool
) -> None:
    assert valid(Header(received_ns=received_ns)) is expected


@pytest.mark.parametrize(
    "received_ns",
    [SEND_NS - 1, CAPTURE_NS + MAX_AGE_NS + 1],
    ids=["before-the-send", "age-above-max"],
)
def test_without_a_shared_clock_the_receive_time_is_never_compared(received_ns: int) -> None:
    assert valid(Header(received_ns=received_ns, shared_clock=False))
