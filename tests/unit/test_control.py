"""Tests for the robot-side command rules: the stale and order verdicts and the watchdog."""

from collections.abc import Callable
from enum import StrEnum

import pytest

from robot.control import CommandOutcome, ModelControl, Verdict, WatchdogState

# Every time below is in nanoseconds on the robot's clock, and every command is judged
# against these values unless a test says otherwise.
SESSION = 5
STALE_NS = 150
WATCHDOG_NS = 300
START_NS = 1_000

# Commands as (frame number, capture time, receive time).
WATCHDOG_TIMELINE = (
    # Applied at age 100; the 100 ns since the start count as not covered. Ok up to 1_300.
    (1, 1_000, 1_100),
    # Applied before the first one expired: no trip. Ok up to 1_500.
    (2, 1_200, 1_250),
    # Applied 200 ns after the second one expired: one trip. Ok up to 1_900.
    (3, 1_600, 1_700),
    # Stale at age 160, so the third stays in effect.
    (4, 1_640, 1_800),
)

# Commands as (session, frame number, capture time, receive time).
MIXED_SEQUENCE = (
    (SESSION, 1, 1_000, 1_100),  # applied
    (SESSION, 1, 1_100, 1_180),  # out of order: frame number equal to the applied one
    (SESSION + 1, 2, 1_150, 1_220),  # out of order: another session
    (SESSION, 2, 1_150, 1_250),  # applied, replacing frame 1 at age 250
    (SESSION, 3, 1_300, 1_500),  # stale at age 200
    (SESSION, 4, 1_500, 1_600),  # applied, replacing frame 2 (not the stale one) at age 450
)

# A second command, received at 1_200 after frame 1 was applied at 1_100, with each verdict it
# can get: (session, frame number, capture time, verdict).
SECOND_COMMAND_WITH_EACH_VERDICT = pytest.mark.parametrize(
    ("session", "frame_number", "capture_ns", "verdict"),
    [
        (SESSION, 2, 1_150, Verdict.APPLIED),
        (SESSION, 2, 1_200 - STALE_NS - 1, Verdict.STALE),
        (SESSION, 1, 1_150, Verdict.OUT_OF_ORDER),
        (SESSION + 1, 2, 1_150, Verdict.OUT_OF_ORDER),
    ],
    ids=["applied", "stale", "out-of-order", "other-session"],
)

QUERIES: tuple[Callable[[ModelControl, int], object], ...] = (
    ModelControl.watchdog_state,
    ModelControl.watchdog_trips,
    ModelControl.watchdog_expired_ns,
)
QUERY_IDS = ["state", "trips", "expired"]


def new_control() -> ModelControl:
    return ModelControl(SESSION, STALE_NS, WATCHDOG_NS, START_NS)


def watchdog_at(control: ModelControl, now_ns: int) -> tuple[WatchdogState, int, int]:
    return (
        control.watchdog_state(now_ns),
        control.watchdog_trips(now_ns),
        control.watchdog_expired_ns(now_ns),
    )


@pytest.mark.parametrize(
    ("members", "strings"),
    [
        # common/protocol.py cannot import robot/, so it checks the watchdog_state of telemetry
        # against its own copy of these strings and refuses a report with any other.
        (list(WatchdogState), ["none", "ok", "expired"]),
        (list(Verdict), ["applied", "stale", "out_of_order"]),
    ],
    ids=["watchdog-state", "verdict"],
)
def test_watchdog_states_and_verdicts_are_exactly_these_strings(
    members: list[StrEnum], strings: list[str]
) -> None:
    assert [member.value for member in members] == strings


@pytest.mark.parametrize(
    ("age_ns", "verdict"),
    [(0, Verdict.APPLIED), (STALE_NS, Verdict.APPLIED), (STALE_NS + 1, Verdict.STALE)],
    ids=["fresh", "at-the-threshold", "above-the-threshold"],
)
def test_a_command_is_stale_only_when_its_age_is_above_the_threshold(
    age_ns: int, verdict: Verdict
) -> None:
    outcome = new_control().on_command(SESSION, 1, 2_000 - age_ns, 2_000)

    assert (outcome.verdict, outcome.freshness_ns) == (verdict, age_ns)


@pytest.mark.parametrize(
    ("frame_number", "verdict"),
    [(9, Verdict.OUT_OF_ORDER), (10, Verdict.OUT_OF_ORDER), (11, Verdict.APPLIED)],
    ids=["lower", "equal", "higher"],
)
def test_a_frame_number_not_above_the_last_applied_one_is_out_of_order(
    frame_number: int, verdict: Verdict
) -> None:
    control = new_control()
    control.on_command(SESSION, 10, 1_000, 1_010)

    assert control.on_command(SESSION, frame_number, 1_020, 1_030).verdict is verdict


@pytest.mark.parametrize("session", [SESSION - 1, SESSION + 1])
def test_a_command_of_another_session_is_out_of_order(session: int) -> None:
    outcome = new_control().on_command(session, 1, 1_000, 1_010)

    assert outcome.verdict is Verdict.OUT_OF_ORDER


@pytest.mark.parametrize(
    ("session", "frame_number"),
    [(SESSION + 1, 11), (SESSION, 10), (SESSION + 1, 9)],
    ids=["other-session", "equal-frame-number", "other-session-and-lower-frame-number"],
)
def test_a_command_both_stale_and_out_of_order_counts_as_stale(
    session: int, frame_number: int
) -> None:
    control = new_control()
    control.on_command(SESSION, 10, 1_000, 1_010)

    outcome = control.on_command(session, frame_number, 1_020, 1_021 + STALE_NS)

    assert outcome.verdict is Verdict.STALE
    assert (control.stale, control.out_of_order) == (1, 0)


@pytest.mark.parametrize(
    ("session", "capture_ns"),
    [(SESSION, 1_020 - STALE_NS - 1), (SESSION + 1, 1_015)],
    ids=["stale", "other-session"],
)
def test_a_rejected_command_does_not_raise_the_last_applied_frame_number(
    session: int, capture_ns: int
) -> None:
    control = new_control()
    control.on_command(SESSION, 5, 1_005, 1_010)
    control.on_command(session, 8, capture_ns, 1_020)

    assert control.on_command(SESSION, 7, 1_025, 1_030).verdict is Verdict.APPLIED


@pytest.mark.parametrize(
    ("frame_number", "capture_ns", "receive_ns"),
    [(6, 1_050, 1_051 + STALE_NS), (4, 1_200, 1_210)],
    ids=["stale", "out-of-order"],
)
def test_a_rejected_command_does_not_replace_the_command_in_effect(
    frame_number: int, capture_ns: int, receive_ns: int
) -> None:
    control = new_control()
    control.on_command(SESSION, 5, 1_000, 1_010)
    control.on_command(SESSION, frame_number, capture_ns, receive_ns)

    assert control.watchdog_state(1_000 + WATCHDOG_NS + 1) is WatchdogState.EXPIRED


def test_a_watchdog_limit_below_the_stale_threshold_is_refused() -> None:
    with pytest.raises(ValueError, match="below stale_ns"):
        ModelControl(SESSION, STALE_NS, STALE_NS - 1, START_NS)


def test_an_equal_watchdog_limit_keeps_a_command_at_the_stale_threshold_ok() -> None:
    control = ModelControl(SESSION, STALE_NS, STALE_NS, START_NS)

    outcome = control.on_command(SESSION, 1, 1_100, 1_100 + STALE_NS)

    assert outcome.verdict is Verdict.APPLIED
    assert control.watchdog_state(1_100 + STALE_NS) is WatchdogState.OK


def test_a_command_received_before_the_session_start_raises() -> None:
    with pytest.raises(ValueError, match="now_ns=999 is before"):
        new_control().on_command(SESSION, 1, 990, 999)


@SECOND_COMMAND_WITH_EACH_VERDICT
def test_a_command_received_before_the_previous_one_raises_and_counts_nothing(
    session: int, frame_number: int, capture_ns: int, verdict: Verdict
) -> None:
    control = new_control()
    control.on_command(SESSION, 1, 1_000, 1_100)
    second = control.on_command(session, frame_number, capture_ns, 1_200)

    with pytest.raises(ValueError, match="now_ns=1199 is before"):
        control.on_command(SESSION, 3, 1_150, 1_199)

    assert second.verdict is verdict
    assert control.received == 2


@SECOND_COMMAND_WITH_EACH_VERDICT
@pytest.mark.parametrize("query", QUERIES, ids=QUERY_IDS)
def test_a_watchdog_query_before_the_last_command_raises(
    query: Callable[[ModelControl, int], object],
    session: int,
    frame_number: int,
    capture_ns: int,
    verdict: Verdict,
) -> None:
    control = new_control()
    control.on_command(SESSION, 1, 1_000, 1_100)
    second = control.on_command(session, frame_number, capture_ns, 1_200)

    with pytest.raises(ValueError, match="now_ns=1199 is before"):
        query(control, 1_199)

    assert second.verdict is verdict


@pytest.mark.parametrize("query", QUERIES, ids=QUERY_IDS)
def test_a_watchdog_query_before_the_session_start_raises(
    query: Callable[[ModelControl, int], object],
) -> None:
    with pytest.raises(ValueError, match="now_ns=999 is before"):
        query(new_control(), 999)


@pytest.mark.parametrize(
    ("commands_received", "now_ns", "state", "trips", "expired_ns"),
    [
        (0, 1_000, WatchdogState.NONE, 0, 0),
        (0, 1_100, WatchdogState.NONE, 0, 100),
        (1, 1_100, WatchdogState.OK, 0, 100),
        (2, 1_500, WatchdogState.OK, 0, 100),
        (2, 1_501, WatchdogState.EXPIRED, 1, 101),
        (2, 1_700, WatchdogState.EXPIRED, 1, 300),
        (3, 1_700, WatchdogState.OK, 1, 300),
        (4, 1_900, WatchdogState.OK, 1, 300),
        (4, 1_901, WatchdogState.EXPIRED, 2, 301),
        (4, 2_000, WatchdogState.EXPIRED, 2, 400),
    ],
)
def test_the_watchdog_follows_a_timeline_of_commands(
    commands_received: int, now_ns: int, state: WatchdogState, trips: int, expired_ns: int
) -> None:
    control = new_control()
    for frame_number, capture_ns, receive_ns in WATCHDOG_TIMELINE[:commands_received]:
        control.on_command(SESSION, frame_number, capture_ns, receive_ns)

    assert watchdog_at(control, now_ns) == (state, trips, expired_ns)


def test_a_session_whose_every_command_is_stale_is_never_covered() -> None:
    control = new_control()
    control.on_command(SESSION, 1, 1_000, 1_151)
    control.on_command(SESSION, 2, 1_100, 1_400)

    assert watchdog_at(control, 1_500) == (WatchdogState.NONE, 0, 500)
    assert control.stale == 2


@pytest.mark.parametrize(
    ("now_ns", "state", "trips", "expired_ns"),
    [
        (1_050 + WATCHDOG_NS, WatchdogState.OK, 0, 100),
        (1_051 + WATCHDOG_NS, WatchdogState.EXPIRED, 1, 101),
        (2_000, WatchdogState.EXPIRED, 1, 750),
    ],
    ids=["at-the-limit", "one-ns-past-the-limit", "long-past-the-limit"],
)
def test_a_session_that_ends_expired_counts_the_expiry_still_in_effect(
    now_ns: int, state: WatchdogState, trips: int, expired_ns: int
) -> None:
    control = new_control()
    control.on_command(SESSION, 1, 1_050, 1_100)

    assert watchdog_at(control, now_ns) == (state, trips, expired_ns)


@pytest.mark.parametrize(
    ("receive_ns", "trips", "expired_ns"),
    [(1_050 + WATCHDOG_NS, 0, 100), (1_051 + WATCHDOG_NS, 1, 101)],
    ids=["at-the-limit", "one-ns-past-the-limit"],
)
def test_a_command_replaced_only_after_its_watchdog_limit_counts_one_trip(
    receive_ns: int, trips: int, expired_ns: int
) -> None:
    control = new_control()
    control.on_command(SESSION, 1, 1_050, 1_100)
    control.on_command(SESSION, 2, receive_ns - 10, receive_ns)

    assert watchdog_at(control, receive_ns) == (WatchdogState.OK, trips, expired_ns)


def test_gap_follows_every_received_command_and_replaced_age_only_applied_ones() -> None:
    control = new_control()

    outcomes = [
        control.on_command(session, frame_number, capture_ns, receive_ns)
        for session, frame_number, capture_ns, receive_ns in MIXED_SEQUENCE
    ]

    assert outcomes == [
        CommandOutcome(Verdict.APPLIED, freshness_ns=100, gap_ns=None, replaced_age_ns=None),
        CommandOutcome(Verdict.OUT_OF_ORDER, freshness_ns=80, gap_ns=80, replaced_age_ns=None),
        CommandOutcome(Verdict.OUT_OF_ORDER, freshness_ns=70, gap_ns=40, replaced_age_ns=None),
        CommandOutcome(Verdict.APPLIED, freshness_ns=100, gap_ns=30, replaced_age_ns=250),
        CommandOutcome(Verdict.STALE, freshness_ns=200, gap_ns=250, replaced_age_ns=None),
        CommandOutcome(Verdict.APPLIED, freshness_ns=100, gap_ns=100, replaced_age_ns=450),
    ]


def test_counters_count_each_verdict_and_received_is_their_sum() -> None:
    control = new_control()

    for session, frame_number, capture_ns, receive_ns in MIXED_SEQUENCE:
        control.on_command(session, frame_number, capture_ns, receive_ns)

    assert (control.received, control.applied, control.stale, control.out_of_order) == (6, 3, 1, 2)
