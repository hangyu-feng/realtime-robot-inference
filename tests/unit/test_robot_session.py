"""Tests for one robot session: ticks, the skip and stall rules, commands and telemetry."""

from pathlib import Path

import pytest

from common.commands import ClsCommand, DepthCommand, encode_command
from common.protocol import (
    PROTOCOL_VERSION,
    CommandHeader,
    FrameHeader,
    ProtocolError,
    StageDurations,
    Telemetry,
    Welcome,
    decode_frame,
    decode_telemetry,
    encode_command_message,
)
from robot.samples import SampleLog, load_samples
from robot.schedule import CaptureSchedule
from robot.session import RobotSession
from robot.settings import RobotSettings

MS = 1_000_000
SECOND = 1_000_000_000
SESSION = 5
# Ticks at 0, 10 ms, 20 ms, ...
SCHEDULE = CaptureSchedule(start_ns=0, period_ns=10 * MS)
JPEGS = (b"\xff\xd8\xffone", b"\xff\xd8\xfftwo", b"\xff\xd8\xffthree")
SETTINGS = RobotSettings(stall_timeout_ms=30.0)
BODIES = {
    "cls": encode_command(ClsCommand(class_id=7, confidence=0.5)),
    "depth": encode_command(DepthCommand(0.1, 0.2, 0.3, 0.2, 0.8)),
}


def session_from(
    start_ns: int = 0,
    settings: RobotSettings = SETTINGS,
    shared_clock: bool = True,
    samples: SampleLog | None = None,
) -> RobotSession:
    welcome = Welcome(PROTOCOL_VERSION, SESSION, shared_clock)
    return RobotSession("r01", settings, welcome, SCHEDULE, JPEGS, start_ns, samples)


def command(
    model: str, frame_number: int, capture_ns: int, result_ns: int = 0, session: int = SESSION
) -> bytes:
    header = CommandHeader(
        model=model,
        session=session,
        frame_number=frame_number,
        capture_ns=capture_ns,
        batch_size=2,
        result_ns=result_ns,
        stages=StageDurations(1, 2, 3, 4, 5, 6, 7, 8),
    )
    return encode_command_message(header, BODIES[model])


def report(session: RobotSession, now_ns: int, reconnects: int = 0) -> Telemetry:
    return decode_telemetry(session.telemetry(now_ns, reconnects))


def sent_frame(session: RobotSession, now_ns: int) -> tuple[FrameHeader, bytes]:
    """Tick on a free path, and return the header and the JPEG of the frame the tick sends."""
    message = session.tick(now_ns, path_busy=False)
    assert message is not None
    header, jpeg = decode_frame(message)
    return header, bytes(jpeg)


def frame_counts(session: RobotSession, now_ns: int) -> tuple[int, int, int]:
    telemetry = report(session, now_ns)
    return telemetry.frames_captured, telemetry.frames_sent, telemetry.frames_skipped_send


def test_the_first_frame_after_a_welcome_is_the_next_due_tick() -> None:
    session = session_from(start_ns=25 * MS)

    assert session.next_frame == 3
    assert session.tick(29 * MS, path_busy=False) is None
    header, _ = sent_frame(session, 30 * MS)
    assert header.frame_number == 3
    assert frame_counts(session, 30 * MS) == (1, 1, 0)


def test_a_session_that_begins_before_the_schedule_starts_with_frame_zero() -> None:
    schedule = CaptureSchedule(start_ns=50 * MS, period_ns=10 * MS)
    session = RobotSession(
        "r01", SETTINGS, Welcome(PROTOCOL_VERSION, SESSION, True), schedule, JPEGS, 0, None
    )

    assert session.next_frame == 0
    assert session.tick(49 * MS, path_busy=False) is None
    header, _ = sent_frame(session, 50 * MS)
    assert header == FrameHeader(SESSION, 0, 50 * MS, 50 * MS)


def test_a_frame_carries_its_session_number_scheduled_capture_time_and_send_time() -> None:
    session = session_from()

    header, jpeg = sent_frame(session, 43 * MS)

    assert header == FrameHeader(
        session=SESSION, frame_number=4, capture_ns=40 * MS, send_ns=43 * MS
    )
    assert jpeg == JPEGS[4 % 3]


def test_a_late_wake_sends_only_the_newest_due_frame_and_counts_the_ticks_it_passed(
    tmp_path: Path,
) -> None:
    samples = SampleLog(tick_capacity=16, command_capacity=16)
    session = session_from(samples=samples)

    header, _ = sent_frame(session, 45 * MS)

    assert header.frame_number == 4
    assert session.next_frame == 5
    assert frame_counts(session, 45 * MS) == (4, 1, 3)
    samples.save(tmp_path / "robot_r01.npz", {})
    _, ticks, _ = load_samples(tmp_path / "robot_r01.npz")
    assert ticks.tolist() == [
        (1, 10 * MS, 0, 2),
        (2, 20 * MS, 0, 2),
        (3, 30 * MS, 0, 2),
        (4, 40 * MS, 45 * MS, 0),
    ]


def test_a_tick_is_handled_once() -> None:
    session = session_from()
    session.tick(10 * MS, path_busy=False)

    assert session.tick(19 * MS, path_busy=False) is None
    assert frame_counts(session, 19 * MS) == (1, 1, 0)


def test_a_busy_path_skips_the_tick_and_a_frame_number_skipped_is_never_sent_late(
    tmp_path: Path,
) -> None:
    samples = SampleLog(tick_capacity=16, command_capacity=16)
    session = session_from(samples=samples)

    assert session.tick(10 * MS, path_busy=True) is None
    header, _ = sent_frame(session, 20 * MS)

    assert header.frame_number == 2
    assert frame_counts(session, 20 * MS) == (2, 1, 1)
    samples.save(tmp_path / "robot_r01.npz", {})
    _, ticks, _ = load_samples(tmp_path / "robot_r01.npz")
    assert ticks.tolist() == [(1, 10 * MS, 0, 1), (2, 20 * MS, 20 * MS, 0)]


def test_the_path_has_stalled_once_it_was_busy_at_every_tick_for_the_stall_timeout() -> None:
    session = session_from()
    session.tick(10 * MS, path_busy=True)
    session.tick(20 * MS, path_busy=True)
    session.tick(30 * MS, path_busy=True)

    assert not session.stalled(40 * MS - 1)
    assert session.stalled(40 * MS)


def test_a_frame_that_leaves_ends_the_stall_and_the_next_starts_at_its_first_skip() -> None:
    session = session_from()
    session.tick(10 * MS, path_busy=True)
    session.tick(40 * MS, path_busy=False)
    assert not session.stalled(100 * MS)

    session.tick(50 * MS, path_busy=True)

    assert not session.stalled(80 * MS - 1)
    assert session.stalled(80 * MS)


def test_a_session_with_no_skipped_tick_has_not_stalled() -> None:
    session = session_from()
    session.tick(10 * MS, path_busy=False)

    assert not session.stalled(10 * SECOND)


@pytest.mark.parametrize(
    "ticks",
    [
        [(10, False), (20, False), (30, False)],
        [(10, True), (20, True), (30, False), (40, True)],
        [(15, False), (95, False), (96, True), (300, True), (301, False)],
        [(5, False), (10, True), (1000, False), (1010, False)],
    ],
)
def test_every_tick_captured_is_either_sent_or_skipped(ticks: list[tuple[int, bool]]) -> None:
    session = session_from()
    counts = []
    for now_ms, path_busy in ticks:
        session.tick(now_ms * MS, path_busy)
        counts.append(frame_counts(session, now_ms * MS))

    assert all(captured == sent + skipped for captured, sent, skipped in counts)
    assert [captured for captured, _, _ in counts] == [now_ms // 10 for now_ms, _ in ticks]


def test_the_fire_lag_of_each_sent_frame_is_recorded_in_milliseconds() -> None:
    session = session_from()
    session.tick(10 * MS, path_busy=False)
    session.tick(20 * MS + 2_500_000, path_busy=False)
    session.tick(30 * MS, path_busy=True)

    fire_lag_ms = report(session, 30 * MS).fire_lag_ms

    assert (fire_lag_ms.count, fire_lag_ms.min_ms, fire_lag_ms.max_ms) == (2, 0.0, 2.5)


def test_the_first_report_is_due_one_second_after_the_session_starts() -> None:
    session = session_from(start_ns=250 * MS)

    assert not session.telemetry_due(1250 * MS - 1)
    assert session.telemetry_due(1250 * MS)


@pytest.mark.parametrize(
    ("sent_ns", "next_due_ns"),
    [
        (1 * SECOND, 2 * SECOND),
        (1 * SECOND + 20 * MS, 2 * SECOND),
        (2 * SECOND - 1, 2 * SECOND),
        (3 * SECOND + 500 * MS, 4 * SECOND),
        # A report sent before its due time, as the last one of a session is.
        (400 * MS, 1 * SECOND),
    ],
)
def test_a_report_moves_the_next_one_by_whole_seconds_past_its_own_time(
    sent_ns: int, next_due_ns: int
) -> None:
    session = session_from()

    session.telemetry(sent_ns, 0)

    assert not session.telemetry_due(next_due_ns - 1)
    assert session.telemetry_due(next_due_ns)


def test_a_report_carries_its_time_the_reconnects_and_every_subscribed_model() -> None:
    session = session_from()

    telemetry = report(session, 1 * SECOND, reconnects=3)

    assert (telemetry.ts_mono_ns, telemetry.reconnects) == (1 * SECOND, 3)
    assert set(telemetry.models) == {"cls", "depth"}


def test_a_session_subscribed_to_one_model_reports_that_model_only() -> None:
    settings = RobotSettings(
        models=("depth",), stale_ms={"depth": 150.0}, watchdog_ms={"depth": 300.0}
    )
    session = session_from(settings=settings)

    assert set(report(session, 1 * SECOND).models) == {"depth"}
    assert set(session.controls) == {"depth"}


def test_commands_get_the_verdicts_of_the_stale_and_order_rules(tmp_path: Path) -> None:
    samples = SampleLog(tick_capacity=16, command_capacity=16)
    session = session_from(samples=samples)

    # Depth is stale above 150 ms: applied, then an older frame, then one too old.
    session.receive(command("depth", 5, capture_ns=50 * MS, result_ns=90 * MS), 100 * MS)
    session.receive(command("depth", 4, capture_ns=40 * MS, result_ns=110 * MS), 120 * MS)
    session.receive(command("depth", 6, capture_ns=60 * MS, result_ns=215 * MS), 220 * MS)
    session.receive(command("depth", 30, capture_ns=300 * MS, result_ns=330 * MS), 340 * MS)

    depth = report(session, 340 * MS).models["depth"]
    assert (
        depth.commands_received,
        depth.commands_applied,
        depth.commands_out_of_order,
        depth.commands_stale,
    ) == (4, 2, 1, 1)
    samples.save(tmp_path / "robot_r01.npz", {})
    commands, _, _ = load_samples(tmp_path / "robot_r01.npz")
    assert commands[["session", "model", "verdict", "batch_size", "frame_number"]].tolist() == [
        (SESSION, 1, 0, 2, 5),
        (SESSION, 1, 2, 2, 4),
        (SESSION, 1, 1, 2, 6),
        (SESSION, 1, 0, 2, 30),
    ]
    assert commands[["capture_ns", "receive_ns", "result_ns"]].tolist() == [
        (50 * MS, 100 * MS, 90 * MS),
        (40 * MS, 120 * MS, 110 * MS),
        (60 * MS, 220 * MS, 215 * MS),
        (300 * MS, 340 * MS, 330 * MS),
    ]
    assert commands["stage_us"].tolist() == [[1, 2, 3, 4, 5, 6, 7, 8]] * 4


def test_a_command_of_another_session_is_out_of_order_and_recorded_under_this_session(
    tmp_path: Path,
) -> None:
    samples = SampleLog(tick_capacity=16, command_capacity=16)
    session = session_from(samples=samples)

    session.receive(command("cls", 5, capture_ns=50 * MS, session=SESSION - 1), 100 * MS)

    assert report(session, 100 * MS).models["cls"].commands_out_of_order == 1
    samples.save(tmp_path / "robot_r01.npz", {})
    commands, _, _ = load_samples(tmp_path / "robot_r01.npz")
    assert commands[["session", "verdict"]].tolist() == [(SESSION, 2)]


def test_each_command_s_durations_are_recorded_as_nanoseconds_over_a_million() -> None:
    session = session_from()

    session.receive(command("depth", 5, capture_ns=50 * MS, result_ns=97 * MS), 100 * MS + 1)
    session.receive(command("depth", 4, capture_ns=40 * MS, result_ns=118 * MS), 120 * MS)
    session.receive(command("depth", 9, capture_ns=90 * MS, result_ns=125 * MS), 133 * MS)

    depth = report(session, 133 * MS).models["depth"]
    # Every command has a freshness, every command after the first a gap, and the one applied
    # over an earlier applied command that command's peak age.
    assert (depth.freshness_ms.count, depth.freshness_ms.min_ms, depth.freshness_ms.max_ms) == (
        3,
        43.0,
        80.0,
    )
    assert depth.freshness_ms.sum_ms == (50 * MS + 1) / 1e6 + 80.0 + 43.0
    assert (depth.command_gap_ms.count, depth.command_gap_ms.min_ms) == (2, 13.0)
    assert depth.command_gap_ms.max_ms == (20 * MS - 1) / 1e6
    assert (depth.command_age_peak_ms.count, depth.command_age_peak_ms.max_ms) == (1, 83.0)
    assert (depth.deliver_ms.count, depth.deliver_ms.min_ms, depth.deliver_ms.max_ms) == (
        3,
        2.0,
        8.0,
    )


def test_the_models_are_judged_and_counted_apart() -> None:
    session = session_from()

    session.receive(command("cls", 5, capture_ns=50 * MS), 100 * MS)
    session.receive(command("depth", 5, capture_ns=50 * MS), 300 * MS)

    models = report(session, 300 * MS).models
    assert (models["cls"].commands_applied, models["cls"].commands_stale) == (1, 0)
    assert (models["depth"].commands_applied, models["depth"].commands_stale) == (0, 1)
    assert models["cls"].command_gap_ms.count == 0


@pytest.mark.parametrize(("shared_clock", "deliver_count"), [(True, 1), (False, 0)])
def test_the_deliver_time_is_recorded_only_on_a_shared_clock(
    shared_clock: bool, deliver_count: int
) -> None:
    session = session_from(shared_clock=shared_clock)

    session.receive(command("cls", 5, capture_ns=50 * MS, result_ns=90 * MS), 100 * MS)

    cls = report(session, 100 * MS).models["cls"]
    assert cls.deliver_ms.count == deliver_count
    assert cls.freshness_ms.count == 1


def test_a_report_carries_each_model_s_counters_and_watchdog_as_its_control_has_them() -> None:
    session = session_from()
    session.receive(command("depth", 5, capture_ns=50 * MS), 100 * MS)
    session.receive(command("depth", 4, capture_ns=40 * MS), 120 * MS)
    session.receive(command("cls", 5, capture_ns=50 * MS), 130 * MS)
    # Depth's watchdog limit is 300 ms, so at 900 ms its one applied command has expired.
    now_ns = 900 * MS

    models = report(session, now_ns).models

    for model, control in session.controls.items():
        entry = models[model]
        assert (
            entry.commands_received,
            entry.commands_applied,
            entry.commands_stale,
            entry.commands_out_of_order,
            entry.watchdog_trips,
            entry.watchdog_expired_ms,
            entry.watchdog_state,
        ) == (
            control.received,
            control.applied,
            control.stale,
            control.out_of_order,
            control.watchdog_trips(now_ns),
            control.watchdog_expired_ns(now_ns) // MS,
            control.watchdog_state(now_ns).value,
        )
    assert (models["depth"].watchdog_state, models["depth"].watchdog_trips) == ("expired", 1)
    assert models["depth"].watchdog_expired_ms == 100 + 550
    assert (models["cls"].watchdog_state, models["cls"].watchdog_expired_ms) == ("ok", 130)


def test_the_watchdog_counts_from_the_start_of_the_session() -> None:
    session = session_from(start_ns=2 * SECOND)

    depth = report(session, 2 * SECOND + 400 * MS).models["depth"]

    assert (depth.watchdog_state, depth.watchdog_trips, depth.watchdog_expired_ms) == (
        "none",
        0,
        400,
    )


@pytest.mark.parametrize(
    "message",
    [
        '{"type":"welcome","protocol_version":1,"session":5,"shared_clock":true}',
        "",
        b"",
        b"\x02",
        command("cls", 5, capture_ns=0)[:-1],
        command("cls", 5, capture_ns=0) + b"\x00",
        b"\x01" + command("cls", 5, capture_ns=0)[1:],
        b"\x02\x09" + command("cls", 5, capture_ns=0)[2:],
    ],
)
def test_a_message_that_is_not_a_command_raises_a_protocol_error(message: str | bytes) -> None:
    session = session_from()

    with pytest.raises(ProtocolError):
        session.receive(message, 100 * MS)


def test_a_command_of_a_model_the_session_did_not_subscribe_to_raises_a_protocol_error() -> None:
    settings = RobotSettings(models=("cls",), stale_ms={"cls": 500.0}, watchdog_ms={"cls": 1000.0})
    session = session_from(settings=settings)

    with pytest.raises(ProtocolError, match="did not subscribe"):
        session.receive(command("depth", 5, capture_ns=50 * MS), 100 * MS)

    assert report(session, 100 * MS).models["cls"].commands_received == 0
