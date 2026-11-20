"""Tests for the pure parts of the transport benchmark: flags, the robots' schedule, errors."""

import argparse
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.transport_bench import (
    SaturateSpec,
    Schedule,
    TransportBenchError,
    leaf_messages,
    main,
    parse_saturate_spec,
)
from common.clock import FakeClock

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

# A second of 30 fps from time 0: tick k is due at k * NS_PER_S // 30 plus the
# robot's phase, so ticks 1 to 5 of phase 0 are due at 33_333_333, 66_666_666,
# 100_000_000, 133_333_333 and 166_666_666 ns, and the last of its 30 ticks at
# 966_666_666 ns.
SECOND_AT_30_FPS = Schedule(start_ns=0, end_ns=NS_PER_S, fps=30)


@st.composite
def robot_schedules(draw: st.DrawFn) -> tuple[Schedule, int]:
    """Draw a schedule at least one frame period long, and a robot's phase within one period."""
    fps = draw(st.integers(min_value=1, max_value=240))
    period_ns = NS_PER_S // fps
    start_ns = draw(st.integers(min_value=0, max_value=10**15))
    duration_ns = draw(st.integers(min_value=period_ns, max_value=10 * NS_PER_S))
    phase_ns = draw(st.integers(min_value=0, max_value=period_ns - 1))
    return Schedule(start_ns, start_ns + duration_ns, fps), phase_ns


@st.composite
def sends(draw: st.DrawFn) -> tuple[Schedule, int, int, int]:
    """Draw a schedule, a robot's phase, a tick of that robot, and when its send ended."""
    schedule, phase_ns = draw(robot_schedules())
    tick = draw(st.integers(min_value=0, max_value=schedule.ticks(phase_ns) - 1))
    # From a uvloop timer that fires early to a stall of seconds.
    delay_ns = draw(st.integers(min_value=-2 * NS_PER_MS, max_value=3 * NS_PER_S))
    return schedule, phase_ns, tick, schedule.due_ns(phase_ns, tick) + delay_ns


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("cls:1", SaturateSpec(model="cls", threads=1, pacing_share=None)),
        ("depth:2", SaturateSpec(model="depth", threads=2, pacing_share=None)),
        ("depth:1:0.8", SaturateSpec(model="depth", threads=1, pacing_share=0.8)),
        ("cls:1:1.5", SaturateSpec(model="cls", threads=1, pacing_share=1.5)),
        # 0 threads leaves the count to ONNX Runtime, as bench.model_bench allows.
        ("cls:0", SaturateSpec(model="cls", threads=0, pacing_share=None)),
    ],
)
def test_spec_gives_the_model_threads_and_pacing_share(text: str, expected: SaturateSpec) -> None:
    assert parse_saturate_spec(text) == expected


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "expected MODEL:THREADS"),
        ("cls", "expected MODEL:THREADS"),
        ("cls:1:0.5:2", "expected MODEL:THREADS"),
        ("detect:1", "unknown model 'detect'"),
        ("CLS:1", "unknown model 'CLS'"),
        (":1", "unknown model ''"),
        ("cls:one", "'one' is not a whole number"),
        ("cls:", "'' is not a whole number"),
        ("cls:-1", "must be at least 0"),
        ("cls:1:half", "'half' is not a number"),
        ("cls:1:0", "above 0"),
        ("cls:1:-0.5", "above 0"),
        ("cls:1:nan", "above 0"),
        ("cls:1:inf", "above 0"),
    ],
)
def test_malformed_spec_is_refused_with_the_reason(text: str, message: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match=message):
        parse_saturate_spec(text)


def test_malformed_spec_is_a_usage_error_that_names_the_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["websocket", "--loop", "uvloop", "--seed", "0", "--out", str(tmp_path / "report.json")]

    with pytest.raises(SystemExit) as raised:
        main([*argv, "--saturate", "cls:1", "--saturate", "detect:1"], FakeClock())

    assert raised.value.code == 2
    assert "argument --saturate: unknown model 'detect'" in capsys.readouterr().err


@pytest.mark.parametrize(
    "command",
    [["dispatch", "--loop", "asyncio"], ["websocket", "--loop", "uvloop", "--seed", "0"]],
)
def test_out_that_is_a_directory_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], command: list[str]
) -> None:
    with pytest.raises(SystemExit) as raised:
        main([*command, "--out", str(tmp_path)], FakeClock())

    assert raised.value.code == 2
    assert f"argument --out: {tmp_path} is a directory" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("nice", "message"), [("20", "must be 0 to 19, got 20"), ("-1", "must be 0 to 19, got -1")]
)
def test_worker_nice_outside_the_range_of_the_loads_is_a_usage_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], nice: str, message: str
) -> None:
    argv = ["websocket", "--loop", "uvloop", "--seed", "0", "--out", str(tmp_path / "report.json")]

    with pytest.raises(SystemExit) as raised:
        main([*argv, "--worker-nice", nice], FakeClock())

    assert raised.value.code == 2
    assert f"argument --worker-nice: {message}" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("schedule", "phase_ns", "ticks"),
    [
        # Tick 30 is due exactly at the end, so it is not part of the schedule.
        (SECOND_AT_30_FPS, 0, 30),
        (SECOND_AT_30_FPS, 1, 30),
        (SECOND_AT_30_FPS, NS_PER_S // 30 - 1, 30),
        (Schedule(start_ns=0, end_ns=NS_PER_S + 1, fps=30), 0, 31),
        # One frame period: only tick 0 is due, 1 ns before the end.
        (Schedule(start_ns=7, end_ns=7 + NS_PER_S // 30, fps=30), NS_PER_S // 30 - 1, 1),
        (Schedule(start_ns=NS_PER_S, end_ns=7 * NS_PER_S // 2, fps=1), NS_PER_S // 2, 2),
    ],
)
def test_ticks_are_those_due_before_the_end(schedule: Schedule, phase_ns: int, ticks: int) -> None:
    assert schedule.ticks(phase_ns) == ticks


@given(robot_schedules())
def test_ticks_end_with_the_last_tick_due_before_the_end(
    schedule_and_phase: tuple[Schedule, int],
) -> None:
    schedule, phase_ns = schedule_and_phase

    ticks = schedule.ticks(phase_ns)

    assert (
        schedule.due_ns(phase_ns, ticks - 1) < schedule.end_ns <= schedule.due_ns(phase_ns, ticks)
    )


@pytest.mark.parametrize(
    ("phase_ns", "tick", "sent_ns", "next_tick"),
    [
        pytest.param(0, 0, 1 * NS_PER_MS, 1, id="a quick send skips nothing"),
        pytest.param(0, 0, 70 * NS_PER_MS, 3, id="a slow send skips the ticks due while it sends"),
        pytest.param(0, 0, 33_333_333, 1, id="a tick due as the send ends is not skipped"),
        # Tick 1 is due at 33.3 ms and goes out 100 ms late; ticks 2 to 4 came
        # due before it had gone, so they are skipped and do not follow in a burst.
        pytest.param(0, 1, 133_400_000, 5, id="a late wake sends no burst"),
        pytest.param(10 * NS_PER_MS, 1, 143_400_000, 5, id="a late wake of a later phase"),
        # A uvloop timer can fire before its tick is due: a negative fire lag.
        pytest.param(0, 1, 32 * NS_PER_MS, 2, id="an early wake skips nothing"),
        pytest.param(0, 29, 967 * NS_PER_MS, 30, id="the last tick ends the schedule"),
        pytest.param(
            0, 28, 1_200 * NS_PER_MS, 30, id="a send past the end skips only ticks before it"
        ),
    ],
)
def test_next_tick_is_the_first_due_once_the_send_has_ended(
    phase_ns: int, tick: int, sent_ns: int, next_tick: int
) -> None:
    assert SECOND_AT_30_FPS.next_tick(phase_ns, tick, sent_ns) == next_tick


@given(sends())
def test_every_skipped_tick_came_due_before_the_send_ended(
    send: tuple[Schedule, int, int, int],
) -> None:
    schedule, phase_ns, tick, sent_ns = send

    following = schedule.next_tick(phase_ns, tick, sent_ns)

    ticks = schedule.ticks(phase_ns)
    assert tick < following <= ticks
    assert all(
        schedule.due_ns(phase_ns, skipped) < sent_ns for skipped in range(tick + 1, following)
    )
    assert following == ticks or schedule.due_ns(phase_ns, following) >= sent_ns


def test_a_failure_is_reported_by_its_own_message_however_deep_its_groups() -> None:
    failure = ExceptionGroup(
        "unhandled errors in a TaskGroup",
        [
            ExceptionGroup(
                "unhandled errors in a TaskGroup",
                [TransportBenchError("a robot's connection closed during the schedule")],
            ),
            TransportBenchError("the receiver exited with status 1"),
        ],
    )

    assert leaf_messages(failure) == [
        "a robot's connection closed during the schedule",
        "the receiver exited with status 1",
    ]
