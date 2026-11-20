"""Tests for the robot's sample log and its file."""

from pathlib import Path

import numpy as np
import pytest

from common.protocol import STAGE_NAMES, CommandHeader, StageDurations
from robot.control import Verdict
from robot.samples import (
    COMMAND_DTYPE,
    TICK_DTYPE,
    VERDICTS,
    SampleLog,
    TickOutcome,
    load_samples,
)

STAGES = StageDurations(11, 12, 13, 14, 15, 16, 17, 18)


def header(model: str, frame_number: int, capture_ns: int = 0, result_ns: int = 0) -> CommandHeader:
    return CommandHeader(
        model=model,
        session=99,
        frame_number=frame_number,
        capture_ns=capture_ns,
        batch_size=4,
        result_ns=result_ns,
        stages=STAGES,
    )


def test_the_row_layouts_are_fixed_and_little_endian() -> None:
    assert COMMAND_DTYPE.names == (
        "session",
        "model",
        "verdict",
        "batch_size",
        "frame_number",
        "capture_ns",
        "send_ns",
        "receive_ns",
        "result_ns",
        "stage_us",
    )
    assert [COMMAND_DTYPE[name].str for name in COMMAND_DTYPE.names[:9]] == [
        "<u4",
        "|u1",
        "|u1",
        "<u2",
        "<u8",
        "<i8",
        "<i8",
        "<i8",
        "<i8",
    ]
    assert COMMAND_DTYPE["stage_us"].subdtype == (np.dtype("<u4"), (len(STAGE_NAMES),))
    assert TICK_DTYPE.descr == [
        ("frame_number", "<u8"),
        ("capture_ns", "<i8"),
        ("send_ns", "<i8"),
        ("outcome", "|u1"),
    ]


def test_the_tick_outcomes_and_verdicts_have_their_codes() -> None:
    assert [(outcome.name, outcome.value) for outcome in TickOutcome] == [
        ("SENT", 0),
        ("SKIPPED_UNSENT", 1),
        ("MISSED_LATE", 2),
    ]
    assert VERDICTS == (Verdict.APPLIED, Verdict.STALE, Verdict.OUT_OF_ORDER)


def test_ticks_are_saved_as_recorded(tmp_path: Path) -> None:
    log = SampleLog(tick_capacity=8, command_capacity=8)
    log.tick(3, 300, 305, TickOutcome.SENT)
    log.tick(4, 400, 0, TickOutcome.SKIPPED_UNSENT)
    log.tick(5, 500, 0, TickOutcome.MISSED_LATE)

    log.save(tmp_path / "robot_a.npz", {})

    _, ticks, _ = load_samples(tmp_path / "robot_a.npz")
    assert ticks.tolist() == [(3, 300, 305, 0), (4, 400, 0, 1), (5, 500, 0, 2)]


@pytest.mark.parametrize(
    ("model", "model_index"),
    [("cls", 0), ("depth", 1)],
)
@pytest.mark.parametrize(
    ("verdict", "verdict_code"),
    [(Verdict.APPLIED, 0), (Verdict.STALE, 1), (Verdict.OUT_OF_ORDER, 2)],
)
def test_a_command_is_saved_with_its_header_its_receive_time_and_its_verdict(
    tmp_path: Path, model: str, model_index: int, verdict: Verdict, verdict_code: int
) -> None:
    log = SampleLog(tick_capacity=8, command_capacity=8)

    log.command(7, header(model, 3, capture_ns=300, result_ns=340), 350, verdict)
    log.save(tmp_path / "robot_a.npz", {})

    commands, _, _ = load_samples(tmp_path / "robot_a.npz")
    scalars = [
        "session",
        "model",
        "verdict",
        "batch_size",
        "frame_number",
        "capture_ns",
        "send_ns",
        "receive_ns",
        "result_ns",
    ]
    assert commands[scalars].tolist() == [(7, model_index, verdict_code, 4, 3, 300, 0, 350, 340)]
    assert commands["stage_us"].tolist() == [[11, 12, 13, 14, 15, 16, 17, 18]]


def test_saving_fills_each_command_s_send_time_from_the_tick_of_its_frame(tmp_path: Path) -> None:
    log = SampleLog(tick_capacity=8, command_capacity=8)
    log.tick(10, 1000, 1001, TickOutcome.SENT)
    log.tick(11, 1100, 0, TickOutcome.SKIPPED_UNSENT)
    log.tick(13, 1300, 1304, TickOutcome.SENT)
    log.command(1, header("depth", 13), 1400, Verdict.APPLIED)
    log.command(1, header("cls", 10), 1401, Verdict.APPLIED)
    log.command(1, header("cls", 13), 1402, Verdict.APPLIED)

    log.save(tmp_path / "robot_a.npz", {})

    commands, _, _ = load_samples(tmp_path / "robot_a.npz")
    assert commands["frame_number"].tolist() == [13, 10, 13]
    assert commands["send_ns"].tolist() == [1304, 1001, 1304]


@pytest.mark.parametrize("frame_number", [9, 12, 14])
def test_a_command_of_a_frame_with_no_tick_keeps_send_time_zero(
    tmp_path: Path, frame_number: int
) -> None:
    log = SampleLog(tick_capacity=8, command_capacity=8)
    log.tick(10, 1000, 1001, TickOutcome.SENT)
    log.tick(13, 1300, 1304, TickOutcome.SENT)
    log.command(1, header("cls", frame_number), 1400, Verdict.APPLIED)

    log.save(tmp_path / "robot_a.npz", {})

    commands, _, _ = load_samples(tmp_path / "robot_a.npz")
    assert commands["send_ns"].tolist() == [0]


def test_a_command_is_saved_when_no_tick_was_recorded(tmp_path: Path) -> None:
    log = SampleLog(tick_capacity=8, command_capacity=8)
    log.command(1, header("cls", 10), 1400, Verdict.STALE)

    log.save(tmp_path / "robot_a.npz", {})

    commands, ticks, _ = load_samples(tmp_path / "robot_a.npz")
    assert commands["send_ns"].tolist() == [0]
    assert len(ticks) == 0


def test_a_full_tick_array_drops_the_row_and_counts_it(tmp_path: Path) -> None:
    log = SampleLog(tick_capacity=2, command_capacity=2)
    log.tick(1, 100, 100, TickOutcome.SENT)
    log.tick(2, 200, 200, TickOutcome.SENT)

    log.tick(3, 300, 300, TickOutcome.SENT)
    log.tick(4, 400, 400, TickOutcome.SENT)
    log.save(tmp_path / "robot_a.npz", {})

    _, ticks, _ = load_samples(tmp_path / "robot_a.npz")
    assert ticks["frame_number"].tolist() == [1, 2]
    assert (log.dropped_ticks, log.dropped_commands) == (2, 0)


def test_a_full_command_array_drops_the_row_and_counts_it(tmp_path: Path) -> None:
    log = SampleLog(tick_capacity=2, command_capacity=2)
    log.command(1, header("cls", 1), 150, Verdict.APPLIED)
    log.command(1, header("cls", 2), 250, Verdict.APPLIED)

    log.command(1, header("cls", 3), 350, Verdict.APPLIED)
    log.save(tmp_path / "robot_a.npz", {})

    commands, _, _ = load_samples(tmp_path / "robot_a.npz")
    assert commands["frame_number"].tolist() == [1, 2]
    assert (log.dropped_ticks, log.dropped_commands) == (0, 1)


def test_an_empty_log_saves_two_empty_arrays_of_the_row_layouts(tmp_path: Path) -> None:
    SampleLog(tick_capacity=4, command_capacity=4).save(tmp_path / "robot_a.npz", {})

    commands, ticks, _ = load_samples(tmp_path / "robot_a.npz")
    assert (len(commands), commands.dtype) == (0, COMMAND_DTYPE)
    assert (len(ticks), ticks.dtype) == (0, TICK_DTYPE)


def test_the_meta_comes_back_as_it_was_saved(tmp_path: Path) -> None:
    meta: dict[str, object] = {
        "robot_id": "a",
        "seed": 3,
        "settings": {"uplink": {"delay_ms": 20.0, "jitter_ms": 0.0}},
        "model_keys": ["cls", "depth"],
        "dropped_ticks": 0,
    }

    SampleLog(tick_capacity=1, command_capacity=1).save(tmp_path / "robot_a.npz", meta)

    assert load_samples(tmp_path / "robot_a.npz")[2] == meta


def test_a_file_with_another_row_layout_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "robot_a.npz"
    np.savez(
        path,
        commands=np.zeros(1, dtype=[("frame_number", "<u8")]),
        ticks=np.zeros(1, dtype=TICK_DTYPE),
        meta=np.array("{}"),
    )

    with pytest.raises(ValueError, match="row layout"):
        load_samples(path)
