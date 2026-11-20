"""Tests for the fake and the null runtime."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.commands import MODEL_KEYS, ClsCommand, DepthCommand
from common.worker_protocol import MAX_BATCH_FRAMES
from workers.models.fake import FakeRunner
from workers.runtime import BatchOutcome, FrameOutcome

FRAME_NS = 4_000_000
# How a frame that the fake serves starts: the JPEG start-of-image marker and the first byte
# of the marker after it.
JPEG_START = b"\xff\xd8\xff"
JPEG = JPEG_START + b"\xe0 rest of the frame"
UNDECODED = FrameOutcome(None, 0, 0, 0)
DEPTH_COMMAND = DepthCommand(left=0.0, center=0.0, right=0.0, steering=0.0, speed=1.0)


def cls_outcome(frame_bytes: int) -> FrameOutcome:
    return FrameOutcome(ClsCommand(class_id=frame_bytes, confidence=1.0), 0, 0, 0)


def ignore_wait(duration_s: float) -> None:
    """Stand in for time.sleep where a test does not look at the wait."""


def test_cls_outcomes_follow_the_frames_in_order_with_their_lengths_as_class_ids() -> None:
    waits_s: list[float] = []
    runner = FakeRunner("cls", FRAME_NS, waits_s.append)

    outcome = runner.run_batch([JPEG_START + bytes(10), JPEG_START, JPEG_START + bytes(300)])

    assert outcome == BatchOutcome(
        (cls_outcome(13), cls_outcome(3), cls_outcome(303)), 3 * FRAME_NS
    )
    assert waits_s == [3 * FRAME_NS / 1e9]


@pytest.mark.parametrize(
    ("frame_bytes", "class_id"), [(65_535, 65_535), (65_536, 0), (70_000, 70_000 - 65_536)]
)
def test_cls_class_id_is_the_frame_length_within_its_u16(frame_bytes: int, class_id: int) -> None:
    runner = FakeRunner("cls", FRAME_NS, ignore_wait)

    outcome = runner.run_batch([JPEG_START + bytes(frame_bytes - len(JPEG_START))])

    assert outcome.frames == (FrameOutcome(ClsCommand(class_id, 1.0), 0, 0, 0),)


def test_depth_outcomes_are_the_full_speed_command_for_every_frame() -> None:
    runner = FakeRunner("depth", FRAME_NS, ignore_wait)

    outcome = runner.run_batch([JPEG, JPEG + b"longer"])

    assert outcome == BatchOutcome((FrameOutcome(DEPTH_COMMAND, 0, 0, 0),) * 2, 2 * FRAME_NS)


@pytest.mark.parametrize(
    "unmarked",
    [
        pytest.param(b"", id="empty"),
        pytest.param(b"\xff\xd8", id="start-of-image-only"),
        pytest.param(b"\xff\xd8\x00\xe0", id="start-of-image-then-no-marker"),
        pytest.param(b"\x89PNG\r\n\x1a\n", id="png"),
    ],
)
def test_frame_that_does_not_start_like_a_jpeg_fails_alone(unmarked: bytes) -> None:
    waits_s: list[float] = []
    runner = FakeRunner("cls", FRAME_NS, waits_s.append)

    outcome = runner.run_batch([JPEG, unmarked, JPEG])

    assert outcome == BatchOutcome(
        (cls_outcome(len(JPEG)), UNDECODED, cls_outcome(len(JPEG))), 2 * FRAME_NS
    )
    assert waits_s == [2 * FRAME_NS / 1e9]


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_batch_without_a_jpeg_does_not_wait_and_reports_no_inference(model: str) -> None:
    waits_s: list[float] = []
    runner = FakeRunner(model, FRAME_NS, waits_s.append)

    outcome = runner.run_batch([b"", b"\x00\x01"])

    assert outcome == BatchOutcome((UNDECODED, UNDECODED), 0)
    assert waits_s == []


@pytest.mark.parametrize("model", MODEL_KEYS)
def test_null_runtime_commands_every_frame_without_a_wait(model: str) -> None:
    waits_s: list[float] = []
    runner = FakeRunner(model, 0, waits_s.append)

    outcome = runner.run_batch([JPEG] * 3)

    assert [frame.command is not None for frame in outcome.frames] == [True] * 3
    assert outcome.inference_ns == 0
    assert waits_s == []


@pytest.mark.parametrize(
    ("model", "frame_ns", "message"),
    [
        pytest.param("pose", 0, "unknown model 'pose'", id="unknown-model"),
        pytest.param("", 0, "unknown model ''", id="empty-model"),
        pytest.param("cls", -1, "frame_ns must not be negative", id="negative-frame-time"),
    ],
)
def test_runner_refuses_an_unknown_model_or_a_negative_frame_time(
    model: str, frame_ns: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        FakeRunner(model, frame_ns, ignore_wait)


UNMARKED = st.binary(max_size=64).filter(lambda frame: not frame.startswith(JPEG_START))
MARKED = st.binary(max_size=64).map(lambda body: JPEG_START + body)
FRAMES = st.lists(st.one_of(UNMARKED, MARKED), max_size=MAX_BATCH_FRAMES)
FRAME_TIMES_NS = st.integers(min_value=1, max_value=10**9)


@given(
    model=st.sampled_from(MODEL_KEYS),
    frame_ns=FRAME_TIMES_NS,
    jpegs=FRAMES.filter(lambda jpegs: any(jpeg.startswith(JPEG_START) for jpeg in jpegs)),
)
def test_runner_waits_once_for_the_frames_that_start_like_a_jpeg(
    model: str, frame_ns: int, jpegs: list[bytes]
) -> None:
    waits_s: list[float] = []

    outcome = FakeRunner(model, frame_ns, waits_s.append).run_batch(jpegs)

    passing = [jpeg.startswith(JPEG_START) for jpeg in jpegs]
    assert [frame.command is not None for frame in outcome.frames] == passing
    assert outcome.inference_ns == frame_ns * sum(passing)
    assert waits_s == [frame_ns * sum(passing) / 1e9]
    assert {
        (frame.decode_ns, frame.preprocess_ns, frame.postprocess_ns) for frame in outcome.frames
    } <= {(0, 0, 0)}


@given(
    model=st.sampled_from(MODEL_KEYS),
    # No time per frame, or no frame to spend it on.
    batch=st.one_of(
        st.tuples(st.just(0), FRAMES),
        st.tuples(FRAME_TIMES_NS, st.lists(UNMARKED, max_size=MAX_BATCH_FRAMES)),
    ),
)
def test_runner_with_nothing_to_wait_for_never_calls_the_wait(
    model: str, batch: tuple[int, list[bytes]]
) -> None:
    frame_ns, jpegs = batch
    waits_s: list[float] = []

    outcome = FakeRunner(model, frame_ns, waits_s.append).run_batch(jpegs)

    passing = [jpeg.startswith(JPEG_START) for jpeg in jpegs]
    assert [frame.command is not None for frame in outcome.frames] == passing
    assert outcome.inference_ns == 0
    assert waits_s == []
