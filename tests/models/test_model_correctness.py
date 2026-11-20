"""Tests that run the real models: batching, fault isolation, session threads and command ranges."""

import os
from collections.abc import Iterator, Sequence
from dataclasses import astuple
from pathlib import Path

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from robot.frames import DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from workers.models import MODELS
from workers.runtime import (
    FrameOutcome,
    ModelLoadError,
    ModelRunner,
    SessionSettings,
    configure_process_threads,
)

FRAME_COUNT = 4
STEP_NS = 1_000
SETTINGS = SessionSettings(intra_op_threads=1, allow_spinning=False)

# Command fields are of order 1, so one tolerance bounds both the absolute and
# the relative difference between two runs of the same frame.
COMMAND_TOLERANCE = 1e-4
# A batch whose rows are swapped or repeated can agree with the single runs only
# if two frames have commands within twice the tolerance of each other. The
# frames must lie this many tolerances apart, which leaves a margin.
FRAME_SEPARATION_TOLERANCES = 10

IMAGENET_CLASS_COUNT = 1000
# The lowest and the highest value of each field, in the order of the dataclass.
CLS_FIELD_RANGE = ((0, 1 / IMAGENET_CLASS_COUNT), (IMAGENET_CLASS_COUNT - 1, 1))
DEPTH_FIELD_RANGE = ((0, 0, 0, -1, 0), (1, 1, 1, 1, 1))

# The kernel lists one entry per thread of this process here.
TASK_DIRECTORY = "/proc/self/task"


class SteppingClock:
    """A clock that advances one step per reading.

    A stage then lasts exactly one step, unless it brackets a reading that
    belongs to another stage.
    """

    def __init__(self) -> None:
        self._now_ns = 0

    def now_ns(self) -> int:
        self._now_ns += STEP_NS
        return self._now_ns


def empty(jpeg: bytes) -> bytes:
    return b""


def truncated(jpeg: bytes) -> bytes:
    return jpeg[: len(jpeg) // 2]


def field_matrix(frames: Sequence[FrameOutcome]) -> NDArray[np.float64]:
    """Stack the command fields of frames that all decoded, one row per frame."""
    rows: list[tuple[float, ...]] = []
    for frame in frames:
        assert frame.command is not None
        rows.append(astuple(frame.command))
    return np.array(rows, dtype=np.float64)


def smallest_separation(fields: NDArray[np.float64]) -> float:
    """Return the smallest distance between two rows: their largest difference in one field."""
    distances = np.abs(fields[:, np.newaxis] - fields[np.newaxis, :]).max(axis=2)
    return float(distances[np.triu_indices(len(fields), k=1)].min())


@pytest.fixture(scope="module")
def jpegs() -> tuple[bytes, ...]:
    return build_frame_set(FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).jpegs


@pytest.fixture(scope="module")
def cls_runner(model_dir: Path) -> ModelRunner:
    spec = MODELS["cls"]
    return ModelRunner(spec, model_dir / spec.batched_file, SETTINGS, SteppingClock())


@pytest.fixture(scope="module")
def depth_runner(model_dir: Path) -> ModelRunner:
    spec = MODELS["depth"]
    return ModelRunner(spec, model_dir / spec.batched_file, SETTINGS, SteppingClock())


@pytest.fixture(scope="module")
def published_depth_runner(model_dir: Path) -> ModelRunner:
    spec = MODELS["depth"]
    return ModelRunner(spec, model_dir / spec.published_file, SETTINGS, SteppingClock())


@pytest.fixture
def restore_opencv_threads() -> Iterator[None]:
    yield
    # A negative count returns OpenCV to its default.
    cv2.setNumThreads(-1)


def test_depth_batch_gives_each_frame_the_command_of_its_single_run(
    depth_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    singles = field_matrix([depth_runner.run_batch([jpeg]).frames[0] for jpeg in jpegs])

    batch = field_matrix(depth_runner.run_batch(jpegs).frames)

    assert smallest_separation(singles) > FRAME_SEPARATION_TOLERANCES * COMMAND_TOLERANCE
    assert batch == pytest.approx(singles, abs=COMMAND_TOLERANCE)


def test_published_and_batched_depth_graphs_agree_on_single_frames(
    depth_runner: ModelRunner, published_depth_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    published = [published_depth_runner.run_batch([jpeg]).frames[0] for jpeg in jpegs]

    batched = [depth_runner.run_batch([jpeg]).frames[0] for jpeg in jpegs]

    assert field_matrix(batched) == pytest.approx(field_matrix(published), abs=COMMAND_TOLERANCE)


def test_cls_batch_gives_each_frame_the_class_and_confidence_of_its_single_run(
    cls_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    singles = field_matrix([cls_runner.run_batch([jpeg]).frames[0] for jpeg in jpegs])

    batch = field_matrix(cls_runner.run_batch(jpegs).frames)

    assert smallest_separation(singles) > FRAME_SEPARATION_TOLERANCES * COMMAND_TOLERANCE
    # Class ids are whole numbers, so the tolerance admits only equal ones.
    assert batch == pytest.approx(singles, abs=COMMAND_TOLERANCE)


@pytest.mark.parametrize("corrupt_index", [0, 1, 2])
def test_corrupt_frame_fails_alone_in_its_batch(
    cls_runner: ModelRunner, jpegs: tuple[bytes, ...], corrupt_index: int
) -> None:
    good = jpegs[:2]
    singles = [cls_runner.run_batch([jpeg]).frames[0] for jpeg in good]
    batch = [*good[:corrupt_index], truncated(jpegs[2]), *good[corrupt_index:]]

    frames = list(cls_runner.run_batch(batch).frames)

    assert frames.pop(corrupt_index).command is None
    assert field_matrix(frames) == pytest.approx(field_matrix(singles), abs=COMMAND_TOLERANCE)


def test_batch_without_a_decodable_frame_runs_no_inference(
    cls_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    outcome = cls_runner.run_batch([empty(jpegs[0]), truncated(jpegs[1])])

    assert [frame.command for frame in outcome.frames] == [None, None]
    assert outcome.inference_ns == 0


def test_each_stage_is_timed_alone_on_the_injected_clock(
    cls_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    outcome = cls_runner.run_batch([jpegs[0], truncated(jpegs[1]), jpegs[2]])

    stages = [
        (frame.decode_ns, frame.preprocess_ns, frame.postprocess_ns) for frame in outcome.frames
    ]
    assert stages == [(STEP_NS, STEP_NS, STEP_NS), (STEP_NS, 0, 0), (STEP_NS, STEP_NS, STEP_NS)]
    assert outcome.inference_ns == STEP_NS


def test_max_batch_size_is_read_from_the_graph(
    cls_runner: ModelRunner, depth_runner: ModelRunner, published_depth_runner: ModelRunner
) -> None:
    assert cls_runner.max_batch_size is None
    assert depth_runner.max_batch_size is None
    assert published_depth_runner.max_batch_size == 1


def test_fixed_batch_graph_refuses_more_frames_than_it_accepts(
    published_depth_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    with pytest.raises(ValueError, match="2 frames decoded, but the depth graph accepts 1"):
        published_depth_runner.run_batch(jpegs[:2])


def test_fixed_batch_graph_counts_only_the_frames_that_decode(
    published_depth_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    single = published_depth_runner.run_batch([jpegs[0]]).frames[0]

    undecodable, decoded = published_depth_runner.run_batch([truncated(jpegs[1]), jpegs[0]]).frames

    assert undecodable.command is None
    assert decoded.command == single.command


def test_depth_command_fields_are_within_their_ranges(
    depth_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    lowest, highest = DEPTH_FIELD_RANGE

    fields = field_matrix(depth_runner.run_batch(jpegs).frames)

    assert ((fields >= lowest) & (fields <= highest)).all()


def test_cls_command_fields_are_within_their_ranges(
    cls_runner: ModelRunner, jpegs: tuple[bytes, ...]
) -> None:
    lowest, highest = CLS_FIELD_RANGE

    fields = field_matrix(cls_runner.run_batch(jpegs).frames)

    assert ((fields >= lowest) & (fields <= highest)).all()


@pytest.mark.parametrize("intra_op_threads", [1, 2, 3])
def test_session_starts_one_thread_fewer_than_its_intra_op_setting(
    model_dir: Path, intra_op_threads: int
) -> None:
    spec = MODELS["cls"]
    settings = SessionSettings(intra_op_threads=intra_op_threads, allow_spinning=False)
    threads_before = len(os.listdir(TASK_DIRECTORY))

    runner = ModelRunner(spec, model_dir / spec.batched_file, settings, SteppingClock())
    threads_started = len(os.listdir(TASK_DIRECTORY)) - threads_before
    # A session's threads end with it. A failed assertion would keep this one
    # alive in its traceback, and its threads could then end inside the count
    # of the next case.
    del runner

    # The thread that calls the session does a share of its work.
    assert threads_started == intra_op_threads - 1


def test_missing_model_file_fails_to_load(tmp_path: Path) -> None:
    with pytest.raises(ModelLoadError, match=r"cannot load model 'cls' from .*absent\.onnx"):
        ModelRunner(MODELS["cls"], tmp_path / "absent.onnx", SETTINGS, SteppingClock())


def test_file_that_is_not_a_model_fails_to_load(tmp_path: Path) -> None:
    model_path = tmp_path / "not_a_model.onnx"
    model_path.write_bytes(b"not a model")

    with pytest.raises(ModelLoadError, match=r"cannot load model 'depth' from .*not_a_model\.onnx"):
        ModelRunner(MODELS["depth"], model_path, SETTINGS, SteppingClock())


@pytest.mark.usefixtures("restore_opencv_threads")
def test_configure_process_threads_leaves_opencv_one_thread() -> None:
    configure_process_threads()

    assert cv2.getNumThreads() == 1
