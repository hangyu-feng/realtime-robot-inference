"""Tests for the synthetic frame set: resolution, determinism, motion and size statistics."""

import cv2
import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st

from common.protocol import FRAME_HEIGHT_PX, FRAME_WIDTH_PX
from robot.frames import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SEED,
    FrameEncodeError,
    FrameSet,
    FrameSetStats,
    build_frame_set,
)

FRAME_SHAPE = (FRAME_HEIGHT_PX, FRAME_WIDTH_PX, 3)

# imdecode accepts every image format OpenCV knows, so decoding alone does not
# show that a frame is a JPEG. These markers open and close a JPEG stream.
JPEG_START_OF_IMAGE = b"\xff\xd8"
JPEG_END_OF_IMAGE = b"\xff\xd9"

# The design target for the mean frame size at the default quality.
TARGET_MEAN_BYTES_MIN = 25_000
TARGET_MEAN_BYTES_MAX = 60_000

# Between two frames of a still scene, sensor noise and JPEG artefacts change
# each pixel by fewer levels than this, so a larger change over this many pixels
# means that objects moved.
MOVED_PIXEL_LEVELS = 64
MOVED_PIXELS_MIN = 1_000


@pytest.fixture(scope="module")
def default_frame_set() -> FrameSet:
    return build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)


def decode(jpeg: bytes) -> npt.NDArray[np.uint8]:
    # IMREAD_COLOR would expand a greyscale JPEG to three channels. With
    # IMREAD_UNCHANGED the channel count is the JPEG's own, so the decoded shape
    # shows whether a frame is in colour.
    image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    assert image is not None
    return np.asarray(image, dtype=np.uint8)


def test_every_frame_decodes_to_a_colour_image_at_the_frame_resolution(
    default_frame_set: FrameSet,
) -> None:
    shapes = [decode(jpeg).shape for jpeg in default_frame_set.jpegs]

    assert shapes == [FRAME_SHAPE] * DEFAULT_FRAME_COUNT


def test_every_frame_is_a_complete_jpeg_stream(default_frame_set: FrameSet) -> None:
    assert all(jpeg.startswith(JPEG_START_OF_IMAGE) for jpeg in default_frame_set.jpegs)
    assert all(jpeg.endswith(JPEG_END_OF_IMAGE) for jpeg in default_frame_set.jpegs)


# Each example renders a whole scene, so the number of examples is kept small.
@settings(max_examples=20)
@given(
    count=st.integers(min_value=1, max_value=3),
    seed=st.integers(min_value=0),
    jpeg_quality=st.integers(min_value=1, max_value=100),
)
@example(count=1, seed=0, jpeg_quality=1)
@example(count=1, seed=0, jpeg_quality=100)
def test_valid_arguments_give_the_requested_frames(
    count: int, seed: int, jpeg_quality: int
) -> None:
    frame_set = build_frame_set(count, seed, jpeg_quality)

    assert frame_set.jpeg_quality == jpeg_quality
    assert [decode(jpeg).shape for jpeg in frame_set.jpegs] == [FRAME_SHAPE] * count


def test_equal_arguments_give_equal_frames() -> None:
    first = build_frame_set(3, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    second = build_frame_set(3, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)

    assert first == second


def test_another_seed_gives_other_frames() -> None:
    first = build_frame_set(3, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    second = build_frame_set(3, DEFAULT_SEED + 1, DEFAULT_JPEG_QUALITY)

    assert set(first.jpegs).isdisjoint(second.jpegs)


def test_frames_differ_from_each_other(default_frame_set: FrameSet) -> None:
    assert len(set(default_frame_set.jpegs)) == DEFAULT_FRAME_COUNT


def test_objects_move_between_consecutive_frames(default_frame_set: FrameSet) -> None:
    images = [decode(jpeg).astype(np.int16) for jpeg in default_frame_set.jpegs]
    # Robots cycle through the set, so the last frame is followed by the first.
    following = images[1:] + images[:1]

    moved_pixels = [
        int((np.abs(after - before).max(axis=2) > MOVED_PIXEL_LEVELS).sum())
        for before, after in zip(images, following, strict=True)
    ]

    assert min(moved_pixels) >= MOVED_PIXELS_MIN


def test_a_higher_quality_gives_larger_frames() -> None:
    low = build_frame_set(2, DEFAULT_SEED, 50).stats()
    high = build_frame_set(2, DEFAULT_SEED, 90).stats()

    assert low.max_bytes < high.min_bytes


def test_stats_summarize_the_frame_sizes() -> None:
    frame_set = FrameSet(jpegs=(bytes(10), bytes(40), bytes(13)), jpeg_quality=70)

    assert frame_set.stats() == FrameSetStats(
        count=3, jpeg_quality=70, min_bytes=10, mean_bytes=21.0, max_bytes=40
    )


def test_stats_of_the_default_set_match_its_frames(default_frame_set: FrameSet) -> None:
    sizes = sorted(len(jpeg) for jpeg in default_frame_set.jpegs)

    stats = default_frame_set.stats()

    assert (stats.count, stats.jpeg_quality) == (DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY)
    assert (stats.min_bytes, stats.max_bytes) == (sizes[0], sizes[-1])
    assert stats.min_bytes <= stats.mean_bytes <= stats.max_bytes


@pytest.mark.parametrize("seed", [DEFAULT_SEED, DEFAULT_SEED + 1, DEFAULT_SEED + 2])
def test_mean_frame_size_is_in_the_target_range_at_the_default_quality(seed: int) -> None:
    stats = build_frame_set(DEFAULT_FRAME_COUNT, seed, DEFAULT_JPEG_QUALITY).stats()

    assert TARGET_MEAN_BYTES_MIN <= stats.mean_bytes <= TARGET_MEAN_BYTES_MAX


@pytest.mark.parametrize(
    ("count", "seed", "jpeg_quality", "argument"),
    [
        (0, DEFAULT_SEED, DEFAULT_JPEG_QUALITY, "count"),
        (-1, DEFAULT_SEED, DEFAULT_JPEG_QUALITY, "count"),
        (1, -1, DEFAULT_JPEG_QUALITY, "seed"),
        (1, DEFAULT_SEED, 0, "jpeg_quality"),
        (1, DEFAULT_SEED, 101, "jpeg_quality"),
    ],
)
def test_an_invalid_argument_raises_value_error(
    count: int, seed: int, jpeg_quality: int, argument: str
) -> None:
    with pytest.raises(ValueError, match=argument):
        build_frame_set(count, seed, jpeg_quality)


def test_a_failed_encode_raises_instead_of_yielding_an_empty_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cv2, "imencode", lambda *_: (False, np.empty(0, dtype=np.uint8)))

    with pytest.raises(FrameEncodeError, match="frame 0"):
        build_frame_set(1, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
