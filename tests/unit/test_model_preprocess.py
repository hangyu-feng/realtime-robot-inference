"""Tests for a frame's way to a model input: JPEG decoding, then each model's preprocessing."""

import struct
from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray

from robot.frames import DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from workers.models import MODELS
from workers.runtime import FrameDecodeError, decode_jpeg

FRAME_SIZES = [pytest.param(480, 640, id="full"), pytest.param(240, 320, id="half")]
FULL_SIZE_SHAPE = (480, 640, 3)
HALF_SIZE_SHAPE = (240, 320, 3)

# JPEG markers, from ITU-T T.81, table B.1. All but the restart marker open a
# segment: the marker, a 2-byte length and a payload.
BASELINE_FRAME_HEADER = b"\xff\xc0"
HUFFMAN_TABLE = b"\xff\xc4"
ARITHMETIC_CONDITIONING = b"\xff\xcc"
RESTART = b"\xff\xd0"
NUMBER_OF_LINES = b"\xff\xdc"
RESTART_INTERVAL = b"\xff\xdd"
APPLICATION_1 = b"\xff\xe1"
COMMENT = b"\xff\xfe"

# Values of the EXIF orientation tag: the image as stored, and a quarter turn to apply.
EXIF_UPRIGHT = 1
EXIF_QUARTER_TURN = 6

IMAGENET_MEAN_RGB = np.array([0.485, 0.456, 0.406])
IMAGENET_STD_RGB = np.array([0.229, 0.224, 0.225])

# Per plane, the values that an all-black and an all-white frame map to.
PLANE_RANGES = {
    "cls": (-IMAGENET_MEAN_RGB / IMAGENET_STD_RGB, (1.0 - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB),
    "depth": (np.zeros(3), np.ones(3)),
}

CONSTANT_FRAME_BGR = (10, 128, 250)
CONSTANT_FRAME_RGB = np.array(CONSTANT_FRAME_BGR[::-1]) / 255.0

FLOAT32_TOLERANCE = 1e-6
# One and a half 8-bit levels on the [0, 1] scale. Resizing rounds to whole levels.
LEVEL_TOLERANCE = 1.5 / 255.0
# In normalized units, about one and a half 8-bit levels: both pipelines round
# to whole levels, and the resized long side has a fractional length, so the
# crop of the reference sits a fraction of a pixel off center.
RESIZE_THEN_CROP_TOLERANCE = 0.025


def encode(image: NDArray[np.uint8], extension: str, parameters: Sequence[int] = ()) -> bytes:
    encoded_ok, encoded = cv2.imencode(extension, image, parameters)
    assert encoded_ok
    return encoded.tobytes()


def segment(marker: bytes, payload: bytes) -> bytes:
    # The length field counts itself and the payload.
    return marker + struct.pack(">H", 2 + len(payload)) + payload


def first_segment(jpeg: bytes, marker: bytes) -> bytes:
    offset = jpeg.index(marker)
    (length_bytes,) = struct.unpack_from(">H", jpeg, offset + len(marker))
    return jpeg[offset : offset + len(marker) + length_bytes]


def exif_orientation_segment(orientation: int) -> bytes:
    """Build an application segment whose EXIF data holds only the orientation tag."""
    # A little-endian TIFF header that points at one directory of one entry:
    # tag 0x0112 (orientation), type 3 (16-bit), one value. No directory follows.
    tiff = struct.pack("<2sHI", b"II", 42, 8)
    directory = struct.pack("<HHHIHHI", 1, 0x0112, 3, 1, orientation, 0, 0)
    return segment(APPLICATION_1, b"Exif\x00\x00" + tiff + directory)


def before_frame_header(jpeg: bytes, inserted: bytes) -> bytes:
    offset = jpeg.index(BASELINE_FRAME_HEADER)
    return jpeg[:offset] + inserted + jpeg[offset:]


def declaring(jpeg: bytes, width_px: int, height_px: int) -> bytes:
    """Rewrite the frame header to declare this size, leaving the image data as it is."""
    # A baseline frame header is the marker, a 2-byte length and the sample
    # precision, followed by the height and the width as big-endian uint16.
    size_offset = jpeg.index(BASELINE_FRAME_HEADER) + 5
    return jpeg[:size_offset] + struct.pack(">HH", height_px, width_px) + jpeg[size_offset + 4 :]


def empty(jpeg: bytes) -> bytes:
    return b""


def truncated(jpeg: bytes) -> bytes:
    return jpeg[: len(jpeg) // 2]


def without_its_last_byte(jpeg: bytes) -> bytes:
    return jpeg[:-1]


def zeroed_after_its_signature(jpeg: bytes) -> bytes:
    return jpeg[:3] + bytes(len(jpeg) - 3)


def cut_inside_its_frame_header(jpeg: bytes) -> bytes:
    return jpeg[: jpeg.index(BASELINE_FRAME_HEADER) + 7]


def with_a_second_frame_header(jpeg: bytes) -> bytes:
    # The size check reads the first frame header, so it holds only as long as
    # the decoder refuses a frame that goes on to declare another size.
    header = first_segment(jpeg, BASELINE_FRAME_HEADER)
    other_header = first_segment(declaring(jpeg, 1280, 960), BASELINE_FRAME_HEADER)
    return jpeg.replace(header, header + other_header, 1)


def at_half_resolution(jpeg: bytes) -> bytes:
    return encode(decode_jpeg(jpeg, half_size=True), ".jpg")


def as_png(jpeg: bytes) -> bytes:
    return encode(decode_jpeg(jpeg, half_size=False), ".png")


def noise_frame(height: int, width: int) -> NDArray[np.uint8]:
    return np.random.default_rng(0).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def ramp_frame(height: int, width: int) -> NDArray[np.uint8]:
    """Blue rises from left to right, green from top to bottom, and red is their mean."""
    across = np.broadcast_to(np.linspace(0.0, 1.0, width)[np.newaxis, :], (height, width))
    down = np.broadcast_to(np.linspace(0.0, 1.0, height)[:, np.newaxis], (height, width))
    planes = np.stack((across, down, (across + down) / 2.0), axis=-1)
    return np.rint(255.0 * planes).astype(np.uint8)


def resize_then_center_crop(image_bgr: NDArray[np.uint8]) -> NDArray[np.float64]:
    """The documented MobileNet pipeline, one step at a time."""
    scale = 256 / min(image_bgr.shape[:2])
    resized = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    top = (resized.shape[0] - 224) // 2
    left = (resized.shape[1] - 224) // 2
    crop_rgb = resized[top : top + 224, left : left + 224, ::-1]
    normalized: NDArray[np.float64] = (crop_rgb / 255.0 - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB
    return normalized.transpose(2, 0, 1)


@pytest.fixture(scope="module")
def jpeg() -> bytes:
    """Return one frame as the robots send it."""
    return build_frame_set(1, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).jpegs[0]


@pytest.mark.parametrize(
    ("half_size", "shape"),
    [
        pytest.param(False, FULL_SIZE_SHAPE, id="full"),
        pytest.param(True, HALF_SIZE_SHAPE, id="half"),
    ],
)
def test_decode_returns_an_8_bit_frame_of_three_channels(
    jpeg: bytes, half_size: bool, shape: tuple[int, int, int]
) -> None:
    image = decode_jpeg(jpeg, half_size=half_size)

    assert image.shape == shape
    assert image.dtype == np.uint8


@pytest.mark.parametrize(
    "parameters",
    [
        pytest.param([cv2.IMWRITE_JPEG_PROGRESSIVE, 1], id="progressive"),
        pytest.param([cv2.IMWRITE_JPEG_OPTIMIZE, 1], id="optimized Huffman tables"),
        pytest.param([cv2.IMWRITE_JPEG_RST_INTERVAL, 8], id="restart intervals"),
    ],
)
def test_decode_accepts_other_jpeg_encodings_of_a_frame(jpeg: bytes, parameters: list[int]) -> None:
    reencoded = encode(decode_jpeg(jpeg, half_size=False), ".jpg", parameters)

    image = decode_jpeg(reencoded, half_size=False)

    assert image.shape == FULL_SIZE_SHAPE


@pytest.mark.parametrize(
    ("half_size", "shape"),
    [
        pytest.param(False, FULL_SIZE_SHAPE, id="full"),
        pytest.param(True, HALF_SIZE_SHAPE, id="half"),
    ],
)
def test_decode_gives_a_grayscale_frame_three_equal_channels(
    jpeg: bytes, half_size: bool, shape: tuple[int, int, int]
) -> None:
    # DESIGN.md 1.4 does not list a grayscale JPEG among the frames refused, so one of the frame
    # size is served, and the colour decode repeats its one channel.
    # OpenCV's stubs type the result of cvtColor as any number; it is the 8-bit plane.
    grey_plane = cv2.cvtColor(decode_jpeg(jpeg, half_size=False), cv2.COLOR_BGR2GRAY)
    grey = encode(grey_plane.astype(np.uint8), ".jpg")

    image = decode_jpeg(grey, half_size=half_size)

    assert image.shape == shape
    assert (image[..., 0] == image[..., 1]).all()
    assert (image[..., 1] == image[..., 2]).all()


@pytest.mark.parametrize(
    "inserted",
    [
        pytest.param(segment(COMMENT, b"camera 7"), id="comment"),
        pytest.param(exif_orientation_segment(EXIF_UPRIGHT), id="application data"),
        pytest.param(segment(RESTART_INTERVAL, bytes(2)), id="restart interval"),
        pytest.param(segment(ARITHMETIC_CONDITIONING, b"\x00\x10"), id="arithmetic conditioning"),
    ],
)
def test_decode_skips_the_segments_that_may_precede_the_frame_header(
    jpeg: bytes, inserted: bytes
) -> None:
    image = decode_jpeg(before_frame_header(jpeg, inserted), half_size=False)

    assert image.shape == FULL_SIZE_SHAPE


def test_decode_accepts_huffman_tables_ahead_of_the_frame_header(jpeg: bytes) -> None:
    tables_first = before_frame_header(jpeg, first_segment(jpeg, HUFFMAN_TABLE))

    image = decode_jpeg(tables_first, half_size=False)

    assert image.shape == FULL_SIZE_SHAPE


def test_decode_ignores_a_frame_header_inside_another_segment(jpeg: bytes) -> None:
    other_header = first_segment(declaring(jpeg, 16_000, 16_000), BASELINE_FRAME_HEADER)
    with_other_header_in_a_comment = before_frame_header(jpeg, segment(COMMENT, other_header))

    image = decode_jpeg(with_other_header_in_a_comment, half_size=False)

    assert image.shape == FULL_SIZE_SHAPE


@pytest.mark.parametrize(
    "damage",
    [
        empty,
        truncated,
        without_its_last_byte,
        zeroed_after_its_signature,
        cut_inside_its_frame_header,
        with_a_second_frame_header,
        at_half_resolution,
        as_png,
    ],
)
@pytest.mark.parametrize(
    "half_size", [pytest.param(False, id="full"), pytest.param(True, id="half")]
)
def test_decode_rejects_a_frame_that_is_not_an_intact_jpeg_of_the_frame_size(
    jpeg: bytes, damage: Callable[[bytes], bytes], half_size: bool
) -> None:
    with pytest.raises(FrameDecodeError):
        decode_jpeg(damage(jpeg), half_size=half_size)


@pytest.mark.parametrize(
    ("width_px", "height_px"),
    [
        pytest.param(641, 480, id="641x480"),
        pytest.param(640, 481, id="640x481"),
        pytest.param(16_000, 16_000, id="16000x16000"),
    ],
)
def test_decode_refuses_a_wrong_declared_size_without_running_the_decoder(
    jpeg: bytes, width_px: int, height_px: int, capfd: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(FrameDecodeError, match=f"frame header declares {width_px}x{height_px}"):
        decode_jpeg(declaring(jpeg, width_px, height_px), half_size=False)

    # The decoder allocates the image that the header declares, and it writes a
    # line to standard error when the frame holds less image data than that.
    assert capfd.readouterr().err == ""


def test_decode_reads_the_declared_size_from_the_header_that_the_decoder_reads(jpeg: bytes) -> None:
    right_header = first_segment(jpeg, BASELINE_FRAME_HEADER)
    oversized = before_frame_header(declaring(jpeg, 16_000, 16_000), segment(COMMENT, right_header))

    with pytest.raises(FrameDecodeError, match="frame header declares 16000x16000"):
        decode_jpeg(oversized, half_size=False)


@pytest.mark.parametrize(
    "inserted",
    [
        pytest.param(b"\xff", id="fill byte"),
        pytest.param(b"\x00", id="stray byte"),
        pytest.param(RESTART, id="restart marker"),
        pytest.param(COMMENT + b"\x00\x01", id="comment shorter than its length field"),
        pytest.param(segment(NUMBER_OF_LINES, bytes(2)), id="number of lines"),
    ],
)
def test_decode_refuses_what_the_decoder_would_skip_on_its_way_to_the_frame_header(
    jpeg: bytes, inserted: bytes
) -> None:
    # The decoder passes over each of these and decodes the frame. They are
    # refused all the same, because the size check has to read the header
    # exactly as the decoder will. The restart marker shows what is at stake: it
    # has no length, so a check that took the next two bytes for one would skip
    # the very frame header that the decoder goes on to read.
    with pytest.raises(FrameDecodeError, match="does not start a JPEG segment"):
        decode_jpeg(before_frame_header(jpeg, inserted), half_size=False)


def test_decode_rejects_a_frame_that_its_exif_orientation_turns_on_its_side(jpeg: bytes) -> None:
    turned = before_frame_header(jpeg, exif_orientation_segment(EXIF_QUARTER_TURN))

    with pytest.raises(FrameDecodeError, match="frame decodes to 480x640"):
        decode_jpeg(turned, half_size=False)


def test_decode_reports_a_refusal_by_opencv_as_a_decode_error(
    jpeg: bytes, monkeypatch: pytest.MonkeyPatch
) -> None:
    opencv_imdecode = cv2.imdecode
    # OpenCV raises for an empty buffer. No frame that passes the header check
    # is known to make it raise, so it is handed that buffer in the frame's place.
    monkeypatch.setattr(cv2, "imdecode", lambda buffer, flags: opencv_imdecode(buffer[:0], flags))

    with pytest.raises(FrameDecodeError, match="OpenCV refused the frame"):
        decode_jpeg(jpeg, half_size=False)


@pytest.mark.parametrize(("key", "side_px"), [("cls", 224), ("depth", 256)])
@pytest.mark.parametrize(("height", "width"), FRAME_SIZES)
def test_preprocess_returns_a_contiguous_float32_tensor_in_the_model_layout(
    key: str, side_px: int, height: int, width: int
) -> None:
    tensor = MODELS[key].preprocess(noise_frame(height, width))

    assert tensor.shape == (3, side_px, side_px)
    assert tensor.dtype == np.float32
    assert tensor.flags.c_contiguous


@pytest.mark.parametrize("key", ["cls", "depth"])
@pytest.mark.parametrize(("height", "width"), FRAME_SIZES)
def test_preprocess_stays_within_the_range_of_black_to_white(
    key: str, height: int, width: int
) -> None:
    lowest, highest = PLANE_RANGES[key]

    tensor = MODELS[key].preprocess(noise_frame(height, width))

    assert (tensor.min(axis=(1, 2)) >= lowest - FLOAT32_TOLERANCE).all()
    assert (tensor.max(axis=(1, 2)) <= highest + FLOAT32_TOLERANCE).all()


@pytest.mark.parametrize(
    ("key", "expected_planes"),
    [
        ("cls", (CONSTANT_FRAME_RGB - IMAGENET_MEAN_RGB) / IMAGENET_STD_RGB),
        ("depth", CONSTANT_FRAME_RGB),
    ],
)
@pytest.mark.parametrize(("height", "width"), FRAME_SIZES)
def test_preprocess_maps_a_constant_bgr_frame_to_constant_rgb_planes(
    key: str, expected_planes: NDArray[np.float64], height: int, width: int
) -> None:
    frame = np.full((height, width, 3), CONSTANT_FRAME_BGR, dtype=np.uint8)

    tensor = MODELS[key].preprocess(frame)

    expected = np.broadcast_to(expected_planes[:, np.newaxis, np.newaxis], tensor.shape)
    assert tensor == pytest.approx(expected, abs=FLOAT32_TOLERANCE)


@pytest.mark.parametrize(("height", "width"), FRAME_SIZES)
def test_cls_preprocess_matches_resize_then_center_crop(height: int, width: int) -> None:
    frame = ramp_frame(height, width)

    tensor = MODELS["cls"].preprocess(frame)

    assert tensor == pytest.approx(resize_then_center_crop(frame), abs=RESIZE_THEN_CROP_TOLERANCE)


@pytest.mark.parametrize(("height", "width"), FRAME_SIZES)
def test_depth_preprocess_resizes_the_whole_frame(height: int, width: int) -> None:
    # The center of output pixel i lies at this source coordinate, as a fraction of the frame.
    across = np.clip(((np.arange(256) + 0.5) * width / 256 - 0.5) / (width - 1), 0.0, 1.0)
    down = np.clip(((np.arange(256) + 0.5) * height / 256 - 0.5) / (height - 1), 0.0, 1.0)
    expected_blue = np.broadcast_to(across[np.newaxis, :], (256, 256))
    expected_green = np.broadcast_to(down[:, np.newaxis], (256, 256))

    red, green, blue = MODELS["depth"].preprocess(ramp_frame(height, width))

    assert blue == pytest.approx(expected_blue, abs=LEVEL_TOLERANCE)
    assert green == pytest.approx(expected_green, abs=LEVEL_TOLERANCE)
    assert red == pytest.approx((expected_blue + expected_green) / 2.0, abs=LEVEL_TOLERANCE)
