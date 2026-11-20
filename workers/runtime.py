"""ONNX Runtime sessions with explicit thread settings, JPEG decoding, and one batch per call.

The model workers and the model benchmark both run their batches through this
module, so the benchmark measures the code that serves.
"""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import cv2
import numpy as np
import onnxruntime as ort
from numpy.typing import NDArray
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from common.clock import Clock
from common.commands import Command
from workers.models import ModelSpec

FRAME_WIDTH_PX = 640
FRAME_HEIGHT_PX = 480
OPENCV_THREAD_COUNT = 1

# A JPEG is the start-of-image marker and then segments. A segment is a marker
# of two bytes, the first of them 0xFF, and a length that counts its own two
# bytes and the payload. The segment that the standard calls the frame header
# declares the image: its payload opens with the sample precision, the height
# and the width. JPEG is big-endian.
_JPEG_START_OF_IMAGE = b"\xff\xd8"
_MARKER_PREFIX = 0xFF
_MARKER_BYTES = 2
_LENGTH_FIELD_BYTES = 2
_FRAME_HEADER_START = struct.Struct(">BBHBHH")
# Marker codes from ITU-T T.81, table B.1. The frame headers are the SOFn
# markers. 0xC4, 0xC8 and 0xCC lie in their range and are something else.
_FRAME_HEADER_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
# What the standard lets precede the frame header: Huffman tables (0xC4),
# arithmetic conditioning (0xCC), quantization tables (0xDB), the restart
# interval (0xDD), application data (0xE0 to 0xEF) and comments (0xFE).
_BEFORE_FRAME_HEADER_MARKERS = frozenset({0xC4, 0xCC, 0xDB, 0xDD, 0xFE, *range(0xE0, 0xF0)})

# The imdecode flag for each decode size, and the factor it divides the frame size by.
_FULL_SIZE_DECODE = (cv2.IMREAD_COLOR, 1)
_HALF_SIZE_DECODE = (cv2.IMREAD_REDUCED_COLOR_2, 2)

# ONNX Runtime's exceptions share no base class below Exception. These are the
# ones it raises for a file that is missing, unreadable or not a model, and for
# a graph it cannot build or has no CPU kernels for.
_MODEL_LOAD_ERRORS = (
    ort_state.NoSuchFile,
    ort_state.InvalidProtobuf,
    ort_state.InvalidArgument,
    ort_state.InvalidGraph,
    ort_state.NotImplemented,
    ort_state.Fail,
)


class ModelLoadError(Exception):
    """A model file is missing, or ONNX Runtime cannot build a session from it."""


class FrameDecodeError(Exception):
    """Frame bytes are not a JPEG that decodes to the expected resolution."""


@dataclass(frozen=True)
class SessionSettings:
    # 0 lets ONNX Runtime size the pool from the cores it sees. That is a
    # control experiment and never a default.
    intra_op_threads: int
    allow_spinning: bool


@dataclass(frozen=True, slots=True)
class FrameOutcome:
    """One frame of a batch. The command is None when the frame did not decode.

    A stage that the frame never reached has a duration of 0.
    """

    command: Command | None
    decode_ns: int
    preprocess_ns: int
    postprocess_ns: int


@dataclass(frozen=True, slots=True)
class BatchOutcome:
    """One batch, with its frames in the order they were given.

    The inference duration covers assembling the batch tensor and the session
    run. It is 0 when no frame decoded.
    """

    frames: tuple[FrameOutcome, ...]
    inference_ns: int


def configure_process_threads() -> None:
    """Set OpenCV's thread count. Call once in every process that decodes."""
    # Left alone, OpenCV sizes its pool from the CPUs it detects: the whole
    # quota of the container at best, every host core at worst. A process that
    # decodes is budgeted one working thread.
    cv2.setNumThreads(OPENCV_THREAD_COUNT)


def _declared_size_px(data: bytes) -> tuple[int, int]:
    """Return the width and height that a JPEG's frame header declares, decoding nothing."""
    # imdecode picks its decoder from the leading bytes. A frame that this walk
    # accepts starts the way OpenCV's JPEG decoder asks, so a robot cannot reach
    # the other image formats that OpenCV was built with.
    if not data.startswith(_JPEG_START_OF_IMAGE):
        raise FrameDecodeError("frame does not start with the JPEG start-of-image marker")
    offset = len(_JPEG_START_OF_IMAGE)
    while True:
        try:
            # A segment before the frame header is followed by the frame header
            # at the least, so reading this far never overruns a JPEG that has one.
            fields = _FRAME_HEADER_START.unpack_from(data, offset)
        except struct.error as error:
            raise FrameDecodeError("frame ends before a JPEG frame header") from error
        prefix, marker, length_bytes, _, height_px, width_px = fields
        if prefix == _MARKER_PREFIX and marker in _FRAME_HEADER_MARKERS:
            return width_px, height_px
        # The decoder is more lenient than this walk: it also passes over fill
        # bytes, stray bytes, markers that stand alone and segments that belong
        # elsewhere. A frame that needs any of that is refused, because a walk
        # that skips differently from the decoder can be shown one frame header
        # while the decoder reads another. Fill bytes before a marker are legal
        # JPEG and are refused too: libjpeg, which encodes every frame here,
        # writes the segments back to back.
        if (
            prefix != _MARKER_PREFIX
            or marker not in _BEFORE_FRAME_HEADER_MARKERS
            or length_bytes < _LENGTH_FIELD_BYTES
        ):
            raise FrameDecodeError(
                f"byte {offset} of the frame does not start a JPEG segment that may precede "
                "the JPEG frame header"
            )
        offset += _MARKER_BYTES + length_bytes


def decode_jpeg(data: bytes, *, half_size: bool) -> NDArray[np.uint8]:
    """Decode one frame to BGR, at half the resolution when asked, and check its size."""
    # The decoder allocates the image that the header declares before it reads
    # any image data, and fills in what a short frame leaves out. Checked only
    # after decoding, the size would bound nothing: the cost of a frame would
    # be set by what it declares, not by its bytes.
    width_px, height_px = _declared_size_px(data)
    if (width_px, height_px) != (FRAME_WIDTH_PX, FRAME_HEIGHT_PX):
        raise FrameDecodeError(
            f"JPEG frame header declares {width_px}x{height_px}, "
            f"expected {FRAME_WIDTH_PX}x{FRAME_HEIGHT_PX}"
        )
    flags, reduction = _HALF_SIZE_DECODE if half_size else _FULL_SIZE_DECODE
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), flags)
    except cv2.error as error:
        # OpenCV returns None for data it cannot decode, and raises when the size
        # it reads is over a limit of its own (OPENCV_IO_MAX_IMAGE_PIXELS and the
        # like). No frame that passes the header check is known to get that far,
        # but one that did must still fail alone.
        raise FrameDecodeError(f"OpenCV refused the frame: {error.err}") from error
    if image is None:
        raise FrameDecodeError("frame is not a decodable JPEG")
    # The decoder turns the image as its EXIF orientation says, which the frame
    # header does not show, so the decoded shape is checked as well.
    expected_shape = (FRAME_HEIGHT_PX // reduction, FRAME_WIDTH_PX // reduction, 3)
    if image.shape != expected_shape:
        height, width = image.shape[:2]
        raise FrameDecodeError(
            f"frame decodes to {width}x{height}, expected {expected_shape[1]}x{expected_shape[0]}"
        )
    # Both flags make OpenCV return 8-bit BGR, which its type stubs do not express.
    return cast(NDArray[np.uint8], image)


class ModelRunner:
    """One ONNX Runtime session, with its model's preprocessing and postprocessing.

    max_batch_size is the batch size that a fixed-batch graph is built for, and
    None for a graph that takes any batch size.
    """

    def __init__(
        self, spec: ModelSpec, model_path: Path, settings: SessionSettings, clock: Clock
    ) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = settings.intra_op_threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        # A spinning thread burns CPU quota while it waits for work.
        options.add_session_config_entry(
            "session.intra_op.allow_spinning", "1" if settings.allow_spinning else "0"
        )
        try:
            # The provider is named because this build also lists AzureExecutionProvider.
            session = ort.InferenceSession(
                model_path, sess_options=options, providers=["CPUExecutionProvider"]
            )
        except _MODEL_LOAD_ERRORS as error:
            raise ModelLoadError(
                f"cannot load model {spec.key!r} from {model_path}: {error}"
            ) from error
        self._spec = spec
        self._clock = clock
        self._session = session
        model_input = session.get_inputs()[0]
        self._input_name: str = model_input.name
        # ONNX Runtime reports a fixed dimension as its size and a symbolic one as its name.
        batch_dimension = model_input.shape[0]
        self.max_batch_size: int | None = (
            batch_dimension if isinstance(batch_dimension, int) else None
        )

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        """Run one batch. A frame that does not decode fails alone; the rest share one run."""
        clock = self._clock
        tensors: list[NDArray[np.float32]] = []
        # Per frame: decode and preprocess durations, and its row in the batch
        # tensor, which is None when the frame did not decode.
        prepared: list[tuple[int, int, int | None]] = []
        for jpeg in jpegs:
            start_ns = clock.now_ns()
            try:
                image = decode_jpeg(jpeg, half_size=False)
            except FrameDecodeError:
                prepared.append((clock.now_ns() - start_ns, 0, None))
                continue
            decoded_ns = clock.now_ns()
            tensors.append(self._spec.preprocess(image))
            prepared.append((decoded_ns - start_ns, clock.now_ns() - decoded_ns, len(tensors) - 1))

        if self.max_batch_size is not None and len(tensors) > self.max_batch_size:
            raise ValueError(
                f"{len(tensors)} frames decoded, but the {self._spec.key} graph "
                f"accepts {self.max_batch_size} per run"
            )

        inference_ns = 0
        rows: Sequence[NDArray[np.float32]] = ()
        if tensors:
            start_ns = clock.now_ns()
            rows = self._session.run(None, {self._input_name: np.stack(tensors)})[0]
            inference_ns = clock.now_ns() - start_ns

        frames: list[FrameOutcome] = []
        for decode_ns, preprocess_ns, row in prepared:
            if row is None:
                frames.append(FrameOutcome(None, decode_ns, 0, 0))
                continue
            start_ns = clock.now_ns()
            command = self._spec.postprocess(rows[row])
            frames.append(
                FrameOutcome(command, decode_ns, preprocess_ns, clock.now_ns() - start_ns)
            )
        return BatchOutcome(tuple(frames), inference_ns)
