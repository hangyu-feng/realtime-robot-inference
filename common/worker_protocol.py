"""Messages between the gateway and a model worker, and the length prefix that frames them.

The link is a Unix stream socket. Every message is a length prefix and then that many
payload bytes, the first of which names the kind. A reader reads exactly the 4 prefix
bytes, checks them with payload_length before it allocates anything, then reads exactly
the payload. This module only builds and parses prefixes and payloads; it does no I/O.
Messages are typed, and nothing is pickled.

One message goes each way per cycle (DESIGN.md 4.5): the gateway sends a batch request,
and the worker's result for it is also the worker's request for the next batch, so a
worker process sends no ready message after the one that ends its warm-up.

All integers are little-endian.

    prefix            "<I"         4 bytes   payload length, 1 to MAX_PAYLOAD_BYTES

    WorkerReady, kind 1, worker to gateway
      header          "<B"         1 byte    kind
                                             then a UTF-8 JSON object whose keys are the
                                             fields of WorkerReady; the keys of t_b_ns
                                             and t_b_cpu_ns are batch sizes in decimal

    BatchRequest, kind 2, gateway to worker
      header          "<BQH"      11 bytes   kind, batch_id, count
      per frame       "<I"         4 bytes   JPEG length, then the JPEG bytes

    BatchResult, kind 3, worker to gateway
      header          "<BQqqQQH"  43 bytes   kind, batch_id, received_ns, sent_ns,
                                             inference_ns, cpu_ns, count
      per frame       "<BQQQB"    26 bytes   status, decode_ns, preprocess_ns,
                                             postprocess_ns, body length, then the
                                             command body

Why each width:

- kind, u8: three kinds exist.
- batch_id, u64: the scheduler's Batch.batch_id, which counts up from 1 for the life of
  a scheduler. A u64 never wraps, so the id needs no wrap-around rule.
- count, u16: 1 to MAX_BATCH_FRAMES, in the width of the command header's batch_size.
- JPEG length, u32: a JPEG of at most max_message_bytes, inside MAX_PAYLOAD_BYTES.
- received_ns and sent_ns, i64: the worker's monotonic clock when it had the whole
  request and just before it writes the result. They are signed, as in the robot
  protocol, so that their difference stays in the type.
- inference_ns, cpu_ns, decode_ns, preprocess_ns and postprocess_ns, u64: durations,
  never negative.
- status, u8: a FrameStatus.
- body length, u8: the command bodies of common/commands.py are 6 or 20 bytes.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import IntEnum
from itertools import pairwise

from common.commands import CommandFormatError, decode_command

# The payload length that starts every message. A u32 holds MAX_PAYLOAD_BYTES with room to
# spare, and a fixed width lets a reader read exactly this much before it knows anything else.
LENGTH_PREFIX = struct.Struct("<I")

# Above every batch size a worker calibrates (DESIGN.md 6.2: up to 8), with room for the
# batch-cap comparison, and inside the u16 frame count of a request and a result.
MAX_BATCH_FRAMES = 64

# The most a reader holds for one payload. cloud/config.py refuses at startup a
# configuration whose largest batch request, built from frames of max_message_bytes, would
# exceed it (with the defaults the largest request is 8 frames of up to 1 MiB, about 8 MiB),
# so a batch built from frames that passed ingress always fits. encode_batch_request relies
# on that check and does not check the bound again.
MAX_PAYLOAD_BYTES = 64 * 2**20

_READY_KIND = 1
_REQUEST_KIND = 2
_RESULT_KIND = 3

_REQUEST_HEADER = struct.Struct("<BQH")
_JPEG_LENGTH = struct.Struct("<I")
_RESULT_HEADER = struct.Struct("<BQqqQQH")
_FRAME_RESULT = struct.Struct("<BQQQB")

# A batch size as a key of a calibration map, in its one decimal spelling. int() would
# also take a sign, spaces, leading zeros, underscores and non-ASCII digits.
_BATCH_SIZE_BY_KEY = {str(size): size for size in range(1, MAX_BATCH_FRAMES + 1)}

# The range of the u64 durations of a batch result. json.loads returns integers of up to
# 4300 digits, and the gateway converts calibrated times to float milliseconds, which
# raises OverflowError for an integer beyond the float range.
_MAX_DURATION_NS = 2**64 - 1

# A SHA-256 digest as hashlib's hexdigest() writes it.
_SHA256_HEX = re.compile("[0-9a-f]{64}")


class WorkerProtocolError(Exception):
    """A length prefix or a payload on a worker link is malformed."""


class FrameStatus(IntEnum):
    """What became of one frame. The gateway counts failures by kind for workers.<m>.errors."""

    OK = 0
    DECODE_ERROR = 1
    INFERENCE_ERROR = 2


# The model file is final when this message is sent. Nothing the gateway sends over the
# link can change it, and at the first start the gateway computes the batch cap only
# afterwards, from t_b_ns. So the worker chooses its file. Where the file depends on the
# cap (DESIGN.md 4.6: the published MiDaS file at cap 1, and the batched copy above it and
# whenever forced_batch_cap is set, so that every arm of the batch-cap comparison runs the
# same graph), the worker applies cloud.scheduler.plan_model to its own calibration before
# it sends this message, and the gateway gets the same cap from the reported table.
# A worker started again after a failure does not choose again: the gateway starts it with
# the cap fixed at the first start, and it loads the file for that cap. The cap is fixed for
# the run (DESIGN.md 1.3 rule 9, 6.2), and the scheduler, whose configuration is fixed when
# it is built, keeps dispatching batches up to it; a new calibration could otherwise give
# cap 1 and the published file, which refuses larger batches (ModelRunner.run_batch raises
# ValueError).
@dataclass(frozen=True, slots=True)
class WorkerReady:
    """A worker's first message, sent once after warm-up.

    t_b_ns and t_b_cpu_ns map each calibrated batch size to the wall time and the CPU time
    of one batch of that size. Their keys are the calibrated sizes, which start at 1 and
    ascend; there is no separate list of sizes, and workers.<m>.batch_sizes reports these
    keys. threads 0 is the runtime's default thread count. model_file and model_sha256 are
    both None for the fake and the null runtime, which load no file.
    """

    model: str
    pid: int
    model_file: str | None
    model_sha256: str | None
    threads: int
    t_b_ns: Mapping[int, int]
    t_b_cpu_ns: Mapping[int, int]


_READY_KEYS = frozenset(field.name for field in fields(WorkerReady))


@dataclass(frozen=True, slots=True)
class BatchRequest:
    """A batch for a worker. The JPEGs are slices of the payload they were read from."""

    batch_id: int
    jpegs: tuple[memoryview, ...]


@dataclass(frozen=True, slots=True)
class FrameResult:
    """One frame of a batch result.

    command_body is the command's body as common/commands.py encodes it, and empty unless
    the status is OK.
    """

    status: FrameStatus
    command_body: bytes
    decode_ns: int
    preprocess_ns: int
    postprocess_ns: int


@dataclass(frozen=True, slots=True)
class BatchResult:
    """A worker's results for one batch, with the frames in request order.

    It is also the worker's request for the next batch. inference_ns is the batch's
    BatchOutcome.inference_ns (assembling the batch tensor and the session run), and cpu_ns
    the process CPU time the batch used.
    """

    batch_id: int
    received_ns: int
    sent_ns: int
    inference_ns: int
    cpu_ns: int
    frames: tuple[FrameResult, ...]


def payload_length(prefix: bytes) -> int:
    """Read a length prefix, refusing a length that no payload may have."""
    length_bytes: int = LENGTH_PREFIX.unpack(prefix)[0]
    if not 1 <= length_bytes <= MAX_PAYLOAD_BYTES:
        raise WorkerProtocolError(
            f"payload length {length_bytes} is outside 1 to {MAX_PAYLOAD_BYTES} bytes"
        )
    return length_bytes


def batch_request_bytes(frame_count: int, jpeg_bytes: int) -> int:
    """Return the payload length of a request for frame_count JPEGs of jpeg_bytes each."""
    return _REQUEST_HEADER.size + frame_count * (_JPEG_LENGTH.size + jpeg_bytes)


def encode_worker_ready(ready: WorkerReady) -> bytes:
    """Return the whole message, length prefix included."""
    document = {
        "model": ready.model,
        "pid": ready.pid,
        "model_file": ready.model_file,
        "model_sha256": ready.model_sha256,
        "threads": ready.threads,
        "t_b_ns": {str(size): duration_ns for size, duration_ns in ready.t_b_ns.items()},
        "t_b_cpu_ns": {str(size): duration_ns for size, duration_ns in ready.t_b_cpu_ns.items()},
    }
    payload = bytes((_READY_KIND,)) + json.dumps(document, separators=(",", ":")).encode()
    return LENGTH_PREFIX.pack(len(payload)) + payload


def encode_batch_request(
    batch_id: int, jpegs: Sequence[bytes | memoryview]
) -> list[bytes | memoryview]:
    """Return the whole message, length prefix included, as buffers for one writelines()."""
    # Each JPEG goes out as the caller's own object, never copied into one large buffer.
    # Its length is counted in bytes, which is what writelines() sends: len() of a view
    # counts its items, and they may be wider than a byte.
    # The payload is not checked against MAX_PAYLOAD_BYTES: cloud/config.py refuses at
    # startup a configuration whose largest request would exceed it.
    frame_buffers: list[bytes | memoryview] = []
    payload_bytes = _REQUEST_HEADER.size
    for jpeg in jpegs:
        jpeg_bytes = memoryview(jpeg).nbytes
        frame_buffers.extend((_JPEG_LENGTH.pack(jpeg_bytes), jpeg))
        payload_bytes += _JPEG_LENGTH.size + jpeg_bytes
    prefix_and_header = LENGTH_PREFIX.pack(payload_bytes) + _REQUEST_HEADER.pack(
        _REQUEST_KIND, batch_id, len(jpegs)
    )
    return [prefix_and_header, *frame_buffers]


def decode_batch_request(payload: memoryview) -> BatchRequest:
    """Parse a batch request without copying: each JPEG is a slice of the payload."""
    try:
        kind, batch_id, count = _REQUEST_HEADER.unpack_from(payload)
    except struct.error as error:
        raise WorkerProtocolError(
            f"batch request of {len(payload)} bytes is shorter than its header"
        ) from error
    if kind != _REQUEST_KIND:
        raise WorkerProtocolError(f"message kind {kind} is not a batch request")
    if not 1 <= count <= MAX_BATCH_FRAMES:
        raise WorkerProtocolError(
            f"batch request of {count} frames, expected 1 to {MAX_BATCH_FRAMES}"
        )
    jpegs: list[memoryview] = []
    offset = _REQUEST_HEADER.size
    for _ in range(count):
        try:
            jpeg_bytes: int = _JPEG_LENGTH.unpack_from(payload, offset)[0]
        except struct.error as error:
            raise WorkerProtocolError(
                f"batch request ends inside the frame length at byte {offset}"
            ) from error
        start = offset + _JPEG_LENGTH.size
        offset = start + jpeg_bytes
        if offset > len(payload):
            raise WorkerProtocolError(
                f"the frame at byte {start} runs past the end of the batch request"
            )
        # workers/runtime.py takes bytes (decode_jpeg calls bytes.startswith), so the
        # worker loop passes bytes(view) for each frame: one copy, outside the timed decode.
        jpegs.append(payload[start:offset])
    if offset != len(payload):
        raise WorkerProtocolError(
            f"batch request is {len(payload)} bytes, but its frames end at byte {offset}"
        )
    return BatchRequest(batch_id, tuple(jpegs))


def encode_batch_result(result: BatchResult) -> bytes:
    """Return the whole message, length prefix included."""
    payload = bytearray(
        _RESULT_HEADER.pack(
            _RESULT_KIND,
            result.batch_id,
            result.received_ns,
            result.sent_ns,
            result.inference_ns,
            result.cpu_ns,
            len(result.frames),
        )
    )
    for frame in result.frames:
        payload += _FRAME_RESULT.pack(
            frame.status,
            frame.decode_ns,
            frame.preprocess_ns,
            frame.postprocess_ns,
            len(frame.command_body),
        )
        payload += frame.command_body
    return LENGTH_PREFIX.pack(len(payload)) + payload


def decode_worker_message(payload: memoryview, model: str) -> WorkerReady | BatchResult:
    """Parse a message from the worker of this model, copying command bodies out of it."""
    # The batch in flight is not known here: the caller compares batch_id and the number
    # of frames with it.
    if not payload:
        raise WorkerProtocolError("empty message from a worker")
    kind = payload[0]
    if kind == _READY_KIND:
        return _decode_worker_ready(payload, model)
    if kind == _RESULT_KIND:
        return _decode_batch_result(payload, model)
    raise WorkerProtocolError(f"message kind {kind} is not one that a worker sends")


def _decode_worker_ready(payload: memoryview, model: str) -> WorkerReady:
    try:
        # JSONDecodeError and UnicodeDecodeError are ValueErrors too. A plain ValueError
        # comes from an integer of more than 4300 digits, RecursionError from deep nesting.
        document = json.loads(str(payload[1:], "utf-8"))
    except (ValueError, RecursionError) as error:
        raise WorkerProtocolError(f"ready message is not UTF-8 JSON: {error}") from error
    if not isinstance(document, dict) or document.keys() != _READY_KEYS:
        raise WorkerProtocolError(
            f"ready message is not an object with exactly the keys {sorted(_READY_KEYS)}"
        )
    if document["model"] != model:
        raise WorkerProtocolError(f"ready message on the {model} link names another model")
    pid = document["pid"]
    if type(pid) is not int or pid < 1:
        raise WorkerProtocolError("pid is not an integer of at least 1")
    threads = document["threads"]
    if type(threads) is not int or threads < 0:
        raise WorkerProtocolError("threads is not an integer of at least 0")
    model_file, model_sha256 = _model_file(document["model_file"], document["model_sha256"])
    t_b_ns = _calibration(document["t_b_ns"], "t_b_ns")
    t_b_cpu_ns = _calibration(document["t_b_cpu_ns"], "t_b_cpu_ns")
    if list(t_b_cpu_ns) != list(t_b_ns):
        raise WorkerProtocolError("t_b_ns and t_b_cpu_ns do not have the same batch sizes")
    return WorkerReady(model, pid, model_file, model_sha256, threads, t_b_ns, t_b_cpu_ns)


def _model_file(model_file: object, model_sha256: object) -> tuple[str | None, str | None]:
    """Check the served file's name and hash: both null, or both set."""
    if model_file is None and model_sha256 is None:
        return None, None
    if model_file is None or model_sha256 is None:
        raise WorkerProtocolError("model_file and model_sha256 are not both null or both set")
    if not isinstance(model_file, str) or not model_file:
        raise WorkerProtocolError("model_file is not a non-empty string")
    if not isinstance(model_sha256, str) or _SHA256_HEX.fullmatch(model_sha256) is None:
        raise WorkerProtocolError("model_sha256 is not 64 lowercase hexadecimal characters")
    return model_file, model_sha256


def _calibration(entries: object, name: str) -> dict[int, int]:
    """Read one calibration map, whose batch sizes start at 1 and ascend."""
    if not isinstance(entries, dict):
        raise WorkerProtocolError(f"{name} is not an object")
    calibration: dict[int, int] = {}
    for key, duration_ns in entries.items():
        if key not in _BATCH_SIZE_BY_KEY:
            raise WorkerProtocolError(
                f"{name} has a key that is not a batch size from 1 to {MAX_BATCH_FRAMES}"
            )
        if type(duration_ns) is not int or not 0 <= duration_ns <= _MAX_DURATION_NS:
            raise WorkerProtocolError(
                f"{name} has a value that is not an integer from 0 to {_MAX_DURATION_NS}"
            )
        calibration[_BATCH_SIZE_BY_KEY[key]] = duration_ns
    sizes = list(calibration)
    if sizes[:1] != [1] or any(smaller >= larger for smaller, larger in pairwise(sizes)):
        raise WorkerProtocolError(f"the batch sizes of {name} do not start at 1 and ascend")
    return calibration


def _decode_batch_result(payload: memoryview, model: str) -> BatchResult:
    try:
        header = _RESULT_HEADER.unpack_from(payload)
    except struct.error as error:
        raise WorkerProtocolError(
            f"batch result of {len(payload)} bytes is shorter than its header"
        ) from error
    _, batch_id, received_ns, sent_ns, inference_ns, cpu_ns, count = header
    if not 1 <= count <= MAX_BATCH_FRAMES:
        raise WorkerProtocolError(
            f"batch result of {count} frames, expected 1 to {MAX_BATCH_FRAMES}"
        )
    if sent_ns < received_ns:
        raise WorkerProtocolError("batch result sent before its request was received")
    frames: list[FrameResult] = []
    offset = _RESULT_HEADER.size
    for _ in range(count):
        frame, offset = _decode_frame_result(payload, offset, model)
        frames.append(frame)
    if offset != len(payload):
        raise WorkerProtocolError(
            f"batch result is {len(payload)} bytes, but its frames end at byte {offset}"
        )
    return BatchResult(batch_id, received_ns, sent_ns, inference_ns, cpu_ns, tuple(frames))


def _decode_frame_result(payload: memoryview, offset: int, model: str) -> tuple[FrameResult, int]:
    """Parse the frame result at offset, and return it with the offset after it."""
    try:
        status_value, decode_ns, preprocess_ns, postprocess_ns, body_bytes = (
            _FRAME_RESULT.unpack_from(payload, offset)
        )
    except struct.error as error:
        raise WorkerProtocolError(f"batch result ends inside the frame at byte {offset}") from error
    try:
        status = FrameStatus(status_value)
    except ValueError as error:
        raise WorkerProtocolError(
            f"unknown frame status {status_value} at byte {offset}"
        ) from error
    start = offset + _FRAME_RESULT.size
    end = start + body_bytes
    if end > len(payload):
        raise WorkerProtocolError(
            f"the command body at byte {start} runs past the end of the batch result"
        )
    # Copied, so that the result outlives the buffer the payload was read into.
    body = bytes(payload[start:end])
    if status is FrameStatus.OK:
        try:
            # Decoded only to check the body against the model: the gateway forwards the
            # body unchanged.
            decode_command(model, body)
        except CommandFormatError as error:
            raise WorkerProtocolError(f"frame at byte {offset}: {error}") from error
    elif body:
        raise WorkerProtocolError(f"frame at byte {offset} failed but carries a command body")
    return FrameResult(status, body, decode_ns, preprocess_ns, postprocess_ns), end
