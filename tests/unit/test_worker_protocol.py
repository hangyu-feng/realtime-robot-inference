"""Tests for the messages between the gateway and a model worker."""

import json
import struct
from collections.abc import Callable
from dataclasses import replace
from functools import partial

import pytest

from common.commands import ClsCommand, DepthCommand, encode_command
from common.worker_protocol import (
    LENGTH_PREFIX,
    MAX_BATCH_FRAMES,
    MAX_PAYLOAD_BYTES,
    BatchRequest,
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
    batch_request_bytes,
    decode_batch_request,
    decode_worker_message,
    encode_batch_request,
    encode_batch_result,
    encode_worker_ready,
    payload_length,
)

# Sizes and offsets from the layouts in the docstring of common/worker_protocol.py.
REQUEST_HEADER_BYTES = 11
RESULT_HEADER_BYTES = 43
FRAME_RESULT_BYTES = 26
# In a frame result, the body length follows the status byte and three u64 durations.
BODY_LENGTH_OFFSET = 25

CLS_BODY = encode_command(ClsCommand(class_id=7, confidence=0.5))
DEPTH_BODY = encode_command(
    DepthCommand(left=0.25, center=0.5, right=0.75, steering=-0.5, speed=0.5)
)
SHA256 = "0123456789abcdef" * 4
JPEGS = (b"\xff\xd8first", b"\xff\xd8second")

READY = WorkerReady(
    model="depth",
    pid=4321,
    model_file="midas_small_batched.onnx",
    model_sha256=SHA256,
    threads=1,
    t_b_ns={1: 10, 2: 20, 4: 40, 8: 80},
    t_b_cpu_ns={1: 11, 2: 21, 4: 41, 8: 81},
)

DELIVERED = FrameResult(FrameStatus.OK, CLS_BODY, decode_ns=10, preprocess_ns=20, postprocess_ns=30)
UNDECODABLE = FrameResult(
    FrameStatus.DECODE_ERROR, b"", decode_ns=5, preprocess_ns=0, postprocess_ns=0
)
NOT_INFERRED = FrameResult(
    FrameStatus.INFERENCE_ERROR, b"", decode_ns=10, preprocess_ns=20, postprocess_ns=0
)
RESULT = BatchResult(
    batch_id=12,
    received_ns=1_000,
    sent_ns=2_000,
    inference_ns=300,
    cpu_ns=400,
    frames=(DELIVERED, UNDECODABLE, NOT_INFERRED),
)


def payload_of(message: bytes) -> memoryview:
    """Check that a message's length prefix counts the rest of it, and return the rest."""
    assert payload_length(message[: LENGTH_PREFIX.size]) == len(message) - LENGTH_PREFIX.size
    return memoryview(message)[LENGTH_PREFIX.size :]


def overwritten(payload: memoryview, offset: int, data: bytes) -> memoryview:
    """Return a copy of the payload with data written at offset."""
    copy = bytearray(payload)
    copy[offset : offset + len(data)] = data
    return memoryview(copy)


REQUEST_PAYLOAD = payload_of(b"".join(encode_batch_request(5, JPEGS)))
RESULT_PAYLOAD = payload_of(encode_batch_result(RESULT))
READY_PAYLOAD = payload_of(encode_worker_ready(READY))
READY_KIND_BYTE = bytes(READY_PAYLOAD[:1])
READY_DOCUMENT = json.loads(bytes(READY_PAYLOAD[1:]))

DECODERS: list[tuple[str, memoryview, Callable[[memoryview], object]]] = [
    ("request", REQUEST_PAYLOAD, decode_batch_request),
    ("result", RESULT_PAYLOAD, partial(decode_worker_message, model="cls")),
    ("ready", READY_PAYLOAD, partial(decode_worker_message, model=READY.model)),
]


def test_layouts_have_the_documented_sizes() -> None:
    one_frame_result = replace(RESULT, frames=(UNDECODABLE,))

    assert LENGTH_PREFIX.size == 4
    assert batch_request_bytes(0, 0) == REQUEST_HEADER_BYTES
    assert len(payload_of(encode_batch_result(one_frame_result))) == (
        RESULT_HEADER_BYTES + FRAME_RESULT_BYTES
    )


def test_batch_request_is_little_endian_in_the_documented_field_order() -> None:
    message = b"".join(encode_batch_request(0x0102030405060708, [b"\xff\xd8"]))

    assert message == bytes.fromhex("11000000 02 0807060504030201 0100 02000000 ffd8")


def test_batch_result_is_little_endian_in_the_documented_field_order() -> None:
    frame = FrameResult(
        FrameStatus.OK,
        encode_command(ClsCommand(class_id=0x0102, confidence=0.5)),
        decode_ns=6,
        preprocess_ns=7,
        postprocess_ns=8,
    )
    result = BatchResult(
        0x0102030405060708, received_ns=-2, sent_ns=3, inference_ns=4, cpu_ns=5, frames=(frame,)
    )

    assert encode_batch_result(result) == bytes.fromhex(
        "4b000000 03 0807060504030201 feffffffffffffff 0300000000000000 0400000000000000"
        " 0500000000000000 0100"
        " 00 0600000000000000 0700000000000000 0800000000000000 06 0201 0000003f"
    )


def test_ready_message_is_its_kind_byte_then_compact_json() -> None:
    ready = WorkerReady(
        "cls",
        pid=7,
        model_file=None,
        model_sha256=None,
        threads=0,
        t_b_ns={1: 5},
        t_b_cpu_ns={1: 6},
    )
    text = (
        b'{"model":"cls","pid":7,"model_file":null,"model_sha256":null,"threads":0,'
        b'"t_b_ns":{"1":5},"t_b_cpu_ns":{"1":6}}'
    )

    assert encode_worker_ready(ready) == LENGTH_PREFIX.pack(1 + len(text)) + b"\x01" + text


@pytest.mark.parametrize(
    "ready",
    [
        pytest.param(READY, id="with-file"),
        pytest.param(
            replace(READY, model="cls", model_file=None, model_sha256=None, threads=0),
            id="without-file",
        ),
        pytest.param(
            replace(
                READY,
                t_b_ns={1: 0, MAX_BATCH_FRAMES: 2**64 - 1},
                t_b_cpu_ns={1: 2**64 - 1, MAX_BATCH_FRAMES: 0},
            ),
            id="extreme-sizes-and-times",
        ),
    ],
)
def test_ready_message_survives_the_link(ready: WorkerReady) -> None:
    assert decode_worker_message(payload_of(encode_worker_ready(ready)), ready.model) == ready


@pytest.mark.parametrize(
    "jpegs",
    [
        pytest.param([JPEGS[0], memoryview(JPEGS[1]), b""], id="bytes-view-and-empty"),
        pytest.param([b"\xff\xd8"] * MAX_BATCH_FRAMES, id="most-frames"),
    ],
)
def test_batch_request_survives_the_link(jpegs: list[bytes | memoryview]) -> None:
    message = b"".join(encode_batch_request(2**64 - 1, jpegs))

    assert decode_batch_request(payload_of(message)) == BatchRequest(
        2**64 - 1, tuple(memoryview(jpeg) for jpeg in jpegs)
    )


def test_batch_request_carries_every_byte_of_a_view_with_wide_items() -> None:
    jpeg = memoryview(JPEGS[1]).cast("I")

    request = decode_batch_request(payload_of(b"".join(encode_batch_request(1, [jpeg]))))

    assert [bytes(view) for view in request.jpegs] == [JPEGS[1]]


@pytest.mark.parametrize(
    ("model", "result"),
    [
        pytest.param("cls", RESULT, id="cls"),
        pytest.param(
            "depth",
            replace(RESULT, frames=(replace(DELIVERED, command_body=DEPTH_BODY), UNDECODABLE)),
            id="depth",
        ),
        pytest.param(
            "cls",
            BatchResult(
                batch_id=2**64 - 1,
                received_ns=-(2**63),
                sent_ns=2**63 - 1,
                inference_ns=2**64 - 1,
                cpu_ns=2**64 - 1,
                frames=(replace(UNDECODABLE, decode_ns=2**64 - 1),) * MAX_BATCH_FRAMES,
            ),
            id="extremes",
        ),
        pytest.param("cls", replace(RESULT, received_ns=7, sent_ns=7), id="sent-when-received"),
    ],
)
def test_batch_result_survives_the_link(model: str, result: BatchResult) -> None:
    assert decode_worker_message(payload_of(encode_batch_result(result)), model) == result


@pytest.mark.parametrize(
    ("frame_count", "jpeg_bytes"), [(1, 0), (1, 1000), (8, 2**10), (MAX_BATCH_FRAMES, 3)]
)
def test_batch_request_bytes_is_the_payload_length_of_an_encoded_request(
    frame_count: int, jpeg_bytes: int
) -> None:
    message = b"".join(encode_batch_request(1, [bytes(jpeg_bytes)] * frame_count))

    assert len(payload_of(message)) == batch_request_bytes(frame_count, jpeg_bytes)


def test_encoded_request_holds_the_callers_jpeg_objects() -> None:
    jpegs: list[bytes | memoryview] = [JPEGS[0], memoryview(JPEGS[1])]

    buffers = encode_batch_request(1, jpegs)

    assert len(buffers) == 1 + 2 * len(jpegs)
    assert all(buffer is jpeg for buffer, jpeg in zip(buffers[2::2], jpegs, strict=True))


def test_decoded_request_jpegs_are_slices_of_the_payload_buffer() -> None:
    message = b"".join(encode_batch_request(1, JPEGS))

    request = decode_batch_request(memoryview(message)[LENGTH_PREFIX.size :])

    assert [bytes(jpeg) for jpeg in request.jpegs] == list(JPEGS)
    assert all(jpeg.obj is message for jpeg in request.jpegs)


def test_decoded_result_does_not_depend_on_the_payload_buffer() -> None:
    buffer = bytearray(RESULT_PAYLOAD)
    with memoryview(buffer) as payload:
        result = decode_worker_message(payload, "cls")

    buffer[:] = bytes(len(buffer))

    assert result == RESULT


@pytest.mark.parametrize("length_bytes", [1, MAX_PAYLOAD_BYTES])
def test_payload_length_within_bounds_is_returned(length_bytes: int) -> None:
    assert payload_length(LENGTH_PREFIX.pack(length_bytes)) == length_bytes


@pytest.mark.parametrize("length_bytes", [0, MAX_PAYLOAD_BYTES + 1, 2**32 - 1])
def test_payload_length_out_of_bounds_is_refused(length_bytes: int) -> None:
    with pytest.raises(WorkerProtocolError, match=f"payload length {length_bytes} is outside"):
        payload_length(LENGTH_PREFIX.pack(length_bytes))


@pytest.mark.parametrize("kind", [0, 1, 3, 4, 255])
def test_request_decoder_refuses_every_other_kind(kind: int) -> None:
    with pytest.raises(WorkerProtocolError, match=f"message kind {kind} is not a batch request"):
        decode_batch_request(overwritten(REQUEST_PAYLOAD, 0, bytes((kind,))))


@pytest.mark.parametrize("kind", [0, 2, 4, 255])
def test_worker_message_decoder_refuses_kinds_that_a_worker_never_sends(kind: int) -> None:
    with pytest.raises(WorkerProtocolError, match=f"message kind {kind} is not one that a worker"):
        decode_worker_message(overwritten(RESULT_PAYLOAD, 0, bytes((kind,))), "cls")


@pytest.mark.parametrize(
    ("payload", "decode"),
    [
        pytest.param(payload[:length_bytes], decode, id=f"{name}-cut-to-{length_bytes}")
        for name, payload, decode in DECODERS
        for length_bytes in range(len(payload))
    ],
)
def test_payload_cut_short_at_any_byte_is_refused(
    payload: memoryview, decode: Callable[[memoryview], object]
) -> None:
    with pytest.raises(WorkerProtocolError):
        decode(payload)


@pytest.mark.parametrize(
    ("payload", "decode"),
    [
        pytest.param(memoryview(bytes(payload) + b"x"), decode, id=name)
        for name, payload, decode in DECODERS
    ],
)
def test_payload_with_trailing_bytes_is_refused(
    payload: memoryview, decode: Callable[[memoryview], object]
) -> None:
    with pytest.raises(WorkerProtocolError):
        decode(payload)


@pytest.mark.parametrize("frame_count", [0, MAX_BATCH_FRAMES + 1])
def test_request_with_a_frame_count_out_of_range_is_refused(frame_count: int) -> None:
    payload = payload_of(b"".join(encode_batch_request(1, [b"\xff\xd8"] * frame_count)))

    with pytest.raises(WorkerProtocolError, match=f"batch request of {frame_count} frames"):
        decode_batch_request(payload)


@pytest.mark.parametrize("frame_count", [0, MAX_BATCH_FRAMES + 1])
def test_result_with_a_frame_count_out_of_range_is_refused(frame_count: int) -> None:
    payload = payload_of(encode_batch_result(replace(RESULT, frames=(UNDECODABLE,) * frame_count)))

    with pytest.raises(WorkerProtocolError, match=f"batch result of {frame_count} frames"):
        decode_worker_message(payload, "cls")


@pytest.mark.parametrize("jpeg_bytes", [len(JPEGS[0]) + 1, 2**32 - 1])
def test_request_frame_running_past_the_end_is_refused(jpeg_bytes: int) -> None:
    payload = payload_of(b"".join(encode_batch_request(1, JPEGS[:1])))
    length_field = struct.pack("<I", jpeg_bytes)

    with pytest.raises(WorkerProtocolError, match="runs past the end"):
        decode_batch_request(overwritten(payload, REQUEST_HEADER_BYTES, length_field))


@pytest.mark.parametrize("body_bytes", [len(CLS_BODY) + 1, 255])
def test_result_body_running_past_the_end_is_refused(body_bytes: int) -> None:
    payload = payload_of(encode_batch_result(replace(RESULT, frames=(DELIVERED,))))
    offset = RESULT_HEADER_BYTES + BODY_LENGTH_OFFSET

    with pytest.raises(WorkerProtocolError, match="runs past the end"):
        decode_worker_message(overwritten(payload, offset, bytes((body_bytes,))), "cls")


def test_result_sent_before_its_request_was_received_is_refused() -> None:
    payload = payload_of(encode_batch_result(replace(RESULT, received_ns=10, sent_ns=9)))

    with pytest.raises(WorkerProtocolError, match="sent before its request was received"):
        decode_worker_message(payload, "cls")


@pytest.mark.parametrize("status", [3, 255])
def test_frame_with_an_unknown_status_is_refused(status: int) -> None:
    payload = payload_of(encode_batch_result(replace(RESULT, frames=(UNDECODABLE,))))

    with pytest.raises(WorkerProtocolError, match=f"unknown frame status {status}"):
        decode_worker_message(overwritten(payload, RESULT_HEADER_BYTES, bytes((status,))), "cls")


@pytest.mark.parametrize(
    ("model", "frame", "message"),
    [
        pytest.param(
            "cls",
            replace(DELIVERED, command_body=CLS_BODY[:-1]),
            "cls command body of 5 bytes",
            id="body-too-short",
        ),
        pytest.param(
            "cls", replace(DELIVERED, command_body=b""), "cls command body of 0 bytes", id="no-body"
        ),
        pytest.param(
            "cls",
            replace(DELIVERED, command_body=DEPTH_BODY),
            "cls command body of 20 bytes",
            id="depth-body-on-the-cls-link",
        ),
        pytest.param(
            "depth", DELIVERED, "depth command body of 6 bytes", id="cls-body-on-the-depth-link"
        ),
        pytest.param(
            "cls",
            replace(UNDECODABLE, command_body=CLS_BODY),
            "failed but carries a command body",
            id="body-on-a-decode-error",
        ),
        pytest.param(
            "cls",
            replace(NOT_INFERRED, command_body=CLS_BODY),
            "failed but carries a command body",
            id="body-on-an-inference-error",
        ),
    ],
)
def test_frame_with_a_body_that_breaks_a_rule_is_refused(
    model: str, frame: FrameResult, message: str
) -> None:
    payload = payload_of(encode_batch_result(replace(RESULT, frames=(frame,))))

    with pytest.raises(WorkerProtocolError, match=message):
        decode_worker_message(payload, model)


@pytest.mark.parametrize(
    "text",
    [
        pytest.param(b"", id="empty"),
        # Valid JSON apart from its encoding, so that only the strict UTF-8 decoding refuses it.
        pytest.param(
            json.dumps(
                {**READY_DOCUMENT, "model_file": "mid\xffas.onnx"}, ensure_ascii=False
            ).encode("latin-1"),
            id="latin-1",
        ),
        pytest.param(b"{", id="not-json"),
        pytest.param(b"\xef\xbb\xbf{}", id="byte-order-mark"),
        pytest.param(b"[" * 10**6, id="deeply-nested"),
        pytest.param(b'{"pid":' + b"1" * 5000 + b"}", id="integer-of-5000-digits"),
    ],
)
def test_ready_message_that_is_not_utf8_json_is_refused(text: bytes) -> None:
    with pytest.raises(WorkerProtocolError, match="not UTF-8 JSON"):
        decode_worker_message(memoryview(READY_KIND_BYTE + text), READY.model)


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param([READY_DOCUMENT], "not an object", id="not-an-object"),
        pytest.param(
            {key: value for key, value in READY_DOCUMENT.items() if key != "pid"},
            "exactly the keys",
            id="missing-key",
        ),
        pytest.param({**READY_DOCUMENT, "extra": 1}, "exactly the keys", id="extra-key"),
        pytest.param({**READY_DOCUMENT, "model": "cls"}, "names another model", id="other-model"),
        pytest.param({**READY_DOCUMENT, "model": "pose"}, "names another model", id="no-model"),
        pytest.param({**READY_DOCUMENT, "pid": 0}, "pid", id="pid-zero"),
        pytest.param({**READY_DOCUMENT, "pid": True}, "pid", id="pid-bool"),
        pytest.param({**READY_DOCUMENT, "pid": 1.0}, "pid", id="pid-float"),
        pytest.param({**READY_DOCUMENT, "threads": -1}, "threads", id="threads-negative"),
        pytest.param({**READY_DOCUMENT, "threads": False}, "threads", id="threads-bool"),
        pytest.param(
            {**READY_DOCUMENT, "model_sha256": None},
            "both null or both set",
            id="file-without-hash",
        ),
        pytest.param(
            {**READY_DOCUMENT, "model_file": None}, "both null or both set", id="hash-without-file"
        ),
        pytest.param({**READY_DOCUMENT, "model_file": ""}, "model_file", id="file-name-empty"),
        pytest.param({**READY_DOCUMENT, "model_file": 7}, "model_file", id="file-name-number"),
        pytest.param(
            {**READY_DOCUMENT, "model_sha256": SHA256.upper()}, "model_sha256", id="hash-uppercase"
        ),
        pytest.param(
            {**READY_DOCUMENT, "model_sha256": SHA256[:-1]}, "model_sha256", id="hash-too-short"
        ),
        pytest.param(
            {**READY_DOCUMENT, "model_sha256": SHA256 + "0"}, "model_sha256", id="hash-too-long"
        ),
        pytest.param(
            {**READY_DOCUMENT, "model_sha256": "g" * 64}, "model_sha256", id="hash-not-hex"
        ),
        pytest.param(
            {**READY_DOCUMENT, "t_b_cpu_ns": {"1": 11, "2": 21}},
            "same batch sizes",
            id="map-keys-mismatched",
        ),
    ],
)
def test_ready_message_breaking_a_rule_is_refused(document: object, message: str) -> None:
    payload = memoryview(READY_KIND_BYTE + json.dumps(document).encode())

    with pytest.raises(WorkerProtocolError, match=message):
        decode_worker_message(payload, READY.model)


@pytest.mark.parametrize("name", ["t_b_ns", "t_b_cpu_ns"])
@pytest.mark.parametrize(
    ("entries", "rule"),
    [
        pytest.param([10], "is not an object", id="not-an-object"),
        pytest.param({}, "do not start at 1 and ascend", id="no-sizes"),
        pytest.param({"2": 1, "4": 1}, "do not start at 1 and ascend", id="sizes-from-2"),
        pytest.param(
            {"1": 1, "4": 1, "2": 1}, "do not start at 1 and ascend", id="sizes-unordered"
        ),
        pytest.param({"1": 1, "01": 1}, "has a key that is not a batch size", id="leading-zero"),
        pytest.param({"1": 1, "+2": 1}, "has a key that is not a batch size", id="size-signed"),
        pytest.param({"0": 1}, "has a key that is not a batch size", id="size-zero"),
        pytest.param(
            {"1": 1, str(MAX_BATCH_FRAMES + 1): 1},
            "has a key that is not a batch size",
            id="size-above-the-bound",
        ),
    ],
)
def test_calibration_map_breaking_a_rule_is_refused(name: str, entries: object, rule: str) -> None:
    payload = memoryview(READY_KIND_BYTE + json.dumps({**READY_DOCUMENT, name: entries}).encode())

    with pytest.raises(WorkerProtocolError, match=f"{name} {rule}"):
        decode_worker_message(payload, READY.model)


@pytest.mark.parametrize("name", ["t_b_ns", "t_b_cpu_ns"])
@pytest.mark.parametrize(
    "duration_ns",
    [
        pytest.param(-1, id="negative"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(2**64, id="above-u64"),
    ],
)
def test_calibrated_time_that_is_not_a_u64_integer_is_refused(
    name: str, duration_ns: object
) -> None:
    # Only the time of size 1 changes, so the two maps keep the same batch sizes and the
    # value rule is the only one the message breaks.
    document = {**READY_DOCUMENT, name: {**READY_DOCUMENT[name], "1": duration_ns}}
    payload = memoryview(READY_KIND_BYTE + json.dumps(document).encode())

    with pytest.raises(WorkerProtocolError, match=f"{name} has a value that is not an integer"):
        decode_worker_message(payload, READY.model)
