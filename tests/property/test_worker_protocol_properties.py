"""Property tests for the messages between the gateway and a model worker."""

import contextlib
import json

from hypothesis import given
from hypothesis import strategies as st

from common.commands import MODEL_KEYS, ClsCommand, DepthCommand, encode_command
from common.worker_protocol import (
    LENGTH_PREFIX,
    MAX_BATCH_FRAMES,
    BatchRequest,
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
    decode_batch_request,
    decode_worker_message,
    encode_batch_request,
    encode_batch_result,
    encode_worker_ready,
)

# The kind byte of a ready message, from the layout in common/worker_protocol.py.
READY_KIND_BYTE = b"\x01"

models = st.sampled_from(MODEL_KEYS)
u64s = st.integers(min_value=0, max_value=2**64 - 1)
i64s = st.integers(min_value=-(2**63), max_value=2**63 - 1)
sha256_hexes = st.from_regex("[0-9a-f]{64}", fullmatch=True)
# Bodies are compared as bytes, so every float32 will do, NaN and the infinities included.
float32s = st.floats(width=32)
command_bodies = {
    "cls": st.builds(ClsCommand, st.integers(0, 2**16 - 1), float32s).map(encode_command),
    "depth": st.builds(DepthCommand, float32s, float32s, float32s, float32s, float32s).map(
        encode_command
    ),
}


def payload_of(message: bytes) -> bytes:
    return message[LENGTH_PREFIX.size :]


@st.composite
def worker_readies(draw: st.DrawFn, model: str) -> WorkerReady:
    sizes = sorted({1} | draw(st.sets(st.integers(2, MAX_BATCH_FRAMES), max_size=8)))
    model_file, model_sha256 = draw(
        st.just((None, None)) | st.tuples(st.text(min_size=1), sha256_hexes)
    )
    return WorkerReady(
        model=model,
        pid=draw(st.integers(min_value=1)),
        model_file=model_file,
        model_sha256=model_sha256,
        threads=draw(st.integers(min_value=0)),
        t_b_ns={size: draw(u64s) for size in sizes},
        t_b_cpu_ns={size: draw(u64s) for size in sizes},
    )


def frame_results(model: str) -> st.SearchStrategy[FrameResult]:
    delivered = st.builds(
        FrameResult, st.just(FrameStatus.OK), command_bodies[model], u64s, u64s, u64s
    )
    failed = st.builds(
        FrameResult,
        st.sampled_from([FrameStatus.DECODE_ERROR, FrameStatus.INFERENCE_ERROR]),
        st.just(b""),
        u64s,
        u64s,
        u64s,
    )
    return delivered | failed


@st.composite
def batch_results(draw: st.DrawFn, model: str) -> BatchResult:
    received_ns, sent_ns = sorted(draw(st.lists(i64s, min_size=2, max_size=2)))
    return BatchResult(
        batch_id=draw(u64s),
        received_ns=received_ns,
        sent_ns=sent_ns,
        inference_ns=draw(u64s),
        cpu_ns=draw(u64s),
        frames=tuple(draw(st.lists(frame_results(model), min_size=1, max_size=MAX_BATCH_FRAMES))),
    )


batch_requests = st.builds(
    BatchRequest,
    batch_id=u64s,
    jpegs=st.lists(
        st.binary(max_size=32).map(lambda jpeg: memoryview(jpeg)),
        min_size=1,
        max_size=MAX_BATCH_FRAMES,
    ).map(tuple),
)
request_payloads = batch_requests.map(
    lambda request: payload_of(b"".join(encode_batch_request(request.batch_id, request.jpegs)))
)


def worker_payloads(model: str) -> st.SearchStrategy[bytes]:
    """Valid payloads from the worker of this model: ready messages and batch results."""
    messages = worker_readies(model).map(encode_worker_ready) | batch_results(model).map(
        encode_batch_result
    )
    return messages.map(payload_of)


@st.composite
def flipped(draw: st.DrawFn, payloads: st.SearchStrategy[bytes]) -> bytes:
    payload = bytearray(draw(payloads))
    for index in draw(st.lists(st.integers(0, len(payload) - 1), min_size=1, max_size=4)):
        payload[index] ^= draw(st.integers(1, 255))
    return bytes(payload)


@st.composite
def truncated(draw: st.DrawFn, payloads: st.SearchStrategy[bytes]) -> bytes:
    payload = draw(payloads)
    return payload[: draw(st.integers(0, len(payload) - 1))]


@st.composite
def extended(draw: st.DrawFn, payloads: st.SearchStrategy[bytes]) -> bytes:
    return draw(payloads) + draw(st.binary(min_size=1, max_size=16))


def mutated(payloads: st.SearchStrategy[bytes]) -> st.SearchStrategy[bytes]:
    """Valid payloads with some bytes flipped, or cut short, or extended."""
    return flipped(payloads) | truncated(payloads) | extended(payloads)


# Arbitrary bytes, and arbitrary bytes behind each of the three kind bytes, so that some
# reach the parsing past the kind check.
arbitrary_payloads = st.binary() | st.builds(
    lambda kind, rest: bytes((kind,)) + rest, st.integers(1, 3), st.binary()
)

json_values = st.recursive(
    st.none() | st.booleans() | st.integers() | st.floats() | st.text(),
    lambda children: (
        st.lists(children, max_size=3) | st.dictionaries(st.text(), children, max_size=3)
    ),
    max_leaves=8,
)
size_keys = st.sampled_from(["0", "1", "2", "4", "01", "+1", " 1", str(MAX_BATCH_FRAMES + 1)])
calibrations = (
    st.dictionaries(size_keys | st.text(), st.integers() | json_values, max_size=4) | json_values
)
# For each field of a ready message, values close to valid ones as well as arbitrary ones.
field_values = {
    "model": models | json_values,
    "pid": st.integers() | json_values,
    "model_file": st.none() | st.text() | json_values,
    "model_sha256": st.none() | sha256_hexes | json_values,
    "threads": st.integers() | json_values,
    "t_b_ns": calibrations,
    "t_b_cpu_ns": calibrations,
}


@st.composite
def ready_documents_with_one_field_replaced(draw: st.DrawFn, model: str) -> dict[str, object]:
    """Return the JSON document of a valid ready message with one field's value replaced."""
    # One field at a time, so that the checks of the fields before it pass and every rule
    # is reached.
    ready = draw(worker_readies(model))
    document: dict[str, object] = json.loads(payload_of(encode_worker_ready(ready))[1:])
    field = draw(st.sampled_from(sorted(field_values)))
    document[field] = draw(field_values[field])
    return document


@given(ready=models.flatmap(worker_readies))
def test_ready_message_survives_the_link(ready: WorkerReady) -> None:
    payload = memoryview(payload_of(encode_worker_ready(ready)))

    assert decode_worker_message(payload, ready.model) == ready


@given(request=batch_requests)
def test_batch_request_survives_the_link(request: BatchRequest) -> None:
    message = b"".join(encode_batch_request(request.batch_id, request.jpegs))

    assert decode_batch_request(memoryview(payload_of(message))) == request


@given(model=models, data=st.data())
def test_batch_result_survives_the_link(model: str, data: st.DataObject) -> None:
    result = data.draw(batch_results(model))

    assert decode_worker_message(memoryview(payload_of(encode_batch_result(result))), model) == (
        result
    )


@given(payload=arbitrary_payloads)
def test_request_decoder_raises_only_protocol_errors_on_arbitrary_bytes(payload: bytes) -> None:
    with contextlib.suppress(WorkerProtocolError):
        decode_batch_request(memoryview(payload))


@given(payload=arbitrary_payloads, model=models)
def test_worker_message_decoder_raises_only_protocol_errors_on_arbitrary_bytes(
    payload: bytes, model: str
) -> None:
    with contextlib.suppress(WorkerProtocolError):
        decode_worker_message(memoryview(payload), model)


@given(payload=mutated(request_payloads))
def test_request_decoder_raises_only_protocol_errors_on_mutated_payloads(payload: bytes) -> None:
    with contextlib.suppress(WorkerProtocolError):
        decode_batch_request(memoryview(payload))


@given(model=models, data=st.data())
def test_worker_message_decoder_raises_only_protocol_errors_on_mutated_payloads(
    model: str, data: st.DataObject
) -> None:
    payload = data.draw(mutated(worker_payloads(model)))

    with contextlib.suppress(WorkerProtocolError):
        decode_worker_message(memoryview(payload), model)


@given(model=models, data=st.data())
def test_ready_decoder_raises_only_protocol_errors_on_arbitrary_field_values(
    model: str, data: st.DataObject
) -> None:
    document = data.draw(ready_documents_with_one_field_replaced(model))
    payload = memoryview(READY_KIND_BYTE + json.dumps(document).encode())

    with contextlib.suppress(WorkerProtocolError):
        decode_worker_message(payload, model)
