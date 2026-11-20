"""Property tests for the wire protocol between a robot and the gateway."""

import contextlib
import json
from collections.abc import Callable
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.commands import MODEL_KEYS, ClsCommand, Command, DepthCommand, encode_command
from common.histogram import Histogram
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    MAX_COUNTER,
    MAX_FRAME_RATE_HZ,
    MAX_THRESHOLD_MS,
    MIN_FRAME_RATE_HZ,
    PROTOCOL_VERSION,
    ROBOT_ID_PATTERN,
    STAGE_NAMES,
    CommandHeader,
    FrameHeader,
    Hello,
    ModelTelemetry,
    ProtocolError,
    StageDurations,
    Telemetry,
    Welcome,
    decode_command_message,
    decode_frame,
    decode_hello,
    decode_telemetry,
    decode_welcome,
    encode_command_message,
    encode_frame,
    encode_hello,
    encode_telemetry,
    encode_welcome,
)

U16 = st.integers(min_value=0, max_value=2**16 - 1)
U32 = st.integers(min_value=0, max_value=2**32 - 1)
U64 = st.integers(min_value=0, max_value=2**64 - 1)
I64 = st.integers(min_value=-(2**63), max_value=2**63 - 1)
COUNTERS = st.integers(min_value=0, max_value=MAX_COUNTER)
THRESHOLDS_MS = st.floats(min_value=0.0, max_value=MAX_THRESHOLD_MS, exclude_min=True)
# Printable ASCII runs from the space to the tilde.
TOKENS = st.none() | st.text(st.characters(min_codepoint=0x20, max_codepoint=0x7E), min_size=1)
# A command body carries float32, so only values that float32 represents come back unchanged.
# NaN is left out because it never compares equal to itself.
WIRE_FLOATS = st.floats(width=32, allow_nan=False)


def recorded(values_ms: list[float]) -> Histogram:
    histogram = Histogram()
    for value_ms in values_ms:
        histogram.record(value_ms)
    return histogram


# Durations inside and outside the histogram range, negative ones included. The bounds keep
# the sum of any list drawn here finite.
HISTOGRAMS = st.lists(st.floats(min_value=-1e300, max_value=1e300), max_size=20).map(recorded)


@st.composite
def hellos(draw: st.DrawFn) -> Hello:
    models = tuple(draw(st.lists(st.sampled_from(MODEL_KEYS), min_size=1, unique=True)))
    return Hello(
        protocol_version=PROTOCOL_VERSION,
        robot_id=draw(st.from_regex(ROBOT_ID_PATTERN, fullmatch=True)),
        token=draw(TOKENS),
        models=models,
        stale_ms={model: draw(THRESHOLDS_MS) for model in models},
        watchdog_ms={model: draw(THRESHOLDS_MS) for model in models},
        frame_rate_hz=draw(st.floats(min_value=MIN_FRAME_RATE_HZ, max_value=MAX_FRAME_RATE_HZ)),
        frame_width_px=FRAME_WIDTH_PX,
        frame_height_px=FRAME_HEIGHT_PX,
    )


WELCOMES = st.builds(
    Welcome,
    protocol_version=st.just(PROTOCOL_VERSION),
    session=st.integers(min_value=1, max_value=2**32 - 1),
    shared_clock=st.booleans(),
)


@st.composite
def model_telemetry(draw: st.DrawFn) -> ModelTelemetry:
    # Each verdict count is at most a third of the largest counter, so their sum, which is
    # commands_received, is a counter too.
    verdict_counts = st.integers(min_value=0, max_value=MAX_COUNTER // 3)
    applied, stale, out_of_order = draw(verdict_counts), draw(verdict_counts), draw(verdict_counts)
    return ModelTelemetry(
        commands_received=applied + stale + out_of_order,
        commands_applied=applied,
        commands_stale=stale,
        commands_out_of_order=out_of_order,
        freshness_ms=draw(HISTOGRAMS),
        command_gap_ms=draw(HISTOGRAMS),
        command_age_peak_ms=draw(HISTOGRAMS),
        deliver_ms=draw(HISTOGRAMS),
        watchdog_trips=draw(COUNTERS),
        watchdog_expired_ms=draw(COUNTERS),
        watchdog_state=draw(st.sampled_from(("none", "ok", "expired"))),
    )


TELEMETRY = st.builds(
    Telemetry,
    ts_mono_ns=COUNTERS,
    frames_captured=COUNTERS,
    frames_sent=COUNTERS,
    frames_skipped_send=COUNTERS,
    fire_lag_ms=HISTOGRAMS,
    reconnects=COUNTERS,
    models=st.dictionaries(st.sampled_from(MODEL_KEYS), model_telemetry()),
)

FRAME_HEADERS = st.builds(FrameHeader, session=U32, frame_number=U64, capture_ns=I64, send_ns=I64)
JPEGS = st.binary().map(lambda tail: b"\xff\xd8" + tail)

STAGES = st.builds(StageDurations, *[U32] * len(STAGE_NAMES))
COMMANDS: dict[str, st.SearchStrategy[Command]] = {
    "cls": st.builds(ClsCommand, U16, WIRE_FLOATS),
    "depth": st.builds(DepthCommand, *[WIRE_FLOATS] * 5),
}


@st.composite
def command_messages(draw: st.DrawFn) -> tuple[CommandHeader, Command]:
    model = draw(st.sampled_from(MODEL_KEYS))
    header = CommandHeader(
        model=model,
        session=draw(U32),
        frame_number=draw(U64),
        capture_ns=draw(I64),
        batch_size=draw(st.integers(min_value=1, max_value=2**16 - 1)),
        result_ns=draw(I64),
        stages=draw(STAGES),
    )
    return header, draw(COMMANDS[model])


# json.loads reads integers of up to 4300 digits and json.dumps writes no longer ones, so a
# longer integer is tested as text, in the unit tests.
INTEGERS = st.integers() | st.builds(
    lambda digits, sign: sign * 10**digits,
    st.integers(min_value=19, max_value=4299),
    st.sampled_from((1, -1)),
)
# json.dumps writes NaN and the infinities as NaN, Infinity and -Infinity, which json.loads
# reads back.
SCALARS = st.none() | st.booleans() | INTEGERS | st.floats() | st.text()
JSON_LIKE = st.recursive(
    SCALARS, lambda children: st.lists(children) | st.dictionaries(st.text(), children)
)
# Characters that JSON gives a meaning to, so that a replaced character often changes a
# number, a string, a literal or the nesting rather than only breaking the syntax.
JSON_CHARACTERS = st.characters() | st.sampled_from('{}[]":,.-+eE0123456789 aflnrstu\\')

TEXT_DECODERS = [
    pytest.param(decode_hello, id="hello"),
    pytest.param(decode_welcome, id="welcome"),
    pytest.param(decode_telemetry, id="telemetry"),
]
VALID_TEXTS = [
    pytest.param(decode_hello, hellos().map(encode_hello), id="hello"),
    pytest.param(decode_welcome, WELCOMES.map(encode_welcome), id="welcome"),
    pytest.param(decode_telemetry, TELEMETRY.map(encode_telemetry), id="telemetry"),
]
BINARY_DECODERS = [
    pytest.param(decode_frame, id="frame"),
    pytest.param(decode_command_message, id="command"),
]
VALID_BINARIES = [
    pytest.param(decode_frame, st.builds(encode_frame, FRAME_HEADERS, JPEGS), id="frame"),
    pytest.param(
        decode_command_message,
        command_messages().map(
            lambda message: encode_command_message(message[0], encode_command(message[1]))
        ),
        id="command",
    ),
]


def key_paths(document: object) -> list[tuple[str, ...]]:
    """Return the path to every value in a JSON document, nested ones included."""
    paths: list[tuple[str, ...]] = []
    if isinstance(document, dict):
        for key, value in document.items():
            paths.append((key,))
            paths.extend((key, *path) for path in key_paths(value))
    return paths


def replaced(document: Any, path: tuple[str, ...], value: object) -> Any:
    """Return a copy of a JSON document with the value at path replaced."""
    if not path:
        return value
    return {**document, path[0]: replaced(document[path[0]], path[1:], value)}


@st.composite
def mutated_text(draw: st.DrawFn, text: str) -> str:
    """Return the text with some characters replaced, then cut short or extended."""
    characters = list(text)
    for index in draw(st.lists(st.integers(min_value=0, max_value=len(text) - 1), max_size=4)):
        characters[index] = draw(JSON_CHARACTERS)
    length = draw(st.integers(min_value=0, max_value=len(characters)))
    return "".join(characters[:length]) + draw(st.text(JSON_CHARACTERS))


@st.composite
def mutated_bytes(draw: st.DrawFn, message: bytes) -> bytes:
    """Return the message with some bytes flipped, then cut short or extended."""
    data = bytearray(message)
    for index in draw(st.lists(st.integers(min_value=0, max_value=len(data) - 1), max_size=4)):
        data[index] ^= draw(st.integers(min_value=1, max_value=255))
    length = draw(st.integers(min_value=0, max_value=len(data)))
    return bytes(data[:length]) + draw(st.binary())


@given(hello=hellos())
def test_hello_survives_the_wire(hello: Hello) -> None:
    assert decode_hello(encode_hello(hello)) == hello


@given(welcome=WELCOMES)
def test_welcome_survives_the_wire(welcome: Welcome) -> None:
    assert decode_welcome(encode_welcome(welcome)) == welcome


@given(telemetry=TELEMETRY)
def test_telemetry_survives_the_wire(telemetry: Telemetry) -> None:
    assert decode_telemetry(encode_telemetry(telemetry)) == telemetry


@given(header=FRAME_HEADERS, jpeg=JPEGS)
def test_frame_survives_the_wire(header: FrameHeader, jpeg: bytes) -> None:
    decoded, view = decode_frame(encode_frame(header, jpeg))

    assert (decoded, bytes(view)) == (header, jpeg)


@given(message=command_messages())
def test_command_survives_the_wire(message: tuple[CommandHeader, Command]) -> None:
    header, command = message

    assert decode_command_message(encode_command_message(header, encode_command(command))) == (
        header,
        command,
    )


@pytest.mark.parametrize("decode", TEXT_DECODERS)
@given(text=st.text())
def test_text_decoder_raises_only_protocol_error_for_any_text(
    decode: Callable[[str], object], text: str
) -> None:
    with contextlib.suppress(ProtocolError):
        decode(text)


@pytest.mark.parametrize("decode", TEXT_DECODERS)
@given(value=JSON_LIKE)
def test_text_decoder_raises_only_protocol_error_for_any_json_value(
    decode: Callable[[str], object], value: object
) -> None:
    with contextlib.suppress(ProtocolError):
        decode(json.dumps(value))


@pytest.mark.parametrize(("decode", "texts"), VALID_TEXTS)
@given(data=st.data())
def test_text_decoder_raises_only_protocol_error_when_a_value_of_a_valid_message_is_replaced(
    decode: Callable[[str], object], texts: st.SearchStrategy[str], data: st.DataObject
) -> None:
    document = json.loads(data.draw(texts))
    path = data.draw(st.sampled_from(key_paths(document)))
    text = json.dumps(replaced(document, path, data.draw(JSON_LIKE)))

    with contextlib.suppress(ProtocolError):
        decode(text)


@pytest.mark.parametrize(("decode", "texts"), VALID_TEXTS)
@given(data=st.data())
def test_text_decoder_raises_only_protocol_error_for_a_mutated_valid_message(
    decode: Callable[[str], object], texts: st.SearchStrategy[str], data: st.DataObject
) -> None:
    text = data.draw(mutated_text(data.draw(texts)))

    with contextlib.suppress(ProtocolError):
        decode(text)


@pytest.mark.parametrize("decode", BINARY_DECODERS)
@given(message=st.binary())
def test_binary_decoder_raises_only_protocol_error_for_any_bytes(
    decode: Callable[[bytes], object], message: bytes
) -> None:
    with contextlib.suppress(ProtocolError):
        decode(message)


@pytest.mark.parametrize(("decode", "messages"), VALID_BINARIES)
@given(data=st.data())
def test_binary_decoder_raises_only_protocol_error_for_a_mutated_valid_message(
    decode: Callable[[bytes], object], messages: st.SearchStrategy[bytes], data: st.DataObject
) -> None:
    message = data.draw(mutated_bytes(data.draw(messages)))

    with contextlib.suppress(ProtocolError):
        decode(message)
