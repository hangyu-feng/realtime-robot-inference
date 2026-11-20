"""Tests for the wire protocol between a robot and the gateway."""

import json
import math
import struct
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import pytest

from common.commands import MODEL_KEYS, ClsCommand, Command, DepthCommand, encode_command
from common.histogram import BUCKET_COUNT, Histogram
from common.protocol import (
    FRAME_HEIGHT_PX,
    FRAME_WIDTH_PX,
    MAX_COUNTER,
    MAX_FRAME_RATE_HZ,
    MAX_TEXT_CHARS,
    MAX_THRESHOLD_MS,
    MIN_FRAME_RATE_HZ,
    PROTOCOL_VERSION,
    STAGE_NAMES,
    CloseCode,
    CommandHeader,
    Fault,
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
    telemetry_to_dict,
)
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS


def recorded(*values_ms: float) -> Histogram:
    histogram = Histogram()
    for value_ms in values_ms:
        histogram.record(value_ms)
    return histogram


def changed(document: dict[str, Any], **changes: object) -> str:
    """Return the JSON text of a message with the given fields replaced or added."""
    return json.dumps({**document, **changes})


def without(document: dict[str, Any], name: str) -> str:
    """Return the JSON text of a message without one field."""
    return json.dumps({key: value for key, value in document.items() if key != name})


# json.dumps never writes 1e999, which json.loads reads as infinity, so it is written over a
# placeholder number that appears nowhere else in a message.
PLACEHOLDER = 12345.678


def with_1e999(text: str) -> str:
    return text.replace(repr(PLACEHOLDER), "1e999")


HELLO = Hello(
    protocol_version=PROTOCOL_VERSION,
    robot_id="robot-01",
    token=None,
    models=("cls", "depth"),
    stale_ms={"cls": 500.0, "depth": 150.0},
    watchdog_ms={"cls": 1000.0, "depth": 300.0},
    frame_rate_hz=30.0,
    frame_width_px=FRAME_WIDTH_PX,
    frame_height_px=FRAME_HEIGHT_PX,
)
HELLO_DOCUMENT = json.loads(encode_hello(HELLO))

WELCOME = Welcome(protocol_version=PROTOCOL_VERSION, session=7, shared_clock=True)
WELCOME_DOCUMENT = json.loads(encode_welcome(WELCOME))

CLS_TELEMETRY = ModelTelemetry(
    commands_received=10,
    commands_applied=7,
    commands_stale=2,
    commands_out_of_order=1,
    freshness_ms=recorded(40.5, 61.25),
    command_gap_ms=recorded(33.25),
    command_age_peak_ms=Histogram(),
    deliver_ms=recorded(0.5),
    watchdog_trips=1,
    watchdog_expired_ms=250,
    watchdog_state="ok",
)
TELEMETRY = Telemetry(
    ts_mono_ns=123_456_789,
    frames_captured=30,
    frames_sent=29,
    frames_skipped_send=1,
    fire_lag_ms=recorded(0.25, 1.5),
    reconnects=0,
    models={"cls": CLS_TELEMETRY, "depth": replace(CLS_TELEMETRY, watchdog_state="expired")},
)
TELEMETRY_DOCUMENT = json.loads(encode_telemetry(TELEMETRY))
CLS_DOCUMENT = TELEMETRY_DOCUMENT["models"]["cls"]
FIRE_LAG_DOCUMENT = TELEMETRY_DOCUMENT["fire_lag_ms"]

TOP_COUNTERS = ("ts_mono_ns", "frames_captured", "frames_sent", "frames_skipped_send", "reconnects")
MODEL_COUNTERS = (
    "commands_received",
    "commands_applied",
    "commands_stale",
    "commands_out_of_order",
    "watchdog_trips",
    "watchdog_expired_ms",
)

JPEG = b"\xff\xd8\xff\xe0jpeg"
FRAME_HEADER = FrameHeader(session=7, frame_number=42, capture_ns=1_000, send_ns=1_500)
FRAME = encode_frame(FRAME_HEADER, JPEG)

CLS_COMMAND = ClsCommand(class_id=7, confidence=0.5)
DEPTH_COMMAND = DepthCommand(left=0.25, center=0.5, right=0.75, steering=-0.5, speed=0.5)
CLS_BODY = encode_command(CLS_COMMAND)
DEPTH_BODY = encode_command(DEPTH_COMMAND)
STAGES = StageDurations(*range(1, 9))
COMMAND_HEADER = CommandHeader(
    model="cls",
    session=7,
    frame_number=42,
    capture_ns=1_000,
    batch_size=4,
    result_ns=2_000,
    stages=STAGES,
)
COMMAND = encode_command_message(COMMAND_HEADER, CLS_BODY)


def overwritten(message: bytes, offset: int, data: bytes) -> bytes:
    """Return a copy of the message with data written at offset."""
    return message[:offset] + data + message[offset + len(data) :]


def test_close_codes_are_those_of_the_design() -> None:
    assert {code.name: code.value for code in CloseCode} == {
        "NORMAL": 1000,
        "SERVER_SHUTDOWN": 1001,
        "PROTOCOL_ERROR": 1002,
        "POLICY": 1008,
        "MESSAGE_TOO_BIG": 1009,
        "OVER_CAPACITY": 1013,
        "SESSION_REPLACED": 4001,
        "SLOW_CONSUMER": 4002,
        "IDLE_TIMEOUT": 4003,
    }


def test_faults_are_the_metrics_labels() -> None:
    assert [str(fault) for fault in Fault] == ["malformed", "version", "bad_hello", "unknown_model"]


# A robot and a gateway built from different code must agree on these values, so changing one
# changes what a peer accepts: it would be another protocol version.
def test_frame_size_and_bounds_are_those_of_version_1() -> None:
    assert (
        PROTOCOL_VERSION,
        FRAME_WIDTH_PX,
        FRAME_HEIGHT_PX,
        MAX_TEXT_CHARS,
        MAX_THRESHOLD_MS,
        MIN_FRAME_RATE_HZ,
        MAX_FRAME_RATE_HZ,
        MAX_COUNTER,
    ) == (1, 640, 480, 65_536, 60_000.0, 1.0, 60.0, 2**63 - 1)


@pytest.mark.parametrize(
    "hello",
    [
        pytest.param(HELLO, id="both-models"),
        pytest.param(
            replace(
                HELLO, models=("depth",), stale_ms={"depth": 150.0}, watchdog_ms={"depth": 300.0}
            ),
            id="one-model",
        ),
        pytest.param(
            replace(HELLO, watchdog_ms={"cls": 1.0, "depth": 1.0}),
            id="watchdog-limit-below-the-stale-threshold",
        ),
        pytest.param(replace(HELLO, models=("depth", "cls")), id="models-in-another-order"),
        pytest.param(
            replace(HELLO, stale_ms=DEFAULT_STALE_MS, watchdog_ms=DEFAULT_WATCHDOG_MS),
            id="default-thresholds",
        ),
        pytest.param(replace(HELLO, robot_id="Az09_.-" * 9 + "x"), id="robot-id-of-64-characters"),
        pytest.param(replace(HELLO, robot_id="7"), id="robot-id-of-one-character"),
        pytest.param(replace(HELLO, token=" ~secret token!"), id="token-of-printable-ascii"),
        pytest.param(replace(HELLO, token="t" * 60_000), id="token-near-the-text-bound"),
        pytest.param(
            replace(
                HELLO,
                stale_ms={"cls": MAX_THRESHOLD_MS, "depth": 5e-324},
                watchdog_ms={"cls": MAX_THRESHOLD_MS, "depth": 5e-324},
            ),
            id="thresholds-at-their-bounds",
        ),
        pytest.param(replace(HELLO, frame_rate_hz=MIN_FRAME_RATE_HZ), id="lowest-frame-rate"),
        pytest.param(replace(HELLO, frame_rate_hz=MAX_FRAME_RATE_HZ), id="highest-frame-rate"),
    ],
)
def test_hello_survives_the_wire(hello: Hello) -> None:
    assert decode_hello(encode_hello(hello)) == hello


def test_hello_reads_integer_thresholds_and_frame_rate_as_floats() -> None:
    hello = decode_hello(
        changed(HELLO_DOCUMENT, stale_ms={"cls": 500, "depth": 150}, frame_rate_hz=30)
    )

    assert hello == HELLO
    assert [type(value) for value in (*hello.stale_ms.values(), hello.frame_rate_hz)] == [
        float,
        float,
        float,
    ]


@pytest.mark.parametrize(
    ("text", "fault", "message"),
    [
        pytest.param("x" * (MAX_TEXT_CHARS + 1), Fault.BAD_HELLO, "longer than", id="too-long"),
        pytest.param(
            encode_hello(HELLO).ljust(MAX_TEXT_CHARS + 1),
            Fault.BAD_HELLO,
            "longer than",
            id="valid-hello-padded-past-the-bound",
        ),
        pytest.param("{", Fault.BAD_HELLO, "not JSON", id="not-json"),
        pytest.param("[" * MAX_TEXT_CHARS, Fault.BAD_HELLO, "not JSON", id="deeply-nested"),
        pytest.param(
            '{"type":"hello","protocol_version":' + "1" * 5000 + "}",
            Fault.BAD_HELLO,
            "not JSON",
            id="integer-of-5000-digits",
        ),
        pytest.param("[]", Fault.BAD_HELLO, "of type hello", id="not-an-object"),
        pytest.param(
            changed(HELLO_DOCUMENT, type="welcome"), Fault.BAD_HELLO, "of type hello", id="welcome"
        ),
        pytest.param(
            without(HELLO_DOCUMENT, "type"), Fault.BAD_HELLO, "of type hello", id="no-type"
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, type="welcome", protocol_version=2),
            Fault.BAD_HELLO,
            "of type hello",
            id="welcome-of-version-2",
        ),
        pytest.param(
            without(HELLO_DOCUMENT, "protocol_version"),
            Fault.BAD_HELLO,
            "protocol_version is missing",
            id="no-version",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version="1"),
            Fault.BAD_HELLO,
            "protocol_version is missing",
            id="version-string",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=True),
            Fault.BAD_HELLO,
            "protocol_version is missing",
            id="version-bool",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=1.0),
            Fault.BAD_HELLO,
            "protocol_version is missing",
            id="version-float",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=2),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-2",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=0),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-0",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=2, models=["cls", "pose"]),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-2-and-an-unknown-model",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, protocol_version=2, advised_fps=10),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-2-and-a-field-of-its-own",
        ),
        pytest.param(
            without(HELLO_DOCUMENT, "token"),
            Fault.BAD_HELLO,
            "exactly the fields",
            id="missing-field",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, advised_fps=10),
            Fault.BAD_HELLO,
            "exactly the fields",
            id="extra-field",
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, advised_fps=10, models=["pose"]),
            Fault.BAD_HELLO,
            "exactly the fields",
            id="extra-field-and-an-unknown-model",
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, robot_id=robot_id), Fault.BAD_HELLO, "robot_id", id=name
            )
            for name, robot_id in [
                ("robot-id-empty", ""),
                ("robot-id-of-65-characters", "a" * 65),
                ("robot-id-with-a-space", "robot 1"),
                ("robot-id-with-a-slash", "robot/1"),
                ("robot-id-ending-in-a-newline", "robot-1\n"),
                ("robot-id-not-ascii", "r\N{LATIN SMALL LETTER O WITH DIAERESIS}bot"),
                ("robot-id-with-a-fullwidth-digit", "robot\N{FULLWIDTH DIGIT ONE}"),
                ("robot-id-number", 7),
                ("robot-id-null", None),
            ]
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, robot_id="robot 1", models=["pose"]),
            Fault.BAD_HELLO,
            "robot_id",
            id="bad-robot-id-and-an-unknown-model",
        ),
        *(
            pytest.param(changed(HELLO_DOCUMENT, token=token), Fault.BAD_HELLO, "token", id=name)
            for name, token in [
                ("token-empty", ""),
                ("token-not-ascii", "s\N{LATIN SMALL LETTER E WITH ACUTE}cret"),
                ("token-with-a-tab", "se\tcret"),
                ("token-with-delete", "secret\x7f"),
                ("token-number", 7),
                ("token-list", ["secret"]),
            ]
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, token="", models=["pose"]),
            Fault.BAD_HELLO,
            "token",
            id="empty-token-and-an-unknown-model",
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, models=models), Fault.BAD_HELLO, "models is not", id=name
            )
            for name, models in [
                ("models-string", "cls"),
                ("models-empty", []),
                ("models-with-a-number", ["cls", 1]),
                ("models-with-a-list", [["cls"]]),
                ("models-repeated", ["cls", "cls"]),
                ("unknown-model-repeated", ["pose", "pose"]),
            ]
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, models=models),
                Fault.UNKNOWN_MODEL,
                "names a model other than",
                id=name,
            )
            for name, models in [
                ("unknown-model", ["pose"]),
                ("known-and-unknown-model", ["cls", "pose"]),
                ("model-key-in-capitals", ["CLS"]),
            ]
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, models=["pose"], **{name: value}),
                Fault.UNKNOWN_MODEL,
                "names a model other than",
                id=f"unknown-model-and-a-bad-{case}",
            )
            for case, name, value in [
                ("threshold", "stale_ms", []),
                ("frame-rate", "frame_rate_hz", 0),
                ("frame-size", "frame_width_px", 0),
            ]
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, **{name: table}),
                Fault.BAD_HELLO,
                f"{name} is not an object",
                id=f"{name}-{case}",
            )
            for name in ("stale_ms", "watchdog_ms")
            for case, table in [
                ("list", [500.0, 150.0]),
                ("missing-a-model", {"cls": 500.0}),
                ("with-a-model-more", {"cls": 500.0, "depth": 150.0, "pose": 1.0}),
            ]
        ),
        pytest.param(
            changed(HELLO_DOCUMENT, models=["depth"]),
            Fault.BAD_HELLO,
            "stale_ms is not an object",
            id="threshold-for-an-unsubscribed-model",
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, **{name: {"cls": value, "depth": 150.0}}),
                Fault.BAD_HELLO,
                f"{name} has a value",
                id=f"{name}-{case}",
            )
            for name in ("stale_ms", "watchdog_ms")
            for case, value in [
                ("zero", 0),
                ("negative", -1.0),
                ("above-the-bound", math.nextafter(MAX_THRESHOLD_MS, math.inf)),
                ("integer-beyond-float-range", 10**400),
                ("bool", True),
                ("string", "500"),
                ("null", None),
                ("nan", math.nan),
                ("infinity", math.inf),
            ]
        ),
        pytest.param(
            with_1e999(changed(HELLO_DOCUMENT, stale_ms={"cls": PLACEHOLDER, "depth": 150.0})),
            Fault.BAD_HELLO,
            "stale_ms has a value",
            id="stale_ms-1e999",
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, frame_rate_hz=rate_hz),
                Fault.BAD_HELLO,
                "frame_rate_hz",
                id=name,
            )
            for name, rate_hz in [
                ("rate-below-the-lowest", math.nextafter(MIN_FRAME_RATE_HZ, 0.0)),
                ("rate-above-the-highest", math.nextafter(MAX_FRAME_RATE_HZ, math.inf)),
                ("rate-zero", 0),
                ("rate-bool", True),
                ("rate-string", "30"),
                ("rate-nan", math.nan),
                ("rate-infinity", math.inf),
            ]
        ),
        pytest.param(
            with_1e999(changed(HELLO_DOCUMENT, frame_rate_hz=PLACEHOLDER)),
            Fault.BAD_HELLO,
            "frame_rate_hz",
            id="rate-1e999",
        ),
        *(
            pytest.param(
                changed(HELLO_DOCUMENT, frame_width_px=width_px, frame_height_px=height_px),
                Fault.BAD_HELLO,
                "frame size",
                id=name,
            )
            for name, width_px, height_px in [
                ("wider", FRAME_WIDTH_PX + 1, FRAME_HEIGHT_PX),
                ("taller", FRAME_WIDTH_PX, FRAME_HEIGHT_PX + 1),
                ("turned", FRAME_HEIGHT_PX, FRAME_WIDTH_PX),
                ("width-float", float(FRAME_WIDTH_PX), FRAME_HEIGHT_PX),
                ("height-string", FRAME_WIDTH_PX, str(FRAME_HEIGHT_PX)),
            ]
        ),
    ],
)
def test_hello_breaking_a_rule_is_refused_with_the_fault_of_the_first_rule(
    text: str, fault: Fault, message: str
) -> None:
    with pytest.raises(ProtocolError, match=message) as caught:
        decode_hello(text)

    assert caught.value.fault is fault


def test_welcome_is_compact_json() -> None:
    assert encode_welcome(WELCOME) == (
        '{"type":"welcome","protocol_version":1,"session":7,"shared_clock":true}'
    )


@pytest.mark.parametrize(
    "welcome",
    [
        pytest.param(WELCOME, id="shared-clock"),
        pytest.param(Welcome(PROTOCOL_VERSION, 1, False), id="first-session"),
        pytest.param(Welcome(PROTOCOL_VERSION, 2**32 - 1, True), id="last-session"),
    ],
)
def test_welcome_survives_the_wire(welcome: Welcome) -> None:
    assert decode_welcome(encode_welcome(welcome)) == welcome


@pytest.mark.parametrize(
    ("text", "fault", "message"),
    [
        pytest.param("x" * (MAX_TEXT_CHARS + 1), Fault.MALFORMED, "longer than", id="too-long"),
        pytest.param("{", Fault.MALFORMED, "not JSON", id="not-json"),
        pytest.param("[" * MAX_TEXT_CHARS, Fault.MALFORMED, "not JSON", id="deeply-nested"),
        pytest.param(
            '{"type":"welcome","session":' + "1" * 5000 + "}",
            Fault.MALFORMED,
            "not JSON",
            id="integer-of-5000-digits",
        ),
        pytest.param("[]", Fault.MALFORMED, "of type welcome", id="not-an-object"),
        pytest.param(
            changed(WELCOME_DOCUMENT, type="hello"), Fault.MALFORMED, "of type welcome", id="hello"
        ),
        pytest.param(
            without(WELCOME_DOCUMENT, "type"), Fault.MALFORMED, "of type welcome", id="no-type"
        ),
        pytest.param(
            without(WELCOME_DOCUMENT, "protocol_version"),
            Fault.MALFORMED,
            "protocol_version is missing",
            id="no-version",
        ),
        pytest.param(
            changed(WELCOME_DOCUMENT, protocol_version="1"),
            Fault.MALFORMED,
            "protocol_version is missing",
            id="version-string",
        ),
        pytest.param(
            changed(WELCOME_DOCUMENT, protocol_version=True),
            Fault.MALFORMED,
            "protocol_version is missing",
            id="version-bool",
        ),
        pytest.param(
            changed(WELCOME_DOCUMENT, protocol_version=2),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-2",
        ),
        pytest.param(
            changed(WELCOME_DOCUMENT, protocol_version=2, advised_fps=10),
            Fault.VERSION,
            "protocol version is not 1",
            id="version-2-and-a-field-of-its-own",
        ),
        pytest.param(
            without(WELCOME_DOCUMENT, "session"),
            Fault.MALFORMED,
            "exactly the fields",
            id="missing-field",
        ),
        pytest.param(
            changed(WELCOME_DOCUMENT, advised_fps=10),
            Fault.MALFORMED,
            "exactly the fields",
            id="advised-frame-rate",
        ),
        *(
            pytest.param(
                changed(WELCOME_DOCUMENT, session=session), Fault.MALFORMED, "session", id=name
            )
            for name, session in [
                ("session-zero", 0),
                ("session-negative", -1),
                ("session-beyond-u32", 2**32),
                ("session-bool", True),
                ("session-float", 7.0),
                ("session-string", "7"),
                ("session-null", None),
            ]
        ),
        *(
            pytest.param(
                changed(WELCOME_DOCUMENT, shared_clock=shared_clock),
                Fault.MALFORMED,
                "shared_clock",
                id=name,
            )
            for name, shared_clock in [
                ("shared-clock-integer", 1),
                ("shared-clock-string", "true"),
                ("shared-clock-null", None),
            ]
        ),
    ],
)
def test_welcome_breaking_a_rule_is_refused_with_its_fault(
    text: str, fault: Fault, message: str
) -> None:
    with pytest.raises(ProtocolError, match=message) as caught:
        decode_welcome(text)

    assert caught.value.fault is fault


def test_telemetry_dict_is_the_design_object_with_histograms_in_their_metrics_form() -> None:
    cls_entry = {
        "commands_received": 10,
        "commands_applied": 7,
        "commands_stale": 2,
        "commands_out_of_order": 1,
        "freshness_ms": CLS_TELEMETRY.freshness_ms.to_dict(),
        "command_gap_ms": CLS_TELEMETRY.command_gap_ms.to_dict(),
        "command_age_peak_ms": CLS_TELEMETRY.command_age_peak_ms.to_dict(),
        "deliver_ms": CLS_TELEMETRY.deliver_ms.to_dict(),
        "watchdog_trips": 1,
        "watchdog_expired_ms": 250,
        "watchdog_state": "ok",
    }

    assert telemetry_to_dict(TELEMETRY) == {
        "ts_mono_ns": 123_456_789,
        "frames_captured": 30,
        "frames_sent": 29,
        "frames_skipped_send": 1,
        "fire_lag_ms": TELEMETRY.fire_lag_ms.to_dict(),
        "reconnects": 0,
        "models": {"cls": cls_entry, "depth": {**cls_entry, "watchdog_state": "expired"}},
    }


def test_telemetry_text_is_its_dict_with_its_type_in_compact_json() -> None:
    expected = {"type": "telemetry", **telemetry_to_dict(TELEMETRY)}

    assert encode_telemetry(TELEMETRY) == json.dumps(expected, separators=(",", ":"))


def full_histogram(bucket_counts: list[int]) -> Histogram:
    """Return a histogram with these counts in its buckets, in index order.

    Its sum, min and max take the most characters a float is written with: a sign, 17
    significant digits and a three-digit exponent.
    """
    longest_float = -1.2345678901234567e-300
    return Histogram.from_dict(
        {
            "count": sum(bucket_counts),
            "sum": longest_float,
            "min": longest_float,
            "max": longest_float,
            "buckets": {str(index): count for index, count in enumerate(bucket_counts)},
        }
    )


def largest_report(bucket_counts: list[int]) -> Telemetry:
    """Return a report with both models, the longest counters and these bucket counts."""
    third = MAX_COUNTER // 3
    return Telemetry(
        ts_mono_ns=MAX_COUNTER,
        frames_captured=MAX_COUNTER,
        frames_sent=MAX_COUNTER,
        frames_skipped_send=MAX_COUNTER,
        fire_lag_ms=full_histogram(bucket_counts),
        reconnects=MAX_COUNTER,
        models={
            model: ModelTelemetry(
                commands_received=3 * third,
                commands_applied=third,
                commands_stale=third,
                commands_out_of_order=third,
                freshness_ms=full_histogram(bucket_counts),
                command_gap_ms=full_histogram(bucket_counts),
                command_age_peak_ms=full_histogram(bucket_counts),
                deliver_ms=full_histogram(bucket_counts),
                watchdog_trips=MAX_COUNTER,
                watchdog_expired_ms=MAX_COUNTER,
                watchdog_state="expired",
            )
            for model in MODEL_KEYS
        },
    )


# The bucket counts written with the most digits within the largest count a histogram may
# have. A count of d digits is at least 10**(d - 1), so each further digit of a count costs
# more: the most digits are 17 in every bucket and 18 in as many as still fit. The rest goes
# to a bucket of 18 digits, which it does not lengthen.
RAISED, LEFT_OVER = divmod(MAX_COUNTER - BUCKET_COUNT * 10**16, 10**17 - 10**16)
WIDEST_BUCKET_COUNTS = (
    [10**17 + LEFT_OVER] + [10**17] * (RAISED - 1) + [10**16] * (BUCKET_COUNT - RAISED)
)


@pytest.mark.parametrize(
    "telemetry",
    [
        pytest.param(TELEMETRY, id="both-models"),
        pytest.param(replace(TELEMETRY, models={"depth": CLS_TELEMETRY}), id="one-model"),
        pytest.param(replace(TELEMETRY, models={}), id="no-model"),
        pytest.param(
            Telemetry(
                ts_mono_ns=0,
                frames_captured=0,
                frames_sent=0,
                frames_skipped_send=0,
                fire_lag_ms=Histogram(),
                reconnects=0,
                models={
                    "cls": ModelTelemetry(
                        commands_received=0,
                        commands_applied=0,
                        commands_stale=0,
                        commands_out_of_order=0,
                        freshness_ms=Histogram(),
                        command_gap_ms=Histogram(),
                        command_age_peak_ms=Histogram(),
                        deliver_ms=Histogram(),
                        watchdog_trips=0,
                        watchdog_expired_ms=0,
                        watchdog_state="none",
                    )
                },
            ),
            id="before-the-first-command",
        ),
    ],
)
def test_telemetry_survives_the_wire(telemetry: Telemetry) -> None:
    assert decode_telemetry(encode_telemetry(telemetry)) == telemetry


@pytest.mark.parametrize(
    "bucket_counts",
    [
        pytest.param([10**12] * BUCKET_COUNT, id="13-digit-count-in-every-bucket"),
        pytest.param(WIDEST_BUCKET_COUNTS, id="widest-counts-a-histogram-allows"),
    ],
)
def test_largest_telemetry_report_fits_the_text_bound_and_survives_the_wire(
    bucket_counts: list[int],
) -> None:
    report = largest_report(bucket_counts)

    text = encode_telemetry(report)

    assert len(text) <= MAX_TEXT_CHARS
    assert decode_telemetry(text) == report


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param("x" * (MAX_TEXT_CHARS + 1), "longer than", id="too-long"),
        pytest.param("{", "not JSON", id="not-json"),
        pytest.param("[" * MAX_TEXT_CHARS, "not JSON", id="deeply-nested"),
        pytest.param(
            '{"type":"telemetry","ts_mono_ns":' + "1" * 5000 + "}",
            "not JSON",
            id="integer-of-5000-digits",
        ),
        pytest.param("[]", "of type telemetry", id="not-an-object"),
        pytest.param(changed(TELEMETRY_DOCUMENT, type="hello"), "of type telemetry", id="hello"),
        pytest.param(without(TELEMETRY_DOCUMENT, "type"), "of type telemetry", id="no-type"),
        pytest.param(
            without(TELEMETRY_DOCUMENT, "reconnects"), "exactly the fields", id="missing-field"
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, frames_dropped=0), "exactly the fields", id="extra-field"
        ),
        *(
            pytest.param(
                changed(TELEMETRY_DOCUMENT, **{name: -1}),
                f"{name} is not an integer",
                id=f"{name}-negative",
            )
            for name in TOP_COUNTERS
        ),
        # A negative verdict count also breaks the sum of the verdicts, whose message names
        # every verdict count, so each case matches the message of the counter rule itself.
        *(
            pytest.param(
                changed(TELEMETRY_DOCUMENT, models={"cls": {**CLS_DOCUMENT, name: -1}}),
                f"{name} is not an integer",
                id=f"{name}-negative",
            )
            for name in MODEL_COUNTERS
        ),
        *(
            pytest.param(
                changed(TELEMETRY_DOCUMENT, frames_sent=value),
                "frames_sent is not an integer",
                id=f"counter-{case}",
            )
            for case, value in [
                ("above-the-largest", MAX_COUNTER + 1),
                ("bool", True),
                ("float", 29.0),
                ("string", "29"),
                ("null", None),
                ("nan", math.nan),
                ("infinity", math.inf),
            ]
        ),
        pytest.param(
            with_1e999(changed(TELEMETRY_DOCUMENT, frames_sent=PLACEHOLDER)),
            "frames_sent is not an integer",
            id="counter-1e999",
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, models=[]), "models is not an object", id="models-list"
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, models={"pose": CLS_DOCUMENT}),
            "models is not an object",
            id="unknown-model",
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, models={"cls": 10}),
            "telemetry is not an object",
            id="model-entry-number",
        ),
        pytest.param(
            changed(
                TELEMETRY_DOCUMENT,
                models={
                    "cls": {
                        key: value for key, value in CLS_DOCUMENT.items() if key != "deliver_ms"
                    }
                },
            ),
            "exactly the fields",
            id="model-entry-missing-field",
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, models={"cls": {**CLS_DOCUMENT, "commands_late": 0}}),
            "exactly the fields",
            id="model-entry-extra-field",
        ),
        *(
            pytest.param(
                changed(
                    TELEMETRY_DOCUMENT,
                    models={"cls": {**CLS_DOCUMENT, "commands_received": received}},
                ),
                "commands_received is not commands_applied",
                id=f"received-{case}-the-sum-of-the-verdicts",
            )
            for case, received in [("above", 11), ("below", 9)]
        ),
        *(
            pytest.param(
                changed(
                    TELEMETRY_DOCUMENT, models={"cls": {**CLS_DOCUMENT, "watchdog_state": state}}
                ),
                "watchdog_state",
                id=f"watchdog-state-{case}",
            )
            for case, state in [
                ("unknown", "tripped"),
                ("in-capitals", "OK"),
                ("number", 1),
                ("null", None),
                ("list", ["ok"]),
            ]
        ),
        pytest.param(
            changed(TELEMETRY_DOCUMENT, fire_lag_ms=[]), "fire_lag_ms", id="histogram-list"
        ),
        pytest.param(
            changed(
                TELEMETRY_DOCUMENT,
                models={"cls": {**CLS_DOCUMENT, "freshness_ms": {**FIRE_LAG_DOCUMENT, "count": 3}}},
            ),
            "freshness_ms",
            id="histogram-count-not-its-buckets",
        ),
        pytest.param(
            changed(
                TELEMETRY_DOCUMENT,
                models={
                    "cls": {
                        **CLS_DOCUMENT,
                        "deliver_ms": {**FIRE_LAG_DOCUMENT, "buckets": {"007": 2}},
                    }
                },
            ),
            "deliver_ms",
            id="histogram-bucket-key-with-leading-zeros",
        ),
        *(
            pytest.param(
                changed(TELEMETRY_DOCUMENT, fire_lag_ms={**FIRE_LAG_DOCUMENT, field: value}),
                "fire_lag_ms",
                id=f"histogram-{field}-{case}",
            )
            for field, case, value in [
                ("sum", "nan", math.nan),
                ("min", "infinity", math.inf),
                ("max", "negative-infinity", -math.inf),
            ]
        ),
        pytest.param(
            with_1e999(
                changed(TELEMETRY_DOCUMENT, fire_lag_ms={**FIRE_LAG_DOCUMENT, "max": PLACEHOLDER})
            ),
            "fire_lag_ms",
            id="histogram-max-1e999",
        ),
    ],
)
def test_telemetry_breaking_a_rule_is_refused_as_malformed(text: str, message: str) -> None:
    with pytest.raises(ProtocolError, match=message) as caught:
        decode_telemetry(text)

    assert caught.value.fault is Fault.MALFORMED


def test_frame_header_is_little_endian_in_the_documented_field_order() -> None:
    header = FrameHeader(
        session=0x01020304, frame_number=0x0102030405060708, capture_ns=-2, send_ns=3
    )

    assert encode_frame(header, b"\xff\xd8") == bytes.fromhex(
        "01 00 0000 04030201 0807060504030201 feffffffffffffff 0300000000000000 ffd8"
    )


def test_command_header_is_little_endian_in_the_documented_field_order() -> None:
    header = CommandHeader(
        model="depth",
        session=0x01020304,
        frame_number=0x0102030405060708,
        capture_ns=-2,
        batch_size=0x0102,
        result_ns=3,
        stages=StageDurations(
            transport_up_us=1,
            slot_wait_us=2,
            dispatch_rt_us=3,
            decode_us=4,
            preprocess_us=5,
            batch_peers_us=6,
            inference_us=7,
            postprocess_us=8,
        ),
    )

    assert encode_command_message(header, DEPTH_BODY) == (
        bytes.fromhex(
            "02 01 0201 04030201 0807060504030201 feffffffffffffff 0300000000000000"
            " 01000000 02000000 03000000 04000000 05000000 06000000 07000000 08000000"
        )
        + DEPTH_BODY
    )


def test_headers_are_32_and_64_bytes() -> None:
    frame_header = encode_frame(FRAME_HEADER, b"")
    command_header = encode_command_message(COMMAND_HEADER, b"")

    assert (len(frame_header), len(command_header)) == (32, 64)


def test_stage_durations_follow_the_wire_order_of_stage_names() -> None:
    assert [field.name for field in fields(StageDurations)] == [
        f"{name}_us" for name in STAGE_NAMES
    ]


@pytest.mark.parametrize(
    ("header", "jpeg"),
    [
        pytest.param(FRAME_HEADER, JPEG, id="typical"),
        pytest.param(
            FrameHeader(
                session=2**32 - 1, frame_number=2**64 - 1, capture_ns=-(2**63), send_ns=2**63 - 1
            ),
            b"\xff\xd8",
            id="extremes-and-the-shortest-jpeg",
        ),
        pytest.param(
            FrameHeader(session=0, frame_number=0, capture_ns=0, send_ns=0),
            b"\xff\xd8" + bytes(100_000),
            id="zeros-and-a-large-jpeg",
        ),
    ],
)
def test_frame_survives_the_wire(header: FrameHeader, jpeg: bytes) -> None:
    decoded, view = decode_frame(encode_frame(header, jpeg))

    assert (decoded, bytes(view)) == (header, jpeg)


def test_decoded_jpeg_is_a_view_into_the_message() -> None:
    _, jpeg = decode_frame(FRAME)

    assert jpeg.obj is FRAME
    assert jpeg == JPEG


@pytest.mark.parametrize(
    ("message", "error"),
    [
        pytest.param(b"", "shorter than its header", id="empty"),
        pytest.param(FRAME[:31], "shorter than its header", id="inside-the-header"),
        pytest.param(FRAME[:32], "shorter than its header", id="header-alone"),
        pytest.param(FRAME[:33], "shorter than its header", id="one-jpeg-byte"),
        pytest.param(overwritten(FRAME, 0, b"\x00"), "kind 0 is not a frame", id="kind-0"),
        pytest.param(overwritten(FRAME, 0, b"\x02"), "kind 2 is not a frame", id="command-kind"),
        pytest.param(overwritten(FRAME, 0, b"\xff"), "kind 255 is not a frame", id="kind-255"),
        pytest.param(overwritten(FRAME, 1, b"\x01"), "reserved", id="reserved-byte"),
        pytest.param(overwritten(FRAME, 2, b"\x01"), "reserved", id="reserved-word-low-byte"),
        pytest.param(overwritten(FRAME, 3, b"\x80"), "reserved", id="reserved-word-high-byte"),
        pytest.param(overwritten(FRAME, 32, b"\xff\xd9"), "start-of-image", id="end-of-image"),
        pytest.param(overwritten(FRAME, 32, b"\x00"), "start-of-image", id="first-marker-byte"),
        pytest.param(overwritten(FRAME, 33, b"\x00"), "start-of-image", id="second-marker-byte"),
    ],
)
def test_frame_breaking_a_rule_is_refused_as_malformed(message: bytes, error: str) -> None:
    with pytest.raises(ProtocolError, match=error) as caught:
        decode_frame(message)

    assert caught.value.fault is Fault.MALFORMED


@pytest.mark.parametrize(
    ("header", "command"),
    [
        pytest.param(COMMAND_HEADER, CLS_COMMAND, id="cls"),
        pytest.param(replace(COMMAND_HEADER, model="depth"), DEPTH_COMMAND, id="depth"),
        pytest.param(
            CommandHeader(
                model="cls",
                session=2**32 - 1,
                frame_number=2**64 - 1,
                capture_ns=-(2**63),
                batch_size=2**16 - 1,
                result_ns=2**63 - 1,
                stages=StageDurations(*[2**32 - 1] * len(STAGE_NAMES)),
            ),
            ClsCommand(class_id=2**16 - 1, confidence=1.0),
            id="extremes",
        ),
    ],
)
def test_command_survives_the_wire(header: CommandHeader, command: Command) -> None:
    message = encode_command_message(header, encode_command(command))

    assert decode_command_message(message) == (header, command)


@pytest.mark.parametrize(
    ("data", "error"),
    [
        pytest.param(b"", "shorter than its header", id="empty"),
        pytest.param(COMMAND[:63], "shorter than its header", id="inside-the-header"),
        pytest.param(overwritten(COMMAND, 0, b"\x00"), "kind 0 is not a command", id="kind-0"),
        pytest.param(overwritten(COMMAND, 0, b"\x01"), "kind 1 is not a command", id="frame-kind"),
        pytest.param(overwritten(COMMAND, 0, b"\x03"), "kind 3 is not a command", id="kind-3"),
        pytest.param(overwritten(COMMAND, 1, b"\x02"), "model id 2", id="model-id-past-the-keys"),
        pytest.param(overwritten(COMMAND, 1, b"\xff"), "model id 255", id="model-id-255"),
        pytest.param(overwritten(COMMAND, 2, b"\x00\x00"), "batch of 0 frames", id="batch-of-0"),
        pytest.param(COMMAND[:64], "cls command body of 0 bytes", id="no-body"),
        pytest.param(COMMAND[:-1], "cls command body of 5 bytes", id="body-cut-short"),
        pytest.param(COMMAND + b"\x00", "cls command body of 7 bytes", id="body-with-a-byte-more"),
        pytest.param(
            encode_command_message(COMMAND_HEADER, DEPTH_BODY),
            "cls command body of 20 bytes",
            id="depth-body-for-cls",
        ),
        pytest.param(
            encode_command_message(replace(COMMAND_HEADER, model="depth"), CLS_BODY),
            "depth command body of 6 bytes",
            id="cls-body-for-depth",
        ),
    ],
)
def test_command_breaking_a_rule_is_refused_as_malformed(data: bytes, error: str) -> None:
    with pytest.raises(ProtocolError, match=error) as caught:
        decode_command_message(data)

    assert caught.value.fault is Fault.MALFORMED


@pytest.mark.parametrize(
    ("encode", "error_type"),
    [
        pytest.param(
            lambda: encode_frame(replace(FRAME_HEADER, session=2**32), JPEG),
            struct.error,
            id="session-beyond-u32",
        ),
        pytest.param(
            lambda: encode_command_message(
                replace(COMMAND_HEADER, stages=replace(STAGES, slot_wait_us=2**32)), CLS_BODY
            ),
            struct.error,
            id="stage-of-71-minutes-or-more",
        ),
        pytest.param(
            lambda: encode_hello(replace(HELLO, frame_rate_hz=math.nan)),
            ValueError,
            id="nan-in-json",
        ),
    ],
)
def test_encoder_raises_for_a_value_its_message_cannot_carry(
    encode: Callable[[], object], error_type: type[Exception]
) -> None:
    with pytest.raises(error_type):
        encode()
