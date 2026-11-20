"""The wire protocol between a robot and the gateway, version 1.

Bytes and text from a robot or from the gateway become trusted types here and nowhere else:
every decoder checks all of its input once and raises ProtocolError, and no other exception,
for anything it refuses. Every encoder is given trusted values, so a value that does not fit
is a programming error and raises.

Text messages are JSON objects with a "type" field and exactly the fields of their dataclass:
hello and telemetry go from robot to cloud, welcome from cloud to robot. They are written as
compact JSON without NaN or infinity.

Binary messages start with a kind byte, and their integers are little-endian. Timestamps are
monotonic nanoseconds, which an i64 holds for over 292 years; they are signed so that the
difference of two readings stays in the type.

frame, kind 1, robot to cloud: a header of struct "<BBHIQqq", 32 bytes, then the JPEG.

    byte  field         type  why this width
    0     kind          u8    one byte names the message: two binary kinds exist
    1     reserved      u8    0; with the u16 after it, puts session at byte 4 and the three
    2     reserved      u16   8-byte fields at bytes 8, 16 and 24, so that every field is
                              naturally aligned in 32 bytes; a decoder refuses any other
                              value, which keeps both free for a later protocol version
    4     session       u32   the registry's connect counter, one step per connect: 2**32 - 1
                              connects take over 49 days at one connect per millisecond
    8     frame_number  u64   a tick counter, which cannot wrap at MAX_FRAME_RATE_HZ
    16    capture_ns    i64   the scheduled capture time, on the robot's clock
    24    send_ns       i64   the robot's clock just before the send

command, kind 2, cloud to robot: a header of struct "<BBHIQqq8I", 64 bytes, then the command
body of common/commands.py, whose length the model fixes.

    byte  field         type     why this width
    0     kind          u8       as in the frame header
    1     model         u8       an index into MODEL_KEYS
    2     batch_size    u16      holds MAX_BATCH_FRAMES (common/worker_protocol.py), and with
                                 the two bytes before it puts session at byte 4, so that the
                                 fields are aligned as in the frame header
    4     session       u32      echoed from the frame, with the frame header's width
    8     frame_number  u64      echoed from the frame, with the frame header's width
    16    capture_ns    i64      echoed from the frame, with the frame header's width
    24    result_ns     i64      the gateway's clock when it had the result
    32    stages        8 x u32  the server's stage durations in whole microseconds, in the
                                 order of STAGE_NAMES

A u32 of microseconds holds about 71 minutes. Under latest-wins every stage is bounded by a
timeout or by one frame period plus a service time; under the FIFO control, slot_wait is
bounded by the queue length over the model's rate, which stays far below that in the one-robot
control run. The gateway computes the durations, so a value that does not fit is a programming
error there, and the encoder raises.
"""

from __future__ import annotations

import json
import re
import struct
from collections.abc import Mapping
from dataclasses import dataclass, fields
from enum import IntEnum, StrEnum
from typing import TypeGuard

from common.commands import MODEL_KEYS, Command, CommandFormatError, decode_command
from common.histogram import RANGE_MAX_MS, Histogram

# The version a hello and a welcome carry. A hello of another version is refused before its
# other fields are read, because they may differ between versions.
PROTOCOL_VERSION = 1

# The one definition of the frame size, because the service serves one resolution: a robot
# declares it in its hello, and a worker refuses a JPEG whose header declares another.
FRAME_WIDTH_PX = 640
FRAME_HEIGHT_PX = 480

# A robot id is a key of /metrics and part of file names, so it is 1 to 64 ASCII letters,
# digits, '_', '-' and '.'. Match it with fullmatch: match would also take a prefix.
ROBOT_ID_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")

# Every text decoder refuses a longer text before it parses it. The bound is sized for the
# largest telemetry report, nine histograms with every bucket non-empty. As encode_telemetry
# writes that report, it takes about 34 000 characters with 13-digit bucket counts and under
# 41 000 with the widest counts a histogram allows, so the bound refuses no report that
# encode_telemetry writes and the other rules accept. That matters because the histograms
# are cumulative within a session, so a report only grows.
MAX_TEXT_CHARS = 65_536

# The top of the histogram range: a robot cannot observe an age above it in its own
# histograms.
MAX_THRESHOLD_MS = RANGE_MAX_MS

# At the top, twice the demo rate: with the 1 Hz telemetry a robot then sends under half the
# default ingress message rate of cloud/config.py (128 per second). At the bottom, one frame
# per second bounds the frame period at one second, which keeps it inside every integer
# conversion.
MIN_FRAME_RATE_HZ = 1.0
MAX_FRAME_RATE_HZ = 60.0

# The largest counter a report may carry: a signed 64-bit integer, which every reader of
# /metrics can hold.
MAX_COUNTER = 2**63 - 1

# The server's stages, in the order of the command header (DESIGN.md 7).
STAGE_NAMES: tuple[str, ...] = (
    "transport_up",
    "slot_wait",
    "dispatch_rt",
    "decode",
    "preprocess",
    "batch_peers",
    "inference",
    "postprocess",
)

_FRAME_KIND = 1
_COMMAND_KIND = 2
_FRAME_HEADER = struct.Struct("<BBHIQqq")
_COMMAND_HEADER = struct.Struct("<BBHIQqq8I")
_JPEG_START_OF_IMAGE = b"\xff\xd8"
# A session number travels as a u32 in every frame header, and the registry's first is 1.
_MAX_SESSION = 2**32 - 1
# The values of robot.control.WatchdogState, which this module cannot import: common/ does
# not depend on robot/.
_WATCHDOG_STATES = frozenset(("none", "ok", "expired"))
# Strict and compact: a NaN or an infinity is a programming error and raises, because every
# decoder here refuses it.
_JSON_ENCODER = json.JSONEncoder(allow_nan=False, separators=(",", ":"))


class CloseCode(IntEnum):
    """The WebSocket close codes of the service (DESIGN.md 4.3).

    RFC 6455 leaves 4000 to 4999 to applications.
    """

    NORMAL = 1000
    SERVER_SHUTDOWN = 1001
    PROTOCOL_ERROR = 1002
    POLICY = 1008  # a bad version, a bad token, the ingress limit
    MESSAGE_TOO_BIG = 1009
    OVER_CAPACITY = 1013  # over the connection cap
    SESSION_REPLACED = 4001  # a hello with the same robot id opened a new session
    SLOW_CONSUMER = 4002
    IDLE_TIMEOUT = 4003


class Fault(StrEnum):
    """Why a decoder refused a message, as a /metrics label.

    version, bad_hello and unknown_model are reasons of connections.rejected, and malformed,
    for a frame, is a reason of frames_rejected.
    """

    MALFORMED = "malformed"
    VERSION = "version"
    BAD_HELLO = "bad_hello"
    UNKNOWN_MODEL = "unknown_model"


class ProtocolError(Exception):
    """A message from a peer breaks the protocol."""

    def __init__(self, fault: Fault, message: str) -> None:
        super().__init__(message)
        self.fault = fault


@dataclass(frozen=True, slots=True)
class Hello:
    """A robot's first message: who it is, what it subscribes to and what it declares.

    stale_ms and watchdog_ms hold one threshold per subscribed model. The frame size is the
    one the robot sends, which must be the one this service serves.
    """

    protocol_version: int
    # ASSUMPTION: A22. Robot ids are unique: a hello with the id of a connected robot replaces
    # that robot's session.
    robot_id: str
    # ASSUMPTION: A21. Real authentication is out of scope: the token is checked only against
    # an optional shared token.
    token: str | None
    # ASSUMPTION: A2. Every frame is offered to each model the robot subscribes to here, which
    # is both models unless the robot names one alone.
    models: tuple[str, ...]
    stale_ms: Mapping[str, float]
    watchdog_ms: Mapping[str, float]
    frame_rate_hz: float
    frame_width_px: int
    frame_height_px: int


@dataclass(frozen=True, slots=True)
class Welcome:
    """The gateway's answer to an accepted hello.

    Every frame of the session carries session. shared_clock says whether robot and gateway
    read one clock, so that timestamps of both can be compared (DESIGN.md 5.4).
    """

    protocol_version: int
    session: int
    shared_clock: bool


@dataclass(frozen=True, slots=True)
class ModelTelemetry:
    """One model's part of a telemetry report (DESIGN.md 10.4).

    watchdog_state is "none", "ok" or "expired".
    """

    commands_received: int
    commands_applied: int
    commands_stale: int
    commands_out_of_order: int
    freshness_ms: Histogram
    command_gap_ms: Histogram
    command_age_peak_ms: Histogram
    deliver_ms: Histogram
    watchdog_trips: int
    watchdog_expired_ms: int
    watchdog_state: str


# ASSUMPTION: A10. The stale count reaches /metrics through this report, which the gateway
# republishes under robots.<id>.telemetry.
@dataclass(frozen=True, slots=True)
class Telemetry:
    """A robot's 1 Hz report, whose JSON field names are those of DESIGN.md 10.4."""

    ts_mono_ns: int
    frames_captured: int
    frames_sent: int
    frames_skipped_send: int
    fire_lag_ms: Histogram
    reconnects: int
    models: Mapping[str, ModelTelemetry]


# ASSUMPTION: A4. The frame number and the capture time travel in this header, not inside
# the JPEG.
# ASSUMPTION: A6. capture_ns is the scheduled tick time, not the time of the send, so a late
# simulator cannot hide latency: send_ns - capture_ns is the fire lag.
@dataclass(frozen=True, slots=True)
class FrameHeader:
    """The header of a frame message. Both times are on the robot's monotonic clock."""

    session: int
    frame_number: int
    capture_ns: int
    send_ns: int


@dataclass(frozen=True, slots=True)
class StageDurations:
    """The server's stage durations for one command, in whole microseconds.

    A command carries durations, not timestamps: the robot adds them to its own two stages,
    fire lag and deliver, and compares the sum with its end-to-end time, without comparing
    the clocks of different machines (DESIGN.md 5.4 and 7).
    """

    transport_up_us: int
    slot_wait_us: int
    dispatch_rt_us: int
    decode_us: int
    preprocess_us: int
    batch_peers_us: int
    inference_us: int
    postprocess_us: int


@dataclass(frozen=True, slots=True)
class CommandHeader:
    """The header of a command message.

    session, frame_number and capture_ns are echoed from the frame the command was computed
    from. result_ns is the gateway's clock when it had the result.
    """

    model: str
    session: int
    frame_number: int
    # ASSUMPTION: A7. Freshness is the robot's receive time minus this capture time, both on
    # the robot's clock, so no two clocks need to agree.
    capture_ns: int
    batch_size: int
    result_ns: int
    stages: StageDurations


# The JSON keys of a text message are its type and the fields of its dataclass.
_HELLO_FIELDS = frozenset(("type", *(field.name for field in fields(Hello))))
_WELCOME_FIELDS = frozenset(("type", *(field.name for field in fields(Welcome))))
_TELEMETRY_FIELDS = frozenset(("type", *(field.name for field in fields(Telemetry))))
_MODEL_TELEMETRY_FIELDS = frozenset(field.name for field in fields(ModelTelemetry))


def encode_hello(hello: Hello) -> str:
    return _JSON_ENCODER.encode(
        {
            "type": "hello",
            "protocol_version": hello.protocol_version,
            "robot_id": hello.robot_id,
            "token": hello.token,
            "models": list(hello.models),
            "stale_ms": dict(hello.stale_ms),
            "watchdog_ms": dict(hello.watchdog_ms),
            "frame_rate_hz": hello.frame_rate_hz,
            "frame_width_px": hello.frame_width_px,
            "frame_height_px": hello.frame_height_px,
        }
    )


def decode_hello(text: str) -> Hello:
    """Read a robot's hello, with the fault of the first rule it breaks.

    The rules are checked in this order, so that the fault is the most specific one: the
    message itself; protocol_version, missing or not an integer (BAD_HELLO) or another version
    (VERSION); the set of fields; the robot id and the token; the models, where a model this
    service does not serve is UNKNOWN_MODEL; the thresholds; the frame rate and size. Every
    other fault is BAD_HELLO.
    """
    document = _read_object(text, "hello", Fault.BAD_HELLO)
    _check_version(document, Fault.BAD_HELLO)
    _check_fields(document, _HELLO_FIELDS, Fault.BAD_HELLO, "hello")
    robot_id = document["robot_id"]
    if not isinstance(robot_id, str) or ROBOT_ID_PATTERN.fullmatch(robot_id) is None:
        raise ProtocolError(
            Fault.BAD_HELLO, "robot_id is not 1 to 64 ASCII letters, digits, '_', '-' or '.'"
        )
    token = document["token"]
    # The server compares the token with the shared token as ASCII bytes. MAX_TEXT_CHARS
    # bounds its length, and cloud/config.py sets no other bound on the shared token, so no
    # length rule here may refuse one.
    if not (
        token is None
        or (isinstance(token, str) and token != "" and token.isascii() and token.isprintable())
    ):
        raise ProtocolError(
            Fault.BAD_HELLO, "token is neither null nor a non-empty printable ASCII string"
        )
    models = document["models"]
    # Every element is known to be a string before the set is built: a list is unhashable.
    if (
        not isinstance(models, list)
        or not models
        or not all(isinstance(model, str) for model in models)
        or len(set(models)) != len(models)
    ):
        raise ProtocolError(Fault.BAD_HELLO, "models is not a non-empty list of distinct strings")
    if not all(model in MODEL_KEYS for model in models):
        raise ProtocolError(
            Fault.UNKNOWN_MODEL, f"models names a model other than {', '.join(MODEL_KEYS)}"
        )
    subscribed = tuple(models)
    # A watchdog limit below the stale threshold is not refused here: the server only stores
    # and echoes the limit, and the robot's own watchdog refuses that combination.
    stale_ms = _read_thresholds(document["stale_ms"], subscribed, "stale_ms")
    watchdog_ms = _read_thresholds(document["watchdog_ms"], subscribed, "watchdog_ms")
    frame_rate_hz = document["frame_rate_hz"]
    # A comparison with NaN is false, and the bounds refuse infinity.
    if not _is_number(frame_rate_hz) or not (
        MIN_FRAME_RATE_HZ <= frame_rate_hz <= MAX_FRAME_RATE_HZ
    ):
        raise ProtocolError(
            Fault.BAD_HELLO,
            f"frame_rate_hz is not a number from {MIN_FRAME_RATE_HZ} to {MAX_FRAME_RATE_HZ}",
        )
    width_px = document["frame_width_px"]
    height_px = document["frame_height_px"]
    if (
        type(width_px) is not int
        or type(height_px) is not int
        or (width_px, height_px) != (FRAME_WIDTH_PX, FRAME_HEIGHT_PX)
    ):
        raise ProtocolError(
            Fault.BAD_HELLO, f"the frame size is not {FRAME_WIDTH_PX}x{FRAME_HEIGHT_PX} pixels"
        )
    return Hello(
        protocol_version=PROTOCOL_VERSION,
        robot_id=robot_id,
        token=token,
        models=subscribed,
        stale_ms=stale_ms,
        watchdog_ms=watchdog_ms,
        frame_rate_hz=float(frame_rate_hz),
        frame_width_px=FRAME_WIDTH_PX,
        frame_height_px=FRAME_HEIGHT_PX,
    )


def encode_welcome(welcome: Welcome) -> str:
    return _JSON_ENCODER.encode(
        {
            "type": "welcome",
            "protocol_version": welcome.protocol_version,
            "session": welcome.session,
            "shared_clock": welcome.shared_clock,
        }
    )


def decode_welcome(text: str) -> Welcome:
    """Read the gateway's welcome: VERSION for another protocol version, else MALFORMED."""
    document = _read_object(text, "welcome", Fault.MALFORMED)
    _check_version(document, Fault.MALFORMED)
    _check_fields(document, _WELCOME_FIELDS, Fault.MALFORMED, "welcome")
    session = document["session"]
    if type(session) is not int or not 1 <= session <= _MAX_SESSION:
        raise ProtocolError(Fault.MALFORMED, f"session is not an integer from 1 to {_MAX_SESSION}")
    shared_clock = document["shared_clock"]
    if type(shared_clock) is not bool:
        raise ProtocolError(Fault.MALFORMED, "shared_clock is not a boolean")
    return Welcome(PROTOCOL_VERSION, session, shared_clock)


def telemetry_to_dict(telemetry: Telemetry) -> dict[str, object]:
    """Return the report as the object of DESIGN.md 10.4, without its type field.

    The gateway stores this once per report, so a /metrics scrape copies it and never
    serializes a histogram.
    """
    return {
        "ts_mono_ns": telemetry.ts_mono_ns,
        "frames_captured": telemetry.frames_captured,
        "frames_sent": telemetry.frames_sent,
        "frames_skipped_send": telemetry.frames_skipped_send,
        "fire_lag_ms": telemetry.fire_lag_ms.to_dict(),
        "reconnects": telemetry.reconnects,
        "models": {
            model: {
                "commands_received": entry.commands_received,
                "commands_applied": entry.commands_applied,
                "commands_stale": entry.commands_stale,
                "commands_out_of_order": entry.commands_out_of_order,
                "freshness_ms": entry.freshness_ms.to_dict(),
                "command_gap_ms": entry.command_gap_ms.to_dict(),
                "command_age_peak_ms": entry.command_age_peak_ms.to_dict(),
                "deliver_ms": entry.deliver_ms.to_dict(),
                "watchdog_trips": entry.watchdog_trips,
                "watchdog_expired_ms": entry.watchdog_expired_ms,
                "watchdog_state": entry.watchdog_state,
            }
            for model, entry in telemetry.models.items()
        },
    }


def encode_telemetry(telemetry: Telemetry) -> str:
    return _JSON_ENCODER.encode({"type": "telemetry", **telemetry_to_dict(telemetry)})


def decode_telemetry(text: str) -> Telemetry:
    """Read a robot's telemetry report; MALFORMED for anything wrong.

    What this accepts is republished in /metrics as strict JSON: every integer is a JSON
    integer from 0 to MAX_COUNTER, every histogram passes Histogram.from_dict, and each
    model's commands_received is its applied, stale and out-of-order commands together
    (DESIGN.md 10.4).
    """
    document = _read_object(text, "telemetry", Fault.MALFORMED)
    _check_fields(document, _TELEMETRY_FIELDS, Fault.MALFORMED, "telemetry")
    entries = document["models"]
    if not isinstance(entries, dict) or not entries.keys() <= set(MODEL_KEYS):
        raise ProtocolError(
            Fault.MALFORMED, f"telemetry models is not an object keyed by {', '.join(MODEL_KEYS)}"
        )
    models: dict[str, ModelTelemetry] = {}
    for model, entry in entries.items():
        if not isinstance(entry, dict):
            raise ProtocolError(Fault.MALFORMED, "a model's telemetry is not an object")
        _check_fields(entry, _MODEL_TELEMETRY_FIELDS, Fault.MALFORMED, "a model's telemetry")
        watchdog_state = entry["watchdog_state"]
        if not isinstance(watchdog_state, str) or watchdog_state not in _WATCHDOG_STATES:
            raise ProtocolError(
                Fault.MALFORMED,
                f"watchdog_state is not one of {', '.join(sorted(_WATCHDOG_STATES))}",
            )
        model_telemetry = ModelTelemetry(
            commands_received=_read_counter(entry, "commands_received"),
            commands_applied=_read_counter(entry, "commands_applied"),
            commands_stale=_read_counter(entry, "commands_stale"),
            commands_out_of_order=_read_counter(entry, "commands_out_of_order"),
            freshness_ms=_read_histogram(entry, "freshness_ms"),
            command_gap_ms=_read_histogram(entry, "command_gap_ms"),
            command_age_peak_ms=_read_histogram(entry, "command_age_peak_ms"),
            deliver_ms=_read_histogram(entry, "deliver_ms"),
            watchdog_trips=_read_counter(entry, "watchdog_trips"),
            watchdog_expired_ms=_read_counter(entry, "watchdog_expired_ms"),
            watchdog_state=watchdog_state,
        )
        if model_telemetry.commands_received != (
            model_telemetry.commands_applied
            + model_telemetry.commands_stale
            + model_telemetry.commands_out_of_order
        ):
            raise ProtocolError(
                Fault.MALFORMED,
                "commands_received is not "
                "commands_applied + commands_stale + commands_out_of_order",
            )
        models[model] = model_telemetry
    return Telemetry(
        ts_mono_ns=_read_counter(document, "ts_mono_ns"),
        frames_captured=_read_counter(document, "frames_captured"),
        frames_sent=_read_counter(document, "frames_sent"),
        frames_skipped_send=_read_counter(document, "frames_skipped_send"),
        fire_lag_ms=_read_histogram(document, "fire_lag_ms"),
        reconnects=_read_counter(document, "reconnects"),
        models=models,
    )


def encode_frame(header: FrameHeader, jpeg: bytes) -> bytes:
    return (
        _FRAME_HEADER.pack(
            _FRAME_KIND,
            0,
            0,
            header.session,
            header.frame_number,
            header.capture_ns,
            header.send_ns,
        )
        + jpeg
    )


def decode_frame(message: bytes) -> tuple[FrameHeader, memoryview]:
    """Read a frame message. The JPEG is a view into the message, not a copy.

    The JPEG check is a prefilter only. The worker walks the JPEG header and refuses any frame
    that does not declare exactly FRAME_WIDTH_PX x FRAME_HEIGHT_PX before it decodes
    (workers/runtime.py), and decoding happens there, after the scheduler has selected the
    frame. How the header fields relate to the connection (the session, the frame order and
    the timestamps) is checked by cloud/admission.py, which has the connection's state.
    """
    if len(message) < _FRAME_HEADER.size + len(_JPEG_START_OF_IMAGE):
        raise ProtocolError(
            Fault.MALFORMED,
            f"frame message of {len(message)} bytes is shorter than its header and a JPEG marker",
        )
    kind, reserved_byte, reserved_word, session, frame_number, capture_ns, send_ns = (
        _FRAME_HEADER.unpack_from(message)
    )
    if kind != _FRAME_KIND:
        raise ProtocolError(Fault.MALFORMED, f"message kind {kind} is not a frame")
    if reserved_byte or reserved_word:
        raise ProtocolError(Fault.MALFORMED, "a reserved field of the frame header is not 0")
    if not message.startswith(_JPEG_START_OF_IMAGE, _FRAME_HEADER.size):
        raise ProtocolError(
            Fault.MALFORMED, "frame payload does not start with the JPEG start-of-image marker"
        )
    header = FrameHeader(session, frame_number, capture_ns, send_ns)
    return header, memoryview(message)[_FRAME_HEADER.size :]


def encode_command_message(header: CommandHeader, body: bytes) -> bytes:
    """Return the message for a command body as the worker encoded it.

    The gateway passes the worker's body on unchanged: it never decodes and re-encodes a
    command.
    """
    stages = header.stages
    return (
        _COMMAND_HEADER.pack(
            _COMMAND_KIND,
            MODEL_KEYS.index(header.model),
            header.batch_size,
            header.session,
            header.frame_number,
            header.capture_ns,
            header.result_ns,
            stages.transport_up_us,
            stages.slot_wait_us,
            stages.dispatch_rt_us,
            stages.decode_us,
            stages.preprocess_us,
            stages.batch_peers_us,
            stages.inference_us,
            stages.postprocess_us,
        )
        + body
    )


def decode_command_message(data: bytes) -> tuple[CommandHeader, Command]:
    """Read a command message on the robot. The body must have its model's exact length."""
    try:
        kind, model_id, batch_size, session, frame_number, capture_ns, result_ns, *stages_us = (
            _COMMAND_HEADER.unpack_from(data)
        )
    except struct.error as error:
        raise ProtocolError(
            Fault.MALFORMED, f"command message of {len(data)} bytes is shorter than its header"
        ) from error
    if kind != _COMMAND_KIND:
        raise ProtocolError(Fault.MALFORMED, f"message kind {kind} is not a command")
    if model_id >= len(MODEL_KEYS):
        raise ProtocolError(Fault.MALFORMED, f"model id {model_id} is not an index of MODEL_KEYS")
    if batch_size == 0:
        raise ProtocolError(Fault.MALFORMED, "command of a batch of 0 frames")
    model = MODEL_KEYS[model_id]
    try:
        command = decode_command(model, data[_COMMAND_HEADER.size :])
    except CommandFormatError as error:
        raise ProtocolError(Fault.MALFORMED, str(error)) from error
    header = CommandHeader(
        model=model,
        session=session,
        frame_number=frame_number,
        capture_ns=capture_ns,
        batch_size=batch_size,
        result_ns=result_ns,
        stages=StageDurations(*stages_us),
    )
    return header, command


def _read_object(text: str, message_type: str, fault: Fault) -> dict[str, object]:
    """Parse a text message and check that its type field is message_type."""
    if len(text) > MAX_TEXT_CHARS:
        raise ProtocolError(
            fault, f"text message of {len(text)} characters is longer than {MAX_TEXT_CHARS}"
        )
    try:
        # JSONDecodeError is a ValueError. A plain ValueError comes from an integer of more
        # than 4300 digits, and RecursionError from deeply nested brackets.
        document = json.loads(text)
    except (ValueError, RecursionError) as error:
        raise ProtocolError(fault, f"{message_type} message is not JSON: {error}") from error
    # No message here or below quotes a value from the input, which can be as long as the
    # text itself.
    if not isinstance(document, dict) or document.get("type") != message_type:
        raise ProtocolError(fault, f"message is not a JSON object of type {message_type}")
    return document


def _check_version(document: Mapping[str, object], fault: Fault) -> None:
    """Refuse another protocol version with VERSION, and a missing or malformed one with fault.

    It is checked before the other fields, which may differ between versions.
    """
    version = document.get("protocol_version")
    if type(version) is not int:
        raise ProtocolError(fault, "protocol_version is missing or not an integer")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(Fault.VERSION, f"protocol version is not {PROTOCOL_VERSION}")


def _check_fields(
    document: Mapping[str, object], names: frozenset[str], fault: Fault, what: str
) -> None:
    if document.keys() != names:
        raise ProtocolError(
            fault, f"{what} does not have exactly the fields {', '.join(sorted(names))}"
        )


def _is_number(value: object) -> TypeGuard[int | float]:
    """Tell whether a JSON value is a number, which a bool never is."""
    return isinstance(value, int | float) and not isinstance(value, bool)


def _read_thresholds(table: object, models: tuple[str, ...], name: str) -> dict[str, float]:
    """Read a hello's threshold for each subscribed model."""
    if not isinstance(table, dict) or table.keys() != set(models):
        raise ProtocolError(
            Fault.BAD_HELLO, f"{name} is not an object keyed by exactly the subscribed models"
        )
    thresholds: dict[str, float] = {}
    for model in models:
        threshold_ms = table[model]
        # A comparison with NaN is false, and the upper bound refuses infinity.
        if not _is_number(threshold_ms) or not 0 < threshold_ms <= MAX_THRESHOLD_MS:
            raise ProtocolError(
                Fault.BAD_HELLO, f"{name} has a value not above 0 and at most {MAX_THRESHOLD_MS}"
            )
        thresholds[model] = float(threshold_ms)
    return thresholds


def _read_counter(document: Mapping[str, object], name: str) -> int:
    value = document[name]
    if type(value) is not int or not 0 <= value <= MAX_COUNTER:
        raise ProtocolError(Fault.MALFORMED, f"{name} is not an integer from 0 to {MAX_COUNTER}")
    return value


def _read_histogram(document: Mapping[str, object], name: str) -> Histogram:
    data = document[name]
    try:
        return Histogram.from_dict(data)
    except ValueError as error:
        raise ProtocolError(Fault.MALFORMED, f"{name}: {error}") from error
