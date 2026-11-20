"""Who may connect, how much a connection may send, and whether a frame fits its connection.

Pure logic on plain values: the I/O layer passes a hello's token and the count of open sessions,
the size and the one gateway clock reading of every message, and the fields of every decoded
frame header, and acts on the answers.
"""

from __future__ import annotations

import hmac
from dataclasses import dataclass
from enum import StrEnum

_NS_PER_S = 1_000_000_000


class RejectReason(StrEnum):
    """A refusal of a handshake that this module decides, as a reason of connections.rejected.

    The other refusals of a handshake come from common/protocol.py (Fault: version, bad_hello,
    unknown_model) and from the handshake timeout of the I/O layer.
    """

    AUTH = "auth"
    CAPACITY = "capacity"


def admission_refusal(
    token: str | None, *, shared_token: str | None, connected: int, max_connections: int
) -> RejectReason | None:
    """Return why a hello is refused, or None to admit it. The token is checked before the cap.

    connected is the number of open sessions, not counting the session this robot id would
    replace.
    """
    # ASSUMPTION: A21. Real authentication is out of scope. The shared token is an optional hook,
    # and with shared_token None no token is checked.
    # Checked first, the token keeps a client without it from learning whether the gateway is
    # full. Both strings are printable ASCII (decode_hello and cloud/config.py check it), and
    # compare_digest keeps the comparison time from telling how much of the token matched.
    if shared_token is not None and (
        token is None
        or not hmac.compare_digest(token.encode("ascii"), shared_token.encode("ascii"))
    ):
        return RejectReason.AUTH
    # ASSUMPTION: A22. A hello for a connected robot id replaces that session, which connected
    # leaves out, so a reconnect is never refused for capacity.
    # No rule refuses robots to protect a minimum rate per robot: a refused robot is a starved
    # robot (DESIGN.md 8.1). The cap is a safety valve, and fair rotation is the main mechanism.
    if connected >= max_connections:
        return RejectReason.CAPACITY
    return None


@dataclass(slots=True)
class _TokenBucket:
    """A token bucket whose level is a whole number of units, units_per_token of them to a token.

    The level is kept in exact integer arithmetic, never in accumulated floats, so the bound a
    bucket sets (its capacity plus its rate over any interval) holds exactly, and a sender at the
    rate is never refused by a rounding error. A float setting is an exact binary fraction,
    numerator / denominator, and with rate_denominator * burst_denominator * 10**9 units to a
    token both the refill of one nanosecond and the capacity are whole numbers of units.
    """

    units_per_token: int
    refill_units_per_ns: int
    capacity_units: int
    level_units: int

    @classmethod
    def full(cls, rate_per_s: float, burst_s: float) -> _TokenBucket:
        """Return a bucket that holds burst_s seconds of its rate and starts full."""
        rate_numerator, rate_denominator = rate_per_s.as_integer_ratio()
        burst_numerator, burst_denominator = burst_s.as_integer_ratio()
        capacity_units = rate_numerator * burst_numerator * _NS_PER_S
        return cls(
            units_per_token=rate_denominator * burst_denominator * _NS_PER_S,
            refill_units_per_ns=rate_numerator * burst_denominator,
            capacity_units=capacity_units,
            level_units=capacity_units,
        )

    def refill(self, elapsed_ns: int) -> None:
        self.level_units = min(
            self.capacity_units, self.level_units + self.refill_units_per_ns * elapsed_ns
        )

    def holds(self, tokens: int) -> bool:
        return self.level_units >= tokens * self.units_per_token

    def take(self, tokens: int) -> None:
        self.level_units -= tokens * self.units_per_token


class IngressLimiter:
    """The ingress limit of one connection: a token bucket of messages and one of bytes.

    Each bucket holds at most burst_s seconds of its rate and starts full. The caller passes
    every message the connection receives, text or binary, through admit before it decodes it,
    so a telemetry report is never parsed unmetered. After a refusal it closes the connection
    with the policy close code, 1008 (closed reason ingress_limit), and a refused frame also
    counts in frames_rejected as rate_limited.
    """

    # The limit protects the event loop's time and the gateway's memory, which every other
    # robot's frames wait behind. It is not what makes the inference share fair: the mailbox
    # is, where under the latest-wins policy a robot's newer frame replaces its pending one, so
    # with the skip rule off a robot that sends faster within the limit gains no share
    # (DESIGN.md 6.1).

    def __init__(
        self, max_messages_per_s: float, max_bytes_per_s: float, burst_s: float, now_ns: int
    ) -> None:
        self._messages = _TokenBucket.full(max_messages_per_s, burst_s)
        self._bytes = _TokenBucket.full(max_bytes_per_s, burst_s)
        self._last_ns = now_ns

    def admit(self, message_bytes: int, now_ns: int) -> bool:
        """Take one message and message_bytes bytes, or nothing when either bucket holds less.

        Both buckets are first refilled for the time since the previous call, or since the
        limiter was made before the first. ValueError when now_ns is before that time.
        """
        if now_ns < self._last_ns:
            raise ValueError(
                f"now_ns={now_ns} is before the previous call at {self._last_ns}: "
                "a monotonic clock never goes back"
            )
        elapsed_ns = now_ns - self._last_ns
        self._last_ns = now_ns
        self._messages.refill(elapsed_ns)
        self._bytes.refill(elapsed_ns)
        if not (self._messages.holds(1) and self._bytes.holds(message_bytes)):
            return False
        self._messages.take(1)
        self._bytes.take(message_bytes)
        return True


def frame_header_valid(
    session: int,
    frame_number: int,
    capture_ns: int,
    send_ns: int,
    received_ns: int,
    *,
    connection_session: int,
    last_frame_number: int | None,
    shared_clock: bool,
    max_age_ns: int,
) -> bool:
    """Check a decoded frame header against its connection, so that everything after can trust it.

    The arguments are the header fields as plain integers. connection_session is the session
    number of this connection's welcome, and last_frame_number the number of the last frame this
    connection accepted, None before the first. received_ns is the one gateway clock reading
    taken when the message arrived: the same value the caller stores in Frame.received_ns and
    subtracts send_ns from for the uplink. Checked against a later reading, a send_ns stamped
    just after the stored receive time would pass and give a negative uplink, which no stage
    duration can carry.

    This checks the frame against its own connection only. That the connection still holds the
    robot's current session is the caller's check against the registry, in the same step and
    before it puts the frame (the docstring of cloud.registry.RobotRegistry states it), so a
    frame that a replaced connection had already read never reaches the mailbox.

    A frame that fails is not put in the mailbox and counts in frames_rejected as malformed, the
    label of a frame the decoder refuses. Whether the session also closes the connection is the
    I/O layer's decision.
    """
    # Messages carry the session number (DESIGN.md 1.4).
    if session != connection_session:
        return False
    # Within a session frame numbers strictly increase. That is the precondition of
    # Mailbox.put, so a model never serves a frame number twice or out of order (DESIGN.md 5.1,
    # 6.1). An honest robot's frames meet it on a transport that keeps their order, as
    # WebSocket over TCP does.
    if last_frame_number is not None and frame_number <= last_frame_number:
        return False
    # ASSUMPTION: A6. capture_ns is the frame's scheduled tick, and a robot sends a frame only
    # once its tick is due, so an honest fire lag, send_ns - capture_ns, is never negative.
    if not 0 <= send_ns - capture_ns <= max_age_ns:
        return False
    # ASSUMPTION: A20. On a shared clock robot and gateway read one monotonic clock, so the
    # robot's timestamps can be compared with received_ns. Without one they never are
    # (DESIGN.md 5.4).
    if not shared_clock:
        return True
    return send_ns <= received_ns and received_ns - capture_ns <= max_age_ns
