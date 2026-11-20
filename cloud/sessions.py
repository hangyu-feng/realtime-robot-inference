"""The gateway's open sessions and every synchronous step that changes them.

The steps are a robot's hello, with its admission and the replacement of an older session of
the same robot id; the ingress step of one message; and the end of a session. This is pure
logic: no asyncio, no sockets, no clock reading. Time arrives as now_ns arguments, the I/O
layer injects the callables wake and on_membership_change, and it calls each step as one
synchronous call with no await inside it, which is what the caller rules of
cloud.registry.RobotRegistry require.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from cloud.admission import IngressLimiter, RejectReason, admission_refusal, frame_header_valid
from cloud.config import ServerConfig
from cloud.counters import HANDSHAKE_TIMEOUT, OVERSIZE, RATE_LIMITED, CloseReason, ConnectionTotals
from cloud.downlink import DownlinkQueue
from cloud.mailbox import Frame, Mailbox
from cloud.registry import RobotRecord, RobotRegistry
from common.protocol import (
    PROTOCOL_VERSION,
    CloseCode,
    Fault,
    Hello,
    ProtocolError,
    Welcome,
    decode_frame,
    decode_hello,
    decode_telemetry,
    encode_welcome,
    telemetry_to_dict,
)

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
# The WebSocket library fails a connection with 1007 for a text message that is not UTF-8.
# CloseCode lists only the codes this service sends itself.
_INVALID_DATA_CODE = 1007

# The close code that ends each kind of session, or None where the connection is aborted.
CLOSE_CODES: Mapping[CloseReason, CloseCode | None] = MappingProxyType(
    {
        # The library has already answered the peer's close frame.
        CloseReason.CLIENT_CLOSE: None,
        # The peer went without a close frame.
        CloseReason.ABNORMAL: None,
        CloseReason.IDLE_TIMEOUT: CloseCode.IDLE_TIMEOUT,
        # A slow consumer is ended only by the writer's slow-consumer timeout, which can fire
        # only while the library's send waits for its buffer to drain, and with a write limit
        # of 0 that wait happens only while bytes are unsent. A close frame would queue behind
        # bytes the robot is not reading, so the connection is aborted instead, and
        # CloseCode.SLOW_CONSUMER stays reserved in the protocol without being sent.
        CloseReason.SLOW_CONSUMER: None,
        CloseReason.INGRESS_LIMIT: CloseCode.POLICY,
        # After 1009 or 1007 the library has already sent its own close frame.
        CloseReason.PROTOCOL_ERROR: CloseCode.PROTOCOL_ERROR,
        CloseReason.REPLACED: CloseCode.SESSION_REPLACED,
        CloseReason.SHUTDOWN: CloseCode.SERVER_SHUTDOWN,
        # The state of the connection is unknown after a fault in the gateway's own code.
        CloseReason.INTERNAL_ERROR: None,
    }
)

# The close code that answers a refused hello, by its label in connections.rejected.
_REFUSAL_CODES: Mapping[str, CloseCode] = MappingProxyType(
    {
        # A first message that is not a well-formed hello breaks the protocol.
        Fault.BAD_HELLO: CloseCode.PROTOCOL_ERROR,
        Fault.VERSION: CloseCode.POLICY,
        Fault.UNKNOWN_MODEL: CloseCode.POLICY,
        RejectReason.AUTH: CloseCode.POLICY,
        RejectReason.CAPACITY: CloseCode.OVER_CAPACITY,
        # No hello within the handshake timeout is an idle timeout of the handshake, and the
        # robot must retry it: a policy code would stop it for good.
        HANDSHAKE_TIMEOUT: CloseCode.IDLE_TIMEOUT,
    }
)


@dataclass(eq=False, slots=True)
class Session:
    """One accepted connection of a robot, from its welcome until it ends.

    number is the session number of the welcome. wake wakes the session's handler; it is
    called when the session ends and when commands were queued for it. Equality is identity:
    two sessions of one robot id are different sessions.
    """

    robot_id: str
    number: int
    record: RobotRecord
    downlink: DownlinkQueue
    limiter: IngressLimiter
    wake: Callable[[], None]
    last_frame_number: int | None = None
    end_reason: CloseReason | None = None

    @property
    def ended(self) -> bool:
        return self.end_reason is not None


@dataclass(frozen=True, slots=True)
class Opened:
    """An accepted hello: the new session and its encoded welcome."""

    session: Session
    welcome: str


@dataclass(frozen=True, slots=True)
class Refused:
    """A refused hello: its label in connections.rejected and the code that closes it."""

    reason: str
    code: CloseCode


class SessionTable:
    """The open sessions by robot id, and the steps that open, feed and end them.

    sessions[robot_id] exists exactly while registry.get(robot_id) is connected, and it holds
    the session whose number is the record's session. Both change only inside hello and end,
    each with no await, so a session that has not ended is its robot's current session: that
    is the check the caller rules of cloud.registry.RobotRegistry ask for before a frame is put
    or a session is closed.
    """

    def __init__(
        self,
        config: ServerConfig,
        registry: RobotRegistry,
        mailbox: Mailbox,
        totals: ConnectionTotals,
        on_membership_change: Callable[[], None],
    ) -> None:
        self._config = config
        self._registry = registry
        self._mailbox = mailbox
        self._totals = totals
        self._on_membership_change = on_membership_change
        self._sessions: dict[str, Session] = {}
        self._sessions_view = MappingProxyType(self._sessions)
        self._max_frame_age_ns = round(config.max_frame_age_ms * _NS_PER_MS)

    @property
    def sessions(self) -> Mapping[str, Session]:
        """Return the open sessions by robot id, a live read-only view."""
        return self._sessions_view

    def hello(
        self, message: str | bytes, wake: Callable[[], None], now_ns: int
    ) -> Opened | Refused:
        """Open a session for a connection's first message, or refuse it.

        wake becomes the new session's wake. Admission runs before any eviction, so a refused
        hello leaves an open session of the same robot id untouched. Every refusal counts in
        connections.rejected.
        """
        if isinstance(message, bytes):
            return self._refuse(Fault.BAD_HELLO.value)
        try:
            hello = decode_hello(message)
        except ProtocolError as error:
            return self._refuse(error.fault.value)
        replaced = self._sessions.get(hello.robot_id)
        refusal = admission_refusal(
            hello.token,
            shared_token=self._config.shared_token,
            connected=self._registry.connected_count() - (0 if replaced is None else 1),
            max_connections=self._config.max_connections,
        )
        if refusal is not None:
            return self._refuse(refusal.value)
        if replaced is not None:
            # ASSUMPTION: A22. Robot ids are unique, so a second connection with the same id
            # replaces the first session, which ends before the new one opens.
            self.end(replaced, CloseReason.REPLACED, now_ns)
        return self._open(hello, len(message), wake, now_ns)

    def refuse_handshake_timeout(self) -> Refused:
        """Refuse a connection that sent no hello within the handshake timeout."""
        return self._refuse(HANDSHAKE_TIMEOUT)

    def ingest(
        self, session: Session, message: str | bytes, received_ns: int, rtt_ns: int | None
    ) -> bool:
        """Run the ingress step of one message; return whether a frame went into the mailbox.

        received_ns is the one gateway clock reading taken when the message arrived, and rtt_ns
        the connection's last ping round trip. When a frame went in, the caller pokes the
        dispatchers of the models the robot subscribes to.
        """
        # A connection may still return a message the library had queued before its session
        # ended. It counts nothing, and a replaced session's frame never reaches the mailbox
        # entry of the session that replaced it.
        if session.ended:
            return False
        # A text message counts its characters, which are its bytes for valid telemetry: JSON
        # escapes every other character, and the decoder refuses what it does not know.
        session.record.bytes_in += len(message)
        self._totals.bytes_in += len(message)
        if isinstance(message, bytes):
            return self._ingest_frame(session, message, received_ns, rtt_ns)
        self._ingest_text(session, message, received_ns, rtt_ns)
        return False

    def lost(
        self, session: Session, peer_code: int | None, local_code: int | None, now_ns: int
    ) -> None:
        """End a session whose connection has closed, with the reason its close codes give.

        peer_code is the code of the close frame the peer sent first, and local_code the code
        this side sent first, by this program or by the library. Neither is set when the
        connection ended without a close frame either way.
        """
        if peer_code is not None:
            reason = CloseReason.CLIENT_CLOSE
        elif local_code in (CloseCode.PROTOCOL_ERROR, _INVALID_DATA_CODE):
            reason = CloseReason.PROTOCOL_ERROR
        elif local_code == CloseCode.MESSAGE_TOO_BIG:
            reason = CloseReason.PROTOCOL_ERROR
            if not session.ended:
                # The library refused the message before its type was known, so it counts as
                # a frame, rejected as oversize.
                session.record.frames_in += 1
                session.record.frames_rejected += 1
                self._totals.frames_rejected[OVERSIZE] += 1
        else:
            # No close frame either way, as when the peer was killed, or 1011 from the
            # library's keepalive.
            reason = CloseReason.ABNORMAL
        self.end(session, reason, now_ns)

    def end(self, session: Session, reason: CloseReason, now_ns: int) -> None:
        """End the session once: the first reason wins, and a later call returns at once.

        The close sequence of cloud.registry.RobotRegistry counts every frame still pending and
        every command still queued. The session's handler is woken last, and it closes the
        connection. Every accepted session ends here exactly once, with one reason, so
        connections.accepted = connected_count + sum(connections.closed).
        """
        if session.ended:
            return
        session.end_reason = reason
        self._mailbox.close(session.robot_id)
        session.downlink.drop_all()
        self._registry.disconnect(session.robot_id, now_ns)
        del self._sessions[session.robot_id]
        self._totals.closed[reason] += 1
        self._on_membership_change()
        session.wake()

    def _refuse(self, reason: str) -> Refused:
        self._totals.rejected[reason] += 1
        return Refused(reason, _REFUSAL_CODES[reason])

    def _open(
        self, hello: Hello, hello_chars: int, wake: Callable[[], None], now_ns: int
    ) -> Opened:
        """Run the open sequence of cloud.registry.RobotRegistry for an admitted hello."""
        record = self._registry.connect(
            hello.robot_id,
            hello.models,
            {model: round(stale_ms * _NS_PER_MS) for model, stale_ms in hello.stale_ms.items()},
            {
                model: round(watchdog_ms * _NS_PER_MS)
                for model, watchdog_ms in hello.watchdog_ms.items()
            },
            round(_NS_PER_S / hello.frame_rate_hz),
            now_ns,
        )
        self._mailbox.open(record)
        config = self._config
        session = Session(
            robot_id=hello.robot_id,
            number=record.session,
            record=record,
            downlink=DownlinkQueue(config.downlink_queue_commands),
            limiter=IngressLimiter(
                config.ingress_max_messages_per_s,
                config.ingress_max_bytes_per_s,
                config.ingress_burst_s,
                now_ns,
            ),
            wake=wake,
        )
        self._sessions[hello.robot_id] = session
        self._totals.accepted += 1
        # The hello is a message received from the robot, counted as text is.
        record.bytes_in += hello_chars
        self._totals.bytes_in += hello_chars
        self._on_membership_change()
        return Opened(
            session, encode_welcome(Welcome(PROTOCOL_VERSION, record.session, config.shared_clock))
        )

    def _ingest_frame(
        self, session: Session, message: bytes, received_ns: int, rtt_ns: int | None
    ) -> bool:
        record = session.record
        # Counted before anything can refuse the frame, so that
        # frames_in = frames_rejected + frames_accepted holds after every message.
        record.frames_in += 1
        if not session.limiter.admit(len(message), received_ns):
            record.frames_rejected += 1
            self._totals.frames_rejected[RATE_LIMITED] += 1
            self.end(session, CloseReason.INGRESS_LIMIT, received_ns)
            return False
        try:
            header, jpeg = decode_frame(message)
        except ProtocolError:
            valid = False
        else:
            valid = frame_header_valid(
                header.session,
                header.frame_number,
                header.capture_ns,
                header.send_ns,
                received_ns,
                connection_session=session.number,
                last_frame_number=session.last_frame_number,
                shared_clock=self._config.shared_clock,
                max_age_ns=self._max_frame_age_ns,
            )
        if not valid:
            # One bad frame fails alone (DESIGN.md 1.4): each header is checked on its own, so
            # it says nothing about the next frame, and the ingress limit bounds what a stream
            # of bad frames can cost.
            record.frames_rejected += 1
            self._totals.frames_rejected[Fault.MALFORMED.value] += 1
            return False
        if self._config.shared_clock:
            # ASSUMPTION: A20. Robot and gateway read one monotonic clock, so the uplink is
            # measured, not estimated.
            transport_up_ns = received_ns - header.send_ns
        else:
            # Half the last ping round trip, and 0 until the session's first one returns,
            # which errs toward serving in the skip rule.
            transport_up_ns = 0 if rtt_ns is None else rtt_ns // 2
        # The JPEG stays a view into the message: nothing is copied and nothing is decoded.
        frame = Frame(
            header.frame_number,
            header.capture_ns,
            received_ns,
            header.send_ns - header.capture_ns,
            transport_up_ns,
            jpeg,
        )
        self._mailbox.put(session.robot_id, frame)
        record.frames_accepted += 1
        session.last_frame_number = header.frame_number
        return True

    def _ingest_text(
        self, session: Session, message: str, received_ns: int, rtt_ns: int | None
    ) -> None:
        # Metered before it is decoded, so a report is never parsed unmetered.
        if not session.limiter.admit(len(message), received_ns):
            self.end(session, CloseReason.INGRESS_LIMIT, received_ns)
            return
        try:
            telemetry = decode_telemetry(message)
        except ProtocolError:
            # A robot that sends malformed text, a second hello included, does not speak the
            # protocol.
            self.end(session, CloseReason.PROTOCOL_ERROR, received_ns)
            return
        # ASSUMPTION: A10. The stale count reaches /metrics through this report. It is kept as
        # the object /metrics publishes, so a scrape copies it and serializes no histogram.
        session.record.telemetry = telemetry_to_dict(telemetry)
        # /metrics shows the round trip as it was when the last report arrived (DESIGN.md 10.3).
        session.record.rtt_ns = rtt_ns
