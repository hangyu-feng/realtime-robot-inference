"""One robot session, from its welcome to its end: ticks, commands and telemetry.

The session decides what each tick sends, applies the skip and stall rules, judges every
command and keeps the counters and histograms of the telemetry report. The caller reads the
clock once per event and passes the reading; nothing here does I/O.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from common.histogram import Histogram
from common.protocol import (
    Fault,
    FrameHeader,
    ModelTelemetry,
    ProtocolError,
    Telemetry,
    Welcome,
    decode_command_message,
    encode_frame,
    encode_telemetry,
)
from robot.control import ModelControl
from robot.samples import SampleLog, TickOutcome
from robot.schedule import CaptureSchedule
from robot.settings import RobotSettings

_NS_PER_MS = 1_000_000
# The telemetry report is sent at 1 Hz (DESIGN.md 10.4).
_TELEMETRY_PERIOD_NS = 1_000_000_000


@dataclass(frozen=True, slots=True)
class _ModelHistograms:
    """One model's duration histograms of the telemetry report."""

    freshness_ms: Histogram = field(default_factory=Histogram)
    command_gap_ms: Histogram = field(default_factory=Histogram)
    command_age_peak_ms: Histogram = field(default_factory=Histogram)
    deliver_ms: Histogram = field(default_factory=Histogram)


class RobotSession:
    """One session of a robot: its frames out, its commands in, and its report.

    start_ns is the robot clock when the welcome arrived. Counters and histograms start at
    zero with the session. The frame numbers come from the robot's one capture schedule, so
    they go on rising from one session to the next.
    """

    def __init__(
        self,
        robot_id: str,
        settings: RobotSettings,
        welcome: Welcome,
        schedule: CaptureSchedule,
        jpegs: Sequence[bytes],
        start_ns: int,
        samples: SampleLog | None,
    ) -> None:
        self.robot_id = robot_id
        self.number = welcome.session
        self._shared_clock = welcome.shared_clock
        self._schedule = schedule
        self._jpegs = jpegs
        self._samples = samples
        self._stall_ns = round(settings.stall_timeout_ms * _NS_PER_MS)
        self._controls = {
            model: ModelControl(
                welcome.session,
                round(settings.stale_ms[model] * _NS_PER_MS),
                round(settings.watchdog_ms[model] * _NS_PER_MS),
                start_ns,
            )
            for model in settings.models
        }
        self._histograms = {model: _ModelHistograms() for model in settings.models}
        self._fire_lag_ms = Histogram()
        self._frames_captured = 0
        self._frames_sent = 0
        self._frames_skipped_send = 0
        # A new session never sends a tick that was due before it began.
        self._next_frame = schedule.latest_due(start_ns) + 1
        self._telemetry_due_ns = start_ns + _TELEMETRY_PERIOD_NS
        # The time of the first of the ticks skipped in a row because the path was busy.
        self._busy_since_ns: int | None = None

    @property
    def next_frame(self) -> int:
        """The frame number of the next tick this session has not yet handled."""
        return self._next_frame

    @property
    def controls(self) -> Mapping[str, ModelControl]:
        """Each subscribed model's command state, read-only."""
        return MappingProxyType(self._controls)

    def tick(self, now_ns: int, path_busy: bool) -> bytes | None:
        """Handle every tick due at now_ns; return the frame message to send, if any.

        Only the newest due tick can be sent. It is skipped when path_busy says that the
        previous frame has not left yet.
        """
        due = self._schedule.latest_due(now_ns)
        if due < self._next_frame:
            return None
        # The ticks before the newest due one were passed while the loop was late. They
        # count as captured and skipped, and a frame number skipped is never sent late.
        for passed in range(self._next_frame, due):
            self._frames_captured += 1
            self._frames_skipped_send += 1
            self._sample_tick(passed, 0, TickOutcome.MISSED_LATE)
        self._next_frame = due + 1
        self._frames_captured += 1
        if path_busy:
            self._frames_skipped_send += 1
            self._sample_tick(due, 0, TickOutcome.SKIPPED_UNSENT)
            if self._busy_since_ns is None:
                self._busy_since_ns = now_ns
            return None
        self._busy_since_ns = None
        self._frames_sent += 1
        # ASSUMPTION: A6. The capture time is the scheduled tick, not this send, so a late
        # simulator cannot hide latency. The difference is the fire lag.
        capture_ns = self._schedule.capture_ns(due)
        self._fire_lag_ms.record((now_ns - capture_ns) / 1e6)
        self._sample_tick(due, now_ns, TickOutcome.SENT)
        # ASSUMPTION: A4. The frame number and the capture time travel in the header, not
        # inside the JPEG, so the pre-encoded frames can be cycled.
        header = FrameHeader(self.number, due, capture_ns, now_ns)
        return encode_frame(header, self._jpegs[due % len(self._jpegs)])

    def stalled(self, now_ns: int) -> bool:
        """Tell whether the path has been busy at every tick for the stall timeout or longer."""
        return self._busy_since_ns is not None and now_ns - self._busy_since_ns >= self._stall_ns

    def telemetry_due(self, now_ns: int) -> bool:
        return now_ns >= self._telemetry_due_ns

    def telemetry(self, now_ns: int, reconnects: int) -> str:
        """Return the report of DESIGN.md 10.4 as of now_ns, and set the next one's due time."""
        models = {
            model: ModelTelemetry(
                commands_received=control.received,
                commands_applied=control.applied,
                commands_stale=control.stale,
                commands_out_of_order=control.out_of_order,
                freshness_ms=self._histograms[model].freshness_ms,
                command_gap_ms=self._histograms[model].command_gap_ms,
                command_age_peak_ms=self._histograms[model].command_age_peak_ms,
                deliver_ms=self._histograms[model].deliver_ms,
                watchdog_trips=control.watchdog_trips(now_ns),
                watchdog_expired_ms=control.watchdog_expired_ns(now_ns) // _NS_PER_MS,
                watchdog_state=control.watchdog_state(now_ns).value,
            )
            for model, control in self._controls.items()
        }
        # Whole periods, to the first due time after now: a late report does not move the
        # later ones, and a report sent before its due time leaves that time as it is.
        periods = (now_ns - self._telemetry_due_ns) // _TELEMETRY_PERIOD_NS + 1
        self._telemetry_due_ns += periods * _TELEMETRY_PERIOD_NS
        return encode_telemetry(
            Telemetry(
                ts_mono_ns=now_ns,
                frames_captured=self._frames_captured,
                frames_sent=self._frames_sent,
                frames_skipped_send=self._frames_skipped_send,
                fire_lag_ms=self._fire_lag_ms,
                reconnects=reconnects,
                models=models,
            )
        )

    def receive(self, message: str | bytes, receive_ns: int) -> None:
        """Judge a command received at receive_ns, and record it.

        ProtocolError for anything else: after the welcome the gateway sends only commands
        of the subscribed models.
        """
        if isinstance(message, str):
            raise ProtocolError(Fault.MALFORMED, "a text message after the welcome")
        # ASSUMPTION: A3. The robot acts on a command by making it the command in effect,
        # which is the state its model's ModelControl keeps. The command's content changes
        # nothing in the simulation, so it is decoded, checked and dropped.
        header, _ = decode_command_message(message)
        control = self._controls.get(header.model)
        if control is None:
            raise ProtocolError(
                Fault.MALFORMED,
                f"a command of {header.model}, which this session did not subscribe to",
            )
        # ASSUMPTION: A7. Freshness is this receive time minus the capture time the command
        # echoes, both on the robot's clock.
        outcome = control.on_command(
            header.session, header.frame_number, header.capture_ns, receive_ns
        )
        # Each duration is the quotient of its nanoseconds by 1e6, which is correctly
        # rounded, so the same division over the samples gives the same float.
        histograms = self._histograms[header.model]
        histograms.freshness_ms.record(outcome.freshness_ns / 1e6)
        if outcome.gap_ns is not None:
            histograms.command_gap_ms.record(outcome.gap_ns / 1e6)
        if outcome.replaced_age_ns is not None:
            histograms.command_age_peak_ms.record(outcome.replaced_age_ns / 1e6)
        # result_ns is the gateway's clock, so the difference means something only when the
        # welcome said that both read one clock.
        if self._shared_clock:
            histograms.deliver_ms.record((receive_ns - header.result_ns) / 1e6)
        if self._samples is not None:
            self._samples.command(self.number, header, receive_ns, outcome.verdict)

    def _sample_tick(self, frame_number: int, send_ns: int, outcome: TickOutcome) -> None:
        if self._samples is not None:
            self._samples.tick(
                frame_number, self._schedule.capture_ns(frame_number), send_ns, outcome
            )
