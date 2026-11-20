"""What a robot does with one model's command stream: the stale and order rules, and the watchdog.

The caller passes the robot clock's reading with every command and every query. The
watchdog is computed exactly from the event times and that reading, with no periodic tick.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Verdict(StrEnum):
    """What the robot did with a received command."""

    APPLIED = "applied"
    STALE = "stale"
    OUT_OF_ORDER = "out_of_order"


class WatchdogState(StrEnum):
    """A model's watchdog state, reported in telemetry as watchdog_state.

    common/protocol.py checks telemetry against the same three strings, because common/
    does not depend on robot/.
    """

    NONE = "none"
    OK = "ok"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class CommandOutcome:
    """The verdict on a received command, and the durations that telemetry records for it.

    freshness_ns is the receive time minus the capture time. gap_ns is the time since the
    previous command of this model received in this session, whatever its verdict or the
    session number it carries, and None for the first. replaced_age_ns is set only when this
    command is applied over an earlier applied command: the earlier command's age now, which
    is the peak age it reached.
    """

    verdict: Verdict
    freshness_ns: int
    gap_ns: int | None
    replaced_age_ns: int | None


class ModelControl:
    """One model's command stream within one session: a verdict on each command, and the watchdog.

    A command is stale when its age at receipt is above stale_ns, and out of order when it
    belongs to another session or its frame number is not above the last applied one; a
    command that is both counts as stale. Any other command is applied and becomes the
    command in effect. The command in effect, captured at c, is expired whenever
    now_ns - c > watchdog_ns, until a newer command is applied; a rejected command does not
    replace it.
    """

    def __init__(self, session: int, stale_ns: int, watchdog_ns: int, start_ns: int) -> None:
        # ASSUMPTION: A9. The thresholds are ours to choose. The robot declares them per model
        # in the handshake, and enforces the same values here.
        if watchdog_ns < stale_ns:
            # A command is accepted at any age up to stale_ns, so under a smaller watchdog
            # limit it could be applied already expired.
            raise ValueError(
                f"watchdog_ns={watchdog_ns} is below stale_ns={stale_ns}: a command accepted "
                "at the stale threshold would be applied already expired"
            )
        self._session = session
        self._stale_ns = stale_ns
        self._watchdog_ns = watchdog_ns
        self._start_ns = start_ns
        self._last_receive_ns: int | None = None
        self._applied_count = 0
        self._stale_count = 0
        self._out_of_order_count = 0
        # The command in effect. Both are None until the first command is applied.
        self._applied_frame_number: int | None = None
        self._applied_capture_ns: int | None = None
        # The watchdog's history from start_ns to the moment the command in effect was
        # applied. A query adds the interval from that moment to its own now_ns.
        self._closed_trips = 0
        self._closed_expired_ns = 0

    @property
    def received(self) -> int:
        # Every received command gets exactly one verdict.
        return self._applied_count + self._stale_count + self._out_of_order_count

    @property
    def applied(self) -> int:
        return self._applied_count

    @property
    def stale(self) -> int:
        # ASSUMPTION: A10. The stale count is reported in the robot's log and through the
        # 1 Hz telemetry (commands_stale) that the gateway includes in /metrics.
        return self._stale_count

    @property
    def out_of_order(self) -> int:
        return self._out_of_order_count

    def on_command(
        self, session: int, frame_number: int, capture_ns: int, now_ns: int
    ) -> CommandOutcome:
        """Judge a command received at now_ns, and apply it unless it is stale or out of order."""
        self._check_time(now_ns)
        # ASSUMPTION: A7. Freshness is the receive time minus the capture time, both on the
        # robot's clock: capture to receive, as the brief defines it, with no clock
        # synchronization needed.
        freshness_ns = now_ns - capture_ns
        gap_ns = None if self._last_receive_ns is None else now_ns - self._last_receive_ns
        self._last_receive_ns = now_ns
        if freshness_ns > self._stale_ns:
            self._stale_count += 1
            return CommandOutcome(Verdict.STALE, freshness_ns, gap_ns, None)
        if session != self._session or (
            self._applied_frame_number is not None and frame_number <= self._applied_frame_number
        ):
            self._out_of_order_count += 1
            return CommandOutcome(Verdict.OUT_OF_ORDER, freshness_ns, gap_ns, None)
        replaced_age_ns: int | None = None
        if self._applied_capture_ns is None:
            self._closed_expired_ns += now_ns - self._start_ns
        else:
            replaced_age_ns = now_ns - self._applied_capture_ns
            # The replaced command was applied at an age of at most stale_ns <= watchdog_ns,
            # so it was ok then. It expired once if its age passed the limit before now, and
            # it was expired for the part of its peak age above the limit.
            expired_ns = replaced_age_ns - self._watchdog_ns
            if expired_ns > 0:
                self._closed_trips += 1
                self._closed_expired_ns += expired_ns
        self._applied_count += 1
        self._applied_frame_number = frame_number
        self._applied_capture_ns = capture_ns
        return CommandOutcome(Verdict.APPLIED, freshness_ns, gap_ns, replaced_age_ns)

    def watchdog_state(self, now_ns: int) -> WatchdogState:
        """Return the state at now_ns: none until a command is applied, then expired or ok."""
        self._check_time(now_ns)
        if self._applied_capture_ns is None:
            return WatchdogState.NONE
        if now_ns - self._applied_capture_ns > self._watchdog_ns:
            return WatchdogState.EXPIRED
        return WatchdogState.OK

    def watchdog_trips(self, now_ns: int) -> int:
        """Count the changes from ok to expired up to now_ns, including one still in effect."""
        open_trips = 1 if self.watchdog_state(now_ns) is WatchdogState.EXPIRED else 0
        return self._closed_trips + open_trips

    def watchdog_expired_ns(self, now_ns: int) -> int:
        """Return the time from start_ns to now_ns during which no unexpired command was held."""
        self._check_time(now_ns)
        # Coverage is 1 - watchdog_expired_ns(now_ns) / (now_ns - start_ns). The time before
        # the first applied command counts as not covered, so a robot that never applies a
        # command has coverage 0, not 1.
        if self._applied_capture_ns is None:
            return now_ns - self._start_ns
        open_expired_ns = max(0, now_ns - self._applied_capture_ns - self._watchdog_ns)
        return self._closed_expired_ns + open_expired_ns

    def _check_time(self, now_ns: int) -> None:
        """Raise ValueError when now_ns is before start_ns or before the last command's receipt."""
        last_event_ns = self._start_ns if self._last_receive_ns is None else self._last_receive_ns
        if now_ns < last_event_ns:
            raise ValueError(
                f"now_ns={now_ns} is before the last event at {last_event_ns}: "
                "the robot clock never goes back"
            )
