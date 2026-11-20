"""The open-loop capture schedule of a robot: when each of its frames is due."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CaptureSchedule:
    """Ticks at start_ns + k * period_ns for the frame numbers k = 0, 1, 2, ...

    The schedule is open loop: a robot that wakes late does not move the later ticks. It
    sends only the latest due frame, and the ticks it passed count as captured.
    """

    start_ns: int
    period_ns: int

    def capture_ns(self, frame_number: int) -> int:
        """Return the scheduled time of the tick that captures this frame."""
        # ASSUMPTION: A6. The capture time is the scheduled tick time, not the actual send
        # time, so a late simulator cannot hide latency (no coordinated omission). The
        # difference is reported as fire lag.
        return self.start_ns + frame_number * self.period_ns

    def latest_due(self, now_ns: int) -> int:
        """Return the highest frame number whose capture time is at or before now_ns."""
        # Frame numbers start at 0, so -1 means that no tick is due yet.
        return max(-1, (now_ns - self.start_ns) // self.period_ns)
