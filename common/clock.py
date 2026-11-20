"""The injectable source of monotonic time.

The I/O layers read a Clock once per event and pass the reading as now_ns to the
scheduling, queueing, protocol and robot-control logic, which never reads the
operating system's clock itself, so tests and the simulated-clock capacity model
move time by hand.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    """A source of monotonic time in nanoseconds."""

    def now_ns(self) -> int: ...


class MonotonicClock:
    """The host's monotonic clock."""

    def now_ns(self) -> int:
        # ASSUMPTION: A20. Every process on one host reads the same CLOCK_MONOTONIC timeline, so
        # stage timestamps taken in different processes can be subtracted. perf_counter is
        # not used because the clock behind it is an implementation detail of the interpreter.
        return time.monotonic_ns()


class FakeClock:
    """A clock that moves only when told to."""

    def __init__(self, start_ns: int = 0) -> None:
        self._now_ns = start_ns

    def now_ns(self) -> int:
        return self._now_ns

    def advance_ns(self, delta_ns: int) -> None:
        if delta_ns < 0:
            raise ValueError(f"a monotonic clock never goes back, got delta_ns={delta_ns}")
        self._now_ns += delta_ns
