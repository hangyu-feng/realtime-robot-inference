"""A virtual-time event loop for the async unit tests.

Time on this loop stands still while any callback is ready and jumps straight to the next timer
when every task waits, so asyncio.timeout, TaskGroups and cancellation behave as on a real loop
while a test never waits on the wall clock.
"""

from __future__ import annotations

import asyncio
import contextvars
import itertools
import selectors
from collections.abc import Callable, Coroutine
from typing import Any, Self

# Past this much virtual time a test is taken to never end: the longest test runs a session for
# two virtual hours. A test that awaits what nothing completes, beside a task that keeps a
# periodic timer, would otherwise spin for ever.
_MAX_VIRTUAL_TIME_NS = 24 * 3_600 * 1_000_000_000


class _VirtualTimeSelector(selectors.DefaultSelector):
    """The real selector, polled without waiting: a wait it is asked for jumps the clock."""

    def __init__(self) -> None:
        super().__init__()
        # Whole nanoseconds, so that an instant is one value however the float sums that reach
        # it round.
        self.time_ns = 0

    def select(self, timeout: float | None = None) -> list[tuple[selectors.SelectorKey, int]]:
        # Polled with timeout 0, so what wakes the loop through its self-pipe, a signal or a
        # callback scheduled from another thread, still arrives.
        events = super().select(0)
        if events:
            return events
        if timeout is None:
            raise RuntimeError(
                "deadlock: no timer is set and no callback is ready, so nothing can ever wake "
                "the virtual-time loop"
            )
        time_ns = self.time_ns + round(timeout * 1e9)
        if time_ns > _MAX_VIRTUAL_TIME_NS:
            # The clock stays where it was, so the cleanup that follows still runs.
            raise RuntimeError(
                f"virtual time passed {_MAX_VIRTUAL_TIME_NS // 3_600_000_000_000} hours: the "
                "test awaits something that never comes beside a timer that keeps firing"
            )
        self.time_ns = time_ns
        return events


class _Deadline(float):
    """A timer's deadline, which sorts after every deadline set earlier for the same instant.

    asyncio keeps its timers in a heap ordered by their deadlines alone, and a heap gives equal
    deadlines no order.
    """

    __slots__ = ("order",)

    order: int

    def __new__(cls, when_s: float, order: int) -> Self:
        deadline = super().__new__(cls, when_s)
        deadline.order = order
        return deadline

    def __lt__(self, other: float, /) -> bool:
        if isinstance(other, _Deadline) and self == other:
            return self.order < other.order
        return super().__lt__(other)


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose time() is a virtual clock that starts at 0.0.

    When no callback is ready the clock jumps by exactly the time to the next timer. Timers due
    at one instant fire in the order they were set, as they do on uvloop, the services' loop.
    An await that nothing can ever complete raises RuntimeError at once, so a test that would
    hang fails instead. Beside a task that keeps a periodic timer, such as a robot's tick loop
    or an active loop-lag probe, the clock runs on instead, and the loop raises RuntimeError
    once it would pass a virtual day. A task that never waits for a timer, such as one
    looping on sleep(0), keeps the clock still and is not detected. Real sockets and threads
    must not be used on this loop: their events take real time, which the virtual clock does
    not see, and a wait for them is reported as a deadlock.
    """

    def __init__(self) -> None:
        # The loop drops its own reference to the selector when it closes; time() keeps this one.
        self._virtual_selector = _VirtualTimeSelector()
        self._timers_set = itertools.count()
        super().__init__(self._virtual_selector)

    def time(self) -> float:
        return self._virtual_selector.time_ns / 1e9

    def call_at[*Ts](
        self,
        when: float,
        callback: Callable[[*Ts], object],
        *args: *Ts,
        context: contextvars.Context | None = None,
    ) -> asyncio.TimerHandle:
        # The clock stands still while callbacks run, so timers set by different callbacks
        # often share a deadline. Rounded to the clock's whole nanoseconds, an instant reached
        # by two float sums is one deadline, and the count orders the timers that share it.
        deadline = _Deadline(round(when * 1e9) / 1e9, next(self._timers_set))
        return super().call_at(deadline, callback, *args, context=context)


class LoopClock:
    """The running loop's clock, in nanoseconds.

    The Clock of every virtual-time test, so the stamps the code under test takes and the
    timers it sets read one timeline.
    """

    def now_ns(self) -> int:
        return round(asyncio.get_running_loop().time() * 1e9)


def run_virtual[T](main: Coroutine[Any, Any, T]) -> T:
    """Run main on a new virtual-time loop, for a synchronous test that drives async code."""
    return asyncio.run(main, loop_factory=VirtualTimeLoop)
