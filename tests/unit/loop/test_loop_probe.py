"""Tests for the event-loop lag probe, on virtual time."""

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest

from cloud.metrics import probe_loop_lag
from common.clock import Clock
from common.histogram import Histogram
from tests.support.virtual_time import LoopClock, run_virtual

NS_PER_MS = 1_000_000
PERIOD_MS = 20
PERIOD_NS = PERIOD_MS * NS_PER_MS
PERIOD_S = PERIOD_MS / 1_000


class StallingClock:
    """The loop's clock plus the stalls a test adds.

    A stall is time that passes while the loop runs nothing: the clock moves on, and every
    timer that came due meanwhile fires late.
    """

    def __init__(self) -> None:
        self._stalled_ns = 0

    def now_ns(self) -> int:
        return LoopClock().now_ns() + self._stalled_ns

    def stall(self, duration_ns: int) -> None:
        self._stalled_ns += duration_ns


@asynccontextmanager
async def probing(clock: Clock, lag_ms: Histogram, active: asyncio.Event) -> AsyncIterator[None]:
    """Run the probe beside the body, then cancel it and check that the cancellation left it."""
    probe = asyncio.create_task(probe_loop_lag(clock, lag_ms, PERIOD_NS, active))
    try:
        yield
    finally:
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe


async def test_an_inactive_probe_takes_no_sample() -> None:
    lag_ms = Histogram()

    async with probing(LoopClock(), lag_ms, asyncio.Event()):
        await asyncio.sleep(10 * PERIOD_S)

    assert lag_ms.count == 0


def test_an_inactive_probe_sets_no_timer() -> None:
    async def wait_beside_a_parked_probe() -> None:
        async with probing(LoopClock(), Histogram(), asyncio.Event()):
            # Nothing sets this Event. With the probe parked, no timer is left to wake the
            # loop, which the virtual-time loop reports as a deadlock; a probe that kept a
            # timer would keep the loop turning for ever instead.
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="deadlock"):
        run_virtual(wait_beside_a_parked_probe())


def test_a_probe_parks_without_a_timer_once_it_is_no_longer_active() -> None:
    lag_ms = Histogram()

    async def wait_beside_a_probe_that_parks() -> None:
        active = asyncio.Event()
        active.set()
        async with probing(LoopClock(), lag_ms, active):
            await asyncio.sleep(2.5 * PERIOD_S)
            active.clear()
            # Nothing sets this Event: once the probe's last timer has fired, none is left.
            await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="deadlock"):
        run_virtual(wait_beside_a_probe_that_parks())
    assert lag_ms.count == 2


async def test_an_active_probe_takes_one_sample_per_period() -> None:
    lag_ms = Histogram()
    active = asyncio.Event()
    active.set()

    async with probing(LoopClock(), lag_ms, active):
        await asyncio.sleep(5.5 * PERIOD_S)

    assert (lag_ms.count, lag_ms.max_ms) == (5, 0.0)


async def test_the_first_target_is_one_period_after_the_activation() -> None:
    lag_ms = Histogram()
    active = asyncio.Event()
    counts: list[int] = []

    async with probing(LoopClock(), lag_ms, active):
        await asyncio.sleep(0.007)
        active.set()
        await asyncio.sleep(PERIOD_S - 0.001)
        counts.append(lag_ms.count)
        await asyncio.sleep(0.002)
        counts.append(lag_ms.count)

    assert (counts, lag_ms.max_ms) == ([0, 1], 0.0)


async def test_a_stall_of_five_periods_gives_one_sample() -> None:
    clock = StallingClock()
    lag_ms = Histogram()
    active = asyncio.Event()
    active.set()

    async with probing(clock, lag_ms, active):
        await asyncio.sleep(1.5 * PERIOD_S)
        clock.stall(5 * PERIOD_NS)
        await asyncio.sleep(3 * PERIOD_S)

    # One sample before the stall, one for the timer the stall made late, and one per period
    # after it. Without the rule, every target the stall passed would add a late sample.
    assert (lag_ms.count, lag_ms.sum_ms, lag_ms.max_ms) == (4, 5.0 * PERIOD_MS, 5.0 * PERIOD_MS)


async def test_a_wake_up_shortly_after_its_target_does_not_shift_the_next_target() -> None:
    clock = StallingClock()
    lag_ms = Histogram()
    active = asyncio.Event()
    active.set()

    async with probing(clock, lag_ms, active):
        await asyncio.sleep(0.5 * PERIOD_S)
        clock.stall(5 * NS_PER_MS)
        # To 2 ms after the second target, which is two periods after the start on the clock:
        # 10 ms have passed on the loop and 5 ms in the stall.
        await asyncio.sleep(0.027)

    # The first timer fires 5 ms late. A second target one period after that wake-up would
    # still be 3 ms away.
    assert (lag_ms.count, lag_ms.sum_ms, lag_ms.max_ms) == (2, 5.0, 5.0)


async def test_the_first_sample_after_a_park_of_ten_periods_is_below_one_period() -> None:
    lag_ms = Histogram()
    active = asyncio.Event()
    active.set()
    counts: list[int] = []

    async with probing(LoopClock(), lag_ms, active):
        await asyncio.sleep(2.5 * PERIOD_S)
        active.clear()
        await asyncio.sleep(10 * PERIOD_S)
        counts.append(lag_ms.count)
        active.set()
        await asyncio.sleep(1.5 * PERIOD_S)
        counts.append(lag_ms.count)

    assert counts == [2, 3]
    assert lag_ms.max_ms is not None
    assert lag_ms.max_ms < PERIOD_MS


async def test_a_timer_that_fires_after_the_last_robot_left_is_not_sampled() -> None:
    lag_ms = Histogram()
    active = asyncio.Event()
    active.set()

    async with probing(LoopClock(), lag_ms, active):
        await asyncio.sleep(1.5 * PERIOD_S)
        active.clear()
        await asyncio.sleep(3 * PERIOD_S)

    assert lag_ms.count == 1
