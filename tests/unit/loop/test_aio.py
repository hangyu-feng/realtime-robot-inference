"""Tests for the timer rule and the services' event loop."""

import asyncio
import sys

import pytest

from common.aio import run, sleep_until
from tests.support.virtual_time import LoopClock


class HalfSpeedClock:
    """Half the loop's time: every timer fires before this clock reaches its deadline.

    uvloop's timers can fire early in the same way.
    """

    def now_ns(self) -> int:
        return LoopClock().now_ns() // 2


async def test_sleep_until_returns_at_the_deadline() -> None:
    clock = LoopClock()
    deadline_ns = clock.now_ns() + 25_000_000

    await sleep_until(clock, deadline_ns)

    assert clock.now_ns() == deadline_ns


@pytest.mark.parametrize("passed_ns", [0, 1, 7_000_000_000])
async def test_sleep_until_returns_at_once_for_a_deadline_already_reached(passed_ns: int) -> None:
    clock = LoopClock()
    start_ns = clock.now_ns()

    await sleep_until(clock, start_ns - passed_ns)

    assert clock.now_ns() == start_ns


async def test_sleep_until_waits_a_whole_millisecond_for_a_deadline_closer_than_that() -> None:
    clock = LoopClock()
    start_ns = clock.now_ns()

    await sleep_until(clock, start_ns + 400_000)

    assert clock.now_ns() == start_ns + 1_000_000


async def test_sleep_until_reads_the_clock_again_after_a_timer_that_fired_early() -> None:
    clock = HalfSpeedClock()
    deadline_ns = clock.now_ns() + 10_000_000

    await sleep_until(clock, deadline_ns)

    assert clock.now_ns() == deadline_ns


@pytest.mark.skipif(sys.platform != "linux", reason="uvloop is the services' loop on Linux only")
def test_run_runs_main_on_uvloop_on_linux() -> None:
    async def loop_module() -> str:
        return type(asyncio.get_running_loop()).__module__

    assert run(loop_module()) == "uvloop"
