"""Tests for the virtual-time loop that the async unit tests run on."""

import asyncio
import signal
import sys

import pytest

from common.aio import sleep_until
from tests.support.virtual_time import LoopClock, VirtualTimeLoop, run_virtual


async def test_async_tests_in_this_directory_run_on_the_virtual_time_loop() -> None:
    assert isinstance(asyncio.get_running_loop(), VirtualTimeLoop)


async def test_a_one_hour_sleep_moves_the_clock_by_exactly_an_hour() -> None:
    loop = asyncio.get_running_loop()
    start_s = loop.time()

    await asyncio.sleep(3_600)

    assert loop.time() == start_s + 3_600


async def test_the_clock_jumps_from_each_timer_to_the_next() -> None:
    clock = LoopClock()
    start_ns = clock.now_ns()
    woken_ns: list[int] = []

    async def wake_after(delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        woken_ns.append(clock.now_ns() - start_ns)

    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(wake_after(2.0))
        tasks.create_task(wake_after(0.25))
        tasks.create_task(wake_after(1.0))

    assert woken_ns == [250_000_000, 1_000_000_000, 2_000_000_000]


async def test_timers_set_for_one_deadline_fire_in_the_order_they_were_set() -> None:
    loop = asyncio.get_running_loop()
    fired: list[int] = []

    for index in range(8):
        loop.call_later(0.05, fired.append, index)
    await asyncio.sleep(0.2)

    assert fired == list(range(8))


async def test_a_timer_set_later_for_the_same_instant_fires_later() -> None:
    loop = asyncio.get_running_loop()
    fired: list[str] = []

    loop.call_later(0.8, fired.append, "set at 0.0 s for 0.8 s")
    await asyncio.sleep(0.7)
    # 0.7 + 0.1 is 0.7999999999999999 as a float, a deadline before the first timer's 0.8.
    loop.call_later(0.1, fired.append, "set at 0.7 s for 0.8 s")
    await asyncio.sleep(0.3)

    assert fired == ["set at 0.0 s for 0.8 s", "set at 0.7 s for 0.8 s"]


async def test_tasks_that_sleep_until_the_same_deadlines_wake_in_one_order_at_every_tick() -> None:
    clock = LoopClock()
    ticks, period_ns = 4, 10_000_000
    start_ns = clock.now_ns() + period_ns
    woken: list[list[int]] = [[] for _ in range(ticks)]

    async def wake_at_every_tick(index: int) -> None:
        for tick in range(ticks):
            await sleep_until(clock, start_ns + tick * period_ns)
            woken[tick].append(index)

    async with asyncio.TaskGroup() as tasks:
        for index in range(5):
            tasks.create_task(wake_at_every_tick(index))

    assert woken == [[0, 1, 2, 3, 4]] * ticks


async def test_the_clock_reads_an_instant_exactly_where_the_float_sum_of_the_sleeps_does_not() -> (
    None
):
    loop = asyncio.get_running_loop()

    await asyncio.sleep(0.1)
    await asyncio.sleep(0.2)

    # 0.1 + 0.2 is 0.30000000000000004 as a float.
    assert loop.time() == 0.3


async def test_the_loop_clock_rounds_where_the_float_product_falls_short_of_a_nanosecond() -> None:
    await asyncio.sleep(1.001)

    # 1.001 * 1e9 is 1000999999.9999999 as a float: truncated, it reads a nanosecond early.
    assert LoopClock().now_ns() == 1_001_000_000


@pytest.mark.skipif(sys.platform == "win32", reason="asyncio has no signal handlers on Windows")
async def test_a_signal_reaches_its_handler_without_the_clock_moving() -> None:
    loop = asyncio.get_running_loop()
    arrived = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, arrived.set)
    start_s = loop.time()

    signal.raise_signal(signal.SIGTERM)
    async with asyncio.timeout(3_600):
        # Completed by the signal handler, which the loop runs once it has read its self-pipe.
        await arrived.wait()

    assert loop.time() == start_s


async def test_a_timeout_fires_at_its_virtual_deadline() -> None:
    clock = LoopClock()
    start_ns = clock.now_ns()

    with pytest.raises(TimeoutError):
        async with asyncio.timeout(2.5):
            # Nothing sets this Event: only the timeout ends the wait.
            await asyncio.Event().wait()

    assert clock.now_ns() - start_ns == 2_500_000_000


def test_run_virtual_returns_what_main_returns_and_starts_its_clock_at_zero() -> None:
    async def read_clock_after_one_second() -> int:
        await asyncio.sleep(1)
        return LoopClock().now_ns()

    assert run_virtual(read_clock_after_one_second()) == 1_000_000_000


def test_an_await_that_nothing_can_complete_raises_instead_of_hanging() -> None:
    async def wait_for_ever() -> None:
        # Nothing sets this Event and no timer is set: the loop has nothing that could wake it.
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="deadlock"):
        run_virtual(wait_for_ever())


def test_an_await_that_never_ends_beside_a_periodic_timer_fails_after_a_virtual_day() -> None:
    async def wait_beside_a_ticker() -> None:
        loop = asyncio.get_running_loop()

        def tick() -> None:
            loop.call_later(60, tick)

        tick()
        # Nothing sets this Event, and the ticker's timer keeps the clock moving.
        await asyncio.Event().wait()

    with pytest.raises(RuntimeError, match="virtual time passed 24 hours"):
        run_virtual(wait_beside_a_ticker())
