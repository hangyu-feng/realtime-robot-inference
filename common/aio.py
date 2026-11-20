"""The services' event loop and their one timer rule."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Coroutine
from typing import Any

from common.clock import Clock

if sys.platform == "linux":
    import uvloop


def run[T](main: Coroutine[Any, Any, T]) -> T:
    """Run main to completion on uvloop on Linux, where the services run, else on asyncio's loop."""
    if sys.platform == "linux":
        return asyncio.run(main, loop_factory=uvloop.new_event_loop)
    return asyncio.run(main)


async def sleep_until(clock: Clock, deadline_ns: int) -> None:
    """Return once the clock reads deadline_ns or later; at once for a deadline already passed."""
    # uvloop rounds a delay to whole milliseconds, counts it from a loop time truncated to the
    # millisecond, and turns a delay below half a millisecond into call_soon, so a timer can
    # fire before its deadline. Sleeping whole milliseconds, rounded up, and then reading the
    # clock again means this never returns early and never spins.
    remaining_ns = deadline_ns - clock.now_ns()
    while remaining_ns > 0:
        await asyncio.sleep(-(-remaining_ns // 1_000_000) / 1000)
        remaining_ns = deadline_ns - clock.now_ns()
