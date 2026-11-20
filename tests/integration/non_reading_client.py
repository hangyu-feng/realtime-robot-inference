"""A robot that sends frames at its declared rate and never reads a command.

Run as `python -m tests.integration.non_reading_client --url URL --robot-id ID --duration-s S`.
Its socket's receive buffer is the robot's 4 KiB, which the kernel doubles, and the library
reads 16 messages ahead, its default, then stops reading, so the gateway's commands to it soon
stay unsent. It exits 0 when a send finds the connection closed, 1 when its duration passes
with the connection still open, and 2 for bad arguments. It prints nothing per frame.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from common import aio
from common.clock import MonotonicClock
from common.transport import TransportClosed
from tests.integration.harness import NS_PER_S, ROBOT_SETTINGS, open_raw_session, raw_frame


async def _send_until_closed(url: str, robot_id: str, duration_s: float) -> int:
    clock = MonotonicClock()
    connection, session = await open_raw_session(url, robot_id)
    period_ns = round(NS_PER_S / ROBOT_SETTINGS.frame_rate_hz)
    start_ns = clock.now_ns()
    end_ns = start_ns + round(duration_s * NS_PER_S)
    frame_number = 0
    # Only the gateway can close the connection: this side never reads, so it never learns
    # of a close but through a send that fails.
    while (tick_ns := start_ns + frame_number * period_ns) < end_ns:
        await aio.sleep_until(clock, tick_ns)
        try:
            connection.send_nowait(raw_frame(session, frame_number))
        except TransportClosed:
            return 0
        frame_number += 1
    connection.abort()
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tests.integration.non_reading_client",
        description="Send frames to the gateway at the robot's rate, and never read.",
    )
    parser.add_argument("--url", required=True, help="the gateway's ws:// address")
    parser.add_argument("--robot-id", required=True)
    parser.add_argument("--duration-s", type=float, required=True)
    # A bad argument exits with 2 here.
    arguments = parser.parse_args(argv)
    return aio.run(_send_until_closed(arguments.url, arguments.robot_id, arguments.duration_s))


if __name__ == "__main__":
    sys.exit(main())
