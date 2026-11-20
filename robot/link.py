"""The seeded link model: one delay line per direction of a robot's connection.

A line holds each message from the moment it is pushed until its release time. The caller
passes the clock's reading, and sleeps until the next release itself.
"""

from __future__ import annotations

import math
import random
from collections import deque

from robot.settings import LinkSettings


def line_capacity(settings: LinkSettings, messages_per_s: float) -> int:
    """Return the messages a line holds: what its longest delay covers at that rate, plus slack."""
    return math.ceil((settings.delay_ms + settings.jitter_ms) / 1000 * messages_per_s) + 2


class DelayLine:
    """Messages in flight in one direction, released in the order they were pushed.

    Bounded by capacity: push refuses a message while the line is full. The robot then skips
    the tick on the uplink, and stops reading on the downlink.
    """

    def __init__(self, settings: LinkSettings, capacity: int, rng: random.Random) -> None:
        self._delay_ns = round(settings.delay_ms * 1_000_000)
        self._jitter_ms = settings.jitter_ms
        self._capacity = capacity
        self._rng = rng
        # Each message with its release time, which never decreases along the line.
        self._messages: deque[tuple[int, str | bytes]] = deque()
        self._last_release_ns: int | None = None

    def __len__(self) -> int:
        return len(self._messages)

    def full(self) -> bool:
        return len(self._messages) >= self._capacity

    def push(self, message: str | bytes, now_ns: int) -> int:
        """Take a message at now_ns and return its release time; ValueError when full."""
        if self.full():
            raise ValueError(f"the delay line already holds its {self._capacity} messages")
        release_ns = (
            now_ns + self._delay_ns + round(self._rng.uniform(0.0, self._jitter_ms) * 1_000_000)
        )
        # TCP keeps order: a message with a short jitter still arrives after the one ahead
        # of it.
        if self._last_release_ns is not None:
            release_ns = max(release_ns, self._last_release_ns)
        self._last_release_ns = release_ns
        self._messages.append((release_ns, message))
        return release_ns

    def next_release_ns(self) -> int | None:
        """Return the release time of the message at the head; None while the line is empty."""
        return self._messages[0][0] if self._messages else None

    def pop_due(self, now_ns: int) -> list[str | bytes]:
        """Take every message whose release time is at or before now_ns, in order."""
        due: list[str | bytes] = []
        while self._messages and self._messages[0][0] <= now_ns:
            due.append(self._messages.popleft()[1])
        return due
