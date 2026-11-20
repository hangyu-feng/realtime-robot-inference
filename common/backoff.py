"""Exponential backoff with jitter, for robot reconnects and worker restarts."""

from __future__ import annotations

import random


class Backoff:
    """Pauses between retries that double from min_ns up to max_ns, with jitter.

    Each pause is its ceiling times a factor drawn uniformly from [0.5, 1]. The jitter keeps
    robots, or worker restarts, that failed together from retrying together.
    """

    def __init__(self, min_ns: int, max_ns: int, rng: random.Random) -> None:
        if not 0 < min_ns <= max_ns:
            raise ValueError(
                f"a backoff needs 0 < min_ns <= max_ns, got min_ns={min_ns}, max_ns={max_ns}"
            )
        self._min_ns = min_ns
        self._max_ns = max_ns
        self._rng = rng
        self._attempt = 0

    def next_ns(self) -> int:
        """Return the pause before the next retry."""
        ceiling_ns = min(self._max_ns, self._min_ns << self._attempt)
        delay_ns = round(ceiling_ns * self._rng.uniform(0.5, 1.0))
        # The count stops growing at the cap, so the shifted value stays small however long the
        # retries go on.
        if ceiling_ns < self._max_ns:
            self._attempt += 1
        return delay_ns

    def reset(self) -> None:
        """Start again from min_ns, after a success."""
        self._attempt = 0
