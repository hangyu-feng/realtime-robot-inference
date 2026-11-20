"""Default freshness thresholds per model, and where they come from.

A robot declares its thresholds in the handshake. These are the values it
declares by default, and the values the server budgets against before any
robot has connected.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

# ASSUMPTION: A9. The brief leaves the thresholds to us. They come from what a
# control loop can tolerate, not from what the system achieves.
# Depth is a displacement budget: at 1 m/s a robot moves 15 cm in 150 ms, and
# steering and speed must not act on older geometry.
# Classification changes slowly: what is in view rarely changes within half a second.
DEFAULT_STALE_MS: Mapping[str, float] = MappingProxyType({"cls": 500.0, "depth": 150.0})

# Twice the stale threshold: a command that was fresh enough to accept may age
# by one more threshold before the robot stops acting on it.
DEFAULT_WATCHDOG_MS: Mapping[str, float] = MappingProxyType({"cls": 1000.0, "depth": 300.0})
