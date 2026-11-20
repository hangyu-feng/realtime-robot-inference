"""The robot's tunables, each with its default and its unit.

A value is checked once, when the settings are built at startup. A bad value raises
ConfigError there, naming the field, and the robot code trusts the values afterwards.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from urllib.parse import urlsplit

from common.commands import MODEL_KEYS
from common.errors import ConfigError
from common.protocol import MAX_FRAME_RATE_HZ, MAX_THRESHOLD_MS, MIN_FRAME_RATE_HZ
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS

# The event loop times a wait in whole milliseconds and runs one below half a millisecond at
# once, so a shorter wait would not be the one configured. From this floor up a wait is at
# least 1 000 000 ns after conversion, which the backoff and the timers rely on.
_MIN_WAIT_MS = 1
_WAIT_FIELDS = (
    "connect_timeout_ms",
    "welcome_timeout_ms",
    "close_timeout_ms",
    "stall_timeout_ms",
    "reconnect_min_ms",
    "reconnect_max_ms",
)
_BYTE_FIELDS = ("sndbuf_bytes", "notsent_lowat_bytes", "rcvbuf_bytes", "max_message_bytes")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


@dataclass(frozen=True)
class LinkSettings:
    """The link model of one direction: every message is held for the delay plus a jitter.

    The jitter of a message is drawn uniformly from 0 to jitter_ms. Both default to 0, which
    is no link model at all: the messages go straight to the socket, or come straight from it.
    """

    delay_ms: float = 0.0
    jitter_ms: float = 0.0

    def __post_init__(self) -> None:
        for name in ("delay_ms", "jitter_ms"):
            value = getattr(self, name)
            _require(
                math.isfinite(value) and value >= 0, f"{name} must be a finite number, 0 or more"
            )

    @property
    def enabled(self) -> bool:
        """Whether this direction holds messages back at all."""
        return self.delay_ms > 0 or self.jitter_ms > 0


@dataclass(frozen=True)
class RobotSettings:
    """Every setting of one simulated robot.

    The command line sets the address, the token, the models, the thresholds, the frame rate
    and the link model. No flag, plan key or environment variable sets the four timeouts,
    the two reconnect pauses or the four byte sizes: only code does, so they are checked
    against their floor and get no further range.
    """

    # The gateway's default port, on this host: by default the fleet runs in the gateway's
    # container.
    url: str = "ws://127.0.0.1:8080/ws"
    # ASSUMPTION: A2. Every frame is offered to both models. A robot may subscribe to one
    # alone, and then gets that model's commands only.
    models: tuple[str, ...] = MODEL_KEYS
    # ASSUMPTION: A9. The thresholds are ours to choose: the robot declares these in its hello
    # and enforces the same values on the commands it receives. Each has exactly the
    # subscribed models as its keys.
    stale_ms: Mapping[str, float] = field(default_factory=lambda: DEFAULT_STALE_MS)
    watchdog_ms: Mapping[str, float] = field(default_factory=lambda: DEFAULT_WATCHDOG_MS)
    # ASSUMPTION: A16. Every demo sends 30 frames per second on the wire.
    frame_rate_hz: float = 30.0
    token: str | None = None
    uplink: LinkSettings = field(default_factory=LinkSettings)
    downlink: LinkSettings = field(default_factory=LinkSettings)

    # Name resolution, the TCP connect and the upgrade together.
    connect_timeout_ms: float = 5000.0
    # The gateway answers a hello at once, so a welcome this late means it is not serving.
    welcome_timeout_ms: float = 5000.0
    # A closing handshake that has not ended by then is dropped, as is the wait for the
    # gateway's close code after a send found the connection closed.
    close_timeout_ms: float = 1000.0
    # A gateway that has taken no frame for this long is not serving the robot. It mirrors
    # the gateway's idle timeout in the other direction.
    stall_timeout_ms: float = 3000.0
    # The pause before a reconnect doubles from the minimum to the maximum, with jitter, so
    # robots that lost the gateway together do not all return together.
    reconnect_min_ms: float = 100.0
    reconnect_max_ms: float = 5000.0

    # The request bench/transport_bench.py uses for a stalled path that holds about two
    # frames of the default frame set. Linux doubles it.
    sndbuf_bytes: int = 40 * 1024
    # The kernel stops taking new frame bytes once this much is unsent.
    notsent_lowat_bytes: int = 16 * 1024
    # The gateway sends only commands, each under 100 bytes.
    rcvbuf_bytes: int = 4096
    # The largest message a robot accepts: a welcome or a command.
    max_message_bytes: int = 4096

    def __post_init__(self) -> None:
        # Frozen settings must not share a mutable mapping with their caller: a later change
        # to the caller's dict would change settings that have passed every rule here.
        object.__setattr__(self, "stale_ms", MappingProxyType(dict(self.stale_ms)))
        object.__setattr__(self, "watchdog_ms", MappingProxyType(dict(self.watchdog_ms)))
        try:
            address = urlsplit(self.url)
            # Reading the port parses it.
            port = address.port
        except ValueError as error:
            raise ConfigError(f"url must be ws://HOST[:PORT]/PATH: {error}") from error
        # TLS ends in front of the service, so a robot speaks plain ws. The robot authenticates
        # with the hello's token, and the WebSocket client refuses a fragment and a user name
        # without a password only when it dials, after the fleet has started.
        _require(
            address.scheme == "ws"
            and bool(address.hostname)
            and port != 0
            and not address.fragment
            and address.username is None,
            "url must be ws://HOST[:PORT]/PATH",
        )
        # The hello carries the token as JSON text, and the gateway compares it as ASCII.
        _require(
            self.token is None
            or (self.token != "" and self.token.isascii() and self.token.isprintable()),
            "token must be printable ASCII and not empty",
        )
        _require(
            len(self.models) > 0
            and len(set(self.models)) == len(self.models)
            and set(self.models) <= set(MODEL_KEYS),
            f"models must be one or more distinct keys of {', '.join(MODEL_KEYS)}",
        )
        self._check_thresholds()
        # A comparison with NaN is false, and the bounds refuse infinity.
        _require(
            MIN_FRAME_RATE_HZ <= self.frame_rate_hz <= MAX_FRAME_RATE_HZ,
            f"frame_rate_hz must be from {MIN_FRAME_RATE_HZ} to {MAX_FRAME_RATE_HZ}, "
            "the rates a hello may declare",
        )
        for name in _WAIT_FIELDS:
            _require(getattr(self, name) >= _MIN_WAIT_MS, f"{name} must be at least {_MIN_WAIT_MS}")
        _require(
            self.reconnect_max_ms >= self.reconnect_min_ms,
            "reconnect_max_ms must be at least reconnect_min_ms",
        )
        for name in _BYTE_FIELDS:
            _require(getattr(self, name) >= 1, f"{name} must be at least 1")

    def _check_thresholds(self) -> None:
        for name in ("stale_ms", "watchdog_ms"):
            thresholds: Mapping[str, float] = getattr(self, name)
            _require(
                thresholds.keys() == set(self.models),
                f"{name} must have exactly the subscribed models as its keys",
            )
            for model, threshold_ms in thresholds.items():
                # The bound a hello puts on a threshold. The command line sets these, and a
                # value the gateway would refuse stops the fleet here, at startup, instead.
                _require(
                    math.isfinite(threshold_ms) and 0 < threshold_ms <= MAX_THRESHOLD_MS,
                    f"{name}.{model} must be finite, above 0 and at most {MAX_THRESHOLD_MS:g}",
                )
        for model in self.models:
            # A command is accepted at any age up to the stale threshold. Under a smaller
            # watchdog limit it could be applied already expired.
            _require(
                self.watchdog_ms[model] >= self.stale_ms[model],
                f"watchdog_ms.{model} must be at least stale_ms.{model}",
            )
