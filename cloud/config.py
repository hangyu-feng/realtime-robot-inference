"""Server configuration: every tunable of the gateway, with its default and its unit.

A value is checked once, when the configuration is built at startup. A bad value
raises ConfigError there, naming the field, and everything downstream trusts the
values and the cross-field rules checked here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from fractions import Fraction
from types import MappingProxyType

from common.commands import MODEL_KEYS
from common.errors import ConfigError
from common.protocol import MAX_FRAME_RATE_HZ, MIN_FRAME_RATE_HZ
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from common.worker_protocol import MAX_BATCH_FRAMES, MAX_PAYLOAD_BYTES, batch_request_bytes

_MAX_NICE = 19
_MAX_PORT = 65_535
_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
# A duration is converted to integer nanoseconds, round(value_ms * 1_000_000), and meets
# readings of the monotonic clock, which the wire carries as an i64. Within that range, about
# 292 years, the conversion is a finite integer that the type holds.
_MAX_DURATION_NS = 2**63 - 1


class QueuePolicy(StrEnum):
    LATEST = "latest"  # one newest-frame slot per robot: the design
    FIFO = "fifo"  # a bounded queue per robot and model: the control experiment


class ModelRuntime(StrEnum):
    ONNX = "onnx"
    FAKE = "fake"  # waits fake_frame_ms per frame: deterministic scheduling tests
    NULL = "null"  # returns at once: isolates transport and scheduling


class Topology(StrEnum):
    # ASSUMPTION: A1. The brief puts the whole system in one container, so the robots
    # share it by default. The split topology gives them a container of their own.
    SINGLE = "single"
    SPLIT = "split"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _finite(name: str, value: float) -> float:
    """Return a setting as a float, refusing a value that is not a finite number."""
    # mypy accepts an int where a float is declared, and float() raises OverflowError for an
    # int beyond the float range, which json.loads returns for a long run of digits.
    try:
        number = float(value)
    except OverflowError as error:
        raise ConfigError(f"{name} must be a finite number") from error
    _require(math.isfinite(number), f"{name} must be a finite number")
    return number


def _require_positive(name: str, value: float) -> None:
    _require(_finite(name, value) > 0, f"{name} must be above 0")


def _require_not_negative(name: str, value: float) -> None:
    _require(_finite(name, value) >= 0, f"{name} must be 0 or more")


def _require_duration(name: str, value: float, ns_per_unit: int) -> None:
    _require(
        value * ns_per_unit <= _MAX_DURATION_NS,
        f"{name} must be at most {_MAX_DURATION_NS} ns, the range of the clock",
    )


@dataclass(frozen=True)
class ModelConfig:
    """The settings of one model's worker and scheduler."""

    # The thresholds the server budgets against before any robot has connected. A robot
    # declares its own in the handshake; those drive its skip rule and its robot-side rules.
    stale_ms: float
    watchdog_ms: float
    runtime: ModelRuntime = ModelRuntime.ONNX
    # Intra-op threads of the worker's session. 0 leaves the count to ONNX Runtime, which
    # sizes its pool from the host's cores: the control experiment, never a default.
    threads: int = 1
    # Sizes calibrated at warm-up (DESIGN.md 6.2). The batch cap is the largest of them
    # whose batch time fits the freshness budget, unless forced_batch_cap is set.
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8)
    # Set only by the batch-cap comparison of DESIGN.md section 12; one of batch_sizes.
    forced_batch_cap: int | None = None
    # The paced worker is the one elastic consumer: it gives way when the total demand
    # would exceed the CPU quota (DESIGN.md 6.4).
    paced: bool = False
    # Service time per frame of the fake runtime, which only waits.
    fake_frame_ms: float = 0.0

    def __post_init__(self) -> None:
        _require_positive("stale_ms", self.stale_ms)
        _require_positive("watchdog_ms", self.watchdog_ms)
        _require_not_negative("fake_frame_ms", self.fake_frame_ms)
        _require_duration("stale_ms", self.stale_ms, _NS_PER_MS)
        _require_duration("watchdog_ms", self.watchdog_ms, _NS_PER_MS)
        _require_duration("fake_frame_ms", self.fake_frame_ms, _NS_PER_MS)
        # A command is accepted at any age up to the stale threshold. Under a smaller
        # watchdog limit it could be applied already expired.
        _require(self.watchdog_ms >= self.stale_ms, "watchdog_ms must be at least stale_ms")
        _require(self.threads >= 0, "threads must be 0 (the runtime's default) or more")
        _require(
            len(self.batch_sizes) > 0
            and self.batch_sizes[0] == 1
            and all(a < b for a, b in zip(self.batch_sizes, self.batch_sizes[1:], strict=False)),
            "batch_sizes must start at 1 and ascend",
        )
        _require(
            self.batch_sizes[-1] <= MAX_BATCH_FRAMES,
            f"batch_sizes must not exceed {MAX_BATCH_FRAMES}, the frames of one worker request",
        )
        _require(
            self.forced_batch_cap is None or self.forced_batch_cap in self.batch_sizes,
            "forced_batch_cap must be one of batch_sizes: only calibrated sizes have a batch time",
        )


def _default_models() -> dict[str, ModelConfig]:
    return {
        "cls": ModelConfig(
            stale_ms=DEFAULT_STALE_MS["cls"], watchdog_ms=DEFAULT_WATCHDOG_MS["cls"]
        ),
        # Depth is expected to be saturated from the first robot (DESIGN.md 6.3, H1),
        # so it is the consumer that can give way.
        "depth": ModelConfig(
            stale_ms=DEFAULT_STALE_MS["depth"],
            watchdog_ms=DEFAULT_WATCHDOG_MS["depth"],
            paced=True,
        ),
    }


# Fields that must be finite and above 0, and fields that must be finite and not negative.
_POSITIVE_FIELDS = (
    "skip_max_unserved_ms",
    "pacing_min_share_cores",
    "ingress_max_messages_per_s",
    "ingress_max_bytes_per_s",
    "ingress_burst_s",
    "max_frame_age_ms",
    "handshake_timeout_ms",
    "idle_timeout_ms",
    "inference_timeout_ms",
    "slow_consumer_timeout_ms",
    "warmup_timeout_ms",
    "shutdown_grace_ms",
    "ping_interval_ms",
    "robot_retention_s",
    "loop_probe_ms",
)
_NOT_NEGATIVE_FIELDS = ("overhead_allowance_ms", "pacing_headroom_cores", "io_cpu_ms_per_frame")
# Durations, with the nanoseconds in their unit.
_DURATION_FIELDS = {
    "skip_max_unserved_ms": _NS_PER_MS,
    "overhead_allowance_ms": _NS_PER_MS,
    "io_cpu_ms_per_frame": _NS_PER_MS,
    "ingress_burst_s": _NS_PER_S,
    "max_frame_age_ms": _NS_PER_MS,
    "handshake_timeout_ms": _NS_PER_MS,
    "idle_timeout_ms": _NS_PER_MS,
    "inference_timeout_ms": _NS_PER_MS,
    "slow_consumer_timeout_ms": _NS_PER_MS,
    "warmup_timeout_ms": _NS_PER_MS,
    "shutdown_grace_ms": _NS_PER_MS,
    "ping_interval_ms": _NS_PER_MS,
    "robot_retention_s": _NS_PER_S,
    "loop_probe_ms": _NS_PER_MS,
}
# Whole numbers that must be at least 1.
_COUNT_FIELDS = (
    "fifo_capacity_frames",
    "max_connections",
    "max_message_bytes",
    "socket_rcvbuf_bytes",
    "socket_sndbuf_bytes",
    "downlink_queue_commands",
    "max_robot_entries",
)


@dataclass(frozen=True)
class ServerConfig:
    """Every setting of the gateway and its workers."""

    host: str = "0.0.0.0"
    port: int = 8080
    topology: Topology = Topology.SINGLE
    # ASSUMPTION: A20. Robots and gateway run on one host and read one monotonic clock.
    shared_clock: bool = True

    queue_policy: QueuePolicy = QueuePolicy.LATEST
    # Under the FIFO control no frame may be dropped within the 30 s steady run, so that
    # freshness can grow for the whole run (DESIGN.md section 12): 900 frames is the whole
    # run of a 30 fps robot. Only that one-robot control runs FIFO, and the queue stays
    # bounded (DESIGN.md 1.3, rule 6).
    fifo_capacity_frames: int = 900
    skip_rule: bool = True
    # The skip rule stops passing over a robot that has gone this long unserved by the
    # model. It equals the longest default watchdog limit (classification, 1 s): a robot
    # that one model has passed over that long holds no unexpired command of that model,
    # because neither default watchdog limit is above 1 s, so serving a frame that is known
    # to be stale costs it nothing more, while a shorter interval would spend the worker on
    # doomed frames more often.
    skip_max_unserved_ms: float = 1000.0
    # ASSUMPTION: A16. Every demo sends 30 fps. The freshness budget of the batch cap is
    # computed for this rate before any robot declares its own.
    frame_rate_hz: float = 30.0
    # Handling time outside the worker, without link delay: the dispatch round trip and the
    # delivery on one host. One allowance serves the batch-cap budget and the skip rule.
    overhead_allowance_ms: float = 10.0
    models: Mapping[str, ModelConfig] = field(default_factory=_default_models)

    # Workers run below the gateway and the simulator, so under a cpuset limit, where the
    # kernel does not freeze the container, the I/O processes preempt them (DESIGN.md 6.4).
    worker_nice: int = 10
    # Quota kept unassigned by the feed-forward pacing share. The quota is enforced per
    # 100 ms period, and the predicted demand of the other consumers comes from declared
    # frame rates, calibrated worker costs and a configured I/O cost, so 10 ms per period
    # absorbs their error before it becomes a container-wide freeze, at a cost of at most
    # 0.1 core to the paced worker.
    pacing_headroom_cores: float = 0.1
    # The paced worker keeps serving even when the others' predicted demand fills the quota.
    pacing_min_share_cores: float = 0.25
    # CPU time per frame of everything outside the workers that runs under the quota: the
    # gateway and, in the single topology, the robot fleet. The pacing share reserves it,
    # so a value below the real cost lets the container's demand pass the quota, and the
    # kernel then freezes every process in it: err high. The default is a placeholder until
    # a gated run of bench.transport_bench sets it for the hardware the gateway runs on.
    # The capacity model reads its value from its input file.
    io_cpu_ms_per_frame: float = 0.15

    # A safety valve, not the main mechanism (DESIGN.md 8.1).
    max_connections: int = 64
    # ASSUMPTION: A21. Real authentication is out of scope. A shared token is an optional
    # hook, off by default.
    shared_token: str | None = None

    # Every message a connection sends, text or binary, passes the ingress limit before it
    # is decoded. A robot sends 30 frames and one telemetry report per second; the defaults
    # hold a robot to about four times that message rate and to 8 MiB/s, over four times
    # the bytes of 30 fps at the largest frame size the frame set aims at (60 KB), so an
    # honest robot whose frames arrive bunched after a stall is never refused, while one
    # that floods costs the event loop a bounded amount. The burst holds one second of
    # each rate, which also lets one message of max_message_bytes through.
    ingress_max_messages_per_s: float = 128.0
    ingress_max_bytes_per_s: float = 8 * 2**20
    ingress_burst_s: float = 1.0
    # A 640x480 JPEG at the default quality is tens of kilobytes; 1 MiB leaves room for any
    # honest encoder and bounds the receive buffer of one message.
    max_message_bytes: int = 2**20
    # Far above every threshold: refuses frame timestamps that are garbage, nothing else.
    max_frame_age_ms: float = 10_000.0

    handshake_timeout_ms: float = 5_000.0
    idle_timeout_ms: float = 3_000.0
    inference_timeout_ms: float = 5_000.0
    slow_consumer_timeout_ms: float = 1_000.0
    warmup_timeout_ms: float = 120_000.0
    shutdown_grace_ms: float = 2_000.0

    # Linux doubles a requested size for its own bookkeeping, and a loopback segment can
    # carry a whole frame past TCP_NOTSENT_LOWAT, so the buffer sizes are what bound a
    # stalled path. With frames of about 40 KB, 40 KiB requested here and at the robot
    # leaves about two frames on a path whose reader has stopped (DESIGN.md 5.1).
    socket_rcvbuf_bytes: int = 40_960
    socket_sndbuf_bytes: int = 40_960
    # Commands of both models waiting for one robot's connection. The oldest is dropped
    # when it is full: a newer command of a model supersedes an older one, and a robot
    # whose queue fills is not reading, which the slow-consumer timeout then ends.
    downlink_queue_commands: int = 8
    ping_interval_ms: float = 2_000.0

    robot_retention_s: float = 300.0
    max_robot_entries: int = 256
    loop_probe_ms: float = 20.0

    def __post_init__(self) -> None:
        # A frozen config must not share a mutable mapping with its caller: a later change to
        # the caller's dict would change a config that has passed every rule here.
        object.__setattr__(self, "models", MappingProxyType(dict(self.models)))
        for name in _POSITIVE_FIELDS:
            _require_positive(name, getattr(self, name))
        for name in _NOT_NEGATIVE_FIELDS:
            _require_not_negative(name, getattr(self, name))
        for name, ns_per_unit in _DURATION_FIELDS.items():
            _require_duration(name, getattr(self, name), ns_per_unit)
        for name in _COUNT_FIELDS:
            _require(getattr(self, name) >= 1, f"{name} must be at least 1")
        _require(1 <= self.port <= _MAX_PORT, f"port must be between 1 and {_MAX_PORT}")
        _require(
            0 <= self.worker_nice <= _MAX_NICE, f"worker_nice must be between 0 and {_MAX_NICE}"
        )
        # The configured rate stands for a declared one: the batch-cap budget assumes it before
        # any robot has declared its own, and the capacity model's robots declare it. So it has
        # the bounds a hello puts on a declared rate.
        _require(
            MIN_FRAME_RATE_HZ <= self.frame_rate_hz <= MAX_FRAME_RATE_HZ,
            f"frame_rate_hz must be from {MIN_FRAME_RATE_HZ} to {MAX_FRAME_RATE_HZ}, "
            "the rates a robot may declare",
        )
        # A hello is validated against MODEL_KEYS alone, and a command names its model by
        # its index in MODEL_KEYS, so every key needs its settings here.
        _require(
            set(self.models) == set(MODEL_KEYS),
            f"models must have exactly the keys {', '.join(MODEL_KEYS)}",
        )
        # The paced worker gets what the quota leaves after every other consumer, which is
        # defined for one elastic consumer only.
        _require(
            sum(model.paced for model in self.models.values()) <= 1,
            "at most one model may be paced",
        )
        for key, model in self.models.items():
            # A worker reads a request only up to MAX_PAYLOAD_BYTES. A batch built from frames
            # that passed ingress must always fit, or one robot's legal frames would stop a
            # worker; the request encoder relies on this check and does not repeat it.
            _require(
                batch_request_bytes(model.batch_sizes[-1], self.max_message_bytes)
                <= MAX_PAYLOAD_BYTES,
                f"models.{key}: {model.batch_sizes[-1]} frames of max_message_bytes do not fit "
                f"one worker request of {MAX_PAYLOAD_BYTES} bytes",
            )
        # The skip rule judges a robot's newest frame. Under FIFO the head is the oldest,
        # and skipping it would hide the growth the control exists to show.
        _require(
            self.queue_policy is QueuePolicy.LATEST or not self.skip_rule,
            "queue_policy fifo needs skip_rule off",
        )
        # Only disconnected robots are dropped from the robot table, so it must have room
        # for every open connection.
        _require(
            self.max_robot_entries >= self.max_connections,
            "max_robot_entries must be at least max_connections",
        )
        # A full bucket must hold one message of every size the transport accepts, or a
        # legal message could never pass. The ingress limiter keeps its bucket levels in
        # exact arithmetic, so these rules are checked in it too: a float product can round
        # up onto the threshold while the exact bucket stays a rounding error below it.
        _require(
            Fraction(self.ingress_max_bytes_per_s) * Fraction(self.ingress_burst_s)
            >= self.max_message_bytes,
            "ingress_max_bytes_per_s * ingress_burst_s must be at least max_message_bytes",
        )
        _require(
            Fraction(self.ingress_max_messages_per_s) * Fraction(self.ingress_burst_s) >= 1,
            "ingress_max_messages_per_s * ingress_burst_s must be at least 1",
        )
        # A hello carries the token as JSON text and the check compares bytes.
        _require(
            self.shared_token is None
            or (
                self.shared_token != ""
                and self.shared_token.isascii()
                and self.shared_token.isprintable()
            ),
            "shared_token must be printable ASCII and not empty",
        )


def config_echo(config: ServerConfig, *, version: str, protocol_version: int) -> dict[str, object]:
    """Build the config object of /metrics (DESIGN.md 10.8): every setting but the token."""
    return {
        "version": version,
        "protocol_version": protocol_version,
        "listen": {"host": config.host, "port": config.port},
        "topology": config.topology.value,
        "shared_clock": config.shared_clock,
        "queue_policy": config.queue_policy.value,
        "fifo_capacity_frames": config.fifo_capacity_frames,
        "frame_rate_hz": config.frame_rate_hz,
        "overhead_allowance_ms": config.overhead_allowance_ms,
        "skip_rule": config.skip_rule,
        "skip_max_unserved_ms": config.skip_max_unserved_ms,
        "models": {
            key: {
                "runtime": model.runtime.value,
                "threads": model.threads,
                "batch_sizes": list(model.batch_sizes),
                "forced_batch_cap": model.forced_batch_cap,
                "paced": model.paced,
                "stale_ms": model.stale_ms,
                "watchdog_ms": model.watchdog_ms,
                "fake_frame_ms": model.fake_frame_ms,
            }
            for key, model in config.models.items()
        },
        "pacing": {
            "headroom_cores": config.pacing_headroom_cores,
            "min_share_cores": config.pacing_min_share_cores,
            "io_cpu_ms_per_frame": config.io_cpu_ms_per_frame,
        },
        "worker_nice": config.worker_nice,
        "max_connections": config.max_connections,
        "auth_required": config.shared_token is not None,
        "ingress": {
            "max_messages_per_s": config.ingress_max_messages_per_s,
            "max_bytes_per_s": config.ingress_max_bytes_per_s,
            "burst_s": config.ingress_burst_s,
            "max_message_bytes": config.max_message_bytes,
            "max_frame_age_ms": config.max_frame_age_ms,
        },
        "timeouts_ms": {
            "handshake": config.handshake_timeout_ms,
            "idle": config.idle_timeout_ms,
            "inference": config.inference_timeout_ms,
            "slow_consumer": config.slow_consumer_timeout_ms,
            "warmup": config.warmup_timeout_ms,
            "shutdown_grace": config.shutdown_grace_ms,
        },
        "socket": {
            "rcvbuf_bytes": config.socket_rcvbuf_bytes,
            "sndbuf_bytes": config.socket_sndbuf_bytes,
        },
        "downlink_queue_commands": config.downlink_queue_commands,
        "ping_interval_ms": config.ping_interval_ms,
        "robot_retention_s": config.robot_retention_s,
        "max_robot_entries": config.max_robot_entries,
        "loop_probe_ms": config.loop_probe_ms,
    }
