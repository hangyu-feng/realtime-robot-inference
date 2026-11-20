"""Tests for the server configuration: defaults, every validation rule, and the config echo."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import fields, replace
from typing import Any

import pytest

from cloud.config import (
    ModelConfig,
    ModelRuntime,
    QueuePolicy,
    ServerConfig,
    Topology,
    config_echo,
)
from common.commands import MODEL_KEYS
from common.errors import ConfigError
from common.protocol import MAX_FRAME_RATE_HZ, MIN_FRAME_RATE_HZ
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from common.worker_protocol import MAX_BATCH_FRAMES, MAX_PAYLOAD_BYTES, batch_request_bytes

# Overrides and changed values are typed Any: mypy checks the value type of a ** dict
# passed to dataclasses.replace against every field, and these cases mix field types.

DEFAULT = ServerConfig()
CLS = DEFAULT.models["cls"]
DEPTH = DEFAULT.models["depth"]

# The most of each unit that fits 2**63 - 1 ns, the range of the clock's readings.
LONGEST_MS = 9_223_372_036_854.0
LONGEST_S = 9_223_372_036.0


def models(cls: ModelConfig = CLS, depth: ModelConfig = DEPTH) -> dict[str, ModelConfig]:
    return {"cls": cls, "depth": depth}


# Both models in batches of one frame, with the largest message whose request of one frame
# still fits MAX_PAYLOAD_BYTES.
ONE_FRAME_MODELS = models(
    cls=replace(CLS, batch_sizes=(1,)), depth=replace(DEPTH, batch_sizes=(1,))
)
LARGEST_ONE_FRAME_MESSAGE_BYTES = MAX_PAYLOAD_BYTES - batch_request_bytes(1, 0)


def echo(config: ServerConfig) -> dict[str, object]:
    return config_echo(config, version="test", protocol_version=1)


def reported(config: ServerConfig, path: str) -> object:
    """Return what /metrics shows at a dotted path of the echo, such as "listen.port"."""
    document = json.loads(json.dumps(echo(config), allow_nan=False))
    for key in path.split("."):
        document = document[key]
    return document


def test_default_config_has_every_model_with_the_default_thresholds() -> None:
    assert set(DEFAULT.models) == set(MODEL_KEYS)
    assert {key: model.stale_ms for key, model in DEFAULT.models.items()} == DEFAULT_STALE_MS
    assert {key: model.watchdog_ms for key, model in DEFAULT.models.items()} == (
        DEFAULT_WATCHDOG_MS
    )


def test_default_config_paces_depth_only() -> None:
    assert [key for key, model in DEFAULT.models.items() if model.paced] == ["depth"]


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        pytest.param({"stale_ms": 0.0}, "stale_ms", id="stale-zero"),
        pytest.param({"stale_ms": float("nan")}, "stale_ms", id="stale-nan"),
        pytest.param({"stale_ms": 10**400}, "stale_ms", id="stale-int-beyond-float"),
        pytest.param(
            {"stale_ms": LONGEST_MS + 1, "watchdog_ms": LONGEST_MS + 1},
            "stale_ms",
            id="stale-beyond-clock",
        ),
        pytest.param({"watchdog_ms": float("inf")}, "watchdog_ms", id="watchdog-infinite"),
        pytest.param({"watchdog_ms": 499.0}, "watchdog_ms", id="watchdog-below-stale"),
        pytest.param({"watchdog_ms": LONGEST_MS + 1}, "watchdog_ms", id="watchdog-beyond-clock"),
        pytest.param({"threads": -1}, "threads", id="threads-negative"),
        pytest.param({"batch_sizes": ()}, "batch_sizes", id="sizes-empty"),
        pytest.param({"batch_sizes": (2, 4)}, "batch_sizes", id="sizes-without-1"),
        pytest.param({"batch_sizes": (1, 4, 2)}, "batch_sizes", id="sizes-descending"),
        pytest.param({"batch_sizes": (1, 2, 2)}, "batch_sizes", id="sizes-repeated"),
        pytest.param(
            {"batch_sizes": (1, MAX_BATCH_FRAMES + 1)}, "batch_sizes", id="sizes-above-bound"
        ),
        pytest.param({"forced_batch_cap": 3}, "forced_batch_cap", id="forced-cap-uncalibrated"),
        pytest.param({"fake_frame_ms": -1.0}, "fake_frame_ms", id="fake-time-negative"),
        pytest.param({"fake_frame_ms": 10**400}, "fake_frame_ms", id="fake-time-int-beyond-float"),
        pytest.param(
            {"fake_frame_ms": LONGEST_MS + 1}, "fake_frame_ms", id="fake-time-beyond-clock"
        ),
    ],
)
def test_model_config_refuses_a_bad_value_naming_the_field(
    overrides: dict[str, Any], field_name: str
) -> None:
    with pytest.raises(ConfigError, match=field_name):
        replace(CLS, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"watchdog_ms": 500.0}, id="watchdog-equal-to-stale"),
        pytest.param({"threads": 0}, id="runtime-default-threads"),
        pytest.param({"forced_batch_cap": 4}, id="forced-cap-calibrated"),
        pytest.param({"batch_sizes": (1,)}, id="batch-of-one-only"),
        pytest.param({"batch_sizes": (1, MAX_BATCH_FRAMES)}, id="sizes-up-to-bound"),
        pytest.param({"runtime": ModelRuntime.FAKE, "fake_frame_ms": 5.0}, id="fake-runtime"),
        pytest.param(
            {"stale_ms": LONGEST_MS, "watchdog_ms": LONGEST_MS, "fake_frame_ms": LONGEST_MS},
            id="durations-filling-the-clock",
        ),
        pytest.param({"stale_ms": 1.0, "watchdog_ms": 1.0}, id="thresholds-of-one-millisecond"),
        pytest.param({"threads": 64}, id="most-threads"),
    ],
)
def test_model_config_accepts_a_boundary_value(overrides: dict[str, Any]) -> None:
    replace(CLS, **overrides)


@pytest.mark.parametrize(
    ("overrides", "field_name"),
    [
        pytest.param({"stale_ms": math.nextafter(1.0, 0.0)}, "stale_ms", id="stale-one-step-below"),
        pytest.param({"stale_ms": 5e-7}, "stale_ms", id="stale-rounding-to-no-nanoseconds"),
        pytest.param(
            {"stale_ms": 1.0, "watchdog_ms": math.nextafter(1.0, 0.0)},
            "watchdog_ms",
            id="watchdog-one-step-below",
        ),
        pytest.param(
            {"stale_ms": 1.0, "watchdog_ms": 5e-7},
            "watchdog_ms",
            id="watchdog-rounding-to-no-nanoseconds",
        ),
    ],
)
def test_model_config_refuses_a_threshold_below_one_millisecond(
    overrides: dict[str, Any], field_name: str
) -> None:
    with pytest.raises(ConfigError, match=f"^{field_name} must be at least 1$"):
        replace(CLS, **overrides)


@pytest.mark.parametrize("threads", [65, 2**31])
def test_model_config_refuses_more_threads_than_the_most_it_may_start(threads: int) -> None:
    with pytest.raises(ConfigError, match=r"^threads must be at most 64$"):
        replace(CLS, threads=threads)


SERVER_POSITIVE_FIELDS = [
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
    "close_timeout_ms",
    "ping_timeout_ms",
    "worker_stop_timeout_ms",
    "worker_restart_min_ms",
    "worker_restart_max_ms",
    "ping_interval_ms",
    "robot_retention_s",
    "loop_probe_ms",
]
SERVER_NOT_NEGATIVE_FIELDS = [
    "overhead_allowance_ms",
    "pacing_headroom_cores",
    "io_cpu_ms_per_frame",
]
SERVER_COUNT_FIELDS = [
    "fifo_capacity_frames",
    "max_connections",
    "max_message_bytes",
    "socket_rcvbuf_bytes",
    "socket_sndbuf_bytes",
    "receive_queue_frames",
    "calibration_runs",
    "downlink_queue_commands",
    "max_robot_entries",
]
SERVER_DURATIONS = [
    ("skip_max_unserved_ms", LONGEST_MS),
    ("overhead_allowance_ms", LONGEST_MS),
    ("io_cpu_ms_per_frame", LONGEST_MS),
    ("ingress_burst_s", LONGEST_S),
    ("max_frame_age_ms", LONGEST_MS),
    ("handshake_timeout_ms", LONGEST_MS),
    ("idle_timeout_ms", LONGEST_MS),
    ("inference_timeout_ms", LONGEST_MS),
    ("slow_consumer_timeout_ms", LONGEST_MS),
    ("warmup_timeout_ms", LONGEST_MS),
    ("shutdown_grace_ms", LONGEST_MS),
    ("close_timeout_ms", LONGEST_MS),
    ("ping_timeout_ms", LONGEST_MS),
    ("worker_stop_timeout_ms", LONGEST_MS),
    ("worker_restart_min_ms", LONGEST_MS),
    ("worker_restart_max_ms", LONGEST_MS),
    ("ping_interval_ms", LONGEST_MS),
    ("robot_retention_s", LONGEST_S),
    ("loop_probe_ms", LONGEST_MS),
]
# Each duration at the longest it may be, with the settings that must grow with it: the ping
# timeout is at least twice the idle timeout, which so fills at most half the clock, and the
# restart maximum is at least the minimum.
GROWING_WITH = {
    "idle_timeout_ms": {"idle_timeout_ms": LONGEST_MS / 2, "ping_timeout_ms": LONGEST_MS},
    "worker_restart_min_ms": {
        "worker_restart_min_ms": LONGEST_MS,
        "worker_restart_max_ms": LONGEST_MS,
    },
}
LONGEST_DURATIONS = [
    pytest.param(GROWING_WITH.get(name, {name: longest}), id=name)
    for name, longest in SERVER_DURATIONS
]
# Each wait and threshold with the shortest it may be, 1 ms in its unit, and the nanoseconds
# in that unit.
SERVER_WAITS = [
    ("skip_max_unserved_ms", 1.0, 1_000_000),
    ("ingress_burst_s", 0.001, 1_000_000_000),
    ("max_frame_age_ms", 1.0, 1_000_000),
    ("handshake_timeout_ms", 1.0, 1_000_000),
    ("idle_timeout_ms", 1.0, 1_000_000),
    ("inference_timeout_ms", 1.0, 1_000_000),
    ("slow_consumer_timeout_ms", 1.0, 1_000_000),
    ("warmup_timeout_ms", 1.0, 1_000_000),
    ("shutdown_grace_ms", 1.0, 1_000_000),
    ("close_timeout_ms", 1.0, 1_000_000),
    ("ping_timeout_ms", 1.0, 1_000_000),
    ("worker_stop_timeout_ms", 1.0, 1_000_000),
    ("worker_restart_min_ms", 1.0, 1_000_000),
    ("worker_restart_max_ms", 1.0, 1_000_000),
    ("ping_interval_ms", 1.0, 1_000_000),
    ("robot_retention_s", 0.001, 1_000_000_000),
    ("loop_probe_ms", 1.0, 1_000_000),
]
# The settings that must move with a wait at its shortest. A burst of 1 ms still holds one
# message of each bucket, and the restart minimum is at most the maximum. The ping timeout is
# at least twice the idle timeout, so the shortest it can be is 2 ms.
MOVING_WITH = {
    "ingress_burst_s": {
        "ingress_max_messages_per_s": 1024.0,
        "ingress_max_bytes_per_s": float(2**30),
    },
    "ping_timeout_ms": {"idle_timeout_ms": 1.0, "ping_timeout_ms": 2.0},
    "worker_restart_max_ms": {"worker_restart_min_ms": 1.0},
}
# The most a whole-number setting may be: the C int that socket options and the runtime take.
LARGEST_COUNT = 2**31 - 1


@pytest.mark.parametrize("field_name", SERVER_POSITIVE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(0.0, id="zero"),
        pytest.param(-1.0, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(10**400, id="int-beyond-float"),
    ],
)
def test_server_config_refuses_a_rate_size_or_time_that_is_not_a_finite_number_above_zero(
    field_name: str, value: Any
) -> None:
    with pytest.raises(ConfigError, match=field_name):
        replace(DEFAULT, **{field_name: value})


@pytest.mark.parametrize("field_name", SERVER_NOT_NEGATIVE_FIELDS)
@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-0.5, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(10**400, id="int-beyond-float"),
    ],
)
def test_server_config_refuses_a_negative_or_non_finite_allowance(
    field_name: str, value: Any
) -> None:
    with pytest.raises(ConfigError, match=field_name):
        replace(DEFAULT, **{field_name: value})


@pytest.mark.parametrize("field_name", SERVER_NOT_NEGATIVE_FIELDS)
def test_server_config_accepts_a_zero_allowance(field_name: str) -> None:
    changes: dict[str, Any] = {field_name: 0.0}
    replace(DEFAULT, **changes)


@pytest.mark.parametrize("changes", LONGEST_DURATIONS)
def test_server_config_accepts_each_duration_at_the_longest_the_clock_and_its_rules_allow(
    changes: dict[str, Any],
) -> None:
    replace(DEFAULT, **changes)


@pytest.mark.parametrize(("field_name", "longest"), SERVER_DURATIONS)
def test_server_config_refuses_a_duration_beyond_the_clock(field_name: str, longest: float) -> None:
    changes: dict[str, Any] = {field_name: longest + 1}
    with pytest.raises(ConfigError, match=field_name):
        replace(DEFAULT, **changes)


@pytest.mark.parametrize(("field_name", "shortest", "ns_per_unit"), SERVER_WAITS)
def test_server_config_accepts_each_wait_at_one_millisecond_or_the_shortest_its_rules_allow(
    field_name: str, shortest: float, ns_per_unit: int
) -> None:
    changes: dict[str, Any] = {field_name: shortest, **MOVING_WITH.get(field_name, {})}
    config = replace(DEFAULT, **changes)
    assert round(getattr(config, field_name) * ns_per_unit) >= 1_000_000


@pytest.mark.parametrize(
    ("field_name", "shortest"), [(name, shortest) for name, shortest, _ in SERVER_WAITS]
)
@pytest.mark.parametrize(
    "below",
    [
        pytest.param(lambda shortest: 0.0, id="zero"),
        pytest.param(lambda shortest: 5e-7, id="5e-7-of-the-unit"),
        pytest.param(lambda shortest: math.nextafter(shortest, 0.0), id="one-step-below"),
    ],
)
def test_server_config_refuses_a_wait_below_one_millisecond(
    field_name: str, shortest: float, below: Callable[[float], float]
) -> None:
    changes: dict[str, Any] = {field_name: below(shortest)}
    with pytest.raises(ConfigError, match=f"^{field_name} must be at least {shortest:g}$"):
        replace(DEFAULT, **changes)


@pytest.mark.parametrize("value", [math.nextafter(0.01, 0.0), 5e-324])
def test_server_config_refuses_a_minimum_share_below_a_hundredth_of_a_core(value: float) -> None:
    with pytest.raises(ConfigError, match=r"^pacing_min_share_cores must be at least 0\.01$"):
        replace(DEFAULT, pacing_min_share_cores=value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(math.nextafter(MIN_FRAME_RATE_HZ, 0.0), id="below-minimum"),
        pytest.param(math.nextafter(MAX_FRAME_RATE_HZ, math.inf), id="above-maximum"),
        pytest.param(0.0, id="zero"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinite"),
        pytest.param(10**400, id="int-beyond-float"),
    ],
)
def test_server_config_refuses_a_frame_rate_that_a_robot_could_not_declare(value: Any) -> None:
    with pytest.raises(ConfigError, match="frame_rate_hz"):
        replace(DEFAULT, frame_rate_hz=value)


@pytest.mark.parametrize("value", [MIN_FRAME_RATE_HZ, MAX_FRAME_RATE_HZ])
def test_server_config_accepts_a_frame_rate_at_either_bound(value: float) -> None:
    replace(DEFAULT, frame_rate_hz=value)


@pytest.mark.parametrize("field_name", SERVER_COUNT_FIELDS)
def test_server_config_refuses_a_count_below_one(field_name: str) -> None:
    changes: dict[str, Any] = {field_name: 0}
    with pytest.raises(ConfigError, match=field_name):
        replace(DEFAULT, **changes)


@pytest.mark.parametrize("field_name", SERVER_COUNT_FIELDS)
def test_server_config_accepts_a_count_of_one(field_name: str) -> None:
    # With one connection, so that the robot table may hold one entry.
    changes: dict[str, Any] = {"max_connections": 1, field_name: 1}
    replace(DEFAULT, **changes)


@pytest.mark.parametrize("field_name", SERVER_COUNT_FIELDS)
def test_server_config_refuses_a_count_that_does_not_fit_a_c_int(field_name: str) -> None:
    changes: dict[str, Any] = {field_name: LARGEST_COUNT + 1}
    with pytest.raises(ConfigError, match=f"^{field_name} must be from 1 to 2147483647$"):
        replace(DEFAULT, **changes)


# Without max_message_bytes, which the payload of one worker request bounds far lower.
@pytest.mark.parametrize(
    "field_name", [name for name in SERVER_COUNT_FIELDS if name != "max_message_bytes"]
)
def test_server_config_accepts_a_count_that_fills_a_c_int(field_name: str) -> None:
    # With a robot table as large, so that it has room for as many connections.
    changes: dict[str, Any] = {"max_robot_entries": LARGEST_COUNT, field_name: LARGEST_COUNT}
    replace(DEFAULT, **changes)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        pytest.param({"port": -1}, "port", id="port-negative"),
        pytest.param({"port": 65_536}, "port", id="port-above-range"),
        pytest.param(
            {"ping_timeout_ms": math.nextafter(2 * DEFAULT.idle_timeout_ms, 0.0)},
            "ping_timeout_ms",
            id="ping-timeout-just-below-twice-idle",
        ),
        pytest.param(
            {"worker_restart_max_ms": math.nextafter(DEFAULT.worker_restart_min_ms, 0.0)},
            "worker_restart_max_ms",
            id="restart-maximum-below-minimum",
        ),
        pytest.param({"worker_nice": -1}, "worker_nice", id="nice-negative"),
        pytest.param({"worker_nice": 20}, "worker_nice", id="nice-above-range"),
        pytest.param({"models": {"cls": CLS}}, "models", id="model-missing"),
        pytest.param({"models": {**models(), "pose": CLS}}, "models", id="model-unknown"),
        pytest.param(
            {"models": models(cls=replace(CLS, paced=True))}, "paced", id="two-paced-models"
        ),
        pytest.param({"queue_policy": QueuePolicy.FIFO}, "skip_rule", id="fifo-with-skip-rule"),
        pytest.param(
            {"max_connections": 300}, "max_robot_entries", id="fewer-entries-than-connections"
        ),
        pytest.param(
            {"ingress_max_bytes_per_s": float(2**19)},
            "max_message_bytes",
            id="byte-bucket-below-one-message",
        ),
        # Each float product below rounds to the threshold, while the exact one is a
        # rounding error short of it.
        pytest.param(
            {"ingress_max_bytes_per_s": float(3 * 2**20), "ingress_burst_s": 1 / 3},
            "max_message_bytes",
            id="byte-bucket-a-rounding-error-below-one-message",
        ),
        pytest.param(
            {"ingress_max_messages_per_s": 0.5},
            "ingress_max_messages_per_s",
            id="message-bucket-below-one-message",
        ),
        pytest.param(
            {
                "ingress_max_messages_per_s": 30.0,
                "ingress_burst_s": 1 / 30,
                "ingress_max_bytes_per_s": float(2**26),
            },
            "ingress_max_messages_per_s",
            id="message-bucket-a-rounding-error-below-one-message",
        ),
        pytest.param(
            {
                "max_message_bytes": 2**24,
                "ingress_max_bytes_per_s": float(2**24),
            },
            "models.cls",
            id="batch-request-above-payload-bound",
        ),
        pytest.param(
            {
                "models": ONE_FRAME_MODELS,
                "max_message_bytes": LARGEST_ONE_FRAME_MESSAGE_BYTES + 1,
                "ingress_max_bytes_per_s": float(MAX_PAYLOAD_BYTES),
            },
            "models.cls",
            id="batch-request-one-byte-above-payload-bound",
        ),
        pytest.param({"shared_token": ""}, "shared_token", id="token-empty"),
        pytest.param({"shared_token": "café"}, "shared_token", id="token-not-ascii"),
        pytest.param({"shared_token": "a\nb"}, "shared_token", id="token-not-printable"),
    ],
)
def test_server_config_refuses_an_inconsistent_setting(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ConfigError, match=match):
        replace(DEFAULT, **overrides)


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"port": 0}, id="port-chosen-by-the-kernel"),
        pytest.param({"port": 65_535}, id="highest-port"),
        pytest.param(
            {"ping_timeout_ms": 2 * DEFAULT.idle_timeout_ms}, id="ping-timeout-twice-idle"
        ),
        pytest.param(
            {"worker_restart_max_ms": DEFAULT.worker_restart_min_ms},
            id="restart-maximum-equal-to-minimum",
        ),
        pytest.param({"worker_nice": 0}, id="lowest-nice"),
        pytest.param({"worker_nice": 19}, id="highest-nice"),
        pytest.param({"pacing_min_share_cores": 0.01}, id="smallest-minimum-share"),
        pytest.param({"queue_policy": QueuePolicy.FIFO, "skip_rule": False}, id="fifo-control"),
        pytest.param({"models": models(depth=replace(DEPTH, paced=False))}, id="no-paced-model"),
        pytest.param({"models": models(cls=replace(CLS, threads=0))}, id="unpaced-default-threads"),
        pytest.param(
            {"models": models(cls=replace(CLS, threads=0), depth=replace(DEPTH, threads=0))},
            id="default-threads-control",
        ),
        pytest.param({"max_connections": 256}, id="entries-equal-to-connections"),
        pytest.param({"ingress_max_bytes_per_s": float(2**20)}, id="byte-bucket-of-one-message"),
        pytest.param(
            {"ingress_max_messages_per_s": 1.0, "ingress_burst_s": 1.0},
            id="message-bucket-of-one-message",
        ),
        pytest.param(
            {
                "models": ONE_FRAME_MODELS,
                "max_message_bytes": LARGEST_ONE_FRAME_MESSAGE_BYTES,
                "ingress_max_bytes_per_s": float(MAX_PAYLOAD_BYTES),
            },
            id="batch-request-filling-the-payload-bound",
        ),
        pytest.param({"shared_token": "s3cret-token"}, id="token"),
        pytest.param({"topology": Topology.SPLIT}, id="split-topology"),
    ],
)
def test_server_config_accepts_a_consistent_setting(overrides: dict[str, Any]) -> None:
    replace(DEFAULT, **overrides)


def test_server_config_keeps_its_models_when_the_caller_changes_the_mapping_it_passed() -> None:
    passed = models()
    config = replace(DEFAULT, models=passed)
    passed["cls"] = replace(CLS, paced=True)
    passed["pose"] = CLS
    assert dict(config.models) == models()


# One change per setting, with the key of the echo that reports it and what /metrics then
# shows there, from a baseline with the skip rule off so that the FIFO policy is valid on
# its own.
BASELINE = replace(DEFAULT, skip_rule=False)
SERVER_CHANGES: list[tuple[str, Any, str, object]] = [
    ("host", "127.0.0.1", "listen.host", "127.0.0.1"),
    ("port", 9000, "listen.port", 9000),
    ("topology", Topology.SPLIT, "topology", "split"),
    ("shared_clock", False, "shared_clock", False),
    ("queue_policy", QueuePolicy.FIFO, "queue_policy", "fifo"),
    ("fifo_capacity_frames", 10, "fifo_capacity_frames", 10),
    ("skip_rule", True, "skip_rule", True),
    ("skip_max_unserved_ms", 500.0, "skip_max_unserved_ms", 500.0),
    ("frame_rate_hz", 15.0, "frame_rate_hz", 15.0),
    ("overhead_allowance_ms", 5.0, "overhead_allowance_ms", 5.0),
    ("models", models(cls=replace(CLS, threads=2)), "models.cls.threads", 2),
    ("worker_nice", 5, "worker_nice", 5),
    ("pacing_headroom_cores", 0.2, "pacing.headroom_cores", 0.2),
    ("pacing_min_share_cores", 0.5, "pacing.min_share_cores", 0.5),
    ("io_cpu_ms_per_frame", 0.3, "pacing.io_cpu_ms_per_frame", 0.3),
    ("max_connections", 32, "max_connections", 32),
    ("shared_token", "s3cret-token", "auth_required", True),
    ("ingress_max_messages_per_s", 64.0, "ingress.max_messages_per_s", 64.0),
    ("ingress_max_bytes_per_s", float(4 * 2**20), "ingress.max_bytes_per_s", float(4 * 2**20)),
    ("ingress_burst_s", 2.0, "ingress.burst_s", 2.0),
    ("max_message_bytes", 2**19, "ingress.max_message_bytes", 2**19),
    ("max_frame_age_ms", 5_000.0, "ingress.max_frame_age_ms", 5_000.0),
    ("handshake_timeout_ms", 1_000.0, "timeouts_ms.handshake", 1_000.0),
    ("idle_timeout_ms", 1_000.0, "timeouts_ms.idle", 1_000.0),
    ("inference_timeout_ms", 1_000.0, "timeouts_ms.inference", 1_000.0),
    ("slow_consumer_timeout_ms", 500.0, "timeouts_ms.slow_consumer", 500.0),
    ("warmup_timeout_ms", 60_000.0, "timeouts_ms.warmup", 60_000.0),
    ("shutdown_grace_ms", 1_000.0, "timeouts_ms.shutdown_grace", 1_000.0),
    ("close_timeout_ms", 500.0, "timeouts_ms.close", 500.0),
    ("ping_timeout_ms", 60_000.0, "timeouts_ms.ping", 60_000.0),
    ("worker_stop_timeout_ms", 1_000.0, "timeouts_ms.worker_stop", 1_000.0),
    ("worker_restart_min_ms", 250.0, "worker_restart_ms.min", 250.0),
    ("worker_restart_max_ms", 20_000.0, "worker_restart_ms.max", 20_000.0),
    ("calibration_runs", 5, "calibration_runs", 5),
    ("socket_rcvbuf_bytes", 65_536, "socket.rcvbuf_bytes", 65_536),
    ("socket_sndbuf_bytes", 65_536, "socket.sndbuf_bytes", 65_536),
    ("receive_queue_frames", 8, "receive_queue_frames", 8),
    ("downlink_queue_commands", 4, "downlink_queue_commands", 4),
    ("ping_interval_ms", 1_000.0, "ping_interval_ms", 1_000.0),
    ("robot_retention_s", 60.0, "robot_retention_s", 60.0),
    ("max_robot_entries", 128, "max_robot_entries", 128),
    ("loop_probe_ms", 10.0, "loop_probe_ms", 10.0),
]
# One change per model setting, with what /metrics then shows under models.<m>.<setting>.
# Each value is valid for either model.
MODEL_CHANGES: list[tuple[str, Any, object]] = [
    ("stale_ms", 100.0, 100.0),
    ("watchdog_ms", 2_000.0, 2_000.0),
    ("runtime", ModelRuntime.NULL, "null"),
    ("threads", 2, 2),
    ("batch_sizes", (1, 2), [1, 2]),
    ("forced_batch_cap", 2, 2),
    ("paced", True, True),
    ("fake_frame_ms", 1.0, 1.0),
]


def test_every_server_setting_has_an_echo_case() -> None:
    assert [name for name, _, _, _ in SERVER_CHANGES] == [
        field.name for field in fields(ServerConfig)
    ]


def test_every_model_setting_has_an_echo_case() -> None:
    assert [name for name, _, _ in MODEL_CHANGES] == [field.name for field in fields(ModelConfig)]


def test_echo_of_the_default_config_lists_every_setting_with_its_default() -> None:
    assert echo(DEFAULT) == {
        "version": "test",
        "protocol_version": 1,
        "listen": {"host": "0.0.0.0", "port": 8080},
        "topology": "single",
        "shared_clock": True,
        "queue_policy": "latest",
        "fifo_capacity_frames": 900,
        "frame_rate_hz": 30.0,
        "overhead_allowance_ms": 10.0,
        "skip_rule": True,
        "skip_max_unserved_ms": 1_000.0,
        "models": {
            "cls": {
                "runtime": "onnx",
                "threads": 1,
                "batch_sizes": [1, 2, 4, 8],
                "forced_batch_cap": None,
                "paced": False,
                "stale_ms": DEFAULT_STALE_MS["cls"],
                "watchdog_ms": DEFAULT_WATCHDOG_MS["cls"],
                "fake_frame_ms": 0.0,
            },
            "depth": {
                "runtime": "onnx",
                "threads": 1,
                "batch_sizes": [1, 2, 4, 8],
                "forced_batch_cap": None,
                "paced": True,
                "stale_ms": DEFAULT_STALE_MS["depth"],
                "watchdog_ms": DEFAULT_WATCHDOG_MS["depth"],
                "fake_frame_ms": 0.0,
            },
        },
        "pacing": {"headroom_cores": 0.1, "min_share_cores": 0.25, "io_cpu_ms_per_frame": 0.26},
        "worker_nice": 0,
        "max_connections": 64,
        "auth_required": False,
        "ingress": {
            "max_messages_per_s": 128.0,
            "max_bytes_per_s": 8 * 2**20,
            "burst_s": 1.0,
            "max_message_bytes": 2**20,
            "max_frame_age_ms": 10_000.0,
        },
        "timeouts_ms": {
            "handshake": 5_000.0,
            "idle": 3_000.0,
            "inference": 5_000.0,
            "slow_consumer": 1_000.0,
            "warmup": 120_000.0,
            "shutdown_grace": 2_000.0,
            "close": 1_000.0,
            "ping": 30_000.0,
            "worker_stop": 2_000.0,
        },
        "socket": {"rcvbuf_bytes": 40_960, "sndbuf_bytes": 4096},
        "receive_queue_frames": 4,
        "downlink_queue_commands": 8,
        "calibration_runs": 3,
        "worker_restart_ms": {"min": 500.0, "max": 10_000.0},
        "ping_interval_ms": 2_000.0,
        "robot_retention_s": 300.0,
        "max_robot_entries": 256,
        "loop_probe_ms": 20.0,
    }


@pytest.mark.parametrize(("field_name", "value", "path", "shown"), SERVER_CHANGES)
def test_echo_reports_each_server_setting_under_its_own_key(
    field_name: str, value: Any, path: str, shown: object
) -> None:
    assert reported(BASELINE, path) != shown
    assert reported(replace(BASELINE, **{field_name: value}), path) == shown


@pytest.mark.parametrize("key", MODEL_KEYS)
@pytest.mark.parametrize(("field_name", "value", "shown"), MODEL_CHANGES)
def test_echo_reports_each_model_setting_under_its_own_key(
    key: str, field_name: str, value: Any, shown: object
) -> None:
    # Neither model is paced in the baseline, so that either may be.
    unpaced = {name: replace(model, paced=False) for name, model in DEFAULT.models.items()}
    changed = {**unpaced, key: replace(unpaced[key], **{field_name: value})}
    path = f"models.{key}.{field_name}"
    assert reported(replace(BASELINE, models=unpaced), path) != shown
    assert reported(replace(BASELINE, models=changed), path) == shown


def test_echo_reports_whether_a_token_is_required_and_never_the_token() -> None:
    token = "s3cret-token"
    with_token = echo(replace(DEFAULT, shared_token=token))
    assert with_token["auth_required"] is True
    assert echo(DEFAULT)["auth_required"] is False
    assert token not in json.dumps(with_token)
