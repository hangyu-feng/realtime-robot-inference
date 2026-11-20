"""Tests for the robot's settings: the defaults, and every rule that refuses a value."""

import math
import random
from typing import Any

import pytest

from common.backoff import Backoff
from common.errors import ConfigError
from common.protocol import MAX_THRESHOLD_MS
from robot.settings import LinkSettings, RobotSettings


def test_the_defaults_subscribe_to_both_models_at_30_fps_with_the_default_thresholds() -> None:
    settings = RobotSettings()

    assert settings.url == "ws://127.0.0.1:8080/ws"
    assert settings.models == ("cls", "depth")
    assert settings.stale_ms == {"cls": 500.0, "depth": 150.0}
    assert settings.watchdog_ms == {"cls": 1000.0, "depth": 300.0}
    assert settings.frame_rate_hz == 30.0
    assert settings.token is None
    assert settings.uplink == LinkSettings(0.0, 0.0)
    assert settings.downlink == LinkSettings(0.0, 0.0)


def test_the_default_waits_and_byte_sizes() -> None:
    settings = RobotSettings()

    assert (
        settings.connect_timeout_ms,
        settings.welcome_timeout_ms,
        settings.close_timeout_ms,
        settings.stall_timeout_ms,
        settings.reconnect_min_ms,
        settings.reconnect_max_ms,
    ) == (5000.0, 5000.0, 1000.0, 3000.0, 100.0, 5000.0)
    assert (
        settings.sndbuf_bytes,
        settings.notsent_lowat_bytes,
        settings.rcvbuf_bytes,
        settings.max_message_bytes,
    ) == (40_960, 16_384, 4096, 4096)


def test_a_backoff_builds_from_the_default_reconnect_pauses() -> None:
    settings = RobotSettings()

    backoff = Backoff(
        round(settings.reconnect_min_ms * 1_000_000),
        round(settings.reconnect_max_ms * 1_000_000),
        random.Random(0),
    )

    assert 50_000_000 <= backoff.next_ns() <= 100_000_000


def test_a_backoff_builds_from_the_shortest_reconnect_pauses() -> None:
    settings = RobotSettings(reconnect_min_ms=1.0, reconnect_max_ms=1.0)

    backoff = Backoff(
        round(settings.reconnect_min_ms * 1_000_000),
        round(settings.reconnect_max_ms * 1_000_000),
        random.Random(0),
    )

    assert 500_000 <= backoff.next_ns() <= 1_000_000


def test_a_robot_may_subscribe_to_one_model_with_that_model_s_thresholds() -> None:
    settings = RobotSettings(
        models=("depth",), stale_ms={"depth": 150.0}, watchdog_ms={"depth": 150.0}
    )

    assert settings.models == ("depth",)


def test_the_settings_do_not_share_the_caller_s_threshold_mapping() -> None:
    stale_ms = {"cls": 500.0, "depth": 150.0}
    settings = RobotSettings(stale_ms=stale_ms)

    stale_ms["depth"] = 99_999.0

    assert settings.stale_ms == {"cls": 500.0, "depth": 150.0}


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"url": "http://127.0.0.1:8080/ws"}, "url must be"),
        ({"url": "wss://127.0.0.1:8080/ws"}, "url must be"),
        ({"url": "ws:///ws"}, "url must be"),
        ({"url": "ws://127.0.0.1:port/ws"}, "url must be"),
        ({"url": "ws://127.0.0.1:0/ws"}, "url must be"),
        ({"url": "ws://127.0.0.1:9/ws#frag"}, "url must be"),
        ({"url": "ws://someone@127.0.0.1:9/ws"}, "url must be"),
        ({"url": "ws://someone:secret@127.0.0.1:9/ws"}, "url must be"),
        ({"models": ()}, "models must be"),
        ({"models": ("cls", "cls")}, "models must be"),
        ({"models": ("cls", "depth", "pose")}, "models must be"),
        ({"stale_ms": {"cls": 500.0}}, "stale_ms must have exactly"),
        ({"watchdog_ms": {"cls": 1000.0, "depth": 300.0, "pose": 1.0}}, "watchdog_ms must have"),
        ({"stale_ms": {"cls": 0.0, "depth": 150.0}}, "stale_ms.cls must be"),
        ({"stale_ms": {"cls": -1.0, "depth": 150.0}}, "stale_ms.cls must be"),
        ({"stale_ms": {"cls": 500.0, "depth": math.nan}}, "stale_ms.depth must be"),
        ({"watchdog_ms": {"cls": 1000.0, "depth": math.inf}}, "watchdog_ms.depth must be"),
        (
            {"watchdog_ms": {"cls": math.nextafter(MAX_THRESHOLD_MS, math.inf), "depth": 300.0}},
            "watchdog_ms.cls must be",
        ),
        ({"watchdog_ms": {"cls": math.nan, "depth": 300.0}}, "watchdog_ms.cls must be"),
        (
            {"stale_ms": {"cls": 500.0, "depth": 300.5}},
            "watchdog_ms.depth must be at least stale_ms.depth",
        ),
        ({"frame_rate_hz": 0.5}, "frame_rate_hz must be"),
        ({"frame_rate_hz": 60.5}, "frame_rate_hz must be"),
        ({"frame_rate_hz": math.nan}, "frame_rate_hz must be"),
        ({"token": ""}, "token must be"),
        ({"token": "café"}, "token must be"),
        ({"token": "line\nbreak"}, "token must be"),
        ({"connect_timeout_ms": 0.999}, "connect_timeout_ms must be at least 1"),
        ({"welcome_timeout_ms": 0.0}, "welcome_timeout_ms must be at least 1"),
        ({"close_timeout_ms": -1.0}, "close_timeout_ms must be at least 1"),
        ({"stall_timeout_ms": math.nan}, "stall_timeout_ms must be at least 1"),
        ({"reconnect_min_ms": 4.9e-7}, "reconnect_min_ms must be at least 1"),
        ({"reconnect_max_ms": 0.5}, "reconnect_max_ms must be at least 1"),
        (
            {"reconnect_min_ms": 200.0, "reconnect_max_ms": 100.0},
            "reconnect_max_ms must be at least reconnect_min_ms",
        ),
        ({"sndbuf_bytes": 0}, "sndbuf_bytes must be at least 1"),
        ({"notsent_lowat_bytes": 0}, "notsent_lowat_bytes must be at least 1"),
        ({"rcvbuf_bytes": -4096}, "rcvbuf_bytes must be at least 1"),
        ({"max_message_bytes": 0}, "max_message_bytes must be at least 1"),
    ],
)
def test_a_value_outside_its_rule_is_refused_naming_the_field(
    changes: dict[str, Any], message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        RobotSettings(**changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"stale_ms": {"cls": 500.0, "depth": 5e-324}},
        {"watchdog_ms": {"cls": MAX_THRESHOLD_MS, "depth": 300.0}},
        {"stale_ms": {"cls": 500.0, "depth": 300.0}},
        {"frame_rate_hz": 1.0},
        {"frame_rate_hz": 60.0},
        {"token": "s3cret token"},
        {"connect_timeout_ms": 1.0},
        {"reconnect_min_ms": 250.0, "reconnect_max_ms": 250.0},
        {"rcvbuf_bytes": 1},
        {"url": "ws://system/ws"},
    ],
)
def test_a_value_at_the_edge_of_its_rule_is_accepted(changes: dict[str, Any]) -> None:
    settings = RobotSettings(**changes)

    assert {name: getattr(settings, name) for name in changes} == changes


@pytest.mark.parametrize(
    ("delay_ms", "jitter_ms", "message"),
    [
        (-0.001, 0.0, "delay_ms must be"),
        (math.nan, 0.0, "delay_ms must be"),
        (math.inf, 0.0, "delay_ms must be"),
        (0.0, -1.0, "jitter_ms must be"),
        (0.0, math.nan, "jitter_ms must be"),
        (0.0, math.inf, "jitter_ms must be"),
    ],
)
def test_a_link_delay_or_jitter_that_is_negative_or_not_finite_is_refused(
    delay_ms: float, jitter_ms: float, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        LinkSettings(delay_ms, jitter_ms)


@pytest.mark.parametrize(
    ("delay_ms", "jitter_ms", "enabled"),
    [(0.0, 0.0, False), (20.0, 0.0, True), (0.0, 5.0, True), (20.0, 5.0, True)],
)
def test_a_direction_has_a_link_model_only_with_a_delay_or_a_jitter(
    delay_ms: float, jitter_ms: float, enabled: bool
) -> None:
    assert LinkSettings(delay_ms, jitter_ms).enabled is enabled
