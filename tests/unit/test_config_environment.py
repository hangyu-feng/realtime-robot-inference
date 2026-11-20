"""Tests for building the server configuration from the environment."""

from __future__ import annotations

import json
import random
import re
from dataclasses import fields, replace

import pytest

from cloud.config import (
    ModelConfig,
    ModelRuntime,
    QueuePolicy,
    ServerConfig,
    Topology,
    load_config,
)
from common.backoff import Backoff
from common.commands import MODEL_KEYS
from common.errors import ConfigError

DEFAULT = ServerConfig()
CLS = DEFAULT.models["cls"]
DEPTH = DEFAULT.models["depth"]

# Every setting but these two takes a plain JSON value: the topology has its own variable,
# and the models take an object per model.
PLAIN_SETTINGS = [
    field.name for field in fields(ServerConfig) if field.name not in {"topology", "models"}
]
MODEL_SETTINGS = [field.name for field in fields(ModelConfig)]


def overrides(document: object) -> dict[str, str]:
    return {"RRI_OVERRIDES": json.dumps(document)}


def test_an_empty_environment_gives_the_default_config() -> None:
    assert load_config({}) == DEFAULT


@pytest.mark.parametrize("topology", list(Topology))
def test_the_topology_variable_sets_the_topology(topology: Topology) -> None:
    assert load_config({"RRI_TOPOLOGY": topology.value}) == replace(DEFAULT, topology=topology)


def test_an_empty_topology_variable_gives_the_default_topology() -> None:
    assert load_config({"RRI_TOPOLOGY": ""}) == DEFAULT


@pytest.mark.parametrize("value", ["both", "SPLIT", " single"])
def test_an_unknown_topology_is_refused_naming_the_variable(value: str) -> None:
    with pytest.raises(ConfigError, match=r"^RRI_TOPOLOGY must be one of single, split$"):
        load_config({"RRI_TOPOLOGY": value})


@pytest.mark.parametrize("value", ["", "   ", "\n\t "])
def test_blank_overrides_give_the_default_config(value: str) -> None:
    assert load_config({"RRI_OVERRIDES": value}) == DEFAULT


def test_topology_and_overrides_apply_together() -> None:
    environment = {"RRI_TOPOLOGY": "split", **overrides({"skip_rule": False})}
    assert load_config(environment) == replace(DEFAULT, topology=Topology.SPLIT, skip_rule=False)


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        pytest.param({"skip_rule": False}, replace(DEFAULT, skip_rule=False), id="bool"),
        pytest.param({"port": 9000}, replace(DEFAULT, port=9000), id="int"),
        pytest.param(
            {"idle_timeout_ms": 2000}, replace(DEFAULT, idle_timeout_ms=2000.0), id="int-for-float"
        ),
        pytest.param(
            {"idle_timeout_ms": 2000.5},
            replace(DEFAULT, idle_timeout_ms=2000.5),
            id="float-for-float",
        ),
        pytest.param({"host": "127.0.0.1"}, replace(DEFAULT, host="127.0.0.1"), id="str"),
        pytest.param(
            {"queue_policy": "fifo", "skip_rule": False},
            replace(DEFAULT, queue_policy=QueuePolicy.FIFO, skip_rule=False),
            id="string-for-enum",
        ),
        pytest.param(
            {"shared_token": "s3cret-token"},
            replace(DEFAULT, shared_token="s3cret-token"),
            id="value-for-optional",
        ),
        pytest.param({"shared_token": None}, DEFAULT, id="null-for-optional"),
        pytest.param(
            {"models": {"depth": {"threads": 2}}},
            replace(DEFAULT, models={"cls": CLS, "depth": replace(DEPTH, threads=2)}),
            id="model-setting-merged-over-its-default",
        ),
        pytest.param(
            {"models": {"cls": {"batch_sizes": [1, 2]}}},
            replace(DEFAULT, models={"cls": replace(CLS, batch_sizes=(1, 2)), "depth": DEPTH}),
            id="list-for-tuple",
        ),
        pytest.param(
            {"models": {"cls": {"forced_batch_cap": 4}}},
            replace(DEFAULT, models={"cls": replace(CLS, forced_batch_cap=4), "depth": DEPTH}),
            id="integer-for-optional-integer",
        ),
        pytest.param(
            {"models": {"cls": {"forced_batch_cap": None}}}, DEFAULT, id="null-for-optional-integer"
        ),
        pytest.param(
            {
                "models": {
                    "cls": {"runtime": "fake", "fake_frame_ms": 2},
                    "depth": {"runtime": "fake", "fake_frame_ms": 40},
                }
            },
            replace(
                DEFAULT,
                models={
                    "cls": replace(CLS, runtime=ModelRuntime.FAKE, fake_frame_ms=2.0),
                    "depth": replace(DEPTH, runtime=ModelRuntime.FAKE, fake_frame_ms=40.0),
                },
            ),
            id="both-models",
        ),
        pytest.param({"models": {}}, DEFAULT, id="no-model-changed"),
        pytest.param({}, DEFAULT, id="empty-object"),
    ],
)
def test_overrides_are_converted_by_the_declared_type_and_merged_over_the_defaults(
    document: object, expected: ServerConfig
) -> None:
    assert load_config(overrides(document)) == expected


def test_an_integer_given_for_a_float_setting_becomes_a_float() -> None:
    assert type(load_config(overrides({"idle_timeout_ms": 2000})).idle_timeout_ms) is float


@pytest.mark.parametrize("name", PLAIN_SETTINGS)
def test_every_setting_but_the_topology_takes_its_default_as_json(name: str) -> None:
    assert load_config(overrides({name: getattr(DEFAULT, name)})) == DEFAULT


@pytest.mark.parametrize("key", MODEL_KEYS)
@pytest.mark.parametrize("name", MODEL_SETTINGS)
def test_every_model_setting_takes_its_default_as_json(key: str, name: str) -> None:
    default = getattr(DEFAULT.models[key], name)
    assert load_config(overrides({"models": {key: {name: default}}})) == DEFAULT


@pytest.mark.parametrize(
    ("text", "message"),
    [
        pytest.param('{"port": true}', "port must be an integer", id="bool-for-int"),
        pytest.param('{"port": 80.0}', "port must be an integer", id="float-for-int"),
        pytest.param('{"port": "80"}', "port must be an integer", id="string-for-int"),
        pytest.param('{"port": null}', "port must be an integer", id="null-for-int"),
        pytest.param(
            '{"idle_timeout_ms": true}', "idle_timeout_ms must be a number", id="bool-for-float"
        ),
        pytest.param(
            '{"idle_timeout_ms": "3000"}',
            "idle_timeout_ms must be a number",
            id="string-for-float",
        ),
        pytest.param(
            '{"idle_timeout_ms": ' + "1" * 400 + "}",
            "idle_timeout_ms must be a finite number",
            id="integer-beyond-float",
        ),
        pytest.param(
            '{"idle_timeout_ms": 1e400}',
            "idle_timeout_ms must be a finite number",
            id="float-beyond-range",
        ),
        pytest.param('{"skip_rule": 1}', "skip_rule must be true or false", id="int-for-bool"),
        pytest.param(
            '{"skip_rule": "false"}', "skip_rule must be true or false", id="str-for-bool"
        ),
        pytest.param('{"host": 1}', "host must be a string", id="int-for-str"),
        pytest.param(
            '{"queue_policy": "lifo"}',
            "queue_policy must be one of latest, fifo",
            id="unknown-enum-value",
        ),
        pytest.param(
            '{"queue_policy": 1}', "queue_policy must be one of latest, fifo", id="int-for-enum"
        ),
        pytest.param(
            '{"shared_token": 1}', "shared_token must be a string", id="int-for-optional-str"
        ),
        pytest.param(
            '{"models": {"depth": {"threads": "2"}}}',
            "models.depth.threads must be an integer",
            id="model-setting-of-the-wrong-type",
        ),
        pytest.param(
            '{"models": {"cls": {"batch_sizes": 1}}}',
            "models.cls.batch_sizes must be a list of integers",
            id="int-for-tuple",
        ),
        pytest.param(
            '{"models": {"cls": {"batch_sizes": [true, 2]}}}',
            "models.cls.batch_sizes must be a list of integers",
            id="list-of-bools-for-tuple",
        ),
        pytest.param(
            '{"models": {"cls": {"batch_sizes": [1, 2.0]}}}',
            "models.cls.batch_sizes must be a list of integers",
            id="list-of-floats-for-tuple",
        ),
        pytest.param(
            '{"models": {"cls": {"forced_batch_cap": 4.0}}}',
            "models.cls.forced_batch_cap must be an integer",
            id="float-for-optional-integer",
        ),
        pytest.param(
            '{"models": {"cls": {"runtime": "gpu"}}}',
            "models.cls.runtime must be one of onnx, fake, null",
            id="unknown-model-enum-value",
        ),
        pytest.param('{"speed": 1}', "speed is not a setting", id="unknown-key"),
        pytest.param(
            '{"models": {"pose": {}}}',
            "models.pose is not a model; the models are cls, depth",
            id="unknown-model-key",
        ),
        pytest.param(
            '{"models": {"cls": {"speed": 1}}}',
            "models.cls.speed is not a model setting",
            id="unknown-model-setting",
        ),
        pytest.param(
            '{"models": []}', "models must be an object keyed by model", id="models-not-object"
        ),
        pytest.param(
            '{"models": {"cls": 1}}', "models.cls must be an object", id="model-not-object"
        ),
        pytest.param(
            '{"models": {"cls": {"threads": -1}}}',
            "models.cls: threads must be 0",
            id="model-rule-after-merge",
        ),
        pytest.param(
            '{"topology": "split"}', "topology is set by RRI_TOPOLOGY only", id="topology"
        ),
        pytest.param("[1]", "must be a JSON object", id="not-an-object"),
        pytest.param('"text"', "must be a JSON object", id="string-not-an-object"),
        pytest.param("{", "not valid JSON", id="invalid-json"),
        pytest.param('{"port": ' + "1" * 5000 + "}", "not valid JSON", id="5000-digit-integer"),
        pytest.param("[" * 100_000 + "]" * 100_000, "not valid JSON", id="deep-nesting"),
    ],
)
def test_bad_overrides_are_refused_naming_the_variable_and_the_key_path(
    text: str, message: str
) -> None:
    with pytest.raises(ConfigError, match=f"^RRI_OVERRIDES: {re.escape(message)}"):
        load_config({"RRI_OVERRIDES": text})


@pytest.mark.parametrize(
    ("document", "message"),
    [
        pytest.param(
            {"worker_restart_min_ms": 1e-7, "worker_restart_max_ms": 1e-7},
            "worker_restart_min_ms must be at least 1",
            id="restart-pauses-below-one-millisecond",
        ),
        pytest.param(
            {"loop_probe_ms": 1e-6},
            "loop_probe_ms must be at least 1",
            id="probe-period-of-one-nanosecond",
        ),
        pytest.param(
            {"robot_retention_s": 0.0005},
            "robot_retention_s must be at least 0.001",
            id="retention-below-one-millisecond",
        ),
        pytest.param(
            {"models": {"depth": {"stale_ms": 1e-7}}},
            "models.depth: stale_ms must be at least 1",
            id="threshold-below-one-millisecond",
        ),
        pytest.param(
            {"socket_rcvbuf_bytes": 2**31},
            "socket_rcvbuf_bytes must be from 1 to 2147483647",
            id="buffer-size-beyond-a-c-int",
        ),
        pytest.param(
            {"calibration_runs": 10**30},
            "calibration_runs must be from 1 to 2147483647",
            id="count-beyond-a-c-int",
        ),
        pytest.param(
            {"models": {"cls": {"threads": 2**31}}},
            "models.cls: threads must be at most 64",
            id="threads-beyond-a-c-int",
        ),
    ],
)
def test_an_override_outside_the_range_of_its_setting_is_refused_naming_the_field(
    document: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=f"^RRI_OVERRIDES: {re.escape(message)}$"):
        load_config(overrides(document))


def test_the_shortest_restart_pauses_make_a_backoff() -> None:
    config = load_config(overrides({"worker_restart_min_ms": 1, "worker_restart_max_ms": 1}))
    backoff = Backoff(
        round(config.worker_restart_min_ms * 1_000_000),
        round(config.worker_restart_max_ms * 1_000_000),
        random.Random(0),
    )
    assert 500_000 <= backoff.next_ns() <= 1_000_000


def test_a_cross_field_rule_applies_after_the_merge() -> None:
    with pytest.raises(
        ConfigError, match=r"^RRI_OVERRIDES: queue_policy fifo needs skip_rule off$"
    ):
        load_config(overrides({"queue_policy": "fifo"}))
