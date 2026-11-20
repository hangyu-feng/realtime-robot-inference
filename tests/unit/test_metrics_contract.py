"""Tests that a rendered /metrics snapshot has exactly the fields DESIGN.md section 10 states."""

from collections.abc import Callable
from pathlib import Path

import pytest

from common.protocol import encode_telemetry
from tests.support.contract import (
    contract_fields,
    contract_types,
    snapshot_fields,
    type_violations,
)
from tests.unit.test_metrics import (
    TELEMETRY,
    World,
    depth_down_world,
    serving_world,
    warming_world,
)

DESIGN = (Path(__file__).resolve().parents[2] / "DESIGN.md").read_text(encoding="utf-8")

# A design text with the shapes the parser reads, around text it must pass over.
DESIGN_SAMPLE = """
## 10. Metrics contract

`GET /metrics`, in prose before the tables, names no field.

### 10.1 Conventions

| Field | Meaning |
|---|---|
| `not_a_field` | A table outside the contract's sections |

### 10.2 Top level

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `schema` | int | constant | Version, see `elsewhere` |
| `uptime_s` | float or null | gauge | |
| `robots` | object keyed by robot id | | |
| `config` | object | constant | |

### 10.3 `robots.<id>`: server-side fields

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `bytes_in`, `bytes_out` | int | counter | Two fields in one row |
| `models.<m>` | object | | Below |
| `telemetry` | object | | |

`robots.<id>.models.<m>`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `in_flight` | int | gauge | |

```
in_flight = `no_field_in_a_code_block`
```

`in_flight = 0`. A sentence that starts with code is not a table's label.

### 10.4 `robots.<id>.telemetry`: robot-reported fields

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `frames_sent` | int | counter | |
| `models.<m>.commands_received` | int | counter | |
| `models.<m>.freshness_ms` | H | histogram | |

### 10.7 Connections, cgroup

`connections`:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `closed` | object, reason to int | counter | Reasons: `client_close`, `abnormal` |

`cgroup`, read from the container's own files:

| Field | Type | Kind | Meaning |
|---|---|---|---|
| `version` | int | constant | |

### 10.8 `config`: config echo

| Field | Meaning |
|---|---|
| `protocol_version` | 1 |
| `listen` | `host`, `port` |
| `topology` | `single` or `split` |
| `queue_policy`, `fifo_capacity_frames` | `latest` or `fifo`, and the queue bound |
| `models.<m>` | `runtime`, `forced_batch_cap` (null unless `something` sets it), `paced` |
| `worker_restart_ms` | `min`, `max`: the bounds of the backoff, in `ms` |
| `shared_clock` | Whether the clocks agree |
| `rcvbuf_bytes`, `sndbuf_bytes` | `requested`, `granted` |

### 10.9 `later`: a subsection after the config echo

| Field | Meaning |
|---|---|
| `not_in_the_contract` | |

## 11. Run-output contract

| Field | Meaning |
|---|---|
| `not_a_metric` | A table after the contract |

### 11.2 `scenarios`: not part of it

| Field | Meaning |
|---|---|
| `not_a_metric_either` | |
"""


def all_reporting_world() -> World:
    """Return the serving gateway after every robot, the disconnected one included, reported."""
    world = serving_world()
    report = encode_telemetry(TELEMETRY)
    now_ns = world.clock.now_ns()
    world.table.ingest(world.table.sessions["b"], report, now_ns, None)
    # The record of c keeps the report of the session it ended with.
    world.registry.get("c").telemetry = world.registry.get("a").telemetry
    return world


def test_contract_fields_reads_each_table_into_the_path_of_its_object() -> None:
    assert contract_fields(DESIGN_SAMPLE) == {
        "": frozenset(("schema", "uptime_s", "robots", "config")),
        "robots.<id>": frozenset(("bytes_in", "bytes_out", "models", "telemetry")),
        "robots.<id>.models.<m>": frozenset(("in_flight",)),
        "robots.<id>.telemetry": frozenset(("frames_sent", "models")),
        "robots.<id>.telemetry.models.<m>": frozenset(("commands_received", "freshness_ms")),
        "connections": frozenset(("closed",)),
        "cgroup": frozenset(("version",)),
        "config": frozenset(
            (
                "protocol_version",
                "listen",
                "topology",
                "queue_policy",
                "fifo_capacity_frames",
                "models",
                "worker_restart_ms",
                "shared_clock",
                "rcvbuf_bytes",
                "sndbuf_bytes",
            )
        ),
        "config.listen": frozenset(("host", "port")),
        "config.models.<m>": frozenset(("runtime", "forced_batch_cap", "paced")),
        "config.worker_restart_ms": frozenset(("min", "max")),
    }


def test_contract_types_reads_the_type_of_every_scalar_outside_the_config_echo() -> None:
    assert contract_types(DESIGN_SAMPLE) == {
        "": {"schema": "int", "uptime_s": "float or null"},
        "robots.<id>": {"bytes_in": "int", "bytes_out": "int"},
        "robots.<id>.models.<m>": {"in_flight": "int"},
        "robots.<id>.telemetry": {"frames_sent": "int"},
        "robots.<id>.telemetry.models.<m>": {"commands_received": "int"},
        "cgroup": {"version": "int"},
    }


def test_type_violations_tells_a_bool_from_an_int_and_an_int_from_a_float() -> None:
    types = {
        "": {"schema": "int", "uptime_s": "float or null", "ratio": "float"},
        "robots.<id>.models.<m>": {"in_flight": "int", "rtt_ms": "float or null"},
        "cgroup": {"version": "int"},
    }
    snapshot: dict[str, object] = {
        "schema": True,
        "uptime_s": None,
        "ratio": 3,
        "robots": {"a": {"models": {"cls": {"in_flight": 1, "rtt_ms": 2}}}},
        "cgroup": {"version": None},
    }

    assert type_violations(snapshot, types) == [
        "schema: True is not int",
        "ratio: 3 is not float",
        "robots.<id>.models.<m>.rtt_ms: 2 is not float or null",
        "cgroup.version: None is not int",
    ]


def test_snapshot_fields_reads_every_instance_into_one_path_and_passes_over_values() -> None:
    snapshot: dict[str, object] = {
        "schema": 1,
        "robots": {
            "a": {
                "session": 1,
                "stale_ms": {"cls": 150.0},
                "models": {"cls": {"in_flight": 0}, "depth": {"in_flight": 1}},
                "telemetry": {
                    "fire_lag_ms": {"count": 0, "buckets": {}},
                    "models": {"cls": {"commands_received": 3}},
                },
            },
            "b": {"session": 2, "connected": True, "models": {}},
        },
        "models": {"cls": {"batches": 1, "batch_size": {"1": 1}}},
        "stages": {"depth": {"slot_wait_ms": {"count": 0}}},
        "connections": {"closed": {"client_close": 0}},
        "workers": {"cls": {"errors": {"decode": 0}, "t_b_ms": None}},
        "cgroup": {"version": 2},
        "processes": {"gateway": {"pid": 10}, "worker_cls": None},
        "loop": {"lag_ms": {"count": 0}},
        "config": {
            "version": "test",
            "listen": {"host": "0.0.0.0", "port": 8080},
            "models": {"cls": {"threads": 1}, "depth": {"paced": True}},
        },
    }

    assert snapshot_fields(snapshot) == {
        "": frozenset(snapshot),
        "robots.<id>": frozenset(("session", "stale_ms", "models", "telemetry", "connected")),
        "robots.<id>.models.<m>": frozenset(("in_flight",)),
        "robots.<id>.telemetry": frozenset(("fire_lag_ms", "models")),
        "robots.<id>.telemetry.models.<m>": frozenset(("commands_received",)),
        "models.<m>": frozenset(("batches", "batch_size")),
        "stages.<m>": frozenset(("slot_wait_ms",)),
        "connections": frozenset(("closed",)),
        "workers.<m>": frozenset(("errors", "t_b_ms")),
        "cgroup": frozenset(("version",)),
        "processes.<p>": frozenset(("pid",)),
        "loop": frozenset(("lag_ms",)),
        "config": frozenset(("version", "listen", "models")),
        "config.listen": frozenset(("host", "port")),
        "config.models.<m>": frozenset(("threads", "paced")),
    }


def test_snapshot_fields_refuses_a_value_where_the_contract_has_an_object() -> None:
    with pytest.raises(ValueError, match="connections is not an object"):
        snapshot_fields({"connections": 3})


def test_a_snapshot_of_full_state_has_exactly_the_fields_of_the_design_at_every_path() -> None:
    snapshot = all_reporting_world().snapshot()

    assert snapshot_fields(snapshot) == contract_fields(DESIGN)


@pytest.mark.parametrize("robot_id", ["a", "b", "c"])
def test_each_robot_entry_alone_has_exactly_the_fields_of_the_design(robot_id: str) -> None:
    snapshot = all_reporting_world().snapshot()
    robots = snapshot["robots"]
    assert isinstance(robots, dict)

    alone = snapshot | {"robots": {robot_id: robots[robot_id]}}

    assert snapshot_fields(alone) == contract_fields(DESIGN)


@pytest.mark.parametrize(
    "build",
    [all_reporting_world, warming_world, depth_down_world],
    ids=["full-state", "warm-up", "depth-down"],
)
def test_every_scalar_of_a_snapshot_has_the_json_type_of_the_design(
    build: Callable[[], World],
) -> None:
    types = contract_types(DESIGN)

    violations = type_violations(build().snapshot(), types)

    assert types["robots.<id>.models.<m>"]["frames_pending"] == "int"
    assert types["robots.<id>"]["rtt_ms"] == "float or null"
    assert types["cgroup"]["memory_sock_bytes"] == "int or null"
    assert violations == []


def test_a_snapshot_taken_during_warm_up_has_the_fields_of_the_design_at_every_path_it_has() -> (
    None
):
    contract = contract_fields(DESIGN)

    fields = snapshot_fields(warming_world().snapshot())

    # Nothing is connected and the depth worker has no process yet, so no robot path exists;
    # every other object is whole, with nulls where a value is not known yet.
    assert fields == {path: names for path, names in contract.items() if "robots." not in path}
