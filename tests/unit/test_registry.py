"""Tests for the robot table: session numbers, retention, the cap, and what a reconnect keeps."""

import pytest

from cloud.registry import ModelCounters, RobotRecord, RobotRegistry

# Times are in nanoseconds on the gateway's clock.
RETENTION_NS = 1_000
MAX_ENTRIES = 3
FRAME_PERIOD_NS = 40
STALE_NS = {"cls": 500, "depth": 150}
WATCHDOG_NS = {"cls": 1_000, "depth": 300}
BOTH = ("cls", "depth")


def connect(
    registry: RobotRegistry, robot_id: str, now_ns: int, subscribed: tuple[str, ...] = BOTH
) -> RobotRecord:
    return registry.connect(
        robot_id,
        subscribed,
        {model: STALE_NS[model] for model in subscribed},
        {model: WATCHDOG_NS[model] for model in subscribed},
        FRAME_PERIOD_NS,
        now_ns,
    )


def kept_ids(registry: RobotRegistry) -> list[str]:
    return sorted(record.robot_id for record in registry.records())


def test_session_numbers_count_up_from_one_across_ids_and_reconnects() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)

    sessions = [connect(registry, "a", 0).session, connect(registry, "b", 1).session]
    registry.disconnect("a", 2)
    sessions.append(connect(registry, "a", 3).session)

    assert sessions == [1, 2, 3]


def test_an_id_that_returns_after_its_record_expired_gets_a_new_record_and_session() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    old = connect(registry, "a", 0)
    old.models["cls"].frames_received = 5
    registry.disconnect("a", 10)

    new = connect(registry, "a", 11 + RETENTION_NS)

    assert new is not old
    assert new.session == 2
    assert new.models["cls"] == ModelCounters(last_command_ns=11 + RETENTION_NS)


def test_an_id_that_returns_after_its_record_was_evicted_gets_an_unused_session_number() -> None:
    registry = RobotRegistry(RETENTION_NS, max_entries=1)
    connect(registry, "a", 0)
    registry.disconnect("a", 1)
    connect(registry, "b", 2)
    registry.disconnect("b", 3)

    returned = connect(registry, "a", 4)

    assert (returned.session, kept_ids(registry)) == (3, ["a"])


@pytest.mark.parametrize(
    ("disconnected_for_ns", "ids"),
    [(RETENTION_NS, ["a", "b"]), (RETENTION_NS + 1, ["b"])],
    ids=["at-the-retention-time", "past-the-retention-time"],
)
def test_a_connect_drops_records_disconnected_for_longer_than_the_retention_time(
    disconnected_for_ns: int, ids: list[str]
) -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    connect(registry, "a", 0)
    registry.disconnect("a", 100)

    connect(registry, "b", 100 + disconnected_for_ns)

    assert kept_ids(registry) == ids


def test_a_full_table_drops_the_record_disconnected_longest_then_by_robot_id() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    for now_ns, robot_id in enumerate("abc"):
        connect(registry, robot_id, now_ns)
    registry.disconnect("c", 10)
    registry.disconnect("b", 10)
    registry.disconnect("a", 11)

    tables = []
    for now_ns, robot_id in [(12, "d"), (13, "e"), (14, "f")]:
        connect(registry, robot_id, now_ns)
        tables.append(kept_ids(registry))

    assert tables == [["a", "c", "d"], ["a", "d", "e"], ["d", "e", "f"]]


def test_a_known_id_gets_its_own_record_back_from_a_full_table_and_nothing_is_dropped() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    records = [connect(registry, robot_id, now_ns) for now_ns, robot_id in enumerate("abc")]
    records[0].models["cls"].last_served_seq = 4
    for now_ns, robot_id in enumerate("abc", start=10):
        registry.disconnect(robot_id, now_ns)

    again = connect(registry, "a", 20)

    assert again is records[0]
    assert again.models["cls"].last_served_seq == 4
    assert kept_ids(registry) == ["a", "b", "c"]


def test_a_new_id_raises_when_every_record_of_a_full_table_is_connected() -> None:
    registry = RobotRegistry(RETENTION_NS, max_entries=2)
    connect(registry, "a", 0)
    connect(registry, "b", 1)

    with pytest.raises(RuntimeError, match="all 2 robot records are connected"):
        connect(registry, "c", 2)

    assert kept_ids(registry) == ["a", "b"]
    registry.disconnect("a", 3)
    assert connect(registry, "c", 4).session == 3


def test_a_connected_id_cannot_connect_again() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = connect(registry, "a", 0)

    with pytest.raises(ValueError, match="'a' is connected"):
        connect(registry, "a", 5)

    assert (record.session, record.connected_ns) == (1, 0)


def test_a_reconnect_keeps_counters_and_fairness_memory_and_clears_session_fields() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = connect(registry, "a", 0, ("cls",))
    counters = record.models["cls"]
    counters.frames_received = 7
    counters.frames_served = 4
    counters.commands_sent = 3
    counters.last_served_seq = 3
    counters.last_served_ns = 20
    counters.last_command_ns = 30
    counters.max_service_gap_ns = 9
    record.bytes_in = 100
    record.frames_accepted = 7
    record.rtt_ns = 5
    record.telemetry = {"frames_sent": 7}
    registry.disconnect("a", 50)

    again = registry.connect("a", BOTH, {"cls": 400, "depth": 100}, WATCHDOG_NS, 20, 60)

    assert again is record
    assert again.models["cls"] is counters
    assert (again.session, again.connected_ns, again.disconnected_ns) == (2, 60, None)
    assert (again.subscribed, again.stale_ns, again.watchdog_ns, again.frame_period_ns) == (
        BOTH,
        {"cls": 400, "depth": 100},
        WATCHDOG_NS,
        20,
    )
    assert again.models == {
        "cls": ModelCounters(
            frames_received=7,
            frames_served=4,
            commands_sent=3,
            last_served_seq=3,
            last_served_ns=20,
            last_command_ns=60,
            max_service_gap_ns=9,
        ),
        "depth": ModelCounters(last_command_ns=60),
    }
    assert (again.bytes_in, again.frames_accepted, again.rtt_ns, again.telemetry) == (
        100,
        7,
        None,
        None,
    )


def test_a_new_record_has_zero_counters_and_a_service_gap_start_for_each_subscribed_model() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)

    record = connect(registry, "a", 5, ("depth",))

    assert record.models == {"depth": ModelCounters(last_command_ns=5)}
    assert (record.bytes_in, record.bytes_out, record.frames_in) == (0, 0, 0)
    assert (record.frames_rejected, record.frames_accepted) == (0, 0)
    assert (record.rtt_ns, record.telemetry) == (None, None)


def test_a_reconnect_starts_the_service_gap_only_for_the_models_it_subscribes_to() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = connect(registry, "a", 0, BOTH)
    registry.disconnect("a", 40)

    connect(registry, "a", 50, ("cls",))

    last_command_ns = {model: counters.last_command_ns for model, counters in record.models.items()}
    assert last_command_ns == {"cls": 50, "depth": None}


def test_disconnect_ends_the_session_and_keeps_what_the_io_layer_set() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = connect(registry, "a", 0)
    record.rtt_ns = 7
    record.telemetry = {"frames_sent": 3}

    registry.disconnect("a", 25)

    assert (record.connected, record.disconnected_ns) == (False, 25)
    assert (record.rtt_ns, record.telemetry) == (7, {"frames_sent": 3})


def test_disconnecting_an_unknown_id_raises_key_error() -> None:
    with pytest.raises(KeyError):
        RobotRegistry(RETENTION_NS, MAX_ENTRIES).disconnect("a", 0)


def test_disconnecting_a_disconnected_id_raises_value_error() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    connect(registry, "a", 0)
    registry.disconnect("a", 1)

    with pytest.raises(ValueError, match="'a' is not connected"):
        registry.disconnect("a", 2)


def test_get_returns_the_record_and_raises_key_error_for_an_unknown_id() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    record = connect(registry, "a", 0)

    assert registry.get("a") is record
    with pytest.raises(KeyError):
        registry.get("b")


def test_records_holds_connected_and_disconnected_records_and_connected_count_the_first() -> None:
    registry = RobotRegistry(RETENTION_NS, MAX_ENTRIES)
    for now_ns, robot_id in enumerate("abc"):
        connect(registry, robot_id, now_ns)
    registry.disconnect("b", 5)

    assert kept_ids(registry) == ["a", "b", "c"]
    assert registry.connected_count() == 2
