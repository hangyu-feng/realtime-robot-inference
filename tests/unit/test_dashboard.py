"""Tests for the dashboard's view logic: points, the series, the replay choice and /data."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

from bench.dashboard import (
    HISTORY_POINTS,
    PROCESS_NAMES,
    STATUS_OK,
    DashboardError,
    Point,
    Series,
    ViewError,
    data_document,
    point,
    read_run,
    read_snapshot,
    replay_run_id,
)
from bench.runfiles import EVENTS_FILE, Event, EventName
from bench.scenarios import disconnect, late_joiner, worker_kill
from bench.tables import topology_runs
from cloud.config import Topology
from common.commands import MODEL_KEYS
from common.histogram import Histogram
from common.protocol import ModelTelemetry, Telemetry, telemetry_to_dict
from tests.support.runs import (
    EVERY_FRAMES,
    FRAME_RATE_HZ,
    NS_PER_S,
    SCRAPE_LAG_NS,
    T0_NS,
    SyntheticRun,
    failed_scrape,
    metrics,
    process,
    robot_entry,
    run_spec,
    synthetic_run,
    worker_kill_run,
    write_run,
)

_CONFIG = run_spec().server_config
_EARLIER_NS = T0_NS + 5 * NS_PER_S
# Two and a half seconds, so a rate that forgets to divide by the span is wrong.
_ELAPSED_NS = 2_500_000_000


def _metrics(ts_ns: int, changes: Mapping[str, object] | None = None) -> dict[str, Any]:
    return metrics(ts_ns, changes, config=_CONFIG)


def _point(
    earlier: Mapping[str, object], later: Mapping[str, object], elapsed_ns: int = _ELAPSED_NS
) -> Point:
    return point(
        read_snapshot(_metrics(_EARLIER_NS, earlier)),
        read_snapshot(_metrics(_EARLIER_NS + elapsed_ns, later)),
        T0_NS,
    )


def _histogram(*values_ms: float) -> Histogram:
    histogram = Histogram()
    for value_ms in values_ms:
        histogram.record(value_ms)
    return histogram


def _telemetry(freshness: Histogram, commands_stale: int = 0) -> dict[str, object]:
    """Return a robot's report in which every model saw the same commands."""
    model = ModelTelemetry(
        commands_received=freshness.count,
        commands_applied=freshness.count - commands_stale,
        commands_stale=commands_stale,
        commands_out_of_order=0,
        freshness_ms=freshness,
        command_gap_ms=Histogram(),
        command_age_peak_ms=Histogram(),
        deliver_ms=Histogram(),
        watchdog_trips=0,
        watchdog_expired_ms=0,
        watchdog_state="ok",
    )
    return telemetry_to_dict(
        Telemetry(
            ts_mono_ns=_EARLIER_NS,
            frames_captured=0,
            frames_sent=0,
            frames_skipped_send=0,
            fire_lag_ms=Histogram(),
            reconnects=0,
            models=dict.fromkeys(MODEL_KEYS, model),
        )
    )


def _robot(
    *,
    session: int = 1,
    connected: bool = True,
    frames_in: int = 0,
    commands_sent: Mapping[str, int] | None = None,
    freshness: Histogram | None = None,
    commands_stale: int = 0,
) -> dict[str, Any]:
    entry = robot_entry(connected=connected, frames_in=frames_in, commands_sent=commands_sent)
    entry["session"] = session
    if freshness is not None:
        entry["telemetry"] = _telemetry(freshness, commands_stale)
    return entry


@pytest.mark.parametrize(
    ("earlier", "later", "read", "expected"),
    [
        pytest.param(
            {"models.cls.frames": 100},
            {"models.cls.frames": 150},
            lambda p: p.models["cls"].served_per_s,
            20.0,
            id="served",
        ),
        pytest.param(
            {"robots.a": _robot(frames_in=30, commands_sent={"depth": 10})},
            {"robots.a": _robot(frames_in=80, commands_sent={"depth": 20})},
            lambda p: p.models["depth"].overwritten_per_s,
            16.0,
            id="overwritten",
        ),
        pytest.param(
            {"models.cls.frames": 100, "models.cls.batches": 40},
            {"models.cls.frames": 150, "models.cls.batches": 50},
            lambda p: p.models["cls"].mean_batch_size,
            5.0,
            id="mean batch size",
        ),
        pytest.param(
            {"robots.a": _robot(frames_in=30, commands_sent={"cls": 10})},
            {"robots.a": _robot(frames_in=80, commands_sent={"cls": 35})},
            lambda p: p.robots["a"].models["cls"].commands_per_s,
            10.0,
            id="commands",
        ),
        pytest.param(
            {"cgroup.cpu_usage_us": 1_000_000},
            {"cgroup.cpu_usage_us": 3_500_000},
            lambda p: p.cpu_cores["container"],
            1.0,
            id="container cpu",
        ),
        pytest.param(
            {"processes.gateway": asdict(process(1, cpu_ms=1_000))},
            {"processes.gateway": asdict(process(1, cpu_ms=2_250))},
            lambda p: p.cpu_cores["gateway"],
            0.5,
            id="gateway cpu",
        ),
    ],
)
def test_a_rate_is_the_rise_of_its_counter_over_the_span_between_the_snapshots(
    earlier: Mapping[str, object],
    later: Mapping[str, object],
    read: Callable[[Point], float | None],
    expected: float,
) -> None:
    assert read(_point(earlier, later)) == pytest.approx(expected)


def test_a_robot_absent_from_the_earlier_snapshot_counts_from_zero() -> None:
    later = _point({}, {"robots.a": _robot(frames_in=40, commands_sent={"cls": 25})})

    assert later.robots["a"].models["cls"].commands_per_s == pytest.approx(10.0)
    assert later.models["cls"].overwritten_per_s == pytest.approx(6.0)


def test_a_new_session_reads_its_freshness_from_its_own_report_alone() -> None:
    old_session = _robot(session=1, freshness=_histogram(*[20.0] * 100))
    new_session = _robot(session=2, freshness=_histogram(*[80.0] * 10))

    reading = _point({"robots.a": old_session}, {"robots.a": new_session})

    expected = _histogram(80.0).quantile_ms(0.95)
    assert reading.robots["a"].models["cls"].freshness_p95_ms == expected


def test_freshness_is_read_from_the_window_between_two_reports_of_a_session() -> None:
    earlier = _robot(freshness=_histogram(*[20.0] * 100))
    later = _robot(freshness=_histogram(*[20.0] * 100, *[80.0] * 10), commands_stale=3)

    reading = _point({"robots.a": earlier}, {"robots.a": later}).robots["a"].models["depth"]

    assert reading.freshness_p95_ms == _histogram(80.0).quantile_ms(0.95)
    assert reading.commands_stale == 3


@pytest.mark.parametrize(
    ("earlier", "later", "read"),
    [
        pytest.param(
            {"robots.a": _robot(freshness=_histogram(20.0))},
            {"robots.a": _robot(freshness=_histogram(20.0))},
            lambda p: p.robots["a"].models["cls"].freshness_p95_ms,
            id="freshness, no command in the window",
        ),
        pytest.param(
            {"robots.a": _robot()},
            {"robots.a": _robot()},
            lambda p: p.robots["a"].models["cls"].freshness_p95_ms,
            id="freshness, no report",
        ),
        pytest.param(
            {"robots.a": _robot()},
            {"robots.a": _robot()},
            lambda p: p.robots["a"].models["cls"].commands_stale,
            id="stale count, no report",
        ),
        pytest.param({}, {}, lambda p: p.loop_lag_p99_ms, id="loop lag, no sample"),
        pytest.param({}, {}, lambda p: p.models["cls"].mean_batch_size, id="no batch"),
        pytest.param(
            {"cgroup.cpu_usage_us": None},
            {"cgroup.cpu_usage_us": None},
            lambda p: p.cpu_cores["container"],
            id="no cpu counter",
        ),
    ],
)
def test_a_value_with_nothing_to_read_it_from_is_null(
    earlier: Mapping[str, object],
    later: Mapping[str, object],
    read: Callable[[Point], object],
) -> None:
    assert read(_point(earlier, later)) is None


def test_the_loop_lag_is_the_p99_of_the_samples_between_the_snapshots() -> None:
    earlier = _histogram(*[0.5] * 100).to_dict()
    later = _histogram(*[0.5] * 100, *[0.5] * 98, 9.0, 9.0).to_dict()

    reading = _point({"loop.lag_ms": earlier}, {"loop.lag_ms": later})

    assert reading.loop_lag_p99_ms == _histogram(9.0).quantile_ms(0.99)


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        pytest.param(asdict(process(12, cpu_ms=100)), asdict(process(13, cpu_ms=200)), id="pid"),
        pytest.param(None, asdict(process(13, cpu_ms=200)), id="none before"),
        pytest.param(asdict(process(12, cpu_ms=100)), None, id="none after"),
    ],
)
def test_a_worker_whose_process_changed_or_is_missing_has_no_cpu_reading(
    earlier: dict[str, object] | None, later: dict[str, object] | None
) -> None:
    reading = _point({"processes.worker_depth": earlier}, {"processes.worker_depth": later})

    assert reading.cpu_cores["worker_depth"] is None
    assert reading.cpu_cores["worker_cls"] == 0.0


@pytest.mark.parametrize(
    ("earlier", "later", "events"),
    [
        pytest.param({}, {"robots.a": _robot()}, ("robot a connected",), id="absent before"),
        pytest.param(
            {"robots.a": _robot(connected=False)},
            {"robots.a": _robot(session=2)},
            ("robot a connected",),
            id="not connected before",
        ),
        pytest.param(
            {"robots.a": _robot(session=1)},
            {"robots.a": _robot(session=2)},
            ("robot a disconnected", "robot a connected"),
            id="another session before",
        ),
        pytest.param(
            {"robots.a": _robot()},
            {"robots.a": _robot(connected=False)},
            ("robot a disconnected",),
            id="disconnected",
        ),
        pytest.param(
            {"workers.depth.restarts": 0},
            {"workers.depth.restarts": 1},
            ("worker depth restarted",),
            id="worker restarted",
        ),
        pytest.param({"robots.a": _robot()}, {"robots.a": _robot()}, (), id="nothing happened"),
    ],
)
def test_a_point_names_the_robots_that_came_and_went_and_the_workers_that_restarted(
    earlier: Mapping[str, object], later: Mapping[str, object], events: tuple[str, ...]
) -> None:
    assert _point(earlier, later).events == events


@pytest.mark.parametrize(
    ("earlier", "later"),
    [
        pytest.param({"models.cls.frames": 100}, {"models.cls.frames": 99}, id="frames"),
        pytest.param(
            {"robots.a": _robot(frames_in=10, commands_sent={"cls": 10})},
            {"robots.a": _robot(frames_in=10, commands_sent={"cls": 9})},
            id="commands sent",
        ),
        pytest.param(
            {"cgroup.cpu_usage_us": 2_000}, {"cgroup.cpu_usage_us": 1_000}, id="cgroup cpu"
        ),
        pytest.param(
            {"loop.lag_ms": _histogram(0.5, 0.5).to_dict()},
            {"loop.lag_ms": _histogram(0.5).to_dict()},
            id="loop lag histogram",
        ),
        pytest.param(
            {"workers.depth.restarts": 2}, {"workers.depth.restarts": 1}, id="worker restarts"
        ),
    ],
)
def test_a_counter_that_went_down_is_a_view_error(
    earlier: Mapping[str, object], later: Mapping[str, object]
) -> None:
    with pytest.raises(ViewError, match="went down"):
        _point(earlier, later)


@pytest.mark.parametrize("elapsed_ns", [0, -NS_PER_S])
def test_a_span_that_is_not_positive_is_a_view_error(elapsed_ns: int) -> None:
    with pytest.raises(ViewError, match="clock moved"):
        _point({}, {}, elapsed_ns)


@pytest.mark.parametrize(
    "changes",
    [
        pytest.param({"ts_mono_ns": "soon"}, id="string clock"),
        pytest.param({"connections": None}, id="null object"),
        pytest.param({"workers": []}, id="list for an object"),
        pytest.param({"models.cls.frames": True}, id="bool count"),
        pytest.param({"models.cls.frames": -1}, id="negative count"),
        pytest.param({"cgroup.cpu_limit_cores": float("nan")}, id="nan"),
        pytest.param({"loop.lag_ms": {"count": 1}}, id="bad histogram"),
        pytest.param({"robots.a": {"session": 1, "connected": True}}, id="robot without models"),
        pytest.param({"robots.a": _robot() | {"connected": 1}}, id="int for a flag"),
    ],
)
def test_a_malformed_snapshot_is_a_view_error(changes: Mapping[str, object]) -> None:
    with pytest.raises(ViewError, match=r"docs/metrics\.md"):
        read_snapshot(_metrics(_EARLIER_NS, changes))


def _counting(k: int, frames_before: int = 0) -> dict[str, Any]:
    """Return the snapshot at second k of a gateway that serves ten cls frames a second."""
    return _metrics(T0_NS + k * NS_PER_S, {"models.cls.frames": frames_before + 10 * k})


def test_a_failure_then_two_snapshots_give_one_point_after_a_gap() -> None:
    series = Series(T0_NS)
    for k in (0, 1):
        series.add_snapshot(_counting(k))

    series.add_failure("timed out")
    series.add_snapshot(_counting(3))
    series.add_snapshot(_counting(4))

    assert [item.t_s for item in series.points] == [1.0, 4.0]
    assert series.points[-1].models["cls"].served_per_s == pytest.approx(10.0)


def test_a_view_error_then_two_snapshots_give_one_point_after_a_gap() -> None:
    series = Series(T0_NS)
    for k in (0, 1):
        series.add_snapshot(_counting(k, frames_before=1_000))

    # The gateway restarted: its counters start again from zero.
    for k in (2, 3, 4):
        series.add_snapshot(_counting(k))

    assert [item.t_s for item in series.points] == [1.0, 4.0]


@pytest.mark.parametrize("count", [HISTORY_POINTS - 1, HISTORY_POINTS, HISTORY_POINTS + 7])
def test_the_series_keeps_the_latest_points_up_to_its_bound(count: int) -> None:
    series = Series(T0_NS)
    for k in range(count + 1):
        series.add_snapshot(_counting(k))

    kept = min(count, HISTORY_POINTS)
    assert [item.t_s for item in series.points] == [
        float(k) for k in range(count - kept + 1, count + 1)
    ]


def test_the_status_is_ok_after_a_point_and_the_reason_after_a_failure() -> None:
    series = Series(T0_NS)
    for k in (0, 1):
        series.add_snapshot(_counting(k))
    after_point = series.status

    series.add_failure("gateway not reachable")

    assert (after_point, series.status) == (STATUS_OK, "gateway not reachable")


def test_a_view_error_becomes_the_status() -> None:
    series = Series(T0_NS)
    series.add_snapshot(_counting(1, frames_before=100))

    series.add_snapshot(_counting(2))

    assert "models.cls.frames went down" in series.status


def test_without_an_origin_the_first_snapshot_is_time_zero() -> None:
    series = Series(None)
    for k in (3, 4):
        series.add_snapshot(_counting(k))

    assert [item.t_s for item in series.points] == [1.0]


def _played(run: SyntheticRun) -> Series:
    series = Series(T0_NS)
    for scrape in run.scrapes:
        if scrape.metrics is not None:
            series.add_snapshot(scrape.metrics)
    return series


def test_a_whole_run_gives_each_robot_the_rates_it_was_served_at() -> None:
    scenario = late_joiner()
    series = _played(synthetic_run(scenario, run_spec("late_joiner")))

    # The points whose span lies inside the robot's run, away from its start and its end.
    rates = {
        (plan.robot_id, model): [
            item.robots[plan.robot_id].models[model].commands_per_s
            for item in series.points
            if plan.start_s + 1 < item.t_s < scenario.duration_s
        ]
        for plan in scenario.robots
        for model in MODEL_KEYS
    }
    expected = {
        (plan.robot_id, model): FRAME_RATE_HZ / EVERY_FRAMES[model]
        for plan in scenario.robots
        for model in MODEL_KEYS
    }
    assert {key: min(values) for key, values in rates.items()} == pytest.approx(expected, abs=1)
    assert {key: max(values) for key, values in rates.items()} == pytest.approx(expected, abs=1)
    assert series.status == STATUS_OK


@pytest.mark.parametrize(
    ("run", "event", "at_s"),
    [
        pytest.param(
            synthetic_run(late_joiner(), run_spec("late_joiner")),
            "robot b connected",
            10.0,
            id="late joiner",
        ),
        # The gateway closes the killed robot's session after the scrape planned at 20 s.
        pytest.param(
            synthetic_run(disconnect(), run_spec("disconnect")),
            "robot c disconnected",
            21.0,
            id="disconnect",
        ),
        # The worker is killed just before the scrape planned at 20 s.
        pytest.param(
            worker_kill_run(worker_kill(), run_spec("worker_kill")),
            "worker depth restarted",
            20.0,
            id="worker kill",
        ),
    ],
)
def test_a_whole_run_marks_its_event_on_the_first_scrape_that_shows_it(
    run: SyntheticRun, event: str, at_s: float
) -> None:
    series = _played(run)

    marked = [item.t_s for item in series.points if event in item.events]
    assert marked == [pytest.approx(at_s + SCRAPE_LAG_NS / NS_PER_S)]


def _runs(**counted: tuple[str, ...]) -> dict[tuple[str, Topology], tuple[str, ...]]:
    """Return topology_runs's mapping in which only the given keys have counted single runs."""
    return topology_runs(None) | {(key, Topology.SINGLE): ids for key, ids in counted.items()}


def test_the_replay_plays_the_first_counted_single_run_of_its_key() -> None:
    runs = _runs(late_joiner=("single-1", "single-2")) | {
        ("late_joiner", Topology.SPLIT): ("split-0",)
    }

    assert replay_run_id(runs, "late_joiner") == "single-1"


@pytest.mark.parametrize("key", ["nonsense", "steady"])
def test_a_key_without_a_counted_single_run_names_the_keys_that_have_one(key: str) -> None:
    runs = _runs(late_joiner=("single-1",), massive=("single-2",)) | {
        ("steady", Topology.SPLIT): ("split-0",)
    }

    with pytest.raises(DashboardError, match=r"the keys that have one: late_joiner, massive$"):
        replay_run_id(runs, key)


def test_a_run_directory_reads_back_as_its_t0_and_its_successful_scrapes(tmp_path: Path) -> None:
    run = synthetic_run(late_joiner(), run_spec("late_joiner"))
    scrapes = (*run.scrapes[:3], failed_scrape(2, T0_NS + 2 * NS_PER_S), *run.scrapes[3:])
    write_run(tmp_path, replace(run, scrapes=scrapes))

    stored = read_run(tmp_path)

    assert stored.t0_mono_ns == T0_NS
    assert stored.snapshots == tuple(scrape.metrics for scrape in run.scrapes)


def test_a_run_directory_that_does_not_start_with_t0_cannot_be_replayed(tmp_path: Path) -> None:
    write_run(tmp_path, synthetic_run(late_joiner(), run_spec("late_joiner")))
    (tmp_path / EVENTS_FILE).write_text(Event(EventName.SCENARIO_END, 30.0).to_json() + "\n")

    with pytest.raises(DashboardError, match="not the t0 event"):
        read_run(tmp_path)


def test_a_missing_run_directory_cannot_be_replayed(tmp_path: Path) -> None:
    with pytest.raises(DashboardError, match="cannot be read"):
        read_run(tmp_path / "absent")


def _refuse(constant: str) -> None:
    raise AssertionError(f"{constant} is not strict JSON")


def test_the_data_document_holds_the_fields_of_the_view_as_strict_json() -> None:
    series = Series(T0_NS)
    series.add_snapshot(_metrics(T0_NS, {"robots.a": _robot()}))
    series.add_snapshot(
        _metrics(T0_NS + NS_PER_S, {"robots.a": _robot(), "processes.worker_depth": None})
    )

    body = data_document("replay of late_joiner", series.status, series.points)

    document = json.loads(body, parse_constant=_refuse)
    (item,) = document["points"]
    assert document.keys() == {"source", "status", "points"}
    assert item.keys() == {
        "t_s",
        "connections",
        "memory_bytes",
        "loop_lag_p99_ms",
        "cpu_limit_cores",
        "cpu_cores",
        "models",
        "robots",
        "events",
    }
    assert item["cpu_cores"].keys() == {"container", *PROCESS_NAMES}
    assert item["models"]["depth"].keys() == {
        "served_per_s",
        "overwritten_per_s",
        "mean_batch_size",
        "stale_ms",
        "worker_state",
        "worker_restarts",
    }
    assert item["robots"]["a"] == {
        "connected": True,
        "models": {
            model: {"commands_per_s": 0.0, "freshness_p95_ms": None, "commands_stale": None}
            for model in MODEL_KEYS
        },
    }
    assert (item["cpu_cores"]["worker_depth"], item["loop_lag_p99_ms"]) == (None, None)
