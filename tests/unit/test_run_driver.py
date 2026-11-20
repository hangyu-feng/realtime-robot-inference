"""Tests of the scenario driver's plan of a run, without processes: what it starts, kills,
scrapes and reports, and the directories it refuses."""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.run import (
    FIRST_SCRAPE_K,
    HAND_RUN_SEED,
    SCRAPE_TIMEOUT_S,
    DriverError,
    RunDirectoryError,
    exit_status,
    final_counts_arrived,
    fleet_command,
    fleet_config,
    hand_run_config,
    kill_targets,
    last_scrape_k,
    main,
    next_scrape_k,
    prepare,
    report_lines,
    scrape_record,
    server_ready_event,
)
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    EVENTS_FILE,
    PLANS_DIRECTORY,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    Event,
    EventName,
    RunConfig,
)
from bench.runplan import MODEL_BENCH_KEY, fleet_plans, planned_events, run_id
from bench.scenarios import disconnect, late_joiner, steady
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from robot.control import Verdict
from robot.settings import LinkSettings, RobotSettings
from tests.support.runs import command_rows, merged, process, run_spec

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000
T0_NS = 5_000 * NS_PER_S
RUN_DIR = Path("/app/results/raw/disconnect/run")


def test_the_fleet_config_holds_the_timeline_the_robots_defaults_and_the_planned_events() -> None:
    fleet = fleet_config(disconnect())

    assert fleet.scenario == disconnect()
    assert fleet.frame_rate_hz == RobotSettings().frame_rate_hz
    assert fleet.models == RobotSettings().models
    assert fleet.stale_ms == DEFAULT_STALE_MS
    assert fleet.watchdog_ms == DEFAULT_WATCHDOG_MS
    assert fleet.planned_events == planned_events(disconnect())


def test_a_hand_run_has_no_spec_no_run_id_and_the_fleet_s_default_seed() -> None:
    config = hand_run_config(steady())

    assert config.hand_run
    assert (config.spec, config.run_id, config.seed) == (None, None, HAND_RUN_SEED)
    assert config.fleet == fleet_config(steady())


def test_a_fleet_process_runs_its_plan_with_the_spec_s_phases_links_and_the_robot_settings() -> (
    None
):
    spec = replace(run_spec("disconnect"), uplink=LinkSettings(5.0, 1.5))
    process = fleet_plans(disconnect(), T0_NS, 7)[1]

    command = fleet_command(
        process, RUN_DIR, "ws://system:8080/ws", fleet_config(disconnect()), spec, "python"
    )

    assert command == [
        "python",
        "-m",
        "robot.fleet",
        "--url",
        "ws://system:8080/ws",
        "--plan",
        str(RUN_DIR / PLANS_DIRECTORY / "fleet_c.json"),
        "--samples-dir",
        str(RUN_DIR / SAMPLES_DIRECTORY),
        "--seed",
        str(process.seed),
        "--phase",
        "random",
        "--frame-rate-hz",
        "30.0",
        "--models",
        "cls,depth",
        "--stale-ms",
        "cls=500.0,depth=150.0",
        "--watchdog-ms",
        "cls=1000.0,depth=300.0",
        "--uplink-delay-ms",
        "5.0",
        "--uplink-jitter-ms",
        "1.5",
        "--downlink-delay-ms",
        "0.0",
        "--downlink-jitter-ms",
        "0.0",
    ]


def test_a_hand_run_s_fleet_takes_random_phases_and_no_link_model() -> None:
    process = fleet_plans(steady(), T0_NS, HAND_RUN_SEED)[0]

    command = fleet_command(
        process, RUN_DIR, "ws://127.0.0.1:8080/ws", fleet_config(steady()), None, "python"
    )

    assert command[command.index("--phase") + 1] == "random"
    assert command[command.index("--seed") + 1] == str(HAND_RUN_SEED)
    assert [
        command[command.index(flag) + 1] for flag in ("--uplink-delay-ms", "--downlink-jitter-ms")
    ] == ["0.0", "0.0"]


def test_a_spec_that_runs_no_robots_has_no_fleet_command() -> None:
    spec = replace(run_spec(MODEL_BENCH_KEY), phase_mode=None)
    process = fleet_plans(steady(), T0_NS, 1)[0]

    with pytest.raises(DriverError, match="runs no robots"):
        fleet_command(process, RUN_DIR, "ws://x:1/ws", fleet_config(steady()), spec, "python")


@pytest.mark.parametrize(
    ("scenario", "targets"),
    [(disconnect(), {"c": "fleet_c"}), (steady(), {}), (late_joiner(), {})],
    ids=["disconnect", "steady", "late joiner"],
)
def test_a_robot_the_scenario_kills_is_killed_through_the_process_that_runs_it_alone(
    scenario: Any, targets: dict[str, str]
) -> None:
    assert kill_targets(fleet_plans(scenario, T0_NS, 1), scenario) == targets


@pytest.mark.parametrize(("duration_s", "last_k"), [(30.0, 30), (300.0, 300), (6.5, 6)])
def test_the_last_scrape_is_at_the_last_whole_second_of_the_run(
    duration_s: float, last_k: int
) -> None:
    assert FIRST_SCRAPE_K == -1
    assert last_scrape_k(duration_s) == last_k


@pytest.mark.parametrize(
    ("k", "now_ns", "next_k"),
    [
        (-1, T0_NS - NS_PER_S + 2 * NS_PER_MS, 0),
        (3, T0_NS + 3 * NS_PER_S + 2 * NS_PER_MS, 4),
        (3, T0_NS + 5 * NS_PER_S + 500 * NS_PER_MS, 6),
        (3, T0_NS + 5 * NS_PER_S, 5),
        (-1, T0_NS - 300 * NS_PER_MS, 0),
    ],
    ids=[
        "on time before t0",
        "on time",
        "late past two seconds",
        "late to a whole second",
        "early",
    ],
)
def test_the_next_scrape_is_the_next_whole_second_that_has_not_passed(
    k: int, now_ns: int, next_k: int
) -> None:
    assert next_scrape_k(k, T0_NS, now_ns) == next_k


def test_a_scrape_answered_in_time_keeps_its_metrics() -> None:
    processes = {"driver": process(20)}

    scrape = scrape_record(2, T0_NS, T0_NS + NS_PER_MS, {"ts_mono_ns": 1}, None, processes, None)

    assert scrape.ok
    assert scrape.metrics == {"ts_mono_ns": 1}
    assert scrape.fleet_processes == processes


def test_a_scrape_answered_later_than_its_limit_is_failed() -> None:
    late_ns = T0_NS + round(SCRAPE_TIMEOUT_S * NS_PER_S) + 1

    scrape = scrape_record(2, T0_NS, late_ns, {"ts_mono_ns": 1}, None, {}, None)

    assert not scrape.ok
    assert scrape.error is not None
    assert "longer than 1 s" in scrape.error


def test_a_scrape_that_failed_keeps_its_error() -> None:
    scrape = scrape_record(None, T0_NS, T0_NS + 1, None, "connection refused", {}, None)

    assert (scrape.ok, scrape.error, scrape.k) == (False, "connection refused", None)


def test_the_gateway_became_ready_at_its_own_clock_less_its_uptime() -> None:
    metrics = {"ts_mono_ns": T0_NS - NS_PER_S, "uptime_s": 4.0}

    assert server_ready_event(metrics, T0_NS) == Event(EventName.SERVER_READY, -5.0)


def test_a_gateway_that_is_not_ready_gives_no_ready_event() -> None:
    assert server_ready_event({"ts_mono_ns": T0_NS, "uptime_s": None}, T0_NS) is None


def _counts(cls: int, depth: int) -> dict[str, Any]:
    return {"models": {"cls": {"commands_received": cls}, "depth": {"commands_received": depth}}}


SETTLED_METRICS: dict[str, Any] = {
    "robots": {
        "a": {"connected": False, "session": 2, "telemetry": _counts(2, 1)},
        "c": {"connected": False, "session": 1},
    }
}
# Robot a's commands: one of an earlier session, then two classifier commands and one depth
# command in the session that its record names.
A_COMMANDS = merged(
    command_rows("cls", [1, 2, 3], [11, 12, 13], session=2, frame_numbers=[4, 5, 6])[:2],
    command_rows("depth", [3], [14], session=2, frame_numbers=[6]),
    command_rows("cls", [0], [10], session=1),
)


def test_the_final_counts_are_in_when_every_robot_is_gone_and_each_survivor_s_counts_match() -> (
    None
):
    assert final_counts_arrived(SETTLED_METRICS, ["a", "c"], {"a": A_COMMANDS}, ("cls", "depth"))


def _set(robot_id: str, key: str, value: object) -> Callable[[dict[str, Any]], None]:
    def edit(metrics: dict[str, Any]) -> None:
        metrics["robots"][robot_id][key] = value

    return edit


def _forget(robot_id: str) -> Callable[[dict[str, Any]], None]:
    def edit(metrics: dict[str, Any]) -> None:
        del metrics["robots"][robot_id]

    return edit


def _drop(robot_id: str, key: str) -> Callable[[dict[str, Any]], None]:
    def edit(metrics: dict[str, Any]) -> None:
        del metrics["robots"][robot_id][key]

    return edit


@pytest.mark.parametrize(
    "edit",
    [
        _set("a", "connected", True),
        _set("c", "connected", True),
        _forget("c"),
        _drop("a", "telemetry"),
        _set("a", "telemetry", _counts(2, 0)),
        _set("a", "telemetry", _counts(3, 1)),
        _set("a", "session", 1),
    ],
    ids=[
        "a survivor still connected",
        "the killed robot still connected",
        "a robot the gateway never saw",
        "no telemetry yet",
        "a model's count behind",
        "a count of every session",
        "the samples of another session",
    ],
)
def test_the_final_counts_are_not_in_while_the_gateway_lags_the_robots(
    edit: Callable[[dict[str, Any]], None],
) -> None:
    metrics = copy.deepcopy(SETTLED_METRICS)
    edit(metrics)

    assert not final_counts_arrived(metrics, ["a", "c"], {"a": A_COMMANDS}, ("cls", "depth"))


@pytest.mark.parametrize(
    ("returncodes", "status"),
    [
        ({"fleet": 0, "fleet_c": -9}, 0),
        ({"fleet": 0}, 0),
        ({"fleet": 1, "fleet_c": -9}, 1),
        ({"fleet": 0, "fleet_c": 0}, 1),
        ({"fleet": 0, "fleet_c": 1}, 1),
        ({"fleet": None, "fleet_c": -9}, 1),
        ({"fleet": -9, "fleet_c": -9}, 1),
    ],
    ids=[
        "clean with the kill",
        "clean without a kill",
        "a survivor failed",
        "the killed robot outlived its kill",
        "the killed robot failed first",
        "a survivor still running",
        "a survivor killed",
    ],
)
def test_the_driver_succeeds_when_survivors_exit_0_and_killed_robots_die_of_sigkill(
    returncodes: dict[str, int | None], status: int
) -> None:
    assert exit_status(returncodes, {"fleet_c"}) == status


def test_a_hand_run_reports_each_robot_s_commands_freshness_and_stale_count_per_model() -> None:
    capture_ns = [0, NS_PER_S, 2 * NS_PER_S, 3 * NS_PER_S]
    freshness_ns = [10 * NS_PER_MS, 20 * NS_PER_MS, 30 * NS_PER_MS, 40 * NS_PER_MS]
    rows = command_rows(
        "cls",
        capture_ns,
        np.add(capture_ns, freshness_ns),
        verdicts=[Verdict.APPLIED, Verdict.STALE, Verdict.APPLIED, Verdict.OUT_OF_ORDER],
    )

    assert report_lines({"a": rows}, ("cls", "depth")) == [
        "robot a cls: 4 commands, freshness ms p50 25.0 p95 38.5 p99 39.7 max 40.0, 1 stale",
        "robot a depth: 0 commands",
    ]


@pytest.mark.parametrize("held", [DONE_FILE, EVENTS_FILE, SCRAPES_FILE])
def test_a_directory_that_holds_a_run_is_refused(tmp_path: Path, held: str) -> None:
    (tmp_path / held).write_text("")

    with pytest.raises(RunDirectoryError, match=f"already holds a run: it has {held}"):
        prepare(tmp_path, steady())


def test_a_hand_run_writes_its_config_and_the_directories_of_its_plans_and_samples(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "steady" / "manual"

    config, fleet = prepare(run_dir, steady())

    assert config == hand_run_config(steady())
    assert fleet == fleet_config(steady())
    assert RunConfig.from_json((run_dir / CONFIG_FILE).read_text()) == config
    assert (run_dir / PLANS_DIRECTORY).is_dir()
    assert (run_dir / SAMPLES_DIRECTORY).is_dir()


def test_a_planned_run_is_played_from_the_config_the_runner_wrote(tmp_path: Path) -> None:
    spec = run_spec("disconnect")
    written = RunConfig(spec, run_id(spec), spec.repeat, None, fleet_config(disconnect()))
    (tmp_path / CONFIG_FILE).write_text(written.to_json())

    config, fleet = prepare(tmp_path, None)

    assert config == written
    assert fleet == fleet_config(disconnect())


def test_a_directory_with_a_config_takes_no_scenario(tmp_path: Path) -> None:
    (tmp_path / CONFIG_FILE).write_text(hand_run_config(steady()).to_json())

    with pytest.raises(RunDirectoryError, match="--scenario is for a directory without one"):
        prepare(tmp_path, steady())


def test_a_directory_without_a_config_needs_a_scenario(tmp_path: Path) -> None:
    with pytest.raises(RunDirectoryError, match="give --scenario for a hand run"):
        prepare(tmp_path, None)


def test_the_config_of_a_run_without_robots_is_refused(tmp_path: Path) -> None:
    spec = run_spec(MODEL_BENCH_KEY)
    (tmp_path / CONFIG_FILE).write_text(RunConfig(spec, run_id(spec), 1, None, None).to_json())

    with pytest.raises(RunDirectoryError, match="a run without robots"):
        prepare(tmp_path, None)


def test_the_driver_exits_2_for_a_directory_that_holds_a_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / DONE_FILE).write_text("")

    assert main(["--run-dir", str(tmp_path)]) == 2
    assert "already holds a run" in capsys.readouterr().err


def test_an_unknown_scenario_key_is_a_bad_argument(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exit_info:
        main(["--run-dir", str(tmp_path), "--scenario", "sweep_n2"])

    assert exit_info.value.code == 2
