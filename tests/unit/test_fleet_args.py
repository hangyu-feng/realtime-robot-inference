"""Tests for the fleet: its command line and plan file, its robots' times, samples and exit."""

import json
import random
from pathlib import Path
from typing import Any

import pytest

from common.errors import ConfigError
from common.protocol import STAGE_NAMES
from robot.fleet import (
    FleetPlan,
    Phase,
    RobotPlan,
    exit_status,
    main,
    parse_arguments,
    phases_ns,
    read_plan,
    robot_runs,
    sample_capacities,
    save_samples,
)
from robot.samples import SampleLog, TickOutcome, load_samples
from robot.schedule import CaptureSchedule
from robot.settings import LinkSettings, RobotSettings

SECOND = 1_000_000_000
# The period at the default 30 frames per second.
PERIOD_NS = 33_333_333


def write_plan(tmp_path: Path, document: object) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def plan_document(**changes: Any) -> dict[str, Any]:
    document: dict[str, Any] = {
        "t0_mono_ns": 5_000_000_000,
        "duration_s": 30.0,
        "robots": [
            {"robot_id": "a", "start_s": 0.0, "stop_s": None},
            {"robot_id": "b", "start_s": 10.0, "stop_s": 20.5},
        ],
    }
    return {**document, **changes}


def test_a_robot_count_names_the_robots_and_runs_them_all_for_the_duration() -> None:
    arguments = parse_arguments(["--robots", "3", "--duration-s", "7.5"])

    assert arguments.plan == FleetPlan(
        t0_mono_ns=None,
        duration_s=7.5,
        robots=(
            RobotPlan("r01", 0.0, None),
            RobotPlan("r02", 0.0, None),
            RobotPlan("r03", 0.0, None),
        ),
    )


def test_the_ids_of_a_large_fleet_keep_two_digits() -> None:
    arguments = parse_arguments(["--robots", "99", "--duration-s", "1"])

    ids = [robot.robot_id for robot in arguments.plan.robots]
    assert (ids[0], ids[8], ids[9], ids[98]) == ("r01", "r09", "r10", "r99")
    assert ids == sorted(ids)


def test_without_other_flags_the_robots_have_the_default_settings() -> None:
    arguments = parse_arguments(["--robots", "1", "--duration-s", "1"])

    assert arguments.settings == RobotSettings()
    assert (arguments.phase, arguments.seed, arguments.samples_dir) == (Phase.RANDOM, 0, None)


def test_a_plan_file_gives_the_time_zero_the_duration_and_each_robot_s_times(
    tmp_path: Path,
) -> None:
    path = write_plan(tmp_path, plan_document())

    arguments = parse_arguments(["--plan", str(path)])

    assert arguments.plan == FleetPlan(
        t0_mono_ns=5_000_000_000,
        duration_s=30.0,
        robots=(RobotPlan("a", 0.0, None), RobotPlan("b", 10.0, 20.5)),
    )


def test_a_plan_without_a_time_zero_and_with_whole_numbers_is_accepted(tmp_path: Path) -> None:
    document = plan_document(
        t0_mono_ns=None,
        duration_s=30,
        robots=[{"robot_id": "r-1.a_B", "start_s": 0, "stop_s": 30}],
    )

    plan = read_plan(write_plan(tmp_path, document))

    assert plan == FleetPlan(None, 30.0, (RobotPlan("r-1.a_B", 0.0, 30.0),))


def test_the_flags_set_the_robots_settings(tmp_path: Path) -> None:
    arguments = parse_arguments(
        [
            *("--url", "ws://system:8080/ws"),
            *("--robots", "2", "--duration-s", "3"),
            *("--frame-rate-hz", "15"),
            *("--phase", "aligned", "--seed", "42"),
            *("--stale-ms", "cls=400,depth=120", "--watchdog-ms", "depth=250"),
            *("--uplink-delay-ms", "20", "--uplink-jitter-ms", "2"),
            *("--downlink-delay-ms", "30", "--downlink-jitter-ms", "3"),
            *("--samples-dir", str(tmp_path / "samples"), "--token", "secret"),
        ]
    )

    assert arguments.settings == RobotSettings(
        url="ws://system:8080/ws",
        stale_ms={"cls": 400.0, "depth": 120.0},
        watchdog_ms={"cls": 1000.0, "depth": 250.0},
        frame_rate_hz=15.0,
        token="secret",
        uplink=LinkSettings(20.0, 2.0),
        downlink=LinkSettings(30.0, 3.0),
    )
    assert (arguments.phase, arguments.seed) == (Phase.ALIGNED, 42)
    assert arguments.samples_dir == tmp_path / "samples"


def test_a_robot_subscribed_to_one_model_declares_that_model_s_thresholds_only() -> None:
    arguments = parse_arguments(
        ["--robots", "1", "--duration-s", "1", "--models", "depth", "--stale-ms", "depth=100"]
    )

    assert arguments.settings.models == ("depth",)
    assert arguments.settings.stale_ms == {"depth": 100.0}
    assert arguments.settings.watchdog_ms == {"depth": 300.0}


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--robots", "3"],
        ["--duration-s", "5"],
        ["--robots", "0", "--duration-s", "5"],
        ["--robots", "100", "--duration-s", "5"],
        ["--robots", "two", "--duration-s", "5"],
        ["--robots", "3", "--duration-s", "0"],
        ["--robots", "3", "--duration-s", "-1"],
        ["--robots", "3", "--duration-s", "nan"],
        ["--robots", "3", "--duration-s", "inf"],
        ["--robots", "3", "--duration-s", "1e300"],
        ["--robots", "3", "--duration-s", "5", "--plan", "plan.json"],
        ["--plan", "plan.json", "--duration-s", "5"],
        ["--plan", "no-such-plan.json"],
        ["--robots", "1", "--duration-s", "5", "--url", "http://127.0.0.1:8080/ws"],
        ["--robots", "1", "--duration-s", "5", "--frame-rate-hz", "0"],
        ["--robots", "1", "--duration-s", "5", "--frame-rate-hz", "120"],
        ["--robots", "1", "--duration-s", "5", "--phase", "staggered"],
        ["--robots", "1", "--duration-s", "5", "--seed", "x"],
        ["--robots", "1", "--duration-s", "5", "--models", "cls,pose"],
        ["--robots", "1", "--duration-s", "5", "--models", "cls,cls"],
        ["--robots", "1", "--duration-s", "5", "--models", ""],
        ["--robots", "1", "--duration-s", "5", "--models", "cls", "--stale-ms", "depth=100"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=fast"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=100,depth=120"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=0"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=nan"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=60001"],
        ["--robots", "1", "--duration-s", "5", "--stale-ms", "depth=400"],
        ["--robots", "1", "--duration-s", "5", "--watchdog-ms", "cls=100"],
        ["--robots", "1", "--duration-s", "5", "--watchdog-ms", "pose=100"],
        ["--robots", "1", "--duration-s", "5", "--uplink-delay-ms", "-1"],
        ["--robots", "1", "--duration-s", "5", "--uplink-jitter-ms", "nan"],
        ["--robots", "1", "--duration-s", "5", "--downlink-delay-ms", "inf"],
        ["--robots", "1", "--duration-s", "5", "--downlink-jitter-ms", "-0.5"],
        ["--robots", "1", "--duration-s", "5", "--token", ""],
    ],
)
def test_a_bad_command_line_exits_with_status_2(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        parse_arguments(argv)

    assert exit_info.value.code == 2


@pytest.mark.parametrize(
    ("document", "message"),
    [
        ([], "a plan must be an object"),
        ({"duration_s": 30.0, "robots": []}, "a plan must be an object"),
        (plan_document(extra=1), "a plan must be an object"),
        (plan_document(t0_mono_ns=1.5), "t0_mono_ns must be"),
        (plan_document(t0_mono_ns=True), "t0_mono_ns must be"),
        (plan_document(t0_mono_ns=-1), "t0_mono_ns must be"),
        (plan_document(t0_mono_ns=2**61 + 1), "t0_mono_ns must be"),
        (plan_document(duration_s=0), "duration_s must be above 0"),
        (plan_document(duration_s=-1.0), "duration_s must be a number"),
        (plan_document(duration_s="30"), "duration_s must be a number"),
        (plan_document(duration_s=True), "duration_s must be a number"),
        (plan_document(duration_s=None), "duration_s must be a number"),
        (plan_document(duration_s=10**400), "duration_s must be a number"),
        (plan_document(robots=[]), "robots must be a list of at least one robot"),
        (plan_document(robots={"robot_id": "a"}), "robots must be a list"),
        (plan_document(robots=["a"]), "a robot must be an object"),
        (plan_document(robots=[{"robot_id": "a", "start_s": 0.0}]), "a robot must be an object"),
        (
            plan_document(robots=[{"robot_id": "a b", "start_s": 0.0, "stop_s": None}]),
            "robot_id must be",
        ),
        (
            plan_document(robots=[{"robot_id": "", "start_s": 0.0, "stop_s": None}]),
            "robot_id must be",
        ),
        (
            plan_document(robots=[{"robot_id": 7, "start_s": 0.0, "stop_s": None}]),
            "robot_id must be",
        ),
        (
            plan_document(robots=[{"robot_id": "a" * 65, "start_s": 0.0, "stop_s": None}]),
            "robot_id must be",
        ),
        (
            plan_document(
                robots=[
                    {"robot_id": "a", "start_s": 0.0, "stop_s": None},
                    {"robot_id": "a", "start_s": 1.0, "stop_s": None},
                ]
            ),
            "robots must have distinct ids",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": -0.5, "stop_s": None}]),
            "a: start_s must be a number",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": "0", "stop_s": None}]),
            "a: start_s must be a number",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": 30.0, "stop_s": None}]),
            "a: start_s must be below duration_s",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": 5.0, "stop_s": 5.0}]),
            "a: stop_s must be above start_s",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": 5.0, "stop_s": 30.5}]),
            "a: stop_s must be above start_s and at most duration_s",
        ),
        (
            plan_document(robots=[{"robot_id": "a", "start_s": 5.0, "stop_s": False}]),
            "a: stop_s must be a number",
        ),
    ],
)
def test_a_plan_that_breaks_a_rule_is_refused_naming_it(
    tmp_path: Path, document: object, message: str
) -> None:
    with pytest.raises(ConfigError, match=message):
        read_plan(write_plan(tmp_path, document))


@pytest.mark.parametrize(
    "content",
    [
        b"",
        b"{",
        b'{"t0_mono_ns": null, "duration_s": 30.0, "robots": [}',
        b"[" * 100_000,
        b"\xff\xfe{}",
        b'{"t0_mono_ns": 1' + b"0" * 5000 + b"}",
    ],
)
def test_a_plan_file_that_is_not_json_is_refused(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "plan.json"
    path.write_bytes(content)

    with pytest.raises(ConfigError, match="is not JSON"):
        read_plan(path)


def test_a_plan_file_that_cannot_be_read_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read the plan"):
        read_plan(tmp_path / "missing.json")


def test_a_bad_plan_exits_with_status_2(tmp_path: Path) -> None:
    path = write_plan(tmp_path, plan_document(duration_s=0))

    with pytest.raises(SystemExit) as exit_info:
        parse_arguments(["--plan", str(path)])

    assert exit_info.value.code == 2


def test_random_phases_are_the_seed_s_draws_below_one_period_in_plan_order() -> None:
    rng = random.Random(42)

    phases = phases_ns(Phase.RANDOM, 42, 50, PERIOD_NS)

    assert phases == [rng.randrange(PERIOD_NS) for _ in range(50)]
    assert all(0 <= phase_ns < PERIOD_NS for phase_ns in phases)
    assert len(set(phases)) == 50


def test_the_same_seed_gives_the_same_phases_and_another_seed_gives_others() -> None:
    assert phases_ns(Phase.RANDOM, 7, 10, PERIOD_NS) == phases_ns(Phase.RANDOM, 7, 10, PERIOD_NS)
    assert phases_ns(Phase.RANDOM, 7, 10, PERIOD_NS) != phases_ns(Phase.RANDOM, 8, 10, PERIOD_NS)


def test_a_larger_fleet_keeps_the_phases_of_the_robots_before() -> None:
    assert phases_ns(Phase.RANDOM, 7, 10, PERIOD_NS)[:5] == phases_ns(Phase.RANDOM, 7, 5, PERIOD_NS)


def test_aligned_phases_are_all_zero() -> None:
    assert phases_ns(Phase.ALIGNED, 7, 4, PERIOD_NS) == [0, 0, 0, 0]


def test_each_robot_runs_from_its_start_to_its_stop_after_the_time_zero(tmp_path: Path) -> None:
    path = write_plan(tmp_path, plan_document(t0_mono_ns=None))

    runs = robot_runs(parse_arguments(["--plan", str(path)]), t0_ns=7 * SECOND)

    # Robot b runs from 10 s to 20.5 s, and robot a, which has no stop, to the end at 30 s.
    assert [(run.robot_id, run.start_ns, run.stop_ns) for run in runs] == [
        ("a", 7 * SECOND, 37 * SECOND),
        ("b", 17 * SECOND, 27_500_000_000),
    ]


@pytest.mark.parametrize("phase", list(Phase))
def test_each_robot_s_first_tick_is_its_start_plus_its_phase(tmp_path: Path, phase: Phase) -> None:
    path = write_plan(tmp_path, plan_document(t0_mono_ns=None))
    arguments = parse_arguments(["--plan", str(path), "--phase", phase.value, "--seed", "42"])

    runs = robot_runs(arguments, t0_ns=7 * SECOND)

    first_ns, second_ns = phases_ns(phase, 42, 2, PERIOD_NS)
    assert [run.schedule for run in runs] == [
        CaptureSchedule(7 * SECOND + first_ns, PERIOD_NS),
        CaptureSchedule(17 * SECOND + second_ns, PERIOD_NS),
    ]


def test_each_robot_draws_from_a_generator_seeded_with_the_seed_and_its_id(tmp_path: Path) -> None:
    path = write_plan(tmp_path, plan_document(t0_mono_ns=None))

    runs = robot_runs(parse_arguments(["--plan", str(path), "--seed", "42"]), t0_ns=7 * SECOND)

    assert [run.rng.getstate() for run in runs] == [
        random.Random("42:a:backoff").getstate(),
        random.Random("42:b:backoff").getstate(),
    ]


@pytest.mark.parametrize(
    ("argv", "capacities"),
    [
        (["--duration-s", "2"], (90, 180)),
        (["--duration-s", "10", "--frame-rate-hz", "29.97"], (330, 660)),
        (["--duration-s", "7.5", "--frame-rate-hz", "15", "--models", "depth"], (128, 128)),
    ],
)
def test_a_sample_log_holds_every_tick_of_the_run_and_a_second_more_and_a_command_per_model(
    argv: list[str], capacities: tuple[int, int]
) -> None:
    arguments = parse_arguments(["--robots", "3", *argv])

    assert sample_capacities(arguments) == capacities


def test_each_robot_s_log_is_saved_with_the_meta_of_its_run(tmp_path: Path) -> None:
    arguments = parse_arguments(
        [
            *("--robots", "2", "--duration-s", "2", "--seed", "7"),
            *("--uplink-delay-ms", "20", "--token", "secret"),
        ]
    )
    runs = robot_runs(arguments, t0_ns=5 * SECOND)
    logs = {
        "r01": SampleLog(tick_capacity=1, command_capacity=1),
        "r02": SampleLog(tick_capacity=1, command_capacity=1),
    }
    logs["r01"].tick(1, 5_033_333_333, 5_033_400_000, TickOutcome.SENT)
    logs["r02"].tick(1, 5_033_333_333, 0, TickOutcome.SKIPPED_UNSENT)
    logs["r02"].tick(2, 5_066_666_666, 5_066_700_000, TickOutcome.SENT)

    save_samples(tmp_path, arguments, runs, logs)

    files = [load_samples(tmp_path / f"robot_{robot_id}.npz") for robot_id in ("r01", "r02")]
    assert [ticks.tolist() for _, ticks, _ in files] == [
        [(1, 5_033_333_333, 5_033_400_000, 0)],
        [(1, 5_033_333_333, 0, 1)],
    ]
    # Every setting but the token.
    settings = {
        "url": "ws://127.0.0.1:8080/ws",
        "models": ["cls", "depth"],
        "stale_ms": {"cls": 500.0, "depth": 150.0},
        "watchdog_ms": {"cls": 1000.0, "depth": 300.0},
        "frame_rate_hz": 30.0,
        "uplink": {"delay_ms": 20.0, "jitter_ms": 0.0},
        "downlink": {"delay_ms": 0.0, "jitter_ms": 0.0},
        "connect_timeout_ms": 5000.0,
        "welcome_timeout_ms": 5000.0,
        "close_timeout_ms": 1000.0,
        "stall_timeout_ms": 3000.0,
        "reconnect_min_ms": 100.0,
        "reconnect_max_ms": 5000.0,
        "sndbuf_bytes": 40960,
        "notsent_lowat_bytes": 16384,
        "rcvbuf_bytes": 4096,
        "max_message_bytes": 4096,
    }
    run_meta = {
        "seed": 7,
        "phase": "random",
        "settings": settings,
        "period_ns": PERIOD_NS,
        "model_keys": ["cls", "depth"],
        "stage_names": list(STAGE_NAMES),
        "verdict_names": ["applied", "stale", "out_of_order"],
        "dropped_commands": 0,
    }
    assert [meta for _, _, meta in files] == [
        {
            **run_meta,
            "robot_id": "r01",
            "schedule_start_ns": runs[0].schedule.start_ns,
            "dropped_ticks": 0,
        },
        {
            **run_meta,
            "robot_id": "r02",
            "schedule_start_ns": runs[1].schedule.start_ns,
            "dropped_ticks": 1,
        },
    ]


@pytest.mark.parametrize(
    ("fatal_codes", "status"),
    [([None], 0), ([None, None, None], 0), ([4001], 1), ([None, 1008, None], 1)],
)
def test_the_fleet_exits_1_when_a_robot_stopped_on_a_final_close_and_0_otherwise(
    fatal_codes: list[int | None], status: int
) -> None:
    assert exit_status(fatal_codes) == status


def test_a_samples_directory_that_cannot_be_created_exits_2_before_any_robot_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")

    # A run that got as far as its robots would last its duration and return 0 or 1.
    status = main(
        ["--robots", "1", "--duration-s", "3600", "--samples-dir", str(blocker / "samples")]
    )

    assert status == 2
    assert capsys.readouterr().err.startswith("robot.fleet: cannot create --samples-dir")


def test_the_samples_directory_is_checked_before_the_plan_is_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    plan = write_plan(tmp_path, plan_document())

    status = main(["--plan", str(plan), "--samples-dir", str(blocker)])

    assert status == 2
    assert len(capsys.readouterr().err.splitlines()) == 1
