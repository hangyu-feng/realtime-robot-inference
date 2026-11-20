"""Tests for the scenario timelines of DESIGN.md 11.2."""

import pytest

from bench.scenarios import (
    RobotPlan,
    Scenario,
    Window,
    disconnect,
    late_joiner,
    massive,
    soak,
    steady,
    sweep,
)


def test_steady_runs_one_robot_for_30_s() -> None:
    assert steady() == Scenario(
        "steady", 30.0, (RobotPlan("a", 0.0, None),), (Window("all", 0.0, 30.0),)
    )


def test_late_joiner_starts_b_at_10_s_with_windows_before_and_after_the_join() -> None:
    assert late_joiner() == Scenario(
        "late_joiner",
        30.0,
        (RobotPlan("a", 0.0, None), RobotPlan("b", 10.0, None)),
        (Window("all", 0.0, 30.0), Window("before", 0.0, 10.0), Window("after", 10.0, 30.0)),
    )


def test_disconnect_stops_c_at_20_s_while_the_survivors_run_to_the_end() -> None:
    assert disconnect() == Scenario(
        "disconnect",
        40.0,
        (RobotPlan("a", 0.0, None), RobotPlan("b", 0.0, None), RobotPlan("c", 0.0, 20.0)),
        (Window("all", 0.0, 40.0), Window("before", 0.0, 20.0), Window("after", 20.0, 40.0)),
    )


@pytest.mark.parametrize(
    ("robot_count", "robot_ids"),
    [(1, ["r01"]), (3, ["r01", "r02", "r03"]), (10, [f"r{number:02d}" for number in range(1, 11)])],
)
def test_sweep_numbers_its_robots_with_two_digits_and_runs_them_for_30_s(
    robot_count: int, robot_ids: list[str]
) -> None:
    assert sweep(robot_count) == Scenario(
        "sweep",
        30.0,
        tuple(RobotPlan(robot_id, 0.0, None) for robot_id in robot_ids),
        (Window("all", 0.0, 30.0),),
    )


def test_sweep_takes_up_to_99_robots_whose_ids_still_have_two_digits() -> None:
    assert sweep(99).robots[-1] == RobotPlan("r99", 0.0, None)


@pytest.mark.parametrize("robot_count", [-1, 0, 100])
def test_sweep_refuses_a_robot_count_outside_1_to_99(robot_count: int) -> None:
    with pytest.raises(ValueError, match="1 to 99 robots"):
        sweep(robot_count)


@pytest.mark.parametrize(
    ("scenario", "duration_s"), [(massive(), 30.0), (soak(), 300.0)], ids=["massive", "soak"]
)
def test_massive_and_soak_run_50_distinct_robots_for_the_whole_scenario(
    scenario: Scenario, duration_s: float
) -> None:
    assert (len(scenario.robots), scenario.duration_s) == (50, duration_s)
    assert len({plan.robot_id for plan in scenario.robots}) == 50
    assert {(plan.start_s, plan.stop_s) for plan in scenario.robots} == {(0.0, None)}
    assert scenario.windows == (Window("all", 0.0, duration_s),)


@pytest.mark.parametrize(
    "scenario",
    [steady(), late_joiner(), sweep(5), disconnect(), massive(), soak()],
    ids=["steady", "late_joiner", "sweep", "disconnect", "massive", "soak"],
)
def test_every_scenario_has_a_window_all_that_covers_the_run(scenario: Scenario) -> None:
    assert Window("all", 0.0, scenario.duration_s) in scenario.windows
