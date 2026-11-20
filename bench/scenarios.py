"""The scenario timelines of DESIGN.md 11.2, as data.

The capacity model simulates them and the run harness plays them against the service, so a
prediction and a measurement of one demo describe the same robots, start and stop times and
windows.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotPlan:
    """When one robot runs: from start_s until stop_s, or to the end of the run when it is None."""

    robot_id: str
    start_s: float
    stop_s: float | None


@dataclass(frozen=True)
class Window:
    """A named span of a run over which results are reported."""

    name: str
    start_s: float
    end_s: float


@dataclass(frozen=True)
class WorkerKill:
    """A SIGKILL of one model's worker process at a planned time from t0."""

    model: str
    at_s: float


@dataclass(frozen=True)
class Scenario:
    """A demo's timeline. Its windows always include one named 'all' that covers the whole run."""

    name: str
    duration_s: float
    robots: tuple[RobotPlan, ...]
    windows: tuple[Window, ...]
    # The worker the scenario kills; None in every scenario that kills none.
    fault: WorkerKill | None = None


# ASSUMPTION: A18. Five demos are done, as the brief lists them, though its evaluation text
# says four: steady, late_joiner, sweep, disconnect and massive, whose memory check is the soak.
def steady() -> Scenario:
    """Return one robot for 30 s (DESIGN.md 9.1)."""
    return Scenario("steady", 30.0, (RobotPlan("a", 0.0, None),), (Window("all", 0.0, 30.0),))


def late_joiner() -> Scenario:
    """Return robot a alone for 10 s, then robots a and b for 20 s (DESIGN.md 9.2)."""
    return Scenario(
        "late_joiner",
        30.0,
        (RobotPlan("a", 0.0, None), RobotPlan("b", 10.0, None)),
        (Window("all", 0.0, 30.0), Window("before", 0.0, 10.0), Window("after", 10.0, 30.0)),
    )


def sweep(robot_count: int) -> Scenario:
    """Return one run of the scaling sweep: robot_count robots for 30 s (DESIGN.md 9.3).

    ValueError for a count outside 1 to 99.
    """
    return _fleet("sweep", robot_count, 30.0)


def disconnect() -> Scenario:
    """Return robots a, b and c for 20 s, then a and b for 20 s more (DESIGN.md 9.4).

    The survivors are the robots that run to the end.
    """
    return Scenario(
        "disconnect",
        40.0,
        (
            RobotPlan("a", 0.0, None),
            RobotPlan("b", 0.0, None),
            # ASSUMPTION: A11. The stop is a SIGKILL of the robot's process, with no clean
            # close. The runner decides how to stop it; the simulation disconnects it at stop_s.
            RobotPlan("c", 0.0, 20.0),
        ),
        (Window("all", 0.0, 40.0), Window("before", 0.0, 20.0), Window("after", 20.0, 40.0)),
    )


def worker_kill() -> Scenario:
    """Return robots a and b for 40 s, with the depth worker killed at 20 s (DESIGN.md 11.2).

    Depth is mid-batch whenever robots stream, so the kill falls inside a batch.
    """
    # The timeline is the disconnect's, with two robots: windows of equal length before and
    # after the kill, and time after it for a restart that loads and calibrates a real model.
    return Scenario(
        "worker_kill",
        40.0,
        (RobotPlan("a", 0.0, None), RobotPlan("b", 0.0, None)),
        (Window("all", 0.0, 40.0), Window("before", 0.0, 20.0), Window("after", 20.0, 40.0)),
        fault=WorkerKill("depth", 20.0),
    )


def massive() -> Scenario:
    """Return 50 robots for 30 s (DESIGN.md 9.5)."""
    return _fleet("massive", 50, 30.0)


def soak() -> Scenario:
    """Return 50 robots for 5 minutes, the memory check of the massive demo (DESIGN.md 9.5)."""
    return _fleet("soak", 50, 300.0)


def _fleet(name: str, robot_count: int, duration_s: float) -> Scenario:
    """Return robots r01, r02, ... that all run for the whole scenario."""
    # The ids carry two digits, so they sort in robot order.
    if not 1 <= robot_count <= 99:
        raise ValueError(f"a fleet has 1 to 99 robots, got {robot_count}")
    robots = tuple(RobotPlan(f"r{number:02d}", 0.0, None) for number in range(1, robot_count + 1))
    return Scenario(name, duration_s, robots, (Window("all", 0.0, duration_s),))
