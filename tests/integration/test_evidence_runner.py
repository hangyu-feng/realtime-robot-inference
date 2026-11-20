"""The evidence runner end to end: a short serving run in each topology, on the fake models.

It builds the image, starts containers and reads the tree with git, so it is marked docker:
make test leaves it out, and make test-docker runs it on the host.
"""

from __future__ import annotations

import gzip
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytest

from bench import docker_ops
from bench.evidence import (
    BUILD_TIMEOUT_S,
    DOCKER_TIMEOUT_S,
    PROJECT,
    Invocation,
    Outcome,
    RunResult,
    is_finished,
    prepare_invocation,
    repository_state,
    run_compose,
    run_one,
)
from bench.gates import EXPECTED_LIMITS, suite_gate_record
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    EVENTS_FILE,
    GATE_FILE,
    INSIDE_ENVIRONMENT_FILE,
    PLANS_DIRECTORY,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    SUMMARY_FILE,
    Done,
    Event,
    EventName,
    Gate,
    ModelSet,
    RunEnvironment,
    RunSpec,
    Scrape,
)
from bench.runplan import evidence_plan, run_id
from bench.scenarios import RobotPlan, Scenario, Window
from bench.summarize import summary_is_current
from cloud.config import Topology

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs the host's docker CLI"),
]

NS_PER_S = 1_000_000_000
# A short disconnect: three robots, the third killed half way, so that scrapes, a kill and
# the windows on both sides of it all fit in a few seconds.
DURATION_S = 6.0
KILL_S = 3.0
SCENARIO = Scenario(
    "disconnect",
    DURATION_S,
    (RobotPlan("a", 0.0, None), RobotPlan("b", 0.0, None), RobotPlan("c", 0.0, KILL_S)),
    (
        Window("all", 0.0, DURATION_S),
        Window("before", 0.0, KILL_S),
        Window("after", KILL_S, DURATION_S),
    ),
)
# A tests gate that passed, built as the runner builds one from the suite's output.
PASSING_SUITE_OUTPUT = b"5 passed in 1.00s\n"


@dataclass(frozen=True)
class Played:
    """One run, in one topology, played twice: the second invocation finds it finished."""

    invocation: Invocation
    spec: RunSpec
    first: RunResult
    second: RunResult

    @property
    def run_dir(self) -> Path:
        return self.invocation.results / self.first.path


@pytest.fixture(scope="module")
def invocation(tmp_path_factory: pytest.TempPathFactory) -> Invocation:
    """Build the image, and take a passing tests gate for it without running the suite."""
    results = tmp_path_factory.mktemp("results")
    state = repository_state()
    compose = run_compose(Topology.SINGLE, None, {}, results, state.commit)
    image_digest = docker_ops.build(compose, BUILD_TIMEOUT_S)
    tests_passed = suite_gate_record(
        PASSING_SUITE_OUTPUT, 0, image_digest, state.code_digest, state.commit, state.dirty
    )
    return prepare_invocation(results, image_digest, tests_passed, state)


@pytest.fixture(scope="module", params=list(Topology), ids=str)
def played(request: pytest.FixtureRequest, invocation: Invocation) -> Played:
    topology: Topology = request.param
    (spec,) = (
        spec
        for spec in evidence_plan(1, ModelSet.FAKE, invocation.tests_passed.code_digest)
        if spec.scenario_key == "disconnect" and spec.topology is topology
    )
    first = run_one(spec, SCENARIO, None, invocation)
    second = run_one(spec, SCENARIO, None, invocation)
    return Played(invocation, spec, first, second)


def _events(run_dir: Path) -> list[Event]:
    return [Event.from_json(line) for line in (run_dir / EVENTS_FILE).read_text().splitlines()]


def _scrapes(run_dir: Path) -> list[Scrape]:
    with gzip.open(run_dir / SCRAPES_FILE, "rt", encoding="utf-8") as file:
        return [Scrape.from_json(line) for line in file]


def test_a_run_finishes_gated_with_its_summary(played: Played) -> None:
    assert played.first.outcome is Outcome.FINISHED, played.first.line()
    assert played.first.detail == "gated"
    assert played.first.path == f"raw/disconnect/{run_id(played.spec)}"


def test_a_run_writes_every_file_of_its_layout(played: Played) -> None:
    run_dir = played.run_dir
    names = [
        CONFIG_FILE,
        ENVIRONMENT_FILE,
        INSIDE_ENVIRONMENT_FILE,
        GATE_FILE,
        EVENTS_FILE,
        SCRAPES_FILE,
        DONE_FILE,
        SUMMARY_FILE,
        f"{PLANS_DIRECTORY}/fleet.json",
        f"{PLANS_DIRECTORY}/fleet_c.json",
        f"{SAMPLES_DIRECTORY}/robot_a.npz",
        f"{SAMPLES_DIRECTORY}/robot_b.npz",
    ]

    assert [name for name in names if not (run_dir / name).is_file()] == []
    # A killed robot writes no samples.
    assert not (run_dir / SAMPLES_DIRECTORY / "robot_c.npz").exists()


def test_a_finished_run_has_a_valid_done_and_a_current_summary(played: Played) -> None:
    done = Done.from_json((played.run_dir / DONE_FILE).read_text())

    assert is_finished(played.run_dir)
    assert summary_is_current(played.run_dir, done)


def test_the_gateway_echoed_the_fake_overrides_and_every_gate_passed(played: Played) -> None:
    gate = Gate.from_json((played.run_dir / GATE_FILE).read_text())

    assert gate.config_ok is not None
    assert (gate.config_ok.ok, gate.config_ok.differences) == (True, ())
    assert gate.limits_ok.ok
    assert gate.fresh.ok
    assert gate.image_built.built
    assert gate.gated


def test_a_second_invocation_skips_a_finished_run(played: Played) -> None:
    name = played.run_dir.name

    assert played.second == RunResult(played.first.path, Outcome.SKIPPED, "")
    assert [path.name for path in played.run_dir.parent.glob(f"{name}*")] == [name]


def test_the_environment_holds_the_gateway_s_limits_and_the_robots_container_s_apart(
    played: Played,
) -> None:
    environment = RunEnvironment.from_json((played.run_dir / ENVIRONMENT_FILE).read_text())
    split = played.spec.topology is Topology.SPLIT

    assert environment.inside.cgroup.cpu_limit_cores == EXPECTED_LIMITS.cpu_limit_cores
    assert (environment.robots_cgroup is not None) == split
    assert {scrape.fleet_cgroup is not None for scrape in _scrapes(played.run_dir)} == {split}


def test_every_file_of_the_run_belongs_to_the_host_user(played: Played) -> None:
    paths = [played.run_dir, *played.run_dir.rglob("*")]

    assert [path for path in paths if path.stat().st_uid != os.getuid()] == []


def test_the_kill_is_an_event_at_the_time_its_signal_was_sent(played: Played) -> None:
    events = _events(played.run_dir)
    kills = [event for event in events if event.name is EventName.ROBOT_KILL]

    assert events[0].name is EventName.T0
    assert [event.robot_id for event in kills] == ["c"]
    # Read on the clock once the kill has returned, so later than planned, never at it.
    assert kills[0].t_s > KILL_S


def test_the_final_poll_finds_the_gateway_s_last_counts(played: Played) -> None:
    (final_poll,) = [
        event for event in _events(played.run_dir) if event.name is EventName.FINAL_POLL
    ]

    assert final_poll.settled is True


def test_the_scrapes_are_taken_at_the_whole_seconds_of_time_zero(played: Played) -> None:
    t0_ns = _events(played.run_dir)[0].t0_mono_ns
    scrapes = _scrapes(played.run_dir)
    assert t0_ns is not None

    assert [scrape.k for scrape in scrapes] == [*range(-1, round(DURATION_S) + 1), None]
    assert all(scrape.ok for scrape in scrapes)
    early = [
        scrape.k
        for scrape in scrapes
        if scrape.k is not None and scrape.driver_before_ns < t0_ns + scrape.k * NS_PER_S
    ]
    assert early == []


def test_a_run_without_done_is_kept_as_an_attempt_and_played_again(played: Played) -> None:
    (played.run_dir / DONE_FILE).unlink()

    again = run_one(played.spec, SCENARIO, None, played.invocation)

    assert again.outcome is Outcome.FINISHED, again.line()
    assert (
        again.detail == f"the directory before it is kept as {played.run_dir.name}.attempt1; gated"
    )
    assert (played.run_dir.parent / f"{played.run_dir.name}.attempt1").is_dir()
    assert is_finished(played.run_dir)
    assert docker_ops.project_containers(PROJECT, DOCKER_TIMEOUT_S) == []


def test_a_build_of_an_unchanged_tree_gives_the_same_image(invocation: Invocation) -> None:
    # The tests gate is stored by image digest, so it is found again only if this holds.
    compose = run_compose(
        Topology.SINGLE, None, {}, invocation.results, invocation.tests_passed.commit
    )

    assert docker_ops.build(compose, BUILD_TIMEOUT_S) == invocation.image_digest
