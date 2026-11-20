"""A short worker-kill run through the evidence runner, in each topology, on the fake models.

The fault helper kills the depth worker from the gateway's container while the scenario
driver plays, and the run finishes gated with the helper's fault event. It builds the image
and starts containers, so it is marked docker: make test leaves it out, and make
test-docker runs it on the host.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from bench.gates import suite_gate_record
from bench.runfiles import EVENTS_FILE, GATE_FILE, SUMMARY_FILE, Event, EventName, Gate, ModelSet
from bench.runplan import evidence_plan
from bench.scenarios import RobotPlan, Scenario, Window, WorkerKill
from cloud.config import Topology

pytestmark = [
    pytest.mark.docker,
    pytest.mark.skipif(shutil.which("docker") is None, reason="needs the host's docker CLI"),
]

# Two robots, and the depth worker killed early enough that its restart, on the fake models,
# ends well before the run does.
DURATION_S = 8.0
KILL_S = 3.0
SCENARIO = Scenario(
    "worker_kill",
    DURATION_S,
    (RobotPlan("a", 0.0, None), RobotPlan("b", 0.0, None)),
    (
        Window("all", 0.0, DURATION_S),
        Window("before", 0.0, KILL_S),
        Window("after", KILL_S, DURATION_S),
    ),
    fault=WorkerKill("depth", KILL_S),
)
# A tests gate that passed, built as the runner builds one from the suite's output.
PASSING_SUITE_OUTPUT = b"5 passed in 1.00s\n"


@dataclass(frozen=True)
class Played:
    """One worker-kill run, in one topology."""

    result: RunResult
    run_dir: Path


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
        if spec.scenario_key == "worker_kill" and spec.topology is topology
    )
    result = run_one(spec, SCENARIO, None, invocation)
    return Played(result, invocation.results / result.path)


def _summary(played: Played) -> dict[str, Any]:
    summary: dict[str, Any] = json.loads((played.run_dir / SUMMARY_FILE).read_text())
    return summary


def test_a_worker_kill_run_finishes_gated_and_leaves_no_container(played: Played) -> None:
    gate = Gate.from_json((played.run_dir / GATE_FILE).read_text())

    assert (played.result.outcome, played.result.detail) == (Outcome.FINISHED, "gated"), (
        played.result.line()
    )
    assert is_finished(played.run_dir)
    assert gate.gated
    assert docker_ops.project_containers(PROJECT, DOCKER_TIMEOUT_S) == []


def test_the_fault_event_names_the_depth_worker_killed_soon_after_its_planned_time(
    played: Played,
) -> None:
    events = [
        Event.from_json(line) for line in (played.run_dir / EVENTS_FILE).read_text().splitlines()
    ]

    (fault,) = (event for event in events if event.name is EventName.FAULT)
    assert fault.model == "depth"
    assert isinstance(fault.pid, int)
    # Read on the clock once the kill has returned, so at the planned time or later.
    assert KILL_S <= fault.t_s < KILL_S + 1.0
    # The runner adds it once the driver has finished.
    assert events[-1] == fault


def test_the_depth_worker_alone_restarted_and_served_depth_again(played: Played) -> None:
    summary = _summary(played)

    (restarted,) = (check for check in summary["checks"] if check["name"] == "worker_restarted")
    assert summary["fault"]["outage_ms"] > 0
    assert restarted["passed"] is True, restarted
    assert restarted["observed"]["restarts"] == {"cls": 0, "depth": 1}
