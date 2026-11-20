"""Properties of the plan of the evidence, over any repeats, specs, code files and timelines."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from hypothesis import given, settings
from hypothesis import strategies as st

from bench.runfiles import EventName
from bench.runplan import (
    MODEL_BENCH_KEY,
    ModelSet,
    RunSpec,
    code_digest,
    code_paths,
    evidence_plan,
    fleet_plans,
    only,
    planned_events,
    remaining,
    run_id,
    spec_digest,
)
from bench.scenarios import RobotPlan, Scenario, Window
from cloud.config import Topology
from robot.settings import LinkSettings

CODE_DIGEST = "c0de" * 16
SERVING_KEYS = [
    "steady",
    "late_joiner",
    "sweep_n1",
    "sweep_n3",
    "sweep_n5",
    "sweep_n10",
    "disconnect",
    "worker_kill",
    "massive",
    "soak",
]
ENTRIES = sorted(
    ["model_bench:single", *(f"{key}:{topology}" for key in SERVING_KEYS for topology in Topology)]
)
RUNS_PER_REPEAT = len(ENTRIES)
# Built once for the properties of a given plan: a plan resolves the gateway's config per run.
PLAN = evidence_plan(4, ModelSet.FAKE, CODE_DIGEST)

# Two digests that start alike, and one that does not: a run id holds the start alone.
_DIGESTS = st.sampled_from(["c0de" * 16, "c0dec0de" + "0" * 56, "feed" * 16])
# Few values, so that two draws are often equal; 2 and 2.0 are one value, as are 0 and -0.0.
_SETTINGS = st.sampled_from([0, 1, 2, 2.0, 2.5, -0.0, "fifo", "latest", None])
_CONFIGS = st.dictionaries(
    st.sampled_from(["queue_policy", "threads", "frame_rate_hz"]),
    _SETTINGS | st.lists(_SETTINGS, max_size=2) | st.dictionaries(st.just("depth"), _SETTINGS),
)
_LINKS = st.builds(
    LinkSettings, delay_ms=st.sampled_from([0.0, 20.0]), jitter_ms=st.sampled_from([0.0, 5.0])
)
_COMPARISONS = st.sampled_from(
    [(None, None), ("cls_cap_1", "a"), ("cls_cap_1", "b"), ("depth_cap_2", "a")]
)


@st.composite
def _specs(draw: st.DrawFn) -> RunSpec:
    candidate, arm = draw(_COMPARISONS)
    return RunSpec(
        scenario_key=draw(st.sampled_from(["steady", "massive"])),
        topology=draw(st.sampled_from(Topology)),
        phase_mode=draw(st.sampled_from(["random", "aligned", None])),
        model_set=draw(st.sampled_from(ModelSet)),
        server_config=draw(_CONFIGS),
        uplink=draw(_LINKS),
        downlink=draw(_LINKS),
        extra_compose=draw(st.sampled_from([None, "e" * 64])),
        candidate=candidate,
        arm=arm,
        code_digest=draw(_DIGESTS),
        repeat=draw(st.integers(min_value=1, max_value=3)),
    )


@st.composite
def _scenarios(draw: st.DrawFn) -> Scenario:
    duration_s = draw(st.floats(min_value=1.0, max_value=600.0))
    robot_ids = draw(
        st.lists(st.text(alphabet="abcdr0123", min_size=1, max_size=3), min_size=1, unique=True)
    )
    starts_s = st.sampled_from([0.0, duration_s / 4])
    stops_s = st.sampled_from([None, duration_s / 2, duration_s])
    robots = tuple(RobotPlan(robot_id, draw(starts_s), draw(stops_s)) for robot_id in robot_ids)
    return Scenario("drawn", duration_s, robots, (Window("all", 0.0, duration_s),))


def _topologies_by_key(runs: list[RunSpec]) -> dict[str, list[Topology]]:
    """Return the topologies of each serving scenario of one repeat, in the order they run."""
    order: dict[str, list[Topology]] = {key: [] for key in SERVING_KEYS}
    for spec in runs[1:]:
        order[spec.scenario_key].append(spec.topology)
    return order


def _sweep_counts(runs: list[RunSpec]) -> list[int]:
    """Return the robot counts of the sweep runs of one repeat, in the order they run."""
    keys = dict.fromkeys(spec.scenario_key for spec in runs)
    return [int(key.removeprefix("sweep_n")) for key in keys if key.startswith("sweep_n")]


@given(first=_specs(), second=_specs())
def test_two_specs_share_a_digest_exactly_when_they_differ_in_nothing_but_repeat_and_code(
    first: RunSpec, second: RunSpec
) -> None:
    same_run = replace(second, repeat=first.repeat, code_digest=first.code_digest) == first

    assert (spec_digest(first) == spec_digest(second)) is same_run


@given(first=_specs(), second=_specs())
def test_two_specs_share_a_run_id_exactly_when_they_share_digest_code_start_and_repeat(
    first: RunSpec, second: RunSpec
) -> None:
    first_parts = (spec_digest(first), first.code_digest[:8], first.repeat)
    second_parts = (spec_digest(second), second.code_digest[:8], second.repeat)

    assert (run_id(first) == run_id(second)) is (first_parts == second_parts)


@given(spec=_specs())
def test_a_spec_read_back_from_its_json_keeps_its_id(spec: RunSpec) -> None:
    assert run_id(RunSpec.from_json(spec.to_json())) == run_id(spec)


_FILES = st.dictionaries(st.text(alphabet="ab/.", max_size=6), st.binary(max_size=6), max_size=4)


@given(first=_FILES, second=_FILES)
def test_two_sets_of_code_files_share_a_digest_exactly_when_they_are_the_same(
    first: dict[str, bytes], second: dict[str, bytes]
) -> None:
    assert (code_digest(first) == code_digest(second)) is (first == second)


_DIRECTORIES = st.sampled_from(["", "cloud/", "bench/", "tests/", "docs/cloud/", "workers/models/"])
_NAMES = st.sampled_from(["run.py", "evidence.py", "main.py", "Dockerfile", "pyproject.toml"])
_PATHS = st.lists(st.builds(str.__add__, _DIRECTORIES, _NAMES))


@given(listed=_PATHS)
def test_the_code_paths_are_a_sorted_selection_that_selecting_again_does_not_change(
    listed: list[str],
) -> None:
    paths = code_paths(listed)

    assert paths == sorted(set(paths))
    assert set(paths) <= set(listed)
    assert code_paths(paths) == paths
    assert not {"pyproject.toml", "bench/evidence.py", "tests/run.py"} & set(paths)


@settings(max_examples=20)
@given(repeats=st.integers(min_value=0, max_value=4), model_set=st.sampled_from(ModelSet))
def test_every_repeat_runs_the_model_benchmark_first_and_each_scenario_once_per_topology(
    repeats: int, model_set: ModelSet
) -> None:
    plan = evidence_plan(repeats, model_set, CODE_DIGEST)
    by_repeat = [
        plan[start : start + RUNS_PER_REPEAT] for start in range(0, len(plan), RUNS_PER_REPEAT)
    ]

    assert len(plan) == repeats * RUNS_PER_REPEAT
    assert [{spec.repeat for spec in runs} for runs in by_repeat] == [
        {repeat} for repeat in range(1, repeats + 1)
    ]
    assert [
        sorted(f"{spec.scenario_key}:{spec.topology}" for spec in runs) for runs in by_repeat
    ] == [ENTRIES] * repeats
    assert [(runs[0].scenario_key, runs[0].model_set) for runs in by_repeat] == [
        (MODEL_BENCH_KEY, ModelSet.REAL)
    ] * repeats
    assert {spec.model_set for runs in by_repeat for spec in runs[1:]} <= {model_set}
    assert len({run_id(spec) for spec in plan}) == len(plan)


@given(repeat=st.integers(min_value=1, max_value=3))
def test_consecutive_repeats_run_the_topologies_and_the_sweep_in_opposite_orders(
    repeat: int,
) -> None:
    one = PLAN[(repeat - 1) * RUNS_PER_REPEAT : repeat * RUNS_PER_REPEAT]
    next_one = PLAN[repeat * RUNS_PER_REPEAT : (repeat + 1) * RUNS_PER_REPEAT]

    assert _topologies_by_key(one) == {
        key: topologies[::-1] for key, topologies in _topologies_by_key(next_one).items()
    }
    assert _sweep_counts(one) == _sweep_counts(next_one)[::-1]
    assert sorted(_sweep_counts(one)) == [1, 3, 5, 10]


@given(finished=st.sets(st.integers(min_value=0, max_value=len(PLAN) - 1)))
def test_remaining_is_the_plan_without_its_finished_runs_in_the_plan_s_order(
    finished: set[int],
) -> None:
    finished_ids = {run_id(PLAN[index]) for index in finished}

    assert remaining(PLAN, finished_ids) == [
        spec for index, spec in enumerate(PLAN) if index not in finished
    ]


@given(entries=st.sets(st.sampled_from(ENTRIES)))
def test_only_is_the_plan_s_runs_of_the_entries_in_the_plan_s_order(entries: set[str]) -> None:
    assert only(PLAN, entries) == [
        spec for spec in PLAN if f"{spec.scenario_key}:{spec.topology}" in entries
    ]


@given(
    scenario=_scenarios(),
    t0_mono_ns=st.integers(min_value=0, max_value=2**61),
    seed=st.integers(min_value=0),
)
def test_every_robot_is_in_one_fleet_plan_and_a_killed_robot_is_alone_in_its_own(
    scenario: Scenario, t0_mono_ns: int, seed: int
) -> None:
    processes = fleet_plans(scenario, t0_mono_ns, seed)
    planned = [robot for process in processes for robot in process.plan.robots]
    alone = [process for process in processes if process.name != "fleet"]
    killed = [robot.robot_id for robot in scenario.robots if robot.stop_s is not None]

    assert sorted(planned, key=lambda robot: robot.robot_id) == sorted(
        (replace(robot, stop_s=None) for robot in scenario.robots),
        key=lambda robot: robot.robot_id,
    )
    assert [[robot.robot_id for robot in process.plan.robots] for process in alone] == [
        [robot_id] for robot_id in killed
    ]
    assert [process.name for process in alone] == [f"fleet_{robot_id}" for robot_id in killed]
    assert len(processes) - len(alone) == min(1, len(scenario.robots) - len(killed))
    assert {(process.plan.t0_mono_ns, process.plan.duration_s) for process in processes} == {
        (t0_mono_ns, scenario.duration_s)
    }
    assert {process.seed for process in processes if process.name == "fleet"} <= {seed}


@given(scenario=_scenarios())
def test_the_planned_events_follow_the_timeline_in_time_order(scenario: Scenario) -> None:
    events = planned_events(scenario)
    names = Counter((event.name, event.robot_id) for event in events)
    robots = scenario.robots

    assert [event.t_s for event in events] == sorted(event.t_s for event in events)
    assert (events[-1].name, events[-1].t_s) == (EventName.SCENARIO_END, scenario.duration_s)
    assert set(names.values()) == {1}
    assert names.keys() == {
        (EventName.SCENARIO_END, None),
        *((EventName.ROBOT_START, robot.robot_id) for robot in robots),
        *((EventName.ROBOT_JOIN, robot.robot_id) for robot in robots if robot.start_s > 0),
        *((EventName.ROBOT_STOP, robot.robot_id) for robot in robots if robot.stop_s is None),
        *((EventName.ROBOT_KILL, robot.robot_id) for robot in robots if robot.stop_s is not None),
    }
    assert {
        (event.robot_id, event.t_s) for event in events if event.name is EventName.ROBOT_KILL
    } == {(robot.robot_id, robot.stop_s) for robot in robots if robot.stop_s is not None}
    assert {
        (event.robot_id, event.t_s) for event in events if event.name is EventName.ROBOT_START
    } == {(robot.robot_id, robot.start_s) for robot in robots}
