"""Tests for the plan of the evidence: run identity, the code digest, order and fleet plans."""

from __future__ import annotations

import ast
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import pytest

from bench.candidates import CANDIDATES, Candidate
from bench.runfiles import PHASE_MODES, Event, EventName, FleetPlan
from bench.runplan import (
    MODEL_BENCH_KEY,
    FleetProcess,
    ModelSet,
    RunSpec,
    code_digest,
    code_paths,
    compare_plan,
    evidence_plan,
    fleet_plans,
    only,
    overrides_for,
    planned_events,
    remaining,
    resolved_config,
    run_id,
    scenario_for,
    spec_digest,
)
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
    worker_kill,
)
from cloud.config import ServerConfig, Topology, config_echo
from common.errors import ConfigError
from common.protocol import PROTOCOL_VERSION
from robot.fleet import read_plan
from robot.settings import LinkSettings

REPOSITORY = Path(__file__).resolve().parents[2]

CODE_DIGEST = "c0de" * 16
OTHER_CODE_DIGEST = "feed" * 16

FAKE_PLAN = evidence_plan(2, ModelSet.FAKE, CODE_DIGEST)
# The first serving run of the plan: steady, in the single topology.
SPEC = FAKE_PLAN[1]
COMPARISON_SPEC = replace(SPEC, candidate="cls_cap_1", arm="a")
CANDIDATE_BY_NAME = {candidate.name: candidate for candidate in CANDIDATES}

FAKE_OVERRIDES = {
    "models": {
        "cls": {"runtime": "fake", "fake_frame_ms": 2.0},
        "depth": {"runtime": "fake", "fake_frame_ms": 40.0, "batch_sizes": [1]},
    }
}

REPEAT_1 = [
    "model_bench:single",
    "steady:single",
    "steady:split",
    "late_joiner:single",
    "late_joiner:split",
    "sweep_n1:single",
    "sweep_n1:split",
    "sweep_n3:single",
    "sweep_n3:split",
    "sweep_n5:single",
    "sweep_n5:split",
    "sweep_n10:single",
    "sweep_n10:split",
    "disconnect:single",
    "disconnect:split",
    "worker_kill:single",
    "worker_kill:split",
    "massive:single",
    "massive:split",
    "soak:single",
    "soak:split",
]
REPEAT_2 = [
    "model_bench:single",
    "steady:split",
    "steady:single",
    "late_joiner:split",
    "late_joiner:single",
    "sweep_n10:split",
    "sweep_n10:single",
    "sweep_n5:split",
    "sweep_n5:single",
    "sweep_n3:split",
    "sweep_n3:single",
    "sweep_n1:split",
    "sweep_n1:single",
    "disconnect:split",
    "disconnect:single",
    "worker_kill:split",
    "worker_kill:single",
    "massive:split",
    "massive:single",
    "soak:split",
    "soak:single",
]

# One robot that runs to the end, one that joins late, and two that the scenario kills.
TWO_KILLS = Scenario(
    "two_kills",
    40.0,
    (
        RobotPlan("a", 0.0, None),
        RobotPlan("b", 5.0, None),
        RobotPlan("c", 0.0, 20.0),
        RobotPlan("d", 10.0, 30.0),
    ),
    (Window("all", 0.0, 40.0),),
)


def _entries(plan: list[RunSpec]) -> list[str]:
    return [f"{spec.scenario_key}:{spec.topology}" for spec in plan]


def _module_file(name: str) -> str | None:
    """Return the repository file of a dotted module or package name; None for any other name."""
    stem = name.replace(".", "/")
    files = [
        path for path in (f"{stem}.py", f"{stem}/__init__.py") if (REPOSITORY / path).is_file()
    ]
    return files[0] if files else None


def _imported_files(path: str) -> set[str]:
    """Return the repository modules that a module's import statements name.

    Imports under TYPE_CHECKING are left out: they are not executed. A package counts where
    an import names it, not for every module inside it.
    """
    names: list[str] = []
    pending: list[ast.AST] = [ast.parse((REPOSITORY / path).read_text(encoding="utf-8"))]
    while pending:
        node = pending.pop()
        if isinstance(node, ast.If) and ast.unparse(node.test).endswith("TYPE_CHECKING"):
            pending.extend(node.orelse)
            continue
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
        pending.extend(ast.iter_child_nodes(node))
    return {file for name in names if (file := _module_file(name)) is not None}


def _reachable_files(entry: str) -> set[str]:
    """Return an entry module and every repository module it imports, directly or not."""
    reached: set[str] = set()
    pending = [entry]
    while pending:
        path = pending.pop()
        if path not in reached:
            reached.add(path)
            pending.extend(_imported_files(path))
    return reached


@pytest.mark.parametrize(
    ("key", "scenario"),
    [
        ("steady", steady()),
        ("late_joiner", late_joiner()),
        ("sweep_n1", sweep(1)),
        ("sweep_n3", sweep(3)),
        ("sweep_n5", sweep(5)),
        ("sweep_n10", sweep(10)),
        ("disconnect", disconnect()),
        ("worker_kill", worker_kill()),
        ("massive", massive()),
        ("soak", soak()),
    ],
)
def test_a_scenario_key_names_its_timeline(key: str, scenario: Scenario) -> None:
    assert scenario_for(key) == scenario


@pytest.mark.parametrize("key", ["model_bench", "sweep", "sweep_n2", "sweep_n05", ""])
def test_a_key_that_is_no_serving_scenario_is_refused(key: str) -> None:
    with pytest.raises(ValueError, match="is not a serving scenario"):
        scenario_for(key)


def test_the_real_model_set_changes_no_setting() -> None:
    assert overrides_for(ModelSet.REAL, {}) == {}
    assert overrides_for(ModelSet.REAL, {"skip_rule": False}) == {"skip_rule": False}


def test_the_fake_model_set_gives_both_models_a_fixed_service_time_and_depth_one_size() -> None:
    assert overrides_for(ModelSet.FAKE, {}) == FAKE_OVERRIDES


def test_the_null_model_set_gives_both_models_the_null_runtime() -> None:
    assert overrides_for(ModelSet.NULL, {}) == {
        "models": {"cls": {"runtime": "null"}, "depth": {"runtime": "null"}}
    }


def test_given_overrides_are_laid_over_those_of_the_model_set_field_by_field() -> None:
    given = {"skip_rule": False, "models": {"depth": {"fake_frame_ms": 80.0, "threads": 2}}}

    assert overrides_for(ModelSet.FAKE, given) == {
        "skip_rule": False,
        "models": {
            "cls": {"runtime": "fake", "fake_frame_ms": 2.0},
            "depth": {"runtime": "fake", "fake_frame_ms": 80.0, "batch_sizes": [1], "threads": 2},
        },
    }
    assert given == {"skip_rule": False, "models": {"depth": {"fake_frame_ms": 80.0, "threads": 2}}}


def test_a_model_that_only_the_given_overrides_name_is_kept() -> None:
    given = {"models": {"cls": {"forced_batch_cap": 1}}}

    assert overrides_for(ModelSet.REAL, given) == given


def test_the_resolved_config_is_the_echo_of_the_gateway_s_config_without_a_version() -> None:
    assert resolved_config(Topology.SINGLE, {}) == config_echo(
        ServerConfig(), version="", protocol_version=PROTOCOL_VERSION
    )


def test_the_resolved_config_of_the_fake_set_holds_its_settings() -> None:
    models: Any = resolved_config(Topology.SPLIT, overrides_for(ModelSet.FAKE, {}))["models"]

    assert (models["cls"]["runtime"], models["cls"]["fake_frame_ms"]) == ("fake", 2.0)
    assert (models["depth"]["runtime"], models["depth"]["fake_frame_ms"]) == ("fake", 40.0)
    assert models["depth"]["batch_sizes"] == [1]


def test_overrides_that_the_gateway_would_refuse_are_refused() -> None:
    with pytest.raises(ConfigError, match="no_such_setting is not a setting"):
        resolved_config(Topology.SINGLE, {"no_such_setting": 1})


def test_a_spec_digest_is_the_start_of_the_sha_256_of_the_spec_without_repeat_and_code() -> None:
    spec = RunSpec(
        scenario_key="steady",
        topology=Topology.SINGLE,
        phase_mode="random",
        model_set=ModelSet.FAKE,
        server_config={"b": 2.0, "a": [1, 2.5]},
        uplink=LinkSettings(),
        downlink=LinkSettings(delay_ms=20.0, jitter_ms=0.5),
        extra_compose=None,
        candidate=None,
        arm=None,
        code_digest=CODE_DIGEST,
        repeat=3,
    )
    canonical = (
        '{"arm":null,"candidate":null,"downlink":{"delay_ms":20,"jitter_ms":0.5},'
        '"extra_compose":null,"model_set":"fake","phase_mode":"random","scenario_key":"steady",'
        '"server_config":{"a":[1,2.5],"b":2},"topology":"single",'
        '"uplink":{"delay_ms":0,"jitter_ms":0}}'
    )

    assert spec_digest(spec) == hashlib.sha256(canonical.encode()).hexdigest()[:12]


def test_a_run_id_joins_the_topology_the_spec_digest_the_code_digest_and_the_repeat() -> None:
    split = replace(SPEC, topology=Topology.SPLIT, repeat=12)

    assert run_id(SPEC) == f"single-{spec_digest(SPEC)}-c0dec0de-r1"
    assert run_id(split) == f"split-{spec_digest(split)}-c0dec0de-r12"
    assert re.fullmatch(r"single-[0-9a-f]{12}-c0dec0de-r1", run_id(SPEC))


def test_a_spec_read_back_from_its_json_has_the_same_id() -> None:
    assert run_id(RunSpec.from_json(SPEC.to_json())) == run_id(SPEC)


SPEC_CHANGES: list[Mapping[str, Any]] = [
    {"scenario_key": "late_joiner"},
    {"topology": Topology.SPLIT},
    {"phase_mode": "aligned"},
    {"model_set": ModelSet.NULL},
    {"server_config": {**SPEC.server_config, "skip_rule": False}},
    {"server_config": resolved_config(Topology.SINGLE, overrides_for(ModelSet.REAL, {}))},
    {"uplink": LinkSettings(delay_ms=20.0)},
    {"downlink": LinkSettings(jitter_ms=5.0)},
    {"extra_compose": "e" * 64},
    {"candidate": "cls_cap_1", "arm": "a"},
]


@pytest.mark.parametrize("change", SPEC_CHANGES, ids=lambda change: "+".join(change))
def test_a_change_to_any_field_of_a_spec_changes_its_id(change: Mapping[str, Any]) -> None:
    changed = replace(SPEC, **change)

    assert spec_digest(changed) != spec_digest(SPEC)
    assert run_id(changed) != run_id(SPEC)


def test_the_two_arms_of_a_comparison_have_different_ids() -> None:
    assert spec_digest(replace(COMPARISON_SPEC, arm="b")) != spec_digest(COMPARISON_SPEC)


def test_every_field_of_a_spec_but_the_repeat_and_the_code_is_in_its_digest() -> None:
    changed_fields = {name for change in SPEC_CHANGES for name in change}

    assert changed_fields | {"repeat", "code_digest"} == {field.name for field in fields(RunSpec)}


def test_repeats_of_a_spec_share_its_digest_and_differ_in_the_last_part_of_their_ids() -> None:
    second = replace(SPEC, repeat=2)

    assert spec_digest(second) == spec_digest(SPEC)
    assert run_id(SPEC).removesuffix("-r1") == run_id(second).removesuffix("-r2")
    assert run_id(SPEC) != run_id(second)


def test_a_code_change_changes_the_id_and_not_the_spec_digest() -> None:
    other_code = replace(SPEC, code_digest=OTHER_CODE_DIGEST)

    assert spec_digest(other_code) == spec_digest(SPEC)
    assert run_id(other_code) != run_id(SPEC)


@pytest.mark.parametrize(
    "setting", ["overhead_allowance_ms", "pacing_headroom_cores", "skip_max_unserved_ms"]
)
def test_an_override_written_as_2_and_as_2_0_gives_one_id(setting: str) -> None:
    whole = replace(SPEC, server_config=resolved_config(Topology.SINGLE, {setting: 2}))
    decimal = replace(SPEC, server_config=resolved_config(Topology.SINGLE, {setting: 2.0}))

    assert whole == decimal
    assert run_id(whole) == run_id(decimal)


def test_specs_that_hold_one_number_as_2_and_as_2_0_have_one_id() -> None:
    whole = replace(SPEC, server_config={**SPEC.server_config, "overhead_allowance_ms": 2})
    decimal = replace(SPEC, server_config={**SPEC.server_config, "overhead_allowance_ms": 2.0})

    assert run_id(whole) == run_id(decimal)


@pytest.mark.parametrize("model_set", list(ModelSet))
def test_a_split_spec_differs_from_the_single_spec_in_exactly_the_topology_of_its_config(
    model_set: ModelSet,
) -> None:
    single, split = evidence_plan(1, model_set, CODE_DIGEST)[1:3]
    differing = {
        name
        for name in single.server_config
        if single.server_config[name] != split.server_config[name]
    }

    assert (single.scenario_key, split.scenario_key) == ("steady", "steady")
    assert single.server_config.keys() == split.server_config.keys()
    assert differing == {"topology"}
    assert (single.server_config["topology"], split.server_config["topology"]) == (
        "single",
        "split",
    )


def test_the_code_digest_does_not_depend_on_the_order_the_files_are_given_in() -> None:
    files = {"cloud/main.py": b"main", "Dockerfile": b"FROM python", "bench/run.py": b"run"}

    assert code_digest(files) == code_digest(dict(reversed(files.items())))
    assert re.fullmatch(r"[0-9a-f]{64}", code_digest(files))


@pytest.mark.parametrize(
    "changed",
    [
        {"cloud/main.py": b"main ", "robot/fleet.py": b"fleet"},
        {"cloud/main.py": b"main", "robot/fleet.py": b"fleet", "robot/link.py": b""},
        {"cloud/main.py": b"main"},
        {"cloud/main.py": b"main", "robot/fleet2.py": b"fleet"},
        {"cloud/main.py": b"fleet", "robot/fleet.py": b"main"},
        {"cloud/main.pyr": b"main", "obot/fleet.py": b"fleet"},
        {"cloud/main.py": b"mainrobot/fleet.pyfleet"},
    ],
    ids=[
        "a content",
        "an added empty file",
        "a removed file",
        "a renamed file",
        "contents swapped",
        "a letter moved between a path and the next",
        "one file holding the rest",
    ],
)
def test_a_change_to_the_code_files_changes_the_code_digest(changed: dict[str, bytes]) -> None:
    files = {"cloud/main.py": b"main", "robot/fleet.py": b"fleet"}

    assert code_digest(changed) != code_digest(files)


def test_the_code_paths_are_the_service_the_bench_entry_points_and_the_image_files() -> None:
    listed = [
        "workers/models/fake.py",
        "scripts/fetch_models.py",
        "robot/fleet.py",
        "common/aio.py",
        "cloud/main.py",
        "bench/__init__.py",
        "bench/run.py",
        "bench/fault.py",
        "bench/live_check.py",
        "bench/runplan.py",
        "bench/runfiles.py",
        "bench/scenarios.py",
        "bench/model_bench.py",
        "bench/transport_bench.py",
        "bench/stats.py",
        "bench/environment.py",
        "requirements.txt",
        "docker-compose.yml",
        "docker-compose.split.yml",
        "Dockerfile",
    ]

    assert code_paths(listed) == sorted(listed)
    assert code_paths([*listed, *listed]) == sorted(listed)


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "uv.lock",
        "Makefile",
        "DESIGN.md",
        ".dockerignore",
        "bench/evidence.py",
        "bench/summarize.py",
        "bench/candidates.py",
        "bench/gates.py",
        "bench/dashboard.py",
        "bench/dashboard.html",
        "tests/unit/test_runplan.py",
        "docs/cloud/notes.md",
        "cloud.py",
        "results/raw/steady/config.json",
    ],
)
def test_a_file_that_does_not_decide_what_a_run_does_is_not_a_code_path(path: str) -> None:
    assert code_paths([path, "cloud/main.py"]) == ["cloud/main.py"]


@pytest.mark.parametrize(
    "entry",
    ["bench/run.py", "bench/model_bench.py", "bench/transport_bench.py"],
)
def test_the_code_paths_hold_every_repository_module_that_an_entry_point_imports(
    entry: str,
) -> None:
    modules = _reachable_files(entry)

    assert {entry, "bench/environment.py", "common/cgroup.py"} <= modules
    assert code_paths(modules) == sorted(modules)


def test_the_code_paths_hold_every_repository_module_that_the_fault_injector_imports() -> None:
    modules = _reachable_files("bench/fault.py")

    assert {"bench/fault.py", "bench/live_check.py", "bench/runfiles.py"} <= modules
    assert code_paths(modules) == sorted(modules)


def test_the_module_walk_follows_imports_but_not_those_under_type_checking() -> None:
    imported = _imported_files("bench/runplan.py")

    assert {"bench/runfiles.py", "bench/scenarios.py", "cloud/config.py"} <= imported
    assert "bench/candidates.py" not in imported


@pytest.mark.parametrize("model_set", [ModelSet.REAL, ModelSet.FAKE])
def test_two_repeats_run_in_this_order(model_set: ModelSet) -> None:
    plan = evidence_plan(2, model_set, CODE_DIGEST)

    assert _entries(plan) == [*REPEAT_1, *REPEAT_2]
    assert [spec.repeat for spec in plan] == [1] * len(REPEAT_1) + [2] * len(REPEAT_2)


@pytest.mark.parametrize("model_set", list(ModelSet))
def test_the_model_benchmark_runs_on_the_real_models_whatever_the_model_set(
    model_set: ModelSet,
) -> None:
    model_bench, *serving = evidence_plan(1, model_set, CODE_DIGEST)

    assert model_bench == RunSpec(
        scenario_key=MODEL_BENCH_KEY,
        topology=Topology.SINGLE,
        phase_mode=None,
        model_set=ModelSet.REAL,
        server_config=config_echo(ServerConfig(), version="", protocol_version=PROTOCOL_VERSION),
        uplink=LinkSettings(),
        downlink=LinkSettings(),
        extra_compose=None,
        candidate=None,
        arm=None,
        code_digest=CODE_DIGEST,
        repeat=1,
    )
    assert {spec.model_set for spec in serving} == {model_set}


def test_a_serving_run_of_the_evidence_has_random_phases_and_the_config_of_its_set() -> None:
    steady_single = evidence_plan(1, ModelSet.FAKE, CODE_DIGEST)[1]

    assert steady_single == RunSpec(
        scenario_key="steady",
        topology=Topology.SINGLE,
        phase_mode="random",
        model_set=ModelSet.FAKE,
        server_config=resolved_config(Topology.SINGLE, FAKE_OVERRIDES),
        uplink=LinkSettings(),
        downlink=LinkSettings(),
        extra_compose=None,
        candidate=None,
        arm=None,
        code_digest=CODE_DIGEST,
        repeat=1,
    )
    assert steady_single.phase_mode in PHASE_MODES


def test_every_run_of_a_plan_has_its_own_id_and_is_read_back_from_its_json() -> None:
    plan = evidence_plan(3, ModelSet.FAKE, CODE_DIGEST)

    assert len({run_id(spec) for spec in plan}) == len(plan)
    assert [RunSpec.from_json(spec.to_json()) for spec in plan] == plan


def test_a_third_repeat_runs_in_the_order_of_the_first() -> None:
    plan = evidence_plan(3, ModelSet.FAKE, CODE_DIGEST)

    assert _entries(plan) == [*REPEAT_1, *REPEAT_2, *REPEAT_1]


def test_no_repeats_plan_no_run() -> None:
    assert evidence_plan(0, ModelSet.REAL, CODE_DIGEST) == []


def test_only_restricts_a_plan_to_its_entries_and_keeps_the_plan_s_order() -> None:
    selected = only(FAKE_PLAN, ["massive:split", "model_bench:single", "steady:single"])

    assert [(f"{spec.scenario_key}:{spec.topology}", spec.repeat) for spec in selected] == [
        ("model_bench:single", 1),
        ("steady:single", 1),
        ("massive:split", 1),
        ("model_bench:single", 2),
        ("steady:single", 2),
        ("massive:split", 2),
    ]


@pytest.mark.parametrize(
    "entry", ["stedy:single", "steady", "steady:both", "model_bench:split", "single:steady"]
)
def test_only_refuses_an_entry_that_no_run_of_the_plan_has(entry: str) -> None:
    with pytest.raises(ValueError, match=f"no run of the plan is {entry}"):
        only(FAKE_PLAN, ["steady:single", entry])


def test_remaining_leaves_out_the_finished_runs_and_keeps_the_plan_s_order() -> None:
    finished = {run_id(FAKE_PLAN[0]), run_id(FAKE_PLAN[3]), run_id(FAKE_PLAN[-1]), "another-run"}

    assert remaining(FAKE_PLAN, finished) == [*FAKE_PLAN[1:3], *FAKE_PLAN[4:-1]]


def test_remaining_is_the_whole_plan_before_any_run_and_empty_after_the_last() -> None:
    assert remaining(FAKE_PLAN, set()) == FAKE_PLAN
    assert remaining(FAKE_PLAN, {run_id(spec) for spec in FAKE_PLAN}) == []


def test_a_run_of_other_code_is_not_a_finished_run_of_the_plan() -> None:
    other_code = evidence_plan(2, ModelSet.FAKE, OTHER_CODE_DIGEST)

    assert remaining(FAKE_PLAN, {run_id(spec) for spec in other_code}) == FAKE_PLAN


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda candidate: candidate.name)
def test_a_comparison_runs_its_arms_in_the_order_a_b_b_a_with_repeats_numbered_per_arm(
    candidate: Candidate,
) -> None:
    plan = compare_plan(candidate, CODE_DIGEST)

    assert [(spec.arm, spec.repeat) for spec in plan] == [("a", 1), ("b", 1), ("b", 2), ("a", 2)]
    assert {spec.candidate for spec in plan} == {candidate.name}
    assert {(spec.scenario_key, spec.topology, spec.model_set) for spec in plan} == {
        (candidate.scenario_key, candidate.topology, candidate.model_set)
    }
    assert {(spec.phase_mode, spec.code_digest, spec.extra_compose) for spec in plan} == {
        ("random", CODE_DIGEST, None)
    }
    assert len({run_id(spec) for spec in plan}) == len(plan)
    assert [RunSpec.from_json(spec.to_json()) for spec in plan] == plan


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda candidate: candidate.name)
def test_each_arm_of_a_comparison_has_the_config_of_its_overrides(candidate: Candidate) -> None:
    a_first, b_first, b_second, a_second = compare_plan(candidate, CODE_DIGEST)
    a_config = resolved_config(candidate.topology, candidate.a_overrides)
    b_config = resolved_config(candidate.topology, candidate.b_overrides)

    assert (a_first.server_config, a_second.server_config) == (a_config, a_config)
    assert (b_first.server_config, b_second.server_config) == (b_config, b_config)
    assert a_config != b_config


@pytest.mark.parametrize(("name", "b_cap"), [("depth_cap_2", 2), ("depth_cap_4", 4)])
def test_arm_a_of_a_depth_cap_comparison_forces_the_default_cap_of_1(name: str, b_cap: int) -> None:
    caps = [
        (spec.arm, spec.server_config["models"]["depth"]["forced_batch_cap"])
        for spec in compare_plan(CANDIDATE_BY_NAME[name], CODE_DIGEST)
    ]

    assert caps == [("a", 1), ("b", b_cap), ("b", b_cap), ("a", 1)]


def test_the_same_arm_a_of_two_comparisons_gives_two_runs() -> None:
    cap_2 = compare_plan(CANDIDATE_BY_NAME["depth_cap_2"], CODE_DIGEST)[0]
    cap_4 = compare_plan(CANDIDATE_BY_NAME["depth_cap_4"], CODE_DIGEST)[0]

    assert cap_2.server_config == cap_4.server_config
    assert run_id(cap_2) != run_id(cap_4)


def test_a_comparison_run_is_not_the_evidence_run_of_the_same_scenario_and_config() -> None:
    arm_a = compare_plan(CANDIDATE_BY_NAME["fifo_vs_latest"], CODE_DIGEST)[0]
    steady_single = evidence_plan(1, ModelSet.REAL, CODE_DIGEST)[1]

    assert replace(arm_a, candidate=None, arm=None) == steady_single
    assert run_id(arm_a) != run_id(steady_single)


def test_the_robots_that_run_to_the_end_share_one_fleet_process_with_the_run_s_seed() -> None:
    assert fleet_plans(late_joiner(), 5_000_000_000, 3) == [
        FleetProcess(
            "fleet",
            FleetPlan(5_000_000_000, 30.0, (RobotPlan("a", 0.0, None), RobotPlan("b", 10.0, None))),
            3,
        )
    ]


def test_a_robot_that_the_scenario_kills_runs_alone_with_no_stop_time_and_its_own_seed() -> None:
    fleet, fleet_c = fleet_plans(disconnect(), 5_000_000_000, 2)

    assert fleet == FleetProcess(
        "fleet",
        FleetPlan(5_000_000_000, 40.0, (RobotPlan("a", 0.0, None), RobotPlan("b", 0.0, None))),
        2,
    )
    assert fleet_c == FleetProcess(
        "fleet_c",
        FleetPlan(5_000_000_000, 40.0, (RobotPlan("c", 0.0, None),)),
        int(hashlib.sha256(b"2:c").hexdigest()[:8], 16),
    )


def test_the_plans_of_killed_robots_have_no_stop_time_and_distinct_seeds() -> None:
    processes = fleet_plans(TWO_KILLS, 7_000_000_000, 1)

    assert [process.name for process in processes] == ["fleet", "fleet_c", "fleet_d"]
    assert [process.plan.robots for process in processes] == [
        (RobotPlan("a", 0.0, None), RobotPlan("b", 5.0, None)),
        (RobotPlan("c", 0.0, None),),
        (RobotPlan("d", 10.0, None),),
    ]
    assert len({process.seed for process in processes}) == len(processes)
    assert {(process.plan.t0_mono_ns, process.plan.duration_s) for process in processes} == {
        (7_000_000_000, 40.0)
    }


def test_the_seed_of_a_killed_robot_follows_the_run_s_seed() -> None:
    first = fleet_plans(disconnect(), 5_000_000_000, 1)[1]
    second = fleet_plans(disconnect(), 5_000_000_000, 2)[1]

    assert first.seed != second.seed
    assert fleet_plans(disconnect(), 9_000_000_000, 1)[1].seed == first.seed


def test_a_scenario_whose_robots_are_all_killed_has_no_shared_fleet_process() -> None:
    scenario = Scenario("kill", 10.0, (RobotPlan("a", 0.0, 5.0),), (Window("all", 0.0, 10.0),))

    assert [process.name for process in fleet_plans(scenario, 1_000_000_000, 1)] == ["fleet_a"]


@pytest.mark.parametrize("scenario", [steady(), late_joiner(), disconnect(), massive(), TWO_KILLS])
def test_the_fleet_reads_every_plan_file_as_it_is_written(
    scenario: Scenario, tmp_path: Path
) -> None:
    processes = fleet_plans(scenario, 5_000_000_000, 1)
    for process in processes:
        (tmp_path / f"{process.name}.json").write_text(process.plan.to_json(), encoding="utf-8")

    read = [asdict(read_plan(tmp_path / f"{process.name}.json")) for process in processes]

    assert read == [asdict(process.plan) for process in processes]
    assert json.loads(processes[0].plan.to_json())["t0_mono_ns"] == 5_000_000_000


def test_the_planned_events_of_the_late_joiner_hold_b_s_start_and_its_join() -> None:
    assert planned_events(late_joiner()) == (
        Event(EventName.ROBOT_START, 0.0, robot_id="a"),
        Event(EventName.ROBOT_START, 10.0, robot_id="b"),
        Event(EventName.ROBOT_JOIN, 10.0, robot_id="b"),
        Event(EventName.ROBOT_STOP, 30.0, robot_id="a"),
        Event(EventName.ROBOT_STOP, 30.0, robot_id="b"),
        Event(EventName.SCENARIO_END, 30.0),
    )


def test_the_planned_events_of_the_disconnect_hold_the_kill_of_c_and_no_stop_of_it() -> None:
    assert planned_events(disconnect()) == (
        Event(EventName.ROBOT_START, 0.0, robot_id="a"),
        Event(EventName.ROBOT_START, 0.0, robot_id="b"),
        Event(EventName.ROBOT_START, 0.0, robot_id="c"),
        Event(EventName.ROBOT_KILL, 20.0, robot_id="c"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="a"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="b"),
        Event(EventName.SCENARIO_END, 40.0),
    )


def test_the_planned_events_of_the_worker_kill_hold_no_fault() -> None:
    # The runner appends the fault, which the gateway's container times, after the driver.
    assert planned_events(worker_kill()) == (
        Event(EventName.ROBOT_START, 0.0, robot_id="a"),
        Event(EventName.ROBOT_START, 0.0, robot_id="b"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="a"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="b"),
        Event(EventName.SCENARIO_END, 40.0),
    )


def test_planned_events_are_in_time_order_whatever_the_order_of_the_robots() -> None:
    assert planned_events(TWO_KILLS) == (
        Event(EventName.ROBOT_START, 0.0, robot_id="a"),
        Event(EventName.ROBOT_START, 0.0, robot_id="c"),
        Event(EventName.ROBOT_START, 5.0, robot_id="b"),
        Event(EventName.ROBOT_JOIN, 5.0, robot_id="b"),
        Event(EventName.ROBOT_START, 10.0, robot_id="d"),
        Event(EventName.ROBOT_JOIN, 10.0, robot_id="d"),
        Event(EventName.ROBOT_KILL, 20.0, robot_id="c"),
        Event(EventName.ROBOT_KILL, 30.0, robot_id="d"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="a"),
        Event(EventName.ROBOT_STOP, 40.0, robot_id="b"),
        Event(EventName.SCENARIO_END, 40.0),
    )
