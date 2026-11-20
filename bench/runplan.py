"""The plan of the evidence: which runs there are, in which order, and what identifies each.

A run is identified by its run spec, and its id is derived from the spec, so a change to the
code or to any setting gives new ids and a finished run is recognized when the plan is run
again. Everything here is pure: the digest of the code is computed from the files' bytes,
which the caller reads, because a process inside the image has no git to list them with.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Any

from bench.runfiles import Event, EventName, FleetPlan
from bench.runfiles import ModelSet as ModelSet
from bench.runfiles import RunSpec as RunSpec
from bench.scenarios import (
    RobotPlan,
    Scenario,
    disconnect,
    late_joiner,
    massive,
    soak,
    steady,
    sweep,
    worker_kill,
)
from cloud.config import OVERRIDES_VARIABLE, TOPOLOGY_VARIABLE, Topology, config_echo, load_config
from common.protocol import PROTOCOL_VERSION
from robot.settings import LinkSettings

if TYPE_CHECKING:
    # For the annotation only: the scenario driver imports this module inside the image, and
    # what it imports is in the code digest, which the candidates are not. A candidate's
    # settings reach a run's identity through the resolved config of its spec.
    from bench.candidates import Candidate

MODEL_BENCH_KEY = "model_bench"
# The robot counts of the scaling sweep (DESIGN.md 9.3), each a scenario key of its own.
SWEEP_ROBOT_COUNTS = (1, 3, 5, 10)

_BEFORE_SWEEP: Mapping[str, Callable[[], Scenario]] = {"steady": steady, "late_joiner": late_joiner}
_SWEEP: Mapping[str, Callable[[], Scenario]] = {
    f"sweep_n{robot_count}": partial(sweep, robot_count) for robot_count in SWEEP_ROBOT_COUNTS
}
_AFTER_SWEEP: Mapping[str, Callable[[], Scenario]] = {
    "disconnect": disconnect,
    "worker_kill": worker_kill,
    "massive": massive,
    "soak": soak,
}
_SCENARIOS: Mapping[str, Callable[[], Scenario]] = {**_BEFORE_SWEEP, **_SWEEP, **_AFTER_SWEEP}

# The phase mode of every planned run: each robot's phase is drawn from the run's seed.
_RANDOM_PHASES = "random"

# What a model set changes in each model's settings. The fake service times put the pair in
# the two regimes the design expects of the real models (DESIGN.md 6.3): 2 ms per frame
# leaves the classifier unsaturated at a few robots, and 40 ms per frame is 25 frames per
# second, below the 30 fps of one robot, so depth is saturated from the first. Fake depth
# calibrates batch size 1 alone, so its batch cap is 1, as the published depth graph's is
# (DESIGN.md 4.6).
_MODEL_SET_SETTINGS: Mapping[ModelSet, Mapping[str, Mapping[str, Any]]] = {
    ModelSet.REAL: {},
    ModelSet.FAKE: {
        "cls": {"runtime": "fake", "fake_frame_ms": 2.0},
        "depth": {"runtime": "fake", "fake_frame_ms": 40.0, "batch_sizes": [1]},
    },
    ModelSet.NULL: {"cls": {"runtime": "null"}, "depth": {"runtime": "null"}},
}

_CODE_DIRECTORIES = ("cloud/", "common/", "robot/", "workers/", "scripts/")
# The scenario driver, the fault injector, the benchmarks and what they import from bench/ (the
# package's own __init__ runs on every such import), then what builds and starts the image.
# pyproject.toml is not among them: the image installs requirements.txt, and the tool settings
# it holds reach the tests gate through the image's digest.
_CODE_FILES = frozenset(
    (
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
        "Dockerfile",
        "docker-compose.yml",
        "docker-compose.split.yml",
        "requirements.txt",
    )
)

# A comparison's runs, as (arm, repeat): A, B, B, A, so that a drift over the four runs
# cancels between the arms. Repeats are numbered per arm.
_COMPARISON_ORDER = (("a", 1), ("b", 1), ("b", 2), ("a", 2))


def scenario_for(key: str) -> Scenario:
    """Return the timeline of a serving scenario key; ValueError for any other key."""
    if key not in _SCENARIOS:
        raise ValueError(f"{key} is not a serving scenario; they are {', '.join(_SCENARIOS)}")
    return _SCENARIOS[key]()


def overrides_for(model_set: ModelSet, overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Return a run's RRI_OVERRIDES object: the model set's settings under the given overrides.

    A model's settings are merged field by field, and a given field wins.
    """
    set_models = _MODEL_SET_SETTINGS[model_set]
    given_models: Mapping[str, Mapping[str, Any]] = overrides.get("models", {})
    merged = {name: value for name, value in overrides.items() if name != "models"}
    models = {
        model: {**set_models.get(model, {}), **given_models.get(model, {})}
        for model in dict.fromkeys((*set_models, *given_models))
    }
    if models:
        merged["models"] = models
    return merged


def resolved_config(topology: Topology, overrides: Mapping[str, Any]) -> dict[str, object]:
    """Return the config echo of a gateway started with these overrides, without its version.

    The config is built as the gateway builds it, from the environment that the compose file
    of the topology gives it, so a setting written as 2 and as 2.0 resolves to one config.
    ConfigError for overrides that the gateway would refuse.
    """
    environment = {TOPOLOGY_VARIABLE: topology.value, OVERRIDES_VARIABLE: json.dumps(overrides)}
    return config_echo(load_config(environment), version="", protocol_version=PROTOCOL_VERSION)


def _whole_as_integer(text: str) -> int | float:
    number = float(text)
    return int(number) if number.is_integer() else number


def spec_digest(spec: RunSpec) -> str:
    """Return the 12 hex digits that identify a run spec apart from its repeat and its code."""
    # Specs that compare equal must give one digest, and 2 equals 2.0, so a whole number is
    # written in one form whichever way the spec holds it.
    document = json.loads(spec.to_json(), parse_float=_whole_as_integer)
    del document["repeat"], document["code_digest"]
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def run_id(spec: RunSpec) -> str:
    """Return the name of a run's directory. Repeats of a spec differ in the last part alone."""
    return f"{spec.topology}-{spec_digest(spec)}-{spec.code_digest[:8]}-r{spec.repeat}"


def code_paths(listed: Iterable[str]) -> list[str]:
    """Return, sorted, the files that decide what a run does, out of the repository's files.

    The paths are relative to the repository root, with "/" between their parts, as git
    lists them.
    """
    return sorted(
        path for path in set(listed) if path.startswith(_CODE_DIRECTORIES) or path in _CODE_FILES
    )


def code_digest(files: Mapping[str, bytes]) -> str:
    """Return the SHA-256 over the paths and the contents of the code files, in path order."""
    digest = hashlib.sha256()
    for path, content in sorted(files.items()):
        # Each part is preceded by its length, so no two sets of files give the same bytes.
        for part in (path.encode(), content):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
    return digest.hexdigest()


def _spec(
    scenario_key: str,
    topology: Topology,
    model_set: ModelSet,
    overrides: Mapping[str, Any],
    code_digest: str,
    repeat: int,
) -> RunSpec:
    """Return the spec of a serving run of the evidence: random phases, the default link model."""
    return RunSpec(
        scenario_key=scenario_key,
        topology=topology,
        phase_mode=_RANDOM_PHASES,
        model_set=model_set,
        server_config=resolved_config(topology, overrides_for(model_set, overrides)),
        uplink=LinkSettings(),
        downlink=LinkSettings(),
        extra_compose=None,
        candidate=None,
        arm=None,
        code_digest=code_digest,
        repeat=repeat,
    )


def evidence_plan(repeats: int, model_set: ModelSet, code_digest: str) -> list[RunSpec]:
    """Return the evidence runs of every repeat, in the order they run (DESIGN.md 11.2).

    A repeat starts with the model benchmark, always on the real models: it measures the
    model files, and its tables feed the predictions, which no demo run may feed. Then
    every serving scenario runs in both topologies. Even repeats run the sweep's counts
    and the two topologies in the opposite order, so two repeats give the order A B B A
    (DESIGN.md 9.6).
    """
    plan: list[RunSpec] = []
    for repeat in range(1, repeats + 1):
        odd = repeat % 2 == 1
        sweep_keys = tuple(_SWEEP) if odd else tuple(_SWEEP)[::-1]
        topologies = (Topology.SINGLE, Topology.SPLIT) if odd else (Topology.SPLIT, Topology.SINGLE)
        model_bench = _spec(
            MODEL_BENCH_KEY, Topology.SINGLE, ModelSet.REAL, {}, code_digest, repeat
        )
        # The model benchmark runs no robot, so it has no phase mode.
        plan.append(replace(model_bench, phase_mode=None))
        plan.extend(
            _spec(scenario_key, topology, model_set, {}, code_digest, repeat)
            for scenario_key in (*_BEFORE_SWEEP, *sweep_keys, *_AFTER_SWEEP)
            for topology in topologies
        )
    return plan


def compare_plan(candidate: Candidate, code_digest: str) -> list[RunSpec]:
    """Return a candidate's four runs in the order A, B, B, A."""
    arm_overrides = {"a": candidate.a_overrides, "b": candidate.b_overrides}
    return [
        replace(
            _spec(
                candidate.scenario_key,
                candidate.topology,
                candidate.model_set,
                arm_overrides[arm],
                code_digest,
                repeat,
            ),
            candidate=candidate.name,
            arm=arm,
        )
        for arm, repeat in _COMPARISON_ORDER
    ]


def only(plan: Sequence[RunSpec], entries: Collection[str]) -> list[RunSpec]:
    """Restrict a plan to the entries, each KEY:TOPOLOGY, keeping the plan's order.

    ValueError for an entry that no run of the plan has.
    """
    in_plan = {f"{spec.scenario_key}:{spec.topology}" for spec in plan}
    unknown = sorted(set(entries) - in_plan)
    if unknown:
        raise ValueError(f"no run of the plan is {', '.join(unknown)}")
    return [spec for spec in plan if f"{spec.scenario_key}:{spec.topology}" in entries]


def remaining(plan: Sequence[RunSpec], finished_ids: Collection[str]) -> list[RunSpec]:
    """Return the runs of a plan that are not finished, in the plan's order."""
    return [spec for spec in plan if run_id(spec) not in finished_ids]


@dataclass(frozen=True)
class FleetProcess:
    """One fleet process of a serving run.

    The name is that of its plan file in plans/ and of its entry in a scrape's processes.
    """

    name: str
    plan: FleetPlan
    seed: int


def fleet_plans(scenario: Scenario, t0_mono_ns: int, seed: int) -> list[FleetProcess]:
    """Return the fleet processes of a scenario: their plans, all with one time zero.

    The robots that run to the end share the process "fleet", which takes the run's seed.
    """
    processes: list[FleetProcess] = []
    to_the_end = tuple(robot for robot in scenario.robots if robot.stop_s is None)
    if to_the_end:
        plan = FleetPlan(t0_mono_ns, scenario.duration_s, to_the_end)
        processes.append(FleetProcess("fleet", plan, seed))
    for robot in scenario.robots:
        if robot.stop_s is not None:
            # ASSUMPTION: A11. A stop time is a SIGKILL of the robot's own process, so the
            # robot runs alone, and its plan has no stop time: only the kill ends it. Its
            # process has a seed of its own, so its phase is a draw independent of the others'.
            alone = (RobotPlan(robot.robot_id, robot.start_s, None),)
            own_seed = int(hashlib.sha256(f"{seed}:{robot.robot_id}".encode()).hexdigest()[:8], 16)
            processes.append(
                FleetProcess(
                    f"fleet_{robot.robot_id}",
                    FleetPlan(t0_mono_ns, scenario.duration_s, alone),
                    own_seed,
                )
            )
    return processes


def planned_events(scenario: Scenario) -> tuple[Event, ...]:
    """Return the events a scenario's timeline fixes in advance, in time order.

    Every robot has a robot_start at its planned start, and one that starts after time zero
    a robot_join there as well. It then has a robot_kill at its stop time, or a robot_stop
    at the end of the run. The scenario_end is last.
    """
    events: list[Event] = []
    for robot in scenario.robots:
        events.append(Event(EventName.ROBOT_START, robot.start_s, robot_id=robot.robot_id))
        if robot.start_s > 0:
            events.append(Event(EventName.ROBOT_JOIN, robot.start_s, robot_id=robot.robot_id))
        if robot.stop_s is None:
            events.append(Event(EventName.ROBOT_STOP, scenario.duration_s, robot_id=robot.robot_id))
        else:
            events.append(Event(EventName.ROBOT_KILL, robot.stop_s, robot_id=robot.robot_id))
    events.append(Event(EventName.SCENARIO_END, scenario.duration_s))
    # The sort is stable, so events at one time keep the order above.
    return tuple(sorted(events, key=lambda event: event.t_s))
