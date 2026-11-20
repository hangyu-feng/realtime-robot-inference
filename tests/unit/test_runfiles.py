"""Tests for the records of a run's files: what each writes, reads back and refuses."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Protocol, Self

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.environment import collect_environment
from bench.runfiles import (
    DONE_FILE,
    SAMPLES_DIRECTORY,
    SUMMARY_FILE,
    ConfigGate,
    Done,
    Event,
    EventName,
    Fact,
    Facts,
    FleetConfig,
    FleetPlan,
    FreshGate,
    Gate,
    GatedLimits,
    ImageGate,
    InsideEnvironment,
    LimitsGate,
    ModelSet,
    RunConfig,
    RunEnvironment,
    RunFileError,
    RunSpec,
    Scrape,
    SuiteGate,
    done_files,
)
from bench.scenarios import RobotPlan, disconnect, worker_kill
from cloud.config import ServerConfig, Topology, config_echo
from common.cgroup import CgroupStats, Limits
from common.commands import MODEL_KEYS
from common.procstat import ProcessStats
from common.protocol import PROTOCOL_VERSION
from common.thresholds import DEFAULT_STALE_MS, DEFAULT_WATCHDOG_MS
from robot.fleet import read_plan
from robot.frames import FrameSetStats
from robot.settings import LinkSettings


class Record(Protocol):
    def to_json(self) -> str: ...

    @classmethod
    def from_json(cls, text: str, /) -> Self: ...


CODE_DIGEST = "c0de" * 16
IMAGE_DIGEST = "sha256:" + "1a" * 32
RUN_ID = "single-0123456789ab-c0dec0de-r2"

SPEC = RunSpec(
    scenario_key="disconnect",
    topology=Topology.SINGLE,
    phase_mode="random",
    model_set=ModelSet.FAKE,
    server_config=config_echo(ServerConfig(), version="", protocol_version=PROTOCOL_VERSION),
    uplink=LinkSettings(),
    downlink=LinkSettings(delay_ms=20.0, jitter_ms=5.0),
    extra_compose=None,
    candidate=None,
    arm=None,
    code_digest=CODE_DIGEST,
    repeat=2,
)
COMPARISON_SPEC = replace(SPEC, extra_compose="e" * 64, candidate="depth_cap_2", arm="b")
MODEL_BENCH_SPEC = replace(
    SPEC, scenario_key="model_bench", phase_mode=None, model_set=ModelSet.REAL
)

FLEET = FleetConfig(
    scenario=disconnect(),
    frame_rate_hz=30.0,
    models=MODEL_KEYS,
    stale_ms=DEFAULT_STALE_MS,
    watchdog_ms=DEFAULT_WATCHDOG_MS,
    planned_events=(
        Event(EventName.ROBOT_START, 0.0, robot_id="a"),
        Event(EventName.ROBOT_START, 0.0, robot_id="b"),
        Event(EventName.ROBOT_START, 0.0, robot_id="c"),
        Event(EventName.ROBOT_KILL, 20.0, robot_id="c"),
        Event(EventName.SCENARIO_END, 40.0),
    ),
)
CONFIG = RunConfig(spec=SPEC, run_id=RUN_ID, seed=2, extra_compose_path=None, fleet=FLEET)
COMPARISON_CONFIG = replace(CONFIG, spec=COMPARISON_SPEC, extra_compose_path="one_cpu.yml")
MODEL_BENCH_CONFIG = replace(CONFIG, spec=MODEL_BENCH_SPEC, fleet=None)
HAND_RUN_CONFIG = RunConfig(spec=None, run_id=None, seed=0, extra_compose_path=None, fleet=FLEET)
WORKER_KILL_CONFIG = replace(
    CONFIG,
    spec=replace(SPEC, scenario_key="worker_kill"),
    fleet=replace(FLEET, scenario=worker_kill()),
)

PLAN = FleetPlan(
    t0_mono_ns=5_000_000_000,
    duration_s=40.0,
    robots=(RobotPlan("a", 0.0, None), RobotPlan("b", 10.0, None), RobotPlan("c", 0.0, 20.0)),
)

EVENTS = [
    Event(EventName.T0, 0.0, t0_mono_ns=5_000_000_000),
    Event(EventName.SERVER_READY, -7.25),
    Event(EventName.ROBOT_START, 0.0, robot_id="a"),
    Event(EventName.ROBOT_JOIN, 10.0, robot_id="b"),
    Event(EventName.ROBOT_KILL, 20.000125, robot_id="c"),
    Event(EventName.ROBOT_STOP, 40.0, robot_id="a"),
    Event(EventName.SCENARIO_END, 40.0),
    Event(EventName.FINAL_POLL, 40.5, settled=True, waited_s=0.3),
]
FAULT_EVENT = Event(EventName.FAULT, 20.000375, model="depth", pid=4242)

PROCESS = ProcessStats(
    pid=41, cpu_user_ms=1200, cpu_system_ms=300, rss_bytes=64 << 20, threads=3, fds=9, nice=0
)
ROBOTS_CGROUP = CgroupStats(
    cpu_usage_us=1_500_000,
    cpu_user_us=1_200_000,
    cpu_system_us=300_000,
    nr_periods=None,
    nr_throttled=None,
    throttled_us=None,
    memory_current_bytes=96 << 20,
    memory_peak_bytes=None,
    memory_anon_bytes=80 << 20,
    memory_file_bytes=8 << 20,
    memory_sock_bytes=1 << 20,
    oom_kills=0,
)
METRICS = {
    "schema": 1,
    "ts_mono_ns": 6_000_000_123,
    "uptime_s": 8.25,
    "connections": {"current": 3, "accepted": 3, "closed": {"abnormal": 0}},
    "workers": {"depth": {"pacing_share": None, "batch_sizes": [1, 2, 4]}},
}
SCRAPE = Scrape(
    k=1,
    driver_before_ns=6_000_000_000,
    driver_after_ns=6_000_900_000,
    metrics=METRICS,
    error=None,
    fleet_processes={"fleet": PROCESS, "fleet_c": replace(PROCESS, pid=42), "driver": PROCESS},
    fleet_cgroup=None,
)
SPLIT_SCRAPE = replace(SCRAPE, fleet_cgroup=ROBOTS_CGROUP)
FINAL_SCRAPE = replace(SCRAPE, k=None, fleet_processes={"driver": PROCESS})
FAILED_SCRAPE = replace(SCRAPE, metrics=None, error="timed out after 1 s")

LIMITS = Limits(
    version=2,
    cpu_limit_cores=2.0,
    cpu_period_us=100_000,
    memory_limit_bytes=4 << 30,
    memory_swap_limit_bytes=0,
    cpuset="0-23",
)
INSIDE = InsideEnvironment(
    cpu_model="Example(R) Processor 9000 @ 2.00GHz",
    cpu_count_visible=24,
    machine="x86_64",
    kernel="6.6.0-example",
    cgroup=LIMITS,
    python="3.14.0",
    packages={"numpy": "2.5.5", "uvloop": None},
    model_sha256={"cls.onnx": "ab" * 32},
    start_utc="2026-01-02T03:04:05+00:00",
    frames=FrameSetStats(
        count=32, jpeg_quality=80, min_bytes=18_000, mean_bytes=19_500.5, max_bytes=21_000
    ),
)
ENVIRONMENT = RunEnvironment(
    inside=INSIDE,
    host_memory_total_bytes=64 << 30,
    engine="docker",
    engine_version="29.0.0",
    commit="f" * 40,
    dirty=False,
    code_digest=CODE_DIGEST,
    image_digest=IMAGE_DIGEST,
    robots_cgroup=None,
)
SPLIT_ENVIRONMENT = replace(
    ENVIRONMENT, robots_cgroup=replace(LIMITS, cpu_limit_cores=None, memory_limit_bytes=None)
)

EXPECTED_LIMITS = GatedLimits(
    cpu_limit_cores=2.0, memory_limit_bytes=4 << 30, memory_swap_limit_bytes=0
)
SUITE_GATE = SuiteGate(
    passed=True,
    image_digest=IMAGE_DIGEST,
    code_digest=CODE_DIGEST,
    commit="f" * 40,
    dirty=False,
    passed_count=640,
    failed_count=0,
    skipped_count=0,
    deselected_count=4,
    gate_file="gates/1a1a1a1a1a1a-c0dec0de.json",
    output_sha256="0b" * 32,
)
GATE = Gate(
    tests_passed=SUITE_GATE,
    image_built=ImageGate(built=True, digest=IMAGE_DIGEST, containers={"rri-system": IMAGE_DIGEST}),
    limits_ok=LimitsGate(ok=True, expected=EXPECTED_LIMITS, read=EXPECTED_LIMITS),
    fresh=FreshGate(ok=True, new_containers=True, accepted_before_t0=0),
    config_ok=ConfigGate(ok=True, differences=()),
)
MODEL_BENCH_GATE = replace(
    GATE,
    fresh=FreshGate(ok=True, new_containers=True, accepted_before_t0=None),
    config_ok=None,
)
FAILED_GATES = [
    replace(GATE, tests_passed=replace(SUITE_GATE, passed=False, failed_count=1)),
    replace(GATE, image_built=replace(GATE.image_built, built=False)),
    replace(
        GATE,
        limits_ok=LimitsGate(
            ok=False,
            expected=EXPECTED_LIMITS,
            read=GatedLimits(
                cpu_limit_cores=1.0, memory_limit_bytes=4 << 30, memory_swap_limit_bytes=None
            ),
        ),
    ),
    replace(GATE, fresh=FreshGate(ok=False, new_containers=True, accepted_before_t0=1)),
    replace(GATE, config_ok=ConfigGate(ok=False, differences=("models.depth.forced_batch_cap",))),
]

RUN_FILES = {
    "config.json": CONFIG.to_json().encode(),
    "gate.json": GATE.to_json().encode(),
    "plans/fleet.json": PLAN.to_json().encode(),
    "events.jsonl": b"".join(event.to_json().encode() + b"\n" for event in EVENTS),
}
DONE = Done(files=done_files(RUN_FILES), summary_sha256=None)
SUMMARIZED_DONE = replace(DONE, summary_sha256="5e" * 32)

FACT_KEY = "steady.cls.commands_per_s"
FACT = Fact(
    value=29.5,
    unit="1/s",
    runs=3,
    min=29.25,
    max=30.0,
    run_ids=("single-0123456789ab-c0dec0de-r1", RUN_ID, "single-0123456789ab-c0dec0de-r3"),
    dirty_runs=1,
    failed_checks={RUN_ID: ("simulator_valid",)},
    attempts=4,
    failed_attempts=1,
)
EMPTY_FACT = Fact(
    value=None,
    unit="ms",
    runs=0,
    min=None,
    max=None,
    run_ids=(),
    dirty_runs=0,
    failed_checks={},
    attempts=0,
    failed_attempts=0,
)
FACTS = Facts(
    model_set=ModelSet.FAKE,
    code_digest=CODE_DIGEST,
    counted_run_ids=FACT.run_ids,
    facts={FACT_KEY: FACT, "steady.depth.freshness_ms.p95": EMPTY_FACT},
)

RECORDS: list[Record] = [
    SPEC,
    COMPARISON_SPEC,
    MODEL_BENCH_SPEC,
    CONFIG,
    COMPARISON_CONFIG,
    MODEL_BENCH_CONFIG,
    HAND_RUN_CONFIG,
    WORKER_KILL_CONFIG,
    PLAN,
    *EVENTS,
    FAULT_EVENT,
    SCRAPE,
    SPLIT_SCRAPE,
    FINAL_SCRAPE,
    FAILED_SCRAPE,
    INSIDE,
    ENVIRONMENT,
    SPLIT_ENVIRONMENT,
    SUITE_GATE,
    GATE,
    MODEL_BENCH_GATE,
    *FAILED_GATES,
    DONE,
    SUMMARIZED_DONE,
    FACTS,
]


def _record_name(record: Record) -> str:
    return type(record).__name__


def _edited(record: Record, path: tuple[str | int, ...], value: object) -> str:
    """Return the record's JSON with the value at the path replaced, or added."""
    document: Any = json.loads(record.to_json())
    holder = document
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return json.dumps(document)


def _without(record: Record, path: tuple[str | int, ...]) -> str:
    """Return the record's JSON without the key at the path."""
    document: Any = json.loads(record.to_json())
    holder = document
    for key in path[:-1]:
        holder = holder[key]
    del holder[path[-1]]
    return json.dumps(document)


@pytest.mark.parametrize("record", RECORDS, ids=_record_name)
def test_a_record_read_back_from_its_json_is_the_same_record(record: Record) -> None:
    assert type(record).from_json(record.to_json()) == record


@pytest.mark.parametrize("record", RECORDS, ids=_record_name)
def test_a_record_read_back_writes_the_same_text_again(record: Record) -> None:
    assert type(record).from_json(record.to_json()).to_json() == record.to_json()


@pytest.mark.parametrize(
    ("record", "path", "value", "message"),
    [
        (CONFIG, ("seed",), 1.5, "config.json: seed must be an integer"),
        (CONFIG, ("spec", "repeat"), True, "config.json: spec.repeat must be an integer"),
        (
            CONFIG,
            ("spec", "topology"),
            "both",
            "config.json: spec.topology must be one of single, split",
        ),
        (
            CONFIG,
            ("spec", "model_set"),
            "onnx",
            "config.json: spec.model_set must be one of real, fake, null",
        ),
        (
            CONFIG,
            ("spec", "phase_mode"),
            "staggered",
            "config.json: spec.phase_mode must be one of random, aligned",
        ),
        (COMPARISON_CONFIG, ("spec", "arm"), "c", "config.json: spec.arm must be one of a, b"),
        (
            CONFIG,
            ("spec", "arm"),
            "a",
            "config.json: spec.arm must be null exactly when candidate is",
        ),
        (
            CONFIG,
            ("spec", "server_config"),
            [],
            "config.json: spec.server_config must be an object",
        ),
        (
            CONFIG,
            ("spec", "uplink", "delay_ms"),
            "0",
            "config.json: spec.uplink.delay_ms must be a",
        ),
        (CONFIG, ("spec", "downlink", "jitter_ms"), -1.0, "config.json: spec.downlink: jitter_ms"),
        (
            CONFIG,
            ("hand_run",),
            True,
            "config.json: hand_run must be true exactly when spec is null",
        ),
        (CONFIG, ("run_id",), None, "config.json: run_id must be null exactly when spec is"),
        (HAND_RUN_CONFIG, ("run_id",), RUN_ID, "config.json: run_id must be null exactly when"),
        (CONFIG, ("fleet", "models"), "cls", "config.json: fleet.models must be a list"),
        (CONFIG, ("fleet", "stale_ms", "cls"), None, "config.json: fleet.stale_ms.cls must be a"),
        (
            CONFIG,
            ("fleet", "scenario", "robots", 2, "stop_s"),
            "20",
            "config.json: fleet.scenario.robots[2].stop_s must be a number",
        ),
        (
            CONFIG,
            ("fleet", "scenario", "windows", 0, "name"),
            0,
            "config.json: fleet.scenario.windows[0].name must be a string",
        ),
        (
            CONFIG,
            ("fleet", "planned_events", 3, "robot_id"),
            3,
            "config.json: fleet.planned_events[3].robot_id must be a string",
        ),
        (
            CONFIG,
            ("fleet", "planned_events", 4, "event"),
            "robot_crash",
            "config.json: fleet.planned_events[4].event must be one of t0, server_ready",
        ),
        (
            WORKER_KILL_CONFIG,
            ("fleet", "scenario", "fault", "at_s"),
            "20",
            "config.json: fleet.scenario.fault.at_s must be a number",
        ),
        (
            WORKER_KILL_CONFIG,
            ("fleet", "scenario", "fault"),
            "depth",
            "config.json: fleet.scenario.fault must be an object",
        ),
        (FAULT_EVENT, ("pid",), "4242", "a line of events.jsonl: pid must be an integer"),
        (FAULT_EVENT, ("model",), None, "a line of events.jsonl: model must be a string"),
        (PLAN, ("t0_mono_ns",), None, "a plan of plans/: t0_mono_ns must be an integer"),
        (
            PLAN,
            ("robots", 0, "robot_id"),
            7,
            "a plan of plans/: robots[0].robot_id must be a string",
        ),
        (EVENTS[0], ("t0_mono_ns",), 5.0, "a line of events.jsonl: t0_mono_ns must be an integer"),
        (EVENTS[4], ("t_s",), None, "a line of events.jsonl: t_s must be a number"),
        (EVENTS[6], ("t_s",), 10**400, "a line of events.jsonl: t_s must be within the range of"),
        (EVENTS[7], ("settled",), 1, "a line of events.jsonl: settled must be true or false"),
        (SCRAPE, ("k",), "1", "a line of scrapes.jsonl.gz: k must be an integer"),
        (SCRAPE, ("metrics",), [], "a line of scrapes.jsonl.gz: metrics must be an object"),
        (FAILED_SCRAPE, ("error",), None, "a line of scrapes.jsonl.gz: error must be a string"),
        (
            SCRAPE,
            ("fleet", "processes", "driver", "fds"),
            None,
            "a line of scrapes.jsonl.gz: fleet.processes.driver.fds must be an integer",
        ),
        (
            SPLIT_SCRAPE,
            ("fleet", "cgroup", "cpu_usage_us"),
            "1",
            "a line of scrapes.jsonl.gz: fleet.cgroup.cpu_usage_us must be an integer",
        ),
        (INSIDE, ("cgroup", "version"), None, "env_inside.json: cgroup.version must be an integer"),
        (INSIDE, ("frames", "mean_bytes"), "big", "env_inside.json: frames.mean_bytes must be a"),
        (INSIDE, ("packages", "numpy"), 2, "env_inside.json: packages.numpy must be a string"),
        (ENVIRONMENT, ("dirty",), 0, "env.json: dirty must be true or false"),
        (ENVIRONMENT, ("kernel",), None, "env.json: kernel must be a string"),
        (
            SPLIT_ENVIRONMENT,
            ("robots_cgroup", "cpuset"),
            3,
            "env.json: robots_cgroup.cpuset must be a string",
        ),
        (SUITE_GATE, ("dirty",), "no", "a tests gate: dirty must be true or false"),
        (
            GATE,
            ("tests_passed", "passed_count"),
            "640",
            "gate.json: tests_passed.passed_count must be an integer",
        ),
        (
            GATE,
            ("image_built", "containers", "rri-system"),
            7,
            "gate.json: image_built.containers.rri-system must be a string",
        ),
        (
            GATE,
            ("limits_ok", "read", "memory_limit_bytes"),
            4.5,
            "gate.json: limits_ok.read.memory_limit_bytes must be an integer",
        ),
        (
            GATE,
            ("fresh", "accepted_before_t0"),
            False,
            "gate.json: fresh.accepted_before_t0 must be an integer",
        ),
        (
            GATE,
            ("config_ok", "differences"),
            "topology",
            "gate.json: config_ok.differences must be a list",
        ),
        (GATE, ("gated",), False, "gate.json: gated must be true exactly when every gate"),
        (
            FAILED_GATES[3],
            ("gated",),
            True,
            "gate.json: gated must be true exactly when every gate",
        ),
        (DONE, ("files", "config.json"), None, "DONE: files.config.json must be a string"),
        (DONE, ("summary_sha256",), 5, "DONE: summary_sha256 must be a string"),
        (FACTS, ("model_set",), "onnx", "facts.json: model_set must be one of real, fake, null"),
        (FACTS, ("counted_run_ids", 1), 2, "facts.json: counted_run_ids[1] must be a string"),
        (FACTS, ("facts", FACT_KEY, "value"), [1], f"facts.json: facts.{FACT_KEY}.value must be a"),
        (FACTS, ("facts", FACT_KEY, "min"), "low", f"facts.json: facts.{FACT_KEY}.min must be a"),
        (FACTS, ("facts", FACT_KEY, "max"), True, f"facts.json: facts.{FACT_KEY}.max must be a"),
        (FACTS, ("facts", FACT_KEY, "runs"), 3.0, f"facts.json: facts.{FACT_KEY}.runs must be an"),
        (
            FACTS,
            ("facts", FACT_KEY, "failed_checks", RUN_ID, 0),
            None,
            f"facts.json: facts.{FACT_KEY}.failed_checks.{RUN_ID}[0] must be a string",
        ),
    ],
)
def test_a_malformed_field_is_refused_by_name(
    record: Record, path: tuple[str | int, ...], value: object, message: str
) -> None:
    with pytest.raises(RunFileError) as refusal:
        type(record).from_json(_edited(record, path, value))

    assert message in str(refusal.value)


@pytest.mark.parametrize(
    ("record", "path", "message"),
    [
        (SPEC, ("code_digest",), "a run spec: code_digest is missing"),
        (CONFIG, ("hand_run",), "config.json: hand_run is missing"),
        (CONFIG, ("spec", "extra_compose"), "config.json: spec.extra_compose is missing"),
        (CONFIG, ("fleet", "scenario", "duration_s"), "config.json: fleet.scenario.duration_s is"),
        (CONFIG, ("fleet", "scenario", "fault"), "config.json: fleet.scenario.fault is missing"),
        (
            WORKER_KILL_CONFIG,
            ("fleet", "scenario", "fault", "model"),
            "config.json: fleet.scenario.fault.model is missing",
        ),
        (FAULT_EVENT, ("pid",), "a line of events.jsonl: pid is missing"),
        (FAULT_EVENT, ("model",), "a line of events.jsonl: model is missing"),
        (PLAN, ("robots", 2, "stop_s"), "a plan of plans/: robots[2].stop_s is missing"),
        (EVENTS[4], ("robot_id",), "a line of events.jsonl: robot_id is missing"),
        (EVENTS[0], ("event",), "a line of events.jsonl: event is missing"),
        (SCRAPE, ("metrics",), "a line of scrapes.jsonl.gz: metrics is missing"),
        (FAILED_SCRAPE, ("error",), "a line of scrapes.jsonl.gz: error is missing"),
        (SCRAPE, ("fleet", "processes"), "a line of scrapes.jsonl.gz: fleet.processes is missing"),
        (INSIDE, ("frames",), "env_inside.json: frames is missing"),
        (ENVIRONMENT, ("cgroup", "cpuset"), "env.json: cgroup.cpuset is missing"),
        (ENVIRONMENT, ("robots_cgroup",), "env.json: robots_cgroup is missing"),
        (GATE, ("config_ok",), "gate.json: config_ok is missing"),
        (GATE, ("limits_ok", "expected", "cpu_limit_cores"), "gate.json: limits_ok.expected.cpu_"),
        (DONE, ("summary_sha256",), "DONE: summary_sha256 is missing"),
        (FACTS, ("facts", FACT_KEY, "dirty_runs"), f"facts.json: facts.{FACT_KEY}.dirty_runs is"),
    ],
)
def test_a_missing_field_is_refused_by_name(
    record: Record, path: tuple[str | int, ...], message: str
) -> None:
    with pytest.raises(RunFileError) as refusal:
        type(record).from_json(_without(record, path))

    assert message in str(refusal.value)


@pytest.mark.parametrize(
    ("record", "path", "message"),
    [
        (CONFIG, ("scenario",), "config.json: scenario is not a field of it"),
        (CONFIG, ("spec", "seed"), "config.json: spec.seed is not a field of it"),
        (PLAN, ("robots", 0, "phase"), "a plan of plans/: robots[0].phase is not a field of it"),
        (EVENTS[0], ("robot_id",), "a line of events.jsonl: robot_id is not a field of it"),
        (EVENTS[6], ("settled",), "a line of events.jsonl: settled is not a field of it"),
        (SCRAPE, ("error",), "a line of scrapes.jsonl.gz: error is not a field of it"),
        (FAILED_SCRAPE, ("metrics",), "a line of scrapes.jsonl.gz: metrics is not a field of it"),
        (SCRAPE, ("fleet", "limits"), "a line of scrapes.jsonl.gz: fleet.limits is not a field"),
        (ENVIRONMENT, ("inside",), "env.json: inside is not a field of it"),
        (GATE, ("tests_passed", "errors_count"), "gate.json: tests_passed.errors_count is not a"),
        (DONE, ("samples",), "DONE: samples is not a field of it"),
        (FACTS, ("facts", FACT_KEY, "median"), f"facts.json: facts.{FACT_KEY}.median is not a"),
    ],
)
def test_an_unknown_field_is_refused_by_name(
    record: Record, path: tuple[str | int, ...], message: str
) -> None:
    with pytest.raises(RunFileError) as refusal:
        type(record).from_json(_edited(record, path, None))

    assert message in str(refusal.value)


@pytest.mark.parametrize(
    ("record_type", "text", "message"),
    [
        (RunConfig, "", "config.json is not JSON"),
        (Gate, '{"gated": tru', "gate.json is not JSON"),
        (Done, '{"files": {}, "summary_sha256": NaN}', "DONE is not JSON"),
        (Facts, '{"facts": {"a.b": {"value": Infinity}}}', "facts.json is not JSON"),
        (Event, '{"event": "scenario_end", "t_s": 1e999}', "a line of events.jsonl is not JSON"),
        (Scrape, "[" * 100_000, "a line of scrapes.jsonl.gz is not JSON"),
        (FleetPlan, "1" * 5000, "a plan of plans/ is not JSON"),
        (Done, "[]", "DONE must be an object"),
        (InsideEnvironment, "null", "env_inside.json must be an object"),
        (Event, '"t0"', "a line of events.jsonl must be an object"),
    ],
)
def test_text_that_is_not_the_json_object_of_its_file_is_refused(
    record_type: type[Record], text: str, message: str
) -> None:
    with pytest.raises(RunFileError) as refusal:
        record_type.from_json(text)

    assert message in str(refusal.value)


@pytest.mark.parametrize("number", [float("nan"), float("inf"), float("-inf")])
def test_a_number_that_is_not_finite_is_never_written(number: float) -> None:
    facts = replace(FACTS, facts={FACT_KEY: replace(FACT, value=number)})
    scrape = replace(SCRAPE, metrics={"uptime_s": number})

    with pytest.raises(RunFileError, match=r"facts\.json cannot be written"):
        facts.to_json()
    with pytest.raises(RunFileError, match=r"a line of scrapes\.jsonl\.gz cannot be written"):
        scrape.to_json()


def test_a_whole_file_ends_with_a_newline_and_a_line_holds_none() -> None:
    assert CONFIG.to_json().endswith("}\n")
    assert "\n" not in SPLIT_SCRAPE.to_json()
    assert "\n" not in EVENTS[0].to_json()


def test_records_with_the_same_entries_in_another_order_write_the_same_bytes() -> None:
    reordered_facts = replace(FACTS, facts=dict(sorted(FACTS.facts.items(), reverse=True)))
    reordered_done = replace(DONE, files=dict(sorted(DONE.files.items(), reverse=True)))

    assert reordered_facts.to_json() == FACTS.to_json()
    assert reordered_done.to_json() == DONE.to_json()


def test_specs_whose_server_configs_write_one_number_as_2_and_as_2_0_are_equal() -> None:
    whole = RunSpec.from_json(_edited(SPEC, ("server_config", "frame_rate_hz"), 2))
    decimal = RunSpec.from_json(_edited(SPEC, ("server_config", "frame_rate_hz"), 2.0))

    assert whole == decimal


def test_a_config_marks_a_hand_run_by_its_missing_spec() -> None:
    assert json.loads(HAND_RUN_CONFIG.to_json())["hand_run"] is True
    assert json.loads(CONFIG.to_json())["hand_run"] is False
    assert RunConfig.from_json(HAND_RUN_CONFIG.to_json()).hand_run
    assert not RunConfig.from_json(CONFIG.to_json()).hand_run


def test_a_config_holds_the_worker_its_scenario_kills_and_null_where_it_kills_none() -> None:
    def fault(config: RunConfig) -> object:
        return json.loads(config.to_json())["fleet"]["scenario"]["fault"]

    assert fault(WORKER_KILL_CONFIG) == {"model": "depth", "at_s": 20.0}
    assert fault(CONFIG) is None


def test_the_fleet_reads_a_plan_file_as_it_is_written(tmp_path: Path) -> None:
    plan_path = tmp_path / "fleet.json"
    plan_path.write_text(PLAN.to_json(), encoding="utf-8")

    assert asdict(read_plan(plan_path)) == asdict(PLAN)


@pytest.mark.parametrize(
    ("event", "line"),
    [
        (EVENTS[0], '{"event":"t0","t_s":0.0,"t0_mono_ns":5000000000}'),
        (EVENTS[1], '{"event":"server_ready","t_s":-7.25}'),
        (EVENTS[2], '{"event":"robot_start","t_s":0.0,"robot_id":"a"}'),
        (EVENTS[3], '{"event":"robot_join","t_s":10.0,"robot_id":"b"}'),
        (EVENTS[4], '{"event":"robot_kill","t_s":20.000125,"robot_id":"c"}'),
        (EVENTS[5], '{"event":"robot_stop","t_s":40.0,"robot_id":"a"}'),
        (EVENTS[6], '{"event":"scenario_end","t_s":40.0}'),
        (EVENTS[7], '{"event":"final_poll","t_s":40.5,"settled":true,"waited_s":0.3}'),
        (FAULT_EVENT, '{"event":"fault","t_s":20.000375,"model":"depth","pid":4242}'),
    ],
)
def test_an_event_line_holds_its_name_its_time_and_the_fields_of_its_kind(
    event: Event, line: str
) -> None:
    assert event.to_json() == line
    assert Event.from_json(line) == event


def test_an_event_time_written_as_a_whole_number_is_read_as_seconds() -> None:
    event = Event.from_json('{"event": "scenario_end", "t_s": 40}')

    assert event == Event(EventName.SCENARIO_END, 40.0)
    assert type(event.t_s) is float


def test_a_scrape_that_succeeded_holds_the_metrics_as_received_and_no_error() -> None:
    line = json.loads(SPLIT_SCRAPE.to_json())

    assert line == {
        "k": 1,
        "driver_before_ns": 6_000_000_000,
        "driver_after_ns": 6_000_900_000,
        "ok": True,
        "metrics": METRICS,
        "fleet": {
            "processes": {
                "fleet": asdict(PROCESS),
                "fleet_c": asdict(replace(PROCESS, pid=42)),
                "driver": asdict(PROCESS),
            },
            "cgroup": asdict(ROBOTS_CGROUP),
        },
    }
    assert Scrape.from_json(SPLIT_SCRAPE.to_json()).ok


def test_a_scrape_that_failed_holds_its_error_and_no_metrics() -> None:
    line = json.loads(FAILED_SCRAPE.to_json())

    assert line["ok"] is False
    assert line["error"] == "timed out after 1 s"
    assert "metrics" not in line
    assert not Scrape.from_json(FAILED_SCRAPE.to_json()).ok


def test_a_scrape_of_the_single_topology_has_no_cgroup_of_a_robots_container() -> None:
    assert "cgroup" not in json.loads(SCRAPE.to_json())["fleet"]
    assert Scrape.from_json(SCRAPE.to_json()).fleet_cgroup is None


def test_the_final_scrape_has_a_null_planned_second() -> None:
    assert json.loads(FINAL_SCRAPE.to_json())["k"] is None


_JSON_VALUES = st.recursive(
    st.none()
    | st.booleans()
    | st.integers()
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20,
)


@given(metrics=st.dictionaries(st.text(), _JSON_VALUES))
def test_any_metrics_object_comes_back_from_a_scrape_line_unchanged(
    metrics: dict[str, object],
) -> None:
    scrape = replace(SCRAPE, metrics=metrics)

    assert Scrape.from_json(scrape.to_json()) == scrape


def test_the_inside_environment_reads_what_the_environment_block_and_the_frame_set_give() -> None:
    block = collect_environment(None)

    inside = InsideEnvironment.from_json(json.dumps({**block, "frames": asdict(INSIDE.frames)}))

    assert json.loads(inside.to_json()) == {**block, "frames": asdict(INSIDE.frames)}


def test_the_run_environment_holds_the_inside_fields_beside_its_own() -> None:
    document = json.loads(SPLIT_ENVIRONMENT.to_json())

    assert document == {
        **json.loads(INSIDE.to_json()),
        "host_memory_total_bytes": 64 << 30,
        "engine": "docker",
        "engine_version": "29.0.0",
        "commit": "f" * 40,
        "dirty": False,
        "code_digest": CODE_DIGEST,
        "image_digest": IMAGE_DIGEST,
        "robots_cgroup": asdict(replace(LIMITS, cpu_limit_cores=None, memory_limit_bytes=None)),
    }


def test_a_gate_file_with_every_gate_passed_is_read_as_gated() -> None:
    text = """
    {"tests_passed": {"passed": true, "image_digest": "sha256:1a", "code_digest": "c0de",
                      "commit": "ffff", "dirty": false, "passed_count": 640, "failed_count": 0,
                      "skipped_count": 0, "deselected_count": 4,
                      "gate_file": "gates/1a-c0de.json", "output_sha256": "0b"},
     "image_built": {"built": true, "digest": "sha256:1a",
                     "containers": {"rri-system": "sha256:1a"}},
     "limits_ok": {"ok": true,
                   "expected": {"cpu_limit_cores": 2.0, "memory_limit_bytes": 4294967296,
                                "memory_swap_limit_bytes": 0},
                   "read": {"cpu_limit_cores": 2, "memory_limit_bytes": 4294967296,
                            "memory_swap_limit_bytes": 0}},
     "fresh": {"ok": true, "new_containers": true, "accepted_before_t0": 0},
     "config_ok": {"ok": true, "differences": []},
     "gated": true}
    """

    gate = Gate.from_json(text)

    assert gate.gated
    assert gate.limits_ok.read == gate.limits_ok.expected
    assert gate.image_built.containers == {"rri-system": "sha256:1a"}
    assert gate.config_ok == ConfigGate(ok=True, differences=())


def test_a_gate_without_a_config_gate_is_gated_when_the_other_gates_passed() -> None:
    assert MODEL_BENCH_GATE.gated
    assert json.loads(MODEL_BENCH_GATE.to_json())["config_ok"] is None
    assert json.loads(MODEL_BENCH_GATE.to_json())["gated"] is True


@pytest.mark.parametrize("gate", FAILED_GATES)
def test_a_gate_with_one_failed_gate_is_not_gated(gate: Gate) -> None:
    assert not gate.gated
    assert json.loads(gate.to_json())["gated"] is False


def test_the_tests_gate_of_a_run_is_the_record_of_its_gates_file() -> None:
    assert json.loads(GATE.to_json())["tests_passed"] == json.loads(SUITE_GATE.to_json())


def test_done_holds_the_sha_256_of_each_file_by_its_path() -> None:
    files = {"config.json": b"{}\n", "plans/fleet.json": b"plan"}

    assert done_files(files) == {
        "config.json": hashlib.sha256(b"{}\n").hexdigest(),
        "plans/fleet.json": hashlib.sha256(b"plan").hexdigest(),
    }


@pytest.mark.parametrize(
    "path", [SUMMARY_FILE, DONE_FILE, f"{SAMPLES_DIRECTORY}/robot_a.npz", "samples/more/x.npz"]
)
def test_done_leaves_out_the_samples_the_summary_and_itself(path: str) -> None:
    assert done_files({**RUN_FILES, path: b"anything"}) == done_files(RUN_FILES)


@pytest.mark.parametrize(
    "path", ["samples.json", "summary.json.tmp", "reports/summary.json", "plans/samples/a.npz"]
)
def test_done_covers_a_file_whose_name_only_resembles_one_it_leaves_out(path: str) -> None:
    assert done_files({path: b"x"}) == {path: hashlib.sha256(b"x").hexdigest()}


def test_a_done_with_a_null_summary_hash_is_valid_when_its_files_match() -> None:
    done = Done.from_json(DONE.to_json())

    assert done.summary_sha256 is None
    assert done.valid_for(RUN_FILES)


def test_a_done_stays_valid_when_the_summary_and_the_samples_are_written() -> None:
    later_files = {
        **RUN_FILES,
        SUMMARY_FILE: b"{}",
        DONE_FILE: SUMMARIZED_DONE.to_json().encode(),
        f"{SAMPLES_DIRECTORY}/robot_a.npz": b"samples",
    }

    assert SUMMARIZED_DONE.valid_for(later_files)


@pytest.mark.parametrize(
    "files",
    [
        {**RUN_FILES, "gate.json": RUN_FILES["gate.json"] + b" "},
        {path: content for path, content in RUN_FILES.items() if path != "events.jsonl"},
        {**RUN_FILES, "reports/extra.json": b"{}"},
        {**RUN_FILES, "summary.json.tmp": b"{}"},
        {},
    ],
    ids=["changed", "removed", "added", "left behind", "empty"],
)
def test_a_done_is_not_valid_for_files_that_differ_from_the_ones_it_recorded(
    files: dict[str, bytes],
) -> None:
    assert not DONE.valid_for(files)


_PATHS = st.text(alphabet="abc/._", min_size=1, max_size=12)


@given(files=st.dictionaries(_PATHS, st.binary(max_size=16)))
def test_a_done_made_from_any_files_is_valid_for_them(files: dict[str, bytes]) -> None:
    assert Done(done_files(files), None).valid_for(files)


@given(
    files=st.dictionaries(_PATHS, st.binary(max_size=16), min_size=1),
    data=st.data(),
    suffix=st.binary(min_size=1, max_size=4),
)
def test_a_change_to_any_file_a_done_covers_makes_it_invalid(
    files: dict[str, bytes], data: st.DataObject, suffix: bytes
) -> None:
    done = Done(done_files(files), None)
    path = data.draw(st.sampled_from(sorted(done.files)), label="changed path")

    assert not done.valid_for({**files, path: files[path] + suffix})


@pytest.mark.parametrize(
    "value", [50, 50.0, 0.0004, True, "between 5 and 10 robots", None], ids=repr
)
def test_a_fact_keeps_its_value_and_the_kind_of_its_value(
    value: int | float | str | bool | None,
) -> None:
    facts = replace(FACTS, facts={FACT_KEY: replace(FACT, value=value)})

    read = Facts.from_json(facts.to_json()).facts[FACT_KEY].value

    assert read == value
    assert type(read) is type(value)


def test_a_fact_keeps_whole_extremes_whole() -> None:
    facts = replace(FACTS, facts={FACT_KEY: replace(FACT, value=50, min=50, max=51)})

    read = Facts.from_json(facts.to_json()).facts[FACT_KEY]

    assert (type(read.min), type(read.max)) == (int, int)


def test_a_fact_with_no_counted_run_has_a_null_value_and_null_extremes() -> None:
    document = json.loads(FACTS.to_json())["facts"]["steady.depth.freshness_ms.p95"]

    assert document == {
        "value": None,
        "unit": "ms",
        "runs": 0,
        "min": None,
        "max": None,
        "run_ids": [],
        "dirty_runs": 0,
        "failed_checks": {},
        "attempts": 0,
        "failed_attempts": 0,
    }


def test_the_facts_file_records_its_model_set_its_code_digest_and_its_counted_runs() -> None:
    document = json.loads(FACTS.to_json())

    assert document["model_set"] == "fake"
    assert document["code_digest"] == CODE_DIGEST
    assert document["counted_run_ids"] == list(FACT.run_ids)
    assert document["facts"][FACT_KEY]["failed_checks"] == {RUN_ID: ["simulator_valid"]}
