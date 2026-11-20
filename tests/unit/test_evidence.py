"""Tests of the evidence runner: the commands, the overrides, the compose project, the resume
rule and the environment of a run, and what it gives a fake docker program, run by run and
signal by signal."""

from __future__ import annotations

import hashlib
import json
import os
import signal
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from pathlib import Path, PurePosixPath

import pytest

from bench import evidence
from bench.aggregate import run_path
from bench.candidates import CANDIDATES, Candidate
from bench.docker_ops import STATE_FORMAT
from bench.environment import collect_environment
from bench.evidence import (
    COMPOSE_FILES,
    CONTAINER_RESULTS,
    DEFAULT_REPEATS,
    PROJECT,
    READY_MARGIN_S,
    EvidenceError,
    Interrupted,
    Invocation,
    Outcome,
    RepositoryState,
    RunFailure,
    RunResult,
    SignalWatch,
    build_parser,
    candidates_named,
    driver_command,
    failed_gates,
    is_finished,
    keep_as_attempt,
    main,
    model_bench_commands,
    model_bench_environment,
    planned_runs,
    ready_timeout_s,
    run_compose,
    run_environment,
    run_one,
    run_overrides,
    suite_gate,
)
from bench.gates import stored_suite_gate, suite_gate_file, suite_gate_record
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    GATE_FILE,
    Done,
    Gate,
    InsideEnvironment,
    ModelSet,
    RunConfig,
    RunSpec,
    SuiteGate,
)
from bench.runplan import (
    MODEL_BENCH_KEY,
    compare_plan,
    evidence_plan,
    only,
    resolved_config,
    run_id,
)
from bench.scenarios import Scenario, steady
from bench.summarize import REPORTS_DIRECTORY, summary_is_current
from cloud.config import ServerConfig, Topology
from common.cgroup import Limits
from robot.frames import FrameSetStats
from tests.support.fake_docker import FakeDocker, install_fake_docker
from tests.support.runs import (
    CODE_DIGEST,
    IMAGE_DIGEST,
    LIMITS,
    gate,
    model_bench_reports,
    run_directory,
    run_spec,
    synthetic_run,
    write_run,
)

REPORTS = PurePosixPath("/app/results/raw/model_bench/run/reports")
DRIVER_LIMITS = Limits(
    version=2,
    cpu_limit_cores=None,
    cpu_period_us=100_000,
    memory_limit_bytes=2 * 2**30,
    memory_swap_limit_bytes=0,
    cpuset="0-31",
)
INSIDE = InsideEnvironment(
    cpu_model="Synthetic CPU",
    cpu_count_visible=32,
    machine="x86_64",
    kernel="6.6.0-synthetic",
    cgroup=DRIVER_LIMITS,
    python="3.14.0",
    packages={"numpy": "2.0.0"},
    model_sha256={},
    start_utc="2026-01-01T00:00:00+00:00",
    frames=FrameSetStats(
        count=16, jpeg_quality=80, min_bytes=30_000, mean_bytes=40_000.0, max_bytes=50_000
    ),
)
STATE = RepositoryState(commit="0123abc", dirty=True, code_digest=CODE_DIGEST)


def _invocation(results: Path) -> Invocation:
    spec = run_spec("steady")
    return Invocation(
        results=results,
        image_digest=IMAGE_DIGEST,
        tests_passed=gate(spec).tests_passed,
        commit=STATE.commit,
        dirty=STATE.dirty,
        engine_version="29.0.0",
        host_memory_total_bytes=32 * 2**30,
        user="1000:1000",
    )


def test_the_model_benchmark_times_each_graph_at_one_and_two_threads_then_the_rest() -> None:
    names = [name for name, _command in model_bench_commands(3, REPORTS)]

    assert names == [
        "cls_t1_published",
        "cls_t2_published",
        "depth_t1_published",
        "depth_t1_batched",
        "depth_t2_published",
        "depth_t2_batched",
        "stages",
        "dispatch",
        "websocket",
    ]


def test_each_benchmark_command_writes_its_report_under_the_run_s_reports() -> None:
    commands = dict(model_bench_commands(3, REPORTS))

    assert commands["depth_t2_published"] == [
        *("python", "-m", "bench.model_bench", "batch", "--model", "depth"),
        *("--graph", "published", "--threads", "2", "--batch-sizes", "1"),
        *("--out", str(REPORTS / "depth_t2_published.json")),
    ]
    assert commands["cls_t1_published"][-4:] == [
        "--batch-sizes",
        "1,2,4,8",
        "--out",
        str(REPORTS / "cls_t1_published.json"),
    ]
    assert commands["dispatch"] == [
        *("python", "-m", "bench.transport_bench", "dispatch", "--loop", "uvloop"),
        *("--out", str(REPORTS / "dispatch.json")),
    ]
    assert commands["websocket"] == [
        *("python", "-m", "bench.transport_bench", "websocket"),
        *("--robots", "50", "--fps", "30", "--duration-s", "20", "--loop", "uvloop"),
        *("--seed", "3", "--out", str(REPORTS / "websocket.json")),
    ]


@pytest.mark.parametrize("model_set", list(ModelSet))
def test_the_overrides_of_an_evidence_run_resolve_to_its_spec_s_config(
    model_set: ModelSet,
) -> None:
    for spec in evidence_plan(1, model_set, CODE_DIGEST):
        assert resolved_config(spec.topology, run_overrides(spec)) == spec.server_config


@pytest.mark.parametrize("candidate", CANDIDATES, ids=[candidate.name for candidate in CANDIDATES])
def test_the_overrides_of_a_comparison_run_resolve_to_its_arm_s_config(
    candidate: Candidate,
) -> None:
    for spec in compare_plan(candidate, CODE_DIGEST):
        assert resolved_config(spec.topology, run_overrides(spec)) == spec.server_config


def test_all_names_every_candidate_in_order_and_a_name_names_one() -> None:
    assert candidates_named("all") == list(CANDIDATES)
    assert candidates_named(CANDIDATES[2].name) == [CANDIDATES[2]]


def test_a_name_that_is_no_candidate_is_refused() -> None:
    with pytest.raises(ValueError, match="fifo is not a candidate"):
        candidates_named("fifo")


@pytest.mark.parametrize(
    ("topology", "extra", "files", "profiles"),
    [
        (Topology.SINGLE, None, ("docker-compose.yml",), ()),
        (Topology.SPLIT, None, ("docker-compose.split.yml",), ("robots",)),
        (Topology.SINGLE, Path("/x/one_cpu.yml"), ("docker-compose.yml", "one_cpu.yml"), ()),
    ],
    ids=["single", "split", "single with an extra file"],
)
def test_a_run_s_compose_project_has_its_topology_s_file_then_the_extra_one(
    topology: Topology, extra: Path | None, files: tuple[str, ...], profiles: tuple[str, ...]
) -> None:
    compose = run_compose(topology, extra, {"skip_rule": False}, Path("/results"), "abc")

    assert tuple(path.name for path in compose.files) == files
    assert compose.profiles == profiles
    assert compose.variables == {
        "RRI_OVERRIDES": '{"skip_rule": false}',
        "RRI_RESULTS": "/results",
        "GIT_COMMIT": "abc",
    }


def test_the_gateway_has_every_model_s_warm_up_and_a_margin_to_be_ready() -> None:
    config = run_spec("steady").server_config

    expected_s = 2 * ServerConfig().warmup_timeout_ms / 1000 + READY_MARGIN_S
    assert ready_timeout_s(config) == expected_s


def test_the_driver_of_a_single_run_reaches_the_gateway_in_its_own_container() -> None:
    spec = run_spec("steady")

    assert driver_command(spec) == [
        *("python", "-m", "bench.run", "--run-dir"),
        f"{CONTAINER_RESULTS}/raw/steady/{run_id(spec)}",
    ]


def test_the_driver_of_a_split_run_reaches_the_gateway_by_its_service_name() -> None:
    spec = run_spec("disconnect", topology=Topology.SPLIT)

    assert driver_command(spec)[5:] == [
        *("--url", "ws://system:8080/ws", "--metrics-url", "http://system:8080/metrics"),
        "--robots-cgroup",
    ]


def test_the_driver_of_a_comparison_run_writes_under_its_candidate_and_arm() -> None:
    spec = compare_plan(CANDIDATES[0], CODE_DIGEST)[1]

    assert driver_command(spec)[4] == (
        f"{CONTAINER_RESULTS}/raw/compare/{CANDIDATES[0].name}/b/{run_id(spec)}"
    )


def test_an_unfinished_directory_is_kept_as_the_next_free_attempt(tmp_path: Path) -> None:
    run_dir = tmp_path / "steady" / "single-x-r1"
    run_dir.mkdir(parents=True)
    (tmp_path / "steady" / "single-x-r1.attempt1").mkdir()

    kept = keep_as_attempt(run_dir)

    assert kept == tmp_path / "steady" / "single-x-r1.attempt2"
    assert kept.is_dir()
    assert not run_dir.exists()


def test_a_run_without_a_directory_keeps_nothing(tmp_path: Path) -> None:
    assert keep_as_attempt(tmp_path / "single-x-r1") is None


def test_a_run_with_a_valid_done_is_finished(tmp_path: Path) -> None:
    run = synthetic_run(steady(), run_spec("steady"))
    run_dir = run_directory(tmp_path, run.spec)
    write_run(run_dir, run)

    assert is_finished(run_dir)


def _change_a_raw_file(run_dir: Path) -> None:
    (run_dir / GATE_FILE).write_text("{}")


def _leave_a_file(run_dir: Path) -> None:
    (run_dir / "leftover.tmp").write_text("")


def _remove_done(run_dir: Path) -> None:
    (run_dir / DONE_FILE).unlink()


def _break_done(run_dir: Path) -> None:
    (run_dir / DONE_FILE).write_text("{")


@pytest.mark.parametrize(
    "damage",
    [_change_a_raw_file, _leave_a_file, _remove_done, _break_done],
    ids=["a raw file changed", "a file left behind", "no DONE", "a malformed DONE"],
)
def test_a_run_whose_done_does_not_hold_is_not_finished(
    tmp_path: Path, damage: Callable[[Path], None]
) -> None:
    run = synthetic_run(steady(), run_spec("steady"))
    run_dir = run_directory(tmp_path, run.spec)
    write_run(run_dir, run)
    damage(run_dir)

    assert not is_finished(run_dir)


def test_a_gate_that_failed_is_named() -> None:
    spec = run_spec("steady")

    assert failed_gates(gate(spec, gated=False)) == ["limits_ok"]
    assert failed_gates(gate(spec)) == []


def test_a_single_run_records_the_driver_s_container_and_no_robots_container(
    tmp_path: Path,
) -> None:
    spec = run_spec("steady")
    scrapes = synthetic_run(steady(), spec).scrapes

    environment = run_environment(spec, INSIDE, scrapes, _invocation(tmp_path))

    assert environment.inside == INSIDE
    assert environment.robots_cgroup is None
    assert (environment.commit, environment.dirty) == (STATE.commit, STATE.dirty)
    assert (environment.code_digest, environment.image_digest) == (CODE_DIGEST, IMAGE_DIGEST)


def test_a_split_run_records_the_gateway_s_limits_and_the_robots_container_apart(
    tmp_path: Path,
) -> None:
    spec = run_spec("steady", topology=Topology.SPLIT)
    scrapes = synthetic_run(steady(), spec).scrapes

    environment = run_environment(spec, INSIDE, scrapes, _invocation(tmp_path))

    assert environment.inside == replace(INSIDE, cgroup=LIMITS)
    assert environment.robots_cgroup == DRIVER_LIMITS


def test_a_split_run_without_a_successful_scrape_has_no_environment(tmp_path: Path) -> None:
    spec = run_spec("steady", topology=Topology.SPLIT)

    with pytest.raises(RunFailure, match="no scrape succeeded"):
        run_environment(spec, INSIDE, (), _invocation(tmp_path))


def _specs(argv: list[str]) -> tuple[list[RunSpec], Path | None]:
    return planned_runs(build_parser().parse_args(argv), CODE_DIGEST)


def test_a_run_command_plays_three_repeats_of_the_real_plan_by_default() -> None:
    specs, extra = _specs(["run"])

    assert specs == evidence_plan(DEFAULT_REPEATS, ModelSet.REAL, CODE_DIGEST)
    assert extra is None


def test_only_keeps_the_entries_named_in_the_plan_s_order() -> None:
    specs, _extra = _specs(
        ["run", "--repeats", "1", "--models", "fake", "--only", "massive:split,steady:single"]
    )

    assert [(spec.scenario_key, spec.topology) for spec in specs] == [
        ("steady", Topology.SINGLE),
        ("massive", Topology.SPLIT),
    ]
    assert {spec.model_set for spec in specs} == {ModelSet.FAKE}


def test_an_extra_compose_file_is_part_of_every_spec_by_its_hash(tmp_path: Path) -> None:
    extra = tmp_path / "one_cpu.yml"
    extra.write_text("services:\n  system:\n    cpus: 1.0\n")

    specs, path = _specs(
        ["run", "--repeats", "1", "--only", "steady:single", "--extra-compose", str(extra)]
    )

    assert path == extra.resolve()
    assert [spec.extra_compose for spec in specs] == [
        hashlib.sha256(extra.read_bytes()).hexdigest()
    ]


@pytest.mark.parametrize(("name", "count"), [("fifo_vs_latest", 4), ("all", 4 * len(CANDIDATES))])
def test_a_comparison_plays_each_candidate_s_four_runs(name: str, count: int) -> None:
    specs, _extra = _specs(["compare", name])

    assert len(specs) == count
    assert specs[:4] == compare_plan(candidates_named(name)[0], CODE_DIGEST)


def test_an_entry_that_the_plan_does_not_have_is_refused() -> None:
    with pytest.raises(ValueError, match="no run of the plan is sweep_n2:single"):
        _specs(["run", "--only", "sweep_n2:single"])


def test_the_model_benchmark_records_the_environment_of_its_first_report(tmp_path: Path) -> None:
    spec = run_spec(MODEL_BENCH_KEY, model_set=ModelSet.REAL)
    environment_block = collect_environment(None)
    report = {
        "kind": "model_bench.batch",
        "environment": environment_block,
        "frames": asdict(INSIDE.frames),
    }
    report_path = tmp_path / "cls_t1_published.json"
    report_path.write_text(json.dumps(report))

    environment = model_bench_environment(spec, report_path, _invocation(tmp_path))

    assert environment.inside.kernel == environment_block["kernel"]
    assert environment.inside.frames == INSIDE.frames
    assert environment.robots_cgroup is None
    assert (environment.commit, environment.image_digest) == (STATE.commit, IMAGE_DIGEST)


def test_a_report_without_an_environment_is_a_failed_run(tmp_path: Path) -> None:
    report_path = tmp_path / "cls_t1_published.json"
    report_path.write_text(json.dumps({"kind": "model_bench.batch"}))

    with pytest.raises(RunFailure, match="has no environment to record"):
        model_bench_environment(run_spec(MODEL_BENCH_KEY), report_path, _invocation(tmp_path))


@pytest.fixture
def restored_signals() -> Iterator[None]:
    """Give back, after the test, the handlers that a SignalWatch of the test replaces."""
    previous = {number: signal.getsignal(number) for number in SignalWatch.HANDLED}
    yield
    for number, handler in previous.items():
        signal.signal(number, handler)


@pytest.mark.usefixtures("restored_signals")
def test_the_first_signal_stops_the_runner_and_the_others_are_ignored() -> None:
    watch = SignalWatch()
    with pytest.raises(Interrupted) as interrupted:
        os.kill(os.getpid(), signal.SIGHUP)
    os.kill(os.getpid(), signal.SIGTERM)

    assert interrupted.value.signal_number == signal.SIGHUP
    assert watch.received == signal.SIGHUP
    assert {signal.getsignal(number) for number in SignalWatch.HANDLED} == {signal.SIG_IGN}


PASSED_OUTPUT = b"12 passed, 3 deselected in 9.00s\n"
FAILED_OUTPUT = b"1 failed, 11 passed, 3 deselected in 9.00s\n"


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path / "bin", monkeypatch)


def _gate_path(results: Path) -> Path:
    return results / suite_gate_file(IMAGE_DIGEST, STATE.code_digest)


def _store_gate(results: Path, output: bytes, exit_status: int) -> SuiteGate:
    """Store a tests gate under the results root, as an earlier invocation left it."""
    record = suite_gate_record(
        output, exit_status, IMAGE_DIGEST, STATE.code_digest, STATE.commit, STATE.dirty
    )
    path = _gate_path(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.to_json())
    path.with_suffix(".txt").write_bytes(output)
    return record


def _suite_gate(results: Path) -> SuiteGate:
    compose = run_compose(Topology.SINGLE, None, {}, results, STATE.commit)
    return suite_gate(compose, results, IMAGE_DIGEST, STATE)


def test_the_suite_runs_in_the_image_as_its_user_and_its_gate_is_stored(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    fake_docker.answer(["run"], stdout=PASSED_OUTPUT.decode())

    record = _suite_gate(results)

    assert record.passed
    assert (record.passed_count, record.deselected_count) == (12, 3)
    assert stored_suite_gate(results, IMAGE_DIGEST, STATE.code_digest) == record
    container = f"rri-tests-{_gate_path(results).stem}"
    # A container of that name that an invocation killed before its cleanup is removed first.
    clear, run, remove = fake_docker.argvs()
    assert clear == ["rm", "--force", container]
    assert run[run.index("run") :] == [
        *("run", "--no-TTY", "--name", container, "system"),
        *("pytest", "-q", "-p", "no:cacheprovider", "-rs", "-m", "not docker"),
    ]
    assert remove == ["rm", "--force", container]


def test_a_stored_gate_that_passed_is_taken_without_running_the_suite(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    stored = _store_gate(results, PASSED_OUTPUT, 0)

    assert _suite_gate(results) == stored
    assert fake_docker.calls() == []


@pytest.mark.parametrize("earlier_failures", [0, 2])
def test_a_stored_gate_that_failed_is_run_again_and_its_output_kept(
    tmp_path: Path, fake_docker: FakeDocker, earlier_failures: int
) -> None:
    results = tmp_path / "results"
    _store_gate(results, FAILED_OUTPUT, 1)
    gate_path = _gate_path(results)
    for k in range(1, earlier_failures + 1):
        gate_path.with_name(f"{gate_path.stem}.fail{k}.txt").write_bytes(b"older\n")
    fake_docker.answer(["run"], stdout=PASSED_OUTPUT.decode())

    record = _suite_gate(results)

    kept = gate_path.with_name(f"{gate_path.stem}.fail{earlier_failures + 1}.txt")
    assert record.passed
    assert kept.read_bytes() == FAILED_OUTPUT
    assert gate_path.with_suffix(".txt").read_bytes() == PASSED_OUTPUT


def test_a_suite_that_fails_stops_the_invocation_and_its_gate_is_stored_as_failed(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    fake_docker.answer(["run"], stdout=FAILED_OUTPUT.decode(), exit_status=1)

    with pytest.raises(EvidenceError, match="the tests gate failed"):
        _suite_gate(results)

    stored = stored_suite_gate(results, IMAGE_DIGEST, STATE.code_digest)
    assert stored is not None
    assert (stored.passed, stored.failed_count) == (False, 1)


def test_suite_output_without_a_count_line_stops_the_invocation_with_no_gate(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    fake_docker.answer(["run"], stdout="ImportError while loading conftest\n", exit_status=4)

    with pytest.raises(EvidenceError, match="no count line"):
        _suite_gate(results)

    assert _gate_path(results).with_suffix(".txt").is_file()
    assert not _gate_path(results).exists()


@pytest.mark.parametrize(
    ("result", "line"),
    [
        (RunResult("raw/steady/x", Outcome.SKIPPED, ""), "raw/steady/x: finished before, skipped"),
        (
            RunResult("raw/steady/x", Outcome.FINISHED, "not gated: limits_ok"),
            "raw/steady/x: finished; not gated: limits_ok",
        ),
    ],
)
def test_a_run_s_line_names_its_directory_and_what_became_of_it(
    result: RunResult, line: str
) -> None:
    assert result.line() == line


GATEWAY_ID = "a" * 64
ROBOTS_ID = "b" * 64
STALE_ID = "5" * 64
# The environment block of every benchmark report: a container with the limits the gate
# expects.
REPORT_ENVIRONMENT = {
    key: value
    for key, value in json.loads(replace(INSIDE, cgroup=LIMITS).to_json()).items()
    if key != "frames"
}


def _serve(fake: FakeDocker) -> None:
    """Answer as an engine whose containers start, print their ready line and run to the end.

    The rules a test gives before these win over them.
    """
    fake.answer(["--quiet", "system"], stdout=f"{GATEWAY_ID}\n")
    fake.answer(["{{.Id}}"], stdout=f"{ROBOTS_ID}\n")
    fake.answer(["logs", "--follow"], stdout="ready: listening on 0.0.0.0:8080\n", block=True)
    fake.answer([STATE_FORMAT], stdout="running 0 0 false\n")
    fake.answer(["{{.Image}}"], stdout=f"{IMAGE_DIGEST}\n")


def _driver_files(
    results: Path, spec: RunSpec, scenario: Scenario, staging: Path
) -> dict[Path, bytes]:
    """Return what the scenario driver and the fleet write in a run of the spec that went well.

    These are the files of a synthetic run but those the runner writes, each at its place in
    the run's directory under the results root.
    """
    write_run(staging, synthetic_run(scenario, spec))
    runner_files = {CONFIG_FILE, ENVIRONMENT_FILE, GATE_FILE, DONE_FILE}
    return {
        results / run_path(spec) / path.relative_to(staging): path.read_bytes()
        for path in staging.rglob("*")
        if path.is_file() and path.name not in runner_files
    }


def _compose_argv(spec: RunSpec, results: Path, *arguments: str) -> list[str]:
    """Return a compose command line of the run's project, as the fake docker records it."""
    compose = run_compose(spec.topology, None, run_overrides(spec), results, STATE.commit)
    return compose.command(*arguments)[1:]


# How the runner lists the project's containers before it removes them.
LISTINGS = [
    ["ps", "--all", "--quiet", "--filter", f"label=com.docker.compose.project={PROJECT}"],
    ["ps", "--all", "--quiet", "--filter", "name=^rri-"],
]


def test_a_serving_run_starts_fresh_plays_as_the_host_user_and_finishes_with_its_summary(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    spec = run_spec("steady")
    fake_docker.answer(
        [f"label=com.docker.compose.project={PROJECT}"], stdout=f"{STALE_ID}\n", occurrence=1
    )
    fake_docker.answer(
        ["bench.run"], files=_driver_files(results, spec, steady(), tmp_path / "staging")
    )
    _serve(fake_docker)

    result = run_one(spec, steady(), None, _invocation(results))

    run_dir = results / result.path
    down = _compose_argv(spec, results, "down", "--remove-orphans")
    as_host_user = ("--no-TTY", "--user", "1000:1000", "system")
    assert (result.outcome, result.detail) == (Outcome.FINISHED, "gated")
    assert summary_is_current(run_dir, Done.from_json((run_dir / DONE_FILE).read_text()))
    assert fake_docker.argvs() == [
        *LISTINGS,
        down,
        ["rm", "--force", STALE_ID],
        _compose_argv(spec, results, "up", "--detach", "--force-recreate", "system"),
        _compose_argv(spec, results, "ps", "--all", "--quiet", "system"),
        ["logs", "--follow", GATEWAY_ID],
        _compose_argv(spec, results, "exec", *as_host_user, *driver_command(spec)),
        ["container", "inspect", "--format", STATE_FORMAT, GATEWAY_ID],
        ["container", "inspect", "--format", "{{.Image}}", GATEWAY_ID],
        *LISTINGS,
        down,
    ]


def test_a_run_whose_driver_fails_stays_without_done_and_is_kept_as_an_attempt_when_played_again(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    spec = run_spec("steady")
    fake_docker.answer(
        ["bench.run"],
        stdout="robot.fleet exited with 1\n",
        exit_status=1,
        files=_driver_files(results, spec, steady(), tmp_path / "staging"),
    )
    _serve(fake_docker)
    invocation = _invocation(results)

    first = run_one(spec, steady(), None, invocation)
    second = run_one(spec, steady(), None, invocation)

    run_dir = results / first.path
    attempt = run_dir.with_name(f"{run_dir.name}.attempt1")
    assert first.outcome is Outcome.FAILED
    assert first.detail.startswith("the driver exited with 1:\nrobot.fleet exited with 1\n")
    assert second.detail.startswith(f"the directory before it is kept as {attempt.name}; ")
    assert [
        (attempt / CONFIG_FILE).is_file(),
        (attempt / DONE_FILE).exists(),
        (run_dir / DONE_FILE).exists(),
    ] == [True, False, False]


@pytest.mark.parametrize(
    ("state", "described"),
    [
        ("exited 137 0 true\n", "exited, exit code 137, 0 restarts, out of memory true"),
        ("running 0 1 false\n", "running, exit code 0, 1 restarts, out of memory false"),
    ],
    ids=["exited", "restarted"],
)
def test_a_split_run_whose_gateway_did_not_run_to_the_end_stays_without_done(
    tmp_path: Path, fake_docker: FakeDocker, state: str, described: str
) -> None:
    results = tmp_path / "results"
    spec = run_spec("steady", topology=Topology.SPLIT)
    fake_docker.answer([STATE_FORMAT, GATEWAY_ID], stdout=state)
    fake_docker.answer(["logs", "--tail"], stdout="worker cls: exited with -9\n")
    fake_docker.answer(
        ["bench.run"], files=_driver_files(results, spec, steady(), tmp_path / "staging")
    )
    _serve(fake_docker)

    result = run_one(spec, steady(), None, _invocation(results))

    assert result.outcome is Outcome.FAILED
    assert result.detail == (
        f"the system container did not run to the end: {described}\n"
        "gateway log:\nworker cls: exited with -9"
    )
    assert not (results / result.path / DONE_FILE).exists()


def _reports_written(results: Path, spec: RunSpec) -> dict[str, tuple[str, dict[Path, bytes]]]:
    """Return, per benchmark command in order, its report's path inside and what it writes."""
    reports = PurePosixPath(run_path(spec)) / REPORTS_DIRECTORY
    # The support module names a batch report after its kind as well.
    documents = {
        name.removeprefix("batch_"): {
            **document,
            "environment": REPORT_ENVIRONMENT,
            "frames": asdict(INSIDE.frames),
        }
        for name, document in model_bench_reports().items()
    }
    return {
        name: (
            str(CONTAINER_RESULTS / reports / f"{name}.json"),
            {results / reports / f"{name}.json": json.dumps(documents[name]).encode()},
        )
        for name, _command in model_bench_commands(spec.repeat, CONTAINER_RESULTS / reports)
    }


def _write_reports_when_run(fake: FakeDocker, results: Path, spec: RunSpec) -> None:
    """Have each benchmark command write its report through the results mount."""
    for out, files in _reports_written(results, spec).values():
        fake.answer([out], files=files)


def test_a_model_benchmark_runs_each_command_as_the_host_user_in_a_container_of_its_own(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    spec = run_spec(MODEL_BENCH_KEY, model_set=ModelSet.REAL, repeat=2)
    _write_reports_when_run(fake_docker, results, spec)
    _serve(fake_docker)

    result = run_one(spec, None, None, _invocation(results))

    recorded = Gate.from_json((results / result.path / GATE_FILE).read_text())
    names = list(_reports_written(results, spec))
    containers = [f"rri-bench-{run_id(spec)}-{k}" for k in range(1, len(names) + 1)]
    runs = [argv for argv in fake_docker.argvs() if "run" in argv]
    removals = [argv[2:] for argv in fake_docker.argvs() if argv[:2] == ["rm", "--force"]]
    websocket = runs[names.index("websocket")]
    assert (result.outcome, result.detail) == (Outcome.FINISHED, "gated")
    assert recorded.image_built.containers == dict.fromkeys(names, IMAGE_DIGEST)
    assert [argv[argv.index("--name") + 1] for argv in runs] == containers
    assert {argv[argv.index("--user") + 1] for argv in runs} == {"1000:1000"}
    assert websocket[websocket.index("--seed") + 1] == "2"
    assert removals == [[container] for container in containers]


def test_a_benchmark_command_that_fails_fails_the_run_without_done(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    spec = run_spec(MODEL_BENCH_KEY, model_set=ModelSet.REAL)
    written = _reports_written(results, spec)
    out, files = written["dispatch"]
    fake_docker.answer([out], stdout="dispatch: the worker exited\n", exit_status=1, files=files)
    _write_reports_when_run(fake_docker, results, spec)
    _serve(fake_docker)

    result = run_one(spec, None, None, _invocation(results))

    container = f"rri-bench-{run_id(spec)}-{list(written).index('dispatch') + 1}"
    assert result.outcome is Outcome.FAILED
    assert result.detail == "dispatch exited with 1:\ndispatch: the worker exited"
    assert not (results / result.path / DONE_FILE).exists()
    assert fake_docker.argvs()[-1] == ["rm", "--force", container]


@pytest.fixture
def engine(
    fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch, restored_signals: None
) -> FakeDocker:
    """A fake docker that builds the image and passes the suite, for main over STATE's tree."""
    fake_docker.answer(["config", "--images"], stdout="realtime-robot-inference:dev\n")
    fake_docker.answer(["image", "inspect"], stdout=f"{IMAGE_DIGEST}\n")
    fake_docker.answer(["pytest"], stdout=PASSED_OUTPUT.decode())
    fake_docker.answer(["version"], stdout="29.0.0\n")
    fake_docker.answer(["info"], stdout=f"{32 * 2**30}\n")
    # The commit, the dirt and the code digest of the tree come from git.
    monkeypatch.setattr(evidence, "repository_state", lambda: STATE)
    return fake_docker


def _run_directories_played(argvs: list[list[str]]) -> list[str]:
    """Return each run's directory inside the containers, in the order the runs began.

    A serving run begins with its driver, the model benchmark with its first command.
    """
    drivers = {
        index: argv[argv.index("--run-dir") + 1]
        for index, argv in enumerate(argvs)
        if "bench.run" in argv
    }
    benchmarks = {
        index: str(PurePosixPath(argv[argv.index("--out") + 1]).parent.parent)
        for index, argv in enumerate(argvs)
        if "--out" in argv
    }
    return [directory for _index, directory in sorted({**drivers, **benchmarks}.items())]


@pytest.mark.parametrize(
    ("command", "plan"),
    [
        (
            ["run", "--repeats", "2", "--models", "fake"],
            evidence_plan(2, ModelSet.FAKE, CODE_DIGEST),
        ),
        (
            ["compare", "all"],
            [spec for candidate in CANDIDATES for spec in compare_plan(candidate, CODE_DIGEST)],
        ),
    ],
    ids=["evidence", "comparisons"],
)
def test_the_runner_plays_its_plan_in_order_each_run_with_its_spec_and_its_seed(
    tmp_path: Path, engine: FakeDocker, command: list[str], plan: list[RunSpec]
) -> None:
    results = tmp_path / "results"
    # Every run fails once it has begun, so the next one follows at once.
    engine.answer(["bench.run"], exit_status=1)
    engine.answer(["bench.model_bench"], exit_status=1)
    _serve(engine)

    status = main([*command, "--results", str(results)])

    configs = [
        RunConfig.from_json((results / run_path(spec) / CONFIG_FILE).read_text()) for spec in plan
    ]
    assert status == 1
    assert _run_directories_played(engine.argvs()) == [
        str(CONTAINER_RESULTS / run_path(spec)) for spec in plan
    ]
    assert [(config.spec, config.run_id, config.seed) for config in configs] == [
        (spec, run_id(spec), spec.repeat) for spec in plan
    ]


def test_an_extra_compose_file_is_laid_over_the_topology_s_and_its_path_recorded(
    tmp_path: Path, engine: FakeDocker
) -> None:
    results = tmp_path / "results"
    extra = tmp_path / "one_cpu.yml"
    extra.write_text("services:\n  system:\n    cpus: 1.0\n")
    engine.answer(["bench.run"], exit_status=1)
    _serve(engine)

    main(
        [
            *("run", "--results", str(results), "--repeats", "1", "--models", "fake"),
            *("--only", "steady:single", "--extra-compose", str(extra)),
        ]
    )

    (planned,) = only(evidence_plan(1, ModelSet.FAKE, CODE_DIGEST), ["steady:single"])
    spec = replace(planned, extra_compose=hashlib.sha256(extra.read_bytes()).hexdigest())
    config = RunConfig.from_json((results / run_path(spec) / CONFIG_FILE).read_text())
    up = next(argv for argv in engine.argvs() if "up" in argv)
    assert (config.spec, config.extra_compose_path) == (spec, str(extra.resolve()))
    assert up[3:7] == ["-f", str(COMPOSE_FILES[Topology.SINGLE]), "-f", str(extra.resolve())]


TWO_RUNS = ["--repeats", "1", "--models", "fake", "--only", "steady:single,steady:split"]


def test_a_signal_during_the_removal_of_a_run_s_containers_lets_the_removal_finish(
    tmp_path: Path, engine: FakeDocker
) -> None:
    results = tmp_path / "results"
    engine.answer([f"label=com.docker.compose.project={PROJECT}"], stdout=f"{GATEWAY_ID}\n")
    # The first down clears the project before the run, the second removes the run's
    # containers after it.
    engine.answer(["down"], occurrence=2, signal_number=signal.SIGINT)
    engine.answer(["bench.run"], exit_status=1)
    _serve(engine)

    status = main(["run", "--results", str(results), *TWO_RUNS])

    argvs = engine.argvs()
    down = next(argv for argv in argvs if "down" in argv)
    cut_short = [index for index, argv in enumerate(argvs) if argv == down][1]
    assert status == 128 + signal.SIGINT
    assert argvs[cut_short + 1 :] == [*LISTINGS, down, ["rm", "--force", GATEWAY_ID]]
    assert len([argv for argv in argvs if "bench.run" in argv]) == 1


def test_a_signal_stops_the_runner_after_its_run_even_when_the_removal_then_fails(
    tmp_path: Path, engine: FakeDocker
) -> None:
    results = tmp_path / "results"
    engine.answer(["bench.run"], signal_number=signal.SIGTERM)
    engine.answer(["down"], occurrence=2, stderr="cannot stop the container\n", exit_status=1)
    _serve(engine)

    status = main(["run", "--results", str(results), *TWO_RUNS])

    assert status == 128 + signal.SIGTERM
    assert len([argv for argv in engine.argvs() if "bench.run" in argv]) == 1


@pytest.mark.usefixtures("restored_signals")
def test_a_signal_during_the_removal_of_the_tests_container_lets_the_removal_finish(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    fake_docker.answer(["run"], stdout=PASSED_OUTPUT.decode())
    # The first removal clears the name before the suite runs; the second follows it.
    fake_docker.answer(["rm"], occurrence=2, signal_number=signal.SIGHUP)
    SignalWatch()

    with pytest.raises(Interrupted):
        _suite_gate(results)

    container = f"rri-tests-{_gate_path(results).stem}"
    assert fake_docker.argvs()[2:] == [["rm", "--force", container]] * 2


@pytest.mark.usefixtures("restored_signals")
def test_a_signal_during_the_removal_of_a_benchmark_container_lets_the_removal_finish(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    spec = run_spec(MODEL_BENCH_KEY, model_set=ModelSet.REAL)
    container = f"rri-bench-{run_id(spec)}-1"
    fake_docker.answer(["rm", "--force", container], occurrence=1, signal_number=signal.SIGINT)
    _serve(fake_docker)
    SignalWatch()

    with pytest.raises(Interrupted):
        run_one(spec, None, None, _invocation(tmp_path / "results"))

    assert fake_docker.argvs()[-2:] == [["rm", "--force", container]] * 2
