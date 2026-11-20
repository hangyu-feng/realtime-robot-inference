"""Tests of the evidence runner's fault runs: the fault helper's event, and what a worker-kill
run gives a fake docker program."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from bench import evidence
from bench.aggregate import run_path
from bench.docker_ops import STATE_FORMAT
from bench.evidence import (
    CONTAINER_RESULTS,
    Invocation,
    Outcome,
    RunFailure,
    fault_event,
    run_compose,
    run_one,
    run_overrides,
)
from bench.runfiles import (
    CONFIG_FILE,
    DONE_FILE,
    ENVIRONMENT_FILE,
    EVENTS_FILE,
    GATE_FILE,
    SUMMARY_FILE,
    Done,
    Event,
    EventName,
    RunSpec,
)
from bench.scenarios import worker_kill
from bench.summarize import summary_is_current
from cloud.config import Topology
from tests.support.fake_docker import FakeDocker, install_fake_docker
from tests.support.runs import IMAGE_DIGEST, gate, run_spec, worker_kill_run, write_run

FAULT = Event(EventName.FAULT, 20.0004, model="depth", pid=12)
FAULT_LINE = FAULT.to_json() + "\n"
COMMIT = "0123abc"
GATEWAY_ID = "a" * 64
ROBOTS_ID = "b" * 64


def test_the_fault_event_is_the_one_line_the_helper_printed() -> None:
    assert fault_event(0, FAULT_LINE.encode()) == FAULT


@pytest.mark.parametrize(
    ("exit_status", "stdout", "reason"),
    [
        (1, b"", "the fault helper exited with 1"),
        (0, b"", "printed 0 lines"),
        (0, (FAULT_LINE * 2).encode(), "printed 2 lines"),
        (0, b"killed\n", "is not JSON"),
        (0, b'{"event": "fault", "t_s": 20.0, "model": "depth"}\n', "pid"),
        (0, b'{"event": "t0", "t_s": 0.0, "t0_mono_ns": 5}\n', "a t0 event, not a fault event"),
    ],
    ids=["failed", "silent", "two lines", "no JSON", "no pid", "another event"],
)
def test_anything_but_one_fault_line_from_a_helper_that_exited_0_fails_the_run(
    exit_status: int, stdout: bytes, reason: str
) -> None:
    with pytest.raises(RunFailure, match=reason):
        fault_event(exit_status, stdout)


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path / "bin", monkeypatch)


def _invocation(results: Path, spec: RunSpec) -> Invocation:
    return Invocation(
        results=results,
        image_digest=IMAGE_DIGEST,
        tests_passed=gate(spec).tests_passed,
        commit=COMMIT,
        dirty=True,
        engine_version="29.0.0",
        host_memory_total_bytes=32 * 2**30,
        user="1000:1000",
    )


def _serve(fake: FakeDocker, results: Path, spec: RunSpec, staging: Path) -> None:
    """Answer as an engine whose containers run to the end, with a driver that plays a worker
    kill: it writes the files of a synthetic one, but for the fault line the runner adds.

    The rules a test gives before these win over them.
    """
    played = worker_kill_run(worker_kill(), spec)
    write_run(staging, replace(played, events=played.events[:-1]))
    runner_files = {CONFIG_FILE, ENVIRONMENT_FILE, GATE_FILE, DONE_FILE}
    driver_files = {
        results / run_path(spec) / path.relative_to(staging): path.read_bytes()
        for path in staging.rglob("*")
        if path.is_file() and path.name not in runner_files
    }
    fake.answer(["bench.run"], files=driver_files)
    fake.answer(["--quiet", "system"], stdout=f"{GATEWAY_ID}\n")
    fake.answer(["{{.Id}}"], stdout=f"{ROBOTS_ID}\n")
    fake.answer(["logs", "--follow"], stdout="ready: listening on 0.0.0.0:8080\n", block=True)
    fake.answer([STATE_FORMAT], stdout="running 0 0 false\n")
    fake.answer(["{{.Image}}"], stdout=f"{IMAGE_DIGEST}\n")


def _helper_argvs(fake: FakeDocker) -> list[list[str]]:
    return [argv for argv in fake.argvs() if "bench.fault" in argv]


def _events(run_dir: Path) -> list[Event]:
    return [Event.from_json(line) for line in (run_dir / EVENTS_FILE).read_text().splitlines()]


@pytest.mark.parametrize("topology", list(Topology), ids=str)
def test_a_worker_kill_run_adds_the_helper_s_fault_event_and_finishes_with_its_summary(
    tmp_path: Path, fake_docker: FakeDocker, topology: Topology
) -> None:
    results = tmp_path / "results"
    spec = run_spec("worker_kill", topology=topology)
    fake_docker.answer(["bench.fault"], stdout=FAULT_LINE)
    _serve(fake_docker, results, spec, tmp_path / "staging")

    result = run_one(spec, worker_kill(), None, _invocation(results, spec))

    run_dir = results / result.path
    compose = run_compose(topology, None, run_overrides(spec), results, COMMIT)
    in_gateway = ("exec", "--no-TTY", "system", "python", "-m", "bench.fault")
    summary = json.loads((run_dir / SUMMARY_FILE).read_text())
    (restarted,) = (check for check in summary["checks"] if check["name"] == "worker_restarted")
    assert (result.outcome, result.detail) == (Outcome.FINISHED, "gated"), result.line()
    # Without --user: the helper runs as the image's user, in the gateway's container.
    assert _helper_argvs(fake_docker) == [
        [*compose.command(*in_gateway)[1:], "--run-dir", str(CONTAINER_RESULTS / result.path)]
    ]
    assert _events(run_dir)[-1] == FAULT
    assert [event.name for event in _events(run_dir)].count(EventName.FAULT) == 1
    assert summary_is_current(run_dir, Done.from_json((run_dir / DONE_FILE).read_text()))
    assert restarted["passed"] is True


@pytest.mark.parametrize(
    ("stderr", "exit_status", "detail"),
    [
        (
            "bench.fault: no t0 line in events.jsonl within 60 s\n",
            1,
            "the fault helper exited with 1; what it wrote:\n"
            "bench.fault: no t0 line in events.jsonl within 60 s",
        ),
        ("", 0, "the fault helper printed 0 lines, not one fault event; what it wrote:\n"),
    ],
    ids=["failed", "silent"],
)
def test_a_worker_kill_run_whose_helper_gives_no_fault_event_stays_without_done(
    tmp_path: Path, fake_docker: FakeDocker, stderr: str, exit_status: int, detail: str
) -> None:
    results = tmp_path / "results"
    spec = run_spec("worker_kill")
    fake_docker.answer(["bench.fault"], stderr=stderr, exit_status=exit_status)
    _serve(fake_docker, results, spec, tmp_path / "staging")

    result = run_one(spec, worker_kill(), None, _invocation(results, spec))

    assert (result.outcome, result.detail) == (Outcome.FAILED, detail)
    assert not (results / result.path / DONE_FILE).exists()


def test_a_helper_still_running_when_the_driver_fails_is_killed_and_the_driver_s_failure_kept(
    tmp_path: Path, fake_docker: FakeDocker
) -> None:
    results = tmp_path / "results"
    spec = run_spec("worker_kill")
    # Blocks until it is killed; a helper left running would hold the run open.
    fake_docker.answer(["bench.fault"], block=True)
    fake_docker.answer(["bench.run"], stdout="robot.fleet exited with 1\n", exit_status=1)
    _serve(fake_docker, results, spec, tmp_path / "staging")

    result = run_one(spec, worker_kill(), None, _invocation(results, spec))

    assert result.outcome is Outcome.FAILED
    assert result.detail.startswith("the driver exited with 1:\nrobot.fleet exited with 1\n")
    assert len(_helper_argvs(fake_docker)) == 1


def test_a_helper_that_has_not_exited_soon_after_the_driver_fails_the_run(
    tmp_path: Path, fake_docker: FakeDocker, monkeypatch: pytest.MonkeyPatch
) -> None:
    results = tmp_path / "results"
    spec = run_spec("worker_kill")
    monkeypatch.setattr(evidence, "FAULT_EXIT_TIMEOUT_S", 0.5)
    fake_docker.answer(["bench.fault"], stdout=FAULT_LINE, block=True)
    _serve(fake_docker, results, spec, tmp_path / "staging")

    result = run_one(spec, worker_kill(), None, _invocation(results, spec))

    assert (result.outcome, result.detail) == (
        Outcome.FAILED,
        "the fault helper did not exit within 0.5 s of the driver's end",
    )
    assert not (results / result.path / DONE_FILE).exists()
