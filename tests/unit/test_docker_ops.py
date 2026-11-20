"""Tests of the layer over the docker CLI, against a fake docker program on the PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from bench import docker_ops
from bench.docker_ops import STATE_FORMAT, Completed, Compose, ContainerState, DockerError
from tests.support.fake_docker import FakeDocker, install_fake_docker

COMPOSE = Compose(
    project="rri-test",
    files=(Path("/repo/docker-compose.split.yml"), Path("/elsewhere/one_cpu.yml")),
    profiles=("robots",),
    variables={"RRI_OVERRIDES": '{"skip_rule": false}', "RRI_RESULTS": "/results"},
)
PREFIX = [
    "compose",
    "--project-name",
    "rri-test",
    "-f",
    "/repo/docker-compose.split.yml",
    "-f",
    "/elsewhere/one_cpu.yml",
    "--profile",
    "robots",
]
# Long enough for the fake to start, and short enough that a test that waits for it is quick.
SHORT_TIMEOUT_S = 0.5


@pytest.fixture
def fake_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    return install_fake_docker(tmp_path / "bin", monkeypatch)


def test_a_compose_command_names_the_project_its_files_and_its_profiles() -> None:
    assert COMPOSE.command("up", "system") == ["docker", *PREFIX, "up", "system"]


def test_the_build_returns_the_digest_of_the_one_image_the_files_name(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["config", "--images"], stdout="rri:dev\nrri:dev\n")
    fake_docker.answer(["image", "inspect"], stdout="sha256:" + "ab" * 32 + "\n")

    digest = docker_ops.build(COMPOSE, SHORT_TIMEOUT_S * 10)

    assert digest == "sha256:" + "ab" * 32
    assert fake_docker.argvs() == [
        [*PREFIX, "build"],
        [*PREFIX, "config", "--images"],
        ["image", "inspect", "--format", "{{.Id}}", "rri:dev"],
    ]
    assert fake_docker.calls()[0].variables["RRI_RESULTS"] == "/results"


def test_the_build_refuses_files_that_name_more_than_one_image(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["config", "--images"], stdout="rri:dev\nother:dev\n")

    with pytest.raises(DockerError, match="name 2 images"):
        docker_ops.build(COMPOSE, SHORT_TIMEOUT_S * 10)


def test_a_command_that_fails_raises_with_its_standard_error(fake_docker: FakeDocker) -> None:
    fake_docker.answer(
        ["up"], stderr="Bind for 0.0.0.0:8080 failed: port is already allocated\n", exit_status=1
    )

    with pytest.raises(DockerError, match=r"exited with 1: Bind for .* port is already allocated"):
        docker_ops.up(COMPOSE, "system", SHORT_TIMEOUT_S * 10)


def test_a_command_that_outlives_its_timeout_raises_with_what_it_wrote(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["down"], stderr="Stopping system\n", block=True)

    with pytest.raises(DockerError, match=r"did not finish within 0\.5 s: Stopping system"):
        docker_ops.down(COMPOSE, SHORT_TIMEOUT_S)


def test_a_foreground_command_that_outlives_its_timeout_raises(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["exec"], stdout="t0 written\n", block=True)

    with pytest.raises(DockerError, match=r"did not finish within 0\.5 s: t0 written"):
        docker_ops.exec_in_container("rri-robots-x", "1000:1000", ["sleep"], SHORT_TIMEOUT_S)


def test_up_and_down_name_the_service_and_remove_the_orphans(fake_docker: FakeDocker) -> None:
    docker_ops.up(COMPOSE, "system", SHORT_TIMEOUT_S * 10)
    docker_ops.down(COMPOSE, SHORT_TIMEOUT_S * 10)

    assert fake_docker.argvs() == [
        [*PREFIX, "up", "--detach", "--force-recreate", "system"],
        [*PREFIX, "down", "--remove-orphans"],
    ]
    assert fake_docker.calls()[0].variables["RRI_OVERRIDES"] == '{"skip_rule": false}'


def test_the_project_containers_are_those_of_its_label_and_of_the_runner_s_names(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["label=com.docker.compose.project=rri-test"], stdout="aaa\nbbb\n")
    fake_docker.answer(["name=^rri-"], stdout="bbb\nccc\n")

    assert docker_ops.project_containers("rri-test", SHORT_TIMEOUT_S * 10) == ["aaa", "bbb", "ccc"]


@pytest.mark.parametrize(
    ("containers", "expected"),
    [([], []), (["aaa", "bbb"], [["rm", "--force", "aaa", "bbb"]])],
    ids=["none", "two"],
)
def test_remove_forces_the_containers_out_and_runs_nothing_for_none(
    fake_docker: FakeDocker, containers: list[str], expected: list[list[str]]
) -> None:
    docker_ops.remove(containers, SHORT_TIMEOUT_S * 10)

    assert fake_docker.argvs() == expected


@pytest.mark.parametrize(
    ("user", "user_flags"),
    [("1000:1000", ["--user", "1000:1000"]), (None, [])],
    ids=["user", "image's user"],
)
def test_a_named_run_runs_in_the_foreground_and_returns_its_status_and_output(
    fake_docker: FakeDocker, user: str | None, user_flags: list[str]
) -> None:
    fake_docker.answer(
        ["run"], stdout="1 failed, 3 passed in 1.00s\n", stderr="warn\n", exit_status=1
    )

    completed = docker_ops.run_named(
        COMPOSE, "system", "rri-tests-x", ["pytest", "-q"], user, SHORT_TIMEOUT_S * 10
    )

    assert completed == Completed(1, b"1 failed, 3 passed in 1.00s\nwarn\n")
    assert fake_docker.argvs() == [
        [*PREFIX, "run", "--no-TTY", *user_flags, "--name", "rri-tests-x", "system", "pytest", "-q"]
    ]


def test_a_detached_run_returns_the_id_of_its_container(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["container", "inspect"], stdout="f" * 64 + "\n")

    container = docker_ops.run_detached(
        COMPOSE, "robots", "rri-robots-x", ["sleep", "infinity"], SHORT_TIMEOUT_S * 10
    )

    assert container == "f" * 64
    assert fake_docker.argvs() == [
        [*PREFIX, "run", "--detach", "--name", "rri-robots-x", "robots", "sleep", "infinity"],
        ["container", "inspect", "--format", "{{.Id}}", "rri-robots-x"],
    ]


def test_the_two_execs_run_as_the_user_in_the_service_and_in_the_container(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["exec"], stdout="done\n")

    in_service = docker_ops.exec_in_service(
        COMPOSE, "system", "1000:1000", ["python", "-m", "bench.run"], SHORT_TIMEOUT_S * 10
    )
    in_container = docker_ops.exec_in_container(
        "rri-robots-x", "1000:1000", ["python", "-m", "bench.run"], SHORT_TIMEOUT_S * 10
    )

    assert in_service == in_container == Completed(0, b"done\n")
    assert fake_docker.argvs() == [
        [*PREFIX, "exec", "--no-TTY", "--user", "1000:1000", "system", "python", "-m", "bench.run"],
        [
            "exec",
            "--interactive",
            "--user",
            "1000:1000",
            "rri-robots-x",
            "python",
            "-m",
            "bench.run",
        ],
    ]


def test_a_command_started_in_a_service_runs_as_the_image_s_user_with_its_streams_apart(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["exec"], stdout='{"event":"fault"}\n', stderr="killed\n", exit_status=3)

    process = docker_ops.start_in_service(COMPOSE, "system", ["python", "-m", "bench.fault"])
    with process:
        stdout, stderr = process.communicate(timeout=SHORT_TIMEOUT_S * 10)

    assert (process.returncode, stdout, stderr) == (3, b'{"event":"fault"}\n', b"killed\n")
    assert fake_docker.argvs() == [
        [*PREFIX, "exec", "--no-TTY", "system", "python", "-m", "bench.fault"]
    ]
    assert fake_docker.calls()[0].variables["RRI_OVERRIDES"] == '{"skip_rule": false}'


def test_a_command_that_cannot_start_in_a_service_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(DockerError, match="cannot run docker compose"):
        docker_ops.start_in_service(COMPOSE, "system", ["python", "-m", "bench.fault"])


@pytest.mark.parametrize("stdout", ["", "aaa\nbbb\n"], ids=["none", "two"])
def test_a_service_without_exactly_one_container_is_an_error(
    fake_docker: FakeDocker, stdout: str
) -> None:
    fake_docker.answer(["ps"], stdout=stdout)

    with pytest.raises(DockerError, match="expected one line for the container of system"):
        docker_ops.service_container(COMPOSE, "system", SHORT_TIMEOUT_S * 10)


def test_the_image_of_a_container_is_read_by_inspecting_it(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["{{.Image}}"], stdout="sha256:" + "cd" * 32 + "\n")

    assert docker_ops.image_of("aaa", SHORT_TIMEOUT_S * 10) == "sha256:" + "cd" * 32
    assert fake_docker.argvs() == [["container", "inspect", "--format", "{{.Image}}", "aaa"]]


def test_the_state_of_a_container_is_read_by_inspecting_it(fake_docker: FakeDocker) -> None:
    fake_docker.answer([STATE_FORMAT], stdout="exited 137 2 true\n")

    assert docker_ops.container_state("aaa", SHORT_TIMEOUT_S * 10) == ContainerState(
        status="exited", exit_code=137, restart_count=2, oom_killed=True
    )
    assert fake_docker.argvs() == [["container", "inspect", "--format", STATE_FORMAT, "aaa"]]


@pytest.mark.parametrize(
    "stdout",
    ["running 0 0\n", "running zero 0 false\n", "running 0 0 maybe\n"],
    ids=["a field short", "an exit code that is no number", "a flag that is no boolean"],
)
def test_a_state_that_cannot_be_read_is_an_error(fake_docker: FakeDocker, stdout: str) -> None:
    fake_docker.answer([STATE_FORMAT], stdout=stdout)

    with pytest.raises(DockerError, match="a state of aaa that cannot be read"):
        docker_ops.container_state("aaa", SHORT_TIMEOUT_S * 10)


def test_the_host_memory_is_the_engine_s_byte_count(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["info"], stdout="33520001024\n")

    assert docker_ops.host_memory_total_bytes(SHORT_TIMEOUT_S * 10) == 33_520_001_024
    assert fake_docker.argvs() == [["info", "--format", "{{.MemTotal}}"]]


def test_a_host_memory_that_is_not_a_byte_count_is_an_error(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["info"], stdout="32GiB\n")

    with pytest.raises(DockerError, match="not a byte count: 32GiB"):
        docker_ops.host_memory_total_bytes(SHORT_TIMEOUT_S * 10)


def test_the_engine_version_is_the_server_s(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["version"], stdout="29.10.0\n")

    assert docker_ops.engine_version(SHORT_TIMEOUT_S * 10) == "29.10.0"
    assert fake_docker.argvs() == [["version", "--format", "{{.Server.Version}}"]]


def test_the_log_tail_holds_both_streams(fake_docker: FakeDocker) -> None:
    fake_docker.answer(["logs", "--tail"], stdout="listening on 0.0.0.0:8080\n", stderr="boom\n")

    assert (
        docker_ops.log_tail("aaa", 30, SHORT_TIMEOUT_S * 10) == "listening on 0.0.0.0:8080\nboom\n"
    )
    assert fake_docker.argvs() == [["logs", "--tail", "30", "aaa"]]


def test_waiting_for_a_line_returns_the_first_that_starts_with_the_prefix(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(
        ["logs", "--follow"],
        stdout=(
            "listening on 0.0.0.0:8080\nready: listening on 0.0.0.0:8080; cls cap 8\n  metrics\n"
        ),
        block=True,
    )

    line = docker_ops.wait_for_line("aaa", "ready:", SHORT_TIMEOUT_S * 10)

    assert line == "ready: listening on 0.0.0.0:8080; cls cap 8"
    assert fake_docker.argvs() == [["logs", "--follow", "aaa"]]


def test_waiting_for_a_line_of_a_container_that_stops_first_is_an_error(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["logs", "--follow"], stdout="gateway: cannot listen\n", exit_status=2)

    with pytest.raises(DockerError, match="ended without a line starting with 'ready:'"):
        docker_ops.wait_for_line("aaa", "ready:", SHORT_TIMEOUT_S * 10)


def test_waiting_for_a_line_that_never_comes_is_an_error_after_the_timeout(
    fake_docker: FakeDocker,
) -> None:
    fake_docker.answer(["logs", "--follow"], stdout="listening on 0.0.0.0:8080\n", block=True)

    with pytest.raises(DockerError, match=r"no line starting with 'ready:' .* within 0\.5 s"):
        docker_ops.wait_for_line("aaa", "ready:", SHORT_TIMEOUT_S)


def test_a_docker_program_that_cannot_start_is_an_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(DockerError, match="cannot run docker version"):
        docker_ops.engine_version(SHORT_TIMEOUT_S * 10)
