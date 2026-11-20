"""The thin layer over the docker CLI that the evidence runner drives its containers with.

Each function runs one docker command with the timeout its caller gives. A command that
cannot start, exits non-zero where only success is expected, or outlives its timeout raises
DockerError, whose message keeps the command's standard error. What to run, and when, is
the runner's decision.
"""

from __future__ import annotations

import os
import selectors
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

DOCKER = "docker"
# Compose sets this label on every container of a project, one-off run containers included.
PROJECT_LABEL = "com.docker.compose.project"
# The runner names every container it starts by name with this prefix.
CONTAINER_PREFIX = "rri-"
_READ_BYTES = 65_536
_NS_PER_S = 1_000_000_000
# A killed process is reaped at once; this only bounds a wait that the kernel ends.
REAP_TIMEOUT_S = 5.0
# The lines of standard error a DockerError quotes: the end holds the reason.
_ERROR_LINES = 20
# What docker container inspect prints of a container's state, on one line.
STATE_FORMAT = "{{.State.Status}} {{.State.ExitCode}} {{.RestartCount}} {{.State.OOMKilled}}"
_BOOLEANS = {"true": True, "false": False}


class DockerError(Exception):
    """A docker command could not start, failed or did not finish within its timeout."""


@dataclass(frozen=True)
class Completed:
    """A command that ran to its end in the foreground: its exit status and its output."""

    exit_status: int
    # Standard output and standard error, interleaved as the command wrote them.
    output: bytes


@dataclass(frozen=True)
class ContainerState:
    """Whether a container runs, and how it ended or restarted, as the engine reports it."""

    # created, running, paused, restarting, removing, exited or dead
    status: str
    exit_code: int
    restart_count: int
    oom_killed: bool


@dataclass(frozen=True)
class Compose:
    """One compose project as the runner drives it: its name, its files and its variables."""

    project: str
    # The topology's own file first; a further file is laid over it.
    files: tuple[Path, ...]
    # Profiles whose services a command may name.
    profiles: tuple[str, ...]
    # The variables the compose files interpolate, added to the runner's own environment.
    variables: Mapping[str, str]

    def command(self, *arguments: str) -> list[str]:
        """Return a docker compose command line for this project."""
        files = [part for path in self.files for part in ("-f", str(path))]
        profiles = [part for profile in self.profiles for part in ("--profile", profile)]
        return [DOCKER, "compose", "--project-name", self.project, *files, *profiles, *arguments]

    def environment(self) -> dict[str, str]:
        return {**os.environ, **self.variables}


def _tail(data: bytes | None) -> str:
    text = (data or b"").decode("utf-8", errors="replace").strip()
    return "\n".join(text.splitlines()[-_ERROR_LINES:])


def _quoted(command: Sequence[str]) -> str:
    return " ".join(command)


def _run(
    command: Sequence[str], timeout_s: float, environment: Mapping[str, str] | None = None
) -> bytes:
    """Run a command that must succeed, and return its standard output."""
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=timeout_s,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DockerError(
            f"{_quoted(command)} did not finish within {timeout_s:g} s: {_tail(error.stderr)}"
        ) from error
    except OSError as error:
        raise DockerError(f"cannot run {_quoted(command)}: {error}") from error
    if completed.returncode != 0:
        raise DockerError(
            f"{_quoted(command)} exited with {completed.returncode}: {_tail(completed.stderr)}"
        )
    return completed.stdout


def _lines(output: bytes) -> list[str]:
    return [line.strip() for line in output.decode("utf-8").splitlines() if line.strip()]


def _single(output: bytes, what: str) -> str:
    lines = _lines(output)
    if len(lines) != 1:
        raise DockerError(f"expected one line for {what}, got {len(lines)}: {lines}")
    return lines[0]


def run_foreground(
    command: Sequence[str], timeout_s: float, environment: Mapping[str, str] | None = None
) -> Completed:
    """Run a command whose exit status the caller judges, and return it with the output.

    DockerError only when the command cannot start or outlives its timeout; it is then killed.
    """
    try:
        completed = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            env=environment,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise DockerError(
            f"{_quoted(command)} did not finish within {timeout_s:g} s: {_tail(error.stdout)}"
        ) from error
    except OSError as error:
        raise DockerError(f"cannot run {_quoted(command)}: {error}") from error
    return Completed(completed.returncode, completed.stdout)


def image_digest(image: str, timeout_s: float) -> str:
    """Return the digest of a local image, as an image's id: "sha256:" and 64 hex digits."""
    return _single(
        _run([DOCKER, "image", "inspect", "--format", "{{.Id}}", image], timeout_s), image
    )


def build(compose: Compose, timeout_s: float) -> str:
    """Build the image of the compose files and return its digest.

    The services of the files share one image, so the build and the digest are of that image.
    """
    environment = compose.environment()
    _run(compose.command("build"), timeout_s, environment)
    images = set(_lines(_run(compose.command("config", "--images"), timeout_s, environment)))
    if len(images) != 1:
        raise DockerError(f"the compose files name {len(images)} images, not one: {sorted(images)}")
    return image_digest(images.pop(), timeout_s)


def project_containers(project: str, timeout_s: float) -> list[str]:
    """Return the ids of every container of the project, and of every container the runner named.

    Stopped containers are included: any of them would be removed before a run.
    """
    by_label = _run(
        [DOCKER, "ps", "--all", "--quiet", "--filter", f"label={PROJECT_LABEL}={project}"],
        timeout_s,
    )
    by_name = _run(
        [DOCKER, "ps", "--all", "--quiet", "--filter", f"name=^{CONTAINER_PREFIX}"], timeout_s
    )
    return list(dict.fromkeys([*_lines(by_label), *_lines(by_name)]))


def up(compose: Compose, service: str, timeout_s: float) -> None:
    """Start the service in a new container, detached, whatever ran before."""
    _run(
        compose.command("up", "--detach", "--force-recreate", service),
        timeout_s,
        compose.environment(),
    )


def down(compose: Compose, timeout_s: float) -> None:
    """Stop and remove the project's containers, those of services no file names included."""
    _run(compose.command("down", "--remove-orphans"), timeout_s, compose.environment())


def remove(containers: Sequence[str], timeout_s: float) -> None:
    """Remove containers, running or not; one that no longer exists is not an error."""
    if containers:
        _run([DOCKER, "rm", "--force", *containers], timeout_s)


def run_detached(
    compose: Compose, service: str, name: str, command: Sequence[str], timeout_s: float
) -> str:
    """Start a one-off container of the service under a name, detached; return its id."""
    _run(
        compose.command("run", "--detach", "--name", name, service, *command),
        timeout_s,
        compose.environment(),
    )
    return container_id(name, timeout_s)


def run_named(
    compose: Compose,
    service: str,
    name: str,
    command: Sequence[str],
    user: str | None,
    timeout_s: float,
) -> Completed:
    """Run a one-off container of the service in the foreground, as user (uid:gid) if given.

    The container is kept when it exits, so that it can be inspected; the caller removes it.
    """
    as_user = [] if user is None else ["--user", user]
    return run_foreground(
        compose.command("run", "--no-TTY", *as_user, "--name", name, service, *command),
        timeout_s,
        compose.environment(),
    )


def exec_in_service(
    compose: Compose, service: str, user: str, command: Sequence[str], timeout_s: float
) -> Completed:
    """Run a command in the service's running container, as user (uid:gid), in the foreground."""
    return run_foreground(
        compose.command("exec", "--no-TTY", "--user", user, service, *command),
        timeout_s,
        compose.environment(),
    )


def start_in_service(
    compose: Compose, service: str, command: Sequence[str]
) -> subprocess.Popen[bytes]:
    """Start a command in the service's running container as the image's user, in the background.

    Its standard input is closed, and its standard output and standard error are piped apart.
    The caller waits for it or kills it. DockerError when it cannot start.
    """
    argv = compose.command("exec", "--no-TTY", service, *command)
    try:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=compose.environment(),
        )
    except OSError as error:
        raise DockerError(f"cannot run {_quoted(argv)}: {error}") from error


def exec_in_container(
    container: str, user: str, command: Sequence[str], timeout_s: float
) -> Completed:
    """Run a command in a running container, as user (uid:gid), in the foreground."""
    return run_foreground(
        [DOCKER, "exec", "--interactive", "--user", user, container, *command], timeout_s
    )


def service_container(compose: Compose, service: str, timeout_s: float) -> str:
    """Return the id of the service's one container."""
    return _single(
        _run(compose.command("ps", "--all", "--quiet", service), timeout_s, compose.environment()),
        f"the container of {service}",
    )


def container_id(container: str, timeout_s: float) -> str:
    """Return the full id of a container named by its name or by a prefix of its id."""
    return _single(
        _run([DOCKER, "container", "inspect", "--format", "{{.Id}}", container], timeout_s),
        container,
    )


def image_of(container: str, timeout_s: float) -> str:
    """Return the digest of the image a container was created from."""
    return _single(
        _run([DOCKER, "container", "inspect", "--format", "{{.Image}}", container], timeout_s),
        container,
    )


def container_state(container: str, timeout_s: float) -> ContainerState:
    """Return the state of a container, running or not."""
    line = _single(
        _run([DOCKER, "container", "inspect", "--format", STATE_FORMAT, container], timeout_s),
        container,
    )
    try:
        status, exit_code, restart_count, oom_killed = line.split()
        return ContainerState(status, int(exit_code), int(restart_count), _BOOLEANS[oom_killed])
    except (ValueError, KeyError) as error:
        raise DockerError(
            f"docker gave a state of {container} that cannot be read: {line}"
        ) from error


def log_tail(container: str, lines: int, timeout_s: float) -> str:
    """Return the last lines of a container's log, both of its streams."""
    completed = run_foreground([DOCKER, "logs", "--tail", str(lines), container], timeout_s)
    return completed.output.decode("utf-8", errors="replace")


def engine_version(timeout_s: float) -> str:
    """Return the version of the docker engine."""
    return _single(
        _run([DOCKER, "version", "--format", "{{.Server.Version}}"], timeout_s), "the version"
    )


def host_memory_total_bytes(timeout_s: float) -> int:
    """Return the memory of the host the engine runs its containers on."""
    text = _single(_run([DOCKER, "info", "--format", "{{.MemTotal}}"], timeout_s), "MemTotal")
    try:
        return int(text)
    except ValueError as error:
        raise DockerError(
            f"docker info gave a MemTotal that is not a byte count: {text}"
        ) from error


def wait_for_line(container: str, prefix: str, timeout_s: float) -> str:
    """Follow a container's log until a line that starts with prefix, and return that line.

    DockerError when the log ends first, because the container stopped, or when the time
    runs out.
    """
    command = [DOCKER, "logs", "--follow", container]
    deadline_ns = time.monotonic_ns() + round(timeout_s * _NS_PER_S)
    try:
        process = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except OSError as error:
        raise DockerError(f"cannot run {_quoted(command)}: {error}") from error
    # Leaving the block closes the pipe.
    with process:
        try:
            # stdout=PIPE gives the process a stdout.
            assert process.stdout is not None
            return _read_until(process.stdout.fileno(), prefix, deadline_ns, container, timeout_s)
        finally:
            process.kill()
            process.wait(timeout=REAP_TIMEOUT_S)


def _read_until(
    descriptor: int, prefix: str, deadline_ns: int, container: str, timeout_s: float
) -> str:
    """Read a container's log from a pipe until a line that starts with prefix; return it."""
    pending = b""
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining_s = (deadline_ns - time.monotonic_ns()) / _NS_PER_S
            if remaining_s <= 0 or not selector.select(remaining_s):
                raise DockerError(
                    f"no line starting with {prefix!r} in the log of {container} "
                    f"within {timeout_s:g} s"
                )
            chunk = os.read(descriptor, _READ_BYTES)
            if not chunk:
                raise DockerError(
                    f"the log of {container} ended without a line starting with {prefix!r}: "
                    "the container stopped"
                )
            *lines, pending = (pending + chunk).split(b"\n")
            for line in lines:
                text = line.decode("utf-8", errors="replace").rstrip("\r")
                if text.startswith(prefix):
                    return text
