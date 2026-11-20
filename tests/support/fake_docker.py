"""A fake docker program on the PATH, for the tests of what drives the docker CLI.

The fake is a real executable: it records each command line and the variables the compose
files read, then does what the first matching rule says, as docker would: it writes the
files a container would write through its mounts, sends a signal to its caller, prints, exits
with a status or blocks until it is killed.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest

# -I and -S: no site packages and no environment, only the standard library, so that each
# command starts quickly.
_PROGRAM = """#!{python} -IS
import base64, json, os, select, sys

argv = sys.argv[1:]
names = ("RRI_OVERRIDES", "RRI_RESULTS", "GIT_COMMIT")
variables = {{name: os.environ.get(name) for name in names}}
log_path = os.environ["FAKE_DOCKER_LOG"]
with open(log_path, "a") as log:
    log.write(json.dumps({{"argv": argv, "variables": variables}}) + "\\n")
with open(os.environ["FAKE_DOCKER_RULES"]) as file:
    rules = json.load(file)


def holds(call, match):
    return all(token in call for token in match)


def occurrence(match):
    with open(log_path) as log:
        return sum(holds(json.loads(line)["argv"], match) for line in log)


for rule in rules:
    if not holds(argv, rule["match"]):
        continue
    if rule["occurrence"] is not None and occurrence(rule["match"]) != rule["occurrence"]:
        continue
    for path, content in rule["files"].items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as file:
            file.write(base64.b64decode(content))
    if rule["signal"] is not None:
        os.kill(os.getppid(), rule["signal"])
    sys.stdout.write(rule["stdout"])
    sys.stdout.flush()
    sys.stderr.write(rule["stderr"])
    sys.stderr.flush()
    if rule["block"]:
        # Blocks until the caller kills it, as docker logs --follow does.
        select.select([], [], [])
    sys.exit(rule["exit"])
"""


@dataclass(frozen=True)
class Call:
    """One command line the fake was given, without the program's name."""

    argv: list[str]
    variables: dict[str, str | None]


class FakeDocker:
    """The fake docker program and what it was asked."""

    def __init__(self, directory: Path) -> None:
        self._log = directory / "calls.jsonl"
        self._rules_path = directory / "rules.json"
        self._rules: list[dict[str, object]] = []
        self._rules_path.write_text("[]")

    def answer(
        self,
        match: Sequence[str],
        *,
        stdout: str = "",
        stderr: str = "",
        exit_status: int = 0,
        block: bool = False,
        occurrence: int | None = None,
        files: Mapping[Path, bytes] | None = None,
        signal_number: int | None = None,
    ) -> None:
        """Answer every command line that holds all of match's tokens; earlier rules win.

        With an occurrence, the rule answers only the command line that is the occurrence-th
        (from 1) to hold them. Before it prints, the command writes files, at their absolute
        paths, and sends signal_number to the process that ran it. A command line that no
        rule answers exits 0 with no output.
        """
        self._rules.append(
            {
                "match": list(match),
                "stdout": stdout,
                "stderr": stderr,
                "exit": exit_status,
                "block": block,
                "occurrence": occurrence,
                "files": {
                    str(path): base64.b64encode(content).decode("ascii")
                    for path, content in (files or {}).items()
                },
                "signal": signal_number,
            }
        )
        self._rules_path.write_text(json.dumps(self._rules))

    def calls(self) -> list[Call]:
        if not self._log.exists():
            return []
        return [Call(**json.loads(line)) for line in self._log.read_text().splitlines()]

    def argvs(self) -> list[list[str]]:
        return [call.argv for call in self.calls()]


def install_fake_docker(directory: Path, monkeypatch: pytest.MonkeyPatch) -> FakeDocker:
    """Put a fake docker program in directory, first on the PATH; return it."""
    directory.mkdir()
    program = directory / "docker"
    program.write_text(_PROGRAM.format(python=sys.executable))
    program.chmod(0o755)
    monkeypatch.setenv("PATH", f"{directory}{os.pathsep}{os.environ.get('PATH', '')}")
    monkeypatch.setenv("FAKE_DOCKER_LOG", str(directory / "calls.jsonl"))
    monkeypatch.setenv("FAKE_DOCKER_RULES", str(directory / "rules.json"))
    return FakeDocker(directory)
