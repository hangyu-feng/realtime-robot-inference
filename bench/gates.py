"""The gates of a run (DESIGN.md 11.3): what each one compares, and the record of them all.

A run counts only when every gate that applies to it passed. The comparisons are pure: they
take what was read and return the gate's part of gate.json. The readers at the end open the
files those readings come from: a run's scrapes, the model benchmark's reports, and the
stored result of the test suite.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import zlib
from collections.abc import Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.runfiles import (
    SCRAPES_FILE,
    ConfigGate,
    FreshGate,
    Gate,
    GatedLimits,
    ImageGate,
    LimitsGate,
    RunFileError,
    Scrape,
    SuiteGate,
)

# ASSUMPTION: A19. The CPU limit is a quota of 2 CPUs of time per period.
# ASSUMPTION: A24. The memory limit of 4 GB is read as 4 GiB of memory with no swap on top.
EXPECTED_LIMITS = GatedLimits(
    cpu_limit_cores=2.0, memory_limit_bytes=4 * 2**30, memory_swap_limit_bytes=0
)

_GATES_DIRECTORY = "gates"
_REPORTS_DIRECTORY = "reports"
_DIGEST_PREFIX = "sha256:"

# The last line of pytest -q: counts separated by commas, then the duration, which is also
# given as h:mm:ss from a minute on. A count's name is one word or two ("subtests passed").
_COUNT = r"\d+ [a-z]+(?: [a-z]+)?"
_COUNT_LINE = re.compile(
    rf"(?P<counts>{_COUNT}(?:, {_COUNT})*) in \d+\.\d+s(?: \(\d+:\d\d:\d\d\))?"
)
# pytest's name for each count, in the singular and in the plural where it has one.
_COUNT_FIELDS = {
    "passed": "passed",
    "failed": "failed",
    "error": "errors",
    "errors": "errors",
    "skipped": "skipped",
    "xfailed": "xfailed",
    "xpassed": "xpassed",
    "deselected": "deselected",
}
# Counted by pytest, and no part of the gate.
_IGNORED_COUNTS = ("warning", "warnings")


class GateError(Exception):
    """What a gate is computed from cannot be read: test output with no count line, a file."""


@dataclass(frozen=True)
class PytestReport:
    """The counts of the final line of a pytest run."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    xfailed: int = 0
    xpassed: int = 0
    deselected: int = 0


def parse_pytest_report(text: str) -> PytestReport:
    """Read the counts of the final count line of pytest -q's output.

    GateError for output without such a line, and for a count this gate does not know: an
    outcome it cannot judge must not pass it.
    """
    matches = [
        match for line in text.splitlines() if (match := _COUNT_LINE.fullmatch(line.strip()))
    ]
    if not matches:
        raise GateError("the test output has no count line")
    counts: dict[str, int] = {}
    for part in matches[-1]["counts"].split(", "):
        number, name = part.split(" ", 1)
        if name in _IGNORED_COUNTS:
            continue
        if name not in _COUNT_FIELDS:
            raise GateError(f"the count line of the test output has an unknown count: {part}")
        counts[_COUNT_FIELDS[name]] = int(number)
    return PytestReport(**counts)


def tests_gate(report: PytestReport, exit_status: int) -> bool:
    """Whether the test suite passed: tests passed, and none failed or was left out.

    A deselected test does not count against the gate: the tests that need the host's
    docker CLI are not selected inside the image. A skipped test does, whatever its
    reason, and so does an expected failure: the evidence rests on every selected test.
    """
    return (
        exit_status == 0
        and report.passed > 0
        and report.failed == 0
        and report.errors == 0
        and report.skipped == 0
        and report.xfailed == 0
        and report.xpassed == 0
    )


def suite_gate_file(image_digest: str, code_digest: str) -> str:
    """Return the path, under the results root, of the tests gate of an image and a code digest.

    The suite's output is beside it, under the same name with the suffix .txt.
    """
    image = image_digest.removeprefix(_DIGEST_PREFIX)
    return f"{_GATES_DIRECTORY}/{image[:12]}-{code_digest[:8]}.json"


def suite_gate_record(
    output: bytes, exit_status: int, image_digest: str, code_digest: str, commit: str, dirty: bool
) -> SuiteGate:
    """Return the tests gate of one run of the suite, from its output as stored.

    The record keeps four of the count line's numbers. The errors, the expected failures and
    the unexpected passes, which fail the gate as well, are in the output, whose hash the
    record holds. GateError for output without a count line.
    """
    report = parse_pytest_report(output.decode("utf-8", errors="replace"))
    return SuiteGate(
        passed=tests_gate(report, exit_status),
        image_digest=image_digest,
        code_digest=code_digest,
        commit=commit,
        dirty=dirty,
        passed_count=report.passed,
        failed_count=report.failed,
        skipped_count=report.skipped,
        deselected_count=report.deselected,
        gate_file=suite_gate_file(image_digest, code_digest),
        output_sha256=hashlib.sha256(output).hexdigest(),
    )


def _cores(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        return float(value)
    except OverflowError:
        # An integer beyond the float range is no CPU limit.
        return None


def _byte_count(value: object) -> int | None:
    # A bool is an int to isinstance, and true is not a byte count.
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def limits_ok(read: Mapping[str, object] | None) -> LimitsGate:
    """Compare the limits read inside a container with the expected ones, exactly.

    read is a cgroup object as /metrics and a benchmark report hold it; None when there is
    none. A limit that is absent, null or not a number of its kind counts as not read, and
    fails the gate: cgroup v1 reports no swap limit, so a run on it is not gated.
    """
    cgroup = {} if read is None else read
    limits = GatedLimits(
        cpu_limit_cores=_cores(cgroup.get("cpu_limit_cores")),
        memory_limit_bytes=_byte_count(cgroup.get("memory_limit_bytes")),
        memory_swap_limit_bytes=_byte_count(cgroup.get("memory_swap_limit_bytes")),
    )
    return LimitsGate(ok=limits == EXPECTED_LIMITS, expected=EXPECTED_LIMITS, read=limits)


def _all_new(container_ids: Collection[str], earlier_ids: Collection[str]) -> bool:
    # docker prints an id in full or cut to its first digits, so two ids name one container
    # when one starts with the other.
    return len(container_ids) > 0 and not any(
        new.startswith(earlier) or earlier.startswith(new)
        for new in container_ids
        for earlier in earlier_ids
    )


def fresh_ok(
    container_ids: Collection[str], earlier_ids: Collection[str], accepted_before_t0: int | None
) -> FreshGate:
    """Whether a serving run started fresh.

    Every container of the run is new: none of their ids was among the project's containers
    listed before the run. And the gateway had accepted no connection at the scrape before
    time zero; None, for a scrape that failed, fails the gate.
    """
    new_containers = _all_new(container_ids, earlier_ids)
    return FreshGate(
        ok=new_containers and accepted_before_t0 == 0,
        new_containers=new_containers,
        accepted_before_t0=accepted_before_t0,
    )


def _differences(
    expected: Mapping[str, Any], read: Mapping[str, Any], prefix: str
) -> Iterator[str]:
    """Yield the dotted path of every value that differs, or that only one side has."""
    for key in sorted(expected.keys() | read.keys()):
        path = f"{prefix}{key}"
        if key not in expected or key not in read:
            yield path
        elif isinstance(expected[key], Mapping) and isinstance(read[key], Mapping):
            yield from _differences(expected[key], read[key], f"{path}.")
        elif expected[key] != read[key]:
            yield path


def config_ok(
    spec_config: Mapping[str, Any], scraped_config: Mapping[str, Any] | None
) -> ConfigGate:
    """Whether the config a gateway echoed is the resolved server config of the run's spec.

    The echo's version is the commit, which is no part of a spec, so it is left out. Values
    are compared with ==, so 2 equals 2.0. An override that did not reach the gateway and
    a topology that differs both fail here. None, for a run with no successful scrape,
    fails with no path to name.
    """
    if scraped_config is None:
        return ConfigGate(ok=False, differences=())
    echoed = {**scraped_config, "version": ""} if "version" in scraped_config else scraped_config
    differences = tuple(_differences(spec_config, echoed, ""))
    return ConfigGate(ok=not differences, differences=differences)


def _image_gate(image_digest: str, container_images: Mapping[str, str]) -> ImageGate:
    """Whether every container of a run has the image that this invocation built."""
    return ImageGate(
        built=len(container_images) > 0
        and all(digest == image_digest for digest in container_images.values()),
        digest=image_digest,
        containers=dict(container_images),
    )


def _entry(document: object, *keys: str) -> object:
    """Return the value at the keys of nested JSON objects; None where there is none."""
    for key in keys:
        if not isinstance(document, Mapping) or key not in document:
            return None
        document = document[key]
    return document


def gate_record(
    tests_passed: SuiteGate,
    image_digest: str,
    container_images: Mapping[str, str],
    container_ids: Collection[str],
    earlier_ids: Collection[str],
    spec_config: Mapping[str, Any],
    scrapes: Sequence[Scrape],
) -> Gate:
    """Return the gates of a serving run.

    image_digest is that of the image built in this invocation, and container_images maps
    each container of the run to the digest of its image. The limits and the config are
    those of the first scrape that succeeded, and the connections accepted before time
    zero those of the scrape one second before it.
    """
    first = next((scrape.metrics for scrape in scrapes if scrape.metrics is not None), None)
    before_t0 = next((scrape.metrics for scrape in scrapes if scrape.k == -1), None)
    cgroup = _entry(first, "cgroup")
    config = _entry(first, "config")
    accepted = _entry(before_t0, "connections", "accepted")
    return Gate(
        tests_passed=tests_passed,
        image_built=_image_gate(image_digest, container_images),
        limits_ok=limits_ok(cgroup if isinstance(cgroup, Mapping) else None),
        fresh=fresh_ok(
            container_ids,
            earlier_ids,
            accepted if isinstance(accepted, int) and not isinstance(accepted, bool) else None,
        ),
        config_ok=config_ok(spec_config, config if isinstance(config, Mapping) else None),
    )


def model_bench_gate_record(
    tests_passed: SuiteGate,
    image_digest: str,
    container_images: Mapping[str, str],
    container_ids: Collection[str],
    earlier_ids: Collection[str],
    report_cgroups: Sequence[Mapping[str, object] | None],
) -> Gate:
    """Return the gates of a model benchmark run, each of whose commands has its own container.

    Every report's limits must be the expected ones; the record shows the first reading
    that is not, or the first of all. No gateway runs, so no connection count is part of
    the fresh gate, and there is no config gate.
    """
    readings = [limits_ok(cgroup) for cgroup in report_cgroups] or [limits_ok(None)]
    new_containers = _all_new(container_ids, earlier_ids)
    return Gate(
        tests_passed=tests_passed,
        image_built=_image_gate(image_digest, container_images),
        limits_ok=next((reading for reading in readings if not reading.ok), readings[0]),
        fresh=FreshGate(ok=new_containers, new_containers=new_containers, accepted_before_t0=None),
        config_ok=None,
    )


def read_scrapes(run_dir: Path) -> list[Scrape]:
    """Read every line of a run's scrapes.

    GateError when the file cannot be read, RunFileError for a line that is malformed.
    """
    path = run_dir / SCRAPES_FILE
    try:
        with gzip.open(path, "rt", encoding="utf-8") as file:
            lines = file.read().splitlines()
    except (OSError, EOFError, zlib.error, UnicodeDecodeError) as error:
        # A file that is not gzip raises BadGzipFile, an OSError; one cut short, EOFError.
        raise GateError(f"cannot read {path}: {error}") from error
    return [Scrape.from_json(line) for line in lines]


def read_report_cgroups(run_dir: Path) -> list[Mapping[str, object] | None]:
    """Read the cgroup object of the environment block of each report of a model benchmark run.

    None for a report without one. GateError when a report cannot be read.
    """
    cgroups: list[Mapping[str, object] | None] = []
    for path in sorted((run_dir / _REPORTS_DIRECTORY).glob("*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, RecursionError) as error:
            # A JSONDecodeError and a UnicodeDecodeError are ValueErrors; deeply nested
            # brackets raise RecursionError.
            raise GateError(f"cannot read {path}: {error}") from error
        cgroup = _entry(report, "environment", "cgroup")
        cgroups.append(cgroup if isinstance(cgroup, Mapping) else None)
    return cgroups


def stored_suite_gate(results_root: Path, image_digest: str, code_digest: str) -> SuiteGate | None:
    """Return the stored tests gate of an image and a code digest, when it can be trusted.

    None when there is none, when the file is malformed or is the gate of another image or
    code, or when the hash it holds is not that of the output stored beside it: the suite
    then runs again. GateError when a file exists and cannot be read.
    """
    gate_path = results_root / suite_gate_file(image_digest, code_digest)
    try:
        gate = SuiteGate.from_json(gate_path.read_text(encoding="utf-8"))
        output = gate_path.with_suffix(".txt").read_bytes()
    except FileNotFoundError, RunFileError, UnicodeDecodeError:
        return None
    except OSError as error:
        raise GateError(f"cannot read the tests gate {gate_path}: {error}") from error
    trusted = (
        gate.image_digest == image_digest
        and gate.code_digest == code_digest
        and gate.output_sha256 == hashlib.sha256(output).hexdigest()
    )
    return gate if trusted else None
