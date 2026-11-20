"""Tests for the gates of a run: each comparison, the record of them all, and the file readers."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest

# The module is used by its name: pytest would collect an imported tests_gate as a test.
from bench import gates
from bench.gates import (
    EXPECTED_LIMITS,
    GateError,
    PytestReport,
    config_ok,
    fresh_ok,
    gate_record,
    limits_ok,
    model_bench_gate_record,
    parse_pytest_report,
    read_report_cgroups,
    read_scrapes,
    stored_suite_gate,
    suite_gate_file,
    suite_gate_record,
)
from bench.runfiles import (
    ConfigGate,
    FreshGate,
    Gate,
    ImageGate,
    LimitsGate,
    RunFileError,
    Scrape,
    SuiteGate,
)
from bench.runplan import resolved_config
from cloud.config import Topology
from common.cgroup import read_limits

REPORTS = Path(__file__).parent / "pytest_reports"

IMAGE_DIGEST = "sha256:" + "1a2b3c4d5e6f" + "7" * 52
OTHER_IMAGE_DIGEST = "sha256:" + "1a2b3c4d5e6f" + "8" * 52
CODE_DIGEST = "c0de" * 16
COMMIT = "f" * 40

GIB = 2**30
V2_LIMITS = {"cpu.max": "200000 100000", "memory.max": str(4 * GIB), "memory.swap.max": "0"}
CGROUP: Mapping[str, object] = {
    "version": 2,
    "cpu_limit_cores": 2.0,
    "cpu_period_us": 100_000,
    "memory_limit_bytes": 4 * GIB,
    "memory_swap_limit_bytes": 0,
    "cpuset": "0-23",
}

SPEC_CONFIG = resolved_config(Topology.SINGLE, {})
ECHOED_CONFIG = {**SPEC_CONFIG, "version": "0123abc"}

SUITE_GATE = suite_gate_record(
    (REPORTS / "passed.txt").read_bytes(), 0, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=False
)
CONTAINER_IMAGES = {"rri-system-1": IMAGE_DIGEST}
CONTAINER_IDS = ["c1"]
EARLIER_IDS = ["b1", "b2"]


def _metrics(accepted: int) -> dict[str, Any]:
    """Return the parts of a /metrics object that the gates read, with the service's counters."""
    return {
        "ts_mono_ns": 6_000_000_000,
        # The cgroup object holds the usage counters beside the limits.
        "cgroup": {**CGROUP, "cpu_usage_us": 1_500_000},
        "config": ECHOED_CONFIG,
        "connections": {"current": accepted, "accepted": accepted},
    }


def _scrape(k: int | None, metrics: Mapping[str, Any] | None) -> Scrape:
    return Scrape(
        k=k,
        driver_before_ns=6_000_000_000,
        driver_after_ns=6_000_900_000,
        metrics=metrics,
        error=None if metrics is not None else "timed out after 1 s",
        fleet_processes={},
        fleet_cgroup=None,
    )


SCRAPES = [_scrape(-1, _metrics(0)), _scrape(0, _metrics(1)), _scrape(None, _metrics(1))]


def _serving_gate(
    scrapes: list[Scrape],
    container_images: Mapping[str, str] = CONTAINER_IMAGES,
    spec_config: Mapping[str, Any] = SPEC_CONFIG,
) -> Gate:
    return gate_record(
        SUITE_GATE, IMAGE_DIGEST, container_images, CONTAINER_IDS, EARLIER_IDS, spec_config, scrapes
    )


def _write_tree(root: Path, files: Mapping[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_the_expected_limits_are_2_cores_4_gib_and_no_swap() -> None:
    assert asdict(EXPECTED_LIMITS) == {
        "cpu_limit_cores": 2.0,
        "memory_limit_bytes": 4_294_967_296,
        "memory_swap_limit_bytes": 0,
    }


def test_a_cgroup_v2_reading_of_the_expected_limits_with_no_swap_passes(tmp_path: Path) -> None:
    root = _write_tree(tmp_path, {"cgroup.controllers": "cpu memory", **V2_LIMITS})

    gate = limits_ok(asdict(read_limits(root)))

    assert gate == LimitsGate(ok=True, expected=EXPECTED_LIMITS, read=EXPECTED_LIMITS)


def test_a_cgroup_v1_reading_fails_because_it_reports_no_swap_limit(tmp_path: Path) -> None:
    root = _write_tree(
        tmp_path,
        {
            "cpu/cpu.cfs_quota_us": "200000",
            "cpu/cpu.cfs_period_us": "100000",
            "memory/memory.limit_in_bytes": str(4 * GIB),
        },
    )

    gate = limits_ok(asdict(read_limits(root)))

    assert not gate.ok
    assert gate.read == replace(EXPECTED_LIMITS, memory_swap_limit_bytes=None)


def test_a_whole_cpu_limit_written_as_an_integer_is_the_same_limit() -> None:
    assert limits_ok({**CGROUP, "cpu_limit_cores": 2}).ok


@pytest.mark.parametrize(
    ("field", "value", "read"),
    [
        ("cpu_limit_cores", 1.0, 1.0),
        ("cpu_limit_cores", 2.5, 2.5),
        ("cpu_limit_cores", None, None),
        ("cpu_limit_cores", "2", None),
        ("cpu_limit_cores", True, None),
        ("cpu_limit_cores", 10**400, None),
        ("memory_limit_bytes", 2 * GIB, 2 * GIB),
        ("memory_limit_bytes", 4 * GIB + 1, 4 * GIB + 1),
        ("memory_limit_bytes", None, None),
        ("memory_limit_bytes", 4294967296.0, None),
        ("memory_swap_limit_bytes", GIB, GIB),
        ("memory_swap_limit_bytes", None, None),
        ("memory_swap_limit_bytes", False, None),
    ],
)
def test_a_limit_that_differs_or_was_not_read_fails_and_is_recorded_as_read(
    field: str, value: object, read: float | None
) -> None:
    gate = limits_ok({**CGROUP, field: value})

    assert not gate.ok
    assert gate.expected == EXPECTED_LIMITS
    assert asdict(gate.read) == {**asdict(EXPECTED_LIMITS), field: read}


@pytest.mark.parametrize("read", [None, {}, {"cpu_limit_cores": 2.0}])
def test_limits_that_are_absent_fail(read: Mapping[str, object] | None) -> None:
    assert not limits_ok(read).ok


# Output of the installed pytest, captured from small suites run with the gate's own options
# (-q -p no:cacheprovider -rs -m "not docker"), each with the exit status of its run.
@pytest.mark.parametrize(
    ("name", "exit_status", "report", "passes"),
    [
        ("passed", 0, PytestReport(passed=2, deselected=1), True),
        ("long", 0, PytestReport(passed=1, deselected=1), True),
        ("failed", 1, PytestReport(failed=1, passed=2, deselected=1), False),
        ("error", 1, PytestReport(passed=2, deselected=1, errors=1), False),
        ("skipped", 0, PytestReport(passed=2, skipped=1, deselected=1), False),
        ("xfailed", 0, PytestReport(passed=2, deselected=1, xfailed=1), False),
        ("xpassed", 0, PytestReport(passed=2, deselected=1, xpassed=1), False),
        ("all_deselected", 5, PytestReport(deselected=1), False),
        (
            "mixed",
            1,
            PytestReport(
                failed=1, passed=10, skipped=1, deselected=5, xfailed=1, xpassed=1, errors=1
            ),
            False,
        ),
    ],
)
def test_the_counts_are_those_of_the_final_line_of_output_captured_from_pytest(
    name: str, exit_status: int, report: PytestReport, passes: bool
) -> None:
    output = (REPORTS / f"{name}.txt").read_text(encoding="utf-8")

    assert parse_pytest_report(output) == report
    assert gates.tests_gate(report, exit_status) is passes


@pytest.mark.parametrize(
    ("text", "report"),
    [
        ("3 passed in 0.10s", PytestReport(passed=3)),
        ("  3 passed, 2 warnings in 0.10s  ", PytestReport(passed=3)),
        ("1 passed, 1 warning in 0.10s", PytestReport(passed=1)),
        ("2 passed, 2 errors in 1.50s", PytestReport(passed=2, errors=2)),
        ("1 error in 0.19s", PytestReport(errors=1)),
        ("640 passed, 4 deselected in 3600.25s (1:00:00)", PytestReport(passed=640, deselected=4)),
        (
            "9 passed in 0.10s\nprinted by a test\n1 failed, 8 passed in 0.20s\n",
            PytestReport(failed=1, passed=8),
        ),
        (".. [100%]\n2 passed in 0.10s\nsys:1: a warning at exit", PytestReport(passed=2)),
    ],
)
def test_the_last_count_line_of_the_output_is_the_one_read(text: str, report: PytestReport) -> None:
    assert parse_pytest_report(text) == report


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ((REPORTS / "usage_error.txt").read_text(encoding="utf-8"), "no count line"),
        ("", "no count line"),
        ("no tests ran in 0.01s", "no count line"),
        ("3 passed", "no count line"),
        ("the suite: 3 passed in 0.10s", "no count line"),
        ("3 passed, 1 rerun in 0.10s", "unknown count: 1 rerun"),
        ("3 passed, 2 subtests passed in 0.10s", "unknown count: 2 subtests passed"),
    ],
)
def test_output_without_a_count_line_it_can_judge_is_an_error(text: str, message: str) -> None:
    with pytest.raises(GateError, match=message):
        parse_pytest_report(text)


def test_the_tests_gate_passes_with_passed_and_deselected_tests_and_exit_status_zero() -> None:
    assert gates.tests_gate(PytestReport(passed=640, deselected=4), 0)


@pytest.mark.parametrize(
    ("report", "exit_status"),
    [
        (PytestReport(passed=640), 1),
        (PytestReport(passed=640, failed=1), 0),
        (PytestReport(passed=640, errors=1), 0),
        (PytestReport(passed=640, skipped=1), 0),
        (PytestReport(passed=640, xfailed=1), 0),
        (PytestReport(passed=640, xpassed=1), 0),
        (PytestReport(deselected=4), 0),
    ],
    ids=[
        "exit status",
        "failure",
        "error",
        "skip",
        "expected failure",
        "unexpected pass",
        "no passed test",
    ],
)
def test_the_tests_gate_fails_on_anything_but_a_clean_pass(
    report: PytestReport, exit_status: int
) -> None:
    assert not gates.tests_gate(report, exit_status)


def test_the_gate_file_is_named_by_the_starts_of_the_image_digest_and_the_code_digest() -> None:
    assert suite_gate_file(IMAGE_DIGEST, CODE_DIGEST) == "gates/1a2b3c4d5e6f-c0dec0de.json"


def test_the_record_of_a_passing_suite_holds_its_counts_and_the_hash_of_its_output() -> None:
    output = (REPORTS / "passed.txt").read_bytes()

    gate = suite_gate_record(output, 0, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=False)

    assert gate == SuiteGate(
        passed=True,
        image_digest=IMAGE_DIGEST,
        code_digest=CODE_DIGEST,
        commit=COMMIT,
        dirty=False,
        passed_count=2,
        failed_count=0,
        skipped_count=0,
        deselected_count=1,
        gate_file="gates/1a2b3c4d5e6f-c0dec0de.json",
        output_sha256=hashlib.sha256(output).hexdigest(),
    )


@pytest.mark.parametrize(
    ("name", "exit_status"), [("mixed", 1), ("skipped", 0), ("xfailed", 0), ("passed", 2)]
)
def test_the_record_of_a_suite_that_did_not_pass_cleanly_is_not_passed(
    name: str, exit_status: int
) -> None:
    output = (REPORTS / f"{name}.txt").read_bytes()

    gate = suite_gate_record(output, exit_status, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=True)

    assert not gate.passed
    assert gate.dirty


def test_the_record_keeps_the_failed_the_skipped_and_the_deselected_counts() -> None:
    output = (REPORTS / "mixed.txt").read_bytes()

    gate = suite_gate_record(output, 1, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=False)

    assert (gate.passed_count, gate.failed_count, gate.skipped_count, gate.deselected_count) == (
        10,
        1,
        1,
        5,
    )


def test_no_record_is_made_of_output_without_a_count_line() -> None:
    output = (REPORTS / "usage_error.txt").read_bytes()

    with pytest.raises(GateError, match="no count line"):
        suite_gate_record(output, 4, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=False)


def _store(results_root: Path, gate: SuiteGate, output: bytes) -> None:
    gate_path = results_root / gate.gate_file
    gate_path.parent.mkdir(parents=True)
    gate_path.write_text(gate.to_json(), encoding="utf-8")
    gate_path.with_suffix(".txt").write_bytes(output)


def test_a_stored_gate_is_trusted_when_its_hash_is_that_of_the_output_beside_it(
    tmp_path: Path,
) -> None:
    _store(tmp_path, SUITE_GATE, (REPORTS / "passed.txt").read_bytes())

    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) == SUITE_GATE


def test_a_stored_gate_that_failed_is_returned_as_failed(tmp_path: Path) -> None:
    output = (REPORTS / "failed.txt").read_bytes()
    failed = suite_gate_record(output, 1, IMAGE_DIGEST, CODE_DIGEST, COMMIT, dirty=False)
    _store(tmp_path, failed, output)

    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) == failed


def test_there_is_no_stored_gate_before_the_suite_ran(tmp_path: Path) -> None:
    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) is None


def test_a_stored_gate_is_not_trusted_when_its_output_changed(tmp_path: Path) -> None:
    _store(tmp_path, SUITE_GATE, (REPORTS / "passed.txt").read_bytes() + b"\n")

    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) is None


def test_a_stored_gate_is_not_trusted_without_its_output(tmp_path: Path) -> None:
    _store(tmp_path, SUITE_GATE, b"")
    (tmp_path / SUITE_GATE.gate_file).with_suffix(".txt").unlink()

    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) is None


def test_a_stored_gate_that_is_malformed_is_not_trusted(tmp_path: Path) -> None:
    _store(tmp_path, SUITE_GATE, (REPORTS / "passed.txt").read_bytes())
    (tmp_path / SUITE_GATE.gate_file).write_text('{"passed": true}', encoding="utf-8")

    assert stored_suite_gate(tmp_path, IMAGE_DIGEST, CODE_DIGEST) is None


def test_the_stored_gate_of_another_image_under_the_same_file_name_is_not_trusted(
    tmp_path: Path,
) -> None:
    _store(tmp_path, SUITE_GATE, (REPORTS / "passed.txt").read_bytes())

    assert suite_gate_file(OTHER_IMAGE_DIGEST, CODE_DIGEST) == SUITE_GATE.gate_file
    assert stored_suite_gate(tmp_path, OTHER_IMAGE_DIGEST, CODE_DIGEST) is None


def test_new_containers_and_no_connection_before_time_zero_are_fresh() -> None:
    assert fresh_ok(["c1", "c2"], ["b1"], 0) == FreshGate(
        ok=True, new_containers=True, accepted_before_t0=0
    )


@pytest.mark.parametrize(
    ("container_ids", "earlier_ids", "accepted_before_t0", "new_containers"),
    [
        (["c1", "b1"], ["b1", "b2"], 0, False),
        (["c1", "0f" * 32], ["0f" * 6], 0, False),
        (["0f" * 6], ["0f" * 32, "b2"], 0, False),
        ([], ["b1"], 0, False),
        (["c1"], [], 1, True),
        (["c1"], [], None, True),
    ],
    ids=[
        "a container from before the run",
        "a container listed before the run by the start of its id",
        "a container listed before the run by its full id",
        "no container",
        "a connection before time zero",
        "no scrape before time zero",
    ],
)
def test_a_run_that_did_not_start_fresh_fails_the_fresh_gate(
    container_ids: list[str],
    earlier_ids: list[str],
    accepted_before_t0: int | None,
    new_containers: bool,
) -> None:
    assert fresh_ok(container_ids, earlier_ids, accepted_before_t0) == FreshGate(
        ok=False, new_containers=new_containers, accepted_before_t0=accepted_before_t0
    )


def test_equal_configs_with_different_versions_pass() -> None:
    assert config_ok(SPEC_CONFIG, ECHOED_CONFIG) == ConfigGate(ok=True, differences=())


def test_a_number_written_as_2_and_as_2_0_is_one_value() -> None:
    spec_config = {**SPEC_CONFIG, "overhead_allowance_ms": 2}
    echoed_config = {**ECHOED_CONFIG, "overhead_allowance_ms": 2.0}

    assert config_ok(spec_config, echoed_config).ok


def test_a_nested_field_that_differs_fails_and_is_named() -> None:
    forced = resolved_config(Topology.SINGLE, {"models": {"depth": {"forced_batch_cap": 1}}})

    assert config_ok(forced, ECHOED_CONFIG) == ConfigGate(
        ok=False, differences=("models.depth.forced_batch_cap",)
    )


def test_a_split_spec_against_the_echo_of_a_single_gateway_fails_on_the_topology() -> None:
    split = resolved_config(Topology.SPLIT, {})

    assert config_ok(split, ECHOED_CONFIG) == ConfigGate(ok=False, differences=("topology",))


def test_every_path_that_differs_is_named_in_path_order() -> None:
    changed = resolved_config(Topology.SINGLE, {"skip_rule": False, "pacing_headroom_cores": 0.5})
    echoed_config = {**changed, "version": "0123abc", "added": 1}
    del echoed_config["worker_nice"]

    assert config_ok(SPEC_CONFIG, echoed_config).differences == (
        "added",
        "pacing.headroom_cores",
        "skip_rule",
        "worker_nice",
    )


def test_an_echo_without_a_version_differs_in_it() -> None:
    echoed_config = {key: value for key, value in ECHOED_CONFIG.items() if key != "version"}

    assert config_ok(SPEC_CONFIG, echoed_config).differences == ("version",)


def test_a_run_with_no_successful_scrape_fails_the_config_gate() -> None:
    assert config_ok(SPEC_CONFIG, None) == ConfigGate(ok=False, differences=())


def test_a_serving_run_whose_scrapes_show_the_planned_run_is_gated() -> None:
    gate = _serving_gate(SCRAPES)

    assert gate == Gate(
        tests_passed=SUITE_GATE,
        image_built=ImageGate(built=True, digest=IMAGE_DIGEST, containers=CONTAINER_IMAGES),
        limits_ok=LimitsGate(ok=True, expected=EXPECTED_LIMITS, read=EXPECTED_LIMITS),
        fresh=FreshGate(ok=True, new_containers=True, accepted_before_t0=0),
        config_ok=ConfigGate(ok=True, differences=()),
    )
    assert gate.gated
    assert Gate.from_json(gate.to_json()) == gate


def test_a_serving_run_with_no_successful_scrape_fails_the_limits_fresh_and_config_gates() -> None:
    gate = _serving_gate([_scrape(-1, None), _scrape(0, None)])

    assert not gate.gated
    assert gate.limits_ok == limits_ok(None)
    assert gate.fresh == FreshGate(ok=False, new_containers=True, accepted_before_t0=None)
    assert gate.config_ok == ConfigGate(ok=False, differences=())


def test_the_limits_and_the_config_are_read_from_the_first_scrape_that_succeeded() -> None:
    later = {**_metrics(1), "cgroup": {**CGROUP, "cpu_limit_cores": 1.0}, "config": {}}

    gate = _serving_gate([_scrape(-1, None), _scrape(0, _metrics(1)), _scrape(1, later)])

    assert gate.limits_ok.ok
    assert gate.config_ok == ConfigGate(ok=True, differences=())
    assert gate.fresh == FreshGate(ok=False, new_containers=True, accepted_before_t0=None)
    assert not gate.gated


def test_a_connection_accepted_before_time_zero_fails_the_fresh_gate_of_a_run() -> None:
    gate = _serving_gate([_scrape(-1, _metrics(1)), _scrape(0, _metrics(1))])

    assert gate.fresh == FreshGate(ok=False, new_containers=True, accepted_before_t0=1)
    assert not gate.gated


@pytest.mark.parametrize(
    "metrics",
    [{}, {"cgroup": [], "config": "none", "connections": {"accepted": True}}],
    ids=["no entries", "entries of the wrong kind"],
)
def test_metrics_without_the_entries_the_gates_read_fail_them(metrics: dict[str, Any]) -> None:
    gate = _serving_gate([_scrape(-1, metrics)])

    assert gate.limits_ok == limits_ok(None)
    assert gate.fresh == FreshGate(ok=False, new_containers=True, accepted_before_t0=None)
    assert gate.config_ok == ConfigGate(ok=False, differences=())


def test_an_override_that_did_not_reach_the_gateway_fails_the_config_gate_of_a_run() -> None:
    planned = resolved_config(Topology.SINGLE, {"skip_rule": False})

    gate = _serving_gate(SCRAPES, spec_config=planned)

    assert gate.config_ok == ConfigGate(ok=False, differences=("skip_rule",))
    assert not gate.gated


@pytest.mark.parametrize(
    "container_images",
    [{"rri-system-1": IMAGE_DIGEST, "rri-robots": OTHER_IMAGE_DIGEST}, {}],
    ids=["a container of another image", "no container"],
)
def test_a_run_whose_containers_do_not_all_have_the_built_image_fails_the_image_gate(
    container_images: dict[str, str],
) -> None:
    gate = _serving_gate(SCRAPES, container_images=container_images)

    assert gate.image_built == ImageGate(
        built=False, digest=IMAGE_DIGEST, containers=container_images
    )
    assert not gate.gated


def test_a_failed_tests_gate_fails_the_run() -> None:
    failed = replace(SUITE_GATE, passed=False, failed_count=1)

    gate = gate_record(
        failed, IMAGE_DIGEST, CONTAINER_IMAGES, CONTAINER_IDS, EARLIER_IDS, SPEC_CONFIG, SCRAPES
    )

    assert not gate.gated


def _model_bench_gate(report_cgroups: list[Mapping[str, object] | None]) -> Gate:
    return model_bench_gate_record(
        SUITE_GATE, IMAGE_DIGEST, CONTAINER_IMAGES, CONTAINER_IDS, EARLIER_IDS, report_cgroups
    )


def test_a_model_benchmark_run_is_gated_without_a_config_gate_or_a_connection_count() -> None:
    gate = _model_bench_gate([CGROUP, CGROUP])

    assert gate == Gate(
        tests_passed=SUITE_GATE,
        image_built=ImageGate(built=True, digest=IMAGE_DIGEST, containers=CONTAINER_IMAGES),
        limits_ok=LimitsGate(ok=True, expected=EXPECTED_LIMITS, read=EXPECTED_LIMITS),
        fresh=FreshGate(ok=True, new_containers=True, accepted_before_t0=None),
        config_ok=None,
    )
    assert gate.gated
    assert Gate.from_json(gate.to_json()) == gate


def test_one_report_with_other_limits_fails_a_model_benchmark_run_and_is_the_one_shown() -> None:
    one_core = {**CGROUP, "cpu_limit_cores": 1.0}

    gate = _model_bench_gate([CGROUP, one_core, CGROUP])

    assert gate.limits_ok == limits_ok(one_core)
    assert not gate.gated


@pytest.mark.parametrize("report_cgroups", [[], [CGROUP, None]], ids=["no report", "no limits"])
def test_a_model_benchmark_run_without_the_limits_of_every_report_is_not_gated(
    report_cgroups: list[Mapping[str, object] | None],
) -> None:
    gate = _model_bench_gate(report_cgroups)

    assert gate.limits_ok == limits_ok(None)
    assert not gate.gated


def test_a_model_benchmark_container_from_before_the_run_fails_its_fresh_gate() -> None:
    gate = model_bench_gate_record(
        SUITE_GATE, IMAGE_DIGEST, CONTAINER_IMAGES, ["c1", "b2"], EARLIER_IDS, [CGROUP]
    )

    assert gate.fresh == FreshGate(ok=False, new_containers=False, accepted_before_t0=None)
    assert not gate.gated


def test_the_scrapes_of_a_run_are_read_back_line_by_line(tmp_path: Path) -> None:
    lines = "".join(scrape.to_json() + "\n" for scrape in SCRAPES)
    (tmp_path / "scrapes.jsonl.gz").write_bytes(gzip.compress(lines.encode(), mtime=0))

    assert read_scrapes(tmp_path) == SCRAPES


def test_a_run_without_a_scrapes_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(GateError, match="cannot read"):
        read_scrapes(tmp_path)


@pytest.mark.parametrize(
    "content",
    [
        b"not gzip",
        gzip.compress(SCRAPES[0].to_json().encode() * 50, mtime=0)[:-12],
        gzip.compress(b"\xff\n", mtime=0),
    ],
    ids=["not gzip", "cut short", "not text"],
)
def test_a_scrapes_file_that_cannot_be_read_is_an_error(tmp_path: Path, content: bytes) -> None:
    (tmp_path / "scrapes.jsonl.gz").write_bytes(content)

    with pytest.raises(GateError, match="cannot read"):
        read_scrapes(tmp_path)


def test_a_scrape_line_that_is_malformed_is_refused_by_its_field(tmp_path: Path) -> None:
    (tmp_path / "scrapes.jsonl.gz").write_bytes(gzip.compress(b'{"k": 1}\n', mtime=0))

    with pytest.raises(RunFileError, match="driver_before_ns is missing"):
        read_scrapes(tmp_path)


def test_the_limits_of_each_report_are_read_in_the_order_of_the_file_names(tmp_path: Path) -> None:
    one_core = {**CGROUP, "cpu_limit_cores": 1.0}
    _write_tree(
        tmp_path,
        {
            "reports/stages.json": json.dumps({"environment": {"cgroup": one_core}}),
            "reports/batch_cls_t1.json": json.dumps({"environment": {"cgroup": CGROUP}}),
            "reports/dispatch.json": json.dumps({"kind": "dispatch", "environment": {}}),
            "reports/notes.txt": "not a report",
        },
    )

    assert read_report_cgroups(tmp_path) == [CGROUP, None, one_core]


def test_a_run_without_reports_has_no_limits_to_read(tmp_path: Path) -> None:
    assert read_report_cgroups(tmp_path) == []


def test_a_report_that_is_not_json_is_an_error(tmp_path: Path) -> None:
    _write_tree(tmp_path, {"reports/stages.json": '{"environment": '})

    with pytest.raises(GateError, match="cannot read"):
        read_report_cgroups(tmp_path)
