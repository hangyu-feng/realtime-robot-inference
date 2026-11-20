"""Tests for the table of DESIGN.md 9.1 rebuilt from /metrics, from synthetic results roots."""

from __future__ import annotations

import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.aggregate import Current, build
from bench.metrics_table import (
    QUANTITIES,
    RESOLUTION,
    TABLE_FILE,
    Cell,
    RunComparison,
    compare_runs,
    main,
    within_max_tolerance,
    within_rate_tolerance,
    within_resolution,
)
from bench.numbers import check_rebuilt
from bench.runfiles import SCRAPES_FILE, ModelSet, RunSpec
from bench.scenarios import steady
from bench.summarize import summarizer_digest
from bench.tables import write_tables
from common.commands import MODEL_KEYS
from common.histogram import BUCKETS_PER_OCTAVE, RANGE_MAX_MS, RANGE_MIN_MS, Histogram
from common.protocol import ModelTelemetry, Telemetry, telemetry_to_dict
from robot.control import Verdict
from robot.samples import VERDICTS
from tests.support.runs import (
    CODE_DIGEST,
    FRESHNESS_MS,
    NS_PER_MS,
    OTHER_CODE_DIGEST,
    Rows,
    SyntheticRun,
    edit_metrics,
    run_spec,
    synthetic_run,
    write_attempt,
    write_finished_run,
)

# The one robot of the steady scenario.
_ROBOT = "a"
# The lower edge of a histogram bucket that is a whole number of nanoseconds, and the upper
# edge the histogram reports for a value there.
_EDGE_BUCKET = 88
_EDGE_MS = RANGE_MIN_MS * 2 ** (_EDGE_BUCKET / BUCKETS_PER_OCTAVE)
_UPPER_EDGE_MS = RANGE_MIN_MS * 2 ** ((_EDGE_BUCKET + 1) / BUCKETS_PER_OCTAVE)
_ONE_NS_MS = 1 / NS_PER_MS
_ROW = {quantity.label: index for index, quantity in enumerate(QUANTITIES)}
_RATE = "Commands per second"
_CELLS_PER_RUN = len(MODEL_KEYS) * len(QUANTITIES)


def _model_report(rows: Rows) -> ModelTelemetry:
    """Return one model's part of a report, recording the commands as robot.session does."""
    freshness = Histogram()
    for capture_ns, receive_ns in zip(
        rows["capture_ns"].tolist(), rows["receive_ns"].tolist(), strict=True
    ):
        freshness.record((receive_ns - capture_ns) / 1e6)
    applied = rows[rows["verdict"] == VERDICTS.index(Verdict.APPLIED)]
    # An applied command's peak age is its age when the next applied command replaces it.
    peaks = Histogram()
    for capture_ns, replaced_ns in zip(
        applied["capture_ns"][:-1].tolist(), applied["receive_ns"][1:].tolist(), strict=True
    ):
        peaks.record((replaced_ns - capture_ns) / 1e6)
    return ModelTelemetry(
        commands_received=rows.size,
        commands_applied=applied.size,
        commands_stale=0,
        commands_out_of_order=0,
        freshness_ms=freshness,
        command_gap_ms=Histogram(),
        command_age_peak_ms=peaks,
        deliver_ms=Histogram(),
        watchdog_trips=0,
        watchdog_expired_ms=0,
        watchdog_state="ok",
    )


def _report(commands: Rows, at_ns: int) -> dict[str, object]:
    """Return the report a robot sends when it stops, covering the commands of its session."""
    return telemetry_to_dict(
        Telemetry(
            ts_mono_ns=at_ns,
            frames_captured=0,
            frames_sent=0,
            frames_skipped_send=0,
            fire_lag_ms=Histogram(),
            reconnects=0,
            models={
                model: _model_report(commands[commands["model"] == MODEL_KEYS.index(model)])
                for model in MODEL_KEYS
            },
        )
    )


def _with_report(run: SyntheticRun, commands: Rows) -> SyntheticRun:
    """Return the run with the robot's last report, covering commands, in its final scrape."""

    def add(k: int | None, metrics: dict[str, Any]) -> None:
        if k is None:
            metrics["robots"][_ROBOT]["telemetry"] = _report(commands, metrics["ts_mono_ns"])

    return edit_metrics(run, add)


def _run(cls_freshness_ms: float, spec: RunSpec, gated: bool = True) -> SyntheticRun:
    """Return a steady run that serves every cls command cls_freshness_ms after its capture."""
    return synthetic_run(
        steady(), spec, freshness_ms={**FRESHNESS_MS, "cls": cls_freshness_ms}, gated=gated
    )


def _reported_run(
    cls_freshness_ms: float = FRESHNESS_MS["cls"],
    spec: RunSpec | None = None,
    gated: bool = True,
) -> SyntheticRun:
    """Return a steady run whose robot's last report records exactly its samples."""
    run = _run(cls_freshness_ms, spec or run_spec("steady"), gated)
    return _with_report(run, run.samples[_ROBOT].commands)


def _failed(run: SyntheticRun, k: int | None) -> SyntheticRun:
    """Return the run with its scrape planned at k, or its final one for None, failed."""
    scrapes = tuple(
        replace(scrape, metrics=None, error="timed out") if scrape.k == k else scrape
        for scrape in run.scrapes
    )
    return replace(run, scrapes=scrapes)


def _build(results: Path) -> None:
    assert build(results, results, ModelSet.FAKE, Current(CODE_DIGEST, summarizer_digest())) == 0


def _counted(results: Path, *runs: SyntheticRun) -> list[Path]:
    """Write the runs as finished runs, aggregate them, and return their directories."""
    directories = [write_finished_run(results, run) for run in runs]
    _build(results)
    return directories


def _table(results: Path, out: Path | None = None) -> str:
    """Write the table with the command and return its text."""
    arguments = ["--results", str(results), *(["--out", str(out)] if out else [])]
    assert main(arguments) == 0
    return ((out or results) / "tables" / TABLE_FILE).read_bytes().decode("utf-8")


def _cells(comparison: RunComparison, model: str) -> dict[str, Cell]:
    return dict(zip(_ROW, comparison.cells[model], strict=True))


def test_every_cell_of_a_run_whose_last_report_records_its_samples_agrees(
    tmp_path: Path,
) -> None:
    _counted(tmp_path, _reported_run())

    (comparison,) = compare_runs(tmp_path)

    assert [cell.agrees for cells in comparison.cells.values() for cell in cells] == [
        True
    ] * _CELLS_PER_RUN
    assert comparison.notes == ()
    assert _table(tmp_path).splitlines()[-1] == (
        f"Every cell of every counted run agrees ({_CELLS_PER_RUN} cells, counted runs: 1)."
    )


def test_a_quantile_whose_sample_sits_at_its_bucket_s_lower_edge_agrees_one_bucket_above_it(
    tmp_path: Path,
) -> None:
    _counted(tmp_path, _reported_run(_EDGE_MS))

    (comparison,) = compare_runs(tmp_path)

    # The report's value is one whole bucket width above the samples': the rule's boundary.
    assert pytest.approx(RESOLUTION) == _UPPER_EDGE_MS / _EDGE_MS
    assert _cells(comparison, "cls")["Freshness p50 (ms)"] == Cell(_UPPER_EDGE_MS, _EDGE_MS, True)
    assert _cells(comparison, "cls")["Freshness max (ms)"] == Cell(_EDGE_MS, _EDGE_MS, True)


def test_a_quantile_one_nanosecond_beyond_one_bucket_above_its_sample_disagrees(
    tmp_path: Path,
) -> None:
    # The samples are one nanosecond below the edge, and the report records the edge itself.
    below = _run(_EDGE_MS - _ONE_NS_MS, run_spec("steady"))
    at_edge = _run(_EDGE_MS, run_spec("steady"))
    _counted(tmp_path, _with_report(below, at_edge.samples[_ROBOT].commands))

    (comparison,) = compare_runs(tmp_path)

    cls = _cells(comparison, "cls")
    assert {label: cell.agrees for label, cell in cls.items()} == {
        "Freshness p50 (ms)": False,
        "Freshness p95 (ms)": False,
        "Freshness p99 (ms)": False,
        "Freshness max (ms)": True,
        "Commands per second": True,
        "Peak command age p95 (ms)": True,
    }
    assert cls["Freshness p50 (ms)"].metrics == _UPPER_EDGE_MS
    assert cls["Freshness p50 (ms)"].samples == pytest.approx(_EDGE_MS - _ONE_NS_MS, rel=1e-12)
    assert all(cell.agrees for cell in comparison.cells["depth"])
    assert _table(tmp_path).splitlines()[-1] == (
        f"Not every cell agrees: 3 of {_CELLS_PER_RUN} cells do not (counted runs: 1)."
    )


def test_a_run_whose_final_scrape_failed_has_no_histogram_cell_and_keeps_its_rates(
    tmp_path: Path,
) -> None:
    _counted(tmp_path, _failed(_reported_run(), None))

    (comparison,) = compare_runs(tmp_path)

    # Only the rate has a /metrics value, and it agrees.
    given = [(label == _RATE, label == _RATE) for label in _ROW]
    assert {
        model: [(cell.metrics is not None, cell.agrees) for cell in comparison.cells[model]]
        for model in MODEL_KEYS
    } == dict.fromkeys(MODEL_KEYS, given)
    assert comparison.notes == (
        "The final scrape failed: no histogram cell has a `/metrics` value.",
    )


def test_a_failed_scrape_at_the_window_s_start_moves_the_rates_to_the_next_scrape(
    tmp_path: Path,
) -> None:
    _counted(tmp_path, _failed(_reported_run(), 0))

    (comparison,) = compare_runs(tmp_path)

    assert comparison.span is not None
    assert (comparison.span.earlier.k, comparison.span.later.k) == (1, 30)
    assert [_cells(comparison, model)[_RATE].agrees for model in MODEL_KEYS] == [True, True]
    assert (
        "Rates between the scrapes planned at 1 s and 30 s, 29.000 s apart on the gateway's "
        "clock; the samples' window is 30 s." in _table(tmp_path).splitlines()
    )


def test_the_same_results_give_the_same_bytes(tmp_path: Path) -> None:
    results = tmp_path / "results"
    _counted(
        results,
        _reported_run(),
        _reported_run(_EDGE_MS, run_spec("steady", repeat=2)),
        _failed(_reported_run(spec=run_spec("steady", repeat=3)), None),
    )

    assert _table(results, tmp_path / "first") == _table(results, tmp_path / "second")
    assert _table(results) == _table(results, tmp_path / "first")


def test_runs_that_do_not_count_beside_the_counted_one_change_nothing(tmp_path: Path) -> None:
    _counted(tmp_path, _reported_run())
    before = _table(tmp_path)
    write_finished_run(tmp_path, _reported_run(_EDGE_MS, run_spec("steady", repeat=2), False))
    write_finished_run(
        tmp_path, _reported_run(_EDGE_MS, run_spec("steady", code_digest=OTHER_CODE_DIGEST))
    )
    write_attempt(tmp_path, _reported_run(_EDGE_MS, run_spec("steady", repeat=3)))
    _build(tmp_path)

    assert len(list((tmp_path / "raw" / "steady").iterdir())) == 4
    assert _table(tmp_path) == before


def test_a_results_root_without_facts_writes_one_row_not_run(tmp_path: Path) -> None:
    lines = _table(tmp_path).splitlines()

    # The label column, then three columns per model.
    assert "| not run |" + "  |" * (3 * len(MODEL_KEYS)) in lines
    assert lines[-1] == "No counted `steady` run: no cell was compared."


def test_a_counted_run_whose_scrapes_are_gone_is_reported_with_no_cell_agreeing(
    tmp_path: Path,
) -> None:
    (run_dir,) = _counted(tmp_path, _reported_run())
    (run_dir / SCRAPES_FILE).unlink()

    lines = _table(tmp_path).splitlines()

    (comparison,) = compare_runs(tmp_path)
    assert not any(cell.agrees for cells in comparison.cells.values() for cell in cells)
    assert comparison.notes == (
        f"The run's files cannot be read: {SCRAPES_FILE}: No such file or directory.",
    )
    assert lines[-1] == (
        f"Not every cell agrees: {_CELLS_PER_RUN} of {_CELLS_PER_RUN} cells do not "
        "(counted runs: 1)."
    )


def test_an_unreadable_facts_file_is_refused_with_exit_1(tmp_path: Path) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "facts.json").write_text("{")

    assert main(["--results", str(tmp_path)]) == 1
    assert not (tmp_path / "tables").exists()


def test_the_number_check_rebuilds_this_table_beside_the_others(
    evidence_root: Path, tmp_path: Path
) -> None:
    results = tmp_path / "results"
    shutil.copytree(evidence_root, results)
    write_tables(results, results)
    _table(results)

    assert check_rebuilt(results, ModelSet.FAKE, "recorded") == []


@pytest.mark.parametrize(
    ("metrics_ms", "samples_ms", "agrees"),
    [
        (_UPPER_EDGE_MS, _EDGE_MS, True),
        (_EDGE_MS, _EDGE_MS, True),
        (_EDGE_MS * RESOLUTION * (1 + 1e-6), _EDGE_MS, False),
        (_EDGE_MS * (1 - 1e-6), _EDGE_MS, False),
    ],
)
def test_a_quantile_agrees_from_its_sample_to_one_bucket_width_above_it(
    metrics_ms: float, samples_ms: float, agrees: bool
) -> None:
    assert within_resolution(metrics_ms, samples_ms) is agrees


@given(
    values_ms=st.lists(
        st.floats(min_value=RANGE_MIN_MS, max_value=RANGE_MAX_MS), min_size=1, max_size=50
    ),
    q=st.floats(min_value=0.0, max_value=1.0),
)
def test_a_histogram_quantile_agrees_with_the_sample_of_its_rank(
    values_ms: list[float], q: float
) -> None:
    histogram = Histogram()
    for value_ms in values_ms:
        histogram.record(value_ms)
    rank = max(1, math.ceil(q * len(values_ms)))

    assert within_resolution(histogram.quantile_ms(q), sorted(values_ms)[rank - 1])


@pytest.mark.parametrize(
    ("metrics_ms", "agrees"),
    [(64.0 + 2**-10, True), (64.0 - 2**-10, True), (64.0 + 2**-9, False), (64.0 - 2**-9, False)],
)
def test_a_maximum_agrees_within_one_microsecond(metrics_ms: float, agrees: bool) -> None:
    assert within_max_tolerance(metrics_ms, 64.0) is agrees


@pytest.mark.parametrize(
    ("metrics_per_s", "span_s", "agrees"),
    [
        # With the span as long as the window, one command at each end: 2 / 30 per second.
        (30.0 + 2**-4, 30.0, True),
        (30.0 - 2**-4, 30.0, True),
        (30.0 + 2**-3, 30.0, False),
        # A span a second short adds the second's commands at the samples' rate: 32 / 29.
        (30.0 + 1.0, 29.0, True),
        (30.0 - 1.0, 29.0, True),
        (30.0 + 1.125, 29.0, False),
    ],
)
def test_rates_agree_within_the_span_difference_and_one_command_at_each_end(
    metrics_per_s: float, span_s: float, agrees: bool
) -> None:
    assert within_rate_tolerance(metrics_per_s, 30.0, span_s, 30.0) is agrees
