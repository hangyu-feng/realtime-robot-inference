"""The table of DESIGN.md 9.1 rebuilt from the /metrics scrapes of each counted steady run.

For every counted steady run, each cell of the table that /metrics can give is rebuilt from
the run's scrapes.jsonl.gz alone and set beside the value its summary computes from the
robot's samples, with whether the two agree within what the /metrics value resolves. The run
ids come from facts.json, as bench.tables takes them; raw/ is never listed. A cell that does
not agree is a finding of the table, not a failure of the command.

    python -m bench.metrics_table --results DIR [--out DIR]

writes OUT/tables/steady_from_metrics.md, OUT being the results root unless --out names
another directory. It exits 0 whatever the cells show, and 1 only when facts.json cannot be
read or lacks a fact.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bench.render import NULL_TEXT, RenderError, read_facts
from bench.runfiles import SCRAPES_FILE, SUMMARY_FILE, RunFileError, Scrape
from bench.tables import NOT_RUN, RAW_DIRECTORY, TABLES_DIRECTORY, TablesError, topology_runs
from cloud.config import Topology
from common.commands import MODEL_KEYS
from common.histogram import BUCKETS_PER_OCTAVE, Histogram

TABLE_FILE = "steady_from_metrics.md"
SCENARIO_KEY = "steady"
# The summary's window that the table of DESIGN.md 9.1 reads.
WINDOW = "all"
# A quantile read from a histogram is the upper edge of the bucket that holds its sample, so
# it lies at most one bucket width above the sample.
RESOLUTION = 2 ** (1 / BUCKETS_PER_OCTAVE)
# The bucket edges are powers computed in floating point and a value is placed in its bucket
# by a logarithm, so a value at an edge can land a few rounding steps either side of it. The
# relative slack covers those steps and is far below the width of a bucket.
ROUNDING = 1e-9
# A histogram's maximum is the recorded value itself, the same quotient of nanoseconds by 1e6
# that the summary takes from the samples, so the two differ only when they are of different
# commands.
MAX_TOLERANCE_MS = 0.001
# A command can fall into one count and not the other at each end of the span: the gateway
# counts it when it hands it to the connection, the robot when it receives it.
EDGE_COMMANDS = 2
# The telemetry histograms the table reads (DESIGN.md 10.4).
HISTOGRAM_FIELDS = ("freshness_ms", "command_age_peak_ms")
# Each value is shown to the microsecond, the resolution of the rule for the maximum, so a
# reader can check a verdict against the two values beside it.
DECIMALS = 3

_NS_PER_S = 1_000_000_000


class MetricsTableError(Exception):
    """A counted run's scrapes or summary do not hold what the table is rebuilt from."""


@dataclass(frozen=True)
class Robot:
    """One robot's entry of a scrape: the commands handed to it, and its last report."""

    commands_sent: Mapping[str, int]
    # Each model's histograms by their field name; None before the session's first report.
    report: Mapping[str, Mapping[str, Histogram]] | None


@dataclass(frozen=True)
class Reading:
    """A scrape that succeeded, as the table reads it."""

    # The planned second from t0; None for the final scrape, taken after the fleet has left.
    k: int | None
    ts_ns: int  # the gateway's clock at the snapshot
    robots: Mapping[str, Robot]


@dataclass(frozen=True)
class Span:
    """The two scrapes a rate is taken between, and the length of the window of the samples."""

    earlier: Reading
    later: Reading
    window_s: float

    @property
    def span_s(self) -> float:
        return (self.later.ts_ns - self.earlier.ts_ns) / _NS_PER_S

    def commands_sent(self, model: str) -> int:
        """Return how many commands of the model the gateway handed to the robots in the span."""
        earlier = self.earlier.robots
        # A robot that connected after the earlier scrape has no entry there: it had been
        # handed nothing.
        return sum(
            _sent(robot, model) - (_sent(earlier[robot_id], model) if robot_id in earlier else 0)
            for robot_id, robot in self.later.robots.items()
        )


@dataclass(frozen=True)
class Quantile:
    """A cell read from a histogram as the quantile q."""

    field: str
    q: float


@dataclass(frozen=True)
class Maximum:
    """A cell read from a histogram's maximum."""

    field: str


@dataclass(frozen=True)
class Rate:
    """A cell read from the commands_sent counters over a span."""


@dataclass(frozen=True)
class Quantity:
    """A row of the table: its label, its path in the summary and its /metrics source."""

    label: str
    # Under models.<m>.<WINDOW> of summary.json.
    path: tuple[str, ...]
    source: Quantile | Maximum | Rate


# The rows of DESIGN.md 9.1 that /metrics gives, in its order. The freshness trend is not
# among them: its slope is fitted against each command's receive time, which no histogram of
# /metrics holds.
QUANTITIES = (
    Quantity("Freshness p50 (ms)", ("freshness_ms", "p50"), Quantile("freshness_ms", 0.5)),
    Quantity("Freshness p95 (ms)", ("freshness_ms", "p95"), Quantile("freshness_ms", 0.95)),
    Quantity("Freshness p99 (ms)", ("freshness_ms", "p99"), Quantile("freshness_ms", 0.99)),
    Quantity("Freshness max (ms)", ("freshness_ms", "max"), Maximum("freshness_ms")),
    Quantity("Commands per second", ("commands_per_s",), Rate()),
    Quantity(
        "Peak command age p95 (ms)",
        ("command_age_peak_ms", "p95"),
        Quantile("command_age_peak_ms", 0.95),
    ),
)


@dataclass(frozen=True)
class Cell:
    """One cell rebuilt from /metrics, the value from the samples, and whether they agree."""

    metrics: float | None
    samples: float | None
    agrees: bool


@dataclass(frozen=True)
class RunComparison:
    """The cells of one counted run."""

    run_id: str
    # By model, one cell per entry of QUANTITIES, in its order.
    cells: Mapping[str, tuple[Cell, ...]]
    # The scrapes the rates are taken between; None where the window does not hold two.
    span: Span | None
    # Why cells have no /metrics value, as sentences.
    notes: tuple[str, ...]


def within_resolution(metrics_ms: float, samples_ms: float) -> bool:
    """Whether a quantile read from a histogram lies within one bucket at or above the sample's."""
    at_or_above = samples_ms <= metrics_ms * (1 + ROUNDING)
    within_one_bucket = metrics_ms <= samples_ms * RESOLUTION * (1 + ROUNDING)
    return at_or_above and within_one_bucket


def within_max_tolerance(metrics_ms: float, samples_ms: float) -> bool:
    return abs(metrics_ms - samples_ms) <= MAX_TOLERANCE_MS


def within_rate_tolerance(
    metrics_per_s: float, samples_per_s: float, span_s: float, window_s: float
) -> bool:
    """Whether a counter's rate over span_s and the samples' over window_s are the same rate.

    They may differ by the commands that the difference between the two lengths holds at the
    samples' rate, and by one command at each end of the span.
    """
    tolerance_per_s = (samples_per_s * abs(span_s - window_s) + EDGE_COMMANDS) / span_s
    return abs(metrics_per_s - samples_per_s) <= tolerance_per_s


def _sent(robot: Robot, model: str) -> int:
    # A robot that does not subscribe to the model has no counters of it: it was handed none.
    return robot.commands_sent.get(model, 0)


def _whole_number(value: object) -> int:
    # A bool is an int to isinstance, and true is not a count.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{value!r} is not a whole number")
    return value


def _robot(entry: Any) -> Robot:
    """Return a robots.<id> entry of /metrics (DESIGN.md 10.3 and 10.4) as the table reads it."""
    telemetry = entry.get("telemetry")
    return Robot(
        commands_sent={
            str(model): _whole_number(counters["commands_sent"])
            for model, counters in entry["models"].items()
        },
        report=None
        if telemetry is None
        else {
            str(model): {field: Histogram.from_dict(fields[field]) for field in HISTOGRAM_FIELDS}
            for model, fields in telemetry["models"].items()
        },
    )


def _reading(k: int | None, metrics: Mapping[str, Any]) -> Reading:
    try:
        return Reading(
            k,
            _whole_number(metrics["ts_mono_ns"]),
            {str(robot_id): _robot(entry) for robot_id, entry in metrics["robots"].items()},
        )
    except (KeyError, TypeError, AttributeError, ValueError) as error:
        scrape = "the final scrape" if k is None else f"the scrape planned at {k} s"
        raise MetricsTableError(
            f"{scrape} does not hold /metrics as DESIGN.md 10 describes it: {error!r}"
        ) from error


def readings(scrapes: Iterable[Scrape]) -> tuple[Reading, ...]:
    """Return the scrapes that succeeded, read; MetricsTableError for a malformed one."""
    return tuple(
        _reading(scrape.k, scrape.metrics) for scrape in scrapes if scrape.metrics is not None
    )


def scrape_span(readings: Iterable[Reading], start_s: float, end_s: float) -> Span | None:
    """Return the first and the last scrape planned within the window; None for fewer than two."""
    inside = sorted(
        (
            reading
            for reading in readings
            if reading.k is not None and start_s <= reading.k <= end_s
        ),
        key=lambda reading: reading.ts_ns,
    )
    return Span(inside[0], inside[-1], end_s - start_s) if len(inside) > 1 else None


def final_histograms(final: Reading | None, model: str) -> Mapping[str, Histogram] | None:
    """Return the model's histograms in the robot's last report, which the final scrape holds.

    The robot sends that report when it stops, and it covers every command of its session.
    None when the final scrape failed or holds no report of the model; MetricsTableError when
    it holds reports of more than one robot, which a steady run does not have.
    """
    if final is None:
        return None
    reports = [
        robot.report[model]
        for robot in final.robots.values()
        if robot.report is not None and model in robot.report
    ]
    if len(reports) > 1:
        raise MetricsTableError(
            f"the final scrape holds reports of {len(reports)} robots, and a steady run has one"
        )
    return reports[0] if reports else None


def _summary_number(summary: Mapping[str, Any], path: Sequence[str]) -> float | None:
    value: object = summary
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise MetricsTableError(f"the summary has no {'.'.join(path)}")
        value = value[key]
    if value is None or (isinstance(value, int | float) and not isinstance(value, bool)):
        return value
    raise MetricsTableError(f"the summary's {'.'.join(path)} is not a number")


def _required_number(summary: Mapping[str, Any], path: Sequence[str]) -> float:
    value = _summary_number(summary, path)
    if value is None:
        raise MetricsTableError(f"the summary's {'.'.join(path)} is null")
    return value


def _histogram_cell(
    histograms: Mapping[str, Histogram] | None,
    samples: float | None,
    read: Callable[[Mapping[str, Histogram]], float | None],
    within: Callable[[float, float], bool],
) -> Cell:
    if histograms is None:
        return Cell(None, samples, False)
    metrics = read(histograms)
    if metrics is None or samples is None:
        # Both are absent where the model received no command in the run.
        return Cell(metrics, samples, metrics is None and samples is None)
    return Cell(metrics, samples, within(metrics, samples))


def _cell(
    quantity: Quantity,
    model: str,
    summary: Mapping[str, Any],
    histograms: Mapping[str, Histogram] | None,
    span: Span | None,
) -> Cell:
    path = ("models", model, WINDOW, *quantity.path)
    match quantity.source:
        case Quantile(field, q):
            return _histogram_cell(
                histograms,
                _summary_number(summary, path),
                lambda found: found[field].quantile_ms(q) if found[field].count else None,
                within_resolution,
            )
        case Maximum(field):
            return _histogram_cell(
                histograms,
                _summary_number(summary, path),
                lambda found: found[field].max_ms,
                within_max_tolerance,
            )
        case Rate():
            samples = _required_number(summary, path)
            if span is None:
                return Cell(None, samples, False)
            metrics = span.commands_sent(model) / span.span_s
            return Cell(
                metrics,
                samples,
                within_rate_tolerance(metrics, samples, span.span_s, span.window_s),
            )


def _notes(final: Reading | None, span: Span | None) -> tuple[str, ...]:
    notes = []
    if final is None:
        notes.append("The final scrape failed: no histogram cell has a `/metrics` value.")
    elif all(robot.report is None for robot in final.robots.values()):
        notes.append(
            "The final scrape holds no robot's report: no histogram cell has a `/metrics` value."
        )
    if span is None:
        notes.append(
            "Fewer than two successful scrapes are planned within the window: no rate has a "
            "`/metrics` value."
        )
    return tuple(notes)


def compare_run(
    run_id: str, readings: Sequence[Reading], summary: Mapping[str, Any]
) -> RunComparison:
    """Rebuild the cells of one run from its scrapes, each beside the value of its summary."""
    window = ("windows", WINDOW)
    start_s = _required_number(summary, (*window, "start_s"))
    span = scrape_span(readings, start_s, _required_number(summary, (*window, "end_s")))
    final = next((reading for reading in readings if reading.k is None), None)
    cells = {
        model: tuple(
            _cell(quantity, model, summary, final_histograms(final, model), span)
            for quantity in QUANTITIES
        )
        for model in MODEL_KEYS
    }
    return RunComparison(run_id, cells, span, _notes(final, span))


def unreadable(run_id: str, reason: str) -> RunComparison:
    """Return a run whose files cannot be read: no cell has a value, and none agrees."""
    cells = {model: tuple(Cell(None, None, False) for _ in QUANTITIES) for model in MODEL_KEYS}
    return RunComparison(run_id, cells, None, (f"The run's files cannot be read: {reason}.",))


_TITLE = "Freshness under steady load, from `/metrics`"
_INTRO = (
    "DESIGN.md 9.1 for every counted `steady` run, in both topologies, rebuilt from the run's "
    "`/metrics` scrapes alone and set beside the value its summary computes from the robot's "
    "samples."
)
_RULES = (
    "Freshness p50, p95, p99 and max come from the robot's `freshness_ms` histogram, and the "
    "peak command age from its `command_age_peak_ms` histogram, both in the report the robot "
    "sent when it stopped, which the final scrape holds. A quantile read from a histogram is "
    "the upper edge of the bucket that holds the sample of rank ceil(q x count): it agrees "
    "when it is at or above the sample value and at most 2^(1/8) times it, one bucket width. "
    "A histogram's maximum is the recorded value itself: it agrees within 1 µs. Commands per "
    "second is the rise of `commands_sent`, the gateway's count of the commands it handed to "
    "the robot, between the first and the last successful scrape planned within the window, "
    "over the time S between the two on the gateway's clock, where the samples count over the "
    "window's length L. The robot's own `commands_received` is not used: it stands as of the "
    "robot's last report, up to a report period before the scrape. The two agree "
    "within (rate x |S - L| + 2) / S: the commands that the difference between S and L holds "
    "at the samples' rate, and one command at each end of the span, which the gateway counts "
    "when it hands the command over and the robot when it receives it."
)
_CAVEATS = (
    "The samples' percentiles interpolate between neighbouring samples, so where those lie "
    "far apart, as in a sparse tail, a cell can fall outside its rule while the histogram is "
    "right (DESIGN.md 10.1). The report covers the robot's whole session, which can hold "
    "commands received after the window's end and before the robot stopped. The peak-age "
    "histogram holds the peak of every command that a later one replaced, where the samples "
    "also count the last command's age at the window's end. The freshness trend is not "
    "rebuilt: its slope is fitted against each command's receive time, and no histogram of "
    "`/metrics` holds that time."
)
_HEADER = (
    "Quantity",
    *(
        f"`{model}` {column}"
        for model in MODEL_KEYS
        for column in ("`/metrics`", "samples", "agrees")
    ),
)


def _markdown_row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _value_text(value: float | None) -> str:
    return NULL_TEXT if value is None else f"{value:.{DECIMALS}f}"


def _cell_texts(cell: Cell) -> tuple[str, str, str]:
    return _value_text(cell.metrics), _value_text(cell.samples), "yes" if cell.agrees else "no"


def _row(quantity: Quantity, index: int, comparison: RunComparison) -> tuple[str, ...]:
    return (
        quantity.label,
        *(text for model in MODEL_KEYS for text in _cell_texts(comparison.cells[model][index])),
    )


def _span_line(span: Span) -> str:
    return (
        f"Rates between the scrapes planned at {span.earlier.k} s and {span.later.k} s, "
        f"{span.span_s:.{DECIMALS}f} s apart on the gateway's clock; the samples' window is "
        f"{span.window_s:g} s."
    )


def _run_lines(comparison: RunComparison) -> list[str]:
    lines = [f"## `{comparison.run_id}`", "", _markdown_row(_HEADER)]
    lines.append(_markdown_row(["---"] * len(_HEADER)))
    lines += [
        _markdown_row(_row(quantity, index, comparison))
        for index, quantity in enumerate(QUANTITIES)
    ]
    lines.append("")
    if comparison.span is not None:
        lines.append(_span_line(comparison.span))
    lines += comparison.notes
    return [*lines, ""]


def agreement_line(comparisons: Sequence[RunComparison]) -> str:
    """Return the line that says whether every cell of every counted run agrees."""
    if not comparisons:
        return f"No counted `{SCENARIO_KEY}` run: no cell was compared."
    cells = [
        cell
        for comparison in comparisons
        for model_cells in comparison.cells.values()
        for cell in model_cells
    ]
    differing = sum(not cell.agrees for cell in cells)
    counted = f"counted runs: {len(comparisons)}"
    if differing == 0:
        return f"Every cell of every counted run agrees ({len(cells)} cells, {counted})."
    return f"Not every cell agrees: {differing} of {len(cells)} cells do not ({counted})."


def render_table(comparisons: Sequence[RunComparison]) -> str:
    """Return the text of steady_from_metrics.md; the same comparisons give the same bytes."""
    lines = [f"# {_TITLE}", "", _INTRO, "", _RULES, "", _CAVEATS, ""]
    for comparison in comparisons:
        lines += _run_lines(comparison)
    if not comparisons:
        lines += ["## Counted runs", "", _markdown_row(_HEADER)]
        lines.append(_markdown_row(["---"] * len(_HEADER)))
        lines += [_markdown_row([NOT_RUN, *[""] * (len(_HEADER) - 1)]), ""]
    return "\n".join([*lines, agreement_line(comparisons), ""])


def _reason(error: Exception) -> str:
    # An OSError's text names the file by its absolute path, which would make the table
    # depend on where the results root lies.
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    return str(error)


def load_run(run_dir: Path) -> tuple[tuple[Reading, ...], dict[str, Any]]:
    """Return a run's successful scrapes and its summary; MetricsTableError if one is unreadable."""
    try:
        summary = json.loads((run_dir / SUMMARY_FILE).read_bytes())
    except (OSError, ValueError) as error:
        # A UnicodeDecodeError and a JSONDecodeError are ValueErrors.
        raise MetricsTableError(f"{SUMMARY_FILE}: {_reason(error)}") from error
    if not isinstance(summary, dict):
        raise MetricsTableError(f"{SUMMARY_FILE} does not hold an object")
    try:
        with gzip.open(run_dir / SCRAPES_FILE, "rt", encoding="utf-8") as file:
            scrapes = [Scrape.from_json(line) for line in file]
    except (OSError, EOFError, ValueError, RunFileError) as error:
        # A file that is not gzip is an OSError, and one cut short an EOFError.
        raise MetricsTableError(f"{SCRAPES_FILE}: {_reason(error)}") from error
    return readings(scrapes), summary


def _compare_stored(results: Path, run_id: str) -> RunComparison:
    try:
        run_readings, summary = load_run(results / RAW_DIRECTORY / SCENARIO_KEY / run_id)
        return compare_run(run_id, run_readings, summary)
    except MetricsTableError as error:
        # The boundary of one run: it is reported with no cell compared, and the others are
        # still compared.
        return unreadable(run_id, str(error))


def compare_runs(results: Path) -> tuple[RunComparison, ...]:
    """Return the comparison of every counted steady run of a results root, by run id.

    RenderError when facts.json cannot be read, TablesError when it lacks a fact of a serving
    scenario.
    """
    runs = topology_runs(read_facts(results))
    run_ids = sorted(run_id for topology in Topology for run_id in runs[SCENARIO_KEY, topology])
    return tuple(_compare_stored(results, run_id) for run_id in run_ids)


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m bench.metrics_table",
        description=f"Write {TABLES_DIRECTORY}/{TABLE_FILE}: DESIGN.md 9.1 rebuilt from /metrics.",
    )
    parser.add_argument(
        "--results", type=Path, default=Path("results"), help="results root (default: %(default)s)"
    )
    parser.add_argument(
        "--out", type=Path, help="the directory to write tables/ in (default: the results root)"
    )
    arguments = parser.parse_args(argv)
    try:
        comparisons = compare_runs(arguments.results)
    except (RenderError, TablesError) as error:
        print(f"bench.metrics_table: {error}", file=sys.stderr)
        return 1
    directory = (arguments.out or arguments.results) / TABLES_DIRECTORY
    directory.mkdir(parents=True, exist_ok=True)
    (directory / TABLE_FILE).write_bytes(render_table(comparisons).encode("utf-8"))
    print(f"bench.metrics_table: {agreement_line(comparisons)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
