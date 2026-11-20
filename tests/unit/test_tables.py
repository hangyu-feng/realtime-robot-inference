"""Tests for the generated tables of DESIGN.md 11.5, from synthetic results roots."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bench.candidates import CANDIDATES
from bench.facts import REGISTRY
from bench.runfiles import SUMMARY_FILE, Fact, Facts, ModelSet
from bench.tables import NOT_RUN, TablesError, main, render_tables, topology_runs, write_tables
from cloud.config import Topology

_TABLE_FILES = (
    "breakdown_n5.md",
    "disconnect.md",
    "env.md",
    "faults.md",
    "late_joiner.md",
    "ledger.md",
    "massive.md",
    "memory_budget.md",
    "predicted_vs_measured.md",
    "steady.md",
    "sweep.md",
    "topology.md",
)
_SINGLE = Topology.SINGLE
_SPLIT = Topology.SPLIT
_CODE = "c0ffee00"

type _Runs = Mapping[tuple[str, Topology], Sequence[str]]


def _run_id(topology: Topology, scenario: str, repeat: int) -> str:
    spec = hashlib.sha256(scenario.encode()).hexdigest()[:12]
    return f"{topology}-{spec}-{_CODE}-r{repeat}"


def _fact(
    value: float | str | None,
    run_ids: Sequence[str] = (),
    lowest: float | None = None,
    highest: float | None = None,
    unit: str = "ms",
) -> Fact:
    return Fact(value, unit, len(run_ids), lowest, highest, tuple(run_ids), 0, {}, len(run_ids), 0)


def _facts(runs: _Runs, *replacements: tuple[str, Fact]) -> Facts:
    """Return every key of the registry, with a value where its scenario key and topology ran.

    A key that belongs to no scenario key takes the runs given for the scenario key "".
    """
    facts: dict[str, Fact] = {}
    for key, entry in REGISTRY.items():
        run_ids = runs.get((entry.scenario_key or "", entry.topology or _SINGLE), ())
        ran = _fact(2.0, run_ids, 1.0, 3.0, entry.unit)
        facts[key] = ran if run_ids else _fact(None, unit=entry.unit)
    facts.update(replacements)
    counted = sorted({run_id for run_ids in runs.values() for run_id in run_ids})
    return Facts(ModelSet.FAKE, "digest", tuple(counted), facts)


def _summary(
    cls_rate: float = 30.0,
    fleet_cores: tuple[tuple[str, float], ...] = (("fleet", 0.25),),
    throttled_ms_per_s: float = 0.5,
) -> dict[str, Any]:
    def model(rate: float, p95_ms: float) -> dict[str, Any]:
        return {
            "all": {"commands_per_s_per_robot": {"mean": rate}, "freshness_ms": {"p95": p95_ms}}
        }

    cpu = {
        "cores": {"gateway": 0.5, "driver": 0.125, **dict(fleet_cores)},
        "throttled_ms_per_s": throttled_ms_per_s,
    }
    return {
        "models": {"cls": model(cls_rate, 4.0), "depth": model(cls_rate / 3, 90.0)},
        "cpu": {"all": cpu, "before": cpu},
        "validity": {"fire_lag_ms": {"p99": 0.75}},
    }


def _every_summary(facts: Facts) -> dict[tuple[str, str], dict[str, Any]]:
    """Return a summary for every run the topology tables read."""
    return {
        (scenario, run_id): _summary()
        for (scenario, _), run_ids in topology_runs(facts).items()
        for run_id in run_ids
    }


def _write_results(
    root: Path,
    facts: Facts | None,
    summaries: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> None:
    """Write facts.json, and each summary under raw/<scenario_key>/<run_id>/.

    Without summaries given, every run the topology tables read has one.
    """
    if facts is not None:
        (root / "summary").mkdir(parents=True)
        (root / "summary" / "facts.json").write_text(facts.to_json())
    every = {} if facts is None else _every_summary(facts)
    for (scenario, run_id), summary in (every if summaries is None else summaries).items():
        directory = root / "raw" / scenario / run_id
        directory.mkdir(parents=True)
        (directory / SUMMARY_FILE).write_text(json.dumps(summary))


def _tables(root: Path, out: Path | None = None) -> dict[str, bytes]:
    write_tables(root, out or root)
    directory = (out or root) / "tables"
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def _render(facts: Facts) -> dict[str, str]:
    summaries = {run_id: summary for (_, run_id), summary in _every_summary(facts).items()}
    return render_tables(facts, summaries)


def _line(text: str, start: str) -> str:
    """Return the one line of a table file that starts with start."""
    (line,) = (line for line in text.splitlines() if line.startswith(start))
    return line


def _section(text: str, heading: str) -> str:
    """Return one table of a file, from its heading to the next."""
    start = text.index(f"## {heading}\n")
    end = text.find("\n## ", start + 1)
    return text[start : None if end == -1 else end]


_STEADY_RUNS = {("steady", _SINGLE): [_run_id(_SINGLE, "steady", repeat) for repeat in (1, 2, 3)]}


def test_every_table_of_design_11_5_is_written(tmp_path: Path) -> None:
    _write_results(tmp_path, _facts({}))

    assert tuple(_tables(tmp_path)) == _TABLE_FILES


def test_every_table_is_filled_from_the_summaries_and_the_facts_that_the_pipeline_writes(
    evidence_root: Path, tmp_path: Path
) -> None:
    texts = {name: text.decode() for name, text in _tables(evidence_root, tmp_path).items()}

    # The comparisons ran fifo_vs_latest alone.
    not_run = [
        (name, line)
        for name, text in texts.items()
        for line in text.splitlines()
        if f"| {NOT_RUN} |" in line
    ]
    others = [candidate.name for candidate in CANDIDATES if candidate.name != "fifo_vs_latest"]
    assert not_run == [("ledger.md", _line(texts["ledger.md"], f"| `{name}` |")) for name in others]
    assert _line(texts["faults.md"], "Counted runs:") == (
        "Counted runs: 1 of `disconnect`, 1 of `worker_kill`."
    )


@pytest.mark.parametrize(
    ("name", "heading", "key", "row"),
    [
        ("steady.md", "One robot", "steady.depth.freshness_ms.p99", "| Freshness p99 (ms) |"),
        ("steady.md", "One robot", "predict.steady.cls.freshness_ms.p50", "| Freshness p50 (ms) |"),
        (
            "late_joiner.md",
            "Per robot and window",
            "late_joiner.after.b.cls.commands_per_s",
            "| B | after |",
        ),
        (
            "late_joiner.md",
            "After the join",
            "late_joiner.b.depth.first_command_ms",
            "| B's first command",
        ),
        ("late_joiner.md", "After the join", "late_joiner.cls.settle_s", "| A and B settle"),
        ("sweep.md", "`depth`", "sweep.n10.depth.batch_size.mean", "| 10 |"),
        ("sweep.md", "CPU", "sweep.n3.cpu.throttled_ms_per_s", "| 3 |"),
        (
            "sweep.md",
            "Inflection point, from the sweep and the 50-robot run",
            "sweep.depth.capacity_fps",
            "| `depth` |",
        ),
        (
            "disconnect.md",
            "Survivors",
            "disconnect.after.survivors.cls.freshness_ms.p95",
            "| Survivors' `cls` p95",
        ),
        ("disconnect.md", "After the kill", "disconnect.detect_ms", "| Time from the kill"),
        (
            "massive.md",
            "Per model",
            "massive.depth.command_gap_bound_ms",
            "| Command gap bound (ms) |",
        ),
        (
            "massive.md",
            "Connections, CPU and memory",
            "massive.memory.sock_mb",
            "| Socket memory at the end (MB) |",
        ),
        (
            "massive.md",
            "Within the watchdog limits, from the sweep and the 50-robot run",
            "massive.cls.useful_robots",
            "| Robots served within",
        ),
        ("massive.md", "Soak, 5 minutes", "soak.tasks.delta", "| Change in event-loop tasks |"),
        (
            "breakdown_n5.md",
            "Stages, all values in ms",
            "sweep.n5.depth.stage.batch_peers_ms.p95",
            "| `batch_peers` |",
        ),
        (
            "breakdown_n5.md",
            "Stages, all values in ms",
            "sweep.n5.cls.residual_ms.mean",
            "| Residual |",
        ),
        ("memory_budget.md", "At 50 robots", "massive.worker_depth.rss_mb", "| Worker `depth` |"),
        ("env.md", "Test machine", "env.repeats", "| Repeats"),
        (
            "predicted_vs_measured.md",
            "Every prediction",
            "steady.cls.commands_per_s",
            "| `steady.cls.commands_per_s` |",
        ),
        ("ledger.md", "Candidates", "ledger.depth_unpaced.main_delta", "| `depth_unpaced` |"),
    ],
)
def test_a_measured_cell_shows_its_value_with_the_lowest_and_highest_run_value(
    name: str, heading: str, key: str, row: str
) -> None:
    scenarios = {entry.scenario_key or "" for entry in REGISTRY.values()}
    runs = {(scenario, _SINGLE): [_run_id(_SINGLE, scenario, 1)] for scenario in scenarios}
    facts = _facts(runs, (key, _fact(77.7, runs["steady", _SINGLE], 70.1, 80.2)))

    table = _section(_render(facts)[name], heading)

    assert "| 77.7 (70.1 to 80.2) |" in _line(table, row)


def test_a_fact_without_run_values_shows_its_value_alone() -> None:
    kernel = _fact("6.6.87", _STEADY_RUNS["steady", _SINGLE], unit="")

    text = _render(_facts(_STEADY_RUNS, ("env.kernel", kernel)))["env.md"]

    assert _line(text, "| Kernel |") == "| Kernel | 6.6.87 |"


def test_a_table_states_the_number_of_its_counted_runs() -> None:
    text = _render(_facts(_STEADY_RUNS))["steady.md"]

    assert "\nCounted runs: 3.\n" in text


def test_a_table_with_predictions_states_the_runs_behind_them() -> None:
    model_bench = [_run_id(_SINGLE, "model_bench", repeat) for repeat in (1, 2)]

    text = _render(_facts({**_STEADY_RUNS, ("", _SINGLE): model_bench}))["steady.md"]

    assert "\nCounted runs: 3.\nCounted runs behind the predictions: 2.\n" in text


def test_a_table_of_predictions_alone_is_filled_when_runs_stand_behind_them() -> None:
    model_bench = [_run_id(_SINGLE, "model_bench", 1)]

    text = _section(_render(_facts({("", _SINGLE): model_bench}))["massive.md"], "Predicted")

    assert _line(text, "| Commands/s per robot, mean |") == (
        "| Commands/s per robot, mean | 2.00 (1.00 to 3.00) | 2.00 (1.00 to 3.00) |"
    )
    assert _line(text, "Counted runs behind the predictions:").endswith(": 1.")


def test_a_sweep_row_states_the_runs_of_its_robot_count() -> None:
    runs = {
        ("sweep_n1", _SINGLE): [_run_id(_SINGLE, "sweep_n1", 1), _run_id(_SINGLE, "sweep_n1", 2)],
        ("sweep_n3", _SINGLE): [_run_id(_SINGLE, "sweep_n3", 1)],
    }

    table = _section(_render(_facts(runs))["sweep.md"], "`cls`")

    assert [_line(table, f"| {count} | ").split(" | ")[1] for count in (1, 3, 5, 10)] == [
        "2",
        "1",
        "0",
        "0",
    ]


@pytest.mark.parametrize("name", _TABLE_FILES)
def test_without_facts_every_table_is_written_as_not_run(tmp_path: Path, name: str) -> None:
    text = _tables(tmp_path)[name].decode()

    assert f"| {NOT_RUN} |" in text


@pytest.mark.parametrize(
    "name", ["late_joiner.md", "sweep.md", "disconnect.md", "massive.md", "breakdown_n5.md"]
)
def test_a_table_whose_scenario_has_no_counted_run_says_not_run(name: str) -> None:
    text = _render(_facts(_STEADY_RUNS))[name]

    assert (f"| {NOT_RUN} |" in text, "(1.00 to 3.00)" in text) == (True, False)


def test_a_fault_case_without_a_counted_run_of_its_scenario_is_written_as_not_run() -> None:
    text = _render(_facts(_STEADY_RUNS))["faults.md"]

    assert _section(text, "Cases").splitlines()[2:] == [
        "| Case | Scenario | Checks | Recovery |",
        "| --- | --- | --- | --- |",
        "| A robot's process killed with SIGKILL | `disconnect` | not run |  |",
        "| The `depth` worker killed with SIGKILL | `worker_kill` | not run |  |",
        "",
        "Counted runs: 0 of `disconnect`, 0 of `worker_kill`.",
        "Failed checks: none.",
    ]


_DISCONNECT_RUNS = [_run_id(_SINGLE, "disconnect", repeat) for repeat in (1, 2)]
_WORKER_KILL_RUNS = [_run_id(_SINGLE, "worker_kill", repeat) for repeat in (1, 2, 3)]


def _fault_facts(*replacements: tuple[str, Fact]) -> Facts:
    """Return facts with counted runs of both fault scenarios, and values for the fault table."""
    rate = "commands_per_s_per_robot.mean"
    return _facts(
        {("disconnect", _SINGLE): _DISCONNECT_RUNS, ("worker_kill", _SINGLE): _WORKER_KILL_RUNS},
        ("disconnect.detect_ms", _fact(52.5, _DISCONNECT_RUNS, 50.0, 55.0)),
        ("disconnect.depth.recovery_s", _fact(3, _DISCONNECT_RUNS, 2, 4, "s (1 s bins)")),
        (
            f"worker_kill.before.cls.{rate}",
            _fact(30.0, _WORKER_KILL_RUNS, 29.5, 30.1, "commands/s"),
        ),
        (f"worker_kill.after.cls.{rate}", _fact(29.8, _WORKER_KILL_RUNS, 29.0, 30.0, "commands/s")),
        ("worker_kill.depth.outage_ms", _fact(1520.0, _WORKER_KILL_RUNS, 1480.0, 1610.0)),
        *replacements,
    )


def test_the_fault_table_shows_each_case_with_its_checks_and_its_recovery() -> None:
    text = _render(_fault_facts())["faults.md"]

    assert _section(text, "Cases").splitlines()[4:] == [
        "| A robot's process killed with SIGKILL | `disconnect` "
        "| `closed_abnormal`, `killed_robot_released` "
        "| session closed after 52.5 (50.0 to 55.0) ms; "
        "survivors' `depth` rate risen after 3 (2 to 4) s |",
        "| The `depth` worker killed with SIGKILL | `worker_kill` "
        "| `worker_restarted`; `cls` commands/s per robot before / after: "
        "30.0 (29.5 to 30.1) / 29.8 (29.0 to 30.0) "
        "| `depth` commands again after 1520 (1480 to 1610) ms |",
        "",
        "Counted runs: 2 of `disconnect`, 3 of `worker_kill`.",
        "Failed checks: none.",
    ]


def test_a_fault_case_whose_scenario_did_not_run_is_not_run_beside_one_that_did() -> None:
    facts = _facts({("worker_kill", _SINGLE): _WORKER_KILL_RUNS})

    text = _section(_render(facts)["faults.md"], "Cases")

    assert _line(text, "| A robot's") == (
        "| A robot's process killed with SIGKILL | `disconnect` | not run |  |"
    )
    assert _line(text, "| The `depth` worker").endswith("ms |")
    assert _line(text, "Counted runs:") == "Counted runs: 0 of `disconnect`, 3 of `worker_kill`."


_WORKER_KILL_FAILED = {
    _WORKER_KILL_RUNS[0]: ("worker_restarted", "simulator_valid"),
    _WORKER_KILL_RUNS[2]: ("no_internal_error",),
}


def _failing(worker_kill_failed: Mapping[str, tuple[str, ...]]) -> Facts:
    """Return the fault facts with a check failed in a disconnect run, and these in worker_kill."""
    detect = _fact(52.5, _DISCONNECT_RUNS, 50.0, 55.0)
    outage = _fact(1520.0, _WORKER_KILL_RUNS, 1480.0, 1610.0)
    return _fault_facts(
        (
            "disconnect.detect_ms",
            replace(detect, failed_checks={_DISCONNECT_RUNS[1]: ("closed_abnormal",)}),
        ),
        ("worker_kill.depth.outage_ms", replace(outage, failed_checks=worker_kill_failed)),
    )


def test_the_fault_table_lists_the_checks_that_failed_in_its_counted_runs() -> None:
    text = _render(_failing(_WORKER_KILL_FAILED))["faults.md"]

    listed = {
        _DISCONNECT_RUNS[1]: "`closed_abnormal`",
        _WORKER_KILL_RUNS[0]: "`worker_restarted`, `simulator_valid`",
        _WORKER_KILL_RUNS[2]: "`no_internal_error`",
    }
    expected = "; ".join(f"{run_id}: {checks}" for run_id, checks in sorted(listed.items()))
    assert _line(text, "Failed checks:") == f"Failed checks: {expected}."


def test_the_fault_table_is_the_same_bytes_whatever_order_its_failed_checks_come_in() -> None:
    reversed_order = dict(reversed(_WORKER_KILL_FAILED.items()))

    assert (
        _render(_failing(_WORKER_KILL_FAILED))["faults.md"]
        == _render(_failing(reversed_order))["faults.md"]
    )


def test_the_topology_table_has_a_worker_kill_section_after_the_disconnect() -> None:
    facts = _facts({("worker_kill", _SINGLE): _WORKER_KILL_RUNS})

    text = _render(facts)["topology.md"]

    headings = [line for line in text.splitlines() if line.startswith("## ")]
    assert headings[headings.index("## `disconnect`") + 1] == "## `worker_kill`"
    assert _line(_section(text, "`worker_kill`"), "Counted runs:") == (
        "Counted runs: 3 in `single`, 0 in `split`."
    )


def test_the_ledger_lists_every_candidate_with_its_outcome() -> None:
    ran = CANDIDATES[0].name
    outcomes = [
        (f"ledger.{candidate.name}.outcome", _fact("not run", unit="")) for candidate in CANDIDATES
    ]
    outcome = _fact("accepted", ("compare-a-1", "compare-b-1"), unit="")

    facts = _facts(_STEADY_RUNS, *outcomes, (f"ledger.{ran}.outcome", outcome))
    text = _render(facts)["ledger.md"]

    assert [
        _line(text, f"| `{candidate.name}` |").rsplit(" | ", 2)[1:] for candidate in CANDIDATES
    ] == [["2", "accepted |"], *(["0", "not run |"] for _ in CANDIDATES[1:])]


def test_without_facts_the_ledger_lists_every_candidate_as_not_run(tmp_path: Path) -> None:
    text = _tables(tmp_path)["ledger.md"].decode()

    assert all(
        _line(text, f"| `{candidate.name}` |").endswith(f"| {NOT_RUN} |")
        for candidate in CANDIDATES
    )


@pytest.mark.parametrize(
    ("measured", "marked"),
    [
        # Against a prediction of 10.0: gaps of 1.5, 2.0, 2.6 and -2.6, each beside a spread
        # of 2.0; a gap equal to the spread is not larger than it.
        (_fact(11.5, ("r",), 10.5, 12.5), ""),
        (_fact(12.0, ("r",), 11.0, 13.0), ""),
        (_fact(12.6, ("r",), 12.0, 14.0), "yes"),
        (_fact(7.4, ("r",), 7.0, 9.0), "yes"),
        (_fact(None), ""),
    ],
)
def test_a_gap_larger_than_the_spread_is_marked_for_explanation(
    measured: Fact, marked: str
) -> None:
    key = "steady.cls.commands_per_s"
    facts = _facts(_STEADY_RUNS, (f"predict.{key}", _fact(10.0)), (key, measured))

    line = _line(_render(facts)["predicted_vs_measured.md"], f"| `{key}` |")

    assert line.endswith(f" | {marked} |")


def test_the_gap_is_the_measurement_minus_the_prediction() -> None:
    key = "massive.cls.commands_per_s_per_robot.mean"
    measured = _fact(8.0, ("r",), 7.5, 9.0)
    facts = _facts(_STEADY_RUNS, (f"predict.{key}", _fact(8.5)), (key, measured))

    line = _line(_render(facts)["predicted_vs_measured.md"], f"| `{key}` |")

    assert line.split(" | ")[1:6] == ["8.50", "8.00 (7.50 to 9.00)", "-0.500", "1.50", "1"]


def test_the_topology_table_shows_each_scenario_from_the_summaries_of_its_counted_runs(
    tmp_path: Path,
) -> None:
    single = [_run_id(_SINGLE, "massive", repeat) for repeat in (1, 2, 3)]
    split = [_run_id(_SPLIT, "massive", 1)]
    summaries = {
        ("massive", single[0]): _summary(cls_rate=8.0, fleet_cores=(("fleet", 0.5),)),
        ("massive", single[1]): _summary(cls_rate=9.0, fleet_cores=(("fleet", 0.25),)),
        ("massive", single[2]): _summary(cls_rate=12.0, fleet_cores=(("fleet", 0.75),)),
        # A killed robot's own fleet process counts; the scenario driver does not.
        ("massive", split[0]): _summary(
            cls_rate=10.0, fleet_cores=(("fleet", 0.25), ("fleet_c", 0.5))
        ),
    }
    facts = _facts({("massive", _SINGLE): single, ("massive", _SPLIT): split})
    _write_results(tmp_path, facts, summaries)

    text = _tables(tmp_path)["topology.md"].decode()
    massive = _section(text, "`massive`")

    assert (
        _line(massive, "| `cls` commands/s per robot |"),
        _line(massive, "| Simulator CPU (cores) |"),
        _line(massive, "Counted runs:"),
    ) == (
        "| `cls` commands/s per robot | 9.00 (8.00 to 12.0) | 10.0 (10.0 to 10.0) |",
        "| Simulator CPU (cores) | 0.500 (0.250 to 0.750) | 0.750 (0.750 to 0.750) |",
        "Counted runs: 3 in `single`, 1 in `split`.",
    )
    assert _line(_section(text, "`soak`"), "| not run |") == "| not run |  |  |"


def _disconnect_summary(killed_cores: float) -> dict[str, Any]:
    """Return a disconnect run's summary, whose killed robot's fleet process used killed_cores.

    That process has exited by the last scrape of the whole run, so its cores there are null.
    """
    summary = _summary(fleet_cores=(("fleet", 0.25), ("fleet_c", killed_cores)))
    whole_run = summary["cpu"]["all"]
    summary["cpu"]["all"] = {**whole_run, "cores": {**whole_run["cores"], "fleet_c": None}}
    return summary


def test_the_simulator_cpu_of_the_disconnect_is_read_before_its_robot_is_killed(
    tmp_path: Path,
) -> None:
    run_ids = [_run_id(_SINGLE, "disconnect", repeat) for repeat in (1, 2)]
    summaries = {
        ("disconnect", run_ids[0]): _disconnect_summary(0.125),
        ("disconnect", run_ids[1]): _disconnect_summary(0.375),
    }
    _write_results(tmp_path, _facts({("disconnect", _SINGLE): run_ids}), summaries)

    text = _tables(tmp_path)["topology.md"].decode()

    line = _line(_section(text, "`disconnect`"), "| Simulator CPU")
    assert line.split(" | ")[:2] == [
        "| Simulator CPU before the kill (cores)",
        "0.500 (0.375 to 0.625)",
    ]


def test_the_same_results_give_the_same_bytes(tmp_path: Path) -> None:
    runs = {("massive", _SINGLE): [_run_id(_SINGLE, "massive", 1)], **_STEADY_RUNS}
    _write_results(tmp_path / "results", _facts(runs))

    first = _tables(tmp_path / "results", tmp_path / "first")
    second = _tables(tmp_path / "results", tmp_path / "second")

    assert first == second


def test_run_directories_that_facts_do_not_count_change_no_byte(tmp_path: Path) -> None:
    counted = [_run_id(_SINGLE, "massive", repeat) for repeat in (1, 2)]
    _write_results(tmp_path, _facts({("massive", _SINGLE): counted}))
    before = _tables(tmp_path)
    # Beside the counted runs: a run whose gate failed, a run of other code and an attempt
    # that holds a summary, each with values the counted runs do not have, and a hand run.
    other = _summary(cls_rate=1.0, fleet_cores=(("fleet", 1.0),), throttled_ms_per_s=9.0)
    _write_results(
        tmp_path,
        None,
        {
            ("massive", _run_id(_SINGLE, "massive-gate-failed", 1)): other,
            ("massive", f"{_SINGLE}-{'1' * 12}-{'2' * 8}-r1"): other,
            ("massive", f"{counted[0]}.attempt1"): other,
        },
    )
    (tmp_path / "raw" / "massive" / "manual").mkdir()
    (tmp_path / "raw" / "massive" / "manual" / "config.json").write_text("{}")

    assert _tables(tmp_path) == before


def test_a_fact_the_tables_show_that_facts_json_lacks_is_named() -> None:
    facts = _facts(_STEADY_RUNS)
    absent = ("massive.fleet.rss_mb", "steady.cls.stale_ratio")
    lacking = replace(
        facts, facts={key: fact for key, fact in facts.facts.items() if key not in absent}
    )

    with pytest.raises(TablesError, match=r"massive\.fleet\.rss_mb, steady\.cls\.stale_ratio"):
        render_tables(lacking, {})


def test_a_counted_run_without_its_summary_is_an_error(tmp_path: Path) -> None:
    run_id = _run_id(_SINGLE, "massive", 1)
    _write_results(tmp_path, _facts({("massive", _SINGLE): [run_id]}), {})

    with pytest.raises(TablesError, match=run_id):
        write_tables(tmp_path, tmp_path)


def test_the_command_writes_under_out_and_exits_0(tmp_path: Path) -> None:
    _write_results(tmp_path / "results", _facts(_STEADY_RUNS))

    status = main(["--results", str(tmp_path / "results"), "--out", str(tmp_path / "out")])

    written = sorted(path.name for path in (tmp_path / "out" / "tables").iterdir())
    assert (status, written) == (0, list(_TABLE_FILES))


def test_the_command_exits_1_and_names_the_problem(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_id = _run_id(_SINGLE, "steady", 1)
    _write_results(tmp_path, _facts({("steady", _SINGLE): [run_id]}), {})

    status = main(["--results", str(tmp_path)])

    assert (status, run_id in capsys.readouterr().err) == (1, True)
