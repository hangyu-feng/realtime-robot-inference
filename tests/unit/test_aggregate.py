"""Tests for the aggregation: which runs count, how they become facts, and the two commands."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from bench.aggregate import (
    EXCLUDED_FILE,
    FAILED_CHECKS_FILE,
    LEDGER_FILE,
    NOT_RUN,
    PREDICTIONS_FILE,
    SUMMARY_DIRECTORY,
    T_B_FILE,
    AggregateError,
    Current,
    Recorded,
    build,
    lookup,
    main,
    predictions_document,
    tree_code_digest,
)
from bench.candidates import CANDIDATES, Candidate, outcome
from bench.capacity_model import parse_input, predict
from bench.facts import BREAKDOWN_STAGES, REGISTRY, Source
from bench.runfiles import (
    DONE_FILE,
    ENVIRONMENT_FILE,
    EVENTS_FILE,
    FACTS_FILE,
    SAMPLES_DIRECTORY,
    SCRAPES_FILE,
    SUMMARY_FILE,
    Done,
    Fact,
    Facts,
    ModelSet,
    RunEnvironment,
    RunSpec,
    done_files,
)
from bench.runplan import MODEL_BENCH_KEY, code_digest, compare_plan, evidence_plan, run_id
from bench.scenarios import (
    RobotPlan,
    Scenario,
    Window,
    disconnect,
    late_joiner,
    massive,
    steady,
    sweep,
    worker_kill,
)
from bench.summarize import run_files, summarizer_digest
from cloud.config import ServerConfig, Topology
from common.commands import MODEL_KEYS
from tests.support import runs

CURRENT = Current(runs.CODE_DIGEST, summarizer_digest())
# A run with a made-up summary needs no robots to speak of: its files only have to be whole.
TINY = Scenario("tiny", 2.0, (RobotPlan("a", 0.0, None),), (Window("all", 0.0, 2.0),))
# The scenarios whose rules read the shape of the timeline: a robot that joins, one that is
# killed. Every other scenario key can stand on TINY.
SHAPED: Mapping[str, Scenario] = {"late_joiner": late_joiner(), "disconnect": disconnect()}
NO_CHECKS: Mapping[str, object] = {"checks": []}
FIFO = next(candidate for candidate in CANDIDATES if candidate.name == "fifo_vs_latest")
# The scenario whose synthetic run stands for each scenario key where the summarizer's own
# summary is read. The soak's keys are read the same way from a shorter run of fewer robots.
SCENARIOS: Mapping[str, Scenario] = {
    "steady": steady(),
    "late_joiner": late_joiner(),
    "sweep_n5": sweep(5),
    "disconnect": disconnect(),
    "massive": massive(),
    "soak": replace(sweep(3), name="soak"),
}


@cache
def _plan(model_set: ModelSet, code_digest: str) -> tuple[RunSpec, ...]:
    return tuple(evidence_plan(1, model_set, code_digest))


def plan_spec(
    scenario_key: str,
    topology: Topology = Topology.SINGLE,
    *,
    model_set: ModelSet = ModelSet.FAKE,
    code_digest: str = runs.CODE_DIGEST,
    repeat: int = 1,
) -> RunSpec:
    """Return the evidence plan's spec for a scenario key and topology, as one of its repeats."""
    (spec,) = (
        spec
        for spec in _plan(model_set, code_digest)
        if spec.scenario_key == scenario_key and spec.topology is topology
    )
    return replace(spec, repeat=repeat)


def nested(values: Mapping[str, object]) -> dict[str, Any]:
    """Return a summary that holds each value at its dotted path, and no failed check."""
    document: dict[str, Any] = {"checks": []}
    for path, value in values.items():
        *parents, last = path.split(".")
        target = document
        for key in parents:
            target = target.setdefault(key, {})
        target[last] = value
    return document


def write_given(
    results: Path,
    spec: RunSpec,
    summary: Mapping[str, object] = NO_CHECKS,
    *,
    scenario: Scenario = TINY,
    gated: bool = True,
) -> Path:
    """Write a finished run of the spec whose summary is the given one."""
    return runs.write_finished_run(
        results, runs.synthetic_run(scenario, spec, gated=gated), summary
    )


def rewrite(run_dir: Path, name: str, text: str) -> None:
    """Replace a file of a finished run, with a DONE that covers the file as it now is."""
    (run_dir / name).write_bytes(text.encode())
    done = Done.from_json((run_dir / DONE_FILE).read_text(encoding="utf-8"))
    recorded = replace(done, files=done_files(run_files(run_dir)))
    (run_dir / DONE_FILE).write_bytes(recorded.to_json().encode())


def read_facts(root: Path) -> Facts:
    return Facts.from_json((root / SUMMARY_DIRECTORY / FACTS_FILE).read_text(encoding="utf-8"))


def built(
    results: Path, model_set: ModelSet = ModelSet.FAKE, counting: Current | Recorded = CURRENT
) -> Mapping[str, Fact]:
    """Build the facts of a results root in place, and return them."""
    assert build(results, results, model_set, counting) == 0
    return read_facts(results).facts


def listed(results: Path, name: str) -> list[dict[str, Any]]:
    document = json.loads((results / SUMMARY_DIRECTORY / name).read_text(encoding="utf-8"))
    listing: list[dict[str, Any]] = document["runs"]
    return listing


def files_under(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def steady_summary(commands_per_s: float) -> dict[str, Any]:
    return nested({"models.cls.all.commands_per_s": commands_per_s})


def _hand_run(results: Path) -> None:
    runs.write_hand_run(results, runs.synthetic_run(TINY, plan_spec("steady")))


def _without_done(results: Path) -> None:
    (write_given(results, plan_spec("steady", repeat=2)) / DONE_FILE).unlink()


def _changed_after_done(results: Path) -> None:
    events = write_given(results, plan_spec("steady", repeat=2)) / EVENTS_FILE
    events.write_bytes(events.read_bytes() + b"\n")


def _unreadable_done(results: Path) -> None:
    (write_given(results, plan_spec("steady", repeat=2)) / DONE_FILE).write_bytes(b"{")


def _gate_failed(results: Path) -> None:
    write_given(results, plan_spec("steady", repeat=2), gated=False)


def _other_code(results: Path) -> None:
    write_given(results, plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST))


def _off_plan(results: Path) -> None:
    write_given(results, replace(plan_spec("steady"), extra_compose="e" * 64))


def _attempt(results: Path) -> None:
    run = runs.synthetic_run(TINY, plan_spec("steady", repeat=2))
    runs.write_attempt(results, run, NO_CHECKS)


def _copy(results: Path) -> None:
    run_dir = write_given(results, plan_spec("steady", repeat=2))
    run_dir.rename(run_dir.with_name("a-copy"))


def _everything_wrong(results: Path) -> None:
    spec = replace(plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST), extra_compose="e" * 64)
    run_dir = write_given(results, spec, gated=False)
    (run_dir / DONE_FILE).unlink()
    run_dir.rename(run_dir.with_name(f"{run_dir.name}.attempt2"))


def _unfinished_with_broken_files(results: Path) -> None:
    run_dir = write_given(results, plan_spec("steady", repeat=2))
    (run_dir / DONE_FILE).unlink()
    (run_dir / "config.json").write_bytes(b'{"hand_run": fal')
    (run_dir / "gate.json").unlink()


@pytest.mark.parametrize(
    ("write_excluded", "reasons"),
    [
        (_hand_run, ["hand run", "no DONE"]),
        (_without_done, ["no DONE"]),
        (_changed_after_done, ["DONE does not match the files"]),
        (_unreadable_done, ["DONE does not match the files"]),
        (_gate_failed, ["gate failed: limits_ok"]),
        (_other_code, [f"another code digest: {runs.OTHER_CODE_DIGEST}"]),
        (_off_plan, ["spec not in the plan"]),
        (_attempt, ["failed attempt"]),
        (_copy, ["spec not in the plan: not in the directory of its spec"]),
        (
            _everything_wrong,
            [
                "no DONE",
                "gate failed: limits_ok",
                f"another code digest: {runs.OTHER_CODE_DIGEST}",
                "spec not in the plan",
                "failed attempt",
            ],
        ),
        # An interrupted run may stop inside any file: it is not counted, and not an error.
        (_unfinished_with_broken_files, ["no DONE"]),
    ],
)
def test_an_excluded_run_is_listed_with_every_reason_that_applies_in_order(
    tmp_path: Path, write_excluded: Callable[[Path], None], reasons: list[str]
) -> None:
    counted = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    write_excluded(tmp_path)

    facts = built(tmp_path)

    assert [run["reasons"] for run in listed(tmp_path, EXCLUDED_FILE)] == [reasons]
    assert facts["steady.cls.commands_per_s"].run_ids == (counted.name,)
    assert read_facts(tmp_path).counted_run_ids == (counted.name,)


def test_a_run_of_another_spec_of_other_code_a_comparison_and_a_hand_run_enter_no_demo_fact(
    tmp_path: Path,
) -> None:
    counted = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    other = steady_summary(999.0)
    write_given(tmp_path, replace(plan_spec("steady"), extra_compose="e" * 64), other)
    write_given(tmp_path, plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST), other)
    comparison = write_given(tmp_path, compare_plan(FIFO, runs.CODE_DIGEST)[0], other)
    runs.write_hand_run(tmp_path, runs.synthetic_run(TINY, plan_spec("steady")))

    facts = built(tmp_path)

    assert facts["steady.cls.commands_per_s"].value == 30.0
    assert facts["steady.cls.commands_per_s"].run_ids == (counted.name,)
    assert {fact.run_ids for key, fact in facts.items() if key.startswith("steady.")} == {
        (counted.name,)
    }
    # The comparison run is counted, for the ledger alone.
    assert read_facts(tmp_path).counted_run_ids == tuple(sorted((counted.name, comparison.name)))
    assert facts["ledger.fifo_vs_latest.outcome"].run_ids == (comparison.name,)
    assert len(listed(tmp_path, EXCLUDED_FILE)) == 3


@pytest.mark.parametrize(
    ("key", "scenario_key", "path", "run_values", "expected", "lowest", "highest"),
    [
        (
            "steady.cls.freshness_ms.p95",
            "steady",
            "models.cls.all.freshness_ms.p95",
            [9.0, 1.0, 4.0],
            4.0,
            1.0,
            9.0,
        ),
        (
            "steady.cls.freshness_ms.p95",
            "steady",
            "models.cls.all.freshness_ms.p95",
            [9.0, 1.0, 4.0, 5.0],
            4.5,
            1.0,
            9.0,
        ),
        # A statistic that is itself an extreme takes the extreme of every run.
        (
            "steady.cls.freshness_ms.max",
            "steady",
            "models.cls.all.freshness_ms.max",
            [9.0, 1.0, 4.0],
            9.0,
            1.0,
            9.0,
        ),
        (
            "massive.depth.commands_per_s_per_robot.min",
            "massive",
            "models.depth.all.commands_per_s_per_robot.min",
            [0.9, 0.1, 0.4],
            0.1,
            0.1,
            0.9,
        ),
        (
            "massive.cls.fairness.min_over_mean",
            "massive",
            "models.cls.all.fairness.min_over_mean",
            [0.9, 0.5, 0.7],
            0.5,
            0.5,
            0.9,
        ),
        (
            "massive.connections.rejected",
            "massive",
            "connections.rejected",
            [0, 2, 1],
            1,
            0,
            2,
        ),
    ],
)
def test_a_fact_combines_the_values_of_its_counted_runs_by_its_rule(
    tmp_path: Path,
    key: str,
    scenario_key: str,
    path: str,
    run_values: list[float],
    expected: float,
    lowest: float,
    highest: float,
) -> None:
    for repeat, value in enumerate(run_values, start=1):
        write_given(tmp_path, plan_spec(scenario_key, repeat=repeat), nested({path: value}))

    fact = built(tmp_path)[key]

    assert (fact.value, fact.min, fact.max, fact.runs) == (
        expected,
        lowest,
        highest,
        len(run_values),
    )
    assert len(fact.run_ids) == len(run_values)


def test_the_gap_bound_is_the_smallest_bound_of_any_run(tmp_path: Path) -> None:
    for repeat, bound_ms in enumerate((900.0, 700.0, 800.0), start=1):
        check = {
            "name": "gaps_within_bound.depth",
            "passed": True,
            "observed": {"bound_ms": bound_ms},
        }
        write_given(tmp_path, plan_spec("massive", repeat=repeat), {"checks": [check]})

    fact = built(tmp_path)["massive.depth.command_gap_bound_ms"]

    assert (fact.value, fact.min, fact.max) == (700.0, 700.0, 900.0)


@pytest.mark.parametrize(
    ("counts", "expected", "kind"), [([50, 50], 50, int), ([50, 51], 50.5, float)]
)
def test_a_whole_number_fact_is_an_int_whenever_its_combined_value_is_integral(
    tmp_path: Path, counts: list[int], expected: float, kind: type
) -> None:
    for repeat, count in enumerate(counts, start=1):
        summary = nested({"models.cls.all.commands_received": count})
        write_given(tmp_path, plan_spec("steady", repeat=repeat), summary)

    fact = built(tmp_path)["steady.cls.commands_received"]

    assert (fact.value, type(fact.value)) == (expected, kind)
    assert (type(fact.min), type(fact.max)) == (int, int)


def test_a_run_without_a_value_leaves_the_fact_to_the_runs_that_have_one(tmp_path: Path) -> None:
    write_given(tmp_path, plan_spec("steady", repeat=1), steady_summary(30.0))
    write_given(tmp_path, plan_spec("steady", repeat=2), steady_summary(20.0))
    write_given(tmp_path, plan_spec("steady", repeat=3), nested({"models.cls.all": None}))

    fact = built(tmp_path)["steady.cls.commands_per_s"]

    assert (fact.value, fact.min, fact.max, fact.runs) == (25.0, 20.0, 30.0, 3)


def test_a_value_that_is_no_number_is_an_unreadable_input(tmp_path: Path) -> None:
    summary = nested({"models.cls.all.commands_per_s": "thirty"})
    write_given(tmp_path, plan_spec("steady"), summary)

    with pytest.raises(AggregateError, match=r"models\.cls\.all\.commands_per_s"):
        build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)


def breakdown_summary(e2e_ms: float, stage_ms: float, first_stage_ms: float) -> dict[str, Any]:
    """Return a 5-robot summary whose classifier stages each take stage_ms, but for the first."""
    means = dict.fromkeys(BREAKDOWN_STAGES, stage_ms) | {BREAKDOWN_STAGES[0]: first_stage_ms}
    values: dict[str, object] = {
        "models.cls.all.e2e_ms.mean": e2e_ms,
        "models.cls.all.stage_sum_ms.mean": sum(means.values()),
        "models.cls.all.residual_ms.mean": e2e_ms - sum(means.values()),
    }
    for stage, mean_ms in means.items():
        values[f"models.cls.all.stage.{stage}_ms.mean"] = mean_ms
        values[f"models.cls.all.stage.{stage}_ms.p50"] = mean_ms / 2
        values[f"models.cls.all.stage.{stage}_ms.p95"] = mean_ms * 2
    return nested(values)


def test_every_breakdown_key_of_a_model_comes_from_the_run_with_the_median_end_to_end_mean(
    tmp_path: Path,
) -> None:
    # Medians taken key by key would come from different runs: the first stage's from the
    # third run, every other stage's from the first, the end-to-end mean's from the second.
    summaries = [
        breakdown_summary(30.0, 2.0, 1.0),
        breakdown_summary(20.0, 1.0, 9.0),
        breakdown_summary(10.0, 3.0, 5.0),
    ]
    for repeat, summary in enumerate(summaries, start=1):
        write_given(tmp_path, plan_spec("sweep_n5", repeat=repeat), summary)
    median_run = lookup(summaries[1], ("models", "cls", "all"))

    facts = built(tmp_path)

    breakdown = {
        key: entry
        for key, entry in REGISTRY.items()
        if entry.source is Source.BREAKDOWN and key.startswith("sweep.n5.cls.")
    }
    assert {key: facts[key].value for key in breakdown} == {
        key: lookup(median_run, entry.path[3:]) for key, entry in breakdown.items()
    }
    stage_means = [facts[f"sweep.n5.cls.stage.{stage}_ms.mean"].value for stage in BREAKDOWN_STAGES]
    assert sum(mean for mean in stage_means if isinstance(mean, float)) == pytest.approx(
        facts["sweep.n5.cls.stage_sum_ms.mean"].value
    )
    assert len(stage_means) == 10
    first_stage = facts[f"sweep.n5.cls.stage.{BREAKDOWN_STAGES[0]}_ms.mean"]
    assert (first_stage.value, first_stage.min, first_stage.max, first_stage.runs) == (
        9.0,
        1.0,
        9.0,
        3,
    )
    assert len(first_stage.run_ids) == 3
    # The other model has no command in these summaries, and so no breakdown.
    assert facts["sweep.n5.depth.e2e_ms.mean"].value is None


def test_the_breakdown_of_two_runs_comes_from_the_one_with_the_lower_end_to_end_mean(
    tmp_path: Path,
) -> None:
    write_given(tmp_path, plan_spec("sweep_n5", repeat=1), breakdown_summary(30.0, 2.0, 1.0))
    write_given(tmp_path, plan_spec("sweep_n5", repeat=2), breakdown_summary(20.0, 1.0, 9.0))

    facts = built(tmp_path)

    assert facts["sweep.n5.cls.e2e_ms.mean"].value == 20.0
    assert facts[f"sweep.n5.cls.stage.{BREAKDOWN_STAGES[0]}_ms.mean"].value == 9.0
    assert facts[f"sweep.n5.cls.stage.{BREAKDOWN_STAGES[1]}_ms.p95"].value == 2.0


def test_a_fact_carries_the_failed_checks_the_attempts_and_the_dirty_runs_behind_it(
    tmp_path: Path,
) -> None:
    failing = {
        "checks": [
            {"name": "conservation", "passed": False, "observed": None},
            {"name": "no_worker_restart", "passed": True, "observed": 0},
            # A check without a criterion only records what was observed.
            {"name": "memory_growth", "passed": None, "observed": {}},
            {"name": "simulator_valid", "passed": False, "observed": None},
        ]
    }
    failed = write_given(tmp_path, plan_spec("steady", repeat=1), failing)
    dirty = write_given(tmp_path, plan_spec("steady", repeat=2))
    environment = RunEnvironment.from_json((dirty / ENVIRONMENT_FILE).read_text(encoding="utf-8"))
    rewrite(dirty, ENVIRONMENT_FILE, replace(environment, dirty=True).to_json())
    runs.write_attempt(tmp_path, runs.synthetic_run(TINY, plan_spec("steady", repeat=3)), NO_CHECKS)
    write_given(tmp_path, plan_spec("steady", repeat=4), gated=False)
    # Runs of other specs and of other code are no attempts at this fact.
    write_given(tmp_path, replace(plan_spec("steady"), extra_compose="e" * 64))
    write_given(tmp_path, plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST))
    write_given(tmp_path, plan_spec("steady", Topology.SPLIT))

    fact = built(tmp_path)["steady.cls.commands_per_s"]

    assert fact.failed_checks == {failed.name: ("conservation", "simulator_valid")}
    assert (fact.runs, fact.dirty_runs, fact.attempts, fact.failed_attempts) == (2, 1, 4, 2)
    assert listed(tmp_path, FAILED_CHECKS_FILE) == [
        {
            "run_id": failed.name,
            "path": f"raw/steady/{failed.name}",
            "checks": ["conservation", "simulator_valid"],
        }
    ]


def test_every_key_of_the_registry_is_a_fact_when_one_scenario_alone_has_counted_runs(
    tmp_path: Path,
) -> None:
    counted = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))

    assert build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT) == 0

    facts = read_facts(tmp_path)
    assert set(facts.facts) == set(REGISTRY)
    assert (facts.model_set, facts.code_digest) == (ModelSet.FAKE, runs.CODE_DIGEST)
    assert facts.facts["massive.cls.commands_per_s_per_robot.mean"] == Fact(
        value=None,
        unit="commands/s",
        runs=0,
        min=None,
        max=None,
        run_ids=(),
        dirty_runs=0,
        failed_checks={},
        attempts=0,
        failed_attempts=0,
    )
    assert {
        key
        for key, fact in facts.facts.items()
        if fact.value is not None and not key.startswith(("env.", "ledger."))
    } == {"steady.cls.commands_per_s"}
    # Without a counted model benchmark run nothing is predicted.
    assert facts.facts["predict.steady.cls.commands_per_s"].value is None
    assert facts.facts["model_bench.cls.batch_cap"].value is None
    assert sorted(path.name for path in (tmp_path / SUMMARY_DIRECTORY).iterdir()) == [
        EXCLUDED_FILE,
        FACTS_FILE,
        FAILED_CHECKS_FILE,
    ]
    assert facts.facts["env.repeats"].value == 0
    assert facts.facts["env.kernel"].run_ids == (counted.name,)


def test_the_environment_facts_are_those_of_the_run_that_started_last(tmp_path: Path) -> None:
    write_given(tmp_path, plan_spec("steady", repeat=1))
    later = write_given(tmp_path, plan_spec("steady", repeat=2))
    environment = RunEnvironment.from_json((later / ENVIRONMENT_FILE).read_text(encoding="utf-8"))
    inside = replace(
        environment.inside, start_utc="2026-06-01T00:00:00+00:00", kernel="6.9.0-later"
    )
    rewrite(
        later, ENVIRONMENT_FILE, replace(environment, inside=inside, engine_version="28").to_json()
    )

    facts = built(tmp_path)

    assert facts["env.kernel"].value == "6.9.0-later"
    assert facts["env.engine"].value == "docker 28"
    assert facts["env.cpu_model"].value == "Synthetic CPU"
    assert (facts["env.kernel"].min, facts["env.kernel"].max) == (None, None)


def test_the_repeats_are_the_smallest_number_of_counted_runs_of_any_planned_run(
    tmp_path: Path,
) -> None:
    plan = _plan(ModelSet.NULL, runs.CODE_DIGEST)
    serving = [spec for spec in plan if spec.scenario_key != MODEL_BENCH_KEY]
    for repeat in (1, 2):
        runs.write_model_bench_run(
            tmp_path, plan_spec(MODEL_BENCH_KEY, model_set=ModelSet.NULL, repeat=repeat)
        )
        for spec in serving:
            scenario = SHAPED.get(spec.scenario_key, TINY)
            write_given(tmp_path, replace(spec, repeat=repeat), scenario=scenario)
    write_given(tmp_path, plan_spec("steady", model_set=ModelSet.NULL, repeat=3))

    fact = built(tmp_path, ModelSet.NULL)["env.repeats"]

    assert (fact.value, type(fact.value), fact.runs) == (2, int, 2)
    # On the null model set nothing is predicted, with or without a model benchmark run.
    assert not (tmp_path / SUMMARY_DIRECTORY / T_B_FILE).exists()
    assert not (tmp_path / SUMMARY_DIRECTORY / PREDICTIONS_FILE).exists()


def test_an_empty_results_root_writes_nothing_and_exits_0(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "raw" / "steady").mkdir(parents=True)

    statuses = [
        build(tmp_path, tmp_path / "out", ModelSet.FAKE, CURRENT),
        main(["build", "--results", str(tmp_path), "--models", "fake"]),
        main(["build", "--results", str(tmp_path), "--models", "fake", "--digest", "recorded"]),
        main(["build", "--results", str(tmp_path / "absent"), "--models", "real"]),
    ]

    assert statuses == [0, 0, 0, 0]
    assert files_under(tmp_path) == {}
    assert "no run directory" in capsys.readouterr().out


@pytest.mark.parametrize(
    "counting",
    [
        Current(runs.OTHER_CODE_DIGEST, summarizer_digest()),
        Recorded(runs.OTHER_CODE_DIGEST, frozenset()),
    ],
)
def test_run_directories_of_which_none_counts_write_nothing_and_exit_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], counting: Current | Recorded
) -> None:
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    write_given(tmp_path, plan_spec("steady", repeat=2), gated=False)
    built(tmp_path)
    before = files_under(tmp_path)
    capsys.readouterr()

    status = build(tmp_path, tmp_path / "out", ModelSet.FAKE, counting)

    assert status == 2
    assert files_under(tmp_path) == before
    out = capsys.readouterr().out
    # Both digests, the reasons with their counts, and the two ways on.
    assert runs.CODE_DIGEST in out
    assert runs.OTHER_CODE_DIGEST in out
    assert "2 excluded: another code digest" in out
    assert "1 excluded: gate failed" in out
    assert "make evidence" in out
    assert "make report DIGEST=recorded" in out


def test_no_counted_run_and_facts_of_the_same_code_names_no_way_on(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    built(tmp_path)
    (run_dir / DONE_FILE).unlink()
    capsys.readouterr()

    status = build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)

    assert status == 2
    out = capsys.readouterr().out
    assert "1 excluded: no DONE" in out
    assert "make evidence" not in out


def test_recorded_mode_without_facts_counts_no_run_and_exits_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    before = files_under(tmp_path)

    status = main(["build", "--results", str(tmp_path), "--models", "fake", "--digest", "recorded"])

    assert status == 2
    assert files_under(tmp_path) == before
    assert FACTS_FILE in capsys.readouterr().out


def test_one_counted_run_of_the_code_is_enough_to_write(tmp_path: Path) -> None:
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    other = plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST)
    counted = write_given(tmp_path, other, steady_summary(20.0))

    counting = Current(runs.OTHER_CODE_DIGEST, summarizer_digest())
    assert build(tmp_path, tmp_path / "out", ModelSet.FAKE, counting) == 0

    facts = read_facts(tmp_path / "out")
    assert (facts.code_digest, facts.counted_run_ids) == (runs.OTHER_CODE_DIGEST, (counted.name,))
    assert facts.facts["steady.cls.commands_per_s"].value == 20.0
    # Everything is written under --out, and nothing beside the runs.
    assert not (tmp_path / SUMMARY_DIRECTORY).exists()
    assert (tmp_path / "out" / LEDGER_FILE).is_file()


def scale_summary(model: str, utilization: float | None, peak_age_ms: float) -> dict[str, Any]:
    return nested(
        {
            f"models.{model}.all.utilization": utilization,
            f"models.{model}.all.capacity_fps": None if utilization is None else 600.0,
            f"models.{model}.all.command_age_peak_ms.p95": peak_age_ms,
        }
    )


def test_the_inflection_and_the_useful_robots_are_read_off_the_medians_of_the_scale_points(
    tmp_path: Path,
) -> None:
    # The classifier's utilization crosses 1 between 5 and 10 robots, and its peak command
    # age passes its watchdog limit of 1000 ms at 50.
    measured = {
        "sweep_n1": (0.05, 40.0),
        "sweep_n3": (0.15, 50.0),
        "sweep_n5": (0.25, 60.0),
        "sweep_n10": (1.5, 400.0),
        "massive": (2.5, 1500.0),
    }
    for scenario_key, (utilization, peak_ms) in measured.items():
        write_given(tmp_path, plan_spec(scenario_key), scale_summary("cls", utilization, peak_ms))
    # A second run at 10 robots moves that point's medians: utilization 1.25, capacity 600.
    write_given(tmp_path, plan_spec("sweep_n10", repeat=2), scale_summary("cls", 1.0, 2000.0))

    facts = built(tmp_path)

    assert facts["sweep.cls.capacity_fps"].value == 600.0
    assert facts["sweep.cls.inflection_n"].value == 600.0 / ServerConfig().frame_rate_hz
    assert facts["sweep.cls.inflection_between"].value == "between 5 and 10 robots"
    # The median peak age at 10 robots, 1200 ms, is above the limit: 5 is the largest within.
    assert (facts["massive.cls.useful_robots"].value, facts["massive.cls.useful_robots"].unit) == (
        5,
        "robots",
    )
    inflection_n = facts["sweep.cls.inflection_n"]
    assert (inflection_n.runs, inflection_n.min, inflection_n.max) == (1, None, None)
    assert len(inflection_n.run_ids) == 6
    # Depth has no utilization at any point, and the split topology no counted run.
    assert facts["sweep.depth.inflection_n"].value is None
    assert facts["massive.depth.useful_robots"].value is None
    assert facts["split.sweep.cls.inflection_between"].value is None
    assert facts["split.massive.cls.useful_robots"].runs == 0


@pytest.mark.parametrize(("model", "expected"), [("depth", 3), ("cls", 50)])
def test_each_model_is_held_to_its_own_default_watchdog_limit(
    tmp_path: Path, model: str, expected: int
) -> None:
    # The same peak ages lie within the classifier's default limit of 1000 ms at every point,
    # and pass depth's limit of 300 ms above 3 robots.
    peaks_ms = {
        "sweep_n1": 100.0,
        "sweep_n3": 200.0,
        "sweep_n5": 400.0,
        "sweep_n10": 600.0,
        "massive": 900.0,
    }
    for scenario_key, peak_ms in peaks_ms.items():
        write_given(tmp_path, plan_spec(scenario_key), scale_summary(model, None, peak_ms))

    facts = built(tmp_path)

    assert facts[f"massive.{model}.useful_robots"].value == expected


def test_a_scale_point_without_a_counted_run_leaves_the_cross_run_keys_null(
    tmp_path: Path,
) -> None:
    for scenario_key in ("sweep_n1", "sweep_n3", "sweep_n10", "massive"):
        write_given(tmp_path, plan_spec(scenario_key), scale_summary("cls", 1.5, 10.0))

    facts = built(tmp_path)

    assert facts["sweep.cls.capacity_fps"].value is None
    assert facts["sweep.cls.inflection_between"].value is None
    assert facts["massive.cls.useful_robots"].value is None
    assert facts["sweep.cls.capacity_fps"].runs == 0


def bins_summary(per_robot: Mapping[str, Sequence[float]], model: str = "depth") -> dict[str, Any]:
    return nested(
        {
            f"series.commands_per_bin.{robot_id}.{model}": bins
            for robot_id, bins in per_robot.items()
        }
    )


def test_the_settle_time_is_counted_in_whole_seconds_from_the_join(tmp_path: Path) -> None:
    # Robot b joins at 10 s and has caught up with a in the fourth second after it.
    first = [30] * 10 + [30, 25, 20, 15] + [15] * 16
    joiner = [0] * 10 + [0, 5, 10, 15] + [15] * 16
    summary = bins_summary({"a": first, "b": joiner})
    write_given(tmp_path, plan_spec("late_joiner"), summary, scenario=late_joiner())

    facts = built(tmp_path)

    fact = facts["late_joiner.depth.settle_s"]
    assert (fact.value, type(fact.value), fact.unit) == (4, int, "s (1 s bins)")
    # The summary holds no classifier series, so the classifier has no answer.
    assert facts["late_joiner.cls.settle_s"].value is None


@pytest.mark.parametrize(
    "per_robot",
    [
        {"a": [30] * 12, "b": [0] * 12},
        {"a": [30.5] * 30, "b": [0.5] * 10 + [15.5] * 20},
    ],
    ids=["shorter than the run", "counts that are not whole"],
)
def test_a_command_series_that_is_short_or_not_whole_counts_is_an_unreadable_input(
    tmp_path: Path, per_robot: Mapping[str, Sequence[float]]
) -> None:
    summary = bins_summary(per_robot)
    write_given(tmp_path, plan_spec("late_joiner"), summary, scenario=late_joiner())

    with pytest.raises(AggregateError, match=r"series\.commands_per_bin\.a\.depth"):
        build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)


def test_the_recovery_time_is_counted_in_whole_seconds_from_the_kill(tmp_path: Path) -> None:
    # Three robots at 5 per second each. After the kill at 20 s the survivors are predicted
    # 7.5, and their mean count first reaches nine tenths of it, 6.75, in the third second.
    summary = bins_summary(
        {"a": [5] * 20 + [5, 6, 7] + [8] * 17, "b": [5] * 20 + [5, 7, 7] + [7] * 17}
    )
    summary["survivors"] = {"depth": {"before": {"commands_per_s_per_robot": 5.0}}}
    write_given(tmp_path, plan_spec("disconnect"), summary, scenario=disconnect())

    facts = built(tmp_path)

    recovery = facts["disconnect.depth.recovery_s"]
    assert (recovery.value, type(recovery.value)) == (3, int)
    # The classifier has no rate before the kill in this summary, and so no answer.
    assert facts["disconnect.cls.recovery_s"].value is None


@pytest.mark.parametrize(
    ("closed_abnormal", "expected"), [(True, 50.0), (False, None), (None, None)]
)
def test_the_detection_time_is_given_only_where_the_abnormal_close_was_seen(
    tmp_path: Path, closed_abnormal: bool | None, expected: float | None
) -> None:
    check = {"name": "closed_abnormal", "passed": closed_abnormal, "observed": {"detect_ms": 50.0}}
    write_given(tmp_path, plan_spec("disconnect"), {"checks": [check]}, scenario=disconnect())

    fact = built(tmp_path)["disconnect.detect_ms"]

    assert fact.value == expected


@pytest.mark.parametrize("scenario_key", sorted(SCENARIOS))
def test_every_summary_path_of_the_registry_is_a_number_the_summarizer_writes(
    tmp_path: Path, scenario_key: str
) -> None:
    run = runs.synthetic_run(SCENARIOS[scenario_key], plan_spec(scenario_key))
    run_dir = runs.write_finished_run(tmp_path, run)
    summary = json.loads((run_dir / SUMMARY_FILE).read_bytes())
    read = {
        key: entry.path
        for key, entry in REGISTRY.items()
        if entry.source in (Source.SUMMARY, Source.BREAKDOWN)
        and (entry.scenario_key, entry.topology) == (scenario_key, Topology.SINGLE)
    }

    values = {key: lookup(summary, path) for key, path in read.items()}

    assert len(read) > 0
    assert {key: value for key, value in values.items() if not isinstance(value, int | float)} == {}


def test_the_facts_of_the_summarizers_own_summaries_have_their_known_answers(
    tmp_path: Path,
) -> None:
    for scenario_key in ("late_joiner", "disconnect"):
        run = runs.synthetic_run(SCENARIOS[scenario_key], plan_spec(scenario_key))
        runs.write_finished_run(tmp_path, run)

    facts = built(tmp_path)

    # Every synthetic robot gets a classifier command per frame and a depth command every
    # third frame from its first frame on, so two robots are level in the first second.
    assert facts["late_joiner.depth.settle_s"].value == 1
    assert facts["late_joiner.after.b.cls.commands_per_s"].value == pytest.approx(30.0)
    # B's first command of a model arrives one synthetic freshness after its join.
    assert facts["late_joiner.b.depth.first_command_ms"].value == pytest.approx(
        runs.FRESHNESS_MS["depth"]
    )
    assert facts["late_joiner.b.cls.first_command_ms"].value == pytest.approx(
        runs.FRESHNESS_MS["cls"]
    )
    assert facts["disconnect.detect_ms"].value == pytest.approx(50.0)
    assert facts["disconnect.before.survivors.depth.commands_per_s_per_robot"].value == (
        pytest.approx(10.0)
    )
    # The synthetic survivors keep their rate after the kill, so they never reach the rise.
    assert facts["disconnect.depth.recovery_s"].value is None
    assert read_failed_checks(tmp_path) == []


def read_failed_checks(results: Path) -> list[dict[str, Any]]:
    return listed(results, FAILED_CHECKS_FILE)


def test_the_worker_kill_facts_are_the_medians_of_their_counted_runs(tmp_path: Path) -> None:
    run_values = [(1500.0, 30.0, 29.0), (1700.0, 29.0, 31.0), (1600.0, 28.0, 30.0)]
    for repeat, (outage_ms, before, after) in enumerate(run_values, start=1):
        summary = nested(
            {
                "fault.outage_ms": outage_ms,
                "models.cls.before.commands_per_s_per_robot.mean": before,
                "models.cls.after.commands_per_s_per_robot.mean": after,
            }
        )
        write_given(tmp_path, runs.run_spec("worker_kill", repeat=repeat), summary)
    split = runs.run_spec("worker_kill", topology=Topology.SPLIT)
    write_given(tmp_path, split, nested({"fault.outage_ms": 2000.0}))

    facts = built(tmp_path)

    outage = facts["worker_kill.depth.outage_ms"]
    assert (outage.value, outage.min, outage.max, outage.runs, outage.unit) == (
        1600.0,
        1500.0,
        1700.0,
        3,
        "ms",
    )
    assert facts["worker_kill.before.cls.commands_per_s_per_robot.mean"].value == 29.0
    assert facts["worker_kill.after.cls.commands_per_s_per_robot.mean"].value == 30.0
    assert facts["split.worker_kill.depth.outage_ms"].value == 2000.0
    # No summary holds a depth rate, though three runs stand behind the key.
    depth_before = facts["worker_kill.before.depth.commands_per_s_per_robot.mean"]
    assert (depth_before.value, depth_before.runs) == (None, 3)


def test_the_worker_kill_facts_of_the_summarizers_own_summary_are_numbers(tmp_path: Path) -> None:
    run = runs.worker_kill_run(worker_kill(), runs.run_spec("worker_kill"))
    runs.write_finished_run(tmp_path, run)

    facts = built(tmp_path)

    keys = [
        key
        for key, entry in REGISTRY.items()
        if (entry.scenario_key, entry.topology) == ("worker_kill", Topology.SINGLE)
    ]
    assert len(keys) == 5
    assert {key: facts[key].value for key in keys if not isinstance(facts[key].value, float)} == {}
    # Every synthetic robot gets a classifier command per frame, before the kill and after it.
    rates = [
        facts[f"worker_kill.{window}.cls.commands_per_s_per_robot.mean"].value
        for window in ("before", "after")
    ]
    assert rates == pytest.approx([30.0, 30.0])
    assert read_failed_checks(tmp_path) == []


def test_the_facts_are_built_without_reading_the_scrapes_or_the_samples(tmp_path: Path) -> None:
    run_dirs = [
        runs.write_finished_run(tmp_path, runs.synthetic_run(SCENARIOS[key], plan_spec(key)))
        for key in ("late_joiner", "disconnect")
    ]
    built(tmp_path)
    before = files_under(tmp_path / SUMMARY_DIRECTORY)
    for run_dir in run_dirs:
        rewrite(run_dir, SCRAPES_FILE, "")
        shutil.rmtree(run_dir / SAMPLES_DIRECTORY)

    built(tmp_path)

    assert files_under(tmp_path / SUMMARY_DIRECTORY) == before


@pytest.fixture(scope="module")
def bench_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A results root with two model benchmark runs, built on the real model set."""
    results = tmp_path_factory.mktemp("bench")
    spec = plan_spec(MODEL_BENCH_KEY, model_set=ModelSet.REAL)
    runs.write_model_bench_run(results, spec, runs.model_bench_reports(1.0))
    runs.write_model_bench_run(results, replace(spec, repeat=2), runs.model_bench_reports(1.5))
    assert build(results, results, ModelSet.REAL, CURRENT) == 0
    return results


def test_the_model_bench_facts_are_medians_of_the_runs_tables(bench_root: Path) -> None:
    facts = read_facts(bench_root).facts

    # The synthetic classifier takes 4 ms per frame in one run and 6 ms in the other.
    assert facts["model_bench.cls.t1.b4.wall_ms"].value == pytest.approx(20.0)
    assert facts["model_bench.cls.t2.b4.wall_ms"].value == pytest.approx(10.0)
    assert facts["model_bench.cls.t1.b4.cpu_ms"].value == pytest.approx(20.0)
    wall = facts["model_bench.cls.t1.b4.wall_ms"]
    assert [wall.min, wall.max] == pytest.approx([16.0, 24.0])
    # Depth at one thread and one frame is the published graph's 60 ms per frame; every
    # other entry the batched copy's, which the builder makes a twentieth slower.
    assert facts["model_bench.depth.t1.b1.wall_ms"].value == pytest.approx(75.0)
    assert facts["model_bench.depth.t1.b2.wall_ms"].value == pytest.approx(75.0 * 1.05 * 2)
    assert facts["model_bench.depth.t2.b1.wall_ms"].value == pytest.approx(75.0 * 1.05 / 2)
    assert facts["model_bench.decode.full_ms"].value == pytest.approx(2.5)
    assert facts["model_bench.decode.half_ms"].value == pytest.approx(1.25)
    assert facts["model_bench.cls.preprocess_ms"].value == pytest.approx(0.625)
    assert facts["model_bench.depth.postprocess_ms"].value == pytest.approx(0.0625)
    # The p50 of the dispatch round trip of one frame: the builder's 0.98 of its mean.
    assert facts["model_bench.dispatch_rt_ms"].value == pytest.approx(0.1 * 0.98 * 1.25)


def test_the_batch_cap_is_planned_from_the_one_thread_table_and_the_default_config(
    bench_root: Path,
) -> None:
    facts = read_facts(bench_root).facts
    # The default budgets: the stale threshold less one frame period and the allowance.
    config = ServerConfig()
    period_ms = 1000.0 / config.frame_rate_hz
    budget_ms = {
        model: config.models[model].stale_ms - period_ms - config.overhead_allowance_ms
        for model in MODEL_KEYS
    }

    # The classifier's largest batch, 8 frames of 5 ms, fits its budget; no depth batch fits.
    assert budget_ms["cls"] > 8 * 5.0
    assert 75.0 < budget_ms["depth"] < 2 * 75.0 * 1.05
    cap = facts["model_bench.cls.batch_cap"]
    assert (cap.value, type(cap.value), cap.runs) == (8, int, 2)
    assert facts["model_bench.depth.batch_cap"].value == 1
    assert facts["model_bench.depth.margin_ms"].value == pytest.approx(budget_ms["depth"] - 75.0)


def test_the_capacity_models_input_is_built_from_the_model_benchmark_alone(
    bench_root: Path,
) -> None:
    t_b = json.loads((bench_root / SUMMARY_DIRECTORY / T_B_FILE).read_bytes())
    run_ids = read_facts(bench_root).counted_run_ids

    assert all(name in t_b["label"] for name in run_ids)
    assert len(run_ids) == 2
    # A batch costs its own CPU time and that of its dispatch on both sides, at its size.
    assert t_b["models"]["cls"]["threads"]["1"]["4"] == {
        "wall_ms": pytest.approx(20.0),
        "cpu_ms": pytest.approx(20.0 + (40.0 + 30.0) * 4 * 1.25 / 1000),
    }
    assert t_b["models"]["depth"]["threads"]["1"]["1"]["wall_ms"] == pytest.approx(75.0)
    assert sorted(t_b["models"]["depth"]["threads"]) == ["1", "2"]
    # Each transport leg is the WebSocket round trip, and each dispatch leg the largest
    # dispatch round trip of any size.
    assert t_b["overheads_ms"] == {
        "transport_up": pytest.approx(0.5),
        "deliver": pytest.approx(0.5),
        "dispatch_request": pytest.approx(0.1 * 8 * 1.25),
        "dispatch_result": pytest.approx(0.1 * 8 * 1.25),
    }
    assert t_b["io_cpu_ms_per_frame"] == pytest.approx((60.0 + 50.0) * 1.25 / 1000)


def test_the_predictions_do_not_depend_on_the_measured_io_cost_except_in_the_cpu_demand(
    bench_root: Path,
) -> None:
    summary = bench_root / SUMMARY_DIRECTORY
    t_b = json.loads((summary / T_B_FILE).read_bytes())
    written = json.loads((summary / PREDICTIONS_FILE).read_bytes())
    facts = read_facts(bench_root).facts
    costlier = {**t_b, "io_cpu_ms_per_frame": t_b["io_cpu_ms_per_frame"] * 4}

    again = predictions_document(ServerConfig(), json.dumps(costlier))

    predicted = {key for key, entry in REGISTRY.items() if entry.source is Source.PREDICTION}
    assert set(written["facts"]) == predicted
    assert {key: facts[key].value for key in predicted} == written["facts"]
    assert again["facts"] == written["facts"]
    curves, curves_again = written["curves"], dict(lookup_object(again, "curves"))
    assert curves_again.pop("cpu_demand_cores") != curves.pop("cpu_demand_cores")
    assert curves_again == curves
    # The share is paced with the configured cost; the measured one is the file's.
    assert written["inputs"]["configured_io_cpu_ms_per_frame"] == ServerConfig().io_cpu_ms_per_frame
    assert written["inputs"]["io_cpu_ms_per_frame"] == t_b["io_cpu_ms_per_frame"]
    assert lookup(again, ("inputs", "io_cpu_ms_per_frame")) == costlier["io_cpu_ms_per_frame"]
    assert facts["predict.steady.cls.commands_per_s"].runs == 2
    assert facts["predict.steady.cls.commands_per_s"].min is None


def test_the_predictions_are_made_for_a_quota_of_2_cores_with_seed_0_for_1_to_50_robots(
    bench_root: Path,
) -> None:
    summary = bench_root / SUMMARY_DIRECTORY
    written = json.loads((summary / PREDICTIONS_FILE).read_bytes())
    config = ServerConfig()
    measured = parse_input((summary / T_B_FILE).read_bytes(), config.models)

    expected = predict(
        config,
        measured.tables,
        measured.overheads,
        measured.io_cpu_ms_per_frame,
        2.0,
        0,
        range(1, 51),
        measured.label,
    )

    assert written == json.loads(json.dumps(expected))


def lookup_object(document: object, key: str) -> Mapping[str, Any]:
    value = lookup(document, (key,))
    assert isinstance(value, dict)
    return value


def test_the_fake_input_takes_the_fake_service_times_and_keeps_the_measured_overheads(
    bench_root: Path, tmp_path: Path
) -> None:
    real = json.loads((bench_root / SUMMARY_DIRECTORY / T_B_FILE).read_bytes())

    assert build(bench_root, tmp_path, ModelSet.FAKE, CURRENT) == 0

    fake = json.loads((tmp_path / SUMMARY_DIRECTORY / T_B_FILE).read_bytes())
    facts = read_facts(tmp_path).facts
    # The fake models wait 2 ms and 40 ms per frame, at one thread, and use no CPU time.
    assert fake["models"] == {
        "cls": {
            "threads": {
                "1": {str(size): {"wall_ms": 2.0 * size, "cpu_ms": 0.0} for size in (1, 2, 4, 8)}
            }
        },
        "depth": {
            "threads": {
                "1": {str(size): {"wall_ms": 40.0 * size, "cpu_ms": 0.0} for size in (1, 2, 4, 8)}
            }
        },
    }
    assert fake["label"].startswith("fake")
    assert (fake["overheads_ms"], fake["io_cpu_ms_per_frame"]) == (
        real["overheads_ms"],
        real["io_cpu_ms_per_frame"],
    )
    # One fake depth frame takes 40 ms, so one robot gets 25 commands per second.
    assert facts["predict.steady.depth.commands_per_s"].value == pytest.approx(25.0, rel=0.05)
    # The model benchmark's own facts still come from the runs on the real models.
    assert facts["model_bench.cls.t1.b4.wall_ms"].value == pytest.approx(20.0)


def comparison_summary(candidate: Candidate, values: Mapping[str, float]) -> dict[str, Any]:
    assert set(values) == {candidate.main.path, *(guard.path for guard in candidate.guards)}
    return nested(values)


def test_the_ledger_holds_each_metric_per_arm_with_both_spreads_and_the_outcome(
    tmp_path: Path,
) -> None:
    trend, rate = FIFO.main.path, FIFO.guards[0].path
    plan = compare_plan(FIFO, runs.CODE_DIGEST)
    arm_values = [
        {trend: 0.0, rate: 15.0},
        {trend: 20.0, rate: 15.0},
        {trend: 21.0, rate: 14.0},
        {trend: 0.5, rate: 15.5},
    ]
    for spec, values in zip(plan, arm_values, strict=True):
        write_given(tmp_path, spec, comparison_summary(FIFO, values))
    # A candidate with runs in one arm alone is not run.
    cls_cap = next(candidate for candidate in CANDIDATES if candidate.name == "cls_cap_1")
    lone = write_given(tmp_path, compare_plan(cls_cap, runs.CODE_DIGEST)[0])
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))

    facts = built(tmp_path)

    lines = [
        json.loads(line)
        for line in (tmp_path / LEDGER_FILE).read_text(encoding="utf-8").splitlines()
    ]
    assert [line["name"] for line in lines] == sorted(candidate.name for candidate in CANDIDATES)
    line = next(line for line in lines if line["name"] == FIFO.name)
    assert line["run_ids"] == {
        "a": sorted(run_id(spec) for spec in plan if spec.arm == "a"),
        "b": sorted(run_id(spec) for spec in plan if spec.arm == "b"),
    }
    assert line["metrics"][trend] == {
        "better": "lower",
        "a_median": 0.25,
        "b_median": 20.5,
        "change": 20.25,
        "a_spread": 0.5,
        "b_spread": 1.0,
        "threshold": pytest.approx(1000.0 / 30.0 / 30.0),
    }
    assert line["metrics"][rate]["change"] == pytest.approx(-0.75)
    expected = outcome(
        FIFO, {trend: [0.0, 0.5], rate: [15.0, 15.5]}, {trend: [20.0, 21.0], rate: [15.0, 14.0]}
    )
    assert (line["outcome"], expected.value) == ("rejected_regression", "rejected_regression")
    assert (line["hypothesis"], line["scenario_key"], line["b_overrides"], line["main"]) == (
        FIFO.hypothesis,
        "steady",
        dict(FIFO.b_overrides),
        {"path": trend, "better": "lower"},
    )
    assert facts["ledger.fifo_vs_latest.outcome"].value == "rejected_regression"
    assert facts["ledger.fifo_vs_latest.main_delta"].value == 20.25
    assert facts["ledger.fifo_vs_latest.baseline_spread"].value == 0.5
    assert (
        facts["ledger.fifo_vs_latest.main_delta"].runs,
        facts["ledger.fifo_vs_latest.main_delta"].unit,
    ) == (
        2,
        "ms/s",
    )
    assert len(facts["ledger.fifo_vs_latest.outcome"].run_ids) == 4
    not_run = next(line for line in lines if line["name"] == "cls_cap_1")
    assert (not_run["outcome"], not_run["metrics"], not_run["run_ids"]) == (
        NOT_RUN,
        {},
        {"a": [lone.name], "b": []},
    )
    assert facts["ledger.cls_cap_1.outcome"].value == NOT_RUN
    assert facts["ledger.cls_cap_1.main_delta"].value is None
    assert facts["ledger.depth_unpaced.outcome"].value == NOT_RUN
    # A comparison run is behind no demo fact.
    assert len(facts["steady.cls.commands_per_s"].run_ids) == 1


def test_a_comparison_run_without_a_metric_is_an_unreadable_input(tmp_path: Path) -> None:
    for spec in compare_plan(FIFO, runs.CODE_DIGEST):
        write_given(tmp_path, spec, nested({FIFO.main.path: 1.0}))

    with pytest.raises(AggregateError, match=r"models\.depth\.all\.commands_per_s"):
        build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)


def stale_summary(commands_per_s: float) -> dict[str, Any]:
    """Return a summary that another version of the summarizer wrote."""
    return {**steady_summary(commands_per_s), "summarizer_digest": "0" * 64}


def test_a_counted_run_whose_summary_is_not_current_stops_the_build_and_is_named(
    tmp_path: Path,
) -> None:
    write_given(tmp_path, plan_spec("steady", repeat=1), steady_summary(30.0))
    other_summarizer = write_given(tmp_path, plan_spec("steady", repeat=2), stale_summary(20.0))
    missing = write_given(tmp_path, plan_spec("steady", repeat=3), steady_summary(10.0))
    (missing / SUMMARY_FILE).unlink()
    edited = write_given(tmp_path, plan_spec("steady", repeat=4), steady_summary(10.0))
    (edited / SUMMARY_FILE).write_bytes(b'{"checks": []}\n')
    before = files_under(tmp_path)

    with pytest.raises(AggregateError, match=r"python -m bench\.summarize") as raised:
        build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)

    named = [run_dir.name in str(raised.value) for run_dir in (other_summarizer, missing, edited)]
    assert named == [True, True, True]
    assert files_under(tmp_path) == before


def test_an_uncounted_run_whose_summary_is_not_current_does_not_stop_the_build(
    tmp_path: Path,
) -> None:
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    write_given(tmp_path, plan_spec("steady", repeat=2), stale_summary(20.0), gated=False)
    other_code = plan_spec("steady", code_digest=runs.OTHER_CODE_DIGEST)
    (write_given(tmp_path, other_code) / SUMMARY_FILE).unlink()

    assert built(tmp_path)["steady.cls.commands_per_s"].value == 30.0


def test_recorded_mode_counts_the_recorded_runs_with_their_summaries_as_they_are(
    tmp_path: Path,
) -> None:
    first = write_given(tmp_path, plan_spec("steady", repeat=1), steady_summary(30.0))
    second = write_given(tmp_path, plan_spec("steady", repeat=2), stale_summary(20.0))
    # A finished, gated run of the plan that the facts did not record is not counted.
    unrecorded = write_given(tmp_path, plan_spec("steady", repeat=3), steady_summary(1.0))
    counting = Recorded(runs.CODE_DIGEST, frozenset((first.name, second.name)))

    fact = built(tmp_path, counting=counting)["steady.cls.commands_per_s"]

    assert (fact.value, fact.run_ids) == (25.0, tuple(sorted((first.name, second.name))))
    assert listed(tmp_path, EXCLUDED_FILE) == [
        {"path": f"raw/steady/{unrecorded.name}", "reasons": ["not recorded in facts.json"]}
    ]


def test_recorded_mode_through_the_command_rebuilds_the_recorded_facts(tmp_path: Path) -> None:
    write_given(tmp_path, plan_spec("steady", repeat=1), steady_summary(30.0))
    write_given(tmp_path, plan_spec("steady", repeat=2), gated=False)
    built(tmp_path)
    recorded = files_under(tmp_path / SUMMARY_DIRECTORY)[FACTS_FILE]
    out = tmp_path / "rebuilt"

    status = main(
        [
            "build",
            "--results",
            str(tmp_path),
            "--models",
            "fake",
            "--out",
            str(out),
            "--digest",
            "recorded",
        ]
    )

    assert status == 0
    assert (out / SUMMARY_DIRECTORY / FACTS_FILE).read_bytes() == recorded


def test_recorded_mode_refuses_a_summary_that_does_not_match_its_done(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    built(tmp_path)
    (run_dir / SUMMARY_FILE).write_bytes(b'{"checks": []}\n')
    before = files_under(tmp_path)

    status = main(["build", "--results", str(tmp_path), "--models", "fake", "--digest", "recorded"])

    assert status == 1
    assert run_dir.name in capsys.readouterr().err
    assert files_under(tmp_path) == before


@pytest.mark.parametrize(
    "damage",
    [
        lambda run_dir: (run_dir / DONE_FILE).unlink(),
        lambda run_dir: (run_dir / EVENTS_FILE).write_bytes(b""),
        lambda run_dir: run_dir.rename(run_dir.with_name("elsewhere")),
    ],
    ids=["no DONE", "a file changed", "the directory is gone"],
)
def test_recorded_mode_refuses_a_recorded_run_that_is_missing_or_not_valid(
    tmp_path: Path, damage: Callable[[Path], object]
) -> None:
    run_dir = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    damage(run_dir)

    with pytest.raises(AggregateError, match=run_dir.name):
        build(
            tmp_path,
            tmp_path,
            ModelSet.FAKE,
            Recorded(runs.CODE_DIGEST, frozenset((run_dir.name,))),
        )


def test_the_code_digest_of_a_tree_leaves_out_a_listed_file_that_the_tree_has_lost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "cloud").mkdir()
    (tmp_path / "cloud" / "present.py").write_bytes(b"x = 1\n")

    def git_listing(arguments: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        # git lists a tracked file that was deleted from the working tree.
        listed = "cloud/gone.py\ncloud/present.py\n"
        return subprocess.CompletedProcess(arguments, 0, stdout=listed, stderr="")

    # The image that runs the tests has no git.
    monkeypatch.setattr(subprocess, "run", git_listing)

    assert tree_code_digest(tmp_path) == code_digest({"cloud/present.py": b"x = 1\n"})


def test_a_finished_run_with_an_unreadable_file_is_an_unreadable_input(tmp_path: Path) -> None:
    run_dir = write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    rewrite(run_dir, "gate.json", "{}")

    with pytest.raises(AggregateError, match=r"gate\.json"):
        build(tmp_path, tmp_path, ModelSet.FAKE, CURRENT)


def test_a_build_removes_the_predictions_of_an_earlier_build_that_it_does_not_make(
    tmp_path: Path,
) -> None:
    write_given(tmp_path, plan_spec("steady"), steady_summary(30.0))
    summary = tmp_path / SUMMARY_DIRECTORY
    summary.mkdir()
    (summary / T_B_FILE).write_bytes(b"{}")
    (summary / PREDICTIONS_FILE).write_bytes(b"{}")

    built(tmp_path)

    assert sorted(path.name for path in summary.iterdir()) == [
        EXCLUDED_FILE,
        FACTS_FILE,
        FAILED_CHECKS_FILE,
    ]


def write_templates(directory: Path, design: str) -> Path:
    directory.mkdir()
    (directory / "DESIGN.tmpl.md").write_bytes(design.encode())
    (directory / "README.tmpl.md").write_bytes(b"No measured value here.\n")
    return directory


NULL_KEY_TEMPLATE = "Rate `{{steady.cls.commands_per_s}}`, useful {{massive.cls.useful_robots}}.\n"


@pytest.mark.parametrize(("model_set", "status"), [(ModelSet.REAL, 1), (ModelSet.FAKE, 0)])
def test_status_fails_for_a_null_templated_key_on_the_real_model_set_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], model_set: ModelSet, status: int
) -> None:
    results = tmp_path / "results"
    write_given(results, plan_spec("steady", model_set=model_set), steady_summary(30.0))
    write_given(results, plan_spec("steady", model_set=model_set, repeat=2), gated=False)
    built(results, model_set)
    templates = write_templates(tmp_path / "docs", NULL_KEY_TEMPLATE)
    capsys.readouterr()

    assert main(["status", "--results", str(results), "--templates", str(templates)]) == status

    out = capsys.readouterr().out
    assert "templated keys that are null: 1\n  massive.cls.useful_robots\n" in out
    assert "excluded runs: 1\n" in out
    assert "gate failed: limits_ok" in out
    assert "runs with a failed check: 0\n" in out


def test_status_fails_for_a_failed_check_on_any_model_set(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    summary = {
        **steady_summary(30.0),
        "checks": [{"name": "conservation", "passed": False, "observed": None}],
    }
    failed = write_given(results, plan_spec("steady"), summary)
    built(results)
    templates = write_templates(tmp_path / "docs", "Rate `{{steady.cls.commands_per_s}}`.\n")
    capsys.readouterr()

    assert main(["status", "--results", str(results), "--templates", str(templates)]) == 1

    out = capsys.readouterr().out
    assert f"runs with a failed check: 1\n  raw/steady/{failed.name}: conservation\n" in out
    assert "templated keys that are null: 0\n" in out


def test_status_passes_when_no_check_failed_and_no_templated_key_is_null(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    results = tmp_path / "results"
    dirty = write_given(results, plan_spec("steady", model_set=ModelSet.REAL), steady_summary(30.0))
    environment = RunEnvironment.from_json((dirty / ENVIRONMENT_FILE).read_text(encoding="utf-8"))
    rewrite(dirty, ENVIRONMENT_FILE, replace(environment, dirty=True).to_json())
    built(results, ModelSet.REAL)
    templates = write_templates(tmp_path / "docs", "Rate `{{steady.cls.commands_per_s}}`.\n")
    capsys.readouterr()

    assert main(["status", "--results", str(results), "--templates", str(templates)]) == 0
    assert "rest on a run of a dirty tree" in capsys.readouterr().out


def test_status_names_a_template_that_is_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_given(tmp_path / "results", plan_spec("steady"), steady_summary(30.0))
    built(tmp_path / "results")

    status = main(
        ["status", "--results", str(tmp_path / "results"), "--templates", str(tmp_path / "docs")]
    )

    assert status == 1
    assert "DESIGN.tmpl.md" in capsys.readouterr().err


def test_status_of_a_results_root_without_facts_has_nothing_to_fail(tmp_path: Path) -> None:
    assert main(["status", "--results", str(tmp_path), "--templates", str(tmp_path)]) == 0
