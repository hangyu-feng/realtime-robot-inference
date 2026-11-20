"""The plots, drawn from synthetic results roots, and the curves they are computed from."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.plots import (
    Plot,
    PlotInputError,
    coverage_curve,
    load_inputs,
    main,
    render_plots,
    rolling_p95,
    stale_ratio_curve,
)
from bench.runfiles import (
    ARMS,
    ConfigGate,
    Done,
    Fact,
    Facts,
    FreshGate,
    Gate,
    GatedLimits,
    ImageGate,
    LimitsGate,
    ModelSet,
    RunConfig,
    SuiteGate,
)
from common.commands import MODEL_KEYS

# Every plot of DESIGN.md 11.5.
PLOT_FILES = (
    "steady_freshness.png",
    "late_joiner_throughput.png",
    "sweep_throughput.png",
    "sweep_freshness.png",
    "sweep_batch_size.png",
    "sweep_cpu.png",
    "disconnect_throughput.png",
    "massive_fairness.png",
    "massive_memory.png",
    "soak_memory.png",
    "breakdown_n5.png",
    "stale_ratio_vs_threshold.png",
    "coverage_vs_watchdog.png",
    "model_tb.png",
    "fifo_vs_latest.png",
    "delay_sweep.png",
)
# The plots of experiments that no scenario runs yet.
NEVER_RUN = ("delay_sweep.png",)
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

CODE_DIGEST = "c0de" * 16
OTHER_CODE_DIGEST = "beef" * 16


def fleet(count: int) -> tuple[str, ...]:
    return tuple(f"r{number:02d}" for number in range(1, count + 1))


# The serving scenario keys and the robots whose values their summaries hold. The disconnect
# run's killed robot writes no samples, so its summary has the two survivors.
SCENARIO_ROBOTS: Mapping[str, tuple[str, ...]] = {
    "steady": ("a",),
    "late_joiner": ("a", "b"),
    "sweep_n1": fleet(1),
    "sweep_n3": fleet(3),
    "sweep_n5": fleet(5),
    "sweep_n10": fleet(10),
    "disconnect": ("a", "b"),
    "massive": fleet(50),
    "soak": fleet(50),
}
STAGES = (
    "fire_lag",
    "transport_up",
    "slot_wait",
    "dispatch_rt",
    "decode",
    "preprocess",
    "batch_peers",
    "inference",
    "postprocess",
    "deliver",
)


def sweep_fact_keys(count: int) -> list[str]:
    prefix = f"sweep.n{count}"
    metrics = ("commands_per_s_per_robot.mean", "freshness_ms.p50", "freshness_ms.p95")
    return [
        *(f"{prefix}.{model}.{metric}" for model in MODEL_KEYS for metric in metrics),
        *(f"{prefix}.{model}.batch_size.mean" for model in MODEL_KEYS),
        f"{prefix}.cpu.utilization",
        f"{prefix}.cpu.throttled_ms_per_s",
    ]


# The facts the plots read, by the scenario key whose runs they come from: the registry's keys
# of DESIGN.md 11.4 that the plots draw, and one key per scenario that names its runs.
FACT_KEYS: Mapping[str, list[str]] = {
    "steady": [f"steady.{model}.commands_received" for model in MODEL_KEYS],
    "late_joiner": [f"late_joiner.before.a.{model}.commands_per_s" for model in MODEL_KEYS],
    "sweep_n1": sweep_fact_keys(1),
    "sweep_n3": sweep_fact_keys(3),
    "sweep_n5": [
        *sweep_fact_keys(5),
        *(f"sweep.n5.{model}.stage.{stage}_ms.mean" for model in MODEL_KEYS for stage in STAGES),
        *(f"sweep.n5.{model}.residual_ms.mean" for model in MODEL_KEYS),
        *(f"sweep.n5.{model}.e2e_ms.mean" for model in MODEL_KEYS),
    ],
    "sweep_n10": sweep_fact_keys(10),
    "disconnect": [
        f"disconnect.before.survivors.{model}.commands_per_s_per_robot" for model in MODEL_KEYS
    ],
    "massive": [
        f"massive.{model}.{metric}"
        for model in MODEL_KEYS
        for metric in (
            "commands_received",
            "commands_per_s_per_robot.mean",
            "freshness_ms.p50",
            "freshness_ms.p95",
        )
    ],
    "soak": [f"soak.{model}.commands_per_s_per_robot.mean" for model in MODEL_KEYS],
    "model_bench": [
        "model_bench.dispatch_rt_ms",
        *(
            f"model_bench.{model}.t{threads}.b{size}.{metric}"
            for model in MODEL_KEYS
            for threads in (1, 2)
            for size in (1, 2, 4, 8)
            for metric in ("wall_ms", "cpu_ms")
        ),
    ],
}
TOPOLOGY_PREFIXES = {"single": "", "split": "split."}

type Counted = Mapping[tuple[str, str], Sequence[str]]


def run_id(
    key: str, topology: str = "single", repeat: int = 1, code: str = CODE_DIGEST, variant: str = ""
) -> str:
    spec_digest = hashlib.sha256(f"{key}:{topology}:{variant}".encode()).hexdigest()[:12]
    return f"{topology}-{spec_digest}-{code[:8]}-r{repeat}"


def summary(robot_ids: Sequence[str], level: float) -> dict[str, Any]:
    """Return a serving run's summary with every field the plots read; level shifts each value."""
    t_s = [0.1 * index for index in range(60)]
    return {
        "robots": {
            robot_id: {
                model: {
                    "all": {
                        "commands_per_s": level + index,
                        "freshness_ms": {"p50": 10 * level + index, "p95": 20 * level + index},
                    }
                }
                for model in MODEL_KEYS
            }
            for index, robot_id in enumerate(robot_ids)
        },
        "models": {
            model: {
                "all": {"batch_size": {"distribution": {"1": 10 * level, "2": 3, "4": 1}}},
                "curves": {
                    "freshness_counts": [0, level, 2, 0, 1],
                    "age_time_ms": [5, level, 3, 1],
                    "uncovered_ms": 2,
                },
            }
            for model in MODEL_KEYS
        },
        "cpu": {
            "all": {
                "cores": {
                    "gateway": 0.1 * level,
                    "worker_cls": 0.2,
                    "worker_depth": 0.9,
                    "fleet": 0.05,
                    "driver": 0.01,
                }
            }
        },
        "series": {
            "freshness": {
                model: {"t_s": t_s, "freshness_ms": [level + index % 7 for index in range(60)]}
                for model in MODEL_KEYS
            },
            "commands_per_bin": {
                robot_id: {model: [level + second for second in range(5)] for model in MODEL_KEYS}
                for robot_id in robot_ids
            },
            "memory": {
                "t_s": [float(second) for second in range(5)],
                "anon_sock_mb": [100 + level * second for second in range(5)],
            },
        },
    }


def write_summary(run_directory: Path, document: Mapping[str, Any]) -> bytes:
    run_directory.mkdir(parents=True)
    text = json.dumps(document, indent=2, sort_keys=True).encode()
    (run_directory / "summary.json").write_bytes(text)
    return text


def fact(run_ids: Sequence[str], value: float) -> Fact:
    return Fact(
        value=value if run_ids else None,
        unit="unit",
        runs=len(run_ids),
        min=value - 0.5 if run_ids else None,
        max=value + 0.5 if run_ids else None,
        run_ids=tuple(run_ids),
        dirty_runs=0,
        failed_checks={},
        attempts=0,
        failed_attempts=0,
    )


def write_facts(root: Path, counted: Counted, omit: str | None = None) -> None:
    facts: dict[str, Fact] = {}
    for key, keys in FACT_KEYS.items():
        for topology, prefix in TOPOLOGY_PREFIXES.items():
            run_ids = counted.get((key, topology), ())
            for index, fact_key in enumerate(keys):
                facts[prefix + fact_key] = fact(run_ids, 1.0 + index)
    facts.pop(omit or "", None)
    counted_ids = tuple(sorted(run_id for run_ids in counted.values() for run_id in run_ids))
    record = Facts(ModelSet.FAKE, CODE_DIGEST, counted_ids, facts)
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "facts.json").write_text(record.to_json())


def write_ledger(root: Path, run_ids: Mapping[str, Sequence[str]]) -> None:
    line = {
        "name": "fifo_vs_latest",
        "run_ids": {arm: list(run_ids.get(arm, ())) for arm in ARMS},
        "outcome": "accepted" if len(run_ids) == len(ARMS) else "not run",
    }
    (root / "ledger.jsonl").write_text(json.dumps(line, sort_keys=True) + "\n")


def write_predictions(root: Path) -> None:
    robots = list(range(1, 51))
    curves: dict[str, object] = {
        "robots": robots,
        "cpu_demand_cores": [0.04 * count for count in robots],
    }
    for model in MODEL_KEYS:
        curves[model] = {
            "commands_per_s_per_robot": [30 / count for count in robots],
            "freshness_ms_p50": [3.0 + count for count in robots],
            "freshness_ms_p95": [5.0 + count for count in robots],
            # The model's curve has no value where the simulation served nothing.
            "batch_size_mean": [None, *(1.0 + count / 10 for count in robots[1:])],
        }
    document = {"label": "synthetic", "inputs": {}, "facts": {}, "curves": curves}
    (root / "summary" / "predictions.json").write_text(json.dumps(document))


def write_results(
    root: Path,
    repeats: Sequence[int] = (1,),
    split: bool = False,
    arms: Sequence[str] = ARMS,
    predictions: bool = True,
) -> Counted:
    """Write a results root with counted runs of every scenario key; return the counted ids."""
    counted: dict[tuple[str, str], list[str]] = {}
    for topology in ("single", "split") if split else ("single",):
        for key, robot_ids in SCENARIO_ROBOTS.items():
            ids = [run_id(key, topology, repeat) for repeat in repeats]
            counted[key, topology] = ids
            for index, counted_id in enumerate(ids):
                level = 1 + index + (10 if topology == "split" else 0)
                write_summary(root / "raw" / key / counted_id, summary(robot_ids, level))
        counted["model_bench", topology] = [
            run_id("model_bench", topology, repeat) for repeat in repeats
        ]
    fifo_ids: dict[str, list[str]] = {}
    for index, arm in enumerate(arms):
        fifo_ids[arm] = [run_id("steady", repeat=repeat, variant=arm) for repeat in (1, 2)]
        for repeat_index, arm_id in enumerate(fifo_ids[arm]):
            arm_summary = summary(SCENARIO_ROBOTS["steady"], 30 * (index + 1) + repeat_index)
            write_summary(root / "raw" / "compare" / "fifo_vs_latest" / arm / arm_id, arm_summary)
    write_facts(root, counted)
    write_ledger(root, fifo_ids)
    if predictions:
        write_predictions(root)
    return counted


def failed_gate() -> Gate:
    """Return a gate whose limits were not the design's: one CPU instead of two."""
    expected = GatedLimits(2.0, 4 * 2**30, 0)
    digest = "sha256:" + "0" * 64
    return Gate(
        tests_passed=SuiteGate(
            True, digest, CODE_DIGEST, "commit", False, 1, 0, 0, 0, "gates/x.json", "0" * 64
        ),
        image_built=ImageGate(True, digest, {}),
        limits_ok=LimitsGate(False, expected, GatedLimits(1.0, 4 * 2**30, 0)),
        fresh=FreshGate(True, True, 0),
        config_ok=ConfigGate(True, ()),
    )


def finish(run_directory: Path, summary_bytes: bytes) -> None:
    done = Done(files={}, summary_sha256=hashlib.sha256(summary_bytes).hexdigest())
    (run_directory / "DONE").write_text(done.to_json())


def add_uncounted_runs(root: Path, counted: Counted) -> None:
    """Add, beside the counted runs of each serving key, runs that aggregation does not count.

    Each holds values that differ from the counted runs': a finished run whose gate failed, a
    finished run of other code, an attempt with a summary, and a hand run without one.
    """
    for key, robot_ids in SCENARIO_ROBOTS.items():
        directory = root / "raw" / key
        gate_failed = directory / run_id(key, variant="one cpu")
        finish(gate_failed, write_summary(gate_failed, summary(robot_ids, 99)))
        (gate_failed / "gate.json").write_text(failed_gate().to_json())
        other_code = directory / run_id(key, code=OTHER_CODE_DIGEST)
        finish(other_code, write_summary(other_code, summary(robot_ids, 77)))
        write_summary(directory / f"{counted[key, 'single'][0]}.attempt1", summary(robot_ids, 55))
        hand_run = directory / "manual"
        (hand_run / "plans").mkdir(parents=True)
        (hand_run / "samples").mkdir()
        hand_config = RunConfig(spec=None, run_id=None, seed=0, extra_compose_path=None, fleet=None)
        (hand_run / "config.json").write_text(hand_config.to_json())


def render(root: Path) -> dict[str, Plot]:
    return render_plots(load_inputs(root))


@pytest.fixture(scope="module")
def drawn(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Plot]:
    root = tmp_path_factory.mktemp("results")
    write_results(root)
    return render(root)


def test_every_plot_of_the_design_is_written_as_a_png(tmp_path: Path) -> None:
    write_results(tmp_path)

    assert main(["--results", str(tmp_path)]) == 0

    written = sorted(path.name for path in (tmp_path / "plots").iterdir())
    assert written == sorted(PLOT_FILES)
    assert all(
        (tmp_path / "plots" / name).read_bytes().startswith(PNG_SIGNATURE) for name in written
    )


@pytest.mark.parametrize("name", [name for name in PLOT_FILES if name not in NEVER_RUN])
def test_a_plot_with_counted_runs_is_drawn(drawn: dict[str, Plot], name: str) -> None:
    assert drawn[name].drawn


@pytest.mark.parametrize("name", NEVER_RUN)
def test_an_experiment_without_a_scenario_says_not_run(drawn: dict[str, Plot], name: str) -> None:
    assert not drawn[name].drawn


def test_every_plot_is_drawn_from_the_summaries_and_the_facts_that_the_pipeline_writes(
    evidence_root: Path,
) -> None:
    plots = render(evidence_root)

    assert [name for name, plot in plots.items() if not plot.drawn] == list(NEVER_RUN)


def test_the_same_inputs_give_the_same_bytes(tmp_path: Path, drawn: dict[str, Plot]) -> None:
    write_results(tmp_path)

    assert render(tmp_path) == drawn


def test_runs_beside_the_counted_ones_change_no_byte(
    tmp_path: Path, drawn: dict[str, Plot]
) -> None:
    counted = write_results(tmp_path)
    add_uncounted_runs(tmp_path, counted)

    assert render(tmp_path) == drawn


@pytest.mark.parametrize(("repeats", "expected_repeat"), [((2, 1), 1), ((10, 2), 2)])
def test_one_run_plots_draw_the_single_run_with_the_lowest_repeat(
    tmp_path: Path, repeats: tuple[int, ...], expected_repeat: int
) -> None:
    write_results(tmp_path, repeats=repeats, split=True)

    inputs = load_inputs(tmp_path)

    assert inputs is not None
    expected = {key: run_id(key, "single", expected_repeat) for key in SCENARIO_ROBOTS}
    assert {key: run.run_id for key, run in inputs.one_run.items()} == expected
    level = 1 + repeats.index(expected_repeat)
    assert inputs.one_run["steady"].document == summary(SCENARIO_ROBOTS["steady"], level)


def test_the_fifo_plot_draws_the_first_repeat_of_each_arm(tmp_path: Path) -> None:
    write_results(tmp_path)

    inputs = load_inputs(tmp_path)

    assert inputs is not None
    expected = {arm: run_id("steady", repeat=1, variant=arm) for arm in ARMS}
    assert {arm: run.run_id for arm, run in inputs.fifo_arms.items()} == expected


def test_every_plot_says_not_run_without_counted_runs(tmp_path: Path) -> None:
    write_facts(tmp_path, {})
    write_ledger(tmp_path, {})

    plots = render(tmp_path)

    assert sorted(plots) == sorted(PLOT_FILES)
    assert not any(plot.drawn for plot in plots.values())


def test_every_plot_says_not_run_in_a_results_root_without_facts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--results", str(tmp_path)]) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines == [f"plots/{name}: not run" for name in PLOT_FILES]
    assert sorted(path.name for path in (tmp_path / "plots").iterdir()) == sorted(PLOT_FILES)


def test_the_fifo_plot_is_not_run_without_both_arms(tmp_path: Path) -> None:
    write_results(tmp_path, arms=("a",))

    plots = render(tmp_path)

    assert not plots["fifo_vs_latest.png"].drawn
    assert plots["steady_freshness.png"].drawn


def test_every_plot_with_counted_runs_is_drawn_without_predictions(tmp_path: Path) -> None:
    write_results(tmp_path, predictions=False)

    plots = render(tmp_path)

    assert [name for name, plot in plots.items() if not plot.drawn] == list(NEVER_RUN)


@pytest.mark.parametrize(
    "key",
    [
        "sweep.n5.cls.stage.decode_ms.mean",
        "massive.depth.freshness_ms.p95",
        "steady.cls.commands_received",
    ],
)
def test_a_missing_fact_is_named(tmp_path: Path, key: str) -> None:
    counted = write_results(tmp_path)
    write_facts(tmp_path, counted, omit=key)

    with pytest.raises(PlotInputError, match=key.replace(".", r"\.")):
        render(tmp_path)


def test_a_counted_run_without_its_summary_is_named(tmp_path: Path) -> None:
    write_results(tmp_path)
    missing = run_id("massive")
    (tmp_path / "raw" / "massive" / missing / "summary.json").unlink()

    with pytest.raises(PlotInputError, match=missing):
        load_inputs(tmp_path)


def test_a_summary_without_a_field_a_plot_draws_is_named(tmp_path: Path) -> None:
    write_results(tmp_path)
    path = tmp_path / "raw" / "soak" / run_id("soak") / "summary.json"
    document = json.loads(path.read_text())
    del document["series"]["memory"]
    path.write_text(json.dumps(document))

    with pytest.raises(PlotInputError, match=r"series\.memory\.t_s is missing"):
        render(tmp_path)


def test_a_malformed_facts_file_fails_the_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary" / "facts.json").write_text("{")

    assert main(["--results", str(tmp_path)]) == 1
    assert "facts.json" in capsys.readouterr().err
    assert not (tmp_path / "plots").exists()


def test_the_stale_ratio_counts_the_commands_at_or_above_each_threshold() -> None:
    counts = np.array([2.0, 0.0, 1.0, 1.0])

    assert stale_ratio_curve(counts).tolist() == [1.0, 0.5, 0.5, 0.25]


def test_the_coverage_counts_the_time_below_each_limit_over_all_time() -> None:
    age_time_ms = np.array([10.0, 20.0, 30.0])

    assert coverage_curve(age_time_ms, 40.0).tolist() == pytest.approx([0.0, 0.1, 0.3, 0.6])


@pytest.mark.parametrize(
    "curve",
    [
        lambda: stale_ratio_curve(np.zeros(3)),
        lambda: coverage_curve(np.zeros(3), 0.0),
    ],
)
def test_a_curve_without_commands_or_time_is_empty(curve: Any) -> None:
    assert curve().size == 0


def test_the_rolling_p95_takes_the_trailing_window_of_each_sample() -> None:
    t_s = np.array([0.0, 0.5, 1.0, 2.5])
    values = np.array([1.0, 2.0, 3.0, 4.0])

    assert rolling_p95(t_s, values, 1.0).tolist() == pytest.approx([1.0, 1.95, 2.95, 4.0])


counts_lists = st.lists(st.integers(min_value=0, max_value=1000), min_size=1, max_size=50)
# Rounding in the cumulative sums may leave a share this far outside its exact value.
SHARE_TOLERANCE = 1e-12


@given(counts_lists.filter(lambda counts: sum(counts) > 0))
def test_the_stale_ratio_falls_from_one_and_stays_a_share(counts: list[int]) -> None:
    curve = stale_ratio_curve(np.array(counts, dtype=np.float64))

    assert curve.size == len(counts)
    assert curve[0] == pytest.approx(1.0)
    assert np.all((curve >= 0) & (curve <= 1 + SHARE_TOLERANCE))
    assert np.all(np.diff(curve) <= SHARE_TOLERANCE)


@given(counts_lists, st.integers(min_value=1, max_value=1000))
def test_the_coverage_rises_from_zero_to_the_covered_share(
    age_time_ms: list[int], uncovered_ms: int
) -> None:
    curve = coverage_curve(np.array(age_time_ms, dtype=np.float64), float(uncovered_ms))

    assert curve.size == len(age_time_ms) + 1
    assert curve[0] == 0
    assert np.all(np.diff(curve) >= -SHARE_TOLERANCE)
    assert curve[-1] == pytest.approx(sum(age_time_ms) / (sum(age_time_ms) + uncovered_ms))
