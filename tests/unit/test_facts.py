"""Tests for the fact keys: the placeholder, the registry, and the rules that combine runs."""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from bench.candidates import CANDIDATES
from bench.facts import (
    BREAKDOWN_STAGES,
    PACING_CONTROL,
    PACING_PRICE_PATH,
    PLACEHOLDER,
    RECOVERY_SHARE,
    REGISTRY,
    SCALE_POINTS,
    Combine,
    Inflection,
    ScalePoint,
    Source,
    bench_graph,
    breakdown_run,
    combine_rule,
    combined,
    gap_bound_run,
    inflection,
    metric_unit,
    placeholder_keys,
    recovery_s,
    settle_s,
    stored,
    useful_robots,
)
from cloud.config import Topology
from common.commands import MODEL_KEYS

REPOSITORY = Path(__file__).resolve().parents[2]
FRAME_RATE_HZ = 30.0
# The robot counts of the scale points: the sweep's four and the massive run's.
COUNTS = tuple(robot_count for _, robot_count in SCALE_POINTS)


def points(*utilizations: float | None) -> list[ScalePoint]:
    """Return the scale points with these utilizations and a capacity of 100 frames/s per robot."""
    return [
        ScalePoint(robot_count, utilization, 100.0 * robot_count)
        for robot_count, utilization in zip(COUNTS, utilizations, strict=True)
    ]


@pytest.mark.parametrize(
    ("text", "rendered"),
    [
        ("{{steady.cls.freshness_ms.p50}} ms", "X ms"),
        ("`{{steady.cls.freshness_ms.p50}}` ms", "X ms"),
        ("| `{{sweep.n5.cpu.utilization}}` / `{{sweep.n10.cpu.utilization}}` |", "| X / X |"),
        # A backtick on one side only is the edge of a longer code span, and stays.
        ("`{{env.kernel}} and more`", "`X and more`"),
        ("`kernel {{env.kernel}}`", "`kernel X`"),
        ("`a {{env.kernel}}` `{{env.engine}}`", "`a X` X"),
    ],
)
def test_a_placeholder_takes_its_backticks_only_when_it_has_both(text: str, rendered: str) -> None:
    assert PLACEHOLDER.sub("X", text) == rendered


@pytest.mark.parametrize(
    "text",
    ["{{Steady.cls}}", "{{1steady}}", "{{steady cls}}", "{{}}", "{steady.cls}", "{{steady.cls}"],
)
def test_text_that_is_not_a_lowercase_key_in_double_braces_is_no_placeholder(text: str) -> None:
    assert PLACEHOLDER.search(text) is None


def test_placeholder_keys_lists_each_key_once_in_order_of_appearance() -> None:
    text = "`{{b.two}}` then {{a.one}} and `{{b.two}}` again, `{{c_3.x}} in a span`"

    assert placeholder_keys(text) == ["b.two", "a.one", "c_3.x"]


@pytest.mark.parametrize(
    ("key", "rule"),
    [
        ("steady.cls.freshness_ms.p95", Combine.MEDIAN),
        ("steady.cls.freshness_ms.max", Combine.LARGEST),
        ("massive.depth.command_gap_ms.max", Combine.LARGEST),
        ("massive.cls.commands_per_s_per_robot.max", Combine.LARGEST),
        ("massive.cls.commands_per_s_per_robot.min", Combine.SMALLEST),
        ("split.massive.depth.watchdog_coverage.min", Combine.SMALLEST),
        ("soak.cls.fairness.min_over_mean", Combine.SMALLEST),
        ("massive.cls.commands_per_s_per_robot.mean", Combine.MEDIAN),
        ("massive.cls.fairness.jain", Combine.MEDIAN),
    ],
)
def test_a_key_combines_its_runs_by_the_rule_of_its_statistic(key: str, rule: Combine) -> None:
    assert REGISTRY[key].combine is rule
    assert combine_rule(key) is rule


@pytest.mark.parametrize(
    ("key", "values", "expected"),
    [
        # Odd and even counts of runs: the middle value, and the mean of the two middle ones.
        ("steady.cls.freshness_ms.p95", [9.0, 1.0, 4.0], 4.0),
        ("steady.cls.freshness_ms.p95", [9.0, 1.0, 4.0, 5.0], 4.5),
        # The extreme is not the middle run's value in any of these.
        ("steady.cls.freshness_ms.max", [9.0, 1.0, 4.0], 9.0),
        ("massive.cls.commands_per_s_per_robot.min", [9.0, 1.0, 4.0], 1.0),
        ("massive.cls.fairness.min_over_mean", [0.9, 0.5, 0.7], 0.5),
    ],
)
def test_run_values_combine_by_the_rule_of_the_key(
    key: str, values: list[float], expected: float
) -> None:
    assert combined(values, REGISTRY[key]) == expected


def test_a_whole_number_key_stores_the_median_of_equal_integers_as_an_int() -> None:
    value = combined([50, 50], REGISTRY["steady.cls.commands_received"])

    assert (value, type(value)) == (50, int)


def test_a_whole_number_key_keeps_a_median_that_is_not_integral_as_a_float() -> None:
    value = combined([50, 51], REGISTRY["steady.cls.commands_received"])

    assert (value, type(value)) == (50.5, float)


def test_a_measured_key_stores_an_integral_value_as_a_float() -> None:
    value = combined([3, 3], REGISTRY["steady.cls.freshness_ms.p95"])

    assert (value, type(value)) == (3.0, float)


def test_combining_no_run_values_is_refused() -> None:
    with pytest.raises(ValueError, match=r"steady\.cls\.freshness_ms\.p95"):
        combined([], REGISTRY["steady.cls.freshness_ms.p95"])


@pytest.mark.parametrize(
    ("value", "integer", "expected", "kind"),
    [(7.0, True, 7, int), (7.5, True, 7.5, float), (7, False, 7.0, float), (7, True, 7, int)],
)
def test_stored_keeps_the_kind_of_number_its_key_has(
    value: float, integer: bool, expected: float, kind: type
) -> None:
    result = stored(value, integer)

    assert (result, type(result)) == (expected, kind)


@pytest.mark.parametrize(
    ("e2e_mean_ms", "chosen"),
    [
        ({"run-1": 30.0, "run-2": 10.0, "run-3": 20.0}, "run-3"),
        # An even count: the lower of the two middle runs.
        ({"run-1": 30.0, "run-2": 10.0}, "run-2"),
        ({"run-1": 40.0, "run-2": 10.0, "run-3": 20.0, "run-4": 30.0}, "run-3"),
        # Two runs share the median: the smaller run id.
        ({"run-3": 20.0, "run-2": 20.0, "run-1": 10.0, "run-4": 30.0}, "run-2"),
        ({"run-1": 5.0}, "run-1"),
    ],
)
def test_the_breakdown_comes_from_the_run_with_the_median_end_to_end_mean(
    e2e_mean_ms: dict[str, float], chosen: str
) -> None:
    assert breakdown_run(e2e_mean_ms) == chosen


def test_a_breakdown_without_a_run_is_refused() -> None:
    with pytest.raises(ValueError, match="no run"):
        breakdown_run({})


@given(means=st.lists(st.integers(min_value=0, max_value=5), min_size=1, max_size=9))
def test_the_breakdown_run_has_the_lower_middle_rank_and_the_smallest_id_at_it(
    means: list[int],
) -> None:
    e2e_mean_ms = {f"run-{index}": float(mean) for index, mean in enumerate(means)}

    chosen = breakdown_run(e2e_mean_ms)

    chosen_ms = e2e_mean_ms[chosen]
    below = sum(mean_ms < chosen_ms for mean_ms in e2e_mean_ms.values())
    at_or_below = sum(mean_ms <= chosen_ms for mean_ms in e2e_mean_ms.values())
    assert below <= (len(means) - 1) // 2 < at_or_below
    assert chosen == min(run_id for run_id, mean_ms in e2e_mean_ms.items() if mean_ms == chosen_ms)


@pytest.mark.parametrize(
    ("gap_max_ms", "chosen"),
    [
        ({"run-1": 9396.0, "run-2": 10686.0, "run-3": 9473.0}, "run-2"),
        # Two runs share the longest gap: the smaller run id.
        ({"run-3": 300.0, "run-2": 300.0, "run-1": 200.0}, "run-2"),
        ({"run-1": 5.0}, "run-1"),
    ],
)
def test_the_gap_bound_comes_from_the_run_with_the_longest_gap(
    gap_max_ms: dict[str, float], chosen: str
) -> None:
    assert gap_bound_run(gap_max_ms) == chosen


def test_a_gap_bound_without_a_run_is_refused() -> None:
    with pytest.raises(ValueError, match="no run"):
        gap_bound_run({})


@given(
    gaps=st.lists(
        st.tuples(st.floats(min_value=0.0, max_value=1e5), st.floats(min_value=0.0, max_value=1e5)),
        min_size=1,
        max_size=9,
    )
)
def test_the_longest_gap_is_never_shown_above_the_bound_when_every_run_met_its_own(
    gaps: list[tuple[float, float]],
) -> None:
    gap_max_ms = {f"run-{index}": gap_ms for index, (gap_ms, _) in enumerate(gaps)}
    bound_ms = {f"run-{index}": gap_ms + slack_ms for index, (gap_ms, slack_ms) in enumerate(gaps)}

    chosen = gap_bound_run(gap_max_ms)

    assert gap_max_ms[chosen] == max(gap_max_ms.values())
    assert gap_max_ms[chosen] <= bound_ms[chosen]


@given(values=st.lists(st.floats(min_value=-1e6, max_value=1e6), min_size=1, max_size=9))
def test_a_combined_value_lies_within_the_run_values_and_an_extreme_is_one_of_them(
    values: list[float],
) -> None:
    median, largest, smallest = (
        combined(values, REGISTRY[key])
        for key in (
            "steady.cls.freshness_ms.p95",
            "steady.cls.freshness_ms.max",
            "massive.cls.commands_per_s_per_robot.min",
        )
    )

    assert min(values) <= median <= max(values)
    assert (largest, smallest) == (max(values), min(values))


@given(
    peaks_ms=st.lists(
        st.floats(min_value=0.0, max_value=2000.0), min_size=len(COUNTS), max_size=len(COUNTS)
    )
)
def test_useful_robots_is_the_end_of_the_run_of_counts_within_the_limit_from_one_robot(
    peaks_ms: list[float],
) -> None:
    limit_ms = 1000.0
    points = list(zip(COUNTS, peaks_ms, strict=True))

    within = useful_robots(points, limit_ms)

    assert within is not None
    assert all(peak_ms <= limit_ms for count, peak_ms in points if count <= within)
    assert next((peak_ms for count, peak_ms in points if count > within), math.inf) > limit_ms


@pytest.mark.parametrize(
    ("utilizations", "expected"),
    [
        # Saturated at the first point: the capacity there, and no lower measured count.
        (
            (1.2, 3.0, 5.0, 10.0, 50.0),
            Inflection(100.0, 100.0 / FRAME_RATE_HZ, "below 1 robot", 1),
        ),
        # At a middle point of the sweep.
        (
            (0.2, 0.6, 1.0, 2.0, 10.0),
            Inflection(500.0, 500.0 / FRAME_RATE_HZ, "between 3 and 5 robots", 5),
        ),
        # Only at the massive run.
        (
            (0.1, 0.2, 0.3, 0.6, 3.0),
            Inflection(5000.0, 5000.0 / FRAME_RATE_HZ, "between 10 and 50 robots", 50),
        ),
        # At no point, and with a point whose utilization is null.
        ((0.1, 0.2, 0.3, 0.6, 0.9), None),
        ((0.2, 0.6, None, 2.0, 10.0), None),
        ((None, 0.6, 1.5, 2.0, 10.0), None),
    ],
)
def test_the_inflection_is_at_the_lowest_saturated_scale_point(
    utilizations: tuple[float | None, ...], expected: Inflection | None
) -> None:
    assert inflection(points(*utilizations), FRAME_RATE_HZ) == expected


def test_a_scale_point_without_a_capacity_gives_no_inflection() -> None:
    measured = points(0.2, 0.6, 1.2, 2.0, 10.0)
    measured[4] = ScalePoint(COUNTS[4], 10.0, None)

    assert inflection(measured, FRAME_RATE_HZ) is None


@pytest.mark.parametrize(
    ("peaks_ms", "expected"),
    [
        # Every point within the limit of 300 ms: the largest count measured.
        ((50.0, 100.0, 150.0, 200.0, 300.0), 50),
        # A middle point above it.
        ((50.0, 100.0, 301.0, 400.0, 900.0), 3),
        # One robot is already above it.
        ((301.0, 400.0, 500.0, 600.0, 900.0), 0),
        # A point within the limit above one that is not does not count.
        ((50.0, 400.0, 150.0, 200.0, 250.0), 1),
        # A point with no counted run has no median.
        ((50.0, 100.0, None, 200.0, 250.0), None),
        ((50.0, 400.0, 500.0, 600.0, None), None),
    ],
)
def test_useful_robots_is_the_largest_count_measured_within_the_watchdog_limit(
    peaks_ms: tuple[float | None, ...], expected: int | None
) -> None:
    assert useful_robots(list(zip(COUNTS, peaks_ms, strict=True)), 300.0) == expected


@pytest.mark.parametrize(
    ("first_counts", "second_counts", "expected"),
    [
        # Equal from the first bin: one second, the resolution of the bins.
        ([15, 15, 15], [15, 15, 15], 1),
        # A difference of one command is counting.
        ([15, 16, 15], [16, 15, 15], 1),
        # Below ten commands a bin, a tenth of the mean is under one command, and one
        # command is still the tolerance.
        ([8, 8, 8], [7, 7, 7], 1),
        # A difference equal to a tenth of the mean, of 105 and 95, is within the tolerance.
        ([105, 105, 105], [95, 95, 95], 1),
        # The joiner catches up in its fourth second.
        ([30, 25, 20, 15, 15], [0, 5, 10, 15, 15], 4),
        # Above one command, a tenth of the mean count is the tolerance: of a mean of 32 a
        # difference of 4 is outside it, and of 31.5 a difference of 3 is inside.
        ([34, 33, 30], [30, 30, 30], 2),
        # A bin that differs later starts the count again after it.
        ([15, 15, 30, 15], [15, 15, 0, 15], 4),
        # Not equal in the last bin, and no bin at all: no answer.
        ([15, 15, 30], [15, 15, 0], None),
        ([], [], None),
    ],
)
def test_two_robots_settle_at_the_first_bin_from_which_their_counts_stay_equal(
    first_counts: list[int], second_counts: list[int], expected: int | None
) -> None:
    assert settle_s(first_counts, second_counts) == expected


def test_settle_refuses_series_of_different_lengths() -> None:
    with pytest.raises(ValueError):
        settle_s([1, 2], [1])


@pytest.mark.parametrize(
    ("counts", "expected"),
    [
        # Three robots at 5 per second each: the two survivors are predicted 7.5, and have
        # recovered at 0.9 of it, 6.75.
        ([5.0, 6.0, 7.0, 7.5], 3),
        ([6.75, 5.0], 1),
        ([5.0, 6.5, 6.74], None),
        ([], None),
    ],
)
def test_the_survivors_recover_at_the_first_bin_near_the_predicted_rate(
    counts: list[float], expected: int | None
) -> None:
    assert RECOVERY_SHARE * 3 / 2 * 5.0 == 6.75
    assert recovery_s(counts, 5.0, 3) == expected


@pytest.mark.parametrize(
    ("model", "threads", "batch_size", "graph"),
    [
        ("cls", 1, 1, "published"),
        ("cls", 2, 8, "published"),
        ("depth", 1, 1, "published"),
        ("depth", 1, 2, "batched"),
        ("depth", 2, 1, "batched"),
        ("depth", 2, 8, "batched"),
    ],
)
def test_only_the_one_thread_cap_of_one_reads_the_published_depth_graph(
    model: str, threads: int, batch_size: int, graph: str
) -> None:
    assert bench_graph(model, threads, batch_size) == graph
    assert REGISTRY[f"model_bench.{model}.t{threads}.b{batch_size}.wall_ms"].path == (
        "tables",
        model,
        f"t{threads}",
        graph,
        str(batch_size),
        "wall_ms",
    )


def test_the_batched_depth_graph_is_also_read_at_the_one_thread_cap_of_one() -> None:
    entry = REGISTRY["model_bench.depth.batched.t1.b1.wall_ms"]

    assert (entry.path, entry.unit, entry.combine) == (
        ("tables", "depth", "t1", "batched", "1", "wall_ms"),
        "ms",
        Combine.MEDIAN,
    )


@pytest.mark.parametrize(
    ("path", "unit"),
    [
        ("models.depth.all.freshness_trend_ms_per_s", "ms/s"),
        ("models.cls.all.freshness_ms.p95", "ms"),
        ("cpu.all.throttled_ms_per_s", "ms/s"),
        ("models.cls.all.commands_per_s_per_robot.mean", "commands/s"),
        ("models.cls.all.batch_size.mean", "frames"),
        ("models.cls.all.batch_size.share_1", "ratio"),
        ("models.cls.all.fairness.min_over_mean", "ratio"),
        ("validity.fire_lag_ms.p99", "ms"),
    ],
)
def test_the_unit_of_a_summary_path_is_that_of_its_metric(path: str, unit: str) -> None:
    assert metric_unit(path) == unit


def test_a_path_that_ends_in_no_known_metric_has_no_unit() -> None:
    with pytest.raises(KeyError, match=r"models\.cls\.all\.robots"):
        metric_unit("models.cls.all.robots")


@pytest.mark.parametrize("template", ["DESIGN.tmpl.md", "README.tmpl.md"])
def test_every_placeholder_of_a_template_is_a_key_of_the_registry(template: str) -> None:
    text = (REPOSITORY / "docs" / template).read_text(encoding="utf-8")

    assert [key for key in placeholder_keys(text) if key not in REGISTRY] == []


def test_every_key_matches_the_placeholder_grammar_and_names_itself() -> None:
    unmatched = [
        key
        for key, entry in REGISTRY.items()
        if placeholder_keys(f"`{{{{{key}}}}}`") != [key] or entry.key != key
    ]

    assert unmatched == []


def test_every_key_of_a_serving_scenario_has_a_twin_in_the_split_topology() -> None:
    single = {key for key, entry in REGISTRY.items() if entry.topology is Topology.SINGLE}
    split = {key for key, entry in REGISTRY.items() if entry.topology is Topology.SPLIT}
    serving = {key for key in single if not key.startswith("model_bench.")}

    assert split == {f"split.{key}" for key in serving}
    assert all(
        REGISTRY[f"split.{key}"].path == REGISTRY[key].path
        and REGISTRY[f"split.{key}"].scenario_key == REGISTRY[key].scenario_key
        for key in serving
    )


def test_the_registry_holds_every_key_family_of_the_design_document() -> None:
    families = [
        "steady.cls.freshness_ms.p99",
        "steady.depth.commands_received",
        "steady.cls.freshness_trend_ms_per_s",
        "late_joiner.before.a.cls.commands_per_s",
        "late_joiner.after.b.depth.freshness_ms.p95",
        "late_joiner.depth.settle_s",
        "late_joiner.b.depth.first_command_ms",
        "sweep.n10.cls.commands_per_s_per_robot.max",
        "sweep.n3.depth.batch_size.share_1",
        "sweep.n1.cls.utilization",
        "sweep.n5.cpu.throttled_ms_per_s",
        "sweep.cls.capacity_fps",
        "sweep.depth.inflection_n",
        "sweep.depth.inflection_between",
        "sweep.n5.cls.stage.fire_lag_ms.p50",
        "sweep.n5.depth.stage.deliver_ms.p95",
        "sweep.n5.depth.stage_sum_ms.mean",
        "sweep.n5.cls.e2e_ms.mean",
        "sweep.n5.cls.residual_ms.mean",
        "disconnect.after.survivors.depth.commands_per_s_per_robot",
        "disconnect.before.survivors.cls.freshness_ms.p95",
        "disconnect.depth.recovery_s",
        "disconnect.detect_ms",
        "worker_kill.depth.outage_ms",
        "worker_kill.after.cls.commands_per_s_per_robot.mean",
        "massive.cls.commands_received",
        "massive.depth.fairness.jain",
        "massive.depth.command_gap_ms.max",
        "massive.depth.command_gap_bound_ms",
        "massive.cls.watchdog_coverage.min",
        "massive.cls.useful_robots",
        "massive.stage.transport_up_ms.p99",
        "massive.stage.dispatch_rt_ms.p50",
        "massive.cpu.utilization",
        "massive.bytes_in_per_s",
        "massive.connections.rejected",
        "massive.gateway.cpu_ms_per_frame",
        "massive.gateway.loop_lag_ms.p99",
        "massive.fleet.rss_mb",
        "massive.fleet.fire_lag_ms.p99",
        "massive.worker_depth.rss_mb",
        "massive.memory.anon_sock_mb_start",
        "soak.memory.slope_mb_per_min",
        "soak.fds.delta",
        "soak.tasks.delta",
        "soak.depth.fairness.min_over_mean",
        "soak.cls.commands_per_s_per_robot.min",
        "model_bench.depth.t2.b8.cpu_ms",
        "model_bench.cls.batch_cap",
        "model_bench.depth.margin_ms",
        "model_bench.decode.half_ms",
        "model_bench.cls.preprocess_ms",
        "model_bench.depth.postprocess_ms",
        "model_bench.dispatch_rt_ms",
        "env.cpu_model",
        "env.kernel",
        "env.engine",
        "env.repeats",
        "split.massive.fleet.cpu_cores",
    ]

    assert [key for key in families if key not in REGISTRY] == []


def test_the_worker_kill_keys_read_the_outage_the_kill_and_each_models_rate_around_it() -> None:
    keys = {
        key: (entry.topology, entry.source, entry.path, entry.unit, entry.combine, entry.integer)
        for key, entry in REGISTRY.items()
        if entry.scenario_key == "worker_kill"
    }

    rate = ("commands_per_s_per_robot", "mean")
    assert keys == {
        **{
            f"{prefix}worker_kill.depth.outage_ms": (
                topology,
                Source.SUMMARY,
                ("fault", "outage_ms"),
                "ms",
                Combine.MEDIAN,
                False,
            )
            for prefix, topology in (("", Topology.SINGLE), ("split.", Topology.SPLIT))
        },
        # The latest kill of the runs bounds how late every kill was.
        **{
            f"{prefix}worker_kill.depth.kill_late_ms.max": (
                topology,
                Source.SUMMARY,
                ("fault", "late_ms"),
                "ms",
                Combine.LARGEST,
                False,
            )
            for prefix, topology in (("", Topology.SINGLE), ("split.", Topology.SPLIT))
        },
        **{
            f"{prefix}worker_kill.{window}.{model}.commands_per_s_per_robot.mean": (
                topology,
                Source.SUMMARY,
                ("models", model, window, *rate),
                "commands/s",
                Combine.MEDIAN,
                False,
            )
            for prefix, topology in (("", Topology.SINGLE), ("split.", Topology.SPLIT))
            for window in ("before", "after")
            for model in MODEL_KEYS
        },
    }
    assert len(keys) == 12


def test_the_breakdown_holds_three_statistics_of_each_of_the_ten_stages_per_model() -> None:
    breakdown = [entry for entry in REGISTRY.values() if entry.source is Source.BREAKDOWN]
    stage_keys = {
        f"sweep.n5.{model}.stage.{stage}_ms.{statistic}"
        for model in MODEL_KEYS
        for stage in BREAKDOWN_STAGES
        for statistic in ("mean", "p50", "p95")
    }
    totals = {
        f"sweep.n5.{model}.{total}.mean"
        for model in MODEL_KEYS
        for total in ("stage_sum_ms", "e2e_ms", "residual_ms")
    }

    assert len(BREAKDOWN_STAGES) == 10
    assert {entry.key for entry in breakdown if entry.topology is Topology.SINGLE} == (
        stage_keys | totals
    )
    assert {entry.scenario_key for entry in breakdown} == {"sweep_n5"}


def test_every_candidate_has_its_three_ledger_keys_and_pacing_its_price_per_arm() -> None:
    ledger = {
        key: (entry.path, entry.unit)
        for key, entry in REGISTRY.items()
        if entry.source is Source.LEDGER
    }

    assert ledger == {
        **{
            f"ledger.{candidate.name}.{field}": ((candidate.name, field), unit)
            for candidate in CANDIDATES
            for field, unit in (
                ("outcome", ""),
                ("main_delta", metric_unit(candidate.main.path)),
                ("baseline_spread", metric_unit(candidate.main.path)),
            )
        },
        **{
            f"ledger.depth_unpaced.depth_rate.{arm}_median": (
                (
                    "depth_unpaced",
                    "metrics",
                    "models.depth.all.commands_per_s_per_robot.mean",
                    f"{arm}_median",
                ),
                "commands/s",
            )
            for arm in ("a", "b")
        },
    }


def test_the_control_that_turns_pacing_off_guards_the_depth_rate_its_price_is_read_from() -> None:
    (control,) = (candidate for candidate in CANDIDATES if candidate.name == PACING_CONTROL)

    assert PACING_PRICE_PATH in {guard.path for guard in control.guards}


@pytest.mark.parametrize(
    "key",
    [
        "steady.cls.commands_received",
        "massive.depth.stale_count",
        "steady.depth.out_of_order_count",
        "massive.connections.accepted",
        "massive.connections.rejected",
        "soak.fds.delta",
        "soak.tasks.delta",
        "model_bench.cls.batch_cap",
        "massive.cls.useful_robots",
        "late_joiner.depth.settle_s",
        "disconnect.cls.recovery_s",
        "env.repeats",
    ],
)
def test_counts_caps_and_times_read_from_bins_are_whole_number_keys(key: str) -> None:
    assert REGISTRY[key].integer


@pytest.mark.parametrize(
    "key",
    [
        "steady.cls.commands_per_s",
        "sweep.cls.inflection_n",
        "disconnect.detect_ms",
        "massive.cls.stale_ratio",
        "model_bench.depth.margin_ms",
        "late_joiner.b.cls.first_command_ms",
    ],
)
def test_measured_times_rates_and_ratios_are_not_whole_number_keys(key: str) -> None:
    assert not REGISTRY[key].integer


@pytest.mark.parametrize(
    ("key", "unit"),
    [
        ("massive.cls.stale_ratio", "ratio"),
        ("massive.cls.fairness.jain", "ratio"),
        ("massive.depth.watchdog_coverage.mean", "ratio"),
        ("sweep.n5.cpu.utilization", "ratio"),
        ("late_joiner.depth.settle_s", "s (1 s bins)"),
        ("disconnect.depth.recovery_s", "s (1 s bins)"),
        ("sweep.cls.inflection_n", "robots"),
        ("sweep.cls.inflection_between", ""),
        ("predict.steady.depth.freshness_ms.p95", "ms"),
        ("predict.massive.cls.commands_per_s_per_robot.mean", "commands/s"),
        ("massive.memory.sock_mb", "MB"),
    ],
)
def test_a_key_has_the_unit_its_value_is_rendered_with(key: str, unit: str) -> None:
    assert REGISTRY[key].unit == unit
