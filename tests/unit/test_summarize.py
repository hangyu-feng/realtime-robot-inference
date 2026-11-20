"""Tests for the per-run summary: each quantity against an answer worked out by hand."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from bench.candidates import CANDIDATES, Candidate
from bench.runfiles import (
    DONE_FILE,
    EVENTS_FILE,
    SAMPLES_DIRECTORY,
    SUMMARY_FILE,
    Done,
    Event,
    EventName,
    ModelSet,
)
from bench.runplan import compare_plan, evidence_plan
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
from bench.summarize import (
    MEMORY_SLOPE_SKIP_S,
    REPORTS_DIRECTORY,
    REPOSITORY_ROOT,
    SummaryError,
    age_time_ms,
    command_gaps,
    dispatch_times_ns,
    fairness,
    in_window,
    least_squares_slope,
    main,
    measured_cycle_ns,
    module_files,
    peak_ages_ns,
    summarize_model_bench,
    summarize_run,
    summarizer_digest,
    summary_of,
    watchdog_coverage,
)
from cloud.config import Topology
from common.commands import MODEL_KEYS
from common.histogram import Histogram
from common.protocol import STAGE_NAMES
from robot.control import Verdict
from robot.samples import VERDICTS, TickOutcome
from tests.support.runs import (
    CODE_DIGEST,
    FRAME_PERIOD_NS,
    KILL_LATE_NS,
    NS_PER_MS,
    NS_PER_S,
    OTHER_CODE_DIGEST,
    RESTARTED_PID,
    T0_NS,
    WORKER_OUTAGE_NS,
    SyntheticRun,
    command_rows,
    edit_metrics,
    failed_scrape,
    histogram,
    merged,
    model_bench_reports,
    run_directory,
    run_spec,
    store_summary,
    synthetic_run,
    tick_rows,
    worker_kill_run,
    write_attempt,
    write_finished_run,
    write_hand_run,
    write_model_bench_run,
)

type Edit = Callable[[int | None, dict[str, Any]], None]

MS = NS_PER_MS
S = NS_PER_S


def at_ns(t_s: float) -> int:
    """Return a time of a synthetic run, given in seconds from its t0."""
    return T0_NS + round(t_s * S)


def always(k: int | None) -> bool:
    return True


def final(k: int | None) -> bool:
    return k is None


def from_second(second: int) -> Callable[[int | None], bool]:
    """Match the scrapes planned at or after a second, and the final one."""
    return lambda k: k is None or k >= second


def setting(values: Mapping[str, object], when: Callable[[int | None], bool] = always) -> Edit:
    """Return an edit that puts each value at its dotted path, in the scrapes that when matches.

    A value that is a function is called with the scrape's planned second.
    """

    def edit(k: int | None, metrics: dict[str, Any]) -> None:
        if not when(k):
            return
        for path, value in values.items():
            *parents, last = path.split(".")
            target = metrics
            for key in parents:
                target = target[key]
            target[last] = value(k) if callable(value) else value

    return edit


def adding(values: Mapping[str, int], when: Callable[[int | None], bool] = always) -> Edit:
    """Return an edit that adds to the counter at each dotted path, where when matches."""

    def edit(k: int | None, metrics: dict[str, Any]) -> None:
        if not when(k):
            return
        for path, amount in values.items():
            *parents, last = path.split(".")
            target = metrics
            for key in parents:
                target = target[key]
            target[last] += amount

    return edit


def check(summary: Mapping[str, Any], name: str) -> Any:
    (found,) = (entry for entry in summary["checks"] if entry["name"] == name)
    return found


def with_samples(run: SyntheticRun, robot_id: str, *commands: Any) -> SyntheticRun:
    """Return the run with one robot's commands replaced; its ticks stay."""
    robot = run.samples[robot_id]
    return replace(
        run, samples={**run.samples, robot_id: replace(robot, commands=merged(*commands))}
    )


def steady_run() -> SyntheticRun:
    return synthetic_run(steady(), run_spec("steady"))


def massive_run(overrides: Mapping[str, Any] | None = None) -> SyntheticRun:
    return synthetic_run(massive(), run_spec("massive", overrides=overrides))


SHORT_SOAK = Scenario(
    "soak",
    10.0,
    (RobotPlan("r01", 0.0, None), RobotPlan("r02", 0.0, None)),
    (Window("all", 0.0, 10.0),),
)


@pytest.mark.parametrize(
    ("time_ns", "inside"), [(99, False), (100, True), (199, True), (200, False)]
)
def test_a_window_holds_its_start_and_not_its_end(time_ns: int, inside: bool) -> None:
    assert in_window(np.array([time_ns]), 100, 200).tolist() == [inside]


def test_a_command_belongs_to_the_window_that_holds_its_receive_time() -> None:
    run = synthetic_run(late_joiner(), run_spec("late_joiner"))
    # Captured in the window before, and received at the instant the window after starts.
    receive = [at_ns(10.0) - 1, at_ns(10.0)]
    run = with_samples(run, "a", command_rows("cls", [at_ns(9.9), at_ns(9.95)], receive))

    models = summary_of(run.run)["models"]

    assert models["cls"]["before"]["commands_received"] == 1
    # The command received at the instant the window starts, and b's 600.
    assert models["cls"]["after"]["commands_received"] == 1 + 600


def test_a_counter_rate_is_over_the_scrape_span_and_a_command_rate_over_the_window() -> None:
    summary: Any = summary_of(steady_run().run)

    # The scrapes of seconds 0 and 29 bound the 30 s window, and both come 2 ms late.
    assert summary["windows"]["all"] == {"start_s": 0.0, "end_s": 30.0, "span_s": 29.0}
    # 30 frames of 40 000 bytes each second.
    assert summary["gateway"]["all"]["bytes_in_per_s"] == pytest.approx(1_200_000)
    assert summary["cpu"]["all"]["cores"]["worker_cls"] == pytest.approx(0.15)
    assert summary["models"]["cls"]["all"]["commands_per_s"] == pytest.approx(900 / 30)


def test_failed_scrapes_shorten_the_span_and_leave_the_rates() -> None:
    run = steady_run()
    scrapes = tuple(
        failed_scrape(line.k, at_ns(line.k)) if line.k in (0, 1, 2) else line
        for line in run.scrapes
    )

    summary: Any = summary_of(replace(run, scrapes=scrapes).run)

    assert summary["windows"]["all"]["span_s"] == 26.0
    assert summary["gateway"]["all"]["bytes_in_per_s"] == pytest.approx(1_200_000)
    assert summary["validity"]["failed_scrapes"] == 3


@pytest.mark.parametrize(
    ("scenario", "key", "second", "span_s"),
    [
        # The window before ends where robot c is killed, so a scrape at that instant is out.
        (disconnect(), "disconnect", 20, 19.0),
        # No robot is killed where this window ends, so a scrape at its end bounds it.
        (late_joiner(), "late_joiner", 10, 10.0 - 0.002),
    ],
)
def test_a_window_that_ends_at_a_kill_takes_its_last_scrape_before_it(
    scenario: Scenario, key: str, second: int, span_s: float
) -> None:
    run = synthetic_run(scenario, run_spec(key))
    run = edit_metrics(run, setting({"ts_mono_ns": at_ns(second)}, lambda k: k == second))

    summary: Any = summary_of(run.run)

    assert summary["windows"]["before"]["span_s"] == pytest.approx(span_s)


def test_gaps_count_the_open_intervals_at_both_ends() -> None:
    gaps = command_gaps(np.array([1 * S, 2 * S, 4 * S]), 0, 30 * S)

    assert gaps.first_ns == 1 * S
    assert gaps.closed_ns.tolist() == [1 * S, 2 * S]
    assert gaps.last_ns == 26 * S
    assert gaps.longest_ns == 26 * S


def test_a_robot_without_a_command_has_one_gap_of_its_whole_time() -> None:
    gaps = command_gaps(np.empty(0, np.int64), 5 * S, 30 * S)

    assert (gaps.first_ns, gaps.closed_ns.size, gaps.last_ns) == (25 * S, 0, 25 * S)


def test_a_peak_age_runs_to_the_next_command_and_the_last_to_the_end() -> None:
    capture = np.array([900 * MS, 1_900 * MS, 3_900 * MS])
    receive = np.array([1_000 * MS, 2_000 * MS, 4_000 * MS])

    peaks = peak_ages_ns(capture, receive, 30 * S)

    assert peaks.tolist() == [1_100 * MS, 2_100 * MS, 26_100 * MS]


def test_the_summary_reports_the_open_intervals_of_the_gap_and_of_the_peak_age() -> None:
    receive = [at_ns(1.0), at_ns(2.0), at_ns(4.0)]
    capture = [time - 100 * MS for time in receive]
    run = with_samples(steady_run(), "a", command_rows("cls", capture, receive))

    summary: Any = summary_of(run.run)
    window = summary["models"]["cls"]["all"]

    assert window["command_gap_ms"] == pytest.approx({"p50": 1500.0, "p95": 1950.0, "max": 26000.0})
    assert window["first_gap_ms"] == {"max": pytest.approx(1000.0)}
    assert window["command_age_peak_ms"]["max"] == pytest.approx(26100.0)
    assert window["command_age_peak_ms"]["p50"] == pytest.approx(2100.0)
    robot = summary["robots"]["a"]["cls"]["all"]
    assert robot["first_gap_ms"] == pytest.approx(1000.0)
    assert robot["command_gap_ms"] == {"max": pytest.approx(26000.0)}


def served_late(run: SyntheticRun, robot_id: str, delay_s: float) -> SyntheticRun:
    """Return the run without the robot's cls commands received in delay_s after its start."""
    (plan,) = (plan for plan in run.fleet.scenario.robots if plan.robot_id == robot_id)
    rows = run.samples[robot_id].commands
    early = (rows["model"] == MODEL_KEYS.index("cls")) & (
        rows["receive_ns"] < at_ns(plan.start_s + delay_s)
    )
    return with_samples(run, robot_id, rows[~early])


def test_the_largest_first_gap_of_a_window_is_over_the_robots_present_for_all_of_it() -> None:
    run = served_late(synthetic_run(late_joiner(), run_spec("late_joiner")), "b", 0.5)

    summary: Any = summary_of(run.run)
    cls = summary["models"]["cls"]

    # b joins at 10 s, and its first cls command is that of its 16th frame, 20 ms old.
    b_first_gap_ms = (15 * FRAME_PERIOD_NS + 20 * MS) / MS
    assert summary["robots"]["b"]["cls"]["all"]["first_gap_ms"] == pytest.approx(b_first_gap_ms)
    # Over the whole run only a is present, and its first command comes 20 ms after t0.
    assert cls["all"]["first_gap_ms"] == {"max": pytest.approx(20.0)}
    assert cls["after"]["first_gap_ms"] == {"max": pytest.approx(b_first_gap_ms)}


def test_coverage_leaves_out_the_start_and_the_time_after_a_command_expires() -> None:
    # Ages at receipt of 100, 50 and 100 ms, under a limit of 1 s, over 10 s.
    capture = np.array([1_900 * MS, 2_950 * MS, 6_900 * MS])
    receive = np.array([2_000 * MS, 3_000 * MS, 7_000 * MS])

    coverage = watchdog_coverage(capture, receive, 1 * S, 0, 10 * S)

    # The first covers 0.9 s until it expires, the second 0.95 s, the last 0.9 s; the 2 s
    # before the first and the times between an expiry and the next command are not covered.
    assert coverage == pytest.approx((0.9 + 0.95 + 0.9) / 10)


def test_coverage_counts_a_command_that_was_in_effect_when_the_window_began() -> None:
    capture = np.array([4_900 * MS, 9_950 * MS])
    receive = np.array([5_000 * MS, 10_500 * MS])

    coverage = watchdog_coverage(capture, receive, 20 * S, 10 * S, 11 * S)

    assert coverage == pytest.approx(1.0)


def test_the_summary_counts_a_command_received_before_a_window_toward_its_coverage() -> None:
    summary: Any = summary_of(synthetic_run(late_joiner(), run_spec("late_joiner")).run)

    # Robot a gets a cls command every frame, so the one it received just before 10 s is in
    # effect when the window after the join begins.
    assert summary["robots"]["a"]["cls"]["after"]["watchdog_coverage"] == pytest.approx(1.0)
    # b joins at 10 s and goes 20 ms of its 20 s without a command.
    assert summary["models"]["cls"]["after"]["watchdog_coverage"] == pytest.approx(
        {"min": 0.999, "mean": (1 + 0.999) / 2}
    )


def test_the_summary_takes_the_watchdog_limit_the_robots_declared() -> None:
    receive = [at_ns(10.0)]
    run = with_samples(steady_run(), "a", command_rows("depth", [at_ns(9.9)], receive))

    summary: Any = summary_of(run.run)

    # One command, 100 ms old, under depth's 300 ms limit: 200 ms of 30 s.
    assert summary["models"]["depth"]["all"]["watchdog_coverage"] == pytest.approx(
        {"min": 0.2 / 30, "mean": 0.2 / 30}
    )


def test_a_killed_robot_is_left_out_of_the_statistics_from_samples() -> None:
    summary: Any = summary_of(synthetic_run(disconnect(), run_spec("disconnect")).run)
    before = summary["models"]["cls"]["before"]

    assert set(summary["robots"]) == {"a", "b"}
    # Three robots sent frames, and two of them left samples.
    assert before["robots"] == 3
    assert before["commands_per_s"] == pytest.approx(60.0)
    assert before["commands_per_s_per_robot"] == pytest.approx(
        {"min": 30.0, "mean": 30.0, "max": 30.0}
    )
    assert before["utilization"] == pytest.approx(3 * 30 / 200)


@pytest.mark.parametrize(("window", "robots"), [("before", 3), ("after", 2), ("all", 2)])
def test_a_killed_robot_is_present_only_in_the_windows_that_end_by_its_kill(
    window: str, robots: int
) -> None:
    summary: Any = summary_of(synthetic_run(disconnect(), run_spec("disconnect")).run)
    cls = summary["models"]["cls"][window]

    # Robot c is killed at 20 s, where the window before ends.
    assert cls["robots"] == robots
    # Each robot present asks for 30 frames a second, of a capacity of 200.
    assert cls["utilization"] == pytest.approx(robots * 30 / 200)


def test_survivors_are_reported_per_window_in_a_scenario_that_kills() -> None:
    run = synthetic_run(disconnect(), run_spec("disconnect"))
    # Robot b gets a command each second, 300 ms after its capture.
    receive = [at_ns(second + 0.3) for second in range(40)]
    capture = [at_ns(second) for second in range(40)]
    run = with_samples(run, "b", command_rows("cls", capture, receive))

    survivors: Any = summary_of(run.run)["survivors"]

    assert survivors["cls"]["before"]["commands_per_s_per_robot"] == pytest.approx((30 + 1) / 2)
    assert survivors["cls"]["after"]["commands_per_s_per_robot"] == pytest.approx((30 + 1) / 2)
    # 600 commands at 20 ms and 20 at 300 ms: the 95th percentile is among the fast ones.
    assert survivors["cls"]["before"]["freshness_ms"] == {"p95": pytest.approx(20.0)}
    # Robot b now receives no depth command, and a ten each second.
    assert survivors["depth"]["after"]["commands_per_s_per_robot"] == pytest.approx((10 + 0) / 2)


def test_a_scenario_that_kills_no_robot_reports_no_survivors() -> None:
    assert summary_of(steady_run().run)["survivors"] == {}


@pytest.mark.parametrize(
    ("rates", "jain", "min_over_mean"),
    [
        ([2.0, 2.0, 2.0, 2.0], 1.0, 1.0),
        ([4.0, 0.0, 0.0, 0.0], 0.25, 0.0),
        ([1.0, 3.0], 0.8, 0.5),
    ],
)
def test_fairness_is_the_jain_index_and_the_lowest_rate_over_the_mean(
    rates: list[float], jain: float, min_over_mean: float
) -> None:
    assert fairness(rates) == pytest.approx({"jain": jain, "min_over_mean": min_over_mean})


@pytest.mark.parametrize("rates", [[], [5.0], [0.0, 0.0]])
def test_fairness_is_null_for_one_robot_and_for_no_command(rates: list[float]) -> None:
    assert fairness(rates) == {"jain": None, "min_over_mean": None}


def test_the_summary_computes_fairness_over_the_present_robots() -> None:
    run = synthetic_run(sweep(3), run_spec("sweep_n3"))
    receive = [at_ns(second + 0.02) for second in range(30)]
    run = with_samples(
        run, "r01", command_rows("cls", [time - 20 * MS for time in receive], receive)
    )

    window: Any = summary_of(run.run)["models"]["cls"]["all"]

    # Rates of 1, 30 and 30 commands per second.
    assert window["fairness"] == pytest.approx(
        {"jain": 61**2 / (3 * (1 + 900 + 900)), "min_over_mean": 3 / 61}
    )
    assert window["commands_per_s_per_robot"]["min"] == pytest.approx(1.0)


def test_batch_statistics_come_from_the_difference_of_the_window_scrapes() -> None:
    slow = Histogram()
    for value_ms in [5.0] * 10 + [50.0] * 90:
        slow.record(value_ms)
    expected = Histogram()
    expected.record(50.0)
    run = edit_metrics(
        steady_run(),
        setting(
            {
                "models.cls.batch_size": lambda k: (
                    {"1": 10, "2": 5} if k == 0 else {"1": 40, "2": 15, "4": 10}
                ),
                "models.cls.batch_ms": lambda k: histogram(5.0, 10) if k == 0 else slow.to_dict(),
            }
        ),
    )

    window: Any = summary_of(run.run)["models"]["cls"]["all"]

    # 30 batches of one frame, 10 of two and 10 of four between the two scrapes.
    assert window["batch_size"] == {
        "distribution": {"1": 30, "2": 10, "4": 10},
        "mean": pytest.approx(90 / 50),
        "share_1": pytest.approx(0.6),
    }
    # The 90 batches of the window all took 50 ms, read as the upper edge of its bucket.
    assert window["batch_ms"] == {
        "p50": expected.quantile_ms(0.5),
        "p95": expected.quantile_ms(0.95),
    }


def test_capacity_of_an_unpaced_worker_is_its_threads_over_its_cpu_time_per_frame() -> None:
    window: Any = summary_of(steady_run().run)["models"]["depth"]["all"]

    # 20 CPU-ms per frame on one thread: 50 frames per second, for one robot at 30.
    assert window["cpu_ms_per_frame"] == pytest.approx(20.0)
    assert (window["pacing_share"], window["share_cores"]) == (None, 1)
    assert window["capacity_fps"] == pytest.approx(50.0)
    assert window["utilization"] == pytest.approx(0.6)
    assert window["capacity_note"] is None


def test_capacity_of_a_paced_worker_takes_its_share_at_the_later_scrape() -> None:
    run = edit_metrics(
        steady_run(), setting({"workers.depth.pacing_share": lambda k: 0.25 if k == 0 else 0.5})
    )

    window: Any = summary_of(run.run)["models"]["depth"]["all"]

    assert window["pacing_share"] == 0.5
    assert window["capacity_fps"] == pytest.approx(25.0)
    assert window["utilization"] == pytest.approx(1.2)


def test_capacity_with_the_runtimes_own_thread_count_is_bounded_by_the_cpu_limit() -> None:
    run = edit_metrics(steady_run(), setting({"workers.depth.threads": 0}))

    window: Any = summary_of(run.run)["models"]["depth"]["all"]

    assert window["share_cores"] == 2.0
    assert window["capacity_fps"] == pytest.approx(100.0)


@pytest.mark.parametrize(
    ("values", "note"),
    [
        (
            {"processes.worker_depth.cpu_user_ms": 7, "processes.worker_depth.cpu_system_ms": 3},
            "used no CPU time",
        ),
        ({"models.depth.frames": 0}, "served no frame"),
        ({"processes.worker_depth": None}, "not one live process"),
    ],
)
def test_capacity_is_null_with_a_note_where_the_cost_per_frame_is_not_known(
    values: Mapping[str, object], note: str
) -> None:
    window: Any = summary_of(edit_metrics(steady_run(), setting(values)).run)["models"]["depth"][
        "all"
    ]

    assert (window["capacity_fps"], window["utilization"]) == (None, None)
    assert note in window["capacity_note"]


def test_a_worker_that_restarted_inside_the_window_has_no_cores_and_no_capacity() -> None:
    # From second 15 the depth worker's counters are those of a new process.
    restarted = setting({"processes.worker_depth.pid": RESTARTED_PID}, from_second(15))

    summary: Any = summary_of(edit_metrics(steady_run(), restarted).run)

    window = summary["models"]["depth"]["all"]
    assert (window["cpu_ms_per_frame"], window["capacity_fps"], window["utilization"]) == (
        None,
        None,
        None,
    )
    assert "not one live process" in window["capacity_note"]
    assert summary["cpu"]["all"]["cores"]["worker_depth"] is None


def cpu_time_in(field: str) -> Edit:
    """Return an edit that puts all the CPU time of the gateway and the workers in one field."""

    def edit(k: int | None, metrics: dict[str, Any]) -> None:
        for counters in metrics["processes"].values():
            total_ms = counters["cpu_user_ms"] + counters["cpu_system_ms"]
            counters.update(cpu_user_ms=0, cpu_system_ms=0)
            counters[field] = total_ms

    return edit


@pytest.mark.parametrize("field", ["cpu_user_ms", "cpu_system_ms"])
def test_cpu_time_is_user_time_plus_system_time(field: str) -> None:
    summary: Any = summary_of(edit_metrics(steady_run(), cpu_time_in(field)).run)

    # The same CPU time, whichever of the two counters holds it.
    assert summary["cpu"]["all"]["cores"]["worker_depth"] == pytest.approx(0.2)
    assert summary["models"]["depth"]["all"]["cpu_ms_per_frame"] == pytest.approx(20.0)
    assert summary["models"]["depth"]["all"]["capacity_fps"] == pytest.approx(50.0)
    assert summary["gateway"]["all"]["cpu_cores"] == pytest.approx(0.003)
    assert summary["gateway"]["all"]["cpu_ms_per_frame"] == pytest.approx(0.1)


def test_the_stage_means_add_up_to_the_end_to_end_mean() -> None:
    window: Any = summary_of(steady_run().run)["models"]["depth"]["all"]

    assert window["e2e_ms"] == {"mean": pytest.approx(80.0)}
    assert window["stage_sum_ms"] == {"mean": pytest.approx(80.0)}
    assert window["residual_ms"] == {"mean": pytest.approx(0.0, abs=1e-9)}
    assert window["stage"]["fire_lag_ms"] == pytest.approx(
        dict.fromkeys(("mean", "p50", "p95", "p99"), 0.2)
    )
    assert window["stage"]["slot_wait_ms"]["mean"] == pytest.approx(58.0)
    assert list(window["stage"]) == [
        "fire_lag_ms",
        "transport_up_ms",
        "slot_wait_ms",
        "dispatch_rt_ms",
        "decode_ms",
        "preprocess_ms",
        "batch_peers_ms",
        "inference_ms",
        "postprocess_ms",
        "deliver_ms",
    ]


def test_the_residual_is_the_time_that_no_stage_holds() -> None:
    receive = [at_ns(second + 0.1) for second in range(1, 4)]
    rows = command_rows(
        "cls",
        [time - 100 * MS for time in receive],
        receive,
        fire_lag_ns=1 * MS,
        deliver_ns=2 * MS,
        stage_us={"inference": 40_000},
    )
    # 3 ms of each command's slot wait go missing.
    rows["stage_us"][:, 1] -= 3_000

    window: Any = summary_of(with_samples(steady_run(), "a", rows).run)["models"]["cls"]["all"]

    assert window["e2e_ms"] == {"mean": pytest.approx(100.0)}
    assert window["stage_sum_ms"] == {"mean": pytest.approx(97.0)}
    assert window["residual_ms"] == {"mean": pytest.approx(3.0)}


def test_a_command_without_a_send_time_is_left_out_of_the_stages() -> None:
    receive = [at_ns(1.1), at_ns(2.1)]
    rows = command_rows("cls", [at_ns(1.0), at_ns(2.0)], receive, fire_lag_ns=1 * MS)
    rows["send_ns"][1] = 0

    window: Any = summary_of(with_samples(steady_run(), "a", rows).run)["models"]["cls"]["all"]

    assert window["stage"]["fire_lag_ms"]["mean"] == pytest.approx(1.0)
    assert window["freshness_ms"]["mean"] == pytest.approx(100.0)


def test_the_pooled_stages_take_the_per_frame_stages_once_per_frame() -> None:
    capture = [at_ns(second) for second in range(1, 5)]
    receive = [time + 50 * MS for time in capture]
    fast = command_rows(
        "cls", capture[:3], receive[:3], stage_us={"transport_up": 1_000, "inference": 4_000}
    )
    # The fourth frame is slow on the uplink, and both models answer it.
    slow = {"transport_up": 9_000, "inference": 4_000}
    late_cls = command_rows("cls", capture[3:], receive[3:], frame_numbers=[4], stage_us=slow)
    late_depth = command_rows(
        "depth",
        capture[3:],
        [receive[3] + 1],
        frame_numbers=[4],
        stage_us={**slow, "inference": 14_000},
    )

    summary: Any = summary_of(with_samples(steady_run(), "a", fast, late_cls, late_depth).run)

    # Four frames, not five commands: (1 + 1 + 1 + 9) / 4.
    assert summary["stage_all"]["transport_up_ms"]["mean"] == pytest.approx(3.0)
    assert summary["stage_all"]["inference_ms"]["mean"] == pytest.approx((4 * 4 + 14) / 5)
    assert summary["stage_all"]["fire_lag_ms"]["mean"] == pytest.approx(0.0)


def test_the_age_time_bins_hold_the_time_spent_at_each_age() -> None:
    capture = np.array([0, 10 * MS])
    receive = np.array([2_500_000, 12_500_000])

    bins, uncovered_ms = age_time_ms(capture, receive, 0, 20 * MS)

    # The first command ages from 2.5 to 12.5 ms and the second from 2.5 to 10 ms.
    assert bins.tolist() == pytest.approx([0, 0, 1.0, 2, 2, 2, 2, 2, 2, 2, 1, 1, 0.5])
    assert uncovered_ms == pytest.approx(2.5)
    assert bins.sum() + uncovered_ms == pytest.approx(20.0)


def test_a_robot_without_an_applied_command_is_uncovered_for_its_whole_time() -> None:
    bins, uncovered_ms = age_time_ms(np.empty(0, np.int64), np.empty(0, np.int64), 0, 20 * MS)

    assert (bins.size, uncovered_ms) == (0, 20.0)


def test_the_curves_cover_the_whole_time_of_every_present_robot() -> None:
    curves: Any = summary_of(synthetic_run(sweep(3), run_spec("sweep_n3")).run)["models"]["depth"][
        "curves"
    ]

    assert sum(curves["age_time_ms"]) + curves["uncovered_ms"] == pytest.approx(3 * 30_000)
    # Three robots wait 80 ms for their first depth command.
    assert curves["uncovered_ms"] == pytest.approx(3 * 80.0)
    # Every command is 80 ms old at its receipt: 300 for each robot.
    assert curves["freshness_counts"] == [0] * 80 + [900]
    assert curves["bin_ms"] == 1


def test_the_curves_leave_out_a_robot_that_is_not_present_for_the_whole_run() -> None:
    curves: Any = summary_of(synthetic_run(late_joiner(), run_spec("late_joiner")).run)["models"][
        "cls"
    ]["curves"]

    # Only a runs from t0, and b joins at 10 s: the curves hold a's 30 s and its 900 cls
    # commands, each 20 ms old.
    assert sum(curves["age_time_ms"]) + curves["uncovered_ms"] == pytest.approx(30_000.0)
    assert curves["uncovered_ms"] == pytest.approx(20.0)
    assert curves["freshness_counts"] == [0] * 20 + [900]


def memory_bytes(megabytes: Callable[[int | None], float]) -> Callable[[int | None], int]:
    return lambda k: round(megabytes(k) * 2**20)


def test_memory_is_measured_before_the_planned_stops() -> None:
    run = synthetic_run(disconnect(), run_spec("disconnect"))
    # One megabyte more each second, and a jump once the robot is killed at 20 s.
    grown = memory_bytes(
        lambda k: 300 + (40 if k is None else k) + (1000 if k is None or k >= 20 else 0)
    )
    run = edit_metrics(run, setting({"cgroup.memory_anon_bytes": grown}))

    memory: Any = summary_of(run.run)["memory"]

    # Socket memory is 1 MB throughout; the scrapes of seconds 5 and 19 bound the growth.
    assert memory["anon_sock_mb_start"] == pytest.approx(306.0)
    assert memory["anon_sock_mb_end"] == pytest.approx(320.0)
    assert memory["anon_sock_mb_max"] == pytest.approx(320.0)
    assert memory["slope_mb_per_min"] == pytest.approx(60.0)
    assert memory["sock_mb"] == pytest.approx(1.0)


def growing_sizes(k: int | None, metrics: dict[str, Any]) -> None:
    """Grow the gateway's RSS, the socket memory and the peak by one MB a second.

    The final scrape, after the fleet has left, holds sizes no earlier scrape reaches.
    """
    second = 100 if k is None else k
    metrics["processes"]["gateway"]["rss_bytes"] = (100 + second) * 2**20
    metrics["cgroup"]["memory_sock_bytes"] = (2 + second) * 2**20
    metrics["cgroup"]["memory_peak_bytes"] = (500 + second) * 2**20


@pytest.mark.parametrize(
    ("scenario", "last_second"),
    [(steady(), 29), (disconnect(), 19)],
    ids=["the end of the run", "a robot killed at 20 s"],
)
def test_the_sizes_are_those_of_the_last_scrape_before_the_end_or_a_planned_stop(
    scenario: Scenario, last_second: int
) -> None:
    run = edit_metrics(synthetic_run(scenario, run_spec(scenario.name)), growing_sizes)

    memory: Any = summary_of(run.run)["memory"]

    assert memory["rss_mb"]["gateway"] == pytest.approx(100 + last_second)
    assert memory["sock_mb"] == pytest.approx(2 + last_second)
    assert memory["memory_peak_bytes"] == (500 + last_second) * 2**20


def test_the_memory_baseline_is_the_first_scrape_after_the_connections_settle() -> None:
    assert MEMORY_SLOPE_SKIP_S == 5.0
    connecting = setting(
        {
            "connections.current": 0,
            "processes.gateway.fds": 20,
            "loop.tasks": 5,
            "cgroup.memory_anon_bytes": 100 * 2**20,
        },
        lambda k: k is not None and k < 5,
    )
    run = edit_metrics(synthetic_run(sweep(3), run_spec("sweep_n3")), connecting)

    memory: Any = summary_of(run.run)["memory"]

    # The scrape at t0 shows no connection yet, and the growth is not measured from it.
    assert (memory["fds_delta"], memory["tasks_delta"]) == (0, 0)
    assert memory["anon_sock_mb_start"] == pytest.approx(301.0)
    assert memory["slope_mb_per_min"] == pytest.approx(0.0)
    # The peak is over every scrape from t0.
    assert memory["anon_sock_mb_max"] == pytest.approx(301.0)


def test_the_deltas_are_the_changes_between_the_two_memory_scrapes() -> None:
    leak = adding(
        {"processes.gateway.fds": 4, "processes.worker_cls.fds": 1, "loop.tasks": 7},
        from_second(20),
    )

    memory: Any = summary_of(edit_metrics(steady_run(), leak).run)["memory"]

    assert (memory["fds_delta"], memory["tasks_delta"]) == (5, 7)


def test_the_memory_growth_is_null_for_a_run_shorter_than_two_settled_scrapes() -> None:
    brief = Scenario("brief", 6.0, (RobotPlan("a", 0.0, None),), (Window("all", 0.0, 6.0),))

    memory: Any = summary_of(synthetic_run(brief, run_spec("steady")).run)["memory"]

    growth = (
        "anon_sock_mb_start",
        "anon_sock_mb_end",
        "slope_mb_per_min",
        "fds_delta",
        "tasks_delta",
    )
    assert {key: memory[key] for key in growth} == dict.fromkeys(growth)
    assert memory["anon_sock_mb_max"] == pytest.approx(301.0)
    assert set(memory["rss_mb"]) == {"driver", "fleet", "gateway", "worker_cls", "worker_depth"}


@pytest.mark.parametrize("second", [5, 29], ids=["the first memory scrape", "the last"])
def test_a_memory_scrape_with_other_connections_than_planned_is_noted(second: int) -> None:
    run = edit_metrics(steady_run(), setting({"connections.current": 0}, lambda k: k == second))

    validity: Any = summary_of(run.run)["validity"]

    assert validity["notes"] == [
        "memory scrapes with other connections than planned: "
        f"0 connections at {second}.0 s, 1 planned"
    ]


def test_a_run_whose_memory_scrapes_show_the_planned_connections_has_no_note() -> None:
    assert summary_of(steady_run().run)["validity"]["notes"] == []


def test_connections_are_counted_from_the_scrape_before_t0() -> None:
    # Seven sessions of an earlier use of the same gateway, all closed before this run.
    earlier = adding({"connections.accepted": 7, "connections.closed.client_close": 7})

    summary: Any = summary_of(edit_metrics(steady_run(), earlier).run)

    assert summary["connections"]["accepted"] == 1
    assert summary["connections"]["closed"]["client_close"] == 1
    assert summary["connections"]["rejected"] == 0


def test_connections_sum_the_refusals_over_their_reasons() -> None:
    refused = adding({"connections.rejected.capacity": 2, "connections.rejected.version": 1}, final)

    connections: Any = summary_of(edit_metrics(steady_run(), refused).run)["connections"]

    assert connections["rejected"] == 3
    assert connections["rejected_by_reason"]["capacity"] == 2


def test_connections_are_null_without_the_final_scrape() -> None:
    run = steady_run()
    scrapes = (*run.scrapes[:-1], failed_scrape(31, at_ns(31.5)))

    summary: Any = summary_of(replace(run, scrapes=scrapes).run)

    assert summary["connections"] is None
    assert summary["validity"]["conservation_ok"] is None
    assert check(summary, "conservation")["passed"] is False


def test_a_run_in_which_nothing_went_wrong_passes_the_checks_of_every_scenario() -> None:
    summary: Any = summary_of(steady_run().run)

    assert [(entry["name"], entry["passed"]) for entry in summary["checks"]] == [
        ("conservation", True),
        ("no_worker_restart", True),
        ("no_internal_error", True),
        ("simulator_valid", True),
        ("no_upward_trend.cls", True),
        ("no_upward_trend.depth", True),
    ]


@pytest.mark.parametrize(
    ("name", "edit", "observed"),
    [
        (
            "conservation",
            adding({"robots.a.frames_accepted": 1}, final),
            ["a: frames_in != frames_rejected + frames_accepted"],
        ),
        ("no_worker_restart", setting({"workers.cls.restarts": 1}, final), {"cls": 1, "depth": 0}),
        ("no_internal_error", setting({"connections.closed.internal_error": 2}, final), 2),
    ],
)
def test_a_check_of_the_final_scrape_fails_on_what_it_guards(
    name: str, edit: Edit, observed: object
) -> None:
    summary: Any = summary_of(edit_metrics(steady_run(), edit).run)

    assert check(summary, name) == {"name": name, "passed": False, "observed": observed}


@pytest.mark.parametrize(
    ("ticks", "fire_lag_p99_ms", "missed"),
    [
        # A fire lag above a tenth of the 33.3 ms frame period.
        (tick_rows([at_ns(second) for second in range(30)], fire_lag_ns=5 * MS), 5.0, 0),
        (
            tick_rows(
                [at_ns(second) for second in range(30)],
                outcomes=[TickOutcome.MISSED_LATE] + [TickOutcome.SENT] * 29,
            ),
            0.0,
            1,
        ),
    ],
)
def test_a_simulator_that_sends_late_or_passes_a_tick_limits_the_run(
    ticks: Any, fire_lag_p99_ms: float, missed: int
) -> None:
    run = steady_run()
    run = replace(run, samples={"a": replace(run.samples["a"], ticks=ticks)})

    summary: Any = summary_of(run.run)

    assert summary["validity"]["simulator_limited"] is True
    assert check(summary, "simulator_valid") == {
        "name": "simulator_valid",
        "passed": False,
        "observed": {"fire_lag_ms_p99": pytest.approx(fire_lag_p99_ms), "missed_late": missed},
    }


def test_validity_counts_the_ticks_by_outcome_and_the_dropped_samples() -> None:
    run = steady_run()
    outcomes = [TickOutcome.SENT] * 3 + [TickOutcome.SKIPPED_UNSENT] * 2
    ticks = tick_rows([at_ns(second) for second in range(5)], fire_lag_ns=1 * MS, outcomes=outcomes)
    robot = replace(run.samples["a"], ticks=ticks, dropped_ticks=4, dropped_commands=6)

    validity: Any = summary_of(replace(run, samples={"a": robot}).run)["validity"]

    assert validity["ticks"] == {"sent": 3, "skipped_unsent": 2, "missed_late": 0}
    assert validity["fire_lag_ms"] == {"p99": pytest.approx(1.0)}
    assert validity["samples_dropped"] == {"ticks": 4, "commands": 6}
    assert validity["simulator_limited"] is False
    assert validity["final_scrape_settled"] is True
    assert validity["throttled_ms_per_s"] == 0.0


def test_a_final_poll_that_did_not_settle_is_reported() -> None:
    run = steady_run()
    *before, final_poll = run.events
    assert final_poll.name is EventName.FINAL_POLL
    unsettled = replace(final_poll, settled=False, waited_s=10.0)

    validity: Any = summary_of(replace(run, events=(*before, unsettled)).run)["validity"]

    assert validity["final_scrape_settled"] is False


def test_the_clock_check_fails_when_the_gateways_reading_is_outside_the_drivers() -> None:
    run = steady_run()
    first = replace(run.scrapes[0], driver_after_ns=run.scrapes[0].driver_before_ns)

    validity: Any = summary_of(replace(run, scrapes=(first, *run.scrapes[1:])).run)["validity"]

    assert validity["shared_clock_ok"] is False


def test_the_clock_check_passes_when_the_gateways_reading_is_between_the_drivers() -> None:
    assert summary_of(steady_run().run)["validity"]["shared_clock_ok"] is True


def test_throttled_time_is_reported_per_second_and_per_period() -> None:
    throttled = setting(
        {
            "cgroup.throttled_us": lambda k: 0 if k == 0 else 58_000,
            "cgroup.nr_throttled": lambda k: 0 if k == 0 else 29,
        }
    )

    summary: Any = summary_of(edit_metrics(steady_run(), throttled).run)

    # 58 ms over the 29 s between the scrapes, in 29 of their 290 periods.
    assert summary["cpu"]["all"]["throttled_ms_per_s"] == pytest.approx(2.0)
    assert summary["cpu"]["all"]["throttled_share"] == pytest.approx(0.1)
    assert summary["validity"]["throttled_ms_per_s"] == pytest.approx(2.0)


def test_cpu_utilization_is_the_cgroups_usage_over_its_limit() -> None:
    cpu: Any = summary_of(steady_run().run)["cpu"]["all"]

    # 0.003 + 0.15 + 0.2 cores of the gateway and the workers, of a limit of 2.
    assert cpu["utilization"] == pytest.approx(0.353 / 2)
    assert cpu["cores"] == pytest.approx(
        {"driver": 0.01, "fleet": 0.1, "gateway": 0.003, "worker_cls": 0.15, "worker_depth": 0.2}
    )
    assert cpu["robots_container_cores"] is None


def test_the_split_topology_reports_the_robots_own_container() -> None:
    run = synthetic_run(steady(), run_spec("steady", topology=Topology.SPLIT))

    cpu: Any = summary_of(run.run)["cpu"]["all"]

    assert cpu["robots_container_cores"] == pytest.approx(0.1)


def test_a_process_that_is_gone_at_the_later_scrape_has_no_cores() -> None:
    cpu: Any = summary_of(synthetic_run(disconnect(), run_spec("disconnect")).run)["cpu"]

    assert cpu["before"]["cores"]["fleet_c"] == pytest.approx(0.02)
    assert cpu["all"]["cores"]["fleet_c"] is None


def test_the_gateway_block_reports_its_cpu_per_frame_and_its_loop_lag() -> None:
    lag = Histogram()
    lag.record(0.5)

    gateway: Any = summary_of(steady_run().run)["gateway"]["all"]

    assert gateway["cpu_cores"] == pytest.approx(0.003)
    assert gateway["cpu_ms_per_frame"] == pytest.approx(0.1)
    assert gateway["loop_lag_ms"] == {"p99": lag.quantile_ms(0.99)}


@pytest.mark.parametrize(
    ("x", "y", "slope"),
    [([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 2.0), ([0.0, 1.0, 2.0, 3.0], [1.0, 0.0, 1.0, 0.0], -0.2)],
)
def test_the_trend_is_the_slope_of_the_least_squares_line(
    x: list[float], y: list[float], slope: float
) -> None:
    assert least_squares_slope(np.array(x), np.array(y)) == pytest.approx(slope)


@pytest.mark.parametrize(("x", "y"), [([], []), ([1.0], [2.0]), ([3.0, 3.0], [1.0, 2.0])])
def test_the_trend_is_null_without_two_different_times(x: list[float], y: list[float]) -> None:
    assert least_squares_slope(np.array(x), np.array(y)) is None


def drifting(ms_per_capture_s: float) -> Any:
    """Return cls commands, one captured each second, each ms_per_capture_s fresher or staler."""
    capture = [at_ns(second) for second in range(30)]
    receive = [at_ns(second) + round((50 + ms_per_capture_s * second) * MS) for second in range(30)]
    return command_rows("cls", capture, receive)


@pytest.mark.parametrize(
    ("ms_per_capture_s", "passed"), [(1.0, True), (-1.0, True), (1.2, False), (-1.2, False)]
)
def test_no_upward_trend_passes_while_the_drift_over_the_run_is_under_a_frame_period(
    ms_per_capture_s: float, passed: bool
) -> None:
    summary: Any = summary_of(with_samples(steady_run(), "a", drifting(ms_per_capture_s)).run)

    # The slope is against receive time, which also moves by the drift: 1 s of capture time
    # is 1 s plus the drift of receive time.
    trend = ms_per_capture_s / (1 + ms_per_capture_s / 1000)
    assert summary["models"]["cls"]["all"]["freshness_trend_ms_per_s"] == pytest.approx(trend)
    # 30 s of drift against a frame period of 33.3 ms.
    assert check(summary, "no_upward_trend.cls") == {
        "name": "no_upward_trend.cls",
        "passed": passed,
        "observed": pytest.approx(trend),
    }


def test_the_late_joiner_check_passes_when_b_got_commands_of_every_model() -> None:
    summary: Any = summary_of(synthetic_run(late_joiner(), run_spec("late_joiner")).run)

    assert check(summary, "b_served_after_join") == {
        "name": "b_served_after_join",
        "passed": True,
        "observed": {"b": {"cls": 600, "depth": 200}},
    }
    # b's first cls command comes 20 ms after it joins.
    assert summary["robots"]["b"]["cls"]["after"]["first_gap_ms"] == pytest.approx(20.0)
    assert "before" not in summary["robots"]["b"]["cls"]


def test_the_late_joiner_check_fails_when_a_model_never_served_b() -> None:
    run = synthetic_run(late_joiner(), run_spec("late_joiner"))
    run = with_samples(run, "b", command_rows("cls", [at_ns(11.0)], [at_ns(11.1)]))

    summary: Any = summary_of(run.run)

    assert check(summary, "b_served_after_join")["passed"] is False
    assert check(summary, "b_served_after_join")["observed"] == {"b": {"cls": 1, "depth": 0}}


def test_the_disconnect_checks_pass_when_the_kill_closed_one_session_as_abnormal() -> None:
    summary: Any = summary_of(synthetic_run(disconnect(), run_spec("disconnect")).run)

    assert check(summary, "killed_robot_released") == {
        "name": "killed_robot_released",
        "passed": True,
        "observed": {"connected": False, "frames_pending": {"cls": 0, "depth": 0}},
    }
    # The gateway closed the session 50 ms after the kill.
    assert check(summary, "closed_abnormal") == {
        "name": "closed_abnormal",
        "passed": True,
        "observed": {"closed": {"abnormal": 1}, "detect_ms": pytest.approx(50.0)},
    }


@pytest.mark.parametrize(
    "values",
    [{"robots.c.models.depth.frames_pending": 1}, {"robots.c.connected": True}],
)
def test_killed_robot_released_fails_while_the_robot_holds_a_slot_or_a_session(
    values: Mapping[str, object],
) -> None:
    run = edit_metrics(synthetic_run(disconnect(), run_spec("disconnect")), setting(values, final))

    assert check(summary_of(run.run), "killed_robot_released")["passed"] is False


@pytest.mark.parametrize(
    ("edit", "closed"),
    [
        # The session ended for another reason.
        (
            adding(
                {"connections.closed.abnormal": -1, "connections.closed.idle_timeout": 1},
                from_second(21),
            ),
            {"idle_timeout": 1},
        ),
        # Another session closed in the same second.
        (
            adding({"connections.closed.slow_consumer": 1}, from_second(21)),
            {"abnormal": 1, "slow_consumer": 1},
        ),
    ],
)
def test_closed_abnormal_fails_unless_exactly_one_abnormal_close_happened(
    edit: Edit, closed: Mapping[str, int]
) -> None:
    run = edit_metrics(synthetic_run(disconnect(), run_spec("disconnect")), edit)

    found = check(summary_of(run.run), "closed_abnormal")

    assert found["passed"] is False
    assert found["observed"]["closed"] == closed


def test_closed_abnormal_fails_when_no_scrape_shows_the_robot_gone() -> None:
    # Every scrape from t0 on, where the gateway has a record of the robot.
    run = edit_metrics(
        synthetic_run(disconnect(), run_spec("disconnect")),
        setting({"robots.c.connected": True}, from_second(0)),
    )

    assert check(summary_of(run.run), "closed_abnormal") == {
        "name": "closed_abnormal",
        "passed": False,
        "observed": None,
    }


def depth_worker_killed() -> SyntheticRun:
    return worker_kill_run(worker_kill(), run_spec("worker_kill"))


# Where the synthetic worker_kill run kills the depth worker, on the shared clock.
KILL_NS = T0_NS + 20 * S + KILL_LATE_NS
# The stages from a batch's dispatch to its result that with_depth gives each command: 20.8 ms.
DEPTH_STAGES_US = {
    "dispatch_rt": 200,
    "decode": 300,
    "preprocess": 200,
    "inference": 20_000,
    "postprocess": 100,
}
AFTER_DISPATCH_NS = 20_800_000


def with_depth(run: SyntheticRun, robot_id: str, receive_ns: list[int]) -> SyntheticRun:
    """Return the run with the robot's depth commands those received at these times."""
    rows = run.samples[robot_id].commands
    cls = rows[rows["model"] == MODEL_KEYS.index("cls")]
    capture_ns = [time_ns - 80 * MS for time_ns in receive_ns]
    return with_samples(
        run, robot_id, cls, command_rows("depth", capture_ns, receive_ns, stage_us=DEPTH_STAGES_US)
    )


def test_the_fault_block_holds_the_kill_its_lateness_and_the_restarts_of_each_worker() -> None:
    run = depth_worker_killed()

    fault: Any = summary_of(run.run)["fault"]

    assert {key: value for key, value in fault.items() if key != "outage_ms"} == {
        "model": "depth",
        "pid": run.events[-1].pid,
        "planned_s": 20.0,
        "kill_s": pytest.approx(20.0004),
        "late_ms": pytest.approx(0.4),
        "restarts": {"cls": 0, "depth": 1},
    }
    # Depth answers every third frame of a robot, so its first command after the outage comes
    # within three frame periods of the outage's end.
    assert (
        WORKER_OUTAGE_NS / MS <= fault["outage_ms"] < (WORKER_OUTAGE_NS + 3 * FRAME_PERIOD_NS) / MS
    )


def test_the_outage_ends_at_the_first_command_of_a_batch_dispatched_after_the_kill() -> None:
    run = depth_worker_killed()
    # The batch of a's command received 19.8 ms after the kill was dispatched 1 ms before it,
    # to the worker that was killed. b's command at 900 ms is the first of the new worker.
    after_kill_ns = KILL_NS + AFTER_DISPATCH_NS - 1 * MS
    run = with_depth(run, "a", [KILL_NS - 100 * MS, after_kill_ns, KILL_NS + 1_200 * MS])
    run = with_depth(run, "b", [KILL_NS - 50 * MS, KILL_NS + 900 * MS])

    assert summary_of(run.run)["fault"]["outage_ms"] == 900.0


def test_the_outage_is_null_without_a_command_of_a_batch_dispatched_after_the_kill() -> None:
    run = depth_worker_killed()
    run = with_depth(run, "a", [KILL_NS - 100 * MS, KILL_NS + AFTER_DISPATCH_NS - 1 * MS])
    # A command received after the run's end lies outside every window.
    run = with_depth(run, "b", [KILL_NS - 50 * MS, at_ns(40.0) + 100 * MS])

    summary: Any = summary_of(run.run)

    assert summary["fault"]["outage_ms"] is None
    assert check(summary, "worker_restarted")["passed"] is False


def test_a_worker_kill_checks_that_the_worker_alone_restarted_once_and_serves_again() -> None:
    run = depth_worker_killed()

    summary: Any = summary_of(run.run)

    assert [(entry["name"], entry["passed"]) for entry in summary["checks"]] == [
        ("conservation", True),
        ("worker_restarted", True),
        ("no_internal_error", True),
        ("simulator_valid", True),
    ]
    assert check(summary, "worker_restarted")["observed"] == {
        "restarts": {"cls": 0, "depth": 1},
        "state": "ready",
        "pid": RESTARTED_PID,
        "killed_pid": run.events[-1].pid,
        "outage_ms": summary["fault"]["outage_ms"],
    }


@pytest.mark.parametrize(
    "values",
    [
        {"workers.depth.restarts": 2},
        {"workers.depth.restarts": 0},
        {"workers.cls.restarts": 1},
        {"workers.depth.state": "restarting"},
        {"workers.depth.pid": None},
    ],
    ids=["depth twice", "depth never", "cls as well", "not ready", "no pid"],
)
def test_worker_restarted_fails_unless_the_killed_worker_alone_restarted_once_and_is_ready(
    values: Mapping[str, object],
) -> None:
    run = edit_metrics(depth_worker_killed(), setting(values, final))

    assert check(summary_of(run.run), "worker_restarted")["passed"] is False


def test_the_restarts_of_a_worker_kill_are_counted_from_the_scrape_before_t0() -> None:
    # The classifier had restarted once before the run began.
    run = edit_metrics(depth_worker_killed(), adding({"workers.cls.restarts": 1}))

    summary: Any = summary_of(run.run)

    assert summary["fault"]["restarts"] == {"cls": 0, "depth": 1}
    assert check(summary, "worker_restarted")["passed"] is True


def test_worker_restarted_fails_when_the_worker_still_has_the_killed_pid() -> None:
    run = depth_worker_killed()
    run = edit_metrics(run, setting({"workers.depth.pid": run.events[-1].pid}, final))

    assert check(summary_of(run.run), "worker_restarted")["passed"] is False


def test_a_scrape_at_the_instant_of_the_planned_worker_kill_stays_out_of_the_window_before() -> (
    None
):
    run = edit_metrics(
        depth_worker_killed(), setting({"ts_mono_ns": at_ns(20.0)}, lambda k: k == 20)
    )

    windows: Any = summary_of(run.run)["windows"]

    assert windows["before"]["span_s"] == pytest.approx(19.0)
    assert windows["after"]["span_s"] == pytest.approx(19.002)


FAULT = Event(EventName.FAULT, 20.0004, model="depth", pid=12)


@pytest.mark.parametrize(
    ("faults", "message"),
    [
        ((), "needs one fault event in events.jsonl, and has 0"),
        ((FAULT, FAULT), "needs one fault event in events.jsonl, and has 2"),
        ((replace(FAULT, model="cls"),), "the fault event names the cls worker"),
    ],
    ids=["none", "two", "another model"],
)
def test_a_worker_kill_without_one_fault_event_of_its_model_has_no_summary(
    faults: tuple[Event, ...], message: str
) -> None:
    run = depth_worker_killed()
    run = replace(run, events=(*run.events[:-1], *faults))

    with pytest.raises(SummaryError, match=message):
        summary_of(run.run)


@pytest.mark.parametrize("scenario", [steady(), disconnect()], ids=["steady", "disconnect"])
def test_a_run_that_kills_no_worker_has_no_fault_block(scenario: Scenario) -> None:
    summary: Any = summary_of(synthetic_run(scenario, run_spec(scenario.name)).run)

    assert summary["fault"] is None
    assert check(summary, "no_worker_restart")["passed"] is True


def test_summarize_run_reads_a_worker_kill_from_its_files(tmp_path: Path) -> None:
    run = depth_worker_killed()
    run_dir = write_finished_run(tmp_path, run)

    assert summarize_run(run_dir) == {
        "summarizer_digest": summarizer_digest(),
        **summary_of(run.run),
    }


def test_all_accepted_passes_when_every_robot_connected_and_none_was_refused() -> None:
    summary: Any = summary_of(massive_run().run)

    assert check(summary, "all_accepted") == {
        "name": "all_accepted",
        "passed": True,
        "observed": {"accepted": 50, "rejected": 0},
    }


@pytest.mark.parametrize(
    "edit",
    [
        adding({"connections.accepted": -1}, final),
        adding({"connections.rejected.capacity": 1}, final),
    ],
)
def test_all_accepted_fails_on_a_missing_or_a_refused_connection(edit: Edit) -> None:
    summary: Any = summary_of(edit_metrics(synthetic_run(SHORT_SOAK, run_spec("soak")), edit).run)

    assert check(summary, "all_accepted")["passed"] is False


def test_the_dispatch_time_of_a_batch_is_its_result_less_the_stages_after_dispatch() -> None:
    stages = {
        "transport_up": 700,
        "dispatch_rt": 200,
        "decode": 300,
        "preprocess": 400,
        "batch_peers": 1_500,
        "inference": 5_000,
        "postprocess": 100,
    }
    # Two commands of one batch, and one of the batch before it.
    rows = merged(
        command_rows("cls", [0, 0], [50 * MS, 50 * MS], stage_us=stages),
        command_rows("cls", [0], [30 * MS], stage_us=stages),
    )

    dispatch = dispatch_times_ns(rows)

    # Every stage from the dispatch round trip to the postprocessing: 7.5 ms.
    assert dispatch.tolist() == [30 * MS - 7_500_000, 50 * MS - 7_500_000]


def waited_for_peers(run: SyntheticRun, frame_number: int, wait_ns: int) -> SyntheticRun:
    """Return the run with one cls batch delivered wait_ns later, all of it spent with its peers.

    The batch holds every robot's frame of that number. It was dispatched when it would have
    been, and its batch_peers stage holds the wait.
    """
    batch_peers = STAGE_NAMES.index("batch_peers")
    samples = {}
    for robot_id, robot in run.samples.items():
        rows = robot.commands.copy()
        batch = (rows["model"] == MODEL_KEYS.index("cls")) & (rows["frame_number"] == frame_number)
        rows["stage_us"][batch, batch_peers] += wait_ns // 1000
        rows["result_ns"][batch] += wait_ns
        rows["receive_ns"][batch] += wait_ns
        samples[robot_id] = replace(robot, commands=merged(rows))
    return replace(run, samples=samples)


def test_a_batch_that_waited_for_its_peers_keeps_its_dispatch_time_in_the_cycle() -> None:
    # The 50 robots' frame 300 forms one batch, whose result comes 10 ms late.
    run = waited_for_peers(massive_run(), 300, 10 * MS)

    found = check(summary_of(run.run), "gaps_within_bound.cls")

    # The batches are still dispatched one frame period apart.
    assert found["observed"]["cycle_ms"] == pytest.approx(FRAME_PERIOD_NS / MS)
    assert found["observed"]["longest_interval_ms"] == pytest.approx(FRAME_PERIOD_NS / MS)
    assert found["passed"] is True


def test_one_slow_batch_counts_once_in_the_measured_cycle() -> None:
    # Dispatches 10 ms apart, with one interval of 100 ms; four robots at a cap of two wait
    # three cycles.
    intervals_ms = [10, 10, 100, 10, 10, 10]
    dispatch = np.cumsum([0, *intervals_ms]) * MS

    cycle_ns, longest_ns = measured_cycle_ns(dispatch, 150 * MS, robot_count=4, batch_cap=2)

    assert (cycle_ns, longest_ns) == (40 * MS, 100 * MS)


def test_the_measured_cycle_counts_the_interval_to_the_windows_end_and_nothing_after_it() -> None:
    dispatch = np.array([0, 10 * MS, 20 * MS, 500 * MS])

    cycle_ns, longest_ns = measured_cycle_ns(dispatch, 60 * MS, robot_count=1, batch_cap=1)

    # Two cycles: the last dispatch inside the window, and the 40 ms to its end.
    assert (cycle_ns, longest_ns) == (25 * MS, 40 * MS)


def test_fewer_dispatches_than_cycles_give_their_whole_time_over_the_cycles() -> None:
    dispatch = np.array([0, 10 * MS])

    cycle_ns, longest_ns = measured_cycle_ns(dispatch, 30 * MS, robot_count=4, batch_cap=1)

    # Two intervals, of 10 and 20 ms, where a robot waits five cycles; rounded up.
    assert (cycle_ns, longest_ns) == (6 * MS, 20 * MS)


def test_the_measured_cycle_is_rounded_up_to_a_whole_nanosecond() -> None:
    dispatch = np.array([0, 10])

    cycle_ns, longest_ns = measured_cycle_ns(dispatch, 11, robot_count=1, batch_cap=1)

    # Intervals of 10 ns and 1 ns over two cycles: 5.5 ns.
    assert (cycle_ns, longest_ns) == (6, 10)


def test_a_run_without_a_dispatch_has_a_cycle_of_zero() -> None:
    assert measured_cycle_ns(np.empty(0, np.int64), 30 * MS, robot_count=4, batch_cap=1) == (0, 0)


def test_the_gap_check_passes_on_the_measured_cycle_where_every_batch_exceeds_the_calibration() -> (
    None
):
    found = check(summary_of(massive_run().run), "gaps_within_bound.depth")

    # Depth dispatches every 100 ms, five times its calibrated 20 ms with the 0.2 ms legs.
    assert found["passed"] is True
    assert found["observed"] == pytest.approx(
        {
            "largest_gap_ms": 3 * FRAME_PERIOD_NS / MS,
            "largest_first_gap_ms": 80.0,
            # One frame period and 50 / 1 + 1 cycles of three frame periods.
            "bound_ms": (1 + 51 * 3) * FRAME_PERIOD_NS / MS,
            "cycle_ms": 3 * FRAME_PERIOD_NS / MS,
            "longest_interval_ms": 3 * FRAME_PERIOD_NS / MS,
            "calibrated_cycle_ms": 20.4,
        }
    )


def without_depth_between(run: SyntheticRun, start_s: float, end_s: float) -> SyntheticRun:
    """Return the run without the depth commands that every robot received in [start_s, end_s)."""
    depth = MODEL_KEYS.index("depth")
    kept = {}
    for robot_id, robot in run.samples.items():
        rows = robot.commands
        gone = (rows["model"] == depth) & in_window(
            rows["receive_ns"], at_ns(start_s), at_ns(end_s)
        )
        kept[robot_id] = replace(robot, commands=rows[~gone])
    return replace(run, samples=kept)


def test_one_slow_batch_of_a_run_counts_once_in_its_measured_cycle() -> None:
    # Depth dispatches every three frame periods. Without the three batches whose commands
    # arrive in [10.0, 10.3) s, one interval is four of those long.
    run = without_depth_between(massive_run(), 10.0, 10.3)

    found = check(summary_of(run.run), "gaps_within_bound.depth")

    interval_ns = 3 * FRAME_PERIOD_NS
    # 50 robots at a cap of one wait 51 cycles, and 51 consecutive intervals hold the long
    # one once: 4 + 50 intervals over 51, rounded up to a whole nanosecond.
    assert found["observed"]["cycle_ms"] == pytest.approx(math.ceil(54 * interval_ns / 51) / MS)
    assert found["observed"]["longest_interval_ms"] == pytest.approx(4 * interval_ns / MS)
    assert found["passed"] is True


def test_the_calibrated_cycle_follows_the_pacing_shares_of_the_window() -> None:
    # The scrape before t0 and the final scrape lie outside the window, and their smaller
    # share is not the window's.
    shares = {-1: 0.1, 10: 0.5, None: 0.1}
    paced = setting({"workers.depth.pacing_share": lambda k: shares.get(k, 0.25)})

    found = check(summary_of(edit_metrics(massive_run(), paced).run), "gaps_within_bound.depth")

    # 20 CPU-ms at the smallest share of the window, a quarter of a core.
    assert found["observed"]["calibrated_cycle_ms"] == pytest.approx(80.0)


def test_the_gap_check_fails_when_a_robot_waits_longer_than_the_bound() -> None:
    receive = [at_ns(1.0), at_ns(11.0), at_ns(29.9)]
    run = with_samples(
        massive_run(), "r07", command_rows("cls", [time - 20 * MS for time in receive], receive)
    )

    found = check(summary_of(run.run), "gaps_within_bound.cls")

    # The other robots keep the cycle at one frame period: a bound of 1 + 8 periods.
    assert found["passed"] is False
    assert found["observed"]["bound_ms"] == pytest.approx(9 * FRAME_PERIOD_NS / MS)
    assert found["observed"]["largest_gap_ms"] == pytest.approx(18_900.0)
    assert found["observed"]["largest_first_gap_ms"] == pytest.approx(1_000.0)


def test_the_gap_check_holds_the_first_gap_to_the_bound_plus_the_uplink_and_the_delivery() -> None:
    run = massive_run()
    # The first command 300.4 ms after the start: over the bound of 300 ms, under it with
    # the 0.5 ms of the uplink and the 0.5 ms of the delivery.
    capture = np.arange(at_ns(0.2804), at_ns(30.0) - 20 * MS, FRAME_PERIOD_NS)
    rows = command_rows(
        "cls", capture, capture + 20 * MS, deliver_ns=500_000, stage_us={"transport_up": 500}
    )
    late = with_samples(run, "r07", rows)
    later = with_samples(run, "r07", rows[30:])

    passing = check(summary_of(late.run), "gaps_within_bound.cls")
    failing = check(summary_of(later.run), "gaps_within_bound.cls")

    assert passing["observed"]["largest_first_gap_ms"] == pytest.approx(300.4)
    assert passing["passed"] is True
    assert failing["passed"] is False


def skipping(run: SyntheticRun) -> SyntheticRun:
    """Return the run with one cls frame of r01 passed over by the skip rule after t0."""
    return edit_metrics(
        run,
        adding(
            {
                "robots.r01.models.cls.frames_skipped": 1,
                "robots.r01.models.cls.frames_overwritten": -1,
            },
            final,
        ),
    )


@pytest.mark.parametrize(
    ("run", "skip_interval_ms"),
    [
        (massive_run(), 0.0),
        (skipping(massive_run()), 1000.0),
        # With the rule off the counter cannot rise; a bound must not depend on it either way.
        (skipping(massive_run({"skip_rule": False})), 0.0),
    ],
    ids=["no frame skipped", "a frame skipped", "the rule off"],
)
def test_the_skip_interval_enters_the_bound_only_when_a_frame_was_skipped(
    run: SyntheticRun, skip_interval_ms: float
) -> None:
    summary: Any = summary_of(run.run)

    assert check(summary, "gaps_within_bound.cls")["observed"]["bound_ms"] == pytest.approx(
        9 * FRAME_PERIOD_NS / MS + skip_interval_ms
    )
    assert check(summary, "conservation")["passed"] is True


def test_the_gap_check_fails_without_a_batch_cap() -> None:
    run = edit_metrics(massive_run(), setting({"workers.cls.batch_cap": None}))

    assert check(summary_of(run.run), "gaps_within_bound.cls") == {
        "name": "gaps_within_bound.cls",
        "passed": False,
        "observed": None,
    }


def test_the_soak_records_its_memory_growth_without_a_criterion() -> None:
    leak = adding({"loop.tasks": 3}, from_second(8))

    summary: Any = summary_of(edit_metrics(synthetic_run(SHORT_SOAK, run_spec("soak")), leak).run)

    assert check(summary, "memory_growth") == {
        "name": "memory_growth",
        "passed": None,
        "observed": {"slope_mb_per_min": pytest.approx(0.0), "fds_delta": 0, "tasks_delta": 3},
    }
    assert check(summary, "all_accepted")["passed"] is True
    assert check(summary, "gaps_within_bound.depth")["passed"] is True


def test_the_series_hold_what_the_plots_of_one_run_draw() -> None:
    series: Any = summary_of(steady_run().run)["series"]

    assert series["bin_s"] == 1
    assert series["commands_per_bin"]["a"]["cls"] == [30] * 30
    assert series["commands_per_bin"]["a"]["depth"] == [10] * 30
    assert len(series["freshness"]["depth"]["t_s"]) == 300
    assert series["freshness"]["depth"]["t_s"][0] == pytest.approx(0.08)
    assert series["freshness"]["depth"]["freshness_ms"] == pytest.approx([80.0] * 300)
    # The scrapes of seconds 0 to 29, each 2 ms after its second.
    assert series["memory"]["t_s"] == pytest.approx([second + 0.002 for second in range(30)])
    assert series["memory"]["anon_sock_mb"] == pytest.approx([301.0] * 30)


def test_only_the_steady_scenario_keeps_the_freshness_of_every_command() -> None:
    series: Any = summary_of(synthetic_run(sweep(3), run_spec("sweep_n3")).run)["series"]

    assert series["freshness"] == {}
    assert set(series["commands_per_bin"]) == {"r01", "r02", "r03"}


def test_the_run_block_names_the_run() -> None:
    run = synthetic_run(
        sweep(3), run_spec("sweep_n3", topology=Topology.SPLIT, repeat=2), gated=False
    )

    assert summary_of(run.run)["run"] == {
        "scenario": "sweep_n3",
        "run_id": run.config.run_id,
        "repeat": 2,
        "topology": "split",
        "duration_s": 30.0,
        "gated": False,
    }


@pytest.mark.parametrize("model_set", list(ModelSet))
def test_a_synthetic_runs_spec_is_the_evidence_plans_spec(model_set: ModelSet) -> None:
    plan = evidence_plan(2, model_set, CODE_DIGEST)

    built = [
        run_spec(
            spec.scenario_key, topology=spec.topology, model_set=spec.model_set, repeat=spec.repeat
        )
        for spec in plan
    ]

    assert built == plan


@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda candidate: candidate.name)
def test_a_synthetic_comparison_runs_spec_is_its_arms_spec(candidate: Candidate) -> None:
    plan = compare_plan(candidate, CODE_DIGEST)
    overrides: Mapping[str | None, Mapping[str, Any]] = {
        "a": candidate.a_overrides,
        "b": candidate.b_overrides,
    }

    built = [
        run_spec(
            spec.scenario_key,
            topology=spec.topology,
            model_set=spec.model_set,
            overrides=overrides[spec.arm],
            repeat=spec.repeat,
            candidate=spec.candidate,
            arm=spec.arm,
        )
        for spec in plan
    ]

    assert built == plan


def test_the_verdicts_are_counted_and_every_command_counts_for_freshness() -> None:
    receive = [at_ns(second + 0.6) for second in range(1, 5)]
    verdicts = [Verdict.APPLIED, Verdict.STALE, Verdict.OUT_OF_ORDER, Verdict.APPLIED]
    rows = command_rows("cls", [time - 600 * MS for time in receive], receive, verdicts=verdicts)

    window: Any = summary_of(with_samples(steady_run(), "a", rows).run)["models"]["cls"]["all"]

    assert (window["stale_count"], window["out_of_order_count"]) == (1, 1)
    assert window["stale_ratio"] == pytest.approx(0.25)
    assert window["commands_received"] == 4
    assert window["freshness_ms"] == pytest.approx(
        dict.fromkeys(("p50", "p95", "p99", "max", "mean"), 600.0)
    )
    assert window["commands_per_s_per_robot"]["mean"] == pytest.approx(4 / 30)
    assert window["applied_per_s_per_robot"]["mean"] == pytest.approx(2 / 30)
    # Only the applied commands are in effect: the first is replaced by the last.
    assert window["command_age_peak_ms"]["max"] == pytest.approx(26_000.0)


def aged(run: SyntheticRun, robot_id: str, count: int, verdict: Verdict) -> SyntheticRun:
    """Return the run with the robot's last count cls commands captured 600 ms earlier.

    The robot gave those commands the verdict.
    """
    rows = run.samples[robot_id].commands.copy()
    chosen = np.flatnonzero(rows["model"] == MODEL_KEYS.index("cls"))[-count:]
    rows["capture_ns"][chosen] -= 600 * MS
    rows["verdict"][chosen] = VERDICTS.index(verdict)
    return with_samples(run, robot_id, rows)


@pytest.mark.parametrize(
    ("verdict", "stale_count"), [(Verdict.STALE, 60), (Verdict.OUT_OF_ORDER, 0)]
)
def test_a_robots_freshness_counts_its_commands_of_every_verdict(
    verdict: Verdict, stale_count: int
) -> None:
    # Of b's 600 cls commands after its join, the last 60 are 620 ms old, the rest 20 ms.
    run = aged(synthetic_run(late_joiner(), run_spec("late_joiner")), "b", 60, verdict)

    robot: Any = summary_of(run.run)["robots"]["b"]["cls"]["after"]

    assert robot["freshness_ms"] == pytest.approx(
        {"p50": 20.0, "p95": 620.0, "p99": 620.0, "max": 620.0}
    )
    assert robot["stale_count"] == stale_count


@pytest.mark.parametrize(
    ("scenario", "topology"),
    [(disconnect(), Topology.SINGLE), (late_joiner(), Topology.SPLIT)],
    ids=["a killed robot", "the robots' own container"],
)
def test_summarize_run_reads_the_run_from_its_files(
    tmp_path: Path, scenario: Scenario, topology: Topology
) -> None:
    run = synthetic_run(scenario, run_spec(scenario.name, topology=topology))
    run_dir = write_finished_run(tmp_path, run)

    summary = summarize_run(run_dir)

    assert summary == {"summarizer_digest": summarizer_digest(), **summary_of(run.run)}
    assert json.loads((run_dir / SUMMARY_FILE).read_bytes()) == json.loads(json.dumps(summary))


def test_a_hand_run_has_no_summary(tmp_path: Path) -> None:
    run_dir = write_hand_run(tmp_path, steady_run())

    with pytest.raises(SummaryError, match="not a serving run of a plan"):
        summarize_run(run_dir)


def test_the_model_benchmark_summary_holds_the_means_of_its_reports(tmp_path: Path) -> None:
    spec = run_spec("model_bench", model_set=ModelSet.REAL)
    run_dir = write_model_bench_run(tmp_path, spec)

    summary: Any = summarize_model_bench(run_dir)

    assert summary["summarizer_digest"] == summarizer_digest()
    assert summary["tables"]["cls"]["t1"]["published"]["4"] == pytest.approx(
        {"wall_ms": 16.0, "cpu_ms": 16.0}
    )
    assert summary["tables"]["depth"]["t2"] == {
        "published": {"1": pytest.approx({"wall_ms": 30.0, "cpu_ms": 60.0})},
        "batched": {
            str(size): pytest.approx({"wall_ms": 31.5 * size, "cpu_ms": 63.0 * size})
            for size in (1, 2, 4, 8)
        },
    }
    assert summary["stages"]["decode_half"] == pytest.approx({"wall_ms": 1.0, "cpu_ms": 1.0})
    assert summary["dispatch"]["2"] == pytest.approx(
        {"wall_ms_mean": 0.2, "wall_ms_p50": 0.196, "gateway_cpu_us": 80.0, "worker_cpu_us": 60.0}
    )
    assert summary["websocket"] == pytest.approx(
        {
            "round_trip_ms_mean": 0.4,
            "receiver_cpu_us_per_message": 60.0,
            "sender_cpu_us_per_message": 50.0,
        }
    )
    assert json.loads((run_dir / SUMMARY_FILE).read_bytes()) == json.loads(json.dumps(summary))


def broken_reports(
    change: Callable[[dict[str, dict[str, Any]]], object],
) -> dict[str, dict[str, Any]]:
    reports = model_bench_reports()
    change(reports)
    return reports


@pytest.mark.parametrize(
    ("reports", "message"),
    [
        (
            broken_reports(lambda reports: reports.pop("stages")),
            "one model_bench.stages report, and has 0",
        ),
        (
            broken_reports(lambda reports: reports.update(again=reports["dispatch"])),
            "one transport_bench.dispatch report, and has 2",
        ),
        (
            broken_reports(lambda reports: reports.update(again=reports["batch_cls_t1_published"])),
            "repeats the table of another batch report",
        ),
        (
            broken_reports(
                lambda reports: reports.update(
                    other={**reports["websocket"], "kind": "transport_bench.other"}
                )
            ),
            "reports of other kinds: transport_bench.other",
        ),
        (
            broken_reports(lambda reports: reports["websocket"]["results"].pop("sender")),
            "websocket.json is not a report of its kind",
        ),
    ],
)
def test_a_model_benchmark_run_with_other_reports_than_its_own_is_refused(
    tmp_path: Path, reports: Mapping[str, Mapping[str, Any]], message: str
) -> None:
    spec = run_spec("model_bench", model_set=ModelSet.REAL)
    run_dir = write_model_bench_run(tmp_path, spec)
    shutil.rmtree(run_dir / REPORTS_DIRECTORY)
    (run_dir / REPORTS_DIRECTORY).mkdir()
    for name, report in reports.items():
        (run_dir / REPORTS_DIRECTORY / f"{name}.json").write_text(json.dumps(report))

    with pytest.raises(SummaryError, match=message):
        summarize_model_bench(run_dir)


def test_the_digest_covers_this_module_and_what_it_imports_in_the_repository() -> None:
    files = module_files(REPOSITORY_ROOT, "bench/summarize.py")

    assert {
        "bench/__init__.py",
        "bench/summarize.py",
        "bench/capacity_model.py",
        "bench/simulation.py",
        "bench/runfiles.py",
        "bench/stats.py",
        "bench/live_check.py",
        "cloud/scheduler.py",
        "robot/samples.py",
        "common/histogram.py",
    } <= set(files)
    assert not any(path.startswith("tests/") for path in files)
    assert files["bench/stats.py"] == (REPOSITORY_ROOT / "bench" / "stats.py").read_bytes()
    assert len(summarizer_digest()) == 64


def test_the_imports_are_followed_through_packages_functions_and_relative_names(
    tmp_path: Path,
) -> None:
    sources = {
        "app/__init__.py": "",
        "app/start.py": "import json\nfrom app import near\nfrom .deep import leaf\n",
        "app/near.py": "def later():\n    import other.inside\n",
        "app/deep/__init__.py": "from .. import sibling\n",
        "app/deep/leaf.py": "import numpy\n",
        "app/sibling.py": "",
        "app/unused.py": "",
        "other/__init__.py": "",
        "other/inside.py": "",
    }
    for path, source in sources.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(source)

    files = module_files(tmp_path, "app/start.py")

    assert set(files) == set(sources) - {"app/unused.py"}


def summary_hash(run_dir: Path) -> str:
    return hashlib.sha256((run_dir / SUMMARY_FILE).read_bytes()).hexdigest()


def done(run_dir: Path) -> Done:
    return Done.from_json((run_dir / DONE_FILE).read_text())


STALE = {"summarizer_digest": "0" * 64, "made_up": True}


def missing_summary(run_dir: Path) -> None:
    (run_dir / SUMMARY_FILE).unlink()


def other_digest(run_dir: Path) -> None:
    store_summary(run_dir, STALE)


def interrupted_rewrite(run_dir: Path) -> None:
    # The summary was replaced and DONE still holds the hash of the one before.
    (run_dir / SUMMARY_FILE).write_bytes(b'{"summarizer_digest":"interrupted"}\n')


@pytest.mark.parametrize("spoil", [missing_summary, other_digest, interrupted_rewrite])
def test_the_command_writes_a_summary_that_is_not_current(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], spoil: Callable[[Path], None]
) -> None:
    run = steady_run()
    run_dir = write_finished_run(tmp_path, run)
    files_before = done(run_dir).files
    spoil(run_dir)

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert json.loads((run_dir / SUMMARY_FILE).read_bytes()) == json.loads(
        json.dumps({"summarizer_digest": summarizer_digest(), **summary_of(run.run)})
    )
    assert done(run_dir).summary_sha256 == summary_hash(run_dir)
    assert done(run_dir).files == files_before
    assert capsys.readouterr().out == f"raw/steady/{run_dir.name}: summary written\n"
    # Nothing is left behind where the temporary files stood.
    assert [path.name for path in run_dir.parent.iterdir()] == [run_dir.name]


def test_the_command_leaves_a_current_summary_as_it_is(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A summary that this code would not compute, but that carries its digest.
    run_dir = write_finished_run(tmp_path, steady_run(), {"made_up": True})
    before = (run_dir / SUMMARY_FILE).read_bytes(), (run_dir / DONE_FILE).read_bytes()

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert ((run_dir / SUMMARY_FILE).read_bytes(), (run_dir / DONE_FILE).read_bytes()) == before
    assert capsys.readouterr().out == f"raw/steady/{run_dir.name}: current\n"


def test_the_command_leaves_and_lists_a_stale_serving_run_without_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = write_finished_run(tmp_path, steady_run(), STALE)
    shutil.rmtree(run_dir / SAMPLES_DIRECTORY)
    before = (run_dir / SUMMARY_FILE).read_bytes(), (run_dir / DONE_FILE).read_bytes()

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert ((run_dir / SUMMARY_FILE).read_bytes(), (run_dir / DONE_FILE).read_bytes()) == before
    assert capsys.readouterr().out == (
        f"raw/steady/{run_dir.name}: left: its summary is not current and it has no samples "
        "to write it from\n"
    )


def changed_raw_file(results: Path) -> Path:
    run_dir = write_finished_run(results, steady_run(), STALE)
    with (run_dir / EVENTS_FILE).open("a") as file:
        file.write("\n")
    return run_dir


def no_done(results: Path) -> Path:
    run_dir = write_finished_run(results, steady_run(), STALE)
    (run_dir / DONE_FILE).unlink()
    return run_dir


def attempt(results: Path) -> Path:
    return write_attempt(results, steady_run(), STALE)


def hand_run(results: Path) -> Path:
    return write_hand_run(results, steady_run())


@pytest.mark.parametrize(
    ("write", "outcome"),
    [
        (changed_raw_file, "left: no valid DONE"),
        (no_done, "left: no valid DONE"),
        (attempt, "left: an attempt"),
        (hand_run, "left: no valid DONE"),
    ],
)
def test_the_command_never_touches_a_run_that_is_not_finished(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    write: Callable[[Path], Path],
    outcome: str,
) -> None:
    run_dir = write(tmp_path)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()}

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert {path.name: path.read_bytes() for path in run_dir.iterdir() if path.is_file()} == before
    assert capsys.readouterr().out == f"raw/steady/{run_dir.name}: {outcome}\n"


def test_the_command_writes_a_stale_model_benchmark_summary_from_its_reports_alone(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run_dir = write_model_bench_run(tmp_path, run_spec("model_bench", model_set=ModelSet.REAL))
    store_summary(run_dir, STALE)

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert json.loads((run_dir / SUMMARY_FILE).read_bytes()) == summarize_model_bench(run_dir)
    assert done(run_dir).summary_sha256 == summary_hash(run_dir)
    assert capsys.readouterr().out == f"raw/model_bench/{run_dir.name}: summary written\n"


def test_the_command_finds_the_runs_of_a_comparison(tmp_path: Path) -> None:
    spec = run_spec("steady", model_set=ModelSet.REAL, candidate="fifo_vs_latest", arm="b")
    run_dir = write_finished_run(tmp_path, synthetic_run(steady(), spec), STALE)

    status = main(["--results", str(tmp_path)])

    assert status == 0
    assert run_dir == run_directory(tmp_path, spec)
    assert run_dir.parent == tmp_path / "raw" / "compare" / "fifo_vs_latest" / "b"
    assert (
        json.loads((run_dir / SUMMARY_FILE).read_bytes())["summarizer_digest"]
        == summarizer_digest()
    )


def test_a_run_whose_summary_cannot_be_computed_is_reported_and_the_others_are_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = write_finished_run(tmp_path, steady_run(), STALE)
    (broken / SAMPLES_DIRECTORY / "robot_a.npz").write_bytes(b"not an archive")
    other_code = synthetic_run(steady(), run_spec("steady", code_digest=OTHER_CODE_DIGEST))
    sound = write_finished_run(tmp_path, other_code, STALE)
    broken_before = (broken / SUMMARY_FILE).read_bytes(), (broken / DONE_FILE).read_bytes()

    status = main(["--results", str(tmp_path)])

    output = capsys.readouterr()
    assert status == 1
    assert f"raw/steady/{broken.name}: failed\n" in output.out
    assert f"raw/steady/{sound.name}: summary written\n" in output.out
    assert output.err.startswith(f"bench.summarize: no summary for {broken.name}: ")
    assert "robot_a.npz is not a samples file" in output.err
    # The run keeps its DONE, and with it its place among the finished runs.
    assert (
        (broken / SUMMARY_FILE).read_bytes(),
        (broken / DONE_FILE).read_bytes(),
    ) == broken_before
    assert (
        json.loads((sound / SUMMARY_FILE).read_bytes())["summarizer_digest"] == summarizer_digest()
    )


def test_the_command_exits_0_on_an_empty_results_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--results", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""
