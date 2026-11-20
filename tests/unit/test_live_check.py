"""The done criterion's checks on hand-made /metrics snapshots and sample arrays.

The command line runs through main: each scrape is answered with a hand-made snapshot, and the
idle window's wait is recorded instead of slept.
"""

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from bench import live_check
from bench.live_check import agreement, boundary_violations, conservation_violations, main
from common.commands import MODEL_KEYS
from common.histogram import BUCKETS_PER_OCTAVE, Histogram
from common.protocol import CommandHeader, StageDurations
from robot.control import Verdict
from robot.samples import COMMAND_DTYPE, SampleLog, TickOutcome
from robot.settings import LinkSettings

ROBOT_ID = "r1"
SECOND_ROBOT_ID = "r2"
SESSION = 7
FRAME_PERIOD_NS = 33_333_333
FIRE_LAG_NS = 1_000_000
# Eight stages of 0.1 to 0.8 ms: 3.6 ms in all.
STAGE_US = np.arange(100, 900, 100, dtype=np.uint32)
# result - send - stages: inside [0, 1000) ns, as the floored stages leave it.
RESIDUAL_NS = 200
# Each model's freshness in ms, as (value, number of samples). 201 samples, so that q * 201 is
# never whole and its rounding up matters: q0.5, q0.95 and q0.99 are the samples of ranks 101,
# 191 and 199. Each of those is alone at its value and a factor of about 1.3, three buckets,
# from its neighbours, so moving it by one bucket moves its own quantile and no other.
FRESHNESS_LEVELS_MS = {
    "cls": ((10.0, 100), (13.0, 1), (17.0, 89), (22.0, 1), (29.0, 7), (38.0, 1), (50.0, 2)),
    "depth": ((40.0, 100), (52.0, 1), (68.0, 89), (88.0, 1), (116.0, 7), (152.0, 1), (200.0, 2)),
}
QUANTILE_RANKS = {0.5: 101, 0.95: 191, 0.99: 199}
# The width of one histogram bucket, as a factor.
BUCKET_FACTOR = 2 ** (1 / BUCKETS_PER_OCTAVE)
DOWNLINK_QUEUE_COMMANDS = 8
URL = "http://127.0.0.1:8080"
NO_LINK_MODEL = LinkSettings()
PROCESS_NAMES = ("gateway", "worker_cls", "worker_depth")

Snapshot = dict[str, Any]
Commands = npt.NDArray[np.void]


def freshness_ms(model: str) -> list[float]:
    """Return a model's freshness samples in ms, in increasing order."""
    return [value_ms for value_ms, count in FRESHNESS_LEVELS_MS[model] for _ in range(count)]


def make_commands(session: int = SESSION) -> Commands:
    """Return one session's command samples, FRESHNESS_LEVELS_MS per model."""
    models = [MODEL_KEYS.index(model) for model in MODEL_KEYS for _ in freshness_ms(model)]
    freshness_ns = [
        round(value_ms * 1_000_000) for model in MODEL_KEYS for value_ms in freshness_ms(model)
    ]
    frame_numbers = np.arange(len(models), dtype=np.int64)
    capture_ns = 1_000_000_000 + frame_numbers * FRAME_PERIOD_NS
    commands = np.zeros(len(models), dtype=COMMAND_DTYPE)
    commands["session"] = session
    commands["model"] = models
    commands["frame_number"] = frame_numbers
    commands["capture_ns"] = capture_ns
    commands["send_ns"] = capture_ns + FIRE_LAG_NS
    commands["stage_us"] = STAGE_US
    commands["result_ns"] = capture_ns + FIRE_LAG_NS + 1000 * int(STAGE_US.sum()) + RESIDUAL_NS
    commands["receive_ns"] = capture_ns + np.array(freshness_ns, dtype=np.int64)
    return commands


def histogram_of(values_ms: Sequence[float]) -> dict[str, object]:
    histogram = Histogram()
    for value_ms in values_ms:
        histogram.record(value_ms)
    return histogram.to_dict()


def model_counters(commands_sent: int) -> dict[str, int]:
    """Return one model's gateway counters for a robot that was sent commands_sent commands."""
    # Each term of an identity has a value of its own, so that a term left out or counted twice
    # breaks it; together they need not make a plausible run.
    served = {
        "commands_sent": commands_sent,
        "commands_queued": 1,
        "commands_dropped_downlink": 2,
        "results_dropped_disconnect": 3,
        "frames_failed": 4,
        "in_flight": 5,
    }
    received = {
        "frames_served": sum(served.values()),
        "frames_overwritten": 6,
        "frames_skipped": 7,
        "frames_dropped_disconnect": 8,
        "frames_pending": 9,
    }
    return {**served, **received, "frames_received": sum(received.values())}


def robot_record(commands: Commands) -> dict[str, Any]:
    """Return a robot's /metrics record that agrees with its samples and keeps every identity."""
    rows = {model: commands[commands["model"] == MODEL_KEYS.index(model)] for model in MODEL_KEYS}
    models = {model: model_counters(rows[model].size) for model in MODEL_KEYS}
    # Every accepted frame goes to each subscribed model.
    frames_accepted = models[MODEL_KEYS[0]]["frames_received"]
    frames_rejected = 10
    frames_in = frames_rejected + frames_accepted
    return {
        "session": SESSION,
        "subscribed": list(MODEL_KEYS),
        "frames_in": frames_in,
        "frames_rejected": frames_rejected,
        "frames_accepted": frames_accepted,
        "models": models,
        "telemetry": {
            "frames_sent": frames_in,
            "reconnects": 0,
            "models": {
                model: {
                    "commands_received": rows[model].size,
                    # The quotient the robot records, as it records it.
                    "freshness_ms": histogram_of(
                        [
                            (int(row["receive_ns"]) - int(row["capture_ns"])) / 1e6
                            for row in rows[model]
                        ]
                    ),
                }
                for model in MODEL_KEYS
            },
        },
    }


def make_snapshot(commands: Commands, robot_ids: Sequence[str] = (ROBOT_ID,)) -> Snapshot:
    """Return a /metrics snapshot in which each robot agrees with commands as its samples."""
    return {
        "config": {"shared_clock": True, "downlink_queue_commands": DOWNLINK_QUEUE_COMMANDS},
        "robots": {robot_id: robot_record(commands) for robot_id in robot_ids},
    }


def telemetry_of(
    snapshot: Snapshot, model: str = "cls", robot_id: str = ROBOT_ID
) -> dict[str, Any]:
    entry: dict[str, Any] = snapshot["robots"][robot_id]["telemetry"]["models"][model]
    return entry


def counters_of(snapshot: Snapshot, model: str = "cls") -> dict[str, Any]:
    entry: dict[str, Any] = snapshot["robots"][ROBOT_ID]["models"][model]
    return entry


def test_samples_and_telemetry_that_agree_give_no_problem() -> None:
    commands = make_commands()

    assert agreement(make_snapshot(commands), ROBOT_ID, commands) == []


def test_only_the_samples_of_the_current_session_are_compared() -> None:
    commands = make_commands()
    snapshot = make_snapshot(commands)

    with_old_session = np.concatenate([make_commands(session=SESSION - 1), commands])

    assert agreement(snapshot, ROBOT_ID, with_old_session) == []


def count_off_by_one(snapshot: Snapshot, commands: Commands) -> None:
    telemetry_of(snapshot)["freshness_ms"] = histogram_of([*freshness_ms("cls"), 14.0])


def commands_received_off_by_one(snapshot: Snapshot, commands: Commands) -> None:
    telemetry_of(snapshot)["commands_received"] += 1


def max_off(snapshot: Snapshot, commands: Commands) -> None:
    values_ms = freshness_ms("cls")
    # The same bucket and the same count: only the maximum differs.
    telemetry_of(snapshot)["freshness_ms"] = histogram_of([*values_ms[:-1], values_ms[-1] + 1e-6])


def residual_of_one_microsecond(snapshot: Snapshot, commands: Commands) -> None:
    commands["result_ns"][0] += 1000 - RESIDUAL_NS


def negative_residual(snapshot: Snapshot, commands: Commands) -> None:
    commands["result_ns"][0] -= RESIDUAL_NS + 1


def robot_absent(snapshot: Snapshot, commands: Commands) -> None:
    del snapshot["robots"][ROBOT_ID]


def no_telemetry(snapshot: Snapshot, commands: Commands) -> None:
    del snapshot["robots"][ROBOT_ID]["telemetry"]


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (count_off_by_one, "histogram count"),
        (commands_received_off_by_one, "commands_received"),
        (max_off, "histogram max"),
        (residual_of_one_microsecond, "result - send - stages"),
        (negative_residual, "result - send - stages"),
        (robot_absent, "not in /metrics"),
        (no_telemetry, "no telemetry"),
    ],
)
def test_agreement_reports_each_disagreement_once(
    break_it: Callable[[Snapshot, Commands], None], named: str
) -> None:
    commands = make_commands()
    snapshot = make_snapshot(commands)

    break_it(snapshot, commands)

    assert [named in problem for problem in agreement(snapshot, ROBOT_ID, commands)] == [True]


@pytest.mark.parametrize("model", MODEL_KEYS)
@pytest.mark.parametrize("q", sorted(QUANTILE_RANKS))
@pytest.mark.parametrize("factor", [BUCKET_FACTOR, 1 / BUCKET_FACTOR], ids=["up", "down"])
def test_a_quantile_one_bucket_from_the_sample_of_its_rank_is_reported(
    model: str, q: float, factor: float
) -> None:
    commands = make_commands()
    snapshot = make_snapshot(commands)
    values_ms = freshness_ms(model)

    # Moved up, the histogram's quantile is the upper edge of the next bucket, more than a bucket
    # above the sample; moved down, it is the lower edge of the sample's bucket, below it.
    values_ms[QUANTILE_RANKS[q] - 1] *= factor
    telemetry_of(snapshot, model)["freshness_ms"] = histogram_of(values_ms)

    problems = agreement(snapshot, ROBOT_ID, commands)
    assert [problem.startswith(f"{ROBOT_ID} {model}: histogram q{q} ") for problem in problems] == [
        True
    ]


def test_the_residual_is_not_checked_when_the_clocks_are_not_shared() -> None:
    commands = make_commands()
    snapshot = make_snapshot(commands)
    snapshot["config"]["shared_clock"] = False

    commands["result_ns"][0] += 5_000_000

    assert agreement(snapshot, ROBOT_ID, commands) == []


def test_a_snapshot_that_keeps_every_identity_has_no_violation() -> None:
    assert conservation_violations(make_snapshot(make_commands())) == []


def frames_in_broken(snapshot: Snapshot) -> None:
    snapshot["robots"][ROBOT_ID]["frames_in"] += 1


def frames_received_broken(snapshot: Snapshot) -> None:
    counters_of(snapshot, "depth")["frames_pending"] += 1


def frames_served_broken(snapshot: Snapshot) -> None:
    counters_of(snapshot)["in_flight"] += 1


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (frames_in_broken, "r1: frames_in"),
        (frames_received_broken, "r1 depth: frames_received"),
        (frames_served_broken, "r1 cls: frames_served"),
    ],
)
def test_each_broken_identity_is_reported(break_it: Callable[[Snapshot], None], named: str) -> None:
    snapshot = make_snapshot(make_commands())

    break_it(snapshot)

    assert [named in problem for problem in conservation_violations(snapshot)] == [True]


def test_counters_that_match_across_the_connection_give_no_problem() -> None:
    assert boundary_violations(make_snapshot(make_commands()), ROBOT_ID) == []


@pytest.mark.parametrize("unread_commands", [0, 1, DOWNLINK_QUEUE_COMMANDS])
def test_commands_still_in_flight_at_the_stop_are_tolerated_up_to_the_downlink_queue(
    unread_commands: int,
) -> None:
    snapshot = make_snapshot(make_commands())

    telemetry_of(snapshot)["commands_received"] -= unread_commands

    assert boundary_violations(snapshot, ROBOT_ID) == []


def frames_sent_above(snapshot: Snapshot) -> None:
    snapshot["robots"][ROBOT_ID]["telemetry"]["frames_sent"] += 1


def frames_sent_below(snapshot: Snapshot) -> None:
    snapshot["robots"][ROBOT_ID]["telemetry"]["frames_sent"] -= 1


def commands_received_above_sent(snapshot: Snapshot) -> None:
    telemetry_of(snapshot)["commands_received"] += 1


def commands_received_beyond_the_tolerance(snapshot: Snapshot) -> None:
    telemetry_of(snapshot, "depth")["commands_received"] -= DOWNLINK_QUEUE_COMMANDS + 1


def reconnected(snapshot: Snapshot) -> None:
    snapshot["robots"][ROBOT_ID]["telemetry"]["reconnects"] = 1


@pytest.mark.parametrize(
    ("break_it", "named"),
    [
        (frames_sent_above, "frames_sent"),
        (frames_sent_below, "frames_sent"),
        (commands_received_above_sent, "r1 cls: commands_sent"),
        (commands_received_beyond_the_tolerance, "r1 depth: commands_sent"),
        (reconnected, "reconnects 1"),
    ],
)
def test_boundary_reports_each_mismatch_once(
    break_it: Callable[[Snapshot], None], named: str
) -> None:
    snapshot = make_snapshot(make_commands())

    break_it(snapshot)

    assert [named in problem for problem in boundary_violations(snapshot, ROBOT_ID)] == [True]


def save_samples(
    path: Path, commands: Commands, uplink: LinkSettings, downlink: LinkSettings
) -> None:
    """Write commands to a samples file as a fleet does, with the robot's link model."""
    log = SampleLog(tick_capacity=commands.size, command_capacity=commands.size)
    for row in commands:
        log.tick(
            int(row["frame_number"]), int(row["capture_ns"]), int(row["send_ns"]), TickOutcome.SENT
        )
        header = CommandHeader(
            model=MODEL_KEYS[row["model"]],
            session=int(row["session"]),
            frame_number=int(row["frame_number"]),
            capture_ns=int(row["capture_ns"]),
            batch_size=int(row["batch_size"]),
            result_ns=int(row["result_ns"]),
            stages=StageDurations(*(int(stage_us) for stage_us in row["stage_us"])),
        )
        log.command(int(row["session"]), header, int(row["receive_ns"]), Verdict.APPLIED)
    log.save(path, {"settings": {"uplink": asdict(uplink), "downlink": asdict(downlink)}})


def answer_scrapes(monkeypatch: pytest.MonkeyPatch, *snapshots: Snapshot) -> None:
    """Answer the command's scrapes with these snapshots, one each, in place of a service."""
    scrapes = iter(snapshots)
    monkeypatch.setattr(live_check, "fetch_metrics", lambda url: next(scrapes))


def run_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot: Snapshot,
    uplink: LinkSettings = NO_LINK_MODEL,
    downlink: LinkSettings = NO_LINK_MODEL,
) -> int:
    """Run the agreement command on snapshot, with make_commands() as each robot's samples."""
    for robot_id in snapshot["robots"]:
        save_samples(tmp_path / f"robot_{robot_id}.npz", make_commands(), uplink, downlink)
    answer_scrapes(monkeypatch, snapshot)
    return main(["agreement", "--url", URL, "--samples-dir", str(tmp_path)])


def test_agreement_command_exits_0_when_every_robot_agrees_and_every_identity_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = make_snapshot(make_commands(), (ROBOT_ID, SECOND_ROBOT_ID))

    status = run_agreement(tmp_path, monkeypatch, snapshot)

    report = json.loads(capsys.readouterr().out)
    problems = {robot_id: robot["problems"] for robot_id, robot in report["robots"].items()}
    assert (status, problems, report["conservation"]) == (
        0,
        {ROBOT_ID: [], SECOND_ROBOT_ID: []},
        [],
    )


def test_agreement_command_exits_1_when_only_the_second_robot_disagrees(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = make_snapshot(make_commands(), (ROBOT_ID, SECOND_ROBOT_ID))
    telemetry_of(snapshot, "depth", SECOND_ROBOT_ID)["commands_received"] += 1

    status = run_agreement(tmp_path, monkeypatch, snapshot)

    report = json.loads(capsys.readouterr().out)
    failed = {robot_id: bool(robot["problems"]) for robot_id, robot in report["robots"].items()}
    assert (status, failed) == (1, {ROBOT_ID: False, SECOND_ROBOT_ID: True})


def test_agreement_command_exits_1_when_only_an_identity_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = make_snapshot(make_commands())
    counters_of(snapshot, "depth")["frames_pending"] += 1

    status = run_agreement(tmp_path, monkeypatch, snapshot)

    report = json.loads(capsys.readouterr().out)
    problems = report["robots"][ROBOT_ID]["problems"]
    assert (status, problems, len(report["conservation"])) == (1, [], 1)


@pytest.mark.parametrize(
    ("uplink", "downlink", "status"),
    [
        (NO_LINK_MODEL, NO_LINK_MODEL, 1),
        (LinkSettings(delay_ms=20.0), NO_LINK_MODEL, 0),
        (LinkSettings(jitter_ms=5.0), NO_LINK_MODEL, 0),
        (NO_LINK_MODEL, LinkSettings(delay_ms=30.0), 0),
        (NO_LINK_MODEL, LinkSettings(jitter_ms=5.0), 0),
    ],
    ids=["no-link-model", "uplink-delay", "uplink-jitter", "downlink-delay", "downlink-jitter"],
)
def test_agreement_command_compares_counters_across_the_connection_only_without_a_link_model(
    uplink: LinkSettings,
    downlink: LinkSettings,
    status: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = make_snapshot(make_commands())
    # A frame sent but not taken in: only the comparison across the connection sees it.
    frames_sent_above(snapshot)

    assert run_agreement(tmp_path, monkeypatch, snapshot, uplink, downlink) == status


def test_agreement_command_exits_2_on_a_directory_without_samples(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "plan.json").write_text("{}")

    # No service listens at this URL: the samples are checked before /metrics is read.
    status = main(["agreement", "--url", "http://127.0.0.1:9", "--samples-dir", str(tmp_path)])

    assert (status, capsys.readouterr().out) == (2, "")


def process_entry(pid: int, cpu_user_ms: int, cpu_system_ms: int) -> dict[str, int]:
    return {"pid": pid, "cpu_user_ms": cpu_user_ms, "cpu_system_ms": cpu_system_ms}


def idle_snapshot(
    ts_mono_ns: int,
    connected: int,
    cgroup_cpu_us: int,
    processes: Mapping[str, dict[str, int] | None],
) -> Snapshot:
    """Return the fields of a /metrics snapshot that the idle command reads."""
    return {
        "ts_mono_ns": ts_mono_ns,
        "connections": {"current": connected},
        "cgroup": {"cpu_usage_us": cgroup_cpu_us},
        "processes": dict(processes),
    }


def run_idle(
    monkeypatch: pytest.MonkeyPatch, first: Snapshot, second: Snapshot
) -> tuple[int, list[float]]:
    """Run the idle command over a 10 s window; return its status and the waits it asked for."""
    waits_s: list[float] = []
    answer_scrapes(monkeypatch, first, second)
    monkeypatch.setattr(live_check, "time", SimpleNamespace(sleep=waits_s.append))
    return main(["idle", "--url", URL, "--window-s", "10"]), waits_s


@pytest.mark.parametrize(
    "depth_after", [None, process_entry(13, 4, 1)], ids=["worker-gone", "worker-replaced"]
)
def test_idle_command_reports_the_cpu_used_within_its_window(
    depth_after: dict[str, int] | None,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = idle_snapshot(
        5_000_000_000,
        0,
        7_000_000,
        {
            "gateway": process_entry(10, 500, 100),
            "worker_cls": process_entry(11, 300, 50),
            "worker_depth": process_entry(12, 900, 80),
        },
    )
    second = idle_snapshot(
        15_000_000_000,
        0,
        7_045_000,
        {
            "gateway": process_entry(10, 520, 110),
            "worker_cls": process_entry(11, 302, 51),
            "worker_depth": depth_after,
        },
    )

    status, waits_s = run_idle(monkeypatch, first, second)

    # A worker that is not the same live process at both scrapes has no window.
    assert (status, waits_s) == (0, [10.0])
    assert json.loads(capsys.readouterr().out) == {
        "window_s": 10.0,
        "cpu_ms": {"gateway": 30, "worker_cls": 3, "worker_depth": None},
        "cgroup_cpu_us": 45_000,
        "robots_connected": [0, 0],
    }


@pytest.mark.parametrize(("connected", "status"), [((0, 0), 0), ((1, 0), 1), ((0, 1), 1)])
def test_idle_command_exits_1_when_a_robot_was_connected_at_either_scrape(
    connected: tuple[int, int], status: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    processes = {name: process_entry(pid, 100, 10) for pid, name in enumerate(PROCESS_NAMES, 10)}
    first, second = (
        idle_snapshot(ts_mono_ns, robots, 1_000_000, processes)
        for ts_mono_ns, robots in zip((0, 10_000_000_000), connected, strict=True)
    )

    assert run_idle(monkeypatch, first, second)[0] == status
