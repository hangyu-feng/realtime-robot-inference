"""Smoke tests of the model benchmark: each subcommand runs end to end and writes its report.

Every run is a process of its own, as in a scenario: the benchmark sets OpenCV's
thread count, renices itself and handles SIGTERM, none of which belong in the
test process, even when a broken check lets a run through that should stop.
"""

import json
import os
import select
import signal
import subprocess
import sys
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from bench.model_bench import MAX_NICE, MODEL_DIR_VARIABLE
from common.procstat import read_process_stats
from robot.frames import DEFAULT_FRAME_COUNT

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_NAME = "report.json"
# The exit status with which argparse refuses a command line.
USAGE_ERROR_STATUS = 2

# Bounds on how long a test waits for the benchmark. They end a test only when
# the benchmark hangs.
RUN_TIMEOUT_S = 300
READY_TIMEOUT_S = 300
EXIT_TIMEOUT_S = 120
# Longer than a test waits in all, so a load that ended in time was ended by its signal.
ENDLESS_DURATION_S = 3600

WARMUP_ITERATIONS = 1
MIN_ITERATIONS = 2
STAGE_ITERATIONS = 2
DURATION_S = 0.5
LOAD_BATCH_SIZE = 2
PACING_SHARE_CORES = 0.5
LOAD_NICE = 5

REPORT_KEYS = {"kind", "args", "environment", "frames", "results"}
SUMMARY_KEYS = {"count", "mean", "min", "p50", "p95", "p99", "max"}
BATCH_ENTRY_KEYS = {
    "batch_size",
    "iterations",
    "wall_ms",
    "mean_cpu_ms",
    "mean_decode_ms",
    "mean_preprocess_ms",
    "mean_inference_ms",
    "mean_postprocess_ms",
    "nr_throttled",
    "throttled_us",
}
STAGES = {
    "decode_full",
    "decode_half",
    "preprocess_cls_full",
    "preprocess_cls_half",
    "preprocess_depth_full",
    "preprocess_depth_half",
    "postprocess_cls",
    "postprocess_depth",
}
STAGE_KEYS = {"wall_ms", "mean_cpu_ms"}
LOAD_KEYS = {"batches", "frames", "wall_s", "cpu_s", "duty", "wall_ms", "paced_ms"}


def bench_command(*arguments: str) -> list[str]:
    return [sys.executable, "-m", "bench.model_bench", *arguments]


def run_bench(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        bench_command(*arguments),
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=RUN_TIMEOUT_S,
        check=False,
    )


def wait_until_ready(load: subprocess.Popen[str]) -> None:
    """Wait for the first line of a load, which must say that it is ready."""
    assert load.stdout is not None
    assert select.select([load.stdout], [], [], READY_TIMEOUT_S)[0], "nothing printed"
    assert load.stdout.readline() == "ready\n"


@pytest.fixture
def endless_load(tmp_path: Path) -> Iterator[subprocess.Popen[str]]:
    """Start a niced load that only a signal can end within the test, and kill it at teardown."""
    with subprocess.Popen(
        bench_command(
            "saturate",
            "--model=cls",
            "--graph=batched",
            f"--nice={LOAD_NICE}",
            f"--warmup-iterations={WARMUP_ITERATIONS}",
            f"--duration-s={ENDLESS_DURATION_S}",
            f"--out={tmp_path / REPORT_NAME}",
        ),
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        yield process
        # A test that fails before its signal would otherwise leave the load running.
        process.kill()


@pytest.mark.parametrize(
    ("model", "graph", "batch_sizes"),
    [("cls", "batched", [1, 2]), ("depth", "published", [1])],
)
def test_batch_writes_an_entry_for_each_batch_size(
    tmp_path: Path, model: str, graph: str, batch_sizes: list[int]
) -> None:
    out = tmp_path / REPORT_NAME

    completed = run_bench(
        "batch",
        f"--model={model}",
        f"--graph={graph}",
        f"--batch-sizes={','.join(map(str, batch_sizes))}",
        f"--warmup-iterations={WARMUP_ITERATIONS}",
        f"--min-iterations={MIN_ITERATIONS}",
        "--min-seconds=0",
        f"--out={out}",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(out.read_text())
    assert (set(report), report["kind"]) == (REPORT_KEYS, "model_bench.batch")
    entries = report["results"]
    assert [entry["batch_size"] for entry in entries] == batch_sizes
    assert all(set(entry) == BATCH_ENTRY_KEYS for entry in entries)
    assert all(set(entry["wall_ms"]) == SUMMARY_KEYS for entry in entries)
    assert all(
        entry["wall_ms"]["count"] == entry["iterations"] >= MIN_ITERATIONS for entry in entries
    )


def test_stages_writes_a_summary_of_each_stage(tmp_path: Path) -> None:
    out = tmp_path / REPORT_NAME

    completed = run_bench(
        "stages",
        f"--iterations={STAGE_ITERATIONS}",
        f"--warmup-iterations={WARMUP_ITERATIONS}",
        f"--out={out}",
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(out.read_text())
    assert (set(report), report["kind"]) == (REPORT_KEYS, "model_bench.stages")
    stages = report["results"]
    assert set(stages) == STAGES
    assert all(set(stage) == STAGE_KEYS for stage in stages.values())
    assert all(stage["wall_ms"]["count"] == STAGE_ITERATIONS for stage in stages.values())


def test_saturate_runs_paced_batches_until_its_duration_is_over(tmp_path: Path) -> None:
    out = tmp_path / REPORT_NAME

    completed = run_bench(
        "saturate",
        "--model=cls",
        "--graph=batched",
        f"--batch-size={LOAD_BATCH_SIZE}",
        f"--pacing-share={PACING_SHARE_CORES}",
        f"--warmup-iterations={WARMUP_ITERATIONS}",
        f"--duration-s={DURATION_S}",
        f"--out={out}",
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.splitlines()[0] == "ready"
    report = json.loads(out.read_text())
    assert (set(report), report["kind"]) == (REPORT_KEYS, "model_bench.saturate")
    results = report["results"]
    assert set(results) == LOAD_KEYS
    assert results["wall_s"] >= DURATION_S
    assert results["frames"] == LOAD_BATCH_SIZE * results["batches"]
    # At one intra-op thread a batch is on a core whenever it is not preempted.
    # At half a core it releases the next batch twice its CPU time after it
    # starts, so no cycle waits only if every batch was preempted for as long as
    # it ran.
    assert results["paced_ms"] > 0


def test_saturate_runs_at_the_niceness_it_was_given(endless_load: subprocess.Popen[str]) -> None:
    wait_until_ready(endless_load)

    # At one intra-op thread the batches run on the main thread, whose niceness
    # is the one that the stat file of the process shows.
    assert read_process_stats(endless_load.pid).nice == os.nice(0) + LOAD_NICE


def test_saturate_stops_early_and_writes_its_report_on_sigterm(
    endless_load: subprocess.Popen[str], tmp_path: Path
) -> None:
    wait_until_ready(endless_load)

    endless_load.send_signal(signal.SIGTERM)
    _, stderr = endless_load.communicate(timeout=EXIT_TIMEOUT_S)

    assert endless_load.returncode == 0, stderr
    results = json.loads((tmp_path / REPORT_NAME).read_text())["results"]
    assert set(results) == LOAD_KEYS
    assert results["wall_s"] < ENDLESS_DURATION_S


@pytest.mark.parametrize(
    "arguments",
    [
        ("batch", "--model=depth", "--graph=published", "--batch-sizes=1,2"),
        ("saturate", "--model=depth", "--graph=published", "--batch-size=2"),
    ],
)
def test_published_depth_graph_refuses_a_batch_size_above_one(
    tmp_path: Path, arguments: tuple[str, ...]
) -> None:
    out = tmp_path / REPORT_NAME

    completed = run_bench(*arguments, f"--out={out}")

    assert completed.returncode == USAGE_ERROR_STATUS
    assert "runs batch size 1 only, not 2: use --graph batched" in completed.stderr
    assert not out.exists()


def test_missing_model_file_fails_at_startup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    monkeypatch.setenv(MODEL_DIR_VARIABLE, str(model_dir))
    out = tmp_path / REPORT_NAME

    completed = run_bench("stages", f"--out={out}")

    assert completed.returncode == USAGE_ERROR_STATUS
    assert f"cannot load model 'cls' from {model_dir}" in completed.stderr
    assert not out.exists()


@pytest.mark.parametrize(
    "clear_variable",
    [
        pytest.param(lambda monkeypatch: monkeypatch.delenv(MODEL_DIR_VARIABLE), id="unset"),
        pytest.param(lambda monkeypatch: monkeypatch.setenv(MODEL_DIR_VARIABLE, ""), id="empty"),
    ],
)
def test_model_dir_variable_without_a_value_fails_at_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clear_variable: Callable[[pytest.MonkeyPatch], None],
) -> None:
    clear_variable(monkeypatch)
    out = tmp_path / REPORT_NAME

    completed = run_bench("stages", f"--out={out}")

    assert completed.returncode == USAGE_ERROR_STATUS
    assert f"{MODEL_DIR_VARIABLE} is not set" in completed.stderr
    assert not out.exists()


def test_out_path_that_is_a_directory_is_refused(tmp_path: Path) -> None:
    completed = run_bench("stages", f"--out={tmp_path}")

    assert completed.returncode == USAGE_ERROR_STATUS
    assert f"argument --out: {tmp_path} is a directory" in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "flag"),
    [
        (
            (
                "batch",
                "--model=cls",
                "--graph=batched",
                f"--batch-sizes=1,{DEFAULT_FRAME_COUNT + 1}",
            ),
            "--batch-sizes",
        ),
        (("saturate", "--model=cls", "--graph=batched", f"--nice={MAX_NICE + 1}"), "--nice"),
        (("saturate", "--model=cls", "--graph=batched", "--pacing-share=0"), "--pacing-share"),
    ],
)
def test_argument_out_of_range_is_refused(
    tmp_path: Path, arguments: tuple[str, ...], flag: str
) -> None:
    out = tmp_path / REPORT_NAME

    completed = run_bench(*arguments, f"--out={out}")

    assert completed.returncode == USAGE_ERROR_STATUS
    assert f"argument {flag}:" in completed.stderr
    assert not out.exists()
