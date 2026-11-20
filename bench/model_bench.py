"""Measure the code path of a model worker: batch times, stage costs and a saturating load.

Batches go through ModelRunner.run_batch, the call that a worker serves with, so
the batch cap and the capacity model of DESIGN.md section 6 rest on the code that
serves. A run writes one JSON report, with the keys kind, args, environment,
frames and results, and then prints a short summary. The results:

- batch: an entry per batch size with iterations, wall_ms (a summary of the
  batch times), mean_cpu_ms per batch, mean_decode_ms, mean_preprocess_ms and
  mean_postprocess_ms per frame, mean_inference_ms per batch, and nr_throttled
  and throttled_us, how far those cgroup counters moved during the timed loop.
- stages: decode_<size>, preprocess_<model>_<size> and postprocess_<model>, each
  with wall_ms and the mean_cpu_ms of one call. A size is full or half.
- saturate: batches, frames, wall_s, cpu_s, duty (cpu_s over wall_s), wall_ms
  and paced_ms. It prints "ready" before its loop and ends the loop early on
  SIGTERM, so that another benchmark can run it as the load of a busy worker.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import signal
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from functools import partial
from pathlib import Path
from statistics import fmean
from types import FrameType

import numpy as np
from numpy.typing import NDArray

from bench.environment import collect_environment
from bench.stats import summarize
from common.cgroup import read_stats
from common.clock import Clock, MonotonicClock
from common.commands import Command
from common.errors import ConfigError
from robot.frames import (
    DEFAULT_FRAME_COUNT,
    DEFAULT_JPEG_QUALITY,
    DEFAULT_SEED,
    FrameSet,
    build_frame_set,
)
from workers.models import MODELS, ModelSpec
from workers.runtime import (
    BatchOutcome,
    ModelLoadError,
    ModelRunner,
    SessionSettings,
    configure_process_threads,
    decode_jpeg,
)

MODEL_DIR_VARIABLE = "MODEL_DIR"

# The serving baseline of DESIGN.md section 4.6: one intra-op thread, and no spinning.
DEFAULT_THREADS = 1
DEFAULT_SPINNING = "off"
# The batch sizes that a worker calibrates at warm-up (DESIGN.md section 6.2).
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_WARMUP_ITERATIONS = 5
DEFAULT_MIN_ITERATIONS = 30
DEFAULT_MIN_SECONDS = 2.0
DEFAULT_STAGE_ITERATIONS = 300
DEFAULT_BATCH_SIZE = 1
DEFAULT_NICE = 0
# The lowest priority that Linux has. It turns a higher niceness into this one
# without an error, and the report would then record one that the load never had.
MAX_NICE = 19
DEFAULT_DURATION_S = 30.0

_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000

# Each decode size by its name in the report, as the half_size argument of decode_jpeg.
_DECODE_SIZES = {"full": False, "half": True}


def paced_start_ns(start_ns: int, cpu_ns: int, share_cores: float) -> int:
    """Return the earliest start of the next batch, after one that began at start_ns.

    The batch used cpu_ns of CPU time. From its start to the returned time, CPU
    time over wall time is at or under the share.
    """
    # Whole nanoseconds from exact arithmetic. A float quotient can round down,
    # which would start the next batch early and leave the duty over the share.
    return start_ns + math.ceil(cpu_ns / Fraction(share_cores))


@dataclass(frozen=True, slots=True)
class _Timed[T]:
    """One timed call. The CPU time is that of every thread of the process."""

    result: T
    start_ns: int
    cpu_ns: int


@dataclass(frozen=True)
class _Summary:
    """What a report says about the calls of one kind."""

    wall_ms: dict[str, float]
    mean_cpu_ms: float

    def describe(self) -> str:
        return (
            f"wall mean {self.wall_ms['mean']:.3f} ms, p95 {self.wall_ms['p95']:.3f} ms, "
            f"cpu mean {self.mean_cpu_ms:.3f} ms"
        )


class _Samples:
    """Times the calls of one kind: the wall time of each, and the CPU time of them all."""

    def __init__(self, clock: Clock, process_time_ns: Callable[[], int]) -> None:
        self._clock = clock
        self._process_time_ns = process_time_ns
        # One entry per call. The iteration count or the duration of a run bounds
        # the number of calls, so the list cannot grow past that.
        self.wall_ms: list[float] = []
        self.cpu_ns = 0

    def measure[T](self, call: Callable[[], T]) -> _Timed[T]:
        """Run one call and record its wall time and the CPU time of the process."""
        # Reading the CPU clock is a system call. Inside the pair of wall-clock
        # readings it would count as wall time of the shortest stages.
        cpu_start_ns = self._process_time_ns()
        start_ns = self._clock.now_ns()
        result = call()
        wall_ns = self._clock.now_ns() - start_ns
        cpu_ns = self._process_time_ns() - cpu_start_ns
        self.wall_ms.append(wall_ns / _NS_PER_MS)
        self.cpu_ns += cpu_ns
        return _Timed(result, start_ns, cpu_ns)

    def summary(self) -> _Summary:
        return _Summary(summarize(self.wall_ms), self.cpu_ns / len(self.wall_ms) / _NS_PER_MS)


@dataclass(frozen=True)
class _Measurement:
    """The results that one subcommand reports, and the lines that summarize them."""

    results: object
    summary_lines: list[str]


def _cycle_batches(jpegs: Sequence[bytes], batch_size: int) -> Iterator[tuple[bytes, ...]]:
    """Yield batches without end, each of the next frames of the set."""
    # A batch size is at most the size of the set, so the frames of a batch are
    # distinct. Moving on through the set times every batch size over the same
    # mix of frames, where the first few frames alone would differ between sizes.
    return itertools.batched(itertools.cycle(jpegs), batch_size)


def _load_runner(
    arguments: argparse.Namespace,
    model_dir: Path,
    clock: Clock,
    *,
    allow_spinning: bool,
    batch_sizes: Sequence[int],
) -> ModelRunner:
    """Load the graph that the arguments name, and refuse a batch size it cannot run."""
    spec = MODELS[arguments.model]
    file_name = spec.published_file if arguments.graph == "published" else spec.batched_file
    settings = SessionSettings(intra_op_threads=arguments.threads, allow_spinning=allow_spinning)
    runner = ModelRunner(spec, model_dir / file_name, settings, clock)
    # A graph whose batch dimension is fixed runs that batch size and no other.
    refused = [size for size in batch_sizes if runner.max_batch_size not in (None, size)]
    if refused:
        raise ConfigError(
            f"the {arguments.graph} {arguments.model} graph runs batch size "
            f"{runner.max_batch_size} only, not {', '.join(map(str, refused))}: "
            "use --graph batched"
        )
    return runner


def _counter_delta(before: int | None, after: int | None) -> int | None:
    """Return how far a cgroup counter moved. None where the kernel does not expose it."""
    if before is None or after is None:
        return None
    return after - before


def _stage_means(outcomes: Sequence[BatchOutcome]) -> dict[str, float]:
    """Return the mean of each worker stage: inference per batch, and the others per frame."""
    frames = [frame for outcome in outcomes for frame in outcome.frames]
    return {
        "mean_decode_ms": fmean(frame.decode_ns for frame in frames) / _NS_PER_MS,
        "mean_preprocess_ms": fmean(frame.preprocess_ns for frame in frames) / _NS_PER_MS,
        "mean_inference_ms": fmean(outcome.inference_ns for outcome in outcomes) / _NS_PER_MS,
        "mean_postprocess_ms": fmean(frame.postprocess_ns for frame in frames) / _NS_PER_MS,
    }


def _time_batch_size(
    runner: ModelRunner,
    jpegs: Sequence[bytes],
    batch_size: int,
    arguments: argparse.Namespace,
    clock: Clock,
) -> _Measurement:
    """Warm up and time one batch size, and return its entry of the report."""
    batches = _cycle_batches(jpegs, batch_size)
    for batch in itertools.islice(batches, arguments.warmup_iterations):
        runner.run_batch(batch)
    samples = _Samples(clock, time.process_time_ns)
    # One outcome per timed batch, bounded as the samples are.
    outcomes: list[BatchOutcome] = []
    min_loop_ns = math.ceil(arguments.min_seconds * _NS_PER_S)
    cgroup_before = read_stats()
    loop_start_ns = clock.now_ns()
    while len(outcomes) < arguments.min_iterations or clock.now_ns() - loop_start_ns < min_loop_ns:
        outcomes.append(samples.measure(partial(runner.run_batch, next(batches))).result)
    cgroup_after = read_stats()
    summary = samples.summary()
    # The counters cover the whole container, so a loop during which anything in
    # it was throttled is marked.
    nr_throttled = _counter_delta(cgroup_before.nr_throttled, cgroup_after.nr_throttled)
    throttled_us = _counter_delta(cgroup_before.throttled_us, cgroup_after.throttled_us)
    entry = {
        "batch_size": batch_size,
        "iterations": len(outcomes),
        **asdict(summary),
        **_stage_means(outcomes),
        "nr_throttled": nr_throttled,
        "throttled_us": throttled_us,
    }
    line = (
        f"batch size {batch_size}: {len(outcomes)} iterations, {summary.describe()}, "
        f"nr_throttled {nr_throttled}, throttled_us {throttled_us}"
    )
    return _Measurement(entry, [line])


def _batch_command(
    arguments: argparse.Namespace, model_dir: Path, frame_set: FrameSet, clock: Clock
) -> _Measurement:
    runner = _load_runner(
        arguments,
        model_dir,
        clock,
        allow_spinning=arguments.spinning == "on",
        batch_sizes=arguments.batch_sizes,
    )
    measurements = [
        _time_batch_size(runner, frame_set.jpegs, batch_size, arguments, clock)
        for batch_size in arguments.batch_sizes
    ]
    return _Measurement(
        [measurement.results for measurement in measurements],
        [line for measurement in measurements for line in measurement.summary_lines],
    )


def _output_row(spec: ModelSpec, model_dir: Path, jpeg: bytes, clock: Clock) -> NDArray[np.float32]:
    """Return the output row of one real inference on one frame."""
    rows: list[NDArray[np.float32]] = []

    def keep_row(row: NDArray[np.float32]) -> Command:
        rows.append(row)
        return spec.postprocess(row)

    # A runner hands an output row to the postprocessing of its model and to
    # nothing else, so a model that keeps the row is how one leaves the serving code.
    settings = SessionSettings(intra_op_threads=DEFAULT_THREADS, allow_spinning=False)
    runner = ModelRunner(
        replace(spec, postprocess=keep_row), model_dir / spec.published_file, settings, clock
    )
    runner.run_batch([jpeg])
    (row,) = rows
    return row


def _time_stages(
    jpeg: bytes, rows: Mapping[str, NDArray[np.float32]], samples: defaultdict[str, _Samples]
) -> None:
    """Time one call of every stage on one frame."""
    for size, half_size in _DECODE_SIZES.items():
        decode = partial(decode_jpeg, jpeg, half_size=half_size)
        image = samples[f"decode_{size}"].measure(decode).result
        # A worker preprocesses an image right after decoding it, so the same
        # order is kept here.
        for key, spec in MODELS.items():
            samples[f"preprocess_{key}_{size}"].measure(partial(spec.preprocess, image))
    for key, spec in MODELS.items():
        samples[f"postprocess_{key}"].measure(partial(spec.postprocess, rows[key]))


def _stages_command(
    arguments: argparse.Namespace, model_dir: Path, frame_set: FrameSet, clock: Clock
) -> _Measurement:
    rows = {
        key: _output_row(spec, model_dir, frame_set.jpegs[0], clock) for key, spec in MODELS.items()
    }
    jpegs = itertools.cycle(frame_set.jpegs)
    samples: defaultdict[str, _Samples] = defaultdict(
        partial(_Samples, clock, time.process_time_ns)
    )
    for jpeg in itertools.islice(jpegs, arguments.warmup_iterations):
        _time_stages(jpeg, rows, samples)
    samples.clear()
    for jpeg in itertools.islice(jpegs, arguments.iterations):
        _time_stages(jpeg, rows, samples)
    summaries = {stage: stage_samples.summary() for stage, stage_samples in samples.items()}
    return _Measurement(
        {stage: asdict(summary) for stage, summary in summaries.items()},
        [f"{stage}: {summary.describe()}" for stage, summary in summaries.items()],
    )


@dataclass
class StopRequest:
    """Set by the SIGTERM handler, and read by the load between batches."""

    requested: bool = False

    def request(self, signal_number: int, frame: FrameType | None) -> None:
        self.requested = True


@dataclass(frozen=True)
class LoadResults:
    """What a run of the load reports."""

    batches: int
    frames: int
    wall_s: float
    cpu_s: float
    duty: float  # cpu_s over wall_s
    wall_ms: dict[str, float]  # a summary of the batch times, which leave out the pacing
    paced_ms: float  # the time slept to hold the duty at the share


def _wait_for_share(
    start_ns: int, cpu_ns: int, share_cores: float, clock: Clock, sleep: Callable[[float], None]
) -> int:
    """Sleep until the share lets the next batch start, and return the time slept."""
    wait_start_ns = clock.now_ns()
    release_ns = paced_start_ns(start_ns, cpu_ns, share_cores)
    if release_ns <= wait_start_ns:
        return 0
    # A signal does not cut time.sleep short (PEP 475), so a stop waits for the
    # end of the cycle.
    sleep((release_ns - wait_start_ns) / _NS_PER_S)
    return clock.now_ns() - wait_start_ns


def run_load(
    run_batch: Callable[[tuple[bytes, ...]], object],
    batches: Iterator[tuple[bytes, ...]],
    *,
    duration_s: float,
    share_cores: float | None,
    stop: StopRequest,
    clock: Clock,
    process_time_ns: Callable[[], int],
    sleep: Callable[[float], None],
) -> LoadResults:
    """Run batches until the duration is over or a stop is requested.

    With a share, each batch after the first starts no earlier than paced_start_ns
    allows. A real run passes time.process_time_ns and time.sleep.
    """
    samples = _Samples(clock, process_time_ns)
    frames = 0
    paced_ns = 0
    cpu_start_ns = process_time_ns()
    loop_start_ns = clock.now_ns()
    # In whole nanoseconds: past 2**53 ns of uptime a float rounds them, and a
    # deadline rounded down would end the load before its duration.
    deadline_ns = loop_start_ns + math.ceil(duration_s * _NS_PER_S)
    # The end is checked only after the pacing wait, so the loop is made of whole
    # cycles and its duty can be held against the share, and the batch summary
    # has a sample. The pacing covers the CPU time of the batches; the loop's own
    # bookkeeping between them is not paced.
    while True:
        batch = next(batches)
        timed = samples.measure(partial(run_batch, batch))
        frames += len(batch)
        if share_cores is not None:
            paced_ns += _wait_for_share(timed.start_ns, timed.cpu_ns, share_cores, clock, sleep)
        if stop.requested or clock.now_ns() >= deadline_ns:
            break
    wall_s = (clock.now_ns() - loop_start_ns) / _NS_PER_S
    cpu_s = (process_time_ns() - cpu_start_ns) / _NS_PER_S
    return LoadResults(
        batches=len(samples.wall_ms),
        frames=frames,
        wall_s=wall_s,
        cpu_s=cpu_s,
        duty=cpu_s / wall_s,
        wall_ms=summarize(samples.wall_ms),
        paced_ms=paced_ns / _NS_PER_MS,
    )


def _saturate_command(
    arguments: argparse.Namespace, model_dir: Path, frame_set: FrameSet, clock: Clock
) -> _Measurement:
    # A thread starts with the niceness of the thread that creates it, so the
    # niceness is set before the session and its threads exist.
    os.nice(arguments.nice)
    # A worker never spins (DESIGN.md section 4.6), and this load stands in for one.
    runner = _load_runner(
        arguments, model_dir, clock, allow_spinning=False, batch_sizes=[arguments.batch_size]
    )
    batches = _cycle_batches(frame_set.jpegs, arguments.batch_size)
    for batch in itertools.islice(batches, arguments.warmup_iterations):
        runner.run_batch(batch)
    stop = StopRequest()
    # The handler stays in place, so a second SIGTERM cannot end the process
    # before its report is written.
    signal.signal(signal.SIGTERM, stop.request)
    print("ready", flush=True)
    results = run_load(
        runner.run_batch,
        batches,
        duration_s=arguments.duration_s,
        share_cores=arguments.pacing_share,
        stop=stop,
        clock=clock,
        process_time_ns=time.process_time_ns,
        sleep=time.sleep,
    )
    line = (
        f"{results.batches} batches of {arguments.batch_size} in {results.wall_s:.3f} s, "
        f"duty {results.duty:.3f}, paced {results.paced_ms:.3f} ms"
    )
    return _Measurement(asdict(results), [line])


_COMMANDS: Mapping[str, Callable[[argparse.Namespace, Path, FrameSet, Clock], _Measurement]] = {
    "batch": _batch_command,
    "stages": _stages_command,
    "saturate": _saturate_command,
}


def _whole_number(minimum: int, maximum: int | None = None) -> Callable[[str], int]:
    """Return an argparse type that accepts the whole numbers of a range."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"{text!r} is not a whole number") from error
        if value < minimum or (maximum is not None and value > maximum):
            allowed = f"at least {minimum}" if maximum is None else f"{minimum} to {maximum}"
            raise argparse.ArgumentTypeError(f"must be {allowed}, got {value}")
        return value

    return parse


# A batch is built from distinct frames, so it cannot be larger than the frame set.
_batch_size = _whole_number(1, DEFAULT_FRAME_COUNT)


def _batch_sizes(text: str) -> list[int]:
    return [_batch_size(part) for part in text.split(",")]


def _finite_number(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be finite, got {text}")
    return value


def _seconds(text: str) -> float:
    value = _finite_number(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must not be negative, got {text}")
    return value


def _share_cores(text: str) -> float:
    value = _finite_number(text)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"must be above 0, got {text}")
    return value


def _output_path(text: str) -> Path:
    path = Path(text)
    # Checked before anything runs: a path that cannot take the report would
    # otherwise surface when the report is written, after the whole measurement.
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError(f"the directory {path.parent} does not exist")
    if path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is a directory")
    return path


def _add_runner_flags(command: argparse.ArgumentParser) -> None:
    """Add the flags that name a graph and the threads of its session."""
    command.add_argument("--model", choices=tuple(MODELS), required=True)
    command.add_argument(
        "--graph",
        choices=("published", "batched"),
        required=True,
        help="the published file, or the copy of it that takes any batch size",
    )
    command.add_argument(
        "--threads",
        type=_whole_number(0),
        default=DEFAULT_THREADS,
        help="intra-op threads; 0 leaves the count to ONNX Runtime (default: %(default)s)",
    )


def _add_batch_flags(batch: argparse.ArgumentParser) -> None:
    _add_runner_flags(batch)
    batch.add_argument(
        "--spinning",
        choices=("on", "off"),
        default=DEFAULT_SPINNING,
        help="whether idle intra-op threads spin (default: %(default)s)",
    )
    batch.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="batch sizes to time, separated by commas (default: %(default)s)",
    )
    batch.add_argument(
        "--min-iterations",
        type=_whole_number(1),
        default=DEFAULT_MIN_ITERATIONS,
        help="fewest timed batches per batch size (default: %(default)s)",
    )
    batch.add_argument(
        "--min-seconds",
        type=_seconds,
        default=DEFAULT_MIN_SECONDS,
        help="shortest timed loop per batch size, in seconds (default: %(default)s)",
    )


def _add_saturate_flags(saturate: argparse.ArgumentParser) -> None:
    _add_runner_flags(saturate)
    saturate.add_argument(
        "--batch-size",
        type=_batch_size,
        default=DEFAULT_BATCH_SIZE,
        help="frames per batch (default: %(default)s)",
    )
    saturate.add_argument(
        "--pacing-share",
        type=_share_cores,
        metavar="CORES",
        help="hold CPU time over wall time at or under this many cores; unpaced when absent",
    )
    saturate.add_argument(
        "--nice",
        type=_whole_number(0, MAX_NICE),
        default=DEFAULT_NICE,
        help="how far to raise the niceness of the process (default: %(default)s)",
    )
    saturate.add_argument(
        "--duration-s",
        type=_seconds,
        default=DEFAULT_DURATION_S,
        help="length of the load, in seconds (default: %(default)s)",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.model_bench",
        description="Time the code path of a model worker and write one JSON report.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    batch = commands.add_parser("batch", help="time batches of each batch size")
    _add_batch_flags(batch)
    stages = commands.add_parser("stages", help="time the per-frame stages outside inference")
    stages.add_argument(
        "--iterations",
        type=_whole_number(1),
        default=DEFAULT_STAGE_ITERATIONS,
        help="timed calls of each stage (default: %(default)s)",
    )
    saturate = commands.add_parser("saturate", help="run batches back to back as a load")
    _add_saturate_flags(saturate)
    for command in (batch, stages, saturate):
        command.add_argument(
            "--warmup-iterations",
            type=_whole_number(0),
            default=DEFAULT_WARMUP_ITERATIONS,
            help="untimed iterations that run before the timed ones (default: %(default)s)",
        )
        command.add_argument(
            "--out", type=_output_path, required=True, help="path of the JSON report"
        )
    return parser


def _measure(arguments: argparse.Namespace, clock: Clock) -> tuple[dict[str, object], list[str]]:
    """Run the subcommand that the arguments name. Return its report and its summary lines."""
    directory = os.environ.get(MODEL_DIR_VARIABLE)
    # An empty value would name the working directory.
    if not directory:
        raise ConfigError(
            f"{MODEL_DIR_VARIABLE} is not set: it names the directory of the model files"
        )
    model_dir = Path(directory)
    if not model_dir.is_dir():
        raise ConfigError(f"{MODEL_DIR_VARIABLE}={directory} is not a directory")
    configure_process_threads()
    frame_set = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    environment = collect_environment(model_dir)
    measurement = _COMMANDS[arguments.command](arguments, model_dir, frame_set, clock)
    report = {
        "kind": f"model_bench.{arguments.command}",
        "args": {**vars(arguments), "out": str(arguments.out)},
        "environment": environment,
        "frames": asdict(frame_set.stats()),
        "results": measurement.results,
    }
    return report, measurement.summary_lines


def main(argv: Sequence[str], clock: Clock) -> None:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        report, summary_lines = _measure(arguments, clock)
    except (ConfigError, ModelLoadError) as error:
        parser.error(str(error))
    arguments.out.write_text(json.dumps(report, indent=2) + "\n")
    # The report comes first: a program that runs saturate as a load may stop
    # reading its output once it has seen "ready".
    print(*summary_lines, f"wrote {arguments.out}", sep="\n")


if __name__ == "__main__":
    main(sys.argv[1:], MonotonicClock())
