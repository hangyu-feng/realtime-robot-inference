"""One model's worker process: thread settings, model file, calibration and the request loop.

The gateway starts one worker per model and holds the other end of its link, a Unix stream
socket (cloud/worker_link.py):

    python -m workers.worker --model {cls,depth} --fd N [--batch-cap B]

N is the file descriptor of the worker's end of the link. --batch-cap is given on a restart,
with the cap fixed at the first start. The worker reads the gateway's configuration from the
environment it inherits, warms its model up, sends one WorkerReady, and then answers batch
requests until the gateway closes the link. It has no event loop: one thread reads a request,
runs the batch, writes the result and blocks in the next read.

The exit status is 0 once the gateway has closed the link, and 2 for a bad argument, a bad
configuration or a model that does not load, each of which prints one line
`worker <model>: <reason>` to stderr. Anything else ends the process with a traceback: a
malformed request is the gateway's bug, and any other exception is this worker's.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import signal
import socket
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Protocol

import onnxruntime as ort
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from cloud.config import ModelRuntime, ServerConfig, load_config
from cloud.scheduler import plan_model
from common.clock import Clock, MonotonicClock
from common.commands import MODEL_KEYS, encode_command
from common.errors import ConfigError
from common.worker_protocol import (
    LENGTH_PREFIX,
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
    decode_batch_request,
    encode_batch_result,
    encode_worker_ready,
    payload_length,
)
from robot.frames import DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from workers.models import MODELS, ModelSpec
from workers.models.fake import FakeRunner
from workers.runtime import (
    BatchOutcome,
    ModelLoadError,
    ModelRunner,
    SessionSettings,
    configure_process_threads,
)

MODEL_DIR_VARIABLE = "MODEL_DIR"

# What a session run raises for a batch it cannot run. These share no base class below
# Exception. The batch is the isolation boundary: its frames fail, the gateway counts them in
# workers.<m>.errors.inference, and the worker serves the next batch.
_INFERENCE_ERRORS = (ort_state.Fail, ort_state.InvalidArgument, ort_state.RuntimeException)
# ONNX Runtime's severity levels run from 0 (verbose) to 4 (fatal).
_ORT_FATAL_SEVERITY = 4


class BatchRunner(Protocol):
    """What a worker serves with: ModelRunner, or FakeRunner for the fake and null runtimes."""

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome: ...


def served_file(spec: ModelSpec, batch_cap: int, forced_batch_cap: int | None) -> str:
    """Return the file that serves batches up to batch_cap.

    DESIGN.md 4.6: the published file at cap 1, and the batched copy above it and in every arm
    of the batch-cap comparison, so that every arm runs the same graph. For a model whose
    published file takes any batch size, both names are the same file.
    """
    if batch_cap == 1 and forced_batch_cap is None:
        return spec.published_file
    return spec.batched_file


def calibrate(
    runner: BatchRunner,
    jpegs: Sequence[bytes],
    batch_sizes: Sequence[int],
    runs: int,
    clock: Clock,
    cpu_time_ns: Callable[[], int],
) -> tuple[dict[int, int], dict[int, int]]:
    """Return the median wall time and the median CPU time of one batch of each size.

    The batches of a size are the next frames of the cycled set, from its first frame, as the
    model benchmark builds them. The first batch of each size is not timed: it grows the
    runtime's memory arena to what the size needs. cpu_time_ns is the CPU time of the whole
    process, so it counts every thread of the runtime.
    """
    t_b_ns: dict[int, int] = {}
    t_b_cpu_ns: dict[int, int] = {}
    for batch_size in batch_sizes:
        batches = itertools.batched(itertools.cycle(jpegs), batch_size)
        runner.run_batch(next(batches))
        walls_ns: list[int] = []
        cpus_ns: list[int] = []
        for batch in itertools.islice(batches, runs):
            # Reading the CPU clock is a system call. Inside the pair of wall-clock readings it
            # would count as wall time of the batch.
            cpu_start_ns = cpu_time_ns()
            start_ns = clock.now_ns()
            runner.run_batch(batch)
            walls_ns.append(clock.now_ns() - start_ns)
            cpus_ns.append(cpu_time_ns() - cpu_start_ns)
        t_b_ns[batch_size] = round(statistics.median(walls_ns))
        t_b_cpu_ns[batch_size] = round(statistics.median(cpus_ns))
    return t_b_ns, t_b_cpu_ns


def answer(
    payload: memoryview,
    runner: BatchRunner,
    received_ns: int,
    clock: Clock,
    cpu_time_ns: Callable[[], int],
) -> bytes:
    """Run one batch request and return its result, length prefix included.

    received_ns is when the last byte of the request arrived. A malformed request raises
    WorkerProtocolError: the gateway sent it, and the process ends.
    """
    request = decode_batch_request(payload)
    # workers/runtime.py takes bytes (decode_jpeg calls bytes.startswith): one copy per frame,
    # here, outside the timed decode.
    jpegs = [bytes(view) for view in request.jpegs]
    cpu_start_ns = cpu_time_ns()
    try:
        outcome = runner.run_batch(jpegs)
    except _INFERENCE_ERRORS:
        # Any other exception is a bug and ends the process.
        frames = (FrameResult(FrameStatus.INFERENCE_ERROR, b"", 0, 0, 0),) * len(jpegs)
        inference_ns = 0
    else:
        frames = tuple(
            FrameResult(
                FrameStatus.OK,
                encode_command(frame.command),
                frame.decode_ns,
                frame.preprocess_ns,
                frame.postprocess_ns,
            )
            if frame.command is not None
            # A frame that does not decode reaches no later stage.
            else FrameResult(FrameStatus.DECODE_ERROR, b"", frame.decode_ns, 0, 0)
            for frame in outcome.frames
        )
        inference_ns = outcome.inference_ns
    cpu_ns = cpu_time_ns() - cpu_start_ns
    sent_ns = clock.now_ns()
    return encode_batch_result(
        BatchResult(request.batch_id, received_ns, sent_ns, inference_ns, cpu_ns, frames)
    )


def _read_into(link: socket.socket, buffer: bytearray) -> int:
    """Fill the buffer from the link, and return how many bytes arrived before the link ended."""
    view = memoryview(buffer)
    filled_bytes = 0
    while filled_bytes < len(buffer):
        try:
            read_bytes = link.recv_into(view[filled_bytes:])
        except ConnectionResetError:
            # Linux reports a reset, not the end of the stream, to a reader whose peer closed its
            # end of a Unix socket with data still unread in it. The gateway does that when it
            # stops this worker without reading its last result, as in a shutdown that cancels
            # the batch in flight: the gateway has gone, as at the end of the stream.
            break
        if read_bytes == 0:
            break
        filled_bytes += read_bytes
    return filled_bytes


def serve(
    link: socket.socket, runner: BatchRunner, clock: Clock, cpu_time_ns: Callable[[], int]
) -> None:
    """Answer the gateway's batch requests until it closes the link.

    One request at a time: the gateway keeps one batch in flight per worker and writes the next
    request only once it has the result of the one before, so no request waits here behind the
    batch that runs. A link that ends inside a message raises WorkerProtocolError.
    """
    prefix = bytearray(LENGTH_PREFIX.size)
    while True:
        prefix_bytes = _read_into(link, prefix)
        if prefix_bytes == 0:
            # The link ended between messages: that is how the gateway stops a worker.
            return
        if prefix_bytes < len(prefix):
            raise WorkerProtocolError("the link ended inside a length prefix")
        # payload_length refuses a length of 0 or above MAX_PAYLOAD_BYTES before anything is
        # allocated for the payload.
        payload = bytearray(payload_length(bytes(prefix)))
        if _read_into(link, payload) < len(payload):
            raise WorkerProtocolError(f"the link ended inside a request of {len(payload)} bytes")
        received_ns = clock.now_ns()
        result = answer(memoryview(payload), runner, received_ns, clock, cpu_time_ns)
        try:
            link.sendall(result)
        except BrokenPipeError, ConnectionResetError:
            # The gateway has gone.
            return


def _model_dir() -> Path:
    """Return the directory of the model files, which the environment names."""
    directory = os.environ.get(MODEL_DIR_VARIABLE)
    # An empty value would name the working directory.
    if not directory:
        raise ConfigError(
            f"{MODEL_DIR_VARIABLE} is not set: it names the directory of the model files"
        )
    model_dir = Path(directory)
    if not model_dir.is_dir():
        raise ConfigError(f"{MODEL_DIR_VARIABLE}={directory} is not a directory")
    return model_dir


def _warm_up_onnx(
    model: str,
    config: ServerConfig,
    batch_cap: int | None,
    jpegs: Sequence[bytes],
    clock: Clock,
    cpu_time_ns: Callable[[], int],
) -> tuple[ModelRunner, WorkerReady]:
    """Load the file that serves the model's batch cap, and calibrate it."""
    model_dir = _model_dir()
    spec = MODELS[model]
    model_config = config.models[model]
    # A spinning thread burns CPU quota while it waits for work (DESIGN.md 4.6).
    settings = SessionSettings(intra_op_threads=model_config.threads, allow_spinning=False)
    runs = config.calibration_runs
    t_b_ns: dict[int, int] = {}
    t_b_cpu_ns: dict[int, int] = {}
    loaded_file: str | None = None
    if batch_cap is None:
        # The first start times every size on the batched file, which runs them all, and
        # derives the cap from that table.
        loaded_file = spec.batched_file
        runner = ModelRunner(spec, model_dir / loaded_file, settings, clock)
        t_b_ns, t_b_cpu_ns = calibrate(
            runner, jpegs, model_config.batch_sizes, runs, clock, cpu_time_ns
        )
        batch_cap = plan_model(model, config, t_b_ns).scheduler.batch_cap
    file_name = served_file(spec, batch_cap, model_config.forced_batch_cap)
    if file_name != loaded_file:
        # A restart loads the file of the cap fixed at the first start, and a first start whose
        # cap came out at 1 loads the published file. Either times only the sizes up to the
        # cap: the cap is fixed for the run (DESIGN.md 1.3 rule 9), and the published file
        # refuses larger batches. At a first start the sizes above the cap keep their times from
        # the batched file, each still over the budget, so the gateway derives the same cap
        # from the reported table.
        runner = ModelRunner(spec, model_dir / file_name, settings, clock)
        sizes = [size for size in model_config.batch_sizes if size <= batch_cap]
        served_ns, served_cpu_ns = calibrate(runner, jpegs, sizes, runs, clock, cpu_time_ns)
        t_b_ns.update(served_ns)
        t_b_cpu_ns.update(served_cpu_ns)
    with (model_dir / file_name).open("rb") as served:
        model_sha256 = hashlib.file_digest(served, "sha256").hexdigest()
    ready = WorkerReady(
        model, os.getpid(), file_name, model_sha256, model_config.threads, t_b_ns, t_b_cpu_ns
    )
    return runner, ready


def warm_up(
    model: str,
    config: ServerConfig,
    batch_cap: int | None,
    jpegs: Sequence[bytes],
    clock: Clock,
    cpu_time_ns: Callable[[], int],
    wait: Callable[[float], None],
) -> tuple[BatchRunner, WorkerReady]:
    """Build the runner of the model's runtime, calibrate it on the frames, and report it.

    batch_cap is None at the first start, and at a restart the cap that was fixed then. The
    ONNX runtime loads its file from the directory that the environment names. wait is what
    the fake runtime waits with.
    """
    model_config = config.models[model]
    if model_config.runtime is ModelRuntime.ONNX:
        return _warm_up_onnx(model, config, batch_cap, jpegs, clock, cpu_time_ns)
    frame_ns = (
        round(model_config.fake_frame_ms * 1_000_000)
        if model_config.runtime is ModelRuntime.FAKE
        else 0
    )
    runner = FakeRunner(model, frame_ns, wait)
    # Neither runtime loads a file, so a restart differs only in the sizes it times.
    sizes = [size for size in model_config.batch_sizes if batch_cap is None or size <= batch_cap]
    t_b_ns, t_b_cpu_ns = calibrate(
        runner, jpegs, sizes, config.calibration_runs, clock, cpu_time_ns
    )
    ready = WorkerReady(model, os.getpid(), None, None, model_config.threads, t_b_ns, t_b_cpu_ns)
    return runner, ready


def _prepare(model: str, batch_cap: int | None, clock: Clock) -> tuple[BatchRunner, WorkerReady]:
    """Read the configuration, set the process up, and warm the model up."""
    config = load_config(os.environ)
    batch_sizes = config.models[model].batch_sizes
    if batch_cap is not None and batch_cap not in batch_sizes:
        raise argparse.ArgumentError(
            None,
            f"argument --batch-cap: {batch_cap} is not one of the batch sizes of {model}, "
            f"{', '.join(map(str, batch_sizes))}",
        )
    # On Linux os.nice sets the calling thread's niceness, and a thread starts with that of the
    # thread that creates it. This thread runs the batches, and the runtime's intra-op threads
    # that help it are created after this call, when the model loads, so every thread that runs
    # batches is at the worker's niceness (DESIGN.md 6.4). A thread that importing onnxruntime
    # started earlier stays at the old value and does no work.
    os.nice(config.worker_nice)
    configure_process_threads()
    # ONNX Runtime's own logger writes the error of a failed batch to stderr, which is the
    # gateway's, before it raises it. The result counts the failure, and nothing is logged per
    # batch, so the runtime writes only a fatal error.
    ort.set_default_logger_severity(_ORT_FATAL_SEVERITY)
    # ASSUMPTION: A23. Frames are synthetic, and their content sets the JPEG size, which sets
    # the decode cost. Calibration runs on the set that the fleet sends by default, so it times
    # the JPEG sizes that robots send.
    jpegs = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).jpegs
    return warm_up(model, config, batch_cap, jpegs, clock, time.process_time_ns, time.sleep)


def main(argv: Sequence[str] | None = None) -> int:
    """Run one model's worker, and return the exit status."""
    # ASSUMPTION: A17. Workers are separate processes by choice; the brief does not require it.
    # A process per model isolates the models from each other, and the compute from the
    # gateway's I/O.
    # The gateway decides a worker's life through the link, and its staged shutdown settles
    # the workers' batches before they end. It starts each worker in a session of its own,
    # out of reach of a Ctrl-C in a terminal; this also ignores a SIGINT sent to the pid.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    parser = argparse.ArgumentParser(
        prog="python -m workers.worker",
        description="Serve one model's batches to the gateway over an inherited Unix socket.",
        exit_on_error=False,
    )
    parser.add_argument("--model", choices=MODEL_KEYS, required=True)
    parser.add_argument(
        "--fd", type=int, required=True, help="file descriptor of this worker's end of the link"
    )
    parser.add_argument(
        "--batch-cap", type=int, help="the batch cap fixed at the first start, on a restart"
    )
    try:
        arguments = parser.parse_args(argv)
    except argparse.ArgumentError as error:
        # The command line did not parse, so there is no model to name.
        print(f"worker: {error}", file=sys.stderr)
        return 2
    model: str = arguments.model
    try:
        # The socket takes the descriptor over, and closes it.
        link = socket.socket(fileno=arguments.fd)
    except (OSError, ValueError) as error:
        # Checked before the warm-up, so that a bad descriptor costs no model load.
        print(f"worker {model}: argument --fd: {error}", file=sys.stderr)
        return 2
    with link:
        # serve() is blocking code, and a socket built on a descriptor keeps the descriptor's
        # mode, which the parent may have made non-blocking.
        link.setblocking(True)
        clock = MonotonicClock()
        try:
            runner, ready = _prepare(model, arguments.batch_cap, clock)
        except (argparse.ArgumentError, ConfigError, ModelLoadError) as error:
            print(f"worker {model}: {error}", file=sys.stderr)
            return 2
        try:
            link.sendall(encode_worker_ready(ready))
        except BrokenPipeError, ConnectionResetError:
            # The gateway went away while this worker warmed up.
            return 0
        serve(link, runner, clock, time.process_time_ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
