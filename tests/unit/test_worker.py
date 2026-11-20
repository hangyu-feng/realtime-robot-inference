"""Tests for the worker process: the served file, calibration, requests and the command line."""

import hashlib
import itertools
import json
import os
import re
import signal
import socket
from collections.abc import Buffer, Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO

import cv2
import pytest
from onnxruntime.capi import onnxruntime_pybind11_state as ort_state

from cloud.config import (
    OVERRIDES_VARIABLE,
    TOPOLOGY_VARIABLE,
    ModelConfig,
    ModelRuntime,
    ServerConfig,
)
from cloud.scheduler import plan_model
from common.clock import Clock, FakeClock
from common.commands import ClsCommand, encode_command
from common.worker_protocol import (
    LENGTH_PREFIX,
    MAX_PAYLOAD_BYTES,
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
    decode_worker_message,
    encode_batch_request,
    encode_batch_result,
    payload_length,
)
from robot.frames import DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set
from workers.models import MODELS, ModelSpec
from workers.models.fake import FakeRunner
from workers.runtime import OPENCV_THREAD_COUNT, BatchOutcome, FrameOutcome, SessionSettings
from workers.worker import (
    MODEL_DIR_VARIABLE,
    answer,
    calibrate,
    main,
    serve,
    served_file,
    warm_up,
)

# A bug that leaves serve waiting for bytes that never come fails its test after this long,
# instead of hanging the suite. No test waits for it.
LINK_TIMEOUT_S = 5.0

JPEG_START = b"\xff\xd8\xff"
JPEGS = (JPEG_START + b"first", JPEG_START + b"the second frame", JPEG_START + b"third")

BATCH_ID = 41
# The request's last byte arrived at RECEIVED_NS, and answer() starts at START_NS.
RECEIVED_NS = 5_000
START_NS = 6_000
CPU_START_NS = 900_000
# What a scripted batch takes, in wall time and in CPU time.
RUN_WALL_NS = 3_000_000
RUN_CPU_NS = 2_500_000
# The untimed first run of each size, far slower than the timed ones, so that a median that
# counted it would show.
UNTIMED_NS = (10**9, 10**9)

# Service time per frame of the fake runtime in the request-loop tests.
FRAME_NS = 2_000_000
LINK_START_NS = 1_000_000
# What one read of a slow link takes.
READ_NS = 1_000
# ONNX Runtime's severity levels: 0 verbose, 1 info, 2 warning, 3 error, 4 fatal.
ORT_FATAL_SEVERITY = 4

FIRST_COMMAND = ClsCommand(class_id=3, confidence=0.75)
THIRD_COMMAND = ClsCommand(class_id=9, confidence=0.5)
OUTCOME = BatchOutcome(
    (
        FrameOutcome(FIRST_COMMAND, decode_ns=11, preprocess_ns=12, postprocess_ns=13),
        FrameOutcome(None, decode_ns=21, preprocess_ns=0, postprocess_ns=0),
        FrameOutcome(THIRD_COMMAND, decode_ns=31, preprocess_ns=32, postprocess_ns=33),
    ),
    inference_ns=1_000_000,
)
NOT_INFERRED = FrameResult(FrameStatus.INFERENCE_ERROR, b"", 0, 0, 0)
UNDECODED = FrameResult(FrameStatus.DECODE_ERROR, b"", 0, 0, 0)

NS_PER_MS = 1_000_000
DEFAULTS = ServerConfig()
# Intra-op threads in the warm-up tests. Not the default, so that a session built with another
# count shows.
THREADS = 2
# Service time per frame of the fake runtime in the warm-up tests.
FAKE_FRAME_MS = 2.5
FAKE_FRAME_NS = 2_500_000
CLS_FILE = MODELS["cls"].published_file
PUBLISHED = MODELS["depth"].published_file
BATCHED = MODELS["depth"].batched_file
# Batch times by size, in ms. The default depth budget is the 150 ms threshold less a frame
# period and the 10 ms allowance: it admits four frames at the fast times and one at the slow.
FAST_MS = {1: 20, 2: 40, 4: 80, 8: 160}
SLOW_MS = {1: 90, 2: 180, 4: 360, 8: 720}


class TimedRunner:
    """Takes the wall and CPU time that its script gives the next batch, and records the batch."""

    def __init__(
        self, clock: FakeClock, cpu_clock: FakeClock, times_ns: Iterable[tuple[int, int]]
    ) -> None:
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._times_ns = iter(times_ns)
        self.batches: list[tuple[bytes, ...]] = []

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        wall_ns, cpu_ns = next(self._times_ns)
        self._clock.advance_ns(wall_ns)
        self._cpu_clock.advance_ns(cpu_ns)
        self.batches.append(tuple(jpegs))
        return BatchOutcome((), 0)


class ScriptedRunner:
    """Gives a fixed outcome, and moves the wall and CPU clocks by fixed amounts while it runs."""

    def __init__(self, outcome: BatchOutcome, clock: FakeClock, cpu_clock: FakeClock) -> None:
        self._outcome = outcome
        self._clock = clock
        self._cpu_clock = cpu_clock
        self.batches: list[list[bytes]] = []

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        self.batches.append(list(jpegs))
        self._clock.advance_ns(RUN_WALL_NS)
        self._cpu_clock.advance_ns(RUN_CPU_NS)
        return self._outcome


class FailingRunner:
    """Uses wall and CPU time on a batch, then raises an error from inside it."""

    def __init__(self, error: Exception, clock: FakeClock, cpu_clock: FakeClock) -> None:
        self._error = error
        self._clock = clock
        self._cpu_clock = cpu_clock

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        self._clock.advance_ns(RUN_WALL_NS)
        self._cpu_clock.advance_ns(RUN_CPU_NS)
        raise self._error


class FakeTime:
    """A monotonic clock that a wait moves forward, as time.sleep moves the real one."""

    def __init__(self, start_ns: int) -> None:
        self.clock = FakeClock(start_ns)
        self.waits_s: list[float] = []

    def wait(self, duration_s: float) -> None:
        self.waits_s.append(duration_s)
        self.clock.advance_ns(round(duration_s * 1e9))


class StandInModels:
    """Stands in for ModelRunner: loads no file, and records every load and every batch.

    A batch takes the wall time that the script gives its file and size, and half of that in
    CPU time. A size that the script does not give a file raises KeyError, as a graph refuses
    a batch size it was not built for.
    """

    def __init__(
        self, clock: FakeClock, cpu_clock: FakeClock, batch_ns: Mapping[tuple[str, int], int]
    ) -> None:
        self._clock = clock
        self._cpu_clock = cpu_clock
        self._batch_ns = batch_ns
        self.loads: list[tuple[str, SessionSettings]] = []
        self.runs: list[tuple[str, tuple[bytes, ...]]] = []

    def load(
        self, spec: ModelSpec, model_path: Path, settings: SessionSettings, clock: Clock
    ) -> StandInRunner:
        self.loads.append((model_path.name, settings))
        return StandInRunner(self, model_path.name)

    def run(self, file_name: str, jpegs: Sequence[bytes]) -> BatchOutcome:
        batch_ns = self._batch_ns[file_name, len(jpegs)]
        self._clock.advance_ns(batch_ns)
        self._cpu_clock.advance_ns(batch_ns // 2)
        self.runs.append((file_name, tuple(jpegs)))
        return BatchOutcome((), 0)


class StandInRunner:
    """One file that StandInModels loaded."""

    def __init__(self, models: StandInModels, file_name: str) -> None:
        self._models = models
        self._file_name = file_name

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        return self._models.run(self._file_name, jpegs)


@dataclass(frozen=True)
class OnnxWarmUp:
    """One warm-up of the ONNX runtime: its settings, its script, and what it must do.

    batch_ms scripts the time of one batch by file and size. calibrated lists the files in
    the order they are loaded, each with the sizes timed on it, and the last of them serves.
    t_b_ms is the table reported.
    """

    model: str
    forced_batch_cap: int | None
    batch_cap: int | None
    batch_ms: Mapping[tuple[str, int], int]
    calibrated: Sequence[tuple[str, tuple[int, ...]]]
    t_b_ms: Mapping[int, int]


def on(file_name: str, batch_ms: Mapping[int, int]) -> dict[tuple[str, int], int]:
    """Return the script of one file from its batch times by size."""
    return {(file_name, size): ms for size, ms in batch_ms.items()}


EVERY_SIZE = (1, 2, 4, 8)
FIRST_START_CAP_FOUR = OnnxWarmUp(
    "depth", None, None, on(BATCHED, FAST_MS), [(BATCHED, EVERY_SIZE)], FAST_MS
)
# The published file is slower than the batched copy here, and over the budget.
FIRST_START_CAP_ONE = OnnxWarmUp(
    "depth",
    None,
    None,
    {**on(BATCHED, SLOW_MS), (PUBLISHED, 1): 120},
    [(BATCHED, EVERY_SIZE), (PUBLISHED, (1,))],
    {**SLOW_MS, 1: 120},
)
FIRST_START_FORCED_CAP_ONE = OnnxWarmUp(
    "depth", 1, None, on(BATCHED, FAST_MS), [(BATCHED, EVERY_SIZE)], FAST_MS
)
# Both file names of cls are one file, so its cap of 1 loads nothing more.
CLS_MS = {1: 400, 2: 800, 4: 1600, 8: 3200}
CLS_FIRST_START_CAP_ONE = OnnxWarmUp(
    "cls", None, None, on(CLS_FILE, CLS_MS), [(CLS_FILE, EVERY_SIZE)], CLS_MS
)
RESTART_CAP_ONE = OnnxWarmUp("depth", None, 1, {(PUBLISHED, 1): 80}, [(PUBLISHED, (1,))], {1: 80})
# The slow times, at which a first start would choose a cap of 1: a restart does not choose.
RESTART_CAP_FOUR = OnnxWarmUp(
    "depth",
    None,
    4,
    on(BATCHED, {1: 90, 2: 180, 4: 360}),
    [(BATCHED, (1, 2, 4))],
    {1: 90, 2: 180, 4: 360},
)
RESTART_FORCED_CAP_ONE = OnnxWarmUp("depth", 1, 1, {(BATCHED, 1): 20}, [(BATCHED, (1,))], {1: 20})
CLS_RESTART_CAP_TWO = OnnxWarmUp(
    "cls", None, 2, on(CLS_FILE, {1: 5, 2: 9}), [(CLS_FILE, (1, 2))], {1: 5, 2: 9}
)


class SlowLink(socket.socket):
    """The worker's end of a link, on which every read takes READ_NS of a fake clock."""

    def __init__(self, fileno: int, clock: FakeClock) -> None:
        super().__init__(fileno=fileno)
        self._clock = clock

    def recv_into(self, buffer: Buffer, nbytes: int = 0, flags: int = 0, /) -> int:
        read_bytes = super().recv_into(buffer, nbytes, flags)
        self._clock.advance_ns(READ_NS)
        return read_bytes


def request_payload(batch_id: int, jpegs: Sequence[bytes]) -> memoryview:
    """Return a request as the gateway encodes it, without its length prefix."""
    return memoryview(b"".join(encode_batch_request(batch_id, jpegs)))[LENGTH_PREFIX.size :]


def result_of(message: bytes) -> BatchResult:
    """Decode a whole cls result message as the gateway does, checking its length prefix."""
    assert payload_length(message[: LENGTH_PREFIX.size]) == len(message) - LENGTH_PREFIX.size
    result = decode_worker_message(memoryview(message)[LENGTH_PREFIX.size :], "cls")
    assert isinstance(result, BatchResult)
    return result


def served(jpeg: bytes) -> FrameResult:
    """Return the result of a frame that the fake cls runtime served."""
    command = ClsCommand(class_id=len(jpeg), confidence=1.0)
    return FrameResult(FrameStatus.OK, encode_command(command), 0, 0, 0)


def read_to_end(end: socket.socket) -> bytes:
    """Read until the other end has stopped writing."""
    received = bytearray()
    while chunk := end.recv(65_536):
        received += chunk
    return bytes(received)


def messages_of(stream: bytes) -> list[WorkerReady | BatchResult]:
    """Split what a cls worker wrote into its messages, decoded as the gateway decodes them."""
    messages: list[WorkerReady | BatchResult] = []
    view = memoryview(stream)
    while view:
        end = LENGTH_PREFIX.size + payload_length(bytes(view[: LENGTH_PREFIX.size]))
        messages.append(decode_worker_message(view[LENGTH_PREFIX.size : end], "cls"))
        view = view[end:]
    return messages


def no_wait(duration_s: float) -> None:
    """Stand in for time.sleep where a test does not look at the wait."""


def with_model(model: str, model_config: ModelConfig) -> ServerConfig:
    """Return the default configuration with one model's settings replaced."""
    return replace(DEFAULTS, models={**DEFAULTS.models, model: model_config})


def set_overrides(monkeypatch: pytest.MonkeyPatch, overrides: Mapping[str, object]) -> None:
    """Give main() the environment of a gateway that runs the defaults with these overrides."""
    monkeypatch.delenv(TOPOLOGY_VARIABLE, raising=False)
    monkeypatch.setenv(OVERRIDES_VARIABLE, json.dumps(overrides))


@pytest.fixture
def link() -> Iterator[tuple[socket.socket, socket.socket]]:
    """Return the gateway's end and the worker's end of a link, each with a timeout."""
    gateway_end, worker_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    with gateway_end, worker_end:
        gateway_end.settimeout(LINK_TIMEOUT_S)
        worker_end.settimeout(LINK_TIMEOUT_S)
        yield gateway_end, worker_end


@pytest.fixture
def interrupt_handler_restored() -> Iterator[None]:
    """Give the test runner its SIGINT handler back: main() ignores SIGINT for the process."""
    handler = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, handler)


@pytest.fixture
def model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Return a model directory that the environment names, with content under every file name."""
    for file_name in (CLS_FILE, PUBLISHED, BATCHED):
        (tmp_path / file_name).write_bytes(f"the content of {file_name}".encode())
    monkeypatch.setenv(MODEL_DIR_VARIABLE, str(tmp_path))
    return tmp_path


@pytest.mark.parametrize(
    ("model", "batch_cap", "forced_batch_cap", "file_name"),
    [
        ("depth", 1, None, "midas_small.onnx"),
        ("depth", 1, 1, "midas_small_batched.onnx"),
        ("depth", 4, None, "midas_small_batched.onnx"),
        ("depth", 4, 4, "midas_small_batched.onnx"),
        ("cls", 1, None, "mobilenetv3_small.onnx"),
        ("cls", 1, 1, "mobilenetv3_small.onnx"),
        ("cls", 4, None, "mobilenetv3_small.onnx"),
        ("cls", 4, 4, "mobilenetv3_small.onnx"),
    ],
)
def test_published_file_serves_only_an_unforced_cap_of_one(
    model: str, batch_cap: int, forced_batch_cap: int | None, file_name: str
) -> None:
    assert served_file(MODELS[model], batch_cap, forced_batch_cap) == file_name


@pytest.mark.parametrize(
    ("runs", "times_ns_by_size", "calibration"),
    [
        # Each size has one timed run far slower than the others, in wall time and in CPU time,
        # so that a mean would not be the median. At size 2 they are different runs, so that
        # the two medians cannot come from one run.
        pytest.param(
            3,
            [
                [UNTIMED_NS, (10, 4), (90, 70), (11, 5)],
                [UNTIMED_NS, (200, 9), (20, 90), (21, 8)],
            ],
            ({1: 11, 2: 21}, {1: 5, 2: 9}),
            id="odd-runs",
        ),
        pytest.param(
            4,
            [
                [UNTIMED_NS, (10, 1), (40, 7), (20, 3), (900, 500)],
                [UNTIMED_NS, (600, 2), (6, 4), (8, 800), (10, 6)],
            ],
            ({1: 30, 2: 9}, {1: 5, 2: 5}),
            id="even-runs",
        ),
    ],
)
def test_calibration_is_the_median_of_the_timed_runs_of_each_size(
    runs: int,
    times_ns_by_size: list[list[tuple[int, int]]],
    calibration: tuple[dict[int, int], dict[int, int]],
) -> None:
    clock = FakeClock(7_000)
    cpu_clock = FakeClock(3_000)
    runner = TimedRunner(clock, cpu_clock, itertools.chain.from_iterable(times_ns_by_size))

    assert calibrate(runner, JPEGS, (1, 2), runs, clock, cpu_clock.now_ns) == calibration


def test_calibration_runs_every_size_on_consecutive_frames_from_the_start_of_the_set() -> None:
    clock = FakeClock()
    cpu_clock = FakeClock()
    runner = TimedRunner(clock, cpu_clock, itertools.repeat((1, 1)))
    # Size 1 takes two frames, one untimed run and one timed run, so a cycle shared by the
    # sizes would start size 2 at the third frame.
    first, second, third = JPEGS

    t_b_ns, t_b_cpu_ns = calibrate(runner, JPEGS, (1, 2), 1, clock, cpu_clock.now_ns)

    assert runner.batches == [(first,), (second,), (first, second), (third, first)]
    assert list(t_b_ns) == list(t_b_cpu_ns) == [1, 2]


def test_answer_reports_each_frame_in_request_order_with_the_stamps_and_the_cpu_time() -> None:
    clock = FakeClock(START_NS)
    cpu_clock = FakeClock(CPU_START_NS)
    runner = ScriptedRunner(OUTCOME, clock, cpu_clock)

    message = answer(request_payload(BATCH_ID, JPEGS), runner, RECEIVED_NS, clock, cpu_clock.now_ns)

    assert result_of(message) == BatchResult(
        batch_id=BATCH_ID,
        received_ns=RECEIVED_NS,
        sent_ns=START_NS + RUN_WALL_NS,
        inference_ns=OUTCOME.inference_ns,
        cpu_ns=RUN_CPU_NS,
        frames=(
            FrameResult(FrameStatus.OK, encode_command(FIRST_COMMAND), 11, 12, 13),
            FrameResult(FrameStatus.DECODE_ERROR, b"", 21, 0, 0),
            FrameResult(FrameStatus.OK, encode_command(THIRD_COMMAND), 31, 32, 33),
        ),
    )
    assert runner.batches == [list(JPEGS)]
    assert [type(jpeg) for jpeg in runner.batches[0]] == [bytes] * len(JPEGS)


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ort_state.Fail("the run failed"), id="Fail"),
        pytest.param(ort_state.InvalidArgument("unexpected input"), id="InvalidArgument"),
        pytest.param(ort_state.RuntimeException("a kernel failed"), id="RuntimeException"),
    ],
)
def test_onnx_runtime_error_fails_every_frame_of_its_batch_alone(error: Exception) -> None:
    clock = FakeClock(START_NS)
    cpu_clock = FakeClock(CPU_START_NS)
    runner = FailingRunner(error, clock, cpu_clock)

    message = answer(request_payload(BATCH_ID, JPEGS), runner, RECEIVED_NS, clock, cpu_clock.now_ns)

    assert result_of(message) == BatchResult(
        BATCH_ID,
        RECEIVED_NS,
        START_NS + RUN_WALL_NS,
        inference_ns=0,
        cpu_ns=RUN_CPU_NS,
        frames=(NOT_INFERRED,) * len(JPEGS),
    )


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(ValueError("3 frames decoded, but the graph accepts 1"), id="ValueError"),
        pytest.param(RuntimeError("a bug"), id="RuntimeError"),
        pytest.param(ort_state.EPFail("the provider failed"), id="other-onnx-runtime-error"),
    ],
)
def test_any_other_error_from_a_batch_propagates(error: Exception) -> None:
    clock = FakeClock(START_NS)
    cpu_clock = FakeClock(CPU_START_NS)
    runner = FailingRunner(error, clock, cpu_clock)

    with pytest.raises(type(error)) as raised:
        answer(request_payload(BATCH_ID, JPEGS), runner, RECEIVED_NS, clock, cpu_clock.now_ns)

    assert raised.value is error


def test_malformed_request_raises_before_any_frame_runs() -> None:
    clock = FakeClock(START_NS)
    cpu_clock = FakeClock(CPU_START_NS)
    runner = ScriptedRunner(OUTCOME, clock, cpu_clock)
    # The kind byte of a batch result, where a request belongs.
    payload = memoryview(b"\x03" + bytes(16))

    with pytest.raises(WorkerProtocolError, match="is not a batch request"):
        answer(payload, runner, RECEIVED_NS, clock, cpu_clock.now_ns)

    assert runner.batches == []


def test_serve_answers_requests_in_order_and_returns_when_the_gateway_closes_the_link(
    link: tuple[socket.socket, socket.socket],
) -> None:
    gateway_end, worker_end = link
    fake_time = FakeTime(LINK_START_NS)
    runner = FakeRunner("cls", FRAME_NS, fake_time.wait)
    first, second, third = JPEGS
    unmarked = b"\xff\xd8 not a marker"
    first_request = encode_batch_request(7, [first, unmarked, second])
    second_request = encode_batch_request(8, [third])
    gateway_end.sendall(b"".join([*first_request, *second_request]))
    gateway_end.shutdown(socket.SHUT_WR)

    serve(worker_end, runner, fake_time.clock, FakeClock().now_ns)

    worker_end.shutdown(socket.SHUT_WR)
    # Each batch waits the fake's service time of its frames that start like a JPEG.
    first_sent_ns = LINK_START_NS + 2 * FRAME_NS
    first_frames = (served(first), UNDECODED, served(second))
    first_result = BatchResult(7, LINK_START_NS, first_sent_ns, 2 * FRAME_NS, 0, first_frames)
    second_sent_ns = first_sent_ns + FRAME_NS
    second_result = BatchResult(8, first_sent_ns, second_sent_ns, FRAME_NS, 0, (served(third),))
    assert read_to_end(gateway_end) == (
        encode_batch_result(first_result) + encode_batch_result(second_result)
    )


def test_serve_stamps_a_request_when_its_last_byte_has_arrived(
    link: tuple[socket.socket, socket.socket],
) -> None:
    gateway_end, worker_end = link
    fake_time = FakeTime(LINK_START_NS)
    runner = FakeRunner("cls", FRAME_NS, fake_time.wait)
    first, _, _ = JPEGS
    # Written whole beforehand, so the request arrives in two reads: its prefix, then its payload.
    gateway_end.sendall(b"".join(encode_batch_request(7, [first])))
    gateway_end.shutdown(socket.SHUT_WR)

    with SlowLink(worker_end.detach(), fake_time.clock) as slow_link:
        serve(slow_link, runner, fake_time.clock, FakeClock().now_ns)

    received_ns = LINK_START_NS + 2 * READ_NS
    result = BatchResult(7, received_ns, received_ns + FRAME_NS, FRAME_NS, 0, (served(first),))
    assert read_to_end(gateway_end) == encode_batch_result(result)


@pytest.mark.parametrize(
    "stream",
    [
        pytest.param(LENGTH_PREFIX.pack(0), id="zero-length"),
        pytest.param(LENGTH_PREFIX.pack(MAX_PAYLOAD_BYTES + 1), id="length-over-the-bound"),
        pytest.param(LENGTH_PREFIX.pack(100)[:2], id="end-inside-the-prefix"),
        pytest.param(b"".join(encode_batch_request(1, JPEGS))[:-1], id="end-inside-the-request"),
    ],
)
def test_serve_refuses_a_stream_that_does_not_end_after_whole_messages(
    stream: bytes, link: tuple[socket.socket, socket.socket]
) -> None:
    gateway_end, worker_end = link
    fake_time = FakeTime(LINK_START_NS)
    runner = FakeRunner("cls", FRAME_NS, fake_time.wait)
    gateway_end.sendall(stream)
    gateway_end.shutdown(socket.SHUT_WR)

    with pytest.raises(WorkerProtocolError):
        serve(worker_end, runner, fake_time.clock, FakeClock().now_ns)

    assert fake_time.waits_s == []


def test_serve_returns_when_the_gateway_has_gone_before_the_result_is_written(
    link: tuple[socket.socket, socket.socket],
) -> None:
    gateway_end, worker_end = link
    fake_time = FakeTime(LINK_START_NS)
    runner = FakeRunner("cls", FRAME_NS, fake_time.wait)
    gateway_end.sendall(b"".join(encode_batch_request(1, JPEGS)))
    gateway_end.close()

    serve(worker_end, runner, fake_time.clock, FakeClock().now_ns)

    assert fake_time.waits_s == [len(JPEGS) * FRAME_NS / 1e9]


def test_serve_returns_when_the_gateway_closed_the_link_without_reading_a_result(
    link: tuple[socket.socket, socket.socket],
) -> None:
    gateway_end, worker_end = link
    fake_time = FakeTime(LINK_START_NS)
    runner = FakeRunner("cls", FRAME_NS, fake_time.wait)
    # A result that the gateway never reads: Linux then reports a reset to the worker's read.
    worker_end.sendall(encode_batch_result(BatchResult(1, 0, 0, 0, 0, (UNDECODED,))))
    gateway_end.close()

    serve(worker_end, runner, fake_time.clock, FakeClock().now_ns)

    assert fake_time.waits_s == []


@pytest.mark.parametrize(
    "warm",
    [
        pytest.param(FIRST_START_CAP_FOUR, id="depth-first-start-cap-four"),
        pytest.param(FIRST_START_CAP_ONE, id="depth-first-start-cap-one"),
        pytest.param(FIRST_START_FORCED_CAP_ONE, id="depth-first-start-forced-cap-one"),
        pytest.param(CLS_FIRST_START_CAP_ONE, id="cls-first-start-cap-one"),
        pytest.param(RESTART_CAP_ONE, id="depth-restart-cap-one"),
        pytest.param(RESTART_CAP_FOUR, id="depth-restart-cap-four"),
        pytest.param(RESTART_FORCED_CAP_ONE, id="depth-restart-forced-cap-one"),
        pytest.param(CLS_RESTART_CAP_TWO, id="cls-restart-cap-two"),
    ],
)
def test_onnx_warm_up_serves_the_file_of_the_cap_and_reports_the_times_of_the_sizes_it_ran(
    warm: OnnxWarmUp, model_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_config = replace(
        DEFAULTS.models[warm.model], threads=THREADS, forced_batch_cap=warm.forced_batch_cap
    )
    config = with_model(warm.model, model_config)
    clock = FakeClock(7_000)
    cpu_clock = FakeClock(3_000)
    batch_ns = {key: ms * NS_PER_MS for key, ms in warm.batch_ms.items()}
    models = StandInModels(clock, cpu_clock, batch_ns)
    monkeypatch.setattr("workers.worker.ModelRunner", models.load)

    runner, ready = warm_up(
        warm.model, config, warm.batch_cap, JPEGS, clock, cpu_clock.now_ns, no_wait
    )

    settings = SessionSettings(intra_op_threads=THREADS, allow_spinning=False)
    assert models.loads == [(file_name, settings) for file_name, _ in warm.calibrated]
    assert [(file_name, len(batch)) for file_name, batch in models.runs] == [
        (file_name, size)
        for file_name, sizes in warm.calibrated
        for size in sizes
        for _ in range(1 + config.calibration_runs)
    ]
    served_name, _ = warm.calibrated[-1]
    digest = hashlib.sha256((model_dir / served_name).read_bytes()).hexdigest()
    t_b_ns = {size: ms * NS_PER_MS for size, ms in warm.t_b_ms.items()}
    t_b_cpu_ns = {size: wall_ns // 2 for size, wall_ns in t_b_ns.items()}
    assert ready == WorkerReady(
        warm.model, os.getpid(), served_name, digest, THREADS, t_b_ns, t_b_cpu_ns
    )
    # The gateway refuses a table whose sizes do not ascend.
    assert list(ready.t_b_ns) == list(ready.t_b_cpu_ns) == sorted(t_b_ns)
    runner.run_batch(JPEGS[:1])
    assert models.runs[-1] == (served_name, JPEGS[:1])


@pytest.mark.parametrize(
    ("first_start", "batch_cap"),
    [
        pytest.param(FIRST_START_CAP_FOUR, 4, id="depth-cap-four"),
        pytest.param(FIRST_START_CAP_ONE, 1, id="depth-cap-one"),
        pytest.param(FIRST_START_FORCED_CAP_ONE, 1, id="depth-forced-cap-one"),
        pytest.param(CLS_FIRST_START_CAP_ONE, 1, id="cls-cap-one"),
    ],
)
@pytest.mark.usefixtures("model_dir")
def test_gateway_plans_from_the_table_of_a_first_start_the_cap_whose_file_the_worker_serves(
    first_start: OnnxWarmUp, batch_cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = first_start.model
    spec = MODELS[model]
    forced_batch_cap = first_start.forced_batch_cap
    config = with_model(model, replace(DEFAULTS.models[model], forced_batch_cap=forced_batch_cap))
    clock = FakeClock()
    cpu_clock = FakeClock()
    batch_ns = {key: ms * NS_PER_MS for key, ms in first_start.batch_ms.items()}
    models = StandInModels(clock, cpu_clock, batch_ns)
    monkeypatch.setattr("workers.worker.ModelRunner", models.load)

    _, ready = warm_up(model, config, None, JPEGS, clock, cpu_clock.now_ns, no_wait)

    assert plan_model(model, config, ready.t_b_ns).scheduler.batch_cap == batch_cap
    assert ready.model_file == served_file(spec, batch_cap, forced_batch_cap)


@pytest.mark.parametrize(
    ("batch_cap", "sizes"),
    [
        pytest.param(None, (1, 2, 4, 8), id="first-start"),
        pytest.param(1, (1,), id="restart-cap-one"),
        pytest.param(4, (1, 2, 4), id="restart-cap-four"),
    ],
)
def test_fake_warm_up_loads_no_file_and_times_its_frame_time_at_the_sizes_up_to_the_cap(
    batch_cap: int | None, sizes: tuple[int, ...]
) -> None:
    model_config = replace(
        DEFAULTS.models["cls"],
        runtime=ModelRuntime.FAKE,
        fake_frame_ms=FAKE_FRAME_MS,
        threads=THREADS,
    )
    config = with_model("cls", model_config)
    fake_time = FakeTime(LINK_START_NS)
    first, _, _ = JPEGS

    runner, ready = warm_up(
        "cls", config, batch_cap, JPEGS, fake_time.clock, FakeClock().now_ns, fake_time.wait
    )

    t_b_ns = {size: size * FAKE_FRAME_NS for size in sizes}
    no_cpu_ns = dict.fromkeys(sizes, 0)
    assert ready == WorkerReady("cls", os.getpid(), None, None, THREADS, t_b_ns, no_cpu_ns)
    assert list(ready.t_b_ns) == list(ready.t_b_cpu_ns) == list(sizes)
    assert runner.run_batch([first, first]).inference_ns == 2 * FAKE_FRAME_NS


def test_null_warm_up_loads_no_file_and_times_every_size_without_a_wait() -> None:
    # The frame time is set to show that only the fake runtime waits it.
    model_config = replace(
        DEFAULTS.models["depth"], runtime=ModelRuntime.NULL, fake_frame_ms=FAKE_FRAME_MS
    )
    config = with_model("depth", model_config)
    fake_time = FakeTime(LINK_START_NS)

    runner, ready = warm_up(
        "depth", config, None, JPEGS, fake_time.clock, FakeClock().now_ns, fake_time.wait
    )

    no_time_ns = dict.fromkeys((1, 2, 4, 8), 0)
    threads = DEFAULTS.models["depth"].threads
    assert ready == WorkerReady("depth", os.getpid(), None, None, threads, no_time_ns, no_time_ns)
    assert runner.run_batch(JPEGS).inference_ns == 0
    assert fake_time.waits_s == []


@pytest.mark.usefixtures("interrupt_handler_restored")
def test_main_reports_ready_then_serves_until_the_gateway_closes_the_link(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"models": {"cls": {"runtime": "null"}}})
    gateway_end, worker_end = link
    first, second, _ = JPEGS
    gateway_end.sendall(b"".join(encode_batch_request(7, [first, second])))
    gateway_end.shutdown(socket.SHUT_WR)

    # main() takes the descriptor over, and closes it.
    status = main(["--model", "cls", "--fd", str(worker_end.detach())])

    assert status == 0
    ready, result = messages_of(read_to_end(gateway_end))
    assert isinstance(ready, WorkerReady)
    assert (ready.pid, ready.model_file, ready.model_sha256) == (os.getpid(), None, None)
    assert ready.threads == DEFAULTS.models["cls"].threads
    assert list(ready.t_b_ns) == list(ready.t_b_cpu_ns) == [1, 2, 4, 8]
    assert isinstance(result, BatchResult)
    assert (result.batch_id, result.inference_ns) == (7, 0)
    assert result.frames == (served(first), served(second))


@pytest.mark.usefixtures("interrupt_handler_restored")
def test_main_returns_zero_when_the_gateway_has_gone_before_the_ready_message(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"models": {"cls": {"runtime": "null"}}})
    gateway_end, worker_end = link
    gateway_end.close()

    assert main(["--model", "cls", "--fd", str(worker_end.detach())]) == 0


@pytest.mark.usefixtures("interrupt_handler_restored", "model_dir")
def test_main_raises_the_niceness_before_it_loads_the_model_and_sets_the_decode_threads(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"worker_nice": 7, "models": {"cls": {"batch_sizes": [1]}}})
    models = StandInModels(FakeClock(), FakeClock(), {(CLS_FILE, 1): 0})
    monkeypatch.setattr("workers.worker.ModelRunner", models.load)
    raised: list[tuple[int, int]] = []
    monkeypatch.setattr(os, "nice", lambda increment: raised.append((increment, len(models.loads))))
    # More threads than a worker may decode with, so that the worker's own setting shows.
    cv2.setNumThreads(OPENCV_THREAD_COUNT + 1)
    gateway_end, worker_end = link
    gateway_end.shutdown(socket.SHUT_WR)

    status = main(["--model", "cls", "--fd", str(worker_end.detach())])

    assert status == 0
    # Raised once, by the configured amount, when no model was loaded yet.
    assert raised == [(7, 0)]
    assert cv2.getNumThreads() == OPENCV_THREAD_COUNT


@pytest.mark.usefixtures("interrupt_handler_restored")
def test_main_keeps_the_runtime_from_logging_the_error_of_each_failed_batch(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"models": {"cls": {"runtime": "null"}}})
    severities: list[int] = []
    monkeypatch.setattr("onnxruntime.set_default_logger_severity", severities.append)
    gateway_end, worker_end = link
    gateway_end.shutdown(socket.SHUT_WR)

    status = main(["--model", "cls", "--fd", str(worker_end.detach())])

    # ONNX Runtime's own logger writes an error line before it raises; above its error
    # severity only a fatal error is written. The worker counts a failed batch in its result.
    assert (status, severities) == (0, [ORT_FATAL_SEVERITY])


@pytest.mark.usefixtures("interrupt_handler_restored", "model_dir")
def test_main_calibrates_on_the_frames_that_the_fleet_sends_by_default(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"calibration_runs": 2, "models": {"cls": {"batch_sizes": [1]}}})
    models = StandInModels(FakeClock(), FakeClock(), {(CLS_FILE, 1): 0})
    monkeypatch.setattr("workers.worker.ModelRunner", models.load)
    gateway_end, worker_end = link
    gateway_end.shutdown(socket.SHUT_WR)

    status = main(["--model", "cls", "--fd", str(worker_end.detach())])

    assert status == 0
    frame_set = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    first, second, third = frame_set.jpegs[:3]
    # One untimed run and two timed ones, each of the next frame of the set.
    assert models.runs == [(CLS_FILE, (first,)), (CLS_FILE, (second,)), (CLS_FILE, (third,))]


@pytest.mark.usefixtures("interrupt_handler_restored")
def test_main_serves_over_a_descriptor_it_inherits_in_non_blocking_mode(
    link: tuple[socket.socket, socket.socket], monkeypatch: pytest.MonkeyPatch
) -> None:
    set_overrides(monkeypatch, {"models": {"cls": {"runtime": "null"}}})
    gateway_end, worker_end = link
    worker_end.setblocking(False)
    first, _, _ = JPEGS

    def gateway() -> bytes:
        """Read the worker's ready message, and only then send a request and end the link."""
        prefix = gateway_end.recv(LENGTH_PREFIX.size, socket.MSG_WAITALL)
        ready = gateway_end.recv(payload_length(prefix), socket.MSG_WAITALL)
        gateway_end.sendall(b"".join(encode_batch_request(3, [first])))
        gateway_end.shutdown(socket.SHUT_WR)
        return prefix + ready + read_to_end(gateway_end)

    with ThreadPoolExecutor(max_workers=1) as pool:
        written = pool.submit(gateway)
        status = main(["--model", "cls", "--fd", str(worker_end.detach())])
        ready, result = messages_of(written.result(timeout=LINK_TIMEOUT_S))

    assert status == 0
    assert isinstance(ready, WorkerReady)
    assert isinstance(result, BatchResult)
    assert (result.batch_id, result.frames) == (3, (served(first),))


@pytest.mark.usefixtures("interrupt_handler_restored")
@pytest.mark.parametrize(
    "descriptor",
    [
        pytest.param(lambda open_file: str(open_file.fileno()), id="not-a-socket"),
        pytest.param(lambda open_file: "-1", id="negative"),
    ],
)
def test_main_refuses_a_descriptor_that_is_not_a_socket_before_it_warms_up(
    descriptor: Callable[[BinaryIO], str],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    set_overrides(monkeypatch, {"models": {"cls": {"runtime": "null"}}})
    warmed_up: list[str] = []
    monkeypatch.setattr("workers.worker.warm_up", lambda model, *_: warmed_up.append(model))

    with open(os.devnull, "rb") as not_a_socket:
        status = main(["--model", "cls", "--fd", descriptor(not_a_socket)])

    assert (status, warmed_up) == (2, [])
    assert re.fullmatch(r"worker cls: argument --fd: [^\n]*\n", capsys.readouterr().err)


@pytest.mark.usefixtures("interrupt_handler_restored")
def test_main_ignores_an_interrupt_from_the_terminal() -> None:
    main(["--model", "pose"])

    assert signal.getsignal(signal.SIGINT) == signal.SIG_IGN


@pytest.mark.usefixtures("interrupt_handler_restored")
@pytest.mark.parametrize(
    "argv",
    [
        pytest.param(["--model", "pose", "--fd", "3"], id="unknown-model"),
        pytest.param(["--model", "cls"], id="no-fd"),
        pytest.param(["--model", "cls", "--fd", "three"], id="fd-not-a-number"),
        pytest.param(["--model", "cls", "--fd", "3", "--batch-cap", "1.5"], id="cap-not-whole"),
        pytest.param(["--model", "cls", "--fd", "3", "--threads", "2"], id="unknown-argument"),
    ],
)
def test_main_refuses_a_command_line_that_does_not_parse(
    argv: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(argv) == 2
    assert re.fullmatch(r"worker: [^\n]*\n", capsys.readouterr().err)


@pytest.mark.usefixtures("interrupt_handler_restored")
@pytest.mark.parametrize("batch_cap", ["0", "3", "16"])
def test_main_refuses_a_batch_cap_that_is_not_a_configured_size(
    batch_cap: str,
    link: tuple[socket.socket, socket.socket],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(TOPOLOGY_VARIABLE, raising=False)
    monkeypatch.delenv(OVERRIDES_VARIABLE, raising=False)
    _, worker_end = link

    status = main(["--model", "depth", "--fd", str(worker_end.detach()), "--batch-cap", batch_cap])

    assert status == 2
    assert re.fullmatch(r"worker depth: argument --batch-cap: [^\n]*\n", capsys.readouterr().err)


@pytest.mark.usefixtures("interrupt_handler_restored")
@pytest.mark.parametrize(
    ("variable", "value"),
    [
        pytest.param(
            OVERRIDES_VARIABLE, '{"models": {"depth": {"threads": "two"}}}', id="wrong-type"
        ),
        pytest.param(OVERRIDES_VARIABLE, "{", id="not-json"),
        pytest.param(OVERRIDES_VARIABLE, '{"topology": "split"}', id="topology-in-overrides"),
        pytest.param(TOPOLOGY_VARIABLE, "cluster", id="unknown-topology"),
    ],
)
def test_main_refuses_an_environment_that_is_not_a_valid_configuration(
    variable: str,
    value: str,
    link: tuple[socket.socket, socket.socket],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv(TOPOLOGY_VARIABLE, raising=False)
    monkeypatch.delenv(OVERRIDES_VARIABLE, raising=False)
    monkeypatch.setenv(variable, value)
    _, worker_end = link

    status = main(["--model", "depth", "--fd", str(worker_end.detach())])

    assert status == 2
    assert re.fullmatch(rf"worker depth: {variable}[^\n]*\n", capsys.readouterr().err)
