"""Tests of the worker link against real processes: spawn, messages, death, stop and reap.

Most tests start workers.worker as the gateway does, with the fake runtime, so that no model
file is needed. The others start a stub in its place, which writes the bytes it is given and
then exits or stays alive, so that a test sets exactly what the link reads and when its stream
ends.
"""

import asyncio
import contextlib
import json
import os
import signal
import sys
from collections.abc import AsyncIterator, Callable
from dataclasses import replace
from pathlib import Path

import pytest

from cloud.config import OVERRIDES_VARIABLE, TOPOLOGY_VARIABLE, ServerConfig, load_config
from cloud.worker_link import WorkerFailed, WorkerLink, WorkerStartError, start_worker
from common.clock import MonotonicClock
from common.commands import ClsCommand, decode_command, encode_command
from common.worker_protocol import (
    LENGTH_PREFIX,
    MAX_PAYLOAD_BYTES,
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerReady,
    encode_batch_result,
    encode_worker_ready,
)
from robot.frames import DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set

pytestmark = pytest.mark.skipif(
    sys.platform != "linux",
    reason="the worker link needs Linux: uvloop, an inherited Unix socket pair, signals",
)

REPO_ROOT = Path(__file__).resolve().parents[2]
# A worker answers within seconds. The bound only keeps a hung worker from hanging the suite.
LINK_TIMEOUT_S = 60.0
OVERRIDES = {
    "calibration_runs": 1,
    "models": {
        "cls": {"runtime": "fake", "fake_frame_ms": 2},
        "depth": {"runtime": "fake", "fake_frame_ms": 2},
    },
}
# The class id of a fake cls command is its frame's length in the u16 of the command body.
CLASS_ID_MASK = 0xFFFF

STUB_SAYS_VARIABLE = "STUB_WORKER_SAYS"
STUB_STAYS_VARIABLE = "STUB_WORKER_STAYS"
STUB_WORKER = f'''\
"""Stands in for workers.worker: writes to its link the parts the environment gives it.

It writes the first part at once and each later part after it has read one request. It then
exits, or stays until it is killed, without reading its link again.
"""

import os
import signal
import socket
import struct
import sys

with (
    socket.socket(fileno=int(sys.argv[sys.argv.index("--fd") + 1])) as link,
    link.makefile("rb") as requests,
):
    first, *answers = os.environ["{STUB_SAYS_VARIABLE}"].split()
    link.sendall(bytes.fromhex(first))
    for answer in answers:
        prefix = requests.read({LENGTH_PREFIX.size})
        requests.read(struct.unpack({LENGTH_PREFIX.format!r}, prefix)[0])
        link.sendall(bytes.fromhex(answer))
    while os.environ["{STUB_STAYS_VARIABLE}"]:
        signal.pause()
'''
READY = WorkerReady(
    model="cls",
    pid=4242,
    model_file=None,
    model_sha256=None,
    threads=1,
    t_b_ns={1: 5},
    t_b_cpu_ns={1: 3},
)
RESULT = BatchResult(
    batch_id=1,
    received_ns=10,
    sent_ns=20,
    inference_ns=5,
    cpu_ns=5,
    frames=(FrameResult(FrameStatus.OK, encode_command(ClsCommand(1, 1.0)), 0, 0, 0),),
)


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> ServerConfig:
    """The fake-runtime configuration, set in the environment that the workers inherit."""
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.delenv(TOPOLOGY_VARIABLE, raising=False)
    monkeypatch.setenv(OVERRIDES_VARIABLE, json.dumps(OVERRIDES))
    return load_config(os.environ)


def use_stub_worker(
    monkeypatch: pytest.MonkeyPatch, directory: Path, *says: bytes, stays: bool
) -> None:
    """Make the workers this test starts a stub that writes says to its link.

    The stub writes the first of says at once and each later one after it has read one
    request. It then exits, or with stays it ignores its link and lives until it is killed.
    """
    (directory / "workers").mkdir()
    (directory / "workers" / "__init__.py").write_text("")
    (directory / "workers" / "worker.py").write_text(STUB_WORKER)
    # python -m looks for the module in the working directory first.
    monkeypatch.chdir(directory)
    monkeypatch.setenv(STUB_SAYS_VARIABLE, " ".join(part.hex() for part in says))
    monkeypatch.setenv(STUB_STAYS_VARIABLE, "yes" if stays else "")


@contextlib.asynccontextmanager
async def started(
    config: ServerConfig, on_lost: Callable[[], None], batch_cap: int | None = None
) -> AsyncIterator[WorkerLink]:
    """Start a cls worker for the block, bound the block's waits, and reap the worker after it."""
    link = await start_worker("cls", batch_cap, on_lost, config=config, clock=MonotonicClock())
    try:
        async with asyncio.timeout(LINK_TIMEOUT_S):
            yield link
    finally:
        link.kill()
        await link.stop()


async def test_the_worker_reports_ready_with_the_configured_batch_sizes(
    config: ServerConfig,
) -> None:
    async with started(config, lambda: None) as link:
        ready = await link.ready()

    assert (ready.model, ready.pid) == ("cls", link.pid)
    assert (ready.model_file, ready.model_sha256) == (None, None)
    assert tuple(ready.t_b_ns) == tuple(ready.t_b_cpu_ns) == config.models["cls"].batch_sizes


async def test_the_worker_leads_a_process_group_of_its_own(config: ServerConfig) -> None:
    async with started(config, lambda: None) as link:
        await link.ready()
        groups = (os.getpgid(link.pid), os.getpgid(0))

    # A Ctrl-C in a terminal signals the terminal's foreground process group, which holds the
    # gateway: from the worker's start on, only the gateway decides when the worker ends.
    assert groups == (link.pid, os.getpgid(0))
    assert groups[0] != groups[1]


async def test_a_restart_with_a_batch_cap_calibrates_only_the_sizes_up_to_it(
    config: ServerConfig,
) -> None:
    async with started(config, lambda: None, batch_cap=2) as link:
        ready = await link.ready()

    assert tuple(ready.t_b_ns) == (1, 2)


async def test_a_batch_of_real_frames_returns_each_frame_its_own_command(
    config: ServerConfig,
) -> None:
    # A batch at the largest calibrated size: more bytes than one write to the socket takes.
    jpegs = build_frame_set(8, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).jpegs
    # The gateway writes each frame as a view into the message it arrived in.
    views = [memoryview(b"frame header" + jpeg)[len(b"frame header") :] for jpeg in jpegs]
    clock = MonotonicClock()
    async with started(config, lambda: None) as link:
        await link.ready()
        dispatched_ns = clock.now_ns()
        link.write_batch(7, views)
        result, result_ns = await link.result()
        after_ns = clock.now_ns()
        lost = link.lost

    assert (result.batch_id, lost) == (7, False)
    assert [frame.status for frame in result.frames] == [FrameStatus.OK] * 8
    assert [decode_command("cls", frame.command_body) for frame in result.frames] == [
        ClsCommand(class_id=len(jpeg) & CLASS_ID_MASK, confidence=1.0) for jpeg in jpegs
    ]
    # One monotonic clock in both processes, so the stamps of the two sides are in order.
    assert dispatched_ns <= result.received_ns <= result.sent_ns <= result_ns <= after_ns


async def test_a_frame_without_the_jpeg_marker_fails_alone(config: ServerConfig) -> None:
    first, last = build_frame_set(2, DEFAULT_SEED, DEFAULT_JPEG_QUALITY).jpegs
    async with started(config, lambda: None) as link:
        await link.ready()
        link.write_batch(1, [first, b"\xff\xd8 no marker follows", last])
        result, _ = await link.result()
        link.write_batch(2, [last])
        next_result, _ = await link.result()

    assert [frame.status for frame in result.frames] == [
        FrameStatus.OK,
        FrameStatus.DECODE_ERROR,
        FrameStatus.OK,
    ]
    assert [frame.status for frame in next_result.frames] == [FrameStatus.OK]


async def test_killing_the_process_fails_the_awaited_result_and_reports_the_loss(
    config: ServerConfig,
) -> None:
    lost = asyncio.Event()
    async with started(config, lost.set) as link:
        await link.ready()
        link.write_batch(1, [b"\xff\xd8\xff a frame the worker never answers"])
        link.kill()
        with pytest.raises(WorkerFailed) as failure:
            await link.result()
        reported = (lost.is_set(), link.lost)
        exit_status = await link.stop()

    assert failure.value.kind == "died"
    assert reported == (True, True)
    assert exit_status == -signal.SIGKILL


async def test_stop_reaps_a_worker_that_exits_when_its_link_closes(config: ServerConfig) -> None:
    lost = asyncio.Event()
    async with started(config, lost.set) as link:
        await link.ready()
        exit_statuses = (await link.stop(), await link.stop())
        # Completed by the link when its transport has closed.
        await lost.wait()

    assert exit_statuses == (0, 0)
    # The process has been waited for: its pid names no process any more.
    with pytest.raises(ProcessLookupError):
        os.kill(link.pid, 0)


async def test_a_worker_that_refuses_its_arguments_fails_the_warm_up_with_its_exit_status(
    config: ServerConfig,
) -> None:
    async with started(config, lambda: None, batch_cap=3) as link:
        with pytest.raises(WorkerFailed) as failure:
            await link.ready()
        exit_status = await link.stop()

    assert (failure.value.kind, exit_status) == ("died", 2)


async def test_a_process_that_cannot_be_spawned_is_a_start_error(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "no-such-python"))

    with pytest.raises(WorkerStartError, match="could not be spawned"):
        await start_worker("cls", None, lambda: None, config=config, clock=MonotonicClock())


@pytest.mark.parametrize(
    "says",
    [
        pytest.param(LENGTH_PREFIX.pack(0), id="zero-length"),
        pytest.param(LENGTH_PREFIX.pack(MAX_PAYLOAD_BYTES + 1) + b"x", id="oversized-length"),
        pytest.param(LENGTH_PREFIX.pack(1) + b"\x09", id="unknown-kind"),
        pytest.param(encode_batch_result(RESULT), id="a-result-before-the-ready"),
        pytest.param(encode_worker_ready(READY) * 2, id="two-messages-at-once"),
    ],
)
async def test_a_worker_that_breaks_the_protocol_fails_the_warm_up_as_a_protocol_failure(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, says: bytes
) -> None:
    use_stub_worker(monkeypatch, tmp_path, says, stays=False)
    lost = asyncio.Event()
    async with started(config, lost.set) as link:
        # Completed by the link once the stub has written everything and exited.
        await lost.wait()
        with pytest.raises(WorkerFailed) as failure:
            await link.ready()

    assert failure.value.kind == "protocol"


@pytest.mark.parametrize(
    "says",
    [
        pytest.param(
            encode_worker_ready(READY) + LENGTH_PREFIX.pack(0), id="a-zero-length-after-a-message"
        ),
        pytest.param(encode_worker_ready(READY) * 2, id="two-messages-at-once"),
    ],
)
async def test_a_bad_length_or_a_message_too_many_closes_the_link_while_the_worker_lives(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, says: bytes
) -> None:
    # The stub keeps its end open, so only the link's own close can make the link lost. That
    # close is how an idle dispatcher, which reads no message, learns of the break.
    use_stub_worker(monkeypatch, tmp_path, says, stays=True)
    async with started(config, lambda: None) as link:
        with pytest.raises(WorkerFailed) as failure:
            await link.ready()
        lost = link.lost
        # What an idle dispatcher reads to learn the kind, and so to kill the live worker.
        recorded = link.failure

    assert (failure.value.kind, lost) == ("protocol", True)
    assert recorded is not None
    assert recorded.kind == "protocol"


async def test_a_result_is_stamped_when_its_last_byte_arrives_not_when_it_is_taken(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_stub_worker(
        monkeypatch, tmp_path, encode_worker_ready(READY), encode_batch_result(RESULT), stays=False
    )
    lost = asyncio.Event()
    clock = MonotonicClock()
    async with started(config, lost.set) as link:
        await link.ready()
        link.write_batch(RESULT.batch_id, [b"\xff\xd8\xff a frame"])
        # Completed by the link once the stub has answered and exited. The end of the stream
        # comes after the result's last byte, so the result is in and has not been taken.
        await lost.wait()
        taken_ns = clock.now_ns()
        result, result_ns = await link.result()

    assert result == RESULT
    # A stamp taken when the dispatcher takes the result would put the dispatcher's own
    # wake-up into the dispatch round trip (DESIGN.md 4.5).
    assert result_ns < taken_ns


async def test_a_message_that_arrived_before_the_link_was_lost_is_still_returned(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_stub_worker(monkeypatch, tmp_path, encode_worker_ready(READY), stays=False)
    lost = asyncio.Event()
    async with started(config, lost.set) as link:
        # Completed by the link once the stub has written everything and exited.
        await lost.wait()
        ready = await link.ready()
        with pytest.raises(WorkerFailed) as failure:
            await link.result()
        recorded = link.failure

    assert ready == READY
    assert failure.value.kind == "died"
    assert recorded is not None
    assert recorded.kind == "died"


async def test_stop_kills_a_worker_that_does_not_exit_when_its_link_closes(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_stub_worker(monkeypatch, tmp_path, encode_worker_ready(READY), stays=True)
    impatient = replace(config, worker_stop_timeout_ms=200.0)
    async with started(impatient, lambda: None) as link:
        await link.ready()
        exit_status = await link.stop()

    assert exit_status == -signal.SIGKILL


async def test_a_stop_that_is_cancelled_kills_the_worker_and_propagates(
    config: ServerConfig, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    use_stub_worker(monkeypatch, tmp_path, encode_worker_ready(READY), stays=True)
    # A stop bound far beyond the bound of this test: only the kill of the cancelled stop()
    # can end the worker in time for the second stop() to reap it.
    patient = replace(config, worker_stop_timeout_ms=10 * LINK_TIMEOUT_S * 1_000)
    async with started(patient, lambda: None) as link:
        await link.ready()
        stopping = asyncio.create_task(link.stop())
        # Scheduled behind the task's first step, in which stop() suspends in its first wait.
        asyncio.get_running_loop().call_soon(stopping.cancel)
        with pytest.raises(asyncio.CancelledError):
            await stopping
        exit_status = await link.stop()

    assert exit_status == -signal.SIGKILL
