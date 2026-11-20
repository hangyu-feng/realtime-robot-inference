"""Measure what moving frames costs in Python, and what one dispatch to a worker costs.

Both subcommands are meant to run inside the 2-CPU container, and run on Linux
only. Each writes one JSON report with the keys kind ("transport_bench." and
the name of the subcommand), args (the parsed arguments), environment, frames
(the sizes of the frame set) and results. Every summary is one of bench.stats,
in milliseconds.

websocket runs simulated robots against a null model over WebSocket on
loopback. A receiver process stands in for the gateway: it answers every frame
at once with one command-sized message. A sender process runs the robots, each
on an open-loop schedule, and a robot skips every tick that comes due before
its previous send has ended. Each --saturate first starts the saturate load of
bench.model_bench, so that the frames share the quota with busy workers. The
window is the sender's schedule: once every robot has connected, the parent
reads the CPU counters of every child and of the cgroup and then starts the
schedule, and it reads them again when the schedule has ended. The results are:

    window_s            the length of the window, on the parent's clock
    processes           per child (receiver, sender, saturate_<i>): pid,
                        cpu_user_s, cpu_system_s, cpu_s, and cpu_cores, which
                        is cpu_s over window_s
    cpu_us_per_message  receiver and sender: CPU microseconds per frame sent,
                        each frame being one message up and one reply down
    cgroup              how far cpu_usage_us, nr_periods, nr_throttled and
                        throttled_us moved, None where the kernel has no such
                        counter
    sender              ticks, frames_sent, ticks_skipped, replies_received,
                        the fire_lag_ms and round_trip_ms summaries, socket:
                        sndbuf_bytes and notsent_lowat_bytes as the kernel set
                        them, and stalled_path (below)
    receiver            messages, the loop_lag_ms summary (None when no probe
                        fired in the window), and socket: rcvbuf_bytes and
                        sndbuf_bytes as the kernel set them
    saturate            per load: name, model, threads, pacing_share,
                        report_span and the report of its bench.model_bench
                        run. As report_span says, that run lasts from the
                        load's "ready", before the receiver and the sender
                        start, to its SIGTERM after the window, so only
                        processes.saturate_<i> covers the window alone

The sender measures stalled_path before the window, on a pair of sockets set
up as a robot's and the receiver's are. Once the frame set has passed through
them, the receiving end stops reading, and the robot's end sends frames until
a send stays unfinished. frames_taken is how many whole frames it sent, and
receiver_queued_bytes and robot_unsent_bytes are what the two sockets then
hold: how deep a queue the kernel can hide when the receiver stalls.

Fire lag and loop lag are the lateness of timers, which the two event loops
round differently. uvloop rounds a delay to the nearest millisecond and counts
it from a clock that it truncates to the millisecond, so its timers can fire
up to about 1.5 ms early and its lags can be negative. asyncio rounds its wait
for a timer up to the millisecond, so its timers never fire early. A
difference of a millisecond or so between runs on the two loops is this
rounding, not work done on the loop.

dispatch times the round trip of one gateway-to-worker dispatch over a Unix
socket pair, with the gateway side on an event loop and the worker side on
blocking calls. The results are a list with one entry per batch size:
batch_size, round_trips, wall_ms (the summary of the round trip times), and
gateway_cpu_us_per_round_trip and worker_cpu_us_per_round_trip, the CPU time
of the thread that does each side's work.

receiver, sender and dispatch-worker are the children that the two start. The
help leaves them out.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import itertools
import json
import math
import random
import signal
import socket
import struct
import sys
import tempfile
import time
from collections.abc import Callable, Coroutine, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from websockets.asyncio.client import ClientConnection, connect
from websockets.asyncio.server import ServerConnection, serve
from websockets.exceptions import ConnectionClosed, ConnectionClosedError, InvalidHandshake

from bench.environment import collect_environment
from bench.stats import summarize
from common.cgroup import CgroupStats, read_stats
from common.clock import Clock, MonotonicClock
from common.commands import MODEL_KEYS, DepthCommand, encode_command
from common.errors import ConfigError
from common.procstat import ProcessGoneError, ProcessStats, read_process_stats
from robot.frames import DEFAULT_FRAME_COUNT, DEFAULT_JPEG_QUALITY, DEFAULT_SEED, build_frame_set

LOOPS = ("uvloop", "asyncio")

DEFAULT_ROBOTS = 50
DEFAULT_FPS = 30
DEFAULT_DURATION_S = 20.0
# The SO_RCVBUF and SO_SNDBUF to request, which Linux doubles (socket(7)).
# DESIGN.md section 5.1 sizes the buffers at about two frames, so that the
# kernel holds no more of a robot's path than that, and this request is chosen
# for a stalled path of the default frame set to hold about two frames. The
# kernel, not the request, decides what the path holds, so every report
# records it as stalled_path.
DEFAULT_SOCKET_BUFFER_BYTES = 40 * 1024
DEFAULT_WORKER_NICE = 0
# The lowest priority that Linux has, and the highest niceness that the --nice
# of bench.model_bench takes.
MAX_WORKER_NICE = 19
DEFAULT_BATCH_SIZES = "1,2,4,8"
DEFAULT_ROUND_TRIPS = 2000

# The receiver refuses a longer message, so one message cannot make it hold
# more than this. The sender checks at startup that its longest frame fits.
MAX_MESSAGE_BYTES = 256 * 1024
# A robot's socket takes more of a frame only while less than this much of what
# it holds is unsent. Linux checks the mark only when a send starts a new
# segment, and a segment on loopback can be as large as half the receiver's
# window, so the unsent bytes can pass the mark by up to that much: the mark
# alone does not keep frames out of the kernel. stalled_path records what the
# robot's socket held.
NOTSENT_LOWAT_BYTES = 16 * 1024
LOOP_PROBE_INTERVAL_MS = 10
# Untimed round trips that run before the timed ones of each batch size.
WARMUP_ROUND_TRIPS = 20
# While stalled_path is measured, a send that is still unfinished after this
# long is taken to be stuck behind the receiver, which reads nothing. The wait
# leaves the kernel time to deliver what it has taken, which can make room.
STALLED_SEND_MS = 500

# Each timeout bounds a wait of a healthy run with a wide margin, so it fires
# only when something has gone wrong.
STARTUP_TIMEOUT_S = 60.0
CONNECT_TIMEOUT_S = 30.0
# How far past its planned end the schedule may run.
SCHEDULE_MARGIN_S = 30.0
DRAIN_TIMEOUT_S = 10.0
SHUTDOWN_TIMEOUT_S = 30.0
# From SIGTERM to SIGKILL, when a run that failed stops its children.
STOP_GRACE_S = 5.0
# The round trips of one batch size have this much time each, on average.
ROUND_TRIP_TIMEOUT_S = 0.1

_NS_PER_US = 1_000
_NS_PER_MS = 1_000_000
_NS_PER_S = 1_000_000_000
_US_PER_S = 1_000_000
_MS_PER_S = 1_000

_MODULE = "bench.transport_bench"
_LOOPBACK = "127.0.0.1"
# Linux's number for the option, from linux/tcp.h. Some CPython builds, such as
# the standalone builds that uv installs, do not export socket.TCP_NOTSENT_LOWAT.
_TCP_NOTSENT_LOWAT = 25
# Linux's ioctl for the bytes of a TCP socket not sent yet, from linux/sockios.h.
_SIOCOUTQNSD = 0x894B

# A frame message: the sequence number (the robot's tick), the scheduled capture
# time and the send time, both on the sender's monotonic clock, padded to a
# 32-byte frame header. The JPEG follows.
_FRAME_HEADER = struct.Struct("<QQQ8x")
# A reply: the frame's sequence number and send time, echoed, padded to a
# 64-byte command header. One command body follows.
_REPLY_HEADER = struct.Struct("<QQ48x")
# The longer of the two command bodies, so no reply is shorter than a real command.
_COMMAND_BODY = encode_command(
    DepthCommand(left=0.0, center=0.0, right=0.0, steering=0.0, speed=1.0)
)
_REPLY_BYTES = _REPLY_HEADER.size + len(_COMMAND_BODY)
# One result of a dispatch: a command header and a command body.
_RESULT = bytes(_REPLY_HEADER.size) + _COMMAND_BODY
# A dispatch message is its payload's length, then the payload. A batch payload
# is the frame count, then each frame as its length and its JPEG bytes. A
# results payload is one result per frame.
_LENGTH = struct.Struct("<I")


class TransportBenchError(Exception):
    """A child failed or timed out, or a peer broke the benchmark's protocol."""


@dataclass(frozen=True)
class SaturateSpec:
    """One saturating worker load. Threads 0 leaves the count to ONNX Runtime."""

    model: str
    threads: int
    pacing_share: float | None  # cores; None leaves the load unpaced


def parse_saturate_spec(text: str) -> SaturateSpec:
    """Parse MODEL:THREADS[:PACING_SHARE], the value of --saturate."""
    model, *numbers = text.split(":")
    if len(numbers) not in (1, 2):
        raise argparse.ArgumentTypeError(f"expected MODEL:THREADS[:PACING_SHARE], got {text!r}")
    if model not in MODEL_KEYS:
        raise argparse.ArgumentTypeError(
            f"unknown model {model!r}: expected one of {', '.join(MODEL_KEYS)}"
        )
    threads = _whole_number(0)(numbers[0])
    pacing_share = _positive_number(numbers[1]) if len(numbers) == 2 else None
    return SaturateSpec(model, threads, pacing_share)


def leaf_messages(group: BaseExceptionGroup[BaseException]) -> list[str]:
    """Return the messages of the exceptions in the group, however deep they nest in groups."""
    messages: list[str] = []
    for error in group.exceptions:
        if isinstance(error, BaseExceptionGroup):
            messages += leaf_messages(error)
        else:
            messages.append(str(error))
    return messages


@dataclass
class _Child:
    """A child process of the run, with its output pipe."""

    name: str
    process: asyncio.subprocess.Process
    output: asyncio.StreamReader
    # Set by the supervisor when the child has exited with status 0.
    exited: asyncio.Event = field(default_factory=asyncio.Event)


async def _start_child(
    group: asyncio.TaskGroup,
    children: list[_Child],
    name: str,
    command_line: Sequence[str],
    pass_fds: Sequence[int] = (),
) -> _Child:
    """Start a Python child, add it to the children, and supervise it in the group."""
    # Its standard error is the parent's, so a child's traceback shows as it happens.
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        *command_line,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        pass_fds=pass_fds,
    )
    assert process.stdout is not None, "a child's output goes to a pipe"
    child = _Child(name, process, process.stdout)
    children.append(child)
    group.create_task(_supervise(child))
    return child


async def _supervise(child: _Child) -> None:
    """Fail the run as soon as the child exits with an error, at whatever step that happens."""
    # No timeout of its own: the wait lasts as long as the run, whose every step has one.
    returncode = await child.process.wait()
    if returncode < 0:
        raise TransportBenchError(f"{child.name} was killed by signal {-returncode}")
    if returncode > 0:
        raise TransportBenchError(
            f"{child.name} exited with status {returncode}; its standard error is above"
        )
    child.exited.set()


async def _expect_line(child: _Child, word: str, timeout_s: float) -> list[str]:
    """Wait for the child's next line, which must start with the word; return its other fields."""
    try:
        async with asyncio.timeout(timeout_s):
            line = await child.output.readline()
            if not line:
                # The child has exited. One that failed is reported by its
                # supervisor, which cancels this wait.
                await child.exited.wait()
                raise TransportBenchError(f"{child.name} exited without printing {word!r}")
    except TimeoutError as error:
        raise TransportBenchError(
            f"{child.name} did not print {word!r} within {timeout_s:g} s"
        ) from error
    fields = line.decode(errors="replace").split()
    if fields[:1] != [word]:
        raise TransportBenchError(f"{child.name} printed {line!r} where {word!r} was due")
    return fields[1:]


async def _wait_for_exits(children: Sequence[_Child], timeout_s: float) -> None:
    """Wait until every child has exited with status 0. Its supervisor reports a failed one."""
    try:
        async with asyncio.timeout(timeout_s):
            for child in children:
                await child.exited.wait()
    except TimeoutError as error:
        running = ", ".join(child.name for child in children if not child.exited.is_set())
        raise TransportBenchError(f"{running} did not exit within {timeout_s:g} s") from error


async def _stop_children(children: Sequence[_Child]) -> None:
    """Terminate every child still running, and kill one that outlasts the grace period."""
    running = [child.process for child in children if child.process.returncode is None]
    for process in running:
        process.terminate()
    try:
        async with asyncio.timeout(STOP_GRACE_S):
            for process in running:
                await process.wait()
    except TimeoutError:
        for process in running:
            if process.returncode is None:
                process.kill()
        async with asyncio.timeout(STOP_GRACE_S):
            for process in running:
                await process.wait()


@dataclass(frozen=True)
class _Snapshot:
    """The counters that the parent reads at one edge of the window."""

    time_ns: int
    cgroup: CgroupStats
    processes: dict[str, ProcessStats]


def _snapshot(clock: Clock, children: Sequence[_Child]) -> _Snapshot:
    time_ns = clock.now_ns()
    cgroup = read_stats()
    processes: dict[str, ProcessStats] = {}
    for child in children:
        try:
            processes[child.name] = read_process_stats(child.process.pid)
        except ProcessGoneError as error:
            raise TransportBenchError(f"{child.name} exited before the window ended") from error
    return _Snapshot(time_ns, cgroup, processes)


@dataclass(frozen=True)
class _ProcessCpu:
    """CPU time of one child over the window."""

    pid: int
    cpu_user_s: float
    cpu_system_s: float
    cpu_s: float
    cpu_cores: float


def _process_cpu(before: ProcessStats, after: ProcessStats, window_s: float) -> _ProcessCpu:
    user_s = (after.cpu_user_ms - before.cpu_user_ms) / _MS_PER_S
    system_s = (after.cpu_system_ms - before.cpu_system_ms) / _MS_PER_S
    cpu_s = user_s + system_s
    return _ProcessCpu(after.pid, user_s, system_s, cpu_s, cpu_s / window_s)


_CGROUP_COUNTERS = ("cpu_usage_us", "nr_periods", "nr_throttled", "throttled_us")


def _cgroup_deltas(before: CgroupStats, after: CgroupStats) -> dict[str, int | None]:
    """Return how far each counter moved, or None where the kernel has no such counter."""
    first, last = asdict(before), asdict(after)
    return {
        name: None if first[name] is None or last[name] is None else last[name] - first[name]
        for name in _CGROUP_COUNTERS
    }


def _read_json(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads(path.read_text())
    return report


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _run_on_loop[T](loop: str, main: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run the coroutine that main makes on the named event loop."""
    if loop == "uvloop":
        # Imported only here: only this choice needs it, and it is not installed everywhere.
        import uvloop

        return asyncio.run(main(), loop_factory=uvloop.new_event_loop)
    return asyncio.run(main())


@dataclass(frozen=True)
class Schedule:
    """The open-loop schedule: a robot's tick k is due at start + phase + k / fps, until the end."""

    start_ns: int
    end_ns: int
    fps: int

    def due_ns(self, phase_ns: int, tick: int) -> int:
        return self.start_ns + phase_ns + tick * _NS_PER_S // self.fps

    def ticks(self, phase_ns: int) -> int:
        """Return how many ticks of a robot with this phase are due before the end."""
        return self._first_due_at(phase_ns, self.end_ns)

    def next_tick(self, phase_ns: int, tick: int, sent_ns: int) -> int:
        """Return the tick to send after this one, whose send ended at sent_ns.

        Each tick in between came due before this one had gone, and is skipped:
        latest-wins sending. Return ticks(phase_ns) when no tick is left.
        """
        return min(max(tick + 1, self._first_due_at(phase_ns, sent_ns)), self.ticks(phase_ns))

    def _first_due_at(self, phase_ns: int, time_ns: int) -> int:
        """Return the first tick due at or after the time; 0 or less if tick 0 is."""
        # Tick k is due at or after the time exactly when k * NS_PER_S >= span * fps,
        # span being the time from tick 0, so the first is span * fps / NS_PER_S,
        # rounded up.
        span_ns = time_ns - self.start_ns - phase_ns
        return -(-span_ns * self.fps // _NS_PER_S)


@dataclass
class _Robot:
    """One simulated robot of the sender: its connection, its phase and its counts."""

    connection: ClientConnection
    phase_ns: int
    frames_sent: int = 0
    ticks_skipped: int = 0
    replies_received: int = 0
    sending_done: bool = False
    # Set once the robot has sent its last frame and every reply is back.
    replies_done: asyncio.Event = field(default_factory=asyncio.Event)

    def check_done(self) -> None:
        if self.sending_done and self.replies_received == self.frames_sent:
            self.replies_done.set()


async def _send_frames(
    robot: _Robot,
    schedule: Schedule,
    jpegs: Sequence[bytes],
    clock: Clock,
    fire_lag_ns: list[int],
) -> None:
    """Send the robot's frames on its schedule, skipping each tick due before a send has ended."""
    ticks = schedule.ticks(robot.phase_ns)
    tick = 0
    while tick < ticks:
        due_ns = schedule.due_ns(robot.phase_ns, tick)
        delay_ns = due_ns - clock.now_ns()
        if delay_ns > 0:
            # On uvloop this sleep can end before the due time, as the module
            # docstring says, and the fire lag is then negative. Sleeping again
            # for the rest would spin on timers that uvloop rounds to zero.
            await asyncio.sleep(delay_ns / _NS_PER_S)
        send_ns = clock.now_ns()
        message = _FRAME_HEADER.pack(tick, due_ns, send_ns) + jpegs[tick % len(jpegs)]
        try:
            await robot.connection.send(message)
        except ConnectionClosed as error:
            raise TransportBenchError("a robot's connection closed during the schedule") from error
        fire_lag_ns.append(send_ns - due_ns)
        robot.frames_sent += 1
        following = schedule.next_tick(robot.phase_ns, tick, clock.now_ns())
        robot.ticks_skipped += following - tick - 1
        tick = following
    robot.sending_done = True
    robot.check_done()


async def _read_replies(robot: _Robot, clock: Clock, round_trip_ns: list[int]) -> None:
    """Receive the robot's replies until its connection closes; time each from its frame's send."""
    try:
        async for reply in robot.connection:
            received_ns = clock.now_ns()
            if not isinstance(reply, bytes) or len(reply) != _REPLY_BYTES:
                raise TransportBenchError(f"a reply must be {_REPLY_BYTES} bytes of binary data")
            _, send_ns = _REPLY_HEADER.unpack_from(reply)
            round_trip_ns.append(received_ns - send_ns)
            robot.replies_received += 1
            robot.check_done()
    except ConnectionClosedError as error:
        raise TransportBenchError("a robot's connection closed abnormally") from error


@contextlib.contextmanager
def _robot_socket(buffer_bytes: int) -> Iterator[socket.socket]:
    """Open a non-blocking socket with a robot's small send buffer and low unsent mark."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as robot_socket:
        robot_socket.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        robot_socket.setsockopt(socket.IPPROTO_TCP, _TCP_NOTSENT_LOWAT, NOTSENT_LOWAT_BYTES)
        # Both event loops' transports set it too. It is set here so that the
        # stalled-path measurement, which has no transport, uses it as well.
        robot_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        robot_socket.setblocking(False)
        yield robot_socket


@contextlib.contextmanager
def _listener(buffer_bytes: int) -> Iterator[socket.socket]:
    """Open a socket bound to a free loopback port, with the receiver's small buffers."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        # An accepted connection inherits the buffer sizes and TCP_NODELAY of the listener.
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, buffer_bytes)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, buffer_bytes)
        listener.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        listener.bind((_LOOPBACK, 0))
        yield listener


async def _receive_bytes(connection: socket.socket, size_bytes: int) -> None:
    """Read size_bytes from the connection and drop them."""
    loop = asyncio.get_running_loop()
    # Every read reuses it, so it bounds what one read takes.
    buffer = bytearray(MAX_MESSAGE_BYTES)
    remaining_bytes = size_bytes
    while remaining_bytes > 0:
        received = await loop.sock_recv_into(connection, buffer)
        if received == 0:
            raise TransportBenchError("the robot's end of the stalled path closed early")
        remaining_bytes -= received


async def _measure_stalled_path(buffer_bytes: int, jpegs: Sequence[bytes]) -> dict[str, int]:
    """Fill a robot's path to a receiver that reads nothing, and return what it then holds."""
    # Imported only here, as uvloop is: the module loads on any platform, so
    # that main can refuse one other than Linux with a clear message.
    import fcntl
    import termios

    loop = asyncio.get_running_loop()
    # The frames as a robot sends them, less a few bytes of WebSocket framing.
    messages = [bytes(_FRAME_HEADER.size) + jpeg for jpeg in jpegs]
    with _listener(buffer_bytes) as listener, _robot_socket(buffer_bytes) as robot_socket:
        listener.listen(1)
        listener.setblocking(False)
        await loop.sock_connect(robot_socket, listener.getsockname())
        receiver_socket, _ = await loop.sock_accept(listener)
        with receiver_socket:
            # A receiver's window grows with what it receives, so the frame set
            # passes through once, read as it arrives, before the path is filled.
            async with asyncio.TaskGroup() as group:
                group.create_task(_receive_bytes(receiver_socket, sum(map(len, messages))))
                for message in messages:
                    await loop.sock_sendall(robot_socket, message)
            frames_taken = 0
            # The kernel's buffers are bounded, so a send that stays unfinished ends this.
            for message in itertools.cycle(messages):
                try:
                    async with asyncio.timeout(STALLED_SEND_MS / _MS_PER_S):
                        await loop.sock_sendall(robot_socket, message)
                except TimeoutError:
                    break
                frames_taken += 1
            queued = fcntl.ioctl(receiver_socket.fileno(), termios.FIONREAD, bytes(4))
            unsent = fcntl.ioctl(robot_socket.fileno(), _SIOCOUTQNSD, bytes(4))
    return {
        "frames_taken": frames_taken,
        "receiver_queued_bytes": int.from_bytes(queued, sys.byteorder),
        "robot_unsent_bytes": int.from_bytes(unsent, sys.byteorder),
    }


async def _connect_robot(
    stack: contextlib.AsyncExitStack, robot_socket: socket.socket, port: int
) -> ClientConnection:
    await asyncio.get_running_loop().sock_connect(robot_socket, (_LOOPBACK, port))
    # With a write limit of 0, a send returns once the kernel has taken the
    # whole frame, so an unfinished send means a frame still in this process.
    # The reader only counts, so the receive queue (websockets' default of 16
    # messages, past which it stops reading the socket) stays empty.
    return await stack.enter_async_context(
        connect(
            f"ws://{_LOOPBACK}:{port}/",
            sock=robot_socket,
            compression=None,
            max_size=_REPLY_BYTES,
            write_limit=0,
        )
    )


async def _connect_robots(
    stack: contextlib.AsyncExitStack, arguments: argparse.Namespace
) -> tuple[list[ClientConnection], dict[str, int]]:
    """Connect every robot; return the connections, and the socket settings the kernel applied."""
    robot_sockets = [
        stack.enter_context(_robot_socket(arguments.socket_buffer_bytes))
        for _ in range(arguments.robots)
    ]
    first_socket = robot_sockets[0]
    buffers = {
        "sndbuf_bytes": first_socket.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
        "notsent_lowat_bytes": first_socket.getsockopt(socket.IPPROTO_TCP, _TCP_NOTSENT_LOWAT),
    }
    try:
        async with asyncio.timeout(CONNECT_TIMEOUT_S):
            connections = [
                await _connect_robot(stack, robot_socket, arguments.port)
                for robot_socket in robot_sockets
            ]
    except TimeoutError as error:
        raise TransportBenchError(
            f"the robots did not connect within {CONNECT_TIMEOUT_S:g} s"
        ) from error
    except (OSError, InvalidHandshake) as error:
        raise TransportBenchError(f"a robot could not connect: {error}") from error
    return connections, buffers


async def _run_schedule(
    robots: Sequence[_Robot],
    schedule: Schedule,
    jpegs: Sequence[bytes],
    clock: Clock,
    fire_lag_ns: list[int],
) -> None:
    timeout_s = (schedule.end_ns - clock.now_ns()) / _NS_PER_S + SCHEDULE_MARGIN_S
    try:
        async with asyncio.timeout(timeout_s), asyncio.TaskGroup() as senders:
            for robot in robots:
                senders.create_task(_send_frames(robot, schedule, jpegs, clock, fire_lag_ns))
    except TimeoutError as error:
        raise TransportBenchError(
            f"the schedule ran {SCHEDULE_MARGIN_S:g} s past its end"
        ) from error


async def _drain(robots: Sequence[_Robot]) -> None:
    """Wait until every reply is back, then close every connection."""
    try:
        async with asyncio.timeout(DRAIN_TIMEOUT_S):
            for robot in robots:
                await robot.replies_done.wait()
            for robot in robots:
                await robot.connection.close()
    except TimeoutError as error:
        raise TransportBenchError(
            f"the last replies and the closing handshakes took over {DRAIN_TIMEOUT_S:g} s"
        ) from error


async def _run_robots(
    robots: Sequence[_Robot], arguments: argparse.Namespace, jpegs: Sequence[bytes], clock: Clock
) -> dict[str, object]:
    """Run the schedule when the parent says so, collect every reply, and return the counts."""
    start_signal = asyncio.Event()
    # In place before "connected" is printed, as the parent signals only after reading it.
    asyncio.get_running_loop().add_signal_handler(signal.SIGUSR1, start_signal.set)
    # One sample per frame sent, so the schedule bounds both lists.
    fire_lag_ns: list[int] = []
    round_trip_ns: list[int] = []
    async with asyncio.TaskGroup() as readers:
        for robot in robots:
            readers.create_task(_read_replies(robot, clock, round_trip_ns))
        print("connected", flush=True)
        try:
            async with asyncio.timeout(STARTUP_TIMEOUT_S):
                await start_signal.wait()
        except TimeoutError as error:
            raise TransportBenchError(
                f"no signal to start the schedule within {STARTUP_TIMEOUT_S:g} s"
            ) from error
        start_ns = clock.now_ns()
        duration_ns = round(arguments.duration_s * _NS_PER_S)
        schedule = Schedule(start_ns, start_ns + duration_ns, arguments.fps)
        await _run_schedule(robots, schedule, jpegs, clock, fire_lag_ns)
        print("end", flush=True)
        await _drain(robots)
    return {
        "ticks": sum(schedule.ticks(robot.phase_ns) for robot in robots),
        "frames_sent": sum(robot.frames_sent for robot in robots),
        "ticks_skipped": sum(robot.ticks_skipped for robot in robots),
        "replies_received": sum(robot.replies_received for robot in robots),
        "fire_lag_ms": summarize([lag_ns / _NS_PER_MS for lag_ns in fire_lag_ns]),
        "round_trip_ms": summarize([trip_ns / _NS_PER_MS for trip_ns in round_trip_ns]),
    }


async def _run_sender(arguments: argparse.Namespace, clock: Clock) -> dict[str, object]:
    """Measure a stalled path, connect every robot and run them, and return the sender's report."""
    frame_set = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    longest_bytes = _FRAME_HEADER.size + frame_set.stats().max_bytes
    if longest_bytes > MAX_MESSAGE_BYTES:
        raise ConfigError(
            f"a frame message of {longest_bytes} bytes exceeds the receiver's "
            f"limit of {MAX_MESSAGE_BYTES}"
        )
    try:
        async with asyncio.timeout(STARTUP_TIMEOUT_S):
            stalled_path = await _measure_stalled_path(
                arguments.socket_buffer_bytes, frame_set.jpegs
            )
    except TimeoutError as error:
        raise TransportBenchError(
            f"measuring the stalled path took over {STARTUP_TIMEOUT_S:g} s"
        ) from error
    async with contextlib.AsyncExitStack() as stack:
        connections, buffers = await _connect_robots(stack, arguments)
        random_phases = random.Random(arguments.seed)
        period_ns = _NS_PER_S // arguments.fps
        robots = [
            _Robot(connection, random_phases.randrange(period_ns)) for connection in connections
        ]
        counts = await _run_robots(robots, arguments, frame_set.jpegs, clock)
    return {
        **counts,
        "frames": asdict(frame_set.stats()),
        "socket": buffers,
        "stalled_path": stalled_path,
    }


class _NullModel:
    """Answers each frame at once with one command-sized message: a gateway with free models."""

    def __init__(self) -> None:
        self.messages = 0

    async def answer(self, connection: ServerConnection) -> None:
        # A message that breaks the protocol raises here. websockets logs the
        # error and closes the connection, which fails the sender.
        async for message in connection:
            self.messages += 1
            if not isinstance(message, bytes) or len(message) < _FRAME_HEADER.size:
                raise TransportBenchError(
                    f"a frame message must be binary, with a {_FRAME_HEADER.size}-byte header"
                )
            sequence, _, send_ns = _FRAME_HEADER.unpack_from(message)
            await connection.send(_REPLY_HEADER.pack(sequence, send_ns) + _COMMAND_BODY)


async def _probe_loop_lag(window: asyncio.Event, clock: Clock, lag_ms: list[float]) -> None:
    """Sleep one probe interval at a time, and record how late each wake-up is in the window."""
    # One sample per interval while the window is open, and the parent closes
    # the window at the end of the schedule, so the schedule bounds the list.
    while True:
        due_ns = clock.now_ns() + LOOP_PROBE_INTERVAL_MS * _NS_PER_MS
        await asyncio.sleep(LOOP_PROBE_INTERVAL_MS / _MS_PER_S)
        if window.is_set():
            lag_ms.append((clock.now_ns() - due_ns) / _NS_PER_MS)


async def _serve_until_stopped(
    listener: socket.socket, null_model: _NullModel, stop: asyncio.Event, lifetime_s: float
) -> None:
    port = listener.getsockname()[1]
    # JPEG does not compress further, so compression is off. Per connection,
    # websockets keeps at most 16 received messages (its default); past that it
    # stops reading the socket, and the backlog waits in the kernel buffers and
    # then in the sender. A reply waits once 32 KiB of replies are unsent (its
    # default write limit), and then the connection's frames wait behind it.
    async with serve(
        null_model.answer, sock=listener, compression=None, max_size=MAX_MESSAGE_BYTES
    ):
        print(f"listening {port}", flush=True)
        try:
            async with asyncio.timeout(lifetime_s):
                await stop.wait()
        except TimeoutError as error:
            raise TransportBenchError(f"no stop request within {lifetime_s:g} s") from error


async def _run_receiver(arguments: argparse.Namespace, clock: Clock) -> dict[str, object]:
    """Answer frames and probe the loop until SIGTERM, and return the receiver's report."""
    loop = asyncio.get_running_loop()
    stop = asyncio.Event()
    window = asyncio.Event()
    loop.add_signal_handler(signal.SIGTERM, stop.set)
    # The parent opens the window when the sender's schedule starts, and closes it at its end.
    loop.add_signal_handler(signal.SIGUSR1, window.set)
    loop.add_signal_handler(signal.SIGUSR2, window.clear)
    null_model = _NullModel()
    loop_lag_ms: list[float] = []
    with _listener(arguments.socket_buffer_bytes) as listener:
        buffers = {
            "rcvbuf_bytes": listener.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF),
            "sndbuf_bytes": listener.getsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF),
        }
        async with asyncio.TaskGroup() as group:
            probe = group.create_task(_probe_loop_lag(window, clock, loop_lag_ms))
            await _serve_until_stopped(listener, null_model, stop, arguments.lifetime_s)
            probe.cancel()
    return {
        "messages": null_model.messages,
        "loop_lag_ms": summarize(loop_lag_ms) if loop_lag_ms else None,
        "socket": buffers,
    }


def _child_lifetime_s(load_count: int, duration_s: float) -> float:
    """Return the longest the parent can take, from starting a child, to stop it."""
    # A child bounds its own run by this, so one whose parent has died does not
    # run on for ever. It is every wait of the parent before its last stop, each
    # at its timeout: the "ready" of each load, the receiver's "listening" and
    # the sender's "connected", then the schedule, then two shutdowns.
    startups = load_count + 2
    schedule_s = duration_s + SCHEDULE_MARGIN_S
    return startups * STARTUP_TIMEOUT_S + schedule_s + 2 * SHUTDOWN_TIMEOUT_S


def _named_loads(specs: Sequence[SaturateSpec]) -> list[tuple[str, SaturateSpec]]:
    return [(f"saturate_{index}", spec) for index, spec in enumerate(specs)]


def _report_path(directory: Path, child_name: str) -> Path:
    return directory / f"{child_name}.json"


def _saturate_command_line(
    spec: SaturateSpec, nice: int, lifetime_s: float, out: Path
) -> list[str]:
    command_line = [
        *("-m", "bench.model_bench", "saturate"),
        *("--model", spec.model, "--graph", "published"),
        *("--threads", str(spec.threads), "--batch-size", "1"),
        *("--nice", str(nice), "--duration-s", str(lifetime_s)),
        *("--out", str(out)),
    ]
    if spec.pacing_share is not None:
        command_line += ["--pacing-share", str(spec.pacing_share)]
    return command_line


async def _measure_window(
    sender: _Child,
    receiver: _Child,
    children: Sequence[_Child],
    duration_s: float,
    clock: Clock,
) -> tuple[_Snapshot, _Snapshot]:
    """Read the counters, start the sender's schedule, and read them again when it has ended."""
    await _expect_line(sender, "connected", STARTUP_TIMEOUT_S)
    start = _snapshot(clock, children)
    receiver.process.send_signal(signal.SIGUSR1)
    sender.process.send_signal(signal.SIGUSR1)
    await _expect_line(sender, "end", duration_s + SCHEDULE_MARGIN_S)
    end = _snapshot(clock, children)
    receiver.process.send_signal(signal.SIGUSR2)
    return start, end


async def _run_websocket(
    arguments: argparse.Namespace, clock: Clock, directory: Path
) -> tuple[_Snapshot, _Snapshot]:
    """Start the loads, the receiver and the sender, and read the counters of the window."""
    lifetime_s = _child_lifetime_s(len(arguments.saturate), arguments.duration_s)
    children: list[_Child] = []
    try:
        async with asyncio.TaskGroup() as group:
            start = partial(_start_child, group, children)
            loads = [
                await start(
                    name,
                    _saturate_command_line(
                        spec, arguments.worker_nice, lifetime_s, _report_path(directory, name)
                    ),
                )
                for name, spec in _named_loads(arguments.saturate)
            ]
            for load in loads:
                await _expect_line(load, "ready", STARTUP_TIMEOUT_S)
            receiver = await start(
                "receiver",
                [
                    *("-m", _MODULE, "receiver", "--loop", arguments.loop),
                    *("--socket-buffer-bytes", str(arguments.socket_buffer_bytes)),
                    *("--lifetime-s", str(lifetime_s)),
                    *("--out", str(_report_path(directory, "receiver"))),
                ],
            )
            (port,) = await _expect_line(receiver, "listening", STARTUP_TIMEOUT_S)
            sender = await start(
                "sender",
                [
                    *("-m", _MODULE, "sender", "--loop", arguments.loop, "--port", port),
                    *("--robots", str(arguments.robots), "--fps", str(arguments.fps)),
                    *("--duration-s", str(arguments.duration_s), "--seed", str(arguments.seed)),
                    *("--socket-buffer-bytes", str(arguments.socket_buffer_bytes)),
                    *("--out", str(_report_path(directory, "sender"))),
                ],
            )
            window = await _measure_window(sender, receiver, children, arguments.duration_s, clock)
            # A load writes its report on SIGTERM, and the sender after its last reply.
            for load in loads:
                load.process.terminate()
            await _wait_for_exits([*loads, sender], SHUTDOWN_TIMEOUT_S)
            receiver.process.terminate()
            await _wait_for_exits([receiver], SHUTDOWN_TIMEOUT_S)
    finally:
        await _stop_children(children)
    return window


def _print_websocket_summary(
    window_s: float,
    processes: Mapping[str, _ProcessCpu],
    cpu_us_per_message: Mapping[str, float],
    sender: Mapping[str, Any],
    receiver: Mapping[str, Any],
) -> None:
    print(f"window {window_s:.3f} s")
    for name, cpu in processes.items():
        print(f"{name}: {cpu.cpu_s:.3f} CPU s, {cpu.cpu_cores:.3f} cores")
    print(
        f"CPU per message: sender {cpu_us_per_message['sender']:.1f} us, "
        f"receiver {cpu_us_per_message['receiver']:.1f} us"
    )
    print(
        f"frames sent {sender['frames_sent']}, ticks skipped {sender['ticks_skipped']}, "
        f"replies received {sender['replies_received']}"
    )
    print(
        f"fire lag p99 {sender['fire_lag_ms']['p99']:.3f} ms, "
        f"round trip p50 {sender['round_trip_ms']['p50']:.3f} ms "
        f"and p99 {sender['round_trip_ms']['p99']:.3f} ms"
    )
    loop_lag = receiver["loop_lag_ms"]
    print(
        "receiver loop lag: no probe fired in the window"
        if loop_lag is None
        else f"receiver loop lag p99 {loop_lag['p99']:.3f} ms"
    )
    stalled_path = sender["stalled_path"]
    print(
        f"stalled path: frames taken {stalled_path['frames_taken']}, bytes queued at the "
        f"receiver {stalled_path['receiver_queued_bytes']}, bytes unsent at the robot "
        f"{stalled_path['robot_unsent_bytes']}"
    )


def _websocket_results(
    arguments: argparse.Namespace, directory: Path, window: tuple[_Snapshot, _Snapshot]
) -> tuple[object, dict[str, object]]:
    """Print a summary of the window, and return the frames and the results of the report."""
    start, end = window
    window_s = (end.time_ns - start.time_ns) / _NS_PER_S
    sender = _read_json(_report_path(directory, "sender"))
    receiver = _read_json(_report_path(directory, "receiver"))
    frames = sender.pop("frames")
    processes = {
        name: _process_cpu(stats, end.processes[name], window_s)
        for name, stats in start.processes.items()
    }
    cpu_us_per_message = {
        name: processes[name].cpu_s * _US_PER_S / sender["frames_sent"]
        for name in ("receiver", "sender")
    }
    _print_websocket_summary(window_s, processes, cpu_us_per_message, sender, receiver)
    saturate = [
        {
            "name": name,
            **asdict(spec),
            "report_span": "ready to SIGTERM",
            "report": _read_json(_report_path(directory, name)),
        }
        for name, spec in _named_loads(arguments.saturate)
    ]
    return frames, {
        "window_s": window_s,
        "processes": {name: asdict(cpu) for name, cpu in processes.items()},
        "cpu_us_per_message": cpu_us_per_message,
        "cgroup": _cgroup_deltas(start.cgroup, end.cgroup),
        "sender": sender,
        "receiver": receiver,
        "saturate": saturate,
    }


def _websocket_command(arguments: argparse.Namespace, clock: Clock) -> None:
    if arguments.duration_s * arguments.fps < 1:
        raise ConfigError("--duration-s must be at least one frame period, so every robot sends")
    environment = collect_environment(None)
    with tempfile.TemporaryDirectory(prefix="transport_bench_") as directory:
        window = asyncio.run(_run_websocket(arguments, clock, Path(directory)))
        frames, results = _websocket_results(arguments, Path(directory), window)
    report = {
        "kind": "transport_bench.websocket",
        "args": {
            **vars(arguments),
            "saturate": [asdict(spec) for spec in arguments.saturate],
            "out": str(arguments.out),
        },
        "environment": environment,
        "frames": frames,
        "results": results,
    }
    _write_json(arguments.out, report)
    print(f"wrote {arguments.out}")


async def _round_trip(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, batch: tuple[bytes, ...]
) -> None:
    """Send one batch and read its results, as the gateway does for every dispatch."""
    parts = [_LENGTH.pack(len(batch))]
    for jpeg in batch:
        parts += (_LENGTH.pack(len(jpeg)), jpeg)
    writer.writelines([_LENGTH.pack(sum(map(len, parts))), *parts])
    await writer.drain()
    (results_bytes,) = _LENGTH.unpack(await reader.readexactly(_LENGTH.size))
    if results_bytes != len(batch) * _REPLY_BYTES:
        raise TransportBenchError(
            f"the worker sent {results_bytes} bytes of results for {len(batch)} frames"
        )
    await reader.readexactly(results_bytes)


@dataclass(frozen=True)
class _DispatchTiming:
    """The gateway side of one batch size's timed round trips."""

    batch_size: int
    round_trips: int
    wall_ms: dict[str, float]
    gateway_cpu_us_per_round_trip: float


async def _time_batch_size(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    worker: _Child,
    jpegs: Sequence[bytes],
    batch_size: int,
    round_trips: int,
    clock: Clock,
) -> _DispatchTiming:
    """Run the warm-up and then the timed round trips of one batch size."""
    batches = itertools.batched(itertools.cycle(jpegs), batch_size)
    # One sample per timed round trip, so --round-trips bounds the list.
    wall_ms: list[float] = []
    # One deadline for all the round trips of the batch size: a deadline for
    # each would add its own cost to every round trip measured.
    timeout_s = (WARMUP_ROUND_TRIPS + round_trips) * ROUND_TRIP_TIMEOUT_S
    try:
        async with asyncio.timeout(timeout_s):
            for batch in itertools.islice(batches, WARMUP_ROUND_TRIPS):
                await _round_trip(reader, writer, batch)
            # The event loop's thread does all of the gateway side's work.
            cpu_start_ns = time.thread_time_ns()
            for batch in itertools.islice(batches, round_trips):
                start_ns = clock.now_ns()
                await _round_trip(reader, writer, batch)
                wall_ms.append((clock.now_ns() - start_ns) / _NS_PER_MS)
            cpu_ns = time.thread_time_ns() - cpu_start_ns
    except TimeoutError as error:
        raise TransportBenchError(
            f"{round_trips} round trips of batch size {batch_size} took over {timeout_s:g} s"
        ) from error
    except (asyncio.IncompleteReadError, ConnectionError) as error:
        # A worker that failed is reported by its supervisor, which cancels this wait.
        await _wait_for_exits([worker], SHUTDOWN_TIMEOUT_S)
        raise TransportBenchError(
            "the worker closed its socket in the middle of a round trip"
        ) from error
    return _DispatchTiming(
        batch_size, round_trips, summarize(wall_ms), cpu_ns / round_trips / _NS_PER_US
    )


async def _run_dispatch(
    arguments: argparse.Namespace, jpegs: Sequence[bytes], clock: Clock, directory: Path
) -> list[_DispatchTiming]:
    """Start the worker side and time the round trips of every batch size."""
    children: list[_Child] = []
    gateway_end, worker_end = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        async with asyncio.TaskGroup() as group:
            with gateway_end:
                # The worker gets its own copy of its end. Once the parent's copy
                # is closed, the gateway's end reads an end of file when the worker exits.
                with worker_end:
                    worker = await _start_child(
                        group,
                        children,
                        "dispatch_worker",
                        [
                            *("-m", _MODULE, "dispatch-worker", "--fd", str(worker_end.fileno())),
                            *("--batch-sizes", ",".join(map(str, arguments.batch_sizes))),
                            *("--round-trips", str(arguments.round_trips)),
                            *("--out", str(_report_path(directory, "dispatch_worker"))),
                        ],
                        pass_fds=(worker_end.fileno(),),
                    )
                await _expect_line(worker, "ready", STARTUP_TIMEOUT_S)
                reader, writer = await asyncio.open_unix_connection(sock=gateway_end)
                try:
                    timings = [
                        await _time_batch_size(
                            reader, writer, worker, jpegs, batch_size, arguments.round_trips, clock
                        )
                        for batch_size in arguments.batch_sizes
                    ]
                finally:
                    writer.close()
            await _wait_for_exits([worker], SHUTDOWN_TIMEOUT_S)
    finally:
        await _stop_children(children)
    return timings


def _dispatch_command(arguments: argparse.Namespace, clock: Clock) -> None:
    environment = collect_environment(None)
    frame_set = build_frame_set(DEFAULT_FRAME_COUNT, DEFAULT_SEED, DEFAULT_JPEG_QUALITY)
    with tempfile.TemporaryDirectory(prefix="transport_bench_") as directory:
        timings = _run_on_loop(
            arguments.loop,
            partial(_run_dispatch, arguments, frame_set.jpegs, clock, Path(directory)),
        )
        worker_report = _read_json(_report_path(Path(directory), "dispatch_worker"))
    worker_cpu_us = worker_report["cpu_us_per_round_trip"]
    results = []
    for timing, worker_us in zip(timings, worker_cpu_us, strict=True):
        print(
            f"batch size {timing.batch_size}: round trip p50 {timing.wall_ms['p50']:.3f} ms, "
            f"p95 {timing.wall_ms['p95']:.3f} ms; CPU per round trip: "
            f"gateway {timing.gateway_cpu_us_per_round_trip:.1f} us, worker {worker_us:.1f} us"
        )
        results.append({**asdict(timing), "worker_cpu_us_per_round_trip": worker_us})
    report = {
        "kind": "transport_bench.dispatch",
        "args": {**vars(arguments), "out": str(arguments.out)},
        "environment": environment,
        "frames": asdict(frame_set.stats()),
        "results": results,
    }
    _write_json(arguments.out, report)
    print(f"wrote {arguments.out}")


def _receive_exactly(connection: socket.socket, view: memoryview) -> None:
    """Fill the view from the socket, in as many reads as that takes."""
    while view:
        try:
            received = connection.recv_into(view)
        except OSError as error:
            raise TransportBenchError("reading from the gateway failed") from error
        if received == 0:
            raise TransportBenchError("the gateway closed the socket before the last round trip")
        view = view[received:]


def _check_batch(payload: memoryview, batch_size: int) -> None:
    """Check that a batch payload holds batch_size length-prefixed frames and nothing else."""
    (count,) = _LENGTH.unpack_from(payload)
    offset = _LENGTH.size
    for _ in range(count):
        if offset + _LENGTH.size > len(payload):
            raise TransportBenchError("a batch ends before its last frame")
        (jpeg_bytes,) = _LENGTH.unpack_from(payload, offset)
        offset += _LENGTH.size + jpeg_bytes
    if count != batch_size or offset != len(payload):
        raise TransportBenchError(f"a batch must hold {batch_size} frames and nothing else")


def _serve_round_trip(
    connection: socket.socket, buffer: memoryview, batch_size: int, results: bytes
) -> None:
    """Read one whole batch, check it, and send back one result per frame."""
    _receive_exactly(connection, buffer[: _LENGTH.size])
    (payload_bytes,) = _LENGTH.unpack_from(buffer)
    if not _LENGTH.size <= payload_bytes <= len(buffer):
        raise TransportBenchError(
            f"a batch payload of {payload_bytes} bytes; batch size {batch_size} "
            f"allows {_LENGTH.size} to {len(buffer)}"
        )
    payload = buffer[:payload_bytes]
    _receive_exactly(connection, payload)
    _check_batch(payload, batch_size)
    try:
        connection.sendall(results)
    except OSError as error:
        raise TransportBenchError("writing to the gateway failed") from error


def _serve_batch_size(connection: socket.socket, batch_size: int, round_trips: int) -> float:
    """Serve one batch size's round trips; return the CPU microseconds of each timed one."""
    # Room for the longest payload that a batch of this size may have.
    buffer = memoryview(bytearray(_LENGTH.size + batch_size * (_LENGTH.size + MAX_MESSAGE_BYTES)))
    results = _LENGTH.pack(batch_size * _REPLY_BYTES) + _RESULT * batch_size
    for _ in range(WARMUP_ROUND_TRIPS):
        _serve_round_trip(connection, buffer, batch_size, results)
    # This thread does all of the worker side's work. The process's CPU time
    # would also count other threads, such as a numerical library's idle pool
    # that spins for a while after it starts.
    cpu_start_ns = time.thread_time_ns()
    for _ in range(round_trips):
        _serve_round_trip(connection, buffer, batch_size, results)
    return (time.thread_time_ns() - cpu_start_ns) / round_trips / _NS_PER_US


def _dispatch_worker_command(arguments: argparse.Namespace, clock: Clock) -> None:
    # Blocking calls with no timeout, as a worker waits for its next batch. The
    # gateway's end closes when the parent exits, which ends any wait here.
    with socket.socket(fileno=arguments.fd) as connection:
        # Said once the imports are done, so no round trip waits for them.
        print("ready", flush=True)
        cpu_us = [
            _serve_batch_size(connection, batch_size, arguments.round_trips)
            for batch_size in arguments.batch_sizes
        ]
    _write_json(arguments.out, {"cpu_us_per_round_trip": cpu_us})


def _receiver_command(arguments: argparse.Namespace, clock: Clock) -> None:
    _write_json(
        arguments.out, _run_on_loop(arguments.loop, partial(_run_receiver, arguments, clock))
    )


def _sender_command(arguments: argparse.Namespace, clock: Clock) -> None:
    _write_json(arguments.out, _run_on_loop(arguments.loop, partial(_run_sender, arguments, clock)))


_COMMANDS: Mapping[str, Callable[[argparse.Namespace, Clock], None]] = {
    "websocket": _websocket_command,
    "dispatch": _dispatch_command,
    "receiver": _receiver_command,
    "sender": _sender_command,
    "dispatch-worker": _dispatch_worker_command,
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


def _positive_number(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a number") from error
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a finite number above 0, got {text}")
    return value


def _batch_sizes(text: str) -> list[int]:
    sizes = [_whole_number(1)(part) for part in text.split(",")]
    # A batch is made of distinct frames of the set, as in bench.model_bench.
    if max(sizes) > DEFAULT_FRAME_COUNT:
        raise argparse.ArgumentTypeError(
            f"a batch size is at most {DEFAULT_FRAME_COUNT}, the size of the frame set"
        )
    return sizes


def _output_path(text: str) -> Path:
    path = Path(text)
    # Checked before anything runs: a path that cannot take the report would
    # otherwise surface when the report is written, after the whole measurement.
    if not path.parent.is_dir():
        raise argparse.ArgumentTypeError(f"the directory {path.parent} does not exist")
    if path.is_dir():
        raise argparse.ArgumentTypeError(f"{path} is a directory")
    return path


def _add_loop_flag(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--loop", choices=LOOPS, required=True, help="the event loop of the processes timed"
    )


def _add_fleet_flags(command: argparse.ArgumentParser) -> None:
    """Add the flags that set the robots and their schedule."""
    command.add_argument(
        "--robots",
        type=_whole_number(1),
        default=DEFAULT_ROBOTS,
        help="number of robots (default: %(default)s)",
    )
    command.add_argument(
        "--fps",
        type=_whole_number(1),
        default=DEFAULT_FPS,
        help="frames per second per robot (default: %(default)s)",
    )
    command.add_argument(
        "--duration-s",
        type=_positive_number,
        default=DEFAULT_DURATION_S,
        help="length of the schedule, in seconds (default: %(default)s)",
    )
    command.add_argument(
        "--seed",
        type=_whole_number(0),
        required=True,
        help="seed of the robots' random phases",
    )


def _add_socket_buffer_flag(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--socket-buffer-bytes",
        type=_whole_number(1),
        default=DEFAULT_SOCKET_BUFFER_BYTES,
        help="SO_RCVBUF and SO_SNDBUF to request, which Linux doubles (default: %(default)s)",
    )


def _add_dispatch_flags(command: argparse.ArgumentParser) -> None:
    command.add_argument(
        "--batch-sizes",
        type=_batch_sizes,
        default=DEFAULT_BATCH_SIZES,
        help="batch sizes to time, separated by commas (default: %(default)s)",
    )
    command.add_argument(
        "--round-trips",
        type=_whole_number(1),
        default=DEFAULT_ROUND_TRIPS,
        help="timed round trips per batch size (default: %(default)s)",
    )


def _add_hidden_commands(commands: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the children's subcommands. Without a help text, the help leaves them out."""
    receiver = commands.add_parser("receiver")
    _add_loop_flag(receiver)
    _add_socket_buffer_flag(receiver)
    receiver.add_argument("--lifetime-s", type=_positive_number, required=True)
    sender = commands.add_parser("sender")
    _add_loop_flag(sender)
    _add_fleet_flags(sender)
    _add_socket_buffer_flag(sender)
    sender.add_argument("--port", type=_whole_number(1), required=True)
    worker = commands.add_parser("dispatch-worker")
    _add_dispatch_flags(worker)
    worker.add_argument("--fd", type=_whole_number(0), required=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bench.transport_bench",
        description="Time frame transport and worker dispatch, and write one JSON report.",
    )
    # The metavar names the public subcommands only, so the usage line leaves out the children's.
    commands = parser.add_subparsers(dest="command", required=True, metavar="{websocket,dispatch}")
    websocket = commands.add_parser(
        "websocket", help="robots streaming frames to a null model over WebSocket"
    )
    _add_fleet_flags(websocket)
    _add_loop_flag(websocket)
    websocket.add_argument(
        "--saturate",
        type=parse_saturate_spec,
        action="append",
        default=[],
        metavar="MODEL:THREADS[:PACING_SHARE]",
        help="also run a saturating worker load of this model; may be repeated",
    )
    websocket.add_argument(
        "--worker-nice",
        type=_whole_number(0, MAX_WORKER_NICE),
        default=DEFAULT_WORKER_NICE,
        help="how far the loads raise their niceness (default: %(default)s)",
    )
    _add_socket_buffer_flag(websocket)
    dispatch = commands.add_parser("dispatch", help="round trips of one gateway-to-worker dispatch")
    _add_dispatch_flags(dispatch)
    _add_loop_flag(dispatch)
    _add_hidden_commands(commands)
    for command in commands.choices.values():
        command.add_argument(
            "--out", type=_output_path, required=True, help="path of the JSON report"
        )
    return parser


def main(argv: Sequence[str], clock: Clock) -> None:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    if sys.platform != "linux":
        parser.error(
            "runs on Linux only: it needs uvloop, TCP_NOTSENT_LOWAT, /proc and cgroup files"
        )
    try:
        _COMMANDS[arguments.command](arguments, clock)
    except* ConfigError as errors:
        parser.error("; ".join(leaf_messages(errors)))
    except* TransportBenchError as failures:
        raise SystemExit("transport_bench: " + "; ".join(leaf_messages(failures))) from None


if __name__ == "__main__":
    main(sys.argv[1:], MonotonicClock())
