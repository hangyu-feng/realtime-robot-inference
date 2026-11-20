"""The per-command and per-tick sample log of one robot, and its file.

The exact percentiles of a run come from these samples. The log is two preallocated arrays
that the robot fills while it runs; the file is written once, after the event loop has ended.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from common.commands import MODEL_KEYS
from common.protocol import STAGE_NAMES, CommandHeader
from robot.control import Verdict

# A command row's verdict is an index into this tuple.
VERDICTS: tuple[Verdict, ...] = (Verdict.APPLIED, Verdict.STALE, Verdict.OUT_OF_ORDER)

# One row per received command. model is an index into MODEL_KEYS, and stage_us holds the
# server's stage durations in the order of STAGE_NAMES. send_ns is filled in when the log is
# saved, from the tick that sent the frame.
COMMAND_DTYPE = np.dtype(
    [
        ("session", "<u4"),
        ("model", "u1"),
        ("verdict", "u1"),
        ("batch_size", "<u2"),
        ("frame_number", "<u8"),
        ("capture_ns", "<i8"),
        ("send_ns", "<i8"),
        ("receive_ns", "<i8"),
        ("result_ns", "<i8"),
        ("stage_us", "<u4", (len(STAGE_NAMES),)),
    ]
)
# One row per tick that came due. send_ns is 0 for a tick that was not sent.
TICK_DTYPE = np.dtype(
    [("frame_number", "<u8"), ("capture_ns", "<i8"), ("send_ns", "<i8"), ("outcome", "u1")]
)


class TickOutcome(IntEnum):
    """What the robot did with a tick that came due."""

    SENT = 0
    # The previous frame was still unsent, or the uplink delay line was full.
    SKIPPED_UNSENT = 1
    # The loop woke after this tick's successor was due.
    MISSED_LATE = 2


class SampleLog:
    """The ticks and the commands of one robot process, in the order they happened.

    Each array is bounded by the capacity it is allocated with. A row that finds its array
    full is dropped and counted in dropped_ticks or dropped_commands; the robot's telemetry
    counters and histograms go on.
    """

    def __init__(self, tick_capacity: int, command_capacity: int) -> None:
        # Written once here, so that every page of both arrays is committed before the run
        # starts and the fleet's memory stays flat through a soak. An array that is only
        # reserved would be committed page by page as the rows arrive, which reads as a leak.
        self._ticks = np.empty(tick_capacity, dtype=TICK_DTYPE)
        self._ticks.fill(0)
        self._commands = np.empty(command_capacity, dtype=COMMAND_DTYPE)
        self._commands.fill(0)
        self._tick_count = 0
        self._command_count = 0
        self.dropped_ticks = 0
        self.dropped_commands = 0

    def tick(self, frame_number: int, capture_ns: int, send_ns: int, outcome: TickOutcome) -> None:
        if self._tick_count == len(self._ticks):
            self.dropped_ticks += 1
            return
        self._ticks[self._tick_count] = (frame_number, capture_ns, send_ns, outcome)
        self._tick_count += 1

    def command(
        self, session: int, header: CommandHeader, receive_ns: int, verdict: Verdict
    ) -> None:
        """Record a command received at receive_ns in the given session of this robot."""
        if self._command_count == len(self._commands):
            self.dropped_commands += 1
            return
        stages = header.stages
        self._commands[self._command_count] = (
            session,
            MODEL_KEYS.index(header.model),
            VERDICTS.index(verdict),
            header.batch_size,
            header.frame_number,
            header.capture_ns,
            0,
            receive_ns,
            header.result_ns,
            (
                stages.transport_up_us,
                stages.slot_wait_us,
                stages.dispatch_rt_us,
                stages.decode_us,
                stages.preprocess_us,
                stages.batch_peers_us,
                stages.inference_us,
                stages.postprocess_us,
            ),
        )
        self._command_count += 1

    def save(self, path: Path, meta: Mapping[str, object]) -> None:
        """Write the rows recorded so far, with meta as a JSON text, to one .npz file."""
        ticks = self._ticks[: self._tick_count]
        commands = self._commands[: self._command_count]
        # A command echoes the frame number of the frame it was computed from. Ticks are
        # appended in frame-number order, and a frame number is unique within one robot
        # process because one schedule runs across its sessions, so a binary search finds the
        # tick. A command whose tick was dropped from a full array keeps send_ns 0.
        if len(ticks) > 0:
            index = np.minimum(
                np.searchsorted(ticks["frame_number"], commands["frame_number"]), len(ticks) - 1
            )
            found = ticks["frame_number"][index] == commands["frame_number"]
            commands["send_ns"] = np.where(found, ticks["send_ns"][index], 0)
        np.savez(path, commands=commands, ticks=ticks, meta=np.array(json.dumps(meta)))


def load_samples(
    path: Path,
) -> tuple[npt.NDArray[np.void], npt.NDArray[np.void], dict[str, object]]:
    """Read a file that SampleLog.save wrote: the commands, the ticks and the meta.

    ValueError for a file whose rows have another layout.
    """
    with np.load(path) as archive:
        commands = archive["commands"]
        ticks = archive["ticks"]
        meta = json.loads(str(archive["meta"]))
    if commands.dtype != COMMAND_DTYPE or ticks.dtype != TICK_DTYPE:
        raise ValueError(f"{path} does not have the row layout of this version's samples")
    return commands, ticks, meta
