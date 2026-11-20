"""One model's scheduler and the accounting of its worker's results.

This covers the check of a result against the batch it answers, the stage durations of each
command, the command messages, and their pushes onto the robots' downlink queues. It is pure
logic: no asyncio, no sockets, no clock reading. Time arrives as now_ns arguments and as the
stamps of a result, and the I/O layer calls these functions in synchronous steps, then wakes
the sessions a step returns.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import accumulate, pairwise
from types import MappingProxyType

from cloud.config import ServerConfig
from cloud.downlink import QueuedCommand
from cloud.mailbox import Frame, Mailbox
from cloud.scheduler import Batch, ModelScheduler, plan_model
from cloud.sessions import Session
from common.histogram import Histogram
from common.protocol import STAGE_NAMES, CommandHeader, StageDurations, encode_command_message
from common.worker_protocol import (
    BatchResult,
    FrameResult,
    FrameStatus,
    WorkerProtocolError,
    WorkerReady,
)

_NS_PER_US = 1_000
# The kind of failure each status counts as, the keys of workers.<m>.errors.
_ERROR_KINDS: Mapping[FrameStatus, str] = MappingProxyType(
    {FrameStatus.DECODE_ERROR: "decode", FrameStatus.INFERENCE_ERROR: "inference"}
)


def check_result(batch: Batch, result: BatchResult, result_ns: int) -> None:
    """Raise WorkerProtocolError unless the result answers the batch with consistent times.

    The result must carry the batch's id and one frame per item, its stamps must be in the
    order dispatched_ns <= received_ns <= sent_ns <= result_ns, and every frame's own stages
    must fit in the worker's time for the batch. It runs before ModelScheduler.complete, so a
    refused result leaves the batch in flight for fail().
    """
    if result.batch_id != batch.batch_id:
        raise WorkerProtocolError(
            f"the result is for batch {result.batch_id}, but batch {batch.batch_id} is in flight"
        )
    if len(result.frames) != len(batch.items):
        raise WorkerProtocolError(
            f"the result has {len(result.frames)} frames for a batch of {len(batch.items)}"
        )
    # ASSUMPTION: A20. The gateway and its workers read one monotonic clock, so their stamps
    # are comparable; these checks guard that, and they keep every stage duration from going
    # negative.
    if not batch.dispatched_ns <= result.received_ns <= result.sent_ns <= result_ns:
        raise WorkerProtocolError(
            "the result's stamps are not in the order dispatched, received, sent, result"
        )
    worker_ns = result.sent_ns - result.received_ns
    if any(
        frame.decode_ns + frame.preprocess_ns + frame.postprocess_ns + result.inference_ns
        > worker_ns
        for frame in result.frames
    ):
        raise WorkerProtocolError("a frame's own stages take longer than the worker's batch")


def stage_durations(
    frame: Frame,
    dispatched_ns: int,
    result: BatchResult,
    frame_result: FrameResult,
    result_ns: int,
) -> StageDurations:
    """Return the eight server stages of one command in whole microseconds (DESIGN.md 7).

    The stages are laid end to end from the frame's send start on the gateway's timeline,
    received_ns - transport_up_ns: exact on a shared clock, an estimate otherwise.
    """
    own_ns = (
        frame_result.decode_ns
        + frame_result.preprocess_ns
        + frame_result.postprocess_ns
        + result.inference_ns
    )
    # In the order of STAGE_NAMES, which is the field order of StageDurations.
    durations_ns = (
        frame.transport_up_ns,
        dispatched_ns - frame.received_ns,
        (result.received_ns - dispatched_ns) + (result_ns - result.sent_ns),
        frame_result.decode_ns,
        frame_result.preprocess_ns,
        result.sent_ns - result.received_ns - own_ns,
        result.inference_ns,
        frame_result.postprocess_ns,
    )
    # Each stage is the difference of two cumulative offsets rounded down to the microsecond,
    # so the eight add up to exactly floor(total / 1000). The robot adds them to its own two
    # stages and compares the sum with the freshness it measured; truncating each stage on
    # its own would leave up to 8 microseconds unaccounted for in every command.
    offsets_us = [offset_ns // _NS_PER_US for offset_ns in accumulate(durations_ns, initial=0)]
    return StageDurations(*(later - earlier for earlier, later in pairwise(offsets_us)))


class ModelDispatch:
    """One model's scheduler, its stage histograms and its error counts, for the gateway's life.

    It is built from the first WorkerReady of the run and kept across worker restarts, so the
    fairness memory, the batch ids, the counters and the batch cap survive them. calibration
    is that first WorkerReady; a restarted worker's calibration is not adopted.
    """

    def __init__(
        self,
        model: str,
        config: ServerConfig,
        calibration: WorkerReady,
        mailbox: Mailbox,
        now_ns: int,
    ) -> None:
        """Plan the model from its first calibration and build its scheduler, free from now_ns.

        plan_model raises ValueError when the calibrated sizes are not the configured batch
        sizes, and the caller treats that as a failed first start.
        """
        self.model = model
        self.plan = plan_model(model, config, calibration.t_b_ns)
        self.calibration = calibration
        self.scheduler = ModelScheduler(self.plan.scheduler, mailbox, now_ns)
        self.stages: dict[str, Histogram] = {name: Histogram() for name in STAGE_NAMES}
        self.errors: dict[str, int] = dict.fromkeys(_ERROR_KINDS.values(), 0)

    def complete(
        self,
        batch: Batch,
        result: BatchResult,
        result_ns: int,
        sessions: Mapping[str, Session],
    ) -> list[Session]:
        """Book a worker's result and queue a command for every deliverable frame.

        result_ns is the gateway's clock reading when the result's last byte arrived. Returns
        the sessions that got a command, in batch order; the caller wakes them. A result that
        check_result refuses raises WorkerProtocolError before anything is booked.
        """
        check_result(batch, result, result_ns)
        failed: list[int] = []
        for index, frame_result in enumerate(result.frames):
            if frame_result.status is not FrameStatus.OK:
                failed.append(index)
                self.errors[_ERROR_KINDS[frame_result.status]] += 1
        deliverable = self.scheduler.complete(
            batch, failed, result.sent_ns - result.received_ns, result.cpu_ns, result_ns
        )
        # Every command is pushed in this same synchronous step, so the second conservation
        # identity of cloud/downlink.py holds again when this returns. The stage histograms
        # record only the commands pushed: the population of the robots' own samples.
        woken: list[Session] = []
        for index in deliverable:
            item = batch.items[index]
            frame_result = result.frames[index]
            stages = stage_durations(
                item.frame, batch.dispatched_ns, result, frame_result, result_ns
            )
            for name in STAGE_NAMES:
                self.stages[name].record(getattr(stages, f"{name}_us") / 1e3)
            header = CommandHeader(
                self.model,
                item.session,
                item.frame.frame_number,
                item.frame.capture_ns,
                len(batch.items),
                result_ns,
                stages,
            )
            # The worker's body, which decode_worker_message has checked, goes out unchanged.
            data = encode_command_message(header, frame_result.command_body)
            # complete() returns only items of a connected record whose session is the item's,
            # and cloud.sessions.SessionTable holds a session exactly while its record is
            # connected, so this lookup cannot miss: a KeyError here is a bug.
            session = sessions[item.record.robot_id]
            session.downlink.push(QueuedCommand(data, item.record.models[self.model]))
            # A batch holds at most one frame per robot, so the sessions are distinct.
            woken.append(session)
        return woken
