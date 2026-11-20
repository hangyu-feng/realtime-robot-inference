"""Where frames wait between arrival and dispatch: the only place in the system where they do.

Each open robot holds, per subscribed model, the frames that model has not finished with: under
QueuePolicy.LATEST (the design) at most one, the newest; under QueuePolicy.FIFO (the control
experiment) a bounded queue. The mailbox counts in the records it holds, so that after every
call, for every record and every model in it, the first conservation identity of DESIGN.md
10.3 holds:

    frames_received = frames_served + frames_overwritten + frames_skipped
                      + frames_dropped_disconnect + pending_count
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cloud.config import QueuePolicy
from cloud.registry import RobotRecord


@dataclass(frozen=True, slots=True)
class Frame:
    """One accepted frame, as the scheduler needs it.

    capture_ns is on the robot's clock. received_ns is the gateway's one clock reading taken
    when the message arrived; the ingress check in cloud/admission.py is given the same value.
    fire_lag_ns is send minus capture, on the robot's clock. transport_up_ns is the uplink as
    the I/O layer computes it: received_ns - send_ns on a shared clock, else half the last ping
    round trip, and 0 until the session's first round trip completes (rtt_ns is None), which
    errs toward serving in the skip rule. There is no session field: a robot's mailbox entry
    lives from open() to close(), which is one session.
    """

    frame_number: int
    capture_ns: int
    received_ns: int
    fire_lag_ns: int
    transport_up_ns: int
    jpeg: bytes | memoryview


class Mailbox:
    """The frames of every open robot, per subscribed model."""

    def __init__(self, policy: QueuePolicy, fifo_capacity_frames: int) -> None:
        self._policy = policy
        self._fifo_capacity_frames = fifo_capacity_frames
        self._records: dict[str, RobotRecord] = {}
        # Per open robot id, the frames pending for each model it subscribed to at open().
        self._pending: dict[str, dict[str, deque[Frame]]] = {}

    def open(self, record: RobotRecord) -> None:
        """Open record.robot_id for its subscribed models; ValueError if the id is open.

        The mailbox keeps the record until close() and counts in record.models.
        """
        if record.robot_id in self._records:
            raise ValueError(f"robot {record.robot_id!r} is already open")
        self._records[record.robot_id] = record
        self._pending[record.robot_id] = {model: deque() for model in record.subscribed}

    def close(self, robot_id: str) -> None:
        """Release the frames of robot_id, counting each in frames_dropped_disconnect per model.

        A frame pending for two models counts once for each. KeyError if the id is not open.
        """
        pending = self._pending.pop(robot_id)
        record = self._records.pop(robot_id)
        for model, queue in pending.items():
            record.models[model].frames_dropped_disconnect += len(queue)

    def put(self, robot_id: str, frame: Frame) -> None:
        """Offer a frame of an open robot to every model it subscribes to.

        Precondition, not checked here: between open() and close() a robot's frames are put in
        strictly increasing frame_number order. frame_header_valid (cloud/admission.py) refuses
        every other frame of one connection, and only the connection of the robot's current
        session puts (see cloud.registry.RobotRegistry), so the mailbox does not check again.
        "Has a pending frame" is then the same as DESIGN.md 5.1's rule that a model serves a
        robot only when the slot's frame number is above every frame number that model has
        finished with, and no cursor is stored.
        """
        record = self._records[robot_id]
        # ASSUMPTION: A2. Every frame is offered to each model the robot subscribed to, and each
        # model selects on its own, with its own counters.
        for model, queue in self._pending[robot_id].items():
            counters = record.models[model]
            counters.frames_received += 1
            if self._policy is QueuePolicy.LATEST:
                # Every model's slot holds the same frame object and nothing is copied, so a
                # frame that no model still needs is released as soon as it is replaced.
                if queue:
                    counters.frames_overwritten += 1
                    queue.clear()
                queue.append(frame)
            elif len(queue) < self._fifo_capacity_frames:
                queue.append(frame)
            else:
                # A full queue drops the new frame for this model, and DESIGN.md 10.3 counts it
                # as overwritten. The control shows what a plain FIFO does: the old frames
                # ahead of it are kept and served first, so freshness grows under overload.
                counters.frames_overwritten += 1

    def pending_robots(self, model: str) -> list[RobotRecord]:
        """Return the open records with at least one frame pending for the model, in no order."""
        return [
            self._records[robot_id]
            for robot_id, pending in self._pending.items()
            if pending.get(model)
        ]

    def head(self, robot_id: str, model: str) -> Frame:
        """Return the frame take() would return; ValueError when none is pending.

        Under LATEST that is the pending frame, under FIFO the oldest.
        """
        return self._queue(robot_id, model)[0]

    def take(self, robot_id: str, model: str) -> Frame:
        """Remove and return the head frame for a batch; ValueError when none is pending."""
        frame = self._queue(robot_id, model).popleft()
        counters = self._records[robot_id].models[model]
        counters.frames_served += 1
        counters.in_flight += 1
        return frame

    def skip(self, robot_id: str, model: str) -> None:
        """Remove the head frame, passed over by the skip rule; ValueError when none is pending."""
        self._queue(robot_id, model).popleft()
        self._records[robot_id].models[model].frames_skipped += 1

    def pending_count(self, robot_id: str, model: str) -> int:
        """Return the frames_pending gauge: the frames held for the robot and model.

        It is 0, never an error, for an id that is not open and for a model the open session
        does not subscribe to, because /metrics reads it, and the first identity is checked,
        for every record and every model in it.
        """
        return len(self._pending.get(robot_id, {}).get(model, ()))

    def _queue(self, robot_id: str, model: str) -> deque[Frame]:
        """Return the frames pending for the robot and model; ValueError when there are none."""
        queue = self._pending.get(robot_id, {}).get(model)
        if not queue:
            raise ValueError(f"no frame of robot {robot_id!r} is pending for {model}")
        return queue
