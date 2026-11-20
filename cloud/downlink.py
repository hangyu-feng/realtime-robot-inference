"""The bounded queue of commands waiting for one robot's connection.

A command is counted in its robot's ModelCounters from the moment complete() delivers its
result until it leaves the queue, so that, for every record and model, the second conservation
identity of DESIGN.md 10.3 holds:

    frames_served = commands_sent + commands_queued + commands_dropped_downlink
                    + results_dropped_disconnect + frames_failed + in_flight

It holds after every call, with one exception: ModelScheduler.complete() takes every item of a
batch out of in_flight, and an item to be delivered is counted again only when its command is
pushed here. The caller pushes a QueuedCommand for every index complete() returns in the same
synchronous step, before any other state-changing call on the registry, the mailbox, a
scheduler or a downlink queue, and the identity holds again once the last of them is pushed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from cloud.registry import ModelCounters


@dataclass(frozen=True, slots=True)
class QueuedCommand:
    """An encoded command message, header and body, and the counters of its robot and model."""

    data: bytes
    counters: ModelCounters


class DownlinkQueue:
    """The commands of both models waiting for one robot's connection, oldest first."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._commands: deque[QueuedCommand] = deque()

    def push(self, command: QueuedCommand) -> None:
        if len(self._commands) >= self._capacity:
            # A full queue drops its oldest command: a newer command of a model supersedes an
            # older one, and a robot whose queue fills is not reading, which the slow-consumer
            # timeout then ends.
            dropped = self._commands.popleft()
            dropped.counters.commands_queued -= 1
            dropped.counters.commands_dropped_downlink += 1
        command.counters.commands_queued += 1
        self._commands.append(command)

    def pop(self, now_ns: int) -> QueuedCommand | None:
        """Return the next command to write, or None when the queue is empty.

        The I/O layer pops a command when it writes it, so commands_sent counts writes, and
        the service gap of the command's model is measured here. ValueError when the model has
        no last_command_ns: connect() sets it for every subscribed model, and only commands of
        the current session are pushed.
        """
        if not self._commands:
            return None
        counters = self._commands[0].counters
        if counters.last_command_ns is None:
            raise ValueError("a queued command belongs to a model with no session start")
        counters.commands_queued -= 1
        counters.commands_sent += 1
        counters.max_service_gap_ns = max(
            counters.max_service_gap_ns, now_ns - counters.last_command_ns
        )
        counters.last_command_ns = now_ns
        return self._commands.popleft()

    def drop_all(self) -> None:
        """Drop every queued command when the robot disconnects.

        Each counts in results_dropped_disconnect: a command still queued when its robot went
        is a result that came back after the robot had gone, and commands_dropped_downlink
        stays a pure slow-consumer signal.
        """
        for command in self._commands:
            command.counters.commands_queued -= 1
            command.counters.results_dropped_disconnect += 1
        self._commands.clear()

    def __len__(self) -> int:
        return len(self._commands)
