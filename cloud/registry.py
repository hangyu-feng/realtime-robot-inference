"""The table of robots the gateway knows: session numbers, per-model counters and fairness memory.

The table is bounded and expiring, so reconnect churn cannot grow memory. A record is the whole
robots.<id> entry of /metrics (DESIGN.md 10.3), and section 10.1 bounds that entry by a
retention time after a disconnect and by a cap on the number of entries. So this is the only
per-robot table that outlives a connection: the I/O layer keeps only its open connections,
keyed by robot id, to route commands and to close a replaced session.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(slots=True)
class ModelCounters:
    """The counters of one robot and one model (DESIGN.md 10.3), cumulative per robot id.

    last_served_seq and last_served_ns are the fairness memory, None until this model first
    serves the robot; a reconnect keeps them, so a robot gains no share by reconnecting. The
    two conservation identities over these counters are stated in cloud/mailbox.py and
    cloud/downlink.py. The gauge frames_pending is not a field: it is Mailbox.pending_count.

    max_service_gap_ns holds closed gaps only: a command popped from the downlink queue closes
    the gap before it, and a disconnect closes the session's last interval, so a reconnect
    keeps every gap of the robot's earlier sessions. /metrics reports max_service_gap_ms as
    max(max_service_gap_ns, end_ns - last_command_ns) when last_command_ns is set, where end_ns
    is the scrape time while the robot is connected and disconnected_ns afterwards, so a robot
    that is starved now, or never got a command, shows it.
    """

    frames_received: int = 0
    frames_served: int = 0
    frames_overwritten: int = 0
    frames_skipped: int = 0
    frames_dropped_disconnect: int = 0
    in_flight: int = 0
    commands_queued: int = 0
    commands_sent: int = 0
    commands_dropped_downlink: int = 0
    results_dropped_disconnect: int = 0
    frames_failed: int = 0
    last_served_seq: int | None = None
    last_served_ns: int | None = None
    last_command_ns: int | None = None
    # ASSUMPTION: A12. Starvation is judged by the longest command gap per robot, which the
    # server measures here.
    max_service_gap_ns: int = 0


@dataclass(slots=True)
class RobotRecord:
    """What the gateway keeps about one robot id: its current or last session, and its counters.

    stale_ns and watchdog_ns are the thresholds declared in the hello, one entry per subscribed
    model, and frame_period_ns comes from the declared frame rate. models has an entry for every
    model this id has subscribed to in any session. The I/O layer increments bytes_in,
    bytes_out, frames_in, frames_rejected and frames_accepted, which are cumulative per robot
    id, and sets rtt_ns (the last ping round trip) and telemetry (the last validated report, as
    the DESIGN.md 10.4 object from common.protocol.telemetry_to_dict). Both of those describe
    one session, so they are None until the I/O layer sets them in each session.
    """

    robot_id: str
    session: int
    connected_ns: int
    disconnected_ns: int | None
    subscribed: tuple[str, ...]
    stale_ns: dict[str, int]
    watchdog_ns: dict[str, int]
    frame_period_ns: int
    models: dict[str, ModelCounters]
    bytes_in: int = 0
    bytes_out: int = 0
    frames_in: int = 0
    frames_rejected: int = 0
    frames_accepted: int = 0
    rtt_ns: int | None = None
    telemetry: Mapping[str, object] | None = None

    @property
    def connected(self) -> bool:
        return self.disconnected_ns is None


class RobotRegistry:
    """Every robot id the gateway knows, connected or disconnected within the retention time.

    Three callers drive the registry, the mailbox (cloud/mailbox.py), the downlink queues
    (cloud/downlink.py) and the schedulers (cloud/scheduler.py) in the same way: the gateway,
    the simulated-clock harness and the property tests. Each sequence below is one synchronous
    step, with no other call on these objects in between (in the gateway, no await in between).

    - Open a session: record = registry.connect(...); mailbox.open(record); a new
      DownlinkQueue(config.downlink_queue_commands) for the connection.
    - Close a session: mailbox.close(robot_id); that connection's DownlinkQueue.drop_all();
      registry.disconnect(robot_id, now_ns).
    - A hello for a robot id that is connected now closes the old session, then opens the new
      one.
    - Only the connection whose session is the record's current one calls Mailbox.put or runs
      the close sequence. Before either, in the same step, it checks that registry.get(robot_id)
      is connected and that its session equals the session of this connection's welcome. A
      connection that a new hello has replaced may still return a message it had already read:
      it drops that message, counts nothing for it in the record, and closes; it does not run
      the close sequence, which the replacing hello already ran. Otherwise its late frame would
      land in the new session's mailbox entry with an old frame number, and its close would end
      the new session.
    - When a worker finishes a batch: ModelScheduler.complete(...), then a QueuedCommand pushed
      on its robot's DownlinkQueue for every index complete() returns.

    A record object outlives its removal from the table wherever a batch item still holds it;
    it is then simply a disconnected record.
    """

    def __init__(self, retention_ns: int, max_entries: int) -> None:
        self._retention_ns = retention_ns
        self._max_entries = max_entries
        self._records: dict[str, RobotRecord] = {}
        # One counter for the whole registry, never reset: a result still at a worker is
        # matched to its session by this number, so no number may be used twice, also not
        # after a record was dropped and its id connects again.
        self._last_session = 0

    def connect(
        self,
        robot_id: str,
        subscribed: tuple[str, ...],
        stale_ns: Mapping[str, int],
        watchdog_ns: Mapping[str, int],
        frame_period_ns: int,
        now_ns: int,
    ) -> RobotRecord:
        """Start a session for robot_id, first dropping every record kept past the retention time.

        A known id gets its own record back with the next session number and every counter
        kept. ValueError when the id is connected now; RuntimeError for a new id when every
        record of a full table is connected.
        """
        for expired_id in [
            record.robot_id
            for record in self._records.values()
            if record.disconnected_ns is not None
            and now_ns - record.disconnected_ns > self._retention_ns
        ]:
            del self._records[expired_id]
        record = self._records.get(robot_id)
        if record is None:
            record = self._add(robot_id, subscribed, stale_ns, watchdog_ns, frame_period_ns, now_ns)
        elif record.connected:
            # ASSUMPTION: A22. A second connection with the same id replaces the first session,
            # and the caller closes the old session before it opens the new one.
            raise ValueError(f"robot {robot_id!r} is connected: close its session first")
        else:
            self._last_session += 1
            record.session = self._last_session
            record.connected_ns = now_ns
            record.disconnected_ns = None
            record.subscribed = subscribed
            record.stale_ns = dict(stale_ns)
            record.watchdog_ns = dict(watchdog_ns)
            record.frame_period_ns = frame_period_ns
            # Both describe one session (DESIGN.md 10.4: telemetry restarts with each session).
            record.rtt_ns = None
            record.telemetry = None
        for model in subscribed:
            record.models.setdefault(model, ModelCounters())
        # The longest service gap is measured from the start of the session, so a robot that
        # waits long for its first command shows it.
        for model, counters in record.models.items():
            counters.last_command_ns = now_ns if model in subscribed else None
        return record

    def disconnect(self, robot_id: str, now_ns: int) -> None:
        """End the session of robot_id. KeyError for an unknown id, ValueError when not connected.

        The I/O fields stay as they are, so the last report stays visible for the retention time.
        """
        record = self._records[robot_id]
        if not record.connected:
            raise ValueError(f"robot {robot_id!r} is not connected")
        record.disconnected_ns = now_ns
        # The disconnect ends the session's last interval without a command (DESIGN.md 10.3).
        # A reconnect starts the interval again at its own start, so the gap is closed here or
        # lost.
        for counters in record.models.values():
            if counters.last_command_ns is not None:
                counters.max_service_gap_ns = max(
                    counters.max_service_gap_ns, now_ns - counters.last_command_ns
                )

    def get(self, robot_id: str) -> RobotRecord:
        return self._records[robot_id]

    def records(self) -> Iterable[RobotRecord]:
        """Return every record kept, connected or not."""
        return self._records.values()

    def connected_count(self) -> int:
        return sum(record.connected for record in self._records.values())

    def _add(
        self,
        robot_id: str,
        subscribed: tuple[str, ...],
        stale_ns: Mapping[str, int],
        watchdog_ns: Mapping[str, int],
        frame_period_ns: int,
        now_ns: int,
    ) -> RobotRecord:
        """Add a record for a new id, after dropping the longest disconnected one from a full table.

        The longest disconnected record has the smallest disconnected_ns, then robot_id.
        """
        if len(self._records) >= self._max_entries:
            oldest = min(
                (
                    (record.disconnected_ns, record.robot_id)
                    for record in self._records.values()
                    if record.disconnected_ns is not None
                ),
                default=None,
            )
            if oldest is None:
                # cloud/config.py keeps max_entries at or above the connection cap, and the
                # caller refuses a connection over the cap before it calls connect, so a full
                # table always holds a disconnected record.
                raise RuntimeError(f"all {len(self._records)} robot records are connected")
            del self._records[oldest[1]]
        self._last_session += 1
        record = RobotRecord(
            robot_id=robot_id,
            session=self._last_session,
            connected_ns=now_ns,
            disconnected_ns=None,
            subscribed=subscribed,
            stale_ns=dict(stale_ns),
            watchdog_ns=dict(watchdog_ns),
            frame_period_ns=frame_period_ns,
            models={},
        )
        self._records[robot_id] = record
        return record
