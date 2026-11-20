"""The gateway's metric state that several I/O layers write and the /metrics snapshot reads.

Pure state, no I/O: the session, worker and gateway code update these counters, and
cloud/metrics.py renders them. Every label is present from construction, at 0, so
/metrics always shows every reason.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from cloud.admission import RejectReason
from common.protocol import Fault


class CloseReason(StrEnum):
    """Why an accepted session ended: the labels of connections.closed."""

    CLIENT_CLOSE = "client_close"
    ABNORMAL = "abnormal"
    IDLE_TIMEOUT = "idle_timeout"
    SLOW_CONSUMER = "slow_consumer"
    INGRESS_LIMIT = "ingress_limit"
    PROTOCOL_ERROR = "protocol_error"
    REPLACED = "replaced"
    SHUTDOWN = "shutdown"
    # A fault in the gateway's own code; 0 in a correct run.
    INTERNAL_ERROR = "internal_error"


# Labels decided by the I/O layer rather than by the admission rules or the protocol.
HANDSHAKE_TIMEOUT = "handshake_timeout"
OVERSIZE = "oversize"
RATE_LIMITED = "rate_limited"

REJECT_REASONS: tuple[str, ...] = (
    RejectReason.CAPACITY.value,
    Fault.VERSION.value,
    RejectReason.AUTH.value,
    Fault.BAD_HELLO.value,
    Fault.UNKNOWN_MODEL.value,
    HANDSHAKE_TIMEOUT,
)
FRAME_REJECT_REASONS: tuple[str, ...] = (Fault.MALFORMED.value, OVERSIZE, RATE_LIMITED)


@dataclass(slots=True)
class ConnectionTotals:
    """The counters of connections in /metrics. The current count is the registry's."""

    accepted: int = 0
    closed: dict[str, int] = field(
        default_factory=lambda: {reason.value: 0 for reason in CloseReason}
    )
    rejected: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REJECT_REASONS, 0))
    frames_rejected: dict[str, int] = field(
        default_factory=lambda: dict.fromkeys(FRAME_REJECT_REASONS, 0)
    )
    bytes_in: int = 0
    bytes_out: int = 0


class WorkerState(StrEnum):
    """Where one model's worker is in its lifecycle: the values of workers.<m>.state."""

    # The first start, until its first WorkerReady.
    WARMING = "warming"
    READY = "ready"
    # After a failure, while a new process starts and warms up.
    RESTARTING = "restarting"
    # A restart attempt failed; the next one comes after the backoff.
    FAILED = "failed"


@dataclass(slots=True)
class WorkerStatus:
    """The live state of one model's worker in /metrics."""

    state: WorkerState = WorkerState.WARMING
    # None while no worker process is live.
    pid: int | None = None
    restarts: int = 0
    timeouts: int = 0
