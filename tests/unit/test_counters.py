"""Tests for the gateway's shared metric state: every label of /metrics is there from the start."""

from __future__ import annotations

from cloud.counters import ConnectionTotals, WorkerState, WorkerStatus

# The labels of DESIGN.md 10.7.
CLOSE_LABELS = {
    "client_close",
    "abnormal",
    "idle_timeout",
    "slow_consumer",
    "ingress_limit",
    "protocol_error",
    "replaced",
    "shutdown",
    "internal_error",
}
REJECT_LABELS = {"capacity", "version", "auth", "bad_hello", "unknown_model", "handshake_timeout"}
FRAME_REJECT_LABELS = {"malformed", "oversize", "rate_limited"}


def test_new_connection_totals_hold_every_label_at_zero() -> None:
    totals = ConnectionTotals()
    assert totals.accepted == 0
    assert totals.closed == dict.fromkeys(CLOSE_LABELS, 0)
    assert totals.rejected == dict.fromkeys(REJECT_LABELS, 0)
    assert totals.frames_rejected == dict.fromkeys(FRAME_REJECT_LABELS, 0)
    assert (totals.bytes_in, totals.bytes_out) == (0, 0)


def test_connection_totals_never_share_their_counters() -> None:
    counted = ConnectionTotals()
    counted.closed["abnormal"] += 1
    counted.rejected["capacity"] += 1
    counted.frames_rejected["malformed"] += 1
    fresh = ConnectionTotals()
    assert fresh.closed == dict.fromkeys(CLOSE_LABELS, 0)
    assert fresh.rejected == dict.fromkeys(REJECT_LABELS, 0)
    assert fresh.frames_rejected == dict.fromkeys(FRAME_REJECT_LABELS, 0)


def test_close_reason_labels_are_plain_strings() -> None:
    assert all(type(label) is str for label in ConnectionTotals().closed)


def test_a_new_worker_status_is_warming_with_no_process_and_no_counts() -> None:
    assert WorkerStatus() == WorkerStatus(
        state=WorkerState.WARMING, pid=None, restarts=0, timeouts=0
    )


def test_worker_state_values_are_the_labels_of_the_contract() -> None:
    assert [state.value for state in WorkerState] == ["warming", "ready", "restarting", "failed"]
