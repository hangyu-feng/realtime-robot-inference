"""What a robot connection is, to the gateway and to the robots.

The session code on both ends talks to a Connection and is served by a Listener. The
WebSocket adapter implements them over the library, and the unit tests run the same session
code over an in-memory implementation. This module holds the types only.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


class TransportClosed(Exception):
    """The connection has closed.

    peer_code is the code of the close frame the peer sent, when its close came first.
    local_code is the code this side sent first, by this program or by the library (1002,
    1007, 1009, 1011). Neither is set when the connection ended without a close frame either
    way.
    """

    def __init__(self, peer_code: int | None, local_code: int | None) -> None:
        if peer_code is not None and local_code is not None:
            raise ValueError(
                f"only one side's close comes first, got peer_code={peer_code} "
                f"and local_code={local_code}"
            )
        super().__init__(
            f"the connection has closed (peer_code={peer_code}, local_code={local_code})"
        )
        self.peer_code = peer_code
        self.local_code = local_code


class TransportConnectError(Exception):
    """A connection could not be opened.

    http_status is the status of an answer to the upgrade other than 101, such as 503 while
    the service warms up or is full; None when no such answer came.
    """

    def __init__(self, message: str, http_status: int | None) -> None:
        super().__init__(message)
        self.http_status = http_status


class Connection(Protocol):
    """One message connection to a peer."""

    async def recv(self) -> str | bytes:
        """Return the next message.

        Raises TransportClosed when the connection has closed and no message is left. Safe to
        cancel: a cancelled call loses no message.
        """

    async def send(self, message: str | bytes) -> None:
        """Write the message, then return once the transport holds no more than its write limit.

        Raises TransportClosed.
        """

    def send_nowait(self, message: str | bytes) -> None:
        """Write the message now and never wait.

        Whatever the kernel refuses stays in the transport and counts in unsent_bytes().
        Raises TransportClosed when the connection is not open or the write failed.
        """

    def unsent_bytes(self) -> int:
        """Return the bytes the transport holds that the kernel has not taken."""

    def rtt_ns(self) -> int | None:
        """Return the last ping round trip.

        None before the first one returns, and while the last one measured 0, below the
        resolution of the clock that measures it.
        """

    async def close(self, code: int, reason: str) -> None:
        """Run the closing handshake, bounded by the adapter's close timeout, then abort.

        The abort is a no-op on a connection that has closed, so it drops only one whose
        handshake did not end in time; it also happens when the caller is cancelled.
        Idempotent, and never raises.
        """

    def abort(self) -> None:
        """Drop the connection without a close frame; a no-op once it has closed."""


@dataclass(frozen=True, slots=True)
class HttpReply:
    """The answer to a plain HTTP request."""

    status: int
    # Always JSON.
    body: bytes


# Serves one connection, from the upgrade to its end; the listener runs one per connection.
Handler = Callable[[Connection], Awaitable[None]]
# Answers an HTTP request for a path, given without its query. None lets the request continue
# with the WebSocket upgrade.
HttpHandler = Callable[[str], Awaitable[HttpReply | None]]


class Listener(Protocol):
    """Accepts connections and runs a Handler for each."""

    @property
    def port(self) -> int:
        """The port bound."""

    def stop_accepting(self) -> None:
        """Stop taking new connections; the open ones stay."""

    async def wait_closed(self) -> None:
        """Return when every connection and every handler has gone."""
