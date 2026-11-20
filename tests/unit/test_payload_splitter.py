"""Tests for the splitter that turns a worker's byte stream into whole payloads."""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from cloud.worker_link import PayloadSplitter
from common.worker_protocol import LENGTH_PREFIX, MAX_PAYLOAD_BYTES, WorkerProtocolError


def framed(payload: bytes) -> bytes:
    """Return the payload as it travels: its length prefix, then its bytes."""
    return LENGTH_PREFIX.pack(len(payload)) + payload


@st.composite
def streams_in_chunks(draw: st.DrawFn) -> tuple[list[bytes], list[bytes]]:
    """Draw payloads and one way of cutting their stream into chunks, empty chunks included."""
    payloads = draw(st.lists(st.binary(min_size=1, max_size=64), max_size=8))
    stream = b"".join(framed(payload) for payload in payloads)
    cuts = sorted(draw(st.lists(st.integers(0, len(stream)), max_size=24)))
    chunks = [stream[start:end] for start, end in pairwise([0, *cuts, len(stream)])]
    return payloads, chunks


@given(streams_in_chunks())
def test_any_chunking_of_a_stream_yields_the_same_payloads(
    stream: tuple[list[bytes], list[bytes]],
) -> None:
    payloads, chunks = stream
    splitter = PayloadSplitter()

    split = [bytes(payload) for chunk in chunks for payload in splitter.feed(chunk)]

    assert split == payloads


@given(st.lists(st.binary(min_size=1, max_size=64), min_size=1, max_size=8))
def test_a_chunk_yields_exactly_the_payloads_it_completes(payloads: list[bytes]) -> None:
    splitter = PayloadSplitter()
    stream = b"".join(framed(payload) for payload in payloads)

    before_the_last_byte = splitter.feed(stream[:-1])
    at_the_last_byte = splitter.feed(stream[-1:])

    assert [bytes(payload) for payload in before_the_last_byte] == payloads[:-1]
    assert [bytes(payload) for payload in at_the_last_byte] == payloads[-1:]


def test_a_payload_is_a_copy_that_a_later_chunk_does_not_change() -> None:
    splitter = PayloadSplitter()
    chunk = bytearray(framed(b"first") + framed(b"second")[:6])

    (first,) = splitter.feed(bytes(chunk))
    (second,) = splitter.feed(framed(b"second")[6:])

    assert (bytes(first), bytes(second)) == (b"first", b"second")
    assert first.obj is not second.obj


@pytest.mark.parametrize("length_bytes", [0, MAX_PAYLOAD_BYTES + 1, 2**32 - 1])
@pytest.mark.parametrize("first_chunk_bytes", [0, 1, 3])
def test_a_length_no_payload_may_have_is_refused_at_the_last_byte_of_its_prefix(
    length_bytes: int, first_chunk_bytes: int
) -> None:
    splitter = PayloadSplitter()
    prefix = LENGTH_PREFIX.pack(length_bytes)

    # No payload byte has arrived: the length is refused before anything is buffered for it.
    assert splitter.feed(prefix[:first_chunk_bytes]) == []
    with pytest.raises(WorkerProtocolError, match="payload length"):
        splitter.feed(prefix[first_chunk_bytes:])


def test_the_largest_allowed_length_is_accepted_without_its_payload() -> None:
    splitter = PayloadSplitter()

    assert splitter.feed(LENGTH_PREFIX.pack(MAX_PAYLOAD_BYTES) + b"x") == []


def test_a_bad_length_after_a_whole_payload_is_refused() -> None:
    splitter = PayloadSplitter()

    with pytest.raises(WorkerProtocolError, match="payload length"):
        splitter.feed(framed(b"whole") + LENGTH_PREFIX.pack(0))
