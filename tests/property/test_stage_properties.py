"""Properties of the server's stage durations on a shared clock.

Every command carries eight stages in whole microseconds, and the robot adds them to its own
two, fire lag and deliver, to account for the freshness it measured (DESIGN.md 7).
"""

from dataclasses import astuple, dataclass
from itertools import pairwise

from hypothesis import given
from hypothesis import strategies as st

from cloud.dispatch import check_result, stage_durations
from cloud.mailbox import Frame
from cloud.registry import RobotRecord
from cloud.scheduler import Batch, BatchItem
from common.commands import ClsCommand, encode_command
from common.worker_protocol import BatchResult, FrameResult, FrameStatus

NS_PER_US = 1_000
JPEG = b"\xff\xd8"
BODY = encode_command(ClsCommand(class_id=1, confidence=0.5))
RECORD = RobotRecord(
    robot_id="a",
    session=1,
    connected_ns=0,
    disconnected_ns=None,
    subscribed=("cls",),
    stale_ns={"cls": 1},
    watchdog_ns={"cls": 1},
    frame_period_ns=1,
    models={},
)
# Intervals around a microsecond, where the rounding of each stage decides the most, and up to
# ten seconds.
INTERVALS_NS = st.integers(min_value=0, max_value=3_000) | st.integers(
    min_value=0, max_value=10**10
)


@dataclass(frozen=True)
class Timeline:
    """The clock readings of one command on a shared clock, and its frame's worker stages."""

    capture_ns: int
    send_ns: int
    received_ns: int
    dispatched_ns: int
    worker_received_ns: int
    worker_sent_ns: int
    result_ns: int
    receive_ns: int
    decode_ns: int
    preprocess_ns: int
    inference_ns: int
    postprocess_ns: int

    def frame(self) -> Frame:
        return Frame(
            1,
            self.capture_ns,
            self.received_ns,
            self.send_ns - self.capture_ns,
            self.received_ns - self.send_ns,
            JPEG,
        )

    def frame_result(self) -> FrameResult:
        return FrameResult(
            FrameStatus.OK, BODY, self.decode_ns, self.preprocess_ns, self.postprocess_ns
        )

    def result(self) -> BatchResult:
        return BatchResult(
            1,
            self.worker_received_ns,
            self.worker_sent_ns,
            self.inference_ns,
            0,
            (self.frame_result(),),
        )

    def stages_us(self) -> tuple[int, ...]:
        stages = stage_durations(
            self.frame(), self.dispatched_ns, self.result(), self.frame_result(), self.result_ns
        )
        return astuple(stages)


@st.composite
def timelines(draw: st.DrawFn) -> Timeline:
    """Draw readings in the order the stages take them, with worker stages that fit."""
    capture_ns = draw(st.integers(min_value=-(2**62), max_value=2**62))
    send_ns = capture_ns + draw(INTERVALS_NS)
    received_ns = send_ns + draw(INTERVALS_NS)
    dispatched_ns = received_ns + draw(INTERVALS_NS)
    worker_received_ns = dispatched_ns + draw(INTERVALS_NS)
    worker_ns = draw(INTERVALS_NS)
    # Four cuts split the worker's time for the batch into this frame's decode, preprocess,
    # inference and postprocess, and the time spent on the other frames of the batch.
    cuts_ns = sorted(
        draw(st.lists(st.integers(min_value=0, max_value=worker_ns), min_size=4, max_size=4))
    )
    decode_ns, preprocess_ns, inference_ns, postprocess_ns = (
        later - earlier for earlier, later in pairwise([0, *cuts_ns])
    )
    worker_sent_ns = worker_received_ns + worker_ns
    result_ns = worker_sent_ns + draw(INTERVALS_NS)
    return Timeline(
        capture_ns=capture_ns,
        send_ns=send_ns,
        received_ns=received_ns,
        dispatched_ns=dispatched_ns,
        worker_received_ns=worker_received_ns,
        worker_sent_ns=worker_sent_ns,
        result_ns=result_ns,
        receive_ns=result_ns + draw(INTERVALS_NS),
        decode_ns=decode_ns,
        preprocess_ns=preprocess_ns,
        inference_ns=inference_ns,
        postprocess_ns=postprocess_ns,
    )


@given(timeline=timelines())
def test_the_stages_are_never_negative_and_add_up_to_the_interval_rounded_down(
    timeline: Timeline,
) -> None:
    stages_us = timeline.stages_us()

    assert min(stages_us) >= 0
    assert sum(stages_us) == (timeline.result_ns - timeline.send_ns) // NS_PER_US


@given(timeline=timelines())
def test_the_robot_accounts_for_its_freshness_to_under_a_microsecond(timeline: Timeline) -> None:
    freshness_ns = timeline.receive_ns - timeline.capture_ns
    fire_lag_ns = timeline.send_ns - timeline.capture_ns
    deliver_ns = timeline.receive_ns - timeline.result_ns

    residual_ns = freshness_ns - fire_lag_ns - NS_PER_US * sum(timeline.stages_us()) - deliver_ns

    assert 0 <= residual_ns < NS_PER_US


@given(timeline=timelines())
def test_each_stage_is_within_a_microsecond_of_its_own_interval(timeline: Timeline) -> None:
    own_ns = (
        timeline.decode_ns
        + timeline.preprocess_ns
        + timeline.inference_ns
        + timeline.postprocess_ns
    )
    intervals_ns = (
        timeline.received_ns - timeline.send_ns,
        timeline.dispatched_ns - timeline.received_ns,
        timeline.worker_received_ns
        - timeline.dispatched_ns
        + timeline.result_ns
        - timeline.worker_sent_ns,
        timeline.decode_ns,
        timeline.preprocess_ns,
        timeline.worker_sent_ns - timeline.worker_received_ns - own_ns,
        timeline.inference_ns,
        timeline.postprocess_ns,
    )

    gaps_ns = [
        NS_PER_US * stage_us - interval_ns
        for stage_us, interval_ns in zip(timeline.stages_us(), intervals_ns, strict=True)
    ]

    assert all(-NS_PER_US < gap_ns < NS_PER_US for gap_ns in gaps_ns)


@given(timeline=timelines())
def test_a_result_with_ordered_stamps_and_fitting_stages_passes_the_check(
    timeline: Timeline,
) -> None:
    batch = Batch(1, (BatchItem(RECORD, 1, timeline.frame()),), timeline.dispatched_ns)

    check_result(batch, timeline.result(), timeline.result_ns)
