"""The fake and the null runtime: a runner that serves commands without a model file.

The fake runtime takes a fixed service time per frame, for scheduling tests whose times are
known in advance. The null runtime takes none, which isolates the cost of transport and
scheduling.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from common.commands import MODEL_KEYS, ClsCommand, DepthCommand
from workers.runtime import BatchOutcome, FrameOutcome

# The JPEG start-of-image marker and the first byte of the marker after it. The gateway's
# prefilter checks only the first two bytes, so a frame can pass it and still fail here, which
# takes the per-frame failure path end to end.
_JPEG_START = b"\xff\xd8\xff"
# The class id travels as a u16 in the command body.
_CLASS_ID_MASK = 0xFFFF
# Nothing near: no steering, and full speed.
_DEPTH_COMMAND = DepthCommand(left=0.0, center=0.0, right=0.0, steering=0.0, speed=1.0)


class FakeRunner:
    """A runner for the fake and the null runtime.

    It waits instead of inferring: the fake runtime with a service time per frame, and the null
    runtime with none. A frame that does not start like a JPEG fails as a frame that does not
    decode, and the others share one wait. A batch with nothing to wait for returns at once.

    The class id of a cls command is the frame's length in bytes, so that an end-to-end test can
    check that each frame's own bytes reached the worker. The inference time reported is the
    nominal wait, which keeps the worker's stages within its measured wall time, because
    time.sleep never returns early. The fake uses almost no CPU, so the feed-forward pacing
    share does not engage in fake runs.
    """

    def __init__(self, model: str, frame_ns: int, wait: Callable[[float], None]) -> None:
        if model not in MODEL_KEYS:
            raise ValueError(f"unknown model {model!r}; the models are {', '.join(MODEL_KEYS)}")
        if frame_ns < 0:
            raise ValueError(f"frame_ns must not be negative, got {frame_ns}")
        self._model = model
        self._frame_ns = frame_ns
        self._wait = wait

    def run_batch(self, jpegs: Sequence[bytes]) -> BatchOutcome:
        frames: list[FrameOutcome] = []
        for jpeg in jpegs:
            if not jpeg.startswith(_JPEG_START):
                frames.append(FrameOutcome(None, 0, 0, 0))
            elif self._model == "cls":
                command = ClsCommand(class_id=len(jpeg) & _CLASS_ID_MASK, confidence=1.0)
                frames.append(FrameOutcome(command, 0, 0, 0))
            else:
                frames.append(FrameOutcome(_DEPTH_COMMAND, 0, 0, 0))
        inference_ns = self._frame_ns * sum(frame.command is not None for frame in frames)
        # A sleep of 0 is still a system call that gives up the CPU, and the thread runs again
        # only when the kernel wakes it. The null runtime would then report a batch that waited
        # as one with no inference, so a wait of nothing is not made.
        if inference_ns > 0:
            self._wait(inference_ns / 1e9)
        return BatchOutcome(tuple(frames), inference_ns)
