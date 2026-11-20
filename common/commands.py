"""Control commands derived from model outputs, and their wire bodies.

A command body is little-endian and has a fixed size per model:

    cls    "<Hf"    6 bytes   class_id (uint16), confidence (float32)
    depth  "<5f"   20 bytes   left, center, right, steering, speed (float32 each)

The body does not name its model. The message that carries it does, as an
index into MODEL_KEYS.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# The index of a key is its id on the wire, so the order is part of the protocol.
MODEL_KEYS: tuple[str, ...] = ("cls", "depth")

# ASSUMPTION: A3. A command is a small structure derived from the model output,
# not the output itself. Classification sends the top-1 class and its
# confidence. Depth sends the proximity of the left, center and right zones of
# the view, with the steering and speed derived from them: steering is left
# minus right proximity, and speed is one minus center proximity. The depth map
# is never sent back.


@dataclass(frozen=True, slots=True)
class ClsCommand:
    """The top-1 class, as its index among the model's classes, and its softmax probability."""

    class_id: int
    confidence: float


@dataclass(frozen=True, slots=True)
class DepthCommand:
    """Zone proximities in [0, 1], where 1 is the nearest point in view, and the motion they imply.

    Steering lies in [-1, 1] and is positive to the right. Speed lies in [0, 1].
    """

    left: float
    center: float
    right: float
    steering: float
    speed: float


Command = ClsCommand | DepthCommand

_CLS_BODY = struct.Struct("<Hf")
_DEPTH_BODY = struct.Struct("<5f")


class CommandFormatError(Exception):
    """A command body has the wrong length, or its model key is unknown."""


def encode_command(command: Command) -> bytes:
    if isinstance(command, ClsCommand):
        return _CLS_BODY.pack(command.class_id, command.confidence)
    return _DEPTH_BODY.pack(
        command.left, command.center, command.right, command.steering, command.speed
    )


def decode_command(model: str, body: bytes) -> Command:
    """Decode the body of a command that the message header attributes to this model key."""
    try:
        if model == "cls":
            return ClsCommand(*_CLS_BODY.unpack(body))
        if model == "depth":
            return DepthCommand(*_DEPTH_BODY.unpack(body))
    except struct.error as error:
        raise CommandFormatError(f"{model} command body of {len(body)} bytes: {error}") from error
    raise CommandFormatError(f"unknown model {model!r}")
