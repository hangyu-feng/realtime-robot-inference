"""Tests for the control commands and their wire bodies."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from common.commands import (
    MODEL_KEYS,
    ClsCommand,
    CommandFormatError,
    DepthCommand,
    decode_command,
    encode_command,
)

# A body carries float32, so only values that float32 represents exactly come
# back unchanged. NaN is left out because it never compares equal to itself.
wire_floats = st.floats(width=32, allow_nan=False)


def test_model_ids_on_the_wire_follow_the_key_order() -> None:
    assert MODEL_KEYS == ("cls", "depth")


@given(class_id=st.integers(min_value=0, max_value=2**16 - 1), confidence=wire_floats)
def test_cls_command_survives_the_wire(class_id: int, confidence: float) -> None:
    command = ClsCommand(class_id, confidence)

    assert decode_command("cls", encode_command(command)) == command


@given(fields=st.tuples(wire_floats, wire_floats, wire_floats, wire_floats, wire_floats))
def test_depth_command_survives_the_wire(fields: tuple[float, float, float, float, float]) -> None:
    command = DepthCommand(*fields)

    assert decode_command("depth", encode_command(command)) == command


def test_cls_body_is_a_little_endian_uint16_then_a_float32() -> None:
    body = encode_command(ClsCommand(class_id=0x0102, confidence=0.5))

    assert body == bytes.fromhex("0201 0000003f")


def test_depth_body_is_five_little_endian_float32_in_field_order() -> None:
    body = encode_command(
        DepthCommand(left=1.0, center=0.5, right=0.25, steering=0.75, speed=0.125)
    )

    assert body == bytes.fromhex("0000803f 0000003f 0000803e 0000403f 0000003e")


@pytest.mark.parametrize(
    ("model", "body_bytes"),
    [("cls", 0), ("cls", 5), ("cls", 7), ("cls", 20), ("depth", 0), ("depth", 6), ("depth", 19)],
)
def test_body_of_the_wrong_length_is_rejected(model: str, body_bytes: int) -> None:
    with pytest.raises(CommandFormatError, match=f"{model} command body of {body_bytes} bytes"):
        decode_command(model, bytes(body_bytes))


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(CommandFormatError, match="unknown model 'pose'"):
        decode_command("pose", bytes(6))
