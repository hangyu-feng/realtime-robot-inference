"""Tests for turning each model's output into a command."""

import math

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp
from numpy.typing import NDArray

from common.commands import MODEL_KEYS, DepthCommand
from workers.models import MODELS, midas, mobilenet

SIGMOID_OF_ONE = 1.0 / (1.0 + math.exp(-1.0))

logit_rows = hnp.arrays(
    np.float32,
    st.integers(min_value=1, max_value=64),
    elements=st.floats(min_value=-1024.0, max_value=1024.0, width=32),
)
# Inverse depth is never negative: the MiDaS graph ends in a ReLU.
depth_maps = hnp.arrays(
    np.float32,
    hnp.array_shapes(min_dims=2, max_dims=2, min_side=3, max_side=16),
    elements=st.floats(min_value=0.0, allow_infinity=False, width=32),
)


def zone_map(left: float, center: float, right: float, upper: float) -> NDArray[np.float32]:
    """A 6x6 map: two upper rows at one value, and three zones of two columns below them."""
    depth_map = np.empty((6, 6), dtype=np.float32)
    depth_map[:2] = upper
    depth_map[2:, 0:2] = left
    depth_map[2:, 2:4] = center
    depth_map[2:, 4:6] = right
    return depth_map


def test_every_model_key_on_the_wire_has_a_model() -> None:
    assert tuple(MODELS) == MODEL_KEYS


@pytest.mark.parametrize(
    ("logits", "class_id", "confidence"),
    [
        pytest.param([0.0, math.log(2.0), math.log(5.0)], 2, 5 / 8, id="odds 1:2:5"),
        pytest.param([0.0, 0.0, 0.0, 0.0], 0, 1 / 4, id="uniform"),
        pytest.param([0.0, 64.0, 0.0], 1, 1.0, id="one dominant class"),
        pytest.param([999.0, 1000.0], 1, SIGMOID_OF_ONE, id="large positive logits"),
        pytest.param([-1000.0, -1001.0], 0, SIGMOID_OF_ONE, id="large negative logits"),
    ],
)
def test_cls_postprocess_returns_the_top_class_and_its_softmax_probability(
    logits: list[float], class_id: int, confidence: float
) -> None:
    command = mobilenet.postprocess(np.asarray(logits, dtype=np.float32))

    assert command.class_id == class_id
    assert command.confidence == pytest.approx(confidence, rel=1e-6)


@given(logits=logit_rows)
def test_cls_postprocess_picks_a_largest_logit(logits: NDArray[np.float32]) -> None:
    command = mobilenet.postprocess(logits)

    assert logits[command.class_id] == logits.max()


@given(logits=logit_rows)
def test_cls_confidence_is_a_probability_no_smaller_than_uniform(
    logits: NDArray[np.float32],
) -> None:
    command = mobilenet.postprocess(logits)

    assert 1 / logits.size <= command.confidence <= 1


def test_depth_postprocess_scores_each_zone_of_the_lower_rows_against_the_nearest_point() -> None:
    command = midas.postprocess(zone_map(left=2.0, center=4.0, right=6.0, upper=8.0))

    assert command == DepthCommand(left=0.25, center=0.5, right=0.75, steering=-0.5, speed=0.5)


@pytest.mark.parametrize(
    ("left", "right", "steering"),
    [
        pytest.param(4.0, 1.0, 0.75, id="nearer on the left steers right"),
        pytest.param(1.0, 4.0, -0.75, id="nearer on the right steers left"),
        pytest.param(4.0, 4.0, 0.0, id="equal sides steer straight"),
    ],
)
def test_depth_steering_turns_away_from_the_nearer_side(
    left: float, right: float, steering: float
) -> None:
    command = midas.postprocess(zone_map(left=left, center=1.0, right=right, upper=4.0))

    assert command.steering == steering


@pytest.mark.parametrize(
    ("center", "speed"),
    [
        pytest.param(8.0, 0.0, id="nearest point ahead stops"),
        pytest.param(2.0, 0.75, id="distant point ahead slows"),
        pytest.param(0.0, 1.0, id="nothing ahead runs at full speed"),
    ],
)
def test_depth_speed_falls_as_the_center_gets_nearer(center: float, speed: float) -> None:
    command = midas.postprocess(zone_map(left=8.0, center=center, right=8.0, upper=8.0))

    assert command.speed == speed


def test_depth_postprocess_reports_no_proximity_for_an_all_zero_map() -> None:
    command = midas.postprocess(np.zeros((256, 256), dtype=np.float32))

    assert command == DepthCommand(left=0.0, center=0.0, right=0.0, steering=0.0, speed=1.0)


@pytest.mark.parametrize(
    ("rows", "columns", "proximities"),
    [
        pytest.param(slice(85, 256), slice(0, 85), (1.0, 0.0, 0.0), id="left zone"),
        pytest.param(slice(85, 256), slice(85, 171), (0.0, 1.0, 0.0), id="center zone"),
        pytest.param(slice(85, 256), slice(171, 256), (0.0, 0.0, 1.0), id="right zone"),
        pytest.param(slice(0, 85), slice(0, 256), (0.0, 0.0, 0.0), id="upper rows"),
    ],
)
def test_depth_zones_tile_the_lower_two_thirds_of_a_model_sized_map(
    rows: slice, columns: slice, proximities: tuple[float, float, float]
) -> None:
    depth_map = np.zeros((256, 256), dtype=np.float32)
    depth_map[rows, columns] = 3.0

    command = midas.postprocess(depth_map)

    assert (command.left, command.center, command.right) == proximities


@given(depth_map=depth_maps)
def test_depth_command_fields_stay_within_their_ranges(depth_map: NDArray[np.float32]) -> None:
    command = midas.postprocess(depth_map)

    unit_fields = (command.left, command.center, command.right, command.speed)
    assert min(unit_fields) >= 0.0
    assert max(unit_fields) <= 1.0
    assert -1.0 <= command.steering <= 1.0


@given(depth_map=depth_maps)
def test_mirrored_depth_map_mirrors_the_command(depth_map: NDArray[np.float32]) -> None:
    command = midas.postprocess(depth_map)

    mirrored = midas.postprocess(depth_map[:, ::-1])

    assert (mirrored.left, mirrored.center, mirrored.right) == pytest.approx(
        (command.right, command.center, command.left)
    )
    assert mirrored.steering == pytest.approx(-command.steering)
