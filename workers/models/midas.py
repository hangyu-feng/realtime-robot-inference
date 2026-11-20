"""MiDaS-small: a frame to the model input, and an inverse-depth map to a depth command.

The graph normalizes its own input: its first nodes subtract the ImageNet mean
and divide by the standard deviation. Preprocessing therefore stops at RGB in
[0, 1], resized to 256x256.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from common.commands import DepthCommand

INPUT_SIDE_PX = 256


def preprocess(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Turn one BGR frame into the RGB input in [0, 1], shaped (3, 256, 256)."""
    resized = cv2.resize(image_bgr, (INPUT_SIDE_PX, INPUT_SIDE_PX), interpolation=cv2.INTER_CUBIC)
    planes_rgb = resized.transpose(2, 0, 1)[::-1]
    # C order makes the copy contiguous, so stacking frames into a batch is a plain copy.
    tensor = planes_rgb.astype(np.float32, order="C")
    tensor /= 255.0
    return tensor


def postprocess(output: NDArray[np.float32]) -> DepthCommand:
    """Reduce one inverse-depth map, where larger is nearer, to zone proximities and a motion."""
    height, width = output.shape
    # ASSUMPTION: A3. Avoidance needs only how near the left, center and right
    # thirds of the view are. A zone's proximity is its mean inverse depth over
    # the lower two thirds of the rows, where the path ahead is, relative to the
    # nearest point in the whole map. The side zones have equal widths, so a
    # mirrored scene mirrors the command.
    side_width_px = width // 3
    lower_rows = output[height // 3 :]
    zones = (
        lower_rows[:, :side_width_px],
        lower_rows[:, side_width_px : width - side_width_px],
        lower_rows[:, width - side_width_px :],
    )
    nearest = float(output.max())
    # The graph ends in a ReLU, so the map is never negative and a mean never
    # exceeds the maximum: each proximity lies in [0, 1]. Means are accumulated
    # in float64 because float32 rounding could carry one past the maximum. An
    # all-zero map has no nearest point, and every zone then reports zero.
    left, center, right = (
        float(zone.mean(dtype=np.float64)) / nearest if nearest > 0.0 else 0.0 for zone in zones
    )
    # Positive steering turns right, away from a nearer left side.
    return DepthCommand(left, center, right, steering=left - right, speed=1.0 - center)
