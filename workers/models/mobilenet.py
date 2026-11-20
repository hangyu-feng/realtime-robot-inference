"""MobileNetV3-small: a frame to the model input, and logits to a classification command.

The graph takes normalized RGB, so all preprocessing happens here: resize the
shorter side to 256 pixels, center crop to 224, scale to [0, 1] and normalize
with the ImageNet mean and standard deviation.
"""

from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from common.commands import ClsCommand

INPUT_SIDE_PX = 224
RESIZED_SHORTER_SIDE_PX = 256

# ImageNet statistics per RGB plane, shaped to broadcast over (3, height, width).
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)
# (x / 255 - mean) / std, folded into one multiply and one add per plane.
_PLANE_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_PLANE_OFFSET = (-_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)


def preprocess(image_bgr: NDArray[np.uint8]) -> NDArray[np.float32]:
    """Turn one BGR frame into the normalized RGB input, shaped (3, 224, 224)."""
    height, width = image_bgr.shape[:2]
    # Resizing the shorter side to 256 and cropping 224 keeps a centered square
    # that spans 224/256 of the shorter side. Cropping that square first and
    # resizing once samples the same source positions to within a pixel, and
    # never resizes pixels that the crop discards.
    crop_side_px = min(height, width) * INPUT_SIDE_PX // RESIZED_SHORTER_SIDE_PX
    top = (height - crop_side_px) // 2
    left = (width - crop_side_px) // 2
    resized = cv2.resize(
        image_bgr[top : top + crop_side_px, left : left + crop_side_px],
        (INPUT_SIDE_PX, INPUT_SIDE_PX),
        interpolation=cv2.INTER_CUBIC,
    )
    planes_rgb = resized.transpose(2, 0, 1)[::-1]
    # C order makes the copy contiguous, so stacking frames into a batch is a plain copy.
    tensor = planes_rgb.astype(np.float32, order="C")
    tensor *= _PLANE_SCALE
    tensor += _PLANE_OFFSET
    return tensor


def postprocess(output: NDArray[np.float32]) -> ClsCommand:
    """Return the top-1 class of one row of logits and its softmax probability."""
    class_id = int(output.argmax())
    # Subtracting the top logit from every term leaves no positive exponent, so
    # exp cannot overflow.
    confidence = 1.0 / float(np.exp(output - output[class_id], dtype=np.float64).sum())
    return ClsCommand(class_id, confidence)
