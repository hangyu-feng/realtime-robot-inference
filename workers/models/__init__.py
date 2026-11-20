"""Registry of the served models: their files, preprocessing and postprocessing by model key."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from common.commands import Command
from workers.models import midas, mobilenet


@dataclass(frozen=True)
class ModelSpec:
    """One served model. Both file names are relative to the model directory."""

    key: str
    published_file: str
    # Accepts any batch size. The same as the published file when that one already does.
    batched_file: str
    preprocess: Callable[[NDArray[np.uint8]], NDArray[np.float32]]
    postprocess: Callable[[NDArray[np.float32]], Command]


MODELS: Mapping[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="cls",
            published_file="mobilenetv3_small.onnx",
            batched_file="mobilenetv3_small.onnx",
            preprocess=mobilenet.preprocess,
            postprocess=mobilenet.postprocess,
        ),
        ModelSpec(
            key="depth",
            published_file="midas_small.onnx",
            batched_file="midas_small_batched.onnx",
            preprocess=midas.preprocess,
            postprocess=midas.postprocess,
        ),
    )
}
