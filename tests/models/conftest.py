"""Fixtures for the tests that run the real models. The model files exist only in the image."""

import os
from pathlib import Path

import pytest

from workers.models import MODELS

MODEL_DIR_VARIABLE = "MODEL_DIR"


@pytest.fixture(scope="session", autouse=True)
def model_dir() -> Path:
    """Return the model directory. Skip this package without one; fail on one that lacks a file."""
    value = os.environ.get(MODEL_DIR_VARIABLE)
    if value is None:
        pytest.skip(f"{MODEL_DIR_VARIABLE} is not set: the model files exist only in the image")
    directory = Path(value)
    missing = sorted(
        {
            file
            for spec in MODELS.values()
            for file in (spec.published_file, spec.batched_file)
            if not (directory / file).is_file()
        }
    )
    if missing:
        pytest.fail(f"{MODEL_DIR_VARIABLE}={directory} has no {', '.join(missing)}", pytrace=False)
    return directory
