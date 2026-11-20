"""The done criterion on the real models: one robot's samples agree with the live /metrics.

The model files exist only in the image, so this module skips unless MODEL_DIR names them.
"""

import os
import sys
from pathlib import Path

import pytest

from bench.live_check import agreement, boundary_violations, conservation_violations
from tests.integration.harness import (
    plan,
    robot_commands,
    run_fleets,
    running_service,
    scrape,
    time_zero_ns,
)

MODEL_DIR_VARIABLE = "MODEL_DIR"

pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
    ),
    pytest.mark.skipif(
        os.environ.get(MODEL_DIR_VARIABLE) is None,
        reason=f"{MODEL_DIR_VARIABLE} is not set: the model files exist only in the image",
    ),
]

# Warm-up loads both models and calibrates every batch size, one model after the other; the
# bound only keeps a broken start from hanging the suite.
READY_TIMEOUT_S = 600.0
DURATION_S = 30.0


def test_one_robot_on_the_real_models_agrees_with_metrics(tmp_path: Path) -> None:
    with running_service({}, tmp_path / "gateway", wait_ready=False, base={}) as service:
        service.wait_ready(READY_TIMEOUT_S)
        run_fleets(
            service,
            [(plan(time_zero_ns(), DURATION_S, [("t19", 0.0, None)]), tmp_path / "fleet", ())],
        )
        snapshot = scrape(service)

    commands = robot_commands(tmp_path / "fleet", "t19")
    assert snapshot["robots"]["t19"]["subscribed"] == ["cls", "depth"]
    assert agreement(snapshot, "t19", commands) == []
    assert boundary_violations(snapshot, "t19") == []
    assert conservation_violations(snapshot) == []
