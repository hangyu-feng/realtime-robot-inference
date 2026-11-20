"""The gateway process while its models warm up, and when a model file cannot be loaded."""

import asyncio
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from common.commands import MODEL_KEYS
from common.transport import TransportConnectError
from common.transport_ws import connect_ws
from tests.integration.harness import (
    READY_TIMEOUT_S,
    ROBOT_CLIENT_OPTIONS,
    Service,
    fleet_timeout_s,
    killed_on_exit,
    plan,
    robot_commands,
    running_service,
    scrape,
    start_fleet,
    time_zero_ns,
)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux", reason="the service needs Linux: uvloop, /proc and cgroup files"
)

# Only keeps a broken service from hanging the suite: a worker that cannot load its model
# exits within a second or two.
STARTUP_FAILURE_TIMEOUT_S = 30.0
HEALTHZ_TIMEOUT_S = 5.0
# The cls model file, which a first start loads.
CLS_MODEL_FILE = "mobilenetv3_small.onnx"


def healthz(service: Service) -> tuple[int, Any]:
    """Return the status and the parsed body of GET /healthz."""
    try:
        with urllib.request.urlopen(
            f"{service.http_url}/healthz", timeout=HEALTHZ_TIMEOUT_S
        ) as reply:
            return reply.status, json.loads(reply.read())
    except urllib.error.HTTPError as error:
        with error:
            return error.code, json.loads(error.read())


def test_the_service_accepts_robots_only_once_both_models_are_warm(tmp_path: Path) -> None:
    # Depth calibrates for about 5 s: batches of one and two 400 ms frames, each run four times.
    overrides = {
        "calibration_runs": 3,
        "models": {"depth": {"fake_frame_ms": 400.0, "batch_sizes": [1, 2]}},
    }

    with running_service(overrides, tmp_path / "gateway", wait_ready=False) as service:
        before = scrape(service)
        warming = healthz(service)
        with pytest.raises(TransportConnectError) as refused:
            asyncio.run(connect_ws(service.ws_url, ROBOT_CLIENT_OPTIONS))
        fleet_plan = plan(time_zero_ns(), 12.0, [("t13", 0.0, None)])
        with killed_on_exit(start_fleet(service, fleet_plan, tmp_path / "fleet")) as fleet:
            service.wait_ready(READY_TIMEOUT_S)
            ready = healthz(service)
            fleet_status = fleet.wait(fleet_timeout_s(fleet_plan))
        after = scrape(service)

    status, body = warming
    commands = robot_commands(tmp_path / "fleet", "t13")
    assert status == 503
    assert (body["status"], body["accepting"], body["workers"]["depth"]) == (
        "unavailable",
        False,
        "warming",
    )
    assert refused.value.http_status == 503
    assert fleet_status == 0
    assert set(commands["model"].tolist()) == set(range(len(MODEL_KEYS)))
    # The robot was refused until the gateway was ready, and then never lost its session.
    assert after["robots"]["t13"]["telemetry"]["reconnects"] == 0
    assert after["connections"]["accepted"] - before["connections"]["accepted"] == 1
    assert ready[0] == 200


@pytest.mark.parametrize(
    "model_files",
    [
        pytest.param({}, id="missing"),
        pytest.param({CLS_MODEL_FILE: b"not an onnx model"}, id="corrupt"),
    ],
)
def test_a_missing_or_corrupt_model_file_stops_the_gateway_at_once_with_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model_files: dict[str, bytes]
) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for name, content in model_files.items():
        (model_dir / name).write_bytes(content)
    monkeypatch.setenv("MODEL_DIR", str(model_dir))

    # Both models on ONNX Runtime, with a warm-up timeout far above the bound of this test, so
    # that only the worker's exit can end the start in time.
    with running_service(
        {"warmup_timeout_ms": 600_000},
        tmp_path / "gateway",
        wait_ready=False,
        expect_exit=1,
        base={"calibration_runs": 1},
    ) as service:
        status = service.process.wait(STARTUP_FAILURE_TIMEOUT_S)

    errors = (tmp_path / "gateway" / "gateway.err").read_text()
    assert status == 1
    assert "worker cls: cannot load model 'cls'" in errors
    assert "gateway: cls failed to start" in errors
