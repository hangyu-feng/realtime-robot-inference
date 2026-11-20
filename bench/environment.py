"""Collect the environment block that every benchmark result records.

The block says where a number was measured: the CPU, the kernel, the limits as
read inside the container, the interpreter and library versions, and the hashes
of the model files.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import os
import platform
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from common.cgroup import read_limits

CPUINFO_PATH = Path("/proc/cpuinfo")

# The distributions whose version can change a measured number.
PACKAGES = ("onnxruntime", "numpy", "opencv-python-headless", "websockets", "uvloop", "onnx")

_MODEL_SUFFIX = ".onnx"


def parse_cpu_model(cpuinfo: str) -> str | None:
    """Return the first "model name" of /proc/cpuinfo. Some architectures list none."""
    for line in cpuinfo.splitlines():
        key, _, value = line.partition(":")
        if key.strip() == "model name":
            return value.strip()
    return None


def collect_environment(model_dir: Path | None) -> dict[str, object]:
    """Describe the machine, the limits, the software and the models of a benchmark run."""
    try:
        cpuinfo = CPUINFO_PATH.read_text()
    except FileNotFoundError:
        # Only Linux has /proc/cpuinfo.
        cpu_model = None
    else:
        cpu_model = parse_cpu_model(cpuinfo)

    packages: dict[str, str | None] = {}
    for name in PACKAGES:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None

    model_sha256: dict[str, str] = {}
    if model_dir is not None:
        # iterdir, not glob: a model directory that does not exist must fail, not hash nothing.
        for path in sorted(path for path in model_dir.iterdir() if path.suffix == _MODEL_SUFFIX):
            with path.open("rb") as file:
                model_sha256[path.name] = hashlib.file_digest(file, "sha256").hexdigest()

    return {
        "cpu_model": cpu_model,
        "cpu_count_visible": os.process_cpu_count(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "cgroup": asdict(read_limits()),
        "python": platform.python_version(),
        "packages": packages,
        "model_sha256": model_sha256,
        # Wall-clock time is right here: the timestamp is recorded, never subtracted.
        "start_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
