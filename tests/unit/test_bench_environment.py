"""Tests for the environment block of a benchmark result."""

import json
import platform
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from bench import environment
from bench.environment import collect_environment, parse_cpu_model
from common.cgroup import read_limits

X86_CPUINFO = (
    "processor\t: 0\n"
    "vendor_id\t: ExampleVendor\n"
    "model\t\t: 183\n"
    "model name\t: Example(R) Processor 9000 @ 2.00GHz\n"
    "stepping\t: 1\n"
    "\n"
    "processor\t: 1\n"
    "model name\t: A later processor\n"
)

ARM_CPUINFO = (
    "processor\t: 0\n"
    "BogoMIPS\t: 48.00\n"
    "Features\t: fp asimd evtstrm\n"
    "CPU implementer\t: 0x61\n"
    "CPU part\t: 0x022\n"
)


def test_parse_cpu_model_returns_the_first_model_name() -> None:
    assert parse_cpu_model(X86_CPUINFO) == "Example(R) Processor 9000 @ 2.00GHz"


def test_parse_cpu_model_keeps_a_colon_inside_the_name() -> None:
    assert parse_cpu_model("model name\t: Example: the sequel\n") == "Example: the sequel"


def test_parse_cpu_model_is_none_when_no_model_name_is_listed() -> None:
    assert parse_cpu_model(ARM_CPUINFO) is None


def test_environment_has_exactly_the_contract_keys() -> None:
    assert set(collect_environment(None)) == {
        "cpu_model",
        "cpu_count_visible",
        "machine",
        "kernel",
        "cgroup",
        "python",
        "packages",
        "model_sha256",
        "start_utc",
    }


def test_environment_survives_a_json_round_trip() -> None:
    collected = collect_environment(None)

    assert json.loads(json.dumps(collected)) == collected


def test_cpu_model_is_read_from_cpuinfo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cpuinfo_path = tmp_path / "cpuinfo"
    cpuinfo_path.write_text(X86_CPUINFO)
    monkeypatch.setattr(environment, "CPUINFO_PATH", cpuinfo_path)

    assert collect_environment(None)["cpu_model"] == "Example(R) Processor 9000 @ 2.00GHz"


def test_cpu_model_is_none_where_there_is_no_cpuinfo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(environment, "CPUINFO_PATH", tmp_path / "cpuinfo")

    assert collect_environment(None)["cpu_model"] is None


def test_cpu_count_visible_counts_at_least_one_cpu() -> None:
    cpu_count_visible = collect_environment(None)["cpu_count_visible"]

    assert isinstance(cpu_count_visible, int)
    assert cpu_count_visible >= 1


def test_interpreter_machine_and_kernel_are_identified() -> None:
    collected = collect_environment(None)

    assert (collected["python"], collected["machine"], collected["kernel"]) == (
        platform.python_version(),
        platform.machine(),
        platform.release(),
    )


def test_cgroup_block_holds_the_limits_read_from_the_cgroup() -> None:
    assert collect_environment(None)["cgroup"] == asdict(read_limits())


def test_packages_cover_the_libraries_that_can_change_a_measurement() -> None:
    packages = collect_environment(None)["packages"]

    assert isinstance(packages, dict)
    assert set(packages) == {
        "onnxruntime",
        "numpy",
        "opencv-python-headless",
        "websockets",
        "uvloop",
        "onnx",
    }
    assert packages["numpy"] == np.__version__


def test_package_that_is_not_installed_maps_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(environment, "PACKAGES", ("numpy", "no-such-distribution"))

    assert collect_environment(None)["packages"] == {
        "numpy": np.__version__,
        "no-such-distribution": None,
    }


def test_model_sha256_covers_every_onnx_file_in_the_model_directory(tmp_path: Path) -> None:
    (tmp_path / "depth.onnx").write_bytes(b"abc")
    (tmp_path / "cls.onnx").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"not a model")

    assert collect_environment(tmp_path)["model_sha256"] == {
        "cls.onnx": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "depth.onnx": "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
    }


def test_model_sha256_is_empty_without_a_model_directory() -> None:
    assert collect_environment(None)["model_sha256"] == {}


def test_model_directory_that_does_not_exist_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        collect_environment(tmp_path / "missing")


def test_start_utc_is_an_iso_8601_timestamp_in_utc() -> None:
    start_utc = collect_environment(None)["start_utc"]

    assert isinstance(start_utc, str)
    assert datetime.fromisoformat(start_utc).utcoffset() == timedelta(0)
