from pathlib import Path

from common.cgroup import Limits, format_limits, parse_cpu_max, parse_memory_max, read_limits

FOUR_GIB = 4 * 2**30


def test_parse_cpu_max_with_quota() -> None:
    assert parse_cpu_max("200000 100000") == (2.0, 100000)


def test_parse_cpu_max_without_quota() -> None:
    assert parse_cpu_max("max 100000") == (None, 100000)


def test_parse_memory_max() -> None:
    assert parse_memory_max("4294967296") == FOUR_GIB
    assert parse_memory_max("max") is None


def test_read_limits_v2(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n")
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    (tmp_path / "memory.max").write_text("4294967296\n")
    (tmp_path / "memory.swap.max").write_text("0\n")

    assert read_limits(tmp_path) == Limits(2, 2.0, 100000, FOUR_GIB, 0)


def test_read_limits_v2_unlimited(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n")
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")

    assert read_limits(tmp_path) == Limits(2, None, 100000, None, None)


def test_read_limits_v1(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("4294967296\n")

    assert read_limits(tmp_path) == Limits(1, 2.0, 100000, FOUR_GIB, None)


def test_read_limits_v1_unlimited(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")

    assert read_limits(tmp_path) == Limits(1, None, 100000, None, None)


def test_read_limits_without_cgroup_files(tmp_path: Path) -> None:
    assert read_limits(tmp_path) == Limits(1, None, None, None, None)


def test_format_limits() -> None:
    limited = Limits(2, 2.0, 100000, FOUR_GIB, 0)
    assert format_limits(limited) == "cgroup v2: cpu limit 2 CPUs, memory limit 4 GiB"
    unlimited = Limits(2, None, 100000, None, None)
    assert format_limits(unlimited) == "cgroup v2: cpu limit unlimited, memory limit unlimited"
