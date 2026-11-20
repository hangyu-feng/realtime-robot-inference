"""Tests for reading cgroup limits and usage counters from a fake cgroup tree."""

import re
from dataclasses import asdict, fields, replace
from pathlib import Path

import pytest

from common.cgroup import (
    CgroupError,
    CgroupStats,
    Limits,
    format_limits,
    parse_cpu_max,
    parse_flat_keyed,
    parse_memory_max,
    read_limits,
    read_stats,
)

FOUR_GIB = 4 * 2**30

V2_FILES = {
    "cgroup.controllers": "cpuset cpu io memory hugetlb pids rdma\n",
    "cpu.stat": (
        "usage_usec 22680744\n"
        "user_usec 12995747\n"
        "system_usec 9684996\n"
        "core_sched.force_idle_usec 0\n"
        "nr_periods 310\n"
        "nr_throttled 7\n"
        "throttled_usec 412345\n"
        "nr_bursts 0\n"
        "burst_usec 0\n"
    ),
    "memory.current": "794365952\n",
    "memory.peak": "810352640\n",
    "memory.stat": "anon 97763328\nfile 655941632\nkernel 24129536\nsock 4096\nshmem 819200\n",
    "memory.events": "low 0\nhigh 0\nmax 3\noom 1\noom_kill 1\noom_group_kill 0\n",
}

V2_STATS = CgroupStats(
    cpu_usage_us=22680744,
    cpu_user_us=12995747,
    cpu_system_us=9684996,
    nr_periods=310,
    nr_throttled=7,
    throttled_us=412345,
    memory_current_bytes=794365952,
    memory_peak_bytes=810352640,
    memory_anon_bytes=97763328,
    memory_file_bytes=655941632,
    memory_sock_bytes=4096,
    oom_kills=1,
)

# The same usage as V2_FILES, in the units and names of cgroup v1. The local "cache" and
# "rss" lines differ from the "total_" ones, which are the ones that include child cgroups.
V1_FILES = {
    "cpuacct/cpuacct.usage": "22680744999\n",
    "cpuacct/cpuacct.usage_user": "12995747123\n",
    "cpuacct/cpuacct.usage_sys": "9684996456\n",
    "cpu/cpu.stat": "nr_periods 310\nnr_throttled 7\nthrottled_time 412345678\n",
    "memory/memory.usage_in_bytes": "794365952\n",
    "memory/memory.max_usage_in_bytes": "810352640\n",
    "memory/memory.stat": (
        "cache 1000\n"
        "rss 2000\n"
        "hierarchical_memory_limit 4294967296\n"
        "total_cache 655941632\n"
        "total_rss 97763328\n"
    ),
    "memory/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\noom_kill 1\n",
}


# The fields of the "cgroup" object in the metrics contract, DESIGN.md section 10.7.
METRICS_CGROUP_FIELDS = [
    "version",
    "cpu_limit_cores",
    "cpu_period_us",
    "cpuset",
    "cpu_usage_us",
    "cpu_user_us",
    "cpu_system_us",
    "nr_periods",
    "nr_throttled",
    "throttled_us",
    "memory_limit_bytes",
    "memory_swap_limit_bytes",
    "memory_current_bytes",
    "memory_peak_bytes",
    "memory_anon_bytes",
    "memory_file_bytes",
    "memory_sock_bytes",
    "oom_kills",
]


def write_files(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_limits_and_stats_together_have_the_cgroup_fields_of_the_metrics_contract() -> None:
    names = [field.name for field in (*fields(Limits), *fields(CgroupStats))]

    assert sorted(names) == sorted(METRICS_CGROUP_FIELDS)


def test_parse_cpu_max_with_quota() -> None:
    assert parse_cpu_max("200000 100000") == (2.0, 100000)


def test_parse_cpu_max_without_quota() -> None:
    assert parse_cpu_max("max 100000") == (None, 100000)


def test_parse_memory_max() -> None:
    assert parse_memory_max("4294967296") == FOUR_GIB
    assert parse_memory_max("max") is None


def test_parse_flat_keyed() -> None:
    text = "usage_usec 12\ncore_sched.force_idle_usec 0\n"

    assert parse_flat_keyed(text) == {"usage_usec": 12, "core_sched.force_idle_usec": 0}


@pytest.mark.parametrize("text", ["anon\n", "anon 1 2\n", "anon many\n", "anon 1\n\nfile 2\n"])
def test_parse_flat_keyed_rejects_a_malformed_line(text: str) -> None:
    with pytest.raises(ValueError):
        parse_flat_keyed(text)


def test_read_limits_v2(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n")
    (tmp_path / "cpu.max").write_text("200000 100000\n")
    (tmp_path / "memory.max").write_text("4294967296\n")
    (tmp_path / "memory.swap.max").write_text("0\n")
    (tmp_path / "cpuset.cpus.effective").write_text("0-3,8\n")

    assert read_limits(tmp_path) == Limits(2, 2.0, 100000, FOUR_GIB, 0, "0-3,8")


def test_read_limits_v2_unlimited(tmp_path: Path) -> None:
    (tmp_path / "cgroup.controllers").write_text("cpu memory\n")
    (tmp_path / "cpu.max").write_text("max 100000\n")
    (tmp_path / "memory.max").write_text("max\n")

    assert read_limits(tmp_path) == Limits(2, None, 100000, None, None, None)


def test_read_limits_v1(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpuset").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("200000\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("4294967296\n")
    (tmp_path / "cpuset" / "cpuset.effective_cpus").write_text("0-1\n")

    assert read_limits(tmp_path) == Limits(1, 2.0, 100000, FOUR_GIB, None, "0-1")


def test_read_limits_v1_unlimited(tmp_path: Path) -> None:
    (tmp_path / "cpu").mkdir()
    (tmp_path / "memory").mkdir()
    (tmp_path / "cpu" / "cpu.cfs_quota_us").write_text("-1\n")
    (tmp_path / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
    (tmp_path / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")

    assert read_limits(tmp_path) == Limits(1, None, 100000, None, None, None)


def test_read_limits_without_cgroup_files(tmp_path: Path) -> None:
    assert read_limits(tmp_path) == Limits(1, None, None, None, None, None)


def test_format_limits() -> None:
    limited = Limits(2, 2.0, 100000, FOUR_GIB, 0, "0-3")
    assert format_limits(limited) == "cgroup v2: cpu limit 2 CPUs, memory limit 4 GiB"
    unlimited = Limits(2, None, 100000, None, None, None)
    assert format_limits(unlimited) == "cgroup v2: cpu limit unlimited, memory limit unlimited"


def test_read_stats_v2(tmp_path: Path) -> None:
    write_files(tmp_path, V2_FILES)

    assert read_stats(tmp_path) == V2_STATS


@pytest.mark.parametrize(
    ("missing_file", "unexposed_fields"),
    [
        ("memory.current", ("memory_current_bytes",)),
        ("memory.peak", ("memory_peak_bytes",)),
        ("memory.stat", ("memory_anon_bytes", "memory_file_bytes", "memory_sock_bytes")),
        ("memory.events", ("oom_kills",)),
    ],
)
def test_read_stats_v2_reports_none_for_a_file_the_kernel_does_not_expose(
    tmp_path: Path, missing_file: str, unexposed_fields: tuple[str, ...]
) -> None:
    write_files(tmp_path, V2_FILES)
    (tmp_path / missing_file).unlink()

    assert read_stats(tmp_path) == replace(V2_STATS, **dict.fromkeys(unexposed_fields))


def test_read_stats_v2_without_the_cpu_controller_has_no_throttling_counters(
    tmp_path: Path,
) -> None:
    usage_only = {"cpu.stat": "usage_usec 500\nuser_usec 300\nsystem_usec 200\n"}
    write_files(tmp_path, V2_FILES | usage_only)

    assert read_stats(tmp_path) == replace(
        V2_STATS,
        cpu_usage_us=500,
        cpu_user_us=300,
        cpu_system_us=200,
        nr_periods=None,
        nr_throttled=None,
        throttled_us=None,
    )


def test_read_stats_v2_in_the_root_cgroup_reports_what_the_root_exposes(tmp_path: Path) -> None:
    # Outside a container the mount point is the root cgroup, which has no throttling
    # counters and no memory.current, memory.peak or memory.events.
    root_files = {
        "cgroup.controllers": V2_FILES["cgroup.controllers"],
        "cpu.stat": "usage_usec 500\nuser_usec 300\nsystem_usec 200\n",
        "memory.stat": V2_FILES["memory.stat"],
    }
    write_files(tmp_path, root_files)

    assert read_stats(tmp_path) == CgroupStats(
        cpu_usage_us=500,
        cpu_user_us=300,
        cpu_system_us=200,
        nr_periods=None,
        nr_throttled=None,
        throttled_us=None,
        memory_current_bytes=None,
        memory_peak_bytes=None,
        memory_anon_bytes=97763328,
        memory_file_bytes=655941632,
        memory_sock_bytes=4096,
        oom_kills=None,
    )


def test_read_stats_v2_without_cpu_stat_raises(tmp_path: Path) -> None:
    write_files(tmp_path, V2_FILES)
    (tmp_path / "cpu.stat").unlink()

    with pytest.raises(CgroupError, match=re.escape("cpu.stat")) as raised:
        read_stats(tmp_path)

    assert isinstance(raised.value.__cause__, FileNotFoundError)


def test_read_stats_v1(tmp_path: Path) -> None:
    write_files(tmp_path, V1_FILES)

    assert read_stats(tmp_path) == replace(V2_STATS, memory_sock_bytes=None)


def test_read_stats_v1_without_an_oom_kill_counter(tmp_path: Path) -> None:
    oom_control = {"memory/memory.oom_control": "oom_kill_disable 0\nunder_oom 0\n"}
    write_files(tmp_path, V1_FILES | oom_control)

    assert read_stats(tmp_path) == replace(V2_STATS, memory_sock_bytes=None, oom_kills=None)


def test_read_stats_without_cgroup_files_reports_none_for_every_field(tmp_path: Path) -> None:
    assert set(asdict(read_stats(tmp_path)).values()) == {None}


@pytest.mark.parametrize(
    ("files", "name", "text"),
    [
        (V2_FILES, "cpu.stat", "usage_usec 500\nuser_usec 300\n"),
        (V2_FILES, "cpu.stat", "usage_usec many\nuser_usec 300\nsystem_usec 200\n"),
        (V2_FILES, "memory.current", "\n"),
        (V2_FILES, "memory.peak", "plenty\n"),
        (V2_FILES, "memory.stat", "anon\n"),
        (V2_FILES, "memory.events", "oom_kill 1 2\n"),
        (V1_FILES, "cpuacct/cpuacct.usage", "fast\n"),
        (V1_FILES, "cpu/cpu.stat", "nr_periods\n"),
        (V1_FILES, "memory/memory.stat", "total_rss lots\n"),
    ],
)
def test_read_stats_rejects_a_malformed_file(
    tmp_path: Path, files: dict[str, str], name: str, text: str
) -> None:
    write_files(tmp_path, files | {name: text})

    with pytest.raises(CgroupError, match=re.escape(name)) as raised:
        read_stats(tmp_path)

    assert isinstance(raised.value.__cause__, (KeyError, ValueError))


def test_read_stats_rejects_a_file_that_exists_but_cannot_be_read(tmp_path: Path) -> None:
    write_files(tmp_path, V2_FILES)
    (tmp_path / "memory.current").unlink()
    (tmp_path / "memory.current").mkdir()

    with pytest.raises(CgroupError, match=re.escape("memory.current")) as raised:
        read_stats(tmp_path)

    assert isinstance(raised.value.__cause__, OSError)
