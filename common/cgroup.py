"""Read the container's limits and usage counters from its own cgroup files.

Supports cgroup v2 and falls back to v1. The limits are logged at startup and
checked by the benchmark gate, so every result records the limits it ran under.
The usage counters are read once per metrics scrape and around benchmark windows.
Limits and CgroupStats together are the ``cgroup`` object of the metrics contract
in DESIGN.md, so their field names are published by the metrics endpoint and
recorded in every benchmark result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")

# cgroup v1 reports "no limit" as a huge page-rounded byte count, not a keyword.
_V1_UNLIMITED_BYTES = 1 << 60

_NS_PER_US = 1000


class CgroupError(Exception):
    """A cgroup file is unreadable or malformed, or one that every cgroup has is missing."""


@dataclass(frozen=True)
class Limits:
    """Limits as seen from inside the container. None means unlimited or unknown."""

    version: int
    cpu_limit_cores: float | None
    cpu_period_us: int | None
    memory_limit_bytes: int | None
    memory_swap_limit_bytes: int | None  # swap allowed on top of memory; reported by v2 only
    cpuset: str | None  # effective CPU list, in the kernel's list format ("0-3,8")


@dataclass(frozen=True)
class CgroupStats:
    """Usage of the container's cgroup. None means the kernel does not expose the value here."""

    cpu_usage_us: int | None
    cpu_user_us: int | None
    cpu_system_us: int | None
    nr_periods: int | None
    nr_throttled: int | None
    throttled_us: int | None
    memory_current_bytes: int | None  # includes page cache
    memory_peak_bytes: int | None
    memory_anon_bytes: int | None
    memory_file_bytes: int | None
    memory_sock_bytes: int | None
    oom_kills: int | None


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def parse_cpu_max(text: str) -> tuple[float | None, int]:
    """Parse cgroup v2 ``cpu.max``: "<quota|max> <period>", both in microseconds."""
    quota, period = text.split()
    period_us = int(period)
    if quota == "max":
        return None, period_us
    return int(quota) / period_us, period_us


def parse_memory_max(text: str) -> int | None:
    """Parse a cgroup v2 memory limit: a byte count, or "max" for no limit."""
    return None if text == "max" else int(text)


def parse_flat_keyed(text: str) -> dict[str, int]:
    """Parse a flat-keyed cgroup file: one "<key> <integer>" pair per line."""
    return {key: int(value) for key, value in (line.split() for line in text.splitlines())}


def _read_limits_v2(root: Path) -> Limits:
    cpu_limit_cores: float | None = None
    cpu_period_us: int | None = None
    cpu_text = _read(root / "cpu.max")
    if cpu_text:
        cpu_limit_cores, cpu_period_us = parse_cpu_max(cpu_text)
    memory_text = _read(root / "memory.max")
    swap_text = _read(root / "memory.swap.max")
    return Limits(
        version=2,
        cpu_limit_cores=cpu_limit_cores,
        cpu_period_us=cpu_period_us,
        memory_limit_bytes=parse_memory_max(memory_text) if memory_text else None,
        memory_swap_limit_bytes=parse_memory_max(swap_text) if swap_text else None,
        cpuset=_read(root / "cpuset.cpus.effective"),
    )


def _read_limits_v1(root: Path) -> Limits:
    quota_text = _read(root / "cpu" / "cpu.cfs_quota_us")
    period_text = _read(root / "cpu" / "cpu.cfs_period_us")
    cpu_period_us = int(period_text) if period_text else None
    cpu_limit_cores: float | None = None
    if quota_text and cpu_period_us and int(quota_text) > 0:  # -1 means no quota
        cpu_limit_cores = int(quota_text) / cpu_period_us
    memory_text = _read(root / "memory" / "memory.limit_in_bytes")
    memory_limit_bytes = int(memory_text) if memory_text else None
    if memory_limit_bytes is not None and memory_limit_bytes >= _V1_UNLIMITED_BYTES:
        memory_limit_bytes = None
    return Limits(
        version=1,
        cpu_limit_cores=cpu_limit_cores,
        cpu_period_us=cpu_period_us,
        memory_limit_bytes=memory_limit_bytes,
        memory_swap_limit_bytes=None,
        cpuset=_read(root / "cpuset" / "cpuset.effective_cpus"),
    )


def read_limits(root: Path = CGROUP_ROOT) -> Limits:
    """Return the limits of the cgroup this process runs in."""
    if (root / "cgroup.controllers").exists():
        return _read_limits_v2(root)
    return _read_limits_v1(root)


def _read_exposed(path: Path) -> str | None:
    """Return the file's text, or None for a file the kernel does not expose here."""
    try:
        return path.read_text()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CgroupError(f"cannot read {path}") from error


def _read_single_value(path: Path) -> int | None:
    text = _read_exposed(path)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError as error:
        raise CgroupError(f"{path}: expected one integer, got {text!r}") from error


def _read_flat_keyed(path: Path) -> dict[str, int]:
    """Read a flat-keyed file. One that the kernel does not expose here has no keys."""
    text = _read_exposed(path)
    if text is None:
        return {}
    try:
        return parse_flat_keyed(text)
    except ValueError as error:
        raise CgroupError(f'{path}: expected "<key> <integer>" lines') from error


def _us_from_ns(duration_ns: int | None) -> int | None:
    return None if duration_ns is None else duration_ns // _NS_PER_US


def _read_stats_v2(root: Path) -> CgroupStats:
    cpu_path = root / "cpu.stat"
    try:
        cpu = parse_flat_keyed(cpu_path.read_text())
        usage_us, user_us, system_us = cpu["usage_usec"], cpu["user_usec"], cpu["system_usec"]
    except (OSError, ValueError, KeyError) as error:
        # Every cgroup v2 directory has cpu.stat with these three keys, whether or not the
        # cpu controller is enabled, so their absence means this is not a usable cgroup.
        raise CgroupError(f"{cpu_path} is missing or malformed") from error
    memory = _read_flat_keyed(root / "memory.stat")
    return CgroupStats(
        cpu_usage_us=usage_us,
        cpu_user_us=user_us,
        cpu_system_us=system_us,
        # The throttling counters are present only where the cpu controller is enabled.
        nr_periods=cpu.get("nr_periods"),
        nr_throttled=cpu.get("nr_throttled"),
        throttled_us=cpu.get("throttled_usec"),
        # The memory files exist only where the memory controller is enabled, and not in
        # the root cgroup. memory.peak also needs Linux 5.19.
        memory_current_bytes=_read_single_value(root / "memory.current"),
        memory_peak_bytes=_read_single_value(root / "memory.peak"),
        memory_anon_bytes=memory.get("anon"),
        memory_file_bytes=memory.get("file"),
        memory_sock_bytes=memory.get("sock"),
        oom_kills=_read_flat_keyed(root / "memory.events").get("oom_kill"),
    )


def _read_stats_v1(root: Path) -> CgroupStats:
    throttling = _read_flat_keyed(root / "cpu" / "cpu.stat")
    memory = _read_flat_keyed(root / "memory" / "memory.stat")
    return CgroupStats(
        cpu_usage_us=_us_from_ns(_read_single_value(root / "cpuacct" / "cpuacct.usage")),
        cpu_user_us=_us_from_ns(_read_single_value(root / "cpuacct" / "cpuacct.usage_user")),
        cpu_system_us=_us_from_ns(_read_single_value(root / "cpuacct" / "cpuacct.usage_sys")),
        nr_periods=throttling.get("nr_periods"),
        nr_throttled=throttling.get("nr_throttled"),
        throttled_us=_us_from_ns(throttling.get("throttled_time")),
        memory_current_bytes=_read_single_value(root / "memory" / "memory.usage_in_bytes"),
        memory_peak_bytes=_read_single_value(root / "memory" / "memory.max_usage_in_bytes"),
        # The total_ counters include child cgroups, as every cgroup v2 counter does.
        memory_anon_bytes=memory.get("total_rss"),
        memory_file_bytes=memory.get("total_cache"),
        # cgroup v1 has no counter for socket memory as a whole.
        memory_sock_bytes=None,
        oom_kills=_read_flat_keyed(root / "memory" / "memory.oom_control").get("oom_kill"),
    )


def read_stats(root: Path = CGROUP_ROOT) -> CgroupStats:
    """Return the usage counters of the cgroup this process runs in."""
    if (root / "cgroup.controllers").exists():
        return _read_stats_v2(root)
    return _read_stats_v1(root)


def format_limits(limits: Limits) -> str:
    cpu = "unlimited" if limits.cpu_limit_cores is None else f"{limits.cpu_limit_cores:g} CPUs"
    memory = (
        "unlimited"
        if limits.memory_limit_bytes is None
        else f"{limits.memory_limit_bytes / 2**30:g} GiB"
    )
    return f"cgroup v{limits.version}: cpu limit {cpu}, memory limit {memory}"


def main() -> None:
    print(format_limits(read_limits()))


if __name__ == "__main__":
    main()
